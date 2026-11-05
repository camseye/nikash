"""Hugging Face backend for the Kaggle run (GPU, 4-bit, greedy decoding).

Imports torch and transformers lazily so the rest of the harness and the
smoke test run with the standard library only. The access token for gated
models is passed in by the caller (the notebook reads it from Kaggle
Secrets); it is never read from or written to a file here.

GPU memory. A T4 has no fused (flash) attention kernel, so a forward pass of
q new tokens against k cached tokens can hold a heads × q × k score matrix:
prefilling a long prompt in one pass needs memory that grows with the square
of its length. So:

- a batch whose longest prompt has more than `prefill_chunk` tokens is fed
  to the model `prefill_chunk` tokens at a time, all its prompts together,
  filling the KV cache before decoding starts; memory then grows linearly
  with prompt length. Shorter batches are prefilled in one pass;
- prompts are batched, in order of length, only while the estimated peak
  (`step_bytes`) stays under `memory_fraction` of the GPU memory left after
  loading the model. A prompt too long to fit even alone gets smaller
  chunks. The plan depends only on prompt lengths, the model and the GPU,
  so a rerun on the same hardware batches the same way;
- if the GPU still runs out, a batch is split in half, or a chunk halved,
  and retried; retries are counted in `last_stats`. A CUDA allocation
  failure raised as a plain RuntimeError counts as running out. A batch
  prefilled in chunks that fails any other way is rerun one prompt at a
  time, and counted as `batch_fallbacks`.

Decoding is greedy for every model: `make_greedy` overrides the sampling and
penalty settings each model ships in its generation_config (Qwen2.5 sets
repetition_penalty 1.05, for example) with the values in `GREEDY`, so every
model picks its most likely next token and nothing else. It also turns off
generate()'s automatic torch.compile (which recompiles for every new batch
shape) and the model's own cache type, so Gemma 2 decodes with the same
sliding-window-aware dynamic cache whether its prompt was prefilled in one
pass or in chunks. The CPU start check (tests/test_prefill.py) applies the
same function.

`HFPool` loads one copy of the model per GPU (a 4-bit model of up to 9B
fits on one T4), each in its own worker process that sees only that GPU.
Two copies in threads of one process crashed on Kaggle's T4 x2 with a CUDA
illegal memory access; a process per GPU runs each copy exactly as a lone
model on a lone GPU. The answers don't depend on which copy ran a case.
A worker that gives no answer within its deadline is treated as failed.
"""

from __future__ import annotations

import atexit
import contextlib
import gc
import multiprocessing
import multiprocessing.util  # registers its exit hook now, so the pool's hook (registered later) runs first
import os
import queue
import time
import traceback
import weakref
from dataclasses import dataclass

from ..schema import Case

MIN_CHUNK = 64


@dataclass(frozen=True)
class Dims:
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    hidden: int
    intermediate: int
    vocab: int
    context: int | None


def model_dims(config) -> Dims:
    cfg = getattr(config, "text_config", None) or config
    heads = cfg.num_attention_heads
    return Dims(
        heads=heads,
        kv_heads=getattr(cfg, "num_key_value_heads", None) or heads,
        head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // heads,
        layers=cfg.num_hidden_layers,
        hidden=cfg.hidden_size,
        intermediate=cfg.intermediate_size,
        vocab=cfg.vocab_size,
        context=getattr(cfg, "max_position_embeddings", None),
    )


def step_bytes(d: Dims, batch: int, q_len: int, kv_len: int) -> int:
    """Upper estimate of the peak for one forward pass (fp16 activations):
    `batch` sequences of `q_len` new tokens against `kv_len` cached ones."""
    scores = batch * d.heads * q_len * kv_len * 4     # attention scores and softmax, materialised
    mlp = batch * q_len * d.intermediate * 6          # gate, up and their product
    stream = batch * q_len * d.hidden * 16            # residual stream, norms, q/k/v/o projections
    logits = batch * d.vocab * 4                      # last position only, fp32
    cache = batch * kv_len * d.layers * 2 * d.kv_heads * d.head_dim * 2
    return scores + mlp + stream + logits + cache


def fit_chunk(d: Dims, total_len: int, budget: int, chunk: int) -> int:
    while chunk > MIN_CHUNK and step_bytes(d, 1, chunk, total_len) > budget:
        chunk //= 2
    if step_bytes(d, 1, chunk, total_len) > budget:
        raise ValueError(f"a {total_len}-token sequence needs more than the {budget / 2**30:.1f} GiB GPU budget "
                         f"even in {chunk}-token chunks; reduce WINDOW_CHARS in nikash/benchmarks/window.py")
    return chunk


def plan_batches(lengths: list[int], d: Dims, *, budget: int, max_new_tokens: int, batch_size: int,
                 prefill_chunk: int = 1024) -> list[tuple[list[int], int | None]]:
    """(prompt indices, chunk) pairs covering every prompt once. `chunk` None
    means one batched prefill; otherwise the batch is prefilled together,
    `chunk` tokens at a time."""
    plan: list[tuple[list[int], int | None]] = []
    group: list[int] = []
    group_chunk: int | None = None
    for i in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        n = lengths[i]
        total = n + max_new_tokens
        if d.context and total > d.context:
            raise ValueError(f"a {n}-token prompt plus {max_new_tokens} new tokens exceeds the model's "
                             f"{d.context}-token context; reduce WINDOW_CHARS in nikash/benchmarks/window.py")
        q = min(n, prefill_chunk)
        if step_bytes(d, 1, q, total) > budget:
            if group:
                plan.append((group, group_chunk))
                group = []
            plan.append(([i], fit_chunk(d, total, budget, prefill_chunk)))
            continue
        # Ascending order: n is the longest (padded) length in the group.
        if group and (len(group) == batch_size or step_bytes(d, len(group) + 1, q, total) > budget):
            plan.append((group, group_chunk))
            group = []
        group.append(i)
        group_chunk = prefill_chunk if n > prefill_chunk else None
    if group:
        plan.append((group, group_chunk))
    return plan


def describe_model(model_id: str, name: str | None = None) -> dict:
    name = name or model_id.split("/")[-1]
    return {"id": name.lower(), "name": name, "note": f"{model_id}, 4-bit NF4, greedy"}


# None turns a setting off; do_sample False with one beam is greedy search.
GREEDY = {
    "do_sample": False, "num_beams": 1,
    "temperature": None, "top_p": None, "top_k": None, "min_p": None, "typical_p": None,
    "epsilon_cutoff": None, "eta_cutoff": None, "penalty_alpha": None, "dola_layers": None,
    "repetition_penalty": None, "encoder_repetition_penalty": None, "no_repeat_ngram_size": None,
    "encoder_no_repeat_ngram_size": None, "exponential_decay_length_penalty": None, "sequence_bias": None,
    "guidance_scale": None, "watermarking_config": None, "prompt_lookup_num_tokens": None,
    "cache_implementation": None, "disable_compile": True,
}


def make_greedy(generation) -> None:
    """Sets a model's generation_config to the settings in `GREEDY`, in place. A setting this transformers
    version doesn't have is left out rather than added, except the compile switch."""
    for key, value in GREEDY.items():
        if hasattr(generation, key) or key == "disable_compile":
            setattr(generation, key, value)


def is_oom(exc: BaseException, oom_type: type) -> bool:
    """torch's OutOfMemoryError, or a CUDA or cuBLAS allocation failure raised as a plain RuntimeError."""
    text = str(exc)
    return isinstance(exc, oom_type) or (isinstance(exc, RuntimeError) and ("out of memory" in text or "ALLOC_FAILED" in text))


def eos_ids(generation, tokenizer) -> set[int]:
    ids = getattr(generation, "eos_token_id", None)
    ids = [] if ids is None else [ids] if isinstance(ids, int) else list(ids)
    extra = getattr(tokenizer, "eos_token_id", None)
    return {int(i) for i in ids + ([] if extra is None else [extra])}


def count_cut_off(new, limit: int, stop: set[int]) -> int:
    """Answers that used all `limit` new tokens without ending: their last token is neither an EOS id nor the
    padding that follows an answer which ended earlier."""
    if new.shape[1] < limit:
        return 0
    return sum(1 for token in new[:, -1].tolist() if token not in stop)


def _cuda_index(device) -> int | None:
    if isinstance(device, int):
        return device
    text = str(device)
    if text.startswith("cuda"):
        return int(text.partition(":")[2] or 0)
    return None


class HFModel:
    def __init__(self, model_id: str, *, name: str | None = None, token: str | None = None,
                 revision: str | None = None, load_in_4bit: bool = True, max_new_tokens: int = 384,
                 batch_size: int = 8, prefill_chunk: int = 1024, memory_fraction: float = 0.5,
                 device: int | None = None) -> None:
        """`device`: load the whole model on that GPU; None spreads it over all GPUs. `revision`: a commit of
        the model's repository (weights, tokenizer and chat template); None takes the latest, and
        `self.revision` is then the commit that was loaded."""
        # Must be set before CUDA is initialised; it reduces fragmentation on long prompts.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_id = model_id
        self.name = name or describe_model(model_id)["name"]
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.prefill_chunk = prefill_chunk
        self.device = device
        self.last_stats: dict = {}
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, revision=revision, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant = None
        if load_in_4bit:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        extra = {}
        # Gemma 2 soft-caps its attention logits, which only the eager attention path applies.
        if getattr(AutoConfig.from_pretrained(model_id, token=token, revision=revision), "model_type", "") == "gemma2":
            extra["attn_implementation"] = "eager"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, token=token, revision=revision, quantization_config=quant, dtype=torch.float16,
            device_map="auto" if device is None else {"": device}, **extra,
        )
        self.model.eval()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        make_greedy(self.model.generation_config)

        self.dims = model_dims(self.model.config)
        if device is None:
            devices = {_cuda_index(v) for v in (getattr(self.model, "hf_device_map", None) or {}).values()} - {None}
            devices = devices or {torch.cuda.current_device()}
        else:
            devices = {device}
        self._devices = sorted(devices)
        spare = min(torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i) for i in devices)
        self.memory_budget = int(spare * memory_fraction)

    def describe(self) -> dict:
        return describe_model(self.model_id, self.name)

    def _prompt(self, messages: list[dict]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            # Some chat templates reject a system role; fold it into the first user turn.
            merged = [{"role": "user", "content": f"{messages[0]['content']}\n\n{messages[1]['content']}"}, *messages[2:]]
            return self.tokenizer.apply_chat_template(merged, **kwargs)

    def _free(self) -> None:
        gc.collect()
        self._torch.cuda.empty_cache()

    # The chat template already holds the special tokens, so none are added here.
    def _batched(self, prompts: list[str], limit: int) -> list[str]:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self._torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=limit, do_sample=False,
                                      pad_token_id=self.tokenizer.pad_token_id)
        return self._decode_new(out[:, enc["input_ids"].shape[1]:], limit)

    def _decode_new(self, new, limit: int) -> list[str]:
        stop = eos_ids(self.model.generation_config, self.tokenizer) | {self.tokenizer.pad_token_id}
        self.last_stats["at_token_limit"] += count_cut_off(new, limit, stop)
        return self.tokenizer.batch_decode(new, skip_special_tokens=True)

    def _chunked(self, prompts: list[str], limit: int, chunk: int) -> list[str]:
        torch = self._torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        n = ids.shape[1]
        # Left padding: each prompt's positions count from its first real token, as generate() counts them.
        positions = (mask.long().cumsum(-1) - 1).clamp(min=0)
        cache = None
        with torch.inference_mode():
            # Prefill all but the last token; generate() then sees a cache of n - 1 tokens and decodes from there.
            for start in range(0, n - 1, chunk):
                end = min(start + chunk, n - 1)
                out = self.model(input_ids=ids[:, start:end], attention_mask=mask[:, :end],
                                 position_ids=positions[:, start:end], past_key_values=cache,
                                 use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                del out
            gen = self.model.generate(input_ids=ids, attention_mask=mask, past_key_values=cache,
                                      max_new_tokens=limit, do_sample=False, pad_token_id=self.tokenizer.pad_token_id)
        return self._decode_new(gen[:, n:], limit)

    def _run(self, prompts: list[str], chunk: int | None, limit: int) -> list[str]:
        try:
            if chunk is None:
                return self._batched(prompts, limit)
            return self._chunked(prompts, limit, chunk)
        except Exception as exc:
            oom = is_oom(exc, self._torch.cuda.OutOfMemoryError)
            if not oom and (chunk is None or len(prompts) == 1):
                raise
        # Retry outside the except block: its traceback holds the failed attempt's tensors until it ends.
        self._free()
        if not oom:
            self.last_stats["batch_fallbacks"] += 1
            return [text for p in prompts for text in self._run([p], chunk, limit)]
        self.last_stats["oom_retries"] += 1
        if len(prompts) > 1:
            half = len(prompts) // 2
            return self._run(prompts[:half], chunk, limit) + self._run(prompts[half:], chunk, limit)
        smaller = (chunk or self.prefill_chunk) // 2
        if smaller < MIN_CHUNK:
            raise self._torch.cuda.OutOfMemoryError(f"out of GPU memory even with {MIN_CHUNK}-token prefill chunks")
        return self._run(prompts, smaller, limit)

    def generate_many(self, batch: list[tuple[list[dict], Case]], max_new_tokens: int | None = None) -> list[str]:
        limit = max_new_tokens or self.max_new_tokens
        if not batch:
            self.last_stats = {}
            return []
        cuda = self._torch.cuda
        for i in self._devices:
            cuda.reset_peak_memory_stats(i)
        prompts = [self._prompt(messages) for messages, _case in batch]
        lengths = [len(ids) for ids in self.tokenizer(prompts, add_special_tokens=False)["input_ids"]]
        plan = plan_batches(lengths, self.dims, budget=self.memory_budget, max_new_tokens=limit,
                            batch_size=self.batch_size, prefill_chunk=self.prefill_chunk)
        self.last_stats = {
            "prompt_tokens_mean": round(sum(lengths) / max(1, len(lengths))),
            "prompt_tokens_max": max(lengths, default=0),
            "batches": len(plan),
            "chunked_prompts": sum(len(group) for group, chunk in plan if chunk is not None),
            "oom_retries": 0,
            "batch_fallbacks": 0,
            "at_token_limit": 0,
            "memory_budget_gib": round(self.memory_budget / 2**30, 2),
        }
        outputs = [""] * len(batch)
        # Some 4-bit kernels use the current CUDA device rather than the tensor's.
        with cuda.device(self.device) if self.device is not None else contextlib.nullcontext():
            for group, chunk in plan:
                for i, text in zip(group, self._run([prompts[i] for i in group], chunk, limit)):
                    outputs[i] = text
                self._free()
        self.last_stats["peak_gpu_gib"] = round(max(cuda.max_memory_allocated(i) for i in self._devices) / 2**30, 2)
        return outputs

    def generate(self, messages: list[dict], case: Case) -> str:
        return self.generate_many([(messages, case)])[0]


def split_work(sizes: list[int], workers: int) -> list[list[int]]:
    """Indices dealt largest first to the least-loaded worker, so shares finish together."""
    shares: list[list[int]] = [[] for _ in range(workers)]
    load = [0] * workers
    for i in sorted(range(len(sizes)), key=lambda i: -sizes[i]):
        w = load.index(min(load))
        shares[w].append(i)
        load[w] += sizes[i]
    return shares


def merge_stats(parts: list[tuple[int, dict]]) -> dict:
    """Per-copy `last_stats` (with each copy's prompt count) → one record for the pool."""
    parts = [(n, s) for n, s in parts if n]
    total = sum(n for n, _ in parts)
    return {
        "prompt_tokens_mean": round(sum(n * s["prompt_tokens_mean"] for n, s in parts) / max(1, total)),
        "prompt_tokens_max": max((s["prompt_tokens_max"] for _, s in parts), default=0),
        "batches": sum(s["batches"] for _, s in parts),
        "chunked_prompts": sum(s["chunked_prompts"] for _, s in parts),
        "oom_retries": sum(s["oom_retries"] for _, s in parts),
        "batch_fallbacks": sum(s.get("batch_fallbacks", 0) for _, s in parts),
        "at_token_limit": sum(s.get("at_token_limit", 0) for _, s in parts),
        "memory_budget_gib": min((s["memory_budget_gib"] for _, s in parts), default=0),
    }


def visible_id(device: int) -> str:
    """CUDA_VISIBLE_DEVICES for the worker of `device`, counted within any list already set here."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return visible.split(",")[device].strip() if visible else str(device)


def _serve(factory, visible: str, model_id: str, kwargs: dict, requests, replies) -> None:
    """Worker process: load one model on its one GPU, then answer jobs until sent None."""
    # Before CUDA starts in this process, so its only GPU is cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    parent = multiprocessing.parent_process()
    try:
        model = factory(model_id, device=0, **kwargs)
        replies.put(("ready", {"describe": model.describe(), "name": model.name, "batch_size": model.batch_size,
                               "prefill_chunk": model.prefill_chunk, "revision": getattr(model, "revision", None)}))
        while True:
            try:
                job = requests.get(timeout=10)
            except queue.Empty:
                # The notebook kernel was killed: free the GPU instead of waiting forever.
                if parent is not None and not parent.is_alive():
                    return
                continue
            if job is None:
                return
            messages, limit = job
            texts = model.generate_many([(m, None) for m in messages], limit)
            replies.put(("done", texts, model.last_stats))
    except BaseException:
        # A CUDA error can leave this process's GPU context unusable, so the worker exits and is replaced.
        replies.put(("error", traceback.format_exc()))


@dataclass
class _Worker:
    device: int
    process: multiprocessing.process.BaseProcess
    requests: multiprocessing.queues.Queue
    replies: multiprocessing.queues.Queue


def _reply(worker: _Worker, timeout: float, since: float | None = None) -> tuple:
    """The worker's next reply, or an error once it has died or `timeout` seconds have passed since `since`
    (a time.monotonic() value; default now). A worker past its deadline is killed."""
    deadline = (time.monotonic() if since is None else since) + timeout
    while True:
        try:
            return worker.replies.get(timeout=min(5.0, max(0.05, deadline - time.monotonic())))
        except queue.Empty:
            if not worker.process.is_alive():
                try:
                    return worker.replies.get(timeout=1)
                except queue.Empty:
                    return ("error", f"the worker process for GPU {worker.device} exited with code {worker.process.exitcode}")
            if time.monotonic() >= deadline:
                worker.process.kill()
                worker.process.join(timeout=10)
                return ("error", f"the worker process for GPU {worker.device} gave no answer within {timeout:.0f} seconds")


def _stop(workers: list[_Worker]) -> None:
    for w in workers:
        if w.process.is_alive():
            with contextlib.suppress(OSError, ValueError):
                w.requests.put(None)
    for w in workers:
        w.process.join(timeout=30)
        if w.process.is_alive():
            w.process.kill()
            w.process.join(timeout=10)
        w.requests.cancel_join_thread()
        w.replies.cancel_join_thread()
    workers.clear()


class HFPool:
    """One HFModel per GPU, each in its own process, used like a single model.

    A worker that fails (an exception, the process dying, or no answer
    within its deadline) is restarted once and its share of the batch
    rerun; greedy decoding gives the same answers. A second failure on the
    same share stops the run with the worker's traceback. Call `close()` to
    free the GPUs before loading another model.

    `job_timeout`: seconds a worker may take to answer a share of up to
    `JOB_PROMPTS` prompts, counted from when the share was sent; a larger
    share gets proportionally longer. `load_timeout`: seconds a worker may
    take to load the model (the first load downloads the weights). A hung
    CUDA call never returns, so only these deadlines end the wait for it.
    """

    JOB_PROMPTS = 64

    def __init__(self, model_id: str, *, devices: list[int] | None = None, factory=HFModel,
                 log=print, job_timeout: float = 3600.0, load_timeout: float = 3600.0, **kwargs) -> None:
        self.job_timeout, self.load_timeout = job_timeout, load_timeout
        if devices is None:
            import torch
            devices = list(range(torch.cuda.device_count()))
        self.devices = list(devices)
        if not self.devices:
            raise RuntimeError("no CUDA device; switch the notebook's accelerator to GPU")
        self._spawn = multiprocessing.get_context("spawn")
        self._load = (factory, model_id, kwargs)
        self._log = log
        self._workers: list[_Worker] = []
        self._finalizer = weakref.finalize(self, _stop, self._workers)
        # Workers aren't daemons (a daemon can't start processes, which torch.compile may need),
        # so they are stopped at exit rather than killed.
        atexit.register(_stop, self._workers)
        self.last_stats: dict = {}
        # One at a time: the first load downloads the weights, the others read the local copy.
        try:
            for d in self.devices:
                self._workers.append(self._start(d))
        except BaseException:
            _stop(self._workers)   # a copy already loaded would otherwise hold its GPU
            raise
        self.name, self.batch_size, self.prefill_chunk = (self._info[k] for k in ("name", "batch_size", "prefill_chunk"))
        self.revision = self._info.get("revision")

    def _timeout(self, prompts: int) -> float:
        return self.job_timeout * max(1.0, prompts / self.JOB_PROMPTS)

    def _start(self, device: int) -> _Worker:
        factory, model_id, kwargs = self._load
        requests, replies = self._spawn.Queue(), self._spawn.Queue()
        process = self._spawn.Process(target=_serve, name=f"nikash-gpu{device}",
                                      args=(factory, visible_id(device), model_id, kwargs, requests, replies))
        process.start()
        worker = _Worker(device, process, requests, replies)
        reply = _reply(worker, self.load_timeout)
        if reply[0] != "ready":
            _stop([worker])
            raise RuntimeError(f"the model could not be loaded on GPU {device}:\n{reply[1]}")
        self._info = reply[1]
        return worker

    def describe(self) -> dict:
        return self._info["describe"]

    def generate_many(self, batch: list[tuple[list[dict], Case]], max_new_tokens: int | None = None) -> list[str]:
        if not self._workers:
            raise RuntimeError("this HFPool is closed")
        try:
            return self._generate(batch, max_new_tokens)
        except BaseException:
            # Workers may still hold replies to this batch; never reuse them.
            self.close()
            raise

    def _generate(self, batch, max_new_tokens):
        sizes = [sum(len(m["content"]) for m in messages) for messages, _case in batch]
        shares = split_work(sizes, len(self._workers))
        jobs = [([batch[i][0] for i in share], max_new_tokens) for share in shares]
        sent = time.monotonic()
        for worker, share, job in zip(self._workers, shares, jobs):
            if share:
                worker.requests.put(job)
        outputs = [""] * len(batch)
        parts: list[tuple[int, dict]] = []
        restarts = 0
        for k, (share, job) in enumerate(zip(shares, jobs)):
            if not share:
                parts.append((0, {}))
                continue
            reply = _reply(self._workers[k], self._timeout(len(share)), since=sent)
            if reply[0] != "done":
                restarts += 1
                self._log(f"GPU {self.devices[k]}: worker failed ({reply[1].strip().splitlines()[-1]}); "
                          f"restarting it and rerunning its {len(share)} prompts once")
                _stop([self._workers[k]])
                self._workers[k] = self._start(self.devices[k])
                self._workers[k].requests.put(job)
                reply = _reply(self._workers[k], self._timeout(len(share)))
                if reply[0] != "done":
                    raise RuntimeError(f"GPU {self.devices[k]} failed twice on the same prompts; last error:\n{reply[1]}")
            texts, stats = reply[1], reply[2]
            for i, text in zip(share, texts):
                outputs[i] = text
            parts.append((len(share), stats))
        self.last_stats = {**merge_stats(parts), "gpus": len(self._workers), "worker_restarts": restarts,
                           "peak_gpu_gib_per_device": [s.get("peak_gpu_gib", 0.0) for _, s in parts]}
        return outputs

    def generate(self, messages: list[dict], case: Case) -> str:
        return self.generate_many([(messages, case)])[0]

    def close(self) -> None:
        self._finalizer()

    def __enter__(self) -> HFPool:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
