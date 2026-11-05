"""Batched and chunked prefill against tiny randomly initialised models (CPU, no download).

One model per version 1 architecture (Qwen 2, Llama, Gemma 2), with the backend's
greedy generation settings. In full precision a padded batch, prefilled in one pass
or in chunks, must answer exactly as each prompt alone. On Kaggle the notebook reruns
this file after installing the GPU packages, with NIKASH_REQUIRE_PREFILL_TESTS=1, so
the transformers that will run the models is checked before any download or GPU time.

Skipped where torch or transformers is not installed: the rest of the suite needs
the standard library only. With NIKASH_REQUIRE_PREFILL_TESTS=1 nothing is skipped:
a missing package or a transformers before 4.53 fails instead. The backend's
retry, cut-off and worker-deadline logic is tested with stand-ins and needs neither.
"""

from __future__ import annotations

import gc
import importlib.util
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


def transformers_at_least(version: str) -> bool:
    # From the package metadata: importing transformers (and torch) here would slow every worker process
    # that imports this module for a stand-in.
    if not HAVE_TORCH:
        return False
    from importlib.metadata import version as installed
    return tuple(int(x) for x in installed("transformers").split(".")[:2]) >= tuple(int(x) for x in version.split("."))


if os.environ.get("NIKASH_REQUIRE_PREFILL_TESTS") == "1" and not transformers_at_least("4.53"):
    raise RuntimeError("the prefill tests need torch and transformers 4.53 or later here; none may be skipped")


class IntTokenizer:
    """Prompts are space-separated token ids; 0 is padding. Pads on the left, like the backend's tokenizer."""

    pad_token_id = 0

    def __call__(self, prompts, return_tensors=None, padding=False, add_special_tokens=False):
        import torch
        seqs = [[int(t) for t in p.split()] for p in prompts]
        width = max(map(len, seqs))
        ids = torch.tensor([[0] * (width - len(s)) + s for s in seqs])
        mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in seqs])
        return _Encoding(input_ids=ids, attention_mask=mask)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [" ".join(str(int(t)) for t in row if int(t) != 0) for row in rows]


class _Encoding(dict):
    def to(self, _device):
        return self


SHAPE = dict(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=256, pad_token_id=0, bos_token_id=None, eos_token_id=None)


@unittest.skipUnless(HAVE_TORCH, "needs torch and transformers")
class ChunkedPrefill:
    """Mixin: subclasses set `model()`. Prompt lengths 9-40 with 8-token chunks, so every prompt spans chunks."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch

        from nikash.models.hf_backend import HFModel, make_greedy

        torch.manual_seed(0)
        model = cls.model().eval()
        make_greedy(model.generation_config)
        model.generation_config.eos_token_id = None
        model.generation_config.suppress_tokens = [0]   # a random model may emit the padding id, which decoding drops
        cls.torch = torch
        cls.backend = HFModel.__new__(HFModel)
        cls.backend.model, cls.backend.tokenizer, cls.backend._torch = model, IntTokenizer(), torch
        rng = torch.Generator().manual_seed(1)
        cls.prompts = [" ".join(str(int(t)) for t in torch.randint(1, 97, (n,), generator=rng)) for n in (23, 9, 40, 17)]

    def alone(self, prompt, limit):
        ids = self.torch.tensor([[int(t) for t in prompt.split()]])
        with self.torch.inference_mode():
            out = self.backend.model.generate(input_ids=ids, attention_mask=self.torch.ones_like(ids),
                                              max_new_tokens=limit, do_sample=False, pad_token_id=0)
        return " ".join(str(int(t)) for t in out[0, ids.shape[1]:])

    def test_a_padded_batch_prefilled_in_one_pass_answers_as_each_prompt_alone(self):
        self.backend.last_stats = {"at_token_limit": 0}
        batched = self.backend._batched(self.prompts, 6)
        self.assertEqual(batched, [self.alone(p, 6) for p in self.prompts])
        self.assertEqual(self.backend.last_stats["at_token_limit"], len(self.prompts))

    def test_a_batch_prefilled_in_chunks_answers_as_each_prompt_alone(self):
        self.backend.last_stats = {"at_token_limit": 0}
        batched = self.backend._chunked(self.prompts, 6, chunk=8)
        self.assertEqual(batched, [self.alone(p, 6) for p in self.prompts])
        self.assertEqual(self.backend.last_stats["at_token_limit"], len(self.prompts))

    def test_a_failing_chunked_batch_is_rerun_one_prompt_at_a_time(self):
        backend = self.backend
        backend.last_stats = {"at_token_limit": 0, "batch_fallbacks": 0, "oom_retries": 0}
        backend._free = lambda: None
        real = backend._chunked

        def refuse_batches(prompts, limit, chunk):
            if len(prompts) > 1:
                raise RuntimeError("simulated failure of a padded batch")
            return real(prompts, limit, chunk)

        backend._chunked = refuse_batches
        try:
            out = backend._run(self.prompts, 8, 6)
        finally:
            del backend._chunked
        self.assertEqual(out, [self.alone(p, 6) for p in self.prompts])
        self.assertEqual(backend.last_stats["batch_fallbacks"], 1)


class LlamaPrefillTests(ChunkedPrefill, unittest.TestCase):
    @staticmethod
    def model():
        from transformers import LlamaConfig, LlamaForCausalLM
        return LlamaForCausalLM(LlamaConfig(**SHAPE))


class QwenPrefillTests(ChunkedPrefill, unittest.TestCase):
    @staticmethod
    def model():
        from transformers import Qwen2Config, Qwen2ForCausalLM
        return Qwen2ForCausalLM(Qwen2Config(**SHAPE))


@unittest.skipUnless(transformers_at_least("4.53"),
                     "transformers before 4.53 applies Gemma 2's sliding window wrongly to any chunked prefill")
class GemmaPrefillTests(ChunkedPrefill, unittest.TestCase):
    # A sliding window shorter than the prompts, as Gemma 2's 4,096 tokens are for the longest windows.
    @staticmethod
    def model():
        from transformers import Gemma2Config, Gemma2ForCausalLM
        return Gemma2ForCausalLM(Gemma2Config(**SHAPE, head_dim=16, sliding_window=16, attn_implementation="eager"))


class FakeOOM(RuntimeError):
    pass


class Rows:
    """The few tensor operations the backend's cut-off count uses, on lists."""

    def __init__(self, rows):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, key):
        return SimpleNamespace(tolist=lambda: [row[key[1]] for row in self.rows])


def stand_in(**attrs):
    from nikash.models.hf_backend import HFModel
    backend = HFModel.__new__(HFModel)
    backend._torch = SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeOOM, empty_cache=lambda: None))
    backend.prefill_chunk = 1024
    backend.last_stats = {"at_token_limit": 0, "batch_fallbacks": 0, "oom_retries": 0}
    backend.__dict__.update(attrs)
    return backend


class HangingModel:
    """Stands in for HFModel inside an HFPool worker process (CPU only). Hangs on its first share holding "hang"."""

    def __init__(self, model_id, *, device, marker):
        self.name, self.batch_size, self.prefill_chunk, self.marker = "fake", 8, 1024, marker
        self.last_stats = {}

    def describe(self):
        return {"id": "fake", "name": "fake", "note": "CPU stand-in"}

    def generate_many(self, batch, max_new_tokens=None):
        prompts = [messages[-1]["content"] for messages, _case in batch]
        if "hang" in prompts and not Path(self.marker).exists():
            Path(self.marker).touch()
            time.sleep(3600)
        self.last_stats = {"prompt_tokens_mean": 1, "prompt_tokens_max": 1, "batches": 1, "chunked_prompts": 0,
                           "oom_retries": 0, "memory_budget_gib": 1.0}
        return prompts


class BackendLogicTests(unittest.TestCase):
    """Retries, counts and deadlines, with stand-ins for torch, the model and the GPU workers."""

    def test_cuda_allocation_failures_raised_as_runtime_errors_are_out_of_memory(self):
        for message in ("CUDA error: out of memory", "CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"):
            def short(prompts, limit):
                if len(prompts) > 1:
                    raise RuntimeError(message)
                return [p.upper() for p in prompts]

            backend = stand_in(_batched=short)
            self.assertEqual(backend._run(["a", "b", "c", "d"], None, 6), ["A", "B", "C", "D"])
            self.assertEqual((backend.last_stats["oom_retries"], backend.last_stats["batch_fallbacks"]), (3, 0))

        def broken(prompts, limit):
            raise RuntimeError("shape mismatch")

        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            stand_in(_batched=broken)._run(["a", "b"], None, 6)

    def test_a_failed_batch_is_freed_before_its_prompts_are_rerun(self):
        class Tensors:
            pass

        failed, alive_at_rerun = [], []

        def chunked(prompts, limit, chunk):
            if len(prompts) > 1:
                tensors = Tensors()
                failed.append(weakref.ref(tensors))
                raise ValueError("simulated failure of a padded batch")
            alive_at_rerun.append(any(ref() is not None for ref in failed))
            return list(prompts)

        backend = stand_in(_chunked=chunked, _free=gc.collect)
        self.assertEqual(backend._run(["a", "b", "c"], 8, 6), ["a", "b", "c"])
        self.assertEqual((alive_at_rerun, backend.last_stats["batch_fallbacks"]), ([False] * 3, 1))

    def test_an_answer_is_cut_off_only_if_it_used_every_token_without_ending(self):
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=9, batch_decode=lambda rows, **_kw: [""] * rows.shape[0])
        backend = stand_in(tokenizer=tokenizer, model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[7, 8])))
        rows = Rows([[5, 6, 5, 7],    # ends with an EOS id on its last allowed token
                     [5, 6, 5, 9],    # ends with the tokenizer's EOS
                     [5, 8, 0, 0],    # ended earlier, then padding
                     [5, 6, 5, 6]])   # still going: cut off
        backend._decode_new(rows, 4)
        self.assertEqual(backend.last_stats["at_token_limit"], 1)
        backend._decode_new(rows, 5)
        self.assertEqual(backend.last_stats["at_token_limit"], 1)
        backend.model.generation_config.eos_token_id = 7
        backend._decode_new(Rows([[5, 6, 5, 8], [5, 6, 5, 7]]), 4)
        self.assertEqual(backend.last_stats["at_token_limit"], 2)

    def test_chunked_prompts_counts_prompts_not_batches(self):
        from nikash.models.hf_backend import model_dims

        dims = model_dims(SimpleNamespace(num_attention_heads=4, num_key_value_heads=2, hidden_size=64, num_hidden_layers=2,
                                          intermediate_size=128, vocab_size=97, max_position_embeddings=4096))
        cuda = SimpleNamespace(OutOfMemoryError=FakeOOM, empty_cache=lambda: None, reset_peak_memory_stats=lambda _i: None,
                               max_memory_allocated=lambda _i: 0)
        tokenizer = lambda prompts, add_special_tokens=False: {"input_ids": [p.split() for p in prompts]}  # noqa: E731
        backend = stand_in(_torch=SimpleNamespace(cuda=cuda), tokenizer=tokenizer, dims=dims, memory_budget=2**40,
                           batch_size=2, prefill_chunk=8, max_new_tokens=4, device=None, _devices=[0],
                           _prompt=lambda messages: messages[0]["content"], _run=lambda prompts, chunk, limit: prompts)
        prompts = ["1 " * 20, "1 " * 30, "1 " * 40, "1 2", "1 2 3"]
        self.assertEqual(backend.generate_many([([{"content": p}], None) for p in prompts]), prompts)
        # Batches of 2, by length: the two short prompts in one pass, then the three long ones in chunks.
        self.assertEqual((backend.last_stats["batches"], backend.last_stats["chunked_prompts"]), (3, 3))

    def test_a_hung_worker_is_stopped_at_its_deadline_restarted_and_its_share_rerun(self):
        from nikash.models.hf_backend import HFPool

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        logs = []
        pool = HFPool("fake/model", devices=[0, 1], factory=HangingModel, log=logs.append, job_timeout=3,
                      marker=str(Path(folder.name) / "hung"))
        self.addCleanup(pool.close)
        began = time.monotonic()
        texts = pool.generate_many([([{"role": "user", "content": p}], None) for p in ("hang", "x")])
        self.assertEqual(texts, ["hang", "x"])
        self.assertEqual(pool.last_stats["worker_restarts"], 1)
        self.assertIn("gave no answer within 3 seconds", logs[0])
        self.assertLess(time.monotonic() - began, 60)
        pool.close()
        self.assertEqual(multiprocessing.active_children(), [])


if __name__ == "__main__":
    unittest.main()
