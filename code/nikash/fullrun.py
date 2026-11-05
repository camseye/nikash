"""Version 1 full run: every model x defence x case, in slices, into a Checkpoint."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .checkpoint import Checkpoint
from .defences.prompt_level import quarantine_evidence
from .run import run_matrix
from .schema import Case
from .scorer.summary import build_summary

COUNTS = ("batches", "chunked_prompts", "oom_retries", "batch_fallbacks", "at_token_limit", "worker_restarts")


class Described:
    """A model known only by its description, for summaries built after it was unloaded."""

    def __init__(self, description: dict) -> None:
        self.name = description["name"]
        self._description = description

    def describe(self) -> dict:
        return self._description


def benchmark_meta(code_dir: Path, ids: list[str]) -> list[dict]:
    """The scoreboard's description of each benchmark, so the run summary and the page agree."""
    board = json.loads((Path(code_dir) / "docs" / "data" / "scoreboard.json").read_text(encoding="utf-8"))
    return [b for b in board["benchmarks"] if b["id"] in ids]


def run_model(checkpoint: Checkpoint, model, cases: list[Case], defences: list, *, slice_size: int = 120,
              stop_at: float | None = None, log: Callable[[str], None] = print) -> dict:
    """Asks `model` for every answer the checkpoint lacks, `slice_size` cases at a time, saving each slice.
    Stops before a slice that would start after `stop_at` (a time.time() value)."""
    out: dict = {"defences": {}, "complete": True}
    for defence in defences:
        todo = checkpoint.todo(model.name, defence.id, cases)
        stats = {"cases": 0, "model_calls": 0, "seconds": 0.0, **{k: 0 for k in COUNTS}, "peak_gpu_gib": 0.0}
        out["defences"][defence.id] = stats
        for start in range(0, len(todo), slice_size):
            if stop_at is not None and time.time() > stop_at:
                out["complete"] = False
                out["stopped"] = f"time budget reached during {defence.id}"
                log(f"{model.name}: time budget reached; {len(todo) - start} {defence.id} answers left for the next session")
                return out
            part = todo[start:start + slice_size]
            began = time.time()
            records = run_matrix(part, [defence], [model])
            checkpoint.add(records)
            calls = sum(1 for c in part if defence.needs_model(c))
            last = (getattr(model, "last_stats", None) or {}) if calls else {}
            stats["cases"] += len(part)
            stats["model_calls"] += calls
            stats["seconds"] = round(stats["seconds"] + time.time() - began, 1)
            for k in COUNTS:
                stats[k] += int(last.get(k) or 0)
            stats["peak_gpu_gib"] = max([stats["peak_gpu_gib"], *last.get("peak_gpu_gib_per_device", [])])
            rate = round(stats["model_calls"] / stats["seconds"] * 3600) if stats["model_calls"] and stats["seconds"] else None
            log(f"{model.name} · {defence.id}: {start + len(part)}/{len(todo)} done, {rate} model calls/hour, "
                f"{stats['batches']} batches, {stats['batch_fallbacks']} batch fallbacks, {stats['oom_retries']} "
                f"out-of-memory retries, {stats['at_token_limit']} cut off at the token limit, "
                f"{sum(1 for r in records if not r['ok'])} parse failures in this slice")
        stats["cases_per_hour"] = (round(stats["model_calls"] / stats["seconds"] * 3600)
                                   if stats["model_calls"] and stats["seconds"] else None)
    return out


def progress(checkpoint: Checkpoint, model_names: list[str], defences: list, cases: list[Case]) -> dict:
    """Answers still missing, per model and defence."""
    return {m: {d.id: len(checkpoint.todo(m, d.id, cases)) for d in defences} for m in model_names}


def summarise(checkpoint: Checkpoint, cases: list[Case], defences: list, descriptions: list[dict],
              benchmarks_meta: list[dict], run_name: str) -> dict | None:
    """The run summary over the models whose answers are all in; None until one is."""
    left = progress(checkpoint, [d["name"] for d in descriptions], defences, cases)
    complete = [d for d in descriptions if not any(left[d["name"]].values())]
    if not complete:
        return None
    names = {d["name"] for d in complete}
    ids = {d.id for d in defences}
    records = [r for r in checkpoint.records() if r["model"] in names and r["defence"] in ids]
    summary = build_summary(cases, records, defences, [Described(d) for d in complete], benchmarks_meta,
                            illustrative=False, run_name=run_name)
    summary["models_missing"] = [d["name"] for d in descriptions if d not in complete]
    if "quarantine" in ids:
        summary["quarantine_evidence"] = quarantine_evidence(cases)
    return summary
