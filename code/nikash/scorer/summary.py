"""A run summary (nikash.run-summary/v0): per-cell metrics and ranking agreement, read by build_dashboard.py."""

from __future__ import annotations

from .. import __version__
from .metrics import aggregate, transfer


def build_summary(cases, records, defences, models, benchmarks_meta, *, illustrative: bool, run_name: str) -> dict:
    by_id = {c.id: c for c in cases}
    cells = aggregate(by_id, records)
    model_names = [m.name for m in models]
    defence_ids = [d.id for d in defences]
    tasks = sorted({c.task for c in cases}, key=("triage", "summary").index)
    return {
        "schema": "nikash.run-summary/v0",
        "nikash_version": __version__,
        "run": run_name,
        "illustrative": illustrative,
        "benchmarks": benchmarks_meta,
        "tasks": tasks,
        "defences": [d.describe() for d in defences],
        "models": [m.describe() for m in models],
        "cells": cells,
        "transfer": {t: transfer(by_id, records, cells, t, defence_ids, model_names, n_boot=500) for t in tasks},
    }
