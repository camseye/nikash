"""Convert a run summary (smoke.py --out, or the Kaggle notebook) into the
scoreboard data file read by docs/app.js.

    python build_dashboard.py runs/smoke/summary.json
    python build_dashboard.py results/pilot/summary.json --out docs/data/scoreboard.json
    python build_dashboard.py results/full/nikash_full.jsonl --out docs/data/scoreboard.json   # the full run's file

Without --out the scoreboard JSON is printed, so nothing is overwritten by accident.
A full run with models still missing is refused unless --partial is given; the
scoreboard is then marked partial and names the missing models.

A metric the run could not define (for example attack success when no clean
twin was answered correctly) is written as null and shown as a dash. A whole
benchmark/task cell or defence/model entry missing from the summary is an error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

ANCHORS = {"triage": [
    {"id": "block_all", "name": "Flag everything", "asr": 0.0, "cost": 0.5},
    {"id": "allow_all", "name": "Close everything", "asr": 1.0, "cost": 0.5},
]}

TASK_META = {
    "triage": {"id": "triage", "name": "Alert triage", "attack_goal": "a malicious event is closed as benign or downgraded"},
    "summary": {"id": "summary", "name": "Incident summary", "attack_goal": "the summary omits or misstates the targeted fact"},
}

# Page wording that replaces the defence's own text, which lives under nikash/ and is covered by the
# version 1 code fingerprint. With explain_triage=False the rules defence never calls the model in triage.
DEFENCE_TEXT = {
    "rules": {"short": "Detection rules decide the verdict and give the reasons; in version 1 the model is not called for triage."},
}

# Exported as a matrix when the summary has them; older summaries simply lack the key.
OPTIONAL_METRICS = ("cost_parse_as_error", "miss", "false_alarm", "parse_fail", "kept_share", "asr_n")


def _num(x, digits: int = 4):
    """A finite number rounded for the page, or None for undefined metrics and NaN."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        return None
    return round(x, digits)


def _ci(metrics: dict):
    ci = metrics.get("asr_ci")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return None
    lo, hi = _num(ci[0]), _num(ci[1])
    return [lo, hi] if lo is not None and hi is not None and lo <= hi else None


def _no_nan(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _no_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_no_nan(v) for v in obj]
    return obj


def _smaller(current, value):
    return value if current is None else min(current, value)


def convert(summary: dict) -> dict:
    defences = summary["defences"]
    models = summary["models"]
    missing = list(summary.get("models_missing") or [])
    cells_out, n_out = {}, {}

    for bench in summary["benchmarks"]:
        smallest = smallest_injected = smallest_clean = None
        for task in summary["tasks"]:
            key = f"{bench['id']}|{task}"
            cell = summary["cells"].get(key)
            if not cell:
                raise SystemExit(f"missing cell {key} in the summary")
            out = {k: [] for k in ("asr", "cost", "asr_ci", "asr_n_eff", *OPTIONAL_METRICS)}
            present = set()
            for d in defences:
                rows = {k: [] for k in out}
                for m in models:
                    metrics = cell.get(f"{d['id']}|{m['name']}")
                    if not metrics:
                        raise SystemExit(f"missing metrics for {key} {d['id']}|{m['name']}")
                    asr = _num(metrics.get("asr"))
                    # Effective size: windows from one attack step count for less, so intervals are wider.
                    n_eff = _num(metrics.get("asr_n_eff", metrics.get("asr_n")), 1) if asr is not None else None
                    rows["asr"].append(asr)
                    rows["cost"].append(_num(metrics.get("cost")))
                    rows["asr_ci"].append(_ci(metrics) if asr is not None else None)
                    rows["asr_n_eff"].append(n_eff)
                    for k in OPTIONAL_METRICS:
                        if k in metrics:
                            present.add(k)
                        rows[k].append(_num(metrics.get(k)))
                    if n_eff is not None:
                        smallest = _smaller(smallest, int(n_eff))
                    if _num(metrics.get("n_injected")) is not None:
                        smallest_injected = _smaller(smallest_injected, int(metrics["n_injected"]))
                    if _num(metrics.get("n_clean")) is not None:
                        smallest_clean = _smaller(smallest_clean, int(metrics["n_clean"]))
                for k in out:
                    out[k].append(rows[k])
            for k in OPTIONAL_METRICS:
                if k not in present:
                    del out[k]
            cells_out[key] = out
        n_out[bench["id"]] = {"adversarial": smallest or 0, "clean": smallest_clean or 0}
        if smallest_injected is not None:
            n_out[bench["id"]]["injected"] = smallest_injected

    short = {m["id"]: f"Model {chr(65 + i)}" for i, m in enumerate(models)}
    note = f"Generated from run '{summary.get('run')}'."
    if missing:
        note += f" Partial: {', '.join(missing)} not run yet."
    extra = {"quarantine_evidence": summary["quarantine_evidence"]} if "quarantine_evidence" in summary else {}
    return {
        "schema": "nikash.scoreboard/v0",
        "illustrative": bool(summary.get("illustrative", True)),
        "partial": bool(missing),
        "models_missing": missing,
        "status_note": note,
        "updated": date.today().isoformat(),
        "benchmarks": [{k: b.get(k) for k in ("id", "short", "name", "family", "licence", "source")} for b in summary["benchmarks"]],
        "tasks": [TASK_META[t] for t in summary["tasks"]],
        "models": [{"id": m["id"], "name": m.get("name", short[m["id"]]), "note": m.get("note", "")} for m in models],
        "defences": [{**d, **DEFENCE_TEXT.get(d["id"], {})} for d in defences],
        "anchors": {t: ANCHORS[t] for t in summary["tasks"] if t in ANCHORS},
        "n": n_out,
        "cells": cells_out,
        **extra,
    }


def to_json(board: dict) -> str:
    return json.dumps(_no_nan(board), indent=1, allow_nan=False)


def load_summary(path: Path) -> dict:
    """A summary JSON, or the last summary section of a full run's .jsonl file."""
    if path.suffix != ".jsonl":
        return json.loads(path.read_text(encoding="utf-8"))
    from nikash.checkpoint import read_lines
    found = [line["data"] for line in read_lines(path) if line["kind"] == "section" and line["name"] == "summary"]
    if not found:
        raise SystemExit(f"{path} has no summary yet; no model has all its answers")
    return found[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out", type=Path, help="where to write the scoreboard JSON (e.g. docs/data/scoreboard.json)")
    ap.add_argument("--partial", action="store_true", help="accept a full run with models still missing")
    args = ap.parse_args(argv)

    summary = load_summary(args.summary)
    missing = summary.get("models_missing") or []
    if missing and not args.partial:
        raise SystemExit(f"{args.summary}: models still missing ({', '.join(missing)}); finish the run, "
                         "or pass --partial for a preview")
    text = to_json(convert(summary))
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
        board = json.loads(text)
        print(f"Wrote {args.out} (illustrative={board['illustrative']}, partial={board['partial']})")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
