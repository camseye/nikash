"""End-to-end smoke run: synthetic fixtures, simulated models, the version 1
scope (triage, five defences, no judge).

Standard library only; runs on a CPU in seconds. Its numbers exercise the
pipeline and are never results.

    python smoke.py                       # print the table, write nothing
    python smoke.py --out runs/smoke      # also write records.jsonl and summary.json
    python smoke.py --full                # all six defences, both tasks and the judge (outside the version 1 study)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nikash import __version__, scope
from nikash.benchmarks.fixtures import SMOKE_BENCHMARKS, build_smoke_cases
from nikash.defences import build_defences
from nikash.models import default_mock_models
from nikash.scorer.judge import KeywordJudge
from nikash.scorer.summary import build_summary
from nikash.run import run_matrix


def pct(x) -> str:
    return "  —  " if x is None else f"{x * 100:5.1f}%"


def print_table(summary: dict) -> None:
    models = [m["name"] for m in summary["models"]]
    for key, cell in summary["cells"].items():
        print(f"\n{key}   (model-averaged; attack success is twin-conditioned)")
        print(f"  {'defence':<12} {'attack':>7} {'raw':>7} {'cost':>7} {'parse':>7}")
        for d in summary["defences"]:
            rows = [cell.get(f"{d['id']}|{m}") for m in models]
            rows = [r for r in rows if r]

            def avg(field):
                vals = [r[field] for r in rows if r.get(field) is not None]
                return sum(vals) / len(vals) if vals else None

            print(f"  {d['id']:<12} {pct(avg('asr')):>7} {pct(avg('asr_raw')):>7} {pct(avg('cost')):>7} {pct(avg('parse_fail')):>7}")
    for task, pairs in summary["transfer"].items():
        for pair, t in pairs.items():
            print(f"\nRanking agreement ({task}) {pair}: tau = {t['tau']:.2f}, 95% interval {t['ci'][0]:.2f} to {t['ci'][1]:.2f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-family", type=int, default=10, help="malicious windows per family and task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, help="folder for records.jsonl and summary.json (nothing is written without it)")
    ap.add_argument("--full", action="store_true", help="also run the parts outside the version 1 study: summaries, the judge and the warning defence")
    args = ap.parse_args(argv)

    if args.full:
        cases = build_smoke_cases(per_family=args.per_family, seed=args.seed, tasks=("triage", "summary"))
        defences = build_defences(seed=args.seed, explain_triage=True)
        judge = KeywordJudge()
    else:
        cases = build_smoke_cases(per_family=args.per_family, seed=args.seed, tasks=scope.TASKS)
        defences = build_defences(list(scope.DEFENCES), seed=args.seed)
        judge = None
    models = default_mock_models(seed=args.seed)
    records = run_matrix(cases, defences, models, judge)
    meta = [{"id": k, **v, "licence": "MIT (synthetic)", "source": "nikash/benchmarks/fixtures.py"} for k, v in SMOKE_BENCHMARKS.items()]
    summary = build_summary(cases, records, defences, models, meta, illustrative=True, run_name="smoke")

    print(f"Nikash {__version__} smoke run: {len(cases)} cases × {len(defences)} defences × {len(models)} simulated models "
          f"= {len(records)} answers. Simulated numbers, not results.")
    print_table(summary)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "records.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"\nWrote {args.out / 'records.jsonl'} and {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
