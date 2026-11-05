"""Check a plug-in against the nikash/0 contract, then score it on the smoke cases.

    python plugins/validate.py -- python plugins/count_rules.py
    python plugins/validate.py --strict --per-family 4 -- python my_defence.py

Everything after `--` is the plug-in command. It is started without a
shell and receives only what `Case.to_request()` sends: no labels, no
attack metadata. The smoke cases are synthetic, so the scores check the
plumbing and are not results.

Without --strict, invalid responses are reported but the exit code is 0
whenever the plug-in could be run. With --strict, exit code 0 means every
request got a valid response, and any parse failure or timeout gives
exit code 1. Usage errors give exit code 2.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nikash.benchmarks.fixtures import build_smoke_cases  # noqa: E402
from nikash.protocol import ExternalPlugin, run_plugin  # noqa: E402
from nikash.run import attach_judgements  # noqa: E402
from nikash.scorer.judge import KeywordJudge  # noqa: E402
from nikash.scorer.metrics import aggregate  # noqa: E402


def pct(x) -> str:
    return "   —  " if x is None else f"{x * 100:5.1f}%"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        print("usage: validate.py [options] -- <plug-in command ...>", file=sys.stderr)
        return 2
    split = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="my-plugin", help="name shown in the report")
    ap.add_argument("--per-family", type=int, default=6, help="malicious windows per family and task")
    ap.add_argument("--timeout", type=float, default=30.0, help="seconds per case")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true", help="fail on any parse failure or timeout")
    args = ap.parse_args(argv[:split])
    command = argv[split + 1:]
    if not command:
        print("no plug-in command after --", file=sys.stderr)
        return 2

    cases = build_smoke_cases(per_family=args.per_family, seed=args.seed)
    plugin = ExternalPlugin(command, name=args.name, timeout=args.timeout)
    started = time.perf_counter()
    records = run_plugin(cases, plugin)
    elapsed = time.perf_counter() - started

    errors = Counter(r["error"] for r in records if not r["ok"])
    ok = sum(1 for r in records if r["ok"])
    print(f"Plug-in '{args.name}': {ok}/{len(records)} valid responses in {elapsed:.1f} s "
          f"({elapsed / max(1, len(records)) * 1000:.0f} ms per case).")
    for error, n in errors.most_common():
        print(f"  {n:4d} × {error}")

    attach_judgements(cases, records, KeywordJudge(), {args.name: ExternalPlugin.model_decides})
    cells = aggregate({c.id: c for c in cases}, records)
    print("\nSmoke scores (synthetic cases; attack success is twin-conditioned):")
    print(f"  {'benchmark|task':<24} {'attack':>7} {'cost':>7} {'parse':>7}  {'kept':>7}  scored")
    for key in sorted(cells):
        m = cells[key][f"{args.name}|external"]
        kept = f"{len(m['kept_ids'])}/{m['n_injected']}"
        print(f"  {key:<24} {pct(m['asr']):>7} {pct(m['cost']):>7} {pct(m['parse_fail']):>7}  {kept:>7}  {m['asr_n']}")
    print("\n'kept' counts injected cases whose clean twin was answered correctly; 'scored' counts the kept cases"
          "\nwhose own answer parsed, and attack success is computed over those.")

    if errors and args.strict:
        print("\nFAILED: the plug-in produced invalid responses (see above).")
        return 1
    if errors:
        print(f"\nNot every request got a valid response ({ok}/{len(records)}); exit code 0 because --strict was not given.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
