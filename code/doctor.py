"""Check that this copy of Nikash is complete and runnable. Read-only.

    python doctor.py            # core checks (standard library only)
    python doctor.py --gpu      # also report which GPU packages are importable

It never installs, downloads or writes anything. Each line is a check;
the summary at the end lists what failed and how to fix it.

Typical uses:
  * after `git clone` or `Replicate.ps1`, before the first smoke run;
  * at the top of the Kaggle notebook, before spending GPU time;
  * when a test fails because a sample file is missing (for example a
    `.gitignore` rule that dropped `*.log` files).
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
import unittest
from pathlib import Path

# Read-only includes __pycache__: set before any project module or test is imported.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Matched against file names only; these files are never opened.
SECRET_NAMES = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*", ".netrc", "_netrc", "kaggle.json",
    "credentials*", ".npmrc", ".pypirc", ".hf_token", "hf_token*", ".huggingface_token",
)
# Run outputs are large and private; the scan does not walk them.
SKIP_DIRS = (HERE / "results", HERE / "runs", HERE / "cache")

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = DIM = RESET = ""

_FAILED: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    mark = f"{GREEN}ok  {RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  {mark} {name}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    if not ok:
        _FAILED.append((name, fix))
    return ok


def section(title: str) -> None:
    print(f"\n{title}")


def _raise(exc: Exception):
    raise exc


def core_imports() -> None:
    section("1. Python and the harness")
    v = sys.version_info
    check("Python 3.11 or newer", v >= (3, 11), f"{platform.python_version()} on {platform.system()}",
          "Install Python 3.11+.")
    for mod in ("nikash", "nikash.schema", "nikash.protocol", "nikash.attacks", "nikash.defences",
                "nikash.models.mock_backend", "nikash.scorer.metrics", "nikash.scorer.stats",
                "nikash.benchmarks.camlds", "nikash.benchmarks.loginject", "nikash.benchmarks.parsers"):
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as exc:  # report every import problem, whatever its type
            check(f"import {mod}", False, f"{type(exc).__name__}: {exc}", "Run from the code/ folder of a complete copy.")


def files() -> None:
    section("2. Files a fresh copy must have")
    must = [
        "smoke.py", "build_dashboard.py", "requirements.txt", "CITATION.cff",
        "threat_model.json", "regulatory_map.json", "citations.json",
        "notebooks/cloud_run.ipynb", "plugins/count_rules.py", "plugins/validate.py",
        "docs/index.html", "docs/app.js", "docs/styles.css",
        "docs/data/scoreboard.json", "docs/data/examples.json",
    ]
    for rel in must:
        check(rel, (HERE / rel).is_file(), fix=f"Restore code/{rel} (git checkout, or re-run Replicate.ps1).")

    samples = HERE / "tests" / "samples"
    logs = sorted(samples.rglob("*.log")) if samples.is_dir() else []
    check("test sample logs present", len(logs) >= 7, f"{len(logs)} .log files under tests/samples",
          "A '*.log' ignore rule probably dropped them. Keep '!/code/tests/samples/**/*.log' in .gitignore.")
    li_data = samples / "loginject" / "artifact" / "data"
    li_files = [li_data / p for p in ("adversarial/samples.jsonl", "ground_truth/annotations.jsonl",
                                      "benign/apache_access.jsonl", "benign/ssh_auth.jsonl", "benign/json_api.jsonl")]
    missing = [str(p.relative_to(samples)) for p in li_files if not p.is_file()]
    check("LogInject samples present", not missing, ", ".join(missing) or f"{len(li_files)} files",
          "Restore code/tests/samples/loginject/artifact/data/.")
    # Counted as unittest runs them: a test defined once in a shared base class runs once per subclass.
    n_tests = unittest.defaultTestLoader.discover(str(HERE / "tests")).countTestCases()
    check("unit tests found", n_tests > 0, f"{n_tests} tests", "Restore code/tests/.")


def data_and_site() -> None:
    section("3. Scoreboard data and page security")
    try:
        board = json.loads((HERE / "docs/data/scoreboard.json").read_text(encoding="utf-8"),
                           parse_constant=lambda name: _raise(ValueError(f"{name} is not valid JSON")))
        ok = board.get("schema") == "nikash.scoreboard/v0" and "cells" in board
        flag = "real results" if board.get("illustrative") is False else "illustrative"
        if board.get("partial"):
            flag += ", partial"
        check("scoreboard.json schema", ok, f"{len(board.get('cells', {}))} cells, {flag}",
              "Regenerate with build_dashboard.py.")
    except (OSError, ValueError) as exc:
        check("scoreboard.json readable", False, str(exc), "Regenerate with build_dashboard.py.")
    for name in ("threat_model.json", "regulatory_map.json", "citations.json"):
        try:
            json.loads((HERE / name).read_text(encoding="utf-8"))
            check(f"{name} is valid JSON", True)
        except (OSError, json.JSONDecodeError) as exc:
            check(f"{name} is valid JSON", False, str(exc), f"Fix the syntax in code/{name}.")
    try:
        html = (HERE / "docs/index.html").read_text(encoding="utf-8")
        check("Content-Security-Policy meta present", 'http-equiv="Content-Security-Policy"' in html,
              fix="Restore the CSP meta in docs/index.html.")
        check("no inline style attributes", " style=" not in html, fix="Move inline styles into styles.css.")
        js = (HERE / "docs/app.js").read_text(encoding="utf-8")
        html_sinks = re.findall(r"\.(?:inner|outer)HTML\s*[+]?=|insertAdjacentHTML|document\.write", js)
        check("app.js never writes HTML strings", not html_sinks, ", ".join(sorted(set(html_sinks))),
              fix="Insert data with textContent or createTextNode only.")
    except OSError as exc:
        check("dashboard files readable", False, str(exc), "Restore code/docs/.")


def secret_files(root: Path, skip=SKIP_DIRS) -> list[Path]:
    """Files under root whose name looks like a key, token or credentials file."""
    skip = {Path(p).resolve() for p in skip}
    found = []
    for folder, dirs, names in os.walk(root):
        here = Path(folder)
        dirs[:] = [d for d in dirs if d != ".git" and (here / d).resolve() not in skip]
        found += [here / n for n in names if any(fnmatch.fnmatch(n.lower(), pat) for pat in SECRET_NAMES)]
    return sorted(found)


def secrets() -> None:
    section("4. No secret files in the tree (names only; nothing is opened)")
    bad = secret_files(ROOT)
    check("no token or key files", not bad, ", ".join(str(p.relative_to(ROOT)) for p in bad[:5]),
          "Delete them. The Hugging Face token lives in Kaggle Secrets only.")


def gpu() -> None:
    section("5. GPU packages (reported only; nothing is installed)")
    for mod in ("torch", "transformers", "accelerate", "bitsandbytes"):
        found = importlib.util.find_spec(mod) is not None
        print(f"  {'found  ' if found else 'missing'} {mod}")
    if importlib.util.find_spec("torch") is not None:
        try:
            torch = importlib.import_module("torch")
            print(f"  CUDA available: {torch.cuda.is_available()}")
        except Exception as exc:  # a broken torch install is reported, not raised
            print(f"  torch failed to import: {type(exc).__name__}")
    print(f"  {DIM}The GPU run belongs in notebooks/cloud_run.ipynb on Kaggle.{RESET}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", action="store_true", help="also report GPU packages")
    args = ap.parse_args(argv)
    sys.path.insert(0, str(HERE))
    print(f"Nikash doctor: {HERE}")
    core_imports()
    files()
    data_and_site()
    secrets()
    if args.gpu:
        gpu()
    print()
    if _FAILED:
        print(f"{RED}{len(_FAILED)} check(s) failed:{RESET}")
        for name, fix in _FAILED:
            print(f"  - {name}: {fix}")
        return 1
    print(f"{GREEN}All checks passed.{RESET} Next: python -m unittest discover -s tests, then python smoke.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
