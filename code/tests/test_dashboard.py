"""build_dashboard.py turns a run summary into docs/data/scoreboard.json; doctor.py's secret-file scan."""

from __future__ import annotations

import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_dashboard  # noqa: E402
import doctor  # noqa: E402
import smoke  # noqa: E402

DEFENCES = [{"id": "none", "name": "No defence", "kind": "baseline", "short": "-"},
            {"id": "rules", "name": "Rules", "kind": "architecture", "short": "-"}]
MODELS = [{"id": "m1", "name": "Model A", "note": ""}, {"id": "m2", "name": "Model B", "note": ""}]


def metrics(asr, cost, n_eff=40.0, ci=(0.1, 0.3), **extra):
    return {"asr": asr, "cost": cost, "asr_n": 60, "asr_n_eff": n_eff, "asr_ci": list(ci),
            "n_injected": 64, "n_clean": 50, **extra}


def summary(**overrides):
    cell = {f"{d['id']}|{m['name']}": metrics(0.2 + 0.1 * i + 0.01 * j, 0.1, n_eff=40.0 - j)
            for i, d in enumerate(DEFENCES) for j, m in enumerate(MODELS)}
    out = {
        "run": "test", "illustrative": False, "tasks": ["triage"],
        "benchmarks": [{"id": "a", "short": "A", "name": "Bench A", "family": "f", "licence": "MIT", "source": "s"}],
        "defences": DEFENCES, "models": MODELS, "cells": {"a|triage": cell},
    }
    out.update(overrides)
    return out


def strict_loads(text):
    def refuse(name):
        raise ValueError(name)
    return json.loads(text, parse_constant=refuse)


class ConvertTests(unittest.TestCase):
    def test_interval_and_effective_size_matrices_run_parallel_to_asr(self):
        cell = build_dashboard.convert(summary())["cells"]["a|triage"]
        self.assertEqual(cell["asr"], [[0.2, 0.21], [0.3, 0.31]])
        self.assertEqual(cell["asr_ci"], [[[0.1, 0.3], [0.1, 0.3]], [[0.1, 0.3], [0.1, 0.3]]])
        self.assertEqual(cell["asr_n_eff"], [[40.0, 39.0], [40.0, 39.0]])
        self.assertNotIn("cost_parse_as_error", cell)

    def test_cost_parse_as_error_is_exported_when_the_summary_has_it(self):
        s = summary()
        for m in s["cells"]["a|triage"].values():
            m["cost_parse_as_error"] = 0.125
        self.assertEqual(build_dashboard.convert(s)["cells"]["a|triage"]["cost_parse_as_error"], [[0.125, 0.125]] * 2)

    def test_page_wording_overrides_the_defence_text_without_touching_the_summary(self):
        s = summary()
        board = build_dashboard.convert(s)
        self.assertEqual(board["defences"][1]["short"], build_dashboard.DEFENCE_TEXT["rules"]["short"])
        self.assertEqual((board["defences"][0]["short"], s["defences"][1]["short"]), ("-", "-"))

    def test_caveat_metrics_are_exported_and_stay_aligned_when_some_are_undefined(self):
        s = summary()
        for key, m in s["cells"]["a|triage"].items():
            m.update({"miss": 0.25, "false_alarm": None if key == "rules|Model B" else 0.5, "parse_fail": 0.0, "kept_share": 0.75})
        cell = build_dashboard.convert(s)["cells"]["a|triage"]
        self.assertEqual(cell["false_alarm"], [[0.5, 0.5], [0.5, None]])
        self.assertEqual((cell["miss"], cell["parse_fail"], cell["kept_share"]), ([[0.25] * 2] * 2, [[0.0] * 2] * 2, [[0.75] * 2] * 2))
        self.assertEqual(cell["asr_n"], [[60, 60], [60, 60]])
        self.assertNotIn("kept_ids", cell)

    def test_sample_sizes_name_the_smallest_raw_and_effective_counts(self):
        self.assertEqual(build_dashboard.convert(summary())["n"]["a"], {"adversarial": 39, "clean": 50, "injected": 64})

    def test_an_undefined_metric_is_null_and_does_not_stop_the_build(self):
        s = summary()
        s["cells"]["a|triage"]["rules|Model B"] = metrics(None, None, n_eff=0.0, ci=(0.0, 1.0))
        cell = build_dashboard.convert(s)["cells"]["a|triage"]
        self.assertEqual((cell["asr"][1][1], cell["cost"][1][1], cell["asr_ci"][1][1], cell["asr_n_eff"][1][1]),
                         (None, None, None, None))
        self.assertEqual(build_dashboard.convert(s)["n"]["a"]["adversarial"], 39)

    def test_nan_becomes_null_and_the_output_is_strict_json(self):
        s = summary()
        s["cells"]["a|triage"]["none|Model A"] = metrics(float("nan"), float("inf"), ci=(float("nan"), 0.3))
        s["benchmarks"][0]["extra_score"] = float("nan")
        board = strict_loads(build_dashboard.to_json(build_dashboard.convert(s)))
        cell = board["cells"]["a|triage"]
        self.assertEqual((cell["asr"][0][0], cell["cost"][0][0], cell["asr_ci"][0][0]), (None, None, None))

    def test_a_missing_defence_model_entry_or_cell_fails_loudly(self):
        s = summary()
        del s["cells"]["a|triage"]["rules|Model A"]
        with self.assertRaises(SystemExit) as ctx:
            build_dashboard.convert(s)
        self.assertIn("rules|Model A", str(ctx.exception))
        with self.assertRaises(SystemExit) as ctx:
            build_dashboard.convert(summary(cells={}))
        self.assertIn("a|triage", str(ctx.exception))

    def test_complete_runs_are_not_partial(self):
        board = build_dashboard.convert(summary())
        self.assertEqual((board["partial"], board["models_missing"], board["illustrative"]), (False, [], False))

    def test_a_missing_illustrative_flag_counts_as_illustrative(self):
        s = summary()
        del s["illustrative"]
        self.assertTrue(build_dashboard.convert(s)["illustrative"])


class MainTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, True)

    def build(self, s, *flags):
        path = self.dir / "summary.json"
        path.write_text(json.dumps(copy.deepcopy(s)), encoding="utf-8")
        out = self.dir / "board.json"
        with redirect_stdout(io.StringIO()):
            build_dashboard.main([str(path), "--out", str(out), *flags])
        return strict_loads(out.read_text(encoding="utf-8"))

    def test_partial_boards_are_marked_and_name_the_missing_models(self):
        s = summary(models_missing=["Model C"])
        with self.assertRaises(SystemExit):
            self.build(s)
        board = self.build(s, "--partial")
        self.assertEqual((board["partial"], board["models_missing"]), (True, ["Model C"]))
        self.assertIn("Model C", board["status_note"])
        self.assertFalse(board["illustrative"])

    def test_a_smoke_run_summary_converts_to_a_well_shaped_board(self):
        with redirect_stdout(io.StringIO()):
            smoke.main(["--per-family", "3", "--out", str(self.dir)])
        board = self.build(json.loads((self.dir / "summary.json").read_text(encoding="utf-8")))
        n_def, n_mod = len(board["defences"]), len(board["models"])
        self.assertTrue(board["cells"])
        for cell in board["cells"].values():
            for name in ("asr", "cost", "asr_ci", "asr_n_eff"):
                self.assertEqual([len(row) for row in cell[name]], [n_mod] * n_def, name)
            for row, ci_row in zip(cell["asr"], cell["asr_ci"]):
                for asr, ci in zip(row, ci_row):
                    if asr is not None and ci is not None:
                        self.assertLessEqual(ci[0], asr + 1e-4)
                        self.assertGreaterEqual(ci[1], asr - 1e-4)


class DoctorSecretScanTests(unittest.TestCase):
    def test_secret_names_are_found_and_run_folders_are_skipped(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        for rel in (".env.local", "keys/id_ed25519", "deploy/site.p12", "kaggle.json", "a/.npmrc", "hf_token.txt",
                    "runs/.env", "notes.md", "app.py"):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("x", encoding="utf-8")
        found = {p.relative_to(root).as_posix() for p in doctor.secret_files(root, skip=[root / "runs"])}
        self.assertEqual(found, {".env.local", "keys/id_ed25519", "deploy/site.p12", "kaggle.json", "a/.npmrc", "hf_token.txt"})


if __name__ == "__main__":
    unittest.main()
