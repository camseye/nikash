"""The full run's checkpoint file and resume, with simulated models on synthetic cases."""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_dashboard  # noqa: E402
from nikash import scope  # noqa: E402
from nikash.benchmarks.fixtures import SMOKE_BENCHMARKS, build_smoke_cases  # noqa: E402
from nikash.checkpoint import Checkpoint, case_stub, code_fingerprint, read_lines, run_fingerprints  # noqa: E402
from nikash.defences import build_defences  # noqa: E402
from nikash.fullrun import progress, run_model, summarise  # noqa: E402
from nikash.models import default_mock_models  # noqa: E402


def quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


class FullRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = build_smoke_cases(per_family=3, seed=0, tasks=scope.TASKS)
        cls.defences = build_defences(list(scope.DEFENCES), seed=0)
        cls.models = default_mock_models(seed=0)[:2]
        cls.settings = {"models": [m.name for m in cls.models], "generation": {"batch_size": 8}}
        cls.fp = run_fingerprints(cls.cases, cls.settings)
        cls.meta = [{"id": k, **v, "licence": "MIT (synthetic)", "source": "fixtures"} for k, v in SMOKE_BENCHMARKS.items()]

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def checkpoint(self, name, fingerprint=None):
        return Checkpoint(self.dir / name, fingerprint or self.fp, {"test": True})

    def run_all(self, ck, defences=None, **kw):
        for m in self.models:
            run_model(ck, m, self.cases, defences or self.defences, slice_size=7, log=lambda _m: None, **kw)

    @staticmethod
    def answers(ck):
        return sorted(json.dumps(r, sort_keys=True) for r in ck.records())

    def test_a_run_over_two_sessions_gives_the_answers_of_one_session(self):
        one = self.checkpoint("one.jsonl")
        self.run_all(one)
        first = self.checkpoint("s1.jsonl")
        self.run_all(first, defences=self.defences[:2])
        second = self.checkpoint("s2.jsonl")
        second.resume_from([self.dir / "s1.jsonl"])
        self.assertEqual(second.resumed[str(self.dir / "s1.jsonl")], len(first.records()))
        self.run_all(second)
        self.assertEqual(self.answers(second), self.answers(one))
        self.assertEqual(len(one.records()), len(self.cases) * len(self.defences) * len(self.models))
        # Everything is in the second session's own file, so its output alone is complete.
        self.assertEqual(self.answers(self.checkpoint("s2.jsonl")), self.answers(one))

    def test_a_line_cut_off_by_a_stopped_session_is_skipped(self):
        ck = self.checkpoint("cut.jsonl")
        run_model(ck, self.models[0], self.cases, self.defences[:1], slice_size=7, log=lambda _m: None)
        kept = len(ck.records())
        with open(self.dir / "cut.jsonl", "a", encoding="utf-8") as fh:
            fh.write('{"kind": "record", "case_id": "half a li')
        again = self.checkpoint("cut.jsonl")
        self.assertEqual(len(again.records()), kept)
        run_model(again, self.models[0], self.cases, self.defences[:2], slice_size=7, log=lambda _m: None)
        lines = read_lines(self.dir / "cut.jsonl")
        self.assertEqual(sum(1 for line in lines if line["kind"] == "record"), kept + len(self.cases))

    def test_answers_for_other_cases_are_never_mixed_in(self):
        other = self.checkpoint("other.jsonl", fingerprint={**self.fp, "cases": "0" * 64})
        self.run_all(other, defences=self.defences[:1])
        ck = self.checkpoint("mine.jsonl")
        ck.resume_from([self.dir / "other.jsonl"])
        self.assertEqual((ck.records(), ck.refused), ([], {str(self.dir / "other.jsonl"): "different cases"}))
        with self.assertRaisesRegex(ValueError, "different cases"):
            self.checkpoint("other.jsonl")

    def test_other_library_versions_are_resumed_and_reported(self):
        libs = {"torch": "2.8.0", "transformers": "5.5.0"}
        for name, theirs in (("same-libs.jsonl", libs), ("new-libs.jsonl", {**libs, "transformers": "5.6.0"})):
            earlier = Checkpoint(self.dir / name, self.fp, {"libraries": theirs})
            run_model(earlier, self.models[0], self.cases, self.defences[:1], slice_size=7, log=lambda _m: None)
        ck = Checkpoint(self.dir / "now.jsonl", self.fp, {"libraries": libs})
        ck.resume_from([self.dir / "same-libs.jsonl", self.dir / "new-libs.jsonl"])
        self.assertEqual((ck.refused, len(ck.records())), ({}, len(self.cases)))
        self.assertEqual(ck.changed("libraries", libs), {str(self.dir / "new-libs.jsonl"): {**libs, "transformers": "5.6.0"}})

    def package(self, name, files):
        root = self.dir / name
        for rel, text in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(text.encode("utf-8"))
        return root

    def test_answers_from_other_code_or_settings_are_never_mixed_in(self):
        files = {"a.py": "x = 1\n", "defences/b.py": "PROMPT = 'Decide.'\n"}
        before = run_fingerprints(self.cases, self.settings, self.package("v1", files))
        same = run_fingerprints(self.cases, self.settings, self.package("v1-again", files))
        changed = run_fingerprints(self.cases, self.settings,
                                   self.package("v2", {**files, "defences/b.py": "PROMPT = 'Decide now.'\n"}))
        other_settings = run_fingerprints(self.cases, {**self.settings, "generation": {"batch_size": 4}},
                                          self.package("v1-settings", files))
        self.assertEqual((before, before["cases"]), (same, changed["cases"]))
        earlier = self.checkpoint("earlier.jsonl", fingerprint=before)
        run_model(earlier, self.models[0], self.cases, self.defences[:1], slice_size=7, log=lambda _m: None)
        for name, fp in (("new-code.jsonl", changed), ("new-settings.jsonl", other_settings)):
            ck = self.checkpoint(name, fingerprint=fp)
            ck.resume_from([self.dir / "earlier.jsonl"])
            self.assertEqual((ck.records(), ck.refused),
                             ([], {str(self.dir / "earlier.jsonl"): "different code or generation settings"}))
        ck = self.checkpoint("same-code.jsonl", fingerprint=same)
        ck.resume_from([self.dir / "earlier.jsonl"])
        self.assertEqual((ck.refused, len(ck.records())), ({}, len(self.cases)))
        with self.assertRaisesRegex(ValueError, "different code"):
            self.checkpoint("earlier.jsonl", fingerprint=changed)

    def test_line_endings_do_not_change_the_code_fingerprint(self):
        lf = {"a.py": "x = 1\ny = 2\n", "sub/c.py": "z = 3\n"}
        crlf = {k: v.replace("\n", "\r\n") for k, v in lf.items()}
        self.assertEqual(code_fingerprint({}, self.package("lf", lf)), code_fingerprint({}, self.package("crlf", crlf)))
        self.assertNotEqual(code_fingerprint({}, self.package("lf", lf)),
                            code_fingerprint({}, self.package("moved", {"b.py": lf["a.py"], "sub/c.py": lf["sub/c.py"]})))
        self.assertEqual(code_fingerprint({"k": 1}), code_fingerprint({"k": 1}))

    def test_the_time_budget_stops_before_a_slice(self):
        ck = self.checkpoint("late.jsonl")
        out = run_model(ck, self.models[0], self.cases, self.defences, stop_at=time.time() - 1, log=lambda _m: None)
        self.assertFalse(out["complete"])
        self.assertEqual(ck.records(), [])

    def test_the_summary_covers_only_models_with_every_answer(self):
        ck = self.checkpoint("sum.jsonl")
        run_model(ck, self.models[0], self.cases, self.defences, slice_size=7, log=lambda _m: None)
        run_model(ck, self.models[1], self.cases, self.defences[:3], slice_size=7, log=lambda _m: None)
        left = progress(ck, [m.name for m in self.models], self.defences, self.cases)
        self.assertEqual(sum(left[self.models[0].name].values()), 0)
        self.assertEqual(left[self.models[1].name]["rules"], len(self.cases))
        summary = summarise(ck, self.cases, self.defences, [m.describe() for m in self.models], self.meta, "test")
        self.assertEqual([m["name"] for m in summary["models"]], [self.models[0].name])
        self.assertEqual(summary["models_missing"], [self.models[1].name])
        self.assertFalse(summary["illustrative"])
        quiet(ck.section, "summary", summary)
        out = self.dir / "board.json"
        with self.assertRaises(SystemExit):   # a partial run never reaches the scoreboard by accident
            quiet(build_dashboard.main, [str(self.dir / "sum.jsonl"), "--out", str(out)])
        self.assertFalse(out.exists())
        quiet(build_dashboard.main, [str(self.dir / "sum.jsonl"), "--out", str(out), "--partial"])
        board = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(board["models"]), 1)
        self.assertIn(self.models[1].name, board["status_note"])
        self.assertEqual(board["quarantine_evidence"], summary["quarantine_evidence"])
        self.assertEqual(sum(q["injected"] for q in board["quarantine_evidence"].values()),
                         sum(1 for c in self.cases if c.injected))

    def test_case_stubs_hold_no_log_text(self):
        stub = case_stub(next(c for c in self.cases if c.injected))
        self.assertEqual(set(stub), {"id", "benchmark", "task", "label", "injected", "twin_id", "reference_severity",
                                     "cluster", "incident", "level"})


if __name__ == "__main__":
    unittest.main()
