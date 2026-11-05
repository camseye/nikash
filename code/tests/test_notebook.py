"""Checks on the Kaggle notebook that need no Kaggle: its cells parse, and its commands work on Kaggle's image."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "cloud_run.ipynb"


def code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


class NotebookTests(unittest.TestCase):
    def test_every_code_cell_parses(self):
        for i, src in enumerate(code_cells()):
            ast.parse(src, filename=f"cell {i}")

    def test_unit_tests_are_run_by_file_not_as_a_tests_package(self):
        # Kaggle's image has another top-level package called `tests`; `-m unittest tests.x` imports that one.
        calls = [line for src in code_cells() for line in src.splitlines() if '"unittest"' in line]
        self.assertGreaterEqual(len(calls), 2)
        for line in calls:
            self.assertIn('"discover"', line)
            self.assertIn('"-s", "tests"', line)
            self.assertNotRegex(line, r'"tests\.\w+"')

    def test_the_start_check_runs_before_any_download(self):
        cells = code_cells()
        start = next(i for i, src in enumerate(cells) if "NIKASH_REQUIRE_PREFILL_TESTS" in src)
        downloads = [i for i, src in enumerate(cells) if re.search(r"\bfetch\(\d+\)", src)]
        self.assertTrue(downloads)
        self.assertLess(start, min(downloads))
        self.assertIn('"-p", "test_prefill.py"', cells[start])


if __name__ == "__main__":
    unittest.main()
