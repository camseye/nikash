import subprocess
import sys
import unittest
from pathlib import Path

from nikash.benchmarks.fixtures import build_smoke_cases
from nikash.protocol import ExternalPlugin, run_plugin
from nikash.scorer.metrics import aggregate

CODE = Path(__file__).resolve().parent.parent
EXAMPLE = CODE / "plugins" / "count_rules.py"


class ExamplePluginTest(unittest.TestCase):
    def test_every_response_is_valid(self):
        cases = build_smoke_cases(per_family=3, seed=1)
        records = run_plugin(cases, ExternalPlugin([sys.executable, "-B", str(EXAMPLE)], name="count-rules", timeout=20))
        self.assertEqual(len(records), len(cases))
        self.assertTrue(all(r["ok"] for r in records), [r["error"] for r in records if not r["ok"]])

    def test_injection_cannot_reach_the_rules(self):
        cases = [c for c in build_smoke_cases(per_family=3, seed=2) if c.task == "triage"]
        records = run_plugin(cases, ExternalPlugin([sys.executable, "-B", str(EXAMPLE)], name="count-rules", timeout=20))
        cells = aggregate({c.id: c for c in cases}, records)
        for key, cell in cells.items():
            self.assertEqual(cell["count-rules|external"]["asr"], 0.0, key)

    def test_validator_passes_in_strict_mode(self):
        out = subprocess.run(
            [sys.executable, "-B", str(CODE / "plugins" / "validate.py"), "--strict", "--per-family", "2", "--",
             sys.executable, "-B", str(EXAMPLE)],
            cwd=CODE, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn("valid responses", out.stdout)

    def test_validator_exit_code_matches_its_docstring(self):
        broken = [sys.executable, "-B", "-c", "import sys\nfor line in sys.stdin:\n    print('not json', flush=True)\n"]

        def validate(*flags):
            return subprocess.run(
                [sys.executable, "-B", str(CODE / "plugins" / "validate.py"), *flags, "--per-family", "1", "--", *broken],
                cwd=CODE, capture_output=True, text=True, timeout=120,
            )

        lenient, strict = validate(), validate("--strict")
        self.assertEqual(lenient.returncode, 0, lenient.stdout + lenient.stderr)
        self.assertIn("exit code 0 because --strict was not given", lenient.stdout)
        self.assertEqual(strict.returncode, 1, strict.stdout + strict.stderr)
        self.assertIn("scored", strict.stdout)

    def test_example_counts_an_ssh_attempt_logged_twice_once(self):
        sys.path.insert(0, str(EXAMPLE.parent))
        self.addCleanup(sys.path.remove, str(EXAMPLE.parent))
        import count_rules

        def ssh(kind, port):
            return {"source": "sshd", "host": "files-01.example.test", "attacker_controlled": ["username"],
                    "fields": {"event": kind, "username": "admin", "src_ip": "198.51.100.23", "port": str(port)}}

        twice = [ssh(kind, 40100 + k) for k in range(6) for kind in ("invalid_user", "failed")]
        self.assertEqual(count_rules.findings(twice), [])
        once = [ssh("invalid_user", 40100 + k) for k in range(10)]
        self.assertEqual(len(count_rules.findings(once)), 1)


if __name__ == "__main__":
    unittest.main()
