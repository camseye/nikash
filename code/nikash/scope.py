"""Version 1 study scope (G2 design, amendments of 2026-10-29).

The harness still implements the summary task, the `warning` defence and the
judge; version 2 brings them back by changing these constants, and adds the
held-out benchmark C.
"""

BENCHMARKS = ("loginject", "camlds")
TASKS = ("triage",)
DEFENCES = ("none", "spotlight", "quarantine", "extract", "rules")
MODELS = 3            # three of the four G2 §7 candidates, chosen at G3
PER_BENCHMARK = 200
