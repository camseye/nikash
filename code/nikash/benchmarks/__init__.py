"""Benchmark loaders.

- `parsers`: raw log line → Event for Apache access, sshd, sudo, cron,
  generic syslog, and JSON-lines logs (Suricata, Wazuh).
- `camlds`: Benchmark B, from the extracted CAM-LDS manifestation archives.
- `loginject`: Benchmark A. `inspect()` reports the archive's shape; cases
  are built only from a mapping confirmed against that report.
- `fixtures`: synthetic smoke-test cases, never reported as results.
- `window`: the character budget every window is held to.

The held-out set (Benchmark C, OTRF) is written after the commit that
freezes the defences (G2 design §6).

Archives are downloaded and extracted inside the Kaggle notebook only.
"""
