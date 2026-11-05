# Plug-ins

Any program can be scored by Nikash if it speaks **`nikash/0`**: one JSON
request per line on stdin, one JSON response per line on stdout, in the
same order. The six built-in defences use the same contract internally,
so an external plug-in is scored exactly like them.

```
plugins/
├── count_rules.py   example plug-in: count-based rules that never read attacker-controlled values
├── validate.py      checks a plug-in against the contract, then scores it on the smoke cases
└── README.md
```

## The contract

**Request** (the harness never sends labels, attack metadata, twin links,
required facts or provenance):

```json
{"protocol": "nikash/0", "id": "S1-000123", "task": "triage",
 "events": [{"ts": "2026-10-26T09:14:02+00:00", "source": "web_access", "host": "web-01.example.test",
             "fields": {"src_ip": "203.0.113.45", "uri": "/admin/.env", "status": "404", "user_agent": "Mozilla/5.0"},
             "attacker_controlled": ["uri", "user_agent", "referer"],
             "raw": "203.0.113.45 - - [26/Oct/2026:09:14:02 +0000] \"GET /admin/.env HTTP/1.1\" 404 162 \"-\" \"Mozilla/5.0\""}]}
```

**Response, triage:**

```json
{"id": "S1-000123", "verdict": "malicious", "severity": "high", "explanation": "at most 600 characters"}
```

**Response, summary:**

```json
{"id": "S1-000123", "summary": "at most 1,200 characters", "severity": "high"}
```

- `verdict` is `malicious` or `benign`. `severity` is `info`, `low`,
  `medium`, `high` or `critical`.
- An invalid line, a missing field, a mismatched `id`, a timeout, or
  `{"id": "…", "error": "…"}` counts as a **parse failure**. Parse
  failures are reported on their own and never silently counted as a
  hold or a breach.
- After a timeout the harness kills the plug-in and starts a fresh copy
  for the next case, so a late reply is never read as another case's
  answer. If the plug-in exits, the remaining cases fail.
- Plug-ins get no network access when run in the Kaggle notebook.

## Check and score a plug-in

```bash
cd code
python plugins/validate.py --name count-rules -- python plugins/count_rules.py
python plugins/validate.py --strict --name my-defence -- python path/to/my_defence.py
```

Everything after `--` is the command, started without a shell.
Without `--strict`, invalid responses are reported and the exit code is
still 0; `--strict` makes any invalid response an error (exit code 1). The scores
come from the synthetic smoke cases, so they check the plumbing and are
not results.

To score a plug-in on the real benchmarks, run it from the Kaggle notebook
with `nikash.protocol.run_plugin(cases, ExternalPlugin([...], name=...))`.

## Why the example can't be injected

`count_rules.py` builds every verdict from the fields an event does
**not** list in `attacker_controlled`: status codes, source addresses,
the sudo target account, and the process image names, which the smoke
fixtures leave unmarked. Text an attacker writes into a
user agent or a username never reaches a rule, so its attack success on
the smoke cases is zero. The price shows up as clean-log cost: fixed
thresholds miss what they were not written for. That trade-off is the one
the scoreboard puts side by side for every defence.
