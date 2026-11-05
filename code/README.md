# Nikash harness

```
code/
├── nikash/
│   ├── schema.py              case and event records; nikash/0 request and response checks
│   ├── attacks/               direct instruction · context stitching · instruction-free cover stories; clean twins
│   ├── defences/              prompt_level.py (none, warning, spotlight, quarantine) · model_free.py (extract, rules) · checks.py
│   ├── models/                mock_backend.py (CPU, simulated) · hf_backend.py (Kaggle GPU, 4-bit, greedy)
│   ├── scorer/                metrics.py · stats.py · summary.py (run summary: per-cell metrics, ranking agreement) · judge.py (summaries and explanations; not used in the version 1 study)
│   ├── benchmarks/            parsers.py, camlds.py (B), loginject.py (A: composed windows with clean twins); fixtures.py (synthetic, smoke only); window.py (10,000-character window cap)
│   ├── scope.py               version 1 study scope: benchmarks A and B, triage, five defences, three models, 200 cases per cell
│   ├── run.py                 every model × defence over a case set, then judging if a judge is given
│   ├── fullrun.py             the notebook's full run: every model × defence × case, in slices, asking only missing answers
│   ├── checkpoint.py          nikash_full.jsonl: run line, answer records, named sections; resume checks
│   └── protocol.py            external plug-ins over JSON Lines (stdin/stdout, no shell)
├── plugins/                   count_rules.py (example external defence) · validate.py (checks any plug-in against the protocol)
├── smoke.py                   end-to-end run on synthetic data with simulated models
├── doctor.py                  read-only health check of a checkout
├── build_dashboard.py         run summary or nikash_full.jsonl → docs/data/scoreboard.json
├── notebooks/cloud_run.ipynb  Kaggle T4 × 2: the full version 1 run by default (MODE "full"), or the pilot
├── tests/                     unit tests (standard library only); samples/ holds small synthetic files in each benchmark's layout
├── threat_model.json          attack classes mapped to STRIDE, OWASP LLM Top 10, MITRE ATLAS and ATT&CK analogues
├── regulatory_map.json        what the EU AI Act, NIST AI RMF, NIS2 and ISO/IEC 42001 ask, and what Nikash does and doesn't provide
├── citations.json             every reference, with its role, licence and verification status
├── CITATION.cff               how to cite the software
├── requirements.txt           GPU-only packages; the core needs none
└── docs/                      the scoreboard site
```

## Run locally (no installs)

```bash
cd code
python doctor.py                         # checks Python, files, scoreboard data and the site's security rules
python -m unittest discover -s tests     # standard library only; see the note below on skipped tests
python smoke.py                          # version 1 scope; prints the table, writes nothing
python smoke.py --full                   # also summaries, the judge and the warning defence (outside the version 1 study)
python smoke.py --out runs/smoke         # writes records.jsonl and summary.json (gitignored)
python build_dashboard.py runs/smoke/summary.json    # prints scoreboard JSON; add --out to write it
```

The smoke run uses **simulated models that read the ground truth** to fake
a fallible assistant. Its numbers check the plumbing and are never results.

The GPU-batching tests in `tests/test_prefill.py` run tiny models on the
CPU, so they are skipped where torch and transformers are not installed,
and the Gemma 2 ones also where transformers is older than 4.53. The
notebook runs them again with skipping turned off before any model is
downloaded.

`doctor.py --gpu` also reports whether the GPU packages are importable
(they only need to be, on Kaggle).

## Run on Kaggle

1. **Kaggle:** verify your phone number (Settings → Phone verification);
   without it there is no GPU or internet in notebooks.
2. **Hugging Face:** accept the licences of
   `meta-llama/Llama-3.1-8B-Instruct` (reviewed by Meta) and
   `google/gemma-2-9b-it` on their model pages, then create a **Read**
   access token (Settings → Access Tokens).
3. **Notebook:** Code → New Notebook → File → Import Notebook →
   `notebooks/cloud_run.ipynb`. Session options: Accelerator **GPU T4
   x2**, Internet **on**.
4. **Secret:** Add-ons → Secrets → add `HF_TOKEN` with the token, and
   attach it to the notebook. The notebook never prints or saves it.
5. **Code:** set `REPO_URL` in the settings cell to your clone's address,
   or attach this folder as a private dataset and set `REPO_URL = ""`.
6. **Run:** Save Version → **Save & Run All (Commit)**. Keep the notebook
   private: its output holds benchmark payload text.

The full run (`MODE = "full"`) saves every answer as it goes to one file,
`nikash_full.jsonl`. A session stops cleanly before Kaggle's time limit;
attach that version's output to the next session (Add Input → this
notebook, latest version) and run again, and only the missing answers are
asked. Output from other code, settings or cases is refused rather than
mixed in. When the last cell says every model is complete, put the file in
`results/full/` and run
`python build_dashboard.py results/full/nikash_full.jsonl --out docs/data/scoreboard.json`.
The pilot (`MODE = "pilot"`) writes `nikash_pilot.json` instead.

## Write your own plug-in

Read one JSON request per line on stdin and write one JSON response per
line on stdout (see the scoreboard's §7 and `nikash/schema.py`). Check it
against the protocol on the synthetic smoke cases:

```bash
python plugins/validate.py --strict -- python my_defence.py
```

Score it from Python:

```python
from nikash.protocol import ExternalPlugin, run_plugin
records = run_plugin(cases, ExternalPlugin(["python", "my_defence.py"], name="my-defence"))
```

`plugins/README.md` walks through the contract with a working example.

## Threat model, standards and citations

`threat_model.json`, `regulatory_map.json` and `citations.json` are the
single source for those topics. Each entry records how confident the
mapping is, and the framework IDs are marked for re-checking against the
live pages before publication. `regulatory_map.json` describes what each
text asks for and where Nikash helps; it is not legal advice.

## Cite

See `CITATION.cff`. A paper is in preparation.
