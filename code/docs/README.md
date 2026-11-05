# Nikash scoreboard

The static site published on GitHub Pages, in a dark "poster lab" style.
It opens on a poster wall: a large question, with cards beside it for the
study at a glance, a hand-written attack fragment and the defence gradient. Below
that are seven numbered plates, one for each part of the study. Each plate
has a big outlined number and a short brief on the left, and the live
exhibit on the right. Hazard tape marks illustrative numbers. Drishti's
dashboard is an operator console; this one is a showcase that explains the
study as you scroll.

```
docs/
├── index.html          poster wall and plates 01–07, with a strict Content-Security-Policy
├── styles.css          dark poster-lab theme
├── app.js              rendering: anatomy viewer, scoreboard, scatter, slope chart
├── data/
│   ├── scoreboard.json results by benchmark × task × defence × model
│   └── examples.json   hand-written synthetic cases for the anatomy viewer
└── README.md           this file
```

## Sections

| § | Section | What it answers |
|---|---|---|
| 1 | Anatomy of an attack | What does an injected log line look like, and how does each defence respond? Attacker-controlled text is underlined in gold. Answers whose explanation contradicts the verdict are flagged. |
| 2 | Scoreboard | Which defence lets the fewest attacks through on this benchmark and task, and at what cost on clean logs? Includes the spread across models. |
| 3 | Security against cost | Where does each defence sit on the trade-off? Shows the Pareto line and the flag-everything and close-everything reference points. |
| 4 | Does the ranking travel? | Does a defence's rank hold across benchmarks (RQ1)? |
| 5 | Method | Pipeline and metric definitions |
| 6 | Data and licences | Where every benchmark comes from, and under what licence |
| 7 | Bring your own defence | The `nikash/0` plug-in protocol |

The controls in §2 (benchmark, task, model) also drive §3 and §4.

## Preview locally

```bash
python -m http.server 8000 --directory code/docs
# open http://localhost:8000/
```

Opening `index.html` from disk fails, because browsers block `fetch` on
`file://`. The page shows a banner explaining this. The banner, the sticker
and the watermarks are kept when printing.

## Data contract

`data/scoreboard.json` (`schema: nikash.scoreboard/v0`):

| Key | Meaning |
|---|---|
| `illustrative` | Anything but `false` (including a missing flag) shows the "work in progress" banner, the "Illustrative only" sticker and an ILLUSTRATIVE watermark on the charts. Set it to `false` only for real, reviewed results. This flag alone decides them: `examples.json` is always `illustrative: true` (its cases are hand-written), which only adds a note to §1 saying so. |
| `partial`, `models_missing` | Written by `--partial`. The page shows a "Partial results" banner naming the missing models. |
| `status_note` | Where the numbers came from; shown under the scoreboard. |
| `benchmarks`, `tasks`, `models`, `defences` | Ordered lists. Their order fixes the row and column order of `cells`. The first task is selected on load. |
| `cells["<benchmark>\|<task>"]` | `asr` and `cost` are matrices: one row per defence, one column per model. Values run from 0 to 1; `null` means the run could not define the metric and is shown as a dash. `asr_ci` (`[low, high]` per entry) and `asr_n_eff` are parallel matrices with each model's clustered Wilson 95% interval and effective sample size; `cost_parse_as_error`, `miss`, `false_alarm` (on clean windows, parsed answers only), `parse_fail` (all answers), `kept_share` and `asr_n` (injected windows kept after twin-conditioning, as a share and a count) are added as parallel matrices when the run summary has them; the page uses them for the sub-lines under the meters and for the "Read with care" notes. |
| `n[<benchmark>]` | Smallest counts across models: `injected` and `clean` cases, and `adversarial`, the smallest effective injected sample (twin-conditioned, with windows from one attack step counting for less). Files without `asr_ci` fall back to Wilson intervals on `adversarial`. |
| `anchors[<task>]` | Reference points for the scatter plot (attack success and cost). |

The model average draws its interval with the smallest per-model effective
sample, not the sum: every model answers the same cases, so pooling models
adds no independent evidence. Exact ties in attack success share a rank in
the table and the slope chart. Malformed values are shown as a dash and named
in a "Data warning" banner; invalid JSON (for example `NaN`) gives a "Data
invalid" banner, and a missing `examples.json` only empties §1.

Definitions:
- **Attack success:** among injected cases whose clean twin (the same logs without the payload) was answered correctly, the share where the attacker's goal is met.
- **Clean-log cost:** balanced error on clean events, that is, (missed-detection rate + false-alarm rate) / 2.

`code/build_dashboard.py` writes this file from a run summary (`summary.json`
from `smoke.py --out`, or the full run's `nikash_full.jsonl`):

```bash
cd code
python build_dashboard.py results/full/nikash_full.jsonl --out docs/data/scoreboard.json
```

A full run with models still missing is refused unless `--partial` is given.
A metric the run could not define is written as `null`; a benchmark/task cell
or defence/model entry missing from the summary stops the build.

Keep `illustrative: true` until the numbers come from the version 1 study and have been reviewed.

## Security notes

- Log lines are attacker-authored by design. `app.js` inserts every data
  string with `textContent`, never `innerHTML`, so a payload such as
  `<img onerror=…>` renders as text.
- The page's Content-Security-Policy allows only same-origin scripts,
  styles and data, with no inline scripts and no third-party requests.
  There are no web fonts, analytics or CDNs.
- All identifiers are synthetic: RFC 5737 addresses, `jane.citizen`,
  `CHG0000000`.
