# Chinese length-soft rejudgement: 81 answers

Authorized scope: repair Chinese target-length extraction and use hard bounds
of 50% and 150%, keeping soft boundaries at 75% and 125%. Rejudge only the 81
Chinese nonfailed answers that enter the soft branch under those conditions.
There are 39 answers from reports1 and 42 from reports2, evaluated by each of
the five original judges: 405 successful judgements in total.

The original scripts, saved Penalty CSVs and official Step1–5 outputs remain
read-only inputs. Everything produced by this runner stays in this directory.

The runner compiles the original pure prompt-building and response-parsing
functions using Python AST. This preserves the original 2,000-character
response truncation, soft note, JSON parsing and sampling settings without
importing the original batch program. Thirteen selected answers are truncated.
It avoids the original program's missing pandas import and unrelated batch
discovery/recovery behavior. Only Python's standard library is needed.

Run from the project root:

```bash
python3 -B length_rejudge_20261026/run_rejudge.py prepare
python3 -B length_rejudge_20261026/run_rejudge.py check --base-url http://100.116.124.0:11434
python3 -B length_rejudge_20261026/run_rejudge.py run --base-url http://100.116.124.0:11434
python3 -B length_rejudge_20261026/run_rejudge.py status
```

Run uses four parallel requests within one judge, processing judges in the
original order. The first pending request per judge is performed alone and
saved before the remaining batch is scheduled. Every request has a 120-second
timeout and at most three attempts. It never restarts or reconfigures the server.
Successful results are appended to judgements.jsonl and skipped on resume.

Artifacts:

- `manifest.csv` / `manifest.json`: identities, targets, counts, original
  verdicts and source hashes for exactly the selected 81 answers.
- `request_payloads.jsonl`: all 405 exact request payloads, excluding the URL.
- `preflight.json`: latest connectivity check, available model metadata and
  exact-tag validation. No generation occurs if this check fails.
- `raw_responses/` / `attempts.jsonl`: raw API responses or errors for each
  attempted generation; `judgements.jsonl`: successful parsed results.
- `summary.json` / `judgement_comparison.csv`: progress and label transitions.
- `score_comparison.csv`: final scores before corrections, with only hard
  corrections, and with hard corrections plus the 405 new soft judgements.
- `quality_cells_*.csv` / `final_rankings_*.csv`: isolated comparative results;
  judge weights, ability scores and measured energy stay fixed.
- `score_verification.json`: replay errors against the original official scores.

The 151 newly hard answers are handled deterministically during score analysis;
they are not sent to a judge. Score analysis requires all 405 valid new labels.
New judge inference energy is not added to the tested models' measured energy.
