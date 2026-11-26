# Archived Legacy Code

このディレクトリには、現行パイプラインに一切使われない**過去のワンオフスクリプト**を退避しています。
削除せず、経緯の記録のために保存しています。現行の `run_pipeline.py` はこれらをインポートしません。

These are superseded one-off scripts kept for provenance. The current
`run_pipeline.py` never imports any of them.

## Files

| File | Original location | Why it was archived |
|---|---|---|
| `aggregate_reports.py` | `step4_q_total_output/` | Superseded by the current `step4_q_total.py`; part of the old aggregate/split approach that was replaced during pipeline refactoring. |
| `split_reports.py` | `step4_q_total_output/` | Same as above — companion to `aggregate_reports.py`. |
| `prepare_analysis_data.py` | `step4_q_total_output/` | Referenced a stale input (`q_total_results_mean.csv`) that no longer exists in the current layout; not part of the documented pipeline. |
| `diagnose_matching.py` | `step1_Test calibration/` | Early one-off debugging script for the judge-matching logic; the logic was folded into `pipeline_sources.py`. |
| `test_mapping.py` | `step1_Test calibration/` | Early one-off check of the prompt/response mapping; superseded by the calibration harness. |
| `test_penalty_evaluator.py` | `step1_Test calibration/` | Initial harness for the penalty evaluator; replaced by `penalty_evaluator_batch.py` and the saved judgement records. |

## Note

These scripts may contain paths or assumptions that no longer hold. Do not run
them against the current data layout; they are preserved only to document how
the pipeline evolved.
