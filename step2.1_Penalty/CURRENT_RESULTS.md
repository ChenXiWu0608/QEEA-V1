# 現行の減点判定（2026-10-27）

`Step2/reports{1,2}/Penalty_*.csv`には、中国語81回答・405裁判単位の再判定と、現行の長さhard・失敗回答の確定判定を反映しています。回答本文と質問は変更していません。置換前の記録は`../revision_20261027_length_sync/backup_before_sync.zip`に保管しています。

本評価の長さ比は、0.5未満または1.5超をhard、[0.5,0.75)または(1.25,1.5]をsoft相当の事前通知、[0.75,1.25]を通常判定とします。soft通知は最終ラベルを強制しません。中国語の指定文字数は「数字＋字／万字」から抽出します。機械可読な条件は`length_protocol.json`です。

裁判重みは従来の100件校正値を固定しています。校正時の長文hard上限2.0と、本評価の上限1.5を区別しています。Step1・Step3の保存判定は今回変更していません。

Step4はDatasetとの質問・回答本文の一致を検証し、空回答・失敗マーカー・長さhardを再確認して集計します。空の末尾行と対象外構成・対象外課題は除外します。

- 置換明細：`../revision_20261027_length_sync/penalty_cell_changes.csv`
- 再判定の原応答：`../length_rejudge_20261026/raw_responses/`
- 集計時の確定補正：`../step4_q_total_output/deterministic_penalty_corrections.csv`
- 現行WP：`../step4_q_total_output/q_total_results_reports{1,2}.csv`
- 統一入口：`../run_pipeline.py`（新しいモデル判定は呼び出しません）
