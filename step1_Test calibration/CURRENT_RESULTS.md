# 現行校正（2026-10-19）

勝敗・減点とも100件です。`vs/weight_calculation_results.csv`と`Penalty test/weight_calculation_results.csv`が現行値です。

入口は`../run_pipeline.py`。`calculate_model_weights.py`は`vs/calculate_judge_weights.py`への互換入口です。人間のJSONと裁判CSVは全文の問題ID・回答本文も照合し、空回答を欠落させず、全100件の整合を要求します。

新しい図用数値は`../step5_eqe_output/chart_data/`へ出力します。旧`../chart_data/`の画像・Originファイルを現行図として使用しないでください。
