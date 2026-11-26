import pandas as pd

r1 = pd.read_csv("step4_q_total_output/q_total_results_reports1.csv")
r2 = pd.read_csv("step4_q_total_output/q_total_results_reports2.csv")

r1["report"] = "reports1"
r2["report"] = "reports2"

df = pd.concat([r1, r2], ignore_index=True)

agg_cols = ["WA", "WP", "Q_raw", "Q_total"]
group_cols = ["test_model", "test_step", "substep"]

agg = df.groupby(group_cols, sort=False).agg(
    **{f"{c}_mean": (c, "mean") for c in agg_cols},
    **{f"{c}_range": (c, lambda x: x.max() - x.min()) for c in agg_cols},
    n_anchors_present=("n_anchors_present", "first"),
    n_pen_judges=("n_pen_judges", "first"),
).reset_index()

agg.to_csv("step4_q_total_output/q_total_results_mean.csv", index=False)
print(f"Done: {len(agg)} rows written to q_total_results_mean.csv")
