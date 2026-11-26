import pandas as pd
import re

df = pd.read_csv("step4_q_total_output/q_total_results.csv")

df["report"] = df["test_model"].str.extract(r"(reports[12])$")
df["test_model"] = df["test_model"].str.replace(r"_(reports[12])$", "", regex=True)

for r in ["reports1", "reports2"]:
    out = df[df["report"] == r].drop(columns=["report"])
    out.to_csv(f"step4_q_total_output/q_total_results_{r}.csv", index=False)
    print(f"q_total_results_{r}.csv: {len(out)} rows")
