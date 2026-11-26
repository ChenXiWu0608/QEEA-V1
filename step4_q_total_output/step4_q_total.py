#!/usr/bin/env python3
"""
step4_q_total.py -- Weighted Q_total from Ability + Penalty

Step 4 of the pipeline:
  WA = Sum_j omega_win[j] * Ability_{A,step,j}
  WP = Sum_j omega_pen[j] * Penalty_{A,step,j}
  Q_raw = WA - WP
  Min-Max per substep -> Q_total in [0, 100]
"""

import os, sys, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "ability_csv": "step3_ability_output/ability_scores.csv",
    "win_weight_csv": "step1_Test calibration/vs/weight_calculation_results.csv",
    "pen_weight_csv": "step1_Test calibration/Penalty test/weight_calculation_results.csv",
    "penalty_root": "step2.1_Penalty",
    "out_dir": "step4_q_total_output",

    "substep_split": {
        "Step1": [(1, 100, "ArenaHard"), (101, 150, "LiveCodeBench"), (151, 200, "GPQA")],
    },

    "verdict_map": {"hard": 3.0, "soft": 1.0, "nice": 0.0},
    "invalid_verdicts": ["unknown", "", "nan", "error"],

    "judge_columns": {
        "ministral3_8b", "olmo3_7b", "llama3.1_8b", "gemma4_e4b", "qwen3.5_9b",
    },

    "penalty_exclude_categories": ["code_generation"],
}


# ============================================================
# normalize_model_name -- standardize test_model for merge key
# ============================================================
def normalize_model_name(name):
    name = str(name)
    name = name.replace("-", "_")
    name = re.sub(r'_(true|false)(_|$)', lambda m: '_' + m.group(1).upper() + m.group(2),
                  name, flags=re.IGNORECASE)
    name = re.sub(r'_(true|false)$', lambda m: '_' + m.group(1).upper(),
                  name, flags=re.IGNORECASE)
    return name.lower()


# ============================================================
# load_weights -- parse weight files into (model -> omega) dicts
# ============================================================
def load_weights(csv_path, label):
    if not os.path.isfile(csv_path):
        print(f"  [FATAL] {label} weight file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    if "model" not in df.columns or "omega" not in df.columns:
        print(f"  [FATAL] {label} weight file missing 'model' or 'omega' columns")
        sys.exit(1)
    w = dict(zip(df["model"], df["omega"]))
    expected = {'gemma4_e4b', 'llama3.1_8b', 'ministral3_8b', 'olmo3_7b', 'qwen3.5_9b'}
    if set(w) != expected or len(df) != 5 or not np.isfinite(df.omega).all() or abs(df.omega.sum()-1) > 1e-12:
        raise ValueError(f'{label}: missing/duplicated judge or invalid weight normalization')
    print(f"  Loaded {label} weights: {len(w)} judges")
    for m, v in sorted(w.items()):
        print(f"    {m}: {v:.6f}")
    return w


# ============================================================
# load_ability -- read ability_scores and normalize test_model
# ============================================================
def load_ability(cfg):
    csv_path = cfg["ability_csv"]
    if not os.path.isfile(csv_path):
        print(f"  [FATAL] ability file not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    print(f"  Loaded ability_scores: {len(df)} rows, {df['test_model'].nunique()} models")
    df = df[~df.test_model.str.contains('deepseek_moe', case=False)
            & df.test_model.str.contains(r'_reports[12]$', regex=True)].copy()
    if df.test_model.nunique() != 44 or len(df) != 1100:
        raise ValueError('Expected 22 configurations x 2 reports x 5 substeps x 5 judges')

    df["merge_key"] = df["test_model"].apply(normalize_model_name)

    return df


# ============================================================
# resolve_penalty_substep -- derive substep from test_step + penalty_id
# ============================================================
def resolve_penalty_substep(test_step, penalty_id):
    if test_step != "Step1":
        return test_step

    try:
        item_num = int(penalty_id)
    except (ValueError, TypeError):
        return f"Step1_UNMAPPED_{penalty_id}"

    if test_step in CONFIG["substep_split"]:
        for start, end, substep_name in CONFIG["substep_split"][test_step]:
            if start <= item_num <= end:
                return substep_name

    return f"Step1_UNMAPPED_{item_num}"


# ============================================================
# load_penalty -- scan Penalty directory, load and aggregate
# ============================================================
def load_penalty(cfg):
    from pipeline_sources import effective_penalties
    return effective_penalties(cfg, normalize_model_name)




# ============================================================
# merge_and_compute -- merge ability + penalty, compute WA/WP/Q
# ============================================================
def merge_and_compute(ability_df, penalty_df, omega_win, omega_pen, cfg):
    # 1. Merge at (merge_key, substep, judge_model) level
    merged = ability_df.merge(
        penalty_df,
        on=["merge_key", "substep", "judge_model"],
        how="left",
        suffixes=("", "_pen"),
        validate="one_to_one",
    )

    total_ability = len(ability_df)
    matched = merged["Penalty"].notna().sum()
    if matched != total_ability or merged['Ability'].isna().any() or not merged.n_anchors_present.eq(4).all():
        raise ValueError('Incomplete ability/penalty coverage; refusing to silently omit judges or anchors')
    print(f"\n  Ability rows: {total_ability}")
    print(f"  Matched with penalty: {matched} ({matched / total_ability * 100:.1f}%)")
    print(f"  Unmatched (no penalty): {total_ability - matched}")

    judge_list = sorted(cfg["judge_columns"])

    results = []
    for (mk, ts, ss), grp in merged.groupby(["merge_key", "test_step", "substep"], observed=False):
        if len(grp) == 0:
            continue

        orig_model = grp["test_model"].iloc[0]
        n_anchors = grp["n_anchors_present"].iloc[0] if "n_anchors_present" in grp.columns else 4

        wa = 0.0
        for _, row in grp.iterrows():
            j = row["judge_model"]
            ab = row["Ability"]
            if pd.notna(ab) and j in omega_win:
                wa += omega_win[j] * ab

        wp = 0.0
        n_pen = 0
        for _, row in grp.iterrows():
            j = row["judge_model"]
            pe = row["Penalty"]
            if pd.notna(pe) and j in omega_pen:
                wp += omega_pen[j] * pe
                n_pen += 1

        results.append({
            "merge_key": mk,
            "test_model": orig_model,
            "test_step": ts,
            "substep": ss,
            "WA": wa,
            "WP": wp,
            "Q_raw": wa - wp,
            "n_anchors_present": n_anchors,
            "n_pen_judges": n_pen,
        })

    return pd.DataFrame(results)


# ============================================================
# normalize_Q -- Min-Max per substep -> [0, 100]
# ============================================================
def normalize_Q(df):
    df = df.copy()
    df["Q_total"] = np.nan

    for ss, grp in df.groupby("substep", observed=False):
        valid_mask = grp["n_anchors_present"] >= 4
        valid_q = grp.loc[valid_mask, "Q_raw"].dropna()

        if len(valid_q) < 2:
            print(f"  [WARN] substep '{ss}': fewer than 2 valid Q_raw values, skipping normalization")
            continue

        q_min = valid_q.min()
        q_max = valid_q.max()
        if abs(q_max - q_min) < 1e-12:
            print(f"  [WARN] substep '{ss}': Q_raw range is zero, setting all to 50")
            df.loc[grp.index[valid_mask], "Q_total"] = 50.0
        else:
            idxs = grp.index[valid_mask]
            df.loc[idxs, "Q_total"] = (grp.loc[idxs, "Q_raw"] - q_min) / (q_max - q_min) * 100.0

    return df


# ============================================================
# save_results — report1, report2, および統合(mean+std)を出力
# ============================================================
def save_results(df, cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # test_model から report 情報をパース
    tmp = df.copy()
    tmp["report"] = tmp["test_model"].str.extract(r"_(reports[12])$", expand=False)
    tmp["base_model"] = tmp["test_model"].str.replace(r"_(reports[12])$", "", regex=True)

    out_cols = ["test_model", "test_step", "substep",
                "WA", "WP", "Q_raw", "Q_total",
                "n_anchors_present", "n_pen_judges", "report"]

    # 1. report1 / report2 個別出力（test_model は _reportsN サフィックス保持）
    for r in ["reports1", "reports2"]:
        sub = tmp[tmp["report"] == r][out_cols].copy()
        sub.to_csv(out_dir / f"q_total_results_{r}.csv", index=False, float_format="%.17g")
        print(f"  Saved q_total_results_{r}.csv ({len(sub)} rows)")

    # 2. 統合結果 (mean + std) — base_model でグループ化
    agg_cols = ["WA", "WP", "Q_raw", "Q_total"]
    group_cols = ["base_model", "test_step", "substep"]
    agg = tmp.groupby(group_cols, sort=False).agg(
        **{f"{c}_mean": (c, "mean") for c in agg_cols},
        **{f"{c}_std": (c, "std") for c in agg_cols},
        n_anchors_present=("n_anchors_present", "first"),
        n_pen_judges=("n_pen_judges", "first"),
    ).reset_index().rename(columns={"base_model": "test_model"})
    agg.to_csv(out_dir / "q_total_results.csv", index=False, float_format="%.17g")
    print(f"  Saved q_total_results.csv ({len(agg)} rows, combined mean+std)")


# ============================================================
# self_check
# ============================================================
def self_check(result_df, ability_df, penalty_df, omega_win, omega_pen):
    print("\n" + "=" * 60)
    print("  SELF CHECK")
    print("=" * 60)

    print(f"\n  Merge match rate:")
    print(f"    Ability rows: {len(ability_df)}")
    print(f"    Result rows: {len(result_df)}")
    expected = ability_df.groupby(["merge_key", "substep"], observed=False).ngroups
    print(f"    Expected (unique model x substep): {expected}")
    if len(result_df) == expected:
        print(f"    MATCH OK")
    else:
        print(f"    MISSING {expected - len(result_df)} groups")

    print(f"\n  Substep distribution:")
    for ss, grp in result_df.groupby("substep", observed=False):
        n_nan = grp["Q_total"].isna().sum()
        print(f"    {ss}: {len(grp)} models, Q_total NaN={n_nan}")

    print(f"\n  Q_raw range per substep:")
    for ss, grp in result_df.groupby("substep", observed=False):
        q = grp["Q_raw"].dropna()
        if len(q) > 0:
            print(f"    {ss}: [{q.min():.4f}, {q.max():.4f}]  mean={q.mean():.4f}")
        else:
            print(f"    {ss}: no data")

    print(f"\n  WA vs WP (overall means):")
    mean_wa = result_df["WA"].mean()
    mean_wp = result_df["WP"].mean()
    print(f"    mean(WA) = {mean_wa:.4f}")
    print(f"    mean(WP) = {mean_wp:.4f}")
    print(f"    ratio WP/WA = {mean_wp / mean_wa:.4f}" if mean_wa != 0 else "    WA is zero")
    if mean_wp > mean_wa:
        print(f"    WARNING: mean(WP) > mean(WA), Q_raw may be negative")

    print(f"\n  Penalty judge coverage:")
    for judge in sorted(CONFIG["judge_columns"]):
        subset = penalty_df[penalty_df["judge_model"] == judge]
        if len(subset) > 0:
            avg_items = subset["n_pen_items"].mean()
            total_items = subset["n_pen_items"].sum()
            na_count = subset["Penalty"].isna().sum()
            print(f"    {judge}: {len(subset)} groups, avg items={avg_items:.1f}, "
                  f"NaN penalty groups={na_count}")

    print(f"\n  n_pen_judges distribution:")
    print(f"    {result_df['n_pen_judges'].value_counts().sort_index().to_dict()}")

    neg_count = (result_df["Q_raw"] < 0).sum()
    print(f"\n  Negative Q_raw: {neg_count} / {len(result_df)} rows")
    if neg_count > 0:
        print(f"  (Negative Q_raw means penalty > ability -- expected for some models)")


# ============================================================
# main
# ============================================================
def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    cfg = CONFIG

    print("=" * 60)
    print("  Q_TOTAL CALCULATION (Step 4)")
    print("=" * 60)

    # 1. Load weights
    print("\n--- Loading weights ---")
    omega_win = load_weights(cfg["win_weight_csv"], "win")
    omega_pen = load_weights(cfg["pen_weight_csv"], "penalty")

    # 2. Load ability
    print("\n--- Loading ability scores ---")
    ability = load_ability(cfg)
    print(f"  Models: {ability['test_model'].nunique()}, substeps: {ability['substep'].nunique()}")

    # 3. Load & aggregate penalty
    print("\n--- Loading and aggregating penalty ---")
    penalty = load_penalty(cfg)
    print(f"  Groups: {len(penalty)}, models: {penalty['merge_key'].nunique()}")

    # 4. Merge and compute WA/WP/Q_raw
    print("\n--- Computing WA, WP, Q_raw ---")
    result = merge_and_compute(ability, penalty, omega_win, omega_pen, cfg)
    print(f"  Result rows: {len(result)}")

    # 5. Normalize Q
    print("\n--- Normalizing Q ---")
    result = normalize_Q(result)
    q_total_nan = result["Q_total"].isna().sum()
    print(f"  Q_total NaN: {q_total_nan}")
    if q_total_nan > 0:
        excluded = result[result["Q_total"].isna()]
        for _, row in excluded.iterrows():
            print(f"    {row['test_model']} | {row['substep']}: "
                  f"n_anchors={row['n_anchors_present']}, Q_raw={row['Q_raw']:.4f}")

    # 6. Save
    print("\n--- Saving results ---")
    save_results(result, cfg)

    # 7. Self-check
    self_check(result, ability, penalty, omega_win, omega_pen)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
