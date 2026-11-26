#!/usr/bin/env python3
"""
step5_eqe.py -- Effective Quality per Energy (EQE) Calculation

Step 5 of the pipeline:
  EQ = Q_total (penalty-adjusted effective quality, [0, 100])
  EQE_total = EQ / sum(total_energy_joule)   [EQ/J]
  EQE_gpu   = EQ / sum(gpu_energy_joule)     [EQ/J]

Inputs:
  - step4_q_total_output/q_total_results.csv
  - Dataset/Step{1,2,3}/analysis{1,2}/analysis_*.csv

Output:
  - step5_eqe_output/eqe_{substep}.csv  (5 files, one per substep)
"""

import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "q_total_csv": "step4_q_total_output/q_total_results.csv",
    "energy_root": "Dataset",
    "out_dir": "step5_eqe_output",

    # Step1 test_id -> substep mapping
    "step1_substep_ranges": [
        ("test_0000", "test_0099", "ArenaHard"),
        ("test_0100", "test_0149", "LiveCodeBench"),
        ("test_0150", "test_0199", "GPQA"),
    ],

    # analysis directory -> report_round mapping
    "report_map": {"analysis1": "reports1", "analysis2": "reports2"},

    # Models to exclude
    "exclude_models_containing": ["deepseek_moe"],

    # Penalty data root (for filtering out code_generation tasks in Step3 energy)
    "penalty_root": "step2.1_Penalty",
    "penalty_exclude_categories": ["code_generation"],

    # All substeps in output order
    "substeps": ["ArenaHard", "LiveCodeBench", "GPQA", "Step2", "Step3"],
}


# ============================================================
# Model Name Normalization
# ============================================================
def normalize_name(name: str) -> str:
    """Normalize a model name to lowercase+underscore for matching."""
    return name.strip().lower().replace("-", "_")


def parse_q_total_name(test_model: str):
    """
    Parse Q_total test_model into (merge_key, report).
    
    'qwen3_32b-q4_K_M_False_reports1' -> ('qwen3_32b_q4_k_m_false', 'reports1')
    'deepseek-r1_14b-qwen-distill-q8_0_False_reports2' -> ('deepseek_r1_14b_qwen_distill_q8_0_false', 'reports2')
    """
    name = test_model.strip()
    m = re.search(r'_(reports[12])$', name, re.IGNORECASE)
    if m:
        report = m.group(1).lower()
        base = name[:m.start()]
    else:
        report = "unknown"
        base = name
    return normalize_name(base), report


def parse_energy_filename(fname: str):
    """
    Parse energy filename into (merge_key, report_round_dir).
    
    'analysis_qwen3_32b-q4_K_M_false_20251106_101850.csv' -> ('qwen3_32b_q4_k_m_false', 'analysis1')
    """
    stem = fname.replace(".csv", "")
    # Strip 'analysis_' prefix
    if stem.startswith("analysis_"):
        stem = stem[len("analysis_"):]
    # Strip timestamp suffix: _YYYYMMDD_HHMMSS
    m = re.search(r'_\d{8}_\d{6}$', stem)
    if m:
        base = stem[:m.start()]
    else:
        base = stem
    return normalize_name(base)


# ============================================================
# Load Q_total
# ============================================================
def load_q_total(cfg):
    # per-report ファイルから読み込み
    report_files = [
        cfg["q_total_csv"].replace(".csv", "_reports1.csv"),
        cfg["q_total_csv"].replace(".csv", "_reports2.csv"),
    ]

    chunks = []
    for fpath in report_files:
        if not os.path.isfile(fpath):
            print(f"  [FATAL] Q_total file not found: {fpath}")
            sys.exit(1)
        sub = pd.read_csv(fpath)
        sub["report"] = sub["test_model"].str.extract(r"_(reports[12])$", expand=False)
        sub["base_model"] = sub["test_model"].str.replace(r"_(reports[12])$", "", regex=True)
        chunks.append(sub)

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n[Q_total] Loaded {len(df)} rows, {df['base_model'].nunique()} unique base_models")

    # Exclude models
    mask = pd.Series(True, index=df.index)
    for keyword in cfg["exclude_models_containing"]:
        mask &= ~df["test_model"].str.contains(keyword, case=False, na=False)
    n_removed = len(df) - mask.sum()
    if n_removed:
        df = df[mask].copy()
        print(f"  Excluded {n_removed} rows containing '{keyword}'")

    # merge_key は base_model の正規化版
    df["merge_key"] = df["base_model"].apply(normalize_name)

    print(f"  After exclusion: {len(df)} rows, {df['merge_key'].nunique()} unique merge_keys")
    print(f"  Reports: {sorted(df['report'].unique())}")
    print(f"  Substeps: {sorted(df['substep'].unique())}")

    return df


# ============================================================
# Load Energy Data
# ============================================================
def load_energy(cfg):
    from pipeline_sources import integrate_energy
    return integrate_energy(cfg)




# ============================================================
# Merge Q_total + Energy, Compute EQE
# ============================================================
def compute_eqe(q_df, energy_df, cfg):
    substep_order = cfg["substeps"]

    # Merge on (merge_key, substep, report)
    merged = q_df.merge(
        energy_df,
        on=["merge_key", "substep", "report"],
        how="inner",
        suffixes=("", "_energy"),
        validate="one_to_one",
    )

    n_q = len(q_df)
    n_merged = len(merged)
    print(f"\n[Merge] Q_total rows: {n_q}, matched with energy: {n_merged} ({n_merged/n_q*100:.1f}%)")

    # Check for unmatched
    unmatched_models = q_df[~q_df["test_model"].isin(merged["test_model"])]
    if len(unmatched_models) > 0:
        print(f"  Unmatched Q_total rows: {len(unmatched_models)}")
        for _, row in unmatched_models.iterrows():
            print(f"    {row['merge_key']} | {row['substep']} | {row['report']}")

    # Compute EQE
    merged["EQ"] = merged["Q_total"]
    merged["EQE_total"] = merged["EQ"] / merged["total_energy_joule"]
    merged["EQE_gpu"] = merged["EQ"] / merged["gpu_energy_joule"]
    merged["EQEE"] = merged["EQE_total"] * 1e6
    if len(merged) != len(q_df) or len(merged) != 220:
        raise ValueError('Quality and energy must match one-to-one across all 220 cells')

    print("\n[EQE Summary]")
    for ss in substep_order:
        sub = merged[merged["substep"] == ss]
        if len(sub) == 0:
            continue
        eqe_t = sub["EQE_total"]
        eqe_g = sub["EQE_gpu"]
        print(f"  {ss}: n={len(sub)}, "
              f"EQE_total [{eqe_t.min():.6f}, {eqe_t.max():.6f}] mean={eqe_t.mean():.6f}, "
              f"EQE_gpu [{eqe_g.min():.6f}, {eqe_g.max():.6f}] mean={eqe_g.mean():.6f}")

    return merged


# ============================================================
# Save Outputs — report別 + 統合(mean±std)
# ============================================================
def save_outputs(eqe_df, cfg):
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eqe_df.to_csv(out_dir / 'eqee_cells.csv', index=False, float_format='%.17g')
    final_source = eqe_df.assign(model=eqe_df['merge_key'])
    repeat = final_source.groupby(['model', 'report']).agg(
        EQ=('EQ', 'mean'), EQEE=('EQEE', 'mean'), energy_j=('total_energy_joule', 'sum')).reset_index()
    final = repeat.groupby('model').agg(
        EQ=('EQ', 'mean'), EQEE=('EQEE', 'mean'), EQ_std=('EQ', 'std'),
        EQEE_std=('EQEE', 'std'), energy_j=('energy_j', 'mean')).reset_index()
    final['EQ_rank'] = final.EQ.rank(ascending=False, method='min').astype(int)
    final['EQEE_rank'] = final.EQEE.rank(ascending=False, method='min').astype(int)
    final['DeltaRank'] = final.EQ_rank - final.EQEE_rank
    final['energy_mj_per_repeat'] = final.energy_j / 1e6
    final.sort_values('EQEE_rank').to_csv(out_dir / 'final_rankings.csv', index=False, float_format='%.17g')
    repeat.to_csv(out_dir / 'repeat_scores.csv', index=False, float_format='%.17g')

    substep_order = cfg["substeps"]
    out_cols = ["test_model", "EQ", "total_energy_joule", "gpu_energy_joule",
                "EQE_total", "EQE_gpu", "EQEE", "report"]

    # test_model から report 情報を分離
    tmp = eqe_df.copy()
    tmp["base_model"] = tmp["test_model"].str.replace(r"_(reports[12])$", "", regex=True)

    all_report1_rows = []
    all_report2_rows = []

    for ss in substep_order:
        sub = tmp[tmp["substep"] == ss].copy()

        # report別出力
        for r in ["reports1", "reports2"]:
            rsub = sub[sub["report"] == r].copy()
            rsub["test_model"] = rsub["base_model"]
            r_out = rsub[out_cols].sort_values("test_model")
            fname = f"eqe_{ss}_{r}.csv"
            r_out.to_csv(out_dir / fname, index=False, float_format="%.17g")
            print(f"  Saved {fname} ({len(r_out)} rows)")
            if r == "reports1":
                all_report1_rows.append(rsub.assign(test_model=rsub["base_model"]))
            else:
                all_report2_rows.append(rsub.assign(test_model=rsub["base_model"]))

        # 統合結果 (mean±std)
        agg_cols = ["EQ", "total_energy_joule", "gpu_energy_joule", "EQE_total", "EQE_gpu", "EQEE"]
        agg = sub.groupby("base_model", sort=False).agg(
            **{f"{c}_mean": (c, "mean") for c in agg_cols},
            **{f"{c}_std": (c, "std") for c in agg_cols},
        ).reset_index().rename(columns={"base_model": "test_model"})
        fname = f"eqe_{ss}.csv"
        agg.to_csv(out_dir / fname, index=False, float_format="%.17g")
        print(f"  Saved {fname} ({len(agg)} rows, combined mean±std)")

    # 全substep report別
    for r, rows, label in [("reports1", all_report1_rows, "eqe_all_reports1"),
                           ("reports2", all_report2_rows, "eqe_all_reports2")]:
        if rows:
            all_r = pd.concat(rows, ignore_index=True)
            all_r = all_r[["test_model", "substep", "EQ", "total_energy_joule",
                           "gpu_energy_joule", "EQE_total", "EQE_gpu", "EQEE"]]
            all_r = all_r.sort_values(["substep", "test_model"])
            all_r.to_csv(out_dir / f"{label}.csv", index=False, float_format="%.17g")
            print(f"  Saved {label}.csv ({len(all_r)} rows)")

    # 全substep統合
    all_combined = []
    for ss in substep_order:
        sub = tmp[tmp["substep"] == ss]
        agg_cols = ["EQ", "total_energy_joule", "gpu_energy_joule", "EQE_total", "EQE_gpu", "EQEE"]
        agg = sub.groupby("base_model", sort=False).agg(
            **{f"{c}_mean": (c, "mean") for c in agg_cols},
            **{f"{c}_std": (c, "std") for c in agg_cols},
        ).reset_index()
        agg["substep"] = ss
        all_combined.append(agg)
    all_out = pd.concat(all_combined, ignore_index=True).rename(columns={"base_model": "test_model"})
    all_out = all_out[["test_model", "substep"] + [f"{c}_mean" for c in agg_cols] + [f"{c}_std" for c in agg_cols]]
    all_out = all_out.sort_values(["substep", "test_model"])
    all_out.to_csv(out_dir / "eqe_all.csv", index=False, float_format="%.17g")
    print(f"  Saved eqe_all.csv ({len(all_out)} rows, combined mean±std)")


# ============================================================
# Self-Check
# ============================================================
def self_check(eqe_df, cfg):
    print("\n" + "=" * 60)
    print("  SELF CHECK")
    print("=" * 60)

    substep_order = cfg["substeps"]

    # 1. Check report coverage per (merge_key, substep)
    print("\n--- Report coverage ---")
    cov = eqe_df.groupby(["merge_key", "substep"])["report"].nunique()
    n_2 = (cov == 2).sum()
    n_1 = (cov == 1).sum()
    print(f"  2 reports: {n_2} combos")
    print(f"  1 report:  {n_1} combos")
    if n_1 > 0:
        singles = cov[cov == 1]
        print(f"  Single-report combos:")
        for idx, val in singles.items():
            print(f"    {idx[0]} | {idx[1]}")

    # 2. Models per substep
    print("\n--- Models per substep ---")
    for ss in substep_order:
        sub = eqe_df[eqe_df["substep"] == ss]
        models = sub["test_model"].unique()
        print(f"  {ss}: {len(models)} unique models")

    # 3. Energy range check
    print("\n--- Energy range (J) ---")
    for ss in substep_order:
        sub = eqe_df[eqe_df["substep"] == ss]
        if len(sub) == 0:
            continue
        print(f"  {ss}: total [{sub['total_energy_joule'].min():.0f}, {sub['total_energy_joule'].max():.0f}] J, "
              f"gpu [{sub['gpu_energy_joule'].min():.0f}, {sub['gpu_energy_joule'].max():.0f}] J")

    # 4. Any NaN
    print("\n--- NaN check ---")
    nan_cols = eqe_df[["EQ", "EQE_total", "EQE_gpu"]].isna().sum()
    for col, c in nan_cols.items():
        if c > 0:
            print(f"  {col}: {c} NaN [WARN]")
            # Print affected rows
            affected = eqe_df[eqe_df[col].isna()]
            for _, r in affected.iterrows():
                print(f"    {r['test_model']} | {r['substep']} | {r['report']}")
    if nan_cols.sum() == 0:
        print(f"  No NaN values [OK]")


# ============================================================
# Main
# ============================================================
def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    cfg = CONFIG

    print("=" * 60)
    print("  EQE CALCULATION (Step 5)")
    print("  Effective Quality per Energy")
    print("=" * 60)

    # 1. Load Q_total
    print("\n--- Loading Q_total ---")
    q_df = load_q_total(cfg)

    # 2. Load energy data
    print("\n--- Loading energy data ---")
    energy_df = load_energy(cfg)

    # 3. Merge and compute EQE
    print("\n--- Computing EQE ---")
    eqe_df = compute_eqe(q_df, energy_df, cfg)

    # 4. Save outputs
    print("\n--- Saving outputs ---")
    save_outputs(eqe_df, cfg)

    # 5. Self-check
    self_check(eqe_df, cfg)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
