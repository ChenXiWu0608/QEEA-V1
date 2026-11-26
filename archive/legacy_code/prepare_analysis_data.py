#!/usr/bin/env python3
"""
prepare_analysis_data.py
Merge Q_total_mean + EQE data into clean analysis-ready CSVs.
Handles model naming inconsistencies (hyphens vs underscores, case).

Outputs:
  step4_q_total_output/analysis_data.csv         — merged mean data + model metadata
  step4_q_total_output/analysis_data_by_report.csv — per-report data for error bars
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("step4_q_total_output")

# ============================================================
# 1. Model metadata parsing
# ============================================================

def normalize_name(name: str) -> str:
    """Normalize to lowercase+underscore, strip trailing _reports1/2."""
    n = name.strip().lower().replace("-", "_")
    n = re.sub(r"_(reports[12])$", "", n)
    return n

def parse_model_metadata(name: str) -> dict:
    """
    Parse a normalized model name into structured metadata.
    Produces unique clean model names that distinguish distill sources.
    """
    n = name.lower()

    # Family mapping
    family_map = {
        "deepseek_r1": "DeepSeek-R1",
        "deepseek_moe": "DeepSeek-MoE",
        "gemma3": "Gemma-3",
        "qwen3": "Qwen-3",
    }

    # Extract reasoning mode
    mode = "unknown"
    if n.endswith("_true"):
        mode = "thinking"
        n_base = n[:-5]
    elif n.endswith("_false"):
        mode = "standard"
        n_base = n[:-6]
    else:
        n_base = n

    # Extract quantization
    quant_map = {"fp16": "FP16", "q8_0": "Q8_0", "q4_k_m": "Q4_K_M", "qat": "QAT"}
    quant = "unknown"
    for qk, qv in sorted(quant_map.items(), key=lambda x: -len(x[0])):
        if qk in n_base:
            quant = qv
            break

    # Extract model family
    family = "other"
    for fk, fv in family_map.items():
        if fk in n_base:
            family = fv
            break

    # Extract model size
    size = -1
    m = re.search(r"_(\d+)b", n_base)
    if m:
        size = int(m.group(1))

    # Build a unique clean key and display name
    # Capture distill source for DeepSeek-R1 variants
    variant = ""
    if family == "DeepSeek-R1":
        if "0528_qwen3" in n_base or "qwen_distill" in n_base:
            variant = "Qwen3"
        elif "llama_distill" in n_base:
            variant = "Llama"
        elif "base" in n_base:
            variant = "Base"
    elif family == "Qwen-3":
        if "a3b_instruct" in n_base:
            variant = "a3b"
        elif "a3b_thinking" in n_base:
            variant = "a3b-think"

    # Also capture "latest" suffix for deepseek-moe
    extra = ""
    if "latest" in n_base:
        extra = "latest"

    parts = [family, f"{size}B" if size > 0 else ""]
    if variant:
        parts.append(variant)
    if extra:
        parts.append(extra)
    if quant and quant not in ("unknown",):
        parts.append(quant)
    if mode == "thinking":
        parts.append("think")

    clean_name = "_".join(p for p in parts if p)

    display_parts_dn = [family, f"{size}B" if size > 0 else ""]
    if variant:
        display_parts_dn.append(f"({variant})")
    if extra:
        display_parts_dn.append(f"({extra})")
    if quant and quant not in ("unknown",):
        display_parts_dn.append(quant)
    if mode == "thinking":
        display_parts_dn.append("(thinking)")

    display_name = " ".join(p for p in display_parts_dn if p)

    return {
        "model_clean": clean_name,
        "model_display": display_name,
        "model_family": family,
        "model_size_b": size,
        "quantization": quant.lower(),
        "reasoning_mode": mode,
    }


# ============================================================
# 2. Load and normalize Q_total mean
# ============================================================

def load_q_total_mean():
    df = pd.read_csv(OUT_DIR / "q_total_results_mean.csv")
    print(f"[Q_total_mean] Loaded {len(df)} rows, {df['test_model'].nunique()} raw models")

    # Normalize model names
    meta = df["test_model"].apply(lambda x: parse_model_metadata(normalize_name(x)))
    meta_df = pd.DataFrame(meta.tolist())

    result = pd.concat([df, meta_df], axis=1)
    result = result.drop(columns=["test_model"])

    print(f"  After normalization: {result['model_clean'].nunique()} unique models")
    print(f"  Families: {sorted(result['model_family'].unique())}")
    print(f"  Sizes: {sorted(result['model_size_b'].unique())}")
    print(f"  Quants: {sorted(result['quantization'].unique())}")
    print(f"  Reasoning: {sorted(result['reasoning_mode'].unique())}")

    return result


# ============================================================
# 3. Load and normalize EQE (per-report)
# ============================================================

def load_eqe():
    df = pd.read_csv(OUT_DIR.parent / "step5_eqe_output" / "eqe_all.csv")
    print(f"\n[EQE] Loaded {len(df)} rows, {df['test_model'].nunique()} raw models")

    # Normalize model names using same parser
    meta = df["test_model"].apply(lambda x: parse_model_metadata(normalize_name(x)))
    meta_df = pd.DataFrame(meta.tolist())

    result = pd.concat([df, meta_df], axis=1)
    result = result.drop(columns=["test_model"])

    return result


# ============================================================
# 4. Merge Q_total mean + EQE mean into one analysis dataset
# ============================================================

def merge_mean_datasets(q_df, eqe_df):
    """
    Merge Q_total_mean (already averaged over reports) with EQE (per-report).
    For a unified dataset, average EQE over reports too.
    """
    # Average EQE over reports
    eqe_mean_cols = ["EQE_total", "EQE_gpu"]
    eqe_agg = eqe_df.groupby(
        ["model_clean", "model_display", "model_family", "model_size_b",
         "quantization", "reasoning_mode", "substep"],
        sort=False
    )[eqe_mean_cols].agg(["mean", "std"]).reset_index()
    eqe_agg.columns = ["model_clean", "model_display", "model_family",
                        "model_size_b", "quantization", "reasoning_mode",
                        "substep"] + [f"{c}_{s}" for c in eqe_mean_cols for s in ["mean", "std"]]

    # Merge with Q_total
    merge_cols = ["model_clean", "model_family", "model_size_b",
                  "quantization", "reasoning_mode", "test_step", "substep"]
    merged = q_df.merge(
        eqe_agg,
        on=["model_clean", "model_family", "model_size_b",
            "quantization", "reasoning_mode", "substep"],
        how="left",
        suffixes=("", "_eqe"),
    )

    # Use model_display from q_df
    if "model_display_eqe" in merged.columns:
        merged = merged.drop(columns=["model_display_eqe"])

    print(f"\n[Merge] Q_total: {len(q_df)} rows, EQE agg: {len(eqe_agg)} groups")
    print(f"  Merged: {len(merged)} rows")
    n_unmatched = merged["EQE_total_mean"].isna().sum()
    if n_unmatched:
        print(f"  Unmatched: {n_unmatched} rows")
        for _, row in merged[merged["EQE_total_mean"].isna()].iterrows():
            print(f"    {row['model_clean']} | {row['substep']}")

    return merged


# ============================================================
# 5. Per-report data (for error bars)
# ============================================================

def build_per_report(q_df_raw, eqe_df):
    """
    Build a per-report dataset: each row is one (model, substep, report).
    """
    # Read raw Q_total (not mean)
    q_raw = pd.read_csv(OUT_DIR / "q_total_results.csv")
    q_raw["report"] = q_raw["test_model"].str.extract(r"(reports[12])$")
    q_raw["test_model"] = q_raw["test_model"].str.replace(r"_(reports[12])$", "", regex=True)
    q_raw_clean = q_raw.copy()
    meta = q_raw_clean["test_model"].apply(lambda x: parse_model_metadata(normalize_name(x)))
    meta_df = pd.DataFrame(meta.tolist())
    q_raw_clean = pd.concat([q_raw_clean, meta_df], axis=1)
    q_raw_clean = q_raw_clean.drop(columns=["test_model"])

    # EQE already has report column
    eqe_clean = eqe_df.copy()

    # Merge per-report
    merged = q_raw_clean.merge(
        eqe_clean,
        on=["model_clean", "model_family", "model_size_b",
            "quantization", "reasoning_mode", "substep", "report"],
        how="inner",
        suffixes=("", "_eqe"),
    )
    if "model_display_eqe" in merged.columns:
        merged = merged.drop(columns=["model_display_eqe"])

    # Select key columns
    out_cols = ["model_clean", "model_display", "model_family", "model_size_b",
                "quantization", "reasoning_mode", "test_step", "substep", "report",
                "Q_total", "WA", "WP", "EQE_total", "EQE_gpu"]
    available = [c for c in out_cols if c in merged.columns]
    return merged[available]


# ============================================================
# main
# ============================================================

def main():
    print("=" * 60)
    print("  ANALYSIS DATA PREPARATION")
    print("=" * 60)

    # 1. Load Q_total mean
    q_mean = load_q_total_mean()

    # 2. Load EQE
    eqe = load_eqe()

    # 3. Merge
    merged = merge_mean_datasets(q_mean, eqe)

    # 4. Save merged analysis data
    out_cols = ["model_clean", "model_display", "model_family", "model_size_b",
                "quantization", "reasoning_mode", "test_step", "substep",
                "Q_total_mean", "Q_total_range", "WA_mean", "WA_range",
                "WP_mean", "WP_range", "Q_raw_mean", "Q_raw_range",
                "EQE_total_mean", "EQE_total_std", "EQE_gpu_mean", "EQE_gpu_std"]
    available = [c for c in out_cols if c in merged.columns]
    out = merged[available].sort_values(
        ["model_family", "model_size_b", "quantization", "reasoning_mode", "substep"]
    ).reset_index(drop=True)
    out_path = OUT_DIR / "analysis_data.csv"
    out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n  Saved {out_path} ({len(out)} rows)")

    # 5. Save per-report data
    per_rep = build_per_report(q_mean, eqe)
    per_rep_path = OUT_DIR / "analysis_data_by_report.csv"
    per_rep.to_csv(per_rep_path, index=False, float_format="%.6f")
    print(f"  Saved {per_rep_path} ({len(per_rep)} rows)")

    # 6. Summary
    print(f"\n{'=' * 60}")
    print(f"  Models: {out['model_clean'].nunique()}")
    print(f"  Families: {sorted(out['model_family'].unique())}")
    for f in sorted(out['model_family'].unique()):
        sub = out[out['model_family'] == f]
        print(f"    {f}: {sub['model_clean'].nunique()} models, "
              f"sizes={sorted(sub['model_size_b'].unique())}")
    print(f"  Quantizations: {sorted(out['quantization'].unique())}")
    print(f"  Reasoning modes: {sorted(out['reasoning_mode'].unique())}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
