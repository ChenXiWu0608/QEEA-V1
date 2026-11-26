import pandas as pd
import numpy as np
from scipy import stats
import json
import hashlib
import os

# ==================== Configuration ====================
VS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(VS_DIR)

CSV_FILES = {
    'gemma4_e4b':     'Test_vs_qwen3_5_9b_gemma4_e4b.csv',
    'llama3.1_8b':    'Test_vs_qwen3_5_9b_llama3.1_8b.csv',
    'ministral3_8b':  'Test_vs_qwen3_5_9b_ministral3_8b.csv',
    'olmo3_7b':       'Test_vs_qwen3_5_9b_olmo3_7b.csv',
    'qwen3.5_9b':     'Test_vs_qwen3_5_9b_qwen3.5_9b.csv',
}

HUMAN_JSON = os.path.join(BASE_DIR, "human_comparison_prompts.json")

SOURCES = ['gemma4_e4b', 'llama3.1_8b', 'ministral3_8b', 'olmo3_7b', 'qwen3.5_9b', 'human']

VALID_VERDICTS = {"A>>B", "A>B", "A=B", "A<B", "A<<B"}

# Verdict -> ordinal (from Test perspective, A=Test, B=Baseline)
# vs_1 (A=Test, B=Baseline) direct mapping
VERDICT_TO_ORDINAL_VS1 = {
    "A<<B": 1, "A<B": 2, "A=B": 3, "A>B": 4, "A>>B": 5,
}
# vs_2 (A=Baseline, B=Test) -> invert perspective
VERDICT_TO_ORDINAL_VS2 = {
    "A>>B": 1, "A>B": 2, "A=B": 3, "A<B": 4, "A<<B": 5,
}
# Human (A=Test, B=Baseline) direct mapping
HUMAN_VERDICT_TO_ORDINAL = {
    "<<": 1, "<": 2, "=": 3, ">": 4, ">>": 5,
}

# ==================== Step 0: Load and Validate Data ====================
print("=" * 70)
print("STEP 0: Data Loading & Validation")
print("=" * 70)

# Load human data
with open(HUMAN_JSON, "r", encoding="utf-8") as f:
    human_data = json.load(f)

human_comparisons = human_data["comparisons"]
print(f"Human comparisons loaded: {len(human_comparisons)}")

# Load CSVs
model_data = {}
for name, fname in CSV_FILES.items():
    fpath = os.path.join(VS_DIR, fname)
    df = pd.read_csv(fpath, keep_default_na=False)
    model_data[name] = df
    print(f"  {name}: {len(df)} rows loaded")

# Validate vs_1/vs_2 verdict format
print("\nVerdict format validation:")
for name, df in model_data.items():
    vs1_set = set(df["vs_1"].dropna().unique())
    vs2_set = set(df["vs_2"].dropna().unique())
    vs1_bad = vs1_set - VALID_VERDICTS - {"error"}
    vs2_bad = vs2_set - VALID_VERDICTS - {"error"}
    if vs1_bad or vs2_bad:
        print(f"  ERROR {name}: vs_1 bad={vs1_bad}, vs_2 bad={vs2_bad}")
    else:
        vs1_err = "error" in vs1_set
        vs2_err = "error" in vs2_set
        status = []
        if vs1_err: status.append(f"vs_1 has 'error' ({df['vs_1'].eq('error').sum()} rows)")
        if vs2_err: status.append(f"vs_2 has 'error' ({df['vs_2'].eq('error').sum()} rows)")
        if status:
            print(f"  WARN {name}: {', '.join(status)}")
        else:
            print(f"  OK {name}")

# ==================== Step 1: Content Fingerprint Alignment ====================
print("\n" + "=" * 70)
print("STEP 1: Content Fingerprint Alignment")
print("=" * 70)


def make_fingerprint(row):
    """Create a content-based fingerprint for row matching."""
    tr = row.get("test_response", "")
    br = row.get("baseline_response", "")
    tr = "" if tr is None or tr != tr else str(tr).strip()
    br = "" if br is None or br != br else str(br).strip()
    if not tr and not br:
        return None
    # Use first 300 chars of each response
    return hashlib.md5((tr[:300] + "|" + br[:300]).encode("utf-8")).hexdigest()


# Build human fingerprint map
human_fp_map = {}
human_missing_fp = 0
for idx, comp in enumerate(human_comparisons):
    fp = make_fingerprint(comp)
    if fp:
        if fp in human_fp_map:
            # Duplicate fingerprint - append index to disambiguate
            fp = fp + f"_{idx}"
        human_fp_map[fp] = (idx, comp)
    else:
        human_missing_fp += 1

print(f"Human fingerprints: {len(human_fp_map)} unique, {human_missing_fp} missing (empty responses)")

# Build model fingerprint maps
model_fp_maps = {}
for name, df in model_data.items():
    fp_map = {}
    missing = 0
    for idx, row in df.iterrows():
        fp = make_fingerprint(row)
        if fp:
            # Handle potential duplicates within same CSV
            if fp in fp_map:
                fp = fp + f"_{idx}"
            fp_map[fp] = (idx, row)
        else:
            missing += 1
    model_fp_maps[name] = fp_map
    print(f"  {name}: {len(fp_map)} fingerprints, {missing} missing")

# Find common fingerprints
common_fps = set(human_fp_map.keys())
for name in model_fp_maps:
    common_fps &= set(model_fp_maps[name].keys())

print(f"\nCommon fingerprints (aligned across all 6 sources): {len(common_fps)}")
if len(common_fps) != len(human_comparisons) or len(common_fps) != 100:
    raise ValueError("Calibration must align all 100 items; inspect missing identities")
for fp in common_fps:
    _, human_row = human_fp_map[fp]
    for source, mapping in model_fp_maps.items():
        _, model_row = mapping[fp]
        for column in ("source_step", "test_id", "test_response", "baseline_response"):
            if str(human_row[column]).strip() != str(model_row[column]).strip():
                raise ValueError(f"Full calibration identity mismatch: {source}/{column}")

# ==================== Step 2: Build Verdict Matrix ====================
print("\n" + "=" * 70)
print("STEP 2: Build Verdict Matrix (Ordinal 1-5)")
print("=" * 70)

# For each common fingerprint, extract ordinals from all 6 sources
aligned_records = []
skip_reasons = {"human_na": 0, "model_error_vs1": 0, "model_error_vs2": 0, "human_missing_fp": 0}

for fp in sorted(common_fps):
    human_idx, human_comp = human_fp_map[fp]

    # Human ordinal
    human_verdict = human_comp.get("human_test:baseline", "").strip()
    human_ordinal = HUMAN_VERDICT_TO_ORDINAL.get(human_verdict, np.nan)
    if np.isnan(human_ordinal):
        skip_reasons["human_na"] += 1

    record = {
        "fingerprint": fp[:16],
        "source_step": human_comp.get("source_step", ""),
        "test_id": human_comp.get("test_id", ""),
        "human_ordinal": human_ordinal,
    }

    for name in SOURCES[:-1]:  # exclude human
        _, csv_row = model_fp_maps[name][fp]

        # vs_1 ordinal
        vs1 = str(csv_row.get("vs_1", "")).strip()
        if vs1 in ("error", "nan", ""):
            record[f"{name}_vs1_ord"] = np.nan
            skip_reasons["model_error_vs1"] += 1
        else:
            record[f"{name}_vs1_ord"] = VERDICT_TO_ORDINAL_VS1.get(vs1, np.nan)

        # vs_2 ordinal (inverted perspective)
        vs2 = str(csv_row.get("vs_2", "")).strip()
        if vs2 in ("error", "nan", ""):
            record[f"{name}_vs2_ord"] = np.nan
            skip_reasons["model_error_vs2"] += 1
        else:
            record[f"{name}_vs2_ord"] = VERDICT_TO_ORDINAL_VS2.get(vs2, np.nan)

        # Merge: int((s1 + s2) / 2 + 0.5) -- round-ties-away-from-zero
        s1 = record[f"{name}_vs1_ord"]
        s2 = record[f"{name}_vs2_ord"]
        if not np.isnan(s1) and not np.isnan(s2):
            record[f"{name}_ordinal"] = int((s1 + s2) / 2 + 0.5)
        elif not np.isnan(s1):
            record[f"{name}_ordinal"] = int(s1)  # fallback to vs_1 only
        elif not np.isnan(s2):
            record[f"{name}_ordinal"] = int(s2)  # fallback to vs_2 only
        else:
            record[f"{name}_ordinal"] = np.nan

    aligned_records.append(record)

df_aligned = pd.DataFrame(aligned_records)
print(f"Aligned comparisons: {len(df_aligned)}")
print(f"  Human NA: {skip_reasons['human_na']}")
print(f"  Model error vs_1: {skip_reasons['model_error_vs1']}")
print(f"  Model error vs_2: {skip_reasons['model_error_vs2']}")

# Build verdict_matrix: 6 columns (5 models + human), aligned rows
verdict_matrix = pd.DataFrame()
verdict_matrix["human"] = df_aligned["human_ordinal"]
for name in SOURCES[:-1]:
    verdict_matrix[name] = df_aligned[f"{name}_ordinal"]

# Drop rows where all values are NaN
valid_rows = verdict_matrix.notna().any(axis=1)
verdict_matrix = verdict_matrix[valid_rows].copy()
print(f"\nFinal verdict matrix: {verdict_matrix.shape[0]} rows x {verdict_matrix.shape[1]} columns")
print(f"Valid rows (at least one source non-NaN): {verdict_matrix.shape[0]}")

# ==================== Part 1: Verdict Distribution ====================
print("\n" + "=" * 70)
print("PART 1: Verdict Distribution (1~5 ordinal ratio)")
print("=" * 70)

ordinal_labels = {1: "Test<<Base", 2: "Test<Base", 3: "Tie", 4: "Test>Base", 5: "Test>>Base"}
for source in SOURCES:
    col = "human" if source == "human" else source
    vals = verdict_matrix[col].dropna()
    total = len(vals)
    if total == 0:
        print(f"{source:20s}: NO DATA")
        continue
    parts = []
    for k in range(1, 6):
        cnt = (vals == k).sum()
        pct = cnt / total * 100
        parts.append(f"{ordinal_labels[k]}={cnt:2d}({pct:5.1f}%)")
    print(f"{source:20s}: {', '.join(parts)}")

# ==================== Part 2: 6x6 Spearman Correlation Matrix ====================
print("\n" + "=" * 70)
print("PART 2: 6x6 Spearman Rank Correlation Matrix")
print("=" * 70)

n_sources = len(SOURCES)
corr_matrix_values = np.zeros((n_sources, n_sources))

for i, s1 in enumerate(SOURCES):
    for j, s2 in enumerate(SOURCES):
        c1 = "human" if s1 == "human" else s1
        c2 = "human" if s2 == "human" else s2
        v1 = verdict_matrix[c1].values
        v2 = verdict_matrix[c2].values
        valid_mask = ~(np.isnan(v1) | np.isnan(v2))
        v1_valid = v1[valid_mask]
        v2_valid = v2[valid_mask]
        if len(v1_valid) >= 3:  # minimum for Spearman
            corr, _ = stats.spearmanr(v1_valid, v2_valid)
            corr_matrix_values[i, j] = corr
        else:
            corr_matrix_values[i, j] = np.nan

corr_matrix = pd.DataFrame(corr_matrix_values, index=SOURCES, columns=SOURCES)
print(corr_matrix.round(3).to_string())

# ==================== Part 3: Weight Calculation ====================
print("\n" + "=" * 70)
print("PART 3: Weight Calculation (Method B: r x acc)")
print("=" * 70)

results = []
for model in SOURCES:
    if model == "human":
        continue

    v_model = verdict_matrix[model].values
    v_human = verdict_matrix["human"].values

    valid_mask = ~(np.isnan(v_model) | np.isnan(v_human))
    v_model_valid = v_model[valid_mask]
    v_human_valid = v_human[valid_mask]
    n = len(v_model_valid)

    if n < 3:
        print(f"  {model}: insufficient valid pairs ({n})")
        continue

    # Spearman correlation with human
    r, pval = stats.spearmanr(v_model_valid, v_human_valid)

    # Weighted accuracy (acc_n): delta = |ordinal difference|
    delta = np.abs(v_model_valid - v_human_valid)
    weighted_sum = (
        np.sum(delta == 0) * 1.0
        + np.sum(delta == 1) * 0.5
        + np.sum(delta >= 2) * 0.0
    )
    acc = weighted_sum / n

    # Final score
    score = max(r, 0.0) * acc

    results.append({
        "model": model,
        "n": n,
        "r_spearman": r,
        "p_value": pval,
        "significant": pval < 0.05,
        "acc_weighted": acc,
        "score": score,
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Normalize weights
total_score = results_df["score"].sum()
results_df["omega"] = results_df["score"] / total_score if total_score > 0 else 0

print()
print("=" * 70)
print("FINAL WEIGHTS (omega_n)")
print("=" * 70)
for _, row in results_df.iterrows():
    sig_mark = "*" if row["significant"] else ""
    print(
        f"  {row['model']:20s}: omega = {row['omega']:.4f}  "
        f"(r={row['r_spearman']:.3f}, acc={row['acc_weighted']:.3f}){sig_mark}"
    )
print(f"  Sum of weights: {results_df['omega'].sum():.4f}")

# ==================== Save Results ====================
if not results_df["n"].eq(100).all():
    raise ValueError("Each judge must have 100 valid merged comparisons")
results_df.to_csv(os.path.join(VS_DIR, "weight_calculation_results.csv"), index=False)
corr_matrix.to_csv(os.path.join(VS_DIR, "spearman_correlation_matrix.csv"))
print()
print("Results saved to:")
print("  - weight_calculation_results.csv")
print("  - spearman_correlation_matrix.csv")

# ==================== Extra: pairwise comparison counts ====================
print("\n" + "=" * 70)
print("APPENDIX: Pairwise Valid Comparison Counts")
print("=" * 70)
count_matrix = np.zeros((n_sources, n_sources), dtype=int)
for i, s1 in enumerate(SOURCES):
    for j, s2 in enumerate(SOURCES):
        c1 = "human" if s1 == "human" else s1
        c2 = "human" if s2 == "human" else s2
        v1 = verdict_matrix[c1].values
        v2 = verdict_matrix[c2].values
        valid_mask = ~(np.isnan(v1) | np.isnan(v2))
        count_matrix[i, j] = valid_mask.sum()
count_df = pd.DataFrame(count_matrix, index=SOURCES, columns=SOURCES)
print(count_df.to_string())
