import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

# File paths
files = {
    'gemma4_e4b': 'Test_result_gemma4_e4b.csv',
    'llama3.1_8b': 'Test_result_llama3.1_8b.csv',
    'ministral3_8b': 'Test_result_ministral3_8b.csv',
    'olmo3_7b': 'Test_result_olmo3_7b.csv',
    'qwen3.5_9b': 'Test_result_qwen3.5_9b.csv',
    'human': 'Test_result_human.csv'
}

# Load all files
data = {}
for name, filepath in files.items():
    df = pd.read_csv(BASE_DIR / filepath, keep_default_na=False)
    if len(df) != 100 or not df.penalty_id.is_unique:
        raise ValueError(f"Expected 100 unique penalty items: {filepath}")
    data[name] = df[['penalty_id', 'verdict']].copy()

# Common penalty_ids (all 100 items now that human file is fixed)
common_ids = sorted(set(data['gemma4_e4b']['penalty_id']))

# verdict -> ordinal: hard=1, soft=2, nice=3
verdict_map = {'hard': 1, 'soft': 2, 'nice': 3}
sources = ['gemma4_e4b', 'llama3.1_8b', 'ministral3_8b', 'olmo3_7b', 'qwen3.5_9b', 'human']

# Build aligned verdict matrix - ensure numeric
verdict_dict = {}
for source in sources:
    verdict_dict[source] = {}
    df = data[source]
    for _, row in df.iterrows():
        pid = row['penalty_id']
        if pid in common_ids:
            verdict_dict[source][pid] = verdict_map.get(row['verdict'], np.nan)

# Create DataFrame with proper numeric dtype
verdict_matrix = pd.DataFrame(verdict_dict)
verdict_matrix = verdict_matrix.astype(float)
if len(verdict_matrix) != 100 or not verdict_matrix.notna().all().all():
    raise ValueError("All judges and the human must align on 100 valid penalty items")

print('Verdict matrix dtypes:')
print(verdict_matrix.dtypes)
print()

# ==================== Part 1: Descriptive Statistics ====================
print('='*70)
print('PART 1: Verdict Distribution (hard:soft:nice ratio)')
print('='*70)
for source in sources:
    counts = verdict_matrix[source].value_counts().sort_index()
    total = counts.sum()
    hard = counts.get(1.0, 0)
    soft = counts.get(2.0, 0)
    nice = counts.get(3.0, 0)
    print(f'{source:20s}: hard={hard:2d} ({hard/total*100:5.1f}%), soft={soft:2d} ({soft/total*100:5.1f}%), nice={nice:2d} ({nice/total*100:5.1f}%)')

# ==================== Part 2: Spearman Correlation Matrix ====================
print()
print('='*70)
print('PART 2: 6x6 Spearman Rank Correlation Matrix')
print('='*70)

n_sources = len(sources)
corr_matrix_values = np.zeros((n_sources, n_sources))

for i, s1 in enumerate(sources):
    for j, s2 in enumerate(sources):
        v1 = verdict_matrix[s1].values
        v2 = verdict_matrix[s2].values
        valid_mask = ~(np.isnan(v1) | np.isnan(v2))
        v1_valid = v1[valid_mask]
        v2_valid = v2[valid_mask]
        if len(v1_valid) > 0:
            corr, _ = stats.spearmanr(v1_valid, v2_valid)
            corr_matrix_values[i, j] = corr
        else:
            corr_matrix_values[i, j] = np.nan

corr_matrix = pd.DataFrame(corr_matrix_values, index=sources, columns=sources)
print(corr_matrix.round(3).to_string())

# ==================== Part 3: Weights Calculation ====================
print()
print('='*70)
print('PART 3: Weight Calculation (Method B: r x acc)')
print('='*70)

results = []
for model in sources:
    if model == 'human':
        continue

    v_model = verdict_matrix[model].values
    v_human = verdict_matrix['human'].values

    valid_mask = ~(np.isnan(v_model) | np.isnan(v_human))
    v_model_valid = v_model[valid_mask]
    v_human_valid = v_human[valid_mask]
    n = len(v_model_valid)

    # Spearman correlation with human
    r, pval = stats.spearmanr(v_model_valid, v_human_valid)

    # Weighted accuracy (acc_n)
    delta = np.abs(v_model_valid - v_human_valid)
    weighted_sum = np.sum(delta == 0) * 1.0 + np.sum(delta == 1) * 0.5 + np.sum(delta == 2) * 0.0
    acc = weighted_sum / n

    # Final score
    score = max(r, 0) * acc

    results.append({
        'model': model,
        'n': n,
        'r_spearman': r,
        'p_value': pval,
        'significant': pval < 0.05,
        'acc_weighted': acc,
        'score': score
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Normalize weights
total_score = results_df['score'].sum()
results_df['omega'] = results_df['score'] / total_score if total_score > 0 else 0

print()
print('='*70)
print('FINAL WEIGHTS (omega_n)')
print('='*70)
for _, row in results_df.iterrows():
    sig_mark = '*' if row['significant'] else ''
    print(f"  {row['model']:20s}: omega = {row['omega']:.4f}  (r={row['r_spearman']:.3f}, acc={row['acc_weighted']:.3f}){sig_mark}")

print(f"  Sum of weights: {results_df['omega'].sum():.4f}")

# Save results to CSV
results_df.to_csv(BASE_DIR / 'weight_calculation_results.csv', index=False)
corr_matrix.to_csv(BASE_DIR / 'spearman_correlation_matrix.csv')
print()
print('Results saved to:')
print('  - weight_calculation_results.csv')
print('  - spearman_correlation_matrix.csv')
