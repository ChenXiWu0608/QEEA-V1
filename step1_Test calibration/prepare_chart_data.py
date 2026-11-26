"""Export current chart CSVs from the original Step1--5 outputs.

Historical chart_data PNG/Origin files are retained. Current numeric chart data
are written to step5_eqe_output/chart_data; no audit output is read.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'step5_eqe_output/chart_data'

MODEL_DISPLAY = {
    "qwen3_8b_fp16_false":                      "Qwen3-8B FP16",
    "qwen3_8b_fp16_true":                        "Qwen3-8B FP16 (T)",
    "qwen3_8b_q8_0_false":                       "Qwen3-8B Q8",
    "qwen3_8b_q8_0_true":                        "Qwen3-8B Q8 (T)",
    "qwen3_32b_q4_K_M_false":                    "Qwen3-32B Q4",
    "qwen3_32b_q4_K_M_true":                     "Qwen3-32B Q4 (T)",
    "qwen3_30b_a3b_instruct_2507_q4_K_M_false":  "Qwen3-30B-A3B Q4",
    "qwen3_30b_a3b_thinking_2507_q4_K_M_true":   "Qwen3-30B-A3B Q4 (T)",
    "gemma3_4b_it_fp16_false":                   "Gemma3-4B FP16",
    "gemma3_4b_it_q8_0_false":                   "Gemma3-4B Q8",
    "gemma3_4b_it_q4_K_M_false":                 "Gemma3-4B Q4",
    "gemma3_4b_it_qat_false":                    "Gemma3-4B QAT",
    "deepseek_r1_14b_qwen_distill_q8_0_false":   "DS-R1-14B Q8",
    "deepseek_r1_14b_qwen_distill_q8_0_true":    "DS-R1-14B Q8 (T)",
    "deepseek_r1_7b_qwen_distill_fp16_false":    "DS-R1-7B FP16",
    "deepseek_r1_7b_qwen_distill_fp16_true":     "DS-R1-7B FP16 (T)",
    "deepseek_r1_8b_0528_qwen3_fp16_false":      "DS-R1-8B FP16",
    "deepseek_r1_8b_0528_qwen3_fp16_true":       "DS-R1-8B FP16 (T)",
    "deepseek_r1_8b_0528_qwen3_q8_0_false":      "DS-R1-8B Q8",
    "deepseek_r1_8b_0528_qwen3_q8_0_true":       "DS-R1-8B Q8 (T)",
    "deepseek_r1_8b_llama_distill_fp16_false":   "DS-R1-8B-L FP16",
    "deepseek_r1_8b_llama_distill_fp16_true":    "DS-R1-8B-L FP16 (T)",
}

JUDGE_DISPLAY = {
    "gemma4_e4b":    "Gemma4-E4B",
    "llama3.1_8b":   "Llama3.1-8B",
    "ministral3_8b": "Ministral3-8B",
    "olmo3_7b":      "OLMo3-7B",
    "qwen3.5_9b":    "Qwen3.5-9B",
}
MODEL_DISPLAY = {k.lower(): v for k, v in MODEL_DISPLAY.items()}
SUBSTEPS = ['ArenaHard','GPQA','LiveCodeBench','Step2','Step3']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    win = pd.read_csv(ROOT/'step1_Test calibration/vs/weight_calculation_results.csv')
    pen = pd.read_csv(ROOT/'step1_Test calibration/Penalty test/weight_calculation_results.csv')
    fields = ['model','r_spearman','acc_weighted','omega']
    w = win[fields].rename(columns={'model':'judge','r_spearman':'r_win','acc_weighted':'acc_win','omega':'omega_win'})
    p = pen[fields].rename(columns={'model':'judge','r_spearman':'r_pen','acc_weighted':'acc_pen','omega':'omega_pen'})
    weights = p.merge(w,on='judge',validate='one_to_one')
    weights['judge_display'] = weights.judge.map(JUDGE_DISPLAY)
    weights.to_csv(OUT/'fig1_omega_comparison.csv',index=False)
    cells = pd.read_csv(ROOT/'step5_eqe_output/eqee_cells.csv')
    cells['model'] = cells.merge_key
    cells['EQEE_gpu'] = cells.EQE_gpu*1e6
    final = pd.read_csv(ROOT/'step5_eqe_output/final_rankings.csv').set_index('model')
    for measure,stem in [('EQ','fig2_qtotal'),('EQEE','fig3_eqe')]:
        stats = cells.groupby(['model','substep'])[measure].agg(['mean','std']).reset_index()
        means = stats.pivot(index='model',columns='substep',values='mean')[SUBSTEPS]
        stds = stats.pivot(index='model',columns='substep',values='std')[SUBSTEPS]
        heat = means.copy()
        heat['mean_all'] = final[measure]
        heat['model_display'] = heat.index.map(MODEL_DISPLAY)
        heat = heat.sort_values('mean_all',ascending=False)
        heatname = 'fig2a_qtotal_heatmap.csv' if measure == 'EQ' else 'fig3a_eqe_total_heatmap.csv'
        heat[['model_display']+SUBSTEPS+['mean_all']].to_csv(OUT/heatname,index=False)
        data = pd.DataFrame(index=means.index)
        for step in SUBSTEPS:
            data[step] = means[step]
            data[step+'_std'] = stds[step]
        data['mean_all'] = final[measure]
        # Standard deviation of each repeat's five-task average, including
        # cross-task covariance; RMS of individual task SDs is not equivalent.
        data['mean_all_std'] = final[measure+'_std']
        data['model_display'] = data.index.map(MODEL_DISPLAY)
        data.sort_values('mean_all',ascending=False).to_csv(OUT/(stem+'_data.csv'),index_label='model')
    rank = final.copy()
    rank['model_display'] = rank.index.map(MODEL_DISPLAY)
    rank = rank.rename(columns={'EQ':'Q_mean','EQ_std':'Q_std'})
    rank.sort_values('Q_mean',ascending=False).to_csv(OUT/'fig2b_qtotal_ranking.csv',index_label='model')
    gpu_repeat = cells.groupby(['model','report']).EQEE_gpu.mean().reset_index()
    gpu = gpu_repeat.groupby('model').EQEE_gpu.agg(['mean','std']).rename(columns={'mean':'EQE_gpu_mean','std':'EQE_gpu_std'})
    energy = cells.groupby('model')[['total_energy_joule','gpu_energy_joule']].sum()
    rank = rank.join(gpu)
    rank['nonGPU_pct'] = (1-energy.gpu_energy_joule/energy.total_energy_joule)*100
    rank = rank.rename(columns={'EQEE':'EQE_total_mean','EQEE_std':'EQE_total_std'})
    rank.sort_values('EQE_total_mean',ascending=False).to_csv(OUT/'fig3_eqe_ranking.csv',index_label='model')
    if rank.model_display.isna().any() or len(rank)!=22:
        raise ValueError('Chart metadata does not cover all 22 configurations')
    print(f'Exported current chart tables: {OUT}')


if __name__ == '__main__':
    main()
