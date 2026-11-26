# Open-Source LLM Cost-Optimized Inference (Part 2)
## QEEA: Quality–Energy Efficiency Assessment — Data and Evaluation Pipeline

**公開データ・評価パイプライン（Data & Evaluation Pipeline for the second report）**

| | |
|---|---|
| **Author / 著者** | Wu Chenxi 呉晨㬢 — Tominobe Laboratory, Graduate School of Environment and Energy, Waseda University（早稲田大学 大学院環境・エネルギー研究科 納富研究室） |
| **Google Scholar** | <https://scholar.google.com/citations?user=E4lgbP0AAAAJ&hl> |
| **Status** | This data archive supports the **second report (under review / 査読中)**: *Open-Source LLMのコスト最適化推論第二報：QEEA-品質エネルギー効率評価フレームワーク（Quality-Energy Efficiency Assessment）* |
| **Shared data with** | The **first report (published / 既掲載)**: *Open-Source LLMのコスト最適化推論（第一報）：圧縮技術と推論モードがエネルギー効率に与える影響の評価* — エネルギー・資源学会論文誌 **47(4): 117–126, 2026**, <https://doi.org/10.24778/jjser.47.4_117> |

---

# English

## What this repository is

This repository publishes the **complete data and evaluation pipeline** behind the second report, *QEEA — Quality-Energy Efficiency Assessment*. It contains:

1. **Raw inference and energy data** — model outputs and original power records for **22 open-source LLM configurations** (model × quantization × reasoning mode) across **two independent inference rounds** (`reports1` / `reports2`) and **five task areas**:
   - Step 1 benchmarks: **Arena-Hard, LiveCodeBench, GPQA**
   - Step 2: **long-text generation**
   - Step 3: **agentic workflow** (Dify)
2. **Saved LLM-judgement records** — 5 judge models (`gemma4_e4b`, `llama3.1_8b`, `ministral3_8b`, `olmo3_7b`, `qwen3.5_9b`) calibrated against 100 human-labelled items each for win/loss and penalty; bidirectional comparisons against 4 baseline anchors (`qwen3.5-4b` → weak, `qwen3.5-9b` → mid, `qwen3.5-35b` → strong, `glm-4.7-flash` → cross).
3. **The offline evaluation pipeline** — recomputes every published number (22 configurations × 2 rounds × 5 areas = **220 scoring cells**) from the saved records with **no model API calls**.

### Relationship to the first report (data sharing)

The **raw inference and energy records** in `RAW/`, `Dataset/`, `Baseline/`, and `Test_file/` are the **same records** used in the already-published first report, which analyzed how quantization (FP16 → Q8 → Q4) and reasoning mode affect energy consumption. The second report adds the **quality dimension** (LLM-judge-based scoring, calibration, and the EQ / EQEE framework) on top of those identical energy records. If you only need the energy analysis, the first report and the same raw data are sufficient; the evaluation pipeline (Steps 1–5) is new in the second report.

## Reproducing the results

The pipeline is fully offline: it reads saved judge records and energy logs, and never calls a model API.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

- Runs the six pipeline scripts in order and then executes `verify()`.
- `verify()` re-checks cell counts (220 cells, 22 configurations, 5 judge weights), recomputed values, and SHA-256 hashes of the pinned input files, and writes `step5_eqe_output/pipeline_verification.json`.
- Current headline results (as of the 2026-10-27 revision): top **EQ** = Qwen3-30B-A3B Q4 (90.7002); top **EQEE** = Gemma3-4B Q4 (825.9255 EQ/MJ). See `step5_eqe_output/final_rankings.csv` and `step5_eqe_output/CURRENT_RESULTS.md`.

## Repository layout

```
run_pipeline.py            Entry point: runs Steps 1–5 + verify()
pipeline_sources.py        Shared module (judge list, aliases, helpers, AST-extracted word helpers)
requirements.txt           Python dependencies
evaluation_pipeline.md     Current pipeline specification (Japanese, 2026-10-27)
PROMPTS.md                 Index of judge prompts / judgement criteria (reviewer Q10)

step1_Test calibration/    100-item calibration sets, judge & penalty weight results, chart export
step2.1_Penalty/           Length-penalty saved judgements, length_protocol.json, Penalty.json (prompts)
step2.2_vs_baseline/       Saved bidirectional judgements (22 configs × 2 rounds × 4 baselines × 5 judges)
step3_ability_output/      Ability scoring output (7,040 matched same-question judgements selected)
step4_q_total_output/      Q_total output (WA − WP → EQraw → per-area Min-Max normalization → EQ)
step5_eqe_output/          Final EQ/EQEE, energy integration, rankings, chart_data/, verification JSON
Dataset/Step{1,2,3}/       Annotated inputs (responses + energy), test_power1/ = raw power samples
Baseline/Step{1,2,3}/      Baseline anchor annotations (4 models per step)
RAW/                       Original inference + PMD2 power records
Test_file/                 Inference runners, benchmark datasets (LiveCodeBench, GPQA, Arena-Hard picks)
scripts/                   Data-preparation scripts (RAW → annotated Dataset)
length_rejudge_20261026/   Length re-judgement run (81 responses / 405 judgement units)
archive/legacy_code/       Dead one-off scripts, archived (see archive/LEGACY_README.md)
```

## Judge prompts (判定プロンプトの所在)

The complete judgement criteria and prompts given to the evaluation models are in this
repository; see **[PROMPTS.md](PROMPTS.md)** for exact file paths:

- Win/loss (勝敗判定) system & user prompts: `step2.2_vs_baseline/dataset_vs_baseline_config.py`,
  `step1_Test calibration/comparison_config.py` (`SYSTEM_PROMPT`, `build_user_prompt()`).
- Usability (利用可能性判定) prompts, all 14 task categories: `step2.1_Penalty/Penalty.json`
  (`system_prompt` / `user_prompt_template`), assembled by `step2.1_Penalty/penalty_evaluator_batch.py`.
- Saved judge outputs (prompt + verdict + reason per judgement unit): `step2.1_Penalty/`, `step2.2_vs_baseline/`.

## Evaluation steps

| Step | What happens | Key rule |
|---|---|---|
| **1 — Calibration** | 100 human-labelled items calibrate 5 judge models' win/loss and penalty weights. | Weight ∝ `max(Spearman r, 0) × weighted agreement`, normalized. |
| **2.1 — Penalty** | Length-ratio pre-screening of long-text responses, plus LLM judges for in-band deviations. | `r < 0.5` or `r > 1.5` → hard (no LLM call); soft band `0.5–0.75` / `1.25–1.5` → judges notified; empty/failed → hard. Coefficients: hard = 3, soft = 1, nice = 0. |
| **2.2 — vs. baseline** | Bidirectional comparison of each configuration against 4 baseline anchors, aggregated with difficulty coefficients. | A≫B = 3, A>B = 1, A=B = 0.5, loss = 0. Workflow: 8 questions, code generation excluded. |
| **3 — Ability** | Match identical question + response text across rounds; per-judge ability scores. | Of 7,040 matched judgements, 1 (both directions invalid) is excluded from the average. |
| **4 — Quality EQ** | `WA` (weight-averaged ability) minus `WP` (weight-averaged penalty) → `EQraw`, Min-Max normalized per area over 22×2 values. | `EQraw = WA − WP`; 5 areas × 22 configs × 2 rounds = 44 values per area. |
| **5 — Energy efficiency EQEE** | Integrate original power logs over kept execution windows (including failed trials); `EQEE = EQ / Energy[J] × 10⁶`. | Unit EQ/MJ; per-area, per-round values averaged with equal weight. Final score is **not** total quality ÷ total energy. |

## Documented scope and exclusions

These are intentional, documented choices — the raw data for all of them is still present in this repository:

- **`*_excluded_from_scope/` directories** (`Dataset/Step2/`, `RAW/Step２_Long-text-generation_Data/`, `step2.1_Penalty/Step2/`): the Step-2 `reports3` round is pre-final-round long-text data, **excluded from the paper's two-round scope**. Kept for completeness; never read by the offline pipeline.
- **Code-generation tasks** are excluded from scoring (Step 3/4 rule); their time windows are excluded from energy integration.
- **`deepseek-moe-16b-base`**: inferred and fully present in raw/judgement data, but **excluded from the final EQ/EQEE scoring** (22 of the 23 inferred configurations are scored).
- **One unusable judgement** (both directions invalid) is excluded from the Step-3 average per the pre-existing rule.

## Live LLM scripts (original runs only)

The following scripts were used to *generate* the saved records with local Ollama models. They are preserved for provenance (including their LLM prompts) but **cannot be re-run reproducibly** — the offline pipeline does not need them:

- `step2.1_Penalty/penalty_evaluator_batch.py` (also the AST source of four word-helper functions used by `pipeline_sources.py`)
- `step2.2_vs_baseline/dataset_vs_baseline_comparison.py`
- `step1_Test calibration/vs/Test_vs_baseline_comparison.py`
- `Baseline/Step3/recover_translation_ollama.py`

## Archived dead code

`archive/legacy_code/` holds six superseded one-off scripts (early debugging, an obsolete aggregate/split pair, and a stale data-prep script). See `archive/LEGACY_README.md` for details. They are not imported by the pipeline.

## Citation

If you use this data or pipeline, please cite **both** reports:

```bibtex
@article{wu2026part1,
  author  = {Wu, Chenxi and Tominobe, Shin},
  title   = {Open-Source LLM Cost-Optimized Inference (Part 1): Evaluation of the Effect of
             Compression Techniques and Inference Modes on Energy Efficiency
             (Open-Source LLMのコスト最適化推論（第一報）：圧縮技術と推論モードがエネルギー効率に与える影響の評価)},
  journal = {Journal of the Japan Society of Energy and Resources
             (エネルギー・資源学会論文誌)},
  volume  = {47},
  number  = {4},
  pages   = {117--126},
  year    = {2026},
  doi     = {10.24778/jjser.47.4_117},
}

@misc{wu2026qeea,
  author = {Wu, Chenxi},
  title  = {Open-Source LLM Cost-Optimized Inference (Part 2):
            QEEA -- Quality-Energy Efficiency Assessment
            (Open-Source LLMのコスト最適化推論第二報：QEEA-品質エネルギー効率評価フレームワーク)},
  year   = {2026},
  note   = {Under review. Data and evaluation pipeline: this repository},
}
```

> The BibTeX entry for the second report will gain a DOI once accepted; please update citations accordingly.

## License and data-publication rules

- **Code**: MIT License — see [`LICENSE`](LICENSE).
- **Data**: CC BY 4.0 — see [`DATA_LICENSE.md`](DATA_LICENSE.md). Attribution: Wu Chenxi, Tominobe Laboratory, Waseda University; the two reports above.
- The repository contains only model inference outputs and instrument power records — no personal data.
- Derived data (judge records, calibrated weights, computed rankings) are published under the same terms as the raw data.
- Data is deliberately shared between the two reports; when citing a specific figure, cite the report in which that figure appears.

---

# 日本語

## 本リポジトリについて

本リポジトリは、第二報 *Open-Source LLMのコスト最適化推論第二報：QEEA-品質エネルギー効率評価フレームワーク（Quality-Energy Efficiency Assessment）*（**査読中**）の**全データと評価パイプライン**を公開するものです。著者は早稲田大学 大学院環境・エネルギー研究科 納富研究室の呉晨㬢です（[Google Scholar](https://scholar.google.com/citations?user=E4lgbP0AAAAJ&hl)）。

収録内容：

1. **原始の推論・エネルギーデータ** — オープンソースLLM 22構成（モデル×量子化×推論モード）× 2回独立推論（`reports1`/`reports2`）× 5区分（Arena-Hard、LiveCodeBench、GPQA、長文生成、エージェントワークフロー）のモデル出力と原電力記録。
2. **保存済みLLM判定記録** — 5裁判モデル（gemma4_e4b, llama3.1_8b, ministral3_8b, olmo3_7b, qwen3.5_9b）の勝敗・減点100件校正と、4基準モデル（qwen3.5-4b→弱, qwen3.5-9b→中, qwen3.5-35b→強, glm-4.7-flash→異系統）との双方向比較。
3. **オフライン評価パイプライン** — 保存済み判定から全数値（22構成×2回×5区分＝**220セル**）を再計算。モデルAPI呼び出しは行わない。

## 第一報とのデータ共有

`RAW/`、`Dataset/`、`Baseline/`、`Test_file/` の**原始推論・エネルギー記録は、既掲載の第一報と同一**です。第一報（エネルギー・資源学会論文誌 47(4): 117–126, 2026, [DOI: 10.24778/jjser.47.4_117](https://doi.org/10.24778/jjser.47.4_117)）は量子化（FP16→Q8→Q4）と推論モードのエネルギー消費への影響を分析しており、第二報（QEEA）はその同一エネルギー記録の上に**品質評価（LLM判定・校正・EQ/EQEE枠組み）**を新たに付加したものです。

## 再現手順

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py
```

`run_pipeline.py` は6つのスクリプトを順次実行し、最後に `verify()` でセル数・数値・入力ファイルのSHA-256を検証します。モデルAPIは呼び出しません。

## 判定プロンプトの所在

評価モデルに与えた判定基準とプロンプトの全文は **[PROMPTS.md](PROMPTS.md)** に所在を示します：

- 勝敗判定：`step2.2_vs_baseline/dataset_vs_baseline_config.py` および `step1_Test calibration/comparison_config.py`（`SYSTEM_PROMPT` と `build_user_prompt()`）
- 利用可能性判定（14タスクカテゴリ全件）：`step2.1_Penalty/Penalty.json`（`system_prompt` / `user_prompt_template`）、組み立てロジックは `step2.1_Penalty/penalty_evaluator_batch.py`
- 全判定ユニットのプロンプト・判定結果・理由の記録：`step2.1_Penalty/`、`step2.2_vs_baseline/`

## 引用

本データ・パイプラインを利用される場合は、**両論文の引用**をお願いします：

- 第一報（既掲載）：呉晨曦, 納富信. Open-Source LLMのコスト最適化推論（第一報）：圧縮技術と推論モードがエネルギー効率に与える影響の評価. エネルギー・資源学会論文誌, 47(4), 117–126, 2026. DOI: 10.24778/jjser.47.4_117
- 第二報（査読中）：呉晨㬢. Open-Source LLMのコスト最適化推論第二報：QEEA-品質エネルギー効率評価フレームワーク（Quality-Energy Efficiency Assessment）. 受理後にDOIを追加します。

BibTeXは上記英語セクション参照。

## ライセンス・データ公開ルール

- **コード**：MIT License（[`LICENSE`](LICENSE)）
- **データ**：CC BY 4.0（[`DATA_LICENSE.md`](DATA_LICENSE.md)）— 表示: 呉晨㬢, 早稲田大学大学院環境・エネルギー研究科 納富研究室。上記2論文への言及を推奨。
- 本リポジトリはモデルの推論出力と計測電力記録のみを含み、個人情報を含みません。
- 導出データ（判定記録・校正重み・ランキング）も原始データと同条項で公開します。
- データは両論文で意図的に共有されています。特定の数値を引用する場合は、その数値が登場する論文を引用してください。

## 範囲の明示（除外事項）

- `*_excluded_from_scope/`（Step2の`reports3`）：最終ラウンド以前の長文データであり、論文の2ラウンド範囲から**明示除外**。完全性のため保管し、オフラインパイプラインは読みません。
- コード生成課題はスコアリング・エネルギー積分から除外（Step3/4ルール）。
- `deepseek-moe-16b-base`：推論・判定データには完全収録されているが、最終EQ/EQEEスコアリングからは除外（23構成中22構成をスコアリング）。
- 双方向とも無効の判定1件は既存規則によりStep3平均から除外。
