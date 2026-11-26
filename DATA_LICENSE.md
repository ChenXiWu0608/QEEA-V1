# Data License & Data-Publication Rules

The **data** in this repository (all inference outputs, original power records,
annotated datasets, saved LLM-judgement records, calibrated weights, and computed
rankings) is published under the **Creative Commons Attribution 4.0 International
License (CC BY 4.0)** — <https://creativecommons.org/licenses/by/4.0/>.

## What CC BY 4.0 means in practice

You are free to copy, redistribute, remix, transform, and build upon the data
for **any purpose, including commercial use**, as long as you:

1. **Give appropriate credit** — cite the author(s) and the two reports below.
2. **Indicate if changes were made** — if you derive new quantities, state that
   they are derived and how.
3. **Do not imply endorsement** — you may not suggest that the authors endorse you
   or your use without their permission.

There are **no restrictions** on use beyond the above. The data contains only
model inference outputs and instrument power records; it contains **no personal
data** and is therefore published without additional access controls.

## Attribution (recommended)

> Wu Chenxi, Tominobe Laboratory, Graduate School of Environment and Energy,
> Waseda University — *Open-Source LLM Cost-Optimized Inference* (Part 1 and
> Part 2). Data: <URL of this repository>.

## Third-party components (carve-out)

Small portions of this repository originate from three public benchmarks — **GPQA**
(CC BY 4.0), **LiveCodeBench** (upstream "cc"; problem statements © LeetCode /
AtCoder / Codeforces), and **Arena-Hard v2** (Apache-2.0). Those portions remain under
their original licenses and copyrights and are **not** licensed here. Full attribution,
sources, and compliance notes (removed full GPQA copy; stripped LiveCodeBench test
cases): see [`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md).

## Shared with the first report

The raw inference and energy records in `RAW/`, `Dataset/`, `Baseline/`, and
`Test_file/` are the **same records** used in the already-published first report:

- Wu Chenxi, Tominobe Shin. *Open-Source LLMのコスト最適化推論（第一報）：圧縮技術と推論モードがエネルギー効率に与える影響の評価*.
  エネルギー・資源学会論文誌 (Journal of the Japan Society of Energy and Resources), 47(4): 117–126, 2026.
  DOI: <https://doi.org/10.24778/jjser.47.4_117>

The evaluation pipeline (Steps 1–5, the quality dimension, and the EQ/EQEE
framework) is introduced in the second report, which is **under review**:

- Wu Chenxi. *Open-Source LLMのコスト最適化推論第二報：QEEA-品質エネルギー効率評価フレームワーク（Quality-Energy Efficiency Assessment）*.
  (DOI to be added upon acceptance.)

When citing a **specific figure or number**, cite the report in which that figure
appears. The raw energy data is common to both; the quality scoring and final
EQ/EQEE rankings are unique to the second report.

## Provenance

- Original inference was run locally (Ollama) and power was captured with a PMD2
  power analyzer; see `Test_file/` (runners) and `RAW/` (records).
- LLM-judgement records were generated with the scripts preserved in this
  repository (see the "Live LLM scripts" section of `README.md`); they are kept
  for provenance and prompt reference but are not needed to reproduce the
  published numbers.
- All numbers in `step5_eqe_output/` are recomputed offline from the saved
  records via `python run_pipeline.py`; see `pipeline_verification.json` for the
  verification manifest.
