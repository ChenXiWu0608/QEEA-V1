# Third-Party Data / サードパーティデータ

This repository redistributes small portions of three public benchmarks. Those parts
remain under their **original licenses and copyrights**; the repository-wide MIT (code)
and CC BY 4.0 (our data) licenses do **not** apply to them. Papers citing this work
should cite the benchmark originals listed below (the second report already does).

本リポジトリは3つの公開ベンチマークの一部を含みます。該当部分はそれぞれの**原典のライセンス・著作権**に帰属し、本リポジトリの MIT／CC BY 4.0 は適用されません。

## 1. GPQA

- **Where used:** `Test_file/Step1_Benchmark/datasets/gpqa_picked.jsonl` (48 selected
  questions) and the prompt column of the Step-1 records under `Dataset/Step1/` and
  `RAW/` (unavoidable parts of the experiment record).
- **Original:** Rein et al., *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*,
  arXiv:2311.12022 — <https://github.com/idavidrein/gpqa>,
  HF: <https://huggingface.co/datasets/Idavidrein/gpqa>. License: CC BY 4.0.
- **Note:** the full upstream file `gpqa_main.csv` previously kept here has been
  **removed** in accordance with the upstream authors' anti-contamination condition
  ("do not reveal dataset examples in plain text online"). Obtain the full set from
  the gated HF repository above. The selection script `jsontake.py` is kept for
  provenance; it expects `gpqa_main.csv` to be re-downloaded locally.

## 2. LiveCodeBench

- **Where used:** `Test_file/Step1_Benchmark/datasets/LiveCodeBench_v1_picked.jsonl`
  (50 problems from `release_v1`, prompts only).
- **Original:** Jain et al., *LiveCodeBench: Holistic and Contamination Free
  Evaluation of Large Language Models for Code*, arXiv:2403.07974 —
  <https://github.com/LiveCodeBench/LiveCodeBench> (harness, MIT),
  HF: <https://huggingface.co/datasets/livecodebench/code_generation> (dataset,
  tagged "cc"). Problem statements originate from LeetCode / AtCoder / Codeforces
  and remain © their respective platforms.
- **Note:** the `public_test_cases` / `private_test_cases` fields of the original
  records have been **stripped** (they were never used by our pipeline, which only
  reads the `prompt` field, and belong to the contest platforms).

## 3. Arena-Hard v2 (hard-prompt subset)

- **Where used:** `Test_file/Step1_Benchmark/datasets/arena_hard_v2_picked.jsonl`
  (selected prompts only), and the corresponding Step-1/Step-2 prompt records under
  `Dataset/` and `RAW/`.
- **Original:** Li et al., *From Crowdsourced Data to High-Quality Benchmarks:
  Arena-Hard and BenchBuilder Pipeline*, arXiv:2406.11939 —
  <https://github.com/lmarena/arena-hard-auto>. License: Apache-2.0. Prompts are
  curated from Chatbot Arena conversations.

## Scope of this notice

Model responses, energy records, judgement records, and all derived quantities in this
repository are original outputs of our experiments (© Wu Chenxi, CC BY 4.0). Where such
records contain verbatim benchmark prompt text, that text remains subject to the
licenses above; everything else is not affected.
