# Judge Prompts / 判定プロンプトの所在

日本語：評価モデル（判定者LLM）に与えた判定基準とプロンプトの全文の所在を以下に示します。
ここで参照するファイルの内容が正本であり、論文本文（2.3節）の記述と一致します。

This file is an index of the exact prompts and judgement criteria given to the five judge
LLMs. The referenced source files are authoritative and are the same files the offline
pipeline reads at run time (`run_pipeline.py`).

## 1. Usability judgement (利用可能性判定, absolute nice / soft / hard)

- **Prompt text (verbatim, all 14 task categories):**
  [`step2.1_Penalty/Penalty.json`](step2.1_Penalty/Penalty.json)
  Each entry provides `system_prompt` and `user_prompt_template` (placeholders `{prompt}`,
  `{language}`, `{response}`):
  - `Step1`: `math`, `coding`, `creative_writing`, `LiveCodeBench`, `GPQA`
  - `Step2`: `short_prompt_simple`, `long_prompt_simple`, `short_prompt_complex`, `long_prompt_complex`
  - `Step3`: `workflow_cs_simple`, `workflow_cs_conflict`, `workflow_cs_prediction`, `workflow_research`, `workflow_translation`
- **Driver script (how the prompts are assembled and sent):**
  [`step2.1_Penalty/penalty_evaluator_batch.py`](step2.1_Penalty/penalty_evaluator_batch.py)
  - Word-count penalty rules are injected into the Step-2 system prompt by
    `modify_step2_system_prompt()`; thresholds and coefficients are in
    [`step2.1_Penalty/length_protocol.json`](step2.1_Penalty/length_protocol.json).
  - Automatic `hard` without an LLM call: empty/failed responses, and word-count
    ratio outside `0.5–1.5` (pre-screening).
- Verbatim example (`Penalty.json → Step1 → math`):

```text
# system_prompt
You are evaluating the output of an AI model.

## Judgement rule
First, verify language consistency: if the Test Model response language does not match the Task language, output "verdict": "hard".
Then, evaluate the mathematical reasoning:
- "nice": Final answer is correct AND reasoning steps are logically sound.
- "soft": Correct methodology/setup but calculation error, OR correct answer but flawed/unconvincing reasoning, OR only partially addresses a multi-part question.
- "hard": Completely wrong approach, irrelevant content, or fundamental mathematical misconceptions.
Respond strictly in JSON: {"verdict": "nice|soft|hard", "reason": "max 10 words"}

# user_prompt_template
## Task context
{prompt}

## Task language
{language}

## Test Model response
{response}

## Response Format
Respond strictly in JSON:
{"verdict": "nice|soft|hard", "reason": "max 10 words"}
```

## 2. Win/loss judgement (勝敗判定, pairwise A≫B … A≪B)

- **System prompt (verbatim; identical in both files):**
  - [`step1_Test calibration/comparison_config.py`](step1_Test%20calibration/comparison_config.py) → `SYSTEM_PROMPT`
  - [`step2.2_vs_baseline/dataset_vs_baseline_config.py`](step2.2_vs_baseline/dataset_vs_baseline_config.py) → `SYSTEM_PROMPT`

```text
You are comparing two AI responses to the same customer support question.
Your only task is to judge which response better fulfills the user's question.

## Judgement rule (choose exactly one)
- "A>>B": Response A is decisively better
- "A>B": Response A is somewhat better
- "A=B": Both responses are equivalent in quality
- "A<B": Response B is somewhat better
- "A<<B": Response B is decisively better

Do not penalize for writing style, length, or formatting differences.
Do not consider whether either response has flaws — only judge which one is relatively better.

Important: Always express the verdict from A's perspective.
For example, if B is better, write "A<B" or "A<<B", never "B>A" or "B>>A".
The verdict field must be exactly one of these five strings: "A>>B", "A>B", "A=B", "A<B", "A<<B".
No other values are allowed.

Keep your reason concise, within 50 words total.

Respond strictly in JSON with no extra text:
{"verdict": "", "reason": ""}
```

- **User prompt template:** `build_user_prompt()` in
  [`step2.2_vs_baseline/dataset_vs_baseline_comparison.py`](step2.2_vs_baseline/dataset_vs_baseline_comparison.py):

```text
## Task context
{prompt}

## Response A
{response_a}

## Response B
{response_b}
```

- **Position-bias control:** every pair is judged twice with A/B swapped
  (`dataset_vs_baseline_comparison.py`, `user_prompt_1` / `user_prompt_2`); verdicts are
  aggregated as described in the paper (A≫B = 3, A>B = 1, A=B = 0.5, loss = 0).

## 3. Calibration and human reference

- Human reference judgements and the prompts shown to human raters during calibration:
  [`step1_Test calibration/human_comparison_prompts.json`](step1_Test%20calibration/human_comparison_prompts.json),
  [`step1_Test calibration/penalty_human.json`](step1_Test%20calibration/penalty_human.json),
  [`step1_Test calibration/generate_human_prompts.py`](step1_Test%20calibration/generate_human_prompts.py)
- Saved judge outputs for every judgement unit (prompt + verdict + reason) are in
  `step2.1_Penalty/` and `step2.2_vs_baseline/`, so any prompt can be inspected together
  with the response it produced.
