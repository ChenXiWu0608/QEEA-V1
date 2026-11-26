"""
Generate human comparison prompt JSON library.
Matches Test_dataset.csv against qwen3.5 9b baseline (all 3 steps).
Outputs ALL 100 rows with assembled combined_prompt strings ready for copy-paste.
Prompt assembly follows Test_vs_baseline_comparison.py exactly (no modification).
Only addition: "用中文输出理由\n\n" at the very beginning of each combined prompt.
"""
import os
import json
import pandas as pd

BASE_DIR = r"C:\Users\KongB\Database\Study library\Master's program\投稿論文\Open-Source LLMのコスト最適化推論\Open-Source LLMのコスト最適化推論 Pt.2"
TEST_CSV = os.path.join(BASE_DIR, "Test calibration", "Test_dataset.csv")
OUTPUT_JSON = os.path.join(BASE_DIR, "Test calibration", "human_comparison_prompts.json")

BASELINE_FILES = {
    "Step1": os.path.join(BASE_DIR, "Baseline", "Step1", "test_qwen3_5_9b-q8_0_False_20260410_113731_annotated.csv"),
    "Step2": os.path.join(BASE_DIR, "Baseline", "Step2", "test_qwen3_5_9b-q8_0_False_20260409_214857_annotated.csv"),
    "Step3": os.path.join(BASE_DIR, "Baseline", "Step3", "20260414_143101_dify_qwen3.5-9b_report_annotated.csv"),
}

SYSTEM_PROMPT = """You are comparing two AI responses to the same customer support question.
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
{"verdict": "", "reason": ""}"""

# Exactly the same as build_user_prompt() in Test_vs_baseline_comparison.py lines 341-360
def build_user_prompt(task_context, response_a, response_b):
    return f"""## Task context
{task_context}

## Response A
{response_a}

## Response B
{response_b}"""


def clean_text(val):
    if not isinstance(val, str):
        return ""
    val = val.strip()
    return "" if val.lower() == "nan" else val


def compute_auto_verdict(test_empty, baseline_empty):
    """Same logic as compare_single_row() lines 539-560"""
    if test_empty and baseline_empty:
        return {
            "vs_1": "A=B", "vs_2": "A=B",
            "vs_1_reason": "Both responses are empty",
            "vs_2_reason": "Both responses are empty",
        }
    elif test_empty:
        return {
            "vs_1": "A<<B", "vs_2": "A>>B",
            "vs_1_reason": "Test response is empty, baseline wins",
            "vs_2_reason": "Test response is empty, baseline wins",
        }
    else:
        return {
            "vs_1": "A>>B", "vs_2": "A<<B",
            "vs_1_reason": "Baseline response is empty, test wins",
            "vs_2_reason": "Baseline response is empty, test wins",
        }


def assemble_combined_prompt(user_prompt):
    """Assemble the complete prompt ready for copy-paste into AI chat."""
    return "用中文输出理由\n\n" + SYSTEM_PROMPT + "\n\n" + user_prompt


def main():
    print(f"Loading test dataset: {TEST_CSV}")
    test_df = pd.read_csv(TEST_CSV, dtype={"source_step": str, "test_id": str})

    print("Loading baseline files...")
    baseline_dfs = {}
    for step, path in BASELINE_FILES.items():
        print(f"  {step}: {os.path.basename(path)}")
        bdf = pd.read_csv(path, dtype={"test_id": str, "task_id": str})
        baseline_dfs[step] = bdf
        print(f"    -> {len(bdf)} rows")

    print("\nMatching and assembling prompts...")
    comparisons = []
    stats = {"total": 0, "valid": 0, "auto_verdict": 0,
             "empty_test": 0, "empty_baseline": 0, "no_match": 0}

    for _, test_row in test_df.iterrows():
        stats["total"] += 1
        source_step = str(test_row.get("source_step", ""))
        if source_step not in baseline_dfs:
            stats["no_match"] += 1
            continue

        baseline_df = baseline_dfs[source_step]
        test_id = str(test_row.get("test_id", ""))
        prompt = clean_text(test_row.get("prompt", ""))
        test_response = clean_text(test_row.get("response", ""))

        if not prompt:
            stats["no_match"] += 1
            continue

        matched = None
        if source_step in ["Step1", "Step2"]:
            match_rows = baseline_df[baseline_df["test_id"] == test_id]
            if not match_rows.empty:
                matched = match_rows.iloc[0].to_dict()
        elif source_step == "Step3":
            test_prefix = prompt[:10]
            for _, brow in baseline_df.iterrows():
                bp = clean_text(brow.get("prompt", ""))
                if bp and bp[:10] == test_prefix:
                    matched = brow.to_dict()
                    break

        if matched is None:
            stats["no_match"] += 1
            continue

        baseline_response = clean_text(matched.get("response", ""))

        test_empty = not test_response
        base_empty = not baseline_response
        if test_empty:
            stats["empty_test"] += 1
        if base_empty:
            stats["empty_baseline"] += 1

        is_auto = test_empty or base_empty

        if is_auto:
            verdict = compute_auto_verdict(test_empty, base_empty)
            ai1_verdict = verdict["vs_1"]
            ai1_reason = verdict["vs_1_reason"]
            ai2_verdict = verdict["vs_2"]
            ai2_reason = verdict["vs_2_reason"]
            stats["auto_verdict"] += 1
        else:
            ai1_verdict = ""
            ai1_reason = ""
            ai2_verdict = ""
            ai2_reason = ""
            stats["valid"] += 1

        # Build user_prompt exactly as the comparison script does
        user_prompt_vs_1 = build_user_prompt(prompt, test_response, baseline_response)
        user_prompt_vs_2 = build_user_prompt(prompt, baseline_response, test_response)

        comparisons.append({
            "index": len(comparisons),
            "source_step": source_step,
            "test_id": test_id,
            "category": str(test_row.get("category", "")),
            "subcategory": str(test_row.get("subcategory", "")),
            "language": str(test_row.get("language", "")),
            "needs_human_judgment": not is_auto,
            "auto_verdict_reason": ai1_reason if is_auto else None,
            "task_context": prompt,
            "test_response": test_response,
            "baseline_response": baseline_response,
            "ai1_combined_prompt": assemble_combined_prompt(user_prompt_vs_1),
            "ai2_combined_prompt": assemble_combined_prompt(user_prompt_vs_2),
            "ai1_verdict": ai1_verdict,
            "ai1_reason": ai1_reason,
            "ai2_verdict": ai2_verdict,
            "ai2_reason": ai2_reason,
            "human_final_check": "test    baseline",
        })

    output = {
        "metadata": {
            "test_model": "qwen3.5-9b (Test calibration - Open-Source inference)",
            "baseline_model": "qwen3.5-9b (Baseline - Open-Source inference medium tier)",
            "total_comparisons": len(comparisons),
            "needs_human_judgment": stats["valid"],
            "auto_verdict": stats["auto_verdict"],
            "verdict_options": ["A>>B", "A>B", "A=B", "A<B", "A<<B"],
            "usage": (
                "1. Copy ai1_combined_prompt -> paste to AI1 -> fill ai1_verdict / ai1_reason. "
                "2. Copy ai2_combined_prompt -> paste to AI2 -> fill ai2_verdict / ai2_reason. "
                "3. Compare results, write final judgment in human_final_check (test vs baseline)."
            ),
        },
        "comparisons": comparisons,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Output: human_comparison_prompts.json")
    print(f"  Total:                {stats['total']} rows")
    print(f"  Comparisons:          {len(comparisons)}")
    print(f"  Needs human judgment: {stats['valid']}")
    print(f"  Auto-verdict:         {stats['auto_verdict']}")
    print(f"    - Test empty:        {stats['empty_test']}")
    print(f"    - Baseline empty:    {stats['empty_baseline']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
