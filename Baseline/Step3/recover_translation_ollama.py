"""
Recover missing translation responses in Baseline Step3 CSVs using Ollama.

For each baseline model (qwen3.5-35b, qwen3.5-9b, qwen3.5-4b, glm-4.7-flash),
runs a 3-step refinement pipeline for the two missing translation tasks:
  Step 1: Initial translation
  Step 2: Translation critique
  Step 3: Refined translation

Usage:
    python recover_translation_ollama.py
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://100.116.124.1:11434/api/generate"

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent

TASKS_JSON = PROJECT_ROOT / "Test_file" / "Step3_Workflow" / "tasks.json"

BASELINE_MODEL_MAP = {
    "qwen3.5-35b": "qwen3.5:35b-a3b-q8_0",
    "qwen3.5-9b":  "qwen3.5:9b-q8_0",
    "qwen3.5-4b":  "qwen3.5:4b-q8_0",
    "glm4.7":      "glm-4.7-flash:q8_0",
}

def call_ollama(model, system, prompt):
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"num_ctx": 8192},
    }
    for attempt in range(1, 4):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            print(f"    Attempt {attempt}/3 failed: {e}")
            time.sleep(2)
    return ""

def step1_translate(model, src, tgt, text):
    system = (
        f"This is an {src} to {tgt} translation, "
        f"please provide the {tgt} translation for this text.\n"
        "Do not provide any explanations or text apart from the translation."
    )
    user = f"{src}: {text}\n{tgt}:"
    return call_ollama(model, system, user)

def step2_critique(model, src, tgt, source_text, init_trans):
    system = (
        f"Your task is to carefully read a Original Article and a Initial Translation "
        f"from {src} to {tgt}, and then give constructive criticism and helpful suggestions "
        "to improve the translation.\n\n"
        "When writing suggestions, pay attention to whether there are ways to improve the "
        "translation's\n"
        "(i) accuracy (by correcting errors of addition, mistranslation, omission, "
        "or untranslated text),\n"
        "(ii) fluency (by applying {tgt} grammar, spelling and punctuation rules, "
        "and ensuring there are no unnecessary repetitions),\n"
        "(iii) style (by ensuring the translations reflect the style of the source text "
        "and takes into account any cultural context),\n"
        "(iv) terminology (by ensuring terminology use is consistent and reflects the "
        "source text domain; and by only ensuring you use equivalent idioms {tgt}).\n\n"
        "Write a list of specific, helpful and constructive suggestions for improving "
        "the translation.\n"
        "Each suggestion should address one specific part of the translation.\n"
        "Output only the suggestions and nothing else."
    )
    user = f"Original Article:\n{source_text}\n\nInitial Translation:\n{init_trans}"
    return call_ollama(model, system, user)

def step3_refine(model, src, tgt, source_text, init_trans, suggestions):
    system = (
        f"Your task is to carefully read, then edit, a translation from {src} to {tgt}.\n\n"
        'You need to check the translation against the "Initial Translation," '
        '"Translation Improvement Suggestions," and the "Original Article." '
        'Improve the "Initial Translation" into a better version and output it.\n\n'
        "(i) accuracy (by correcting errors of addition, mistranslation, omission, "
        "or untranslated text),\n"
        "(ii) fluency (by applying {tgt} grammar, spelling and punctuation rules "
        "and ensuring there are no unnecessary repetitions),\n"
        "(iii) style (by ensuring the translations reflect the style of the source text),\n"
        "(iv) terminology (inappropriate for context, inconsistent use), or\n"
        "(v) other errors.\n\n"
        "Output only the new translation and nothing else."
    )
    user = (
        f"Original Article:\n{source_text}\n\n"
        f"Initial Translation:\n{init_trans}\n\n"
        f"Translation Improvement Suggestions:\n{suggestions}"
    )
    return call_ollama(model, system, user)

def load_translation_tasks(tasks_json):
    """Load source texts for task_id=3 (multilingual_translation)."""
    with open(tasks_json, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    for task in tasks:
        if task["task_id"] == 3:
            qs = task["questions"]
            return [
                {"src": qs[0]["source_lang"], "tgt": qs[0]["target_lang"],
                 "text": qs[0]["source_text"]},
                {"src": qs[1]["source_lang"], "tgt": qs[1]["target_lang"],
                 "text": qs[1]["source_text"]},
            ]
    return []

def match_baseline_model(csv_name):
    for key, model in BASELINE_MODEL_MAP.items():
        if key in csv_name:
            return model
    return None

def extract_lang_direction(prompt):
    """Parse prompt like 'ja→en: text...' to get src, tgt."""
    m = re.match(r'^(\w+)→(\w+):\s*(.*)', prompt, re.DOTALL)
    if m:
        return m.group(1), m.group(2)
    return None, None

def main():
    print("=" * 60)
    print("Baseline Step3 Translation Recovery (Ollama)")
    print("=" * 60)

    if not TASKS_JSON.exists():
        print(f"ERROR: tasks.json not found at {TASKS_JSON}")
        sys.exit(1)

    trans_tasks = load_translation_tasks(TASKS_JSON)
    if not trans_tasks:
        print("ERROR: No translation tasks found in tasks.json")
        sys.exit(1)

    csv_files = sorted(SCRIPT_DIR.glob("*_annotated.csv"))
    print(f"Found {len(csv_files)} baseline CSV files\n")

    for csv_path in csv_files:
        model = match_baseline_model(csv_path.name)
        if not model:
            print(f"  SKIP {csv_path.name}: no matching model")
            continue

        print(f"{'='*50}")
        print(f"  File:   {csv_path.name}")
        print(f"  Model:  {model}")
        print(f"{'='*50}")

        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)

        modified = False
        for i, row in enumerate(rows):
            if row.get("task_id") != "3":
                continue
            if row.get("response", "").strip():
                print(f"  Row {i+2}: already has response, skip")
                continue

            prompt = row.get("prompt", "")
            src, tgt = extract_lang_direction(prompt)
            if not src:
                print(f"  Row {i+2}: cannot parse prompt, skip")
                continue

            source_text = None
            for tt in trans_tasks:
                if tt["src"] == src and tt["tgt"] == tgt:
                    source_text = tt["text"]
                    break
            if not source_text:
                print(f"  Row {i+2}: no source text for {src}→{tgt}, skip")
                continue

            print(f"\n  Row {i+2}: {src}→{tgt} ({len(source_text)} chars)")

            print(f"  [1/3] Initial translation...", end=" ", flush=True)
            init = step1_translate(model, src, tgt, source_text)
            if not init:
                print("FAILED")
                continue
            print(f"OK ({len(init)} chars)")

            print(f"  [2/3] Critique...", end=" ", flush=True)
            critique = step2_critique(model, src, tgt, source_text, init)
            if not critique:
                print("SKIP (no suggestions)")
                final = init
            else:
                print(f"OK ({len(critique)} chars)")
                print(f"  [3/3] Refine...", end=" ", flush=True)
                final = step3_refine(model, src, tgt, source_text, init, critique)
                if not final:
                    print("FAILED (using initial)")
                    final = init
                else:
                    print(f"OK ({len(final)} chars)")

            row["response"] = final
            row["response_len"] = str(len(final))
            modified = True
            print(f"  \u2713 Saved ({len(final)} chars)")

        if modified:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\n  \u2713 File updated: {csv_path.name}\n")
        else:
            print(f"  No changes needed\n")

    print("Done!")

if __name__ == "__main__":
    main()
