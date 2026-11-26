"""
Penalty Evaluator Batch Script
Batch processes CSV files from Dataset/Step{1,2,3}/reports{1,2,3} directories
using penalty test prompts from Penalty.json with 5 judge models.

Each input CSV generates a combined output file containing all 5 penalty verdicts.
Supports row-level incremental writing for interruption resilience.

Usage:
    python penalty_evaluator_batch.py           # Process full Dataset
    python penalty_evaluator_batch.py --test    # Process only Penalty/test files

Output:
    Penalty/{step}/{report}/Penalty_{model_name}_{bool}_{timestamp}.csv

Output contains columns:
    - penalty_id, source_step, category, subcategory, language
    - prompt, response
    - {model_short}_verdict, {model_short}_reason (for each of 5 judge models)

Recovery Design (SQLite-backed):
    - Each CSV file gets a SQLite progress DB in the same output directory.
    - Rows are pre-populated at start; each completed (row, model) result is
      UPSERT-ed immediately on receipt - no data loss on interruption.
    - On resume, already-completed (row, model) pairs are skipped.
    - After all 5 models finish, the DB is exported to the final CSV and removed.
    - Models are processed sequentially; rows within a model run in parallel.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
import requests
import pandas as pd
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# ==================== Configuration ====================

# -----------------------------------------------
# Ollama API Configuration (unified center)
# -----------------------------------------------
OLLAMA_CONFIG = {
    "base_url": "http://100.116.124.0:11434",
    "generate_url": "http://100.116.124.0:11434/api/generate",
    "tags_url": "http://100.116.124.0:11434/api/tags",
    "timeout": 120,
    "default_options": {
        "temperature": 0.1,
        "top_p": 1.0,
        "top_k": 0,
        "repeat_penalty": 1.0,
        "num_ctx": 32468,
        "num_predict": 1024,
    }
}

# Aliases for backward compatibility
OLLAMA_URL = OLLAMA_CONFIG["generate_url"]
TIMEOUT_SECONDS = OLLAMA_CONFIG["timeout"]

# Paths relative to script's parent directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Data source configuration
DATA_ROOT = os.path.join(BASE_DIR, "Dataset")
TEST_DATA_ROOT = os.path.join(SCRIPT_DIR, "test")

# Configuration for batch processing main dataset
STEP_CONFIG = {
    "Step1": ["reports1", "reports2"],
    "Step2": ["reports1", "reports2", "reports3"],
    "Step3": ["reports1", "reports2"],
}
PROCESS_ORDER = ["Step1", "Step2", "Step3"]

# Configuration for testing - uses test directory
TEST_PROCESS_ORDER = ["Step1", "Step2", "Step3"]
TEST_STEP_CONFIG = {
    "Step1": ["reports1", "reports2"],
    "Step2": ["reports1", "reports2", "reports3"],
    "Step3": ["reports1", "reports2"],
}

# Output configuration
OUTPUT_ROOT = os.path.join(SCRIPT_DIR)  # Output to Penalty/ directory

# Penalty JSON path
PENALTY_JSON_PATH = os.path.join(SCRIPT_DIR, "Penalty.json")

# Recovery file (file-level recovery)
RECOVERY_FILE = os.path.join(SCRIPT_DIR, ".penalty_recovery_batch.json")

# Processing settings
MAX_PARALLEL = 4          # Max parallel requests per model (single_gpu mode)
MAX_RETRIES = 3

# Concurrency scheduling settings (overridden by --single_gpu / --muti_gpu)
# Default: single_gpu mode
MAX_CONCURRENT_MODELS = 1     # How many models can run simultaneously
MAX_PARALLEL_PER_MODEL = 4    # Max requests per model when running

# Judge models
JUDGE_MODELS = [
    "ministral-3:8b-instruct-2512-q8_0",
    "olmo-3:7b-instruct-q8_0",
    "llama3.1:8b-instruct-q8_0",
    "gemma4:e4b-it-q8_0",
    "qwen3.5:9b-q8_0",
]

# Model name short mapping for output files
MODEL_SHORT_NAME = {
    "ministral-3:8b-instruct-2512-q8_0": "ministral3_8b",
    "olmo-3:7b-instruct-q8_0": "olmo3_7b",
    "llama3.1:8b-instruct-q8_0": "llama3_1_8b",   # dot -> underscore for safe column names
    "gemma4:e4b-it-q8_0": "gemma4_e4b",
    "qwen3.5:9b-q8_0": "qwen3_5_9b",               # dot -> underscore
}


# ==================== SQLite Progress DB ====================
# Each CSV task gets a .progress.db file in its output directory.
# Schema:
#   rows(penalty_id INTEGER PRIMARY KEY, source_step, category, subcategory,
#        language, prompt, response)
#   results(penalty_id INTEGER, model_short TEXT, verdict TEXT, reason TEXT,
#           PRIMARY KEY (penalty_id, model_short))
#   meta(key TEXT PRIMARY KEY, value TEXT)   -- stores output_filename, step, etc.

def db_path_for_task(task_info: dict) -> str:
    """Return the SQLite progress DB path for a given task."""
    model_name = task_info["model_name"]
    bool_val   = task_info.get("bool_value", "")
    step       = task_info["step"]
    bool_display = ""
    if bool_val:
        bool_display = ("_" + (bool_val.capitalize() if step in ("Step1", "Step2") else bool_val.lower()))
    safe_model = re.sub(r'[^\w\-]', '_', model_name)
    db_name = f".progress_{safe_model}{bool_display}.db"
    return os.path.join(task_info["output_dir"], db_name)


def db_connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for concurrent-safe writes."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_init(conn: sqlite3.Connection, rows: list, output_filename: str):
    """
    Create tables and pre-populate base row data.
    Idempotent: safe to call on an existing DB (re-populates missing rows only).
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rows (
            penalty_id   INTEGER PRIMARY KEY,
            source_step  TEXT,
            category     TEXT,
            subcategory  TEXT,
            language     TEXT,
            prompt       TEXT,
            response     TEXT
        );
        CREATE TABLE IF NOT EXISTS results (
            penalty_id   INTEGER NOT NULL,
            model_short  TEXT    NOT NULL,
            verdict      TEXT    DEFAULT '',
            reason       TEXT    DEFAULT '',
            PRIMARY KEY (penalty_id, model_short)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    # Store output filename so we can find it on resume
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('output_filename', ?)", (output_filename,))
    # Pre-populate base rows (INSERT OR IGNORE = safe on resume)
    conn.executemany(
        "INSERT OR IGNORE INTO rows VALUES (?,?,?,?,?,?,?)",
        [
            (
                i + 1,
                row.get("source_step", ""),
                row.get("category", ""),
                row.get("subcategory", ""),
                row.get("language", ""),
                row.get("prompt", ""),
                row.get("response", ""),
            )
            for i, row in enumerate(rows)
        ],
    )
    conn.commit()


def db_upsert_result(conn: sqlite3.Connection, lock: threading.Lock,
                     penalty_id: int, model_short: str, verdict: str, reason: str):
    """Thread-safe upsert of a single (row, model) result."""
    with lock:
        conn.execute(
            "INSERT OR REPLACE INTO results (penalty_id, model_short, verdict, reason) VALUES (?,?,?,?)",
            (penalty_id, model_short, verdict, reason),
        )
        conn.commit()


def db_completed_rows_for_model(conn: sqlite3.Connection, model_short: str) -> set:
    """Return set of penalty_ids already finished for this model."""
    cur = conn.execute(
        "SELECT penalty_id FROM results WHERE model_short=? AND verdict != ''",
        (model_short,),
    )
    return {row[0] for row in cur.fetchall()}


def db_export_csv(conn: sqlite3.Connection, output_path: str, header: list):
    """
    Export the full SQLite progress DB to the final CSV.
    Called once after all models finish.
    """
    # Fetch all base rows
    cur = conn.execute("SELECT * FROM rows ORDER BY penalty_id")
    base_rows = {r[0]: list(r) for r in cur.fetchall()}  # penalty_id -> [fields...]
    base_cols = ["penalty_id", "source_step", "category", "subcategory", "language", "prompt", "response"]

    # Fetch all results
    cur = conn.execute("SELECT penalty_id, model_short, verdict, reason FROM results")
    result_map = {}  # (penalty_id, model_short) -> (verdict, reason)
    for pid, ms, v, r in cur.fetchall():
        result_map[(pid, ms)] = (v, r)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for pid in sorted(base_rows.keys()):
            row_dict = dict(zip(base_cols, base_rows[pid]))
            for model_short, verdict_col, reason_col in MODEL_PENALTY_COLUMNS:
                v, r = result_map.get((pid, model_short), ("", ""))
                row_dict[verdict_col] = v
                row_dict[reason_col]  = r
            writer.writerow(row_dict)


# ==================== Category Mapping ====================
CATEGORY_MAPPING = {
    # Step1 mappings
    ("Step1", "hard_prompt", "math"): ("Step1", "math"),
    ("Step1", "hard_prompt", "coding"): ("Step1", "coding"),
    ("Step1", "creative_writing", "creative_writing"): ("Step1", "creative_writing"),
    ("Step1", "LiveCodeBench", "easy"): ("Step1", "LiveCodeBench"),
    ("Step1", "LiveCodeBench", "medium"): ("Step1", "LiveCodeBench"),
    ("Step1", "LiveCodeBench", "hard"): ("Step1", "LiveCodeBench"),
    ("Step1", "GPQA", None): ("Step1", "GPQA"),
    
    # Step2 mappings
    ("Step2", "short_prompt", "simple"): ("Step2", "short_prompt_simple"),
    ("Step2", "long_prompt", "simple"): ("Step2", "long_prompt_simple"),
    ("Step2", "short_prompt", "complex"): ("Step2", "short_prompt_complex"),
    ("Step2", "long_prompt", "complex"): ("Step2", "long_prompt_complex"),
    
    # Step3 mappings
    # 支持两种 subcategory 格式：
    ("Step3", "customer_support", "workflow_cs_simple"): ("Step3", "workflow_cs_simple"),
    ("Step3", "customer_support", "workflow_cs_conflict"): ("Step3", "workflow_cs_conflict"),
    ("Step3", "customer_support", "workflow_cs_prediction"): ("Step3", "workflow_cs_prediction"),
    ("Step3", "research", "investigation"): ("Step3", "workflow_research"),
    ("Step3", "translation", "multilingual"): ("Step3", "workflow_translation"),
}


# ==================== Word Count Functions ====================
def extract_word_limit(prompt: str) -> int | None:
    """Extract the expected word count from the first sentence of the prompt."""
    if not prompt:
        return None
    
    first_sentence = re.split(r'[.?!]|\n|\r', prompt.strip())[0]
    
    # English: "Write a {num}-word" pattern
    english_match = re.search(r'Write\s+a\s+(\d+)\s*-?\s*word', first_sentence, re.IGNORECASE)
    if english_match:
        return int(english_match.group(1))
    
    # Japanese: "{num} 語で" pattern
    japanese_match = re.search(r'(\d+)\s*語で', first_sentence)
    if japanese_match:
        return int(japanese_match.group(1))
    
    # Chinese: "500字", "2000 字" or "2万字".
    chinese_match = re.search(r'(\d+)\s*(万)?\s*字', first_sentence)
    if chinese_match:
        count = int(chinese_match.group(1))
        if chinese_match.group(2):
            count *= 10000
        return count
    
    return None


def detect_language_type(language: str) -> str:
    """Detect the language type for word counting."""
    if not language:
        return "other"
    
    lang_lower = language.lower()
    
    if lang_lower == "english":
        return "english"
    elif lang_lower in ["chinese", "chinese (simplified)", "chinese (traditional)", "zh", "zh-cn", "zh-tw"]:
        return "chinese"
    elif lang_lower in ["japanese", "jp", "ja"]:
        return "japanese"
    else:
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', language):
            return "chinese"
        return "other"


def count_words(text: str, lang_type: str) -> int:
    """Count words/characters based on language type."""
    if not text:
        return 0
    
    if lang_type == "english":
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        words = [w for w in clean_text.split() if w]
        return len(words)
    else:
        non_whitespace = [c for c in text if not c.isspace()]
        return len(non_whitespace)


def calculate_word_penalty(actual_count: int, expected_count: int) -> str | None:
    """Calculate word count penalty based on the ratio of actual to expected count."""
    if expected_count <= 0:
        return None
    
    ratio = actual_count / expected_count
    
    if ratio < 0.5 or ratio > 1.5:
        return "hard"
    elif ratio < 0.75 or ratio > 1.25:
        return "soft"
    else:
        return None


def modify_step2_system_prompt(original_prompt: str, word_penalty: str | None) -> str:
    """Modify the Step2 system prompt to include word count penalty rules."""
    if word_penalty is None:
        return original_prompt
    
    word_note = f"\n\n## Word Count Penalty Note\nBased on pre-validation, the response has a word count penalty of '{word_penalty}'. Please incorporate this into your final verdict."
    
    lines = original_prompt.split('\n')
    if lines:
        lines.insert(1, word_note)
    
    return '\n'.join(lines)


# ==================== Helper Functions ====================
def load_penalty_prompts():
    """Load penalty prompts from JSON file."""
    with open(PENALTY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_prompt_mapping(source_step, category, subcategory):
    """Get the prompt mapping from CSV fields to Penalty.json keys."""
    cat = category.strip() if category else ""
    sub = subcategory.strip() if subcategory else ""
    
    if source_step == "Step1" and cat.lower() == "gpqa":
        return ("Step1", "GPQA")
    
    key = (source_step, cat, sub if sub else None)
    if key in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[key]
    
    for map_key, map_val in CATEGORY_MAPPING.items():
        if map_key[0] != source_step:
            continue
        if map_key[1].lower() != cat.lower():
            continue
        if map_key[2] is None and (sub == "" or sub is None):
            return map_val
        if map_key[2] and map_key[2].lower() == sub.lower():
            return map_val
    
    return None


def build_user_prompt(template, row, category, subcategory):
    """Build the user prompt by replacing placeholders in template."""
    prompt = template
    prompt = prompt.replace("{prompt}", row.get("prompt", ""))
    language = row.get("language", "") or "English"
    prompt = prompt.replace("{language}", language)
    response_text = row.get("response", "")
    if len(response_text) > 2000:
        response_text = response_text[:2000] + "\n\n[Response truncated at 2000 chars]"
    prompt = prompt.replace("{response}", response_text)
    
    if category and category.strip().lower() == "gpqa":
        prompt = prompt.replace("{correct_answer}", subcategory or "")
    
    return prompt


def call_ollama_api(model, prompt, system, options=None):
    """Call Ollama API and return response data."""
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "think": False,
        "stream": False,
        "options": options if options is not None else OLLAMA_CONFIG["default_options"],
    }
    
    response = requests.post(OLLAMA_CONFIG["generate_url"], json=payload, timeout=OLLAMA_CONFIG["timeout"])
    response.raise_for_status()
    
    return response.json()


def clean_json_string(s):
    """Clean JSON string by removing problematic control characters."""
    result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    result = re.sub(r'[\u00ad\u200b\u200c\u200d\u200e\u200f\u2028\u2029\u202a\u202b\u202c\u202d\u202e\u2060\u206f]', '', result)
    result = re.sub(r'[\u061c]', '', result)
    return result


def parse_llm_json_response(text, debug_mode=True, model_name="", row_index=0, debug_log_file=None):
    """Parse JSON verdict and reason from LLM response text."""
    def log_debug(msg):
        print(msg)
        if debug_mode and debug_log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(debug_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] Model={model_name} Row={row_index + 1}\n")
                    f.write(f"  {msg}\n")
            except:
                pass
    
    def log_failure(raw_text):
        if debug_mode and debug_log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(debug_log_file, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [PARSE FAILURE] Model={model_name} Row={row_index + 1}\n")
                    f.write(f"  RAW_RESPONSE_START\n")
                    f.write(f"  {repr(raw_text[:2000])}\n")
                    f.write(f"  RAW_RESPONSE_END\n")
                    f.write(f"  {'='*60}\n")
            except:
                pass
    
    log_debug(f"[RAW RESPONSE] Length: {len(text)} chars")
    
    if len(text) > 300:
        log_debug(f"[PREVIEW] {repr(text[:300])}...")
    else:
        log_debug(f"[PREVIEW] {repr(text)}")
    
    try:
        # Strategy 1: Extract JSON from markdown code blocks (json-tagged only)
        code_block_pattern = r'```(\w*)\s*([\s\S]*?)```'
        code_block_matches = re.findall(code_block_pattern, text)
        
        if code_block_matches:
            for i, (tag, content) in enumerate(code_block_matches):
                if tag and tag.lower() != 'json':
                    log_debug(f"[Strategy 1] Skip non-JSON code block {i} (tag={tag})")
                    continue
                try:
                    cleaned = content.strip()
                    cleaned = clean_json_string(cleaned)
                    data = json.loads(cleaned)
                    log_debug(f"[Strategy 1] SUCCESS on block {i}")
                    return data.get("verdict", "unknown"), data.get("reason", ""), cleaned
                except (json.JSONDecodeError, ValueError) as e:
                    log_debug(f"[Strategy 1] Parse failed on block {i}: {e}")
                    continue
        
        # Strategy 2: Find JSON by locating { and }
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = text[start:end]
            json_str = clean_json_string(json_str)
            
            try:
                data = json.loads(json_str)
                log_debug(f"[Strategy 2] SUCCESS")
                return data.get("verdict", "unknown"), data.get("reason", ""), json_str
            except (json.JSONDecodeError, ValueError) as e:
                log_debug(f"[Strategy 2] Parse failed: {e}")
                
                # Strategy 3: JSON Repair
                fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                
                def fix_string_escapes(s):
                    result = []
                    in_string = False
                    escape_next = False
                    VALID_ESCAPES = set('"\\/bfnrt')
                    for i, ch in enumerate(s):
                        if escape_next:
                            if in_string and ch not in VALID_ESCAPES:
                                if ch == 'u' and i + 4 < len(s) and all(c in '0123456789abcdefABCDEF' for c in s[i:i + 5]):
                                    result.append(ch)
                                else:
                                    result.append('\\')
                                    result.append(ch)
                            else:
                                result.append(ch)
                            escape_next = False
                            continue
                        if ch == '\\':
                            result.append(ch)
                            escape_next = True
                            continue
                        if ch == '"':
                            in_string = not in_string
                            result.append(ch)
                            continue
                        if in_string and ch == '\n':
                            result.append('\\n')
                            continue
                        if in_string and ch == '\r':
                            result.append('\\r')
                            continue
                        if in_string and ch == '\t':
                            result.append('\\t')
                            continue
                        result.append(ch)
                    if escape_next and in_string:
                        result.append('\\')
                    return ''.join(result)
                
                fixed = fix_string_escapes(fixed)
                
                try:
                    data = json.loads(fixed)
                    log_debug(f"[Strategy 3] JSON REPAIR SUCCESS")
                    return data.get("verdict", "unknown"), data.get("reason", ""), fixed
                except (json.JSONDecodeError, ValueError) as e2:
                    log_debug(f"[Strategy 3] Repair failed: {e2}")
        
        # Strategy 4: Pure regex extraction (strict JSON-like)
        verdict_match = re.search(r'"verdict"\s*:\s*"(nice|soft|hard)"', text)
        if verdict_match:
            verdict = verdict_match.group(1)
            reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            reason = reason_match.group(1) if reason_match else ""
            reason = reason.encode().decode('unicode_escape', errors='replace')
            log_debug(f"[Strategy 4] SUCCESS: verdict={verdict}")
            return verdict, reason, "regex_only"
        
        # Strategy 4b: Loose key-value matching
        loose_match = re.search(r'(?:verdict|judgment|assessment)\s*[=:：]\s*["\']?\s*(nice|soft|hard)\s*["\']?', text, re.IGNORECASE)
        if loose_match:
            verdict = loose_match.group(1).lower()
            log_debug(f"[Strategy 4b] SUCCESS: verdict={verdict}")
            return verdict, "", "loose_regex"
        
        # Strategy 5: Keyword proximity detection
        keyword_pos = -1
        for kw in ['verdict', 'judgment', 'assessment', 'evaluation', 'rating']:
            pos = text.lower().find(kw)
            if pos != -1:
                keyword_pos = pos
                break
        if keyword_pos >= 0:
            nearby = text[max(0, keyword_pos - 50):keyword_pos + 100]
            for v in ['nice', 'soft', 'hard']:
                if re.search(r'\b' + v + r'\b', nearby):
                    log_debug(f"[Strategy 5] SUCCESS: verdict={v}")
                    return v, "", "proximity"
        
        # Strategy 6: Last-resort standalone word detection
        counts = {'nice': 0, 'soft': 0, 'hard': 0}
        for v in counts:
            counts[v] = len(re.findall(r'\b' + v + r'\b', text.lower()))
        total = sum(counts.values())
        if total > 0:
            sorted_v = sorted(counts, key=counts.get, reverse=True)
            top, second = counts[sorted_v[0]], counts[sorted_v[1] if len(sorted_v) > 1 else sorted_v[0]]
            # Confident: top >= 2x second place, or only one verdict type found
            if top >= 2 * second or second == 0:
                log_debug(f"[Strategy 6] SUCCESS: verdict={sorted_v[0]} (counts={counts})")
                return sorted_v[0], "", "frequency"
        
        log_debug(f"[FAILED] All parsing strategies failed")
        log_failure(text)
    
    except Exception as e:
        log_debug(f"[EXCEPTION] {type(e).__name__}: {e}")
        log_failure(text)
    
    return "unknown", "Failed to parse JSON response from model", "parse_failed"


def run_model_batch(conn, db_lock, rows, done_ids, model, model_short, 
                    penalty_prompts, debug_log_file, max_parallel, desc_prefix=""):
    """
    Run all pending rows for a single model in parallel.
    Returns the number of completed rows.
    """
    todo_rows = [(i, row) for i, row in enumerate(rows) if (i + 1) not in done_ids]

    print(f"\n  [{'='*45}]")
    print(f"  Judge model : {model_short}")
    print(f"  Todo / Total: {len(todo_rows)} / {len(rows)}")
    print(f"  [{'='*45}]")

    if not todo_rows:
        print(f"  [SKIP] All rows already done for {model_short}")
        return 0

    # ---- parallel worker per row ----
    def worker_task(row_idx, row, mdl=model, mdl_short=model_short):
        result, error = None, None
        try:
            _, _, result, error = process_single_request(
                row_idx, row, mdl, penalty_prompts, debug_log_file
            )
        except Exception as e:
            error = str(e)

        if result:
            return result
        if error:
            print(f"  [MODEL {mdl_short}] Row {row_idx + 1}: {error}")
        return None

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx = {
            executor.submit(worker_task, i, row): i
            for i, row in todo_rows
        }

        pbar = tqdm(
            total=len(future_to_idx),
            desc=f"{desc_prefix}{model_short[:15]}",
            unit="row",
            bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        completed_count = 0
        for future in as_completed(future_to_idx):
            try:
                result = future.result()
                if result:
                    penalty_id = result["penalty_id"]
                    verdict    = result.get("verdict", "")
                    reason     = result.get("reason", "")

                    # Persist immediately - survives any crash after this line
                    db_upsert_result(conn, db_lock,
                                     penalty_id, model_short, verdict, reason)

                    completed_count += 1
                    if completed_count <= 10 or completed_count % 50 == 0:
                        print(f"  [RESULT] Row {penalty_id} -> {model_short} verdict={verdict}")
            except Exception as e:
                print(f"\n  [ERROR] Task failed: {e}")
            finally:
                pbar.update(1)

        pbar.close()

    # Verify this model is fully complete
    done_after = db_completed_rows_for_model(conn, model_short)
    missing    = len(rows) - len(done_after)
    if missing > 0:
        print(f"  [WARNING] {missing} rows still missing for {model_short} "
              f"(will be left blank in output)")

    return completed_count


def run_sequential_models(conn, db_lock, rows, model_short_list, max_parallel_per_model, 
                          penalty_prompts, debug_log_file, desc_prefix=""):
    """
    Run models one by one, each with parallel workers (single_gpu mode).
    """
    for model in JUDGE_MODELS:
        model_short = model_short_list.get(model, model.replace(":", "_"))
        done_ids = db_completed_rows_for_model(conn, model_short)
        run_model_batch(conn, db_lock, rows, done_ids, model, model_short,
                        penalty_prompts, debug_log_file, max_parallel_per_model, desc_prefix)


def run_concurrent_models(conn, db_lock, rows, model_short_list, max_concurrent, 
                          max_parallel_per_model, penalty_prompts, debug_log_file, desc_prefix=""):
    """
    Run models in batches of max_concurrent, each with parallel workers (multi_gpu mode).
    Models are processed in groups; each group runs concurrently.
    """
    total_models = len(JUDGE_MODELS)
    batch_num = 0
    
    for start_idx in range(0, total_models, max_concurrent):
        batch_num += 1
        batch_models = JUDGE_MODELS[start_idx:start_idx + max_concurrent]
        end_idx = min(start_idx + max_concurrent, total_models)
        
        print(f"\n{'='*60}")
        print(f"  [BATCH {batch_num}] Running models: {[model_short_list.get(m, m.replace(':', '_'))[:15] for m in batch_models]}")
        print(f"{'='*60}")
        
        # Run this batch of models concurrently using a thread per model
        model_threads = []
        thread_lock = threading.Lock()
        
        def run_single_model(model, mdl_short):
            done_ids = db_completed_rows_for_model(conn, mdl_short)
            completed = run_model_batch(conn, db_lock, rows, done_ids, model, mdl_short,
                                       penalty_prompts, debug_log_file, max_parallel_per_model,
                                       desc_prefix)
            with thread_lock:
                print(f"  [BATCH {batch_num}] Model {mdl_short} completed: {completed} rows")
        
        for model in batch_models:
            model_short = model_short_list.get(model, model.replace(":", "_"))
            t = threading.Thread(target=run_single_model, args=(model, model_short))
            model_threads.append(t)
            t.start()
        
        # Wait for all models in this batch to complete
        for t in model_threads:
            t.join()
        
        print(f"  [BATCH {batch_num}] All models in batch completed")
    
    print(f"\n{'='*60}")
    print(f"  All {total_models} judge models completed in {batch_num} batches")
    print(f"{'='*60}")


# Global lock to serialise restart_ollama calls from parallel worker threads
_RESTART_LOCK = threading.Lock()
REPAIR_MODE = False  # set True by --repair to reduce retries

def _is_ollama_remote():
    """Check if Ollama is on a different host (not localhost/127.0.0.1)."""
    url = OLLAMA_CONFIG["base_url"]
    host = url.split("://")[-1].split(":")[0]
    return host not in ("localhost", "127.0.0.1", "::1", "")


def restart_ollama():
    """Restart Ollama service.  No-op when Ollama is on a remote host."""
    # Remote Ollama: we can't restart it from here - just give it time
    if _is_ollama_remote():
        print("    [RESTART] Ollama is on a remote host, cannot restart from here. "
              "Waiting 10s for it to recover...")
        time.sleep(10)
        return True  # optimistic
    with _RESTART_LOCK:
        print("    [RESTART] Attempting to restart local Ollama service...")

        def _run_cmd(cmd, desc):
            try:
                subprocess.run(cmd, check=True, capture_output=True,
                               text=True, encoding='utf-8', errors='replace', timeout=30)
                return True
            except subprocess.TimeoutExpired:
                print(f"    [RESTART] {desc} timed out.")
            except FileNotFoundError:
                print(f"    [RESTART] {desc}: command not found.")
            except Exception as e:
                print(f"    [RESTART] {desc} failed: {e}")
            return False

        def _wait_ollama_up(retries=8, interval=3):
            for _ in range(retries):
                time.sleep(interval)
                try:
                    r = requests.get(OLLAMA_CONFIG["tags_url"], timeout=5)
                    if r.status_code == 200:
                        print("    [RESTART] Ollama is back online.")
                        return True
                except:
                    pass
            print("    [WARNING] Could not verify Ollama is online.")
            return True

        if sys.platform == "win32":
            _run_cmd(["ollama", "stop"], "ollama stop")
            time.sleep(3)
            return _wait_ollama_up()
        else:
            _run_cmd(["sudo", "systemctl", "restart", "ollama"],
                     "systemctl restart ollama")
            time.sleep(5)
            return _wait_ollama_up()


# ==================== File-level Recovery State Management ====================
def load_recovery_state():
    """
    Load recovery state from file.
    Returns a set of completed file paths.
    """
    if os.path.exists(RECOVERY_FILE):
        try:
            with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            return set(state.get("completed_files", []))
        except Exception as e:
            print(f"[WARNING] Failed to load recovery state: {e}")
    return set()


def save_recovery_state(completed_files):
    """Save recovery state to file."""
    state = {
        "completed_files": list(completed_files),
        "last_updated": datetime.now().isoformat()
    }
    with open(RECOVERY_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def clear_recovery_state():
    """Clear recovery state file."""
    if os.path.exists(RECOVERY_FILE):
        os.remove(RECOVERY_FILE)


# ==================== Data Source Discovery ====================
def extract_model_name_and_bool_step12(filename: str) -> tuple:
    """
    Extract model name and bool value from Step1/2 CSV filename.
    
    Format: test_{model_name}_{bool}_{YYYYMMDD}_{HHMMSS}_annotated.csv
    Example: test_qwen3_30b-a3b-instruct-2507-q4_K_M_False_20251106_072259_annotated.csv
    Returns: (model_name, bool_value)
             e.g., ('qwen3_30b-a3b-instruct-2507-q4_K_M', 'False')
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # Remove test_ prefix
    if not name_without_ext.startswith('test_'):
        return (name_without_ext, "")
    
    after_test = name_without_ext[5:]  # Remove 'test_'
    
    # Use regex to find _True_ or _False_ followed by timestamp pattern
    match = re.search(r'_(True|False)_\d{8}_\d{6}_annotated$', after_test)
    if match:
        model_name = after_test[:match.start()]
        bool_val = match.group(1)
        return (model_name, bool_val)
    
    # Fallback: try to find _True_ or _False_ anywhere
    for bool_val in ['True', 'False']:
        separator = f'_{bool_val}_'
        idx = after_test.find(separator)
        if idx != -1:
            model_name = after_test[:idx]
            return (model_name, bool_val)
    
    # Last resort: return everything after test_
    return (after_test, "")


def extract_model_name_and_bool_step3(filename: str) -> tuple:
    """
    Extract model name and bool value from Step3 CSV filename.
    
    Format: {YYYYMMDD}_{HHMMSS}_dify_{model_name}_{bool}_report_annotated.csv
    Example: 20251110_184248_dify_qwen3_30b-a3b-instruct-2507-q4_K_M_false_report_annotated.csv
    Returns: (model_name, bool_value)
             e.g., ('qwen3_30b-a3b-instruct-2507-q4_K_M', 'false')
    """
    name_without_ext = os.path.splitext(filename)[0]
    
    # Find _dify_ position
    dify_idx = name_without_ext.find('_dify_')
    if dify_idx == -1:
        return (name_without_ext, "")
    
    # Remove {timestamp}_dify_ prefix
    after_dify = name_without_ext[dify_idx + 6:]  # Remove '_dify_'
    
    # Find _report_ position (bool value is between dify and report)
    report_idx = after_dify.find('_report')
    if report_idx != -1:
        # Extract model_name_bool before _report_annotated
        before_report = after_dify[:report_idx]
        # Split by _ to get model_name and bool
        parts = before_report.rsplit('_', 1)
        if len(parts) == 2:
            return (parts[0], parts[1])
        return (before_report, "")
    
    return (after_dify, "")


def extract_model_name_and_bool(filename: str, step: str) -> tuple:
    """
    Extract model name and bool value from CSV filename.
    
    Step1/2 format: test_{model_name}_{bool}_{timestamp}_annotated.csv
    Step3 format: {timestamp}_dify_{model_name}_{bool}_report_annotated.csv
    
    Returns: (model_name, bool_value)
    """
    if step in ("Step1", "Step2"):
        return extract_model_name_and_bool_step12(filename)
    elif step == "Step3":
        return extract_model_name_and_bool_step3(filename)
    else:
        return (os.path.splitext(filename)[0], "")


# ==================== CSV Validation (Lightweight) ====================


def discover_csv_files(test_mode=False):
    """
    Scan all Step and reports directories, return task queue.
    Tasks are ordered by Step1->2->3, then by report directory, then by filename.
    
    Args:
        test_mode: If True, use TEST_DATA_ROOT and TEST_STEP_CONFIG
    """
    task_queue = []
    
    # Choose configuration based on mode
    if test_mode:
        process_order = TEST_PROCESS_ORDER
        base_dir = TEST_DATA_ROOT
        step_config = TEST_STEP_CONFIG
    else:
        process_order = PROCESS_ORDER
        step_config = STEP_CONFIG
        base_dir = DATA_ROOT
    
    for step in process_order:
        reports = step_config.get(step, [])
        
        for report in reports:
            if test_mode:
                # Test mode: CSV files are in Penalty/test/StepX/reportsX/
                step_dir = os.path.join(base_dir, step, report) if report else os.path.join(base_dir, step)
            else:
                # Full mode: CSV files are in Dataset/StepX/reportsX
                step_dir = os.path.join(base_dir, step, report)
            
            if not os.path.exists(step_dir):
                print(f"  [WARN] Directory not found: {step_dir}")
                continue
            
            csv_files = sorted([
                f for f in os.listdir(step_dir)
                if f.endswith('.csv') and not f.startswith('.')
            ])
            
            for csv_file in csv_files:
                input_path = os.path.join(step_dir, csv_file)
                
                # Extract model name and bool value
                model_name, bool_val = extract_model_name_and_bool(csv_file, step)
                
                # For full mode, output to Penalty/StepX/reportsX
                # For test mode, output to Penalty/test/StepX
                if test_mode:
                    output_dir = os.path.join(OUTPUT_ROOT, "test", step)
                else:
                    output_dir = os.path.join(OUTPUT_ROOT, step, report)
                
                task_queue.append({
                    "step": step,
                    "report": report,
                    "input_file": csv_file,
                    "input_path": input_path,
                    "output_dir": output_dir,
                    "model_name": model_name,
                    "bool_value": bool_val,
                })
    
    return task_queue


# ==================== CSV Validation (Lightweight) ====================
def validate_csv_header(csv_path):
    """
    Read only CSV header and first 3 rows for validation.
    Returns (headers, sample_rows) tuple.
    """
    required_fields = ['prompt', 'response', 'category', 'subcategory', 'language']
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        if headers is None:
            raise ValueError("CSV file has no header")
        
        missing = [fld for fld in required_fields if fld not in headers]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        
        sample_rows = []
        for i, row in enumerate(reader):
            if i >= 3:
                break
            sample_rows.append(row)
    
    return headers, sample_rows


# ==================== Output Writing ====================
BASE_COLUMNS = ["penalty_id", "source_step", "category", "subcategory", "language", "prompt", "response"]

# Model-specific penalty output columns (verdict and reason for each model)
MODEL_PENALTY_COLUMNS = [
    ("ministral3_8b", "ministral3_8b_verdict", "ministral3_8b_reason"),
    ("olmo3_7b", "olmo3_7b_verdict", "olmo3_7b_reason"),
    ("llama3_1_8b", "llama3_1_8b_verdict", "llama3_1_8b_reason"),
    ("gemma4_e4b", "gemma4_e4b_verdict", "gemma4_e4b_reason"),
    ("qwen3_5_9b", "qwen3_5_9b_verdict", "qwen3_5_9b_reason"),
]


def get_combined_output_header():
    """Get output CSV header with all 5 penalty verdicts and reasons."""
    header = BASE_COLUMNS[:5]  # penalty_id, source_step, category, subcategory, language
    header.extend(["prompt", "response"])  # prompt, response
    
    # Add verdict and reason for each model
    for model_short, verdict_col, reason_col in MODEL_PENALTY_COLUMNS:
        header.append(verdict_col)
        header.append(reason_col)
    
    return header



# ==================== Core Processing Logic ====================
def process_single_request(row_index, row, model, penalty_prompts, debug_log_file):
    """
    Process a single (row, model) pair.
    Returns: (row_index, model, result_dict or None, error_message or None)
    """
    source_step = row.get("source_step", "")
    category = row.get("category", "")
    subcategory = row.get("subcategory", "")
    language = row.get("language", "")
    
    # Empty response: skip LLM call, verdict = "hard"
    response_value = row.get("response", "")
    response_text = "" if pd.isna(response_value) else str(response_value).strip()
    if not response_text or response_text in ("(请求失败)", "（请求失败）"):
        result = {
            "penalty_id": row_index + 1,
            "source_step": source_step,
            "category": category,
            "subcategory": subcategory,
            "language": language,
            "prompt": row.get("prompt", ""),
            "response": response_text,
            "verdict": "hard",
            "reason": "Empty test model response",
            "total_duration": 0,
            "prompt_eval_count": 0,
        }
        return row_index, model, result, None
    
    mapping = get_prompt_mapping(source_step, category, subcategory)
    if not mapping:
        return row_index, model, None, f"No prompt mapping for ({source_step}, {category}, {subcategory})"
    
    json_step, json_cat_key = mapping
    prompts = penalty_prompts.get(json_step, {})
    prompt_data = prompts.get(json_cat_key, None)
    
    if not prompt_data:
        return row_index, model, None, f"No prompt found for step={json_step}, cat={json_cat_key}"
    
    system_prompt = prompt_data["system_prompt"]
    user_prompt_template = prompt_data["user_prompt_template"]
    user_prompt = build_user_prompt(user_prompt_template, row, category, subcategory)
    
    model_short = MODEL_SHORT_NAME.get(model, model.replace(":", "_"))
    
    # Step2 Word Count Check
    word_penalty_verdict = None
    word_limit_expected = None
    word_count_actual = None
    word_ratio = None
    
    if source_step == "Step2":
        word_limit_expected = extract_word_limit(row.get("prompt", ""))
        
        if word_limit_expected is not None:
            lang_type = detect_language_type(language)
            word_count_actual = count_words(row.get("response", ""), lang_type)
            word_penalty_verdict = calculate_word_penalty(word_count_actual, word_limit_expected)
            word_ratio = word_count_actual / word_limit_expected if word_limit_expected > 0 else 0
            
            if word_penalty_verdict == "hard":
                result = {
                    "penalty_id": row_index + 1,
                    "source_step": source_step,
                    "category": category,
                    "subcategory": subcategory,
                    "language": language,
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                    "verdict": "hard",
                    "reason": f"Word count penalty: expected={word_limit_expected}, actual={word_count_actual}, ratio={word_ratio:.2%}",
                    "total_duration": 0,
                    "prompt_eval_count": 0,
                    "word_limit_expected": word_limit_expected,
                    "word_count_actual": word_count_actual,
                    "word_ratio": round(word_ratio, 4),
                    "word_penalty": word_penalty_verdict,
                }
                return row_index, model, result, None
            
            elif word_penalty_verdict == "soft":
                modified_system_prompt = modify_step2_system_prompt(system_prompt, word_penalty_verdict)
                system_prompt = modified_system_prompt
            
            else:
                pass  # Normal LLM call
    
    # Call Ollama API with retry logic (repair mode: at most 1 retry)
    effective_retries = 1 if REPAIR_MODE else MAX_RETRIES
    for attempt in range(1, effective_retries + 1):
        try:
            api_response = call_ollama_api(model, user_prompt, system_prompt)
            llm_text = api_response.get("response", "")
            verdict, reason, raw_json = parse_llm_json_response(
                llm_text,
                debug_mode=(attempt == 1),
                model_name=model_short,
                row_index=row_index,
                debug_log_file=debug_log_file
            )
            
            prompt_eval_count = api_response.get("prompt_eval_count", 0)
            total_duration_ns = api_response.get("total_duration", 0)
            total_duration_ms = total_duration_ns / 1_000_000 if total_duration_ns else 0
            
            result = {
                "penalty_id": row_index + 1,
                "source_step": source_step,
                "category": category,
                "subcategory": subcategory,
                "language": language,
                "prompt": row.get("prompt", ""),
                "response": row.get("response", ""),
                "verdict": verdict,
                "reason": reason,
                "total_duration": round(total_duration_ms, 2),
                "prompt_eval_count": prompt_eval_count,
                "word_limit_expected": word_limit_expected if source_step == "Step2" else "",
                "word_count_actual": word_count_actual if source_step == "Step2" else "",
                "word_ratio": round(word_ratio, 4) if source_step == "Step2" and word_ratio is not None else "",
                "word_penalty": word_penalty_verdict if source_step == "Step2" else "",
            }
            
            return row_index, model, result, None
            
        except requests.exceptions.Timeout:
            if REPAIR_MODE:
                print(f"    [TIMEOUT] Repair mode: skipping (120s timeout)")
                result = {
                    "penalty_id": row_index + 1,
                    "source_step": source_step,
                    "category": category,
                    "subcategory": subcategory,
                    "language": language,
                    "prompt": row.get("prompt", ""),
                    "response": row.get("response", ""),
                    "verdict": "unknown",
                    "reason": "Timeout in repair mode",
                    "total_duration": 0,
                    "prompt_eval_count": 0,
                    "word_limit_expected": word_limit_expected if source_step == "Step2" else "",
                    "word_count_actual": word_count_actual if source_step == "Step2" else "",
                    "word_ratio": round(word_ratio, 4) if source_step == "Step2" and word_ratio is not None else "",
                    "word_penalty": word_penalty_verdict if source_step == "Step2" else "",
                }
                return row_index, model, result, None
            elif attempt < effective_retries:
                print(f"    [TIMEOUT] Attempt {attempt}/{effective_retries}, retrying...")
                time.sleep(2)
            else:
                print(f"    [TIMEOUT] All {effective_retries} attempts failed, restarting Ollama...")
                if restart_ollama():
                    time.sleep(5)
                    try:
                        api_response = call_ollama_api(model, user_prompt, system_prompt)
                        llm_text = api_response.get("response", "")
                        verdict, reason, raw_json = parse_llm_json_response(
                            llm_text, debug_mode=False,
                            model_name=model_short, row_index=row_index,
                            debug_log_file=debug_log_file
                        )
                        prompt_eval_count = api_response.get("prompt_eval_count", 0)
                        total_duration_ns = api_response.get("total_duration", 0)
                        total_duration_ms = total_duration_ns / 1_000_000 if total_duration_ns else 0
                        
                        result = {
                            "penalty_id": row_index + 1,
                            "source_step": source_step,
                            "category": category,
                            "subcategory": subcategory,
                            "language": language,
                            "prompt": row.get("prompt", ""),
                            "response": row.get("response", ""),
                            "verdict": verdict,
                            "reason": reason,
                            "total_duration": round(total_duration_ms, 2),
                            "prompt_eval_count": prompt_eval_count,
                            "word_limit_expected": word_limit_expected if source_step == "Step2" else "",
                            "word_count_actual": word_count_actual if source_step == "Step2" else "",
                            "word_ratio": round(word_ratio, 4) if source_step == "Step2" and word_ratio is not None else "",
                            "word_penalty": word_penalty_verdict if source_step == "Step2" else "",
                        }
                        return row_index, model, result, None
                    except Exception as e2:
                        return row_index, model, None, f"Failed after restart: {e2}"
                else:
                    return row_index, model, None, "Failed to restart Ollama"
        
        except Exception as e:
            if attempt < effective_retries:
                print(f"    [ERROR] Attempt {attempt}/{effective_retries}: {e}, retrying...")
                time.sleep(2)
            else:
                return row_index, model, None, f"Failed after {effective_retries} attempts: {e}"
    
    return row_index, model, None, "Unexpected: exited retry loop"


# ==================== Single CSV File Processing ====================
def process_single_csv(task_info, penalty_prompts, completed_files_set, debug_log_file,
                       max_concurrent_models=1, max_parallel_per_model=4):
    """
    Process a single CSV file with all 5 judge models.

    Args:
        task_info: Dictionary with task metadata
        penalty_prompts: Dictionary with penalty prompt templates
        completed_files_set: Set of already-completed file paths (modified in place)
        debug_log_file: Path to debug log file
        max_concurrent_models: Max number of models running simultaneously (1=sequential, 2=concurrent batches)
        max_parallel_per_model: Max parallel requests per model

    Recovery strategy:
        - A SQLite progress DB is created in the output directory.
        - On each run, already-completed (penalty_id, model_short) pairs are
          loaded from the DB and skipped.
        - Each new result is upserted immediately, so interruptions lose at most
          the currently-in-flight parallel batch.
        - After ALL models finish, the DB is exported to the final CSV and deleted.

    Returns:
        dict: Processing result with stats.
    """
    input_path = task_info["input_path"]
    model_name = task_info["model_name"]
    step       = task_info["step"]

    # ------------------------------------------------------------------
    # File-level skip: already exported to final CSV in a previous run
    # ------------------------------------------------------------------
    if input_path in completed_files_set:
        print(f"  [SKIP] Already processed")
        return {
            "input_file": task_info["input_file"],
            "status": "skipped",
            "rows_processed": 0,
            "models_evaluated": 0,
        }

    # ------------------------------------------------------------------
    # Determine output paths
    # ------------------------------------------------------------------
    bool_val = task_info.get("bool_value", "")
    if bool_val:
        bool_display = bool_val.capitalize() if step in ("Step1", "Step2") else bool_val.lower()
        output_filename = f"Penalty_{model_name}_{bool_display}.csv"
    else:
        output_filename = f"Penalty_{model_name}.csv"

    os.makedirs(task_info["output_dir"], exist_ok=True)
    output_path = os.path.join(task_info["output_dir"], output_filename)
    progress_db  = db_path_for_task(task_info)

    # ------------------------------------------------------------------
    # Validate and load input CSV (streaming to avoid memory spike)
    # ------------------------------------------------------------------
    try:
        headers, sample = validate_csv_header(input_path)
        print(f"  [OK] CSV validated: {len(headers)} columns")
    except Exception as e:
        raise ValueError(f"CSV validation failed: {e}")

    rows = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source_step"] = step
            rows.append(row)

    print(f"  [OK] Loaded {len(rows)} rows")

    # ------------------------------------------------------------------
    # Open / initialise progress DB
    # ------------------------------------------------------------------
    conn      = db_connect(progress_db)
    db_lock   = threading.Lock()          # one lock per connection
    header    = get_combined_output_header()

    db_init(conn, rows, output_filename)

    # ------------------------------------------------------------------
    # Run judge models using the selected scheduling strategy
    # ------------------------------------------------------------------
    desc_prefix = "SINGLE" if max_concurrent_models == 1 else "MULTI"
    
    if max_concurrent_models >= 2:
        # Multi-GPU mode: run models in concurrent batches
        run_concurrent_models(conn, db_lock, rows, MODEL_SHORT_NAME, max_concurrent_models,
                             max_parallel_per_model, penalty_prompts, debug_log_file, desc_prefix)
    else:
        # Single-GPU mode: run models sequentially
        run_sequential_models(conn, db_lock, rows, MODEL_SHORT_NAME, max_parallel_per_model,
                             penalty_prompts, debug_log_file, desc_prefix)

    # ------------------------------------------------------------------
    # Export final CSV from DB
    # ------------------------------------------------------------------
    print(f"\n  [EXPORT] Writing {output_filename} ...")
    db_export_csv(conn, output_path, header)
    conn.close()

    # Remove progress DB now that export succeeded
    try:
        os.remove(progress_db)
    except OSError:
        pass

    print(f"  [OUTPUT] {output_filename} ({len(rows)} rows, {len(JUDGE_MODELS)} models)")

    # Mark file as fully done in file-level recovery
    completed_files_set.add(input_path)
    save_recovery_state(completed_files_set)

    return {
        "input_file": task_info["input_file"],
        "status": "success",
        "rows_processed": len(rows),
        "models_evaluated": len(JUDGE_MODELS),
        "output_file": output_filename,
    }


# ==================== Repair Mode ====================

def scan_repair_tasks():
    """
    Scan all Penalty output CSVs and identify rows with empty or 'unknown' verdicts.
    Returns list of (csv_path, missing_count) tuples for files needing repair.
    """
    repair_list = []
    for step in PROCESS_ORDER:
        for report in STEP_CONFIG.get(step, []):
            report_dir = os.path.join(OUTPUT_ROOT, step, report)
            if not os.path.isdir(report_dir):
                continue
            for fname in sorted(os.listdir(report_dir)):
                if not fname.startswith('Penalty_') or not fname.endswith('.csv'):
                    continue
                fpath = os.path.join(report_dir, fname)
                missing = count_missing_in_csv(fpath)
                if missing > 0:
                    repair_list.append((fpath, missing))
    return repair_list


def _should_skip_repair_row(row):
    """Check if a row should be excluded from repair (Step3 code_generation or empty)."""
    step = row.get('source_step', '').strip()
    cat = row.get('category', '').strip()
    return step == 'Step3' and (cat == 'code_generation' or cat == '')


def count_missing_in_csv(csv_path):
    """Count cells with empty or 'unknown' verdict in a single CSV file."""
    missing = 0
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if _should_skip_repair_row(row):
                    continue
                for _, vcol, _ in MODEL_PENALTY_COLUMNS:
                    v = row.get(vcol, '').strip()
                    if v == '' or v == 'unknown':
                        missing += 1
    except Exception:
        pass
    return missing


def repair_single_csv(csv_path, penalty_prompts, debug_log_file,
                      max_concurrent_models, max_parallel_per_model):
    """
    Repair a single Penalty output CSV by re-evaluating only rows where a
    model's verdict is empty or 'unknown'.  Correct existing verdicts are
    NEVER overwritten.

    Strategy:
      1. Load the CSV into memory.
      2. Identify (penalty_id, model_short) pairs needing repair.
      3. Create a temporary SQLite progress DB pre-populated with all rows.
      4. Pre-mark every GOOD verdict as completed in the DB so it is skipped.
      5. Run the standard model scheduling logic.  Only missing rows are processed.
      6. Patch new results back into the in-memory rows.
      7. Overwrite the original CSV with the repaired data.
      8. Clean up the temp DB.

    Returns the number of (row, model) cells repaired.
    """
    # ------------------------------------------------------------------
    # 1. Load CSV
    # ------------------------------------------------------------------
    csv_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    if not csv_rows:
        return 0

    # ------------------------------------------------------------------
    # 2. Count repairs needed (and which models are affected)
    # ------------------------------------------------------------------
    repair_pairs = set()        # (penalty_id, model_short)
    models_needed = set()       # model_short
    for row in csv_rows:
        if _should_skip_repair_row(row):
            continue
        pid = int(row.get('penalty_id', 0))
        if pid == 0:
            continue
        for model_short, verdict_col, _ in MODEL_PENALTY_COLUMNS:
            v = row.get(verdict_col, '').strip()
            if v == '' or v == 'unknown':
                repair_pairs.add((pid, model_short))
                models_needed.add(model_short)

    if not repair_pairs:
        return 0

    print(f"  [REPAIR] {os.path.basename(csv_path)}: {len(repair_pairs)} cells need fixing "
          f"({len(models_needed)} model(s): {sorted(models_needed)})")

    # ------------------------------------------------------------------
    # 3. Create temp progress DB
    # ------------------------------------------------------------------
    temp_dir = os.path.join(SCRIPT_DIR, ".repair_temp")
    os.makedirs(temp_dir, exist_ok=True)
    db_name = f"repair_{os.path.basename(csv_path).replace('.csv', '')}.db"
    db_path = os.path.join(temp_dir, db_name)

    conn = db_connect(db_path)
    db_lock = threading.Lock()

    # Build input_rows for db_init (penalty_id must equal index+1)
    input_rows = []
    for row in csv_rows:
        input_rows.append({
            "source_step": row.get("source_step", ""),
            "category": row.get("category", ""),
            "subcategory": row.get("subcategory", ""),
            "language": row.get("language", ""),
            "prompt": row.get("prompt", ""),
            "response": row.get("response", ""),
        })

    db_init(conn, input_rows, os.path.basename(csv_path))

    # ------------------------------------------------------------------
    # 4. Pre-populate ALL existing GOOD verdicts (nice/soft/hard) into DB
    #    so run_model_batch sees them as "already done" and skips them.
    # ------------------------------------------------------------------
    for row in csv_rows:
        pid = int(row.get('penalty_id', 0))
        for model_short, verdict_col, reason_col in MODEL_PENALTY_COLUMNS:
            v = row.get(verdict_col, '').strip()
            r = row.get(reason_col, '')
            if v not in ('', 'unknown'):
                db_upsert_result(conn, db_lock, pid, model_short, v, r)

    # ------------------------------------------------------------------
    # 4b. Pre-populate skipped rows (Step3 code_generation) as "already done"
    #     for ALL models so run_model_batch never touches them.
    # ------------------------------------------------------------------
    for row in csv_rows:
        if _should_skip_repair_row(row):
            pid = int(row.get('penalty_id', 0))
            for model_short, _, _ in MODEL_PENALTY_COLUMNS:
                db_upsert_result(conn, db_lock, pid, model_short, 'skip', '')

    # ------------------------------------------------------------------
    # 5. Run only the models that have missing rows
    # ------------------------------------------------------------------
    full_model_to_short = MODEL_SHORT_NAME

    needed_judge_models = []
    for m in JUDGE_MODELS:
        ms = full_model_to_short.get(m, m.replace(":", "_"))
        if ms in models_needed:
            needed_judge_models.append(m)

    desc_prefix = "REPAIR"
    for model in needed_judge_models:
        model_short = full_model_to_short.get(model, model.replace(":", "_"))
        done_ids = db_completed_rows_for_model(conn, model_short)
        run_model_batch(conn, db_lock, input_rows, done_ids, model, model_short,
                        penalty_prompts, debug_log_file, max_parallel_per_model, desc_prefix)

    # ------------------------------------------------------------------
    # 6. Read new results from DB and patch into in-memory CSV
    # ------------------------------------------------------------------
    cur = conn.execute("SELECT penalty_id, model_short, verdict, reason FROM results")
    all_results = {}
    for pid, ms, v, r in cur.fetchall():
        all_results[(pid, ms)] = (v, r)

    repaired_count = 0
    for row in csv_rows:
        pid = int(row.get('penalty_id', 0))
        for model_short, verdict_col, reason_col in MODEL_PENALTY_COLUMNS:
            if (pid, model_short) in repair_pairs:
                new = all_results.get((pid, model_short), (None, None))
                old_v = row.get(verdict_col, '').strip()
                if new[0] and new[0] not in ('', 'unknown'):
                    row[verdict_col] = new[0]
                    row[reason_col] = new[1] if new[1] else ""
                    repaired_count += 1
                elif new[0] == '' or new[0] is None:
                    # Still no result - keep whatever was there before
                    pass

    # ------------------------------------------------------------------
    # 7. Overwrite CSV
    # ------------------------------------------------------------------
    header = get_combined_output_header()
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)

    # ------------------------------------------------------------------
    # 8. Cleanup
    # ------------------------------------------------------------------
    conn.close()
    try:
        os.remove(db_path)
    except OSError:
        pass

    print(f"  [REPAIR OK] {os.path.basename(csv_path)}: {repaired_count} cells repaired")
    return repaired_count


# ==================== Main Entry Point ====================
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Penalty Evaluator - LLM Judge Benchmark (Batch Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python penalty_evaluator_batch.py              # Process full Dataset (default: single_gpu)
  python penalty_evaluator_batch.py --test       # Process only Penalty/test files
  python penalty_evaluator_batch.py --single_gpu  # Single GPU: 1 model, 4 parallel requests
  python penalty_evaluator_batch.py --muti_gpu    # Multi GPU: 2 models, 2 requests each
  
Scheduling modes:
  --single_gpu   (default) Process judge models sequentially, 4 requests per model
                 Use when running on a single GPU
  --muti_gpu     Process judge models in batches of 2, 2 requests per model
                 Use when running on multiple GPUs for better utilization
        """
    )
    parser.add_argument('--test', action='store_true', 
                        help='Run in test mode - only process files in Penalty/test directory')
    parser.add_argument('--repair', action='store_true',
                        help='Repair mode: scan existing Penalty output CSVs and re-evaluate only rows with empty or unknown verdicts')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Single GPU mode: process models sequentially with 4 parallel requests each')
    parser.add_argument('--muti_gpu', action='store_true',
                        help='Multi GPU mode: process models in concurrent batches of 2 with 2 requests each')
    args = parser.parse_args()
    
    test_mode = args.test
    repair_mode = args.repair
    
    # Determine scheduling mode - default to single_gpu if neither flag is set
    if args.muti_gpu and not args.single_gpu:
        max_concurrent_models = 2
        max_parallel_per_model = 2
        scheduling_mode = "MULTI_GPU (2 concurrent models, 2 requests each)"
    else:
        max_concurrent_models = 1
        max_parallel_per_model = 4
        scheduling_mode = "SINGLE_GPU (1 sequential model, 4 requests each)"
    
    # ================================================================
    # REPAIR MODE
    # ================================================================
    if repair_mode:
        global REPAIR_MODE
        REPAIR_MODE = True
        OLLAMA_CONFIG["timeout"] = 60  # shorter timeout for repair
        print("=" * 70)
        print(f"  Penalty Evaluator - REPAIR MODE")
        print(f"  Scheduling mode: {scheduling_mode}")
        print("=" * 70)

        # Load penalty prompts
        print("\n[1/4] Loading penalty prompts...")
        try:
            penalty_prompts = load_penalty_prompts()
            print(f"  OK: Loaded from {PENALTY_JSON_PATH}")
        except Exception as e:
            print(f"  [ERROR] Failed to load penalty prompts: {e}")
            return

        # Verify Ollama connectivity
        print("\n[2/4] Verifying Ollama connectivity...")
        try:
            r = requests.get(OLLAMA_CONFIG["tags_url"], timeout=10)
            if r.status_code == 200:
                print("  OK: Ollama is running")
            else:
                print(f"  [ERROR] Ollama returned status {r.status_code}")
                return
        except Exception as e:
            print(f"  [ERROR] Cannot connect to Ollama: {e}")
            return

        # Scan for CSVs needing repair
        print("\n[3/4] Scanning for missing/unknown verdicts...")
        repair_list = scan_repair_tasks()
        if not repair_list:
            print("  No files need repair. Everything is complete!")
            return

        print(f"  Found {len(repair_list)} file(s) with missing data:")
        total_missing = 0
        for fpath, missing in repair_list:
            print(f"    - {os.path.basename(fpath)}: {missing} cells")
            total_missing += missing
        print(f"  Total: {total_missing} cells to repair across {len(repair_list)} file(s)")

        # Clear debug log
        debug_log = os.path.join(BASE_DIR, "parse_debug_log_repair.txt")
        if os.path.exists(debug_log):
            os.remove(debug_log)

        # Process each file
        print(f"\n[4/4] Repairing...")
        print(f"  Ollama URL: {OLLAMA_CONFIG['generate_url']}")
        print(f"  Scheduling mode: {scheduling_mode}")
        print("-" * 70)

        start_time = time.time()
        total_repaired = 0
        remaining = total_missing
        for i, (fpath, missing_before) in enumerate(repair_list, 1):
            print(f"\n[{i}/{len(repair_list)}] {os.path.basename(fpath)} "
                  f"(still need: {remaining} cells)")
            try:
                repaired = repair_single_csv(
                    fpath, penalty_prompts, debug_log,
                    max_concurrent_models, max_parallel_per_model)
                total_repaired += repaired
                # Re-count this file to get true remaining
                still_missing = count_missing_in_csv(fpath)
                remaining -= (missing_before - still_missing)
                print(f"  [PROGRESS] {remaining} cells remaining across all files")
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"  REPAIR SUMMARY:")
        print(f"  Files processed: {len(repair_list)}")
        print(f"  Cells repaired this session: {total_repaired}")
        print(f"  Cells still remaining: {remaining}")
        print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"{'='*70}")
        return

    # ================================================================
    # STANDARD MODE (original flow continues below)
    # ================================================================
    
    print("=" * 70)
    mode_str = "TEST MODE" if test_mode else "BATCH MODE"
    print(f"  Penalty Evaluator - LLM Judge Benchmark ({mode_str})")
    print(f"  Scheduling mode: {scheduling_mode}")
    print("=" * 70)
    
    # Step 1: Load penalty prompts
    print("\n[1/7] Loading penalty prompts...")
    try:
        penalty_prompts = load_penalty_prompts()
        print(f"  OK: Loaded from {PENALTY_JSON_PATH}")
    except Exception as e:
        print(f"  [ERROR] Failed to load penalty prompts: {e}")
        return
    
    # Step 2: Discover data sources
    print("\n[2/7] Discovering data sources...")
    task_queue = discover_csv_files(test_mode=test_mode)
    print(f"  Found {len(task_queue)} CSV files to process")
    
    for task in task_queue:
        print(f"    - {task['step']}/{task['report']}: {task['input_file']}")
    
    # Step 3: Load recovery state
    print("\n[3/7] Checking recovery state...")
    completed_files = load_recovery_state()
    if completed_files:
        print(f"  Found {len(completed_files)} already completed files")
        print("  These files will be skipped on resume")
    else:
        print("  No recovery state found, starting fresh")
    
    # Step 4: Verify Ollama connectivity
    print("\n[4/7] Verifying Ollama connectivity...")
    try:
        r = requests.get(OLLAMA_CONFIG["tags_url"], timeout=10)
        if r.status_code == 200:
            models_data = r.json()
            available_models = [m["name"] for m in models_data.get("models", [])]
            print(f"  OK: Ollama is running")
            print(f"  Available models: {len(available_models)}")
            
            for model in JUDGE_MODELS:
                model_base = model.split(":")[0]
                if any(model_base in m for m in available_models):
                    print(f"    [OK] {model}")
                else:
                    print(f"    [WARNING] {model} not found in available models")
        else:
            print(f"  [ERROR] Ollama returned status {r.status_code}")
            return
    except Exception as e:
        print(f"  [ERROR] Cannot connect to Ollama: {e}")
        print(f"  Make sure Ollama is running and accessible at {OLLAMA_CONFIG['base_url']}")
        return
    
    # Step 5: Process task queue
    print(f"\n[5/7] Starting batch evaluation...")
    print(f"  Ollama URL: {OLLAMA_CONFIG['generate_url']}")
    print(f"  Judge models: {len(JUDGE_MODELS)}")
    print(f"  Scheduling mode: {scheduling_mode}")
    print(f"  Timeout per request: {OLLAMA_CONFIG['timeout']}s")
    print("-" * 70)
    
    # Clear debug log file at start
    DEBUG_LOG_FILE = os.path.join(BASE_DIR, "parse_debug_log_batch.txt")
    if os.path.exists(DEBUG_LOG_FILE):
        os.remove(DEBUG_LOG_FILE)
    
    # Filter out already completed files
    remaining_tasks = [t for t in task_queue if t["input_path"] not in completed_files]
    skipped_tasks = [t for t in task_queue if t["input_path"] in completed_files]
    
    print(f"  Total files: {len(task_queue)}")
    print(f"  Skipped (completed): {len(skipped_tasks)}")
    print(f"  Remaining: {len(remaining_tasks)}")
    print("-" * 70)
    
    start_time = time.time()
    results = []
    
    for i, task in enumerate(remaining_tasks, 1):
        print(f"\n[{i}/{len(remaining_tasks)}] Processing: {task['input_file']}")
        print(f"  Source: {task['step']}/{task['report']}")
        print(f"  Model: {task['model_name']}")
        
        try:
            result = process_single_csv(task, penalty_prompts, completed_files, DEBUG_LOG_FILE,
                                       max_concurrent_models, max_parallel_per_model)
            results.append(result)
            
            if result['status'] == 'success':
                print(f"  [OK] Output: {result['output_file']}")
            else:
                print(f"  [SKIP] File was already processed")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "input_file": task['input_file'],
                "status": "failed",
                "error": str(e),
            })
    
    # Step 6: Summary
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r.get('status') == 'success')
    skip_count = sum(1 for r in results if r.get('status') == 'skipped')
    fail_count = sum(1 for r in results if r.get('status') == 'failed')
    
    print(f"\n[7/7] Summary:")
    print(f"  Total files processed: {len(results)}")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    
    # Show debug log info
    if os.path.exists(DEBUG_LOG_FILE):
        with open(DEBUG_LOG_FILE, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        failure_count = sum(1 for line in log_lines if "[PARSE FAILURE]" in line)
        print(f"\n  [DEBUG LOG] Total entries: {len(log_lines)}")
        print(f"  [DEBUG LOG] Parse failures: {failure_count}")
        print(f"  [DEBUG LOG] See {DEBUG_LOG_FILE} for details")
    
    print("\n" + "=" * 70)
    print(f"  Batch evaluation completed! (mode: {scheduling_mode})")
    print("=" * 70)


if __name__ == "__main__":
    main()
