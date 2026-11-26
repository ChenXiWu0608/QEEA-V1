"""
Test vs Baseline Comparison Script (Enhanced with SQLite Recovery)

Compares Test_dataset.csv with Baseline files using judge models via Ollama API.
For each baseline file, performs bidirectional comparison (test->baseline and baseline->test)
using multiple judge models.

Output files: Test_vs_{baseline_model}_{judge_model}.csv

Recovery Design (SQLite-backed):
    - Each output CSV task gets a SQLite progress DB (.progress.db) in the output directory.
    - Rows are pre-populated at start; each completed (row_index, direction) result is
      UPSERT-ed immediately on receipt - no data loss on interruption.
    - On resume, already-completed (row_index, direction) pairs are skipped.
    - After all rows finish, the DB is exported to the final CSV and removed.
    - File-level recovery skips entirely completed baseline×judge combinations.

Usage:
    python Test_vs_baseline_comparison.py                                    # Full mode (all baselines)
    python Test_vs_baseline_comparison.py --baseline-model qwen3_5_9b        # Single baseline model only
    python Test_vs_baseline_comparison.py --single_gpu                       # Single GPU config
    python Test_vs_baseline_comparison.py --multi_gpu                        # Multi GPU config

"""

import argparse
import csv
import json
import logging
import os
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    print("requests not installed. Run: pip install requests pandas")
    sys.exit(1)

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from comparison_config import (
    OLLAMA_URL,
    MAX_PARALLEL,
    TIMEOUT_SECONDS,
    MAX_RETRIES,
    JUDGE_MODELS,
    MODEL_SHORT_NAME,
    TEST_DATASET_PATH,
    BASELINE_DIR,
    OUTPUT_DIR,
    BASELINE_FILES,
    SYSTEM_PROMPT,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    MODEL_NAME_ALIASES,
)

# ==================== Recovery Configuration ====================
RECOVERY_FILE = OUTPUT_DIR / ".comparison_recovery_state.json"

# Processing settings (can be overridden by CLI args)
MAX_CONCURRENT_MODELS = 1      # How many model combinations can run simultaneously
MAX_PARALLEL_PER_MODEL = 2     # Max parallel requests per comparison task

# Verdict handling
VALID_VERDICTS = ["A>>B", "A>B", "A=B", "A<B", "A<<B"]
REPAIRABLE_VERDICTS = {"", "error"}

# ==================== Logging Setup ====================
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "comparison.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ==================== SQLite Progress DB ====================
# Each comparison task (baseline × judge) gets a .progress.db file.
# Schema:
#   rows(row_index INTEGER PRIMARY KEY, source_step, test_id, prompt, test_response, baseline_response)
#   results(row_index INTEGER, direction TEXT, verdict TEXT, reason TEXT,
#           PRIMARY KEY (row_index, direction))
#   meta(key TEXT PRIMARY KEY, value TEXT)  -- stores output_filename, baseline_model, judge_model, step

def db_path_for_task(baseline_model: str, judge_model: str, output_dir: Path) -> str:
    """Return the SQLite progress DB path for a given comparison task."""
    judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
    safe_baseline = re.sub(r'[^\w\-]', '_', baseline_model)
    safe_judge = re.sub(r'[^\w\-]', '_', judge_short)
    db_name = f".progress_{safe_baseline}_{safe_judge}.db"
    return str(output_dir / db_name)


def db_connect(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode for concurrent-safe writes."""
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_init(conn: sqlite3.Connection, rows: list, output_filename: str,
            baseline_model: str, judge_model: str, step: str):
    """
    Create tables and pre-populate base row data.
    Idempotent: safe to call on an existing DB (re-populates missing rows only).
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rows (
            row_index        INTEGER PRIMARY KEY,
            source_step      TEXT,
            test_id          TEXT,
            prompt           TEXT,
            test_response    TEXT,
            baseline_response TEXT
        );
        CREATE TABLE IF NOT EXISTS results (
            row_index   INTEGER NOT NULL,
            direction   TEXT    NOT NULL,  -- 'vs_1' or 'vs_2'
            verdict     TEXT    DEFAULT '',
            reason      TEXT    DEFAULT '',
            PRIMARY KEY (row_index, direction)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    # Store metadata
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('output_filename', ?)", (output_filename,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('baseline_model', ?)", (baseline_model,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('judge_model', ?)", (judge_model,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('step', ?)", (step,))
    # Pre-populate base rows (INSERT OR IGNORE = safe on resume)
    conn.executemany(
        "INSERT OR IGNORE INTO rows VALUES (?,?,?,?,?,?)",
        [
            (
                i,
                str(row.get("source_step", "")),
                str(row.get("test_id", "")),
                str(row.get("prompt", "")),
                str(row.get("test_response", "")),
                str(row.get("baseline_response", "")),
            )
            for i, row in enumerate(rows)
        ],
    )
    conn.commit()


def db_upsert_result(conn: sqlite3.Connection, lock: threading.Lock,
                     row_index: int, direction: str, verdict: str, reason: str):
    """Thread-safe upsert of a single (row, direction) result."""
    with lock:
        conn.execute(
            "INSERT OR REPLACE INTO results (row_index, direction, verdict, reason) VALUES (?,?,?,?)",
            (row_index, direction, verdict, reason),
        )
        conn.commit()


def db_completed_rows_for_direction(conn: sqlite3.Connection, direction: str) -> set:
    """Return set of row_indices already finished for this direction."""
    placeholders = ",".join("?" for _ in VALID_VERDICTS)
    cur = conn.execute(
        f"SELECT row_index FROM results WHERE direction=? AND verdict IN ({placeholders})",
        (direction, *VALID_VERDICTS),
    )
    return {row[0] for row in cur.fetchall()}


def db_export_csv(conn: sqlite3.Connection, output_path: Path, baseline_model: str, judge_model: str):
    """
    Export the full SQLite progress DB to the final CSV.
    Called once after all rows finish.
    """
    # Fetch all base rows
    cur = conn.execute("SELECT * FROM rows ORDER BY row_index")
    base_rows = {r[0]: list(r) for r in cur.fetchall()}  # row_index -> [fields...]
    base_cols = ["row_index", "source_step", "test_id", "prompt", "test_response", "baseline_response"]

    # Fetch all results
    cur = conn.execute("SELECT row_index, direction, verdict, reason FROM results")
    result_map = {}  # (row_index, direction) -> (verdict, reason)
    for rid, direction, v, r in cur.fetchall():
        result_map[(rid, direction)] = (v, r)

    # Build header
    header = ["source_step", "test_id", "prompt", "test_response", "baseline_response",
              "vs_1", "vs_2", "vs_1_reason", "vs_2_reason"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for rid in sorted(base_rows.keys()):
            row_dict = dict(zip(base_cols, base_rows[rid]))
            # vs_1: Test vs Baseline (A=Test, B=Baseline)
            v1, r1 = result_map.get((rid, "vs_1"), ("", ""))
            # vs_2: Baseline vs Test (A=Baseline, B=Test)
            v2, r2 = result_map.get((rid, "vs_2"), ("", ""))
            row_dict["vs_1"] = v1
            row_dict["vs_2"] = v2
            row_dict["vs_1_reason"] = r1
            row_dict["vs_2_reason"] = r2
            # Remove internal row_index
            row_dict.pop("row_index", None)
            writer.writerow(row_dict)


# ==================== File-level Recovery State Management ====================
def load_recovery_state() -> set:
    """
    Load recovery state from file.
    Returns a set of completed task keys (baseline_model|judge_model|step).
    """
    if RECOVERY_FILE.exists():
        try:
            with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            return set(state.get("completed_tasks", []))
        except Exception as e:
            logger.warning(f"Failed to load recovery state: {e}")
    return set()


def save_recovery_state(completed_tasks: set):
    """Save recovery state to file."""
    state = {
        "completed_tasks": list(completed_tasks),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RECOVERY_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def clear_recovery_state():
    """Clear recovery state file."""
    if RECOVERY_FILE.exists():
        RECOVERY_FILE.unlink()


# ==================== Data Loading Module ====================
def load_test_dataset(path: Path) -> pd.DataFrame:
    """Load Test_dataset.csv"""
    logger.info(f"Loading test dataset from: {path}")
    df = pd.read_csv(path, dtype={"test_id": str, "source_step": str, "prompt": str, "response": str})
    # Clean test_id to ensure consistent format
    df["test_id"] = df["test_id"].astype(str).str.strip()
    logger.info(f"Test dataset loaded: {len(df)} rows")
    return df


def load_baseline_file(path: Path) -> pd.DataFrame:
    """Load a single baseline CSV file"""
    logger.debug(f"Loading baseline file: {path}")
    df = pd.read_csv(path, dtype={"test_id": str, "task_id": str, "prompt": str, "response": str})
    return df


def extract_model_name(filename: str) -> str:
    """Extract model name from baseline filename.
    
    Examples:
        test_qwen3_5_4b-q8_0_False_20260410_180517_annotated.csv -> qwen3_5_4b
        test_qwen3_5_9b-q8_0_False_20260410_113731_annotated.csv -> qwen3_5_9b
        test_qwen3_5_35b-a3b-q8_0_False_20260409_235358_annotated.csv -> qwen3_5_35b_a3b
        test_glm-4_7-flash_q8_0_False_20260412_011535_annotated.csv -> glm_4_7_flash
        20260414_143101_dify_qwen3.5-9b_report_annotated.csv -> qwen3_5_9b
        20260414_123741_dify_qwen3.5-35b_report_annotated.csv -> qwen3_5_35b
        20260414_150623_dify_qwen3.5-4b_report_annotated.csv -> qwen3_5_4b
        20260414_151930_dify_glm4.7_report_annotated.csv -> glm4_7
    """
    # Remove extension
    name = Path(filename).stem
    
    # For Step3 files: 20260414_143101_dify_qwen3.5-9b_report
    if "dify" in name:
        # Extract model part between "dify_" and "_report"
        match = re.search(r'dify_(.+?)_report', name)
        if match:
            model = match.group(1)
            # Replace dots and hyphens with underscores
            model = model.replace(".", "_").replace("-", "_")
            return model
    
    # For Step1/Step2 files: test_qwen3_5_4b-q8_0_False_...
    # Extract model part after "test_" and before "_False_" or "_True_"
    match = re.search(r'test_(.+?)_(?:False|True)_', name)
    if match:
        model = match.group(1)
        # Remove quantization suffix (e.g., -q8_0)
        model = re.sub(r'-q\d+_?\w*', '', model)
        # Replace dots and hyphens with underscores
        model = model.replace(".", "_")
        return model
    
    # Fallback: return cleaned filename
    return re.sub(r'[_\-](?:False|True|annotated|report|csv).*$', '', name)


# ==================== Data Matching Module ====================
def match_step1_step2(test_row: Any, baseline_df: pd.DataFrame, test_id: str) -> Optional[dict]:
    """Match Step1/Step2 data by test_id"""
    match = baseline_df[baseline_df["test_id"] == test_id]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None


def match_step3(test_row: Any, baseline_df: pd.DataFrame) -> Optional[dict]:
    """Match Step3 data by first 10 characters of prompt"""
    test_prompt = str(test_row.get("prompt", ""))[:10]
    if not test_prompt:
        return None
    
    # Find matching row in baseline
    for _, baseline_row in baseline_df.iterrows():
        baseline_prompt = str(baseline_row.get("prompt", ""))[:10]
        if test_prompt == baseline_prompt:
            return baseline_row.to_dict()
    return None


def get_baseline_files_for_step(step: str) -> list:
    """Get list of baseline files for a specific step"""
    return BASELINE_FILES.get(step, [])


# ==================== Prompt Builder Module ====================
def build_user_prompt(prompt: str, response_a: str, response_b: str) -> str:
    """Build the user prompt for comparison.
    
    Args:
        prompt: The original user question/prompt
        response_a: Response A (to be judged as better or worse)
        response_b: Response B (to be judged as better or worse)
    
    Returns:
        Formatted user prompt string
    """
    user_prompt = f"""## Task context
{prompt}

## Response A
{response_a}

## Response B
{response_b}"""
    return user_prompt


# ==================== Judge Caller Module ====================
def build_ollama_payload(model: str, prompt: str, system: str) -> dict:
    """Build Ollama API request payload"""
    return {
        "model": model,
        "prompt": prompt,
        "system": system,
        "format": "json",
        "think": False,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "num_ctx": 40000,
        }
    }


def call_judge_model(
    model: str,
    prompt: str,
    system: str,
    max_retries: int = MAX_RETRIES,
    timeout: int = TIMEOUT_SECONDS
) -> Optional[dict]:
    """Call Ollama judge model with retry logic (synchronous version).
    
    Args:
        model: Model name to use
        prompt: User prompt
        system: System prompt
        max_retries: Maximum number of retries
        timeout: Request timeout in seconds
    
    Returns:
        Parsed JSON response with verdict and reason, or None on failure
    """
    payload = build_ollama_payload(model, prompt, system)
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout
            )
            if response.status_code != 200:
                logger.warning(f"API returned status {response.status_code} for model {model}")
                continue
              
            data = response.json()
            response_text = data.get("response", "")
              
            result = parse_judge_response(response_text)
            if result and result.get("verdict") in VALID_VERDICTS:
                return result

            # Bare-minimum fallback: extract verdict label from raw text
            verdict = extract_verdict_bare_minimum(response_text)
            if verdict:
                logger.info(f"    Bare-minimum verdict extracted: {verdict}")
                return {"verdict": verdict, "reason": ""}
              
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout calling model {model}, attempt {attempt}/{max_retries}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Network error calling model {model}: {e}, attempt {attempt}/{max_retries}")
        except Exception as e:
            logger.warning(f"Unexpected error calling model {model}: {e}, attempt {attempt}/{max_retries}")
        
        if attempt < max_retries:
            time.sleep(1 * attempt)  # Exponential backoff
    
    logger.error(f"Failed to get valid response from model {model} after {max_retries} retries")
    return None


def normalize_verdict(v: str) -> str:
    """Convert B->A patterns to A->B perspective."""
    mapping = {
        "B>>A": "A<<B", "B>A": "A<B", "B=A": "A=B",
        "B<A": "A>B", "B<<A": "A>>B",
    }
    return mapping.get(v.strip(), v.strip())


def extract_verdict_bare_minimum(text: str) -> Optional[str]:
    """Last-resort: regex-extract verdict label from raw text (A->B or B->A)."""
    m = re.search(r'(A>>B|A>B|A=B|A<B|A<<B|B>>A|B>A|B=A|B<A|B<<A)', text)
    if m:
        return normalize_verdict(m.group(1))
    return None


def repair_json_escapes(json_str: str) -> str:
    """Escape backslashes that are valid text but invalid JSON escapes."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)


def parse_judge_response_fallback(json_str: str) -> Optional[dict]:
    """Regex-based fallback for malformed JSON that still contains the fields."""
    verdict_match = re.search(
        r'"verdict"\s*:\s*"((?:A|B)(?:>>|>|=|<|<<)(?:A|B)|error)"',
        json_str,
    )
    if not verdict_match:
        return None

    reason_match = re.search(r'"reason"\s*:\s*"(.*)"\s*[,}]?\s*$', json_str, re.DOTALL)
    reason = reason_match.group(1) if reason_match else ""
    reason = repair_json_escapes(reason)
    reason = reason.replace(r'\"', '"').replace(r'\\', '\\').strip()
    return {"verdict": normalize_verdict(verdict_match.group(1)), "reason": reason}


def parse_judge_response(response_text: str) -> Optional[dict]:
    """Parse judge model JSON response with robust fallback logic."""
    if not response_text:
        return None

    # Strip markdown code block if present
    text = re.sub(r'^```(?:json)?\s*', '', response_text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()

    # Find first {
    start = text.find('{')
    if start == -1:
        logger.warning(f"No JSON object found in response: {response_text[:100]}..")
        return {"verdict": "error", "reason": "Failed to parse judge response"}

    # Try full JSON parsing first (if closing } exists)
    end = text.rfind('}')
    if end > start:
        json_str = text[start:end + 1]
        for candidate in (json_str, repair_json_escapes(json_str)):
            try:
                result = json.loads(candidate, strict=False)
                verdict = normalize_verdict(result.get("verdict", ""))
                reason = result.get("reason", "")
                return {"verdict": verdict, "reason": reason.strip() if reason else ""}
            except json.JSONDecodeError:
                pass

        fallback = parse_judge_response_fallback(json_str)
        if fallback:
            return fallback

    # Truncated JSON (no closing } or all parse attempts failed) — extract verdict via regex directly
    truncated = text[start:]
    verdict_match = re.search(
        r'"verdict"\s*:\s*"((?:A|B)(?:>>|>|=|<|<<)(?:A|B)|error)"', truncated
    )
    if verdict_match:
        reason_match = re.search(r'"reason"\s*:\s*"(.*)', truncated, re.DOTALL)
        reason = ""
        if reason_match:
            raw = reason_match.group(1)
            if raw.endswith('"') and not raw.endswith('\\"'):
                raw = raw[:-1]
            reason = raw.strip()
        return {"verdict": normalize_verdict(verdict_match.group(1)), "reason": reason}

    logger.warning(f"No JSON object found in response: {response_text[:100]}..")
    return {"verdict": "error", "reason": "Failed to parse judge response"}


# ==================== Comparison Module ====================
def compare_single_row(
    judge_model: str,
    prompt: str,
    test_response: str,
    baseline_response: str,
) -> dict:
    """Compare a single row using the judge model (both directions).
    
    This calls the judge model twice:
    1. vs_1: Test vs Baseline (A = Test, B = Baseline)
    2. vs_2: Baseline vs Test (A = Baseline, B = Test)
    
    If either response is empty, auto-verdict without calling the judge:
    - Empty test_response → Test loses (A<<B / A>>B)
    - Empty baseline_response → Baseline loses (A>>B / A<<B)
    - Both empty → tie (A=B)
    
    Returns:
        Dictionary with comparison results
    """
    test_empty = not test_response
    baseline_empty = not baseline_response
    
    if test_empty or baseline_empty:
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
    
    # Direction 1: Test vs Baseline (A = Test, B = Baseline)
    user_prompt_1 = build_user_prompt(prompt, test_response, baseline_response)
    result_1 = call_judge_model(judge_model, user_prompt_1, SYSTEM_PROMPT)
    
    # Direction 2: Baseline vs Test (A = Baseline, B = Test)
    user_prompt_2 = build_user_prompt(prompt, baseline_response, test_response)
    result_2 = call_judge_model(judge_model, user_prompt_2, SYSTEM_PROMPT)
    
    return {
        "vs_1": result_1.get("verdict", "error") if result_1 else "error",
        "vs_2": result_2.get("verdict", "error") if result_2 else "error",
        "vs_1_reason": result_1.get("reason", "") if result_1 else "",
        "vs_2_reason": result_2.get("reason", "") if result_2 else "",
    }


# ==================== Row Preparation Module ====================
def prepare_rows_for_comparison(
    test_df: pd.DataFrame,
    step_baselines: dict[str, pd.DataFrame],
) -> list:
    """Prepare rows for comparison by matching test data with per-step baseline data.
    
    Iterates ALL test rows. For each row, looks up the source_step to find
    the correct baseline DataFrame, then matches by test_id (Step1/Step2)
    or prompt prefix (Step3).
    
    Args:
        test_df: Test dataset DataFrame (all steps)
        step_baselines: Dict mapping step name -> baseline DataFrame
    
    Returns:
        List of row dictionaries ready for comparison
    """
    rows = []
    stats = {"Step1": {"total": 0, "matched": 0, "no_prompt": 0, "unmatched": 0},
             "Step2": {"total": 0, "matched": 0, "no_prompt": 0, "unmatched": 0},
             "Step3": {"total": 0, "matched": 0, "no_prompt": 0, "unmatched": 0}}
    
    for idx, test_row in test_df.iterrows():
        source_step = test_row.get("source_step", "")
        
        if source_step not in stats:
            continue
        stats[source_step]["total"] += 1
        
        # Find the baseline df for this row's step
        if source_step not in step_baselines:
            stats[source_step]["unmatched"] += 1
            continue
        baseline_df = step_baselines[source_step]
        
        test_id = test_row.get("test_id", "")
        prompt = test_row.get("prompt", "")
        test_response = test_row.get("response", "")
        if not isinstance(test_response, str) or test_response.strip().lower() in ("", "nan"):
            test_response = ""
        
        # Skip if prompt is empty
        if not prompt:
            stats[source_step]["no_prompt"] += 1
            continue
        
        # Match with baseline based on step
        matched_baseline = None
        if source_step in ["Step1", "Step2"]:
            matched_baseline = match_step1_step2(test_row, baseline_df, test_id)
        elif source_step == "Step3":
            matched_baseline = match_step3(test_row, baseline_df)
        
        if matched_baseline is None:
            stats[source_step]["unmatched"] += 1
            continue
        
        stats[source_step]["matched"] += 1
        
        baseline_response = matched_baseline.get("response", "")
        if not isinstance(baseline_response, str) or baseline_response.strip().lower() in ("", "nan"):
            baseline_response = ""
        
        rows.append({
            "row_index": idx,
            "source_step": source_step,
            "test_id": test_id,
            "prompt": prompt,
            "test_response": test_response,
            "baseline_response": baseline_response,
        })
    
    # Log matching statistics
    for step, s in stats.items():
        logger.info(f"  {step}: {s['total']} total, {s['matched']} matched, "
                    f"{s['unmatched']} unmatched, {s['no_prompt']} no-prompt")
    
    return rows


# ==================== Single Task Processing ====================
def run_single_comparison_task(
    test_df: pd.DataFrame,
    step_baselines: dict[str, pd.DataFrame],
    baseline_model: str,
    judge_model: str,
    output_dir: Path,
    progress_db: str,
    completed_rows_lock: threading.Lock,
    completed_rows: dict,
) -> tuple:
    """
    Run comparison for a single baseline×judge combination with SQLite recovery.
    
    Matches all test rows against per-step baseline DataFrames,
    producing a single output file with rows from all steps combined.
    
    Args:
        test_df: Test dataset DataFrame
        step_baselines: Dict mapping step name -> baseline DataFrame
        baseline_model: Baseline model name (canonical)
        judge_model: Judge model name
        output_dir: Output directory
        progress_db: Path to SQLite progress DB
        completed_rows_lock: Lock for thread-safe access to completed_rows
        completed_rows: Dict tracking completed row_indices per direction
    
    Returns:
        (output_path, success_count, skip_count) tuple
    """
    judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
    output_filename = f"Test_vs_{baseline_model}_{judge_short}.csv"
    output_path = output_dir / output_filename
    
    logger.info(f"Starting comparison: baseline='{baseline_model}' vs test using judge='{judge_short}'")
    
    # Prepare rows for comparison across all steps
    rows = prepare_rows_for_comparison(test_df, step_baselines)
    total_rows = len(rows)
    
    if total_rows == 0:
        logger.warning(f"No valid rows for comparison: {baseline_model} × {judge_short}")
        return output_path, 0, 0
    
    logger.info(f"Prepared {total_rows} rows for comparison")
    
    # Initialize or resume SQLite progress DB
    conn = db_connect(progress_db)
    db_init(conn, rows, output_filename, baseline_model, judge_model, "all")
    
    # Load already completed rows for each direction
    done_vs_1 = db_completed_rows_for_direction(conn, "vs_1")
    done_vs_2 = db_completed_rows_for_direction(conn, "vs_2")
    
    logger.info(f"Resumed: vs_1 done={len(done_vs_1)}/{total_rows}, vs_2 done={len(done_vs_2)}/{total_rows}")
    
    # Thread-safe tracking
    with completed_rows_lock:
        completed_rows["vs_1"] = done_vs_1.copy()
        completed_rows["vs_2"] = done_vs_2.copy()
    
    success_count = 0
    skip_count = len(done_vs_1) + len(done_vs_2)
    
    # Process rows using ThreadPoolExecutor for async comparison
    def process_row(row_dict):
        """Process a single row and persist results to SQLite."""
        row_index = row_dict["row_index"]
        prompt = row_dict["prompt"]
        test_response = row_dict["test_response"]
        baseline_response = row_dict["baseline_response"]
        
        # Check if already completed
        if row_index in done_vs_1 and row_index in done_vs_2:
            return None, "skipped"
        
        try:
            result = compare_single_row(
                judge_model, prompt, test_response, baseline_response
            )
            
            # Persist vs_1 result
            db_upsert_result(conn, completed_rows_lock, row_index, "vs_1",
                           result["vs_1"], result["vs_1_reason"])
            with completed_rows_lock:
                completed_rows["vs_1"].add(row_index)
            
            # Persist vs_2 result
            db_upsert_result(conn, completed_rows_lock, row_index, "vs_2",
                           result["vs_2"], result["vs_2_reason"])
            with completed_rows_lock:
                completed_rows["vs_2"].add(row_index)
            
            return result, "success"
        
        except Exception as e:
            logger.error(f"Error processing row {row_index}: {e}")
            # Persist error result
            db_upsert_result(conn, completed_rows_lock, row_index, "vs_1",
                           "error", f"Processing error: {str(e)}")
            db_upsert_result(conn, completed_rows_lock, row_index, "vs_2",
                           "error", f"Processing error: {str(e)}")
            with completed_rows_lock:
                completed_rows["vs_1"].add(row_index)
                completed_rows["vs_2"].add(row_index)
            return None, "error"
    
    # Use ThreadPoolExecutor to process rows
    # Note: Each row processing creates its own event loop for async operations
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PER_MODEL) as executor:
        # Submit all rows
        future_to_row = {
            executor.submit(process_row, row): row
            for row in rows
            if row["row_index"] not in done_vs_1 or row["row_index"] not in done_vs_2
        }
        
        # Process completed futures
        for future in as_completed(future_to_row):
            row = future_to_row[future]
            try:
                result, status = future.result()
                if status == "success":
                    success_count += 1
                    if success_count % 10 == 0:
                        logger.info(f"  Progress: {success_count} rows completed")
                elif status == "error":
                    logger.warning(f"  Row {row['row_index']} completed with errors")
            except Exception as e:
                logger.error(f"  Future exception for row {row['row_index']}: {e}")
    
    # Export final CSV from DB
    logger.info(f"Exporting results to {output_path}")
    db_export_csv(conn, output_path, baseline_model, judge_model)
    conn.close()
    
    # Remove progress DB now that export succeeded
    try:
        if os.path.exists(progress_db):
            os.remove(progress_db)
    except OSError as e:
        logger.warning(f"Failed to remove progress DB: {e}")
    
    logger.info(
        f"Comparison complete: baseline='{baseline_model}', judge='{judge_short}' "
        f"success={success_count}, skipped={skip_count}"
    )
    
    return output_path, success_count, skip_count


# ==================== Main Orchestrator Module ====================
def run_all_comparisons(test_mode: bool = False, baseline_models: Optional[list] = None):
    """Run all comparisons between test dataset and baseline files.
    
    Flow (design requirement order):
    1. Load test dataset
    2. For each judge model
    3.   For each unique baseline model (canonical name)
    4.     Load Step1+Step2+Step3 baseline DataFrames
    5.     Match all test rows against the correct per-step baseline
    6.     Run bidirectional comparison
    7.     Save single output file
    
    Args:
        test_mode: If True, use test mode configuration
        baseline_models: Optional list of canonical baseline model names to process.
                         If None, process all baseline models.
    """
    logger.info("=" * 80)
    logger.info("  Test vs Baseline Comparison Tool (Enhanced with SQLite Recovery)")
    logger.info("=" * 80)
    
    # Load recovery state
    completed_tasks = load_recovery_state()
    if completed_tasks:
        logger.info(f"Loaded {len(completed_tasks)} completed tasks from recovery state")
    
    # Load test dataset
    test_df = load_test_dataset(TEST_DATASET_PATH)
    
    # Track statistics
    total_success = 0
    total_files = 0
    total_skipped = 0
    
    # Step 1: Load all baseline data
    all_baseline_data = {}  # (step, raw_model_name) -> baseline_df
    
    for step in ["Step1", "Step2", "Step3"]:
        baseline_files = get_baseline_files_for_step(step)
        if not baseline_files:
            logger.warning(f"No baseline files found for {step}")
            continue
        
        for baseline_file in baseline_files:
            baseline_path = BASELINE_DIR / baseline_file
            if not baseline_path.exists():
                logger.warning(f"Baseline file not found: {baseline_path}")
                continue
            
            try:
                baseline_df = load_baseline_file(baseline_path)
                model_name = extract_model_name(baseline_file)
                all_baseline_data[(step, model_name)] = baseline_df
                logger.info(f"Loaded baseline: {step}/{model_name} ({len(baseline_df)} rows)")
            except Exception as e:
                logger.error(f"Error loading baseline file {baseline_file}: {e}")
    
    if not all_baseline_data:
        logger.error("No baseline data loaded. Exiting.")
        return
    
    # Step 2: Group raw model names by canonical name
    # e.g. "glm4_7" (Step3) -> "glm-4_7-flash_q8_0" to merge with Step1/Step2
    canonical_baseline_groups = {}  # canonical_name -> {step: df}
    for (step, raw_name), df in all_baseline_data.items():
        canonical = MODEL_NAME_ALIASES.get(raw_name, raw_name)
        if canonical not in canonical_baseline_groups:
            canonical_baseline_groups[canonical] = {}
        canonical_baseline_groups[canonical][step] = df
    
    logger.info(f"Grouped into {len(canonical_baseline_groups)} unique baseline models")
    for canonical, steps in canonical_baseline_groups.items():
        logger.info(f"  {canonical}: steps={list(steps.keys())}")
    
    # Filter to requested baseline models if specified
    if baseline_models:
        filtered = {k: v for k, v in canonical_baseline_groups.items() if k in baseline_models}
        skipped_models = set(canonical_baseline_groups.keys()) - set(filtered.keys())
        if skipped_models:
            logger.info(f"Skipping models not in filter: {sorted(skipped_models)}")
        canonical_baseline_groups = filtered
        if not canonical_baseline_groups:
            logger.error(f"No matching baseline models found. Available: {list(canonical_baseline_groups.keys())}")
            return
    
    # Clear recovery state for targeted models so they are re-processed
    if baseline_models:
        existing_tasks = load_recovery_state()
        tasks_to_remove = {t for t in existing_tasks if any(t.startswith(f"{m}|") for m in baseline_models)}
        if tasks_to_remove:
            remaining = existing_tasks - tasks_to_remove
            save_recovery_state(remaining)
            logger.info(f"Cleared {len(tasks_to_remove)} stale recovery entries for: {baseline_models}")
            # Reload cleared state
            completed_tasks = load_recovery_state()
    
    # Process order: judge model outer loop, baseline inner loop (design requirement)
    for judge_model in JUDGE_MODELS:
        judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing judge model: {judge_short}")
        logger.info(f"{'='*60}")
        
        for baseline_model, step_baselines in canonical_baseline_groups.items():
            task_key = f"{baseline_model}|{judge_model}"
            
            # Check file-level recovery
            if task_key in completed_tasks:
                output_filename = f"Test_vs_{baseline_model}_{judge_short}.csv"
                output_path = OUTPUT_DIR / output_filename
                if output_path.exists():
                    logger.info(f"[SKIP] Task already completed: {baseline_model} × {judge_short}")
                    total_files += 1
                    total_skipped += 1
                    continue
                else:
                    logger.info(f"[RE-RUN] Task marked complete but file missing: {baseline_model} × {judge_short}")
            
            # Generate output paths
            progress_db = db_path_for_task(baseline_model, judge_model, OUTPUT_DIR)
            
            try:
                output_path, success_count, skip_count = run_single_comparison_task(
                    test_df,
                    step_baselines,
                    baseline_model,
                    judge_model,
                    OUTPUT_DIR,
                    progress_db,
                    threading.Lock(),
                    {"vs_1": set(), "vs_2": set()},
                )
                
                if success_count > 0 or skip_count > 0:
                    total_success += success_count
                    total_files += 1
                    
                    # Mark task as completed in recovery state
                    completed_tasks.add(task_key)
                    save_recovery_state(completed_tasks)
                    
                    logger.info(f"Output saved: {output_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to complete comparison for {baseline_model} × {judge_short}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"  COMPARISON COMPLETE")
    logger.info(f"  Total files generated: {total_files}")
    logger.info(f"  Total successful comparisons: {total_success}")
    logger.info(f"  Total skipped (recovered): {total_skipped}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*80}")


# ==================== Entry Point ====================
def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test vs Baseline Comparison Tool (Enhanced with SQLite Recovery)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python Test_vs_baseline_comparison.py                                    # Full mode
  python Test_vs_baseline_comparison.py --baseline-model qwen3_5_9b        # One baseline only
  python Test_vs_baseline_comparison.py --baseline-model qwen3_5_9b --multi_gpu
  python Test_vs_baseline_comparison.py --single_gpu                       # Single GPU

Scheduling modes:
  --single_gpu   (default) Process comparisons sequentially, 2 parallel requests per comparison
                  Use when running on a single GPU
  --multi_gpu    Process 2 comparisons concurrently, 2 requests each
                  Use when running on multiple GPUs for better utilization

Baseline model filtering:
  --baseline-model  Process only matching baseline model(s). Comma-separated for multiple.
                    Clears stale recovery state for targeted models so they re-run.
                    Examples: qwen3_5_9b, qwen3_5_4b, qwen3_5_35b-a3b, glm-4_7-flash_q8_0
        """
    )
    parser.add_argument('--test', action='store_true',
                        help='Run in test mode')
    parser.add_argument('--single_gpu', action='store_true',
                        help='Single GPU mode: sequential comparisons with 2 parallel requests each')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Multi GPU mode: 2 concurrent comparisons with 2 requests each')
    parser.add_argument('--baseline-model', type=str, default=None,
                        help='Process only this baseline model (canonical name, e.g. qwen3_5_9b). '
                             'Clears stale recovery state for this model. Can specify multiple times.')
    args = parser.parse_args()
    
    # Determine scheduling mode
    global MAX_CONCURRENT_MODELS, MAX_PARALLEL_PER_MODEL
    if args.multi_gpu and not args.single_gpu:
        MAX_CONCURRENT_MODELS = 2
        MAX_PARALLEL_PER_MODEL = 2
        scheduling_mode = "MULTI_GPU (2 concurrent comparisons, 2 requests each)"
    else:
        MAX_CONCURRENT_MODELS = 1
        MAX_PARALLEL_PER_MODEL = 2
        scheduling_mode = "SINGLE_GPU (1 sequential comparison, 2 requests each)"
    
    print("=" * 70)
    mode_str = "TEST MODE" if args.test else "FULL MODE"
    print(f"  Test vs Baseline Comparison Tool ({mode_str})")
    print(f"  Scheduling mode: {scheduling_mode}")
    if args.baseline_model:
        print(f"  Baseline filter: {args.baseline_model}")
    print("=" * 70)
    
    # Determine which baseline models to process
    baseline_models = None
    if args.baseline_model:
        baseline_models = [m.strip() for m in args.baseline_model.split(",")]
        logger.info(f"Filtering to baseline models: {baseline_models}")
    
    logger.info(f"Starting Test vs Baseline Comparison...")
    logger.info(f"Mode: {mode_str}, Scheduling: {scheduling_mode}")
    
    # Run the async main function
    run_all_comparisons(test_mode=args.test, baseline_models=baseline_models)
    
    logger.info("All comparisons completed!")


if __name__ == "__main__":
    main()
