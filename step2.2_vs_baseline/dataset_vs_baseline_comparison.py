"""
Dataset vs Baseline Comparison Tool

Compares every Dataset CSV file against every Baseline CSV file of the same
step (Step1/Step2/Step3) using judge models via Ollama API.

Data source:  Dataset/Step{1,2,3}/  (scanned recursively)
Baseline:     Baseline/Step{1,2,3}/
Output:       vs_baseline/{StepX}/{judge_short}/{dataset_model_id}_vs_{baseline_model}.csv

For each comparison, performs bidirectional judgement (dataset->baseline and
baseline->dataset) using multiple judge models.

Recovery Design (SQLite-backed):
    - Each output CSV task gets a SQLite progress DB in the output subdirectory.
    - Rows are pre-populated at start; each completed (row_index, direction)
      result is UPSERT-ed immediately on receipt — no data loss on interruption.
    - On resume, already-completed (row_index, direction) pairs are skipped.
    - After all rows finish, the DB is exported to the final CSV and removed.
    - File-level recovery skips entirely completed dataset×baseline×judge combos.

Usage:
    # Requires: set OLLAMA_NUM_PARALLEL=4 before starting Ollama
    #   Windows (cmd): set OLLAMA_NUM_PARALLEL=4 && ollama serve
    #   Linux:          OLLAMA_NUM_PARALLEL=4 ollama serve
    python dataset_vs_baseline_comparison.py                    # Full mode (default, 4 parallel)
    python dataset_vs_baseline_comparison.py --single_gpu       # Single GPU (4 parallel)
    python dataset_vs_baseline_comparison.py --multi_gpu        # Multi GPU (8 parallel total)
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_sources import failed as response_failed

sys.path.insert(0, str(Path(__file__).parent))
from dataset_vs_baseline_config import (
    OLLAMA_URL,
    MAX_PARALLEL,
    TIMEOUT_SECONDS,
    MAX_RETRIES,
    NUM_PREDICT,
    JUDGE_MODELS,
    MODEL_SHORT_NAME,
    DATASET_DIR,
    BASELINE_DIR,
    OUTPUT_DIR,
    STEPS,
    SYSTEM_PROMPT,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    BASELINE_CANONICAL_NAMES,
)

# ==================== Processing Settings ====================
# Requires OLLAMA_NUM_PARALLEL=4 in Ollama env (set before starting Ollama)
MAX_CONCURRENT_MODELS = 1
MAX_PARALLEL_PER_MODEL = MAX_PARALLEL

RECOVERY_FILE = OUTPUT_DIR / ".dataset_comparison_same_question_v2_recovery_state.json"
VALID_VERDICTS = ["A>>B", "A>B", "A=B", "A<B", "A<<B"]
REPAIRABLE_VERDICTS = {"", "error"}

# ==================== Logging Setup ====================
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "dataset_vs_baseline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ==================== SQLite Progress DB ====================
def db_path_for_task(dataset_model_id: str, baseline_model: str, judge_model: str, output_dir: Path) -> str:
    judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
    safe_dataset = re.sub(r'[^\w\-]', '_', dataset_model_id)
    safe_baseline = re.sub(r'[^\w\-]', '_', baseline_model)
    safe_judge = re.sub(r'[^\w\-]', '_', judge_short)
    db_name = f".progress_same_question_v2_{safe_dataset}_vs_{safe_baseline}_{safe_judge}.db"
    return str(output_dir / db_name)


def output_path_for_task(dataset_model_id: str, baseline_model: str, output_dir: Path) -> Path:
    safe_dataset = re.sub(r'[^\w\-]', '_', dataset_model_id)
    safe_baseline = re.sub(r'[^\w\-]', '_', baseline_model)
    return output_dir / f"{safe_dataset}_vs_{safe_baseline}.csv"


def db_connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def db_init(conn: sqlite3.Connection, rows: list, output_filename: str,
            dataset_model_id: str, baseline_model: str, judge_model: str, step: str):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS rows (
            row_index        INTEGER PRIMARY KEY,
            test_id          TEXT,
            prompt           TEXT,
            dataset_response TEXT,
            baseline_response TEXT
        );
        CREATE TABLE IF NOT EXISTS results (
            row_index   INTEGER NOT NULL,
            direction   TEXT    NOT NULL,
            verdict     TEXT    DEFAULT '',
            reason      TEXT    DEFAULT '',
            PRIMARY KEY (row_index, direction)
        );
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    expected_rows = [(i, str(row.get('test_id', '')), str(row.get('prompt', '')),
                      str(row.get('dataset_response', '')), str(row.get('baseline_response', '')))
                     for i, row in enumerate(rows)]
    existing_rows = conn.execute('SELECT row_index, test_id, prompt, dataset_response, baseline_response FROM rows ORDER BY row_index').fetchall()
    if existing_rows and existing_rows != expected_rows:
        raise ValueError('Saved comparison progress belongs to different question/answer content; use a separate progress DB')
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('output_filename', ?)", (output_filename,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('dataset_model_id', ?)", (dataset_model_id,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('baseline_model', ?)", (baseline_model,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('judge_model', ?)", (judge_model,))
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('step', ?)", (step,))
    conn.executemany(
        "INSERT OR IGNORE INTO rows VALUES (?,?,?,?,?)",
        [
            (i, str(row.get("test_id", "")), str(row.get("prompt", "")),
             str(row.get("dataset_response", "")), str(row.get("baseline_response", "")))
            for i, row in enumerate(rows)
        ],
    )
    conn.commit()


def db_upsert_result(conn: sqlite3.Connection, lock: threading.Lock,
                     row_index: int, direction: str, verdict: str, reason: str):
    with lock:
        conn.execute(
            "INSERT OR REPLACE INTO results (row_index, direction, verdict, reason) VALUES (?,?,?,?)",
            (row_index, direction, verdict, reason),
        )
        conn.commit()


def db_completed_rows_for_direction(conn: sqlite3.Connection, direction: str) -> set:
    placeholders = ",".join("?" for _ in VALID_VERDICTS)
    cur = conn.execute(
        f"SELECT row_index FROM results WHERE direction=? AND verdict IN ({placeholders})",
        (direction, *VALID_VERDICTS),
    )
    return {row[0] for row in cur.fetchall()}


def db_export_csv(conn: sqlite3.Connection, output_path: Path):
    cur = conn.execute("SELECT * FROM rows ORDER BY row_index")
    base_rows = {r[0]: list(r) for r in cur.fetchall()}
    base_cols = ["row_index", "test_id", "prompt", "dataset_response", "baseline_response"]

    cur = conn.execute("SELECT row_index, direction, verdict, reason FROM results")
    result_map = {}
    for rid, direction, v, r in cur.fetchall():
        result_map[(rid, direction)] = (v, r)

    header = ["test_id", "prompt", "dataset_response", "baseline_response",
              "vs_1", "vs_2", "vs_1_reason", "vs_2_reason"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for rid in sorted(base_rows.keys()):
            row_dict = dict(zip(base_cols, base_rows[rid]))
            v1, r1 = result_map.get((rid, "vs_1"), ("", ""))
            v2, r2 = result_map.get((rid, "vs_2"), ("", ""))
            row_dict["vs_1"] = v1
            row_dict["vs_2"] = v2
            row_dict["vs_1_reason"] = r1
            row_dict["vs_2_reason"] = r2
            row_dict.pop("row_index", None)
            writer.writerow(row_dict)


# ==================== File-level Recovery State ====================
def load_recovery_state() -> set:
    if RECOVERY_FILE.exists():
        try:
            with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            return set(state.get("completed_tasks", []))
        except Exception as e:
            logger.warning(f"Failed to load recovery state: {e}")
    return set()


def save_recovery_state(completed_tasks: set):
    state = {
        "completed_tasks": list(completed_tasks),
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(RECOVERY_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def clear_recovery_state():
    if RECOVERY_FILE.exists():
        RECOVERY_FILE.unlink()


# ==================== Dataset & Baseline File Scanning ====================
def extract_dataset_model_info(filepath: Path) -> dict:
    """
    Extract model identity from a Dataset CSV file path.

    Returns:
        dict with keys: step, model_name, true_false, report_round, model_id
    """
    step = filepath.parent.parent.name
    report_round = filepath.parent.name
    stem = filepath.stem

    true_false = ""
    model_name = stem

    if "dify" in stem:
        match = re.search(r'dify_(.+?)_(true|false)_report', stem)
        if match:
            model_name = match.group(1)
            true_false = match.group(2)
            model_name = model_name.replace(".", "_").replace("-", "_")
    else:
        match = re.search(r'test_(.+?)_(True|False)_', stem)
        if match:
            model_name = match.group(1)
            true_false = match.group(2)
            model_name = model_name.replace(".", "_")

    model_id = f"{model_name}_{true_false}_{report_round}" if true_false else f"{model_name}_{report_round}"

    return {
        "step": step,
        "model_name": model_name,
        "true_false": true_false,
        "report_round": report_round,
        "model_id": model_id,
    }


def scan_dataset_files(dataset_dir: Path) -> dict:
    """
    Recursively scan Dataset/Step{1,2,3}/ for CSV files.

    Returns:
        dict: {step: [list of info dicts sorted by model_id]}
    """
    result = {step: [] for step in STEPS}
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return result

    for step_dir in sorted(dataset_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name not in STEPS:
            continue
        step = step_dir.name
        for csv_path in sorted(step_dir.rglob("*.csv")):
            parent = csv_path.parent.name
            if parent not in ("reports1", "reports2"):
                continue
            info = extract_dataset_model_info(csv_path)
            info["csv_path"] = csv_path
            result[step].append(info)

    for step in STEPS:
        result[step].sort(key=lambda x: x["model_id"])

    return result


def normalize_model_name(raw: str) -> str:
    """Normalize separators and strip artifacts, then map to canonical name."""
    name = raw.replace(".", "_").replace("-", "_")
    name = re.sub(r'_+', '_', name).strip('_')
    return BASELINE_CANONICAL_NAMES.get(name, name)


def extract_baseline_model_name(filepath: Path) -> str:
    """
    Extract model name from a Baseline CSV filename, normalized across steps.

    Handles both Step1/2 format (test_<model>_<True/False>_<timestamp>_annotated)
    and Step3 format (<timestamp>_dify_<model>_report).

    Returns a canonical name consistent across all steps.
    """
    name = filepath.stem

    if "dify" in name:
        match = re.search(r'dify_(.+?)_report', name)
        if match:
            return normalize_model_name(match.group(1))

    match = re.search(r'test_(.+?)_(?:False|True)_', name)
    if match:
        model = match.group(1)
        # Remove quantization suffix (can be -q8_0 or _q8_0)
        model = re.sub(r'[-_]q\d+[_\w]*', '', model)
        return normalize_model_name(model)

    # Fallback: clean up remaining patterns
    fallback = re.sub(r'[_\-](?:False|True|annotated|report|csv).*$', '', name)
    return normalize_model_name(fallback)


def scan_baseline_files(baseline_dir: Path) -> dict:
    """
    Scan Baseline/Step{1,2,3}/ for CSV files.

    Returns:
        dict: {step: [list of {model_name, csv_path} sorted by model_name]}
    """
    result = {step: [] for step in STEPS}
    if not baseline_dir.exists():
        logger.error(f"Baseline directory not found: {baseline_dir}")
        return result

    for step_dir in sorted(baseline_dir.iterdir()):
        if not step_dir.is_dir() or step_dir.name not in STEPS:
            continue
        step = step_dir.name
        for csv_path in sorted(step_dir.glob("*.csv")):
            if ".sync-conflict-" in csv_path.name or "recover" in csv_path.stem:
                continue
            model_name = extract_baseline_model_name(csv_path)
            result[step].append({"model_name": model_name, "csv_path": csv_path})

    for step in STEPS:
        result[step].sort(key=lambda x: x["model_name"])

    return result


# ==================== Data Loading ====================
def load_datalist_csv(path: Path) -> pd.DataFrame:
    """Load a single Dataset or Baseline CSV."""
    logger.debug(f"Loading file: {path}")
    df = pd.read_csv(path, dtype=str)
    return df


# ==================== Row Matching Module ====================
def match_and_prepare_rows(dataset_df: pd.DataFrame, baseline_df: pd.DataFrame, step: str) -> list:
    """Match task identifier AND full prompt; keep main-task and question identity."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pipeline_sources import match_questions, clean
    merged = match_questions(dataset_df, baseline_df, step)
    id_col = "task_id" if step == "Step3" else "test_id"
    return [{"row_index": i, "test_id": str(row[id_col]),
             "prompt": row["_prompt_key"],
             "dataset_response": clean(row["response_dataset"]),
             "baseline_response": clean(row["response_baseline"])}
            for i, (_, row) in enumerate(merged.iterrows())]



# ==================== Prompt Builder Module ====================
def build_user_prompt(prompt: str, response_a: str, response_b: str) -> str:
    user_prompt = f"""## Task context
{prompt}

## Response A
{response_a}

## Response B
{response_b}"""
    return user_prompt


# ==================== Judge Caller Module ====================
def build_ollama_payload(model: str, prompt: str, system: str) -> dict:
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
            "num_predict": NUM_PREDICT,
            "num_ctx": 40960,
        }
    }


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


def call_judge_model(
    model: str,
    prompt: str,
    system: str,
    max_retries: int = MAX_RETRIES,
    timeout: Optional[int] = None
) -> Optional[dict]:
    if timeout is None:
        timeout = TIMEOUT_SECONDS
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
            logger.warning(
                f"Timeout calling model {model}, attempt {attempt}/{max_retries}, "
                f"timeout={timeout}s, prompt_chars={len(prompt)}"
            )
        except requests.exceptions.ConnectionError as e:
            logger.warning(
                f"Network error calling model {model}: {e}, attempt {attempt}/{max_retries}, "
                f"prompt_chars={len(prompt)}"
            )
        except Exception as e:
            logger.warning(f"Unexpected error calling model {model}: {e}, attempt {attempt}/{max_retries}")

        if attempt < max_retries:
            time.sleep(1 * attempt)

    logger.error(f"Failed to get valid response from model {model} after {max_retries} retries")
    return None


def parse_judge_response(response_text: str) -> Optional[dict]:
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


def repair_json_escapes(json_str: str) -> str:
    """Escape backslashes that are valid text but invalid JSON escapes."""
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)


def parse_judge_response_fallback(json_str: str) -> Optional[dict]:
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


# ==================== Comparison Module ====================
def compare_single_row(
    judge_model: str,
    prompt: str,
    dataset_response: str,
    baseline_response: str,
) -> dict:
    dataset_empty = response_failed(dataset_response)
    baseline_empty = response_failed(baseline_response)

    if dataset_empty or baseline_empty:
        if dataset_empty and baseline_empty:
            return {
                "vs_1": "A=B", "vs_2": "A=B",
                "vs_1_reason": "Both responses are empty",
                "vs_2_reason": "Both responses are empty",
            }
        elif dataset_empty:
            return {
                "vs_1": "A<<B", "vs_2": "A>>B",
                "vs_1_reason": "Dataset response is empty, baseline wins",
                "vs_2_reason": "Dataset response is empty, baseline wins",
            }
        else:
            return {
                "vs_1": "A>>B", "vs_2": "A<<B",
                "vs_1_reason": "Baseline response is empty, dataset wins",
                "vs_2_reason": "Baseline response is empty, dataset wins",
            }

    user_prompt_1 = build_user_prompt(prompt, dataset_response, baseline_response)
    result_1 = call_judge_model(judge_model, user_prompt_1, SYSTEM_PROMPT)

    user_prompt_2 = build_user_prompt(prompt, baseline_response, dataset_response)
    result_2 = call_judge_model(judge_model, user_prompt_2, SYSTEM_PROMPT)

    return {
        "vs_1": result_1.get("verdict", "error") if result_1 else "error",
        "vs_2": result_2.get("verdict", "error") if result_2 else "error",
        "vs_1_reason": result_1.get("reason", "") if result_1 else "",
        "vs_2_reason": result_2.get("reason", "") if result_2 else "",
    }


def compare_single_direction(
    judge_model: str,
    prompt: str,
    dataset_response: str,
    baseline_response: str,
    direction: str,
) -> tuple[str, str]:
    if direction not in ("vs_1", "vs_2"):
        raise ValueError(f"Unknown comparison direction: {direction}")

    if response_failed(dataset_response) or response_failed(baseline_response):
        result = compare_single_row(judge_model, prompt, dataset_response, baseline_response)
        return result[direction], result[f"{direction}_reason"]

    if direction == "vs_1":
        user_prompt = build_user_prompt(prompt, dataset_response, baseline_response)
    else:
        user_prompt = build_user_prompt(prompt, baseline_response, dataset_response)

    result = call_judge_model(judge_model, user_prompt, SYSTEM_PROMPT)
    if not result:
        return "error", ""
    return result.get("verdict", "error"), result.get("reason", "")


# ==================== Single Task Processing ====================
def run_single_comparison_task(
    dataset_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    step: str,
    dataset_model_id: str,
    baseline_model: str,
    judge_model: str,
    output_dir: Path,
    progress_db: str,
    completed_rows_lock: threading.Lock,
    completed_rows: dict,
) -> tuple:
    judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
    output_path = output_path_for_task(dataset_model_id, baseline_model, output_dir)
    output_filename = output_path.name

    logger.info(
        f"Starting: dataset='{dataset_model_id}' vs baseline='{baseline_model}' "
        f"judge='{judge_short}' (step={step})"
    )

    rows = match_and_prepare_rows(dataset_df, baseline_df, step)
    total_rows = len(rows)

    if total_rows == 0:
        logger.warning(f"No matching rows: {dataset_model_id} vs {baseline_model}")
        return output_path, 0, 0

    logger.info(f"Prepared {total_rows} matched rows")

    conn = db_connect(progress_db)
    db_init(conn, rows, output_filename, dataset_model_id, baseline_model, judge_model, step)

    done_vs_1 = db_completed_rows_for_direction(conn, "vs_1")
    done_vs_2 = db_completed_rows_for_direction(conn, "vs_2")

    logger.info(f"Resumed: vs_1 done={len(done_vs_1)}/{total_rows}, vs_2 done={len(done_vs_2)}/{total_rows}")

    with completed_rows_lock:
        completed_rows["vs_1"] = done_vs_1.copy()
        completed_rows["vs_2"] = done_vs_2.copy()

    success_count = 0
    skip_count = len(done_vs_1) + len(done_vs_2)

    def process_row(row_dict):
        row_index = row_dict["row_index"]
        prompt = row_dict["prompt"]
        dataset_response = row_dict["dataset_response"]
        baseline_response = row_dict["baseline_response"]

        if row_index in done_vs_1 and row_index in done_vs_2:
            return None, "skipped"

        try:
            result = {}

            if row_index not in done_vs_1:
                verdict, reason = compare_single_direction(
                    judge_model, prompt, dataset_response, baseline_response, "vs_1"
                )
                result["vs_1"] = verdict
                result["vs_1_reason"] = reason
                db_upsert_result(conn, completed_rows_lock, row_index, "vs_1", verdict, reason)
                with completed_rows_lock:
                    completed_rows["vs_1"].add(row_index)

            if row_index not in done_vs_2:
                verdict, reason = compare_single_direction(
                    judge_model, prompt, dataset_response, baseline_response, "vs_2"
                )
                result["vs_2"] = verdict
                result["vs_2_reason"] = reason
                db_upsert_result(conn, completed_rows_lock, row_index, "vs_2", verdict, reason)
                with completed_rows_lock:
                    completed_rows["vs_2"].add(row_index)

            return result, "success"

        except Exception as e:
            logger.error(f"Error processing row {row_index}: {e}")
            db_upsert_result(conn, completed_rows_lock, row_index, "vs_1",
                           "error", f"Processing error: {str(e)}")
            db_upsert_result(conn, completed_rows_lock, row_index, "vs_2",
                           "error", f"Processing error: {str(e)}")
            with completed_rows_lock:
                completed_rows["vs_1"].add(row_index)
                completed_rows["vs_2"].add(row_index)
            return None, "error"

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PER_MODEL) as executor:
        future_to_row = {
            executor.submit(process_row, row): row
            for row in rows
            if row["row_index"] not in done_vs_1 or row["row_index"] not in done_vs_2
        }

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

    logger.info(f"Exporting results to {output_path}")
    db_export_csv(conn, output_path)
    conn.close()

    try:
        if os.path.exists(progress_db):
            os.remove(progress_db)
    except OSError as e:
        logger.warning(f"Failed to remove progress DB: {e}")

    logger.info(
        f"Complete: dataset='{dataset_model_id}', baseline='{baseline_model}', judge='{judge_short}' "
        f"success={success_count}, skipped={skip_count}"
    )

    return output_path, success_count, skip_count


# ==================== Completed CSV Repair ====================
def is_repairable_verdict(value: Any) -> bool:
    return str(value or "").strip().lower() in REPAIRABLE_VERDICTS


def normalize_identity_value(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n")


def read_output_csv(output_path: Path) -> tuple[list[str], list[dict]]:
    with open(output_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def validate_completed_csv_alignment(
    output_path: Path,
    fieldnames: list[str],
    csv_rows: list[dict],
    prepared_rows: list[dict],
) -> bool:
    required = [
        "test_id", "prompt", "dataset_response", "baseline_response",
        "vs_1", "vs_2", "vs_1_reason", "vs_2_reason",
    ]
    missing = [col for col in required if col not in fieldnames]
    if missing:
        logger.warning(f"Repair skipped for {output_path}: missing columns {missing}")
        return False

    if len(csv_rows) != len(prepared_rows):
        logger.warning(
            f"Repair skipped for {output_path}: row count mismatch "
            f"csv={len(csv_rows)} matched={len(prepared_rows)}"
        )
        return False

    identity_columns = [
        ("test_id", "test_id"),
        ("prompt", "prompt"),
        ("dataset_response", "dataset_response"),
        ("baseline_response", "baseline_response"),
    ]
    for i, (csv_row, prepared_row) in enumerate(zip(csv_rows, prepared_rows)):
        for csv_col, prepared_key in identity_columns:
            csv_value = normalize_identity_value(csv_row.get(csv_col, ""))
            prepared_value = normalize_identity_value(prepared_row.get(prepared_key, ""))
            if csv_value != prepared_value:
                logger.warning(
                    f"Repair skipped for {output_path}: identity mismatch at row {i} "
                    f"column={csv_col} test_id={csv_row.get('test_id', '')}"
                )
                return False

    return True


def write_repaired_csv(
    output_path: Path,
    fieldnames: list[str],
    rows: list[dict],
    prepared_rows: list[dict],
) -> bool:
    tmp_path = output_path.with_name(output_path.name + ".repair.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    tmp_fieldnames, tmp_rows = read_output_csv(tmp_path)
    if not validate_completed_csv_alignment(tmp_path, tmp_fieldnames, tmp_rows, prepared_rows):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return False

    os.replace(tmp_path, output_path)
    return True


def repair_completed_csv_errors(
    output_path: Path,
    prepared_rows: list[dict],
    step: str,
    dataset_model_id: str,
    dataset_path: Path,
    baseline_model: str,
    baseline_path: Path,
    judge_model: str,
) -> tuple[int, int, int]:
    fieldnames, csv_rows = read_output_csv(output_path)
    if not validate_completed_csv_alignment(output_path, fieldnames, csv_rows, prepared_rows):
        return -1, 0, 0

    repair_targets = []
    for row_index, csv_row in enumerate(csv_rows):
        if is_repairable_verdict(csv_row.get("vs_1")):
            repair_targets.append((row_index, "vs_1"))
        if is_repairable_verdict(csv_row.get("vs_2")):
            repair_targets.append((row_index, "vs_2"))

    if not repair_targets:
        return 0, 0, 0

    judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
    vs_1_count = sum(1 for _, direction in repair_targets if direction == "vs_1")
    vs_2_count = len(repair_targets) - vs_1_count
    logger.info(
        f"  [REPAIR] {dataset_model_id} vs {baseline_model} "
        f"judge='{judge_short}' step={step} errors: vs_1={vs_1_count}, vs_2={vs_2_count}"
    )
    logger.info(f"    dataset_file={dataset_path}")
    logger.info(f"    baseline_file={baseline_path}")
    logger.info(f"    output_file={output_path}")

    fixed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PER_MODEL) as executor:
        future_to_target = {}
        for row_index, direction in repair_targets:
            row = prepared_rows[row_index]
            future = executor.submit(
                compare_single_direction,
                judge_model,
                row["prompt"],
                row["dataset_response"],
                row["baseline_response"],
                direction,
            )
            future_to_target[future] = (row_index, direction)

        for future in as_completed(future_to_target):
            row_index, direction = future_to_target[future]
            try:
                verdict, reason = future.result()
            except Exception as e:
                logger.warning(
                    f"    Repair failed for row={row_index} direction={direction}: {e}"
                )
                continue

            verdict_norm = normalize_verdict(verdict)
            if verdict_norm in VALID_VERDICTS:
                csv_rows[row_index][direction] = verdict_norm
                csv_rows[row_index][f"{direction}_reason"] = reason
                fixed_count += 1
                if fixed_count % 10 == 0:
                    logger.info(f"    Repair progress: fixed {fixed_count}/{len(repair_targets)}")
            else:
                logger.warning(
                    f"    Invalid verdict for row={row_index} direction={direction}: "
                    f"'{verdict}' (len={len(reason)})"
                )

    if fixed_count == 0:
        logger.warning(f"  [REPAIR] No valid replacements produced for {output_path}")
        return len(repair_targets), 0, len(repair_targets)

    remaining_count = sum(
        1
        for csv_row in csv_rows
        for direction in ("vs_1", "vs_2")
        if is_repairable_verdict(csv_row.get(direction))
    )
    if not write_repaired_csv(output_path, fieldnames, csv_rows, prepared_rows):
        logger.warning(f"  [REPAIR] Temporary CSV validation failed for {output_path}")
        return len(repair_targets), 0, len(repair_targets)

    logger.info(
        f"  [REPAIR] Complete: fixed={fixed_count}, remaining_errors={remaining_count}, "
        f"output={output_path}"
    )
    return len(repair_targets), fixed_count, remaining_count


# ==================== Step3 Baseline Empty Fix ====================
def step3_fix_baseline_empty_rows():
    """
    Fix Step3 comparison CSVs where rows were evaluated with empty baseline_response
    (because baseline translation data was missing). Re-runs comparison using the
    now-populated baseline responses.
    """
    logger.info("=" * 80)
    logger.info("  Step3 Baseline Empty Row Fix")
    logger.info("=" * 80)

    dataset_files = scan_dataset_files(DATASET_DIR)
    baseline_files = scan_baseline_files(BASELINE_DIR)

    step_dataset_files = dataset_files.get("Step3", [])
    step_baseline_files = baseline_files.get("Step3", [])

    if not step_dataset_files or not step_baseline_files:
        logger.error("No Step3 dataset or baseline files found. Exiting.")
        return

    total_fixed = 0
    total_checked = 0

    for judge_model in JUDGE_MODELS:
        judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
        step_output_dir = OUTPUT_DIR / "Step3" / judge_short

        for ds_info in step_dataset_files:
            for bl_info in step_baseline_files:
                output_path = output_path_for_task(
                    ds_info["model_id"], bl_info["model_name"], step_output_dir
                )

                if not output_path.exists():
                    continue

                total_checked += 1

                fieldnames, csv_rows = read_output_csv(output_path)

                dataset_df = load_datalist_csv(ds_info["csv_path"])
                baseline_df = load_datalist_csv(bl_info["csv_path"])
                prepared_rows = match_and_prepare_rows(dataset_df, baseline_df, "Step3")

                if len(csv_rows) != len(prepared_rows):
                    logger.warning(
                        f"  Row count mismatch: {output_path.name} "
                        f"csv={len(csv_rows)} prepared={len(prepared_rows)}, skip"
                    )
                    continue

                rows_to_fix = []
                for i, (csv_row, prep_row) in enumerate(zip(csv_rows, prepared_rows)):
                    csv_base = (csv_row.get("baseline_response") or "").strip()
                    cur_base = (prep_row.get("baseline_response") or "").strip()
                    csv_ds = (csv_row.get("dataset_response") or "").strip()
                    cur_ds = (prep_row.get("dataset_response") or "").strip()

                    if csv_base != cur_base:
                        rows_to_fix.append(i)
                    elif csv_ds != cur_ds:
                        rows_to_fix.append(i)

                if not rows_to_fix:
                    continue

                logger.info(
                    f"  [STEP3_FIX] {ds_info['model_id']} vs {bl_info['model_name']} "
                    f"judge={judge_short}: {len(rows_to_fix)} rows flagged"
                )

                fixed = 0
                with ThreadPoolExecutor(max_workers=MAX_PARALLEL_PER_MODEL) as executor:
                    future_to_idx = {}
                    for idx in rows_to_fix:
                        row = prepared_rows[idx]
                        future = executor.submit(
                            compare_single_row, judge_model,
                            row["prompt"], row["dataset_response"], row["baseline_response"],
                        )
                        future_to_idx[future] = idx

                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                        except Exception as e:
                            logger.warning(f"    Fix failed for row {idx}: {e}")
                            continue

                        row = prepared_rows[idx]
                        csv_rows[idx]["baseline_response"] = row["baseline_response"]
                        csv_rows[idx]["dataset_response"] = row["dataset_response"]
                        csv_rows[idx]["vs_1"] = result.get("vs_1", "error")
                        csv_rows[idx]["vs_2"] = result.get("vs_2", "error")
                        csv_rows[idx]["vs_1_reason"] = result.get("vs_1_reason", "")
                        csv_rows[idx]["vs_2_reason"] = result.get("vs_2_reason", "")
                        fixed += 1

                tmp_path = output_path.with_name(output_path.name + ".fix.tmp")
                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(csv_rows)
                os.replace(tmp_path, output_path)

                logger.info(f"    Fixed {fixed}/{len(rows_to_fix)} rows")
                total_fixed += fixed

    logger.info(f"\nStep3 fix complete: checked {total_checked} files, fixed {total_fixed} rows")


# ==================== Main Orchestrator ====================
def run_all_comparisons(repair_completed: bool = False, step3_fix: bool = False):
    if step3_fix:
        step3_fix_baseline_empty_rows()
        return



    completed_tasks = load_recovery_state()
    if completed_tasks:
        logger.info(f"Loaded {len(completed_tasks)} completed tasks from recovery state")

    logger.info("Scanning Dataset files...")
    dataset_files = scan_dataset_files(DATASET_DIR)
    for step in STEPS:
        logger.info(f"  {step}: {len(dataset_files[step])} files")

    logger.info("Scanning Baseline files...")
    baseline_files = scan_baseline_files(BASELINE_DIR)
    for step in STEPS:
        logger.info(f"  {step}: {len(baseline_files[step])} files")

    if not any(dataset_files[s] for s in STEPS):
        logger.error("No Dataset files found. Exiting.")
        return
    if not any(baseline_files[s] for s in STEPS):
        logger.error("No Baseline files found. Exiting.")
        return

    total_files = 0
    total_success = 0
    total_skipped = 0
    total_repair_targets = 0
    total_repaired = 0
    total_remaining_errors = 0

    for judge_model in JUDGE_MODELS:
        judge_short = MODEL_SHORT_NAME.get(judge_model, judge_model.replace(":", "_"))
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing judge model: {judge_short}")
        logger.info(f"{'='*60}")

        for step in STEPS:
            step_dataset_files = dataset_files.get(step, [])
            step_baseline_files = baseline_files.get(step, [])

            if not step_dataset_files or not step_baseline_files:
                logger.info(f"  [SKIP] {step}: no dataset or baseline files")
                continue

            logger.info(f"\n  {step}: {len(step_dataset_files)} dataset × {len(step_baseline_files)} baseline")

            step_output_dir = OUTPUT_DIR / step / judge_short
            step_output_dir.mkdir(parents=True, exist_ok=True)

            for ds_info in step_dataset_files:
                for bl_info in step_baseline_files:
                    task_key = f"{ds_info['model_id']}|{bl_info['model_name']}|{judge_model}|{step}"
                    output_path = output_path_for_task(ds_info["model_id"], bl_info["model_name"], step_output_dir)

                    if output_path.exists() and (task_key in completed_tasks or repair_completed):
                        if repair_completed:
                            try:
                                dataset_df = load_datalist_csv(ds_info["csv_path"])
                                baseline_df = load_datalist_csv(bl_info["csv_path"])
                                rows = match_and_prepare_rows(dataset_df, baseline_df, step)
                                checked, fixed, remaining = repair_completed_csv_errors(
                                    output_path,
                                    rows,
                                    step,
                                    ds_info["model_id"],
                                    ds_info["csv_path"],
                                    bl_info["model_name"],
                                    bl_info["csv_path"],
                                    judge_model,
                                )
                                if checked > 0:
                                    total_repair_targets += checked
                                    total_repaired += fixed
                                    total_remaining_errors += remaining
                                elif checked == 0:
                                    logger.info(f"  [SKIP] {ds_info['model_id']} vs {bl_info['model_name']}")
                                if checked >= 0:
                                    completed_tasks.add(task_key)
                                    save_recovery_state(completed_tasks)
                            except Exception as e:
                                logger.error(
                                    f"Repair failed: {ds_info['model_id']} vs {bl_info['model_name']} "
                                    f"({judge_short}): {e}"
                                )
                        else:
                            logger.info(f"  [SKIP] {ds_info['model_id']} vs {bl_info['model_name']}")
                        total_files += 1
                        total_skipped += 1
                        continue

                    try:
                        dataset_df = load_datalist_csv(ds_info["csv_path"])
                        baseline_df = load_datalist_csv(bl_info["csv_path"])

                        progress_db = db_path_for_task(
                            ds_info["model_id"], bl_info["model_name"],
                            judge_model, step_output_dir
                        )

                        output_path, sc, sk = run_single_comparison_task(
                            dataset_df, baseline_df,
                            step,
                            ds_info["model_id"],
                            bl_info["model_name"],
                            judge_model,
                            step_output_dir,
                            progress_db,
                            threading.Lock(),
                            {"vs_1": set(), "vs_2": set()},
                        )

                        if sc > 0 or sk > 0:
                            total_success += sc
                            total_files += 1
                            completed_tasks.add(task_key)
                            save_recovery_state(completed_tasks)

                    except Exception as e:
                        logger.error(
                            f"Failed: {ds_info['model_id']} vs {bl_info['model_name']} "
                            f"({judge_short}): {e}"
                        )
                        import traceback
                        traceback.print_exc()

    logger.info(f"\n{'='*80}")
    logger.info(f"  COMPARISON COMPLETE")
    logger.info(f"  Total comparison files generated: {total_files}")
    logger.info(f"  Total successful row comparisons: {total_success}")
    logger.info(f"  Total skipped (recovered): {total_skipped}")
    if repair_completed:
        logger.info(f"  Total repair targets found: {total_repair_targets}")
        logger.info(f"  Total repair targets fixed: {total_repaired}")
        logger.info(f"  Total repair errors remaining: {total_remaining_errors}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*80}")


# ==================== Entry Point ====================
def main():
    global MAX_CONCURRENT_MODELS, MAX_PARALLEL_PER_MODEL, TIMEOUT_SECONDS
    parser = argparse.ArgumentParser(
        description="Dataset vs Baseline Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python dataset_vs_baseline_comparison.py                    # Full mode
  python dataset_vs_baseline_comparison.py --single_gpu       # Single GPU
  python dataset_vs_baseline_comparison.py --multi_gpu        # Multi GPU
  python dataset_vs_baseline_comparison.py --repair_completed # Repair error cells in completed CSVs
  python dataset_vs_baseline_comparison.py --repair_completed --parallel 1
  python dataset_vs_baseline_comparison.py --step3_fix        # Fix Step3 rows with empty baseline
        """
    )
    parser.add_argument('--single_gpu', action='store_true',
                        help='Single GPU mode: sequential comparisons')
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Multi GPU mode: 2 concurrent comparisons')
    parser.add_argument('--repair_completed', action='store_true',
                        help='Check existing completed CSVs and rerun only error/blank verdict cells')
    parser.add_argument('--step3_fix', action='store_true',
                        help='Re-run Step3 comparisons where baseline was empty (now populated)')
    parser.add_argument('--timeout', type=int, default=None,
                        help=f'Override Ollama timeout in seconds (default: {TIMEOUT_SECONDS})')
    parser.add_argument('--parallel', type=int, default=None,
                        help=f'Parallel requests per comparison (default: MAX_PARALLEL={MAX_PARALLEL})')
    args = parser.parse_args()

    if args.parallel is not None and args.parallel < 1:
        parser.error("--parallel must be >= 1")

    if args.timeout is not None and args.timeout < 1:
        parser.error("--timeout must be >= 1")

    per_model_parallel = args.parallel if args.parallel is not None else MAX_PARALLEL

    if args.timeout is not None:
        TIMEOUT_SECONDS = args.timeout
        logger.info(f"Timeout overridden to {TIMEOUT_SECONDS}s")

    if args.multi_gpu and not args.single_gpu:
        MAX_CONCURRENT_MODELS = 2
        MAX_PARALLEL_PER_MODEL = per_model_parallel
        scheduling_mode = (
            f"MULTI_GPU (2 concurrent comparisons, {MAX_PARALLEL_PER_MODEL} requests each)"
        )
    else:
        MAX_CONCURRENT_MODELS = 1
        MAX_PARALLEL_PER_MODEL = per_model_parallel
        scheduling_mode = (
            f"SINGLE_GPU (1 sequential comparison, {MAX_PARALLEL_PER_MODEL} requests each)"
        )

    print("=" * 70)
    print(f"  Dataset vs Baseline Comparison Tool")
    print(f"  Scheduling mode: {scheduling_mode}")
    print("=" * 70)

    logger.info(f"Starting Dataset vs Baseline Comparison...")
    logger.info(f"Scheduling: {scheduling_mode}")
    if args.repair_completed:
        logger.info("Repair mode: completed CSVs will be checked for error/blank verdict cells")
    if args.step3_fix:
        logger.info("Step3 fix mode: re-running selections for rows where baseline was empty")

    run_all_comparisons(repair_completed=args.repair_completed, step3_fix=args.step3_fix)

    logger.info("All comparisons completed!")


if __name__ == "__main__":
    main()
