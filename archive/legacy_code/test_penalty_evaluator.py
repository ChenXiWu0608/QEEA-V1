"""
Penalty Evaluator Script
Calls Ollama API to evaluate Test_dataset.csv using penalty test prompts from Penalty.json.
Supports recovery from interruption, progress bar, parallel processing, and auto-restart on timeout.
Results are written immediately with smart ordering - later IDs may wait for earlier IDs to finish writing.

Usage:
    python penalty_evaluator.py
    
Output:
    Test_result_{model_name}.csv files in Test calibration/ folder
    
Recovery:
    If interrupted, rerun the script to resume from where it left off.
"""

import csv
import json
import os
import re
import sys
import time
import requests
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

# ==================== Configuration ====================
OLLAMA_URL = "http://192.168.10.54:11434/api/generate"

# Paths relative to script's parent directory
# Script is located in "Test calibration" folder, Penalty folder is in the parent
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# Paths relative to base directory
CSV_PATH = os.path.join(BASE_DIR, "Test calibration", "Test_dataset.csv")
PENALTY_JSON_PATH = os.path.join(BASE_DIR, "Penalty", "Penalty.json")
OUTPUT_DIR = BASE_DIR
RECOVERY_FILE = os.path.join(OUTPUT_DIR, ".penalty_recovery.json")
DEBUG_LOG_FILE = os.path.join(OUTPUT_DIR, "parse_debug_log.txt")

MAX_PARALLEL = 2
TIMEOUT_SECONDS = 120
MAX_RETRIES = 3

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
    "llama3.1:8b-instruct-q8_0": "llama3.1_8b",
    "gemma4:e4b-it-q8_0": "gemma4_e4b",
    "qwen3.5:9b-q8_0": "qwen3.5_9b",
}


# ==================== Category Mapping ====================
# Maps (source_step, csv_category, csv_subcategory) -> (json_step, json_category_key)
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
    # 1. 旧格式：(category, subcategory) -> (json_step, json_category_key)
    # 2. 新格式：subcategory 直接对应 Penalty.json 中的键名 (如 workflow_cs_simple, workflow_cs_conflict, workflow_cs_prediction)
    ("Step3", "customer_support", "information_query"): ("Step3", "workflow_cs_simple"),
    ("Step3", "customer_support", "workflow_cs_simple"): ("Step3", "workflow_cs_simple"),
    ("Step3", "customer_support", "workflow_cs_conflict"): ("Step3", "workflow_cs_conflict"),
    ("Step3", "customer_support", "workflow_cs_prediction"): ("Step3", "workflow_cs_prediction"),
    ("Step3", "research", "investigation"): ("Step3", "workflow_research"),
    ("Step3", "translation", "multilingual"): ("Step3", "workflow_translation"),
}


# ==================== Word Count Functions ====================
def extract_word_limit(prompt: str) -> int | None:
    """
    Extract the expected word count from the first sentence of the prompt.
    
    Supports:
    - English: "Write a 2000-word ..."
    - Japanese: "...2000 語で..."
    - Chinese: "写一篇 2000 字..." or "2000 字关于..."
    
    Returns the expected word count as an integer, or None if not found.
    """
    if not prompt:
        return None
    
    # Get the first sentence (up to first period, question mark, or newline)
    first_sentence = re.split(r'[.?!]|\n|\r', prompt.strip())[0]
    
    # English: "Write a {num}-word" pattern
    english_match = re.search(r'Write\s+a\s+(\d+)\s*-?\s*word', first_sentence, re.IGNORECASE)
    if english_match:
        return int(english_match.group(1))
    
    # Japanese: "{num} 語で" pattern
    japanese_match = re.search(r'(\d+)\s*語', first_sentence)
    if japanese_match:
        return int(japanese_match.group(1))
    
    # Chinese: "{num} 字" or "{num} 万字" pattern
    chinese_match = re.search(r'(\d+)\s*字', first_sentence)
    if chinese_match:
        count = int(chinese_match.group(1))
        # Handle "万字" (10000)
        if '万' in first_sentence[:chinese_match.start() + 10]:
            count *= 10000
        return count
    
    return None


def detect_language_type(language: str) -> str:
    """
    Detect the language type for word counting.
    
    Returns:
    - "english": Count by words
    - "chinese": Count by characters (for Chinese)
    - "japanese": Count by characters (for Japanese)
    - "other": Default to character counting
    """
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
        # For other languages, default to character counting for Asian languages
        # Try to detect if it contains CJK characters
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', language):
            return "chinese"  # Treat as CJK character counting
        return "other"


def count_words(text: str, lang_type: str) -> int:
    """
    Count words/characters based on language type.
    
    For English: count by words (space-separated)
    For Chinese/Japanese: count by characters
    """
    if not text:
        return 0
    
    if lang_type == "english":
        # Count English words by splitting on whitespace
        # Remove non-word characters except spaces
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        words = [w for w in clean_text.split() if w]
        return len(words)
    else:
        # For Chinese/Japanese: count all characters
        # Filter out whitespace but count all other characters
        non_whitespace = [c for c in text if not c.isspace()]
        return len(non_whitespace)


def calculate_word_penalty(actual_count: int, expected_count: int) -> str | None:
    """
    Calculate word count penalty based on the ratio of actual to expected count.
    
    Rules:
    - actual/expected < 0.5 or > 2.0: "hard"
    - 0.5-0.75 or 1.25-2.0: "soft"
    - 0.75-1.25: None (let LLM decide)
    
    Returns:
    - "hard" if significantly outside range
    - "soft" if moderately outside range
    - None if within acceptable range (LLM should decide)
    """
    if expected_count <= 0:
        return None
    
    ratio = actual_count / expected_count
    
    if ratio < 0.5 or ratio > 2.0:
        return "hard"
    elif ratio < 0.75 or ratio > 1.25:
        return "soft"
    else:
        return None


def modify_step2_system_prompt(original_prompt: str, word_penalty: str | None) -> str:
    """
    Modify the Step2 system prompt to include word count penalty rules.
    
    If word_penalty is "hard" or "soft", add a note to use that verdict.
    """
    if word_penalty is None:
        return original_prompt
    
    # Add a note at the beginning about word count penalty
    word_note = f"\n\n## Word Count Penalty Note\nBased on pre-validation, the response has a word count penalty of '{word_penalty}'. Please incorporate this into your final verdict."
    
    # Insert after the first line (model role definition)
    lines = original_prompt.split('\n')
    if lines:
        lines.insert(1, word_note)
    
    return '\n'.join(lines)


# ==================== Helper Functions ====================
def load_penalty_prompts():
    """Load penalty prompts from JSON file."""
    with open(PENALTY_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_data(limit=100):
    """Load CSV data, limited to first `limit` rows."""
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)
    return rows


def get_prompt_mapping(source_step, category, subcategory):
    """Get the prompt mapping from CSV fields to Penalty.json keys."""
    cat = category.strip() if category else ""
    sub = subcategory.strip() if subcategory else ""
    
    # Special handling for GPQA - subcategory contains correct answer, not a category
    # When category is "GPQA", always return GPQA prompt regardless of subcategory value
    if source_step == "Step1" and cat.lower() == "gpqa":
        return ("Step1", "GPQA")
    
    # Try exact match
    key = (source_step, cat, sub if sub else None)
    if key in CATEGORY_MAPPING:
        return CATEGORY_MAPPING[key]
    
    # Try case-insensitive match
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
    
    # Replace {prompt} with csv prompt
    prompt = prompt.replace("{prompt}", row.get("prompt", ""))
    
    # Replace {language} with csv language (default to English if empty)
    language = row.get("language", "") or "English"
    prompt = prompt.replace("{language}", language)
    
    # Replace {response} with csv response
    prompt = prompt.replace("{response}", row.get("response", ""))
    
    # Special: For GPQA, replace {correct_answer} with subcategory content
    if category and category.strip().lower() == "gpqa":
        prompt = prompt.replace("{correct_answer}", subcategory or "")
    
    return prompt


def call_ollama_api(model, prompt, system):
    """
    Call Ollama API and return response data.
    Raises requests.exceptions.Timeout on timeout.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "think": False,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "num_ctx": 32468,
        }
    }
    
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=TIMEOUT_SECONDS
    )
    response.raise_for_status()
    
    return response.json()


def clean_json_string(s):
    """
    Clean JSON string by removing problematic control characters.
    Removes:
    - ASCII control characters (except newlines, tabs, carriage returns)
    - Unicode control characters (soft hyphen, zero-width characters, etc.)
    """
    # Remove ASCII control characters (keep \n, \t, \r as literal characters, not escaped)
    # Remove U+0000 to U+001F except we'll handle them specially
    result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    
    # Remove problematic Unicode control characters
    # U+00ad: soft hyphen
    # U+200b: zero-width space
    # U+200c: zero-width non-joiner
    # U+200d: zero-width joiner
    # U+200e: left-to-right mark
    # U+200f: right-to-left mark
    # U+2028: line separator
    # U+2029: paragraph separator
    # U+202a-U+202e: various bidi controls
    # U+2060: word joiner
    # U+206f: n-ary operator spacing
    result = re.sub(r'[\u00ad\u200b\u200c\u200d\u200e\u200f\u2028\u2029\u202a\u202b\u202c\u202d\u202e\u2060\u206f]', '', result)
    
    # Also remove some other potentially problematic characters
    # U+061c: Arabic sign half matrah (sometimes appears in mixed content)
    result = re.sub(r'[\u061c]', '', result)
    
    return result


def parse_llm_json_response(text, debug_mode=True, model_name="", row_index=0):
    """
    Parse JSON verdict and reason from LLM response text.
    Enhanced with multiple strategies for JSON extraction and debug logging.
    Returns (verdict, reason, raw_json_or_error) tuple.
    """
    def log_debug(msg):
        """Print debug message and optionally write to log file."""
        print(msg)
        if debug_mode:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] Model={model_name} Row={row_index + 1}\n")
                    f.write(f"  {msg}\n")
            except:
                pass
    
    def log_failure(raw_text):
        """Log failed parsing attempt with raw response."""
        if debug_mode:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] [PARSE FAILURE] Model={model_name} Row={row_index + 1}\n")
                    f.write(f"  RAW_RESPONSE_START\n")
                    f.write(f"  {repr(raw_text[:2000])}\n")
                    f.write(f"  RAW_RESPONSE_END\n")
                    f.write(f"  {'='*60}\n")
            except:
                pass
    
    log_debug(f"[RAW RESPONSE] Length: {len(text)} chars")
    
    # Show preview of raw response
    if len(text) > 300:
        log_debug(f"[PREVIEW] {repr(text[:300])}...")
    else:
        log_debug(f"[PREVIEW] {repr(text)}")
    
    try:
        # Strategy 1: Extract JSON from markdown code blocks
        # Match ```json ... ``` or ``` ... ``` patterns
        # Use non-greedy matching to find the innermost JSON
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        code_block_matches = re.findall(code_block_pattern, text)
        
        if code_block_matches:
            log_debug(f"[Strategy 1] Found {len(code_block_matches)} code block(s)")
            for i, match in enumerate(code_block_matches):
                log_debug(f"[Code Block {i}] Content: {repr(match[:150])}...")
                try:
                    cleaned = match.strip()
                    # Clean problematic control characters
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
            log_debug(f"[Strategy 2] Found JSON at pos {start}:{end}")
            log_debug(f"[JSON Preview] {repr(json_str[:150])}...")
          
            # Clean up control characters
            json_str = clean_json_string(json_str)
          
            try:
                data = json.loads(json_str)
                log_debug(f"[Strategy 2] SUCCESS")
                return data.get("verdict", "unknown"), data.get("reason", ""), json_str
            except (json.JSONDecodeError, ValueError) as e:
                log_debug(f"[Strategy 2] Parse failed: {e}")
              
                # Strategy 3: JSON Repair
                log_debug(f"[Strategy 3] Attempting JSON repair. ..")
      
                # 3a. Fix trailing commas before } or ]
                fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                if fixed != json_str:
                    log_debug(f"[Strategy 3a] Fixed trailing commas")
      
                # 3b. Fix unescaped newlines and tabs within string values
                def fix_string_escapes(s):
                    result = []
                    in_string = False
                    escape_next = False
                    for ch in s:
                        if escape_next:
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
                    return ''.join(result)
      
                fixed = fix_string_escapes(fixed)
      
                # 3c. Try to parse with fixed version
                try:
                    data = json.loads(fixed)
                    log_debug(f"[Strategy 3] JSON REPAIR SUCCESS")
                    return data.get("verdict", "unknown"), data.get("reason", ""), fixed
                except (json.JSONDecodeError, ValueError) as e2:
                    log_debug(f"[Strategy 3] Repair failed: {e2}")
        
                    # 3d. Try to extract just the verdict using regex as fallback
                    log_debug(f"[Strategy 3d] Attempting regex extraction. ..")
                    verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', json_str)
                    if verdict_match:
                        verdict = verdict_match.group(1)
                        reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
                        reason = reason_match.group(1) if reason_match else ""
                        log_debug(f"[Strategy 3d] Regex extracted: verdict={verdict}")
                        return verdict, reason, "regex_fallback"
        
        # Strategy 4: Pure regex extraction (no JSON parsing)
        log_debug(f"[Strategy 4] Trying pure regex extraction. ..")
        
        # Look for verdict field - also handle Unicode quotes
        verdict_match = re.search(r'"verdict"\s*:\s*"[^"]*"', text)
        if verdict_match:
            verdict = verdict_match.group(0)
            verdict = re.sub(r'"verdict"\s*:\s*"', '', verdict)
            verdict = re.sub('"$', '', verdict)
            
            # Look for reason field (handle escaped characters)
            reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            reason = reason_match.group(1) if reason_match else ""
            # Clean up reason
            reason = reason.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r').replace('\\', '\\')
            log_debug(f"[Strategy 4] SUCCESS: verdict={verdict}")
            return verdict, reason, "regex_only"
        
        # All strategies failed
        log_debug(f"[FAILED] All parsing strategies failed")
        log_failure(text)
    
    except Exception as e:
        log_debug(f"[EXCEPTION] {type(e).__name__}: {e}")
        log_failure(text)
    
    return "unknown", "Failed to parse JSON response from model", "parse_failed"


def restart_ollama():
    """Restart Ollama service using systemctl."""
    print("    [RESTART] Attempting to restart Ollama service. ..")
    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", "ollama"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        print("    [RESTART] Ollama service restart command sent successfully.")
        # Wait for Ollama to be ready
        time.sleep(10)
        # Verify Ollama is running
        for _ in range(5):
            try:
                r = requests.get("http://192.168.10.54:11434/api/tags", timeout=5)
                if r.status_code == 200:
                    print("    [RESTART] Ollama is back online.")
                    return True
            except:
                pass
            time.sleep(2)
        print("    [WARNING] Restart command sent, but could not verify Ollama status.")
        return True
    except subprocess.TimeoutExpired:
        print("    [ERROR] Restart command timed out.")
        return False
    except Exception as e:
        print(f"    [ERROR] Failed to restart Ollama: {e}")
        return False


# ==================== Recovery State Management ====================
def load_recovery_state():
    """
    Load recovery state from file.
    Returns a dict: {row_index: set_of_completed_model_names}
    """
    if os.path.exists(RECOVERY_FILE):
        try:
            with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            # Convert lists back to sets
            return {int(k): set(v) for k, v in state.get("completed_rows", {}).items()}
        except Exception as e:
            print(f"[WARNING] Failed to load recovery state: {e}")
    return {}


def save_recovery_state(state):
    """Save recovery state to file."""
    serializable_state = {
        "completed_rows": {str(k): list(v) for k, v in state.items()},
        "last_updated": datetime.now().isoformat()
    }
    with open(RECOVERY_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable_state, f, indent=2)


def clear_recovery_state():
    """Clear recovery state file."""
    if os.path.exists(RECOVERY_FILE):
        os.remove(RECOVERY_FILE)


# ==================== Output Writing ====================
# 每个模型的输出列定义（不含模型名称）
MODEL_OUTPUT_COLUMNS = ["verdict", "reason", "total_duration", "prompt_eval_count"]

# 基础列（所有 CSV 文件共用）- 按要求的顺序排列
BASE_COLUMNS = ["penalty_id", "source_step", "category", "subcategory", "language", "prompt", "response"]

def get_model_output_header(model_short):
    """获取单个模型的输出 CSV 表头。
    顺序：penalty_id,source_step,category,subcategory,language,verdict,reason,total_duration,prompt_eval_count,prompt,response
    """
    return BASE_COLUMNS[:5] + MODEL_OUTPUT_COLUMNS + BASE_COLUMNS[5:]


def write_csv_with_header(output_path, header):
    """写入 CSV 文件头."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()


def append_results_to_csv_sorted(results, output_path, model_short):
    """
    将结果按 penalty_id 排序后写入 CSV 文件。
    先写入表头，然后按 penalty_id 升序写入所有结果。
    """
    if not results:
        return
    
    header = get_model_output_header(model_short)
    # 按 penalty_id 排序
    sorted_results = sorted(results, key=lambda x: x.get("penalty_id", 0))
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)


def create_output_csvs():
    """
    Create all output CSV files with headers.
    每个 CSV 文件只包含对应模型的列，表头不包含模型名称。
    """
    for model_short in [MODEL_SHORT_NAME.get(m, m.replace(":", "_")) for m in JUDGE_MODELS]:
        output_path = os.path.join(OUTPUT_DIR, f"Test_result_{model_short}.csv")
        header = get_model_output_header(model_short)
        write_csv_with_header(output_path, header)
        print(f"  Created: {output_path}")


# ==================== Smart Write Queue Class ====================
class SmartWriteQueue:
    """
    智能写入队列：每个模型独立的写入队列。
    
    核心逻辑：
    1. 每个模型有独立的写入队列
    2. 维护 results 列表缓存结果
    3. 当获取到结果时立即缓存（不直接写入文件）
    4. 每个模型的所有任务完成后，按 penalty_id 排序写入 CSV
    """
    
    def __init__(self, output_path, header):
        self.output_path = output_path
        self.header = header
        self.results = []  # 缓存的结果列表
    
    def __del__(self):
        """析构函数：确保文件关闭"""
        self.flush()
    
    def _ensure_dir_exists(self):
        """确保输出目录存在"""
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _ensure_file_open(self):
        """确保文件目录存在"""
        self._ensure_dir_exists()
    
    def write_result(self, result):
        """
        智能写入结果：
        - 缓存结果到内存
        - 当模型所有任务完成后，按 penalty_id 排序写入 CSV
        """
        self.results.append(result)
        print(f"  [CACHE] Result {result.get('penalty_id')} cached")
    
    def flush(self):
        """将所有缓存的结果按 penalty_id 排序后写入 CSV 文件"""
        # 如果没有缓存的数据，直接返回，避免覆盖已有数据
        if not self.results:
            return
        
        self._ensure_file_open()
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.header, extrasaction='ignore')
            writer.writeheader()
            for result in sorted(self.results, key=lambda x: x.get("penalty_id", 0)):
                writer.writerow(result)
        print(f"  [FLUSH] Wrote {len(self.results)} rows to {self.output_path}")
        self.results.clear()
    
    def get_stats(self):
        """获取写入统计信息"""
        return {
            "cached_count": len(self.results),
            "output_path": self.output_path
        }


# ==================== Core Processing Logic ====================
def process_single_request(row_index, row, model, penalty_prompts):
    """
    Process a single (row, model) pair.
    Returns: (row_index, model, result_dict or None, error_message or None)
    
    For Step2 tasks:
    - Perform word count check before calling LLM
    - If word count is outside 75%-125% range, return hard/soft directly
    - If word count is within 75%-125%, let LLM decide
    """
    source_step = row.get("source_step", "")
    category = row.get("category", "")
    subcategory = row.get("subcategory", "")
    language = row.get("language", "")
    
    # Get prompt mapping
    mapping = get_prompt_mapping(source_step, category, subcategory)
    if not mapping:
        return row_index, model, None, f"No prompt mapping for ({source_step}, {category}, {subcategory})"
    
    json_step, json_cat_key = mapping
    
    # Get prompt data from Penalty.json
    prompts = penalty_prompts.get(json_step, {})
    prompt_data = prompts.get(json_cat_key, None)
    
    if not prompt_data:
        return row_index, model, None, f"No prompt found for step={json_step}, cat={json_cat_key}"
    
    system_prompt = prompt_data["system_prompt"]
    user_prompt_template = prompt_data["user_prompt_template"]
    
    # Build user prompt
    user_prompt = build_user_prompt(user_prompt_template, row, category, subcategory)
    
    # Get model short name for debug logging
    model_short = MODEL_SHORT_NAME.get(model, model.replace(":", "_"))
    
    # === Step2 Word Count Check ===
    word_penalty_verdict = None
    word_limit_expected = None
    word_count_actual = None
    word_ratio = None
    
    if source_step == "Step2":
        # Extract expected word count from prompt
        word_limit_expected = extract_word_limit(row.get("prompt", ""))
        
        if word_limit_expected is not None:
            # Detect language type
            lang_type = detect_language_type(language)
            
            # Count actual words/characters in response
            word_count_actual = count_words(row.get("response", ""), lang_type)
            
            # Calculate penalty based on word count
            word_penalty_verdict = calculate_word_penalty(word_count_actual, word_limit_expected)
            
            # Get ratio for logging
            word_ratio = word_count_actual / word_limit_expected if word_limit_expected > 0 else 0
            
            # 根据惩罚等级决定行为
            if word_penalty_verdict == "hard":
                # Hard 惩罚：直接返回结果，不调用 LLM
                print(f"  [WORD CHECK HARD] Row {row_index + 1}: expected={word_limit_expected}, "
                      f"actual={word_count_actual}, ratio={word_ratio:.2%}, penalty=hard -> return directly")
            
                # 构建硬惩罚结果，不调用 API
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
                    # Word count fields (only for Step2)
                    "word_limit_expected": word_limit_expected,
                    "word_count_actual": word_count_actual,
                    "word_ratio": round(word_ratio, 4),
                    "word_penalty": word_penalty_verdict,
                }
                return row_index, model, result, None
            
            elif word_penalty_verdict == "soft":
                # Soft 惩罚：修改系统提示，调用 LLM 参考
                modified_system_prompt = modify_step2_system_prompt(system_prompt, word_penalty_verdict)
                system_prompt = modified_system_prompt
                print(f"  [WORD CHECK SOFT] Row {row_index + 1}: expected={word_limit_expected}, "
                      f"actual={word_count_actual}, ratio={word_ratio:.2%}, penalty=soft -> call LLM with penalty info")
            
            else:
                # 无惩罚：正常调用 LLM
                print(f"  [WORD CHECK OK] Row {row_index + 1}: expected={word_limit_expected}, "
                      f"actual={word_count_actual}, ratio={word_ratio:.2%} -> call LLM normally")
    
    # Call Ollama API with retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api_response = call_ollama_api(model, user_prompt, system_prompt)
          
            # Parse LLM response
            llm_text = api_response.get("response", "")
            verdict, reason, raw_json = parse_llm_json_response(
                llm_text, 
                debug_mode=(attempt == 1),  # Only debug on first attempt
                model_name=model_short,
                row_index=row_index
            )
          
            # Log raw response for debugging (only when parsing fails)
            if verdict == "unknown":
                print(f"    [PARSE FAILED] Raw response preview:")
                print(f"    [RAW FULL] {repr(llm_text[:500])}")
          
            # Get API metrics
            prompt_eval_count = api_response.get("prompt_eval_count", 0)
            total_duration_ns = api_response.get("total_duration", 0)
            total_duration_ms = total_duration_ns / 1_000_000 if total_duration_ns else 0
          
            # Build result dict
            result = {
                "penalty_id": row_index + 1,  # 1-based index
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
                # Word count fields (only for Step2)
                "word_limit_expected": word_limit_expected if source_step == "Step2" else "",
                "word_count_actual": word_count_actual if source_step == "Step2" else "",
                "word_ratio": round(word_ratio, 4) if source_step == "Step2" and word_ratio is not None else "",
                "word_penalty": word_penalty_verdict if source_step == "Step2" else "",
            }
          
            return row_index, model, result, None
          
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                print(f"    [TIMEOUT] Attempt {attempt}/{MAX_RETRIES}, retrying. ..")
                time.sleep(2)
            else:
                # Final attempt failed, try restart
                print(f"    [TIMEOUT] All {MAX_RETRIES} attempts failed, restarting Ollama. ..")
                if restart_ollama():
                    time.sleep(5)
                    # One more try after restart
                    try:
                        api_response = call_ollama_api(model, user_prompt, system_prompt)
                        llm_text = api_response.get("response", "")
                        verdict, reason, raw_json = parse_llm_json_response(
                            llm_text, 
                            debug_mode=False,
                            model_name=model_short,
                            row_index=row_index
                        )
      
                        # Log raw response for debugging (only when parsing fails)
                        if verdict == "unknown":
                            print(f"    [PARSE FAILED (after restart)] Raw response preview:")
                            print(f"    [RAW FULL] {repr(llm_text[:500])}")
      
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
                            # Word count fields (only for Step2)
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
            if attempt < MAX_RETRIES:
                print(f"    [ERROR] Attempt {attempt}/{MAX_RETRIES}: {e}, retrying. ..")
                time.sleep(2)
            else:
                return row_index, model, None, f"Failed after {MAX_RETRIES} attempts: {e}"
    
    return row_index, model, None, "Unexpected: exited retry loop"


# ==================== Main Entry Point ====================
def main():
    print("=" * 70)
    print("  Penalty Evaluator - LLM Judge Benchmark")
    print("=" * 70)
    
    # Step 1: Load penalty prompts
    print("\n[1/6] Loading penalty prompts. ..")
    try:
        penalty_prompts = load_penalty_prompts()
        print(f"  OK: Loaded from {PENALTY_JSON_PATH}")
    except Exception as e:
        print(f"  [ERROR] Failed to load penalty prompts: {e}")
        return
    
    # Step 2: Load CSV data
    print("\n[2/6] Loading test dataset. ..")
    try:
        rows = load_csv_data(limit=100)
        print(f"  OK: Loaded {len(rows)} rows from {CSV_PATH}")
    except Exception as e:
        print(f"  [ERROR] Failed to load CSV: {e}")
        return
    
    # Step 3: Load recovery state
    print("\n[3/6] Checking recovery state. ..")
    completed_state = load_recovery_state()
    if completed_state:
        total_completed = sum(len(v) for v in completed_state.values())
        print(f"  Found recovery state: {len(completed_state)} rows processed")
        print(f"  Total completed tasks: {total_completed}")
    else:
        print("  No recovery state found, starting fresh")
    
    # Step 4: Verify Ollama connectivity
    print("\n[4/6] Verifying Ollama connectivity. ..")
    try:
        r = requests.get("http://192.168.10.54:11434/api/tags", timeout=10)
        if r.status_code == 200:
            models_data = r.json()
            available_models = [m["name"] for m in models_data.get("models", [])]
            print(f"  OK: Ollama is running")
            print(f"  Available models: {len(available_models)}")
        
            # Check required models
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
        print("  Make sure Ollama is running and accessible at http://192.168.10.54:11434")
        return
    
    # Step 5: Process data
    print("\n[5/6] Starting evaluation. ..")
    print(f"  Ollama URL: {OLLAMA_URL}")
    print(f"  Judge models: {len(JUDGE_MODELS)}")
    print(f"  Parallel workers: {MAX_PARALLEL}")
    print(f"  Timeout per request: {TIMEOUT_SECONDS}s")
    print(f"  Max retries: {MAX_RETRIES}")
    print("-" * 70)
    
    # Clear debug log file at start
    if os.path.exists(DEBUG_LOG_FILE):
        os.remove(DEBUG_LOG_FILE)
    print(f"  Debug log file: {DEBUG_LOG_FILE}")
    
    # 为每个模型创建智能写入队列
    model_write_queues = {}
    write_locks = {}
    for model in JUDGE_MODELS:
        model_short = MODEL_SHORT_NAME.get(model, model.replace(":", "_"))
        output_path = os.path.join(OUTPUT_DIR, f"Test_result_{model_short}.csv")
        header = get_model_output_header(model_short)
        model_write_queues[model_short] = SmartWriteQueue(output_path, header)
        write_locks[model_short] = threading.Lock()
    
    # Calculate progress
    total_tasks = len(rows) * len(JUDGE_MODELS)
    already_completed = sum(len(v) for v in completed_state.values())
    remaining_tasks = total_tasks - already_completed
    
    print(f"  Total tasks: {total_tasks}")
    print(f"  Already completed: {already_completed}")
    print(f"  Remaining: {remaining_tasks}")
    print("-" * 70)
    
    start_time = time.time()
    
    # 逐模型处理：每个模型内并行发送 MAX_PARALLEL 个请求
    # 由于显存限制，同一时间只能加载一个模型
    # Recovery state format: {row_index: set_of_completed_model_names}
    
    # 为每个模型创建进度条
    for model in JUDGE_MODELS:
        model_short = MODEL_SHORT_NAME.get(model, model.replace(":", "_"))
        print(f"\n  {'='*50}")
        print(f"  Processing with model: {model_short}")
        print(f"  {'='*50}")
        
        # 检查该模型已完成多少任务
        completed_for_model = sum(1 for row_idx, models in completed_state.items() if model_short in models)
        remaining_for_model = len(rows) - completed_for_model
        
        if completed_for_model > 0:
            print(f"  Model {model_short}: {completed_for_model} already completed, {remaining_for_model} remaining")
        
    # 为该模型创建线程池，并行处理所有行
        # 为恢复状态修改添加锁
        recovery_lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            # 提交所有待处理任务
            futures = {}
            for i, row in enumerate(rows):
                # 跳过已完成的任务
                if model_short in completed_state.get(i, set()):
                    continue
      
                def worker_task(row_idx=i, row=row, model=model, model_short=model_short):
                    """Process a single row with a single model."""
                    # Double-check if this model has already processed this row
                    if model_short in completed_state.get(row_idx, set()):
                        return None
                      
                    result, error = None, None
                    try:
                        _, _, result, error = process_single_request(row_idx, row, model, penalty_prompts)
                    except Exception as e:
                        import traceback
                        error = f"{str(e)}\n{traceback.format_exc()}"
        
                    if result:
                        # Update recovery state (thread-safe)
                        with recovery_lock:
                            if row_idx not in completed_state:
                                completed_state[row_idx] = set()
                            completed_state[row_idx].add(model_short)
                            save_recovery_state(completed_state)
                      
                        # 使用智能写入队列写入结果
                        # 每个模型有自己的 lock 和 write_queue
                        with write_locks[model_short]:
                            model_write_queues[model_short].write_result(result)
                      
                        return result
                    elif error:
                        print(f"  [MODEL {model_short}] Row {row_idx + 1}: {error}")
                    return None
      
                futures[executor.submit(worker_task)] = i
      
            # 创建进度条
            pbar = tqdm(
                total=len(futures),
                desc=f"{model_short[:15]}. ..",
                unit="row",
                bar_format="{l_bar}{bar:40}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
      
            # 更新已完成数量
            if completed_for_model > 0:
                pbar.update(completed_for_model)
      
            completed_count = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        completed_count += 1
      
                        # 打印返回内容（限制 30 条每模型）
                        if completed_count <= 30:
                            penalty_id = result.get("penalty_id", "N/A")
                            step = result.get("source_step", "N/A")
                            category = result.get("category", "N/A")
                            verdict = result.get("verdict", "N/A")
                            reason = result.get("reason", "N/A")[:100]
                            print(f"  [RESULT] Row {penalty_id} (Step:{step}, Cat:{category}) -> {model_short} verdict={verdict}, reason={reason}. ..")
                   
                    pbar.set_postfix({
                        "total": completed_count,
                        "time": f"{(time.time() - start_time) / 60:.1f}min"
                    })
                except Exception as e:
                    print(f"\n  [ERROR] Task failed: {e}")
                finally:
                    pbar.update(1)
      
            pbar.close()
        
        # Flush the write queue for this model to ensure all writes are complete
        model_write_queues[model_short].flush()
    
    elapsed_time = time.time() - start_time
    
    # Step 6: Finalize
    print("\n[6/6] Finalizing. ..")
    print(f"  Total processing time: {elapsed_time:.2f}s ({elapsed_time/60:.1f}min)")
    
    # Show debug log info
    if os.path.exists(DEBUG_LOG_FILE):
        with open(DEBUG_LOG_FILE, "r", encoding="utf-8") as f:
            log_lines = f.readlines()
        failure_count = sum(1 for line in log_lines if "[PARSE FAILURE]" in line)
        print(f"  [DEBUG LOG] Total entries: {len(log_lines)}")
        print(f"  [DEBUG LOG] Parse failures: {failure_count}")
        print(f"  [DEBUG LOG] See {DEBUG_LOG_FILE} for details")
    
    # Show row count for each output file
    model_short_names = [MODEL_SHORT_NAME.get(m, m.replace(":", "_")) for m in JUDGE_MODELS]
    total_rows = 0
    for model_short in model_short_names:
        output_path = os.path.join(OUTPUT_DIR, f"Test_result_{model_short}.csv")
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                row_count = sum(1 for _ in f) - 1  # minus header
            total_rows += row_count
            print(f"  {model_short}: {row_count} rows written to CSV")
    
    print(f"  Total rows across all files: {total_rows}")
    
    # Clear recovery state on successful completion
    clear_recovery_state()
    print("  Cleared recovery state")
    
    print("\n" + "=" * 70)
    print("  Evaluation completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()