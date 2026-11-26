"""
Test vs Baseline Comparison Configuration

Configuration for comparing Test_dataset.csv with Baseline files
using judge models via Ollama API.
"""

import os
from pathlib import Path

# ==================== Ollama API Configuration ====================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.10.54:11434/api/generate")
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "4"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "120"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# ==================== Judge Models Configuration ====================
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

# ==================== Path Configuration ====================
# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

# Input paths
TEST_DATASET_PATH = SCRIPT_DIR / "Test_dataset.csv"
BASELINE_DIR = BASE_DIR / "Baseline"

# Output path (same directory as test dataset)
OUTPUT_DIR = SCRIPT_DIR

# ==================== Baseline Files Configuration ====================
# Map each step to its baseline CSV files
BASELINE_FILES = {
    "Step1": [
        "Step1/test_qwen3_5_4b-q8_0_False_20260410_180517_annotated.csv",
        "Step1/test_qwen3_5_9b-q8_0_False_20260410_113731_annotated.csv",
        "Step1/test_qwen3_5_35b-a3b-q8_0_False_20260409_235358_annotated.csv",
        "Step1/test_glm-4_7-flash_q8_0_False_20260412_011535_annotated.csv",
    ],
    "Step2": [
        "Step2/test_qwen3_5_4b-q8_0_False_20260409_214430_annotated.csv",
        "Step2/test_qwen3_5_9b-q8_0_False_20260409_214857_annotated.csv",
        "Step2/test_qwen3_5_35b-a3b-q8_0_False_20260409_231145_annotated.csv",
        "Step2/test_glm-4_7-flash_q8_0_False_20260413_211708_annotated.csv",
    ],
    "Step3": [
        "Step3/20260414_143101_dify_qwen3.5-9b_report_annotated.csv",
        "Step3/20260414_123741_dify_qwen3.5-35b_report_annotated.csv",
        "Step3/20260414_150623_dify_qwen3.5-4b_report_annotated.csv",
        "Step3/20260414_151930_dify_glm4.7_report_annotated.csv",
    ],
}

# Model name aliases: maps variant names (e.g. from dify filenames) to canonical names
# Canonical names match what extract_model_name() returns for Step1/Step2 files.
# This ensures Step3 baselines are grouped together with their Step1/Step2 counterparts.
MODEL_NAME_ALIASES = {
    "glm4_7": "glm-4_7-flash_q8_0",
    "qwen3_5_35b": "qwen3_5_35b-a3b",
}

# ==================== Prompt Templates ====================
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

# ==================== Logging Configuration ====================
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
