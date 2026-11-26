"""
Dataset vs Baseline Comparison Configuration

Configuration for comparing Dataset files against Baseline files
using judge models via Ollama API.
"""

import os
from pathlib import Path

# ==================== Ollama API Configuration ====================
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.10.54:11434/api/generate")
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "3"))
TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "300"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "2048"))

# ==================== Judge Models Configuration ====================
JUDGE_MODELS = [
    "ministral-3:8b-instruct-2512-q8_0",
    "olmo-3:7b-instruct-q8_0",
    "llama3.1:8b-instruct-q8_0",
    "gemma4:e4b-it-q8_0",
    "qwen3.5:9b-q8_0",
]

MODEL_SHORT_NAME = {
    "ministral-3:8b-instruct-2512-q8_0": "ministral3_8b",
    "olmo-3:7b-instruct-q8_0": "olmo3_7b",
    "llama3.1:8b-instruct-q8_0": "llama3.1_8b",
    "gemma4:e4b-it-q8_0": "gemma4_e4b",
    "qwen3.5:9b-q8_0": "qwen3.5_9b",
}

# ==================== Path Configuration ====================
SCRIPT_DIR = Path(__file__).parent.resolve()
BASE_DIR = SCRIPT_DIR.parent

DATASET_DIR = BASE_DIR / "Dataset"
BASELINE_DIR = BASE_DIR / "Baseline"
OUTPUT_DIR = SCRIPT_DIR  # vs_baseline directory

# ==================== Steps ====================
STEPS = ["Step1", "Step2", "Step3"]

# ==================== Baseline Model Name Normalization ====================
# Maps extracted filenames to canonical names, so Step1/2/3 baselines are consistent
BASELINE_CANONICAL_NAMES = {
    "qwen3_5_4b": "qwen3_5_4b",
    "qwen3_5_9b": "qwen3_5_9b",
    "qwen3_5_35b": "qwen3_5_35b",
    "qwen3_5_35b_a3b": "qwen3_5_35b",
    "glm4_7": "glm4_7_flash",
    "glm_4_7_flash": "glm4_7_flash",
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
