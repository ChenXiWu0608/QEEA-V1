"""Compatibility entry point for the canonical bidirectional 100-item calibration."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "vs/calculate_judge_weights.py"), run_name="__main__")
