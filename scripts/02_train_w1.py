#!/usr/bin/env python3
"""Run the leakage-safe nested-CV W1 experiment and train the final artifact."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.w1 import run_nested_w1  # noqa: E402


if __name__ == "__main__":
    result = run_nested_w1(ROOT)
    print(f"selected_model_by_mean={result['selected_model_by_mean']}")
    print(f"final_alpha={result['final_alpha']}")
    for row in result["nested_cv"]:
        print(row["model"], f"macro_f1={row['macro_f1_mean']:.6f}±{row['macro_f1_std']:.6f}")

