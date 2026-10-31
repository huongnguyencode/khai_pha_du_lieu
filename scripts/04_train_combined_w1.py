#!/usr/bin/env python3
"""Run repeated nested CV and temporal validation on the harmonized cohort."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.combined_experiment import run_combined_experiment  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    args = parser.parse_args()
    result = run_combined_experiment(ROOT, args.repeats, args.outer_splits, args.inner_splits)
    print(f"best_model={result['best_model']}")
    print(result["paired_comparison"])
