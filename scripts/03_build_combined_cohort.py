#!/usr/bin/env python3
"""Build the harmonized two-cycle NHANES modeling cohort."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.combined_builder import build_combined_cohort  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-data-dir", type=Path, default=Path("/tmp/nhanes_2017_2020"))
    args = parser.parse_args()
    result = build_combined_cohort(ROOT, args.old_data_dir)
    print(f"rows={result['rows']} features={result['feature_count']}")
    print(result["cycle_target_counts"])
