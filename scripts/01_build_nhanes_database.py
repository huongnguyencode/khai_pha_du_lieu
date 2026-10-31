#!/usr/bin/env python3
"""Build the canonical NHANES master and complete modeling table."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.builder import build_all  # noqa: E402


if __name__ == "__main__":
    result = build_all(ROOT)
    print(f"master_rows={result['master_rows']}")
    print(f"modeling_rows={result['modeling_rows']}")
    print(f"target_counts={result['target_counts']}")
    print(f"historical_split_counts={result['historical_split_counts']}")

