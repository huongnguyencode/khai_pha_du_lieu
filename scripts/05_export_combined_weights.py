#!/usr/bin/env python3
"""Export human-readable final weights from the combined model artifact."""

import csv
import pickle
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.combined_experiment import shrink  # noqa: E402


artifact_path = ROOT / "models/nhanes_combined_dual_axis.pkl"
with artifact_path.open("rb") as stream:
    artifact = pickle.load(stream)

rows = []
for axis, key in (("egfr", "normalized_mi_egfr"), ("uacr", "normalized_mi_uacr")):
    normalized = artifact[key]
    alpha = float(artifact["params"][axis]["alpha"])
    final = shrink(normalized, alpha)
    for feature, mi, weight in zip(artifact["features"], normalized, final):
        rows.append({"axis": axis, "feature": feature, "normalized_mi": float(mi), "alpha": alpha, "final_weight": float(weight)})

output = ROOT / "reports/combined_w1/final_axis_weights.csv"
with output.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
print(output)
