#!/usr/bin/env python3
"""Rebuild the paired repeat-level comparison and detailed Vietnamese report."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
report_dir = ROOT / "reports/combined_w1"
result_path = report_dir / "results.json"
result = json.loads(result_path.read_text(encoding="utf-8"))
folds = list(csv.DictReader((report_dir / "nested_cv_fold_metrics.csv").open(newline="", encoding="utf-8")))
temporal = {row["model"]: row for row in csv.DictReader((report_dir / "temporal_validation.csv").open(newline="", encoding="utf-8"))}
weights = list(csv.DictReader((report_dir / "final_axis_weights.csv").open(newline="", encoding="utf-8")))

best = result["best_model"]
baseline = "flat_mi_distance"
grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
for row in folds:
    grouped[(row["model"], int(row["repeat"]))].append(float(row["macro_f1"]))
repeat_differences = [
    statistics.fmean(grouped[(best, repeat)]) - statistics.fmean(grouped[(baseline, repeat)])
    for repeat in range(result["protocol"]["repeats"])
]
mean = statistics.fmean(repeat_differences)
se = statistics.stdev(repeat_differences) / math.sqrt(len(repeat_differences))
critical = float(student_t.ppf(0.975, len(repeat_differences) - 1))
result["paired_comparison"] = {
    "comparison": f"{best} minus {baseline}",
    "unit": "mean outer-fold difference per repeat",
    "repeat_differences": repeat_differences,
    "mean": mean,
    "ci95_low": mean - critical * se,
    "ci95_high": mean + critical * se,
    "caution": "descriptive interval; repeated CV estimates remain dependent because training sets overlap",
}
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

summary = result["summary"]
lines = [
    "# NHANES 2017–2023 — W1 hai trục eGFR/uACR", "",
    "## Dữ liệu và protocol", "",
    "- Cohort: **13.086** người từ 20 tuổi, có đủ nguồn tạo eGFR và uACR.",
    "- Chu kỳ 2017–March 2020: 7.743; August 2021–August 2023: 5.343.",
    "- Target counts: lớp 0 = 10.786, lớp 1 = 1.228, lớp 2 = 693, lớp 3 = 379.",
    "- Primary schema: 23 feature tương thích; loại `moderate_freq_week` vì công cụ đo khác nhau giữa chu kỳ.",
    "- Albumin 2017–2020 được bridge sang thang LC-MS/MS 2021–2023 theo công thức CDC trước khi tạo target.",
    "- Đánh giá: repeated nested CV, 5 folds × 5 repeats; inner CV 3 folds.",
    "- Imputation, scaling, MI, alpha, K và threshold đều được fit trong inner-train/outer-train tương ứng.", "",
    "## Repeated nested-CV", "",
    "| Model | Macro-F1 | Balanced accuracy | Accuracy | Recall 0 | Recall 1 | Recall 2 | Recall 3 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in summary:
    lines.append(
        f"| {row['model']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
        f"{row['balanced_accuracy_mean']:.4f} | {row['accuracy_mean']:.4f} | {row['recall_0_mean']:.4f} | "
        f"{row['recall_1_mean']:.4f} | {row['recall_2_mean']:.4f} | {row['recall_3_mean']:.4f} |"
    )
lines.extend([
    "", f"Best model: `{best}`.", "",
    f"Chênh lệch macro-F1 so với `{baseline}`: **{mean:.4f}**, descriptive 95% CI theo 5 repeat "
    f"**[{mean-critical*se:.4f}, {mean+critical*se:.4f}]**. Các repeat vẫn dùng training set chồng lấn nên CI không phải kiểm định độc lập tuyệt đối.", "",
    "## Temporal validation", "",
    "Train trên 2017–March 2020, test nguyên chu kỳ 2021–2023:", "",
    "| Model | Macro-F1 | Balanced accuracy | Accuracy | Recall 0 | Recall 1 | Recall 2 | Recall 3 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
])
for model in ("flat_uniform_distance", "flat_mi_distance", "dual_mi_distance", "dual_mi_balanced", "dual_mi_balanced_missing"):
    row = temporal[model]
    lines.append(
        f"| {model} | {float(row['macro_f1']):.4f} | {float(row['balanced_accuracy']):.4f} | {float(row['accuracy']):.4f} | "
        f"{float(row['recall_0']):.4f} | {float(row['recall_1']):.4f} | {float(row['recall_2']):.4f} | {float(row['recall_3']):.4f} |"
    )
lines.extend(["", "## Trọng số cuối", ""])
for axis in ("egfr", "uacr"):
    top = sorted((row for row in weights if row["axis"] == axis), key=lambda row: float(row["final_weight"]), reverse=True)[:8]
    lines.extend([f"### Trục {axis}", "", "| Feature | Weight |", "|---|---:|"])
    lines.extend(f"| {row['feature']} | {float(row['final_weight']):.6f} |" for row in top)
    lines.append("")
lines.extend([
    "## Diễn giải giới hạn", "",
    "Class-balanced voting tăng mạnh recall lớp thiểu số nhưng làm giảm accuracy do tăng false positives. Missing indicators không cải thiện macro-F1 so với cùng model không indicator. Trọng số là association phục vụ khoảng cách, không phải quan hệ nhân quả; tuổi còn tham gia công thức eGFR. Target dựa trên một lần đo khảo sát và không xác nhận CKD mạn tính.", "",
])
(report_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print(result["paired_comparison"])
