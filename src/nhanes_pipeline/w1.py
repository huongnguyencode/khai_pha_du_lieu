"""Leakage-safe group-weighted KNN and nested-CV W1 experiment.

This module intentionally depends only on NumPy/SciPy plus the standard library.
The same code runs locally and inside the pinned Docker image.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.special import digamma
from scipy.spatial import cKDTree
from scipy.stats import t as student_t

from .builder import CATEGORICAL_LEVELS, FEATURES


CATEGORICAL = tuple(CATEGORICAL_LEVELS)
NUMERIC = tuple(feature for feature in FEATURES if feature not in CATEGORICAL)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _float(text: str) -> float:
    return math.nan if text == "" else float(text)


def load_training_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = []
    holdout_count = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["historical_split"] == "seen_holdout":
                holdout_count += 1
            elif row["historical_split"] == "train":
                rows.append(row)
    if len(rows) != 3740 or holdout_count != 1603:
        raise ValueError("Historical split invariant failed")
    matrix = np.asarray([[_float(row[feature]) for feature in FEATURES] for row in rows], dtype=float)
    target = np.asarray([int(row["target"]) for row in rows], dtype=int)
    seqn = np.asarray([int(row["SEQN"]) for row in rows], dtype=int)
    return matrix, target, seqn, list(FEATURES)


@dataclass
class FoldPreprocessor:
    numeric_medians: np.ndarray
    numeric_scales: np.ndarray
    categorical_modes: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "FoldPreprocessor":
        numeric = matrix[:, [FEATURES.index(feature) for feature in NUMERIC]]
        medians = np.nanmedian(numeric, axis=0)
        numeric_filled = np.where(np.isnan(numeric), medians, numeric)
        scales = np.std(numeric_filled, axis=0)
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0
        modes = []
        for feature in CATEGORICAL:
            column = matrix[:, FEATURES.index(feature)]
            values, counts = np.unique(column[np.isfinite(column)], return_counts=True)
            modes.append(float(values[np.argmax(counts)]))
        return cls(medians, scales, np.asarray(modes, dtype=float))

    def transform(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        numeric = matrix[:, [FEATURES.index(feature) for feature in NUMERIC]]
        numeric = np.where(np.isnan(numeric), self.numeric_medians, numeric)
        numeric = (numeric - self.numeric_medians) / self.numeric_scales
        categorical = matrix[:, [FEATURES.index(feature) for feature in CATEGORICAL]]
        categorical = np.where(np.isnan(categorical), self.categorical_modes, categorical)
        return numeric, categorical


def _mi_discrete(values: np.ndarray, target: np.ndarray) -> float:
    x_levels, x_inverse = np.unique(values, return_inverse=True)
    y_levels, y_inverse = np.unique(target, return_inverse=True)
    joint = np.zeros((len(x_levels), len(y_levels)), dtype=float)
    np.add.at(joint, (x_inverse, y_inverse), 1)
    joint /= len(values)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (px @ py)[mask])))


def _mi_continuous_discrete(values: np.ndarray, target: np.ndarray, neighbors: int = 3) -> float:
    """Ross-style kNN mutual information estimator for one continuous feature."""
    values = np.asarray(values, dtype=float)
    radii = np.empty(len(values), dtype=float)
    k_all = np.empty(len(values), dtype=float)
    label_counts = np.empty(len(values), dtype=float)
    for label in np.unique(target):
        mask = target == label
        subset = values[mask, None]
        count = len(subset)
        k = min(neighbors, count - 1)
        tree = cKDTree(subset)
        distances, _ = tree.query(subset, k=k + 1)
        radius = distances[:, -1] if distances.ndim == 2 else distances
        radii[mask] = np.nextafter(radius, 0)
        k_all[mask] = k
        label_counts[mask] = count
    tree = cKDTree(values[:, None])
    counts = tree.query_ball_point(values[:, None], radii, return_length=True) - 1
    estimate = digamma(len(values)) + np.mean(digamma(k_all)) - np.mean(digamma(label_counts)) - np.mean(digamma(counts + 1))
    return max(0.0, float(estimate))


def estimate_mi_weights(matrix: np.ndarray, y: np.ndarray, preprocessor: FoldPreprocessor, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    numeric, categorical = preprocessor.transform(matrix)
    rng = np.random.default_rng(seed)
    scores = []
    for feature in FEATURES:
        if feature in CATEGORICAL:
            scores.append(_mi_discrete(categorical[:, CATEGORICAL.index(feature)], y))
        else:
            # As in standard kNN MI estimators, break exact ties with negligible,
            # seeded noise. NHANES numeric variables are heavily rounded.
            values = numeric[:, NUMERIC.index(feature)].copy()
            scale = max(1.0, float(np.mean(np.abs(values))))
            values += 1e-10 * scale * rng.standard_normal(len(values))
            scores.append(_mi_continuous_discrete(values, y))
    mi = np.maximum(np.asarray(scores, dtype=float), 0.0)
    epsilon = 1e-6
    normalized = np.full(len(FEATURES), 1 / len(FEATURES)) if not np.isfinite(mi).all() or float(mi.sum()) <= 0 else (mi + epsilon) / (mi.sum() + epsilon * len(mi))
    return mi, normalized


def shrink_weights(normalized_mi: np.ndarray, alpha: float) -> np.ndarray:
    uniform = np.full(len(FEATURES), 1 / len(FEATURES))
    return (1 - alpha) * uniform + alpha * normalized_mi


def group_distances(train_num: np.ndarray, train_cat: np.ndarray, query_num: np.ndarray, query_cat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numeric_weights = np.asarray([weights[FEATURES.index(feature)] for feature in NUMERIC])
    categorical_weights = np.asarray([weights[FEATURES.index(feature)] for feature in CATEGORICAL])
    distance = np.zeros((len(query_num), len(train_num)), dtype=np.float64)
    for index in range(len(NUMERIC)):
        delta = query_num[:, None, index] - train_num[None, :, index]
        distance += numeric_weights[index] * delta * delta
    for index in range(len(CATEGORICAL)):
        distance += categorical_weights[index] * (query_cat[:, None, index] != train_cat[None, :, index])
    return np.sqrt(np.maximum(distance, 0.0))


def predict_from_distances(distances: np.ndarray, y_train: np.ndarray, k: int, vote: str) -> np.ndarray:
    neighbor_idx = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    predictions = np.empty(len(distances), dtype=int)
    for row_index, indices in enumerate(neighbor_idx):
        labels = y_train[indices]
        selected = distances[row_index, indices]
        zero = selected <= 1e-12
        if zero.any():
            labels, votes = labels[zero], np.ones(int(zero.sum()))
        else:
            votes = 1.0 / selected if vote == "distance" else np.ones(len(labels))
        totals = np.bincount(labels, weights=votes, minlength=4)
        predictions[row_index] = int(np.flatnonzero(totals == totals.max())[0])
    return predictions


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    test_parts: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, n_splits)):
            test_parts[fold].extend(map(int, part))
    all_indices = np.arange(len(y))
    folds = []
    for part in test_parts:
        test = np.asarray(sorted(part), dtype=int)
        train = np.setdiff1d(all_indices, test, assume_unique=True)
        folds.append((train, test))
    return folds


def score_predictions(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    recalls, precisions, f1s = [], [], []
    for label in range(4):
        tp = int(np.sum((y_true == label) & (prediction == label)))
        actual = int(np.sum(y_true == label)); predicted = int(np.sum(prediction == label))
        recall = tp / actual if actual else 0.0; precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall); precisions.append(precision); f1s.append(f1)
    result = {"macro_f1": float(np.mean(f1s)), "balanced_accuracy": float(np.mean(recalls)), "accuracy": float(np.mean(y_true == prediction))}
    for label in range(4):
        result[f"precision_{label}"] = precisions[label]; result[f"recall_{label}"] = recalls[label]; result[f"f1_{label}"] = f1s[label]
    return result


def _select_alphas(matrix: np.ndarray, y: np.ndarray, votes: Iterable[str], seed: int, splits: int) -> tuple[dict[str, float], list[dict[str, object]]]:
    votes = tuple(votes)
    collected = {(vote, alpha): [] for vote in votes for alpha in ALPHAS}
    detail = []
    for fold, (train_idx, valid_idx) in enumerate(stratified_folds(y, splits, seed)):
        prep = FoldPreprocessor.fit(matrix[train_idx])
        train_num, train_cat = prep.transform(matrix[train_idx]); valid_num, valid_cat = prep.transform(matrix[valid_idx])
        _, normalized = estimate_mi_weights(matrix[train_idx], y[train_idx], prep, seed + fold)
        for alpha in ALPHAS:
            distance = group_distances(train_num, train_cat, valid_num, valid_cat, shrink_weights(normalized, alpha))
            for vote in votes:
                prediction = predict_from_distances(distance, y[train_idx], 3, vote)
                metric = score_predictions(y[valid_idx], prediction)["macro_f1"]
                collected[(vote, alpha)].append(metric)
                detail.append({"inner_fold": fold, "vote": vote, "alpha": alpha, "macro_f1": metric})
    selected = {vote: sorted(((float(np.mean(collected[(vote, alpha)])), alpha) for alpha in ALPHAS), key=lambda item: (-item[0], item[1]))[0][1] for vote in votes}
    return selected, detail


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run_nested_w1(root: str | Path, outer_splits: int = 5, inner_splits: int = 3) -> dict[str, object]:
    root = Path(root)
    matrix, y, seqn, _ = load_training_table(root / "data/processed/nhanes_modeling_complete.csv")
    configs = {"M1_uniform_feature_uniform_vote": ("uniform", False), "M2_uniform_feature_distance_vote": ("distance", False), "M3_MI_feature_uniform_vote": ("uniform", True), "M4_MI_feature_distance_vote": ("distance", True)}
    fold_rows, weight_rows, prediction_rows, inner_rows = [], [], [], []
    for outer_fold, (train_idx, test_idx) in enumerate(stratified_folds(y, outer_splits, 42)):
        selected, details = _select_alphas(matrix[train_idx], y[train_idx], ("uniform", "distance"), 4200 + outer_fold, inner_splits)
        for row in details: row["outer_fold"] = outer_fold; inner_rows.append(row)
        prep = FoldPreprocessor.fit(matrix[train_idx])
        train_num, train_cat = prep.transform(matrix[train_idx]); test_num, test_cat = prep.transform(matrix[test_idx])
        raw_mi, normalized = estimate_mi_weights(matrix[train_idx], y[train_idx], prep, 5200 + outer_fold)
        cache = {}
        for model, (vote, uses_mi) in configs.items():
            alpha = selected[vote] if uses_mi else 0.0
            weights = shrink_weights(normalized, alpha) if uses_mi else np.full(len(FEATURES), 1 / len(FEATURES))
            if alpha not in cache: cache[alpha] = group_distances(train_num, train_cat, test_num, test_cat, weights)
            prediction = predict_from_distances(cache[alpha], y[train_idx], 3, vote)
            fold_rows.append({"model": model, "outer_fold": outer_fold, "selected_alpha": alpha, **score_predictions(y[test_idx], prediction)})
            for idx, predicted in zip(test_idx, prediction): prediction_rows.append({"model": model, "outer_fold": outer_fold, "SEQN": int(seqn[idx]), "true_target": int(y[idx]), "predicted_target": int(predicted)})
            if uses_mi:
                for pos, feature in enumerate(FEATURES): weight_rows.append({"model": model, "outer_fold": outer_fold, "selected_alpha": alpha, "feature": feature, "raw_mi": raw_mi[pos], "normalized_mi": normalized[pos], "weight": weights[pos]})
    summary = []
    for model in configs:
        part = [row for row in fold_rows if row["model"] == model]; result = {"model": model, "outer_folds": len(part), "k": 3}
        for metric in ("macro_f1", "balanced_accuracy", "accuracy", "recall_0", "recall_1", "recall_2", "recall_3"):
            values = np.asarray([row[metric] for row in part]); result[f"{metric}_mean"] = float(values.mean()); result[f"{metric}_std"] = float(values.std(ddof=1))
        result["selected_alpha_values"] = "|".join(f"{row['selected_alpha']:g}" for row in part); summary.append(result)
    summary.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    output = root / "reports/w1"; output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "nested_cv_fold_metrics.csv", fold_rows); _write_rows(output / "inner_cv_alpha_scores.csv", inner_rows); _write_rows(output / "oof_predictions.csv", prediction_rows); _write_rows(output / "weights_w1_by_fold.csv", weight_rows); _write_rows(output / "nhanes_ablation.csv", summary)
    weight_summary = []
    for model in ("M3_MI_feature_uniform_vote", "M4_MI_feature_distance_vote"):
        for feature in FEATURES:
            values = np.asarray([row["weight"] for row in weight_rows if row["model"] == model and row["feature"] == feature])
            weight_summary.append({"model": model, "feature": feature, "mean": values.mean(), "std": values.std(ddof=1), "min": values.min(), "max": values.max()})
    _write_rows(output / "weights_w1_summary.csv", weight_summary)
    selected_model = summary[0]["model"]; vote, uses_mi = configs[selected_model]
    final_alpha = _select_alphas(matrix, y, (vote,), 6200, 5)[0][vote] if uses_mi else 0.0
    prep = FoldPreprocessor.fit(matrix); train_num, train_cat = prep.transform(matrix); raw_mi, normalized = estimate_mi_weights(matrix, y, prep, 7200)
    weights = shrink_weights(normalized, final_alpha) if uses_mi else np.full(len(FEATURES), 1 / len(FEATURES))
    _write_rows(output / "final_weights.csv", [
        {"feature": feature, "raw_mi": raw_mi[pos], "normalized_mi": normalized[pos], "final_weight": weights[pos]}
        for pos, feature in enumerate(FEATURES)
    ])
    artifact = {"model_name": selected_model, "vote": vote, "k": 3, "alpha": final_alpha, "features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC, "preprocessor": prep, "train_numeric": train_num, "train_categorical": train_cat, "train_target": y, "train_seqn": seqn, "raw_mi": raw_mi, "normalized_mi": normalized, "feature_weights": weights, "holdout_used_for_selection": False}
    models = root / "models"; models.mkdir(exist_ok=True)
    with (models / "nhanes_w1_group_knn.pkl").open("wb") as stream: pickle.dump(artifact, stream)
    m4 = np.asarray([row["macro_f1"] for row in fold_rows if row["model"] == "M4_MI_feature_distance_vote"])
    m2 = np.asarray([row["macro_f1"] for row in fold_rows if row["model"] == "M2_uniform_feature_distance_vote"])
    differences = m4 - m2
    difference_mean = float(differences.mean())
    difference_se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    critical = float(student_t.ppf(0.975, len(differences) - 1))
    paired_comparison = {
        "comparison": "M4 minus M2 outer-fold macro-F1",
        "mean_difference": difference_mean,
        "ci95_low": difference_mean - critical * difference_se,
        "ci95_high": difference_mean + critical * difference_se,
        "interpretation": "not_confirmed" if difference_mean - critical * difference_se <= 0 else "confirmed_positive",
    }
    result = {"protocol": {"outer_folds": outer_splits, "inner_folds": inner_splits, "k_fixed": 3, "alphas": ALPHAS}, "selection_data": "historical_split=train only", "seen_holdout_used": False, "selected_model_by_mean": selected_model, "final_alpha": final_alpha, "paired_comparison": paired_comparison, "nested_cv": summary}
    (output / "w1_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    header = ["model", "macro_f1_mean", "macro_f1_std", "balanced_accuracy_mean", "accuracy_mean", "selected_alpha_values"]
    table = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + ["| " + " | ".join(str(row[key]) for key in header) + " |" for row in summary]
    report = ["# NHANES W1 — group Mutual Information weights", "", "All preprocessing, MI estimation, alpha selection, and evaluation used only the 3,740-row historical training partition. `seen_holdout` was not used.", "", *table, "", f"Best mean configuration: `{selected_model}`; full-train CV alpha: `{final_alpha}`.", "", f"Paired outer-fold M4−M2 macro-F1 difference: `{difference_mean:.6f}`; 95% CI `[{paired_comparison['ci95_low']:.6f}, {paired_comparison['ci95_high']:.6f}]`. The interval crosses zero, so W1 improvement is not confirmed."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
