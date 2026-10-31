"""Repeated nested-CV experiments for the harmonized two-cycle cohort.

The primary experiment keeps the four-pattern endpoint while also modelling its
two constituent axes (low eGFR and high uACR) with separate MI weight vectors.
Nearest-neighbour search is exact: categorical mismatch distances are embedded
as normalized one-hot Euclidean coordinates and queried with ``cKDTree``.
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
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import t as student_t

from .builder import CATEGORICAL_LEVELS
from .combined_builder import PRIMARY_CATEGORICAL, PRIMARY_FEATURES


ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
K_VALUES = (3, 5, 7, 11, 15)
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
NUMERIC = tuple(feature for feature in PRIMARY_FEATURES if feature not in PRIMARY_CATEGORICAL)
MODELS = (
    "flat_uniform_distance",
    "flat_mi_distance",
    "dual_uniform_distance",
    "dual_mi_distance",
    "dual_mi_balanced",
    "dual_mi_balanced_missing",
)


def _float(text: str) -> float:
    return math.nan if text == "" else float(text)


def load_combined(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    matrix = np.asarray([[_float(row[feature]) for feature in PRIMARY_FEATURES] for row in rows], dtype=float)
    target = np.asarray([int(row["target"]) for row in rows], dtype=int)
    egfr_low = np.asarray([int(row["egfr_low"]) for row in rows], dtype=int)
    uacr_high = np.asarray([int(row["uacr_high"]) for row in rows], dtype=int)
    cycles = np.asarray([row["cycle"] for row in rows])
    row_ids = [row["row_id"] for row in rows]
    if len(rows) != 13086 or len(set(row_ids)) != len(rows):
        raise ValueError("Combined modeling table invariant failed")
    return matrix, target, egfr_low, uacr_high, cycles, row_ids


@dataclass
class CombinedPreprocessor:
    medians: np.ndarray
    scales: np.ndarray
    modes: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "CombinedPreprocessor":
        numeric = matrix[:, [PRIMARY_FEATURES.index(feature) for feature in NUMERIC]]
        medians = np.nanmedian(numeric, axis=0)
        filled = np.where(np.isnan(numeric), medians, numeric)
        scales = np.std(filled, axis=0)
        scales[~np.isfinite(scales) | (scales == 0)] = 1.0
        modes = []
        for feature in PRIMARY_CATEGORICAL:
            column = matrix[:, PRIMARY_FEATURES.index(feature)]
            values, counts = np.unique(column[np.isfinite(column)], return_counts=True)
            if not len(values):
                raise ValueError(f"No observed values for {feature}")
            modes.append(values[np.argmax(counts)])
        return cls(medians, scales, np.asarray(modes, dtype=float))

    def transform(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_num = matrix[:, [PRIMARY_FEATURES.index(feature) for feature in NUMERIC]]
        num_missing = np.isnan(raw_num)
        numeric = (np.where(num_missing, self.medians, raw_num) - self.medians) / self.scales
        raw_cat = matrix[:, [PRIMARY_FEATURES.index(feature) for feature in PRIMARY_CATEGORICAL]]
        cat_missing = np.isnan(raw_cat)
        categorical = np.where(cat_missing, self.modes, raw_cat)
        return numeric, categorical, num_missing, cat_missing


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold, piece in enumerate(np.array_split(indices, n_splits)):
            parts[fold].extend(map(int, piece))
    all_indices = np.arange(len(y))
    result = []
    for part in parts:
        valid = np.asarray(sorted(part), dtype=int)
        train = np.setdiff1d(all_indices, valid, assume_unique=True)
        result.append((train, valid))
    return result


def _mi_discrete(values: np.ndarray, target: np.ndarray) -> float:
    x_levels, x_inv = np.unique(values, return_inverse=True)
    y_levels, y_inv = np.unique(target, return_inverse=True)
    joint = np.zeros((len(x_levels), len(y_levels)), dtype=float)
    np.add.at(joint, (x_inv, y_inv), 1)
    joint /= len(values)
    px, py = joint.sum(axis=1, keepdims=True), joint.sum(axis=0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / (px @ py)[mask])))


def _mi_continuous(values: np.ndarray, target: np.ndarray, neighbors: int = 3) -> float:
    radii = np.empty(len(values)); k_all = np.empty(len(values)); label_counts = np.empty(len(values))
    for label in np.unique(target):
        mask = target == label
        subset = values[mask, None]
        k = min(neighbors, len(subset) - 1)
        distances, _ = cKDTree(subset).query(subset, k=k + 1)
        radii[mask] = np.nextafter(distances[:, -1], 0)
        k_all[mask] = k; label_counts[mask] = len(subset)
    tree = cKDTree(values[:, None])
    counts = tree.query_ball_point(values[:, None], radii, return_length=True) - 1
    estimate = digamma(len(values)) + np.mean(digamma(k_all)) - np.mean(digamma(label_counts)) - np.mean(digamma(counts + 1))
    return max(0.0, float(estimate))


def estimate_weights(matrix: np.ndarray, y: np.ndarray, prep: CombinedPreprocessor, seed: int) -> tuple[np.ndarray, np.ndarray]:
    numeric, categorical, _, _ = prep.transform(matrix)
    rng = np.random.default_rng(seed)
    scores = []
    for feature in PRIMARY_FEATURES:
        if feature in PRIMARY_CATEGORICAL:
            scores.append(_mi_discrete(categorical[:, PRIMARY_CATEGORICAL.index(feature)], y))
        else:
            values = numeric[:, NUMERIC.index(feature)].copy()
            values += 1e-10 * max(1.0, float(np.mean(np.abs(values)))) * rng.standard_normal(len(values))
            scores.append(_mi_continuous(values, y))
    raw = np.maximum(np.asarray(scores), 0)
    epsilon = 1e-6
    normalized = np.full(len(PRIMARY_FEATURES), 1 / len(PRIMARY_FEATURES))
    if np.isfinite(raw).all() and raw.sum() > 0:
        normalized = (raw + epsilon) / (raw.sum() + epsilon * len(raw))
    return raw, normalized


def shrink(normalized: np.ndarray, alpha: float) -> np.ndarray:
    return (1 - alpha) * np.full(len(PRIMARY_FEATURES), 1 / len(PRIMARY_FEATURES)) + alpha * normalized


def embedding(matrix: np.ndarray, prep: CombinedPreprocessor, weights: np.ndarray, add_missing: bool) -> np.ndarray:
    numeric, categorical, num_missing, cat_missing = prep.transform(matrix)
    blocks = []
    for pos, feature in enumerate(NUMERIC):
        weight = weights[PRIMARY_FEATURES.index(feature)]
        value_scale = math.sqrt(weight / 2 if add_missing else weight)
        blocks.append(numeric[:, pos : pos + 1] * value_scale)
        if add_missing:
            blocks.append(num_missing[:, pos : pos + 1].astype(float) * math.sqrt(weight / 2))
    for pos, feature in enumerate(PRIMARY_CATEGORICAL):
        weight = weights[PRIMARY_FEATURES.index(feature)]
        levels = np.asarray(CATEGORICAL_LEVELS[feature], dtype=float)
        one_hot = (categorical[:, pos : pos + 1] == levels[None, :]).astype(float)
        # A category mismatch has squared one-hot distance 2.
        blocks.append(one_hot * math.sqrt(weight / (4 if add_missing else 2)))
        if add_missing:
            blocks.append(cat_missing[:, pos : pos + 1].astype(float) * math.sqrt(weight / 2))
    return np.concatenate(blocks, axis=1)


def query_neighbors(train: np.ndarray, valid: np.ndarray, k: int = max(K_VALUES)) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = cKDTree(train).query(valid, k=k, workers=1)
    if k == 1:
        distances, indices = distances[:, None], indices[:, None]
    return distances, indices


def binary_scores(distances: np.ndarray, indices: np.ndarray, y_train: np.ndarray, k: int, balanced: bool) -> np.ndarray:
    d, labels = distances[:, :k], y_train[indices[:, :k]]
    vote = 1.0 / np.maximum(d, 1e-12)
    zero = d <= 1e-12
    any_zero = zero.any(axis=1)
    vote[any_zero] = zero[any_zero].astype(float)
    if balanced:
        counts = np.bincount(y_train, minlength=2).astype(float)
        factors = len(y_train) / (2 * np.maximum(counts, 1))
        vote *= factors[labels]
    denominator = vote.sum(axis=1)
    return (vote * labels).sum(axis=1) / np.maximum(denominator, 1e-15)


def flat_predictions(distances: np.ndarray, indices: np.ndarray, y_train: np.ndarray, k: int) -> np.ndarray:
    d, labels = distances[:, :k], y_train[indices[:, :k]]
    vote = 1.0 / np.maximum(d, 1e-12)
    zero = d <= 1e-12
    any_zero = zero.any(axis=1)
    vote[any_zero] = zero[any_zero].astype(float)
    totals = np.zeros((len(d), 4), dtype=float)
    for label in range(4):
        totals[:, label] = (vote * (labels == label)).sum(axis=1)
    return np.argmax(totals, axis=1)


def binary_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict[str, float]:
    tp = int(np.sum((y == 1) & (pred == 1))); tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1))); fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1, "balanced_accuracy": (recall + specificity) / 2}
    if score is not None:
        order = np.argsort(-score, kind="mergesort")
        sorted_y = y[order]
        positives = max(int(y.sum()), 1)
        tp_curve = np.cumsum(sorted_y == 1)
        ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(y) - 1]
        tp = tp_curve[ends]
        result["average_precision"] = float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / positives)
    return result


def four_class_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    recalls, precisions, f1s = [], [], []
    result: dict[str, float] = {"accuracy": float(np.mean(y == pred))}
    for label in range(4):
        tp = int(np.sum((y == label) & (pred == label)))
        actual = int(np.sum(y == label)); predicted = int(np.sum(pred == label))
        recall = tp / actual if actual else 0.0; precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall); precisions.append(precision); f1s.append(f1)
        result[f"precision_{label}"] = precision; result[f"recall_{label}"] = recall; result[f"f1_{label}"] = f1
    result["macro_f1"] = float(np.mean(f1s)); result["balanced_accuracy"] = float(np.mean(recalls))
    return result


def _best(scores: dict[tuple, list[float]]) -> tuple:
    return sorted(scores, key=lambda key: (-float(np.mean(scores[key])), key))[0]


def select_all(matrix: np.ndarray, y4: np.ndarray, yg: np.ndarray, ya: np.ndarray, seed: int, splits: int = 3) -> dict[str, object]:
    flat_uniform = {(0.0, k): [] for k in K_VALUES}
    flat_mi = {(alpha, k): [] for alpha in ALPHAS for k in K_VALUES}
    dual: dict[tuple[str, str], dict[tuple, list[float]]] = {}
    for model in MODELS[2:]:
        for axis in ("egfr", "uacr"):
            alphas = (0.0,) if model == "dual_uniform_distance" else ALPHAS
            thresholds = THRESHOLDS if "balanced" in model else (0.5,)
            dual[(model, axis)] = {(alpha, k, threshold): [] for alpha in alphas for k in K_VALUES for threshold in thresholds}

    for fold, (train_idx, valid_idx) in enumerate(stratified_folds(y4, splits, seed)):
        prep = CombinedPreprocessor.fit(matrix[train_idx])
        _, global_mi = estimate_weights(matrix[train_idx], y4[train_idx], prep, seed + 100 + fold)
        axis_mi = {}
        for axis, target in (("egfr", yg), ("uacr", ya)):
            _, axis_mi[axis] = estimate_weights(matrix[train_idx], target[train_idx], prep, seed + (200 if axis == "egfr" else 300) + fold)

        for alpha in ALPHAS:
            weights = shrink(global_mi, alpha)
            tr = embedding(matrix[train_idx], prep, weights, False); va = embedding(matrix[valid_idx], prep, weights, False)
            distances, indices = query_neighbors(tr, va)
            for k in K_VALUES:
                metric = four_class_metrics(y4[valid_idx], flat_predictions(distances, indices, y4[train_idx], k))["macro_f1"]
                flat_mi[(alpha, k)].append(metric)
                if alpha == 0:
                    flat_uniform[(0.0, k)].append(metric)

        for add_missing in (False, True):
            applicable = [model for model in MODELS[2:] if (model.endswith("_missing") == add_missing)]
            if not applicable:
                continue
            for axis, target in (("egfr", yg), ("uacr", ya)):
                for alpha in ALPHAS:
                    weights = shrink(axis_mi[axis], alpha)
                    tr = embedding(matrix[train_idx], prep, weights, add_missing)
                    va = embedding(matrix[valid_idx], prep, weights, add_missing)
                    distances, indices = query_neighbors(tr, va)
                    for model in applicable:
                        if model == "dual_uniform_distance" and alpha != 0:
                            continue
                        balanced = "balanced" in model
                        thresholds = THRESHOLDS if balanced else (0.5,)
                        for k in K_VALUES:
                            score = binary_scores(distances, indices, target[train_idx], k, balanced)
                            for threshold in thresholds:
                                metric = binary_metrics(target[valid_idx], (score >= threshold).astype(int))["f1"]
                                dual[(model, axis)][(alpha, k, threshold)].append(metric)

    selected: dict[str, object] = {
        "flat_uniform_distance": {"alpha": _best(flat_uniform)[0], "k": _best(flat_uniform)[1]},
        "flat_mi_distance": {"alpha": _best(flat_mi)[0], "k": _best(flat_mi)[1]},
    }
    for model in MODELS[2:]:
        selected[model] = {}
        for axis in ("egfr", "uacr"):
            alpha, k, threshold = _best(dual[(model, axis)])
            selected[model][axis] = {"alpha": alpha, "k": k, "threshold": threshold}
    return selected


def evaluate_selected(
    train_matrix: np.ndarray, test_matrix: np.ndarray, train_y4: np.ndarray, test_y4: np.ndarray,
    train_yg: np.ndarray, test_yg: np.ndarray, train_ya: np.ndarray, test_ya: np.ndarray,
    selected: dict[str, object], seed: int,
) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    prep = CombinedPreprocessor.fit(train_matrix)
    rows = []
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    _, global_mi = estimate_weights(train_matrix, train_y4, prep, seed + 10)
    for model in MODELS[:2]:
        params = selected[model]
        normalized = global_mi if model == "flat_mi_distance" else np.full(len(PRIMARY_FEATURES), 1 / len(PRIMARY_FEATURES))
        weights = shrink(normalized, float(params["alpha"]))
        distances, indices = query_neighbors(embedding(train_matrix, prep, weights, False), embedding(test_matrix, prep, weights, False))
        pred = flat_predictions(distances, indices, train_y4, int(params["k"]))
        predictions[model] = (pred, np.full(len(pred), np.nan), np.full(len(pred), np.nan))
        rows.append({"model": model, **four_class_metrics(test_y4, pred)})

    axis_mi = {}
    for axis, target in (("egfr", train_yg), ("uacr", train_ya)):
        _, axis_mi[axis] = estimate_weights(train_matrix, target, prep, seed + (20 if axis == "egfr" else 30))
    for model in MODELS[2:]:
        add_missing = model.endswith("_missing"); balanced = "balanced" in model
        axis_scores = {}; axis_predictions = {}
        for axis, train_target in (("egfr", train_yg), ("uacr", train_ya)):
            params = selected[model][axis]
            normalized = axis_mi[axis] if model != "dual_uniform_distance" else np.full(len(PRIMARY_FEATURES), 1 / len(PRIMARY_FEATURES))
            weights = shrink(normalized, float(params["alpha"]))
            distances, indices = query_neighbors(
                embedding(train_matrix, prep, weights, add_missing), embedding(test_matrix, prep, weights, add_missing)
            )
            score = binary_scores(distances, indices, train_target, int(params["k"]), balanced)
            axis_scores[axis] = score
            axis_predictions[axis] = (score >= float(params["threshold"])).astype(int)
        pred = axis_predictions["uacr"] + 2 * axis_predictions["egfr"]
        predictions[model] = (pred, axis_scores["egfr"], axis_scores["uacr"])
        metric = {"model": model, **four_class_metrics(test_y4, pred)}
        for axis, truth in (("egfr", test_yg), ("uacr", test_ya)):
            for name, value in binary_metrics(truth, axis_predictions[axis], axis_scores[axis]).items():
                metric[f"{axis}_{name}"] = value
        rows.append(metric)
    return rows, predictions


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    keys = list(rows[0])
    for row in rows[1:]:
        keys.extend(key for key in row if key not in keys)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def run_combined_experiment(root: str | Path, repeats: int = 5, outer_splits: int = 5, inner_splits: int = 3) -> dict[str, object]:
    root = Path(root)
    matrix, y4, yg, ya, cycles, row_ids = load_combined(root / "data/processed/nhanes_combined_2017_2023.csv")
    output = root / "reports/combined_w1"; output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []; prediction_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for outer_fold, (train_idx, test_idx) in enumerate(stratified_folds(y4, outer_splits, 42 + repeat * 100)):
            selected = select_all(matrix[train_idx], y4[train_idx], yg[train_idx], ya[train_idx], 10000 + repeat * 100 + outer_fold, inner_splits)
            metrics, predictions = evaluate_selected(
                matrix[train_idx], matrix[test_idx], y4[train_idx], y4[test_idx], yg[train_idx], yg[test_idx],
                ya[train_idx], ya[test_idx], selected, 20000 + repeat * 100 + outer_fold,
            )
            for row in metrics:
                row.update({"repeat": repeat, "outer_fold": outer_fold, "selected_params": json.dumps(selected[row["model"]], sort_keys=True)})
                fold_rows.append(row)
            for model, (pred, score_g, score_a) in predictions.items():
                for pos, idx in enumerate(test_idx):
                    prediction_rows.append({
                        "model": model, "repeat": repeat, "outer_fold": outer_fold, "row_id": row_ids[idx],
                        "cycle": cycles[idx], "true_target": int(y4[idx]), "predicted_target": int(pred[pos]),
                        "egfr_score": "" if np.isnan(score_g[pos]) else float(score_g[pos]),
                        "uacr_score": "" if np.isnan(score_a[pos]) else float(score_a[pos]),
                    })
            print(f"repeat={repeat + 1}/{repeats} outer_fold={outer_fold + 1}/{outer_splits}", flush=True)

    summary = []
    metric_names = ("macro_f1", "balanced_accuracy", "accuracy", "recall_0", "recall_1", "recall_2", "recall_3")
    for model in MODELS:
        part = [row for row in fold_rows if row["model"] == model]
        item: dict[str, object] = {"model": model, "outer_evaluations": len(part)}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in part])
            item[f"{metric}_mean"] = float(values.mean()); item[f"{metric}_std"] = float(values.std(ddof=1))
        for axis in ("egfr", "uacr"):
            for metric in ("f1", "recall", "precision", "average_precision"):
                key = f"{axis}_{metric}"
                values = [float(row[key]) for row in part if key in row]
                item[f"{key}_mean"] = float(np.mean(values)) if values else ""
        summary.append(item)
    summary.sort(key=lambda row: float(row["macro_f1_mean"]), reverse=True)
    _write_rows(output / "nested_cv_fold_metrics.csv", fold_rows)
    _write_rows(output / "oof_predictions.csv", prediction_rows)
    _write_rows(output / "model_summary.csv", summary)

    # Temporal validation: tune only on the older cycle and evaluate on the newer cycle.
    old = np.flatnonzero(cycles == "2017-2020"); new = np.flatnonzero(cycles == "2021-2023")
    temporal_selected = select_all(matrix[old], y4[old], yg[old], ya[old], 31000, inner_splits)
    temporal_rows, _ = evaluate_selected(matrix[old], matrix[new], y4[old], y4[new], yg[old], yg[new], ya[old], ya[new], temporal_selected, 32000)
    for row in temporal_rows:
        row["train_cycle"] = "2017-2020"; row["test_cycle"] = "2021-2023"
        row["selected_params"] = json.dumps(temporal_selected[row["model"]], sort_keys=True)
    _write_rows(output / "temporal_validation.csv", temporal_rows)

    best_model = str(summary[0]["model"])
    final_selected = select_all(matrix, y4, yg, ya, 41000, 5)
    prep = CombinedPreprocessor.fit(matrix)
    artifact: dict[str, object] = {
        "model_name": best_model, "params": final_selected[best_model], "features": PRIMARY_FEATURES,
        "categorical": PRIMARY_CATEGORICAL, "numeric": NUMERIC, "preprocessor": prep,
        "train_matrix": matrix, "train_target": y4, "train_egfr_low": yg, "train_uacr_high": ya,
        "train_row_ids": row_ids, "cycles": cycles,
    }
    if best_model.startswith("flat"):
        _, mi = estimate_weights(matrix, y4, prep, 42000)
        artifact["normalized_mi"] = mi
    else:
        artifact["normalized_mi_egfr"] = estimate_weights(matrix, yg, prep, 42001)[1]
        artifact["normalized_mi_uacr"] = estimate_weights(matrix, ya, prep, 42002)[1]
    model_dir = root / "models"; model_dir.mkdir(exist_ok=True)
    with (model_dir / "nhanes_combined_dual_axis.pkl").open("wb") as stream:
        pickle.dump(artifact, stream)

    comparison_model = "flat_mi_distance"
    best_values = np.asarray([row["macro_f1"] for row in fold_rows if row["model"] == best_model])
    base_values = np.asarray([row["macro_f1"] for row in fold_rows if row["model"] == comparison_model])
    # Use one paired mean difference per repeat. Treating all overlapping outer
    # folds as independent would make the interval spuriously narrow.
    repeat_differences = []
    for repeat in range(repeats):
        best_repeat = [row["macro_f1"] for row in fold_rows if row["model"] == best_model and row["repeat"] == repeat]
        base_repeat = [row["macro_f1"] for row in fold_rows if row["model"] == comparison_model and row["repeat"] == repeat]
        repeat_differences.append(float(np.mean(best_repeat) - np.mean(base_repeat)))
    differences = np.asarray(repeat_differences)
    mean = float(differences.mean()); se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    critical = float(student_t.ppf(0.975, len(differences) - 1)) if len(differences) > 1 else math.nan
    result = {
        "rows": len(matrix), "features": len(PRIMARY_FEATURES), "protocol": {"repeats": repeats, "outer_splits": outer_splits, "inner_splits": inner_splits},
        "best_model": best_model, "final_params": final_selected[best_model], "summary": summary,
        "paired_comparison": {
            "comparison": f"{best_model} minus {comparison_model}", "unit": "mean outer-fold difference per repeat",
            "repeat_differences": repeat_differences, "mean": mean,
            "ci95_low": mean - critical * se, "ci95_high": mean + critical * se,
            "caution": "descriptive interval; repeated CV estimates remain dependent because training sets overlap",
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Combined NHANES 2017–2023 weighted-KNN experiment", "",
        f"Cohort: {len(matrix):,} adults with observed label sources; primary schema: {len(PRIMARY_FEATURES)} harmonized features.", "",
        "Urinary albumin from 2017–March 2020 was bridged to the 2021–2023 LC-MS/MS scale before target construction.", "",
        "| model | macro-F1 mean | balanced accuracy | accuracy |", "|---|---:|---:|---:|",
    ]
    for row in summary:
        report.append(f"| {row['model']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['balanced_accuracy_mean']:.4f} | {row['accuracy_mean']:.4f} |")
    report.extend(["", f"Best repeated nested-CV model: `{best_model}`.", "", f"Paired macro-F1 difference versus `{comparison_model}`: {mean:.4f}, 95% CI [{mean-critical*se:.4f}, {mean+critical*se:.4f}]."])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result
