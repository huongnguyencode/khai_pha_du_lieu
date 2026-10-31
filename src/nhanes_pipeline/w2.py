"""Controlled, fold-local dual-axis KNN ablations and budgeted group search.

A reproduces the historical W1 algorithm; B changes joint thresholds only;
C also tunes K/alpha jointly; D/E perturb C's group weights and retune K and
thresholds. Scores always use W1's balanced distance voting and MI definition.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t as student_t

from .combined_builder import PRIMARY_FEATURES
from .combined_experiment import (
    ALPHAS, K_VALUES, THRESHOLDS as LEGACY_THRESHOLDS,
    CombinedPreprocessor, binary_metrics, binary_scores, embedding,
    estimate_weights, four_class_metrics, load_combined, query_neighbors,
    shrink, stratified_folds, _write_rows,
)

GROUPS = {
    "age_sex": ("RIDAGEYR", "RIAGENDR"),
    "social": ("RIDRETH3", "DMDEDUC2", "INDFMPIR"),
    "body": ("BMXBMI", "BMXWAIST"),
    "blood_pressure": ("sbp_mean", "dbp_mean", "BPQ020", "BPQ150"),
    "history": ("DIQ010", "DIQ160", "BPQ080"),
    "lifestyle": ("current_smoking", "ALQ111", "ALQ121", "ALQ130", "PAD680"),
    "labs": ("LBXTC", "LBDHDD", "LBXSUA", "LBXHGB"),
}
GROUP_INDEX = np.array([next(i for i, names in enumerate(GROUPS.values()) if f in names) for f in PRIMARY_FEATURES])
KS = tuple(sorted(set(K_VALUES) | {21, 31, 51}))
THRESHOLDS = tuple(sorted(set(LEGACY_THRESHOLDS) | {.75, .8, .85, .9, .95}))
LAMBDA = .01
MODELS = ("A_W1_original", "B_joint_thresholds", "C_joint_K_alpha",
          "D_W2_groups", "E_W2_groups_regularized")
CANDIDATE_SEED = 20261031


def candidates():
    """Identity plus 24 fixed sparse alternatives, generated without labels.

    Different numbers and magnitudes of adjustments give different penalties.
    This is a budgeted search, not exhaustive optimization of all group weights.
    """
    rng = np.random.default_rng(CANDIDATE_SEED)
    result = [np.zeros((2, len(GROUPS)))]
    seen = {tuple(result[0].ravel())}
    while len(result) < 25:
        theta = np.zeros(2 * len(GROUPS))
        size = 1 + (len(result) - 1) % 4
        indices = rng.choice(len(theta), size, replace=False)
        theta[indices] = rng.choice([-1., 1.], size) * rng.choice(np.log([1.5, 2.]), size)
        key = tuple(theta)
        if key not in seen:
            result.append(theta.reshape(2, len(GROUPS)))
            seen.add(key)
    return result


def weights(mi, alpha, theta):
    base = shrink(mi, alpha)  # Identical to W1; no hidden uniform floor.
    w = base * np.exp(theta[GROUP_INDEX])
    return w / w.sum()


def joint_score(y, g, a):
    pred = a.astype(int) + 2 * g.astype(int)
    cm = np.bincount(4 * y + pred, minlength=16).reshape(4, 4)
    denominator = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.divide(2 * np.diag(cm), denominator,
                                  out=np.zeros(4), where=denominator > 0)))


def axis_predictions(bank, settings):
    return [bank[a][tuple(settings[a][:2])] >= settings[a][2] for a in range(2)]


def tune_pair(bank, y, initial, thresholds=THRESHOLDS):
    """Two coordinate sweeps; preserve incumbent on ties, deterministic order.

    Each bank is inner OOF only. No outer truth is accepted by selection.
    """
    params = [tuple(p) for p in initial]
    pred = axis_predictions(bank, params)
    value = joint_score(y, *pred)
    for _ in range(2):
        changed = False
        for axis in (0, 1):
            best_value, best_params, best_pred = value, params[axis], pred[axis]
            for (alpha, k), score in sorted(bank[axis].items()):
                for threshold in thresholds:
                    current = score >= threshold
                    trial = joint_score(y, current if axis == 0 else pred[0],
                                        current if axis == 1 else pred[1])
                    if trial > best_value + 1e-12:
                        best_value, best_params, best_pred = trial, (alpha, k, threshold), current
            changed |= best_params != params[axis]
            params[axis], pred[axis], value = best_params, best_pred, best_value
        if not changed:
            break
    return params


def threshold_pair(scores, y, initial):
    best = [tuple(p) for p in initial]
    value = joint_score(y, scores[0] >= best[0][2], scores[1] >= best[1][2])
    for tg in THRESHOLDS:
        for ta in THRESHOLDS:
            current = joint_score(y, scores[0] >= tg, scores[1] >= ta)
            if current > value + 1e-12:
                value = current
                best = [(initial[0][0], initial[0][1], tg), (initial[1][0], initial[1][1], ta)]
    return value, best


def fit_context(x, y, seed):
    """Match W1 evaluate_selected MI seeds, for exact A reproduction."""
    prep = CombinedPreprocessor.fit(x)
    mi = [estimate_weights(x, y // 2, prep, seed + 20)[1],
          estimate_weights(x, y % 2, prep, seed + 30)[1]]
    return prep, mi


def scores_for(x, q, y, prep, mi, axis, alpha, theta, ks):
    w = weights(mi[axis], alpha, theta)
    d, idx = query_neighbors(embedding(x, prep, w, False), embedding(q, prep, w, False), max(ks))
    target = y // 2 if axis == 0 else y % 2
    return {k: binary_scores(d, idx, target, k, True) for k in ks}


def select(x, y, seed, inner=3):
    contexts = []
    bank = [{(alpha, k): np.zeros(len(y)) for alpha in ALPHAS for k in KS} for _ in range(2)]
    legacy_scores = [{} for _ in range(2)]
    for f, (tr, va) in enumerate(stratified_folds(y, inner, seed)):
        prep = CombinedPreprocessor.fit(x[tr])
        mi = [estimate_weights(x[tr], y[tr] // 2, prep, seed + 200 + f)[1],
              estimate_weights(x[tr], y[tr] % 2, prep, seed + 300 + f)[1]]
        contexts.append((tr, va, prep, mi))
        for axis in range(2):
            target = y[va] // 2 if axis == 0 else y[va] % 2
            for alpha in ALPHAS:
                values = scores_for(x[tr], x[va], y[tr], prep, mi, axis, alpha, np.zeros(len(GROUPS)), KS)
                for k, score in values.items():
                    bank[axis][(alpha, k)][va] = score
                    if k in K_VALUES:
                        for threshold in LEGACY_THRESHOLDS:
                            metric = binary_metrics(target, score >= threshold)["f1"]
                            legacy_scores[axis].setdefault((alpha, k, threshold), []).append(metric)
    original = [sorted(s, key=lambda key: (-float(np.mean(s[key])), key))[0] for s in legacy_scores]
    original_score = joint_score(y, *axis_predictions(bank, original))
    scores = [bank[a][original[a][:2]] for a in range(2)]
    b_value, b_params = threshold_pair(scores, y, original)
    c_params = tune_pair(bank, y, b_params)
    c_value = joint_score(y, *axis_predictions(bank, c_params))
    params = {MODELS[0]: {"axes": original, "candidate": 0},
              MODELS[1]: {"axes": b_params, "candidate": 0},
              MODELS[2]: {"axes": c_params, "candidate": 0}}
    history = [{"stage": name, "candidate": 0, "macro_f1": value, "penalty": 0.,
                "objective_regularized": value, "parameters": json.dumps(params[name])}
               for name, value in zip(MODELS[:3], (original_score, b_value, c_value))]
    best = {False: ((c_value, 0), params[MODELS[2]]), True: ((c_value, 0), params[MODELS[2]])}
    cache = {}
    for ci, theta in enumerate(candidates()):
        if ci == 0:
            continue
        candidate_bank = []
        for axis in range(2):
            alpha = c_params[axis][0]
            key = (axis, alpha, tuple(theta[axis]))
            if not np.any(theta[axis]):
                axis_bank = {(alpha, k): bank[axis][(alpha, k)] for k in KS}
            elif key in cache:
                axis_bank = cache[key]
            else:
                axis_bank = {(alpha, k): np.zeros(len(y)) for k in KS}
                for tr, va, prep, mi in contexts:
                    values = scores_for(x[tr], x[va], y[tr], prep, mi, axis, alpha, theta[axis], KS)
                    for k, score in values.items(): axis_bank[(alpha, k)][va] = score
                cache[key] = axis_bank
            candidate_bank.append(axis_bank)
        axes = tune_pair(candidate_bank, y, c_params)
        value = joint_score(y, *axis_predictions(candidate_bank, axes))
        penalty = float(np.sum(theta**2))
        settings = {"axes": axes, "candidate": ci}
        history.append({"stage": "group_search", "candidate": ci, "macro_f1": value,
                        "penalty": penalty, "objective_regularized": value - LAMBDA * penalty,
                        "parameters": json.dumps(settings)})
        for regularized in (False, True):
            objective = value - (LAMBDA * penalty if regularized else 0.)
            key = (objective, -ci)
            if key > best[regularized][0]: best[regularized] = (key, settings)
    params[MODELS[3]] = best[False][1]
    params[MODELS[4]] = best[True][1]
    return params, history


def predict_settings(x, y, q, prep, mi, settings):
    theta = candidates()[settings["candidate"]]
    scores = [scores_for(x, q, y, prep, mi, axis, float(p[0]), theta[axis], (int(p[1]),))[int(p[1])]
              for axis, p in enumerate(settings["axes"])]
    pg = scores[0] >= settings["axes"][0][2]
    pa = scores[1] >= settings["axes"][1][2]
    return pa.astype(int) + 2 * pg.astype(int), scores


def predict(artifact, matrix, model=None):
    """Predict without labels. Returned scores are balanced votes, not calibrated risk."""
    model = model or artifact["model_name"]
    q = np.asarray(matrix, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(artifact["features"]):
        raise ValueError("Prediction requires a 2-D matrix in artifact feature order")
    if artifact["source_hash"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Model source changed; use the module version recorded with this artifact")
    return predict_settings(artifact["train_matrix"], artifact["train_target"], q,
                            artifact["preprocessor"], artifact["mi"], artifact["parameters"][model])


def metrics_for(truth, pred, scores):
    pg, pa = pred // 2, pred % 2
    row = {**four_class_metrics(truth, pred),
           "hamming_loss": float(np.mean(np.r_[pg != truth // 2, pa != truth % 2])),
           "healthy_fp": int(np.sum((truth == 0) & (pred != 0)))}
    row["healthy_fp_per_1000"] = 1000 * row["healthy_fp"] / len(truth)
    for axis, target, p, score in (("egfr", truth // 2, pg, scores[0]), ("uacr", truth % 2, pa, scores[1])):
        row.update({axis + "_" + k: v for k, v in binary_metrics(target, p, score).items()})
        row[axis + "_fp"] = int(np.sum((p == 1) & (target == 0)))
        row[axis + "_fn"] = int(np.sum((p == 0) & (target == 1)))
        row[axis + "_fp_per_1000"] = 1000 * row[axis + "_fp"] / len(truth)
    return row


def evaluate(x, y, q, truth, params, seed):
    prep, mi = fit_context(x, y, seed)
    rows, preds, weight_rows, cache = [], {}, [], {}
    for model, settings in params.items():
        theta = candidates()[settings["candidate"]]
        scores = []
        for axis, (alpha, k, threshold) in enumerate(settings["axes"]):
            key = (axis, alpha, k, tuple(theta[axis]))
            if key not in cache:
                cache[key] = scores_for(x, q, y, prep, mi, axis, alpha, theta[axis], (k,))[k]
            scores.append(cache[key])
            w = weights(mi[axis], alpha, theta[axis])
            ranks = np.argsort(np.argsort(-w, kind="stable"), kind="stable") + 1
            for j, feature in enumerate(PRIMARY_FEATURES):
                weight_rows.append({"model": model, "axis": axis, "feature": feature,
                                    "group": list(GROUPS)[GROUP_INDEX[j]], "weight": float(w[j]),
                                    "rank": int(ranks[j]), "multiplier": float(np.exp(theta[axis, GROUP_INDEX[j]]))})
        pred = (scores[1] >= settings["axes"][1][2]).astype(int) + 2 * (scores[0] >= settings["axes"][0][2]).astype(int)
        rows.append({"model": model, **metrics_for(truth, pred, scores), "parameters": json.dumps(settings)})
        preds[model] = (pred, scores)
    return rows, preds, weight_rows, (prep, mi)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _prediction_rows(preds, va, y, ids, cycles, repeat, fold):
    return [{"model": m, "repeat": repeat, "outer_fold": fold, "row_id": ids[idx],
             "cycle": str(cycles[idx]), "true_target": int(y[idx]), "predicted_target": int(pred[pos]),
             "egfr_score": float(scores[0][pos]), "uacr_score": float(scores[1][pos])}
            for m, (pred, scores) in preds.items() for pos, idx in enumerate(va)]


def _fold_job(root, output, repeat, fold, tr, va, inner):
    started = time.monotonic()
    root, out = Path(root), Path(output)
    folder = out / f"repeat_{repeat}_fold_{fold}"
    folder.mkdir(exist_ok=True)
    done = folder / "metrics.json"
    if done.exists():
        for name in ("weights.csv", "search.csv", "predictions.csv"):
            if not (folder / name).is_file(): raise ValueError(f"Incomplete checkpoint: {folder}")
        return json.loads(done.read_text()), list(csv.DictReader((folder / "weights.csv").open()))
    x, y, _, _, cycles, ids = load_combined(root / "data/processed/nhanes_combined_2017_2023.csv")
    params, history = select(x[tr], y[tr], 10000 + repeat * 100 + fold, inner)
    metrics, preds, wr, _ = evaluate(x[tr], y[tr], x[va], y[va], params, 20000 + repeat * 100 + fold)
    for row in metrics: row.update(repeat=repeat, outer_fold=fold)
    for row in wr: row.update(repeat=repeat, outer_fold=fold)
    _write_rows(folder / "weights.csv", wr)
    _write_rows(folder / "search.csv", history)
    _write_rows(folder / "predictions.csv", _prediction_rows(preds, va, y, ids, cycles, repeat, fold))
    _atomic_json(done, metrics)  # Completion marker written last.
    print(f"repeat={repeat+1} fold={fold+1} seconds={time.monotonic()-started:.1f}", flush=True)
    return metrics, wr


def _sd(values):
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.


def run(root, repeats=5, outer=5, inner=3, jobs=1, output=None):
    if repeats < 1 or min(outer, inner) < 2 or jobs < 1:
        raise ValueError("repeats/jobs >= 1 and outer/inner >= 2 are required")
    root = Path(root).resolve()
    out = Path(output).resolve() if output else root / "reports/w2"
    out.mkdir(parents=True, exist_ok=True)
    x, y, _, _, cycles, ids = load_combined(root / "data/processed/nhanes_combined_2017_2023.csv")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol = {"version": 2, "repeats": repeats, "outer": outer, "inner": inner,
                "ks": KS, "alphas": ALPHAS, "thresholds": THRESHOLDS, "lambda": LAMBDA,
                "candidate_seed": CANDIDATE_SEED, "candidates": [t.tolist() for t in candidates()],
                "source_hash": source_hash,
                "dependency_hash": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes() for name in ("combined_experiment.py", "combined_builder.py", "builder.py"))).hexdigest(),
                "data_hash": hashlib.sha256((root / "data/processed/nhanes_combined_2017_2023.csv").read_bytes()).hexdigest(),
                "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                "models": MODELS, "balanced": True, "mi_uniform_floor": 0.,
                "selection": "A: binary fold-mean F1; B: exhaustive threshold pair; C/D/E: two coordinate sweeps on inner pooled OOF macro-F1",
                "regularization": "lambda * sum(theta**2), relative to C MI weights; lambda fixed before evaluation",
                "group_search": "24 sparse fixed alternatives; C alpha held fixed; K/threshold retuned; includes identity",
                "score_interpretation": "class-balanced neighbor scores, not calibrated probabilities"}
    protocol = json.loads(json.dumps(protocol))
    protocol_path = out / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol/environment changed; choose a new --output directory")
    _atomic_json(protocol_path, protocol)
    tasks = [(str(root), str(out), repeat, fold, tr, va, inner)
             for repeat in range(repeats)
             for fold, (tr, va) in enumerate(stratified_folds(y, outer, 42 + repeat * 100))]
    all_metrics, all_weights = [], []
    if jobs == 1:
        results = [_fold_job(*task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(_fold_job, *task) for task in tasks]
            for future in as_completed(futures): results.append(future.result())
    for metrics, wr in results:
        all_metrics.extend(metrics); all_weights.extend(wr)
    all_metrics.sort(key=lambda r: (r["repeat"], r["outer_fold"], r["model"]))
    all_weights.sort(key=lambda r: (int(r["repeat"]), int(r["outer_fold"]), r["model"], int(r["axis"]), r["feature"]))
    _write_rows(out / "fold_metrics.csv", all_metrics)
    _write_rows(out / "weights_w2_by_fold.csv", all_weights)
    predictions = []
    for task in tasks:
        predictions.extend(csv.DictReader((out / f"repeat_{task[2]}_fold_{task[3]}" / "predictions.csv").open()))
    _write_rows(out / "oof_predictions.csv", predictions)
    summary = []
    metric_names = [k for k in all_metrics[0] if k not in ("model", "parameters", "repeat", "outer_fold")]
    for model in MODELS:
        part = [r for r in all_metrics if r["model"] == model]
        row = {"model": model, "outer_evaluations": len(part)}
        for metric in metric_names:
            a = [float(r[metric]) for r in part]
            row[metric] = float(np.mean(a)); row[metric + "_std"] = _sd(a)
        summary.append(row)
    _write_rows(out / "summary.csv", summary)
    stability = []
    for model in MODELS:
        for axis in range(2):
            for feature in PRIMARY_FEATURES:
                part = [r for r in all_weights if r["model"] == model and int(r["axis"]) == axis and r["feature"] == feature]
                values = [float(r["weight"]) for r in part]
                ranks = [int(r["rank"]) for r in part]
                stability.append({"model": model, "axis": axis, "feature": feature,
                                  "mean": float(np.mean(values)), "std": _sd(values),
                                  "min": min(values), "max": max(values), "mean_rank": float(np.mean(ranks)),
                                  "rank_std": _sd(ranks), "top5_frequency": float(np.mean(np.array(ranks) <= 5))})
    _write_rows(out / "weights_w2_summary.csv", stability)
    comparisons = []
    for current, baseline in ((MODELS[1], MODELS[0]), (MODELS[2], MODELS[1]),
                              (MODELS[3], MODELS[2]), (MODELS[4], MODELS[2]), (MODELS[4], MODELS[0])):
        differences = []
        for repeat in range(repeats):
            means = {m: np.mean([r["macro_f1"] for r in all_metrics if r["model"] == m and r["repeat"] == repeat]) for m in (current, baseline)}
            differences.append(float(means[current] - means[baseline]))
        mean = float(np.mean(differences))
        margin = float(student_t.ppf(.975, repeats-1) * _sd(differences) / np.sqrt(repeats)) if repeats > 1 else None
        comparisons.append({"model": current, "baseline": baseline, "mean_difference": mean,
                            "repeat_differences": differences, "ci95_low": mean-margin if margin is not None else None,
                            "ci95_high": mean+margin if margin is not None else None,
                            "caution": "Descriptive only; repeated CV training sets overlap; no independent significance claim"})
    old = np.flatnonzero(cycles == "2017-2020"); new = np.flatnonzero(cycles == "2021-2023")
    print("Starting temporal selection", flush=True)
    settings, temporal_history = select(x[old], y[old], 31000, inner)
    temporal, temporal_preds, temporal_weights, _ = evaluate(x[old], y[old], x[new], y[new], settings, 32000)
    _write_rows(out / "temporal.csv", temporal)
    _write_rows(out / "temporal_weights.csv", temporal_weights)
    _write_rows(out / "temporal_search.csv", temporal_history)
    _write_rows(out / "temporal_predictions.csv", _prediction_rows(temporal_preds, new, y, ids, cycles, -1, -1))
    print("Starting full-cohort selection", flush=True)
    settings, history = select(x, y, 41000, inner)
    prep, mi = fit_context(x, y, 42000)
    best_model = max(summary, key=lambda r: r["macro_f1"])["model"]
    artifact = {"model_name": best_model, "features": PRIMARY_FEATURES, "groups": GROUPS,
                "preprocessor": prep, "mi": mi, "train_matrix": x, "train_target": y,
                "row_ids": ids, "parameters": settings, "candidates": [t.tolist() for t in candidates()],
                "protocol": protocol, "source_hash": source_hash}
    model_path = (root / "models/nhanes_w2.pkl") if output is None else out / "nhanes_w2.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as f: pickle.dump(artifact, f)
    _atomic_json(out / "final_parameters.json", settings)
    _write_rows(out / "final_search.csv", history)
    final_weights = []
    for model, config in settings.items():
        theta = candidates()[config["candidate"]]
        for axis in range(2):
            w = weights(mi[axis], config["axes"][axis][0], theta[axis])
            final_weights.extend({"model": model, "axis": axis, "feature": feature,
                                  "group": list(GROUPS)[GROUP_INDEX[j]], "weight": float(w[j])}
                                 for j, feature in enumerate(PRIMARY_FEATURES))
    _write_rows(out / "final_weights.csv", final_weights)
    result = {"protocol": protocol, "summary": summary, "comparisons": comparisons,
              "temporal": temporal, "best_model": best_model, "model_path": str(model_path),
              "final_parameters": settings,
              "limitations": ["Exploratory model development on previously studied cohorts",
                              "Temporal cohort was previously seen; not independent external validation",
                              "Single measurement patterns do not establish chronic CKD",
                              "Survey weights/strata/PSU not used; no population inference",
                              "Group search limited to predeclared candidates; no global optimum claim",
                              "Best model chosen after comparing outer results; independent confirmation needed"]}
    _atomic_json(out / "results.json", result)
    write_report(out, result, predictions)
    print(json.dumps([{k: r[k] for k in ("model", "macro_f1", "accuracy")} for r in summary], indent=2), flush=True)
    return result


def write_report(out, result, predictions):
    lines = ["# Cải thiện W2 — NHANES 2017–2023", "",
             "## Thiết kế", "",
             f"Cohort 13.086 người, 23 biến, target bốn pattern eGFR/uACR. Outer {result['protocol']['outer']} folds × {result['protocol']['repeats']} repeats; inner {result['protocol']['inner']} folds. Cùng outer splits W1. Mọi preprocessing, MI và chọn tham số nằm trong train folds.", "",
             "- A: tái lập W1 balanced, chọn F1 riêng từng trục, lưới gốc.",
             "- B: giữ alpha/K của A; chọn cặp threshold theo macro-F1 bốn lớp.",
             "- C: thêm tìm alpha/K theo mục tiêu bốn lớp; K gồm 3,5,7,11,15,21,31,51.",
             "- D: 24 perturbations nhóm cố định và identity; alpha giữ từ C, K/threshold được tune lại.",
             "- E: cùng ứng viên D, objective = inner macro-F1 − 0,01 × sum(theta²), regularize về C.",
             "- Giữ nguyên balanced voting và công thức MI W1; không thêm uniform floor, missing indicator hay đổi cohort.",
             "- B duyệt toàn bộ cặp threshold trên lưới. C/D/E dùng hai lượt coordinate search, không bảo đảm tối ưu toàn cục.", "",
             "## Kết quả nested CV", "",
             "| Model | Macro-F1 ± SD | Accuracy | Precision 1 | Precision 2 | Precision 3 | Recall 3 | FP lớp 0 / 1.000 người |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["summary"]:
        lines.append(f"| {r['model']} | {r['macro_f1']:.4f} ± {r['macro_f1_std']:.4f} | {r['accuracy']:.4f} | {r['precision_1']:.4f} | {r['precision_2']:.4f} | {r['precision_3']:.4f} | {r['recall_3']:.4f} | {r['healthy_fp_per_1000']:.1f} |")
    lines += ["", "SD tính trên outer folds, không phải sai số chuẩn. FP lớp 0 / 1.000 dùng mẫu số toàn bộ người trong fold. Các FP tuyệt đối trong fold_metrics.csv có support từng fold, không so trực tiếp với toàn cohort.", "",
              "## Đóng góp từng bước", ""]
    for r in result["comparisons"]:
        interval = f"[{r['ci95_low']:.4f}; {r['ci95_high']:.4f}]" if r["ci95_low"] is not None else "không tính khi chỉ có một repeat"
        lines.append(f"- {r['model']} − {r['baseline']}: **{r['mean_difference']:+.4f}**, khoảng mô tả 95% theo repeat: {interval}.")
    lines += ["", "Training sets chồng lấn; khoảng trên chỉ mô tả, không xem folds/repeats là các nghiên cứu độc lập.",
              "Đóng góp của trọng số nhóm phải đọc D/E so với C, không quy toàn bộ chênh lệch so với A cho trọng số.", "",
              "## Temporal validation", "",
              "Train 2017–March 2020; tune chỉ trong chu kỳ cũ; test 2021–2023. Chu kỳ mới đã được xem trong nghiên cứu trước, không phải external validation độc lập.", "",
              "| Model | Macro-F1 | Accuracy | Recall 3 | Precision 3 |", "|---|---:|---:|---:|---:|"]
    for r in result["temporal"]:
        lines.append(f"| {r['model']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['recall_3']:.4f} | {r['precision_3']:.4f} |")
    lines += ["", "## Confusion matrix OOF và PR", "",
              "Mỗi người xuất hiện một lần mỗi repeat; không coi tổng support là số người độc lập.", ""]
    pr_rows, pooled, matched = [], [], []
    original_rows = [r for r in predictions if r['model'] == MODELS[0]]
    original_truth = np.array([int(r['true_target']) for r in original_rows])
    original_pred = np.array([int(r['predicted_target']) for r in original_rows])
    original_recall = [binary_metrics(original_truth // 2, original_pred // 2)['recall'],
                       binary_metrics(original_truth % 2, original_pred % 2)['recall']]
    for model in MODELS:
        part = [r for r in predictions if r["model"] == model]
        truth = np.array([int(r["true_target"]) for r in part])
        pred = np.array([int(r["predicted_target"]) for r in part])
        scores = [np.array([float(r[key]) for r in part]) for key in ("egfr_score", "uacr_score")]
        pooled.append({"model": model, **metrics_for(truth, pred, scores)})
        cm = np.bincount(4 * truth + pred, minlength=16).reshape(4, 4)
        lines += [f"### {model}", "", "```text", "True\\Pred     0      1      2      3"]
        lines += [str(i) + "       " + " ".join(f"{n:6d}" for n in row) for i, row in enumerate(cm)]
        lines += ["```", ""]
        # Curves are descriptive OOF diagnostics only, never fed into tuning.
        for axis, score in enumerate(scores):
            target = truth // 2 if axis == 0 else truth % 2
            # Diagnostic threshold at the original model's pooled recall. This
            # uses OOF outcomes for description only and is never deployed.
            order = np.argsort(-score, kind='stable')
            ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score)-1]
            tp = np.cumsum(target[order])[ends]
            recall = tp / max(int(target.sum()), 1)
            eligible = np.flatnonzero(recall >= original_recall[axis])
            index = eligible[0] if len(eligible) else len(ends)-1
            threshold = float(score[order][ends[index]])
            diagnostic = score >= threshold
            matched.append({'model': model, 'axis': axis, 'reference_recall': original_recall[axis],
                            'threshold_descriptive_only': threshold,
                            **binary_metrics(target, diagnostic),
                            'fp': int(np.sum(diagnostic & (target == 0))),
                            'support': len(target)})
            for threshold in np.linspace(0., 1., 101):
                p = score >= threshold
                pr_rows.append({"model": model, "axis": axis, "threshold": float(threshold),
                                **binary_metrics(target, p), "fp": int(np.sum(p & (target == 0))),
                                "fn": int(np.sum(~p & (target == 1))), "support": len(target)})
    _write_rows(out / "pooled_oof_metrics.csv", pooled)
    _write_rows(out / "precision_recall_thresholds.csv", pr_rows)
    _write_rows(out / "matched_recall_diagnostics.csv", matched)
    lines += ['## Precision tại recall tương đương W1', '',
              'Chọn điểm đầu tiên đạt ít nhất recall OOF gộp của A trên từng trục. Đây là phân tích mô tả sau đánh giá; threshold này không được dùng cho model cuối. Do score có ties, recall thực tế có thể vượt mức tham chiếu.', '',
              '| Model | Trục | Recall tham chiếu | Recall đạt được | Precision | FP / 1.000 người |',
              '|---|---|---:|---:|---:|---:|']
    for r in matched:
        lines.append(f"| {r['model']} | {('eGFR', 'uACR')[r['axis']]} | {r['reference_recall']:.4f} | {r['recall']:.4f} | {r['precision']:.4f} | {1000*r['fp']/r['support']:.1f} |")
    lines.append('')
    lines += ["## Kết luận và giới hạn", "",
              f"Cấu hình có macro-F1 outer trung bình cao nhất: **{result['best_model']}**. Đây là lựa chọn sau khi so sánh các phương án; cần dữ liệu mới để xác nhận độc lập.", "",
              "Tăng macro-F1 không tự chứng minh cải thiện lâm sàng. Cần đọc cùng precision/recall từng lớp, average precision từng trục và số false positives. Balanced scores không phải xác suất đã calibration.", "",
              "Giới hạn: target một lần đo không xác nhận CKD mạn tính; không suy luận nhân quả; chưa dùng survey design; nhóm biến là giả định thiết kế; tìm kiếm hữu hạn không bảo đảm tối ưu. Không dùng làm chẩn đoán.", "",
              "## Artifact", "",
              "- protocol.json: lưới, seed, candidates, hash code/data và phiên bản môi trường.",
              "- fold_metrics.csv / oof_predictions.csv: metrics và dự đoán từng fold.",
              "- weights_w2_by_fold.csv / weights_w2_summary.csv: trọng số, rank, tần suất top 5.",
              "- precision_recall_thresholds.csv: đường PR theo threshold, chỉ dùng mô tả.",
              "- matched_recall_diagnostics.csv: precision/FP tại recall tham chiếu W1, chỉ dùng mô tả.",
              "- pooled_oof_metrics.csv: precision/recall/F1, FP/FN gộp với support lặp.",
              "- temporal.csv / temporal_predictions.csv: kết quả chuyển chu kỳ.",
              "- final_parameters.json / final_weights.csv: tham số và trọng số fit toàn cohort.",
              f"- Model: `{result['model_path']}`; gọi `nhanes_pipeline.w2.predict(artifact, X)` với đúng thứ tự feature.",
              "- Checkpoint từng fold ghi completion marker cuối; resume chỉ khi protocol/data/code/môi trường khớp.", ""]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
