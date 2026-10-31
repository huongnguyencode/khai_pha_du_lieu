"""Validate W2 artifacts, OOF partitions, and exact historical W1 reproduction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nhanes_pipeline.w2 import MODELS, metrics_for, predict
from nhanes_pipeline.combined_experiment import load_combined, stratified_folds


def validate(output):
    output = Path(output)
    result = json.loads((output / 'results.json').read_text())
    protocol = result['protocol']
    data_path = ROOT / 'data/processed/nhanes_combined_2017_2023.csv'
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == protocol['data_hash']
    _, y, _, _, cycles, ids = load_combined(data_path)
    expected = dict(zip(ids, zip(y, cycles)))
    rows = list(csv.DictReader((output / 'oof_predictions.csv').open()))
    assert len(rows) == len(y) * protocol['repeats'] * len(MODELS)
    groups = defaultdict(list)
    for row in rows:
        truth, cycle = expected[row['row_id']]
        assert int(row['true_target']) == truth and row['cycle'] == cycle
        groups[row['model'], int(row['repeat']), int(row['outer_fold'])].append(row)
    expected_group_count = len(MODELS) * protocol['repeats'] * protocol['outer']
    assert len(groups) == expected_group_count
    metrics = list(csv.DictReader((output / 'fold_metrics.csv').open()))
    assert len(metrics) == expected_group_count
    index = {(r['model'], int(r['repeat']), int(r['outer_fold'])): r for r in metrics}
    for repeat in range(protocol['repeats']):
        for fold, (_, valid) in enumerate(stratified_folds(y, protocol['outer'], 42 + repeat * 100)):
            valid_ids = {ids[i] for i in valid}
            for model in MODELS:
                key = model, repeat, fold
                part = groups[key]
                assert len(part) == len(valid_ids)
                assert {r['row_id'] for r in part} == valid_ids
                truth = np.array([int(r['true_target']) for r in part])
                pred = np.array([int(r['predicted_target']) for r in part])
                scores = [np.array([float(r[k]) for r in part]) for k in ('egfr_score', 'uacr_score')]
                for name, value in metrics_for(truth, pred, scores).items():
                    assert abs(float(index[key][name]) - value) < 1e-12, (key, name)
    weight_groups = defaultdict(float)
    for row in csv.DictReader((output / 'weights_w2_by_fold.csv').open()):
        key = row['model'], row['repeat'], row['outer_fold'], row['axis']
        assert float(row['weight']) >= 0
        weight_groups[key] += float(row['weight'])
    assert len(weight_groups) == expected_group_count * 2
    assert all(abs(total - 1.) < 1e-12 for total in weight_groups.values())

    historical_difference = None
    temporal_difference = None
    if protocol['outer'] == 5 and protocol['inner'] == 3:
        historical = [r for r in csv.DictReader((ROOT / 'reports/combined_w1/nested_cv_fold_metrics.csv').open())
                      if r['model'] == 'dual_mi_balanced' and int(r['repeat']) < protocol['repeats']]
        assert len(historical) == protocol['repeats'] * protocol['outer']
        differences = []
        for old in historical:
            new = index[MODELS[0], int(old['repeat']), int(old['outer_fold'])]
            for metric in ('macro_f1', 'accuracy', 'balanced_accuracy', 'recall_0', 'recall_1', 'recall_2', 'recall_3'):
                differences.append(abs(float(new[metric]) - float(old[metric])))
        historical_difference = max(differences)
        assert historical_difference < 1e-10, historical_difference
        old_temporal = next(r for r in csv.DictReader((ROOT / 'reports/combined_w1/temporal_validation.csv').open())
                            if r['model'] == 'dual_mi_balanced')
        new_temporal = next(r for r in result['temporal'] if r['model'] == MODELS[0])
        temporal_difference = max(abs(float(old_temporal[name])-new_temporal[name])
                                  for name in ('macro_f1', 'accuracy', 'balanced_accuracy'))
        assert temporal_difference < 1e-10, temporal_difference

    with Path(result['model_path']).open('rb') as stream:
        artifact = pickle.load(stream)
    q = artifact['train_matrix'][:7].copy()
    q[0, :] = np.nan  # All-missing input still follows stored preprocessing.
    pred, scores = predict(artifact, q)
    assert pred.shape == (len(q),) and set(pred) <= {0, 1, 2, 3}
    assert all(np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all() for s in scores)
    answer = {'status': 'passed', 'fold_model_evaluations': len(metrics),
              'oof_prediction_rows': len(rows), 'unique_people': len(y),
              'historical_w1_max_metric_difference': historical_difference,
              'historical_temporal_max_metric_difference': temporal_difference,
              'artifact_predict_with_missing_values': 'passed'}
    (output / 'validation.json').write_text(json.dumps(answer, indent=2) + '\n')
    print(json.dumps(answer, indent=2))
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/w2')
    validate(parser.parse_args().output)
