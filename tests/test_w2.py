from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from nhanes_pipeline import w2
from nhanes_pipeline.builder import CATEGORICAL_LEVELS
from nhanes_pipeline.combined_builder import PRIMARY_CATEGORICAL, PRIMARY_FEATURES
from nhanes_pipeline.combined_experiment import four_class_metrics, shrink, stratified_folds


class TestW2Study(unittest.TestCase):
    def test_identity_preserves_w1_and_group_weights_form_simplex(self):
        mi = np.arange(1, 24, dtype=float)
        mi /= mi.sum()
        for alpha in (0., .5, 1.):
            np.testing.assert_allclose(w2.weights(mi, alpha, np.zeros(7)), shrink(mi, alpha))
            for candidate in w2.candidates():
                for theta in candidate:
                    weight = w2.weights(mi, alpha, theta)
                    self.assertTrue(np.all(weight >= 0))
                    self.assertAlmostEqual(weight.sum(), 1.)

    def test_candidate_search_is_reproducible_and_penalties_vary(self):
        first, second = w2.candidates(), w2.candidates()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), len({tuple(t.ravel()) for t in first}))
        penalties = {round(float(np.sum(t**2)), 8) for t in first[1:]}
        self.assertGreater(len(penalties), 2)
        np.testing.assert_array_equal(first[0], np.zeros((2, 7)))

    def test_joint_threshold_search_matches_exhaustive_four_class_metric(self):
        y = np.array([0, 1, 2, 3] * 8)
        rng = np.random.default_rng(123)
        scores = [rng.random(len(y)), rng.random(len(y))]
        value, params = w2.threshold_pair(scores, y, [(1., 15, .5)] * 2)
        expected = max(four_class_metrics(y, (scores[1] >= ta).astype(int) +
                                         2 * (scores[0] >= tg).astype(int))['macro_f1']
                       for tg in w2.THRESHOLDS for ta in w2.THRESHOLDS)
        self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(value, w2.joint_score(y, scores[0] >= params[0][2], scores[1] >= params[1][2]))

    def test_inner_selection_separates_train_and_validation(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(48, len(PRIMARY_FEATURES)))
        for name in PRIMARY_CATEGORICAL:
            x[:, PRIMARY_FEATURES.index(name)] = rng.choice(CATEGORICAL_LEVELS[name], len(x))
        x[:, 0] = np.arange(len(x))  # Stable row marker, never an actual model feature in production.
        y = np.array([0, 1, 2, 3] * 12)
        folds = stratified_folds(y, 3, 73)
        expected_train = [set(x[tr, 0]) for tr, _ in folds]
        mi_calls, score_calls = [], []

        def fake_mi(train, target, prep, seed):
            ids = set(train[:, 0])
            self.assertIn(ids, expected_train)
            self.assertEqual(len(train), len(target))
            mi_calls.append(ids)
            values = np.full(len(PRIMARY_FEATURES), 1 / len(PRIMARY_FEATURES))
            return values, values

        def fake_scores(train, valid, target, prep, mi, axis, alpha, theta, ks):
            train_ids, valid_ids = set(train[:, 0]), set(valid[:, 0])
            self.assertFalse(train_ids & valid_ids)
            self.assertEqual(train_ids | valid_ids, set(x[:, 0]))
            score_calls.append((train_ids, valid_ids))
            scores = (valid[:, 0] % (5 + axis)) / (5 + axis)
            return {k: scores for k in ks}

        with patch.object(w2, 'estimate_weights', side_effect=fake_mi), \
             patch.object(w2, 'scores_for', side_effect=fake_scores), \
             patch.object(w2, 'candidates', return_value=[np.zeros((2, 7))]):
            settings, history = w2.select(x, y, 73, 3)
        self.assertEqual(len(mi_calls), 6)
        self.assertTrue(score_calls)
        self.assertEqual(set(settings), set(w2.MODELS))
        self.assertGreaterEqual(history[1]['macro_f1'], history[0]['macro_f1'])
        self.assertGreaterEqual(history[2]['macro_f1'], history[1]['macro_f1'])

    def test_false_positive_metrics_distinguish_axes_and_four_patterns(self):
        truth = np.array([0, 0, 1, 2, 3])
        pred = np.array([1, 3, 3, 2, 0])
        scores = [pred // 2, pred % 2]
        metric = w2.metrics_for(truth, pred, scores)
        self.assertEqual(metric['healthy_fp'], 2)
        self.assertEqual(metric['egfr_fp'], 2)
        self.assertEqual(metric['uacr_fp'], 2)
        self.assertEqual(metric['healthy_fp_per_1000'], 400)


if __name__ == '__main__':
    unittest.main()
