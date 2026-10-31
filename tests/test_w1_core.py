from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.builder import FEATURES  # noqa: E402
from nhanes_pipeline.w1 import group_distances, predict_from_distances, shrink_weights  # noqa: E402


class TestW1Core(unittest.TestCase):
    def test_shrinkage_endpoints_and_sum(self) -> None:
        mi = np.arange(1, len(FEATURES) + 1, dtype=float)
        mi /= mi.sum()
        uniform = shrink_weights(mi, 0)
        learned = shrink_weights(mi, 1)
        np.testing.assert_allclose(uniform, np.full(len(FEATURES), 1 / len(FEATURES)))
        np.testing.assert_allclose(learned, mi)
        self.assertAlmostEqual(float(shrink_weights(mi, 0.5).sum()), 1.0)

    def test_categorical_mismatch_contributes_one_group_weight(self) -> None:
        weights = np.zeros(len(FEATURES))
        weights[FEATURES.index("RIAGENDR")] = 1.0
        train_num = np.zeros((1, 13))
        query_num = np.zeros((1, 13))
        train_cat = np.zeros((1, 11))
        query_cat = np.zeros((1, 11))
        query_cat[0, 0] = 1
        distance = group_distances(train_num, train_cat, query_num, query_cat, weights)
        self.assertAlmostEqual(float(distance[0, 0]), 1.0)

    def test_distance_vote_and_zero_distance(self) -> None:
        distances = np.asarray([[0.0, 0.2, 0.3], [0.3, 0.2, 0.1]])
        labels = np.asarray([2, 0, 1])
        prediction = predict_from_distances(distances, labels, 3, "distance")
        self.assertEqual(prediction.tolist(), [2, 1])


if __name__ == "__main__":
    unittest.main()

