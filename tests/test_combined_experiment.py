from __future__ import annotations

import numpy as np
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.combined_experiment import (  # noqa: E402
    binary_scores,
    four_class_metrics,
    stratified_folds,
)


class TestCombinedExperimentCore(unittest.TestCase):
    def test_stratified_folds_partition_every_row(self) -> None:
        y = np.asarray([0] * 20 + [1] * 10 + [2] * 5 + [3] * 5)
        folds = stratified_folds(y, 5, 42)
        observed = np.concatenate([valid for _, valid in folds])
        self.assertEqual(sorted(observed.tolist()), list(range(len(y))))
        self.assertEqual(len(set(observed.tolist())), len(y))

    def test_balanced_binary_vote_changes_prior_effect(self) -> None:
        distances = np.asarray([[1.0, 1.0, 1.0]])
        indices = np.asarray([[0, 1, 2]])
        y = np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        ordinary = binary_scores(distances, indices, y, 3, False)[0]
        balanced = binary_scores(distances, indices, y, 3, True)[0]
        self.assertLess(ordinary, 0.5)
        self.assertGreater(balanced, 0.5)

    def test_four_class_metrics_perfect(self) -> None:
        y = np.asarray([0, 1, 2, 3])
        metrics = four_class_metrics(y, y)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
