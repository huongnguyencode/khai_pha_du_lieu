from __future__ import annotations

import csv
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.combined_builder import (  # noqa: E402
    ALBUMIN_FORWARD_INTERCEPT,
    ALBUMIN_FORWARD_SLOPE,
    PRIMARY_FEATURES,
    _old_bp_medication,
)


class TestCombinedBuilderRules(unittest.TestCase):
    def test_primary_schema_excludes_non_equivalent_activity_frequency(self) -> None:
        self.assertEqual(len(PRIMARY_FEATURES), 23)
        self.assertNotIn("moderate_freq_week", PRIMARY_FEATURES)

    def test_albumin_forward_bridge(self) -> None:
        self.assertAlmostEqual(ALBUMIN_FORWARD_INTERCEPT + ALBUMIN_FORWARD_SLOPE * 10, 9.788)

    def test_legacy_bp_medication_skip_rules(self) -> None:
        self.assertEqual(_old_bp_medication({"BPQ050A": 1}, 1), 1)
        self.assertEqual(_old_bp_medication({"BPQ050A": 2}, 1), 2)
        self.assertEqual(_old_bp_medication({}, 2), 2)
        self.assertEqual(_old_bp_medication({"BPQ040A": 2}, 1), 2)
        self.assertIsNone(_old_bp_medication({"BPQ040A": 1}, 1))


if __name__ == "__main__":
    unittest.main()
