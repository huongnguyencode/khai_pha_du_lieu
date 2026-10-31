from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhanes_pipeline.builder import CATEGORICAL_LEVELS, FEATURES, build_all  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(text: str) -> float | None:
    return None if text == "" else float(text)


class TestNhanesDatabase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        build_all(ROOT)
        cls.modeling = read_csv(ROOT / "data/processed/nhanes_modeling_complete.csv")
        cls.master = read_csv(ROOT / "data/processed/nhanes_master_by_seqn.csv")

    def test_row_counts_and_unique_seqn(self) -> None:
        self.assertEqual(len(self.master), 11933)
        self.assertEqual(len(self.modeling), 5343)
        self.assertEqual(len({row["SEQN"] for row in self.master}), 11933)
        self.assertEqual(len({row["SEQN"] for row in self.modeling}), 5343)

    def test_historical_split_is_preserved(self) -> None:
        train = {row["SEQN"] for row in self.modeling if row["historical_split"] == "train"}
        holdout = {row["SEQN"] for row in self.modeling if row["historical_split"] == "seen_holdout"}
        self.assertEqual(len(train), 3740)
        self.assertEqual(len(holdout), 1603)
        self.assertFalse(train & holdout)

        expected_train = {str(int(float(row["SEQN"]))) for row in read_csv(ROOT / "nhanes_w0_handoff/train_data.csv")}
        expected_holdout = {str(int(float(row["SEQN"]))) for row in read_csv(ROOT / "nhanes_w0_handoff/holdout_locked.csv")}
        self.assertEqual(train, expected_train)
        self.assertEqual(holdout, expected_holdout)

    def test_target_counts_and_formula(self) -> None:
        counts = {target: 0 for target in range(4)}
        for row in self.modeling:
            target = int(row["target"])
            counts[target] += 1
            creatinine = float(row["LBXSCR"])
            age = float(row["RIDAGEYR"])
            female = int(float(row["RIAGENDR"])) == 2
            kappa = 0.7 if female else 0.9
            alpha = -0.241 if female else -0.302
            ratio = creatinine / kappa
            egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.2 * 0.9938**age
            if female:
                egfr *= 1.012
            uacr = float(row["URXUMA"]) / float(row["URXUCR"]) * 100
            expected = int(uacr >= 30) + 2 * int(egfr < 60)
            self.assertEqual(target, expected)
            self.assertAlmostEqual(float(row["egfr_2021_ckdepi"]), egfr, places=10)
            self.assertAlmostEqual(float(row["uacr_mg_g"]), uacr, places=10)
        self.assertEqual(counts, {0: 4418, 1: 471, 2: 311, 3: 143})

    def test_features_match_legacy_except_zero_normalization(self) -> None:
        canonical = {int(float(row["SEQN"])): row for row in self.modeling}
        for path in (ROOT / "nhanes_w0_handoff/train_data.csv", ROOT / "nhanes_w0_handoff/holdout_locked.csv"):
            for legacy in read_csv(path):
                current = canonical[int(float(legacy["SEQN"]))]
                for feature in FEATURES:
                    legacy_value = value(legacy[feature])
                    if legacy_value is not None and abs(legacy_value) < 1e-70:
                        legacy_value = 0.0
                    current_value = value(current[feature])
                    self.assertEqual(current_value is None, legacy_value is None, (legacy["SEQN"], feature))
                    if current_value is not None:
                        self.assertAlmostEqual(current_value, legacy_value, places=10, msg=(legacy["SEQN"], feature))

    def test_derived_feature_rules(self) -> None:
        for row in self.modeling:
            systolic = [value(row[name]) for name in ("BPXOSY1", "BPXOSY2", "BPXOSY3")]
            systolic = [item for item in systolic if item is not None]
            expected_sbp = sum(systolic) / len(systolic) if systolic else None
            self.assertEqual(value(row["sbp_mean"]) is None, expected_sbp is None)
            if expected_sbp is not None:
                self.assertAlmostEqual(float(row["sbp_mean"]), expected_sbp)

            q = value(row["PAD790Q"])
            unit = row["PAD790U"] or None
            factors = {"D": 7, "W": 1, "M": 3 / 13, "Y": 1 / 52}
            expected = q * factors[unit] if q is not None and unit in factors else None
            self.assertEqual(value(row["moderate_freq_week"]) is None, expected is None)
            if expected is not None:
                self.assertAlmostEqual(float(row["moderate_freq_week"]), expected)

    def test_categorical_levels_and_zero_fix(self) -> None:
        forbidden = "5.397605346934028e-79"
        for row in self.modeling:
            self.assertNotIn(forbidden, row.values())
            for feature, allowed in CATEGORICAL_LEVELS.items():
                if row[feature] != "":
                    self.assertIn(int(float(row[feature])), allowed, (row["SEQN"], feature))

    def test_manifest_and_dictionary(self) -> None:
        manifest_path = ROOT / "data/processed/nhanes_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["master_rows"], 11933)
        self.assertEqual(manifest["modeling_rows"], 5343)
        self.assertEqual(len(manifest["source_files"]), 13)
        self.assertTrue(all(len(item["sha256"]) == 64 for item in manifest["source_files"].values()))
        dictionary = read_csv(ROOT / "data/processed/nhanes_data_dictionary.csv")
        model_features = [row["column"] for row in dictionary if row["model_input"] == "1"]
        self.assertEqual(model_features, list(FEATURES))


if __name__ == "__main__":
    unittest.main()

