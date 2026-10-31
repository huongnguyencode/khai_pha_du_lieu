"""Build a harmonized 2017-March 2020 + 2021-2023 NHANES cohort.

The older urinary-albumin measurements are moved onto the 2021-2023
LC-MS/MS scale with the CDC forward bridging equation before uACR and the
four-class target are calculated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from .builder import CATEGORICAL_LEVELS, FEATURES, _derive_modeling_row, _merge_modules
from .xport import read_xport


PRIMARY_FEATURES = tuple(feature for feature in FEATURES if feature != "moderate_freq_week")
PRIMARY_CATEGORICAL = tuple(feature for feature in PRIMARY_FEATURES if feature in CATEGORICAL_LEVELS)
OLD_MODULES = (
    "P_DEMO", "P_BIOPRO", "P_ALB_CR", "P_BMX", "P_BPXO", "P_BPQ",
    "P_DIQ", "P_SMQ", "P_ALQ", "P_PAQ", "P_TCHOL", "P_HDL", "P_CBC",
)
ALBUMIN_FORWARD_INTERCEPT = 1.381
ALBUMIN_FORWARD_SLOPE = 0.8407


def _present(value: object) -> bool:
    return value is not None and value != "" and not (isinstance(value, float) and math.isnan(value))


def _number(value: object) -> float | None:
    return float(value) if _present(value) else None


def _valid(value: object, allowed: Iterable[int]) -> int | None:
    if not _present(value):
        return None
    result = int(float(value))
    return result if result in allowed else None


def _mean(row: dict[str, object], names: Iterable[str]) -> float | None:
    values = [_number(row.get(name)) for name in names]
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_old_modules(data_dir: Path) -> tuple[dict[int, dict[str, object]], dict[str, int]]:
    master: dict[int, dict[str, object]] = {}
    counts: dict[str, int] = {}
    seen_names: set[str] = {"SEQN"}
    for module in OLD_MODULES:
        variables, rows = read_xport(data_dir / f"{module}.xpt")
        counts[module] = len(rows)
        ids = [int(float(row["SEQN"])) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate SEQN in {module}")
        if module == "P_DEMO":
            for seqn in ids:
                master[seqn] = {"SEQN": seqn}
        for source in rows:
            seqn = int(float(source["SEQN"]))
            if seqn not in master:
                raise ValueError(f"{module} SEQN absent from P_DEMO: {seqn}")
            destination = master[seqn]
            for variable in variables:
                if variable.name == "SEQN":
                    continue
                name = variable.name
                if name in seen_names or name in destination:
                    name = f"{name}__{module}"
                destination[name] = source[variable.name]
        seen_names.update(variable.name for variable in variables if variable.name != "SEQN")
    return master, counts


def _old_bp_medication(row: dict[str, object], bpq020: int | None) -> int | None:
    """Harmonize legacy BPQ050A to the later BPQ150 meaning."""
    current = _valid(row.get("BPQ050A"), (1, 2))
    ever_prescribed = _valid(row.get("BPQ040A"), (1, 2))
    if current is not None:
        return current
    if bpq020 == 2 or ever_prescribed == 2:
        return 2
    return None


def _derive_old(row: dict[str, object]) -> dict[str, object] | None:
    age = _number(row.get("RIDAGEYR"))
    sex = _valid(row.get("RIAGENDR"), (1, 2))
    creatinine = _number(row.get("LBXSCR"))
    urine_albumin_old = _number(row.get("URXUMA"))
    urine_creatinine = _number(row.get("URXUCR"))
    if age is None or age < 20 or sex is None:
        return None
    if None in (creatinine, urine_albumin_old, urine_creatinine) or urine_creatinine <= 0:
        return None

    urine_albumin = ALBUMIN_FORWARD_INTERCEPT + ALBUMIN_FORWARD_SLOPE * urine_albumin_old
    female = sex == 2
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = creatinine / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.2 * 0.9938**age
    if female:
        egfr *= 1.012
    uacr = urine_albumin / urine_creatinine * 100
    egfr_low = int(egfr < 60)
    uacr_high = int(uacr >= 30)

    bpq020 = _valid(row.get("BPQ020"), (1, 2))
    alq111 = _valid(row.get("ALQ111"), (1, 2))
    alq121 = _valid(row.get("ALQ121"), range(11))
    inferred_no_alcohol = alq121 is None and alq111 == 2
    if inferred_no_alcohol:
        alq121 = 0
    alq130 = _number(row.get("ALQ130"))
    if alq130 in (777, 999):
        alq130 = None
    if alq130 is None and inferred_no_alcohol:
        alq130 = 0.0

    smq020 = _valid(row.get("SMQ020"), (1, 2))
    smq040 = _valid(row.get("SMQ040"), (1, 2, 3))
    smoking = None
    if smq020 == 2 or (smq020 == 1 and smq040 == 3):
        smoking = 0
    elif smq020 == 1 and smq040 in (1, 2):
        smoking = 1

    pad680 = _number(row.get("PAD680"))
    if pad680 in (7777, 9999):
        pad680 = None
    result: dict[str, object] = {
        "row_id": f"2017-2020:{int(row['SEQN'])}", "SEQN": int(row["SEQN"]),
        "cycle": "2017-2020", "RIDAGEYR": age, "RIAGENDR": sex,
        "RIDRETH3": _valid(row.get("RIDRETH3"), (1, 2, 3, 4, 6, 7)),
        "DMDEDUC2": _valid(row.get("DMDEDUC2"), (1, 2, 3, 4, 5)),
        "INDFMPIR": _number(row.get("INDFMPIR")), "BMXBMI": _number(row.get("BMXBMI")),
        "BMXWAIST": _number(row.get("BMXWAIST")),
        "sbp_mean": _mean(row, ("BPXOSY1", "BPXOSY2", "BPXOSY3")),
        "dbp_mean": _mean(row, ("BPXODI1", "BPXODI2", "BPXODI3")),
        "BPQ020": bpq020, "BPQ150": _old_bp_medication(row, bpq020),
        "DIQ010": _valid(row.get("DIQ010"), (1, 2, 3)),
        "DIQ160": _valid(row.get("DIQ160"), (1, 2)), "current_smoking": smoking,
        "ALQ111": alq111, "ALQ121": alq121, "ALQ130": alq130, "PAD680": pad680,
        "BPQ080": _valid(row.get("BPQ080"), (1, 2)), "LBXTC": _number(row.get("LBXTC")),
        "LBDHDD": _number(row.get("LBDHDD")), "LBXSUA": _number(row.get("LBXSUA")),
        "LBXHGB": _number(row.get("LBXHGB")), "LBXSCR": creatinine,
        "URXUMA_original": urine_albumin_old, "URXUMA_harmonized": urine_albumin,
        "URXUCR": urine_creatinine, "egfr_2021_ckdepi": egfr, "uacr_mg_g": uacr,
        "egfr_low": egfr_low, "uacr_high": uacr_high,
        "target": uacr_high + 2 * egfr_low, "survey_exam_weight": _number(row.get("WTMECPRP")),
        "SDMVSTRA": _number(row.get("SDMVSTRA")), "SDMVPSU": _number(row.get("SDMVPSU")),
    }
    return result


def _read_current(path: Path) -> list[dict[str, object]]:
    output = []
    with path.open(newline="", encoding="utf-8") as stream:
        for source in csv.DictReader(stream):
            row: dict[str, object] = {
                "row_id": f"2021-2023:{int(float(source['SEQN']))}",
                "SEQN": int(float(source["SEQN"])), "cycle": "2021-2023",
            }
            for feature in PRIMARY_FEATURES:
                row[feature] = _number(source.get(feature))
            for name in ("LBXSCR", "URXUCR", "egfr_2021_ckdepi", "uacr_mg_g", "SDMVSTRA", "SDMVPSU"):
                row[name] = _number(source.get(name))
            albumin = _number(source.get("URXUMA"))
            row["URXUMA_original"] = albumin
            row["URXUMA_harmonized"] = albumin
            row["target"] = int(float(source["target"]))
            row["uacr_high"] = int(row["target"]) % 2
            row["egfr_low"] = int(row["target"]) // 2
            row["survey_exam_weight"] = _number(source.get("WTMEC2YR"))
            output.append(row)
    return output


def build_combined_cohort(root: str | Path, old_data_dir: str | Path) -> dict[str, object]:
    root, old_data_dir = Path(root), Path(old_data_dir)
    old_master, raw_counts = _load_old_modules(old_data_dir)
    old_rows = [derived for seqn in sorted(old_master) if (derived := _derive_old(old_master[seqn])) is not None]
    current_rows = _read_current(root / "data/processed/nhanes_modeling_complete.csv")
    rows = old_rows + current_rows
    if len(old_rows) != 7743 or len(current_rows) != 5343 or len(rows) != 13086:
        raise ValueError(f"Combined cohort invariant failed: {len(old_rows)}, {len(current_rows)}")
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate combined row_id")

    columns = (
        "row_id", "SEQN", "cycle", "survey_exam_weight", "SDMVSTRA", "SDMVPSU",
        *PRIMARY_FEATURES, "LBXSCR", "URXUMA_original", "URXUMA_harmonized", "URXUCR",
        "egfr_2021_ckdepi", "uacr_mg_g", "egfr_low", "uacr_high", "target",
    )
    output_dir = root / "data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "nhanes_combined_2017_2023.csv"
    with table_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if not _present(row.get(name)) else row.get(name) for name in columns})

    cycle_target_counts = {
        cycle: {str(label): sum(row["cycle"] == cycle and row["target"] == label for row in rows) for label in range(4)}
        for cycle in ("2017-2020", "2021-2023")
    }
    missing = {
        cycle: {feature: sum(row["cycle"] == cycle and not _present(row.get(feature)) for row in rows) for feature in PRIMARY_FEATURES}
        for cycle in ("2017-2020", "2021-2023")
    }
    manifest = {
        "dataset": "Harmonized NHANES 2017-March 2020 and August 2021-August 2023",
        "rows": len(rows), "features": list(PRIMARY_FEATURES), "feature_count": len(PRIMARY_FEATURES),
        "cycle_rows": {"2017-2020": len(old_rows), "2021-2023": len(current_rows)},
        "cycle_target_counts": cycle_target_counts,
        "albumin_harmonization": {
            "applied_to": "2017-2020 URXUMA",
            "common_scale": "2021-2023 LC-MS/MS",
            "formula": "URXUMA_harmonized = 1.381 + 0.8407 * URXUMA_original",
            "source": "CDC ALB_CR_L analytic notes",
        },
        "bp_medication_harmonization": "BPQ050A/BPQ040A/BPQ020 -> BPQ150-compatible current-use variable",
        "excluded_primary_feature": "moderate_freq_week (measurement instrument not equivalent across cycles)",
        "raw_old_files": {
            f"{module}.xpt": {"rows": raw_counts[module], "sha256": _sha256(old_data_dir / f"{module}.xpt")}
            for module in OLD_MODULES
        },
        "missing_counts_by_cycle": missing,
        "output": {"path": str(table_path.relative_to(root)), "sha256": _sha256(table_path)},
    }
    manifest_path = output_dir / "nhanes_combined_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
