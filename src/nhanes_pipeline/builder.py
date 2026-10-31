"""Build canonical and modeling NHANES tables from the 13 source modules."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

from .xport import Variable, read_xport


MODULES = (
    "DEMO_L", "BIOPRO_L", "ALB_CR_L", "BMX_L", "BPXO_L", "BPQ_L",
    "DIQ_L", "SMQ_L", "ALQ_L", "PAQ_L", "TCHOL_L", "HDL_L", "CBC_L",
)

FEATURES = (
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR",
    "BMXBMI", "BMXWAIST", "sbp_mean", "dbp_mean", "BPQ020", "BPQ150",
    "DIQ010", "DIQ160", "current_smoking", "ALQ111", "ALQ121", "ALQ130",
    "PAD680", "moderate_freq_week", "BPQ080", "LBXTC", "LBDHDD",
    "LBXSUA", "LBXHGB",
)

CATEGORICAL_LEVELS = {
    "RIAGENDR": (1, 2), "RIDRETH3": (1, 2, 3, 4, 6, 7),
    "DMDEDUC2": (1, 2, 3, 4, 5), "BPQ020": (1, 2), "BPQ150": (1, 2),
    "DIQ010": (1, 2, 3), "DIQ160": (1, 2), "current_smoking": (0, 1),
    "ALQ111": (1, 2), "ALQ121": tuple(range(11)), "BPQ080": (1, 2),
}

LABEL_SOURCE_COLUMNS = ("LBXSCR", "URXUMA", "URXUCR", "egfr_2021_ckdepi", "uacr_mg_g")
SURVEY_COLUMNS = ("WTMEC2YR", "WTINT2YR", "SDMVSTRA", "SDMVPSU", "WTPH2YR")
MODELING_COLUMNS = (
    "SEQN", "cycle", "historical_split", "eligible_ckd_pattern",
    *SURVEY_COLUMNS, *FEATURES,
    "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3",
    "SMQ020", "SMQ040", "PAD790Q", "PAD790U",
    *LABEL_SOURCE_COLUMNS, "target",
)


def _present(value: object) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value)) and value != ""


def _number(value: object) -> float | None:
    return float(value) if _present(value) else None


def _valid_code(value: object, allowed: Iterable[int]) -> int | None:
    if not _present(value):
        return None
    integer = int(float(value))
    return integer if integer in allowed else None


def _mean_available(row: dict[str, object], columns: Iterable[str]) -> float | None:
    values = [_number(row.get(column)) for column in columns]
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_historical_split(handoff_dir: Path) -> dict[int, str]:
    split: dict[int, str] = {}
    for filename, label in (("train_data.csv", "train"), ("holdout_locked.csv", "seen_holdout")):
        with (handoff_dir / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                seqn = int(float(row["SEQN"]))
                if seqn in split:
                    raise ValueError(f"SEQN {seqn} occurs in both historical splits")
                split[seqn] = label
    return split


def _merge_modules(data_dir: Path) -> tuple[dict[int, dict[str, object]], dict[str, list[Variable]], dict[str, int]]:
    master: dict[int, dict[str, object]] = {}
    dictionaries: dict[str, list[Variable]] = {}
    row_counts: dict[str, int] = {}
    seen_names: set[str] = {"SEQN"}
    for module in MODULES:
        variables, rows = read_xport(data_dir / f"{module}.xpt")
        dictionaries[module] = variables
        row_counts[module] = len(rows)
        if len({int(float(row["SEQN"])) for row in rows}) != len(rows):
            raise ValueError(f"Duplicate SEQN in {module}")
        if module == "DEMO_L":
            for source in rows:
                seqn = int(float(source["SEQN"]))
                master[seqn] = {"SEQN": seqn}
        for source in rows:
            seqn = int(float(source["SEQN"]))
            if seqn not in master:
                raise ValueError(f"{module} contains SEQN {seqn} absent from DEMO_L")
            destination = master[seqn]
            destination[f"has_{module}"] = 1
            for variable in variables:
                if variable.name == "SEQN":
                    continue
                name = variable.name
                if name in seen_names or name in destination:
                    name = f"{name}__{module}"
                destination[name] = source[variable.name]
        seen_names.update(variable.name for variable in variables if variable.name != "SEQN")
    for row in master.values():
        for module in MODULES:
            row.setdefault(f"has_{module}", 0)
    return master, dictionaries, row_counts


def _derive_modeling_row(raw: dict[str, object], split: str) -> dict[str, object] | None:
    age = _number(raw.get("RIDAGEYR"))
    creatinine = _number(raw.get("LBXSCR"))
    urine_albumin = _number(raw.get("URXUMA"))
    urine_creatinine = _number(raw.get("URXUCR"))
    if age is None or age < 20 or None in (creatinine, urine_albumin, urine_creatinine) or urine_creatinine <= 0:
        return None

    gender = _valid_code(raw.get("RIAGENDR"), (1, 2))
    if gender is None:
        return None
    female = gender == 2
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = creatinine / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.2 * 0.9938**age
    if female:
        egfr *= 1.012
    uacr = urine_albumin / urine_creatinine * 100
    target = int(uacr >= 30) + 2 * int(egfr < 60)

    bpq020 = _valid_code(raw.get("BPQ020"), (1, 2))
    bpq150 = _valid_code(raw.get("BPQ150"), (1, 2))
    if bpq150 is None and bpq020 == 2:
        bpq150 = 2
    alq111 = _valid_code(raw.get("ALQ111"), (1, 2))
    alq121 = _valid_code(raw.get("ALQ121"), range(11))
    alq121_inferred_zero = False
    if alq121 is None and alq111 == 2:
        alq121 = 0
        alq121_inferred_zero = True
    alq130 = _number(raw.get("ALQ130"))
    if alq130 in (777, 999):
        alq130 = None
    if alq130 is None and alq121_inferred_zero:
        alq130 = 0.0

    smq020 = _valid_code(raw.get("SMQ020"), (1, 2))
    smq040 = _valid_code(raw.get("SMQ040"), (1, 2, 3))
    current_smoking: int | None = None
    if smq020 == 2 or (smq020 == 1 and smq040 == 3):
        current_smoking = 0
    elif smq020 == 1 and smq040 in (1, 2):
        current_smoking = 1

    moderate_q = _number(raw.get("PAD790Q"))
    moderate_unit = raw.get("PAD790U") if _present(raw.get("PAD790U")) else None
    factors = {"D": 7.0, "W": 1.0, "M": 3.0 / 13.0, "Y": 1.0 / 52.0}
    moderate_week = moderate_q * factors[moderate_unit] if moderate_q is not None and moderate_unit in factors else None

    row: dict[str, object] = {
        "SEQN": int(raw["SEQN"]), "cycle": "2021-2023", "historical_split": split,
        "eligible_ckd_pattern": 1,
        "RIDAGEYR": age, "RIAGENDR": gender,
        "RIDRETH3": _valid_code(raw.get("RIDRETH3"), (1, 2, 3, 4, 6, 7)),
        "DMDEDUC2": _valid_code(raw.get("DMDEDUC2"), (1, 2, 3, 4, 5)),
        "INDFMPIR": _number(raw.get("INDFMPIR")), "BMXBMI": _number(raw.get("BMXBMI")),
        "BMXWAIST": _number(raw.get("BMXWAIST")),
        "sbp_mean": _mean_available(raw, ("BPXOSY1", "BPXOSY2", "BPXOSY3")),
        "dbp_mean": _mean_available(raw, ("BPXODI1", "BPXODI2", "BPXODI3")),
        "BPQ020": bpq020, "BPQ150": bpq150,
        "DIQ010": _valid_code(raw.get("DIQ010"), (1, 2, 3)),
        "DIQ160": _valid_code(raw.get("DIQ160"), (1, 2)),
        "current_smoking": current_smoking, "ALQ111": alq111, "ALQ121": alq121,
        "ALQ130": alq130,
        "PAD680": None if _number(raw.get("PAD680")) in (7777, 9999) else _number(raw.get("PAD680")),
        "moderate_freq_week": moderate_week,
        "BPQ080": _valid_code(raw.get("BPQ080"), (1, 2)),
        "LBXTC": _number(raw.get("LBXTC")), "LBDHDD": _number(raw.get("LBDHDD")),
        "LBXSUA": _number(raw.get("LBXSUA")), "LBXHGB": _number(raw.get("LBXHGB")),
        "SMQ020": smq020, "SMQ040": smq040, "PAD790Q": moderate_q, "PAD790U": moderate_unit,
        "LBXSCR": creatinine, "URXUMA": urine_albumin, "URXUCR": urine_creatinine,
        "egfr_2021_ckdepi": egfr, "uacr_mg_g": uacr, "target": target,
    }
    for name in SURVEY_COLUMNS:
        row[name] = _number(raw.get(name))
    for name in ("BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"):
        row[name] = _number(raw.get(name))
    return row


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if not _present(row.get(key)) else row.get(key) for key in columns})


def _dictionary_rows(dictionaries: dict[str, list[Variable]]) -> list[dict[str, object]]:
    source = {}
    labels = {}
    for module, variables in dictionaries.items():
        for variable in variables:
            source.setdefault(variable.name, module)
            labels.setdefault(variable.name, variable.label)
    rows = []
    administrative = {"SEQN", "cycle", "historical_split", "eligible_ckd_pattern"}
    for name in MODELING_COLUMNS:
        if name in FEATURES:
            role = "model_feature"
        elif name in LABEL_SOURCE_COLUMNS or name == "target":
            role = "label_source" if name != "target" else "target"
        elif name in SURVEY_COLUMNS:
            role = "survey_design"
        elif name in administrative:
            role = "administrative"
        else:
            role = "feature_source"
        allowed = "|".join(map(str, CATEGORICAL_LEVELS[name])) if name in CATEGORICAL_LEVELS else ""
        rows.append({
            "column": name, "role": role, "model_input": int(name in FEATURES),
            "source_module": source.get(name.split("__")[0], "derived"),
            "source_label": labels.get(name.split("__")[0], "derived/administrative"),
            "data_type": "categorical" if name in CATEGORICAL_LEVELS else "numeric_or_text",
            "allowed_levels": allowed,
            "missing_rule": "special/refusal codes -> missing; imputation is fold-local" if name in FEATURES else "not imputed in canonical table",
        })
    return rows


def build_all(root: str | Path) -> dict[str, object]:
    root = Path(root)
    data_dir = root / "data"
    processed = data_dir / "processed"
    reports = root / "reports"
    handoff_dir = root / "nhanes_w0_handoff"
    master, dictionaries, raw_counts = _merge_modules(data_dir)
    historical_split = _read_historical_split(handoff_dir)

    modeling = []
    for seqn in sorted(master):
        row = _derive_modeling_row(master[seqn], historical_split.get(seqn, ""))
        if row is not None:
            if not row["historical_split"]:
                raise ValueError(f"Eligible SEQN {seqn} missing from historical handoff")
            modeling.append(row)
    if set(historical_split) != {int(row["SEQN"]) for row in modeling}:
        raise ValueError("Raw-derived eligible IDs differ from the historical handoff")

    master_columns = ["SEQN"] + sorted({key for row in master.values() for key in row if key != "SEQN"})
    master_path = processed / "nhanes_master_by_seqn.csv"
    modeling_path = processed / "nhanes_modeling_complete.csv"
    dictionary_path = processed / "nhanes_data_dictionary.csv"
    _write_csv(master_path, [master[key] for key in sorted(master)], master_columns)
    _write_csv(modeling_path, modeling, MODELING_COLUMNS)
    dictionary_rows = _dictionary_rows(dictionaries)
    _write_csv(dictionary_path, dictionary_rows, list(dictionary_rows[0]))

    target_counts = {str(target): sum(row["target"] == target for row in modeling) for target in range(4)}
    split_counts = {name: sum(row["historical_split"] == name for row in modeling) for name in ("train", "seen_holdout")}
    source_files = {}
    for module in MODULES:
        path = data_dir / f"{module}.xpt"
        source_files[path.name] = {"rows": raw_counts[module], "sha256": _sha256(path), "bytes": path.stat().st_size}
    outputs = {
        str(path.relative_to(root)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in (master_path, modeling_path, dictionary_path)
    }
    manifest = {
        "dataset": "NHANES August 2021-August 2023 CKD measurement-pattern study",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "1.0.0",
        "join_key": "SEQN", "join_strategy": "DEMO_L anchored left joins",
        "master_rows": len(master), "modeling_rows": len(modeling),
        "feature_count": len(FEATURES), "features": list(FEATURES),
        "target_counts": target_counts, "historical_split_counts": split_counts,
        "target_definition": "2021 CKD-EPI eGFR threshold 60 and uACR threshold 30 mg/g",
        "historical_holdout_status": "seen; exploratory/history only; excluded from W1 selection",
        "zero_normalization": {
            "rule": "XPORT mathematical zero is emitted as exact 0; legacy 5.397605346934028e-79 is rejected",
            "legacy_handoff_cells_detected": {"ALQ121": 728, "INDFMPIR": 41, "PAD680": 3, "total": 772},
        },
        "source_files": source_files, "outputs": outputs,
    }
    manifest_path = processed / "nhanes_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    missingness = {
        feature: sum(not _present(row.get(feature)) for row in modeling) for feature in FEATURES
    }
    audit = {
        "master_rows": len(master), "modeling_rows": len(modeling), "target_counts": target_counts,
        "split_counts": split_counts, "missing_counts": missingness,
        "seqn_unique_master": len({row["SEQN"] for row in master.values()}),
        "seqn_unique_modeling": len({row["SEQN"] for row in modeling}),
    }
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "nhanes_data_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return manifest

