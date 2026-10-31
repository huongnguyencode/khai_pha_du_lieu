# NHANES 2021–2023 data layout

- The XPT files in this directory are immutable source modules.
- `processed/nhanes_master_by_seqn.csv` is the DEMO-anchored, one-row-per-SEQN master table.
- `processed/nhanes_modeling_complete.csv` is the 5,343-row analysis table. It contains the 24 model features, raw inputs used for derived features, label-source fields, survey design fields, and the historical split marker.
- `processed/nhanes_data_dictionary.csv` defines column roles. Only rows with `model_input=1` may enter the predictor matrix.
- `processed/nhanes_manifest.json` contains source/output checksums and row-count invariants.

`seen_holdout` has already been evaluated in W0. It must not be used to select W1 weights, hyperparameters, missingness rules, or features.

Rebuild from the workspace root:

```bash
PYTHONPATH=src python3 scripts/01_build_nhanes_database.py
python3 -m unittest discover -s tests -v
```

