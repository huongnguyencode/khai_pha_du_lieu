# NHANES group-weighted KNN

This workspace builds a canonical NHANES 2021–2023 database and evaluates group-level Mutual Information feature weights for four-class KNN. It is a research pipeline, not a clinical diagnostic system.

## Docker training

Build and run the complete data audit, tests, nested CV, and train-only model fit:

```bash
docker compose build
docker compose run --rm nhanes-train
```

If Docker Buildx cannot write its default user configuration, use a writable temporary config:

```bash
mkdir -p /tmp/nhanes-docker-config
DOCKER_CONFIG=/tmp/nhanes-docker-config docker build -t nhanes-weighted-knn:local .
docker run --rm \
  -v "$PWD/data/processed:/workspace/data/processed" \
  -v "$PWD/reports:/workspace/reports" \
  -v "$PWD/models:/workspace/models" \
  nhanes-weighted-knn:local
```

Outputs are written to:

- `data/processed/`: master/modeling tables, dictionary, manifest and checksums.
- `reports/w1/`: nested-CV folds, OOF predictions, weights and ablation report.
- `models/nhanes_w1_group_knn.pkl`: train-only fitted research artifact.

The `seen_holdout` partition is never used to select MI weights, alpha, vote mode, or the final train-only artifact.

## Harmonized 2017–2023 experiment

The combined primary analysis uses 13,086 adults with observed serum creatinine
and urine albumin/creatinine across NHANES 2017–March 2020 and 2021–2023. The
older urine-albumin values are bridged to the 2021–2023 LC-MS/MS scale using the
CDC equation. The primary shared schema has 23 features; the non-equivalent
physical-activity frequency item is excluded.

After downloading the 13 official `P_*.xpt` files to a temporary directory:

```bash
PYTHONPATH=src python3 scripts/03_build_combined_cohort.py \
  --old-data-dir /tmp/nhanes_2017_2020

PYTHONPATH=src python3 scripts/04_train_combined_w1.py \
  --repeats 5 --outer-splits 5 --inner-splits 3

PYTHONPATH=src python3 scripts/05_export_combined_weights.py
PYTHONPATH=src python3 scripts/06_finalize_combined_report.py
```

The experiment compares flat four-class KNN with dual-axis eGFR/uACR KNN,
MI feature weights, class-balanced voting, and missing-indicator ablations.
Outputs are stored in `reports/combined_w1/`; the final research artifact is
`models/nhanes_combined_dual_axis.pkl`.

## W2 controlled group-weight optimization

Run the five paired ablations on the same combined cohort and outer folds:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 python3 scripts/07_train_w2.py \
  --repeats 5 --outer 5 --inner 3 --jobs 4
```

A reproduces the original W1 balanced model. B chooses the two thresholds
jointly for four-class macro-F1. C also searches alpha and a larger K grid.
D/E add a fixed budget of sparse group-weight adjustments, with and without
regularization toward C. All configurations keep the original MI formula,
balanced voting, 23 features, and labels. Group search retunes K/thresholds
while retaining C's alpha; it does not claim a global optimum.

Results, paired comparisons, temporal validation, OOF precision/recall
diagnostics, feature-weight stability, and fold checkpoints are written to
`reports/w2/`. The trained artifact is `models/nhanes_w2.pkl`. Interrupted
runs resume only if code/data/environment/protocol hashes agree. Use
`--output reports/w2_separate_protocol` for a different configuration;
that run's model is saved in its output directory.

For inference, load a trusted pickle and call `nhanes_pipeline.w2.predict`:

```python
import pickle
from nhanes_pipeline.w2 import predict

with open('models/nhanes_w2.pkl', 'rb') as stream:
    artifact = pickle.load(stream)
# X must have 23 columns in artifact['features'] order; NaNs are supported.
predicted_patterns, balanced_scores = predict(artifact, X)
```

Balanced scores are not calibrated clinical probabilities. Targets describe
single-measurement patterns; the previously studied temporal cohort does not
provide independent external validation. Read `reports/w2/REPORT.md` for the
actual results and limits of the comparison.

Validate OOF partitions and historical W1 reproduction, then build the
Vietnamese handoff:

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/08_validate_w2.py
python3 scripts/09_finalize_w2_report.py
```

Optional figures require Matplotlib (`python3 scripts/10_plot_w2.py`); they
export standalone PNG/SVG macro-F1 and precision-recall charts. The training
environment is recorded in `reports/w2/protocol.json`; resuming in a different
environment requires a separate output directory.

