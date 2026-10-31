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

