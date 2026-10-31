# PHASE 1 — KAGGLE CKD BASELINE (W0)

> This is an implementation specification for Kaggle Notebook and subsequent handoff to Codex in VS Code. Implement Phase 1 only. Do **not** train W1/W2/W3 in this phase.

## 0. Project objective and terminology

Research task: Chronic Kidney Disease (CKD) binary classification using K-Nearest Neighbors (KNN), with later improvements to feature weights and neighbor voting.

- **W0**: a **uniform, fixed initial feature-weight vector** (not learned). Each of the 24 *original features* has the same weight. This is the baseline. If the features are encoded into more columns, preserve equality **by original feature group** rather than allocating equal weight per one-hot column.
- **W1 (later in Codex)**: feature weights learned from training data, initially via Mutual Information (MI).
- **W2 (later in Codex)**: weights optimized with cross-validation, optionally using distance-weighted voting.
- **W3 (later in Codex)**: mixed-type / missingness-aware metric and cost-sensitive optimization.

**Scope**: use Kaggle to download/audit data, define a held-out test set, implement a leakage-safe W0 KNN baseline, measure its performance and export a reproducibility bundle. Codex must use the **same dataset, split, feature semantics, and test holdout**, but fit all preprocessing anew on each training fold when optimizing W1+.

## 1. Data source and provenance

- Official UCI CKD dataset: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease (dataset ID: `336`).
- Reference repo only (do not assume its preprocessed CSV is leakage-safe): https://github.com/Abdelkerim-Dassi/Chronic-Kidney-Disease-Prediction
- Expected source: approximately **400 records, 24 predictors, two labels (`ckd`, `notckd`)**. Validate actual numbers and field names; fail loudly on unexpected data instead of silently inventing data.
- Preferred path: in Kaggle Notebook, enable **Internet** if available, `pip install ucimlrepo` and `fetch_ucirepo(id=336)`; if Internet is disabled, download original data from UCI to your computer and upload it to Kaggle as an Input dataset. In the offline case, inspect file schema rather than assuming the CSV has the same shape.
- Keep the raw file unchanged. Record origin, fetch date, dataset shape, column names, class balance, missing-value counts and any parsing decisions.

## 2. Kaggle setup

1. Open https://www.kaggle.com/code → **New Notebook**.
2. Set language to Python; **CPU is enough** for 400 rows and KNN (GPU is not needed).
3. Ensure data access using the official UCI fetch route or Kaggle's **Add Input** upload flow.
4. Put final outputs under `/kaggle/working/ckd_w0_artifacts/` and choose **Save Version / Save & Run All** to retain them.
5. Fix `random_state=42`, and document the Python / pandas / scikit-learn / ucimlrepo versions.

## 3. Dataset inspection and cleaning rules

- Normalize leading/trailing whitespace and source representations of missing data (`?`, blank, whitespace plus `?`, tabs before `yes/no/ckd/notckd`).
- Remove a purely administrative `id` column if present, and never treat the target as an input.
- Target: map `ckd` → 1 and `notckd` → 0 **only after checking all unique values**. Stop on unmapped labels.
- Inspect duplicate rows; do not drop duplicates silently. Record a clear policy (e.g., de-duplicate exact records before splitting if justified).
- 24 original predictors should remain in scope. Separate numerical and categorical fields based on the data dictionary, **not** merely pandas `dtype` (many numeric columns may contain `?` strings).
- Suggested numerical original features: `age`, `bp`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc`.
- Suggested categorical/ordinal original features: `sg`, `al`, `su`, `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`. Confirm this grouping against the source schema. `sg`, `al`, `su` are ordinal-ish and can be modeled in sensitivity experiments later; **Phase 1** may one-hot-encode them to avoid arbitrarily assuming equal ordinal spacing.
- UCI and repository aliases may differ (e.g. `wc` vs `wbcc`, `rc` vs `rbcc`); create an explicitly documented mapping instead of silently dropping fields.
- Never impute using `class`, and never fit imputers/encoders/scalers on the whole dataset before data splitting.

## 4. Split discipline (before any fitting)

- Use stratified train/test split **70% / 30%**, `random_state=42`, with saved original row identifiers / indices.
- Hold out test data from all design choices and hyperparameter tuning. The test can be scored **once after baseline W0 is locked**; after that, do not tune future W1/W2/W3 to its results.
- On the training portion, perform `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` cross-validation to select `K` and (optionally) `p` using predeclared scoring `f1` for class CKD (positive = 1), tracking recall and precision as secondary metrics.
- Fit imputers, encoders and scalers **inside a scikit-learn Pipeline** for each fold. For W0 the baseline must use the same feature-group weighting rules later used by Codex.
- For exploratory notebooks, report cross-validation scores only on the training split. Avoid repeated peeking at final test results.

## 5. W0: exactly what is the baseline weight vector?

For the 24 **original** predictor groups:

\[
W_0=(1/24, 1/24, \ldots, 1/24), \quad \sum_j w_j=1.
\]

If original feature `dm` is one-hot encoded into two or more columns, do **not** assign 1/24 to every output column. Implement a deterministic group-aware Euclidean distance:

\[
d_{W_0}(x,z)=\sqrt{\sum_{j=1}^{24} w_j\,\delta_j(x,z)^2}.
\]

- For numeric columns (after median imputation and scaling), use the scalar difference.
- For categorical columns, use a mismatch indicator `δ_j(x,z) = 1` if values differ, else `0`, **or** use a one-hot group squared difference normalized consistently across all features. Choose and document one method, and never mix methods during comparisons.
- A simple consistent Phase 1 implementation can use a `ColumnTransformer` with numeric median+StandardScaler and categorical most_frequent+OneHotEncoder, then group-normalize the one-hot coordinates so **the squared distance contribution for a category mismatch equals 1**. Multiply every original feature-group contribution by `1/24`. This ensures fixed W0 and compatibility with future feature weighting.
- Do not claim these fixed weights were fitted or learned. Save as `weights_w0.csv` with 24 records and columns `feature,weight,source`, where `source="uniform_fixed"`.
- Also include a plain scikit-learn KNN control with standard preprocessing to show whether group-aware weighting changes the baseline metric. Distinguish `knn_plain` from `knn_w0_group_weighted` explicitly.

## 6. Required Phase 1 experiments

### E1. Data audit
Output class balance, missingness by feature, numeric summary, categorical counts, and a short notes document. Record how many rows remained.

### E2. Basic KNN (`knn_plain`)
- `Pipeline(preprocess, KNeighborsClassifier(...))`, with numerical median imputation + standard scaling, categorical most-frequent imputation + one-hot encoding.
- Tune `n_neighbors` on **training CV**, e.g. `[1, 3, 5, 7, 9, 11, 15]`; try uniform votes, Euclidean distance. Record CV mean/std and best K.

### E3. Group-aware W0 KNN (`knn_w0_group_weighted`)
- Use the documented mixed-type per-feature distance with 24 equal original-feature weights and nearest-neighbor majority vote.
- Choose K on training CV with exactly the same folds and search values as E2.
- Check tie handling, zero distances, and missing-data behaviors.

### E4. Locked holdout evaluation
- Once E2/E3 settings are selected, evaluate final locked models on test exactly once.
- Report Accuracy, precision for CKD, recall/sensitivity for CKD, F1 for CKD, specificity for non-CKD, ROC-AUC **only if score outputs are valid**, and confusion matrix (TP, TN, FP, FN).
- Emphasize false negatives (CKD → not CKD) and uncertainty from small N.
- Do not promise or fabricate any accuracy threshold.

## 7. Suggested notebook structure

`01_kaggle_ckd_w0.ipynb` cells:

1. Introduction and reproducibility configuration.
2. Dependencies/imports/version prints.
3. Download/load original UCI data (offline path fallback).
4. Validate source schema and target labels.
5. Normalize missing values and audit.
6. Stratified 70/30 split and export immutable indices.
7. Preprocessing in pipelines (fit train folds only).
8. W0 definition + verification of sum/nonnegativity + feature-group map.
9. Implement/test reference KNN and W0 group-aware distance.
10. Cross-validation for K selection on train only.
11. Lock hyperparameters and evaluate test once.
12. Export artifacts / manifest and print a compact handoff summary.

## 8. Files Kaggle MUST export

All under `/kaggle/working/ckd_w0_artifacts/`:

```
ckd_w0_artifacts/
├── dataset_manifest.json         # source URL, rows/columns, SHA-256 or source identifier
├── environment.json              # versions and seed
├── data_dictionary.csv           # original 24 features, type, allowed levels, handling
├── data_audit.csv                # missing counts, dtypes, descriptive summary
├── split_indices.json            # train/test original record IDs, fixed seed
├── feature_group_mapping.json    # original feature to transformed output columns
├── weights_w0.csv               # 24 original-feature equal weights
├── cv_results.csv               # training CV results for each baseline candidate
├── baseline_metrics.json        # locked holdout test scores and confusion matrix
├── predictions_test.csv         # original row ID, true label, predictions, optional scores
├── model_w0.joblib              # fitted pipeline/model, if serializable
├── requirements_freeze.txt      # reproducible environment
└── HANDOFF_TO_CODEX.md          # next steps and assumptions
```

Save the original raw data separately or include a verified way of downloading it again. A model artifact MUST be accompanied by its preprocessing, metadata, and exact feature order. If a custom Python class is needed, export its `.py` module as well (otherwise a joblib pickle may not load outside the notebook).

## 9. Kaggle-to-Codex handoff protocol

1. Download the output files from the **saved Kaggle Notebook version**.
2. Copy them to `artifacts/kaggle_w0/` inside the VS Code repository; keep these files read-only as the reference baseline.
3. Add original UCI data under `data/raw/` or a `download_data.py` script with checksum verification.
4. Codex must first verify that raw-data checksum, split IDs, feature dictionary, W0 weights, and reproduced CV/test metrics match the manifest. If a version mismatch is present, flag it rather than silently overwriting results.
5. Only then begin **Phase 2**: calculate W1 from fold-specific training data using Mutual Information; improve it to W2 through CV-based optimization of feature weights and neighbor voting. **Do not select W1/W2 using Kaggle's locked test results.**
6. In **Phase 3**, research per-feature or grouped weights with mixed distances, missingness-aware representations and asymmetric error costs. Clearly separate hypotheses from measured outcomes.
7. Keep a new final external dataset, if available, for true independent evaluation; repeated evaluation on Kaggle's holdout after seeing its outcomes will compromise its status as an unseen test.

## 10. Unit tests / acceptance conditions

- `assert len(weights_w0) == 24`
- `assert all(weight >= 0)` and `sum(weights)==1` within floating-point tolerance.
- Re-run same fixed split seed yields same split indices.
- Train and test indices are disjoint, and class balance is reported.
- No fit or transformation statistics are learned on test samples.
- Original categorical mismatch contributes one group's weight, regardless of the number of one-hot columns.
- The K nearest neighbor search gives the same result as a brute-force reference on a toy fixture.
- A manual tiny dataset produces expected predictions for K=1 and K=3.
- Missing or unexpected source columns generate explicit validation errors.
- All exported metrics can be reproduced from `predictions_test.csv`.
- The report distinguishes *KNN baseline performance* from *the fixed W0 weights* and makes no clinical diagnostic claims.

## 11. Kaggle AI / Codex-ready instruction

Copy this instruction to Kaggle Code Assistant or Codex, alongside this Markdown:

> Read `KAGGLE_PHASE1_W0_BASELINE.md` completely. Implement only the Kaggle Phase 1 notebook `01_kaggle_ckd_w0.ipynb`. Use UCI Chronic Kidney Disease dataset ID 336, inspect actual schema, split stratified 70/30 before fitting, create a leakage-safe KNN baseline and a group-aware uniform W0 (24 original features), tune K by CV within training only, evaluate the locked test once, export all specified artifacts to `/kaggle/working/ckd_w0_artifacts/`. Add explicit unit tests and do not fabricate results or produce W1/W2/W3 yet. Make every cell runnable in order on Kaggle CPU.

## 12. Important research cautions

- **W0 is fixed, not learned**; it does not require a GPU and there is no notion of an optimal W0 training run.
- **The weights do not carry clinical meaning** without further evidence.
- **Never use test outcomes to improve W1/W2/W3.** If the goal includes later research, consider nested CV for reporting generalization, plus new external data for true deployment claims.
- **One-hot categorical columns should not inadvertently count as multiple original attributes.** Consistent weight mapping is essential to fair comparisons.
- **Do not directly copy Project B's reported 97–99% results as validated performance.** Recompute actual results under the specified leakage-safe evaluation design.
- This is an educational classification project, **not** a clinical diagnosis tool.
