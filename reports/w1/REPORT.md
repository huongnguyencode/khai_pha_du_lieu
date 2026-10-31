# NHANES W1 — group Mutual Information weights

All preprocessing, MI estimation, alpha selection, and evaluation used only the 3,740-row historical training partition. `seen_holdout` was not used.

| model | macro_f1_mean | macro_f1_std | balanced_accuracy_mean | accuracy_mean | selected_alpha_values |
|---|---|---|---|---|---|
| M4_MI_feature_distance_vote | 0.3437584024585584 | 0.02345921386145907 | 0.3283685860225791 | 0.8058885119596354 | 0.5|0.25|1|0.25|0.25 |
| M2_uniform_feature_distance_vote | 0.34168381847132007 | 0.038397941652754716 | 0.3249731803274861 | 0.8064186200355948 | 0|0|0|0|0 |
| M3_MI_feature_uniform_vote | 0.31270859510738813 | 0.024129547582162875 | 0.29953818872358506 | 0.814442167595628 | 1|1|1|1|0.25 |
| M1_uniform_feature_uniform_vote | 0.3031966425940197 | 0.024585769829084145 | 0.293368332405911 | 0.8131074130047484 | 0|0|0|0|0 |

Best mean configuration: `M4_MI_feature_distance_vote`; full-train CV alpha: `1.0`.

Paired outer-fold M4−M2 macro-F1 difference: `0.002075`; 95% CI `[-0.032418, 0.036567]`. The interval crosses zero, so W1 improvement is not confirmed.
