# NHANES 2017–2023 — W1 hai trục eGFR/uACR

## Dữ liệu và protocol

- Cohort: **13.086** người từ 20 tuổi, có đủ nguồn tạo eGFR và uACR.
- Chu kỳ 2017–March 2020: 7.743; August 2021–August 2023: 5.343.
- Target counts: lớp 0 = 10.786, lớp 1 = 1.228, lớp 2 = 693, lớp 3 = 379.
- Primary schema: 23 feature tương thích; loại `moderate_freq_week` vì công cụ đo khác nhau giữa chu kỳ.
- Albumin 2017–2020 được bridge sang thang LC-MS/MS 2021–2023 theo công thức CDC trước khi tạo target.
- Đánh giá: repeated nested CV, 5 folds × 5 repeats; inner CV 3 folds.
- Imputation, scaling, MI, alpha, K và threshold đều được fit trong inner-train/outer-train tương ứng.

## Repeated nested-CV

| Model | Macro-F1 | Balanced accuracy | Accuracy | Recall 0 | Recall 1 | Recall 2 | Recall 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dual_mi_balanced | 0.4113 ± 0.0157 | 0.4814 | 0.7258 | 0.8128 | 0.2543 | 0.2959 | 0.5626 |
| dual_mi_balanced_missing | 0.4112 ± 0.0137 | 0.4807 | 0.7252 | 0.8119 | 0.2591 | 0.2924 | 0.5594 |
| dual_mi_distance | 0.3684 ± 0.0151 | 0.3444 | 0.8047 | 0.9475 | 0.1070 | 0.1815 | 0.1414 |
| flat_mi_distance | 0.3528 ± 0.0165 | 0.3358 | 0.8033 | 0.9506 | 0.0757 | 0.1443 | 0.1726 |
| dual_uniform_distance | 0.3364 ± 0.0153 | 0.3187 | 0.8044 | 0.9542 | 0.0839 | 0.1302 | 0.1066 |
| flat_uniform_distance | 0.3337 ± 0.0116 | 0.3184 | 0.8044 | 0.9554 | 0.0761 | 0.1186 | 0.1235 |

Best model: `dual_mi_balanced`.

Chênh lệch macro-F1 so với `flat_mi_distance`: **0.0585**, descriptive 95% CI theo 5 repeat **[0.0502, 0.0668]**. Các repeat vẫn dùng training set chồng lấn nên CI không phải kiểm định độc lập tuyệt đối.

## Temporal validation

Train trên 2017–March 2020, test nguyên chu kỳ 2021–2023:

| Model | Macro-F1 | Balanced accuracy | Accuracy | Recall 0 | Recall 1 | Recall 2 | Recall 3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat_uniform_distance | 0.3150 | 0.3004 | 0.8196 | 0.9762 | 0.0552 | 0.0932 | 0.0769 |
| flat_mi_distance | 0.3437 | 0.3221 | 0.8186 | 0.9717 | 0.0594 | 0.0965 | 0.1608 |
| dual_mi_distance | 0.3510 | 0.3284 | 0.8055 | 0.9502 | 0.0955 | 0.1350 | 0.1329 |
| dual_mi_balanced | 0.4048 | 0.4770 | 0.7084 | 0.7868 | 0.3100 | 0.2797 | 0.5315 |
| dual_mi_balanced_missing | 0.4081 | 0.4719 | 0.7163 | 0.7974 | 0.2951 | 0.3055 | 0.4895 |

## Trọng số cuối

### Trục egfr

| Feature | Weight |
|---|---:|
| RIDAGEYR | 0.261843 |
| BPQ150 | 0.116928 |
| LBXSUA | 0.107316 |
| BPQ020 | 0.101955 |
| LBXHGB | 0.057222 |
| DIQ010 | 0.050059 |
| sbp_mean | 0.042963 |
| BMXWAIST | 0.042818 |

### Trục uacr

| Feature | Weight |
|---|---:|
| DIQ010 | 0.200764 |
| sbp_mean | 0.137763 |
| BPQ150 | 0.135500 |
| BPQ020 | 0.135483 |
| RIDAGEYR | 0.128979 |
| ALQ121 | 0.034161 |
| DMDEDUC2 | 0.032911 |
| BPQ080 | 0.031777 |

## Diễn giải giới hạn

Class-balanced voting tăng mạnh recall lớp thiểu số nhưng làm giảm accuracy do tăng false positives. Missing indicators không cải thiện macro-F1 so với cùng model không indicator. Trọng số là association phục vụ khoảng cách, không phải quan hệ nhân quả; tuổi còn tham gia công thức eGFR. Target dựa trên một lần đo khảo sát và không xác nhận CKD mạn tính.
