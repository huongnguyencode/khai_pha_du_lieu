# Cải thiện W2 — NHANES 2017–2023

## Thiết kế

Cohort 13.086 người, 23 biến, target bốn pattern eGFR/uACR. Outer 5 folds × 5 repeats; inner 3 folds. Cùng outer splits W1. Mọi preprocessing, MI và chọn tham số nằm trong train folds.

- A: tái lập W1 balanced, chọn F1 riêng từng trục, lưới gốc.
- B: giữ alpha/K của A; chọn cặp threshold theo macro-F1 bốn lớp.
- C: thêm tìm alpha/K theo mục tiêu bốn lớp; K gồm 3,5,7,11,15,21,31,51.
- D: 24 perturbations nhóm cố định và identity; alpha giữ từ C, K/threshold được tune lại.
- E: cùng ứng viên D, objective = inner macro-F1 − 0,01 × sum(theta²), regularize về C.
- Giữ nguyên balanced voting và công thức MI W1; không thêm uniform floor, missing indicator hay đổi cohort.
- B duyệt toàn bộ cặp threshold trên lưới. C/D/E dùng hai lượt coordinate search, không bảo đảm tối ưu toàn cục.

## Kết quả nested CV

| Model | Macro-F1 ± SD | Accuracy | Precision 1 | Precision 2 | Precision 3 | Recall 3 | FP lớp 0 / 1.000 người |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_W1_original | 0.4113 ± 0.0157 | 0.7258 | 0.2169 | 0.2176 | 0.2144 | 0.5626 | 154.3 |
| B_joint_thresholds | 0.4317 ± 0.0194 | 0.7782 | 0.2568 | 0.2677 | 0.2939 | 0.4291 | 92.6 |
| C_joint_K_alpha | 0.4456 ± 0.0142 | 0.7982 | 0.2834 | 0.3211 | 0.3197 | 0.4121 | 71.0 |
| D_W2_groups | 0.4462 ± 0.0122 | 0.7955 | 0.2732 | 0.3269 | 0.3235 | 0.4106 | 74.3 |
| E_W2_groups_regularized | 0.4446 ± 0.0136 | 0.7981 | 0.2796 | 0.3221 | 0.3209 | 0.4058 | 70.8 |

SD tính trên outer folds, không phải sai số chuẩn. FP lớp 0 / 1.000 dùng mẫu số toàn bộ người trong fold. Các FP tuyệt đối trong fold_metrics.csv có support từng fold, không so trực tiếp với toàn cohort.

## Đóng góp từng bước

- B_joint_thresholds − A_W1_original: **+0.0204**, khoảng mô tả 95% theo repeat: [0.0150; 0.0259].
- C_joint_K_alpha − B_joint_thresholds: **+0.0139**, khoảng mô tả 95% theo repeat: [0.0072; 0.0205].
- D_W2_groups − C_joint_K_alpha: **+0.0006**, khoảng mô tả 95% theo repeat: [-0.0045; 0.0058].
- E_W2_groups_regularized − C_joint_K_alpha: **-0.0010**, khoảng mô tả 95% theo repeat: [-0.0042; 0.0022].
- E_W2_groups_regularized − A_W1_original: **+0.0333**, khoảng mô tả 95% theo repeat: [0.0263; 0.0403].

Training sets chồng lấn; khoảng trên chỉ mô tả, không xem folds/repeats là các nghiên cứu độc lập.
Đóng góp của trọng số nhóm phải đọc D/E so với C, không quy toàn bộ chênh lệch so với A cho trọng số.

## Temporal validation

Train 2017–March 2020; tune chỉ trong chu kỳ cũ; test 2021–2023. Chu kỳ mới đã được xem trong nghiên cứu trước, không phải external validation độc lập.

| Model | Macro-F1 | Accuracy | Recall 3 | Precision 3 |
|---|---:|---:|---:|---:|
| A_W1_original | 0.4048 | 0.7084 | 0.5315 | 0.1910 |
| B_joint_thresholds | 0.4335 | 0.8033 | 0.3217 | 0.2949 |
| C_joint_K_alpha | 0.4409 | 0.8067 | 0.3636 | 0.3514 |
| D_W2_groups | 0.4366 | 0.8065 | 0.3427 | 0.3311 |
| E_W2_groups_regularized | 0.4409 | 0.8067 | 0.3636 | 0.3514 |

## Confusion matrix OOF và PR

Mỗi người xuất hiện một lần mỗi repeat; không coi tổng support là số người độc lập.

### A_W1_original

```text
True\Pred     0      1      2      3
0        43836   5201   2836   2057
1         3272   1561    472    835
2          999    367   1025   1074
3          282    203    344   1066
```

### B_joint_thresholds

```text
True\Pred     0      1      2      3
0        47869   3297   1862    902
1         4054   1291    376    419
2         1509    349    944    663
3          462    241    379    813
```

### C_joint_K_alpha

```text
True\Pred     0      1      2      3
0        49284   2507   1427    712
1         4284   1180    295    381
2         1616    275    981    593
3          508    242    364    781
```

### D_W2_groups

```text
True\Pred     0      1      2      3
0        49070   2807   1344    709
1         4236   1250    287    367
2         1644    291    953    577
3          497    267    353    778
```

### E_W2_groups_regularized

```text
True\Pred     0      1      2      3
0        49298   2563   1370    699
1         4292   1197    285    366
2         1638    288    954    585
3          509    257    360    769
```

## Precision tại recall tương đương W1

Chọn điểm đầu tiên đạt ít nhất recall OOF gộp của A trên từng trục. Đây là phân tích mô tả sau đánh giá; threshold này không được dùng cho model cuối. Do score có ties, recall thực tế có thể vượt mức tham chiếu.

| Model | Trục | Recall tham chiếu | Recall đạt được | Precision | FP / 1.000 người |
|---|---|---:|---:|---:|---:|
| A_W1_original | eGFR | 0.6547 | 0.6547 | 0.3614 | 94.8 |
| A_W1_original | uACR | 0.4561 | 0.4561 | 0.2947 | 134.1 |
| B_joint_thresholds | eGFR | 0.6547 | 0.6547 | 0.3614 | 94.8 |
| B_joint_thresholds | uACR | 0.4561 | 0.4561 | 0.2947 | 134.1 |
| C_joint_K_alpha | eGFR | 0.6547 | 0.6547 | 0.3961 | 81.8 |
| C_joint_K_alpha | uACR | 0.4561 | 0.4561 | 0.3262 | 115.7 |
| D_W2_groups | eGFR | 0.6547 | 0.6547 | 0.3961 | 81.8 |
| D_W2_groups | uACR | 0.4561 | 0.4561 | 0.3232 | 117.3 |
| E_W2_groups_regularized | eGFR | 0.6547 | 0.6547 | 0.3960 | 81.8 |
| E_W2_groups_regularized | uACR | 0.4561 | 0.4561 | 0.3268 | 115.4 |

## Kết luận và giới hạn

Cấu hình có macro-F1 outer trung bình cao nhất: **D_W2_groups**. Đây là lựa chọn sau khi so sánh các phương án; cần dữ liệu mới để xác nhận độc lập.

Tăng macro-F1 không tự chứng minh cải thiện lâm sàng. Cần đọc cùng precision/recall từng lớp, average precision từng trục và số false positives. Balanced scores không phải xác suất đã calibration.

Giới hạn: target một lần đo không xác nhận CKD mạn tính; không suy luận nhân quả; chưa dùng survey design; nhóm biến là giả định thiết kế; tìm kiếm hữu hạn không bảo đảm tối ưu. Không dùng làm chẩn đoán.

## Artifact

- protocol.json: lưới, seed, candidates, hash code/data và phiên bản môi trường.
- fold_metrics.csv / oof_predictions.csv: metrics và dự đoán từng fold.
- weights_w2_by_fold.csv / weights_w2_summary.csv: trọng số, rank, tần suất top 5.
- precision_recall_thresholds.csv: đường PR theo threshold, chỉ dùng mô tả.
- matched_recall_diagnostics.csv: precision/FP tại recall tham chiếu W1, chỉ dùng mô tả.
- pooled_oof_metrics.csv: precision/recall/F1, FP/FN gộp với support lặp.
- temporal.csv / temporal_predictions.csv: kết quả chuyển chu kỳ.
- final_parameters.json / final_weights.csv: tham số và trọng số fit toàn cohort.
- Model: `/home/huongcode/khai_pha/khai_pha_du_lieu/models/nhanes_w2.pkl`; gọi `nhanes_pipeline.w2.predict(artifact, X)` với đúng thứ tự feature.
- Checkpoint từng fold ghi completion marker cuối; resume chỉ khi protocol/data/code/môi trường khớp.
