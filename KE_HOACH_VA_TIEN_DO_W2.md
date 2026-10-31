# Kế hoạch và tiến độ nâng cấp W2 — NHANES 2017–2023

> **ĐÃ HOÀN TẤT W2** — cập nhật 2026-10-31T21:11:37+07:00. Đã tiếp tục từ 12 checkpoints, hoàn thành 25/25 folds, temporal validation, fit model cuối, kiểm chứng, báo cáo và biểu đồ. Giữ nguyên code, dữ liệu và protocol của lượt chạy trước.

## Trạng thái hiện tại

Không còn bước W2 dang dở. Không cần chạy lại huấn luyện. Lệnh resume đã sử dụng:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/07_train_w2.py --repeats 5 --outer 5 --inner 3 --jobs 4
```

Sau đó đã chạy thành công lần lượt scripts 08, 09 và 10 theo kế hoạch. Môi trường: Python 3.12.3, NumPy 1.26.4, SciPy 1.11.4; Matplotlib 3.6.3.

## 1. Mục tiêu

Giữ cohort 13.086 người, 23 feature, target bốn pattern eGFR/uACR. Đánh giá đóng góp riêng của threshold, K/alpha và trọng số nhóm bằng cùng outer splits W1. Không dùng kết quả outer/test để chọn tham số trong từng fold.

## 2. Các bước và trạng thái

- [x] Đọc tài liệu W1, code và kết quả đã lưu.
- [x] Phân tích giới hạn W1: precision thiểu số thấp; K và threshold thường chạm biên lưới; mục tiêu chọn F1 từng trục chưa khớp macro-F1 bốn lớp.
- [x] Triển khai A: tái lập W1 balanced nguyên bản.
- [x] Triển khai B: giữ alpha/K của A, chọn cặp threshold theo macro-F1 bốn lớp và mở rộng lưới threshold.
- [x] Triển khai C: chọn alpha/K theo mục tiêu bốn lớp, thêm K = 21,31,51.
- [x] Triển khai D: tìm trọng số bảy nhóm bằng 24 ứng viên sparse cố định và identity; giữ alpha của C, tune lại K/threshold.
- [x] Triển khai E: cùng ứng viên D, phạt 0,01 × sum(theta²) để regularize về C.
- [x] Sửa penalty để có mức phạt khác nhau theo độ lớn/số nhóm điều chỉnh.
- [x] Giữ nguyên MI W1 và balanced voting; không thêm uniform floor, không đổi schema/nhãn.
- [x] Chạy 21 unit tests: tất cả pass trong Python cục bộ.
- [x] Thêm checkpoint từng fold và kiểm tra hash code/data/protocol/môi trường khi resume.
- [x] Hoàn tất repeated nested CV 5 folds × 5 repeats, inner 3 folds. Đã lưu đủ 25/25 fold checkpoints trong lượt chạy tiếp.
- [x] Hoàn tất temporal validation: train 2017–2020, test 2021–2023.
- [x] Chọn tham số và fit artifact cuối trên toàn cohort.
- [x] Chạy kiểm chứng đầy đủ bằng scripts/08_validate_w2.py.
- [x] Đọc kết quả để phân biệt cải thiện do threshold/K với cải thiện do trọng số nhóm.
- [x] Xuất tài liệu kết quả bằng scripts/09_finalize_w2_report.py.
- [x] Xuất biểu đồ PR và macro-F1 bằng scripts/10_plot_w2.py.

## 3. File đã triển khai

| File | Công dụng |
|---|---|
| src/nhanes_pipeline/w2.py | Tìm tham số trong inner CV, A–E, checkpoint, inference, sinh bảng/báo cáo thô |
| scripts/07_train_w2.py | Chạy huấn luyện; hỗ trợ --jobs và --output |
| scripts/08_validate_w2.py | Kiểm tra OOF, tính lại metrics, so A với W1 đã lưu, kiểm tra inference |
| scripts/09_finalize_w2_report.py | Tạo CAI_THIEN_W2_NHANES_DUAL_AXIS_2017_2023.md sau khi validation pass |
| scripts/10_plot_w2.py | Vẽ PNG/SVG bằng Matplotlib |
| tests/test_w2.py | 5 kiểm thử mới cho trọng số, penalty, threshold, tách train/validation và false positives |
| README.md | Hướng dẫn chạy W2 và dùng model |

## 4. Protocol đang dùng

- Python/NumPy/SciPy thực tế: `{'python': '3.12.3', 'numpy': '1.26.4', 'scipy': '1.11.4'}`.
- Docker socket không truy cập được; lượt này chạy bằng Python cục bộ.
- Outer seed: 42 + repeat × 100; cùng splits W1.
- K: 3,5,7,11,15,21,31,51.
- Alpha: 0; 0,25; 0,5; 0,75; 1.
- Threshold balanced score: 0,3; 0,4; 0,5; 0,6; 0,7; 0,75; 0,8; 0,85; 0,9; 0,95.
- B duyệt toàn bộ cặp threshold. C/D/E tìm theo hai lượt coordinate search, không bảo đảm tối ưu toàn cục.
- D/E có 24 perturbations cố định cộng identity, không phải tối ưu tự do 46 trọng số.
- Tham số/seed/ứng viên/hash: reports/w2/protocol.json.

## 5. Checkpoint đã lưu

**Đã hoàn thành 25/25 outer folds.** Mỗi fold có metrics.json (completion marker), predictions.csv, weights.csv và search.csv. Giữ nguyên reports/w2 để bảo toàn khả năng tái lập.

- [x] Repeat 0: `repeat_0_fold_0`, `repeat_0_fold_1`, `repeat_0_fold_2`, `repeat_0_fold_3`, `repeat_0_fold_4`.
- [x] Repeat 1: `repeat_1_fold_0`, `repeat_1_fold_1`, `repeat_1_fold_2`, `repeat_1_fold_3`, `repeat_1_fold_4`.
- [x] Repeat 2: `repeat_2_fold_0`, `repeat_2_fold_1`, `repeat_2_fold_2`, `repeat_2_fold_3`, `repeat_2_fold_4`.
- [x] Repeat 3: `repeat_3_fold_0`, `repeat_3_fold_1`, `repeat_3_fold_2`, `repeat_3_fold_3`, `repeat_3_fold_4`.
- [x] Repeat 4: `repeat_4_fold_0`, `repeat_4_fold_1`, `repeat_4_fold_2`, `repeat_4_fold_3`, `repeat_4_fold_4`.

Temporal validation và fit model cuối đã hoàn thành. `models/nhanes_w2.pkl` lưu cấu hình D theo quy tắc chọn macro-F1 outer trung bình cao nhất đã triển khai; không đổi lựa chọn sau khi xem temporal.

## 6. Kết quả cuối — đã kiểm chứng

| Model | Số folds | Macro-F1 trung bình | Accuracy trung bình | Macro-F1 temporal |
|---|---:|---:|---:|---:|
| A_W1_original | 25 | 0.4113 | 0.7258 | 0.4048 |
| B_joint_thresholds | 25 | 0.4317 | 0.7782 | 0.4335 |
| C_joint_K_alpha | 25 | 0.4456 | 0.7982 | 0.4409 |
| D_W2_groups | 25 | 0.4462 | 0.7955 | 0.4366 |
| E_W2_groups_regularized | 25 | 0.4446 | 0.7981 | 0.4409 |

- B − A = +0,0204; C − B = +0,0139. Phần lớn cải thiện đến từ threshold và K/alpha.
- D − C = +0,0006; khoảng mô tả 95% theo repeat [−0,0045; +0,0058]. E − C = −0,0010; khoảng [−0,0042; +0,0022]. Chưa thấy lợi ích bổ sung ổn định từ trọng số nhóm. Các khoảng chỉ mô tả do training sets chồng lấn.
- Temporal: D = 0,4366 thấp hơn C = 0,4409; E bằng C. Lợi thế nhóm nhỏ trong nested CV chưa lặp lại khi chuyển chu kỳ.
- D so với A: accuracy tăng 0,7258 → 0,7955; FP từ lớp 0 sang lớp 1–3 giảm 154,3 → 74,3/1.000 người. Precision lớp 3 tăng 0,2144 → 0,3235, nhưng recall lớp 3 giảm 0,5626 → 0,4106. Không diễn giải cải thiện macro-F1 thành tăng mọi chỉ số.
- Tại recall tham chiếu W1: precision eGFR A/C/D/E ≈ 0,3614/0,3961/0,3961/0,3960; precision uACR ≈ 0,2947/0,3262/0,3232/0,3268. B bằng A vì chỉ đổi threshold, không đổi scores. Đây chỉ là phân tích mô tả, không đưa threshold này vào artifact.
- Trọng số: tuổi đứng đầu eGFR; DIQ010 đứng đầu uACR theo mean weight ở C/D/E. D chọn nhiều ứng viên khác nhau, E giữ identity ở 13/25 folds. Đọc rank_std và top5_frequency trong weights_w2_summary.csv; không xem rank thành nhân quả.

## 7. Kiểm chứng và nơi đọc kết quả

- 21/21 unit tests pass trong môi trường cục bộ.
- `reports/w2/validation.json`: passed; 125 model-fold evaluations, 327.150 OOF predictions, 13.086 người. Đối chiếu partition, metrics và trọng số; inference với dữ liệu thiếu pass.
- A khớp W1: sai khác metrics lớn nhất bằng 0 ở cả nested CV và temporal.
- Báo cáo tiếng Việt: `CAI_THIEN_W2_NHANES_DUAL_AXIS_2017_2023.md`.
- Báo cáo thô: `reports/w2/REPORT.md`; kết quả máy đọc: `reports/w2/results.json`.
- Model: `models/nhanes_w2.pkl`.
- Biểu đồ PNG/SVG: `reports/w2/macro_f1_comparison.*` và `reports/w2/precision_recall_curves.*`.
- Log: `reports/w2_training.log`. Lịch sử dừng và tiếp tục: `reports/w2/STOPPED_BY_USER.json` (status hiện là completed).

## 8. Tiêu chí đọc kết quả

1. Kiểm tra A tái lập đúng W1, tất cả người được đánh giá đúng một lần/model/repeat.
2. Đọc B − A để đánh giá chọn threshold chung và lưới threshold mới.
3. Đọc C − B để đánh giá tìm alpha/K và mở rộng K.
4. Đọc D/E − C để đánh giá lợi ích riêng của trọng số nhóm.
5. Đọc precision/recall từng lớp, FP tuyệt đối và FP/1.000 người; không chỉ xem macro-F1.
6. Đọc matched_recall_diagnostics.csv để so precision tại recall tham chiếu W1; các threshold này chỉ mô tả, không đưa vào model cuối.
7. Đọc độ ổn định trọng số/rank qua folds và kết quả temporal.
8. Khoảng theo repeat chỉ mang tính mô tả vì training sets chồng lấn. Nếu W2 nhóm không tốt hơn C, báo đúng kết quả, không quy cải thiện threshold/K cho trọng số.

## 9. Artifact đã xuất sau huấn luyện

- reports/w2/results.json: kết quả và cấu hình cuối.
- reports/w2/REPORT.md: báo cáo thô sinh tự động.
- reports/w2/fold_metrics.csv và oof_predictions.csv.
- reports/w2/temporal.csv và temporal_predictions.csv.
- reports/w2/weights_w2_by_fold.csv và weights_w2_summary.csv.
- reports/w2/final_parameters.json và final_weights.csv.
- reports/w2/precision_recall_thresholds.csv, matched_recall_diagnostics.csv.
- models/nhanes_w2.pkl: artifact nghiên cứu, dùng nhanes_pipeline.w2.predict.
- reports/w2/validation.json: đã xuất, status passed.
- CAI_THIEN_W2_NHANES_DUAL_AXIS_2017_2023.md: đã xuất.

## 10. Giới hạn giữ nguyên

Target một lần đo không xác nhận CKD mạn tính. Balanced score chưa calibration. Temporal cohort đã được xem trong quá trình nghiên cứu trước. Chưa dùng survey weights/strata/PSU; không suy luận dân số hoặc nhân quả, không dùng model làm chẩn đoán.
