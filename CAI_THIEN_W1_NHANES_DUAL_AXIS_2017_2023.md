# Cải thiện W1 trên NHANES 2017–2023 bằng KNN trọng số hai trục eGFR/uACR

## 1. Tóm tắt kết quả

Phiên bản W1 cải thiện mở rộng cohort từ 5.343 lên **13.086 người trưởng thành có nhãn thật**, bằng cách kết hợp:

- NHANES 2017–March 2020: 7.743 người.
- NHANES August 2021–August 2023: 5.343 người.

Thay vì học một vector trọng số chung cho target bốn lớp, mô hình mới học hai vector riêng:

- Trọng số cho trục `eGFR < 60`.
- Trọng số cho trục `uACR ≥ 30 mg/g`.

Hai dự đoán nhị phân sau đó được ghép lại thành bốn nhóm dấu hiệu thận ban đầu.

Kết quả repeated nested CV, 5 folds × 5 repeats:

| Model | Macro-F1 | Balanced accuracy | Accuracy | Recall lớp 1 | Recall lớp 2 | Recall lớp 3 |
|---|---:|---:|---:|---:|---:|---:|
| Flat uniform distance | 0,3337 ± 0,0116 | 0,3184 | 0,8044 | 0,0761 | 0,1186 | 0,1235 |
| Flat MI distance | 0,3528 ± 0,0165 | 0,3358 | 0,8033 | 0,0757 | 0,1443 | 0,1726 |
| Dual-axis MI distance | 0,3684 ± 0,0151 | 0,3444 | 0,8047 | 0,1070 | 0,1815 | 0,1414 |
| **Dual-axis MI + balanced voting** | **0,4113 ± 0,0157** | **0,4814** | 0,7258 | **0,2543** | **0,2959** | **0,5626** |
| Dual-axis MI + balanced + missing indicators | 0,4112 ± 0,0137 | 0,4807 | 0,7252 | 0,2591 | 0,2924 | 0,5594 |

So với flat MI KNN, cấu hình tốt nhất tăng macro-F1 trung bình **0,0585**. Khoảng mô tả 95% tính trên năm chênh lệch repeat là `[0,0502; 0,0668]`. Các training set trong repeated CV vẫn chồng lấn, vì vậy khoảng này không phải kiểm định trên năm nghiên cứu độc lập.

---

## 2. Mục tiêu nghiên cứu

Target của project được tạo từ hai dấu hiệu:

1. eGFR thấp: `eGFR < 60`.
2. Albumin niệu: `uACR ≥ 30 mg/g`.

Hai dấu hiệu nhị phân tạo thành bốn lớp:

| Target | eGFR thấp | uACR cao | Diễn giải |
|---:|---|---|---|
| 0 | Không | Không | Không có dấu hiệu theo hai ngưỡng |
| 1 | Không | Có | Chỉ có albumin niệu |
| 2 | Có | Không | Chỉ có eGFR thấp |
| 3 | Có | Có | Có cả hai dấu hiệu |

Đây là bốn pattern theo phép đo khảo sát, không phải bốn giai đoạn CKD. Target dựa trên một lần đo nên không xác nhận tính mạn tính và không được dùng như chẩn đoán y khoa.

Câu hỏi nghiên cứu của W1 cải thiện:

> Việc học trọng số đặc trưng riêng cho trục eGFR và trục uACR, kết hợp với class-balanced neighbor voting, có cải thiện khả năng nhận diện bốn pattern so với KNN dùng một vector trọng số chung hay không?

---

## 3. Vì sao W1 cũ chưa đủ tốt?

W1 cũ dùng một vector MI cho toàn bộ target bốn lớp. Cách làm này có ba hạn chế.

### 3.1 Hai cơ chế nhãn khác nhau

Feature liên quan đến eGFR thấp không nhất thiết cũng liên quan đến albumin niệu:

- Tuổi, uric acid và hemoglobin có thể cung cấp tín hiệu mạnh hơn cho eGFR thấp.
- Đái tháo đường, huyết áp tâm thu và điều trị tăng huyết áp có thể cung cấp tín hiệu mạnh hơn cho uACR cao.

Một vector chung phải thỏa hiệp giữa hai cơ chế.

### 3.2 Mất cân bằng bốn lớp

Trong cohort mới:

| Lớp | Số mẫu | Tỷ lệ xấp xỉ |
|---:|---:|---:|
| 0 | 10.786 | 82,42% |
| 1 | 1.228 | 9,38% |
| 2 | 693 | 5,30% |
| 3 | 379 | 2,90% |
| **Tổng** | **13.086** | **100%** |

KNN thông thường tìm phần lớn hàng xóm thuộc lớp 0. Distance voting giúp nhưng chưa đủ chống lại chênh lệch mật độ lớp.

### 3.3 Cohort cũ nhỏ

W1 trước đó chỉ tối ưu trên 3.740 mẫu historical train, trong đó lớp 3 có 100 mẫu. Việc mở rộng lên 13.086 người giúp:

- Lớp 1 tăng lên 1.228 mẫu.
- Lớp 2 tăng lên 693 mẫu.
- Lớp 3 tăng lên 379 mẫu.
- MI và lựa chọn siêu tham số ổn định hơn qua folds.

---

## 4. Xây dựng cohort kết hợp

### 4.1 Tiêu chí đưa vào

Một người chỉ được đưa vào supervised cohort khi:

```text
Tuổi ≥ 20
+ có creatinine máu
+ có albumin nước tiểu
+ có creatinine nước tiểu hợp lệ
= có thể tính eGFR, uACR và target thật
```

Không impute các xét nghiệm tạo nhãn và không tạo pseudo-label cho người thiếu nhãn.

### 4.2 Cohort flow

| Chu kỳ | Tổng DEMO | Tuổi ≥20 và đủ nguồn nhãn | Đưa vào modeling |
|---|---:|---:|---:|
| 2017–March 2020 | 15.560 | 7.743 | 7.743 |
| 2021–2023 | 11.933 | 5.343 | 5.343 |
| **Kết hợp** | 27.493 | **13.086** | **13.086** |

Các giá trị SAS missing trong XPT được nhận diện cả khi parser biểu diễn chúng dưới dạng `NaN`. Chúng không được tính nhầm là kết quả xét nghiệm hợp lệ.

### 4.3 Harmonization albumin nước tiểu

NHANES 2017–March 2020 và 2021–2023 dùng hai phương pháp đo albumin nước tiểu khác nhau. Trước khi tạo uACR, albumin của chu kỳ cũ được chuyển sang thang LC-MS/MS mới bằng phương trình CDC:

```text
URXUMA_harmonized = 1.381 + 0.8407 × URXUMA_2017_2020
```

Sau đó:

```text
uACR = URXUMA_harmonized / URXUCR × 100
```

Creatinine nước tiểu và serum creatinine không được điều chỉnh vì CDC không khuyến nghị correction giữa hai chu kỳ này.

### 4.4 Harmonization thuốc huyết áp

Chu kỳ cũ dùng `BPQ050A`, trong khi chu kỳ mới dùng `BPQ150`. Biến chung được tạo theo skip pattern:

- `BPQ050A=1` → Có dùng thuốc.
- `BPQ050A=2` → Không dùng thuốc.
- Chưa từng được thông báo tăng huyết áp → Không dùng thuốc huyết áp.
- Từng có tăng huyết áp nhưng chưa từng được yêu cầu dùng thuốc → Không dùng thuốc.
- Refused/don't know hoặc không thể suy ra → missing.

### 4.5 Schema 23 feature

`moderate_freq_week` bị loại khỏi primary combined analysis vì:

- Chu kỳ cũ đo số ngày hoạt động vừa trong một tuần.
- Chu kỳ mới đo số lần theo ngày/tuần/tháng/năm.
- Nội dung câu hỏi và điều kiện thời lượng không hoàn toàn tương đương.

Primary schema còn 23 feature. Việc loại biến được khai báo rõ, không sửa lịch sử W1 24 feature.

---

## 5. Thiết kế mô hình hai trục

### 5.1 Trục G: eGFR thấp

```text
y_G = 1 nếu eGFR < 60
y_G = 0 nếu eGFR ≥ 60
```

Số mẫu dương:

```text
lớp 2 + lớp 3 = 693 + 379 = 1.072
```

### 5.2 Trục A: uACR cao

```text
y_A = 1 nếu uACR ≥ 30
y_A = 0 nếu uACR < 30
```

Số mẫu dương:

```text
lớp 1 + lớp 3 = 1.228 + 379 = 1.607
```

### 5.3 Ghép lại thành bốn lớp

```python
predicted_target = predicted_uacr_high + 2 * predicted_egfr_low
```

Mô hình vẫn trả về bốn pattern ban đầu, nhưng mỗi trục có:

- Vector MI riêng.
- Alpha riêng.
- K riêng.
- Threshold riêng.

---

## 6. Học trọng số đặc trưng

Với mỗi trục:

```text
w(alpha) = (1 - alpha) × uniform + alpha × normalized_MI
```

Lưới tìm kiếm:

```text
alpha = {0, 0.25, 0.5, 0.75, 1}
K = {3, 5, 7, 11, 15}
threshold = {0.3, 0.4, 0.5, 0.6, 0.7}
```

Siêu tham số cuối:

| Trục | Alpha | K | Threshold |
|---|---:|---:|---:|
| eGFR | 1,0 | 15 | 0,7 |
| uACR | 1,0 | 15 | 0,6 |

Alpha bằng 1 nghĩa là artifact cuối sử dụng hoàn toàn normalized MI của full cohort. K và threshold lớn hơn cấu hình W1 cũ giúp giảm độ nhạy với ba hàng xóm cục bộ và kiểm soát class-balanced scores.

---

## 7. Class-balanced neighbor voting

Với neighbor `i`, phiếu được tính từ khoảng cách và tần suất lớp trong fold-train:

```text
vote_i = 1 / distance_i × inverse_class_frequency(y_i)
```

Class frequency chỉ được tính trên fold-train. Validation/test không tham gia xác định hệ số cân bằng.

Balanced voting làm tăng ảnh hưởng của hàng xóm thuộc nhóm eGFR thấp hoặc uACR cao. Điều này tăng recall lớp thiểu số nhưng cũng tăng false positives.

---

## 8. Tiền xử lý và khoảng cách

Trong mỗi fold:

- Numeric: median imputation và standardization.
- Categorical: mode imputation.
- Numeric distance: squared standardized difference.
- Categorical distance: mismatch được chuẩn hóa để đóng góp đúng một group weight.
- MI, median, mode, scale, class frequency và threshold đều được fit trong train fold.

SEQN, cycle, eGFR, uACR và các xét nghiệm trực tiếp tạo nhãn không được đưa vào vector đầu vào.

Missing-indicator ablation chia group weight giữa giá trị đã impute và cờ missing, không tạo thêm một feature group có trọng số độc lập.

---

## 9. Protocol đánh giá

### 9.1 Repeated nested CV

```text
Outer: 5 folds × 5 repeats
Inner: 3 folds
Stratification: target bốn lớp
Primary metric: macro-F1
```

Outer folds chỉ dùng để đánh giá. Inner folds chọn:

- Alpha.
- K.
- Threshold.

### 9.2 Temporal validation

```text
Train: NHANES 2017–March 2020
Test: NHANES 2021–2023
```

Temporal validation đo khả năng chuyển qua chu kỳ và thay đổi quy trình thu thập. Nó chưa phải external validation hoàn toàn độc lập vì dữ liệu 2021–2023 đã xuất hiện trong quá trình nghiên cứu trước đây.

---

## 10. Phân tích kết quả nested CV

### 10.1 Tác động của MI

Flat MI tăng macro-F1 từ `0,3337` lên `0,3528` so với flat uniform. Trên cohort lớn hơn, MI có tín hiệu cải thiện rõ hơn W1 cũ, nhưng recall các lớp thiểu số vẫn thấp.

### 10.2 Tác động của dual-axis

Dual-axis MI distance đạt macro-F1 `0,3684`, cao hơn flat MI `0,3528`. Việc tách hai trục giúp học được hai cấu trúc feature khác nhau, nhưng nếu không cân bằng voting, model vẫn nghiêng mạnh về lớp 0.

### 10.3 Tác động của balanced voting

Dual-axis MI + balanced voting đạt:

- Macro-F1: `0,4113`.
- Balanced accuracy: `0,4814`.
- Recall lớp 1: `0,2543`.
- Recall lớp 2: `0,2959`.
- Recall lớp 3: `0,5626`.

So với flat MI:

| Metric | Flat MI | Dual balanced | Thay đổi |
|---|---:|---:|---:|
| Macro-F1 | 0,3528 | 0,4113 | +0,0585 |
| Balanced accuracy | 0,3358 | 0,4814 | +0,1456 |
| Recall lớp 1 | 0,0757 | 0,2543 | +0,1785 |
| Recall lớp 2 | 0,1443 | 0,2959 | +0,1515 |
| Recall lớp 3 | 0,1726 | 0,5626 | +0,3900 |
| Accuracy | 0,8033 | 0,7258 | −0,0775 |

Đây không phải cải thiện miễn phí. Model đổi một phần accuracy và specificity để nhận diện thêm các lớp thiểu số.

### 10.4 Missing indicators

Thêm missing indicators đạt macro-F1 `0,4112`, gần như bằng model không indicator `0,4113`. Không có bằng chứng rằng missing indicators cải thiện primary metric trong cấu hình hiện tại.

---

## 11. Confusion matrix OOF gộp

Mỗi người có một OOF prediction trong mỗi repeat, nên bảng dưới có tổng support bằng năm lần cohort. Đây là mô tả gộp, không phải 65.430 người độc lập.

```text
Dual-axis MI + balanced voting

             Pred 0   Pred 1   Pred 2   Pred 3
True 0        43836     5201     2836     2057
True 1         3272     1561      472      835
True 2          999      367     1025     1074
True 3          282      203      344     1066
```

Per-class OOF gộp:

| Lớp | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0 | 0,9059 | 0,8128 | 0,8568 |
| 1 | 0,2129 | 0,2542 | 0,2317 |
| 2 | 0,2192 | 0,2958 | 0,2518 |
| 3 | 0,2118 | 0,5625 | 0,3078 |

Recall lớp 3 tăng mạnh, nhưng precision chỉ khoảng 21%. Model tạo nhiều false positives cho lớp thiểu số; đây là giới hạn quan trọng nhất của balanced voting hiện tại.

---

## 12. Temporal validation

Train trên 2017–March 2020, test trên 2021–2023:

| Model | Macro-F1 | Balanced accuracy | Accuracy | Recall 1 | Recall 2 | Recall 3 |
|---|---:|---:|---:|---:|---:|---:|
| Flat MI | 0,3437 | 0,3221 | 0,8186 | 0,0594 | 0,0965 | 0,1608 |
| Dual MI | 0,3510 | 0,3284 | 0,8055 | 0,0955 | 0,1350 | 0,1329 |
| Dual MI balanced | **0,4048** | **0,4770** | 0,7084 | **0,3100** | 0,2797 | **0,5315** |
| Dual balanced + missing | 0,4081 | 0,4719 | 0,7163 | 0,2951 | **0,3055** | 0,4895 |

Xu hướng cải thiện macro-F1 và recall lớp 3 vẫn tồn tại khi chuyển chu kỳ, nhưng accuracy tiếp tục giảm do false positives.

---

## 13. Trọng số cuối

### 13.1 Trục eGFR thấp

| Hạng | Feature | Weight |
|---:|---|---:|
| 1 | RIDAGEYR | 0,261843 |
| 2 | BPQ150 | 0,116928 |
| 3 | LBXSUA | 0,107316 |
| 4 | BPQ020 | 0,101955 |
| 5 | LBXHGB | 0,057222 |
| 6 | DIQ010 | 0,050059 |
| 7 | sbp_mean | 0,042963 |
| 8 | BMXWAIST | 0,042818 |

Tuổi đứng đầu một phần vì tuổi tham gia trực tiếp công thức eGFR. Không được diễn giải đây là bằng chứng nhân quả hoặc độ quan trọng lâm sàng độc lập.

### 13.2 Trục uACR cao

| Hạng | Feature | Weight |
|---:|---|---:|
| 1 | DIQ010 | 0,200764 |
| 2 | sbp_mean | 0,137763 |
| 3 | BPQ150 | 0,135500 |
| 4 | BPQ020 | 0,135483 |
| 5 | RIDAGEYR | 0,128979 |
| 6 | ALQ121 | 0,034161 |
| 7 | DMDEDUC2 | 0,032911 |
| 8 | BPQ080 | 0,031777 |

Hai vector khác nhau rõ rệt. Uric acid đứng thứ ba ở trục eGFR nhưng gần như bằng 0 ở trục uACR; đái tháo đường đứng đầu ở trục uACR nhưng chỉ thứ sáu ở trục eGFR. Đây là bằng chứng thực nghiệm ủng hộ việc không dùng một vector chung.

---

## 14. Kết luận

W1 cải thiện đạt ba kết quả chính:

1. Mở rộng cohort từ 5.343 lên 13.086 người có nhãn thật mà không tạo pseudo-label.
2. Học hai vector trọng số riêng phù hợp với cấu trúc target eGFR/uACR.
3. Class-balanced voting tăng đáng kể macro-F1, balanced accuracy và recall lớp thiểu số.

Tuy nhiên, model vẫn chưa phù hợp triển khai y tế:

- Precision lớp 1–3 còn thấp.
- Accuracy giảm do tăng false positives.
- Target dựa trên một lần đo, không xác nhận CKD mạn tính.
- Temporal validation chưa phải external validation hoàn toàn độc lập.
- Survey design weights chưa được dùng cho suy luận dân số.

Kết luận nghiên cứu phù hợp:

> Tách target thành hai trục và sử dụng class-balanced distance voting cải thiện ổn định khả năng nhận diện các pattern thiểu số so với flat MI KNN. Cải thiện recall đi kèm chi phí false positives lớn; cần nghiên cứu calibration/threshold và external validation trước khi xem xét ứng dụng.

---

## 15. Công việc tiếp theo

Ưu tiên tiếp theo nên là:

1. Vẽ precision–recall curve cho từng trục và chọn operating point theo mục tiêu cụ thể.
2. Báo cáo false positives tuyệt đối ở từng threshold thay vì chỉ chọn threshold theo F1.
3. So sánh dual-axis KNN với logistic regression, random forest và gradient boosting có class weighting.
4. Chạy sensitivity analysis với biến hoạt động thể lực harmonized gần đúng.
5. Phân tích selection bias giữa người có và không có đủ xét nghiệm tạo nhãn.
6. Tạo external validation bằng một cohort chưa tham gia quá trình thiết kế.
7. Dùng survey weights, strata và PSU trong phân tích mô tả dân số riêng.
8. Không chuyển sang tối ưu tự do 23 trọng số cho đến khi kiểm tra độ ổn định weight qua folds và cycles.

---

## 16. Artifact và khả năng tái lập

- Combined cohort: `data/processed/nhanes_combined_2017_2023.csv`
- Manifest/checksum: `data/processed/nhanes_combined_manifest.json`
- Báo cáo tóm tắt: `reports/combined_w1/REPORT.md`
- Fold metrics: `reports/combined_w1/nested_cv_fold_metrics.csv`
- OOF predictions: `reports/combined_w1/oof_predictions.csv`
- Temporal validation: `reports/combined_w1/temporal_validation.csv`
- Hai vector trọng số: `reports/combined_w1/final_axis_weights.csv`
- Result JSON: `reports/combined_w1/results.json`
- Artifact cuối: `models/nhanes_combined_dual_axis.pkl`
- Xây cohort: `scripts/03_build_combined_cohort.py`
- Train/evaluate: `scripts/04_train_combined_w1.py`
- Export weights: `scripts/05_export_combined_weights.py`
- Tạo báo cáo: `scripts/06_finalize_combined_report.py`

Toàn bộ 16 unit tests của project đã pass trong Docker với NumPy 2.1.3 và SciPy 1.14.1.
