# Phân tích kết quả W0 trên NHANES và kế hoạch cải thiện trọng số

> **Tài liệu hướng dẫn cho Codex trong VS Code.** Đọc tài liệu này cùng CODEX_PROJECT.md. Với thí nghiệm trong tài liệu này, dữ liệu hiện hành là NHANES 2021–2023, không phải UCI CKD. Không áp dụng ngầm các giả định 400 mẫu/24 biến/nhãn nhị phân của kế hoạch UCI. Giữ nguyên tài liệu UCI làm hồ sơ lịch sử; triển khai thí nghiệm NHANES thành cấu hình và báo cáo riêng.

## 1. Mục tiêu

Phân tích hai baseline KNN đã chạy trên bộ NHANES mới, giải thích vì sao chúng yếu ở các lớp thiểu số, rồi thiết kế thí nghiệm W1 học trọng số đặc trưng theo cách chống rò rỉ dữ liệu.

Mục tiêu không phải làm đẹp điểm W0. W0 là mốc trọng số đặc trưng bằng nhau. Mọi thay đổi cách biểu quyết, nhãn, đặc trưng hoặc trọng số phải được ghi thành một cấu hình/ablation riêng để so sánh công bằng.

## 2. Dữ liệu và định nghĩa nhãn

### 2.1 Nguồn và cách tạo mẫu

- Nguồn: CDC/NCHS NHANES, chu kỳ August 2021–August 2023; các tệp XPT được ghép theo SEQN.
- Các bảng đã dùng: DEMO_L, BIOPRO_L, ALB_CR_L, BMX_L, BPXO_L, BPQ_L, DIQ_L, SMQ_L, ALQ_L, PAQ_L, TCHOL_L, HDL_L, CBC_L.
- Số SEQN duy nhất sau ghép: **11.933**.
- Giữ người từ 20 tuổi trở lên và chỉ giữ mẫu có thể tính được eGFR và uACR: **5.343 mẫu**.
- SEQN chỉ là mã ghép/đối chiếu; tuyệt đối không đưa vào đầu vào mô hình.

### 2.2 Nhãn bốn lớp hiện tại

Nhãn được suy ra từ **một lần đo** eGFR và uACR:

| target | Quy tắc tạo nhãn | Cách gọi trong báo cáo |
|---:|---|---|
| 0 | eGFR ≥ 60 và uACR < 30 mg/g | Không có dấu hiệu theo hai chỉ số |
| 1 | eGFR ≥ 60 và uACR ≥ 30 mg/g | Chỉ có albumin niệu |
| 2 | eGFR < 60 và uACR < 30 mg/g | Chỉ có eGFR thấp |
| 3 | eGFR < 60 và uACR ≥ 30 mg/g | Cả hai dấu hiệu |

Đây là **nhóm dấu hiệu theo phép đo khảo sát**, không phải bốn giai đoạn CKD và không xác nhận tính mạn tính. Không gọi lớp 0 là “chắc chắn không mắc CKD”; không trình bày mô hình như công cụ chẩn đoán.

Nhãn dùng creatinine máu để tính eGFR và albumin/creatinine nước tiểu để tính uACR. Các biến trực tiếp dùng tạo nhãn (LBXSCR, URXUMA, URXUCR, egfr, uacr) không nằm trong 24 biến đầu vào để tránh mô hình chỉ tái tạo công thức tạo nhãn. Tuổi và giới vẫn là đầu vào; báo cáo phải ghi rõ chúng cũng tham gia công thức eGFR.

### 2.3 Phân bố lớp

| Lớp | Số mẫu toàn bộ | Tỷ lệ xấp xỉ |
|---|---:|---:|
| 0 — Không có dấu hiệu theo hai chỉ số | 4.418 | 82,7% |
| 1 — Chỉ albumin niệu | 471 | 8,8% |
| 2 — Chỉ eGFR thấp | 311 | 5,8% |
| 3 — Cả hai dấu hiệu | 143 | 2,7% |
| **Tổng** | **5.343** | **100%** |

## 3. Bộ 24 biến đầu vào hiện hành

Codex phải giữ đúng schema này khi tái lập baseline; không tự thêm lại biến đã bỏ hoặc đổi target.

~~~text
RIDAGEYR, RIAGENDR, RIDRETH3, DMDEDUC2, INDFMPIR,
BMXBMI, BMXWAIST, sbp_mean, dbp_mean,
BPQ020, BPQ150, DIQ010, DIQ160, current_smoking,
ALQ111, ALQ121, ALQ130, PAD680, moderate_freq_week, BPQ080,
LBXTC, LBDHDD, LBXSUA, LBXHGB
~~~

- sbp_mean, dbp_mean: trung bình ba lần đo có sẵn BPXOSY1–3 và BPXODI1–3.
- current_smoking: biến dẫn xuất từ SMQ020/SMQ040.
- moderate_freq_week: tần suất hoạt động vừa quy đổi về lần/tuần từ PAD790Q/PAD790U. Nếu lượng đo dương nhưng đơn vị thiếu, giữ là thiếu; không gán 0.
- BPQ080 được dùng thay vigorous_freq_week. Biến vigorous bị loại vì khoảng 54,6% giá trị dẫn xuất thiếu do thiếu đơn vị khi lượng đo dương.
- BPQ150=2 chỉ được suy ra khi BPQ020=2 theo nhánh câu hỏi. Các trường hợp từ chối/không biết vẫn là thiếu.
- Các biến số dùng median imputation và StandardScaler; biến phân loại dùng mode imputation và one-hot encoding. Toàn bộ tham số imputation, encoding, scaling phải được fit bên trong từng fold.
- Các nhóm categorical hiện gồm: RIAGENDR, RIDRETH3, DMDEDUC2, BPQ020, BPQ150, BPQ080, DIQ010, DIQ160, current_smoking, ALQ111, ALQ121.
- Tỷ lệ thiếu sau xử lý mã từ chối/không biết vẫn đáng chú ý: ALQ130 khoảng 26,6%, moderate_freq_week khoảng 19,8%, ALQ111/ALQ121 khoảng 12,5%, INDFMPIR khoảng 12,7%. Không xóa hàng theo complete-case chỉ để tăng tiện lợi; báo cáo tỷ lệ thiếu và giữ imputation trong fold.

## 4. Baseline đã đo

Holdout cố định: stratified 70/30, random_state=42; train **3.740**, test **1.603**. Train class counts [3092, 330, 218, 100]; test class counts [1326, 141, 93, 43] theo thứ tự target 0–3.

### M1 — W0 chính

- Feature weights: đều, mỗi biến gốc có trọng số 1/24.
- Voting: uniform.
- K chọn bằng 5-fold StratifiedKFold trên train: **K=3**.
- CV macro-F1: **0,3155**.
- Holdout: accuracy **0,810**, macro-F1 **0,2863**, balanced accuracy **0,2825**.
- Confusion matrix, hàng là nhãn thật, cột là nhãn dự đoán:

~~~text
[[1283, 25, 16, 2],
 [ 128,  6,  7, 0],
 [  75,  3,  9, 6],
 [  30,  6,  6, 1]]
~~~

Test recall theo lớp: lớp 0 **0,968**; lớp 1 **0,043**; lớp 2 **0,097**; lớp 3 **0,023**.

### M2 — cùng trọng số đặc trưng đều, distance voting

- Feature weights vẫn đều 1/24; thay đổi duy nhất so với M1 là neighbor voting distance.
- CV chọn **K=3**, voting distance; CV macro-F1 **0,3399**.
- Holdout: accuracy **0,795**, macro-F1 **0,2981**, balanced accuracy **0,2932**.
- Confusion matrix:

~~~text
[[1254, 32, 30, 10],
 [ 120,  7, 14,  0],
 [  70,  5, 10,  8],
 [  25,  6,  9,  3]]
~~~

Test recall theo lớp: lớp 0 **0,946**; lớp 1 **0,050**; lớp 2 **0,108**; lớp 3 **0,070**.

### 4.1 Diễn giải

1. M2 tăng holdout macro-F1 khoảng **0,012** và balanced accuracy khoảng **0,011** so với M1; lớp 3 được nhận đúng 3/43 thay vì 1/43. Đây là cải thiện nhỏ, không phải hiệu năng tốt.
2. Accuracy M1/M2 lần lượt **81,0%/79,5%**, thấp hơn baseline luôn đoán lớp 0: 1326/1603 = **82,7%**. Vì vậy accuracy cao bề ngoài không chứng minh nhận diện lớp CKD-evidence tốt.
3. Cả hai cấu hình vẫn bỏ sót đa số lớp 1–3. Nguyên nhân hợp lý gồm mất cân bằng lớp, 24 đặc trưng không trực tiếp chứa các chỉ số tạo nhãn, nhiều biến yếu/thiếu/correlated, phép đo cắt ngang một lần và khoảng cách Euclidean có thể không tách được các nhóm dấu hiệu.
4. CV tốt hơn holdout ở cả M1 và M2; M2 còn được chọn trong một lưới nhiều cấu hình, nên CV best-score có thể lạc quan. Chỉ một lần 5-fold không đủ để khẳng định M2 tổng quát tốt hơn chắc chắn.
5. Không diễn giải việc M2 tăng nhẹ là bằng chứng W1 sẽ tốt. Trọng số feature có thể không cải thiện; nếu không cải thiện, ghi nhận trung thực.

## 5. Tình trạng holdout và chống rò rỉ

Holdout đã được xem cho cả M1 và M2. Vì vậy, trong các phát triển tiếp theo:

- Không dùng test để chọn trọng số, K, alpha/shrinkage, threshold, missingness strategy hoặc danh sách biến.
- Chọn W1 bằng nested stratified CV chỉ trên train: outer folds để ước lượng hiệu năng, inner folds để chọn alpha/K/vote. Tính MI và fit preprocessing riêng bên trong mỗi inner-train fold. Repeated CV có thể bổ sung để đo độ ổn định; repeated CV đơn thuần không thay thế nested CV khi đã chọn siêu tham số.
- Có thể lưu test cũ để tái lập bảng lịch sử, nhưng phải ghi nhãn **exploratory / đã xem**, không gọi là holdout độc lập cuối cùng.
- Nếu cần một ước lượng cuối độc lập, dùng cohort/holdout mới chưa từng xem. Không tạo lại test mới từ cùng dữ liệu sau khi đã xem toàn bộ kết quả rồi gọi nó là độc lập.

## 6. Kế hoạch cải thiện trọng số

### Bước 0 — xác nhận handoff

Đọc các file Kaggle handoff: train_data.csv, holdout_locked.csv, feature_schema.json, weights_w0.csv, cv_m1.csv, cv_m2.csv, baseline_metrics.json. Kiểm tra số dòng, 24 cột đầu vào, class counts và SEQN disjointness với manifest. Chỉ dùng train_data.csv cho W1. Không đọc holdout trong code tối ưu W1.

Nếu handoff hiện thiếu hash hoặc danh sách split đầy đủ, ghi nhận thiếu sót; không tự suy ra/chỉnh lại split. Giữ SEQN làm row ID, không đưa vào mô hình.

### W1 — group-level Mutual Information

Mục tiêu: kiểm tra liệu trọng số học theo biến gốc có giúp hơn W0 không, trong khi giữ các yếu tố khác cố định.

1. Trong **mỗi fold**, fit imputer/encoder/scaler chỉ trên fold-train.
2. Tính MI của từng **biến gốc trong 24 biến** với nhãn đa lớp chỉ trên fold-train. Khai báo đúng biến discrete/continuous; không tính MI trên toàn bộ train trước CV.
3. Không tính MI riêng cho từng cột one-hot rồi coi chúng là biến gốc độc lập. Mỗi nhóm gốc nhận đúng một trọng số.
4. Tạo MI weights không âm, có epsilon smoothing và fallback về đều nếu mọi MI bằng 0. Chuẩn hóa tổng bằng 1.
5. Shrink về W0 để giảm overfit:

~~~text
w(alpha) = (1 - alpha) * uniform_24 + alpha * normalized_MI
~~~

Chọn alpha trong một lưới nhỏ được khai báo trước, ví dụ {0, 0.25, 0.5, 0.75, 1}, chỉ bên trong train CV. Không mặc định alpha=1.

### So sánh ablation theo từng thay đổi

| Mã | Feature weights | Neighbor vote | So với |
|---|---|---|---|
| M1 | đều | uniform | Baseline gốc W0 |
| M2 | đều | distance | M1 để đo riêng tác động voting |
| M3/W1 | MI group/shrunk | uniform | M1 để đo riêng tác động feature weights |
| M4 | MI group/shrunk | distance | M2 để đo tác động feature weights khi đã dùng distance voting |

Giai đoạn đầu, chạy so sánh **K cố định = 3** để cô lập tác động của trọng số. Sau đó, nếu cần, tune K bằng cùng CV cho mỗi cấu hình và báo cáo riêng kết quả tuned. Không thay cùng lúc target, features, K, vote, threshold và weights trong một phép so sánh.

### Metric và mất cân bằng lớp

- Primary metric cho nhãn bốn lớp: **macro-F1**.
- Luôn báo cáo per-class precision/recall/F1/support, balanced accuracy, accuracy, confusion matrix, và dự đoán theo lớp.
- Không thêm oversampling, class-balanced voting hoặc thresholding vào W1 đầu tiên. Nếu nghiên cứu các cách này, đặt thành ablation riêng sau khi có kết quả group weights.
- Có thể làm thêm bài toán nhị phân target=0 so với target thuộc {1,2,3} như phân tích thứ cấp, nhưng giữ bài toán 4 lớp hiện tại làm nhiệm vụ chính của thí nghiệm này. Không đổi nhãn âm thầm.

### W2 — chỉ sau khi có W1 ổn định

Nếu MI group weights có hiệu quả và ổn định qua folds, mới thử tối ưu một số ít trọng số nhóm hoặc bucket. Ràng buộc w_j >= 0, sum(w)=1, regularize về uniform, dùng inner CV/nested CV và giới hạn trial. Không bắt đầu bằng tối ưu tự do 24 tham số trên một split; mẫu lớp 3 chỉ có 100 trong train.

## 7. Khoảng cách và quy tắc nhóm

Với 24 biến gốc:

~~~text
d_w^2(x,z) = sum_g w_g * D_g(x_g,z_g)
~~~

- Numeric: D_g = ((x_g-z_g)/scale_g)^2, scale fit trong fold-train.
- Categorical: mismatch 0/1 hoặc one-hot group được chuẩn hóa để một mismatch đóng góp đúng w_g bất kể số mức.
- SEQN, target, eGFR và uACR không thuộc vector đầu vào.
- Nếu thêm missing indicator, nó phải thuộc group gốc tương ứng và group vẫn chỉ có tổng trọng số w_g; không để cột indicator âm thầm tạo thêm trọng số.
- W0 group weight: mỗi biến gốc 1/24. Nếu mọi nhóm dùng cách chuẩn hóa như nhau, thứ tự hàng xóm tương ứng với tổng khoảng cách group-unweighted; xác nhận bằng kiểm tra toán học/toy case theo project spec.

## 8. Việc Codex cần làm trong VS Code

1. Thêm cấu hình và báo cáo riêng cho NHANES; cập nhật phần mô tả dataset/label, không sửa lịch sử UCI để giả vờ đó là cùng dữ liệu.
2. Tạo pipeline tái lập đọc handoff, kiểm tra schema/IDs, tách preprocessing và group distances, đảm bảo fit fold-local.
3. Implement W1 MI group weights type-aware, epsilon smoothing, uniform fallback, shrinkage alpha.
4. Chạy CV chỉ trên train, lưu fold-wise weights, CV predictions, per-class metrics, mean/std và stability/rank của weights qua folds.
5. Tạo bảng M1–M4; giữ test cũ làm cột exploratory/historical và không dùng nó ra quyết định.
6. Lưu weights_w1_by_fold.csv, weights_w1_summary.csv, nhanes_cv_results.csv, nhanes_ablation.csv, target/data manifest, và báo cáo tiếng Việt. Không bịa số liệu; nếu code chưa chạy thì ghi “chưa đo”.
7. Nêu rõ mô hình hiện có baseline yếu ở lớp thiểu số và NHANES label là pattern theo một lần đo; không tuyên bố đạt khả năng chẩn đoán.

### Prompt ngắn để giao Codex

> Đọc CODEX_PROJECT.md và PHAN_TICH_KET_QUA_NHANES_W0_VA_KE_HOACH_W1_CHO_CODEX.md. Thí nghiệm này dùng NHANES 2021–2023 và handoff 5.343 mẫu/24 biến/4 lớp; đây là nhánh mới, không phải UCI 400 dòng. Xác minh artifact/schema/split trước. Giữ M1/M2 làm baseline lịch sử. Implement W1 bằng Mutual Information theo 24 biến gốc, fit mọi preprocessing và MI bên trong từng train fold, normalize/smooth weights, shrink về uniform, chọn alpha/K bằng inner CV và ước lượng bằng outer CV trên train. Primary metric macro-F1 và báo cáo per-class recall/confusion matrix. Không dùng holdout đã xem để chọn weights hoặc hyperparameters; đánh dấu kết quả holdout cũ exploratory. Không đổi nhãn/feature/schema âm thầm và không bịa kết quả.

## 9. Nguồn và ghi chú

- NHANES continuous data portal: https://wwwn.cdc.gov/nchs/nhanes/
- NHANES 2021–2023 data page: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023
- CDC NHANES CKD definition/surveillance notes: eGFR <60 hoặc uACR ≥30 mg/g từ một lần đo; đây là dấu hiệu khảo sát và có thể không tương đương xác nhận CKD kéo dài.
- UCI CKD dataset (chỉ cho nhánh/đặc tả UCI cũ): https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

**Giới hạn báo cáo:** Các chỉ số ở M1/M2 là kết quả của split hiện tại. Holdout đã được xem; không dùng các con số này như một đánh giá cuối độc lập sau khi đã phát triển W1/W2.

