# CODEX_PROJECT.md — CKD-Weighted-KNN

> **Vai trò:** Đây là đặc tả kỹ thuật và kế hoạch thực hiện cho Codex trong VS Code. Đọc toàn bộ tài liệu này trước khi thay đổi mã. Triển khai theo milestone, chạy kiểm thử và báo cáo kết quả thật. Không tự bịa chỉ số Accuracy hoặc kết quả nghiên cứu.

## 0. Mục tiêu và phạm vi

**Tên đề tài:** Nghiên cứu cải tiến thuật toán K-Nearest Neighbors dựa trên trọng số đặc trưng, trọng số hàng xóm và xử lý dữ liệu hỗn hợp trong hỗ trợ phân loại bệnh thận mạn tính (CKD).

**Bài toán:** Phân loại nhị phân `CKD=1`, `notCKD=0` từ bộ UCI Chronic Kidney Disease (400 bản ghi; 24 đặc trưng; có dữ liệu thiếu, biến liên tục, nhị phân và thứ bậc). Đây là nghiên cứu học máy, **không phải hệ thống chẩn đoán y khoa**.

**Repo tham khảo:** https://github.com/Abdelkerim-Dassi/Chronic-Kidney-Disease-Prediction

**Dataset chính thức:** https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

**Không sao chép máy móc** notebook của repo tham khảo. Cần phân biệt các kết quả tác giả báo cáo với kết quả tái lập; nếu khác nhau, lưu lại cách chia dữ liệu, phiên bản thư viện và lý do có thể xảy ra.

### Các mức công việc

1. **Baseline bắt buộc:** tái lập KNN `k=3`, các đối chứng feature selection RFECV/CFS nếu khả thi; đánh giá lại với pipeline không rò rỉ dữ liệu. Đây là nền để so sánh, không phải mục tiêu nghiên cứu cuối.
2. **Cấp độ 2 — Cải tiến chính:** tự cài đặt Feature-Weighted Distance KNN, Distance-Weighted Voting KNN, mô hình kết hợp hai cơ chế; kiểm chứng công thức bằng unit test.
3. **Cấp độ 3 — Nghiên cứu sâu:** đề xuất/học trọng số nhóm đặc trưng cho dữ liệu hỗn hợp; xử lý thiếu trong pipeline; lựa chọn ngưỡng với mục tiêu hạn chế false negatives nhưng kiểm soát false positives; đánh giá ablation và độ ổn định.

### Câu hỏi nghiên cứu

- **RQ1:** Sau khi loại bỏ rò rỉ dữ liệu, KNN gốc còn đạt hiệu năng thế nào?
- **RQ2:** Trọng số đặc trưng có thay đổi thứ tự hàng xóm và cải thiện Recall/F1/PR-AUC trên CKD không?
- **RQ3:** Biểu quyết theo khoảng cách có cải thiện so với biểu quyết đều không?
- **RQ4:** Cơ chế trọng số theo **nhóm đặc trưng gốc** có ổn định hơn việc gán trọng số độc lập cho mọi cột one-hot không?
- **RQ5:** Có thể giảm FN ở mức specificity/precision chấp nhận được bằng lựa chọn ngưỡng trên validation không?
- **RQ6:** Những cải thiện có ổn định qua nhiều lần chia và khi giả lập tỷ lệ dữ liệu thiếu tăng không?

### Giả thuyết (chưa phải kết luận)

- H1: KNN có trọng số đặc trưng vượt KNN không trọng số trên một hoặc nhiều metric tiền định.
- H2: Distance-weighted voting có thể cải thiện kết quả đối với một số vùng lân cận.
- H3: Học trọng số nhóm biến và sử dụng khoảng cách phù hợp dữ liệu hỗn hợp tốt hơn naive one-hot Euclidean trong một số cấu hình.
- H4: Chọn threshold theo validation có thể giảm FN nhưng sẽ làm thay đổi FP; cần báo cáo cả hai.

---

## 1. Nguyên tắc bắt buộc cho Codex

1. **Không data leakage:** chia dữ liệu trước khi `.fit()` imputer, scaler, encoder, feature selector, MI, trọng số, threshold. Mọi phép học tham số phải nằm trong train fold của cross-validation. Test set không được dùng để chọn thiết kế hay siêu tham số.
2. **Không điền thiếu theo nhãn thật** (`class`). Không sử dụng `class` hoặc `id` như đầu vào. Không fit tiền xử lý trên toàn bộ dataset.
3. **Không tuyên bố kết quả 97.5%, 98.33%, 99.17% là kết quả tái lập** nếu chưa chạy và đo. Đó là kết quả tác giả repo tham khảo báo cáo, có thể khác với kết quả pipeline nghiêm ngặt.
4. **Không ép mọi biến thành số liên tục một cách tùy tiện:** xử lý theo schema. `sg`, `al`, `su` cần thử ít nhất một cách mã hóa có lý giải; biến nominal có thể dùng one-hot; xem xét group weights khi one-hot có nhiều cột.
5. **Không chỉnh monkey-patch phương thức toàn cục của `KNeighborsClassifier` để ghép AdaBoost.** AdaBoost là baseline tham khảo tùy chọn, không phải đóng góp chính. Nếu thực hiện, cần estimator tương thích `sample_weight` và unit test nghiêm túc; nếu không, ghi rõ không tái lập.
6. **Không dùng `RFECV(KNeighborsClassifier(...))` trực tiếp** vì estimator không có `coef_`/`feature_importances_`. Dùng estimator nền phù hợp, kiểm tra nested CV.
7. **Không khẳng định tính mới học thuật tuyệt đối**: Feature-Weighted KNN và Distance-Weighted KNN đã tồn tại. Đóng góp dự kiến là cấu hình/phương pháp học trọng số nhóm hoặc cost-aware pipeline được thiết kế, giải thích, kiểm chứng cẩn thận.
8. **Không trộn metric:** recall lớp CKD là metric chính; luôn đi kèm precision, specificity, F1, balanced accuracy, ROC-AUC, PR-AUC, confusion matrix và FN/FP tuyệt đối. ROC/PR-AUC cần score liên tục, không dùng nhãn nhị phân.
9. **Khi thay đổi code:** thêm/chỉnh test, chạy test, ghi kết quả/ảnh hưởng trong CHANGELOG hoặc log milestone. Không tự bịa metric, hình vẽ, dataset, kết quả ablation.
10. **Bảo mật và đạo đức:** dữ liệu chỉ phục vụ nghiên cứu; không hiển thị ứng dụng như công cụ chẩn đoán đã được chứng nhận; ghi cảnh báo ở UI/README.

---

## 2. Công nghệ

- Python 3.11 hoặc 3.12 (chọn phiên bản hỗ trợ đầy đủ các thư viện, khóa phiên bản trong `requirements.lock` hoặc `uv.lock`).
- NumPy, pandas, scikit-learn, SciPy, matplotlib, seaborn (chỉ dùng offline để vẽ biểu đồ từ kết quả thật), joblib.
- `ucimlrepo` để lấy dataset gốc nếu mạng cho phép; hỗ trợ CSV tải thủ công.
- `pytest`, `ruff`, `mypy` (mypy có thể áp dụng dần), `pyyaml`, `tqdm`.
- `streamlit` cho ứng dụng demo sau khi pipeline và kiểm thử hoàn thành.
- `optuna` **tùy chọn** cho tìm trọng số/phạm vi siêu tham số; có baseline đơn giản trước và giới hạn số trial để tránh overfit 400 mẫu.
- Không dùng GPU, deep learning, cơ sở dữ liệu hoặc API cloud nếu không thật sự cần.

---

## 3. Kiến trúc thư mục bắt buộc

```text
CKD-Weighted-KNN/
├── CODEX_PROJECT.md
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data.yaml
│   ├── baseline.yaml
│   ├── weighted_knn.yaml
│   └── deep_research.yaml
├── data/
│   ├── raw/                 # dữ liệu gốc, không sửa trực tiếp
│   └── README.md
├── src/ckd_knn/
│   ├── __init__.py
│   ├── data.py             # fetch/load, schema, clean text/missing tokens
│   ├── schema.py           # định nghĩa nhóm cột và nhãn
│   ├── split.py            # holdout + CV, indices cố định
│   ├── preprocessing.py    # ColumnTransformer / pipelines
│   ├── metrics.py          # confusion matrix, FN, FP, thresholds
│   ├── evaluate.py         # CV, benchmark, thống kê
│   ├── feature_selection.py # RFECV/CFS, train-only
│   ├── weights.py          # MI, group weights, optimize weights
│   ├── distances.py        # Euclidean, weighted Euclidean, mixed distance
│   ├── models/
│   │   ├── baseline.py
│   │   ├── weighted_knn.py
│   │   └── mixed_cost_knn.py
│   └── plots.py
├── scripts/
│   ├── 01_audit_data.py
│   ├── 02_train_baselines.py
│   ├── 03_feature_selection.py
│   ├── 04_train_weighted_knn.py
│   ├── 05_train_deep_model.py
│   ├── 06_run_ablation.py
│   ├── 07_evaluate_holdout.py
│   └── 08_export_model.py
├── notebooks/
│   └── exploration.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_distances.py
│   ├── test_weighted_knn.py
│   ├── test_metrics.py
│   ├── test_no_leakage.py
│   └── test_reproducibility.py
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── experiment_log.csv
│   ├── baseline.md
│   ├── improvement.md
│   └── deep_research.md
├── models/                 # artifacts, tránh commit file lớn
└── app/
    └── streamlit_app.py
```

`AGENTS.md`: chỉ dẫn ngắn cho Codex: đọc `CODEX_PROJECT.md`, không thay đổi protocol split/test, luôn chạy test và báo cáo files modified. Nếu repo khởi đầu trống, tạo cấu trúc ở trên trong milestone 0.

---

## 4. Data specification và chống rò rỉ

### 4.1 Nguồn dữ liệu

UCI dataset ID 336, 400 mẫu, 24 đặc trưng + nhãn `class`. Lưu file gốc, nguồn tải, thời điểm lấy, checksum SHA256 và số dòng/column trong báo cáo. Không tin mọi biến thể CSV trên mạng đều giống nhau.

- Chuẩn hóa chuỗi `?`, khoảng trắng, tab và ký tự bất thường về NaN.
- Nhãn: chuẩn hóa chuỗi `ckd`, `notckd` thành `1`, `0`. Những nhãn không hợp lệ phải báo lỗi.
- Kiểm tra tên cột và schema thực tế: `wc`/`rc` theo UCI có thể khác `wbcc`/`rbcc` trong repository khác.
- Không xóa dòng/đặc trưng do thiếu dữ liệu mà không lưu thống kê và lý do.
- Kiểm tra trùng lặp toàn bộ bản ghi; nếu có trùng, áp dụng protocol hạn chế train/test leakage, ghi rõ cách xử lý.
- Kiểm tra phân phối nhãn và missingness theo từng cột/lớp **chỉ để mô tả**, không dùng nhãn test để tạo tham số xử lý.

### 4.2 Kiểu biến

- Liên tục/số đếm: `age`, `bp`, `bgr`, `bu`, `sc`, `sod`, `pot`, `hemo`, `pcv`, `wc`, `rc` (xác nhận schema gốc).
- Nominal/binary: `rbc`, `pc`, `pcc`, `ba`, `htn`, `dm`, `cad`, `appet`, `pe`, `ane`.
- Có thứ bậc: `sg`, `al`, `su`; thực nghiệm hai cách biểu diễn khi có cơ sở, ghi giải thích.
- Chỉ số ID `id` (nếu có) phải loại khỏi đầu vào.

### 4.3 Protocol split

- Một holdout **70% train/dev, 30% test**, `stratify=y`, `random_state=42`; lưu row IDs/hashes tại `reports/split_manifest.json`.
- Trong train/dev, sử dụng 5-fold stratified CV để chọn tham số. Với các thí nghiệm cần báo cáo độ ổn định, dùng repeated CV trên train/dev, ví dụ 5 folds × 5 repeats.
- Nếu tối ưu nhiều cấu hình rồi dùng CV scores để tuyên bố performance, phải dùng **nested CV** hoặc đánh giá holdout một lần sau khi khóa mô hình. Không nhìn test rồi chỉnh tham số.
- Ghi rõ mọi seed, số mẫu mỗi tập, phiên bản thư viện.

### 4.4 Tiền xử lý

- Dùng `ColumnTransformer` trong `Pipeline`.
- Numeric: `SimpleImputer(strategy="median", add_indicator=True)` tùy cấu hình; `StandardScaler` hoặc `MinMaxScaler` fit chỉ trên train fold.
- Nominal: `SimpleImputer(strategy="most_frequent")`, `OneHotEncoder(handle_unknown="ignore")`.
- Ordinal: xử lý giá trị thiếu và `OrdinalEncoder` với thứ tự được định nghĩa minh bạch; nếu không chắc về ordinal nghĩa y khoa, so sánh one-hot.
- Sau transform, các feature names / mapping từ cột gốc sang cột encoded phải được lưu để học group weights.
- **KNNImputer** là ablation độc lập và không được fit ngoài train fold. Không nhầm `KNNImputer` với classifier KNN.
- Test `pipeline.fit(X_train,y_train)` và `pipeline.predict(X_test)` ngay cả khi test có category chưa từng xuất hiện ở train.

---

## 5. Milestone 0 — Khởi tạo và data audit

**Nhiệm vụ**

- [ ] Tạo repo, virtual environment, `pyproject.toml`/`requirements.txt`, `README`, `AGENTS.md`, `CODEX_PROJECT.md`.
- [ ] Implement `data.py`, `schema.py`, `01_audit_data.py`.
- [ ] Tải dataset UCI, ghi nguồn, checksum, số mẫu, phân bố nhãn, tỷ lệ missing từng cột, data types, duplicates.
- [ ] Tạo `split_manifest.json` đúng train/test stratified.
- [ ] Viết `test_data.py`, `test_no_leakage.py` cho schema, nhãn, ID, missing tokens.

**Hoàn thành khi:** dữ liệu đọc được, xác nhận 400 mẫu/24 đặc trưng (hoặc báo cáo minh bạch mọi khác biệt), có audit report, test pass.

---

## 6. Milestone 1 — Baseline tái lập Project B

**Mục tiêu:** Tạo baseline chuẩn để so sánh cải tiến, không ép phải giống hệt Accuracy trong README Project B.

### Mô hình

- B0: `DummyClassifier(strategy="most_frequent")` — tối thiểu phải đánh bại baseline trivial.
- B1: `KNeighborsClassifier(n_neighbors=3, weights="uniform", metric="minkowski", p=2)` trong pipeline leakage-safe.
- B2: KNN tối ưu `n_neighbors` trong `[1,3,5,7,9,11,15,21]` với điều kiện K không vượt số mẫu train mỗi fold; thử `p=[1,2]`, `weights=["uniform","distance"]`.
- B3: KNN + RFECV: RFECV dùng estimator có `coef_`/`feature_importances_` (ví dụ `LogisticRegression`/`RandomForest` thích hợp), giữ nguyên train-only và cân nhắc nested CV.
- B4: KNN + CFS: cài đặt hoặc dùng thư viện đáng tin; document chính xác biến thể CFS, cách tính correlation cho dữ liệu mixed, search strategy, và tất cả bước được fit bên trong CV. Nếu không tái lập chính xác CFS từ Project B, đánh dấu `approximate reproduction`, không gắn nhãn `exact reproduction`.

### Đầu ra

- `reports/baseline.md`: nguồn, cấu hình, preprocessing, CV protocol, metric mean±std, holdout (khi đến milestone test cuối), so sánh kết quả tác giả (có nhãn `reported, not reproduced`).
- `reports/tables/baselines.csv`, biểu đồ scores theo K, confusion matrix và các metric.
- Không tuning trên test.

**Hoàn thành khi:** chạy lại bằng CLI được, seed cố định cho kết quả tái lập, có báo cáo baseline và test.

---

## 7. Milestone 2 — Cải tiến Feature-Weighted + Distance-Weighted KNN (trọng tâm)

### 7.1 Công thức thuật toán

**Feature-weighted Euclidean:**

```math
d_w(x,z)=\sqrt{\sum_{j=1}^{p} w_j(x_j-z_j)^2}, \quad w_j\ge0,\quad \sum_j w_j=1.
```

- Nếu `w_j=0` thì đặc trưng không tham gia khoảng cách; giải thích đây có thể là pseudometric trên không gian gốc.
- Khi mọi `w_j` bằng nhau (hoặc tất cả bằng 1), thứ tự hàng xóm phải bằng Euclidean chuẩn đối với cùng dữ liệu tiền xử lý.
- Không hard-code trọng số lâm sàng theo cảm tính.

**Distance-weighted voting:**

```math
v_i=\frac{1}{d_w(x,x_i)+\epsilon}, \quad
s_c(x)=\sum_{i\in N_K(x;w)} v_i\mathbb{1}[y_i=c], \quad
\hat y=\arg\max_c s_c(x).
```

- Trường hợp `distance=0`: định nghĩa ổn định (nếu có hàng xóm khoảng cách 0, chỉ vote trong nhóm khoảng cách 0; hoặc epsilon và giải thích). Test explicit.
- `predict_proba` trả về tỷ lệ **weighted vote**, không gọi là xác suất lâm sàng đã được hiệu chuẩn.
- Xử lý tie-break nhất quán, không phụ thuộc thứ tự input; định nghĩa tie policy rõ ràng.

### 7.2 Bắt buộc tự cài đặt `WeightedKNNClassifier`

API tương thích scikit-learn cơ bản:

```python
model = WeightedKNNClassifier(
    n_neighbors=5,
    feature_weights=None,   # None -> equal weights
    vote_weights="uniform", # hoặc "distance"
    epsilon=1e-8,
    metric="weighted_euclidean",
)
model.fit(X_train_transformed, y_train)
pred = model.predict(X_test_transformed)
score = model.predict_proba(X_test_transformed)
```

- Kế thừa `BaseEstimator`, `ClassifierMixin` theo chuẩn phiên bản sklearn đã khóa; hỗ trợ `get_params`, `set_params`, `classes_`, `n_features_in_`, `fit/predict/predict_proba`.
- Không truy cập dữ liệu labels của test trong `fit`.
- Tách `distances.py` khỏi model và dùng NumPy vectorized/chunking; tránh memory bùng nổ nếu mở rộng dataset.
- Validate K dương, K <= n_train; shape, NaN sau transform, non-negative finite weights, tổng weights >0, duplicate/ties.
- Viết test so sánh với sklearn KNN đối với equal weights và uniform/distance voting trên dữ liệu toy không hòa, không trùng mẫu.
- Tạo visualization toy chứng minh khi thay trọng số thì thứ tự hàng xóm thay đổi.

### 7.3 Thực nghiệm bắt buộc

- M1: KNN gốc (equal feature weights + uniform votes).
- M2: KNN gốc + distance voting.
- M3: feature weights + uniform votes.
- M4: feature weights + distance voting (combined).

**Feature weight initialization:**

- EQUAL: mọi encoded feature có trọng số bằng nhau.
- MI: ước lượng Mutual Information **chỉ trên train fold**, đảm bảo cách khai báo biến discrete/continuous đúng với representation; normalize nonnegative scores, có epsilon-smoothing, zero-score fallback.
- Nếu thêm Random Forest feature importance, là baseline mở rộng và phải report riêng.

**Hyperparameter tuning:** K, `vote_weights`, scaler, cách tính/chuẩn hóa weights chỉ được chọn qua CV train/dev. Giữ search space vừa phải, reproducible, logging đầy đủ.

**Hoàn thành khi:** M1–M4 chạy thành công, unit tests chứng minh công thức, có bảng metric và ablation M1→M4, có file model (sau final refit).

---

## 8. Milestone 3 — Nghiên cứu sâu: Mixed-Type, Missingness, Cost-Aware KNN

**Mục tiêu:** phát triển một biến thể có lý do phương pháp rõ hơn việc bật `weights="distance"` của sklearn.

### 8.1 Nghiên cứu A: Group-aware feature weights

Vấn đề: một biến nominal sau one-hot có thể sinh nhiều cột, từ đó ảnh hưởng của một biến gốc vào khoảng cách không được kiểm soát bằng cách gán trọng số tùy tiện cho từng cột.

**Phương án đề xuất:** trọng số theo **nhóm biến gốc**, ví dụ `htn -> {htn_no, htn_yes}`, `sg -> các mức sg`.

```math
d^2(x,z)=\sum_{g=1}^{G}\lambda_g D_g(x_g,z_g), \quad \lambda_g\ge0, \quad \sum_g\lambda_g=1.
```

- Với numeric: `D_g = ((x_g-z_g)/scale_g)^2`, scaler học từ train.
- Với binary nominal: `D_g = 1[x_g != z_g]` (thử nghiệm baseline; mismatch 0/1).
- Với multi-category nominal: 0 nếu trùng, 1 nếu khác; hoặc scaled one-hot distance nhưng **mỗi biến gốc chỉ có một trọng số nhóm**.
- Với ordinal: normalized numeric distance hoặc ordinal mismatch, so sánh hai lựa chọn.
- Nếu impute khiến hai giá trị được điền giống nhau giả tạo, có thể thử missing-indicator features hoặc missingness penalty **không sử dụng nhãn test**.
- Để so sánh công bằng, giữ tổng trọng số nhóm cố định và báo cáo mỗi thành phần `D_g` riêng.

### 8.2 Nghiên cứu B: Cách học trọng số nhóm

Các baseline:

1. Equal group weights.
2. Group-level Mutual Information dựa trên tập train (định nghĩa aggregation minh bạch).
3. Tối ưu một tập nhỏ các trọng số nhóm bằng validation/CV; ưu tiên group/bucket để hạn chế 24 tham số tự do trên 400 mẫu.

**Giới hạn tránh overfit:** các trọng số không âm, chuẩn hóa sum=1; regularize toward equal weights; optimize trên inner CV; tối đa trial được ghi trong YAML; không nhìn test; kiểm tra stability giữa folds. Không dùng 24 weights tự do làm mặc định.

### 8.3 Nghiên cứu C: Missing data robustness

Ablation độc lập:

- Numeric median + categorical mode.
- Median/mode + missing indicators.
- KNNImputer cho numeric (trong train pipeline) + mode cho categorical.
- Artificial masking: chỉ mask có kiểm soát trên validation/test bản sao, không học gì từ holdout; thử ví dụ 5%, 10%, 20% giá trị quan sát, nhiều seed; không mask target. Ghi rõ đây là missing-at-random simulation, không đồng nghĩa missingness lâm sàng thực tế.

Báo cáo performance theo missing-rate cùng CI/dispersion; không lựa chọn mô hình dựa trên kết quả holdout masked nếu holdout đã được dùng như kiểm định cuối.

### 8.4 Nghiên cứu D: Cost-sensitive/threshold-aware classification

Định nghĩa positive class = CKD.

- Tạo score CKD bằng normalized neighbor votes; threshold mặc định 0.5.
- Xác định threshold trên inner-validation hoặc out-of-fold predictions **chỉ train/dev**.
- Một mục tiêu ví dụ: maximize Recall(CKD) subject to `Specificity >= target_min`, với `target_min` do config định nghĩa; nếu không khả thi, report infeasible chứ không giấu.
- So sánh threshold=0.5 và tuned threshold trên cùng model.
- Bổ sung cost ratio FN:FP (ví dụ 5:1, 10:1) như **kịch bản nghiên cứu**, không coi là chính sách y khoa được xác nhận. Báo cáo sensitivity analysis trên nhiều cost ratio.
- Nếu dùng calibration, fit bằng CV chỉ train/dev; phân biệt weighted neighbor fraction với calibrated probability.

### 8.5 Kết hợp và ablation

- D0: M4 từ milestone 2.
- D1: D0 + group-aware mixed-type distance.
- D2: D1 + learned group weights.
- D3: D2 + missingness handling (một cấu hình đã chọn bằng train CV).
- D4: D3 + threshold optimized for screening constraints.

**Không kết luận D4 tốt hơn chỉ từ accuracy.** Báo cáo Recall, FN, specificity, precision, F1, PR-AUC và biến động qua CV. Thực hiện ablation từng thành phần, ghi số lần fit và chi phí tính toán.

**Hoàn thành khi:** có công thức, giải thích nguồn weights, code kiểm thử, báo cáo ablation D0–D4, phân tích trường hợp thất bại và giới hạn khả năng khái quát.

---

## 9. Evaluation protocol và báo cáo kết quả

### 9.1 Metrics

- Confusion matrix: TP, TN, FP, FN với CKD=positive.
- Accuracy; balanced accuracy; Recall (sensitivity), Precision, F1, Specificity; ROC-AUC; PR-AUC/average precision (nói rõ cách tính).
- Báo cáo metrics theo CV: mean ± std, có thể thêm 95% bootstrap CI thích hợp khi báo cáo holdout; không diễn giải CI quá mức do mẫu nhỏ/selection bias.
- Báo cáo throughput/predict latency, số cột encoded, số nhóm biến gốc, số biến được chọn.
- Threshold phải được chọn trước khi final holdout; báo cáo FP/FN tuyệt đối để tránh bị đánh lừa bởi accuracy.

### 9.2 So sánh công bằng

- Dùng chính xác cùng các fold và split cho các mô hình trong cùng thí nghiệm.
- Không chọn model sau khi nhìn final holdout; nếu thấy lỗi code thì sửa, re-run và ghi rõ lịch sử, không che việc dùng test nhiều lần.
- Lưu experiment configs, dataset hash, seeds, Python/package versions, git commit, timestamp và kết quả dưới `reports/experiment_log.csv`.
- Nếu kết quả không cải thiện, báo cáo trung thực nguyên nhân: dữ liệu ít, leakage được loại bỏ, trọng số overfit, nhiễu nhãn, hoặc giảm feature gây mất thông tin.

### 9.3 File kết quả

- `reports/tables/baselines.csv`
- `reports/tables/weighted_ablation.csv`
- `reports/tables/deep_ablation.csv`
- `reports/tables/fold_metrics.csv`
- `reports/figures/knn_vs_k.png`
- `reports/figures/ablation_recall_specificity.png`
- `reports/figures/roc_pr_curves.png`
- `reports/figures/confusion_matrices.png`
- `reports/figures/missingness_robustness.png`
- `reports/figures/group_weights.png`
- `reports/improvement.md` và `reports/deep_research.md` giải thích bằng tiếng Việt.

Chỉ tạo figure khi có kết quả thật; ảnh phải có title, trục, chú giải và nguồn thí nghiệm.

---

## 10. Tests tối thiểu

- [ ] Khoảng cách weighted Euclidean khớp tính tay cho vector nhỏ.
- [ ] Equal feature weights cho cùng thứ tự nearest neighbors với Euclidean chuẩn; scale factor đồng nhất không ảnh hưởng thứ tự.
- [ ] Distance voting khớp tính tay, bao gồm trường hợp 2/5 hàng xóm gần có thể thắng 3/5 hàng xóm xa.
- [ ] Duplicate zero-distance samples không tạo infinity/NaN; tie resolution deterministic.
- [ ] Sai shape, K không hợp lệ, trọng số âm/NaN/tất cả bằng 0 tạo lỗi rõ ràng.
- [ ] Schema cleaning xử lý `?`, `\t?`, whitespace, nhãn `ckd`/`notckd`.
- [ ] Data leakage test: thay đổi hoàn toàn giá trị trong test không làm thay đổi statistics fit của train preprocessing/weights.
- [ ] RFECV/CFS/MI không `fit` trên test; kiểm tra bằng instrumentation/mock hoặc kiểm tra pipeline/cv fit boundaries.
- [ ] Reproducibility test cùng seed ra cùng chỉ số trên toy data.
- [ ] Threshold tuning không đọc `y_test` và không tối ưu trên holdout.
- [ ] Tests về mixed feature groups: one-hot expansion không vô tình nhân trọng số một biến; symmetric, nonnegative distance, self-distance=0 với vector không thiếu.
- [ ] End-to-end smoke test trên toy dataset synthetic nhỏ, chạy CLI → report → model artifact → predict.

---

## 11. CLI và hướng dẫn chạy (Codex phải hiện thực)

Cung cấp CLI dạng:

```bash
python scripts/01_audit_data.py --config configs/data.yaml
python scripts/02_train_baselines.py --config configs/baseline.yaml
python scripts/03_feature_selection.py --config configs/baseline.yaml
python scripts/04_train_weighted_knn.py --config configs/weighted_knn.yaml
python scripts/05_train_deep_model.py --config configs/deep_research.yaml
python scripts/06_run_ablation.py --config configs/deep_research.yaml
python scripts/07_evaluate_holdout.py --config configs/deep_research.yaml
python scripts/08_export_model.py --config configs/deep_research.yaml
pytest -q
streamlit run app/streamlit_app.py
```

- Script phải có `--help`, exit code đúng, log rõ config/data missing, **không tự tạo kết quả giả nếu không thấy dataset**.
- Final test command phải có cờ/guard tránh chạy lại vô tình. Mọi model trước khi đánh giá test đã phải được chọn và đóng băng cấu hình.
- `README.md` phải ghi rõ cách tạo venv, cài, lấy dữ liệu, lệnh chạy từng milestone trên Windows PowerShell và trên Linux/macOS.

---

## 12. Streamlit demo — chỉ làm sau thí nghiệm

Tính năng:

- Chọn model baseline / weighted / deep đã export thành công.
- Nhập giá trị các đặc trưng hợp lệ; có thể bỏ trống những biến được phép thiếu; validate min/max/type và categories.
- Hiển thị nhãn dự đoán, CKD weighted vote score (không gọi đây là nguy cơ lâm sàng đã hiệu chuẩn), ngưỡng đang dùng và thông điệp giới hạn nghiên cứu.
- Nếu model cho phép, hiển thị top-K hàng xóm dưới dạng bảng thông tin đã ẩn nhận diện, khoảng cách, trọng số vote; không trình bày thông tin cá nhân thật.
- Có trang `About model`: UCI dataset, split, metric thật, giới hạn validation, không dùng để chẩn đoán.
- Tải artifact model từ file riêng cùng schema và version; không refit khi user mở web.

---

## 13. Tiêu chí nghiệm thu

### Cấp độ 2 hoàn thành khi:

- Có baseline leakage-safe và báo cáo so sánh.
- Có `WeightedKNNClassifier` tự cài đặt + tests toán học.
- Có M1–M4, cùng fold/seed, bảng Recall/F1/Specificity/FN.
- Có giải thích rõ Feature Weight vs Neighbor Weight bằng ví dụ số.
- Không tuyên bố cải thiện nếu metric thực nghiệm không xác nhận.

### Cấp độ 3 hoàn thành khi:

- Có group-aware distance xử lý tốt nominal/numeric/ordinal.
- Có chiến lược học group weights có regularization và train-only CV.
- Có nghiên cứu missingness robustness.
- Có threshold/cost-aware experiment và trade-off FN–FP.
- Có D0–D4 ablation + error analysis + limitations, không giả tạo ý nghĩa thống kê.
- Có CLI, test, tài liệu, báo cáo và demo chạy được.

---

## 14. Quy trình làm việc bắt buộc của Codex

**Không cố triển khai toàn bộ trong một lượt nếu thiếu dữ liệu hoặc môi trường.** Làm theo thứ tự, nhưng chủ động triển khai milestone kế tiếp khi milestone hiện tại có thể kiểm chứng được.

1. Đọc `CODEX_PROJECT.md` và kiểm tra workspace. Liệt kê trạng thái: repo trống hay đã có code, tệp dữ liệu nào có thật, Python và thư viện nào sẵn.
2. Lập checklist milestone (0 → 1 → 2 → 3 → demo). Ưu tiên milestone 2/3 nhưng không bỏ baseline.
3. Tạo thiết kế ngắn cho mỗi milestone; implement code sạch, tài liệu và tests tương ứng.
4. Chạy lệnh lint/tests/smoke khi môi trường cho phép, chỉ đưa ra metrics nếu script đã thực sự chạy.
5. Sau mỗi milestone, ghi: file đã tạo/sửa, lệnh chạy, kết quả đo, điều chưa hoàn thành, bước tiếp theo.
6. Nếu không có mạng/dataset: tạo code fetch + hướng dẫn tải thủ công, unit tests dùng dữ liệu synthetic, **không bịa báo cáo trên UCI**.
7. Tránh thay đổi split/protocol âm thầm. Mọi quyết định lớn phải được phản ánh trong YAML và báo cáo.

### Prompt mở đầu để dùng với Codex

> Hãy đọc toàn bộ `CODEX_PROJECT.md` và `AGENTS.md`. Đây là dự án nghiên cứu CKD Weighted KNN. Bắt đầu bằng kiểm tra workspace, khởi tạo cấu trúc, dataset audit, baseline chống leakage, sau đó xây dựng `WeightedKNNClassifier` và hệ thống kiểm thử. Tuân thủ hoàn toàn quy trình split/CV/test và không tự bịa metrics. Chủ động implement và chạy các bài test có thể chạy, ghi báo cáo milestone bằng tiếng Việt. Chưa cần Streamlit cho đến khi hoàn tất thuật toán và thực nghiệm.

---

## 15. Tài liệu tham khảo kỹ thuật

- Repository tham khảo: https://github.com/Abdelkerim-Dassi/Chronic-Kidney-Disease-Prediction
- Dữ liệu CKD chính thức: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease
- RFECV: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
- ColumnTransformer: https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
- Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

**Lưu ý cuối:** Các phương pháp weighted KNN, feature selection và cost-sensitive learning không phải ý tưởng chưa từng tồn tại. Nếu dùng chữ “đề xuất”, cần nêu rõ thành phần kỹ thuật cụ thể do dự án thiết kế, so sánh với baseline/các biến thể phổ biến, và diễn giải kết quả theo bằng chứng thực nghiệm.
