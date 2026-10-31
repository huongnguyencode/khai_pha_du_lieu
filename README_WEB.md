# Thận An — web dự đoán W2 trên Docker, cổng 8090

Ứng dụng tiếng Việt hỗ trợ nhập một người hoặc một danh sách `.xlsx`, chạy đúng artifact `D_W2_groups` và xuất kết quả Excel. Nhóm có dấu hiệu được giải thích pattern kèm gợi ý sức khỏe chung; nhóm 0 chỉ nhận kết quả, không có lời khuyên riêng.

## Chạy ứng dụng

Từ terminal:

```bash
cd /home/huongcode/khai_pha/khai_pha_du_lieu
docker build -f Dockerfile.web -t nhanes-w2-web:local .
docker run -d --name nhanes-w2-web-8090 --init --restart unless-stopped --read-only --tmpfs /tmp:size=64m,mode=1777 --cap-drop ALL --security-opt no-new-privileges:true -p 127.0.0.1:8090:8090 nhanes-w2-web:local
```

Mở **http://localhost:8090**. Chỉ lắng nghe ở localhost; không tự mở ra Internet/LAN. Khi container cùng tên đã tồn tại, dùng `docker start nhanes-w2-web-8090` để chạy lại, không chạy thêm `docker run` trùng tên.

Kiểm tra và xem log:

```bash
docker ps --filter name=nhanes-w2-web-8090
docker logs --tail 80 nhanes-w2-web-8090
curl http://localhost:8090/health
```

Health trả `{"status": "ok", "model": "W2", "features": 23}` khi artifact nạp được. Lần build đầu tải các thư viện Python; lúc dự đoán không cần Internet, API key hoặc dịch vụ AI ngoài.

Dừng / khởi động lại container đã tạo:

```bash
docker stop nhanes-w2-web-8090
docker start nhanes-w2-web-8090
```

### Lựa chọn Docker Compose

File [`compose.web.yaml`](compose.web.yaml) dành riêng cho web; không dùng `compose.yaml` cũ vì đó là pipeline train. Nếu container tạo bằng `docker run` ở trên đang chạy, dừng nó trước để giải phóng cổng 8090.

```bash
docker compose -f compose.web.yaml up -d --build
docker compose -f compose.web.yaml logs --tail 80 nhanes-web
docker compose -f compose.web.yaml down
```

Hai cách chạy là lựa chọn thay thế, không chạy đồng thời trên cùng cổng. Không có bước train khi build/start web. `Dockerfile.web.dockerignore` được dùng riêng để đưa artifact W2 vào image, do `.dockerignore` của pipeline cũ đang loại thư mục models.

## Cách dùng

### Nhập từng người

- Nhập mã/tên tùy chọn và các giá trị trong biểu mẫu. Mỗi trường ghi rõ nhãn, đơn vị hoặc danh mục lựa chọn.
- Tuổi từ 20 đến 120, giới tính và ít nhất 12/23 feature có giá trị là yêu cầu đầu vào của ứng dụng. **Mốc 12 là quy tắc chất lượng nhập liệu, chưa phải ngưỡng được kiểm định lâm sàng.** Tuổi từ 80 được quy về nhóm 80+ theo dữ liệu NHANES.
- Chưa biết một feature thì để trống; không nhập 0 để thay dữ liệu thiếu. Median/mode/scale và trọng số dùng đúng giá trị đã học trong artifact, không fit lại trên thông tin nhập.
- Nút “Điền ví dụ” tạo thông tin minh họa, không phải hồ sơ người bệnh. Nút “Nhập lại” xóa biểu mẫu và kết quả.
- Chọn “Phân tích dấu hiệu” để dự đoán; có thể tải kết quả thành `.xlsx`.

### Nhập Excel

1. Chuyển sang “Danh sách Excel”, tải file mẫu.
2. Điền sheet `Du_lieu`. Hai cột `id`, `name` tùy chọn; giữ đủ tiêu đề 23 feature theo mẫu. Sheet `Tu_dien` giải thích mã và đơn vị; sheet `Huong_dan` giải thích thao tác. Xóa/thay dòng ví dụ trước khi nhập hồ sơ thật.
3. Tải file `.xlsx`, tối đa 10 MB, 1.000 dòng dữ liệu. Chỉ nhận giá trị, không nhận công thức/macro.
4. Ứng dụng ưu tiên sheet `Du_lieu`; nếu không có thì đọc sheet đầu tiên. Bỏ qua cột metadata/nhãn ngoài 23 feature và thông báo số cột bỏ qua. Nhận `row_id` hoặc `SEQN` làm mã nếu không có `id`.
5. Lọc “Có dấu hiệu”, “Chưa gợi ý”, “Lỗi dữ liệu”; tìm theo mã/tên. Dòng lỗi hiển thị số dòng Excel và lỗi cần sửa, không bị tính vào nhóm không dấu hiệu.
6. Tải kết quả với các sheet `Tat_ca`, `Co_dau_hieu`, `Chua_goi_y`, `Loi_du_lieu`, `Luu_y`.

Các trường phân loại dùng **mã số theo từ điển NHANES**, không dùng tùy ý 0/1 hay chữ Có/Không. Ví dụ `DIQ010=1` là có tiền sử đái tháo đường, `=2` là không, `=3` là borderline; `current_smoking` lại dùng 0/1. XLSX mẫu có chú thích và dropdown mã hợp lệ.

Đơn vị cholesterol/HDL/acid uric là **mg/dL**, hemoglobin là **g/dL**. Ứng dụng không tự suy đoán hoặc chuyển từ mmol/L. `INDFMPIR` là tỷ số theo khảo sát Hoa Kỳ, không phải thu nhập bằng tiền; có thể để trống nếu không có dữ liệu tương ứng.

## Diễn giải đầu ra

| Target | Gợi ý của mô hình | Nội dung hiển thị |
|---:|---|---|
| 0 | Chưa gợi ý cả hai dấu hiệu | Chỉ kết quả; không có lời khuyên riêng |
| 1 | Albumin niệu | Giải thích dấu hiệu uACR và hướng dẫn chung |
| 2 | Khả năng lọc giảm | Giải thích dấu hiệu eGFR và hướng dẫn chung |
| 3 | Đồng thời hai dấu hiệu | Giải thích cả hai và hướng dẫn chung |

W2 phân loại pattern từ ngưỡng eGFR <60 và uACR ≥30; **không đo trực tiếp eGFR/uACR của người nhập, không xác nhận bệnh thận mạn, không phân giai đoạn và không dự đoán đái tháo đường**. Nhóm 0 không đồng nghĩa chắc chắn không bệnh. Không hiển thị điểm biểu quyết như phần trăm xác suất bệnh.

Lời khuyên là nội dung quy tắc cố định theo pattern và tiền sử được khai, không mạo danh bác sĩ, không kê đơn hay tự yêu cầu đổi thuốc. Nhắc đái tháo đường chỉ khi người nhập khai `DIQ010=1` và kết quả thận có dấu hiệu. Các nguồn hiển thị cùng kết quả:

- [NIDDK — Managing Chronic Kidney Disease](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/managing).
- [NIDDK — Healthy Eating for Adults with CKD](https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/healthy-eating-adults-chronic-kidney-disease).
- [NIDDK — Managing Diabetes](https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes).

Mô hình nghiên cứu học từ NHANES Hoa Kỳ; chưa có xác nhận hiệu quả trên quần thể Việt Nam. Lời khuyên cần được người có chuyên môn rà soát trước khi dùng trong cơ sở khám chữa bệnh.

## Cấu trúc mã

| File | Chức năng |
|---|---|
| `web/static/index.html`, `styles.css`, `app.js` | Giao diện responsive tiếng Việt, không cần Node khi chạy |
| `web/schema.py` | Nhãn, nhóm, đơn vị, mã và quy tắc kiểm tra 23 feature |
| `web/service.py` | Kiểm dữ liệu, xác minh checksum artifact, gọi W2 và tạo diễn giải |
| `web/excel.py` | Đọc XLSX giới hạn kích thước, tạo mẫu và xuất kết quả |
| `web/app.py` | WSGI API, static assets, healthcheck; Gunicorn phục vụ HTTP |
| `Dockerfile.web`, `requirements.web.txt` | Image web riêng, pin NumPy/SciPy theo môi trường W2 |
| `compose.web.yaml` | Cấu hình chạy web tại localhost:8090 |
| `web/tests/test_web.py` | Kiểm thử inference/Excel/WSGI không cần mở cổng |
| `web/tests/browser_smoke.py` | Kiểm thử Chromium trên web đang chạy |

Ứng dụng chỉ đọc artifact được đóng gói từ dự án, kiểm checksum file và source hash W2, không cho người dùng upload pickle. Hồ sơ/file tải lên được xử lý trong RAM, không ghi database, không lưu vào localStorage hoặc cookie. Excel xuất các trường mã/tên dưới dạng văn bản an toàn, không cho chúng trở thành công thức. Container chạy user không phải root, filesystem chỉ đọc và thư mục tạm RAM.

## Kiểm thử

Kiểm thử trong image web:

```bash
docker run --rm nhanes-w2-web:local python -m unittest web.tests.test_web -v
```

Hoặc dùng Python của máy nếu đã có NumPy/SciPy/openpyxl:

```bash
PYTHONPATH=src:. python3 -m unittest web.tests.test_web -v
node --check web/static/app.js
```

Kiểm thử trình duyệt bằng image riêng, không làm image web cài thêm Chromium:

```bash
docker build -f web/tests/Dockerfile.browser -t nhanes-w2-browser-test:local web/tests
docker run --rm --init --network container:nhanes-w2-web-8090 --shm-size=256m -v /home/huongcode/khai_pha/khai_pha_du_lieu/web/tests:/work/web/tests nhanes-w2-browser-test:local
```

Lệnh browser trên dùng chung network namespace với container `nhanes-w2-web-8090`, nên không phụ thuộc Docker Desktop có bật host networking hay không. Nếu chạy web bằng Compose hoặc tên khác, thay tên container cho đúng. Khi dùng một dịch vụ khác có thể đặt địa chỉ qua biến `NHANES_WEB_URL`. Ảnh được lưu ở `web/tests/screenshots/`, kết quả máy đọc được ở `web/tests/browser_results.json`.

Các kiểm thử dự đoán so sánh wrapper với hàm `w2.predict` gốc để kiểm tra tích hợp, không dùng điểm trên mẫu train để tuyên bố chất lượng lâm sàng. Báo cáo đánh giá mô hình nằm ở [tài liệu W0/W1/W2](PHAN_TICH_TOAN_BO_W0_W1_W2_NHANES_2017_2023.md).
