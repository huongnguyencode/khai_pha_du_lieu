# File Excel khách hàng giả lập để test Thận An

Tất cả hồ sơ đều là **giả lập**. Các nhóm có/không dấu hiệu dưới đây là **đầu ra đã kiểm bằng W2**, không phải kết luận có bệnh thật hoặc chắc chắn không bệnh.

| File | Số người | Target 0 | Target 1 | Target 2 | Target 3 |
|---|---:|---:|---:|---:|---:|
| [01_hon_hop_20_nguoi.xlsx](01_hon_hop_20_nguoi.xlsx) | 20 | 10 | 3 | 3 | 4 |
| [02_chua_goi_y_bat_thuong_10_nguoi.xlsx](02_chua_goi_y_bat_thuong_10_nguoi.xlsx) | 10 | 10 | 0 | 0 | 0 |
| [03_co_dau_hieu_10_nguoi.xlsx](03_co_dau_hieu_10_nguoi.xlsx) | 10 | 0 | 3 | 3 | 4 |
| [04_albumin_nieu_10_nguoi.xlsx](04_albumin_nieu_10_nguoi.xlsx) | 10 | 0 | 10 | 0 | 0 |
| [05_loc_than_giam_10_nguoi.xlsx](05_loc_than_giam_10_nguoi.xlsx) | 10 | 0 | 0 | 10 | 0 |
| [06_ca_hai_dau_hieu_10_nguoi.xlsx](06_ca_hai_dau_hieu_10_nguoi.xlsx) | 10 | 0 | 0 | 0 | 10 |
| [07_co_dau_hieu_kem_tien_su_tieu_duong_10_nguoi.xlsx](07_co_dau_hieu_kem_tien_su_tieu_duong_10_nguoi.xlsx) | 10 | 0 | 5 | 1 | 4 |

Target 0: chưa gợi ý bất thường. Target 1: chỉ albumin niệu. Target 2: chỉ giảm khả năng lọc. Target 3: đồng thời cả hai dấu hiệu.

## Cách dùng

1. Mở **http://localhost:8090** → **Danh sách Excel**.
2. Chọn file cần thử và bấm **Phân tích danh sách**.
3. Xem sheet **Doi_chieu** trong file để đối chiếu từng mã hồ sơ và số lời khuyên mong đợi. Web chỉ đọc **Du_lieu**.
4. Dòng nhóm 0 không có lời khuyên riêng. File 07 gồm 10 dòng có dấu hiệu thận và `DIQ010=1`, dự kiến mỗi dòng có thêm lời khuyên về tiền sử đái tháo đường.

Các file đã đủ đúng số người, không có dòng ví dụ thừa. Có thể sửa Du_lieu, nhưng khi sửa thì bảng Doi_chieu đã lưu không còn bảo đảm khớp.

Bộ hỗn hợp 20 người trộn thứ tự ngẫu nhiên: 10 chưa gợi ý + 10 có dấu hiệu; trong 10 có dấu hiệu gồm 3 nhóm 1, 3 nhóm 2, 4 nhóm 3. Các hồ sơ được dùng lại giữa các file để đối chiếu; toàn bộ bộ file có 40 id khác nhau, không phải 80 người độc lập.

## Kiểm tra và tái tạo

Đã mở lại cả 7 XLSX, kiểm tra đủ 23 feature, số người, mã hợp lệ, nhãn W2, sheet đối chiếu và API upload hiện tại. Không có dòng lỗi hoặc công thức Excel. Không cần train lại model.

Chi tiết số lượng và checksum: [kiem_tra_du_lieu.json](kiem_tra_du_lieu.json).

Tái tạo từ thư mục dự án:

```bash
PYTHONPATH=src:. python3 scripts/12_create_customer_test_workbooks.py
```

Sinh bằng seed cố định 20261101, không sao chép hàng dữ liệu NHANES. Các giá trị là giả định của bộ test, không phải thông số y khoa khuyến nghị. Hồ sơ được chọn sau khi xem đầu ra W2 để đảm bảo đủ các nhánh giao diện; **không dùng bộ này để tính accuracy hoặc chứng minh hiệu quả mô hình**.

## Kiểm thử trực tiếp trên web đang chạy

Đã gửi cả 7 file qua HTTP `/api/predict-xlsx` trong container `nhanes-w2-web-8090` và tải kết quả Excel tương ứng: 7/7 lượt upload, 7/7 lượt xuất kết quả thành công; số người từng nhóm khớp và không có dòng lỗi. Nhóm 0 không có lời khuyên riêng.

Có thể kiểm tra lại từ thư mục dự án khi web đang chạy:

```bash
python3 scripts/13_verify_customer_workbooks_on_web.py
```
