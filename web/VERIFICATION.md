# Kiểm chứng web Thận An

Ngày kiểm tra: 01/11/2026.

## Kết quả đã chạy

- Build thành công image `nhanes-w2-web:local` bằng `Dockerfile.web`.
- Khởi động container `nhanes-w2-web-8090`, publish `127.0.0.1:8090:8090`, có restart policy.
- HTTP `/`, `/app.js`, `/api/schema`, `/health` trả 200 trong mạng container. Health xác nhận W2 và 23 feature.
- **18/18** unit/integration tests chạy thành công trong chính image web; không mở cổng phụ cho các tests này.
- Chromium kiểm tra thành công: nhập từng người, input sai, nhóm 0 không có lời khuyên, XLSX upload/lọc/tìm/xuất, lời khuyên nhóm có dấu hiệu, tên chứa ký tự HTML được hiển thị như văn bản, mobile 390 px không tràn ngang và không có lỗi JavaScript.
- Các kiểm thử dùng dữ liệu minh họa, không ghi thêm hồ sơ người bệnh và không huấn luyện lại mô hình.

Nguồn bằng chứng: [`tests/browser_results.json`](tests/browser_results.json).

Ảnh giao diện:

- [Desktop](tests/screenshots/desktop.png).
- [Mobile](tests/screenshots/mobile.png).
- [Kết quả Excel](tests/screenshots/excel-results.png).
- [Diễn giải nhóm có dấu hiệu](tests/screenshots/positive-guidance.png).

## Cách kiểm thử trình duyệt

Chromium chạy trong image QA riêng và dùng chung network namespace với container web. `networkidle` timeout trong môi trường Docker/Chromium này dù trang và API đã phản hồi; kiểm thử ghi nhận hiện tượng đó trong JSON và xác nhận sẵn sàng qua DOMContentLoaded, các trường schema, health trạng thái sẵn sàng và phản hồi dự đoán thực tế. Không dùng một khoảng chờ cố định để khẳng định ứng dụng hoạt động.

Kiểm tra này không phải xác nhận lâm sàng, không phải kiểm thử tải nhiều người đồng thời và không xác nhận khả năng triển khai public. Phiên bản hiện tại phục vụ tại localhost. Hướng dẫn vận hành và giới hạn dữ liệu nằm ở [`../README_WEB.md`](../README_WEB.md).
