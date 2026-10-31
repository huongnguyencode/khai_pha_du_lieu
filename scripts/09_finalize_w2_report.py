"""Create the Vietnamese W2 handoff from measured and validated results."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/w2'


def main():
    result = json.loads((OUT / 'results.json').read_text())
    validation = json.loads((OUT / 'validation.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Validate W2 artifacts before finalizing the handoff')
    summary = {row['model']: row for row in result['summary']}
    a, b, c, d, e = result['protocol']['models']
    winner = result['best_model']
    full_delta = summary[winner]['macro_f1'] - summary[a]['macro_f1']
    group_delta = summary[e]['macro_f1'] - summary[c]['macro_f1']
    temporal = {row['model']: row for row in result['temporal']}
    temporal_delta = temporal[winner]['macro_f1'] - temporal[a]['macro_f1']
    folds = list(csv.DictReader((OUT / 'fold_metrics.csv').open()))
    counts = {model: Counter(json.loads(row['parameters'])['candidate']
                             for row in folds if row['model'] == model)
              for model in (d, e)}
    lines = ['# Cải thiện W2 trên NHANES 2017–2023', '',
             '## Kết quả chính', '',
             f"Cấu hình có macro-F1 outer trung bình cao nhất là **{winner}**, đạt **{summary[winner]['macro_f1']:.4f} ± {summary[winner]['macro_f1_std']:.4f}**; W1 gốc đạt **{summary[a]['macro_f1']:.4f}**. Chênh lệch **{full_delta:+.4f}**.", '',
             f"Temporal validation: cấu hình này đạt **{temporal[winner]['macro_f1']:.4f}**, chênh lệch **{temporal_delta:+.4f}** so với W1 gốc. Chu kỳ test đã được sử dụng trong quá trình nghiên cứu trước; đây không phải xác nhận external độc lập.", '',
             '## Nguồn cải thiện', '',
             f"- Chọn threshold chung và mở rộng lưới threshold (B − A): **{summary[b]['macro_f1']-summary[a]['macro_f1']:+.4f}** macro-F1.",
             f"- Tìm alpha/K theo mục tiêu bốn lớp và mở rộng K (C − B): **{summary[c]['macro_f1']-summary[b]['macro_f1']:+.4f}**.",
             f"- Trọng số nhóm không regularization (D − C): **{summary[d]['macro_f1']-summary[c]['macro_f1']:+.4f}**.",
             f"- Trọng số nhóm có regularization (E − C): **{group_delta:+.4f}**.", '']
    if group_delta <= 0:
        lines.append('Thí nghiệm này chưa cho thấy lợi ích bổ sung về macro-F1 trung bình của trọng số nhóm có regularization so với C. Không quy lợi ích của threshold/K cho trọng số nhóm, và không kết luận rằng mọi cách tối ưu nhóm đều không hiệu quả.')
    else:
        lines.append('Trọng số nhóm có regularization cải thiện macro-F1 trung bình so với C trong thí nghiệm này. Cần đọc độ biến thiên theo repeat, precision/recall và temporal validation trước khi xem đây là cải thiện ổn định.')
    lines += ['', '## Chi phí false positives', '',
              f"- W1 gốc: accuracy **{summary[a]['accuracy']:.4f}**, FP từ lớp 0 sang lớp 1–3 **{summary[a]['healthy_fp_per_1000']:.1f}/1.000 người**.",
              f"- {winner}: accuracy **{summary[winner]['accuracy']:.4f}**, FP từ lớp 0 sang lớp 1–3 **{summary[winner]['healthy_fp_per_1000']:.1f}/1.000 người**.",
              'Mẫu số là toàn bộ người được đánh giá trong fold. Các chỉ số trên là trung bình outer folds; số FP/FN tuyệt đối và precision/recall từng lớp nằm trong báo cáo chi tiết.', '',
              '## Kiểm chứng và tái lập', '',
              f"- Kiểm tra {validation['fold_model_evaluations']} model-fold evaluations và {validation['oof_prediction_rows']:,} OOF predictions; mỗi người chỉ xuất hiện một lần mỗi model/repeat.",
              f"- Sai khác lớn nhất giữa các metrics A và W1 đã lưu: **{validation['historical_w1_max_metric_difference']}**.",
              '- Đã kiểm tra trọng số không âm, tổng bằng 1; dự đoán từ artifact với dữ liệu thiếu; đối chiếu lại metrics từ predictions.',
              f"- Môi trường thực tế: `{result['protocol']['environment']}`.",
              '- Docker socket không truy cập được trong phiên triển khai; thực nghiệm chạy bằng Python cục bộ và ghi phiên bản vào protocol.', '',
              '## Tần suất chọn ứng viên nhóm', '']
    for model, counter in counts.items():
        lines.append(f'- {model}: `{dict(sorted(counter.items()))}`. Candidate 0 là giữ nguyên trọng số C.')
    lines += ['', 'Không diễn giải hệ số nhóm hoặc thứ hạng feature thành quan hệ nhân quả. Tuổi còn tham gia công thức tạo eGFR.', '',
              '## Chạy lại', '', '```bash',
              'PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v',
              'PYTHONDONTWRITEBYTECODE=1 python3 scripts/07_train_w2.py --repeats 5 --outer 5 --inner 3 --jobs 4',
              'OPENBLAS_NUM_THREADS=1 python3 scripts/08_validate_w2.py',
              'python3 scripts/09_finalize_w2_report.py', '```', '',
              'Giữ cùng code, dữ liệu và môi trường để resume checkpoint. Dùng `--output` khác khi thay protocol. File model cuối fit toàn cohort chỉ phục vụ nghiên cứu, không cung cấp ước lượng hiệu năng độc lập mới.', '',
              '---', '', (OUT / 'REPORT.md').read_text()]
    destination = ROOT / 'CAI_THIEN_W2_NHANES_DUAL_AXIS_2017_2023.md'
    destination.write_text('\n'.join(lines), encoding='utf-8')
    print(destination)


if __name__ == '__main__':
    main()
