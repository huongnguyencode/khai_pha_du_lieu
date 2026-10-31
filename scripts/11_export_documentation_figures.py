"""Render documentation tables/figures from saved data and results; never train models."""
from pathlib import Path
import hashlib
import json
import os
import textwrap

os.environ.setdefault('MPLCONFIGDIR', '/tmp/nhanes-documentation-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'anh'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none'})
COLORS = ['#526777', '#21a179', '#3174ad', '#dc9133', '#9b6aa5']
GENERATED = []

def save(fig, stem):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=190, bbox_inches='tight', facecolor='white')
    GENERATED.append(stem)
    plt.close(fig)

def table(frame, stem, title, note='', widths=None, wrap=36):
    frame.to_csv(OUT / f'{stem}.csv', index=False)
    def cell(v):
        return '\n'.join(textwrap.wrap(str(v), wrap, break_long_words=False, break_on_hyphens=False))
    body = [[cell(v) for v in row] for row in frame.values]
    head = ['\n'.join(textwrap.wrap(str(c), min(wrap, 16), break_long_words=False, break_on_hyphens=False)) for c in frame.columns]
    heights = [max(str(c).count('\n') + 1 for c in head)] + [max(c.count('\n') + 1 for c in row) for row in body]
    fig, ax = plt.subplots(figsize=(15, max(2.7, sum(heights) * .27 + 1.5)))
    ax.axis('off')
    t = ax.table(cellText=body, colLabels=head, cellLoc='left', loc='center', colWidths=widths)
    t.auto_set_font_size(False); t.set_fontsize(10.5)
    for (r,c), obj in t.get_celld().items():
        obj.set_height(heights[r] * .27 / (fig.get_figheight() - 1.3))
        obj.set_edgecolor('#dbe3eb'); obj.PAD = .08
        if r == 0:
            obj.set_facecolor('#17364a'); obj.set_text_props(color='white', weight='bold')
        else: obj.set_facecolor('#eef4f8' if r % 2 else 'white')
    ax.set_title(title, loc='left', fontweight='bold', pad=22, fontsize=15)
    fig.text(.125, .015, note, fontsize=9, color='#425466')
    save(fig, stem)


def cmfig(cm, stem, title, note):
    cm = np.asarray(cm, dtype=int)
    frame = pd.DataFrame(cm, columns=[f'Pred {i}' for i in range(4)])
    frame.insert(0, 'Nhãn thật', [f'True {i}' for i in range(4)])
    frame.to_csv(OUT / f'{stem}.csv', index=False)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im=ax.imshow(pct, cmap='Blues', vmin=0, vmax=1)
    for r in range(4):
        for c in range(4):
            ax.text(c,r,f'{cm[r,c]:,}\n{pct[r,c]:.1%}',ha='center',va='center',
                    color='white' if pct[r,c]>.5 else '#10283b',fontsize=12)
    ax.set(xticks=range(4), yticks=range(4), xlabel='Lớp dự đoán', ylabel='Lớp thực tế', title=title)
    fig.colorbar(im, ax=ax, label='Tỷ lệ theo hàng')
    fig.text(.08, .005, note,fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1)); save(fig,stem)


def main():
    df=pd.read_csv(ROOT/'data/processed/nhanes_combined_2017_2023.csv')
    old=pd.read_csv(ROOT/'data/processed/nhanes_modeling_complete.csv')
    master=pd.read_csv(ROOT/'data/processed/nhanes_master_by_seqn.csv')
    manifest=json.loads((ROOT/'data/processed/nhanes_combined_manifest.json').read_text())
    features=manifest['features']
    w0=json.loads((ROOT/'nhanes_w0_handoff/baseline_metrics.json').read_text())
    w1=pd.read_csv(ROOT/'reports/combined_w1/model_summary.csv')
    w2=pd.read_csv(ROOT/'reports/w2/summary.csv')
    w2 = w2.rename(columns={c: c+'_mean' for c in w2.columns if c not in ('model', 'outer_evaluations') and not c.endswith('_std')})
    temporal=pd.read_csv(ROOT/'reports/w2/temporal.csv')
    predictions=pd.read_csv(ROOT/'reports/w2/oof_predictions.csv')
    assert len(df)==13086 and df.row_id.is_unique
    assert df[features].shape==(13086,23)
    assert np.array_equal(df.target, (df.uacr_mg_g>=30).astype(int)+2*(df.egfr_2021_ckdepi<60).astype(int))
    assert not df[['LBXSCR','URXUMA_harmonized','URXUCR','target']].isna().any().any()
    for name, part in predictions.groupby('model'):
        assert len(part)==65430 and not part.duplicated(['repeat','row_id']).any()
        truth = df.set_index('row_id').target.reindex(part.row_id).to_numpy()
        assert np.array_equal(truth, part.true_target.to_numpy())
        fold_f1, fold_accuracy, fold_fp = [], [], []
        for _, fold in part.groupby(['repeat', 'outer_fold']):
            cm = np.bincount(4*fold.true_target+fold.predicted_target, minlength=16).reshape(4,4)
            fold_f1.append(np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1))))
            fold_accuracy.append(cm.trace()/cm.sum())
            fold_fp.append(1000*(cm[0].sum()-cm[0,0])/cm.sum())
        expected = w2.set_index('model').loc[name]
        assert np.isclose(np.mean(fold_f1), expected.macro_f1_mean)
        assert np.isclose(np.mean(fold_accuracy), expected.accuracy_mean)
        assert np.isclose(np.mean(fold_fp), expected.healthy_fp_per_1000_mean)
    rules=pd.DataFrame([
        [0,'≥ 60','< 30','Không có dấu hiệu theo hai ngưỡng',0,0],
        [1,'≥ 60','≥ 30','Chỉ albumin niệu',0,1],
        [2,'< 60','< 30','Chỉ eGFR thấp',1,0],
        [3,'< 60','≥ 30','Có cả hai dấu hiệu',1,1],
    ],columns=['Target','eGFR (mL/phút/1,73 m²)','uACR (mg/g)','Diễn giải','G','A'])
    table(rules,'01_bang_rule_4_target','Bảng 1. Hai ngưỡng tạo bốn pattern',
          'Target = A + 2 × G. Một lần đo khảo sát; không phải bốn giai đoạn CKD.',[.06,.2,.13,.45,.08,.08])
    flow=pd.DataFrame([
        ['2017–03/2020',15560,7743,7817,'23'],
        ['08/2021–08/2023',11933,5343,6590,'24 lịch sử; 23 kết hợp'],
        ['Kết hợp',27493,13086,14407,'23'],
    ],columns=['Chu kỳ','Số dòng DEMO','Đủ điều kiện / nguồn nhãn','Không vào cohort','Số feature'])
    table(flow,'02_bang_cohort','Bảng 2. Từ dữ liệu khảo sát đến cohort mô hình',
          'Không vào cohort gồm dưới 20 tuổi hoặc không đủ điều kiện nhãn; không đồng nghĩa tất cả đều thiếu xét nghiệm.')
    counts=pd.DataFrame({'Target':range(4)})
    for cycle in ['2017-2020','2021-2023']:
        counts[cycle]=[int(((df.cycle==cycle)&(df.target==i)).sum()) for i in range(4)]
    counts['Tổng']=counts['2017-2020']+counts['2021-2023']
    counts['Tỷ lệ (%)']=(counts['Tổng']/len(df)*100).map(lambda n:f'{n:.2f}')
    table(counts,'03_bang_phan_bo_lop','Bảng 3. Mất cân bằng bốn lớp',
          'Tỷ lệ mẫu không dùng survey weights; không diễn giải thành tỷ lệ dân số Hoa Kỳ.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].bar(counts.Target,counts['Tổng'],color=COLORS[:4]); axes[0].set(xticks=range(4),xlabel='Target',ylabel='Số người',ylim=(0,12500))
    for i,n in enumerate(counts['Tổng']): axes[0].text(i,n+150,f'{n:,}\n{n/len(df):.2%}',ha='center')
    for i,cycle in enumerate(['2017-2020','2021-2023']):
        n=counts[cycle].to_numpy();axes[1].bar(np.arange(4)+(i-.5)*.36,n/n.sum()*100,width=.36,label=cycle,color=COLORS[i])
    axes[1].set(xticks=range(4),xlabel='Target',ylabel='Tỷ lệ mẫu (%)');axes[1].legend()
    fig.suptitle('Phân bố lớp: lớp 0 chiếm 82,42% cohort kết hợp',fontweight='bold')
    fig.tight_layout();save(fig,'04_bieu_do_lech_lop')
    descriptions=[
      ('RIDAGEYR','Tuổi (năm)','Số','Nhân khẩu'),('RIAGENDR','Giới tính khảo sát','Loại','Nhân khẩu'),
      ('RIDRETH3','Nhóm chủng tộc / sắc tộc','Loại','Xã hội'),('DMDEDUC2','Trình độ học vấn','Loại','Xã hội'),
      ('INDFMPIR','Tỷ số thu nhập / ngưỡng nghèo','Số','Xã hội'),('BMXBMI','BMI (kg/m²)','Số','Cơ thể'),
      ('BMXWAIST','Vòng eo (cm)','Số','Cơ thể'),('sbp_mean','Huyết áp tâm thu trung bình (mmHg)','Số','Huyết áp'),
      ('dbp_mean','Huyết áp tâm trương trung bình (mmHg)','Số','Huyết áp'),('BPQ020','Từng được thông báo tăng huyết áp','Loại','Huyết áp'),
      ('BPQ150','Đang dùng thuốc huyết áp (đã đồng nhất)','Loại','Huyết áp'),('DIQ010','Từng được thông báo đái tháo đường','Loại','Tiền sử'),
      ('DIQ160','Từng được thông báo tiền đái tháo đường','Loại','Tiền sử'),('current_smoking','Hiện hút thuốc (dẫn xuất)','Loại','Lối sống'),
      ('ALQ111','Từng uống rượu trong đời','Loại','Lối sống'),('ALQ121','Mã tần suất uống rượu 12 tháng','Loại','Lối sống'),
      ('ALQ130','Số ly trung bình khi uống rượu','Số','Lối sống'),('PAD680','Thời gian ngồi (phút/ngày)','Số','Lối sống'),
      ('BPQ080','Từng được thông báo cholesterol cao','Loại','Tiền sử'),('LBXTC','Cholesterol toàn phần (mg/dL)','Số','Xét nghiệm'),
      ('LBDHDD','HDL cholesterol (mg/dL)','Số','Xét nghiệm'),('LBXSUA','Acid uric máu (mg/dL)','Số','Xét nghiệm'),
      ('LBXHGB','Hemoglobin (g/dL)','Số','Xét nghiệm')]
    dictionary=pd.DataFrame(descriptions,columns=['Feature','Ý nghĩa','Kiểu trong mô hình','Nhóm nội dung'])
    assert dictionary.Feature.tolist()==features
    table(dictionary,'05_bang_23_feature','Bảng 4. Bộ 23 feature dùng chung W1 hai trục / W2',
          'Loại = categorical (mode + mismatch). Số = numeric (median + chia độ lệch chuẩn).',[.18,.49,.16,.17],45)
    missing=pd.DataFrame({'Feature':features})
    for cycle in ['2017-2020','2021-2023']:
        part=df[df.cycle==cycle];missing[cycle+' (%)']=[f'{part[f].isna().mean()*100:.2f}' for f in features]
    missing['Thiếu (n)']=[int(df[f].isna().sum()) for f in features]
    missing['Kết hợp (%)']=[f'{df[f].isna().mean()*100:.2f}' for f in features]
    table(missing,'06_bang_du_lieu_thieu','Bảng 5. Missingness trước imputation','Số liệu tính trực tiếp trên cohort; chưa dùng median/mode của bất kỳ fold nào.')
    top=missing.sort_values('Thiếu (n)',ascending=False).head(10).iloc[::-1]
    fig,ax=plt.subplots(figsize=(10,5.8));ax.barh(top.Feature,top['Kết hợp (%)'].astype(float),color=COLORS[2])
    for j,n in enumerate(top['Kết hợp (%)'].astype(float)):ax.text(n+.3,j,f'{n:.2f}%',va='center')
    ax.set(xlim=(0,30),xlabel='Tỷ lệ thiếu (%)',title='10 feature thiếu nhiều nhất trước imputation');fig.tight_layout();save(fig,'07_bieu_do_missingness')
    examplecols=['row_id','RIDAGEYR','RIAGENDR','BMXBMI','sbp_mean','DIQ010','egfr_2021_ckdepi','uacr_mg_g','target']
    example=pd.concat([df[df.target==i].head(1) for i in range(4)])[examplecols].copy()
    example.columns=['row_id','Tuổi','Giới','BMI','SBP','DIQ010','eGFR','uACR','Target']
    for c in ['Tuổi','BMI','SBP','eGFR','uACR']: example[c]=example[c].map(lambda v:f'{v:.2f}' if pd.notna(v) else 'NA')
    table(example,'08_bang_vi_du_du_lieu','Bảng 6. Bốn dòng thật, mỗi target một dòng',
          'Bảng minh họa chỉ hiển thị một phần schema. eGFR/uACR được giữ để kiểm tra nhãn, không nằm trong X.',wrap=24)
    cleaning=pd.DataFrame([
      ['Khóa / ghép','SEQN duy nhất từng module; DEMO làm bảng gốc','Ngăn nhân bản dòng khi join'],
      ['SAS missing / zero','Missing → NA; zero XPT → đúng 0','Sửa 772 ô zero lỗi ở handoff lịch sử'],
      ['Mã câu hỏi','Chỉ giữ code hợp lệ; 777/999 và 7777/9999 theo biến → NA','Không coi từ chối là số đo'],
      ['Skip logic','Suy ra không dùng thuốc / không hút / không uống khi đủ nhánh','Không gán mọi missing thành 0'],
      ['Đồng nhất chu kỳ','Albumin cũ: 1.381 + 0.8407 × giá trị gốc; thuốc huyết áp quy đổi','Giữ cùng nghĩa trước tạo nhãn / train'],
      ['Nguồn nhãn','Không impute creatinine / albumin để tạo target','Chỉ dùng nhãn từ số đo có thật'],
      ['Numeric trong fold','Median train; chia SD train; SD=0 → 1','Không để đơn vị đo chi phối khoảng cách'],
      ['Categorical trong fold','Mode train; mismatch chuẩn hóa theo feature gốc','Một biến nhiều mức không được nhiều phiếu'],
      ['Bất thường / lệch phải','Không thấy bước clipping, winsorization, log trong pipeline hiện tại','Không viết các bước chưa thực hiện'],
    ],columns=['Bước','Quy tắc đã thực hiện','Mục đích / giới hạn'])
    table(cleaning,'09_bang_lam_sach','Bảng 7. Quy tắc làm sạch và tiền xử lý','Tham chiếu builder.py, combined_builder.py, combined_experiment.py; mọi thống kê fit trong train.',[.18,.48,.34],45)
    historical=[]
    for name,r in w0.items():
        cm=np.array(r['holdout_confusion_matrix']);p=np.diag(cm)/np.maximum(cm.sum(0),1);rec=np.diag(cm)/cm.sum(1);f1=2*p*rec/np.maximum(p+rec,1e-15)
        assert np.isclose(f1.mean(),r['holdout_macro_f1'])
        historical.append([name.replace('M2_equal_features_distance_vote','M2 distance vote'),3,f'{r["cv_macro_f1"]:.4f}',f'{cm.trace()/cm.sum():.4f}',f'{r["holdout_macro_f1"]:.4f}',f'{r["holdout_balanced_accuracy"]:.4f}',f'{rec[3]:.4f}'])
        cmfig(cm,'11_w0_confusion' if name=='M1_W0' else '12_w0_distance_confusion',
              'W0 M1 — holdout lịch sử' if name=='M1_W0' else 'M2 distance — holdout lịch sử',
              'n = 1.603 người. Holdout đã được xem; chỉ dùng mô tả lịch sử.')
    table(pd.DataFrame(historical,columns=['Mô hình','K','CV chọn K','Accuracy holdout','Macro-F1 holdout','Balanced accuracy','Recall 3']),
          '10_bang_ket_qua_w0','Bảng 8. W0 NHANES trên split 3.740 / 1.603','CV chọn K không phải nested CV của W1; không ghép chung thành một đường tăng điểm.')
    initial=pd.read_csv(ROOT/'reports/w1/nhanes_ablation.csv').sort_values('model')
    tab=initial[['model','macro_f1_mean','macro_f1_std','balanced_accuracy_mean','accuracy_mean']].copy().round(4)
    tab['model'] = ['M1: Đều / uniform', 'M2: Đều / distance', 'M3: MI / uniform', 'M4: MI / distance']
    tab.columns = ['Cấu hình', 'Macro-F1', 'SD', 'Balanced accuracy', 'Accuracy']
    table(tab,'13_bang_w1_ban_dau','Bảng 9. W1 ban đầu — nested CV chỉ trên 3.740 người',
          'K cố định = 3; outer 5 folds, inner 3 folds; không dùng seen_holdout. M4 − M2 = +0,0021.')
    tab=w1[['model','macro_f1_mean','macro_f1_std','balanced_accuracy_mean','accuracy_mean','recall_1_mean','recall_2_mean','recall_3_mean']].round(4)
    names = {'flat_uniform_distance': 'Flat đều', 'flat_mi_distance': 'Flat MI',
             'dual_uniform_distance': 'Hai trục đều', 'dual_mi_distance': 'Hai trục MI',
             'dual_mi_balanced': 'Hai trục MI + cân bằng',
             'dual_mi_balanced_missing': 'Cân bằng + cờ thiếu'}
    tab['model'] = tab['model'].map(names)
    tab.columns = ['Cấu hình', 'Macro-F1', 'SD', 'Balanced accuracy', 'Accuracy', 'Recall 1', 'Recall 2', 'Recall 3']
    table(tab,'14_bang_w1_hai_truc','Bảng 10. W1 hai trục — cùng cohort 13.086 người',
          'Mean / SD trên 25 outer folds; mỗi người có OOF một lần mỗi repeat.',
          widths=[.23,.11,.08,.14,.11,.11,.11,.11],wrap=22)
    fig,ax=plt.subplots(figsize=(11,5.5));part=w1.iloc[::-1]
    ax.barh(part.model,part.macro_f1_mean,xerr=part.macro_f1_std,color=COLORS[1],capsize=4)
    for i,n in enumerate(part.macro_f1_mean):ax.text(n+.025,i,f'{n:.4f}',va='center')
    ax.set(xlim=(0,.5),xlabel='Macro-F1 (mean ± SD outer folds)',title='W1: MI, tách trục và balanced voting');fig.tight_layout();save(fig,'15_bieu_do_w1_ablation')
    a=w2.set_index('model').loc['A_W1_original']; d=w2.set_index('model').loc['D_W2_groups']
    tab=w2[['model','macro_f1_mean','macro_f1_std','accuracy_mean','balanced_accuracy_mean','precision_3_mean','recall_3_mean','healthy_fp_per_1000_mean']].round(4)
    tab['model'] = ['A: W1', 'B: Ngưỡng', 'C: K/alpha', 'D: Nhóm', 'E: Nhóm + phạt']
    tab.columns = ['Cấu hình', 'Macro-F1', 'SD', 'Accuracy', 'Balanced accuracy', 'Precision 3', 'Recall 3', 'FP lớp 0 / 1.000']
    table(tab,'16_bang_w2_ablation','Bảng 11. W2 — so sánh có kiểm soát trên cùng outer splits',
          'FP lớp 0 / 1.000 dùng mẫu số toàn bộ người được đánh giá trong fold.',wrap=24)
    fig,ax=plt.subplots(figsize=(10,5.2));bars=ax.bar(list('ABCDE'),w2.macro_f1_mean,yerr=w2.macro_f1_std,color=COLORS,capsize=5)
    for b,n in zip(bars,w2.macro_f1_mean):ax.text(b.get_x()+b.get_width()/2,n+.024,f'{n:.4f}',ha='center')
    ax.set(ylim=(0,.53),xlabel='A: W1 | B: ngưỡng chung | C: K/alpha | D: nhóm | E: nhóm + phạt',ylabel='Macro-F1 (mean ± SD)',title='W2: phần lớn mức tăng đến từ ngưỡng và K/alpha');fig.tight_layout();save(fig,'17_bieu_do_w2_macro_f1')
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axes,['healthy_fp_per_1000_mean','precision_3_mean','recall_3_mean'],['FP từ lớp 0 / 1.000 người','Precision lớp 3','Recall lớp 3']):
        ax.bar(['W1 (A)','W2 (D)'],[a[key],d[key]],color=[COLORS[0],COLORS[3]])
        for i,n in enumerate([a[key],d[key]]):ax.text(i,n*1.025,f'{n:.1f}' if '1000' in key else f'{n:.4f}',ha='center')
        ax.set_title(title);ax.set_ylim(0,max(a[key],d[key])*1.2)
    fig.suptitle('W2 giảm báo động nhầm và tăng precision, đồng thời giảm recall',fontweight='bold');fig.tight_layout();save(fig,'18_bieu_do_danh_doi_w1_w2')
    for model,stem in [('A_W1_original','19_w1_confusion_oof'),('D_W2_groups','20_w2_confusion_oof')]:
        part=predictions[predictions.model==model];cm=np.bincount(4*part.true_target+part.predicted_target,minlength=16).reshape(4,4)
        cmfig(cm,stem,f'{model} — OOF gộp 5 repeats','65.430 lượt dự đoán của 13.086 người; không phải 65.430 người độc lập.')
    temporal_table = temporal[['model','macro_f1','accuracy','balanced_accuracy','precision_3','recall_3','healthy_fp']].round(4)
    temporal_table['model'] = ['A: W1', 'B: Ngưỡng', 'C: K/alpha', 'D: Nhóm', 'E: Nhóm + phạt']
    temporal_table.columns = ['Cấu hình', 'Macro-F1', 'Accuracy', 'Balanced accuracy', 'Precision 3', 'Recall 3', 'FP từ lớp 0 (n)']
    table(temporal_table,
          '21_bang_temporal','Bảng 12. Train 2017–03/2020, test 08/2021–08/2023',
          'n test = 5.343; healthy_fp là số lớp 0 dự đoán sang lớp 1–3. Không phải external validation độc lập.')
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(5);ax.bar(x-.18,w2.macro_f1_mean,.36,color=COLORS[2],label='Repeated nested CV');ax.bar(x+.18,temporal.macro_f1,.36,color=COLORS[3],label='Temporal')
    ax.set(xticks=x,xticklabels=list('ABCDE'),ylim=(0,.55),ylabel='Macro-F1',title='Kết quả chuyển chu kỳ: C/E nhỉnh hơn D');ax.legend();fig.tight_layout();save(fig,'22_bieu_do_temporal')
    ww=pd.read_csv(ROOT/'reports/combined_w1/final_axis_weights.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5.7))
    for ax,axis in zip(axes,['egfr','uacr']):
        part=ww[ww.axis==axis].nlargest(8,'final_weight').iloc[::-1]
        ax.barh(part.feature,part.final_weight,color=COLORS[2 if axis=='egfr' else 1]);ax.set(xlim=(0,.3),title='Trục '+axis,xlabel='Trọng số cuối (full cohort)')
    fig.suptitle('W1: hai trục học hai vector trọng số khác nhau',fontweight='bold');fig.tight_layout();save(fig,'23_bieu_do_trong_so_w1')
    # PR curves from scores; A and B have identical rankings by construction.
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for model,color in zip(['A_W1_original','C_joint_K_alpha','D_W2_groups'],[COLORS[0],COLORS[2],COLORS[3]]):
        part=predictions[predictions.model==model];y=part.true_target.to_numpy()
        for j,(axis,label) in enumerate([('egfr_score','eGFR thấp'),('uacr_score','uACR cao')]):
            truth=y//2 if j==0 else y%2;s=part[axis].to_numpy();order=np.argsort(-s,kind='stable');ends=np.r_[np.flatnonzero(np.diff(s[order])!=0),len(s)-1]
            tp=np.cumsum(truth[order])[ends];rec=tp/truth.sum();prec=tp/(ends+1)
            axes[j].step(np.r_[0,rec],np.r_[1,prec],where='pre',color=color,label=model)
            axes[j].set(xlim=(0,1),ylim=(0,1),xlabel='Recall',ylabel='Precision',title=label);axes[j].grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Precision–recall từ OOF gộp; chỉ dùng phân tích mô tả');fig.tight_layout();save(fig,'24_bieu_do_precision_recall')
    stats={
        'n':len(df),'shape':list(df.shape),'class_counts':df.target.value_counts().sort_index().to_dict(),
        'missing_total':int(df[features].isna().sum().sum()),'rows_any_feature_missing':int(df[features].isna().any(axis=1).sum()),
        'complete_feature_rows':int(df[features].notna().all(axis=1).sum()),
        'new_cycle_adults':int(master.RIDAGEYR.ge(20).sum()),
        'new_cycle_under20':int(master.RIDAGEYR.lt(20).sum()),
        'new_cycle_adults_excluded':int(master.RIDAGEYR.ge(20).sum()-len(old)),
        'feature_numeric_count':12,'feature_categorical_count':11,
        'w1_w2_macro_delta':float(d.macro_f1_mean-a.macro_f1_mean),
        'w1_w2_fp_reduction_relative':float(1-d.healthy_fp_per_1000_mean/a.healthy_fp_per_1000_mean),
        'environment':{'python':__import__('platform').python_version(),'matplotlib':matplotlib.__version__,'pandas':pd.__version__,'numpy':np.__version__},
        'figures':GENERATED,
    }
    sources=['data/processed/nhanes_combined_2017_2023.csv','nhanes_w0_handoff/baseline_metrics.json','reports/w1/nhanes_ablation.csv','reports/combined_w1/model_summary.csv','reports/w2/summary.csv','reports/w2/oof_predictions.csv','reports/w2/temporal.csv']
    stats['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    (OUT/'so_lieu_tai_lieu.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    index=['# Ảnh và bảng dùng trong tài liệu W0 → W1 → W2','','Tạo bằng `python3 scripts/11_export_documentation_figures.py`. Không huấn luyện lại mô hình.','','PNG: chèn vào Word/slide. SVG: vector để xuất/in. CSV: dữ liệu bảng để sửa và dùng lại.','','| Nội dung | PNG | SVG | CSV |','|---|---|---|---|']
    for stem in GENERATED:
        csv=f'[CSV]({stem}.csv)' if (OUT/f'{stem}.csv').exists() else '—'
        index.append(f'| {stem} | [PNG]({stem}.png) | [SVG]({stem}.svg) | {csv} |')
    (OUT/'README.md').write_text('\n'.join(index)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
