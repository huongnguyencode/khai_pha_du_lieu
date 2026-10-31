"""The 23 original features in artifact order; Vietnamese form and Excel metadata."""
from nhanes_pipeline.builder import CATEGORICAL_LEVELS
from nhanes_pipeline.combined_builder import PRIMARY_FEATURES


def numeric(key, label, group, unit, minimum, maximum, example, help_text=''):
    return dict(key=key,label=label,group=group,unit=unit,min=minimum,max=maximum,
                example=example,kind='number',help=help_text,required=key=='RIDAGEYR')


def category(key, label, group, labels, example, help_text=''):
    levels=CATEGORICAL_LEVELS[key]
    return dict(key=key,label=label,group=group,unit='',kind='category',
                options=[dict(value=n,label=s) for n,s in zip(levels,labels)],
                example=example,help=help_text,required=key=='RIAGENDR')

YES_NO=['Có','Không']
FEATURES=[
 numeric('RIDAGEYR','Tuổi','Thông tin cơ bản','năm',20,120,52,'Chỉ dành cho người từ 20 tuổi. Tuổi từ 80 được quy về nhóm 80+ như NHANES.'),
 category('RIAGENDR','Giới tính trong hồ sơ','Thông tin cơ bản',['Nam','Nữ'],2,'Mã giới tính dùng trong dữ liệu huấn luyện.'),
 category('RIDRETH3','Nhóm dân số theo NHANES','Thông tin cơ bản',['Người Mỹ gốc Mexico','Gốc Hispanic khác','Da trắng, không Hispanic','Da đen, không Hispanic','Châu Á, không Hispanic','Khác / đa chủng tộc'],6,'Có thể để trống nếu không có phân loại tương ứng; dữ liệu huấn luyện thuộc khảo sát Hoa Kỳ.'),
 category('DMDEDUC2','Trình độ học vấn','Thông tin cơ bản',['Dưới lớp 9','Lớp 9–11','Tốt nghiệp phổ thông / tương đương','Cao đẳng chưa tốt nghiệp / trung cấp','Tốt nghiệp đại học trở lên'],5),
 numeric('INDFMPIR','Tỷ số thu nhập / ngưỡng nghèo','Thông tin cơ bản','tỷ số',0,5,2.5,'Biến đặc thù NHANES. Để trống nếu không biết; không nhập thu nhập bằng tiền.'),
 numeric('BMXBMI','Chỉ số BMI','Thể trạng & huyết áp','kg/m²',10,100,27.4),
 numeric('BMXWAIST','Vòng eo','Thể trạng & huyết áp','cm',30,250,91),
 numeric('sbp_mean','Huyết áp tâm thu trung bình','Thể trạng & huyết áp','mmHg',50,300,128),
 numeric('dbp_mean','Huyết áp tâm trương trung bình','Thể trạng & huyết áp','mmHg',20,200,78),
 category('BPQ020','Từng được thông báo tăng huyết áp','Tiền sử sức khỏe',YES_NO,2),
 category('BPQ150','Hiện đang dùng thuốc huyết áp','Tiền sử sức khỏe',YES_NO,2),
 category('DIQ010','Từng được thông báo đái tháo đường','Tiền sử sức khỏe',['Có','Không','Ranh giới / borderline'],2,'Tiền sử do bạn cung cấp; ứng dụng không chẩn đoán đái tháo đường.'),
 category('DIQ160','Từng được thông báo tiền đái tháo đường','Tiền sử sức khỏe',YES_NO,2),
 category('current_smoking','Hiện đang hút thuốc','Sinh hoạt',['Không','Có'],0),
 category('ALQ111','Đã từng uống rượu / bia','Sinh hoạt',YES_NO,1),
 category('ALQ121','Tần suất uống trong 12 tháng','Sinh hoạt',['Không uống','Hằng ngày','Gần như hằng ngày','3–4 lần/tuần','2 lần/tuần','1 lần/tuần','2–3 lần/tháng','1 lần/tháng','7–11 lần/năm','3–6 lần/năm','1–2 lần/năm'],8),
 numeric('ALQ130','Số ly trung bình trong ngày có uống','Sinh hoạt','ly',0,100,1),
 numeric('PAD680','Thời gian ngồi mỗi ngày','Sinh hoạt','phút',0,1440,360),
 category('BPQ080','Từng được thông báo cholesterol cao','Tiền sử sức khỏe',YES_NO,2),
 numeric('LBXTC','Cholesterol toàn phần','Xét nghiệm','mg/dL',20,1000,190),
 numeric('LBDHDD','HDL cholesterol','Xét nghiệm','mg/dL',5,200,52),
 numeric('LBXSUA','Acid uric máu','Xét nghiệm','mg/dL',0.1,30,5.5),
 numeric('LBXHGB','Hemoglobin','Xét nghiệm','g/dL',2,25,13.8),
]
assert tuple(f['key'] for f in FEATURES)==PRIMARY_FEATURES
GROUPS=list(dict.fromkeys(f['group'] for f in FEATURES))
MIN_OBSERVED=12  # Product input-quality rule, not a validated clinical cutoff.
DEMO={f['key']:f['example'] for f in FEATURES}
