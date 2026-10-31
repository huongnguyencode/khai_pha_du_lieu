"""Validate inputs and run the bundled W2 artifact; no model training or patient storage."""
import hashlib
import math
import pickle
import threading
from pathlib import Path

import numpy as np
from nhanes_pipeline.w2 import predict
from .schema import FEATURES, MIN_OBSERVED

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'models/nhanes_w2.pkl'
MODEL_SHA256='32333d787c50e03ceb15ce0be724937c089421286cbde982db9a7d2f0c3ac8f3'
MODEL_NAME='D_W2_groups'
LABELS={0:'Chưa gợi ý bất thường theo hai trục',1:'Gợi ý albumin niệu',
        2:'Gợi ý khả năng lọc của thận giảm',3:'Gợi ý giảm khả năng lọc và albumin niệu'}
SOURCES=[
 {'title':'NIDDK: chăm sóc sức khỏe thận','url':'https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/managing'},
 {'title':'NIDDK: ăn uống và bệnh thận','url':'https://www.niddk.nih.gov/health-information/kidney-disease/chronic-kidney-disease-ckd/healthy-eating-adults-chronic-kidney-disease'},
 {'title':'NIDDK: quản lý đái tháo đường','url':'https://www.niddk.nih.gov/health-information/diabetes/overview/managing-diabetes'},
]
NOTICE='Kết quả sàng lọc từ mô hình nghiên cứu, không thay thế chẩn đoán. Cần xét nghiệm và đánh giá của nhân viên y tế để xác nhận.'


class InputError(ValueError):
    def __init__(self,message,details=None):
        super().__init__(message)
        self.details=details or []


def text_value(value, limit=120):
    if value is None:return ''
    return str(value).strip()[:limit]


def validate_record(record):
    if not isinstance(record,dict):raise InputError('Dữ liệu một người phải là một đối tượng chứa các giá trị.')
    values=[];errors=[];missing=[];notes=[]
    for f in FEATURES:
        raw=record.get(f['key'])
        if raw is None or (isinstance(raw,str) and raw.strip().lower() in ('','na','n/a','nan','null')):
            values.append(math.nan);missing.append(f['key'])
            if f['required']:errors.append({'field':f['key'],'message':f"Cần nhập {f['label'].lower()}."})
            continue
        try:
            if isinstance(raw,bool):raise ValueError
            value=float(str(raw).strip())
            if not math.isfinite(value):raise ValueError
        except (ValueError,TypeError):
            values.append(math.nan);errors.append({'field':f['key'],'message':f"{f['label']}: cần số hợp lệ, dùng dấu chấm cho phần thập phân."});continue
        if f['kind']=='category':
            if value not in [o['value'] for o in f['options']]:
                errors.append({'field':f['key'],'message':f"{f['label']}: mã không hợp lệ; xem danh mục trong file mẫu."})
        elif not f['min']<=value<=f['max']:
            errors.append({'field':f['key'],'message':f"{f['label']}: nhập trong khoảng {f['min']}–{f['max']} {f['unit']}."})
        if f['key']=='RIDAGEYR':
            if value!=int(value):errors.append({'field':f['key'],'message':'Tuổi phải là số năm nguyên.'})
            if value>80:
                notes.append('Tuổi được đưa về nhóm 80+ để khớp cách mã hóa dữ liệu huấn luyện.');value=80.
        values.append(value)
    if len(FEATURES)-len(missing)<MIN_OBSERVED:
        errors.append({'field':'_record','message':f'Cần ít nhất {MIN_OBSERVED}/23 thông tin, gồm tuổi và giới tính. Không nhập 0 thay cho giá trị chưa biết.'})
    if errors:raise InputError('Thông tin chưa hợp lệ. Hãy kiểm tra các trường được đánh dấu.',errors)
    data=dict(zip([f['key'] for f in FEATURES],values))
    if math.isfinite(data['sbp_mean']) and math.isfinite(data['dbp_mean']) and data['dbp_mean']>=data['sbp_mean']:
        raise InputError('Huyết áp tâm trương cần nhỏ hơn huyết áp tâm thu.',[{'field':'dbp_mean','message':'Kiểm tra lại hai giá trị huyết áp.'}])
    return np.asarray(values),missing,notes


def interpretation(target,record):
    if target==0:return '',[],[]
    summary={1:'Mô hình gợi ý dấu hiệu albumin trong nước tiểu tăng. Cần đo uACR để kiểm tra.',
             2:'Mô hình gợi ý chức năng lọc có thể giảm. Cần xét nghiệm creatinine máu và tính eGFR để kiểm tra.',
             3:'Mô hình gợi ý đồng thời hai dấu hiệu. Cần kiểm tra creatinine máu/eGFR và uACR.'}[target]
    advice=[
      'Trao đổi với nhân viên y tế về kết quả và việc xét nghiệm xác nhận hoặc theo dõi lại.',
      'Ăn giảm muối, hạn chế thực phẩm chế biến sẵn; trao đổi chế độ đạm, kali và nước theo xét nghiệm.',
      'Ngủ đủ, giữ giờ ngủ đều và vận động phù hợp thể trạng.',
      'Trao đổi trước khi dùng ibuprofen hoặc naproxen; không tự thay đổi thuốc đang được kê.',
    ]
    try:diabetes=float(record.get('DIQ010'))==1
    except (ValueError,TypeError):diabetes=False
    if diabetes:advice.append('Bạn khai có tiền sử đái tháo đường: theo dõi đường huyết và dùng thuốc theo kế hoạch bác sĩ, hạn chế đồ uống có đường.')
    return summary,advice,SOURCES if diabetes else SOURCES[:2]


class PredictionService:
    def __init__(self,path=MODEL_PATH):
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=MODEL_SHA256:
            raise RuntimeError('Artifact W2 không khớp checksum được duyệt.')
        with path.open('rb') as stream:self.artifact=pickle.load(stream)
        if self.artifact['model_name']!=MODEL_NAME or tuple(self.artifact['features'])!=tuple(f['key'] for f in FEATURES):
            raise RuntimeError('Artifact không đúng model hoặc schema W2.')
        self.lock=threading.Lock()
        # Verify source hash and inference at startup, without accepting any untrusted pickle.
        predict(self.artifact,self.artifact['train_matrix'][:1])

    def predict_records(self,records,source_rows=None):
        good=[];matrices=[];results=[]
        for i,record in enumerate(records):
            base={'index':i,'row':source_rows[i] if source_rows else i+1,
                  'id':text_value(record.get('id',record.get('row_id',record.get('SEQN',i+1)))),
                  'name':text_value(record.get('name',''))}
            try:
                matrix,missing,notes=validate_record(record)
                matrices.append(matrix);good.append((record,base,missing,notes))
            except InputError as exc:
                results.append({**base,'status':'invalid','message':str(exc),'errors':exc.details})
        if matrices:
            with self.lock:
                predictions,_=predict(self.artifact,np.stack(matrices),model=MODEL_NAME)
            for target,(record,base,missing,notes) in zip(predictions,good):
                target=int(target);summary,advice,sources=interpretation(target,record)
                results.append({**base,'status':'ok','target':target,'flagged':target!=0,'label':LABELS[target],
                    'egfr_low':bool(target//2),'uacr_high':bool(target%2),
                    'summary':summary,'advice':advice,'sources':sources,'missing_features':missing,
                    'observed_count':23-len(missing),'notes':notes})
        results.sort(key=lambda r:r['index'])
        ok=[r for r in results if r['status']=='ok']
        return {'results':results,'counts':{'total':len(results),'valid':len(ok),'invalid':len(results)-len(ok),
                    'flagged':sum(r['flagged'] for r in ok),'unflagged':sum(not r['flagged'] for r in ok)},'notice':NOTICE}
