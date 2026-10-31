"""Create synthetic XLSX fixtures selected by W2 output, not clinical ground truth."""
from collections import Counter
from copy import deepcopy
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
from openpyxl import load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font, PatternFill, Alignment
from web.excel import template_workbook, parse_xlsx, decorate
from web.schema import DEMO,FEATURES
from web.service import PredictionService,LABELS,MODEL_SHA256
from web.app import Application,XLSX

OUT=ROOT/'khach_hang'
SEED=20261101
KEYS=[f['key'] for f in FEATURES]
BOOKS=[
 ('01_hon_hop_20_nguoi.xlsx','20 hồ sơ trộn: 10 nhóm 0; 3 nhóm 1; 3 nhóm 2; 4 nhóm 3',Counter({0:10,1:3,2:3,3:4})),
 ('02_chua_goi_y_bat_thuong_10_nguoi.xlsx','10 hồ sơ chưa gợi ý bất thường theo W2',Counter({0:10})),
 ('03_co_dau_hieu_10_nguoi.xlsx','10 hồ sơ có dấu hiệu: 3 nhóm 1; 3 nhóm 2; 4 nhóm 3',Counter({1:3,2:3,3:4})),
 ('04_albumin_nieu_10_nguoi.xlsx','10 hồ sơ được W2 dự đoán nhóm 1: chỉ albumin niệu',Counter({1:10})),
 ('05_loc_than_giam_10_nguoi.xlsx','10 hồ sơ được W2 dự đoán nhóm 2: chỉ giảm khả năng lọc',Counter({2:10})),
 ('06_ca_hai_dau_hieu_10_nguoi.xlsx','10 hồ sơ được W2 dự đoán nhóm 3: đồng thời hai dấu hiệu',Counter({3:10})),
 ('07_co_dau_hieu_kem_tien_su_tieu_duong_10_nguoi.xlsx','10 hồ sơ có dấu hiệu thận và khai tiền sử đái tháo đường; kiểm tra lời khuyên bổ sung',None),
]


def generate_candidates(rng,count=1600):
    records=[]
    for _ in range(count):
        age=int(rng.integers(24,81));sex=int(rng.integers(1,3))
        dm=int(rng.choice([1,2,3],p=[.42,.52,.06]));ht=int(rng.choice([1,2],p=[.6,.4]))
        sm=int(rng.choice([0,1],p=[.8,.2]));al=int(rng.choice([1,2],p=[.7,.3]))
        bmi=round(float(rng.uniform(20,37)),1)
        records.append({**DEMO,'RIDAGEYR':age,'RIAGENDR':sex,'RIDRETH3':6,
          'DMDEDUC2':int(rng.integers(2,6)),'INDFMPIR':round(float(rng.uniform(.8,4.5)),2),
          'BMXBMI':bmi,'BMXWAIST':round(float(55+1.5*bmi+rng.uniform(-6,12)),1),
          'sbp_mean':int(rng.integers(105,181)),'dbp_mean':int(rng.integers(62,96)),
          'BPQ020':ht,'BPQ150':int(rng.choice([1,2])) if ht==1 else 2,'DIQ010':dm,
          'DIQ160':None if dm==1 else int(rng.choice([1,2],p=[.3,.7])),
          'current_smoking':sm,'ALQ111':al,'ALQ121':int(rng.integers(5,11)) if al==1 else 0,
          'ALQ130':int(rng.integers(1,4)) if al==1 else 0,'PAD680':int(rng.integers(3,11))*60,
          'BPQ080':int(rng.choice([1,2])),'LBXTC':int(rng.integers(145,271)),
          'LBDHDD':int(rng.integers(32,76)),'LBXSUA':round(float(rng.uniform(3.3,10.8)),1),
          'LBXHGB':round(float(rng.uniform(10.5,16.5)),1)})
    return records


def profile_pools(service,rng):
    candidates=generate_candidates(rng)
    result=service.predict_records(candidates)
    assert result['counts']['invalid']==0
    pools={i:[] for i in range(4)}
    for rec,pred in zip(candidates,result['results']):
        target=pred['target']
        if len(pools[target])<10:pools[target].append(deepcopy(rec))
    assert all(len(rows)==10 for rows in pools.values())
    # Assign identifiers independently of pattern. No input rows are copied from NHANES.
    catalog=[r for rows in pools.values() for r in rows];rng.shuffle(catalog)
    surnames=['Nguyễn','Trần','Lê','Phạm','Hoàng','Vũ','Đặng','Bùi','Đỗ','Hồ']
    male=['Minh An','Quang Bình','Đức Cường','Hoàng Duy','Tuấn Hưng','Gia Khánh','Hữu Lâm','Thành Nam','Anh Phúc','Trung Sơn']
    female=['Ngọc Anh','Thu Hà','Mai Hương','Khánh Linh','Phương Mai','Bảo Ngọc','Thanh Thảo','Minh Thu','Hải Yến','Thùy Trang']
    for index,record in enumerate(catalog,start=1):
        names=male if record['RIAGENDR']==1 else female
        record['id']=f'KH-GL-{index:03d}'
        record['name']=surnames[(index-1)%len(surnames)]+' '+names[(index-1)//len(surnames)]+' (giả lập)'
    positive=pools[1][:3]+pools[2][:3]+pools[3][:4]
    mixed=pools[0][:]+positive
    rng.shuffle(mixed);rng.shuffle(positive)
    diabetic=[r for rows in (pools[1],pools[2],pools[3]) for r in rows if r['DIQ010']==1]
    assert len(diabetic)>=10
    rng.shuffle(diabetic)
    rows_by_book=[mixed,pools[0],positive,pools[1],pools[2],pools[3],diabetic[:10]]
    return rows_by_book


def create_book(path,title,records,payload):
    book=load_workbook(BytesIO(template_workbook()))
    book.properties.title='Thận An — dữ liệu khách hàng giả lập để kiểm thử'
    book.properties.subject=title;book.properties.creator='Thận An — bộ kiểm thử'
    sheet=book['Du_lieu'];sheet.delete_rows(2,sheet.max_row-1)
    for record in records:sheet.append([record.get(k) for k in ['id','name']+KEYS])
    decorate(sheet,{'A':20,'B':42})
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font=Font(name='Arial',size=11,color='1E4A78')
    for j,f in enumerate(FEATURES,start=3):
        for row in range(2,len(records)+2):
            if f['kind']=='number':sheet.cell(row,j).number_format='0' if f['key']=='RIDAGEYR' else '0.00'
    sheet.sheet_properties.tabColor='164C52'
    check=book.create_sheet('Doi_chieu')
    check.append(['Dòng Excel','id','name','Target W2 dự kiến','Kết quả W2 dự kiến','Tiền sử đái tháo đường được khai','Số lời khuyên dự kiến'])
    for record,row in zip(records,payload['results']):
        check.append([row['row'],row['id'],row['name'],row['target'],row['label'],
                      'Có (DIQ010=1)' if record['DIQ010']==1 else 'Không khai có đái tháo đường',len(row['advice'])])
    decorate(check,{'A':16,'B':20,'C':42,'D':20,'E':65,'F':44,'G':22})
    colors={0:'E5F1EA',1:'FFF0D7',2:'E9EFFB',3:'F9E1DA'}
    for row in check.iter_rows(min_row=2):
        row[3].fill=PatternFill('solid',fgColor=colors[row[3].value])
        row[3].comment=Comment('Đầu ra đã chạy bằng artifact D_W2_groups; không phải nhãn bệnh thật. Không dùng để tính accuracy lâm sàng.','Thận An')
    check.sheet_properties.tabColor='C5904B'
    guide=book['Huong_dan'];guide.delete_rows(2,guide.max_row-1)
    instructions=[
      ['Loại dữ liệu','TOÀN BỘ HỒ SƠ LÀ GIẢ LẬP, không phải khách hàng hoặc bệnh nhân thật. Tên, mã và 23 thông tin được sinh để kiểm thử.'],
      ['Mục tiêu file',title],
      ['Cách thử','Mở http://localhost:8090 → Danh sách Excel → chọn file này → Phân tích danh sách.'],
      ['Sheet nhập','Web chỉ đọc Du_lieu. File đã đủ đúng số người; không có dòng ví dụ thừa cần xóa.'],
      ['Đối chiếu','Sheet Doi_chieu ghi kết quả dự kiến của từng id. Các kết quả đã được chạy bằng W2; sheet này không được dùng làm đầu vào.'],
      ['Diễn giải','Target 0: chưa gợi ý; 1: albumin niệu; 2: giảm khả năng lọc; 3: đồng thời cả hai dấu hiệu. Không phải chẩn đoán bệnh hoặc giai đoạn CKD.'],
      ['Nhóm không dấu hiệu','Mong đợi web không hiển thị lời khuyên riêng cho nhóm 0, kể cả khi tiền sử khác có giá trị.'],
      ['Đái tháo đường','DIQ010=1 chỉ là tiền sử được khai. File 07 kiểm tra lời khuyên bổ sung khi đồng thời có dấu hiệu thận; W2 không chẩn đoán đái tháo đường.'],
      ['Ô trống','DIQ160 có thể trống khi hồ sơ khai đã có đái tháo đường; đây là missing có chủ ý. Không nhập 0 thay cho ô thiếu.'],
      ['Có thể sửa','Có thể sửa các ô màu xanh trong Du_lieu để thử tình huống khác. Khi sửa, kết quả dự kiến trong Doi_chieu không còn bảo đảm khớp.'],
      ['Mã và đơn vị','Xem Tu_dien. Dùng đúng mã categorical và đơn vị xét nghiệm. Các cột số là giá trị, không có công thức.'],
      ['Nguồn sinh','Python numpy.default_rng, seed 20261101; lấy mẫu trong các khoảng giả định ở scripts/12_create_customer_test_workbooks.py. Không lấy nguyên hàng dữ liệu train.'],
      ['Chọn hồ sơ','Sinh ứng viên giả lập, chạy W2 và chọn đủ số mẫu theo đầu ra để kiểm tra giao diện. Đây không phải tập test độc lập hoặc nhãn lâm sàng.'],
      ['Model','D_W2_groups; SHA256 artifact: '+MODEL_SHA256],
      ['Dùng lại hồ sơ','Các file có thể dùng lại cùng id để bạn đối chiếu từ bộ hỗn hợp sang bộ từng nhóm. Không cộng số dòng mọi file thành số người độc lập.'],
    ]
    for item in instructions:guide.append(item)
    decorate(guide,{'A':30,'B':120})
    for row in range(2,guide.max_row+1):guide.row_dimensions[row].height=45
    guide.sheet_properties.tabColor='A8BCC2'
    for ws in book:
        ws.sheet_view.showGridLines=False
        ws.print_options.horizontalCentered=True
        for row in ws.iter_rows():
            for cell in row:
                assert cell.data_type not in ('f','e')
    book.save(path);book.close()


def verify_file(path,expected,service):
    data=path.read_bytes();records,source_rows,ignored=parse_xlsx(data)
    assert not ignored
    result=service.predict_records(records,source_rows)
    actual=Counter(r['target'] for r in result['results'])
    assert actual==expected,(path.name,expected,actual)
    assert result['counts']['invalid']==0
    assert len({r['id'] for r in records})==len(records)
    book=load_workbook(BytesIO(data),data_only=False)
    assert book['Du_lieu'].max_row-1==sum(expected.values())
    assert book['Doi_chieu'].max_row-1==len(records)
    for saved,pred in zip(book['Doi_chieu'].iter_rows(min_row=2,values_only=True),result['results']):
        assert saved[1]==pred['id'] and saved[3]==pred['target'] and saved[6]==len(pred['advice'])
        if pred['target']==0:assert not pred['advice']
    for ws in book:
        assert not any(c.data_type in ('f','e') for row in ws for c in row)
    book.close()
    # Exercise the actual upload endpoint without requiring access to a host port.
    status=[]
    env={'PATH_INFO':'/api/predict-xlsx','REQUEST_METHOD':'POST','CONTENT_TYPE':XLSX,
         'CONTENT_LENGTH':str(len(data)),'wsgi.input':BytesIO(data),'HTTP_HOST':'localhost:8090'}
    def start_response(s,h):status.append(s)
    response=json.loads(b''.join(Application(service)(env,start_response)))
    assert status==['200 OK'] and response['counts']==result['counts']
    assert [r['target'] for r in response['results']]==[r['target'] for r in result['results']]
    return {'file':path.name,'people':len(records),'targets':dict(sorted(actual.items())),
            'counts':result['counts'],'sha256':hashlib.sha256(data).hexdigest(),'upload_api':'200 OK'}


def main():
    OUT.mkdir(exist_ok=True)
    service=PredictionService();rng=np.random.default_rng(SEED)
    collections=profile_pools(service,rng)
    entries=[]
    for (filename,title,expected),records in zip(BOOKS,collections):
        payload=service.predict_records(records,list(range(2,len(records)+2)))
        assert payload['counts']['invalid']==0
        actual=Counter(r['target'] for r in payload['results'])
        if expected is None:
            assert all(r['DIQ010']==1 for r in records)
            assert payload['counts']['flagged']==10
            assert all(len(r['advice'])==5 for r in payload['results'])
            expected=actual
        assert actual==expected
        path=OUT/filename
        create_book(path,title,records,payload)
        entries.append(verify_file(path,expected,service))
    manifest={'kind':'synthetic functional-test fixtures; not clinical labels or an independent validation set',
              'seed':SEED,'model':'D_W2_groups','model_sha256':MODEL_SHA256,'files':entries,
              'unique_profiles_across_files':len({r['id'] for rows in collections for r in rows}),
              'verification':'All files reopened, passed current XLSX parser, validation, inference, expected-output checks, and upload WSGI endpoint.'}
    (OUT/'kiem_tra_du_lieu.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    lines=['# File Excel khách hàng giả lập để test Thận An','','Tất cả hồ sơ đều là **giả lập**. Các nhóm có/không dấu hiệu dưới đây là **đầu ra đã kiểm bằng W2**, không phải kết luận có bệnh thật hoặc chắc chắn không bệnh.','','| File | Số người | Target 0 | Target 1 | Target 2 | Target 3 |','|---|---:|---:|---:|---:|---:|']
    for e in entries:
        c=e['targets'];lines.append(f"| [{e['file']}]({e['file']}) | {e['people']} | {c.get(0,0)} | {c.get(1,0)} | {c.get(2,0)} | {c.get(3,0)} |")
    lines+=['','Target 0: chưa gợi ý bất thường. Target 1: chỉ albumin niệu. Target 2: chỉ giảm khả năng lọc. Target 3: đồng thời cả hai dấu hiệu.','','## Cách dùng','','1. Mở **http://localhost:8090** → **Danh sách Excel**.','2. Chọn file cần thử và bấm **Phân tích danh sách**.','3. Xem sheet **Doi_chieu** trong file để đối chiếu từng mã hồ sơ và số lời khuyên mong đợi. Web chỉ đọc **Du_lieu**.','4. Dòng nhóm 0 không có lời khuyên riêng. File 07 gồm 10 dòng có dấu hiệu thận và `DIQ010=1`, dự kiến mỗi dòng có thêm lời khuyên về tiền sử đái tháo đường.','','Các file đã đủ đúng số người, không có dòng ví dụ thừa. Có thể sửa Du_lieu, nhưng khi sửa thì bảng Doi_chieu đã lưu không còn bảo đảm khớp.','','Bộ hỗn hợp 20 người trộn thứ tự ngẫu nhiên: 10 chưa gợi ý + 10 có dấu hiệu; trong 10 có dấu hiệu gồm 3 nhóm 1, 3 nhóm 2, 4 nhóm 3. Các hồ sơ được dùng lại giữa các file để đối chiếu; toàn bộ bộ file có '+str(manifest['unique_profiles_across_files'])+' id khác nhau, không phải '+str(sum(e['people'] for e in entries))+' người độc lập.','','## Kiểm tra và tái tạo','','Đã mở lại cả 7 XLSX, kiểm tra đủ 23 feature, số người, mã hợp lệ, nhãn W2, sheet đối chiếu và API upload hiện tại. Không có dòng lỗi hoặc công thức Excel. Không cần train lại model.','','Chi tiết số lượng và checksum: [kiem_tra_du_lieu.json](kiem_tra_du_lieu.json).','','Tái tạo từ thư mục dự án:','','```bash','PYTHONPATH=src:. python3 scripts/12_create_customer_test_workbooks.py','```','','Sinh bằng seed cố định 20261101, không sao chép hàng dữ liệu NHANES. Các giá trị là giả định của bộ test, không phải thông số y khoa khuyến nghị. Hồ sơ được chọn sau khi xem đầu ra W2 để đảm bảo đủ các nhánh giao diện; **không dùng bộ này để tính accuracy hoặc chứng minh hiệu quả mô hình**.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
