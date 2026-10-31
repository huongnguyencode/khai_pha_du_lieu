"""Bounded, in-memory XLSX input/template/result workbooks."""
from io import BytesIO
from zipfile import ZipFile, BadZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.comments import Comment

from .schema import FEATURES, DEMO, MIN_OBSERVED
from .service import InputError, NOTICE

MAX_ROWS=1000
MAX_UPLOAD=10*1024*1024
MAX_UNCOMPRESSED=50*1024*1024


def parse_xlsx(data):
    if not data or len(data)>MAX_UPLOAD:raise InputError('File cần có dữ liệu và không vượt quá 10 MB.')
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries=archive.infolist()
            if len(entries)>2000 or sum(i.file_size for i in entries)>MAX_UNCOMPRESSED:
                raise InputError('Nội dung Excel sau giải nén quá lớn. Hãy chia nhỏ file.')
            names={i.filename for i in entries}
            if 'xl/workbook.xml' not in names or any('vbaproject' in n.lower() for n in names):
                raise InputError('Chỉ nhận file .xlsx thông thường, không nhận file có macro.')
        book=load_workbook(BytesIO(data),read_only=True,data_only=False,keep_links=False)
    except InputError:raise
    except Exception as exc:
        raise InputError('Không đọc được file .xlsx. Hãy tải file mẫu và lưu lại bằng Excel.') from exc
    try:
        sheet=book['Du_lieu'] if 'Du_lieu' in book.sheetnames else book.worksheets[0]
        if sheet.max_row is None or sheet.max_column is None:
            raise InputError('Không xác định được kích thước sheet. Hãy lưu lại bằng Excel hoặc dùng file mẫu.')
        if sheet.max_row>MAX_ROWS+1:
            raise InputError('Sheet vượt quá 1.000 dòng dữ liệu. Hãy chia nhỏ file và xóa các dòng định dạng thừa.')
        if sheet.max_column>128:raise InputError('File có quá nhiều cột. Chỉ giữ dữ liệu cần nhập, tối đa 128 cột.')
        iterator=sheet.iter_rows(max_col=sheet.max_column)
        header_cells=next(iterator,None)
        if not header_cells:raise InputError('File Excel chưa có dòng tiêu đề.')
        headers=[str(c.value).strip() if c.value is not None else '' for c in header_cells]
        nonempty=[h for h in headers if h]
        if len(nonempty)!=len(set(nonempty)):raise InputError('Có tên cột trùng nhau. Hãy dùng mỗi tên cột một lần.')
        required=[f['key'] for f in FEATURES]
        absent=[key for key in required if key not in headers]
        if absent:raise InputError('Thiếu cột trong file Excel: '+', '.join(absent)+'. Tải mẫu để dùng đúng 23 cột; ô chưa biết có thể để trống.')
        records=[];source_rows=[]
        for row_number,cells in enumerate(iterator,start=2):
            if not any(c.value is not None for c in cells):continue
            if len(records)>=MAX_ROWS:raise InputError('Tối đa 1.000 người mỗi lần. Hãy chia file thành nhiều phần.')
            if any(c.data_type=='f' for c in cells):
                raise InputError(f'Dòng Excel {row_number} có công thức. Hãy dán dưới dạng giá trị trước khi tải lên.')
            record={h:c.value for h,c in zip(headers,cells) if h}
            if 'id' not in record:record['id']=record.get('row_id',record.get('SEQN',str(row_number-1)))
            records.append(record);source_rows.append(row_number)
        if not records:raise InputError('File chỉ có tiêu đề, chưa có người nào để phân tích.')
        ignored=[h for h in nonempty if h not in required+['id','name','SEQN','row_id']]
        return records,source_rows,ignored
    finally:book.close()


def safe_cell(value):
    # Model results are literals, never Excel formulas or external references.
    if isinstance(value,str) and value.startswith(('=','+','-','@','\t','\r')):return "'"+value
    return value


def decorate(sheet,widths=None):
    sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
    for cell in sheet[1]:
        cell.fill=PatternFill('solid',fgColor='164C52');cell.font=Font(name='Arial',bold=True,color='FFFFFF')
        cell.alignment=Alignment(wrap_text=True,vertical='center')
    sheet.row_dimensions[1].height=32
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font=Font(name='Arial',size=11);cell.alignment=Alignment(vertical='top',wrap_text=True)
    for column in sheet.columns:
        letter=column[0].column_letter
        sheet.column_dimensions[letter].width=(widths or {}).get(letter,22)


def finish(book):
    output=BytesIO();book.save(output);return output.getvalue()


def template_workbook():
    book=Workbook();sheet=book.active;sheet.title='Du_lieu'
    keys=['id','name']+[f['key'] for f in FEATURES]
    sheet.append(keys);sheet.append(['VD-001','Ví dụ minh họa — xóa trước khi nhập']+[DEMO[f['key']] for f in FEATURES])
    for column,f in enumerate(FEATURES,start=3):
        description=f['label']+(f" ({f['unit']})" if f['unit'] else '')
        if f['kind']=='category':description+='; '+', '.join(f"{o['value']}={o['label']}" for o in f['options'])
        sheet.cell(1,column).comment=Comment(description,'Thận An')
        if f['kind']=='category':
            dv=DataValidation(type='list',formula1='"'+','.join(str(o['value']) for o in f['options'])+'"',allow_blank=True)
            dv.errorTitle='Mã chưa hợp lệ';dv.error='Xem sheet Tu_dien để chọn mã.';dv.showErrorMessage=True
            sheet.add_data_validation(dv);dv.add(f'{sheet.cell(1,column).column_letter}2:{sheet.cell(1,column).column_letter}{MAX_ROWS+1}')
    decorate(sheet)
    dictionary=book.create_sheet('Tu_dien');dictionary.append(['Cột','Ý nghĩa','Đơn vị','Giá trị hợp lệ','Bắt buộc'])
    for f in FEATURES:
        allowed='; '.join(f"{o['value']} = {o['label']}" for o in f['options']) if f['kind']=='category' else f"{f['min']}–{f['max']}"
        dictionary.append([f['key'],f['label'],f['unit'],allowed,'Có' if f['required'] else 'Có thể để trống'])
    decorate(dictionary,{'B':45,'D':90})
    guide=book.create_sheet('Huong_dan');guide.append(['Nội dung','Hướng dẫn'])
    for row in [
      ['Nhập dữ liệu','Điền sheet Du_lieu. Xóa hoặc thay dòng ví dụ trước khi dự đoán dữ liệu thật.'],
      ['Mã hóa','Giữ nguyên tên 23 cột. Giá trị phân loại dùng số mã trong Tu_dien, không dùng chữ Có/Không.'],
      ['Thông tin tối thiểu',f'Tuổi và giới tính bắt buộc; ít nhất {MIN_OBSERVED}/23 feature có giá trị mỗi người. Đây là quy tắc chất lượng nhập liệu của ứng dụng, chưa phải ngưỡng được kiểm định lâm sàng.'],
      ['Ô thiếu','Để trống khi không biết. Không nhập 0 thay giá trị chưa biết. Dùng dấu chấm làm dấu thập phân nếu nhập dưới dạng chữ.'],
      ['Đơn vị','Cholesterol, HDL và acid uric: mg/dL; hemoglobin: g/dL. Không nhập mmol/L vào cột mg/dL.'],
      ['Giới hạn','Tối đa 1.000 người, file 10 MB; chỉ dùng giá trị, không có công thức hoặc macro.'],
      ['Nguồn dữ liệu','Mô hình học từ người trưởng thành trong NHANES Hoa Kỳ; chưa xác nhận hiệu quả trên quần thể Việt Nam.'],
      ['Kết quả',NOTICE],
      ['Quyền riêng tư','Ứng dụng xử lý file trong bộ nhớ, không lưu hồ sơ tải lên.'],
    ]:guide.append(row)
    decorate(guide,{'A':28,'B':110})
    return finish(book)


def result_workbook(payload):
    book=Workbook();book.remove(book.active)
    headers=['Dòng dữ liệu','Mã','Họ tên','Trạng thái','Target','Kết quả','Nhận định','Gợi ý chăm sóc','Số feature có dữ liệu','Feature thiếu','Ghi chú / lỗi']
    rows=payload['results']
    for title,items in [
      ('Tat_ca',rows),('Co_dau_hieu',[r for r in rows if r.get('flagged')]),
      ('Chua_goi_y',[r for r in rows if r.get('status')=='ok' and not r['flagged']]),
      ('Loi_du_lieu',[r for r in rows if r['status']=='invalid'])]:
        sheet=book.create_sheet(title);sheet.append(headers)
        for r in items:
            error='; '.join(e['message'] for e in r.get('errors',[]))
            values=[r['row'],r['id'],r['name'],'Hợp lệ' if r['status']=='ok' else 'Chưa dự đoán',
                    r.get('target'),r.get('label','Lỗi dữ liệu'),r.get('summary',''),
                    '\n'.join(r.get('advice',[])),r.get('observed_count'),', '.join(r.get('missing_features',[])),
                    error or '; '.join(r.get('notes',[]))]
            sheet.append([safe_cell(v) for v in values])
        decorate(sheet,{'A':16,'B':22,'C':28,'F':48,'G':68,'H':100,'J':50,'K':65})
    guide=book.create_sheet('Luu_y');guide.append(['Nội dung','Diễn giải'])
    guide.append(['Kết quả',NOTICE]);guide.append(['Nhóm 0','Chưa gợi ý bất thường theo hai trục; không xác nhận hoàn toàn không có bệnh. Không có lời khuyên riêng cho nhóm này.'])
    guide.append(['Dòng lỗi','Không được gán vào nhóm có dấu hiệu hoặc chưa gợi ý. Sửa dữ liệu rồi phân tích lại.'])
    guide.append(['Kết luận','Chỉ dự đoán pattern thận, không chẩn đoán đái tháo đường hoặc giai đoạn bệnh.'])
    guide.append(['Điền khuyết','Feature thiếu được điền bằng preprocessing đã học; không huấn luyện lại trên file tải lên.'])
    from .service import SOURCES
    for source in SOURCES:guide.append([source['title'],source['url']])
    decorate(guide,{'A':30,'B':120})
    return finish(book)
