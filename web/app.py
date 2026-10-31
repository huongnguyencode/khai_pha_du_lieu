"""Small WSGI API served by Gunicorn in Docker; also testable without a socket."""
from io import BytesIO
import json
import logging
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

from .schema import FEATURES, GROUPS, DEMO, MIN_OBSERVED
from .service import PredictionService, InputError, NOTICE
from .excel import MAX_UPLOAD, parse_xlsx, template_workbook, result_workbook

STATIC=Path(__file__).parent/'static'
XLSX='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


class Application:
    def __init__(self,service=None):
        self._service=service;self._lock=threading.Lock()

    @property
    def service(self):
        if self._service is None:
            with self._lock:
                if self._service is None:self._service=PredictionService()
        return self._service

    def __call__(self,environ,start_response):
        status=200;extra=[]
        try:status,kind,body,extra=self.dispatch(environ)
        except InputError as exc:
            status=422;kind='application/json; charset=utf-8';body=json.dumps({'error':str(exc),'details':exc.details},ensure_ascii=False).encode()
        except Exception:
            logging.exception('Prediction request failed')
            status=500;kind='application/json; charset=utf-8';body=json.dumps({'error':'Không thể xử lý yêu cầu. Kiểm tra trạng thái mô hình hoặc liên hệ người quản trị.'},ensure_ascii=False).encode()
        phrases={200:'OK',400:'Bad Request',403:'Forbidden',404:'Not Found',405:'Method Not Allowed',413:'Content Too Large',415:'Unsupported Media Type',422:'Unprocessable Entity',500:'Internal Server Error'}
        headers=[('Content-Type',kind),('Content-Length',str(len(body))),('Cache-Control','no-store'),
            ('X-Content-Type-Options','nosniff'),('Referrer-Policy','no-referrer'),
            ('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"),*extra]
        start_response(f'{status} {phrases[status]}',headers)
        return [body]

    def dispatch(self,env):
        path=env.get('PATH_INFO','/');method=env.get('REQUEST_METHOD','GET')
        def js(data,status=200):return status,'application/json; charset=utf-8',json.dumps(data,ensure_ascii=False,allow_nan=False).encode(),[]
        if method=='GET':
            assets={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/styles.css':('styles.css','text/css; charset=utf-8'),'/favicon.svg':('favicon.svg','image/svg+xml')}
            if path in assets:
                name,kind=assets[path];return 200,kind,(STATIC/name).read_bytes(),[]
            if path=='/health':
                _=self.service;return js({'status':'ok','model':'W2','features':23})
            if path=='/api/schema':return js({'features':FEATURES,'groups':GROUPS,'demo':DEMO,'min_observed':MIN_OBSERVED,'notice':NOTICE})
            if path=='/api/template':return 200,XLSX,template_workbook(),[('Content-Disposition','attachment; filename="mau_du_lieu_than_an.xlsx"')]
            return js({'error':'Không tìm thấy đường dẫn.'},404)
        if method!='POST':return js({'error':'Phương thức không được hỗ trợ.'},405)
        if path not in ('/api/predict','/api/predict-xlsx'):return js({'error':'Không tìm thấy đường dẫn.'},404)
        origin=env.get('HTTP_ORIGIN')
        if origin:
            try:u=urlsplit(origin)
            except ValueError:return js({'error':'Nguồn gửi yêu cầu không được chấp nhận.'},403)
            if u.scheme not in ('http','https') or u.netloc!=env.get('HTTP_HOST'):return js({'error':'Nguồn gửi yêu cầu không được chấp nhận.'},403)
        try:length=int(env.get('CONTENT_LENGTH') or 0)
        except ValueError:return js({'error':'Kích thước yêu cầu không hợp lệ.'},400)
        if length<=0:return js({'error':'Yêu cầu chưa có dữ liệu.'},400)
        if length>MAX_UPLOAD:return js({'error':'Yêu cầu vượt quá 10 MB.'},413)
        body=env['wsgi.input'].read(length)
        if len(body)!=length:return js({'error':'Dữ liệu tải lên chưa đầy đủ.'},400)
        if path=='/api/predict':
            if env.get('CONTENT_TYPE','').split(';')[0]!='application/json':return js({'error':'Cần gửi dữ liệu JSON.'},415)
            try:record=json.loads(body)
            except (ValueError,UnicodeDecodeError):return js({'error':'Dữ liệu JSON không hợp lệ.'},400)
            if not isinstance(record,dict):raise InputError('Cần gửi thông tin của một người.')
            payload=self.service.predict_records([record])
            if payload['counts']['invalid']:
                row=payload['results'][0];raise InputError(row['message'],row['errors'])
        else:
            if env.get('CONTENT_TYPE','').split(';')[0] not in (XLSX,'application/octet-stream'):
                return js({'error':'Chỉ nhận nội dung file .xlsx.'},415)
            records,source_rows,ignored=parse_xlsx(body)
            payload=self.service.predict_records(records,source_rows)
            payload['ignored_columns']=ignored
        download=parse_qs(env.get('QUERY_STRING','')).get('download')==['1']
        if download:return 200,XLSX,result_workbook(payload),[('Content-Disposition','attachment; filename="ket_qua_than_an.xlsx"')]
        return js(payload)


application=Application()
