"""Inference, XLSX and WSGI integration checks that require no listening socket."""
from copy import deepcopy
from io import BytesIO
import json
import unittest

import numpy as np
from openpyxl import Workbook,load_workbook
from nhanes_pipeline.w2 import predict
from web.app import Application, XLSX
from web.excel import parse_xlsx, template_workbook, result_workbook, MAX_ROWS, MAX_UPLOAD
from web.schema import FEATURES,DEMO
from web.service import PredictionService,InputError,interpretation,validate_record

KEYS=[f['key'] for f in FEATURES]


def workbook_bytes(headers,rows):
    book=Workbook();sheet=book.active;sheet.title='Du_lieu';sheet.append(headers)
    for row in rows:sheet.append(row)
    out=BytesIO();book.save(out);return out.getvalue()


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service=PredictionService();cls.app=Application(cls.service)
        artifact=cls.service.artifact
        # Include one original record from each target; the test compares inference
        # with the original function, not clinical accuracy on seen training rows.
        ids=[np.flatnonzero(artifact['train_target']==i)[0] for i in range(4)]
        cls.samples=artifact['train_matrix'][ids].copy()
        cls.records=[dict(id=f'CASE-{i}',name=f'Người minh họa {i}',**{k:(float(v) if np.isfinite(v) else None) for k,v in zip(KEYS,row)}) for i,row in enumerate(cls.samples)]

    def call(self,path,method='GET',body=b'',kind='application/json',**extra):
        path,_,query=path.partition('?')
        env={'PATH_INFO':path,'QUERY_STRING':query,'REQUEST_METHOD':method,'CONTENT_LENGTH':str(len(body)),
             'CONTENT_TYPE':kind,'wsgi.input':BytesIO(body),'HTTP_HOST':'localhost:8090',**extra}
        status=[];headers=[]
        def start(s,h):status.append(s);headers.extend(h)
        output=b''.join(self.app(env,start))
        return int(status[0].split()[0]),dict(headers),output

    def post(self,record):return self.call('/api/predict','POST',json.dumps(record).encode())

    def test_original_artifact_predictions_and_feature_order(self):
        expected,_=predict(self.service.artifact,self.samples)
        result=self.service.predict_records(self.records)
        self.assertEqual([r['target'] for r in result['results']],expected.tolist())
        self.assertEqual(tuple(KEYS),tuple(self.service.artifact['features']))
        self.assertEqual(result['counts']['valid'],4)

    def test_partial_missing_uses_fitted_preprocessor(self):
        record=dict(DEMO)
        for key in KEYS[2:7]:record[key]=None
        result=self.service.predict_records([record])['results'][0]
        self.assertEqual(result['observed_count'],18)
        self.assertEqual(len(result['missing_features']),5)
        x,_,_=validate_record(record);expected,_=predict(self.service.artifact,x[None,:])
        self.assertEqual(result['target'],int(expected[0]))

    def test_requires_age_sex_and_enough_information(self):
        with self.assertRaises(InputError):validate_record({'RIDAGEYR':50,'RIAGENDR':1})
        record=dict(DEMO);record['RIAGENDR']=None
        with self.assertRaises(InputError):validate_record(record)

    def test_invalid_code_numeric_and_age(self):
        for key,value in [('RIAGENDR',7),('DIQ010',1.5),('BMXBMI',float('inf')),('RIDAGEYR',19),('RIDAGEYR',50.5),('ALQ130',999),('PAD680',1500),('LBXHGB','wrong')]:
            with self.subTest(key=key,value=value):
                record={**DEMO,key:value}
                with self.assertRaises(InputError):validate_record(record)

    def test_age_topcoding_matches_training_semantics(self):
        row,_,notes=validate_record({**DEMO,'RIDAGEYR':95})
        self.assertEqual(row[0],80);self.assertTrue(notes)

    def test_blood_pressure_consistency(self):
        with self.assertRaises(InputError):validate_record({**DEMO,'sbp_mean':70,'dbp_mean':90})

    def test_no_advice_for_class_zero_even_with_diabetes_history(self):
        summary,advice,sources=interpretation(0,{'DIQ010':1})
        self.assertEqual((summary,advice,sources),('',[],[]))

    def test_guidance_for_each_positive_pattern_and_history(self):
        for target in (1,2,3):
            summary,advice,sources=interpretation(target,{'DIQ010':2})
            self.assertTrue(summary);self.assertEqual(len(advice),4);self.assertEqual(len(sources),2)
            self.assertFalse(any('Bạn khai có tiền sử' in s for s in advice))
        _,advice,sources=interpretation(3,{'DIQ010':1})
        self.assertTrue(any('Bạn khai có tiền sử' in s for s in advice));self.assertEqual(len(sources),3)

    def test_static_page_schema_health_and_csp(self):
        for path in ('/','/app.js','/styles.css','/favicon.svg','/health','/api/schema'):
            with self.subTest(path=path):
                code,headers,body=self.call(path)
                self.assertEqual(code,200);self.assertTrue(body)
                self.assertIn("frame-ancestors 'none'",headers['Content-Security-Policy'])
        self.assertEqual(json.loads(self.call('/health')[2])['model'],'W2')
        schema=json.loads(self.call('/api/schema')[2]);self.assertEqual([f['key'] for f in schema['features']],KEYS)

    def test_single_api_success_and_validation_error(self):
        code,_,body=self.post(DEMO);self.assertEqual(code,200);self.assertEqual(json.loads(body)['counts']['valid'],1)
        code,_,body=self.post({'RIDAGEYR':50});self.assertEqual(code,422);self.assertTrue(json.loads(body)['details'])

    def test_bad_requests_and_same_origin(self):
        self.assertEqual(self.call('/api/predict','POST',b'{')[0],400)
        self.assertEqual(self.call('/api/predict','POST',b'{}',kind='text/plain')[0],415)
        self.assertEqual(self.call('/api/predict','POST',b'{}',HTTP_ORIGIN='https://example.com')[0],403)
        self.assertEqual(self.call('/api/predict','POST',b'{}',HTTP_ORIGIN='http://[')[0],403)
        self.assertEqual(self.call('/api/predict','POST',b'{}',CONTENT_LENGTH=str(MAX_UPLOAD+1))[0],413)
        self.assertEqual(self.call('/unknown')[0],404)
        self.assertEqual(self.call('/api/predict','PUT')[0],405)
        self.assertEqual(self.post([])[0],422)

    def test_template_workbook_and_legend(self):
        code,headers,data=self.call('/api/template');self.assertEqual(code,200);self.assertIn('attachment',headers['Content-Disposition'])
        book=load_workbook(BytesIO(data));self.assertEqual(book.sheetnames,['Du_lieu','Tu_dien','Huong_dan'])
        self.assertEqual([c.value for c in book['Du_lieu'][1]],['id','name']+KEYS)
        records,rows,ignored=parse_xlsx(data);self.assertEqual(len(records),1);self.assertEqual(rows,[2]);self.assertFalse(ignored)
        validate_record(records[0]);book.close()

    def test_excel_batch_keeps_error_rows_separate_and_export(self):
        records=deepcopy(self.records)+[{**DEMO,'id':'ERR','name':'Cần sửa','RIAGENDR':9}]
        headers=['id','name']+KEYS+['target']
        rows=[[r.get(h) for h in headers] for r in records]
        data=workbook_bytes(headers,rows)
        code,_,body=self.call('/api/predict-xlsx','POST',data,kind=XLSX);self.assertEqual(code,200)
        payload=json.loads(body);self.assertEqual(payload['counts']['total'],5);self.assertEqual(payload['counts']['invalid'],1)
        self.assertEqual(payload['counts']['flagged']+payload['counts']['unflagged'],4)
        self.assertEqual(payload['ignored_columns'],['target']);self.assertEqual(payload['results'][-1]['row'],6)
        code,headers,data=self.call('/api/predict-xlsx?download=1','POST',data,kind=XLSX);self.assertEqual(code,200)
        book=load_workbook(BytesIO(data));self.assertEqual(book['Tat_ca'].max_row,6)
        self.assertEqual(book['Loi_du_lieu'].max_row,2)
        self.assertEqual(book['Co_dau_hieu'].max_row-1,payload['counts']['flagged'])
        self.assertEqual(book['Chua_goi_y'].max_row-1,payload['counts']['unflagged'])
        for row in book['Chua_goi_y'].iter_rows(min_row=2):self.assertIsNone(row[7].value)
        book.close()

    def test_export_single_recomputes_from_original_inputs(self):
        code,headers,data=self.call('/api/predict?download=1','POST',json.dumps(DEMO).encode())
        self.assertEqual(code,200);self.assertEqual(headers['Content-Type'],XLSX)
        book=load_workbook(BytesIO(data));self.assertEqual(book['Tat_ca'].max_row,2);book.close()

    def test_excel_formula_injection_is_written_as_literal(self):
        payload=self.service.predict_records([{**DEMO,'id':'=HYPERLINK("https://example.com")','name':'+CMD'}])
        book=load_workbook(BytesIO(result_workbook(payload)),data_only=False)
        cell=book['Tat_ca']['B2'];self.assertNotEqual(cell.data_type,'f');self.assertTrue(cell.value.startswith("'="));book.close()

    def test_malformed_missing_duplicate_headers_and_formula(self):
        with self.assertRaises(InputError):parse_xlsx(b'not xlsx')
        for headers,rows in [(['id'],[['x']]),(KEYS+[KEYS[0]],[[1]*24]),(KEYS,[['=1+1']+list(DEMO.values())[1:]])]:
            with self.subTest(headers=headers):
                with self.assertRaises(InputError):parse_xlsx(workbook_bytes(headers,rows))

    def test_upload_row_limit_and_empty_workbook(self):
        with self.assertRaises(InputError):parse_xlsx(workbook_bytes(KEYS,[]))
        with self.assertRaises(InputError):parse_xlsx(workbook_bytes(KEYS,[[None]*23+[1]]*(MAX_ROWS+1)))

    def test_extra_label_columns_cannot_change_predictions(self):
        code,_,body=self.post({**DEMO,'target':3,'egfr_2021_ckdepi':1,'uacr_mg_g':99999})
        expected=self.service.predict_records([DEMO])['results'][0]['target']
        self.assertEqual(code,200);self.assertEqual(json.loads(body)['results'][0]['target'],expected)


if __name__=='__main__':unittest.main()
