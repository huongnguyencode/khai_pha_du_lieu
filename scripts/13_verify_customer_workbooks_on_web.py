"""Upload the fixture workbooks to a running web, then check its XLSX exports."""
from pathlib import Path
from io import BytesIO
from collections import Counter
import argparse
import json
import urllib.request
from openpyxl import load_workbook


def verify(directory,url):
    manifest=json.loads((directory/'kiem_tra_du_lieu.json').read_text());verified=[]
    for entry in manifest['files']:
        data=(directory/entry['file']).read_bytes()
        headers={'Content-Type':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        req=urllib.request.Request(url+'/api/predict-xlsx',data=data,headers=headers)
        with urllib.request.urlopen(req,timeout=30) as response:
            assert response.status==200;payload=json.loads(response.read())
        assert payload['counts']==entry['counts']
        assert Counter(str(row['target']) for row in payload['results'])==Counter(entry['targets'])
        book=load_workbook(BytesIO(data),data_only=True)
        for expected,actual in zip(book['Doi_chieu'].iter_rows(min_row=2,values_only=True),payload['results']):
            assert expected[1]==actual['id'] and expected[3]==actual['target'] and expected[6]==len(actual['advice'])
            if actual['target']==0:assert not actual['advice']
        book.close()
        req=urllib.request.Request(url+'/api/predict-xlsx?download=1',data=data,headers=headers)
        with urllib.request.urlopen(req,timeout=30) as response:
            assert response.status==200;output=response.read()
        book=load_workbook(BytesIO(output),data_only=True)
        assert book['Tat_ca'].max_row-1==entry['people']
        assert book['Co_dau_hieu'].max_row-1==entry['counts']['flagged']
        assert book['Chua_goi_y'].max_row-1==entry['counts']['unflagged']
        assert book['Loi_du_lieu'].max_row==1
        book.close()
        verified.append({'file':entry['file'],'upload':'passed','download':'passed','counts':payload['counts']})
    return {'status':'passed','files_verified':len(verified),'checks':verified}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,default=Path(__file__).resolve().parent.parent/'khach_hang')
    parser.add_argument('--url',default='http://127.0.0.1:8090')
    args=parser.parse_args()
    print(json.dumps(verify(args.directory,args.url.rstrip('/')),ensure_ascii=False,indent=2))
