"""Real Chromium end-to-end checks against the running Docker web service."""
from io import BytesIO
import json
import os
from pathlib import Path

from openpyxl import Workbook,load_workbook
from playwright.sync_api import sync_playwright,expect,TimeoutError as PlaywrightTimeout

BASE=os.environ.get('NHANES_WEB_URL','http://127.0.0.1:8090')
ROOT=Path(__file__).resolve().parents[2]
SHOTS=ROOT/'web/tests/screenshots';SHOTS.mkdir(exist_ok=True)


def main():
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        context=browser.new_context(viewport={'width':1440,'height':1080},accept_downloads=True)
        page=context.new_page();errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(BASE,wait_until='domcontentloaded')
        # Attempt the skill's idle check; this Docker/Chromium combination can
        # time out despite completed API calls. App readiness is asserted below.
        idle_observed=True
        try:page.wait_for_load_state('networkidle',timeout=3000)
        except PlaywrightTimeout:idle_observed=False
        expect(page.locator('#health-status')).to_have_text('Hệ thống sẵn sàng')
        expect(page.locator('#prediction-form [name="RIDAGEYR"]')).to_be_visible()
        page.screenshot(path=str(SHOTS/'desktop.png'),full_page=True)
        page.locator('#fill-demo').click();expect(page.locator('#filled-count')).to_have_text('23/23')
        page.locator('#predict-single').click();expect(page.locator('#single-result')).to_be_visible()
        expect(page.locator('#single-result .outcome')).to_have_text('Chưa gợi ý bất thường')
        expect(page.locator('#single-result .care-list')).to_have_count(0)
        with page.expect_download() as download:
            page.locator('.export-single').click()
        result_book=load_workbook(BytesIO(Path(download.value.path()).read_bytes()))
        assert result_book['Tat_ca'].max_row==2;result_book.close()
        page.locator('#field-RIDAGEYR').fill('19');page.locator('#predict-single').click()
        expect(page.locator('#global-error')).to_be_visible()
        expect(page.locator('#field-RIDAGEYR')).to_have_attribute('aria-invalid','true')
        page.locator('#reset-form').click();expect(page.locator('#filled-count')).to_have_text('0/23')
        expect(page.locator('#single-empty')).to_be_visible()
        page.locator('#guide-button').click();expect(page.locator('#detail-dialog')).to_be_visible()
        page.locator('#close-dialog').click()

        schema=context.request.get(BASE+'/api/schema').json();demo=schema['demo']
        positive={**demo,'RIDAGEYR':80,'BPQ020':1,'BPQ150':1,'DIQ010':1,
                  'sbp_mean':178,'dbp_mean':88,'LBXSUA':11,'LBXHGB':10,'BMXWAIST':110}
        response=context.request.post(BASE+'/api/predict',data=json.dumps(positive),headers={'Content-Type':'application/json'})
        assert response.ok and response.json()['results'][0]['flagged'],'Positive test fixture must produce a positive pattern.'
        keys=['id','name']+[f['key'] for f in schema['features']]
        book=Workbook();sheet=book.active;sheet.title='Du_lieu';sheet.append(keys)
        for record in [dict(demo,id='NORMAL',name='Ví dụ chưa gợi ý'),dict(positive,id='FLAG',name='Ví dụ có dấu hiệu'),dict(demo,id='ERR',name='Ví dụ lỗi',RIAGENDR=7)]:
            sheet.append([record.get(key) for key in keys])
        out=BytesIO();book.save(out)
        page.locator('[data-mode="batch"]').click()
        page.locator('#excel-file').set_input_files({'name':'kiem_thu.xlsx','mimeType':'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet','buffer':out.getvalue()})
        page.locator('#predict-batch').click();expect(page.locator('#batch-results')).to_be_visible()
        expect(page.locator('#result-rows tr')).to_have_count(3)
        page.locator('[data-filter="flagged"]').click();expect(page.locator('#result-rows tr')).to_have_count(1)
        page.locator('#result-rows button').click();expect(page.locator('#detail-dialog .care-list')).to_be_visible()
        expect(page.locator('#detail-dialog .care-list li')).to_have_count(5)
        page.screenshot(path=str(SHOTS/'positive-guidance.png'),full_page=True)
        page.locator('#close-dialog').click()
        page.locator('[data-filter="unflagged"]').click();expect(page.locator('#result-rows tr')).to_have_count(1)
        page.locator('#result-rows button').click();expect(page.locator('#detail-dialog .care-list')).to_have_count(0)
        page.locator('#close-dialog').click()
        page.locator('[data-filter="invalid"]').click();expect(page.locator('#result-rows tr')).to_have_count(1)
        expect(page.locator('#result-rows')).to_contain_text('ERR')
        page.locator('[data-filter="all"]').click()
        page.locator('#search-results').fill('FLAG');expect(page.locator('#result-rows tr')).to_have_count(1)
        page.locator('#search-results').fill('');expect(page.locator('#result-rows tr')).to_have_count(3)
        page.locator('#search-results').blur();page.evaluate('window.scrollTo(0, 0)')
        page.screenshot(path=str(SHOTS/'excel-results.png'),full_page=True)
        with page.expect_download() as download:
            page.locator('#export-batch').click()
        book=load_workbook(BytesIO(Path(download.value.path()).read_bytes()))
        assert book['Tat_ca'].max_row==4 and book['Co_dau_hieu'].max_row==2
        assert book['Chua_goi_y'].max_row==2 and book['Loi_du_lieu'].max_row==2
        book.close()
        # Client text is always inserted as text, not interpreted as HTML.
        page.locator('[data-mode="single"]').click();page.locator('#fill-demo').click()
        page.locator('#person-name').fill('<img src=x onerror=alert(1)>')
        page.locator('#predict-single').click();expect(page.locator('#single-result')).to_be_visible()
        expect(page.locator('#single-result img')).to_have_count(0)
        expect(page.locator('#single-result .person-heading')).to_contain_text('<img src=x onerror=alert(1)>')
        mobile=context.new_page();mobile.set_viewport_size({'width':390,'height':844})
        mobile.goto(BASE,wait_until='domcontentloaded');expect(mobile.locator('#health-status')).to_have_text('Hệ thống sẵn sàng')
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth'),'Mobile page has horizontal overflow.'
        mobile.screenshot(path=str(SHOTS/'mobile.png'),full_page=True)
        mobile.locator('[data-mode="batch"]').click();expect(mobile.locator('#drop-zone')).to_be_visible()
        assert mobile.evaluate('document.documentElement.scrollWidth <= innerWidth')
        assert not errors,errors
        result={'status':'passed','browser':'Chromium','networkidle_observed':idle_observed,'readiness':'DOMContentLoaded + schema fields + health ready + actual prediction responses','checks':['single prediction','invalid inputs','normal result has no advice','XLSX upload, filters, search and download','positive guidance','XSS text escaping','mobile layout','no JavaScript errors'],'screenshots':['web/tests/screenshots/'+name for name in ('desktop.png','mobile.png','excel-results.png','positive-guidance.png')]}
        (ROOT/'web/tests/browser_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        browser.close()

if __name__=='__main__':main()
