"""Browser contract for discovery and four frozen engineering candidates."""
import json
import shutil
from pathlib import Path

import pytest
from flask import Flask, render_template

playwright=pytest.importorskip('playwright.sync_api')
ROOT=Path(__file__).parents[2]


@pytest.mark.parametrize('today',[True,False])
@pytest.mark.parametrize('case',['valid','empty','failed','unavailable','unauthenticated'])
def test_persistent_display_is_readonly_and_truthful(today,case):
    executable=shutil.which('google-chrome') or shutil.which('chromium')
    if executable is None:pytest.skip('Local Chromium required')
    app=Flask(__name__,template_folder=str(ROOT/'templates'))
    app.config['OPERATOR_UI_PERSISTENT_DISPLAY']=True
    with app.test_request_context():html=render_template('operator_ui_persistent.jinja',persistent_today=today)
    probabilities={'production':.6,'market':.4,'residual_box':.55,'residual_half':.5}
    payload={'schema':'operator_ui_persistent_collector_v1','state':'WAITING_FOR_RACE',
      'status_at':'2026-10-03T05:30:00Z','inventory_at':'2026-10-03T05:19:00Z','inventory_state':'OLDER_INVENTORY',
      'race_count':90,'upcoming':[{'venue':'Fixture venue','race_number':1,'jump_at':'2026-10-03T06:50:00Z'}],
      'forecast_errors':[], 'forecasts':[{'evidence_class':'ENGINEERING','scientific_admission':'CANARY_NOT_VERIFIED',
      'job_id':'fixture','race':{'race_id':'Fixture race','jump_timestamp':'2026-10-03T06:50:00Z'},
      'runners':[{'box':1,'name':'First runner','probabilities':probabilities},
      {'box':2,'name':'Second runner','probabilities':{name:1-p for name,p in probabilities.items()}}],
      'models':{name:None if name=='market' else 'a'*64 for name in probabilities},'published_at':'2026-10-03T06:45:00Z',
      'verified_at':'2026-10-03T06:46:00Z','manifest_sha256':'b'*64}]}
    if case=='empty':payload['forecasts']=[]
    if case=='failed':
        payload.update(state='HOLD',status_reason='OPERATIONAL_PREDICTION_FAILED_PRESERVED_CONSUMPTION',forecasts=[],failed_forecasts=[{
            'status':'FAILED','race':{'race_id':'Dubbo R1','jump_timestamp':'2026-10-03T06:50:00Z'},
            'candidates':{name:{'status':'FAILED','failure':'RESIDUAL_SCORER_FAILED'} for name in probabilities}}])
    if case=='unavailable':payload={'schema':payload['schema'],'state':'UNAVAILABLE','reason':'Evidence unavailable.'}
    requests=[]
    with playwright.sync_playwright() as driver:
        browser=driver.chromium.launch(executable_path=executable,headless=True,args=['--no-sandbox','--disable-background-networking'])
        page=browser.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
        def respond(route):
            request=route.request;path=request.url.removeprefix('http://offline.invalid').split('?',1)[0]
            requests.append((request.method,path))
            assert request.method=='GET'
            if path=='/':return route.fulfill(body=html,content_type='text/html')
            if path.startswith('/static/'):return route.fulfill(path=str(ROOT/path.lstrip('/')))
            if path=='/operator-ui/api/v1/predictions/persistent':return route.fulfill(status=401 if case=='unauthenticated' else 200,body=json.dumps(payload),content_type='application/json')
            raise AssertionError(path)
        page.route('**/*',respond);page.goto('http://offline.invalid/')
        playwright.expect(page.locator('#persistent-status')).not_to_contain_text('Checking')
        if case=='unauthenticated':playwright.expect(page.get_by_role('link',name='Sign in')).to_have_count(1)
        elif case=='unavailable':playwright.expect(page.locator('#persistent-data')).to_be_empty()
        elif case=='failed':
            playwright.expect(page.locator('#persistent-status')).to_contain_text('HOLD')
            playwright.expect(page.locator('.persistent-failure')).to_contain_text('Dubbo R1')
            playwright.expect(page.locator('.persistent-failure')).to_contain_text('RESIDUAL_SCORER_FAILED')
            playwright.expect(page.locator('.persistent-forecast')).to_have_count(0)
            assert 'COMPLETE_BEFORE_CUTOFF' not in page.inner_text('body')
            assert '%' not in page.locator('.persistent-failure').inner_text()
        elif today:
            playwright.expect(page.locator('#persistent-data')).to_contain_text('Fixture venue R1')
            playwright.expect(page.locator('#persistent-data')).to_contain_text('Discovery does not establish')
        elif case=='empty':playwright.expect(page.locator('#persistent-data')).to_contain_text('No verified four-candidate forecasts')
        else:
            playwright.expect(page.locator('.persistent-forecast')).to_have_count(1)
            for label in ['Production model','Normalized WIN market','Residual + box','Residual half']:
                playwright.expect(page.get_by_role('columnheader',name=label,exact=True)).to_have_count(1)
            playwright.expect(page.locator('.persistent-forecast')).to_contain_text('60.0000%')
            playwright.expect(page.locator('.persistent-forecast')).to_contain_text('derived from the verified captured WIN market')
        assert not errors
        browser.close()
