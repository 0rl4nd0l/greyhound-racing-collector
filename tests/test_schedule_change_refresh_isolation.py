"""Only complete replayable canonical timing changes are local exclusions."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from hashlib import sha256
import json

import pytest

from scripts.capture_thedogs_market_history import TimedResponse, _response_receipt
from scripts.refresh_prejump_upcoming import has_unisolated_refresh_failure
from tests.test_empty_eligible_refresh import report_fixture, publish
from tests.test_race_local_quarantine_refresh import reselect


def changed_report(root):
    report = report_fixture(root, eligible=True)
    report.update(upcoming_dir=str(root), quarantine_count=0,
        shared_sportsbet_snapshot={'status':'VALIDATED','payload_sha256':'a'*64})
    first = report['selected_races'][0]
    url = 'https://www.thedogs.com.au/racing/gunnedah/2026-07-19/6'
    candidate = {**first, 'race_url':url, 'race_id':'Race 6 - GUNN - 2026-07-19',
        'race_number':6, 'scheduled_jump_datetime':first['jump_datetime']}
    start = datetime.fromisoformat(report['generated_at']).astimezone(timezone.utc)
    end = start+timedelta(seconds=1)
    new = datetime.fromisoformat(first['jump_datetime'])+timedelta(minutes=2)
    body = f'<formatted-time data-format="datetime_short" data-timestamp="{int(new.timestamp())}"></formatted-time>'.encode()
    response = TimedResponse(requested_url=url,final_url=url,request_start_utc=start,
        request_end_utc=end,status_code=200,headers={'content-type':'text/html','date':format_datetime(end)},body=body)
    evidence = {'schema_version':'canonical_schedule_change_rejection_v1','race_url':url,
        'race_date':candidate['date'],'race_number':6,'discovery_jump':candidate['scheduled_jump_datetime'],
        'canonical_jump':new.isoformat(),'race_page_http':_response_receipt(response,include_body=True)}
    path=root/'schedule-change.json';raw=json.dumps(evidence,sort_keys=True).encode();path.write_bytes(raw)
    report['selected_races'].append(candidate);report['selected_count']=2
    report['downloads'].append({'race_url':url,'success':False,'result':{'success':False,
        'error':'discovery_canonical_jump_changed','schedule_change_evidence':{'path':str(path),'sha256':sha256(raw).hexdigest()}}})
    report['sidecar_metadata_coverage']['races'].append({'race_url':url,'race_id':candidate['race_id'],
        'csv_path':None,'sidecar_path':None,'weather_track_rejected_reasons':['accepted_csv_missing']})
    reselect(report)
    return report


def test_complete_changed_race_is_excluded_without_stopping_unrelated_index(tmp_path):
    report=changed_report(tmp_path)
    assert has_unisolated_refresh_failure(report) is False
    published=publish(tmp_path,tmp_path/'runtime/state.json',report,'changed')
    assert published['status']=='PUBLISHED' and published['race_count']==1
    assert report['selected_count']==2 and len(report['downloads'])==2
    assert report['current_index_metadata_selection']['excluded_race_count']==1
    assert report['downloads'][1]['success'] is False


@pytest.mark.parametrize('defect',['missing_evidence','bare_error','unknown_error','source_denial','request_exhausted','changed_evidence','shared_denial','missing_timestamp'])
def test_incomplete_or_shared_failure_cannot_become_success(tmp_path,defect):
    report=changed_report(tmp_path);result=report['downloads'][1]['result']
    if defect in ('missing_evidence','bare_error'):result.pop('schedule_change_evidence')
    elif defect=='unknown_error':result['error']='unknown'
    elif defect=='source_denial':result['source_http_status']=429
    elif defect=='request_exhausted':result['source_failure_category']='request_cap_exhausted'
    elif defect=='changed_evidence':(tmp_path/'schedule-change.json').write_text('{}')
    elif defect=='shared_denial':report['shared_sportsbet_snapshot']['status']='UNAVAILABLE'
    elif defect=='missing_timestamp':report['selected_races'][1].pop('scheduled_jump_datetime')
    assert has_unisolated_refresh_failure(report) is True


def rewrite_proof(root, report, mutate):
    path=root/'schedule-change.json';value=json.loads(path.read_bytes());mutate(value)
    raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw)
    report['downloads'][1]['result']['schedule_change_evidence']['sha256']=sha256(raw).hexdigest()


@pytest.mark.parametrize('defect',['missing_epoch','conflicting_epochs','invalid_date','wrong_url','redirect','denial','retry_header','stale_response','same_instant','naive','old_alias','included_rejected_race'])
def test_replay_rejects_false_or_incomplete_schedule_proof(tmp_path,defect):
    import base64
    report=changed_report(tmp_path)
    def mutate(v):
        response=v['race_page_http']
        if defect in ('missing_epoch','conflicting_epochs'):
            body=b'<html>No timestamp</html>' if defect=='missing_epoch' else base64.b64decode(response['body_base64'])+b'<formatted-time data-format="datetime_short" data-timestamp="1"></formatted-time>'
            response.update(body_base64=base64.b64encode(body).decode(),body_sha256=sha256(body).hexdigest(),body_bytes=len(body))
        elif defect=='invalid_date':v['race_date']='2026-07-20'
        elif defect=='wrong_url':v['race_url']=v['race_url'].replace('/6','/7')
        elif defect=='redirect':response['final_url']+='-different'
        elif defect=='denial':response['status_code']=403
        elif defect=='retry_header':response['headers']['retry-after']='60'
        elif defect=='stale_response':response['headers']['date']='Thu, 01 Jan 1970 00:00:00 GMT'
        elif defect=='same_instant':
            v['discovery_jump']=v['canonical_jump'];report['selected_races'][1].update(scheduled_jump_datetime=v['canonical_jump'],jump_datetime=v['canonical_jump'])
        elif defect=='naive':v['canonical_jump']='2026-07-19T13:07:00'
        elif defect=='old_alias':report['selected_races'][1]['jump_datetime']='2026-07-19T13:06:00+10:00'
        elif defect=='included_rejected_race':report['current_index_races'].append(report['selected_races'][1])
    rewrite_proof(tmp_path,report,mutate)
    assert has_unisolated_refresh_failure(report) is True


@pytest.mark.parametrize('status',['REFRESH_BUDGET_EXCEEDED','ACQUISITION_INCOMPLETE','REQUEST_CAP_EXHAUSTED','DISCOVERY_FAILED'])
def test_non_success_report_still_never_publishes_partial_index(tmp_path,status):
    report=changed_report(tmp_path);report['status']=status
    assert publish(tmp_path,tmp_path/'runtime/state.json',report,'blocked')['status']=='REJECTED'


def test_only_changed_race_produces_explicit_empty_index_with_zero_csv_quarantines(tmp_path):
    report=changed_report(tmp_path)
    for key in ['selected_races','downloads']:report[key]=report[key][1:]
    report['sidecar_metadata_coverage']['races']=report['sidecar_metadata_coverage']['races'][1:]
    report.update(status='NO_QUALIFIED_RACES',selected_count=1,accepted_csv_count=0,sidecar_count=0)
    reselect(report)
    result=publish(tmp_path,tmp_path/'runtime/state.json',report,'empty')
    assert result['status']=='PUBLISHED' and result['race_count']==0
    assert report['quarantine_count']==0 and report['downloads'][0]['success'] is False


def test_real_download_retains_changed_page_without_attempting_csv(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import upcoming_race_browser as module
    from utils.race_schedule_rejection import complete_schedule_change_rejection
    report=changed_report(tmp_path);candidate=report['selected_races'][1];url=candidate['race_url']
    evidence=json.loads((tmp_path/'schedule-change.json').read_bytes())
    import base64
    response=SimpleNamespace(status_code=200,url=url,headers=evidence['race_page_http']['headers'],
        content=base64.b64decode(evidence['race_page_http']['body_base64']),close=lambda:None)
    calls=[]
    browser=module.UpcomingRaceBrowser.__new__(module.UpcomingRaceBrowser)
    browser.upcoming_dir=str(tmp_path)
    browser.session=SimpleNamespace(get=lambda *a,**k:(calls.append(a[0]) or response))
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return datetime.fromisoformat(report['generated_at']).astimezone(tz)
    monkeypatch.setattr(module,'datetime',Clock)
    monkeypatch.setattr(browser,'extract_detailed_race_info',lambda *a:{'date':candidate['date'],'race_number':6,'url':url})
    monkeypatch.setattr(browser,'_extract_safe_target_metadata_from_page',lambda *a,**k:{})
    monkeypatch.setattr(browser,'_extract_safe_target_grade_from_hint',lambda *a,**k:{})
    monkeypatch.setattr(browser,'_merge_safe_target_metadata',lambda *a,**k:{})
    monkeypatch.setattr(browser,'_extract_safe_weather_track_metadata_from_page',lambda *a,**k:{})
    result=browser.download_race_csv(url,race_info_hint={**candidate,'url':url})
    assert result['success'] is False and result['error']=='discovery_canonical_jump_changed'
    assert complete_schedule_change_rejection(candidate,result,str(tmp_path))
    assert calls==[url]
