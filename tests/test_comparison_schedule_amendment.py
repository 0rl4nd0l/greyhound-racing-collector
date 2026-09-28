from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.amend_comparison_schedule import calendar
from race_collection.live_freshness_contract import digest, encoded
from race_collection.freshness_campaign import Campaign
from scripts.comparison_status import programme, render
from scripts.comparison_notifications import deliver


def test_october_calendar_preserves_local_times_and_dst():
    start, end, closure, slots = calendar('2026-10-01T12:00:00')
    assert start == '2026-10-01T12:00:00+10:00'
    assert end == '2027-01-21T12:00:00+11:00'
    assert closure == '2027-02-04T12:00:00+11:00'
    assert len(slots) == len(set(slots)) == 80
    assert slots[0] == '2026-10-01T13:00:00+10:00'
    assert slots[-1] == '2027-01-20T13:00:00+11:00'
    assert slots[1].endswith('+10:00') and slots[2].endswith('+11:00')
    assert all(datetime.fromisoformat(s).hour == 13 and datetime.fromisoformat(s).weekday()<5 for s in slots)


def campaign(tmp_path):
    base={'schema_version':'collector_engineering_campaign_v1','campaign_id':'SYNTHETIC',
          'max_capture_attempts':12,'max_logical_requests':48000,'max_live_seconds':10800}
    (tmp_path/'authorization.json').write_bytes(encoded(base))
    programme={'schema_version':'collector_persistent_programme_v1','status':'AUTHORIZED_PERSISTENT_PROGRAMME',
        'campaign_id':'SYNTHETIC','programme_id':'SYNTHETIC','authority_reference':'ORIGINAL',
        'prior_effective_authorization_sha256':digest(base),'starts_at':'2026-10-05T12:00:00+11:00',
        'expires_at':'2027-02-08T12:00:00+11:00','max_capture_attempts':1000,
        'max_logical_requests':1304000,'max_live_seconds':580800,
        'initial_counters':{'capture_attempts':0,'logical_requests':0,'live_seconds':0}}
    (tmp_path/'persistent-programme-authority.json').write_bytes(encoded(programme))
    row={'schema_version':'programme_schedule_amendment_v1','authority_reference':'NEW',
         'issued_at':'2026-09-28T10:00:00+00:00','empty_state_sha256':'a'*64,
         'prior_programme_sha256':digest(programme),'programme':{**programme,
         'starts_at':'2026-10-01T12:00:00+10:00','expires_at':'2027-02-04T12:00:00+11:00'}}
    folder=tmp_path/'programme-schedule-amendments';folder.mkdir()
    return programme,row,folder/'0001.json'


def test_amendment_chain_keeps_original_and_counters(tmp_path):
    old,row,path=campaign(tmp_path)
    before=(tmp_path/'persistent-programme-authority.json').read_bytes()
    path.write_bytes(encoded(row))
    current=Campaign(tmp_path)
    assert current.programme == row['programme']
    assert current.programme['initial_counters'] == old['initial_counters']
    assert (tmp_path/'persistent-programme-authority.json').read_bytes()==before


@pytest.mark.parametrize('change', ['budget','counter','root','broken_chain','backdate'])
def test_amendment_rejects_noncalendar_change(tmp_path,change):
    _,row,path=campaign(tmp_path)
    if change=='budget':row['programme']['max_logical_requests']+=1
    if change=='counter':row['programme']['initial_counters']={'capture_attempts':1}
    if change=='root':row['programme']['prediction_root']='/different'
    if change=='broken_chain':row['prior_programme_sha256']='b'*64
    if change=='backdate':row['issued_at']='2026-10-02T00:00:00+00:00'
    path.write_bytes(encoded(row))
    with pytest.raises(ValueError):Campaign(tmp_path)


def test_status_only_projects_counts_and_no_target_values(tmp_path):
    cfg={'state_root':str(tmp_path/'state'),'prediction_root':str(tmp_path/'pred'),
         'programme_id':'invented','slots':['2026-10-01T13:00:00+10:00'],'source_commit':'f'*40}
    claim=tmp_path/'state/slots/001';claim.mkdir(parents=True)
    (claim/'terminal.json').write_text('{"status":"FAILED_RESTORED"}')
    package=claim/'invented-001';package.mkdir()
    (package/'progress.json').write_text(json.dumps({'windows':{'eligible_observed_windows':[{'race_id':'DO_NOT_RELEASE'}]},
        'sample_count':8,'unavailable_samples_including_warmup':2,'winner':'DO_NOT_RELEASE'}))
    row=programme(cfg,{'counts':{'PENDING':1,'CLOSED':2},'oldest_outstanding_jump':'2026-09-28T00:00:00+00:00'},now=datetime(2026,9,28,1,tzinfo=timezone.utc))
    assert row['observed_opportunities']==1 and row['outstanding_results']==1
    assert row['oldest_outstanding_seconds']==3600 and row['sessions']=={'FAILED_RESTORED':1}
    assert 'DO_NOT_RELEASE' not in json.dumps(row)
    assert 'DO_NOT_RELEASE' not in render({'status':'HEALTHY','programme':row})


def test_optional_notification_no_transport_then_idempotent_redacted_delivery(tmp_path):
    cfg=tmp_path/'notification.json';state=tmp_path/'delivery.json'
    value={'status':'ALERT','alerts':['source:hold'],'private':'DO_NOT_RELEASE'}
    def forbidden(*a,**k):pytest.fail('unexpected transport')
    assert deliver(value,cfg,state,post=forbidden)=='LOCAL_ONLY_NO_DESTINATION'
    secret=tmp_path/'endpoint';secret.write_text('https://example.invalid/private-token');secret.chmod(0o600)
    cfg.write_text(json.dumps({'status':'AUTHORIZED_OPERATIONAL_ALERTS','authority_reference':'SYNTHETIC','endpoint_file':str(secret)}))
    calls=[]
    def post(*a,**k):calls.append(k);return SimpleNamespace(status_code=204,close=lambda:None)
    assert deliver(value,cfg,state,post=post)=='DELIVERED'
    assert deliver(value,cfg,state,post=forbidden)=='DELIVERED_UNCHANGED'
    assert 'DO_NOT_RELEASE' not in json.dumps(calls)
    assert 'private-token' not in state.read_text()


@pytest.mark.parametrize('consumed', ['slot','membership','prediction','queue','source','result'])
def test_preparation_rejects_consumption_without_decoding_outcomes(tmp_path,monkeypatch,consumed):
    from scripts import amend_comparison_schedule as module
    import sqlite3
    root=tmp_path/'campaign';root.mkdir()
    original,_,_=campaign(root)
    ledger={'attempts':[],'logical_requests':0,'launches':{}}
    (root/'ledger.json').write_bytes(encoded(ledger))
    source=tmp_path/'source.json';source.write_bytes(encoded({'phase':'OPEN','active':None}))
    plan={'programme_root':str(tmp_path/'members')};pp=tmp_path/'plan.json';pp.write_bytes(encoded(plan))
    result_root=tmp_path/'results';result_root.mkdir()
    result_db=result_root/'official.sqlite3'
    with sqlite3.connect(result_db) as db:
        for table in ('autonomous_official_result_evidence_races','autonomous_official_result_evidence_runners'):
            db.execute('CREATE TABLE '+table+' (private TEXT)')
    auth={'runtime':{'state_root':str(result_root)},'result_database':str(result_db)}
    ap=tmp_path/'authority.json';ap.write_bytes(encoded(auth))
    binding={'plan_sha256':digest(plan),'authority':str(ap),'authority_sha256':digest(auth)}
    bp=tmp_path/'binding.json';bp.write_bytes(encoded(binding))
    cfg={'comparison_plan':str(pp),'comparison_plan_sha256':digest(plan),'result_binding':str(bp),
         'campaign_root':str(root),'programme_authority_sha256':digest(original),
         'source_state':str(source),'state_root':str(tmp_path/'state'),'prediction_root':str(tmp_path/'pred'),
         'source_baseline':{'state_sha256':digest({'phase':'OPEN','active':None})},'lock_path':str(tmp_path/'lock')}
    assert not any(module.empty_state(cfg)['counts'].values())
    if consumed=='result':
        with sqlite3.connect(result_db) as db:db.execute("INSERT INTO autonomous_official_result_evidence_races VALUES ('DO_NOT_DECODE')")
    if consumed=='slot':(tmp_path/'state/slots/001').mkdir(parents=True)
    if consumed=='membership':
        (tmp_path/'members').mkdir();(tmp_path/'members/admission.json').write_text('not decoded')
    if consumed=='prediction':
        (tmp_path/'pred').mkdir();(tmp_path/'pred/sealed').write_text('not decoded')
    if consumed=='queue':
        with sqlite3.connect(result_root/'queue.sqlite3') as db:
            for name in ('jobs','events','requests'):db.execute('CREATE TABLE '+name+' (private TEXT)')
            db.execute("INSERT INTO requests VALUES ('not decoded')")
    if consumed=='source':source.write_text('{"phase":"STOP","active":null}')
    with pytest.raises(ValueError,match='programme_consumed_held_or_changed'):module.empty_state(cfg)
