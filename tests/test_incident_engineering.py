"""Invented incident authority and cumulative accounting; no runtime access."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import pytest


def test_legacy_incident_adjustment_is_not_new_engineering_consumption():
    from race_collection.incident_engineering import incident_usage
    ledger = {'launches': {'prior': {'incident_adjustment_reference':'retained-approved-adjustment',
                                   'charged_seconds':10800}}, 'attempts':[]}
    assert incident_usage(ledger, 'SYNTHETIC') == dict(capture_attempts=0, live_seconds=0,
        prediction=0, results=0, logical_requests=0)


@pytest.mark.parametrize('tag', ['incident_authority','incident_authority_sha256','incident_id','incident_slot'])
def test_partial_incident_consumption_tags_still_fail_closed(tag):
    from race_collection.incident_engineering import incident_usage
    with pytest.raises((ValueError, KeyError, TypeError)):
        incident_usage({'launches':{'new':{tag:'unbound','charged_seconds':1}}}, 'SYNTHETIC')


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def incident(tmp_path):
    from tests.fixtures.incident_engineering_case import make_incident
    return make_incident(tmp_path)

def test_authority_is_hash_bound_prospective_and_non_evaluative(incident):
    from race_collection.incident_engineering import load_incident_authority
    value,ref=incident
    assert load_incident_authority(ref)['slots'][0]['id']=='001'
    value['study_enrolment']=True
    changed=put(Path(ref['path']),value)
    with pytest.raises(ValueError,match='invalid_incident_authority'):load_incident_authority(ref)
    with pytest.raises(ValueError,match='invalid_incident_authority'):load_incident_authority(changed)


def test_campaign_separates_consumption_and_rejects_reused_window(incident,tmp_path,monkeypatch):
    import race_collection.freshness_campaign as module
    from race_collection.live_freshness_contract import digest
    value,ref=incident
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return datetime.fromisoformat('2026-10-01T16:11:00+10:00').astimezone(tz or timezone.utc)
    monkeypatch.setattr(module,'datetime',Clock)
    root=tmp_path/'campaign'
    initial={'schema_version':'collector_engineering_campaign_v1','campaign_id':'SYNTHETIC',
        'max_capture_attempts':12,'max_logical_requests':48000,'max_live_seconds':10800}
    put(root/'authorization.json',initial)
    put(root/'ledger.json',{'campaign_id':'SYNTHETIC','launches':{},'attempts':[],'logical_requests':0,'source_holds':[]})
    put(root/'persistent-programme-authority.json',{'schema_version':'collector_persistent_programme_v1',
        'status':'AUTHORIZED_PERSISTENT_PROGRAMME','campaign_id':'SYNTHETIC','programme_id':'STUDY',
        'prior_effective_authorization_sha256':digest(initial),'authority_reference':'SYNTHETIC_STUDY',
        'starts_at':'2026-10-01T12:00:00+10:00','expires_at':'2027-01-21T12:00:00+11:00',
        'max_capture_attempts':1000,'max_logical_requests':1304000,'max_live_seconds':580800,
        'initial_counters':{'capture_attempts':0,'logical_requests':0,'live_seconds':0}})
    campaign=module.Campaign(root,incident_authority=ref,incident_slot='001')
    now=Clock.now(timezone.utc)
    campaign.begin('one',now=now,deadline=datetime.fromisoformat(value['slots'][0]['cleanup_by']))
    item={'race_id':'Race 1 - SYN - 2026-10-01','capture_window_minutes':10}
    campaign.consume(tmp_path/'claim',item)
    campaign.request();campaign.request(kind='results')
    with pytest.raises(ValueError,match='consumed'):campaign.consume(tmp_path/'other',item)
    campaign.close('one',now=now+timedelta(minutes=2))
    with pytest.raises(ValueError,match='slot_consumed'):
        campaign.begin('another',now=now,deadline=datetime.fromisoformat(value['slots'][0]['cleanup_by']))
    ledger=json.loads((root/'ledger.json').read_bytes())
    assert len(ledger['attempts'])==1 and ledger['logical_requests']==2
    assert module.Campaign(root).programme_usage(ledger)=={'capture_attempts':0,'logical_requests':0,'live_seconds':0}
    study=json.loads(Path(value['study_plan']['path']).read_bytes())
    denied='Race 2 - SYN - 2026-10-01'
    put(Path(study['programme_root'])/value['study_plan']['sha256']/'attempts'/hashlib.sha256(denied.encode()).hexdigest()/'admission.json',{})
    with pytest.raises(ValueError,match='already_admitted'):
        campaign.consume(tmp_path/'denied',{'race_id':denied,'capture_window_minutes':10})
    campaign.incident_window(datetime.fromisoformat('2026-10-02T11:59:59+10:00'),kind='results')
    with pytest.raises(ValueError,match='window_closed'):
        campaign.incident_window(datetime.fromisoformat('2026-10-02T12:00:00+10:00'),kind='results')
    key=ref['sha256']+':001'
    ledger['incident_request_usage'][key]['counts']['results']=72
    put(root/'ledger.json',ledger)
    with pytest.raises(ValueError,match='request_cap_exhausted'):campaign.request(kind='results')
    ledger['incident_request_usage'][key]['incident_authority_sha256']='f'*64
    with pytest.raises(ValueError,match='invalid_incident_consumption'):
        module.Campaign(root).programme_usage(ledger)


@pytest.mark.parametrize('change',['backdate','long_window','overlap','late_cleanup','late_result','study_root','pilot_root','changed_model'])
def test_authority_rejects_scope_expansion(incident,change):
    from race_collection.incident_engineering import load_incident_authority
    value,ref=incident
    if change=='backdate':value['issued_at']='2026-10-01T16:11:00+10:00'
    if change=='long_window':value['slots'][0]['ends_at']='2026-10-01T17:41:00+10:00'
    if change=='overlap':value['slots'][1].update(starts_at='2026-10-01T18:00:00+10:00',ends_at='2026-10-01T19:30:00+10:00')
    if change=='late_cleanup':value['slots'][1]['cleanup_by']='2026-10-01T21:31:00+10:00'
    if change=='late_result':value['result_deadline']='2026-10-02T12:01:00+10:00'
    if change=='study_root':value['prediction_root']=json.loads(Path(value['study_plan']['path']).read_bytes())['prediction_output_roots'][0]
    if change=='pilot_root':value['result_root']=json.loads(Path(value['weekend_authority']['path']).read_bytes())['state_root']
    if change=='changed_model':value['candidate_registry']=put(Path(ref['path']).parent/'other-registry.json',{'other':True})
    ref=put(Path(ref['path']),value)
    with pytest.raises(ValueError,match='invalid_incident_authority'):load_incident_authority(ref)


def test_source_usage_requires_authenticated_lease_and_operation_tags(incident):
    from race_collection.incident_engineering import validate_incident_lease,incident_source_usage
    authority,ref=incident
    start=datetime.fromisoformat(authority['slots'][0]['starts_at']).timestamp()
    row={'incident_authority':ref,'incident_authority_sha256':ref['sha256'],'incident_id':authority['incident_id'],
        'incident_slot':'001','incident_kind':'prediction','reference':'SYNTHETIC_ONLY:slot:001','prior_phase':'OPEN',
        'authorized_at':start-60,'expires_at':start+5400,'max_operations':192,'operation_start':1}
    assert validate_incident_lease(row)['incident_id']=='invented-incident'
    operation={k:row[k] for k in ('incident_authority_sha256','incident_slot','incident_kind')};operation['at']=start
    value={'operations':[{'at':start-3600},operation],'diagnostic_authorizations':[row]}
    assert incident_source_usage(value,0)==1
    assert incident_source_usage(value,2)==0
    value['operations'][1]['incident_slot']='002'
    with pytest.raises(ValueError,match='invalid_incident_source_accounting'):incident_source_usage(value,0)
    value['operations'][1]['incident_slot']='001'
    row['prior_phase']='STOP'
    with pytest.raises(ValueError,match='invalid_incident_source_lease'):incident_source_usage(value,0)


def october2_authority(incident):
    value, ref = incident
    value.update(schema_version='collector_incident_engineering_authority_20261002_v1',
                 issued_at='2026-10-02T12:00:00+10:00',
                 collection_stop_at='2026-10-02T21:00:00+10:00',
                 cleanup_deadline='2026-10-02T21:30:00+10:00',
                 result_deadline='2026-10-04T12:00:00+11:00',
                 max_capture_attempts_per_window=32,
                 max_prediction_logical_requests_per_window=24064,
                 max_python_requests_per_window=24000,
                 max_browser_navigations_per_window=64,
                 max_result_requests_per_window=96,
                 max_result_operations_per_window=96,
                 second_window_requires_demonstrated_correction=True,
                 limits_basis={'refreshes':120, 'python_per_refresh':192, 'auxiliary':960})
    for slot in value['slots']:
        for key in ('starts_at', 'ends_at', 'cleanup_by'):
            slot[key] = slot[key].replace('2026-10-01', '2026-10-02')
    return value, put(Path(ref['path']), value)


def test_october2_profile_preserves_legacy_and_separates_higher_consumption(incident):
    from race_collection.incident_engineering import load_incident_authority, incident_usage
    legacy, oldref = incident
    oldref = put(Path(oldref['path']).with_name('legacy.json'), legacy)
    value, ref = october2_authority(incident)
    assert load_incident_authority(oldref)['max_prediction_logical_requests_per_window'] == 16000
    assert load_incident_authority(ref)['max_prediction_logical_requests_per_window'] == 24064
    ledger = {'incident_request_usage': {}}
    for bound, authority, prediction, results in [(oldref, legacy, 16000, 72), (ref, value, 24064, 96)]:
        ledger['incident_request_usage'][bound['sha256']+':001'] = dict(
            incident_authority=bound, incident_authority_sha256=bound['sha256'],
            incident_id=authority['incident_id'], incident_slot='001',
            counts={'prediction':prediction, 'results':results})
    assert incident_usage(ledger, 'SYNTHETIC')['logical_requests'] == 40232
    assert incident_usage(ledger, 'SYNTHETIC', authority_sha256=ref['sha256'])['logical_requests'] == 24160
    ledger['incident_request_usage'][oldref['sha256']+':001']['counts']['prediction'] += 1
    with pytest.raises(ValueError, match='invalid_incident_request_consumption'):
        incident_usage(ledger, 'SYNTHETIC')


@pytest.mark.parametrize('key,value', [
    ('max_prediction_logical_requests_per_window',24065),
    ('max_python_requests_per_window',24001),
    ('max_browser_navigations_per_window',65),
    ('max_capture_attempts_per_window',33),
    ('max_result_requests_per_window',97),
    ('result_deadline','2026-10-04T12:01:00+11:00'),
    ('second_window_requires_demonstrated_correction',False),
    ('collection_stop_at','2026-10-02T21:01:00+10:00'),
])
def test_october2_profile_rejects_scope_expansion(incident,key,value):
    from race_collection.incident_engineering import load_incident_authority
    authority, ref = october2_authority(incident)
    authority[key] = value
    with pytest.raises(ValueError, match='invalid_incident_authority'):
        load_incident_authority(put(Path(ref['path']),authority))


def test_october2_result_lease_is_separate_and_deadline_bound(incident):
    from race_collection.incident_engineering import validate_incident_lease
    authority, ref = october2_authority(incident)
    start=datetime.fromisoformat(authority['slots'][0]['starts_at']).timestamp()
    row=dict(incident_authority=ref, incident_authority_sha256=ref['sha256'],
             incident_id=authority['incident_id'], incident_slot='001', incident_kind='results',
             reference=authority['authority_reference']+':slot:001:results', prior_phase='OPEN',
             authorized_at=start+5400, expires_at=start+6000, max_operations=96)
    assert validate_incident_lease(row)['result_deadline']=='2026-10-04T12:00:00+11:00'
    row['max_operations']=97
    with pytest.raises(ValueError, match='invalid_incident_source_lease'):
        validate_incident_lease(row)
