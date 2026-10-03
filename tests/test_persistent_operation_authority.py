"""Native accounting restart/exhaustion and source control tests, no requests."""
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
from pathlib import Path
import pytest

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import digest
from race_collection.persistent_authority import load_persistent_allocation, persistent_source_usage, stamp
from tests.fixtures.persistent_operation_case import make_persistent, put
from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked


@pytest.fixture
def case(tmp_path, monkeypatch):
    standing, standing_ref, allocation, ref = make_persistent(tmp_path/'authority')
    class Clock(datetime):
        current = datetime.fromisoformat('2026-10-03T10:00:00+10:00')
        @classmethod
        def now(cls, tz=None): return cls.current.astimezone(tz or timezone.utc)
    monkeypatch.setattr('race_collection.freshness_campaign.datetime', Clock)
    root = tmp_path/'campaign'
    initial = dict(schema_version='collector_engineering_campaign_v1', campaign_id='SYNTHETIC',
                   max_capture_attempts=12,max_logical_requests=48000,max_live_seconds=10800)
    put(root/'authorization.json', initial)
    put(root/'persistent-programme-authority.json', dict(schema_version='collector_persistent_programme_v1',
        status='AUTHORIZED_PERSISTENT_PROGRAMME',campaign_id='SYNTHETIC',programme_id='STUDY',
        prior_effective_authorization_sha256=digest(initial),authority_reference='user:synthetic-study',
        starts_at='2026-10-01T12:00:00+10:00',expires_at='2027-01-21T12:00:00+11:00',
        max_capture_attempts=1000,max_logical_requests=1304000,max_live_seconds=580800,
        initial_counters=dict(capture_attempts=0,logical_requests=0,live_seconds=0)))
    put(root/'ledger.json',dict(campaign_id='SYNTHETIC',attempts=[],launches={},logical_requests=0))
    return dict(root=root,standing=standing,standing_ref=standing_ref,allocation=allocation,ref=ref,clock=Clock)


def campaign(case): return Campaign(case['root'], persistent_allocation=case['ref'])
def ledger(case): return json.loads((case['root']/'ledger.json').read_bytes())
def item(n): return dict(race_id=f'SYNTHETIC-{n}',capture_window_minutes=10)


def test_split_request_caps_restart_exhaustion_and_science_separation(case):
    c = campaign(case)
    for _ in range(3): c.request()
    for _ in range(2): c.request(kind='browser')
    restored = Campaign.from_scope({'campaign_root':str(case['root']),'persistent_allocation':case['ref']})
    before = (case['root']/'ledger.json').read_bytes()
    for kind in ['prediction','python','browser','results']:
        with pytest.raises(ValueError,match='cap_exhausted|results_forbidden'): restored.request(kind=kind)
    assert (case['root']/'ledger.json').read_bytes() == before
    used = restored.persistent_usage(ledger(case),selected=True)
    assert (used['python'],used['browser'],used['prediction'],used['results']) == (3,2,5,0)
    assert ledger(case)['logical_requests'] == 5
    assert Campaign(case['root']).programme_usage(ledger(case)) == dict(capture_attempts=0,logical_requests=0,live_seconds=0)


def test_daily_capture_cap_preserves_consumption_and_duplicate_guard(case):
    c = campaign(case)
    c.consume(case['root']/'one',item(1))
    with pytest.raises(ValueError,match='window_consumed'): c.consume(case['root']/'duplicate',item(1))
    c.consume(case['root']/'two',item(2))
    assert not campaign(case).available()
    with pytest.raises(ValueError,match='persistent_capture_allowance_consumed'): campaign(case).consume(case['root']/'three',item(3))
    assert len(ledger(case)['attempts']) == 2
    assert Campaign(case['root']).programme_usage(ledger(case))['capture_attempts'] == 0


def test_restart_reuses_open_launch_and_does_not_recharge(case):
    c = campaign(case);now=case['clock'].now(timezone.utc);deadline=stamp(case['allocation']['cleanup_by'])
    c.begin('owned-day',now=now,deadline=deadline)
    before=(case['root']/'ledger.json').read_bytes()
    campaign(case).begin('owned-day',now=now+timedelta(minutes=1),deadline=deadline)
    assert (case['root']/'ledger.json').read_bytes()==before
    with pytest.raises(ValueError,match='owner_or_launch'):campaign(case).begin('other',now=now,deadline=deadline)
    c.close('owned-day',now=now+timedelta(minutes=5))
    with pytest.raises(ValueError,match='persistent_launch_consumed'):campaign(case).begin('owned-day',now=now,deadline=deadline)
    assert Campaign(case['root']).programme_usage(ledger(case))['live_seconds']==0


def test_forged_or_changed_daily_receipt_cannot_free_science_or_reset_budget(case):
    c=campaign(case);c.request()
    old=ledger(case)
    old['persistent_operation_request_usage'][case['allocation']['allocation_id']]['persistent_allocation_sha256']='f'*64
    with pytest.raises(ValueError):Campaign(case['root']).programme_usage(old)
    changed=copy.deepcopy(case['allocation']);changed['issued_at']='2026-10-03T08:31:00+10:00'
    new_ref=put(case['root']/'replacement.json',changed)
    replacement=Campaign(case['root'],persistent_allocation=new_ref)
    with pytest.raises(ValueError,match='allocation_changed'):replacement.request()
    assert ledger(case)['logical_requests']==1


def test_source_hold_and_results_remain_blocked(case):
    c=campaign(case);c.hold_source({'reason':'synthetic-denial'})
    with pytest.raises(ValueError,match='source_hold'):c.request()
    with pytest.raises(ValueError,match='source_hold'):c.consume(case['root']/'denied',item(1))
    with pytest.raises(ValueError,match='results_forbidden'):c.persistent_window(kind='results')


@pytest.mark.parametrize('change',['overspend','result_access','date','long','root','naive'])
def test_bad_allocation_rejected(case,change):
    a=case['allocation']
    if change=='overspend':a['caps']['max_python_requests']+=1
    if change=='result_access':a['caps']['max_result_requests']=1
    if change=='date':a['racing_date']='2026-10-04'
    if change=='long':a['ends_at']='2026-10-04T13:00:00+11:00';a['cleanup_by']=a['ends_at']
    if change=='root':a['prediction_root']=case['standing']['protected_roots'][0]
    if change=='naive':a['ends_at']='2026-10-03T23:00:00'
    with pytest.raises(ValueError):load_persistent_allocation(put(case['root']/'bad.json',a))


def test_overnight_dst_window_keeps_racing_date_and_closes(case):
    a=case['allocation'];a['ends_at']='2026-10-04T08:00:00+11:00';a['cleanup_by']='2026-10-04T08:30:00+11:00'
    ref=put(case['root']/'overnight.json',a);c=Campaign(case['root'],persistent_allocation=ref)
    assert c.persistent_window(stamp('2026-10-04T04:00:00+11:00'))['racing_date']=='2026-10-03'
    with pytest.raises(ValueError,match='window_closed'):c.persistent_window(stamp(a['ends_at']))


def source(case):
    path=case['root']/'source.json';now=case['clock'].now(timezone.utc).timestamp()
    put(path,dict(schema='sportsbet_access_v1',access_basis={'status':'permitted','reference':'SYNTHETIC'},
        phase='OPEN',active=None,not_before=0,recovery_attempts=1,denials=[{'status':429,'reason':'retained'}],operations=[]))
    return SportsbetAccess(path,clock=lambda:now)


def grant(access,case):
    a=load_persistent_allocation(case['ref'])
    return access.authorize_diagnostic(reference=a['authority_reference']+':day:'+a['allocation_id'],
        expected_sha256=hashlib.sha256(access.path.read_bytes()).hexdigest(),expires_at=stamp(a['ends_at']).timestamp(),
        max_operations=a['max_source_operations'],rationale='Synthetic daily grant',persistent_allocation=case['ref'])


def test_source_grant_over192_preserves_history_identity_and_restart(case,monkeypatch):
    access=source(case);grant(access,case)
    with pytest.raises(SportsbetAccessBlocked,match='persistent_owner_required'):access.check_admission()
    monkeypatch.setenv('GREYHOUND_PERSISTENT_ALLOCATION_SHA256',case['ref']['sha256'])
    access.check_admission()
    before=access.path.read_bytes()
    with pytest.raises(ValueError,match='day_consumed'):grant(access,case)
    assert access.path.read_bytes()==before
    with access.operation('python'):pass
    state=access.read()
    assert len(state['operations'])==1 and state['recovery_attempts']==1 and len(state['denials'])==1
    assert state['operating_policy']['python_per_60_seconds']==10
    assert state['operating_policy']['browser_per_60_seconds']==1
    assert persistent_source_usage(state,0)==1
    monkeypatch.setenv('GREYHOUND_PERSISTENT_ALLOCATION_SHA256','f'*64)
    with pytest.raises(SportsbetAccessBlocked,match='persistent_owner_required'):access.check_admission()


def test_source_denial_cannot_be_reopened_by_daily_grant(case):
    access=source(case);v=access.read();v['phase']='STOP';access.write(v)
    before=access.path.read_bytes()
    with pytest.raises(SportsbetAccessBlocked,match='requires_open'):grant(access,case)
    assert access.path.read_bytes()==before


def test_next_racing_date_gets_separate_finite_budget_without_resetting_prior(case):
    c=campaign(case)
    for _ in range(3):c.request()
    a=copy.deepcopy(case['allocation']);a.update(racing_date='2026-10-04',
        allocation_id=case['standing_ref']['sha256']+':2026-10-04',
        issued_at='2026-10-04T08:30:00+11:00', starts_at='2026-10-04T09:00:00+11:00',
        ends_at='2026-10-05T01:00:00+11:00',cleanup_by='2026-10-05T01:30:00+11:00')
    for key in ('state_root','prediction_root'):a[key]=str(Path(case['standing'][key])/'days/2026-10-04')
    ref=put(case['root']/'next-date.json',a)
    case['clock'].current=datetime.fromisoformat('2026-10-04T10:00:00+11:00')
    next_day=Campaign(case['root'],persistent_allocation=ref);next_day.request()
    assert ledger(case)['logical_requests']==4
    assert next_day.persistent_usage(ledger(case),selected=True)['python']==1
    assert c.persistent_usage(ledger(case),selected=True)['python']==3
    with pytest.raises(ValueError,match='window_closed'):c.request()
    assert Campaign(case['root']).programme_usage(ledger(case))['logical_requests']==0


def test_scientific_identity_alias_remains_excluded(case):
    c=campaign(case);study=json.loads(Path(case['standing']['study_plan']['path']).read_bytes())
    race='original-study-id';root=Path(study['programme_root'])/case['standing']['study_plan']['sha256']/'attempts'
    put(root/hashlib.sha256(race.encode()).hexdigest()/'admission.json',{})
    capture={**item(1),'race_id_aliases':[race]}
    with pytest.raises(ValueError,match='study_identity_already_admitted'):c.consume(case['root']/'denied',capture)
    assert ledger(case)['attempts']==[]


def test_persistent_source_cap_exhaustion_is_truthful_and_readonly(case,monkeypatch):
    access=source(case);grant(access,case)
    monkeypatch.setenv('GREYHOUND_PERSISTENT_ALLOCATION_SHA256',case['ref']['sha256'])
    value=access.read();row=value['diagnostic_authority']
    value['operations']=[{'at':row['authorized_at'], 'kind':'python',
        'persistent_allocation_sha256':row['persistent_allocation_sha256'],
        'persistent_allocation_id':row['persistent_allocation_id']} for _ in range(row['max_operations'])]
    access.write(value);before=access.path.read_bytes()
    with pytest.raises(SportsbetAccessBlocked,match='bound_reached'):access.check_admission()
    assert access.path.read_bytes()==before
    assert persistent_source_usage(access.read(),0)==400


def test_persistent_grant_preserves_minute_limits(case,monkeypatch):
    access=source(case);grant(access,case)
    monkeypatch.setenv('GREYHOUND_PERSISTENT_ALLOCATION_SHA256',case['ref']['sha256'])
    with access.operation('browser'):pass
    with pytest.raises(SportsbetAccessBlocked,match='operating_policy_stop'):
        with access.operation('browser'):pass
    assert len(access.read()['operations'])==1
    assert access.read()['phase']=='STOP'
    with pytest.raises(SportsbetAccessBlocked,match='requires_open'):grant(access,case)
