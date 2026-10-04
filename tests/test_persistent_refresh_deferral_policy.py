"""Persistent engineering outage ceiling is derived, bound, cumulative and finite."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from race_collection import refresh_deferral_policy as policy
from race_collection.live_freshness_contract import classify_refresh_outage
from race_collection.persistent_capacity import calculate_capacity
from race_collection import persistent_collector as collector
from scripts import run_freshness_rehearsal as run
from tests.fixtures.persistent_operation_case import make_persistent, put
from tests.test_scheduled_refresh_outage import outage
from tests.test_persistent_collector import owner_case, retain_inventory, finish_native
from tests.test_persistent_native import backend
from tests.test_persistent_refresh_recovery import deferred_fixture


def scoped(tmp_path):
    standing, _, allocation, _ = make_persistent(tmp_path/'authority')
    allocation['ends_at']='2026-10-03T09:01:00+10:00'
    allocation['cleanup_by']='2026-10-03T09:02:00+10:00'
    capacity=calculate_capacity(allocation['starts_at'], allocation['ends_at'])
    standing['daily_caps']=capacity['caps']
    standing_ref=put(tmp_path/'authority/standing.json', standing)
    allocation.update(standing_authority=standing_ref,allocation_id=standing_ref['sha256']+':2026-10-03',caps=capacity['caps'])
    allocation['limits_basis']=put(tmp_path/'authority/capacity.json',capacity)
    ref=put(tmp_path/'authority/allocation.json',allocation)
    return {'persistent_allocation':ref},allocation,capacity


def test_authenticated_daily_formula_and_historical_default(tmp_path):
    scope,allocation,capacity=scoped(tmp_path)
    expected=min(allocation['caps']['max_source_operations'],capacity['calculation']['planned_refreshes']+20)
    assert policy.limit_for(scope)==expected==26
    assert policy.limit_for({})==2
    value=policy.policy_for(scope)
    assert value['allocation']==scope['persistent_allocation']
    assert value['capacity']==allocation['limits_basis']
    assert value['standing_authority']==allocation['standing_authority']
    assert value['request_retries_added']==0 and value['provider_permission_extended'] is False


@pytest.mark.parametrize('defect',['capacity_hash','forged_refreshes','forged_recovery','caps','wrong_path','standing','study','missing_capacity'])
def test_forged_or_unbound_capacity_never_establishes_policy(tmp_path,defect):
    scope,a,c=scoped(tmp_path); ref=scope['persistent_allocation']
    if defect=='capacity_hash':Path(a['limits_basis']['path']).write_text('{}')
    elif defect=='missing_capacity':a.pop('limits_basis')
    elif defect=='standing':a['standing_authority']['sha256']='0'*64
    elif defect=='study':
        s=json.loads(Path(a['standing_authority']['path']).read_bytes());s['engineering_only']=False
        a['standing_authority']=put(Path(a['standing_authority']['path']),s);a['allocation_id']=a['standing_authority']['sha256']+':2026-10-03'
    else:
        if defect=='forged_refreshes':c['calculation']['planned_refreshes']+=100
        elif defect=='forged_recovery':c['calculation']['source_operation_recovery_allowance']+=100
        elif defect=='caps':c['caps']['max_source_operations']+=32
        p=Path(a['limits_basis']['path']) if defect!='wrong_path' else tmp_path/'foreign-capacity.json'
        a['limits_basis']=put(p,c)
    scope['persistent_allocation']=put(Path(ref['path']),a)
    with pytest.raises((ValueError,KeyError)):policy.limit_for(scope)


def test_exact_limit_preserves_old_receipt_and_success_never_resets_count(tmp_path):
    scope,a,c=scoped(tmp_path)
    failed=set();old_bytes=None
    for number in range(27):
        output,plan,rid,root,report=outage(tmp_path/'runs',run_id=f'cycle{number:03}_odds_capture')
        # First record predates amendment. Later calls bind this same daily grant.
        if number:plan.update(scope)
        result=run.record_refresh_outage(output,plan,rid,failed)
        assert result is (number<26)
        path=output/'refresh-deferrals'/('cycle000_odds_capture.json')
        if not number:old_bytes=path.read_bytes()
        assert path.read_bytes()==old_bytes
        if 0<number<26:
            value=json.loads((output/'refresh-deferrals'/(rid+'.json')).read_bytes())
            assert value['failed_cycle_count']==number+1
            assert value['maximum_failed_cycles']==26
            policy.verify_record(value,classify_refresh_outage(plan['evidence_root'],rid),allocation_sha=scope['persistent_allocation']['sha256'])
        assert json.loads(report.read_bytes())['status']=='METADATA_COVERAGE_INCOMPLETE'
    assert len(failed)==26
    assert json.loads(old_bytes)['maximum_failed_cycles']==2
    # A success adds no failed receipt and cannot reset the existing count.
    assert not (output/'refresh-deferrals'/'cycle026_odds_capture.json').exists()


@pytest.mark.parametrize('defect',['limit','count','allocation','capacity','standing','no_policy'])
def test_new_receipt_cannot_forge_policy_or_relabel_it_historical(tmp_path,defect):
    scope,a,c=scoped(tmp_path)
    _,plan,rid,_,_=outage(tmp_path/'run');classified=classify_refresh_outage(plan['evidence_root'],rid)
    value={**policy.record_fields(classified,scope),'failed_cycle_count':3}
    if defect=='limit':value['maximum_failed_cycles']+=1
    elif defect=='count':value['failed_cycle_count']=27
    elif defect=='allocation':value['allocation_sha256']='0'*64
    elif defect=='capacity':value['refresh_deferral_policy']['capacity']['sha256']='0'*64
    elif defect=='standing':value['refresh_deferral_policy']['standing_authority']['sha256']='0'*64
    else:value.pop('refresh_deferral_policy')
    with pytest.raises(ValueError):policy.verify_record(value,classified,allocation_sha=scope['persistent_allocation']['sha256'])


@pytest.mark.parametrize('age',[269,270,600])
def test_only_persistent_wait_can_outlive_freshness_but_never_grants_readiness(age):
    current=datetime(2099,1,1,tzinfo=timezone.utc)
    view=lambda:SimpleNamespace(source_generated_at=(current-timedelta(seconds=age)).isoformat())
    assert policy.index_allows_scheduled_wait(view,current,persistent=True)
    assert policy.index_allows_scheduled_wait(view,current,persistent=False) is (age<270)


@pytest.mark.parametrize('code',['CURRENT_INDEX_UNAVAILABLE','CURRENT_INDEX_STALE','DISCOVERY_TIMEOUT','CURRENT_INDEX_PATH_UNSAFE','CURRENT_INDEX_SOURCE_INVALID','CURRENT_INDEX_INVALID'])
def test_known_unavailability_only_strict_provenance_errors_still_raise(code):
    from race_collection.synchronous_manual_capture import CaptureOneRejected
    def read():raise CaptureOneRejected(code)
    current=datetime.now(timezone.utc)
    if code in {'CURRENT_INDEX_UNAVAILABLE','CURRENT_INDEX_STALE','DISCOVERY_TIMEOUT'}:
        assert policy.index_allows_scheduled_wait(read,current,persistent=True)
    else:
        with pytest.raises(CaptureOneRejected):policy.index_allows_scheduled_wait(read,current,persistent=True)
    with pytest.raises(CaptureOneRejected):policy.index_allows_scheduled_wait(read,current,persistent=False)


def test_owner_stays_unavailable_after270_seconds_until_later_verified_refresh(owner_case,monkeypatch):
    c=owner_case
    _,plan,_,retained=deferred_fixture(c.output/'outage')
    c.owner.plan['evidence_root']=plan['evidence_root']
    record=json.loads(retained.read_bytes());record['observed_at']=c.clock[0].isoformat()
    rid=record['run_id']
    record.update(policy.record_fields(classify_refresh_outage(plan['evidence_root'],rid),c.owner.plan))
    ref=put(c.output/'refresh-deferrals'/retained.name,record)
    c.owner.state['refresh_failures']=[ref];c.owner.save()
    c.owner.activate();retain_inventory(c)
    from race_collection import synchronous_manual_capture as capture
    observed=[(c.clock[0]-timedelta(seconds=10)).isoformat()]
    monkeypatch.setattr(capture,'bounded_current_race_index',lambda **k:SimpleNamespace(source_generated_at=observed[0]))
    c.clock[0]+=timedelta(seconds=400)
    c.owner.tick()
    health=json.loads((c.output/'persistent-health.json').read_bytes())
    assert health['forecast_admission_ready'] is False
    assert health['reason'] in {'UPSTREAM_TEMPORARY_UNAVAILABLE', 'CURRENT_INDEX_REFRESH_IN_PROGRESS'}
    assert len(c.children)==1 and not (c.output/'HALT.json').exists()
    finish_native(c,c.children[0]);c.owner.tick()
    assert len(c.children)==2  # Due other lane still receives normal scheduled work.
    finish_native(c,c.children[1]);c.owner.tick()
    assert json.loads((c.output/'persistent-health.json').read_bytes())['forecast_admission_ready'] is False
    observed[0]=c.clock[0].isoformat()
    c.owner.tick()
    assert json.loads((c.output/'persistent-health.json').read_bytes())['forecast_admission_ready'] is True
    assert json.loads(Path(ref['path']).read_bytes())==record
