"""Daily authority crosses the native preparer/execution-contract seam."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path

import pytest

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import digest
from race_collection.persistent_authority import stamp
from scripts import prepare_freshness_rehearsal as prep
from scripts.run_freshness_rehearsal import execution_contract
from scripts.run_comparison_schedule import programme_source_usage
from tests.fixtures.persistent_operation_case import make_persistent, put
from tests.test_short_operational_observation import prepared
from tests.test_persistent_operation_authority import case, campaign, ledger, source, grant


def daily_arguments(prepared,tmp_path,monkeypatch):
    standing,standing_ref,a,ref=make_persistent(tmp_path/'persistent')
    initial=dict(schema_version='collector_engineering_campaign_v1',campaign_id='SYNTHETIC',
        max_capture_attempts=12,max_logical_requests=48000,max_live_seconds=10800)
    put(prepared['campaign_root']/'authorization.json',initial)
    put(prepared['campaign_root']/'ledger.json',dict(campaign_id='SYNTHETIC',attempts=[],launches={},logical_requests=0))
    monkeypatch.setattr('race_collection.freshness_campaign.Campaign',Campaign)
    comparison=dict(status='AUTHORIZED_ENGINEERING',persistent_allocation=ref,candidate_registry=standing['candidate_registry'])
    comp=put(tmp_path/'comparison.json',comparison)
    # Parent integration owns the frozen-comparison loader's new authority branch;
    # this fixture supplies its resolved return without bypassing Campaign/daily validation.
    monkeypatch.setattr('src.predictor.future_comparison.load_plan',lambda path,sha:(json.loads(path.read_bytes()),{}))
    return dict(prepared,start=stamp(a['starts_at']),persistent_allocation=ref,
                prediction_root=Path(a['prediction_root']),comparison_plan=Path(comp['path'])),a,comparison


def test_daily_preparation_preserves_overnight_bounds_and_contract_binding(prepared,tmp_path,monkeypatch):
    args,a,_=daily_arguments(prepared,tmp_path,monkeypatch)
    result=prep.prepare(**args)
    plan=json.loads(Path(result['plan']).read_bytes())
    assert (stamp(plan['ends_at'])-stamp(plan['starts_at'])).total_seconds()==16*3600
    assert plan['persistent_allocation']==args['persistent_allocation']
    assert plan['racing_date']=='2026-10-03'
    assert plan['cleanup_seconds']==1800
    assert plan['max_logical_requests']==3
    assert plan['operational_predictions']['max_jobs']==2
    assert plan['operational_predictions']['result_access'] is False
    contract=execution_contract(plan,{'source_date':'2026-10-03'})
    assert contract['persistent_allocation']==args['persistent_allocation']
    assert contract['frozen_comparison']==plan['frozen_comparison']
    assert contract['racing_date']==contract['source_date']=='2026-10-03'
    assert contract['max_logical_requests']==3


@pytest.mark.parametrize('change',['allocation','model','status'])
def test_preparer_rejects_changed_comparison_binding_before_publication(prepared,tmp_path,monkeypatch,change):
    args,a,comparison=daily_arguments(prepared,tmp_path,monkeypatch)
    if change=='allocation':comparison['persistent_allocation']={**args['persistent_allocation'],'sha256':'f'*64}
    if change=='model':comparison['candidate_registry']={**comparison['candidate_registry'],'sha256':'f'*64}
    if change=='status':comparison['status']='AUTHORIZED'
    put(args['comparison_plan'],comparison)
    with pytest.raises(ValueError,match='persistent_comparison_scope_mismatch'):prep.prepare(**args)
    assert not args['output'].exists()


def test_daily_preparer_rejects_shifted_start_and_profiles(prepared,tmp_path,monkeypatch):
    args,a,_=daily_arguments(prepared,tmp_path,monkeypatch)
    args['start']=stamp('2026-10-03T10:00:00+10:00')
    with pytest.raises(ValueError,match='scope_window_changed'):prep.prepare(**args)
    args['start']=None
    with pytest.raises(ValueError,match='native_comparison_path'):prep.prepare(**args,start_after_minutes=10)
    assert not args['output'].exists()


def test_remaining_capture_jobs_does_not_charge_daily_engineering_to_science(case):
    c=campaign(case)
    c.consume(case['root']/'one',dict(race_id='SYNTHETIC',capture_window_minutes=10))
    assert prep.remaining_capture_jobs(c)==1
    assert prep.remaining_capture_jobs(Campaign(case['root']))==1000
    c.consume(case['root']/'two',dict(race_id='SYNTHETIC2',capture_window_minutes=10))
    with pytest.raises(ValueError,match='persistent_capture_allowance_consumed'):prep.remaining_capture_jobs(c)
    assert prep.remaining_capture_jobs(Campaign(case['root']))==1000


def test_scientific_source_usage_excludes_only_authenticated_daily_operations(case,monkeypatch):
    access=source(case);grant(access,case)
    monkeypatch.setenv('GREYHOUND_PERSISTENT_ALLOCATION_SHA256',case['ref']['sha256'])
    with access.operation('python'):pass
    assert programme_source_usage(access.read(),0,stamp('2026-10-01T13:00:00+10:00'))==0
    value=access.read();value['operations'][0]['persistent_allocation_sha256']='f'*64
    with pytest.raises(ValueError,match='source_accounting_invalid'):
        programme_source_usage(value,0,stamp('2026-10-01T13:00:00+10:00'))
