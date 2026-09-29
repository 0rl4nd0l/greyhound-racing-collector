"""Separate operational authority through real package/contract campaign seams."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest
from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import create_once, digest, FreshnessContract
from scripts.run_freshness_rehearsal import execution_contract
from tests.test_short_operational_observation import prepared


def setup_campaign(root):
    root.mkdir(exist_ok=True)
    base = dict(schema_version='collector_engineering_campaign_v1', campaign_id='synthetic',
                max_capture_attempts=12, max_logical_requests=48000, max_live_seconds=10800)
    create_once(root/'authorization.json', base)
    now = datetime.now(timezone.utc)
    programme = dict(schema_version='collector_persistent_programme_v1',
        status='AUTHORIZED_PERSISTENT_PROGRAMME', campaign_id='synthetic', programme_id='study',
        authority_reference='study-only', prior_effective_authorization_sha256=digest(base),
        starts_at=(now+timedelta(days=2)).isoformat(), expires_at=(now+timedelta(days=126)).isoformat(),
        max_capture_attempts=1012, max_logical_requests=1352000, max_live_seconds=591600,
        initial_counters=dict(capture_attempts=0, logical_requests=0, live_seconds=0))
    create_once(root/'persistent-programme-authority.json', programme)
    (root/'ledger.json').write_text(json.dumps(dict(campaign_id='synthetic', attempts=[],
                                                  logical_requests=0, launches={})))
    return now


def test_explicit_operational_authority_admits_without_study_slots_or_budget(tmp_path):
    now = setup_campaign(tmp_path)
    programme_bytes = (tmp_path/'persistent-programme-authority.json').read_bytes()
    original = Campaign(tmp_path)
    with pytest.raises(ValueError, match='persistent_programme_expired_or_not_started'):
        original.begin('unauthorized', now=now, deadline=now+timedelta(minutes=5))
    operational = Campaign(tmp_path, engineering_authority='user:separate-operational')
    assert operational.value['max_capture_attempts'] == 12
    operational.begin('operational', now=now, deadline=now+timedelta(seconds=7260))
    operational.request()
    operational.consume(tmp_path/'claim', dict(race_id='race', capture_window_minutes=10))
    operational.close('operational', now=now+timedelta(seconds=60))
    ledger = json.loads((tmp_path/'ledger.json').read_bytes())
    assert ledger['logical_requests'] == 1 and len(ledger['attempts']) == 1
    assert ledger['launches']['operational']['charged_seconds'] == 60
    assert original.programme_usage(ledger) == dict(capture_attempts=0, logical_requests=0, live_seconds=0)
    assert 'persistent_request_usage' not in ledger
    assert (tmp_path/'persistent-programme-authority.json').read_bytes() == programme_bytes
    assert not (tmp_path/'slots').exists()
    with pytest.raises(ValueError, match='window_consumed'):
        operational.consume(tmp_path/'retry', dict(race_id='race', capture_window_minutes=10))
    with pytest.raises(ValueError, match='engineering_results_forbidden'):
        operational.request(kind='results')


def test_engineering_cannot_cross_study_start_or_remove_shared_holds(tmp_path):
    now = setup_campaign(tmp_path)
    c = Campaign(tmp_path, engineering_authority='user:separate')
    with pytest.raises(ValueError, match='engineering_window_overlaps_programme'):
        c.begin('overlap', now=now, deadline=now+timedelta(days=3))
    c.hold_source({'status':429})
    for call in (lambda: c.begin('held', now=now, deadline=now+timedelta(minutes=5)),
                 lambda: c.request(),
                 lambda: c.consume(tmp_path/'claim', dict(race_id='race', capture_window_minutes=10))):
        with pytest.raises(ValueError, match='campaign_source_hold'):
            call()


def test_real_preparer_contract_preserves_explicit_scope(prepared, monkeypatch):
    from scripts import prepare_freshness_rehearsal as prep
    import race_collection.freshness_campaign as module
    monkeypatch.setattr(module, 'Campaign', Campaign)
    now = setup_campaign(prepared['campaign_root'])
    prepared['start'] = now+timedelta(minutes=10)
    before = (prepared['campaign_root']/'ledger.json').read_bytes()
    prep.prepare(**prepared, engineering_authority='user:separate-operational')
    plan = json.loads((prepared['output']/'plan.json').read_bytes())
    accounting = {'source_date':prepared['start'].astimezone(__import__('zoneinfo').ZoneInfo('Australia/Melbourne')).date().isoformat()}
    scope = FreshnessContract(execution_contract(plan, accounting))
    scope.campaign.begin(plan['rehearsal_id'], now=now, deadline=now+timedelta(seconds=7260))
    scope.campaign.request()
    assert scope.campaign.programme is None
    assert scope.campaign.value['max_capture_attempts'] == 12
    assert plan['engineering_authority'] == 'user:separate-operational'
    assert plan['campaign_authorization_sha256'] == digest(scope.campaign.value)
    # Dropping the binding cannot silently select the other authority.
    bad = execution_contract(plan, accounting)
    del bad['engineering_authority']
    with pytest.raises(ValueError, match='campaign_authorization_changed'):
        FreshnessContract(bad)
    assert before != (prepared['campaign_root']/'ledger.json').read_bytes()


@pytest.mark.parametrize('changes', [{'comparison_plan':Path('/unused')},
                                     {'prediction_root':Path('/unused')},
                                     {'operational_predictions':False}])
def test_engineering_never_routes_to_study(prepared, changes):
    from scripts.prepare_freshness_rehearsal import prepare
    prepared.update(changes)
    with pytest.raises(ValueError, match='engineering_requires_separate_operational_predictions'):
        prepare(**prepared, engineering_authority='user:separate')
    assert not prepared['output'].exists()
