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


def test_operational_limits_remain_cumulative_and_study_time_stays_gated(tmp_path, monkeypatch):
    import race_collection.freshness_campaign as module
    now = setup_campaign(tmp_path)
    operational = Campaign(tmp_path, engineering_authority='user:separate')
    with operational.ledger() as ledger:
        ledger['logical_requests'] = 47999
    operational.request()
    with pytest.raises(ValueError, match='cap_exhausted'):
        Campaign(tmp_path, engineering_authority='user:another-reference').request()
    for number in range(12):
        operational.consume(tmp_path/str(number), dict(race_id=str(number), capture_window_minutes=10))
    assert not operational.available()
    with pytest.raises(ValueError, match='allowance_consumed'):
        Campaign(tmp_path, engineering_authority='user:another-reference').consume(
            tmp_path/'extra', dict(race_id='extra', capture_window_minutes=10))
    operational.begin('bounded', now=now, deadline=now+timedelta(seconds=10800))
    operational.close('bounded', now=now+timedelta(seconds=10800))
    with pytest.raises(ValueError, match='time_exhausted'):
        operational.begin('extra', now=now, deadline=now+timedelta(seconds=1))
    study = Campaign(tmp_path)
    with pytest.raises(ValueError, match='persistent_programme_expired_or_not_started'):
        study.request()
    class StudyClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now+timedelta(days=3)
    monkeypatch.setattr(module, 'datetime', StudyClock)
    with pytest.raises(ValueError, match='engineering_window_overlaps_programme'):
        operational.request()
    study.request()
    ledger = json.loads((tmp_path/'ledger.json').read_bytes())
    # The 47,999 unclassified historical requests remain charged; only the one
    # newly classified operational request is excluded from study accounting.
    assert study.programme_usage(ledger) == dict(capture_attempts=0, logical_requests=48000, live_seconds=0)
    assert ledger['persistent_request_usage'] == dict(prediction=1, results=0)


def test_finite_provider_allocation_excludes_only_explicit_engineering(tmp_path):
    from utils.sportsbet_access import SportsbetAccess
    from scripts.run_comparison_schedule import programme_source_usage
    import hashlib
    clock = [datetime.now(timezone.utc).timestamp()]
    gate = SportsbetAccess(tmp_path/'source.json', clock=lambda: clock[0])
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    def allocate(reference, **extra):
        gate.authorize_diagnostic(reference=reference,
            expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
            expires_at=clock[0]+3600, max_operations=192, rationale='synthetic', **extra)
    allocate('old-unclassified')
    with gate.operation('python') as op: op.success = True
    allocate('user:separate', engineering_authority='user:separate')
    with gate.operation('python') as op: op.success = True
    start = datetime.fromtimestamp(clock[0]+7200, timezone.utc)
    assert programme_source_usage(gate.read(), 0, start) == 1
    assert len(gate.read()['operations']) == 2
    # The next lease seals the engineering interval; its operations still count.
    allocate('study:slot:1')
    with gate.operation('python') as op: op.success = True
    value = gate.read()
    assert programme_source_usage(value, 0, start) == 2
    assert programme_source_usage(value, 1, start) == 1
    assert programme_source_usage(value, 2, start) == 1
    assert programme_source_usage(value, 3, start) == 0
    # Full final-slot boundary: engineering consumption must not deny slot 80.
    value['operations'] += [{'at':clock[0], 'kind':'python'}] * (79*192-2)
    assert len(value['operations']) == 79*192+1
    assert programme_source_usage(value, 0, start) == 79*192
    value['diagnostic_authorizations'][1]['expires_at'] = start.timestamp()+1
    with pytest.raises(ValueError, match='invalid_preprogramme_source_accounting'):
        programme_source_usage(value, 0, start)


def test_separate_provider_authority_cannot_clear_denial(tmp_path):
    from utils.sportsbet_access import SportsbetAccess
    import hashlib
    gate = SportsbetAccess(tmp_path/'source.json')
    gate.initialize(access_basis={'status':'permitted','reference':'synthetic'})
    gate.retain_denial(403, reason='synthetic')
    before = gate.path.read_bytes()
    with pytest.raises(ValueError, match='invalid_separate_engineering_source_authority'):
        gate.authorize_diagnostic(reference='user:separate', engineering_authority='user:separate',
            expected_sha256=hashlib.sha256(before).hexdigest(),
            expires_at=datetime.now(timezone.utc).timestamp()+3600, max_operations=192, rationale='synthetic')
    assert gate.path.read_bytes() == before
