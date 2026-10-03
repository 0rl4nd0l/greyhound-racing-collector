"""A distinct authorized opportunity window preserves the consumed first window."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from race_collection.incident_engineering import load_incident_authority
from race_collection.live_freshness_contract import digest
from tests.test_incident_engineering import put
from tests.test_october3_incident_authority import october3

SECOND = 'user:20261003-recovery-continuation:window:002'


def second_authority(october3, tmp_path):
    first, _ = october3
    second = deepcopy(first)
    second.update(authority_reference=SECOND, incident_id='invented-october3-second',
        issued_at='2026-10-03T13:20:00+10:00',
        state_root=str(tmp_path/'second-state'), prediction_root=str(tmp_path/'second-predictions'),
        result_root=str(tmp_path/'second-results'), slots=[dict(id='001',
        starts_at='2026-10-03T16:45:00+10:00', ends_at='2026-10-03T18:15:00+10:00',
        cleanup_by='2026-10-03T18:46:00+10:00')])
    return second, put(tmp_path/'second-authority.json', second)


def test_distinct_second_reference_preserves_original_authority(october3, tmp_path):
    first, first_ref = october3
    before = Path(first_ref['path']).read_bytes()
    second, ref = second_authority(october3, tmp_path)
    assert load_incident_authority(ref) == second
    assert load_incident_authority(first_ref) == first
    assert Path(first_ref['path']).read_bytes() == before


@pytest.mark.parametrize('reference', [SECOND+':slot:001', SECOND+'x',
    'user:20261003-recovery-continuation:window:000',
    'user:20261003-recovery-continuation:window:0002',
    'user:20261003-recovery-continuation:window:2',
    'user:20261002-recovery-continuation:window:002'])
def test_only_exact_authorized_second_reference_is_accepted(october3, tmp_path, reference):
    second, ref = second_authority(october3, tmp_path)
    second['authority_reference'] = reference
    with pytest.raises(ValueError, match='invalid_incident_authority'):
        load_incident_authority(put(Path(ref['path']), second))


def test_native_renewal_preserves_first_lease_and_rejects_reuse_and_denial(october3, tmp_path, monkeypatch):
    from scripts import run_comparison_schedule as schedule
    import utils.sportsbet_access as source
    first, first_ref = october3
    second, second_ref = second_authority(october3, tmp_path)
    root = tmp_path/'campaign'
    initial = dict(schema_version='collector_engineering_campaign_v1', campaign_id='SYNTHETIC',
        max_capture_attempts=12, max_logical_requests=48000, max_live_seconds=10800)
    put(root/'authorization.json', initial)
    ledger_ref = put(root/'ledger.json', dict(campaign_id='SYNTHETIC', launches={}, attempts=[],
        logical_requests=0, source_holds=[]))
    programme = dict(schema_version='collector_persistent_programme_v1',
        status='AUTHORIZED_PERSISTENT_PROGRAMME', campaign_id='SYNTHETIC', programme_id='STUDY',
        authority_reference='SYNTHETIC_STUDY', prior_effective_authorization_sha256=digest(initial),
        starts_at='2026-10-01T12:00:00+10:00', expires_at='2027-01-21T12:00:00+11:00',
        max_capture_attempts=1000, max_logical_requests=1304000, max_live_seconds=580800,
        initial_counters=dict(capture_attempts=0,logical_requests=0,live_seconds=0))
    put(root/'persistent-programme-authority.json', programme)
    current = [datetime.fromisoformat('2026-10-03T12:50:00+10:00')]
    original_access = source.SportsbetAccess
    monkeypatch.setattr(source, 'SportsbetAccess', lambda path: original_access(path, clock=lambda:current[0].timestamp()))
    access = source.SportsbetAccess(tmp_path/'source.json')
    access.initialize(access_basis={'status':'permitted','reference':'SYNTHETIC'})
    access.authorize_diagnostic(reference='SYNTHETIC_OLD',
        expected_sha256=hashlib.sha256(access.path.read_bytes()).hexdigest(),
        expires_at=current[0].timestamp()+600, max_operations=1, rationale='SYNTHETIC')
    value = access.read()
    cfg = dict(campaign_root=str(root), programme_authority_sha256=digest(programme),
        source_state=str(access.path), lock_path=str(tmp_path/'collector.lock'),
        authority_reference=first['authority_reference'], incident_authority=first_ref, incident_slot='001',
        max_source_operations=192, source_baseline=dict(denials_sha256=digest(value['denials']),
        recovery_attempts=value['recovery_attempts'], access_basis_sha256=digest(value['access_basis']),
        operating_policy_sha256=digest(value.get('operating_policy')), operation_count=0))
    schedule.renew_source(cfg, '001', now=current[0])
    first_state = deepcopy(access.read())
    first_ledger_bytes = Path(ledger_ref['path']).read_bytes()
    with pytest.raises(ValueError, match='source_slot_already_consumed'):
        schedule.renew_source(cfg, '001', now=current[0])
    assert access.read() == first_state
    current[0] = datetime.fromisoformat('2026-10-03T16:35:00+10:00')
    cfg.update(authority_reference=second['authority_reference'], incident_authority=second_ref)
    schedule.renew_source(cfg, '001', now=current[0])
    after = access.read()
    assert after['diagnostic_authorizations'][:-1] == first_state['diagnostic_authorizations']
    assert after['diagnostic_authorizations'][-1]['reference'] == SECOND+':slot:001'
    for key in ('operations', 'denials', 'recovery_attempts'):
        assert after.get(key, []) == first_state.get(key, [])
    assert Path(ledger_ref['path']).read_bytes() == first_ledger_bytes
    with pytest.raises(ValueError, match='source_slot_already_consumed'):
        schedule.renew_source(cfg, '001', now=current[0])
    access.retain_denial(403, reason='SYNTHETIC')
    with pytest.raises(ValueError, match='source_requires_explicit_disposition'):
        schedule.renew_source(cfg, '001', now=current[0])
    assert access.read()['phase'] == 'STOP'
    assert len(access.read()['denials']) == 1


@pytest.mark.parametrize('suffix', ['001', '002', '003', '999'])
def test_positive_zero_padded_references_allow_fresh_finite_receipts(october3, tmp_path, suffix):
    second, ref = second_authority(october3, tmp_path)
    second['authority_reference'] = 'user:20261003-recovery-continuation:window:'+suffix
    assert load_incident_authority(put(Path(ref['path']), second)) == second
