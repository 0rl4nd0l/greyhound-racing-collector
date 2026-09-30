"""Finite pilot admission and study accounting through the public source gate."""
from datetime import datetime
import hashlib
import json

import pytest

from scripts.run_comparison_schedule import programme_source_usage
from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked


def pilot(tmp_path):
    authority = {
        'schema_version': 'collector_development_pilot_authority_v1',
        'status': 'AUTHORIZED_DEVELOPMENT_PILOT',
        'campaign_id': 'synthetic-campaign',
        'allocation_id': 'development-single-snapshot-20261003-v1',
        'authority_reference': 'synthetic:approved-pilot',
        'allocation_sha256': '1' * 64,
        'prior_effective_authorization_sha256': '2' * 64,
        'state_root': str(tmp_path / 'runtime'),
        'dates': ['2026-10-03', '2026-10-04', '2026-10-10', '2026-10-11'],
        'max_capture_attempts': 24, 'max_attempts_per_date': 6,
        'max_logical_requests': 24000, 'max_logical_requests_per_date': 6000,
        'max_live_seconds': 28800, 'max_live_seconds_per_date': 7200,
        'max_source_operations': 192, 'max_source_operations_per_date': 48,
        'max_result_operations': 72, 'max_result_logical_requests': 720,
        'result_closure_at': '2026-10-25T12:00:00+11:00',
    }
    path = tmp_path / 'authority.json'
    path.write_text(json.dumps(authority))
    ref = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    clock = [datetime.fromisoformat('2026-10-03T12:41:00+10:00').timestamp()]
    gate = SportsbetAccess(tmp_path / 'source.json', clock=lambda: clock[0])
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'synthetic'})
    return gate, ref, clock


def allocate(gate, ref, *, day='2026-10-03', expires='2026-10-03T14:40:00+10:00', cap=48):
    gate.authorize_diagnostic(reference='synthetic:approved-pilot:slot:' + day,
        expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
        expires_at=datetime.fromisoformat(expires).timestamp(), max_operations=cap,
        rationale='synthetic approved separate allocation',
        development_authority=ref, development_slot=day)


def test_pilot_operations_preserve_global_consumption_without_borrowing_study_budget(tmp_path, monkeypatch):
    gate, ref, clock = pilot(tmp_path)
    allocate(gate, ref)
    with pytest.raises(SportsbetAccessBlocked, match='development_owner'):
        with gate.operation('python'):
            pytest.fail('unrelated consumer borrowed pilot allowance')
    monkeypatch.setenv('GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256', ref['sha256'])
    with gate.operation('python') as operation:
        operation.response(200, {})
    value = gate.read()
    assert len(value['operations']) == 1
    assert programme_source_usage(value, 0, datetime.fromisoformat('2026-10-01T12:00:00+10:00')) == 0
    clock[0] = datetime.fromisoformat('2026-10-05T13:00:00+11:00').timestamp()
    monkeypatch.delenv('GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256')
    gate.authorize_diagnostic(reference='study:slot:3', rationale='synthetic study slot',
        expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
        expires_at=clock[0]+3600, max_operations=192)
    monkeypatch.setenv('GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256', ref['sha256'])
    with pytest.raises(SportsbetAccessBlocked, match='development_allocation_not_current'):
        with gate.operation('python'):
            pytest.fail('pilot consumer borrowed study allocation')
    monkeypatch.delenv('GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256')
    with gate.operation('python') as operation:
        operation.response(200, {})
    assert len(gate.read()['operations']) == 2
    assert programme_source_usage(gate.read(), 0, datetime.fromisoformat('2026-10-01T12:00:00+10:00')) == 1


def test_pilot_day_cannot_be_reallocated_after_restart(tmp_path):
    gate, ref, clock = pilot(tmp_path)
    allocate(gate, ref)
    before = gate.path.read_bytes()
    with pytest.raises(ValueError, match='development_slot_consumed'):
        allocate(SportsbetAccess(gate.path, clock=lambda: clock[0]), ref)
    assert gate.path.read_bytes() == before


@pytest.mark.parametrize('cap,expires', [(49, '2026-10-03T14:40:00+10:00'), (48, '2026-10-03T14:40:01+10:00')])
def test_pilot_source_cap_and_deadline_are_not_expandable(tmp_path, cap, expires):
    gate, ref, _ = pilot(tmp_path)
    before = gate.path.read_bytes()
    with pytest.raises(ValueError, match='development'):
        allocate(gate, ref, cap=cap, expires=expires)
    assert gate.path.read_bytes() == before


def test_pilot_authority_cannot_clear_source_stop(tmp_path):
    gate, ref, _ = pilot(tmp_path)
    gate.retain_denial(403, reason='synthetic')
    before = gate.path.read_bytes()
    with pytest.raises((ValueError, SportsbetAccessBlocked), match='development|cooldown'):
        allocate(gate, ref)
    assert gate.path.read_bytes() == before


def test_source_accounting_rejects_relabelled_operation(tmp_path, monkeypatch):
    gate, ref, _ = pilot(tmp_path)
    allocate(gate, ref)
    monkeypatch.setenv('GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256', ref['sha256'])
    with gate.operation('python') as operation:
        operation.response(200, {})
    value = gate.read()
    value['operations'][0]['development_slot'] = '2026-10-04'
    with pytest.raises(ValueError, match='development'):
        programme_source_usage(value, 0, datetime.fromisoformat('2026-10-01T12:00:00+10:00'))


def test_all_forty_eight_operations_stay_consumed_on_restart(tmp_path, monkeypatch):
    gate, ref, clock = pilot(tmp_path)
    allocate(gate, ref)
    monkeypatch.setenv('GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256', ref['sha256'])
    for _ in range(48):
        with gate.operation('python') as operation:
            operation.response(200, {})
        clock[0] += 61
    restarted = SportsbetAccess(gate.path, clock=lambda: clock[0])
    with pytest.raises(SportsbetAccessBlocked, match='diagnostic_bound'):
        with restarted.operation('python'):
            pytest.fail('spent pilot budget was renewed')
    assert len(restarted.read()['operations']) == 48
