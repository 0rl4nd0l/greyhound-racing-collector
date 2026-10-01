"""Source contention must defer the native result queue without hiding holds."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from tests.fixtures.persistent_comparison_case import setup, patched_clock
from utils.sportsbet_access import SportsbetAccess


def test_active_source_defers_native_queue_without_charge_or_health_failure(tmp_path, monkeypatch, capsys):
    from scripts import run_comparison_result_queue as queue
    from scripts.check_comparison_health import evaluate
    from src.predictor import comparison_result_runtime as runtime

    setup(tmp_path)
    clock, scenario = patched_clock(tmp_path)
    monkeypatch.setattr(queue, 'datetime', clock)
    monkeypatch.setattr(runtime, 'datetime', clock)
    monkeypatch.setattr(sys, 'argv', ['result-queue', '--binding', str(tmp_path / 'binding.json')])
    access = SportsbetAccess(tmp_path / 'source.json')
    gate = access.read()
    gate['active'] = 'browser:invented-legitimate-owner'
    gate['denials'] = [{'status': 403, 'reason': 'invented historical denial'}]
    access.write(gate)
    source_before = access.path.read_bytes()
    ledger_before = json.loads((tmp_path / 'campaign/ledger.json').read_bytes())
    reads = []
    native_read = SportsbetAccess.read

    def read_once(source):
        reads.append(source.path)
        return native_read(source)

    monkeypatch.setattr(SportsbetAccess, 'read', read_once)

    value = queue.cycle(tmp_path / 'binding.json')
    assert value['status'] == 'SOURCE_OPERATION_BUSY'
    assert value['counts'] == {'PENDING': 1}
    assert value['request_attempts'] == 0
    now = datetime.fromisoformat(scenario['now'])
    assert evaluate({'at': now.isoformat(), 'status': 'SESSION_RUNNING'}, value,
                    now=now, schedule_active='active') == []
    with sqlite3.connect(tmp_path / 'private/queue.sqlite3') as db:
        before = db.execute('SELECT state,due,attempts FROM jobs').fetchone()
        assert before[0] == 'PENDING' and before[2] == 0
    assert queue.main() == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])['status'] == 'SOURCE_OPERATION_BUSY'
    assert reads == [access.path, access.path]
    with sqlite3.connect(tmp_path / 'private/queue.sqlite3') as db:
        assert db.execute('SELECT state,due,attempts FROM jobs').fetchone() == before
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 0
    assert access.path.read_bytes() == source_before
    assert json.loads((tmp_path / 'campaign/ledger.json').read_bytes()) == ledger_before
    assert not (tmp_path / 'synthetic-requests.jsonl').exists()
    assert not (tmp_path / 'private/attempts').exists()


@pytest.mark.parametrize('restriction', [
    'STOP', 'COOLDOWN', 'RECOVERY', 'UNKNOWN', 'prohibited', 'unresolved',
    'missing', 'invalid', 'future_not_before', 'campaign_hold',
])
def test_source_restrictions_outrank_busy_without_attempt_or_retry(tmp_path, monkeypatch, restriction):
    from scripts import run_comparison_result_queue as queue
    from scripts.check_comparison_health import evaluate
    from src.predictor import comparison_result_runtime as runtime

    setup(tmp_path)
    clock, scenario = patched_clock(tmp_path)
    monkeypatch.setattr(queue, 'datetime', clock)
    monkeypatch.setattr(runtime, 'datetime', clock)
    access = SportsbetAccess(tmp_path / 'source.json')
    gate = access.read()
    gate['active'] = 'browser:invented-active-operation'
    if restriction in {'STOP', 'COOLDOWN', 'RECOVERY', 'UNKNOWN'}:
        gate['phase'] = restriction
        gate['denials'] = [{'status': 429, 'reason': 'invented retained denial'}]
    elif restriction in {'prohibited', 'unresolved'}:
        gate['access_basis']['status'] = restriction
    elif restriction == 'future_not_before':
        gate['not_before'] = clock.now(timezone.utc).timestamp() + 600
    elif restriction == 'invalid':
        gate['active'] = {'invalid': 'owner'}
    access.path.write_text(json.dumps(gate))
    if restriction == 'missing':
        access.path.unlink()
    ledger_path = tmp_path / 'campaign/ledger.json'
    if restriction == 'campaign_hold':
        ledger = json.loads(ledger_path.read_bytes())
        ledger['source_holds'] = [{'status': 403, 'reason': 'invented campaign denial'}]
        ledger_path.write_text(json.dumps(ledger))
    before = json.loads(ledger_path.read_bytes())
    expected = 'SOURCE_HOLD' if restriction == 'campaign_hold' else 'SHARED_SOURCE_HOLD'
    result = queue.cycle(tmp_path / 'binding.json')
    assert result['status'] == expected
    assert result['request_attempts'] == 0 and result['counts'] == {'PENDING': 1}
    with sqlite3.connect(tmp_path / 'private/queue.sqlite3') as db:
        pending = db.execute('SELECT state,due,attempts FROM jobs').fetchone()
    assert queue.cycle(tmp_path / 'binding.json')['status'] == expected
    with sqlite3.connect(tmp_path / 'private/queue.sqlite3') as db:
        assert db.execute('SELECT state,due,attempts FROM jobs').fetchone() == pending
        assert pending[0] == 'PENDING' and pending[2] == 0
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 0
    assert json.loads(ledger_path.read_bytes()) == before
    assert not (tmp_path / 'private/attempts').exists()
    now = datetime.fromisoformat(scenario['now'])
    assert 'results:worker_hold_or_failure' in evaluate(
        {'at': now.isoformat(), 'status': 'NO_SLOT_DUE'}, result,
        now=now, schedule_active='inactive')


@pytest.mark.parametrize('code,accepted', [
    ('SOURCE_OPERATION_BUSY', True), ('SHARED_SOURCE_HOLD', False),
    ('SOURCE_HOLD', False), ('UNKNOWN', False),
])
def test_scheduler_accepts_only_explicit_source_busy_deferral(tmp_path, monkeypatch, code, accepted):
    from scripts import run_comparison_schedule as schedule
    from src.predictor import comparison_result_runtime as runtime
    from tests.test_persistent_schedule import config

    now = datetime.now(timezone.utc)
    cfg = config(tmp_path, monkeypatch, [
        (now - timedelta(days=1)).isoformat(), (now + timedelta(minutes=8)).isoformat()])
    root = Path(cfg['state_root'])
    first = root / 'slots/001'; first.mkdir(parents=True)
    (first / 'terminal.json').write_text('{"status":"COMPLETED"}')
    prediction = tmp_path / 'prediction'; prediction.mkdir()
    cfg['prediction_root'] = str(prediction)
    binding = tmp_path / 'binding.json'; binding.write_text('{}')
    cfg['result_binding'] = str(binding)
    result_root = tmp_path / 'results'; result_root.mkdir()
    (result_root / 'health.json').write_text(json.dumps({'status': code, 'at': now.isoformat()}))
    monkeypatch.setattr(runtime, 'load_runtime', lambda *a, **k: ({}, {}, {'state_root': str(result_root)}))
    monkeypatch.setattr(schedule, 'verify_canary', lambda *a, **k: True)
    monkeypatch.setattr(schedule.shutil, 'disk_usage', lambda _: SimpleNamespace(free=20 * 2**30))

    class AdmissionBoundary(Exception):
        pass

    def renewal(*args, **kwargs):
        raise AdmissionBoundary('No source renewal is performed by this fixture')

    monkeypatch.setattr(schedule, 'renew_source', renewal)
    if accepted:
        with pytest.raises(AdmissionBoundary):
            schedule.tick(tmp_path / 'config')
    else:
        assert schedule.tick(tmp_path / 'config')['status'] == 'RESULT_RETENTION_HOLD'


@pytest.mark.parametrize('code,accepted', [
    ('SOURCE_OPERATION_BUSY', True), ('SHARED_SOURCE_HOLD', False),
    ('SOURCE_HOLD', False), ('UNKNOWN', False),
])
def test_pilot_study_health_accepts_only_explicit_busy_deferral(tmp_path, monkeypatch, code, accepted):
    from tests.test_development_pilot import setup as setup_pilot
    from race_collection import development_pilot as pilot

    _, _, _, config, clock = setup_pilot(tmp_path, monkeypatch)
    health = tmp_path / 'result-busy-health.json'
    health.write_text(json.dumps({'status': code, 'at': clock[0].isoformat(), 'counts': {}, 'oldest_due': None}))
    config['synthetic_result_health'] = pilot.reference(health)
    if accepted:
        pilot.require_study_health(config, clock[0])
    else:
        with pytest.raises(ValueError, match='study_recovery_or_results_have_priority'):
            pilot.require_study_health(config, clock[0])
