"""Synthetic service-to-receipt integration, never a live daemon invocation."""
from datetime import datetime, timedelta
import importlib.util
import contextlib
import io
import json
import multiprocessing
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from race_collection import scheduled_input_retention as retention
from scripts import autonomous_live_odds_capture as capture
from scripts import prepare_input_retention_services as package
from scripts import shadow_autopilot_daemon as daemon
from scripts import shadow_autopilot_v1 as autopilot
from tests.test_prospective_input_retention import generator_files

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('lane_receipt_fixture',
    ROOT / 'tests/race_collection/test_scheduled_forward_corpus.py')
fixtures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)


class CaptureFinished(BaseException):
    """Stop the fixture before any ordinary downstream scorer/result step."""


def installed_units(tmp_path, evidence, db, **full_options):
    directory = tmp_path / 'installed'
    common = dict(service_dir=directory, repo_path=tmp_path / 'old-source',
                  python_path=Path(sys.executable), evidence_root=evidence,
                  db_path=db, lock_path=tmp_path / 'collector.lock')
    daemon.write_service_files(**common, state_path=tmp_path / 'full-state.json',
        odds_capture_state_path=tmp_path / 'odds-state.json',
        shadow_model=tmp_path / 'unused-model', pause_path=tmp_path / 'pause', **full_options)
    daemon.write_odds_capture_service_files(**common, state_path=tmp_path / 'odds-state.json')
    return directory


def prepare(tmp_path, installed, config):
    destination = tmp_path / 'paired'
    with contextlib.redirect_stdout(io.StringIO()):
        assert package.main(['--installed-dir', str(installed), '--output-dir', str(destination),
                             '--repo-path', str(ROOT), '--retention-config', str(config)]) == 0
    return destination


@pytest.fixture
def lane_case(tmp_path, monkeypatch):
    from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence

    def add_page(sidecar):
        page = TimedResponse(requested_url=fixtures.RACE_URL, final_url=fixtures.RACE_URL,
            request_start_utc=fixtures.NOW-timedelta(seconds=1), request_end_utc=fixtures.NOW,
            status_code=200, headers={}, body=b'<html>synthetic pre-race field</html>')
        sidecar['primary_race_page_evidence'] = persist_primary_race_page_evidence(
            artifact_root=Path(sidecar['raw_export_path']).parent.parent,
            race_discovery_key=fixtures.RACE_ID, response=page, canonical_runner_set={})

    case = fixtures._fixture(tmp_path, mutate_sidecar=add_page)
    db = tmp_path / 'synthetic.db'
    with sqlite3.connect(db) as connection:
        connection.executescript("CREATE TABLE race_metadata(race_id TEXT,race_date TEXT);"
            "CREATE TABLE dog_race_data(race_id TEXT,dog_name TEXT,finish_position INTEGER);"
            "INSERT INTO race_metadata VALUES ('prior','2026-08-01');"
            "INSERT INTO dog_race_data VALUES ('prior','Alpha',1),('prior','Bravo',2);")
    config = {'schema_version': 'scheduled_input_retention_v1',
        'authority': {'approved': True, 'scope': retention.SCOPE,
            'approval_reference': 'synthetic-only-authority', 'race_ids': [fixtures.RACE_ID],
            'history_source': str(db), 'not_before': (fixtures.NOW-timedelta(minutes=1)).isoformat(),
            'expires_at': (fixtures.JUMP-timedelta(minutes=10)).isoformat()},
        'output_root': str(tmp_path / 'retained'), 'max_seconds': 15,
        'max_history_source_bytes': 268435456, 'max_bundle_bytes': 16777216,
        'cutoff_seconds_before_jump': 600,
        'static_files': {k: {'path': str(p), 'sha256': h} for k, (p, h) in generator_files(tmp_path).items()}}
    config_path = tmp_path / 'one shared config.json'
    config_path.write_text(json.dumps(config))
    installed = installed_units(tmp_path, case.evidence_root, db)
    paired = prepare(tmp_path, installed, config_path)
    now = fixtures.NOW + timedelta(seconds=5)
    clock = [now]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]

    monkeypatch.setattr(capture, 'datetime', Clock)
    monkeypatch.setattr(retention, '_now', lambda: clock[0])
    # Environmental seams only: temporary ownership, no scheduler or live I/O.
    monkeypatch.setattr(daemon, 'protected_hashes', lambda: {})
    monkeypatch.setattr(autopilot, 'protected_hashes', lambda: {})
    monkeypatch.setattr(daemon, 'copy_if_exists', lambda *a, **k: None)
    monkeypatch.setattr(daemon, 'systemd_deployment_status', lambda **k: {})
    monkeypatch.setattr(daemon, 'full_daemon_odds_window_defer_decision',
                        lambda *a, **k: {'should_defer': False})
    for name in ('acquire_lock_with_odds_capture_retry', 'acquire_lock_with_t2_due_retry'):
        monkeypatch.setattr(daemon, name, lambda **k: {'run_id': k['run_id']})
    for name in ('probe_duplicate_lock', 'probe_stale_lock_cleanup', 'simulate_timeout_recovery'):
        monkeypatch.setattr(daemon, name, lambda *a, **k: {'status': 'PASS'})
    monkeypatch.setattr(daemon, 'release_lock', lambda *a, **k: {'status': 'RELEASED'})
    monkeypatch.setattr(autopilot, 'scheduled_collector_authority',
                        lambda *a, **k: {'run_id': 'synthetic-service-capture'})
    monkeypatch.setattr(autopilot, 'prepare_manual_collector_request', lambda **k: (None, None))
    monkeypatch.setattr(autopilot, 'publish_current_race_index_after_refresh', lambda **k: {})
    # Source fetch and ordinary append are controlled synthetic boundaries. The
    # receipt producer, retention callback, claim, worker and generator are real.
    monkeypatch.setattr(capture, 'build_capture_plan', lambda *a, **k: {
        'races': [case.plan_item], 'generated_at': now.isoformat()})
    monkeypatch.setattr(capture, 'refresh_plan_item_for_time', lambda item, at: item)
    monkeypatch.setattr(capture, 'existing_capture_runner_status', lambda *a, **k: {})
    monkeypatch.setattr(capture, 'block_or_skip_existing_capture_attempt', lambda *a: False)
    monkeypatch.setattr(capture, 'fetch_odds_for_target_race_with_timeout', lambda *a, **k: {})
    monkeypatch.setattr(capture, 'validate_fetched_odds', lambda *a: case.attempt['validation'])
    monkeypatch.setattr(capture, 'append_validated_capture', lambda **k: {
        **case.attempt['append_report'], 'status': 'SUCCESS', 'inserted_rows': 4,
        'capture_timestamp': clock[0].isoformat()})

    real_popen = subprocess.Popen
    def worker_clock(command, **kwargs):
        assert command[2:4] == ['-m', 'race_collection.scheduled_input_retention']
        # Actual private worker in a fresh interpreter, with synthetic wall time.
        code = ('import json,sys; from pathlib import Path; from datetime import datetime; '
                'from race_collection import scheduled_input_retention as m; '
                'm._now=lambda:datetime.fromisoformat(sys.argv[2]); p=Path(sys.argv[1]); '
                '\ntry: result=m._retain(p)'
                '\nexcept Exception as e: result={"status":"REJECTED","reason":m.failure_code(e)}'
                '\n(p.parent/"worker-result.json").write_text(json.dumps(result))')
        return real_popen([sys.executable, '-B', '-c', code, command[-1], clock[0].isoformat()], **kwargs)
    monkeypatch.setattr(retention.subprocess, 'Popen', worker_clock)

    reports = []
    def step(*, name, command, **kwargs):
        if name in ('refresh_prejump_races', 'refresh_odds_capture_candidates'):
            return {'name': name, 'status': 'PASS', 'returncode': 0}
        assert name == 'autonomous_live_odds_capture', 'no scorer or result command permitted'
        index = next(i for i, value in enumerate(command) if value.endswith('/autonomous_live_odds_capture.py'))
        with contextlib.redirect_stdout(io.StringIO()):
            capture.main(command[index+1:])
        out = Path(command[command.index('--output-dir')+1])
        report = json.loads((out / 'autonomous_live_odds_capture_report.json').read_bytes())
        reports.append(report)
        raise CaptureFinished
    monkeypatch.setattr(autopilot, 'step_command', step)

    def run_command(*, name, command, **kwargs):
        assert name in ('autopilot_cycle', 'odds_capture_autopilot_cycle')
        autopilot.main(command[2:])
        pytest.fail('fixture must stop immediately after capture')
    monkeypatch.setattr(daemon, 'run_command', run_command)

    def run(lane, run_id='first', label='retention-configured'):
        clock[0] = now + timedelta(seconds=lane*10 + len(reports))
        service = paired / label / package.SERVICES[lane]
        command = package.service_command(service.read_text())
        with pytest.raises(CaptureFinished):
            daemon.main(command[2:] + ['--run-id', f'{lane}-{run_id}', '--current-time', clock[0].isoformat()])
        return reports[-1]['attempts'][0]
    return run, config_path, tmp_path / 'retained', db


@pytest.mark.parametrize('lane', [0, 1], ids=['full-daemon', 'odds-capture'])
def test_generated_service_reaches_real_receipt_and_retention(lane_case, lane):
    run, config, retained, db = lane_case
    attempt = run(lane)
    assert attempt['status'] == 'APPENDED'
    assert attempt['collector_exact_receipt_publish']['status'] == 'PUBLISHED'
    assert attempt['input_retention']['status'] == 'RETAINED'
    assert len(list(retained.glob('*/terminal.json'))) == 1


@pytest.mark.parametrize('mode', ['disabled', 'expired', 'absent'])
@pytest.mark.parametrize('lane', [0, 1])
def test_service_authority_gate_precedes_history_stat(lane_case, mode, lane):
    run, path, retained, db = lane_case
    config = json.loads(path.read_bytes())
    if mode == 'disabled':
        config['authority']['approved'] = False
    elif mode == 'expired':
        config['authority']['expires_at'] = fixtures.NOW.isoformat()
    path.write_text(json.dumps(config))
    db.unlink()  # An attempted history stat would fail.
    attempt = run(lane, label='default-off' if mode == 'absent' else 'retention-configured')
    assert attempt['status'] == 'APPENDED'
    if mode == 'absent':
        assert 'input_retention' not in attempt
    else:
        assert attempt['input_retention']['reason'] == 'HISTORY_ACCESS_NOT_AUTHORIZED'
    assert not retained.exists()


def test_duplicate_cross_lane_and_restart_keep_one_claim(lane_case):
    run, config, retained, db = lane_case
    assert run(0)['input_retention']['status'] == 'RETAINED'
    for lane, identity in [(1, 'other-lane'), (0, 'restart')]:
        assert run(lane, identity)['input_retention']['reason'] == 'RETENTION_ALREADY_ATTEMPTED'
    assert len(list(retained.glob('*/request.json'))) == 1


def test_concurrent_service_observations_keep_one_worker(lane_case, tmp_path):
    run, config, retained, db = lane_case
    context = multiprocessing.get_context('fork')
    start = context.Event()
    def observe(lane):
        start.wait(10)
        attempt = run(lane, 'concurrent')
        (tmp_path / f'worker-{lane}.json').write_text(json.dumps(attempt['input_retention']))
    processes = [context.Process(target=observe, args=(lane,)) for lane in (0, 1)]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0
    results = [json.loads((tmp_path / f'worker-{lane}.json').read_bytes()) for lane in (0, 1)]
    # Deliberately bypassing the collector lock permits source receipt races;
    # the sole worker may conservatively reject those changing inputs. The
    # required concurrency property is one consumed worker, never a second try.
    assert sum(r.get('reason') == 'RETENTION_ALREADY_ATTEMPTED' for r in results) == 1
    assert len(list(retained.glob('*/request.json'))) == 1
    assert len(list(retained.glob('*/worker-result.json'))) == 1
    assert len(list(retained.glob('*/terminal.json'))) == 1


@pytest.mark.parametrize('lane', [0, 1])
def test_failed_retention_preserves_capture_and_consumes_claim(lane_case, lane):
    run, config, retained, db = lane_case
    db.write_bytes(b'synthetic invalid SQLite input')
    attempt = run(lane)
    assert attempt['status'] == 'APPENDED'
    assert attempt['collector_exact_receipt_publish']['status'] == 'PUBLISHED'
    assert attempt['input_retention']['status'] == 'REJECTED'
    terminal = json.loads(next(retained.glob('*/terminal.json')).read_bytes())
    assert terminal['status'] == 'REJECTED'
    assert 'synthetic invalid' not in json.dumps(terminal)
    assert not list(retained.glob('*/bundle'))
    assert run(1-lane, 'after-failure')['input_retention']['reason'] == 'RETENTION_ALREADY_ATTEMPTED'


def test_package_preserves_both_rollbacks_and_rejects_split_lock(tmp_path):
    installed = installed_units(tmp_path, tmp_path / 'evidence', tmp_path / 'synthetic.db')
    originals = {p.name: p.read_bytes() for p in installed.iterdir()}
    paired = prepare(tmp_path, installed, tmp_path / 'future-config.json')
    assert all((paired / 'rollback' / name).read_bytes() == raw for name, raw in originals.items())
    assert all((installed / name).read_bytes() == raw for name, raw in originals.items())
    assert not (tmp_path / 'future-config.json').exists()
    for name in package.TIMERS:
        assert (paired / 'default-off' / name).read_bytes() == originals[name]
    unit = installed / daemon.ODDS_CAPTURE_SERVICE_NAME
    unit.write_text(unit.read_text().replace(str(tmp_path / 'collector.lock'), str(tmp_path / 'other.lock')))
    with pytest.raises(ValueError, match='share interpreter'):
        package.prepare_services(installed_dir=installed, output_dir=tmp_path / 'invalid',
                                 repo_path=ROOT, retention_config=tmp_path / 'config')
    assert not (tmp_path / 'invalid').exists()


def test_package_preserves_r3_bindings_and_source_bound_access_guard(tmp_path):
    jobs = tmp_path / 'r3 jobs.sqlite3'
    bundles = tmp_path / 'r3 prediction bundles'
    installed = installed_units(tmp_path, tmp_path / 'evidence', tmp_path / 'synthetic.db',
        skip_shadow_run=True, r3_job_store=jobs, r3_prediction_bundles=bundles)
    originals = {p.name: p.read_bytes() for p in installed.iterdir()}
    config = tmp_path / 'future-config.json'
    paired = prepare(tmp_path, installed, config)
    for label in ('default-off', 'retention-configured'):
        full = (paired / label / daemon.SERVICE_NAME).read_text()
        args = daemon.parse_args(package.service_command(full)[2:])
        assert args.skip_shadow_run is True
        assert args.r3_job_store == jobs
        assert args.r3_prediction_bundles == bundles
        assert args.enable_autonomous_result_capture is True
        for name in package.SERVICES:
            unit = (paired / label / name).read_text()
            condition = next(line for line in unit.splitlines() if line.startswith('ExecCondition='))
            assert str(ROOT / 'scripts/check_sportsbet_access.py') in condition
            assert str(tmp_path / 'old-source') not in condition
            lane = daemon.parse_args(package.service_command(unit)[2:])
            assert lane.input_retention_config == (config if label == 'retention-configured' else None)
    assert all((paired / 'rollback' / name).read_bytes() == raw for name, raw in originals.items())
    assert all((installed / name).read_bytes() == raw for name, raw in originals.items())
    assert not config.exists()


@pytest.mark.parametrize('lane', package.SERVICES)
@pytest.mark.parametrize('damage', ['wrong-script', 'missing-condition', 'changed-access-state'])
def test_package_rejects_modified_access_guard(tmp_path, lane, damage):
    installed = installed_units(tmp_path, tmp_path / 'evidence', tmp_path / 'synthetic.db')
    unit = installed / lane
    original = unit.read_text()
    if damage == 'wrong-script':
        modified = original.replace('/scripts/check_sportsbet_access.py', '/scripts/unchecked.py')
    elif damage == 'missing-condition':
        modified = ''.join(line for line in original.splitlines(keepends=True)
                           if not line.startswith('ExecCondition='))
    else:
        modified = original.replace('GREYHOUND_SPORTSBET_ACCESS_STATE=',
                                    'GREYHOUND_SPORTSBET_ACCESS_STATE=/unapproved/')
    assert modified != original
    unit.write_text(modified)
    with pytest.raises(ValueError, match='differs from supported generator'):
        prepare(tmp_path, installed, tmp_path / 'future.json')
    assert not (tmp_path / 'paired').exists()


@pytest.mark.parametrize('damage', ['missing-full-lane', 'already-armed', 'unsupported-setting'])
def test_package_refuses_partial_or_unreviewed_installation(tmp_path, damage):
    installed = installed_units(tmp_path, tmp_path / 'evidence', tmp_path / 'synthetic.db')
    full = installed / daemon.SERVICE_NAME
    if damage == 'missing-full-lane':
        full.unlink()
    elif damage == 'already-armed':
        full.write_text(full.read_text().replace('run-once ', 'run-once --input-retention-config prior.json '))
    else:
        full.write_text(full.read_text().replace('--refresh-limit 6', '--refresh-limit 7'))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare(tmp_path, installed, tmp_path / 'future.json')
    assert not (tmp_path / 'paired').exists()
