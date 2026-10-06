"""Durable sidecar lifecycle tests using fabricated jobs and a fake child clock."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from race_collection import prospective_speed_runtime as runtime


class Clock:
    def __init__(self):
        self.value = datetime(2026, 10, 10, 8, 50, tzinfo=timezone.utc)

    def __call__(self):
        return self.value


def fixture_job(tmp_path, *, race_id='Race 1 - FIXTURE - 2026-10-10', activation=True):
    refs = {}
    for name in ('plan', 'population'):
        refs[name] = runtime.put_new(tmp_path/f'{name}-{hashlib.sha256(race_id.encode()).hexdigest()[:8]}.json', {})
    auth = {'status': 'AUTHORIZED_PROSPECTIVE_DEVELOPMENT' if activation else 'PROPOSED',
        'plan_sha256': refs['plan']['sha256'], 'population_sha256': refs['population']['sha256'],
        'candidate_commit': runtime.inputs.FROZEN_CANDIDATE_COMMIT,
        'additional_source_requests': 0, 'additional_result_requests': 0,
        'development_precedence_verified': True}
    refs['activation'] = runtime.put_new(tmp_path/f'activation-{hashlib.sha256(race_id.encode()).hexdigest()[:8]}.json', auth)
    job = {**refs, 'member': {'race_id': race_id, 'jump_at': '2026-10-10T09:00:00+00:00'},
        'original': {'original_published_complete_at': '2026-10-10T08:45:00+00:00'},
        'forecast_at': '2026-10-10T08:50:00+00:00'}
    return runtime.put_new(tmp_path/f'job-{hashlib.sha256(race_id.encode()).hexdigest()[:8]}.json', job)


def fake_worker(monkeypatch, clock, *, after_write=None, returncode=0, payload_change=None):
    calls = []

    def child(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 90
        assert kwargs['preexec_fn'] is runtime._limits
        if returncode:
            return SimpleNamespace(returncode=returncode)
        job_path = Path(command[command.index('--job')+1])
        output = Path(command[command.index('--output')+1])
        job = json.loads(job_path.read_bytes())
        payload = {'race_id': job['member']['race_id'], 'forecast_at': job['forecast_at'],
            'information_cutoff': job['forecast_at'], 'jump_at': job['member']['jump_at'],
            'predictions': [{'runner_id': 'fixture:1', 'baseline': 0.6, 'baseline_plus_speed': 0.6},
                            {'runner_id': 'fixture:2', 'baseline': 0.4, 'baseline_plus_speed': 0.4}]}
        payload.update(payload_change or {})
        runtime.put_new(output/'forecast.json', payload)
        if after_write:
            after_write(clock)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runtime.subprocess, 'run', child)
    monkeypatch.setattr(runtime, 'utc_now', clock)
    monkeypatch.setattr(runtime.plan, 'forecast_admission', lambda *args, **kwargs: None)
    return calls


def attempt_path(root, job_ref):
    job = json.loads(Path(job_ref['path']).read_bytes())
    return root/hashlib.sha256(job['member']['race_id'].encode()).hexdigest()


def test_replay_restart_preserves_probabilities_and_does_not_spawn_again(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock())
    first = runtime.run_job(job, root, replay=True)
    attempt = attempt_path(root, job)
    preserved = {p.name: p.read_bytes() for p in attempt.iterdir()}
    second = runtime.run_job(job, root, replay=True)
    assert first['status'] == 'REPLAY_NOT_LIVE'
    assert second['status'] == 'ATTEMPT_ALREADY_CONSUMED'
    assert second['original_status'] == 'REPLAY_NOT_LIVE'
    assert len(calls) == 1
    assert preserved == {p.name: p.read_bytes() for p in attempt.iterdir()}


def test_different_job_for_consumed_race_is_identity_failure(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock())
    runtime.run_job(job, root, replay=True)
    changed = json.loads(Path(job['path']).read_bytes())
    changed['forecast_at'] = '2026-10-10T08:51:00+00:00'
    changed_ref = runtime.put_new(tmp_path/'changed-job.json', changed)
    with pytest.raises(ValueError, match='CONSUMED_JOB_IDENTITY_CHANGED'):
        runtime.run_job(changed_ref, root, replay=True)
    assert len(calls) == 1


def test_changed_forecast_is_not_accepted_on_restart(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock())
    runtime.run_job(job, root, replay=True)
    payload = attempt_path(root, job)/'forecast.json'
    value = json.loads(payload.read_bytes()); value['predictions'][0]['baseline'] = 0.7
    payload.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='OUTPUT_CHANGED'):
        runtime.run_job(job, root, replay=True)
    assert len(calls) == 1


def test_interrupted_claim_is_consumed_and_never_recomputed(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'; root.mkdir(mode=0o700)
    attempt = attempt_path(root, job); attempt.mkdir()
    race_id = json.loads(Path(job['path']).read_bytes())['member']['race_id']
    claim = runtime.put_new(attempt/'claim.json', {'race_id': race_id, 'source_job': job,
        'mode': 'REPLAY', 'claimed_at': '2026-10-10T08:50:00+00:00'})
    calls = fake_worker(monkeypatch, Clock())
    assert runtime.run_job(job, root, replay=True)['status'] == 'INTERRUPTED_ATTEMPT'
    second = runtime.run_job(job, root, replay=True)
    assert second['original_status'] == 'INTERRUPTED_ATTEMPT'
    assert calls == []
    terminal = json.loads((attempt/'terminal.json').read_bytes())
    assert terminal['claim'] == claim
    assert not (attempt/'forecast.json').exists()


def test_invalid_activation_rejected_before_consumption_or_launch(tmp_path, monkeypatch):
    job = fixture_job(tmp_path, activation=False)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock())
    with pytest.raises(ValueError, match='ALLOCATION_ACTIVATION_NOT_VERIFIED'):
        runtime.run_job(job, root)
    assert calls == []
    assert not attempt_path(root, job).exists()


def test_payload_completed_after_jump_is_retained_as_late_failure(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock(), after_write=lambda clock: setattr(clock, 'value',
        datetime(2026, 10, 10, 9, 0, 1, tzinfo=timezone.utc)))
    result = runtime.run_job(job, root)
    attempt = attempt_path(root, job)
    assert result['status'] == 'LATE_SPEED_FORECAST'
    assert (attempt/'forecast.json').exists()
    assert not (attempt/'seal.json').exists()
    assert runtime.run_job(job, root)['original_status'] == 'LATE_SPEED_FORECAST'
    assert len(calls) == 1


def test_durable_seal_crossing_jump_is_not_reported_as_success(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    clock = Clock()
    calls = fake_worker(monkeypatch, clock)
    actual_put = runtime.put_new

    def delayed_seal(path, value):
        result = actual_put(path, value)
        if Path(path).name == 'seal.json':
            clock.value = datetime(2026, 10, 10, 9, 0, 1, tzinfo=timezone.utc)
        return result

    monkeypatch.setattr(runtime, 'put_new', delayed_seal)
    assert runtime.run_job(job, root)['status'] == 'LATE_SPEED_SEAL'
    assert runtime.run_job(job, root)['original_status'] == 'LATE_SPEED_SEAL'
    assert len(calls) == 1
    assert (attempt_path(root, job)/'forecast.json').exists()


def test_valid_prejump_seal_chain_is_checked_on_restart(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock())
    assert runtime.run_job(job, root)['status'] == 'SEALED_PREJUMP'
    assert runtime.run_job(job, root)['original_status'] == 'SEALED_PREJUMP'
    assert len(calls) == 1
    completion_path = attempt_path(root, job)/'completion.json'
    completion = json.loads(completion_path.read_bytes())
    completion['completed_at'] = '2026-10-10T09:00:00+00:00'
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match='CONSUMED_SEAL_NOT_PREJUMP'):
        runtime.run_job(job, root)


def test_worker_identity_failure_is_terminal_and_not_retried(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock(), payload_change={'race_id': 'wrong-race'})
    assert runtime.run_job(job, root, replay=True)['status'] == 'SPEED_INTEGRITY_FAILURE'
    assert runtime.run_job(job, root, replay=True)['original_status'] == 'SPEED_INTEGRITY_FAILURE'
    assert len(calls) == 1


def test_failed_race_does_not_prevent_next_independent_race(tmp_path, monkeypatch):
    first = fixture_job(tmp_path)
    second = fixture_job(tmp_path, race_id='Race 2 - FIXTURE - 2026-10-10')
    root = tmp_path/'attempts'
    failed_calls = fake_worker(monkeypatch, Clock(), returncode=1)
    assert runtime.run_job(first, root, replay=True)['status'] == 'SPEED_PROCESSING_FAILED'
    success_calls = fake_worker(monkeypatch, Clock())
    assert runtime.run_job(second, root, replay=True)['status'] == 'REPLAY_NOT_LIVE'
    assert runtime.run_job(first, root, replay=True)['original_status'] == 'SPEED_PROCESSING_FAILED'
    assert len(failed_calls) == len(success_calls) == 1


def test_child_timeout_is_consumed_and_retained(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    fake_worker(monkeypatch, Clock())
    calls = []

    def timeout(command, **kwargs):
        calls.append(command)
        raise runtime.subprocess.TimeoutExpired(command, 90)

    monkeypatch.setattr(runtime.subprocess, 'run', timeout)
    assert runtime.run_job(job, root, replay=True)['status'] == 'SPEED_PROCESSING_TIMEOUT'
    assert runtime.run_job(job, root, replay=True)['original_status'] == 'SPEED_PROCESSING_TIMEOUT'
    assert len(calls) == 1


def test_isolated_child_has_no_network_and_only_attempt_is_host_writable(tmp_path):
    output = tmp_path/'attempt'
    job = {'path': str(tmp_path/'job.json'), 'sha256': 'a'*64}
    command = runtime.isolated_command(job, output, python='/fixture/python')
    assert command[0] == 'bwrap'
    assert '--unshare-net' in command and '--die-with-parent' in command
    ro_index = command.index('--ro-bind')
    assert command[ro_index+1:ro_index+3] == ['/', '/']
    writable = [command[index+1:index+3] for index, value in enumerate(command) if value == '--bind']
    assert writable == [[str(output), str(output)]]
    assert command[command.index('--chdir')+1] == str(Path(runtime.__file__).resolve().parents[1])
    assert command[command.index('-m')+1:command.index('-m')+3] == ['scripts.run_prospective_speed', 'calculate']
    assert command[-4:] == ['--job-sha256', 'a'*64, '--output', str(output)]


def test_simultaneous_worker_cannot_launch_second_attempt(tmp_path, monkeypatch):
    job = fixture_job(tmp_path)
    root = tmp_path/'attempts'
    calls = fake_worker(monkeypatch, Clock())
    with runtime.exclusive(root):
        with pytest.raises(BlockingIOError):
            runtime.run_job(job, root, replay=True)
    assert calls == []
    assert not attempt_path(root, job).exists()
