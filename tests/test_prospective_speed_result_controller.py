"""Controller ownership and restart, with fabricated clocks and no service IO."""
from datetime import datetime
from pathlib import Path

import pytest

from race_collection import prospective_speed_result_controller as controller

NOW = datetime.fromisoformat('2026-10-11T06:00:00+11:00')


def inventory(jump='2026-10-11T08:00:00+11:00'):
    return {'schema_version': 'daily_race_inventory_v1', 'status': 'COMPLETE',
        'source_date': '2026-10-11', 'observed_at': NOW.isoformat(),
        'race_count': 1, 'discovery_failures': [], 'races': [{
            'date': '2026-10-11', 'race_number': 1,
            'url': 'https://www.thedogs.com.au/racing/dubbo/2026-10-11/1',
            'scheduled_jump_datetime': jump}]}


def test_verified_idle_census_has_gap():
    health = {'status': 'WAITING_FOR_RACE', 'children': [], 'at': NOW.isoformat()}
    assert controller.safe_gap(health, inventory(), NOW) == datetime.fromisoformat('2026-10-11T08:00:00+11:00')


@pytest.mark.parametrize('change', ['active', 'child', 'stale', 'near', 'headroom', 'unknown', 'incomplete', 'wrong_day'])
def test_never_pauses_on_incomplete_or_active_view(change):
    health = {'status': 'WAITING_FOR_RACE', 'children': [], 'at': NOW.isoformat()}
    value = inventory()
    if change == 'active': health['status'] = 'ACTIVE_COLLECTION'
    if change == 'child': health['children'] = ['inventory']
    if change == 'stale': health['at'] = '2026-10-11T05:00:00+11:00'
    if change == 'near': value = inventory('2026-10-11T06:59:00+11:00')
    if change == 'headroom': value = inventory('2026-10-11T07:05:00+11:00')
    if change == 'unknown': value['races'][0]['scheduled_jump_datetime'] = None
    if change == 'incomplete': value['discovery_failures'] = ['missing meeting']
    if change == 'wrong_day': value['source_date'] = '2026-10-10'
    with pytest.raises(ValueError): controller.safe_gap(health, value, NOW)


class Host:
    def __init__(self): self.actions = []; self.blocked = False; self.resume_failure = False
    def ready(self):
        self.actions.append('ready')
        if self.blocked: raise ValueError('WAIT_FOR_IDLE_COLLECTOR')
        return {'preparation': {'path': '/fixture', 'sha256': 'a'*64}}
    def stop(self): self.actions.append('stop')
    def quiet(self, intent): self.actions.append('quiet'); return {'fixture': True}
    def resume(self, intent):
        self.actions.append('resume')
        if self.resume_failure: raise ValueError('SOURCE_HOLD')
        return 'RESUMED'


def setup(tmp_path, monkeypatch):
    state = tmp_path/'controller'
    config = {'schema_version': 'prospective_speed_result_controller_v1',
        'status': 'AUTHORIZED_EXISTING_DEVELOPMENT_QUIET_GAPS', 'bridge': {}, 'state_root': str(state)}
    ref = controller.runtime.put_new(tmp_path/'config.json', config)
    monkeypatch.setattr(controller.runtime, 'utc_now', lambda: NOW)
    monkeypatch.setattr(controller.results, 'load_config', lambda ref: ({}, {
        'schema_version': 'prospective_sectional_plan_v1', 'dates': ['2026-10-10', '2026-10-11']}, {}))
    monkeypatch.setattr(controller.results, 'prepare_queue', lambda ref: None)
    due = {'selected_due': 1, 'transport_permitted_by_time_and_budget': True}
    monkeypatch.setattr(controller.results, 'inspect_queue', lambda ref: due.copy())
    host = Host(); calls = []
    def run(ref):
        calls.append('request'); due['selected_due'] = 0
        return {'status': 'RESULT_CLOSED'}
    monkeypatch.setattr(controller.results, 'run_cycle', run)
    return ref, state, host, calls, due


def test_request_only_after_durable_intent_and_quiet_then_resume(tmp_path, monkeypatch):
    ref, state, host, calls, _ = setup(tmp_path, monkeypatch)
    def run(_):
        assert host.actions == ['ready', 'stop', 'quiet']
        assert len(list(state.glob('batch-*/intent.json'))) == 1
        calls.append('request')
        return {'status': 'SOURCE_STOP'}
    monkeypatch.setattr(controller.results, 'run_cycle', run)
    assert controller.cycle(ref, host_factory=lambda _: host)['status'] == 'RESULT_BATCH_COMPLETE_COLLECTOR_RUNNING'
    assert host.actions == ['ready', 'stop', 'quiet', 'resume'] and len(calls) == 1


def test_active_collection_not_stopped(tmp_path, monkeypatch):
    ref, state, host, calls, _ = setup(tmp_path, monkeypatch); host.blocked = True
    assert controller.cycle(ref, host_factory=lambda _: host)['status'] == 'WAIT_FOR_VERIFIED_COLLECTION_GAP'
    assert host.actions == ['ready'] and not calls and not list(state.glob('batch-*'))


def test_failure_still_resumes_without_repeat_request(tmp_path, monkeypatch):
    ref, state, host, calls, _ = setup(tmp_path, monkeypatch)
    def fail(_): calls.append('failed'); raise RuntimeError('transport exception')
    monkeypatch.setattr(controller.results, 'run_cycle', fail)
    with pytest.raises(RuntimeError): controller.cycle(ref, host_factory=lambda _: host)
    assert host.actions[-1] == 'resume' and len(list(state.glob('batch-*/complete.json'))) == 1
    controller.cycle(ref, recover_only=True, host_factory=lambda _: host)
    assert len(calls) == 1


def test_unverified_resume_preserves_intent_then_recovery_only_restores(tmp_path, monkeypatch):
    ref, state, host, calls, _ = setup(tmp_path, monkeypatch); host.resume_failure = True
    with pytest.raises(ValueError, match='SOURCE_HOLD'): controller.cycle(ref, host_factory=lambda _: host)
    assert not list(state.glob('batch-*/complete.json'))
    host.resume_failure = False
    assert controller.cycle(ref, recover_only=True, host_factory=lambda _: host)['status'] == 'RECOVERY_COMPLETE'
    assert len(calls) == 1 and len(list(state.glob('batch-*/complete.json'))) == 1


def test_no_due_work_never_observes_or_pauses_collector(tmp_path, monkeypatch):
    ref, state, host, calls, due = setup(tmp_path, monkeypatch); due['selected_due'] = 0
    assert controller.cycle(ref, host_factory=lambda _: host)['status'] == 'NO_SELECTED_RESULT_DUE'
    assert not host.actions and not calls


def test_amended_start_uses_bound_plan_not_old_october_ten(tmp_path, monkeypatch):
    ref, state, host, calls, due = setup(tmp_path, monkeypatch)
    plan = {'schema_version': 'prospective_sectional_plan_v2', 'dates': ['2026-10-06', '2026-10-07'],
        'selection_windows': [{'local_date': '2026-10-06', 'freeze_at': '2026-10-06T22:00:00+11:00'}]}
    monkeypatch.setattr(controller.results, 'load_config', lambda ref: ({}, plan, {}))
    monkeypatch.setattr(controller.runtime, 'utc_now', lambda: datetime.fromisoformat('2026-10-06T21:59:00+11:00'))
    assert controller.cycle(ref, host_factory=lambda _: host)['status'] == 'BEFORE_DEVELOPMENT_HORIZON'
    assert not host.actions and not calls
    monkeypatch.setattr(controller.runtime, 'utc_now', lambda: datetime.fromisoformat('2026-10-07T06:00:00+11:00'))
    assert controller.cycle(ref, host_factory=lambda _: host)['status'] == 'RESULT_BATCH_COMPLETE_COLLECTOR_RUNNING'
    assert calls == ['request']


def test_partial_intent_cannot_restart_unowned_collector(tmp_path, monkeypatch):
    ref, state, host, calls, _ = setup(tmp_path, monkeypatch)
    (state/'batch-partial').mkdir(parents=True)
    state.chmod(0o700)
    with pytest.raises(ValueError, match='PARTIAL_INTENT'): controller.cycle(ref, recover_only=True, host_factory=lambda _: host)
    assert not host.actions and not calls


def test_native_resume_waits_for_post_start_health(monkeypatch):
    host = object.__new__(controller.Host)
    states = iter([{'ActiveState': 'inactive'}, {'ActiveState': 'active', 'MainPID': '123'},
                   {'ActiveState': 'active', 'MainPID': '123'}])
    monkeypatch.setattr(host, 'service', lambda: next(states))
    monkeypatch.setattr(host, 'quiet', lambda intent: None)
    seen = []
    monkeypatch.setattr(controller.subprocess, 'run', lambda argv, **kwargs: seen.append(argv))
    monkeypatch.setattr(controller.runtime, 'utc_now', lambda: NOW)
    monkeypatch.setattr(controller.time, 'sleep', lambda seconds: seen.append('wait'))
    health = iter([{'status': 'PAUSED', 'preparation': 'same', 'at': NOW.isoformat()},
                   {'status': 'WAITING_FOR_RACE', 'preparation': 'same', 'at': NOW.isoformat()}])
    monkeypatch.setattr(host, 'view', lambda: (next(health), {}, Path('/fixture')))
    assert host.resume({'before': {'preparation': 'same'}}) == 'RESUMED'
    assert 'wait' in seen


def test_native_start_that_fails_initialization_is_not_restored(monkeypatch):
    host = object.__new__(controller.Host)
    states = iter([{'ActiveState': 'inactive'}, {'ActiveState': 'failed', 'MainPID': '0'}])
    monkeypatch.setattr(host, 'service', lambda: next(states))
    monkeypatch.setattr(host, 'quiet', lambda intent: None)
    monkeypatch.setattr(controller.subprocess, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(controller.runtime, 'utc_now', lambda: NOW)
    with pytest.raises(ValueError, match='RESUME_NOT_RUNNING'):
        host.resume({'before': {'preparation': 'same'}})
