import json
from types import SimpleNamespace
from datetime import datetime,timezone
import pytest
from race_collection.persistent_collector import classify_native_dispatch, restore_due_times, record_native_terminal


def evidence(tmp_path, *, action='LIVE_COLLECTION_COMPLETE', status='READY', code=0):
    identity='a'*32
    runtime=tmp_path/'shadow_autopilot_daemon_runtime'
    for child in ('service-terminals','service-lifecycles'):(runtime/child).mkdir(parents=True)
    terminal={'invocation_id':identity,'allocation_sha256':'b'*64,'status':status,'runtime_action':action,'final_verdict':'DAEMON_READY'}
    lifecycle={'invocation_id':identity,'children_reaped':True,'status':'COMPLETE','returncode':code}
    (runtime/'service-terminals'/f'{identity}.json').write_text(json.dumps(terminal))
    (runtime/'service-lifecycles'/f'{identity}.json').write_text(json.dumps(lifecycle))
    return {'invocation_id':identity,'returncode':code},runtime


def test_success_requires_native_terminal_and_reaped_children(tmp_path):
    record,runtime=evidence(tmp_path)
    assert classify_native_dispatch(tmp_path,record,'b'*64)['disposition']=='COMPLETED'
    (runtime/'service-terminals'/('a'*32+'.json')).unlink()
    with pytest.raises(ValueError,match='terminal_missing'):classify_native_dispatch(tmp_path,record,'b'*64)


@pytest.mark.parametrize('code',[0,2])
def test_lock_handoff_is_deferred_not_success_or_failure(tmp_path,code):
    record,_=evidence(tmp_path,action='DEFERRED_FULL_LOCK_HANDOFF',status='SKIPPED_FULL_DAEMON_LOCK_HANDOFF',code=code)
    assert classify_native_dispatch(tmp_path,record,'b'*64)['disposition']=='DEFERRED'


def test_native_failure_cannot_hide_behind_zero_exit(tmp_path):
    record,_=evidence(tmp_path,action='LIVE_TIMING_BUDGET_EXCEEDED',status='FAILED')
    with pytest.raises(ValueError,match='terminal_failure'):classify_native_dispatch(tmp_path,record,'b'*64)


def test_restart_retains_due_times_from_consumed_dispatches():
    state={'dispatches':[{'lane':'full','started_at':'2026-10-03T04:00:00+00:00'},
                         {'lane':'odds','started_at':'2026-10-03T04:01:00+00:00'}]}
    assert restore_due_times(state)=={'full':'2026-10-03T04:15:00+00:00','odds':'2026-10-03T04:02:00+00:00'}
    assert restore_due_times(state)==state['next_due_at']
    state['next_due_at']['full']='2026-10-03T04:01:00+00:00'
    with pytest.raises(ValueError,match='regressed'):restore_due_times(state)


def test_receipt_requires_authenticated_persistent_scope(tmp_path,monkeypatch):
    from race_collection.live_freshness_contract import FreshnessContract
    monkeypatch.setenv('GREYHOUND_SERVICE_INVOCATION','a'*32)
    monkeypatch.setenv('GREYHOUND_PERSISTENT_ALLOCATION_SHA256','b'*64)
    args=SimpleNamespace(live_freshness_contract=tmp_path/'scope',evidence_root=tmp_path)
    monkeypatch.setattr(FreshnessContract,'load',lambda path:SimpleNamespace(value={'persistent_allocation':{'sha256':'c'*64}}))
    with pytest.raises(ValueError,match='authority_mismatch'):record_native_terminal(args,{})
    assert not list(tmp_path.rglob('*.json'))
