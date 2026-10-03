"""Owned atomic publication must finish before the parent reads its index."""
from datetime import datetime, timedelta
import json
from pathlib import Path

import pytest

from race_collection import persistent_collector as collector
from race_collection import synchronous_manual_capture as capture
from race_collection.live_freshness_contract import classify_refresh_outage, create_once
from tests.test_persistent_collector import owner_case, finish_native
from tests.test_persistent_native import backend
from tests.test_scheduled_refresh_outage import outage
from tests.test_empty_eligible_refresh import report_fixture, publish
from tests.race_collection.test_synchronous_manual_capture import _write_publication_evidence


def retained_index(case):
    # Native publication fixture has an intentionally fixed racing date; this
    # read-only observation clock does not reopen or acquire another lease.
    case.clock[0] = datetime.fromisoformat('2026-07-19T12:55:01+10:00')
    evidence = Path(case.owner.plan['evidence_root'])
    state = evidence/'shadow_autopilot_daemon_runtime/odds.json'
    initial = report_fixture(evidence, generated=case.clock[0]-timedelta(seconds=1))
    published = publish(evidence, state, initial, 'published')
    assert published['status'] == 'PUBLISHED', json.dumps(published)
    _write_publication_evidence(evidence, state, published)
    _, plan, run, _, _ = outage(case.output/'prior-outage')
    value = {**classify_refresh_outage(plan['evidence_root'], run),
        'observed_at':(case.clock[0]-timedelta(seconds=10)).isoformat(),
        'failed_cycle_count':1, 'source_evidence_root':plan['evidence_root'],
        'allocation_sha256':case.prepared['allocation_ref']['sha256']}
    target = case.output/'refresh-deferrals'/(run+'.json')
    create_once(target,value)
    case.owner.state['refresh_failures'] = [collector.reference(target)]
    case.owner.save()
    return evidence, capture.current_race_index_path(state)


@pytest.mark.parametrize('lane',['full','odds'])
def test_owned_atomic_publisher_is_drained_before_strict_parent_read(owner_case, monkeypatch, lane):
    c = owner_case; c.owner.activate()
    evidence,index = retained_index(c)
    c.owner.launch(lane, ['/synthetic/publisher'], c.owner.plan['source_root'], {})
    reads=[]
    original=capture._RetainedSafeFiles.validate
    def atomic_replace_while_reading(snapshot):
        # Same native atomic replacement primitive, with a complete valid
        # canonical publication. Its inode changes while the reader owns the
        # old descriptor; the unchanged strict verifier must reject that read.
        reads.append(True)
        capture._atomic_replace_canonical(index,json.loads(index.read_bytes()),evidence_root=evidence)
        original(snapshot)
    monkeypatch.setattr(capture._RetainedSafeFiles,'validate',atomic_replace_while_reading)
    dispatch=[]
    monkeypatch.setattr(c.owner.predictions,'tick',lambda **kwargs:dispatch.append(kwargs['allow_dispatch']))
    c.owner.poll()
    assert dispatch == [False] and not reads
    assert c.owner.state['forecast_admission_ready'] is False
    assert c.owner.state['forecast_admission_reason'] == 'CURRENT_INDEX_REFRESH_IN_PROGRESS'
    assert not (c.output/'HALT.json').exists()
    # Security semantics are unchanged: directly racing the strict reader is
    # still rejected, never reclassified as a valid publication.
    with pytest.raises(capture.CaptureOneRejected) as rejected:
        collector.refresh_outage_pending(c.output,c.owner.plan,c.owner.state['refresh_failures'],c.prepared['allocation_ref']['sha256'])
    assert rejected.value.code == 'CURRENT_INDEX_PATH_UNSAFE'
    assert rejected.value.details['reason'] == 'path_replaced'
    monkeypatch.setattr(capture._RetainedSafeFiles,'validate',original)
    finish_native(c,c.children[0]); c.owner.poll()
    assert dispatch == [False,True]
    assert c.owner.state['forecast_admission_ready'] is True
    assert len(c.owner.state['refresh_failures']) == 1
    # With no owned publisher, an unexpected swap still propagates to HALT.
    monkeypatch.setattr(capture._RetainedSafeFiles,'validate',atomic_replace_while_reading)
    with pytest.raises(capture.CaptureOneRejected) as quiet_rejection:
        c.owner.outage_pending()
    assert quiet_rejection.value.code == 'CURRENT_INDEX_PATH_UNSAFE'


@pytest.mark.parametrize('kind',['symlink','mutated_source','stale'])
def test_quiet_reader_still_rejects_unsafe_or_stale_publications(owner_case, kind):
    c=owner_case;c.owner.activate()
    evidence,index=retained_index(c)
    if kind=='symlink':
        target=index.with_name('unexpected.json');target.write_bytes(index.read_bytes())
        index.unlink();index.symlink_to(target)
        with pytest.raises(capture.CaptureOneRejected):
            capture.bounded_current_race_index(current_time=c.clock[0],timeout_seconds=1,
                index_path=index,evidence_root=evidence,max_age_seconds=270,return_verified_view=True)
        assert c.owner.outage_pending()
        assert c.owner.state['forecast_admission_ready'] is False
        return
    elif kind=='mutated_source':
        packet=json.loads(index.read_bytes())
        source=evidence/packet['source_refresh_report_path']
        source.write_bytes(b'{}')
    else:
        c.clock[0]+=timedelta(seconds=300)
        assert c.owner.outage_pending()
        assert not c.owner.state['forecast_admission_ready']
        return
    with pytest.raises(capture.CaptureOneRejected):
        c.owner.outage_pending()
    assert not (c.output/'HALT.json').exists()  # No owner run loop in this fixture.


def test_capture_rejection_diagnostics_are_scalar_and_allowlisted():
    error = capture.CaptureOneRejected('CURRENT_INDEX_PATH_UNSAFE', path='/retained/index.json',
        reason='path_replaced', payload={'private':'never retained'}, rows=['private'])
    assert collector._safe_capture_diagnostics(error) == {'capture_rejection':{
        'code':'CURRENT_INDEX_PATH_UNSAFE','path':'/retained/index.json','reason':'path_replaced'}}
    assert collector._safe_capture_diagnostics(RuntimeError('other')) == {}
    malformed = capture.CaptureOneRejected('CURRENT_INDEX_PATH_UNSAFE', path={'private':'hidden'},reason='x'*4097)
    assert collector._safe_capture_diagnostics(malformed) == {'capture_rejection':{'code':'CURRENT_INDEX_PATH_UNSAFE'}}


def test_owned_prediction_finishes_index_admission_before_due_publishers_launch(owner_case):
    from types import SimpleNamespace
    from tests.test_persistent_collector import retain_inventory
    c=owner_case;c.owner.activate();retain_inventory(c)
    child=SimpleNamespace(returncode=None)
    child.poll=lambda:child.returncode
    c.owner.predictions.child=child
    c.owner.predictions.log=SimpleNamespace(close=lambda:None)
    due=dict(c.owner.state['next_due_at'])
    assert c.owner.tick()=='RUNNING'
    assert not c.children
    assert c.owner.state['next_due_at']==due  # Waiting consumes no lane launch.
    child.returncode=0
    assert c.owner.tick()=='RUNNING'
    assert c.owner.predictions.child is None
    assert {row['lane'] for row in c.owner.state['dispatches']}=={'full','odds'}
