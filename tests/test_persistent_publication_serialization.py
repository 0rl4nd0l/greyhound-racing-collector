"""A due second lane must not mutate publisher directories before owner reap."""
import json
from pathlib import Path

from race_collection import synchronous_manual_capture as capture
from race_collection import persistent_collector as collector
from tests.test_persistent_collector import owner_case, retain_inventory, finish_native
from tests.test_persistent_native import backend


def test_full_and_odds_progress_without_consuming_waiting_lane(owner_case):
    c=owner_case;c.owner.activate();retain_inventory(c)
    c.owner.tick()
    assert list(c.owner.children)==['full']
    assert c.owner.state['next_due_at'].get('odds') is None
    dispatched=len(c.owner.state['dispatches'])
    c.owner.tick()
    assert len(c.owner.state['dispatches'])==dispatched
    assert c.owner.state['next_due_at'].get('odds') is None
    finish_native(c,c.children[0]);c.owner.tick()
    assert list(c.owner.children)==['odds']
    assert c.owner.state['completed_lanes']['full']==1
    assert c.owner.state['next_due_at']['odds']
    finish_native(c,c.children[1]);c.owner.tick()
    assert c.owner.state['completed_lanes']=={'full':1,'odds':1}


def test_due_odds_cannot_create_sibling_during_owned_full_publication(owner_case,monkeypatch):
    c=owner_case;c.owner.activate();retain_inventory(c)
    root=c.output/'collector/evidence';root.mkdir(parents=True,exist_ok=True)
    c.owner.plan['evidence_root']=str(root)
    target=root/'runtime/current.json';target.parent.mkdir(exist_ok=True)
    c.owner.launch('full',['/synthetic/full'],c.owner.plan['source_root'],{})
    original=collector.subprocess.Popen
    def native_wrapper(command,**kwargs):
        # Match native daemon output-directory creation before the child lane
        # lock is acquired. This is a sibling of the in-progress publication.
        (root/('native-'+kwargs['env']['INVOCATION_ID'])).mkdir()
        return original(command,**kwargs)
    monkeypatch.setattr(collector.subprocess,'Popen',native_wrapper)
    capture._atomic_replace_canonical(target,{'safe':True},evidence_root=root,
        _pre_replace=lambda:c.owner.tick())
    assert json.loads(target.read_bytes())=={'safe':True}
    assert len(c.children)==1 and not list(root.glob('native-*'))
    assert c.owner.state['next_due_at'].get('odds') is None


def test_waiting_lane_preserves_accounting_and_consumed_due_time(owner_case):
    c=owner_case;c.owner.activate();retain_inventory(c);c.owner.tick()
    paths=[Path(c.cfg['campaign_root'])/'ledger.json',Path(c.cfg['source_state'])]
    before={p:p.read_bytes() for p in paths}
    due=dict(c.owner.state['next_due_at']);dispatches=list(c.owner.state['dispatches'])
    for _ in range(3):c.owner.tick()
    assert c.owner.state['next_due_at']==due
    assert c.owner.state['dispatches']==dispatches
    assert all(p.read_bytes()==raw for p,raw in before.items())


def test_overdue_full_cannot_starve_previously_waiting_odds(owner_case):
    from datetime import timedelta
    c=owner_case;c.owner.activate();retain_inventory(c);c.owner.tick()
    c.clock[0]+=timedelta(seconds=901)
    retain_inventory(c)
    finish_native(c,c.children[0]);c.owner.tick()
    assert list(c.owner.children)==['odds']
    assert len(c.owner.state['dispatches'])==2
    finish_native(c,c.children[1]);c.owner.tick()
    assert list(c.owner.children)==['full']
    assert [r['lane'] for r in c.owner.state['dispatches']]==['full','odds','full']


def test_cutoff_does_not_launch_waiting_lane_or_consume_its_due_time(owner_case):
    from datetime import timedelta
    c=owner_case;c.owner.activate();retain_inventory(c);c.owner.tick()
    finish_native(c,c.children[0]);c.owner.poll()
    c.owner.scope.end=c.clock[0]+timedelta(seconds=600)
    due=dict(c.owner.state['next_due_at']);dispatches=len(c.owner.state['dispatches'])
    assert c.owner.tick()=='RUNNING'
    assert not c.owner.children
    assert len(c.owner.state['dispatches'])==dispatches
    assert c.owner.state['next_due_at']==due
    c.clock[0]=c.owner.scope.end
    assert c.owner.tick()=='DAY_ENDED'
