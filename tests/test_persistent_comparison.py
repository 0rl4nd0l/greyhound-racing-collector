from datetime import datetime,timedelta
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from src.predictor.on_demand import canonical_bytes


def run(root,action):
    p=subprocess.run([sys.executable,'-B','-m','tests.fixtures.persistent_comparison_case','--root',str(root),'--action',action],capture_output=True,text=True,timeout=60)
    assert p.returncode==0,p.stderr[-3000:]
    return json.loads(p.stdout.splitlines()[-1])


def scenario(root,response,*,days=0,hours=0):
    path=root/'scenario.json';value=json.loads(path.read_bytes())
    value['response']=response;value['now']=(datetime.fromisoformat(value['now'])+timedelta(days=days,hours=hours)).isoformat()
    path.write_bytes(canonical_bytes(value))


def requests(root):
    path=root/'synthetic-requests.jsonl'
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_exported_prediction_pending_restart_and_exact_result_closure(tmp_path):
    assert run(tmp_path,'setup')['forecasts']==4
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'predictions').rglob('*') if p.is_file()}
    # Both shared ownership boundaries defer without losing/charging the job.
    lock=tmp_path/'collector.lock';lock.write_text('occupied')
    assert run(tmp_path,'queue')['status']=='COLLECTOR_LOCK_BUSY'
    lock.unlink()
    with (tmp_path/'campaign/owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert run(tmp_path,'queue')['status']=='CAMPAIGN_OWNER_BUSY'
    assert requests(tmp_path)==0
    result=run(tmp_path,'queue');assert result['counts']=={'PENDING':1},result
    assert requests(tmp_path)==1
    assert run(tmp_path,'queue')['counts']=={'PENDING':1}
    assert requests(tmp_path)==1
    scenario(tmp_path,'available',hours=3)
    result=run(tmp_path,'queue');assert result['counts']=={'CLOSED':1},result
    assert requests(tmp_path)==2
    assert run(tmp_path,'queue')['counts']=={'CLOSED':1}
    assert requests(tmp_path)==2
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'predictions').rglob('*') if p.is_file()}
    scenario(tmp_path,'available',days=16)
    assert run(tmp_path,'queue')['status']=='CLOSURE_SEALED'
    receipt=(tmp_path/'private/closure/closure.json').read_bytes()
    assert run(tmp_path,'queue')['status']=='CLOSURE_SEALED'
    assert (tmp_path/'private/closure/closure.json').read_bytes()==receipt
    assert requests(tmp_path)==2


def test_denial_persists_across_restarts_without_retry(tmp_path):
    run(tmp_path,'setup');scenario(tmp_path,'denial')
    result=run(tmp_path,'queue');assert result['status']=='SOURCE_HOLD',result
    assert requests(tmp_path)==1
    scenario(tmp_path,'available',days=1)
    assert run(tmp_path,'queue')['status']=='SOURCE_HOLD'
    assert requests(tmp_path)==1
    assert '600' in (tmp_path/'campaign/ledger.json').read_text()


def test_changed_field_quarantines_without_retry(tmp_path):
    run(tmp_path,'setup');scenario(tmp_path,'changed')
    result=run(tmp_path,'queue');assert result['counts']=={'QUARANTINED':1},result
    scenario(tmp_path,'available',days=1)
    assert run(tmp_path,'queue')['counts']=={'QUARANTINED':1}
    assert requests(tmp_path)==1


def test_deadheat_is_retained_not_reopened(tmp_path):
    run(tmp_path,'setup');scenario(tmp_path,'deadheat')
    result=run(tmp_path,'queue');assert result['counts']=={'CLOSED':1},result
    assert run(tmp_path,'queue')['counts']=={'CLOSED':1}
    assert requests(tmp_path)==1


def test_interrupted_attempt_and_partial_closure_are_preserved(tmp_path):
    run(tmp_path,'setup');run(tmp_path,'queue')
    # A crash after evidence commit but before queue update is reconciled, never
    # refetched. Also preserve an incomplete closure staging directory.
    scenario(tmp_path,'available',hours=3);run(tmp_path,'queue')
    with sqlite3.connect(tmp_path/'private/queue.sqlite3') as db:
        db.execute("UPDATE jobs SET state='RUNNING',due='2000-01-01T00:00:00+00:00'")
    assert run(tmp_path,'queue')['counts']=={'CLOSED':1}
    assert requests(tmp_path)==2
    staging=tmp_path/'private/closure-staging-interrupted';staging.mkdir();(staging/'partial').write_bytes(b'preserved')
    scenario(tmp_path,'available',days=16)
    assert run(tmp_path,'queue')['status']=='CLOSURE_SEALED'
    assert (staging/'partial').read_bytes()==b'preserved'
    assert requests(tmp_path)==2


def test_shared_source_stop_blocks_queue_and_direct_collector(tmp_path):
    run(tmp_path,'setup')
    from utils.sportsbet_access import SportsbetAccess
    SportsbetAccess(tmp_path/'source.json').retain_denial(403,reason='SYNTHETIC')
    assert run(tmp_path,'queue')['status']=='SHARED_SOURCE_HOLD'
    assert requests(tmp_path)==0
    # Direct exported collector cannot bypass the same transport/queue guard.
    rid=json.loads((tmp_path/'scenario.json').read_bytes())['race']
    output=tmp_path/'private/attempts/autonomous_official_result_capture_direct'
    output.parent.mkdir(exist_ok=True)
    proc=subprocess.run([sys.executable,'-B','-m','tests.fixtures.persistent_comparison_case',
        '--root',str(tmp_path),'--action','collector','--',
        '--comparison-result-binding',str(tmp_path/'binding.json'),
        '--r3-job-store',str(tmp_path/'predictions-jobs.db'),'--r3-prediction-bundles',str(tmp_path/'predictions'),
        '--db',str(tmp_path/'private/official-results.sqlite3'),'--race-id',rid,
        '--output-dir',str(output),'--evidence-root',str(output.parent),'--execute-db-ingest'],capture_output=True,text=True)
    assert requests(tmp_path)==0
    assert proc.returncode!=0 or 'machine_result_retention' in proc.stdout


def test_private_storage_checks_and_finite_backoff(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from src.predictor.comparison_result_runtime import storage_check
    from scripts.run_comparison_result_queue import next_due
    import pytest
    monkeypatch.setattr('shutil.disk_usage',lambda _:SimpleNamespace(free=1))
    with pytest.raises(ValueError,match='DISK_PRESSURE'):storage_check(tmp_path,{'max_storage_bytes':32*2**30})
    jump=datetime.fromisoformat('2026-10-05T13:00:00+11:00')
    assert next_due(jump,2,jump)==jump+timedelta(days=1)
    assert next_due(jump,3,jump)==jump+timedelta(days=7)
    assert next_due(jump,3,jump+timedelta(days=30))==jump+timedelta(days=30,minutes=20)


def test_reboot_after_closure_publish_reconciles_queue_once(tmp_path):
    run(tmp_path,'setup');run(tmp_path,'queue')
    scenario(tmp_path,'pending',days=16)
    assert run(tmp_path,'queue')['status']=='CLOSURE_SEALED'
    with sqlite3.connect(tmp_path/'private/queue.sqlite3') as db:
        db.execute("UPDATE jobs SET state='RUNNING'")
        db.execute("DELETE FROM events WHERE status='CLOSURE_SEALED'")
    assert run(tmp_path,'queue')['counts']=={'DEADLINE_UNRESOLVED':1}
    run(tmp_path,'queue')
    with sqlite3.connect(tmp_path/'private/queue.sqlite3') as db:
        assert db.execute("SELECT count(*) FROM events WHERE status='CLOSURE_SEALED'").fetchone()[0]==1
    assert requests(tmp_path)==1


def test_missing_jobstore_bootstrap_vs_lost_membership(tmp_path):
    run(tmp_path,'setup')
    jobstore=tmp_path/'predictions-jobs.db';jobstore.rename(tmp_path/'saved-jobs.db')
    assert run(tmp_path,'queue')['status']=='JOB_STORE_MISSING'
    # Move only synthetic admission to model the pre-first-job state.
    claim=next((tmp_path/'comparison-programme').glob('*/attempts/*/admission.json'))
    claim.rename(claim.with_suffix('.saved'))
    assert run(tmp_path,'queue')['status']=='CYCLE_COMPLETE'
    assert requests(tmp_path)==0
