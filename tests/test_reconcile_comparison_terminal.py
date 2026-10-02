"""Explicit retained non-finish closure keeps ranked result evidence unchanged."""
import json
from pathlib import Path
import sqlite3

import pytest

from tests.test_reconcile_comparison_result_identity import case, put, ref, run, assert_quarantined


def nonfinish(case):
    authority=case['authority']
    body=Path(authority['body']['path'])
    body.write_bytes(body.read_bytes().replace(b'>3rd<',b'>DNF<'))
    authority['body']=ref(body)
    response_path=Path(authority['response']['path'])
    response=json.loads(response_path.read_bytes())
    response.update(sha256=authority['body']['sha256'],bytes=body.stat().st_size)
    authority['response']=put(response_path,response)
    authority['mode']='KNOWN_NON_FINISH_ONLY'
    paths=[case['root']/'official-results.sqlite3',Path(case['runtime']['source_state']),
           Path(case['runtime']['campaign_root'])/'ledger.json',
           *(Path(authority[k]['path']) for k in ('body','request','response','failed_report'))]
    return {str(p):p.read_bytes() for p in paths}


def test_explicit_nonfinish_mode_preserves_ranked_db_attempts_and_all_source_artifacts(case,monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session,'request',lambda *a,**k:pytest.fail('network forbidden'))
    before=nonfinish(case)
    result=run(case)
    assert result['result_state']=='RESULT_KNOWN_NON_FINISH'
    assert result['full_order_eligible'] is False and result['provider_requests']==0
    assert result['status']=='KNOWN_NON_FINISH_FROM_RETAINED_IDENTITY_RECONCILIATION'
    assert all(Path(p).read_bytes()==data for p,data in before.items())
    with sqlite3.connect(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone()==('CLOSED_NON_FINISH',1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==1
        assert [r[0] for r in db.execute('SELECT status FROM events ORDER BY id')]==['COLLECTOR_FAILURE','CLOSED_NON_FINISH']
    output=case['root']/'reconciliations'/case['authority']['reconciliation_id']
    after=json.loads((output/'after.json').read_bytes())
    assert after['full_order_eligible'] is False
    record=json.loads(Path(after['terminal_record']['path']).read_bytes())
    assert record['runner_results'][-1]['finish_position'] is None
    assert (output/'committed.json').is_file()
    assert not Path(case['runtime']['lock_path']).exists()


def test_default_full_order_mode_still_rejects_nonfinish(case):
    before=nonfinish(case)
    del case['authority']['mode']
    with pytest.raises(ValueError):run(case)
    assert_quarantined(case)
    assert all(Path(p).read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('when',['before_commit','after_commit'])
def test_nonfinish_interruption_receipt_truthfully_reports_queue_commit(case,monkeypatch,when):
    import scripts.reconcile_comparison_result_identity as module
    before=nonfinish(case)
    original=module.create_once
    seen=[]
    def interrupted(path,value):
        target='after.json' if when=='before_commit' else 'committed.json'
        if Path(path).name==target and not seen:
            seen.append(True)
            raise RuntimeError('invented interrupted receipt')
        return original(path,value)
    monkeypatch.setattr(module,'create_once',interrupted)
    with pytest.raises(RuntimeError):run(case)
    output=case['root']/'reconciliations'/case['authority']['reconciliation_id']
    failure=json.loads((output/'failure.json').read_bytes())
    assert failure['queue_commit_performed'] is (when=='after_commit')
    if when=='before_commit':
        assert_quarantined(case)
        assert not (output/'committed.json').exists()
    else:
        with sqlite3.connect(case['root']/'queue.sqlite3') as db:
            assert db.execute('SELECT state,attempts FROM jobs').fetchone()==('CLOSED_NON_FINISH',1)
        assert json.loads((output/'committed.json').read_bytes())['completion_status']=='PROCESSING_FAILED_AFTER_QUEUE_COMMIT'
    assert all(Path(p).read_bytes()==data for p,data in before.items())


@pytest.mark.parametrize('guard',['unknown_mode','sealed_original'])
def test_nonfinish_mode_is_explicit_and_cannot_reopen_sealed_roots(case,guard):
    before=nonfinish(case)
    if guard=='unknown_mode':case['authority']['mode']='INFER_NONFINISH'
    else:(case['root']/'closure').mkdir()
    with pytest.raises(ValueError):run(case)
    assert_quarantined(case)
    assert all(Path(p).read_bytes()==data for p,data in before.items())
