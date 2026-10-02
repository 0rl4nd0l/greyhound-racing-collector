"""Native private queue state and opaque closure for known non-finish results."""
import hashlib
import json
from datetime import timezone
from pathlib import Path

import pytest

from tests.test_reconcile_comparison_result_identity import case, put
from scripts import run_comparison_result_queue as queue
from scripts.reconcile_comparison_result_identity import RetainedDeadline
from src.predictor.comparison_result_runtime import database


def prepare_nonfinish(case, *, marker=b'DNF'):
    body = Path(case['authority']['body']['path'])
    body.write_bytes(body.read_bytes().replace(b'>3rd<', b'>'+marker+b'<'))
    response_path = Path(case['authority']['response']['path'])
    response = json.loads(response_path.read_bytes())
    response.update(sha256=hashlib.sha256(body.read_bytes()).hexdigest(), bytes=body.stat().st_size)
    put(response_path, response)
    with database(case['root']) as db:
        db.execute("UPDATE jobs SET state='PENDING',due=?", (case['now'].astimezone(timezone.utc).isoformat(),))


def close(case):
    with database(case['root']) as db:
        return queue.close_retained_nonfinish(db, case['root'], case['job'], case['bundle'],
                                              case['runtime'], case['now'])


def test_known_nonfinish_closes_separately_without_request_or_attempt_changes(case):
    prepare_nonfinish(case)
    assert close(case) is True
    with database(case['root']) as db:
        assert tuple(db.execute('SELECT state,attempts FROM jobs').fetchone()) == ('CLOSED_NON_FINISH', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert [r[0] for r in db.execute('SELECT status FROM events ORDER BY id')] == ['COLLECTOR_FAILURE', 'CLOSED_NON_FINISH']
        health = queue.status(case['root'], db, case['now'], 'CYCLE_COMPLETE')
        assert health['counts'] == {'CLOSED_NON_FINISH': 1}
        assert health['oldest_outstanding_jump'] is None
        assert db.execute("SELECT count(*) FROM jobs WHERE state='PENDING'").fetchone()[0] == 0
    assert close(case) is False  # no retry/reacquisition or duplicate event
    import sqlite3
    with sqlite3.connect(case['root']/'official-results.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM autonomous_official_result_evidence_races').fetchone()[0] == 0


@pytest.mark.parametrize('failure', ['unknown', 'mixed', 'source_hold', 'wrong_request_time', 'outside_attempts', 'uncharged'])
def test_bad_or_incomplete_evidence_never_closes(case, failure):
    prepare_nonfinish(case, marker=b'UNKNOWN' if failure == 'unknown' else b'DNF')
    body = Path(case['authority']['body']['path'])
    if failure == 'mixed':
        body.write_bytes(body.read_bytes().replace(b'>2nd<', b'>UNKNOWN<'))
        response_path = Path(case['authority']['response']['path'])
        response = json.loads(response_path.read_bytes())
        response.update(sha256=hashlib.sha256(body.read_bytes()).hexdigest(), bytes=body.stat().st_size)
        put(response_path, response)
    elif failure == 'source_hold':
        put(body.parent/'transport-status.json', {'status':'SOURCE_HOLD'})
    elif failure == 'wrong_request_time':
        path = Path(case['authority']['request']['path'])
        request = json.loads(path.read_bytes()); request['at'] = '2020-01-01T00:00:00+00:00'; put(path, request)
    elif failure == 'outside_attempts':
        with database(case['root']) as db: db.execute('UPDATE requests SET artifact=?',(str(case['root']/'outside'),))
    elif failure == 'uncharged':
        with database(case['root']) as db: db.execute('UPDATE jobs SET attempts=2')
    assert close(case) is False
    with database(case['root']) as db:
        assert db.execute('SELECT state FROM jobs').fetchone()[0] == 'PENDING'
    assert not (case['root']/'terminal-results').exists()


def test_opaque_terminal_closure_copies_exact_proof_and_does_not_change_eligibility(case):
    prepare_nonfinish(case); assert close(case)
    with database(case['root']) as db:
        staging = case['root']/'invented-closure'; staging.mkdir()
        receipts = queue.seal_known_nonfinish(case['root'], db, staging)
    assert len(receipts) == 1 and receipts[0]['full_order_eligible'] is False
    target = staging/'terminal-results'/case['job'].job_id
    for name, expected in receipts[0]['files_sha256'].items():
        assert hashlib.sha256((target/name).read_bytes()).hexdigest() == expected
    assert (target/'body').read_bytes() == Path(case['authority']['body']['path']).read_bytes()
    assert json.loads((target/'record').read_bytes())['full_order_eligible'] is False


def test_tampered_terminal_record_stops_closure(case):
    prepare_nonfinish(case); assert close(case)
    path = case['root']/'terminal-results'/(case['job'].job_id+'.json')
    path.chmod(0o600); path.write_bytes(path.read_bytes()+b' ')
    with database(case['root']) as db:
        with pytest.raises(ValueError): queue.seal_known_nonfinish(case['root'], db, case['root']/'bad-closure')


def test_queue_commit_follows_durable_record_and_caller_transaction(case):
    prepare_nonfinish(case); assert close(case)
    path = case['root']/'terminal-results'/(case['job'].job_id+'.json')
    record = json.loads(path.read_bytes())
    deadline = RetainedDeadline({'expires_at':case['runtime']['expires_at']},None)
    with database(case['root']) as db:
        db.execute("UPDATE jobs SET state='QUARANTINED'")
        db.execute("DELETE FROM events WHERE status='CLOSED_NON_FINISH'"); db.commit()
        db.execute('BEGIN IMMEDIATE')
        queue.record_known_nonfinish(db,case['root'],case['job'],record,case['now'],deadline=deadline)
        assert db.in_transaction
        db.rollback()
        assert db.execute('SELECT state FROM jobs').fetchone()[0] == 'QUARANTINED'
        assert path.exists()  # durable uncommitted evidence is preserved, never a false closure
        queue.record_known_nonfinish(db,case['root'],case['job'],record,case['now'],deadline=deadline)
        db.commit()
        assert db.execute('SELECT state FROM jobs').fetchone()[0] == 'CLOSED_NON_FINISH'


def cycle_clock(case, monkeypatch):
    from scripts import reconcile_comparison_result_identity as reconcile
    monkeypatch.setattr(queue, 'datetime', reconcile.datetime)
    return case['root'].parent/'binding.json'


def test_native_cycle_child_failure_closes_known_nonfinish_and_restart_does_not_launch(case, monkeypatch):
    prepare_nonfinish(case)
    attempt = Path(case['authority']['body']['path']).parent
    retained = {p.name: p.read_bytes() for p in attempt.iterdir() if p.is_file()}
    binding = cycle_clock(case, monkeypatch)
    with database(case['root']) as db:
        db.execute('DELETE FROM requests'); db.execute('UPDATE jobs SET attempts=0')
    launched = []
    class Child:
        def __init__(self, command, **kwargs):
            launched.append(command)
            output = Path(command[command.index('--output-dir')+1]); output.mkdir()
            for name, raw in retained.items(): (output/name).write_bytes(raw)
            with database(case['root']) as db:
                db.execute('UPDATE jobs SET attempts=attempts+1')
                db.execute('INSERT INTO requests(at,race,artifact) VALUES(?,?,?)',
                    (case['authority']['issued_at'], case['job'].input.race_id, str(output/'response-invented')))
        def wait(self, **kwargs): return 2
    monkeypatch.setattr(queue.subprocess, 'Popen', Child)
    first = queue.cycle(binding)
    assert first['counts'] == {'CLOSED_NON_FINISH': 1} and len(launched) == 1
    second = queue.cycle(binding)
    assert second['counts'] == first['counts'] and len(launched) == 1
    with database(case['root']) as db:
        assert db.execute('SELECT status FROM events ORDER BY id DESC LIMIT 1').fetchone()[0] == 'CLOSED_NON_FINISH'
        assert tuple(db.execute('SELECT state,attempts FROM jobs').fetchone()) == ('CLOSED_NON_FINISH', 1)


def test_known_terminal_response_never_masks_existing_full_order_rejection(case, monkeypatch):
    prepare_nonfinish(case)
    binding = cycle_clock(case, monkeypatch)
    from src.predictor.comparison_results import ComparisonResultSource
    monkeypatch.setattr(ComparisonResultSource, 'read', lambda *a,**k: {
        'state':'RESULT_REJECTED','reason':'OFFICIAL_RESULT_RUNNER_IDENTITY_MISMATCH'})
    def forbidden(*a,**k): raise AssertionError('known result must not hide rejected standard evidence')
    monkeypatch.setattr(queue, 'close_retained_nonfinish', forbidden)
    assert queue.cycle(binding)['counts'] == {'QUARANTINED':1}
    assert not (case['root']/'terminal-results').exists()
