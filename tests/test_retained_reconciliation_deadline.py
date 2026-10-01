"""Fresh-clock boundaries on invented native retained-result reconciliation."""
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import sqlite3

import pytest
from tests.test_reconcile_comparison_result_identity import case, run, assert_quarantined


def clock_at(monkeypatch, module, clock):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz) if tz else clock[0].replace(tzinfo=None)
    monkeypatch.setattr(module, 'datetime', Clock)


@pytest.mark.parametrize('boundary', ['authority', 'result'])
@pytest.mark.parametrize('offset', [0, 1])
def test_stale_admission_time_never_allows_expired_protected_read(case, monkeypatch, boundary, offset):
    from scripts import reconcile_comparison_result_identity as module
    expires = datetime.fromisoformat((case['authority'] if boundary == 'authority' else case['runtime'])['expires_at'])
    if boundary == 'result':
        case['authority']['expires_at'] = (expires+timedelta(hours=1)).isoformat()
    clock_at(monkeypatch, module, [expires+timedelta(seconds=offset)])
    monkeypatch.setattr(module, 'evidence_rows', lambda *a, **k: pytest.fail('expired body decode'))
    from src.operator_ui import job_store
    monkeypatch.setattr(job_store, 'JobStore', lambda *a, **k: pytest.fail('expired job database read'))
    with pytest.raises(ValueError):
        run(case)
    assert_quarantined(case)


@pytest.mark.parametrize('seam', ['runtime', 'storage', 'body_read', 'html_title', 'identity', 'result_read', 'after_publication'])
def test_clock_crossing_stops_before_next_protected_stage(case, monkeypatch, seam):
    from scripts import reconcile_comparison_result_identity as module
    from scripts import ingest_results_for_date as ingest
    from src.predictor.comparison_result_runtime import ACTIVE
    expires = datetime.fromisoformat(case['authority']['expires_at'])
    clock = [case['now']]
    clock_at(monkeypatch, module, clock)
    if seam in {'runtime', 'storage'}:
        owner, name = module, 'load_runtime' if seam == 'runtime' else 'storage_check'
    elif seam == 'html_title':
        owner, name = ingest, 'title_from_html'
    elif seam == 'result_read':
        owner, name = module.ComparisonResultSource, 'read'
    elif seam == 'identity':
        from scripts import autonomous_official_result_capture as native
        owner, name = native, 'comparison_runner_identity_error'
    else:
        owner, name = module, 'checked' if seam == 'body_read' else 'create_once'
    original = getattr(owner, name)
    def cross(*args, **kwargs):
        value = original(*args, **kwargs)
        if (seam not in {'body_read', 'after_publication'}
                or seam == 'body_read' and args[0]['path'] == case['authority']['body']['path']
                or seam == 'after_publication' and Path(args[0]).name == 'after.json'):
            clock[0] = expires
        return value
    monkeypatch.setattr(owner, name, cross)
    if seam == 'body_read':
        monkeypatch.setattr(module, 'evidence_rows', lambda *a, **k: pytest.fail('body decoded after guarded read expired'))
    if seam == 'html_title':
        monkeypatch.setattr(ingest, 'rendered_text_from_html', lambda *a: pytest.fail('second parser ran after expiry'))
    if seam == 'identity':
        monkeypatch.setattr(ingest, 'result_validation_error', lambda *a: pytest.fail('position validator ran after expiry'))
    with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
        run(case)
    assert_quarantined(case)
    assert ACTIVE.get() is None
    output = case['root']/'reconciliations'/case['authority']['reconciliation_id']
    assert not (output/'committed.json').exists()
    if seam == 'after_publication':
        assert (output/'after.json').exists() and (output/'failure.json').exists()


def test_native_append_checks_fresh_deadline_before_database_commit(case, monkeypatch):
    from scripts import reconcile_comparison_result_identity as module
    from scripts import autonomous_official_result_capture as native
    from src.predictor.comparison_result_runtime import ACTIVE
    clock = [case['now']]
    clock_at(monkeypatch, module, clock)
    expires = datetime.fromisoformat(case['authority']['expires_at'])
    original = native.insert_official_result_evidence_rows
    def cross(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = expires
        return value
    monkeypatch.setattr(native, 'insert_official_result_evidence_rows', cross)
    with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
        run(case)
    assert_quarantined(case)
    assert ACTIVE.get() is None
    with sqlite3.connect(case['root']/'official-results.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM autonomous_official_result_evidence_races').fetchone()[0] == 0


def test_expiry_during_queue_commit_retains_true_commit_without_private_reread(case, monkeypatch):
    from scripts import reconcile_comparison_result_identity as module
    from src.predictor.comparison_result_runtime import ACTIVE
    clock = [case['now']]
    clock_at(monkeypatch, module, clock)
    expires = datetime.fromisoformat(case['authority']['expires_at'])
    class CrossingCommit(sqlite3.Connection):
        def commit(self):
            super().commit()
            clock[0] = expires
    original = sqlite3.connect
    def connect(database, *args, **kwargs):
        if 'queue.sqlite3' in str(database):
            kwargs['factory'] = CrossingCommit
        return original(database, *args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
        run(case)
    output = case['root']/'reconciliations'/case['authority']['reconciliation_id']
    assert json.loads((output/'failure.json').read_bytes())['queue_commit_performed'] is True
    assert json.loads((output/'committed.json').read_bytes())['completion_status'] == 'PROCESSING_FAILED_AFTER_QUEUE_COMMIT'
    with original(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('CLOSED', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == 2


    assert ACTIVE.get() is None and not Path(case['runtime']['lock_path']).exists()
    retained = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    from src.operator_ui import job_store
    monkeypatch.setattr(job_store, 'JobStore', lambda *a, **k: pytest.fail('expired restart read jobs'))
    monkeypatch.setattr(module, 'evidence_rows', lambda *a, **k: pytest.fail('expired restart decoded results'))
    with pytest.raises(ValueError, match='RESULT_DEADLINE_EXPIRED'):
        run(case)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()} == retained
    with original(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == ('CLOSED', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == 2


@pytest.mark.parametrize('interruption', ['before_commit', 'after_commit', 'after_commit_metadata_unavailable'])
def test_interrupted_queue_commit_is_observed_or_explicitly_unknown(case, monkeypatch, interruption):
    from scripts import reconcile_comparison_result_identity as module
    from src.predictor.comparison_result_runtime import ACTIVE
    clock = [case['now']]
    clock_at(monkeypatch, module, clock)
    expires = datetime.fromisoformat(case['authority']['expires_at'])
    class InterruptedCommit(sqlite3.Connection):
        def commit(self):
            if interruption != 'before_commit':
                super().commit()
            clock[0] = expires
            raise InterruptedError('invented_queue_commit_interruption')
    original = sqlite3.connect
    def connect(database, *args, **kwargs):
        if 'queue.sqlite3' in str(database):
            if 'mode=ro' in str(database) and interruption == 'after_commit_metadata_unavailable':
                raise sqlite3.OperationalError('invented_queue_metadata_unavailable')
            kwargs['factory'] = InterruptedCommit
        return original(database, *args, **kwargs)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    with pytest.raises(InterruptedError):
        run(case)
    output = case['root']/'reconciliations'/case['authority']['reconciliation_id']
    failure = json.loads((output/'failure.json').read_bytes())
    expected = {'before_commit': False, 'after_commit': True, 'after_commit_metadata_unavailable': None}[interruption]
    assert failure['queue_commit_performed'] is expected
    assert failure['queue_commit_attempted'] is True
    assert (output/'committed.json').exists() is (expected is True)
    with original(case['root']/'queue.sqlite3') as db:
        assert db.execute('SELECT state,attempts FROM jobs').fetchone() == (
            'QUARANTINED' if interruption == 'before_commit' else 'CLOSED', 1)
        assert db.execute('SELECT count(*) FROM requests').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM events').fetchone()[0] == (1 if interruption == 'before_commit' else 2)
    assert ACTIVE.get() is None and not Path(case['runtime']['lock_path']).exists()
