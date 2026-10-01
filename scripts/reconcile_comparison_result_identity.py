"""Explicit private reconciliation of one retained, identity-quarantined result.

No acquisition, retry, model change, label write, outcome export or evaluation.
The original request, quarantine and consumed counters remain intact.
"""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess

from race_collection.live_freshness_contract import create_once
from race_collection.synchronous_manual_capture import acquire_collector_lock_no_steal, release_owned_collector_lock
from src.predictor.comparison_result_runtime import ACTIVE, MAX_BODY, load_runtime, storage_check
from src.predictor.comparison_result_scope import authorize_job
from src.predictor.comparison_results import ComparisonResultSource
from src.predictor.future_comparison import stamp
from src.predictor.on_demand import canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
STATUS = 'CLOSED_FROM_RETAINED_IDENTITY_RECONCILIATION'


def require(condition):
    if not condition:
        raise ValueError('retained_identity_reconciliation_not_authorized_or_exact')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference, *, maximum=2*MAX_BODY):
    path = Path(reference['path'])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink())
    require(path.is_file() and path.stat().st_size <= maximum)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference['sha256'])
    return raw


class RetainedDeadline:
    """Fresh expiry checks for private retained reads, with no transport methods."""

    def __init__(self, authority, requested_now):
        self.deadline = stamp(authority['expires_at'])
        self.requested_now = requested_now

    def check_deadline(self):
        current = datetime.now(timezone.utc)
        if self.requested_now is not None:
            current = max(current, self.requested_now)
        if current >= self.deadline:
            raise ValueError('RESULT_DEADLINE_EXPIRED')
        return current

    def call(self, function, *args, **kwargs):
        self.check_deadline()
        value = function(*args, **kwargs)
        self.check_deadline()
        return value


def observed_queue_commit(root, original_job, original_event, output):
    """Classify an interrupted commit from queue metadata only, never outcomes."""
    try:
        with closing(sqlite3.connect((root/'queue.sqlite3').as_uri()+'?mode=ro', uri=True, timeout=2)) as db:
            db.row_factory = sqlite3.Row
            job = db.execute('SELECT race,job,jump,state,due,attempts FROM jobs WHERE job=?',
                             (original_job['job'],)).fetchone()
            event = db.execute('SELECT status,artifact FROM events WHERE race=? ORDER BY id DESC LIMIT 1',
                               (original_job['race'],)).fetchone()
            if job is None or event is None:
                return None
            if (dict(job) == {**original_job, 'state': 'CLOSED'}
                    and dict(event) == {'status': STATUS, 'artifact': str(output)}):
                return True
            if dict(job) == original_job and dict(event) == original_event:
                return False
    except Exception:
        pass
    return None


def evidence_rows(job, bundle, body, url, captured, now, *, deadline):
    """Keep values inside native machine validators; return no public outcomes."""
    from scripts import ingest_results_for_date as ingest
    from scripts.autonomous_official_result_capture import build_artifact_rows, comparison_runner_identity_error
    deadline.check_deadline()
    race = bundle.result['race']
    participants = [{'box_number': r['box'], 'dog_name': r['name']} for r in job.input.ordered_runners]
    candidate = ingest.RaceCandidate(job.input.race_id, race['venue'], race['race_number'], race['race_date'],
        None, job.input.jump_timestamp, None, Path('/unused-sealed-r3-inputs'), participants,
        'JUMPED_AWAITING_RESULT', participant_source='verified_r3_prediction',
        csv_participants=participants, canonical_thedogs_url=race['url'])
    text = deadline.call(body.decode, 'utf-8', errors='strict')
    title = deadline.call(ingest.title_from_html, text)
    rendered = deadline.call(ingest.rendered_text_from_html, text)
    require(not deadline.call(ingest.response_is_forbidden, 200, title, rendered))
    rows = deadline.call(ingest.parse_thedogs_result_html_runner_rows, text)
    expected = {r['box_number']: r['dog_name'] for r in participants}
    require(len({r['box_number'] for r in rows}) == len(rows))
    require({r['box_number']: r['dog_name'] for r in rows if r['box_number'] in expected} == expected)
    selected = deadline.call(ingest.TheDogsResultFetcher(None)._result_from_html, candidate, url, text)
    require(selected is not None and selected.source == 'thedogs_official' and selected.status == 'resulted')
    require(deadline.call(comparison_runner_identity_error, candidate, selected) is None)
    require(not selected.reserve_box_remappings and not selected.rejected_reserve_box_remappings)
    require(set(selected.positions_by_box) == set(expected))
    require(deadline.call(ingest.result_validation_error, candidate, selected) in (None, 'duplicate_first_place_results'))
    require(deadline.call(ingest.finish_positions_follow_competition_ranking, selected.positions_by_box.values()))
    item = {**race, 'start_datetime': job.input.jump_timestamp, 'source': selected.source,
        'status': selected.status, 'source_url': url, 'winner_box': selected.winner_box,
        'winner_name': expected[selected.winner_box], 'box_order': selected.raw_order,
        'participants': participants, 'participant_source': 'verified_r3_prediction',
        'positions': [{'box_number': box, 'dog_name': expected[box], 'finish_position': position}
                      for box, position in selected.positions_by_box.items()]}
    artifacts = deadline.call(build_artifact_rows, {'ingested': [item], 'failed': [], 'scope': {
        'candidate_source': 'authorized_retained_comparison_identity_reconciliation'}}, generated_at=captured)
    deadline.call(ComparisonResultSource._validate, job, bundle, artifacts['race_rows'], artifacts['runner_rows'],
                  deadline.check_deadline())
    return artifacts


def reconcile(authority_path, expected_sha, approval, *, now=None):
    os.umask(0o077)
    authority_ref = {'path': str(authority_path), 'sha256': expected_sha}
    authority = json.loads(checked(authority_ref))
    require(authority['schema_version'] == 'comparison_retained_identity_reconciliation_v1')
    require(authority['status'] == 'AUTHORIZED_RETAINED_IDENTITY_RECONCILIATION')
    require(approval and authority['authority_reference'] == approval)
    require(authority['network_requests_allowed'] is False and authority['outcomes_released'] is False
            and authority['preserve_attempts'] is True)
    deadline = RetainedDeadline(authority, now)
    now = deadline.check_deadline()
    require(stamp(authority['issued_at']) <= now)
    require(subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() == authority['source_commit'])
    binding = json.loads(checked(authority['binding']))
    plan, result_authority, cfg = load_runtime(binding, now=deadline.check_deadline())
    deadline.deadline = min(deadline.deadline, stamp(cfg['expires_at']))
    deadline.check_deadline()
    root = Path(cfg['state_root']); result_db = Path(result_authority['result_database'])
    require(re.fullmatch(r'[a-zA-Z0-9_-]{1,96}', authority['reconciliation_id']))
    output = root/'reconciliations'/authority['reconciliation_id']
    attempt = Path(authority['attempt_directory'])
    require(attempt.parent == root/'attempts' and attempt.resolve() == attempt)
    body_path = Path(authority['body']['path'])
    require(body_path.parent == attempt and body_path.name.startswith('response-') and body_path.suffix == '.body')
    require(Path(authority['request']['path']) == body_path.with_suffix('.request.json'))
    require(Path(authority['response']['path']) == body_path.with_suffix('.json'))
    require(Path(authority['failed_report']['path']) == attempt/'official_result_ingest_dry_run_report.json')
    require(not (attempt/'transport-status.json').exists())
    deadline.call(storage_check, root, cfg)
    with (root/'worker.lock').open('a') as worker, (Path(cfg['campaign_root'])/'owner.lock').open('a') as owner:
        fcntl.flock(worker, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owned = acquire_collector_lock_no_steal(Path(cfg['lock_path']), run_id=authority['reconciliation_id'],
            output_dir=output, phase='retained_result_identity_reconciliation')
        try:
            deadline.check_deadline()
            require(ACTIVE.get() is None)
            token = ACTIVE.set(deadline)
            try:
                return _locked(authority, authority_ref, binding, plan, cfg, result_db, root, output, attempt,
                               body_path, now, deadline)
            finally:
                ACTIVE.reset(token)
        finally:
            release_owned_collector_lock(owned)


def _locked(authority, authority_ref, binding, plan, cfg, result_db, root, output, attempt, body_path, now, deadline):
    from src.operator_ui.job_store import JobStore, Phase, canonical
    from src.operator_ui.r3_api import build_verified_bundle_reader
    from scripts.autonomous_official_result_capture import append_official_result_evidence_to_db
    store = deadline.call(JobStore, Path(cfg['job_store']), readonly=True)
    jobs = [j for j in deadline.call(store.recorded_jobs) if j.job_id == authority['job_id']]
    require(len(jobs) == 1)
    job = jobs[0]
    require(job.phase is Phase.PREDICTION_READY and job.input.race_id == authority['race_id'])
    bundles = Path(cfg['prediction_bundles'])
    deadline.call(authorize_job, job, binding, plan, now=deadline.check_deadline(), prediction_bundles=bundles)
    reader = deadline.call(build_verified_bundle_reader, bundles, store)
    bundle = deadline.call(reader, job)
    require(bundle is not None)
    # Admission and immutable four-forecast verification precede outcome decoding.
    deadline.check_deadline()
    with closing(sqlite3.connect((root/'queue.sqlite3').as_uri()+'?mode=rw', uri=True, timeout=2)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA synchronous=FULL')
        db.execute('BEGIN IMMEDIATE')
        require([r[0] for r in db.execute('SELECT binding FROM identity')] == [hashlib.sha256(canonical_bytes(binding)).hexdigest()])
        row = db.execute('SELECT race,job,jump,state,due,attempts FROM jobs WHERE job=?', (job.job_id,)).fetchone()
        require(row is not None and row['race'] == job.input.race_id and row['jump'] == job.input.jump_timestamp
                and row['state'] == 'QUARANTINED' and row['attempts'] > 0)
        requests = [dict(r) for r in db.execute('SELECT id,at,race,artifact FROM requests ORDER BY id')]
        race_requests = [r for r in requests if r['race'] == job.input.race_id]
        matching = [r for r in race_requests if r['artifact'] == str(body_path.with_suffix(''))]
        require(len(matching) == 1 and race_requests[-1] == matching[0] and len(race_requests) == row['attempts'])
        event = db.execute('SELECT status,artifact FROM events WHERE race=? ORDER BY id DESC LIMIT 1',
            (job.input.race_id,)).fetchone()
        require(event is not None and event['artifact'] == str(attempt)
                and event['status'] in {'COLLECTOR_FAILURE', 'QUARANTINED'})
        request = deadline.call(json.loads, deadline.call(checked, authority['request']))
        response = deadline.call(json.loads, deadline.call(checked, authority['response']))
        require(request['at'] == matching[0]['at'] == response['observed_at'])
        url = bundle.result['race']['url']+'?trial=false'
        require(request['url'] == response['final_url'] == url)
        require(response['status'] == 200 and response['host'] == 'www.thedogs.com.au'
                and set(response['retry_headers']) <= {'date'})
        require(response['content_type'].lower().startswith('text/html'))
        report = deadline.call(json.loads, deadline.call(checked, authority['failed_report']))
        require(report['candidate_count'] == 1 and not report['ingested'] and len(report['failed']) == 1)
        failure = report['failed'][0]
        require(failure['race_id'] == job.input.race_id and failure['errors'] == ['comparison_official_runner_identity_mismatch'])
        body = deadline.call(checked, authority['body'], maximum=MAX_BODY)
        require(response['sha256'] == authority['body']['sha256'] and response['bytes'] == len(body))
        artifacts = evidence_rows(job, bundle, body, url, stamp(request['at']), now, deadline=deadline)
        result_source = deadline.call(ComparisonResultSource, result_db)
        existing = deadline.call(result_source.read, job, bundle, now=deadline.check_deadline())
        require(existing['state'] == 'RESULT_AVAILABLE' or existing.get('reason') == 'OFFICIAL_RESULT_UNAVAILABLE')
        expected_evidence = {'race_rows': artifacts['race_rows'],
            'runner_rows': sorted(artifacts['runner_rows'], key=lambda r:r['box_number'])}
        evidence_sha = hashlib.sha256(canonical(expected_evidence)).hexdigest()
        if existing['state'] == 'RESULT_AVAILABLE':
            require(existing['evidence_sha256'] == evidence_sha)
        preserved_paths = [Path(cfg['campaign_root'])/'ledger.json', Path(cfg['source_state'])]
        preserved = {str(p): sha(p) for p in preserved_paths}
        before = {'at': deadline.check_deadline().isoformat(), 'authority': authority_ref, 'job': dict(row),
            'requests_sha256': hashlib.sha256(canonical_bytes(requests)).hexdigest(),
            'result_database_sha256': sha(result_db), 'preserved': preserved,
            'original_artifacts': {k:authority[k] for k in ('failed_report','request','response','body')},
            'provider_requests': 0, 'outcomes_released': False}
        deadline.check_deadline()
        output.mkdir(parents=True, exist_ok=False, mode=0o700)
        queue_committed = False
        queue_commit_attempted = False
        committed = None
        try:
            deadline.call(create_once, output/'before.json', before)
            appended = deadline.call(append_official_result_evidence_to_db, db_path=result_db, artifact_rows=artifacts,
                output_dir=output, execute=True, allow_dead_heats=True)
            require(appended['status'] in {'APPENDED_OFFICIAL_RESULT_EVIDENCE','NOOP_ALREADY_PRESENT'})
            verified_source = deadline.call(ComparisonResultSource, result_db)
            verified = deadline.call(verified_source.read, job, bundle, now=deadline.check_deadline())
            require(verified['state'] == 'RESULT_AVAILABLE' and verified['evidence_sha256'] == evidence_sha)
            require(all(sha(path) == expected for path,expected in preserved.items()))
            require([dict(r) for r in db.execute('SELECT id,at,race,artifact FROM requests ORDER BY id')] == requests)
            after = {'status': 'VERIFIED_RETAINED_EVIDENCE_PENDING_QUEUE_COMMIT', 'at': deadline.check_deadline().isoformat(), 'job_id': job.job_id, 'race_id': job.input.race_id,
                'authority_sha256': authority_ref['sha256'], 'before_sha256': sha(output/'before.json'),
                'evidence_sha256': evidence_sha, 'original_captured_at': request['at'],
                'result_database_sha256': sha(result_db), 'attempts_preserved': row['attempts'],
                'request_count_preserved': len(requests), 'provider_requests': 0, 'outcomes_released': False}
            deadline.call(create_once, output/'after.json', after)
            db.execute("UPDATE jobs SET state='CLOSED' WHERE job=? AND state='QUARANTINED'", (job.job_id,))
            require(db.execute('SELECT changes()').fetchone()[0] == 1)
            db.execute('INSERT INTO events(at,race,status,artifact) VALUES(?,?,?,?)',
                (deadline.check_deadline().isoformat(), job.input.race_id, STATUS, str(output)))
            committed = {'status': STATUS, 'after_sha256': sha(output/'after.json'),
                'attempts_preserved': row['attempts'], 'provider_requests': 0, 'outcomes_released': False}
            def commit_queue():
                nonlocal queue_committed, queue_commit_attempted
                queue_commit_attempted = True
                db.commit()
                queue_committed = True
            deadline.call(commit_queue)
            deadline.call(create_once, output/'committed.json', committed)
            return {'status': STATUS, **{k:after[k] for k in ('job_id','race_id','evidence_sha256','attempts_preserved','provider_requests','outcomes_released')}}
        except BaseException as exc:
            if not queue_committed:
                try:
                    db.rollback()
                except Exception:
                    queue_committed = None
                if queue_commit_attempted:
                    # SQLite may have committed and then raised (for example a
                    # delivered signal). A fresh read-only connection avoids a
                    # false rollback claim; unavailable metadata stays unknown.
                    queue_committed = observed_queue_commit(root, dict(row), dict(event), output)
            if queue_committed is True and not (output/'committed.json').exists():
                # An irreversible queue commit can cross expiry. Finish only
                # its structural receipt; do not reread results or imply rollback.
                create_once(output/'committed.json', {**committed,
                    'completion_status': 'PROCESSING_FAILED_AFTER_QUEUE_COMMIT'})
            create_once(output/'failure.json', {'at': datetime.now(timezone.utc).isoformat(),
                'failure_class': type(exc).__name__, 'queue_commit_performed': queue_committed,
                'queue_commit_attempted': queue_commit_attempted,
                'provider_requests': 0, 'outcomes_released': False})
            raise


def main():
    from scripts.check_freshness_service import deny_network
    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--approval-id', required=True)
    args = parser.parse_args()
    def stop(signum, frame):
        raise InterruptedError('reconciliation_terminated')
    previous = signal.signal(signal.SIGTERM, stop)
    try:
        try:
            value = reconcile(args.authority, args.authority_sha256, args.approval_id)
        except Exception as exc:
            value = {'status':'RECONCILIATION_REJECTED', 'failure_class':type(exc).__name__, 'outcomes_released':False}
            print(json.dumps(value)); return 2
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(json.dumps(value)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
