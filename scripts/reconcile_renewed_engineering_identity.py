"""Append-only private successor closure for explicitly renewed engineering jobs.

The original expired authority, sealed queue, database and closure stay intact.
Only retained responses are processed; this module exposes no acquisition path.
"""
from contextlib import closing, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from race_collection.live_freshness_contract import create_once
from src.predictor.engineering_result_renewal import load_renewal
from src.predictor.future_comparison import stamp
from src.predictor.on_demand import canonical_bytes
from scripts.reconcile_comparison_result_identity import RetainedDeadline, checked, evidence_rows

ROOT = Path(__file__).resolve().parents[1]
STATUS = 'RENEWED_ENGINEERING_SUCCESSOR_CLOSURE_VERIFIED'


def require(condition):
    if not condition:
        raise ValueError('renewed_successor_not_authorized_or_exact')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scope(reference, approval, *, now):
    value = json.loads(checked(reference))
    require(value.get('schema_version') == 'renewed_engineering_identity_successor_v1'
            and value.get('status') == 'AUTHORIZED_RETAINED_IDENTITY_SUCCESSOR'
            and approval and value.get('authority_reference') == approval
            and value.get('network_requests_allowed') is False
            and type(value.get('max_additional_requests')) is int
            and value['max_additional_requests'] == 0
            and value.get('outcomes_released') is False
            and value.get('preserve_originals') is True)
    renewal, binding, plan, original, cfg = load_renewal(
        value['renewed_authority'], value['renewed_authority_approval'], now=now)
    require(stamp(renewal['issued_at']) <= stamp(value['issued_at']) <= now
            < stamp(value['expires_at']) <= stamp(renewal['expires_at']))
    members = {(r['job_id'], r['race_id']) for r in renewal['jobs']}
    rows = value['jobs']
    require(isinstance(rows, list) and 1 <= len(rows) <= 2
            and len({r['job_id'] for r in rows}) == len(rows)
            and len({r['race_id'] for r in rows}) == len(rows)
            and all((r['job_id'], r['race_id']) in members for r in rows))
    return value, renewal, binding, plan, original, cfg


def preserved_files(value, renewal, binding, plan, original, cfg):
    root = Path(cfg['state_root'])
    closure = sorted(p for p in (root/'closure').glob('*') if p.is_file())
    require(bool(closure))
    mandatory = {str(p) for p in [Path(value['renewed_authority']['path']),
        Path(renewal['original_binding']['path']), Path(binding['authority']), Path(binding['plan']),
        root/'queue.sqlite3', Path(original['result_database']),
        Path(cfg['campaign_root'])/'ledger.json', Path(cfg['source_state']), *closure]}
    refs = value['preserved_originals']
    require(isinstance(refs, list) and len({r['path'] for r in refs}) == len(refs)
            and {r['path'] for r in refs} == mandatory)
    for ref in refs:
        path = Path(ref['path'])
        require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
                and path.is_file() and digest(path) == ref['sha256'])
    require(not any(Path(str(path)+suffix).exists()
        for path in [root/'queue.sqlite3', Path(original['result_database'])]
        for suffix in ('-wal', '-journal')))
    output = Path(value['output_directory'])
    require(output.is_absolute() and output.resolve() == output and not output.is_symlink())
    for protected in [root, Path(cfg['prediction_bundles']), Path(plan['programme_root']),
                      Path(cfg['campaign_root']), ROOT]:
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output))
    require(not any(Path(ref['path']).is_relative_to(output) for ref in refs))
    return output, {ref['path']: ref['sha256'] for ref in refs}


def retained_artifacts(member, job, bundle, queue, root, deadline, bundles):
    from src.predictor.comparison_result_runtime import MAX_BODY
    queued = queue.execute('SELECT race,job,jump,state,attempts FROM jobs WHERE job=?', (job.job_id,)).fetchone()
    require(queued is not None and queued['race'] == job.input.race_id
            and queued['jump'] == job.input.jump_timestamp and queued['state'] == 'QUARANTINED'
            and queued['attempts'] > 0)
    requests = queue.execute('SELECT at,artifact FROM requests WHERE race=? ORDER BY id',
                             (job.input.race_id,)).fetchall()
    require(len(requests) == queued['attempts'])
    attempt = Path(member['attempt_directory']); body_path = Path(member['body']['path'])
    require(attempt.parent == root/'attempts' and attempt.resolve() == attempt
            and body_path.parent == attempt and body_path.name.startswith('response-')
            and body_path.suffix == '.body'
            and Path(member['request']['path']) == body_path.with_suffix('.request.json')
            and Path(member['response']['path']) == body_path.with_suffix('.json')
            and Path(member['failed_report']['path']) == attempt/'official_result_ingest_dry_run_report.json'
            and not (attempt/'transport-status.json').exists()
            and requests[-1]['artifact'] == str(body_path.with_suffix('')))
    request = deadline.call(json.loads, deadline.call(checked, member['request']))
    response = deadline.call(json.loads, deadline.call(checked, member['response']))
    url = bundle.result['race']['url']+'?trial=false'
    require(request['at'] == requests[-1]['at'] == response['observed_at']
            and request['url'] == response['final_url'] == url
            and response['status'] == 200 and response['host'] == 'www.thedogs.com.au'
            and set(response['retry_headers']) <= {'date'}
            and response['content_type'].lower().startswith('text/html'))
    failed = deadline.call(json.loads, deadline.call(checked, member['failed_report']))
    require(failed['candidate_count'] == 1 and not failed['ingested'] and len(failed['failed']) == 1
            and failed['failed'][0]['race_id'] == job.input.race_id
            and failed['failed'][0]['errors'] == ['comparison_official_runner_identity_mismatch'])
    body = deadline.call(checked, member['body'], maximum=MAX_BODY)
    require(response['sha256'] == member['body']['sha256'] and response['bytes'] == len(body))
    artifacts = deadline.call(evidence_rows, job, bundle, body, url, stamp(request['at']),
        deadline.check_deadline(), deadline=deadline, prediction_bundles=bundles)
    return artifacts, queued['attempts']


def reconcile(reference, approval, *, now=None):
    from src.predictor.comparison_result_runtime import ACTIVE
    from src.predictor.comparison_result_scope import authorize_job
    from src.predictor.comparison_results import ComparisonResultSource
    from src.operator_ui.job_store import JobStore, Phase
    from src.operator_ui.r3_api import build_verified_bundle_reader
    from scripts.autonomous_official_result_capture import (
        append_official_result_evidence_to_db, ensure_official_result_evidence_tables)
    os.umask(0o077)
    current = datetime.now(timezone.utc)
    if now is not None:
        current = max(current, now)
    value, renewal, binding, plan, original, cfg = load_scope(reference, approval, now=current)
    deadline = RetainedDeadline(value, now)
    deadline.deadline = min(deadline.deadline, stamp(renewal['expires_at']))
    require(subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip() == value['source_commit'])
    output, preserved = preserved_files(value, renewal, binding, plan, original, cfg)
    require(not output.exists() and ACTIVE.get() is None)
    deadline.check_deadline()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    create_once(output/'before.json', dict(authority=reference, preserved_originals=preserved,
        provider_requests=0, outcomes_released=False))
    token = ACTIVE.set(deadline)
    try:
        root = Path(cfg['state_root']); bundles = Path(cfg['prediction_bundles'])
        store = deadline.call(JobStore, Path(cfg['job_store']), readonly=True)
        reader = deadline.call(build_verified_bundle_reader, bundles, store)
        retained = []
        with closing(sqlite3.connect((root/'queue.sqlite3').as_uri()+'?mode=ro&immutable=1', uri=True)) as queue:
            queue.row_factory = sqlite3.Row
            queue.execute('PRAGMA query_only=ON')
            require([r[0] for r in queue.execute('SELECT binding FROM identity')]
                    == [hashlib.sha256(canonical_bytes(binding)).hexdigest()])
            for member in value['jobs']:
                job = deadline.call(store.get, member['job_id'])
                require(job is not None and job.phase is Phase.PREDICTION_READY
                        and job.input.race_id == member['race_id'])
                deadline.call(authorize_job, job, binding, plan, now=deadline.check_deadline(), prediction_bundles=bundles)
                bundle = deadline.call(reader, job); require(bundle is not None)
                artifacts, attempts = retained_artifacts(member, job, bundle, queue, root, deadline, bundles)
                retained.append((job, bundle, artifacts, attempts))
        new_db = output/'official-results.sqlite3'
        deadline.check_deadline()
        with sqlite3.connect(new_db) as db:
            ensure_official_result_evidence_tables(db)
        rows = []
        for job, bundle, artifacts, attempts in retained:
            appended = deadline.call(append_official_result_evidence_to_db, db_path=new_db,
                artifact_rows=artifacts, output_dir=output, execute=True, allow_dead_heats=True)
            require(appended['status'] == 'APPENDED_OFFICIAL_RESULT_EVIDENCE')
            verified = deadline.call(ComparisonResultSource(new_db).read, job, bundle, now=deadline.check_deadline())
            require(verified['state'] == 'RESULT_AVAILABLE')
            rows.append(dict(job_id=job.job_id, race_id=job.input.race_id,
                state='IDENTITY_VERIFIED_CLOSED_IN_SUCCESSOR', evidence_sha256=verified['evidence_sha256'],
                original_state='QUARANTINED', original_attempts_preserved=attempts))
        deadline.call(load_scope, reference, approval, now=deadline.check_deadline())
        require(all(digest(path) == expected for path, expected in preserved.items()))
        result = dict(status=STATUS, authority=reference, source_commit=value['source_commit'],
            result_database=dict(path=str(new_db), sha256=digest(new_db)), rows=rows,
            closed=len(rows), original_evidence_preserved=True, preserved_originals=preserved,
            provider_requests=0, outcomes_released=False, at=deadline.check_deadline().isoformat())
        deadline.call(create_once, output/'closure.json', result)
        return result
    except BaseException as exc:
        create_once(output/'failure.json', dict(status='SUCCESSOR_CLOSURE_NOT_PUBLISHED',
            failure_class=type(exc).__name__, provider_requests=0, outcomes_released=False))
        raise
    finally:
        ACTIVE.reset(token)


@contextmanager
def mute_private_output():
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with open(os.devnull, 'w') as sink:
        try:
            os.dup2(sink.fileno(), 1); os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(saved[0], 1); os.dup2(saved[1], 2)
            for fd in saved: os.close(fd)


def main():
    from scripts.check_freshness_service import deny_network
    deny_network()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--approval-id', required=True)
    args = parser.parse_args()
    try:
        with mute_private_output():
            result = reconcile(dict(path=str(args.authority), sha256=args.authority_sha256), args.approval_id)
    except BaseException as exc:
        print(json.dumps(dict(status='SUCCESSOR_RECONCILIATION_REJECTED', failure_class=type(exc).__name__, outcomes_released=False)))
        return 2
    print(json.dumps({key:result[key] for key in ('status','closed','provider_requests','outcomes_released')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
