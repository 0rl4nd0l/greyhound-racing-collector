"""Read-only private closure verification under a fresh retained-only authority.

Only scalar counts, identities, hashes and failure categories leave the machine
validators. Old queues, attempts, bodies and sealed closure are never changed.
"""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr, closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from src.predictor.engineering_result_renewal import load_renewal
from src.predictor.future_comparison import stamp, verify_comparison
from src.predictor.on_demand import canonical_bytes
from scripts.reconcile_comparison_result_identity import RetainedDeadline, checked, evidence_rows

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, category):
    if not condition:
        raise ValueError(category)


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


def audit(reference, approval, *, source_commit):
    from src.predictor.comparison_result_runtime import ACTIVE, MAX_BODY
    from src.predictor.comparison_result_scope import authorize_job
    from src.predictor.comparison_results import ComparisonResultSource
    from src.operator_ui.job_store import JobStore, Phase
    from src.operator_ui.r3_api import build_verified_bundle_reader
    actual = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    require(actual == source_commit, 'AUDIT_SOURCE_CHANGED')
    require(subprocess.run(['git','diff','--quiet','HEAD'], cwd=ROOT).returncode == 0, 'AUDIT_SOURCE_DIRTY')
    value, binding, plan, original, cfg = load_renewal(reference, approval, now=datetime.now(timezone.utc))
    deadline = RetainedDeadline(value, None)
    deadline.check_deadline()
    root = Path(cfg['state_root']); result_db = Path(original['result_database'])
    paths = [Path(reference['path']), Path(binding['authority']), Path(binding['plan']), Path(value['original_binding']['path']),
        root/'queue.sqlite3',result_db,Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state'])]
    paths.extend(p for p in (root/'closure').glob('*') if p.is_file())
    preserved = {str(path):digest(path) for path in paths}
    require(not any(Path(str(path)+suffix).exists() for path in [root/'queue.sqlite3',result_db]
                    for suffix in ('-wal','-journal')), 'ACTIVE_RESULT_DATABASE_SIDECAR')
    deadline.check_deadline()
    store = deadline.call(JobStore,Path(cfg['job_store']),readonly=True)
    reader = deadline.call(build_verified_bundle_reader,Path(cfg['prediction_bundles']),store)
    source = deadline.call(ComparisonResultSource,result_db)
    require(ACTIVE.get() is None, 'AUDIT_OTHER_PRIVATE_GUARD_ACTIVE')
    token = ACTIVE.set(deadline)
    rows=[]
    try:
        with closing(sqlite3.connect((root/'queue.sqlite3').as_uri()+'?mode=ro',uri=True)) as queue:
            queue.row_factory=sqlite3.Row;queue.execute('PRAGMA query_only=ON')
            require([r[0] for r in queue.execute('SELECT binding FROM identity')]
                == [hashlib.sha256(canonical_bytes(binding)).hexdigest()], 'ORIGINAL_QUEUE_BINDING_CHANGED')
            request_count=queue.execute('SELECT count(*) FROM requests').fetchone()[0]
            for member in value['jobs']:
                deadline.check_deadline()
                job=deadline.call(store.get,member['job_id'])
                require(job is not None and job.phase is Phase.PREDICTION_READY
                        and job.input.race_id==member['race_id'], 'FROZEN_JOB_IDENTITY_MISMATCH')
                deadline.call(authorize_job,job,binding,plan,now=deadline.check_deadline(),
                    prediction_bundles=Path(cfg['prediction_bundles']))
                verification=deadline.call(verify_comparison,Path(cfg['prediction_bundles']),
                    Path(member['admission']['path']),expected_plan_sha256=binding['plan_sha256'])
                require(verification.get('engineering_evidence') and not verification.get('future_race_evidence'),
                    'ENGINEERING_MEMBERSHIP_VERIFICATION_FAILED')
                queued=queue.execute('SELECT race,job,state,attempts,jump FROM jobs WHERE job=?',(job.job_id,)).fetchone()
                require(queued is not None and queued['race']==member['race_id']
                        and queued['jump']==job.input.jump_timestamp, 'QUEUE_IDENTITY_MISMATCH')
                bundle=deadline.call(reader,job);require(bundle is not None,'VERIFIED_BUNDLE_MISSING')
                row=dict(job_id=job.job_id,race_id=job.input.race_id,state=queued['state'],
                    consumed_attempts=queued['attempts'],four_forecasts_verified=True,
                    prior_boundary=member.get('prior_boundary'))
                if queued['state']=='CLOSED':
                    evidence=deadline.call(source.read,job,bundle,now=deadline.check_deadline())
                    require(evidence['state']=='RESULT_AVAILABLE','CLOSED_RESULT_IDENTITY_REJECTED')
                    row.update(result_state='IDENTITY_VERIFIED_CLOSED',evidence_sha256=evidence['evidence_sha256'])
                    del evidence
                elif queued['state']=='QUARANTINED':
                    requests=queue.execute('SELECT at,artifact FROM requests WHERE race=? ORDER BY id',
                        (job.input.race_id,)).fetchall()
                    require(requests and len(requests)==queued['attempts'],'RETAINED_ATTEMPTS_MISMATCH')
                    request=requests[-1]; stem=Path(request['artifact'])
                    require(stem.parent.parent==root/'attempts' and stem.parent.resolve()==stem.parent,
                        'RETAINED_RESPONSE_PATH_MISMATCH')
                    metadata_path=stem.with_suffix('.json');body_path=stem.with_suffix('.body')
                    metadata=deadline.call(json.loads,metadata_path.read_bytes())
                    request_meta=deadline.call(json.loads,stem.with_suffix('.request.json').read_bytes())
                    url=bundle.result['race']['url']+'?trial=false'
                    require(metadata['status']==200 and metadata['host']=='www.thedogs.com.au'
                        and metadata['final_url']==request_meta['url']==url
                        and metadata['observed_at']==request_meta['at']==request['at']
                        and set(metadata['retry_headers']) <= {'date'}
                        and metadata['content_type'].lower().startswith('text/html'), 'RETAINED_ENVELOPE_REJECTED')
                    body=deadline.call(checked,{'path':str(body_path),'sha256':metadata['sha256']},maximum=MAX_BODY)
                    require(len(body)==metadata['bytes'],'RETAINED_BODY_SIZE_MISMATCH')
                    try:
                        evidence_rows(job,bundle,body,url,stamp(request['at']),deadline.check_deadline(),deadline=deadline)
                        row['result_state']='RETAINED_EVIDENCE_VALIDATED_CLOSURE_NOT_COMMITTED'
                    except ValueError as exc:
                        deadline.check_deadline()
                        row['result_state']='QUARANTINED_NATIVE_VALIDATION_REJECTED'
                        row['failure_class']=type(exc).__name__
                        # Emit code locations, never exception text or private values.
                        tb=exc.__traceback__
                        while tb:
                            if (Path(tb.tb_frame.f_code.co_filename)==ROOT/'scripts/reconcile_comparison_result_identity.py'
                                    and tb.tb_frame.f_code.co_name=='evidence_rows'):
                                row['failed_native_line']=tb.tb_lineno
                            tb=tb.tb_next
                    del body
                    row['retained_body_sha256']=metadata['sha256']
                else:
                    row['result_state']='UNRESOLVED_ORIGINAL_STATE'
                deadline.check_deadline();rows.append(row)
    finally:
        ACTIVE.reset(token)
    require(all(digest(path)==expected for path,expected in preserved.items()),'ORIGINAL_EVIDENCE_CHANGED')
    deadline.check_deadline()
    counts={status:sum(r['result_state']==status for r in rows) for status in sorted({r['result_state'] for r in rows})}
    return dict(schema_version='renewed_engineering_private_closure_audit_v1',
        at=deadline.check_deadline().isoformat(),authority=reference,source_commit=source_commit,
        counts=counts,jobs=len(rows),original_request_count=request_count,provider_requests=0,
        human_outcome_access=False,outcomes_released=False,original_evidence_preserved=True,
        preserved_sha256=preserved,rows=rows)


def publish_result(reference, approval, output, result):
    # Hash-bound metadata validation after private work: never publish using an
    # edited replacement deadline or a changed membership/authority receipt.
    value, _, _, _, _ = load_renewal(reference, approval, now=datetime.now(timezone.utc))
    RetainedDeadline(value, None).check_deadline()
    from race_collection.live_freshness_contract import create_once
    create_once(output, result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--authority-sha256',required=True)
    parser.add_argument('--approval-id',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    os.umask(0o077)
    from scripts.check_freshness_service import deny_network
    deny_network()
    try:
        with mute_private_output():
            result=audit(dict(path=str(args.authority),sha256=args.authority_sha256),
                args.approval_id,source_commit=args.source_commit)
    except Exception as exc:
        print(json.dumps(dict(status='PRIVATE_AUDIT_FAILED',failure_class=type(exc).__name__,outcomes_released=False)))
        return 2
    # No publishing after expiry; the metadata receipt is create-only.
    publish_result(dict(path=str(args.authority),sha256=args.authority_sha256),
                   args.approval_id,args.output,result)
    print(json.dumps({key:result[key] for key in ('counts','jobs','original_request_count','provider_requests','outcomes_released')}))
    return 0

if __name__=='__main__':raise SystemExit(main())
