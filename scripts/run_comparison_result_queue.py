"""One persistent, bounded, outcome-blind result-retention cycle.

Scheduled by the generated systemd timer. Acquisition remains in the existing
exported official-result collector; this module only owns durable work/recovery.
"""
from datetime import datetime, timedelta, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

from race_collection.live_phase_checkpoint import atomic_json
from src.predictor.comparison_result_runtime import database, load_runtime, storage_check
from src.predictor.comparison_result_scope import authorize_job
from src.predictor.future_comparison import stamp
from src.predictor.on_demand import canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'CLOSED', 'QUARANTINED', 'ATTEMPTS_EXHAUSTED', 'DEADLINE_UNRESOLVED'}


def event(db, now, race, status, artifact=None):
    db.execute('INSERT INTO events(at,race,status,artifact) VALUES(?,?,?,?)',
               (now.isoformat(), race, status, str(artifact) if artifact else None))
    db.commit()


def next_due(jump, attempts, now):
    # First T+15m, second T+2h, next day, then weekly. Missed cycles do
    # one overdue attempt, never replay a burst of missed provider requests.
    nominal = jump + (timedelta(hours=2) if attempts == 1 else
                      timedelta(days=1) if attempts == 2 else timedelta(days=7 * (attempts - 2)))
    return max(nominal, now + timedelta(minutes=20)).astimezone(timezone.utc)


def status(root, db, now, code):
    counts = dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state').fetchall())
    value = {'schema_version': 'comparison_retention_health_v1', 'at': now.isoformat(),
             'status': code, 'counts': counts,
             'request_attempts': db.execute('SELECT count(*) FROM requests').fetchone()[0],
             'oldest_due': db.execute("SELECT min(due) FROM jobs WHERE state='PENDING'").fetchone()[0],
             'oldest_outstanding_jump': db.execute("SELECT min(jump) FROM jobs WHERE state NOT IN ('CLOSED')").fetchone()[0],
             'outcomes_released': False}
    atomic_json(root / 'health.json', value)
    return value


def _reconcile_closed_queue(db, now, final):
    db.execute("UPDATE jobs SET state='DEADLINE_UNRESOLVED' WHERE state IN ('PENDING','RUNNING')")
    if not db.execute("SELECT 1 FROM events WHERE status='CLOSURE_SEALED'").fetchone():
        event(db, now, None, 'CLOSURE_SEALED', final)
    db.commit()


def _closure(binding_path, root, db, now, cfg):
    # The previous child may have outlived a killed queue worker. Never seal
    # while its native owner/collector lock still permits result writes.
    with (Path(cfg['campaign_root']) / 'owner.lock').open('a') as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return status(root, db, now, 'CLOSURE_WRITER_BUSY')
        if Path(cfg['lock_path']).exists():
            return status(root, db, now, 'CLOSURE_WRITER_BUSY')
        return _seal_closure(binding_path, root, db, now)


def _seal_closure(binding_path, root, db, now):
    from scripts.seal_comparison_result_closure import seal
    final = root / 'closure'
    if final.exists():
        receipt = json.loads((final / 'closure.json').read_bytes())
        if hashlib.sha256((final / 'official-results.sqlite3').read_bytes()).hexdigest() != receipt['result_database_sha256']:
            raise ValueError('closure_identity_changed')
        _reconcile_closed_queue(db,now,final)
        return status(root, db, now, 'CLOSURE_SEALED')
    # Interrupted staging directories remain private evidence. A subsequent pass
    # seals to a new directory, then publishes exactly once under worker flock.
    staging = root / ('closure-staging-' + uuid.uuid4().hex)
    receipt = seal(binding_path, staging, now=now)
    receipt['result_database'] = str(final / 'official-results.sqlite3')
    atomic_json(staging / 'closure.json', receipt)
    (staging / 'closure.json').chmod(0o400)
    staging.rename(final)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    _reconcile_closed_queue(db,now,final)
    return status(root, db, now, 'CLOSURE_SEALED')


def cycle(binding_path):
    os.umask(0o077)
    now = datetime.now(timezone.utc)
    started=time.monotonic()
    binding = json.loads(binding_path.read_bytes())
    plan, authority, cfg = load_runtime(binding, now=now, allow_closure=True)
    root = Path(cfg['state_root']); root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / 'worker.lock').open('a') as mutex:
        fcntl.flock(mutex, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with database(root) as db:
            def deadline_closure():
                current = datetime.now(timezone.utc)
                if current >= stamp(cfg['expires_at']):
                    return _closure(binding_path, root, db, current, cfg)

            identity = hashlib.sha256(canonical_bytes(binding)).hexdigest()
            prior = db.execute('SELECT binding FROM identity').fetchall()
            if prior and [r[0] for r in prior] != [identity]:
                raise ValueError('queue_binding_changed')
            db.execute('INSERT OR IGNORE INTO identity VALUES(?)', (identity,)); db.commit()
            result_db = Path(authority['result_database'])
            if not result_db.exists():
                from scripts.autonomous_official_result_capture import ensure_official_result_evidence_tables
                with sqlite3.connect(result_db) as results:
                    ensure_official_result_evidence_tables(results)
            if (closed := deadline_closure()) is not None:
                return closed
            storage_check(root, cfg)
            if (closed := deadline_closure()) is not None: return closed
            from src.operator_ui.job_store import JobStore, Phase
            from src.operator_ui.r3_api import build_verified_bundle_reader
            from src.predictor.comparison_results import ComparisonResultSource
            if not Path(cfg['job_store']).exists():
                claims=Path(plan['programme_root'])/binding['plan_sha256']/'attempts'
                if db.execute('SELECT count(*) FROM jobs').fetchone()[0] or any(claims.glob('*/admission.json')):
                    return status(root,db,now,'JOB_STORE_MISSING')
                return status(root,db,now,'CYCLE_COMPLETE')
            if (closed := deadline_closure()) is not None: return closed
            store = JobStore(Path(cfg['job_store']), readonly=True)
            if (closed := deadline_closure()) is not None: return closed
            bundles = Path(cfg['prediction_bundles'])
            read_bundle = build_verified_bundle_reader(bundles, store)
            if (closed := deadline_closure()) is not None: return closed
            jobs = {j.job_id: j for j in store.recorded_jobs()}
            if (closed := deadline_closure()) is not None: return closed
            claims = Path(plan['programme_root']) / binding['plan_sha256'] / 'attempts'
            # Discover only immutable admitted jobs; no broad target-result scan.
            for admission_path in sorted(claims.glob('*/admission.json')):
                if (closed := deadline_closure()) is not None: return closed
                if time.monotonic()-started>300: return status(root,db,now,'CYCLE_BUDGET')
                admission = json.loads(admission_path.read_bytes())
                job = jobs.get(admission['job_id'])
                if job is None or job.phase is not Phase.PREDICTION_READY:
                    continue  # prediction failures retain their original denominator
                race = job.input.race_id
                if db.execute('SELECT 1 FROM jobs WHERE race=?', (race,)).fetchone(): continue
                now = datetime.now(timezone.utc)
                if (closed := deadline_closure()) is not None: return closed
                if now < stamp(job.input.jump_timestamp) + timedelta(minutes=15): continue
                try:
                    authorize_job(job, binding, plan, now=now, prediction_bundles=bundles)
                except (ValueError, KeyError, OSError):
                    if (closed := deadline_closure()) is not None: return closed
                    # Unique rejected claim, no repeated per-cycle event flood.
                    db.execute('INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,0)',
                               (race, job.job_id, job.input.jump_timestamp, 'QUARANTINED', None))
                    event(db, now, race, 'MEMBERSHIP_REJECTED')
                    continue
                if (closed := deadline_closure()) is not None: return closed
                if db.execute('SELECT count(*) FROM jobs').fetchone()[0] >= cfg['max_races']:
                    return status(root, db, now, 'RACE_BUDGET_EXHAUSTED')
                db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,0)',
                           (race, job.job_id, job.input.jump_timestamp, 'PENDING', now.isoformat()))
                event(db, now, race, 'DISCOVERED')
            # A killed process never returns its consumed attempt. Validate retained
            # DB evidence first; otherwise keep backoff and original private log.
            now = datetime.now(timezone.utc)
            if (closed := deadline_closure()) is not None: return closed
            db.execute("UPDATE jobs SET state='PENDING' WHERE state='RUNNING'"); db.commit()
            rows = db.execute("SELECT * FROM jobs WHERE state='PENDING' AND due<=? ORDER BY due,race LIMIT ?",
                              (now.isoformat(), cfg['races_per_cycle'])).fetchall()
            for row in rows:
                if (closed := deadline_closure()) is not None: return closed
                if time.monotonic()-started>300: return status(root,db,now,'CYCLE_BUDGET')
                job = jobs.get(row['job'])
                if job is None:
                    db.execute("UPDATE jobs SET state='QUARANTINED' WHERE race=?", (row['race'],))
                    event(db, now, row['race'], 'JOB_MISSING'); continue
                now = datetime.now(timezone.utc)
                if (closed := deadline_closure()) is not None: return closed
                authorize_job(job, binding, plan, now=now, prediction_bundles=bundles)
                if (closed := deadline_closure()) is not None: return closed
                bundle = read_bundle(job)
                now = datetime.now(timezone.utc)
                if (closed := deadline_closure()) is not None: return closed
                evidence = ComparisonResultSource(result_db).read(job, bundle, now=now)
                if (closed := deadline_closure()) is not None: return closed
                if evidence['state'] == 'RESULT_AVAILABLE':
                    db.execute("UPDATE jobs SET state='CLOSED' WHERE race=?", (row['race'],))
                    event(db, now, row['race'], 'CLOSED_FROM_RETAINED_EVIDENCE'); continue
                if evidence['state'] == 'RESULT_REJECTED':
                    db.execute("UPDATE jobs SET state='QUARANTINED' WHERE race=?", (row['race'],))
                    event(db, now, row['race'], 'RESULT_IDENTITY_REJECTED'); continue
                if row['attempts'] >= cfg['max_attempts_per_race']:
                    db.execute("UPDATE jobs SET state='ATTEMPTS_EXHAUSTED' WHERE race=?", (row['race'],))
                    event(db, now, row['race'], 'ATTEMPTS_EXHAUSTED'); continue
                # Check shared holds/lock before consuming a queue attempt. The
                # child independently acquires both locks and rechecks at request.
                from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
                # Classify one validated snapshot. An active operation is
                # contention, while restrictions and invalid state remain holds.
                try:
                    source = SportsbetAccess(cfg['source_state']).read()
                except SportsbetAccessBlocked:
                    return status(root,db,now,'SHARED_SOURCE_HOLD')
                if (source['access_basis']['status'] != 'permitted'
                        or source['phase'] != 'OPEN' or source['not_before'] > now.timestamp()):
                    return status(root,db,now,'SHARED_SOURCE_HOLD')
                from race_collection.freshness_campaign import Campaign
                with Campaign(cfg['campaign_root'], **{key: cfg[key] for key in
                        ('incident_authority', 'incident_slot') if key in cfg}).ledger() as ledger:
                    if ledger.get('source_holds'):
                        return status(root, db, now, 'SOURCE_HOLD')
                if source['active'] is not None:
                    return status(root, db, now, 'SOURCE_OPERATION_BUSY')
                with (Path(cfg['campaign_root'])/'owner.lock').open('a') as owner:
                    try: fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError: return status(root,db,now,'CAMPAIGN_OWNER_BUSY')
                if Path(cfg['lock_path']).exists():
                    return status(root, db, now, 'COLLECTOR_LOCK_BUSY')
                now = datetime.now(timezone.utc)
                if (closed := deadline_closure()) is not None: return closed
                output = root / 'attempts' / ('autonomous_official_result_capture_' + uuid.uuid4().hex)
                output.parent.mkdir(exist_ok=True)
                attempts = row['attempts'] + 1
                due = next_due(stamp(row['jump']), attempts, now)
                db.execute("UPDATE jobs SET state='RUNNING',due=? WHERE race=?",
                           (due.isoformat(), row['race']))
                event(db, now, row['race'], 'ATTEMPT_DISPATCHED', output)
                command = [sys.executable, '-B', '-m', 'scripts.autonomous_official_result_capture',
                    '--comparison-result-binding', str(binding_path), '--r3-job-store', cfg['job_store'],
                    '--r3-prediction-bundles', cfg['prediction_bundles'], '--db', str(result_db),
                    '--race-id', row['race'], '--output-dir', str(output), '--evidence-root', str(root/'attempts'), '--execute-db-ingest']
                # No outcome-bearing child output reaches journal or agent.
                with output.with_suffix('.log').open('xb') as log:
                    if (closed := deadline_closure()) is not None: return closed
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log)
                    try:
                        code = child.wait(timeout=40)
                    except BaseException:
                        child.terminate()
                        try: child.wait(timeout=5)
                        except subprocess.TimeoutExpired: child.kill(); child.wait()
                        event(db, now, row['race'], 'INTERRUPTED', output)
                        raise
                after = datetime.now(timezone.utc)
                charged = db.execute('SELECT attempts FROM jobs WHERE race=?',(row['race'],)).fetchone()[0]
                if charged == row['attempts']:
                    db.execute('UPDATE jobs SET due=? WHERE race=?',((after+timedelta(minutes=20)).isoformat(),row['race']))
                if (closed := deadline_closure()) is not None: return closed
                evidence = ComparisonResultSource(result_db).read(job, bundle, now=after)
                if (closed := deadline_closure()) is not None: return closed
                state = 'CLOSED' if evidence['state'] == 'RESULT_AVAILABLE' else (
                        'QUARANTINED' if evidence['state'] == 'RESULT_REJECTED' else 'PENDING')
                transport_status = output / 'transport-status.json'
                if transport_status.exists():
                    disposition = json.loads(transport_status.read_bytes())['status']
                    if disposition == 'SOURCE_HOLD':
                        db.execute("UPDATE jobs SET state='PENDING' WHERE race=?", (row['race'],))
                        event(db, after, row['race'], 'SOURCE_HOLD', output)
                        return status(root, db, after, 'SOURCE_HOLD')
                    state = 'QUARANTINED'
                report_path = output / 'official_result_ingest_dry_run_report.json'
                if state == 'PENDING' and report_path.exists():
                    report = json.loads(report_path.read_bytes())
                    errors = [e for failure in report.get('failed', []) for e in failure.get('errors', [])]
                    retryable = {'no_thedogs_positions_found', 'thedogs_http_404', 'thedogs_http_500',
                                 'thedogs_http_502', 'thedogs_http_503', 'thedogs_http_504'}
                    if errors and any(e not in retryable and not e.startswith('thedogs_http_error:') for e in errors):
                        state = 'QUARANTINED'
                db.execute('UPDATE jobs SET state=? WHERE race=?', (state, row['race']))
                event(db, after, row['race'], state if not code else 'COLLECTOR_FAILURE', output)
            if (closed := deadline_closure()) is not None: return closed
            return status(root, db, datetime.now(timezone.utc), 'CYCLE_COMPLETE')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--binding', type=Path, required=True)
    args = p.parse_args()
    def stop(signum, frame): raise InterruptedError('termination')
    signal.signal(signal.SIGTERM, stop)
    try:
        value = cycle(args.binding)
        print(json.dumps(value))
        return 0 if value['status'] in ('CYCLE_COMPLETE', 'CLOSURE_SEALED','COLLECTOR_LOCK_BUSY','CAMPAIGN_OWNER_BUSY','SOURCE_OPERATION_BUSY') else 2
    except Exception as exc:
        # Even exception messages can contain provider data. Only class is public.
        value={'status': 'RESULT_WORKER_FAILED', 'failure_class': type(exc).__name__,
               'at':datetime.now(timezone.utc).isoformat(),'outcomes_released':False}
        try:
            binding=json.loads(args.binding.read_bytes())
            _,_,cfg=load_runtime(binding,now=datetime.now(timezone.utc),allow_closure=True)
            root=Path(cfg['state_root'])
            if root.is_dir(): atomic_json(root/'health.json',value)
        except Exception: pass
        print(json.dumps(value))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
