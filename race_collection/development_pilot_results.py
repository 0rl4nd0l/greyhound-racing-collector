"""Finite, private result checks for the explicitly approved development pilot.

No discovery, comparison-result database, fallback browser or retry loop. Queue
inspection reads identity/status metadata only. Network and target-label access
are reachable only after allocation admission and retained pre-result replay.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from race_collection.development_examples import (
    DevelopmentRejected, admit, canonical, checked, digest, join_result, private_output,
    put, race_key, read, stamp, verify_package,
)

ZONE = ZoneInfo('Australia/Melbourne')
MAX_BODY = 1024 * 1024


def _load(path, pin):
    cfg = read({'path': str(Path(path).absolute()), 'sha256': pin})
    synthetic = cfg.get('status') == 'SYNTHETIC_FIXTURE'
    if not synthetic and any(os.environ.get(name) for name in (
            'SYNTHETIC_DEVELOPMENT_CLOCK','FRESHNESS_FABRICATED_SOURCE','GREYHOUND_SHARED_SNAPSHOT_FIXTURE')):
        raise DevelopmentRejected('RESULT_FIXTURE_ENVIRONMENT_FORBIDDEN')
    if (cfg.get('schema_version') != 'development_pilot_runtime_v1'
            or cfg.get('status') not in {'AUTHORIZED', 'SYNTHETIC_FIXTURE'}
            or not cfg.get('authority_reference')
            or cfg.get('max_result_operations') != 72
            or cfg.get('max_result_transport_requests') != 720
            or cfg.get('max_result_checks_per_race') != 3):
        raise DevelopmentRejected('RESULT_WORKER_DISABLED')
    allocation = read(cfg['allocation'])
    if allocation.get('status') != ('SYNTHETIC_FIXTURE' if synthetic else 'AUTHORIZED'):
        raise DevelopmentRejected('RESULT_ALLOCATION_NOT_AUTHORIZED')
    if not synthetic and (allocation.get('allocation_id') != 'development-single-snapshot-20261003-v1'
            or stamp(cfg['result_closure_at']) != stamp('2026-10-25T12:00:00+11:00')):
        raise DevelopmentRejected('RESULT_ALLOCATION_OUTSIDE_PILOT')
    root = Path(cfg['state_root'])
    if not root.is_absolute() or root.resolve() != root or any(p.is_symlink() for p in (root, *root.parents)):
        raise DevelopmentRejected('RESULT_STATE_PATH_UNSAFE')
    if root.exists():
        info = root.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
            raise DevelopmentRejected('RESULT_STATE_NOT_PRIVATE')
    if not synthetic:
        source = Path(__file__).resolve().parents[1]
        if (Path(cfg['source_root']).resolve() != source
                or Path(cfg['python']).resolve() != Path(sys.executable).resolve()
                or Path(sys.prefix).resolve() != Path(cfg['python']).parent.parent.resolve()
                or subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip() != cfg['source_commit']
                or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()):
            raise DevelopmentRejected('RESULT_RUNTIME_SOURCE_IDENTITY_MISMATCH')
        profile = read(cfg['pilot_campaign_authority'])
        if (profile.get('schema_version') != 'collector_development_pilot_authority_v1'
                or profile.get('status') != 'AUTHORIZED_DEVELOPMENT_PILOT'
                or profile.get('allocation') != cfg['allocation']
                or profile.get('allocation_sha256') != cfg['allocation']['sha256']
                or profile.get('authority_reference') != cfg['authority_reference']
                or allocation.get('authority_reference') != cfg['authority_reference']
                or profile.get('state_root') != str(root)
                or profile.get('max_result_operations') != 72
                or profile.get('max_result_logical_requests') != 720
                or stamp(profile['result_closure_at']) != stamp(cfg['result_closure_at'])):
            raise DevelopmentRejected('RESULT_CAMPAIGN_AUTHORITY_MISMATCH')
    return cfg, allocation, root, synthetic


def _metadata(path, *, max_bytes=32768):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or path.stat().st_size > max_bytes:
        raise DevelopmentRejected('RESULT_METADATA_UNSAFE')
    return json.loads(path.read_bytes())


def _due_times(ready, cfg):
    jump = stamp(ready['jump_at'])
    local = jump.astimezone(ZONE)
    noon = (local + timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    times = [jump + timedelta(minutes=30), noon, stamp(cfg['result_closure_at'])-timedelta(minutes=30)]
    if not times[0] < times[1] < times[2]:
        raise DevelopmentRejected('RESULT_CHECK_SCHEDULE_INVALID')
    return times


def _inventory(cfg, root, now, *, race_key_fn=race_key, predecessor_configs=None):
    # A successor may preserve earlier nominations in the same cumulative
    # ledger. Callers authenticate these exact configurations before passing
    # them; old nominations retain their own result milestones and identity.
    allocations = {cfg['allocation']['sha256']: cfg, **(predecessor_configs or {})}
    ready_rows, due, completed = [], [], 0
    files = sorted((root/'ready').glob('*.json'))
    if len(files) > 24:
        raise DevelopmentRejected('RESULT_READY_BUDGET')
    for path in files:
        ready = _metadata(path)
        if (ready.get('schema_version') != 'development_pilot_capture_ready_v1'
                or ready.get('race_key') != race_key_fn(ready['race_id'])
                or path.name != digest(ready['race_id'].encode())+'.json'
                or ready.get('allocation_sha256') not in allocations):
            raise DevelopmentRejected('RESULT_NOMINATION_IDENTITY_INVALID')
        key = path.stem
        directory = root/'results'/key
        ready_rows.append(ready)
        if (directory/'complete.json').exists():
            completion = _metadata(directory/'complete.json')
            if completion['race_id'] != ready['race_id'] or completion['pre_result_sha256'] != ready['pre_result_sha256']:
                raise DevelopmentRejected('RESULT_COMPLETION_CHANGED')
            completed += 1
            continue
        times = _due_times(ready, allocations[ready['allocation_sha256']])
        if (directory/'official-result.json').exists() and now >= times[0]:
            due.append((times[0], ready, None))
            continue
        attempts = sorted(directory.glob('attempt-*/started.json'))
        if len(attempts) > 3:
            raise DevelopmentRejected('RESULT_ATTEMPT_BUDGET')
        used = {int(p.parent.name.split('-')[1]) for p in attempts}
        # Only the latest elapsed milestone is eligible: late startup never bursts
        # through obsolete checks. A started check is consumed even after a crash.
        elapsed = [n for n, at in enumerate(times) if at <= now]
        if elapsed:
            stage = max(elapsed)
            if stage not in used:
                due.append((times[stage], ready, stage))
            elif now >= times[-1]:
                due.append((times[-1], ready, None))
    attempts = list((root/'results').glob('*/attempt-*/started.json'))
    requests = list((root/'results').glob('*/attempt-*/request.json'))
    if len(attempts) > 72 or len(requests) > 720:
        raise DevelopmentRejected('RESULT_CUMULATIVE_BUDGET')
    return ready_rows, sorted(due, key=lambda r:(r[0], r[1]['race_id'])), completed, len(attempts), len(requests)


def inspect_queue(config_path, config_sha256, *, now=None):
    cfg, _, root, synthetic = _load(config_path, config_sha256)
    rows, due, completed, attempts, requests = _inventory(cfg, root, now or datetime.now(timezone.utc))
    return {'status': 'WORK_DUE' if due else 'NO_WORK', 'ready': len(rows), 'due': len(due),
            'completed': completed, 'operations_consumed': attempts,
            'transport_requests_consumed': requests,
            'rejected': sum(_metadata(p).get('status') in {'NOMINATION_REJECTED','RESULT_JOIN_REJECTED'}
                for p in (root/'results').glob('*/complete.json')),
            'transport_requests_unconfirmed': sum(not (p.parent/'transport-started.json').exists()
                for p in (root/'results').glob('*/attempt-*/request.json')), 'synthetic': synthetic}


def _admitted_packet(ready, cfg):
    access, allocation, member, _ = admit(ready['access_path'], ready['access_sha256'], ready['race_id'])
    if (access['allocation'] != cfg['allocation'] or member['race_key'] != ready['race_key']
            or stamp(member['jump_at']) != stamp(ready['jump_at'])
            or ready['prediction_entry'] != member['entry']
            or ready['job_id'] != member['entry']['job_id']):
        raise DevelopmentRejected('RESULT_READY_ACCESS_MISMATCH')
    output = Path(ready['example_dir'])
    if not output.is_absolute() or output.resolve() != output or not output.is_relative_to(Path(cfg['state_root'])/'sessions'):
        raise DevelopmentRejected('RESULT_EXAMPLE_OUTSIDE_PILOT')
    private_output(output)
    completion = read({'path': str(output/'completion.json'), 'sha256': ready['completion_sha256']})
    if (completion.get('status') != 'SEALED_PRE_RESULT' or completion['race_id'] != ready['race_id']
            or completion['access_sha256'] != ready['access_sha256']
            or completion['pre_result_sha256'] != ready['pre_result_sha256']
            or not stamp(completion['sealed_at']) < stamp(ready['jump_at'])):
        raise DevelopmentRejected('RESULT_READY_SEAL_MISMATCH')
    verify_package(ready['access_path'], ready['access_sha256'], ready['race_id'], output)
    packet = read({'path': str(output/'pre_result.json'), 'sha256': ready['pre_result_sha256']})
    if packet['race']['race_id'] != ready['race_id']:
        raise DevelopmentRejected('RESULT_PACKET_RACE_MISMATCH')
    if packet['synthetic'] != (cfg['status'] == 'SYNTHETIC_FIXTURE'):
        raise DevelopmentRejected('RESULT_SYNTHETIC_AUTHORITY_MISMATCH')
    return packet, allocation


def _shared_stop(cfg):
    from utils.sportsbet_access import SportsbetAccess
    if SportsbetAccess(cfg['source_state']).blocks_restoration():
        raise DevelopmentRejected('RESULT_SHARED_SOURCE_HOLD')
    ledger = _metadata(Path(cfg['campaign_root'])/'ledger.json', max_bytes=8*1024*1024)
    if ledger.get('source_holds'):
        raise DevelopmentRejected('RESULT_SHARED_SOURCE_HOLD')


def _study_priority(cfg, now):
    schedule = read(cfg['study_schedule'])
    for slot in schedule['slots']:
        start = stamp(slot)
        end = start + timedelta(minutes=schedule['session_minutes'])
        if start - timedelta(minutes=10) <= now <= end + timedelta(minutes=5):
            raise DevelopmentRejected('RESULT_STUDY_PRIORITY')
    # Capture and result acquisition share the same outcome-blind recovery,
    # first-session gate and result-backlog priority criteria.
    from race_collection.development_pilot import require_study_health
    try:
        require_study_health(cfg,now)
    except (OSError,ValueError,KeyError,TypeError):
        raise DevelopmentRejected('RESULT_STUDY_RECOVERY_OR_RETENTION_PRIORITY') from None



@contextmanager
def _ownership(cfg, output, now):
    from race_collection.synchronous_manual_capture import (
        acquire_collector_lock_no_steal, release_owned_collector_lock,
    )
    _study_priority(cfg, now)
    _shared_stop(cfg)
    owner = open(Path(cfg['campaign_root'])/'owner.lock', 'a+')
    lock = None
    try:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DevelopmentRejected('RESULT_CAMPAIGN_BUSY') from None
        lock = acquire_collector_lock_no_steal(Path(cfg['lock_path']),
            run_id='development-result-'+output.name, output_dir=output,
            phase='development_pilot_result', acquisition_policy='approved_development_pilot_v1')
        _shared_stop(cfg)
        yield
    finally:
        if lock is not None:
            release_owned_collector_lock(lock)
        owner.close()


def _official_result(packet, markup, observed, url):
    """Use observed official runner names; never fill them from expected names."""
    from scripts.ingest_results_for_date import RaceCandidate, TheDogsResultFetcher, parse_thedogs_result_html_runner_rows
    race = packet['race']
    observed_rows = parse_thedogs_result_html_runner_rows(markup)
    boxes = [r['box_number'] for r in observed_rows if r.get('box_number') is not None]
    if len(boxes) != len(set(boxes)):
        return None, 'AMBIGUOUS', 'OFFICIAL_RESULT_DUPLICATE_BOX'
    candidate = RaceCandidate(race_id=race['race_id'], venue=race['venue'], race_number=race['race_number'],
        race_date=race['race_date'], race_time=None, start_datetime=packet['times']['scheduled_jump_at'],
        sportsbet_url=None, csv_path=Path('/unused'),
        participants=[{'box_number':r['box_number'], 'dog_name':r['display_name']} for r in packet['runners']],
        lifecycle_status='resulted', participant_source='snapshot', canonical_thedogs_url=race['url'])
    parsed = TheDogsResultFetcher(None)._result_from_html(candidate, url, markup)
    if parsed is None:
        return None, 'MISSING', 'NO_OFFICIAL_RESULT_ROWS'
    if not parsed.positions_by_box:
        return None, 'AMBIGUOUS', 'OFFICIAL_RESULT_ROWS_UNRESOLVED'
    names = parsed.dog_names_by_box or {}
    positions = parsed.positions_by_box
    base = {**{k:race[k] for k in ('race_id','race_date','race_number','venue')},
            'source':'thedogs_official', 'source_url':url, 'captured_at':observed.isoformat()}
    rows = [{**base, 'box_number':box, 'dog_name':names.get(box,''), 'finish_position':position,
             'is_winner':position==1} for box,position in positions.items()]
    order = sorted(positions, key=positions.get)
    winner = order[0]
    race_row = {**base, 'status':'resulted', 'start_datetime':packet['times']['scheduled_jump_at'],
        'winner_box':winner, 'winner_name':names.get(winner,''), 'position_count':len(rows),
        'participant_count':len(packet['runners']), 'box_order':order}
    evidence = {'race_rows':[race_row], 'runner_rows':rows}
    from src.operator_ui.journal_results import OfficialResultSource
    job = SimpleNamespace(input=SimpleNamespace(race_id=race['race_id'],
        jump_timestamp=packet['times']['scheduled_jump_at'], ordered_runners=[
            {'box':r['box_number'], 'name':r['display_name'], 'source_native_runner_id':r.get('source_native_runner_id')}
            for r in packet['runners']]))
    bundle = SimpleNamespace(result={'race':race, 'generated_at':packet['times']['prediction_at']})
    try:
        OfficialResultSource._validate(job,bundle,[race_row],rows,observed)
    except ValueError:
        return None, 'AMBIGUOUS', 'OFFICIAL_RESULT_FIELD_UNVERIFIED'
    return evidence, 'OFFICIAL', 'EXACT_OFFICIAL_FIELD'


def _write_raw(path, raw):
    with os.fdopen(os.open(path, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600),'wb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


@contextmanager
def _transport_deadline(enabled):
    if not enabled:
        yield
        return
    def expired(signum, frame):
        raise TimeoutError('RESULT_TRANSPORT_DEADLINE')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 25)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _fetch(cfg, packet, attempt, session, campaign, now):
    from scripts.ingest_results_for_date import THEDOGS_PUBLIC_HEADERS, response_is_forbidden, title_from_html, rendered_text_from_html
    url = packet['race']['url'] + '?trial=false'
    if not url.startswith('https://www.thedogs.com.au/racing/') or '?' in packet['race']['url']:
        raise DevelopmentRejected('RESULT_SOURCE_URL_INVALID')
    # A durable reservation precedes the authoritative campaign charge. Missing
    # acknowledgement never hides consumed/uncertain allowance after a crash.
    put(attempt/'request.json', {'at':now.isoformat(), 'url':url, 'race_id':packet['race']['race_id'],
        'status':'TRANSPORT_RESERVED_BEFORE_CAMPAIGN_CHARGE'})
    if campaign is not None:
        _shared_stop(cfg)
        campaign.request(kind='results')
    put(attempt/'charge.json', {'status':'CAMPAIGN_CHARGED' if campaign is not None else 'SYNTHETIC_NO_CAMPAIGN_CHARGE'})
    put(attempt/'transport-started.json', {'at':now.isoformat(), 'status':'TRANSPORT_MAY_HAVE_STARTED'})
    response = session.get(url, headers={**THEDOGS_PUBLIC_HEADERS, 'Accept-Encoding':'identity'},
                           timeout=(5,20), allow_redirects=False, stream=True)
    try:
        headers = {k.lower():v for k,v in response.headers.items()}
        observed = datetime.now(timezone.utc) if campaign is not None else now
        header_denied = (response.status_code in (401,403,429)
            or any(h in headers for h in ('retry-after','ratelimit-reset','x-ratelimit-reset')))
        def stop():
            observation = {'source':'development_pilot_results', 'at':observed.isoformat(),
                'status_code':response.status_code, 'reason':'PROVIDER_DENIAL_OR_RETRY_GUIDANCE',
                'evidence':str(attempt/'request.json')}
            # Persist the globally shared STOP before potentially failing body IO.
            if campaign is not None: campaign.hold_source(observation)
            put(Path(cfg['state_root'])/'results'/'source-stop.json', observation)
        if header_denied:
            stop()
            put(attempt/'response.json', {'observed_at':observed.isoformat(),
                'status_code':response.status_code, 'url':response.url, 'body_read':False})
            return None, 'MISSING', 'SHARED_SOURCE_STOP', observed
        raw = response.raw.read(MAX_BODY+1, decode_content=False)
        _write_raw(attempt/'response.html', raw)
        observed = datetime.now(timezone.utc) if campaign is not None else now
        put(attempt/'response.json', {'observed_at':observed.isoformat(), 'status_code':response.status_code,
            'url':response.url, 'sha256':digest(raw), 'bytes':len(raw)})
        markup = raw.decode('utf-8', errors='replace')
        if response_is_forbidden(response.status_code,title_from_html(markup),rendered_text_from_html(markup)):
            stop()
            return None, 'MISSING', 'SHARED_SOURCE_STOP', observed
        if (len(raw)>MAX_BODY or response.url != url or response.status_code != 200
                or headers.get('content-encoding','identity') not in ('','identity')
                or 'text/html' not in headers.get('content-type','')):
            return None, 'MISSING', 'OFFICIAL_HTTP_ENVELOPE_UNAVAILABLE', observed
        evidence, disposition, reason = _official_result(packet,markup,observed,url)
        return evidence, disposition, reason, observed
    finally:
        response.close()


def _finish(cfg, allocation, ready, packet, directory, evidence, disposition, reason, now):
    synthetic = packet['synthetic']
    positions = {r['box_number']:r['finish_position'] for r in evidence['runner_rows']} if evidence else {}
    result = {'schema_version':'development_official_result_v1', 'synthetic':synthetic,
        'race_id':ready['race_id'], 'race_key':ready['race_key'], 'disposition':disposition,
        'observed_at':now.isoformat(), 'official_source':'synthetic_official_result_fixture' if synthetic else 'thedogs_official',
        'source_evidence_sha256':digest(canonical(evidence)) if evidence else None,
        'official_evidence':evidence,
        'finishers':{r['identity']:positions[r['box_number']] for r in packet['runners']} if evidence else {},
        'reason':reason}
    result_pin = put(directory/'official-result.json',result)
    authority = {'schema_version':'development_result_authority_v1',
        'status':'SYNTHETIC_FIXTURE' if synthetic else 'AUTHORIZED', 'allocation_id':allocation['allocation_id'],
        'authority_reference':cfg['authority_reference'], 'members':{ready['race_id']:{
            'race_key':ready['race_key'], 'pre_result_sha256':ready['pre_result_sha256'],
            'result':{'path':str(directory/'official-result.json'), 'sha256':result_pin}}}}
    authority_pin = put(directory/'authority.json',authority)
    joined = join_result(ready['access_path'],ready['access_sha256'],ready['race_id'],ready['example_dir'],
                         directory/'authority.json',authority_pin)
    put(directory/'complete.json', {**joined, 'pre_result_sha256':ready['pre_result_sha256']})
    return joined


def _reject(directory, ready, status, error):
    reason = str(error) if isinstance(error,DevelopmentRejected) else type(error).__name__
    value = {'status':status, 'race_id':ready['race_id'], 'pre_result_sha256':ready['pre_result_sha256'],
             'reason':reason, 'trainable':False, 'target_join_completed':False}
    put(directory/'complete.json',value)
    return value


def _finish_or_reject(cfg, allocation, ready, packet, directory, evidence, disposition, reason, now):
    try:
        return _finish(cfg,allocation,ready,packet,directory,evidence,disposition,reason,now)
    except (ValueError,KeyError) as error:
        return _reject(directory,ready,'RESULT_JOIN_REJECTED',error)


def run_cycle(config_path, config_sha256, *, now=None, session=None, campaign=None):
    cfg, allocation, root, synthetic = _load(config_path, config_sha256)
    actual_now = datetime.now(timezone.utc)
    if not synthetic and (now is not None or session is not None or campaign is not None):
        raise DevelopmentRejected('RESULT_TEST_OVERRIDES_FORBIDDEN')
    now = now or actual_now
    status = inspect_queue(config_path, config_sha256, now=now)
    if status['status'] == 'NO_WORK':
        return status
    results = private_output(root/'results')
    put(results/'config-identity.json', {'sha256':config_sha256, 'allocation_sha256':cfg['allocation']['sha256']})
    with open(results/'worker.lock','a+') as worker:
        os.chmod(results/'worker.lock',0o600)
        try: fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return {**status,'status':'WORKER_BUSY'}
        _, due, _, operations, transports = _inventory(cfg,root,now)
        if not due: return inspect_queue(config_path, config_sha256, now=now)
        _, ready, stage = due[0]
        if not synthetic and stage is not None and now < stamp(cfg['result_closure_at']):
            _study_priority(cfg,now)
            _shared_stop(cfg)
        directory = private_output(results/digest(ready['race_id'].encode()))
        try:
            packet, allocation = _admitted_packet(ready,cfg)
        except (ValueError,KeyError) as error:
            return _reject(directory,ready,'NOMINATION_REJECTED',error)
        # A published result always recovers through the same restricted join.
        if (directory/'official-result.json').exists():
            existing = _metadata(directory/'official-result.json')
            return _finish_or_reject(cfg,allocation,ready,packet,directory,existing['official_evidence'],
                           existing['disposition'],existing['reason'],stamp(existing['observed_at']))
        if stage is None or now >= stamp(cfg['result_closure_at']) or operations >= 72 or transports >= 720:
            return _finish_or_reject(cfg,allocation,ready,packet,directory,None,'MISSING','CHECKS_EXHAUSTED_OR_INTERRUPTED',now)
        if (results/'source-stop.json').exists():
            if now >= stamp(cfg['result_closure_at']):
                return _finish_or_reject(cfg,allocation,ready,packet,directory,None,'MISSING','SHARED_SOURCE_STOP_AT_CLOSURE',now)
            return {**status,'status':'SOURCE_STOP'}
        stored = sum(p.stat().st_size for p in root.parent.rglob('*') if p.is_file() and not p.is_symlink())
        if (shutil.disk_usage(results).free < 2*2**30
                or stored + 2*MAX_BODY > allocation.get('max_storage_bytes',10*2**30)):
            raise DevelopmentRejected('RESULT_DISK_PRESSURE')
        attempt = private_output(directory/f'attempt-{stage}')
        def execute():
            # Charge before transport; never delete a started marker on failure.
            put(attempt/'started.json', {'race_id':ready['race_id'], 'at':now.isoformat(),
                'stage':stage, 'pre_result_sha256':ready['pre_result_sha256'],
                'missed_prior_stages':[n for n in range(stage) if not (directory/f'attempt-{n}'/'started.json').exists()]})
            try:
                with _transport_deadline(not synthetic):
                    evidence, disposition, reason, observed = _fetch(cfg,packet,attempt,session,campaign,now)
            except Exception as error:
                evidence, disposition, reason, observed = None,'MISSING','RESULT_CHECK_INTERRUPTED_'+type(error).__name__,now
            put(attempt/'finished.json', {'disposition':disposition, 'reason':reason, 'at':observed.isoformat()})
            if disposition == 'OFFICIAL' or stage == 2:
                return _finish_or_reject(cfg,allocation,ready,packet,directory,evidence,disposition,reason,observed)
            return {'status':'CHECK_RETAINED', 'synthetic':synthetic, 'disposition':disposition,
                    'operations_consumed':operations+1, 'next_check_at':_due_times(ready,cfg)[stage+1].isoformat()}
        if synthetic:
            if session is None: raise DevelopmentRejected('SYNTHETIC_TRANSPORT_REQUIRED')
            return execute()
        from race_collection.freshness_campaign import Campaign
        import requests
        now = datetime.now(timezone.utc)
        if stamp(cfg['result_closure_at'])-now < timedelta(seconds=45):
            return _finish_or_reject(cfg,allocation,ready,packet,directory,None,'MISSING','FINAL_TRANSPORT_WINDOW_EXPIRED',now)
        with _ownership(cfg,attempt,now):
            campaign = Campaign(Path(cfg['campaign_root']), development_authority=cfg['pilot_campaign_authority'])
            with requests.Session() as session:
                return execute()
