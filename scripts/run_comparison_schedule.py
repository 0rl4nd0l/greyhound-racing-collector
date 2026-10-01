"""Persistent admission wrapper around the existing finite collector supervisor.

No second collector or timer implementation. Fixed slots are consumed once;
missed slots are recorded, not made up. An interrupted predecessor must restore
before any new slot. Source STOP is never automatically reopened.
"""
from datetime import datetime, timedelta, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from race_collection.live_freshness_contract import create_once, digest
from race_collection.live_phase_checkpoint import atomic_json
from src.predictor.future_comparison import checked, load_plan, stamp

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    cfg = json.loads(path.read_bytes())
    incident = cfg.get('status') == 'AUTHORIZED_INCIDENT_SCHEDULE'
    if cfg.get('status') not in {'AUTHORIZED_PERSISTENT_SCHEDULE', 'AUTHORIZED_INCIDENT_SCHEDULE'} or not cfg.get('authority_reference'):
        raise ValueError('schedule_not_authorized')
    plan, _ = load_plan(Path(cfg['comparison_plan']), cfg['comparison_plan_sha256'])
    if plan['status'] != ('AUTHORIZED_ENGINEERING' if incident else 'AUTHORIZED'):
        raise ValueError('study_not_authorized')
    slots = [stamp(s) for s in cfg['slots']]
    if (not 1 <= len(slots) <= 80 or slots != sorted(set(slots))
            or any(not stamp(plan['starts_at']) <= s < stamp(plan['ends_at']) for s in slots)
            or (not incident and any((b-a).total_seconds() < 23*3600 for a,b in zip(slots,slots[1:])))
            or cfg['session_minutes'] != 90 or cfg['grace_seconds'] != 300
            or cfg['source_operations_per_session'] != 192
            or cfg['max_source_operations'] != len(slots)*192):
        raise ValueError('invalid_fixed_schedule')
    if incident:
        from race_collection.incident_comparison import validate_incident_plan
        authority = validate_incident_plan(plan)
        slot = next(row for row in authority['slots'] if row['id'] == plan['incident_slot'])
        if (any(cfg.get(key) != plan[key] for key in ('incident_authority', 'incident_slot'))
                or cfg['slots'] != [slot['starts_at']]
                or cfg['authority_reference'] != authority['authority_reference']
                or Path(cfg['state_root']) != Path(authority['state_root']) / 'windows' / slot['id']
                or cfg['prediction_root'] != authority['prediction_root']):
            raise ValueError('incident_schedule_mismatch')
        result_binding = json.loads(Path(cfg['result_binding']).read_bytes())
        if (result_binding['plan'] != cfg['comparison_plan']
                or result_binding['plan_sha256'] != cfg['comparison_plan_sha256']):
            raise ValueError('incident_schedule_result_binding_mismatch')
    current = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(['git','status','--porcelain','--untracked-files=no'], cwd=ROOT, text=True)
    if current != cfg['source_commit'] or dirty:
        raise ValueError('schedule_source_changed')
    from race_collection.persistent_storage import check_mount
    check_mount(cfg['storage_mount'], Path(cfg['state_root']))
    check_mount(cfg['storage_mount'], Path(cfg['prediction_root']))
    if any(not Path(cfg[k]).is_absolute() for k in ('python','history_database','lock_path','reconciliation_roots','installed_dir','campaign_root','source_state','comparison_plan','result_binding')):
        raise ValueError('schedule_paths_must_be_absolute')
    return cfg, plan


def programme_source_usage(value, baseline_count, programme_start):
    """Keep global consumption; exclude only authenticated separate allocations."""
    count = len(value.get('operations', []))
    used = count - baseline_count
    allocations = value.get('diagnostic_authorizations', [])
    for index, allocation in enumerate(allocations):
        if 'engineering_authority' not in allocation:
            continue
        start = allocation['operation_start']
        end = allocations[index+1]['operation_start'] if index+1 < len(allocations) else count
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end <= count
                or not allocation['engineering_authority']
                or allocation['engineering_authority'] != allocation['reference']
                or allocation['prior_phase'] != 'OPEN'
                or not allocation['authorized_at'] < allocation['expires_at'] < programme_start.timestamp()
                or end-start > allocation['max_operations']
                or any(not allocation['authorized_at'] <= row['at'] < allocation['expires_at']
                       for row in value['operations'][start:end])):
            raise ValueError('invalid_preprogramme_source_accounting')
        used -= max(0, end-max(start, baseline_count))
    from race_collection.development_source_authority import development_source_usage
    from race_collection.incident_engineering import incident_source_usage
    return used - development_source_usage(value, baseline_count) - incident_source_usage(value, baseline_count)


def renew_source(cfg, slot, *, now):
    """Only an approved finite programme may renew an expired OPEN lease."""
    from race_collection.freshness_campaign import Campaign
    from utils.sportsbet_access import SportsbetAccess
    incident = cfg.get('incident_authority') is not None
    campaign = Campaign(cfg['campaign_root'], **{key: cfg[key] for key in
        ('incident_authority', 'incident_slot') if key in cfg})
    if digest(campaign.study_programme if incident else campaign.programme) != cfg['programme_authority_sha256']:
        raise ValueError('programme_authority_changed')
    with (campaign.root/'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with campaign.ledger() as ledger:
            if ledger.get('source_holds') or any(not r.get('closed_at') for r in ledger['launches'].values()):
                raise ValueError('campaign_hold_or_unfinished_owner')
        if Path(cfg['lock_path']).exists(): raise ValueError('collector_lock_busy')
        access = SportsbetAccess(cfg['source_state'])
        value = access.read()
        baseline = cfg['source_baseline']
        if (value['phase'] != 'OPEN' or value['active'] is not None or now.timestamp() < value['not_before']
                or digest(value['denials']) != baseline['denials_sha256']
                or value['recovery_attempts'] != baseline['recovery_attempts']
                or digest(value['access_basis']) != baseline['access_basis_sha256']
                or digest(value.get('operating_policy')) != baseline['operating_policy_sha256']):
            raise ValueError('source_requires_explicit_disposition')
        if incident:
            slot = cfg['incident_slot']
        used = (0 if incident else programme_source_usage(value, baseline['operation_count'], stamp(campaign.programme['starts_at'])))
        if not 0 <= used <= cfg['max_source_operations'] - 192:
            raise ValueError('programme_source_budget_exhausted')
        previous = [a for a in value.get('diagnostic_authorizations', [])
                    if a['reference'].startswith(cfg['authority_reference'] + ':slot:')]
        if (not incident and len(previous) >= len(cfg['slots'])) or any(a['reference'].endswith(':slot:'+slot) for a in previous):
            raise ValueError('source_slot_already_consumed')
        prior = hashlib.sha256(access.path.read_bytes()).hexdigest()
        end = (stamp(next(row for row in campaign.incident['slots'] if row['id'] == slot)['ends_at'])
               if incident else now+timedelta(hours=3))
        access.authorize_diagnostic(reference=cfg['authority_reference']+':slot:'+slot,
            expected_sha256=prior, expires_at=end.timestamp(),
            max_operations=192, rationale='Approved finite native collection; OPEN-only slot lease',
            **({'incident_authority': cfg['incident_authority'], 'incident_slot': slot} if incident else {}))
        return {'before_sha256': prior, 'after_sha256': hashlib.sha256(access.path.read_bytes()).hexdigest(),
                'slot': slot, 'operation_start': len(value.get('operations', []))}


def child(command, log):
    with log.open('ab') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=stream)
        try: return proc.wait()
        except BaseException:
            proc.terminate()
            # Supervisor handles SIGTERM with the same bounded restoration as
            # foreground mode. Never launch another owner while it is draining.
            try: proc.wait(timeout=2400)
            except subprocess.TimeoutExpired: pass
            raise


def verify_canary(cfg, result_cfg, root, now):
    import sqlite3
    if cfg.get('incident_acceptance'):
        from race_collection.incident_acceptance import verified_incident_acceptance
        evidence = verified_incident_acceptance(cfg, cfg['incident_acceptance'], now, seal=True)
        if evidence is not None:
            receipt = root/'canary.json'
            if not receipt.exists():
                create_once(receipt, {'status':'CANARY_STRUCTURALLY_VERIFIED', 'at':now.isoformat(),
                    'first_slot':cfg['slots'][0], 'plan_sha256':cfg['comparison_plan_sha256'],
                    'verified_predictions':evidence['verified_predictions'], 'closed_results':evidence['closed_results'],
                    'incident_acceptance':evidence, 'outcomes_released':False})
            return True
        return False
    first=root/'slots/001'
    terminal=first/'terminal.json'
    continuation=None
    if not terminal.exists() or json.loads(terminal.read_bytes()).get('status')!='COMPLETED':
        from race_collection.scientific_session_recovery import completed_continuation
        continuation=completed_continuation(cfg,root)
        if continuation is None:return False
    first_plan=first/(cfg['programme_id']+'-001')/'plan.json'
    admitted_plans={str(first_plan)}
    if continuation:
        admitted_plans.add(continuation['plan'])
        from race_collection.scientific_session_recovery import checked as checked_ref, prior_continuation_plans
        authority=checked_ref(cfg['first_session_continuation'])
        admitted_plans.update(ref['path'] for ref,_ in prior_continuation_plans(authority))
    prediction_root=Path(cfg['prediction_root'])
    verified=[]
    for path in (prediction_root/'dispatches').glob('*.json'):
        dispatch=json.loads(path.read_bytes())
        if dispatch.get('plan') not in admitted_plans:continue
        key=hashlib.sha256(dispatch['race_id'].encode()).hexdigest()
        terminal_path=prediction_root/'races'/key/'terminal.json'
        if terminal_path.exists():
            value=json.loads(terminal_path.read_bytes())
            if value.get('status')=='PREDICTION_READY':verified.append(value['job_id'])
    if not verified:return False
    queue=Path(result_cfg['state_root'])/'queue.sqlite3'
    if not queue.exists():return False
    with sqlite3.connect(queue.as_uri()+'?mode=ro',uri=True) as db:
        closed=sum(bool(db.execute("SELECT 1 FROM jobs WHERE job=? AND state='CLOSED'",(job,)).fetchone()) for job in verified)
    if not closed:return False
    receipt=root/'canary.json'
    if not receipt.exists():
        create_once(receipt,{'status':'CANARY_STRUCTURALLY_VERIFIED','at':now.isoformat(),
            'first_slot':cfg['slots'][0],'verified_predictions':len(verified),'closed_results':closed,
            'plan_sha256':cfg['comparison_plan_sha256'],'outcomes_released':False,
            **({'continuation':continuation} if continuation else {})})
    return True


def prepare_session(cfg, package, slot):
    from scripts.prepare_freshness_rehearsal import prepare
    roots=json.loads(checked(Path(cfg['reconciliation_roots']),cfg['reconciliation_roots_sha256']))
    # Each prior owned immutable package joins the next reconciliation inventory.
    # No old receipt or shared campaign root inventory is edited.
    prior_plans=set(Path(cfg['state_root']).glob('slots/*/*/plan.json'))
    if cfg.get('first_session_continuation'):
        ref=cfg['first_session_continuation'];authority=json.loads(checked(Path(ref['path']),ref['sha256']))
        ref=authority['continuation_plan'];checked(Path(ref['path']),ref['sha256'])
        prior_plans.add(Path(ref['path']))
        from race_collection.scientific_session_recovery import prior_continuation_plans
        prior_plans.update(Path(ref['path']) for ref,_ in prior_continuation_plans(authority))
    for plan_path in sorted(prior_plans):
        prior=json.loads(plan_path.read_bytes())
        for key in ('scheduled_progress','scheduled_reports','phase_checkpoints'):
            roots[key]=sorted(set(roots[key]+[prior['evidence_root']]))
        roots['prior_rehearsals']=sorted(set(roots['prior_rehearsals']+[str(plan_path.parent)]))
    return prepare(output=package,start=slot,python=Path(cfg['python']),
        db=Path(cfg['history_database']),lock=Path(cfg['lock_path']),reconciliation_roots=roots,
        installed_dir=Path(cfg['installed_dir']),campaign_root=Path(cfg['campaign_root']),
        operational_predictions=True,observation_minutes=90,comparison_plan=Path(cfg['comparison_plan']),
        prediction_root=Path(cfg['prediction_root']),
        **{key: cfg[key] for key in ('incident_authority', 'incident_slot') if key in cfg})


def tick(config_path):
    os.umask(0o077)
    cfg, plan = load_config(config_path)
    root = Path(cfg['state_root']); root.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (root/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity = root/'config-identity.json'
        if identity.exists():
            if json.loads(identity.read_bytes())['sha256'] != digest(cfg): raise ValueError('schedule_changed')
        else: create_once(identity, {'sha256':digest(cfg)})
        now = datetime.now(timezone.utc)
        # Recovery precedes expiry/pause checks: already owed cleanup is not
        # cancelled by stopping future admissions. Unknown boot/PID stays held.
        for claim in sorted((root/'slots').glob('*')):
            package = claim/(cfg['programme_id']+'-'+claim.name); pp = package/'plan.json'
            if (claim/'terminal.json').exists(): continue
            if pp.exists() and (package/'restoration.json').exists():
                previous = json.loads(pp.read_bytes())
                code = child([cfg['python'],'-B',str(Path(previous['source_root'])/'scripts/run_freshness_rehearsal.py'),
                    '--plan',str(pp),'--plan-sha256',digest(previous),'--approval-id',cfg['authority_reference'],
                    '--restore-only'],claim/'restore.log')
                if code or not (package/'restored.json').exists():
                    return {'status':'RESTORATION_HELD','outcomes_released':False}
            create_once(claim/'terminal.json',{'status':'INTERRUPTED_CONSUMED','at':now.isoformat()})
        incident_ready = None
        if cfg.get('incident_acceptance'):
            # Continue sealing fresh structural proofs before the private result
            # deadline, even after a canary exists. Later ticks validate only a
            # sealed proof and the native result closure, never expired reads.
            # An admission-only pause must not discard this finite opportunity.
            incident_ready = verify_canary(cfg, None, root, now)
        if (root/'PAUSE_ADMISSIONS').exists(): return {'status':'ADMISSIONS_PAUSED'}
        if now >= stamp(plan['ends_at']): return {'status':'ADMISSION_ENDPOINT_REACHED'}
        if incident_ready is False:
            return {'status':'CANARY_NOT_VERIFIED'}
        elif cfg.get('first_session_continuation') and not (root/'canary.json').exists():
            from race_collection.scientific_session_recovery import completed_continuation
            if cfg.get('incident_acceptance') or completed_continuation(cfg,root) is not None:
                from src.predictor.comparison_result_runtime import load_runtime
                binding=json.loads(Path(cfg['result_binding']).read_bytes())
                _,_,result_cfg=load_runtime(binding,now=now)
                verify_canary(cfg,result_cfg,root,now)
            if not (root/'canary.json').exists():
                # A failed predecessor is not permission to consume tomorrow's
                # admission while its explicit recovery proof remains incomplete.
                return {'status':'CANARY_NOT_VERIFIED'}
        for index, text in enumerate(cfg['slots']):
            slot = stamp(text)
            if now < slot-timedelta(minutes=10): continue
            claim = root/'slots'/f'{index+1:03d}'
            if claim.exists(): continue
            claim.mkdir(parents=True,mode=0o700)
            create_once(claim/'admission.json', {'slot':text,'claimed_at':now.isoformat(),
                'config_sha256':digest(cfg),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
            if now > slot-timedelta(minutes=5):
                create_once(claim/'terminal.json',{'status':'MISSED_SLOT','at':now.isoformat()}); continue
            if (shutil.disk_usage(root).free < 10*2**30 or sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > 40*2**30
                    or sum(p.stat().st_size for p in Path(cfg['prediction_root']).rglob('*') if p.is_file()) > 20*2**30):
                create_once(claim/'terminal.json',{'status':'DISK_HOLD','at':now.isoformat()})
                return {'status':'DISK_HOLD'}
            from src.predictor.comparison_result_runtime import load_runtime
            result_binding=json.loads(Path(cfg['result_binding']).read_bytes())
            _,_,result_cfg=load_runtime(result_binding,now=now)
            result_health_path=Path(result_cfg['state_root'])/'health.json'
            if index > 0:
                if not result_health_path.exists(): return {'status':'RESULT_HEALTH_MISSING'}
                result_health=json.loads(result_health_path.read_bytes())
                if not verify_canary(cfg,result_cfg,root,now):
                    return {'status':'CANARY_NOT_VERIFIED'}
                if (now-stamp(result_health['at']) > timedelta(minutes=45) or result_health['status'] not in {'CYCLE_COMPLETE','COLLECTOR_LOCK_BUSY','CAMPAIGN_OWNER_BUSY'}
                        or result_health.get('oldest_due') and now-stamp(result_health['oldest_due']) > timedelta(days=1)):
                    return {'status':'RESULT_RETENTION_HOLD'}
            atomic_json(root/'health.json',{'status':'SESSION_RUNNING','at':now.isoformat(),'slot':text,'outcomes_released':False})
            create_once(claim/'source-lease.json',renew_source(cfg,str(index+1),now=now))
            package = claim/(cfg['programme_id']+'-'+claim.name)
            prepare_session(cfg, package, slot)
            pp=package/'plan.json'; prepared=json.loads(pp.read_bytes())
            code=child([cfg['python'],'-B',str(Path(prepared['source_root'])/'scripts/run_freshness_rehearsal.py'),
                '--plan',str(pp),'--plan-sha256',digest(prepared),'--approval-id',cfg['authority_reference']],claim/'session.log')
            if not (package/'restored.json').exists():
                return {'status':'RESTORATION_HELD'}
            create_once(claim/'terminal.json',{'status':'COMPLETED' if code==0 else 'FAILED_RESTORED',
                                             'at':datetime.now(timezone.utc).isoformat()})
            return {'status':'SESSION_COMPLETED' if code==0 else 'SESSION_FAILED_RESTORED'}
        return {'status':'NO_SLOT_DUE'}


def health(config_path, value):
    # A malformed/unapproved config is never permission to create arbitrary paths.
    try:
        cfg, _ = load_config(config_path)
        root=Path(cfg['state_root'])
        if root.is_dir():
            atomic_json(root/'health.json',{**value,'at':datetime.now(timezone.utc).isoformat()})
    except Exception:
        pass  # systemd retains failure exit/journal when no valid store is bound


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    def stop(signum,frame):raise InterruptedError('termination')
    signal.signal(signal.SIGTERM,stop)
    try:
        result=tick(args.config);result['outcomes_released']=False
        health(args.config,result)
        print(json.dumps(result))
        return 0 if result['status'] in {'NO_SLOT_DUE','ADMISSIONS_PAUSED','ADMISSION_ENDPOINT_REACHED','SESSION_COMPLETED'} else 2
    except Exception as exc:
        result={'status':'SCHEDULE_FAILED','failure_class':type(exc).__name__,'outcomes_released':False}
        health(args.config,result)
        print(json.dumps(result))
        return 2


if __name__=='__main__':raise SystemExit(main())
