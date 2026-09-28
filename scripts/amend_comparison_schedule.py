"""Prepare a prospective, date-only amendment for a wholly unconsumed programme.

No service or shared authority writes. Install only while all workers are
quiescent, rechecking empty_state and preserving the superseded packet/stores.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
from zoneinfo import ZoneInfo

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import digest, encoded
from src.predictor.future_comparison import checked


def calendar(start):
    local = datetime.fromisoformat(start)
    start = local.replace(tzinfo=ZoneInfo('Australia/Melbourne'))
    if local.tzinfo is not None and local.utcoffset() != start.utcoffset():
        raise ValueError('melbourne_offset_mismatch')
    if start.hour != 12 or start.minute or start.second or start.microsecond:
        raise ValueError('local_noon_start_required')
    end = start + timedelta(days=112)
    slots = [(start + timedelta(days=d)).replace(hour=13).isoformat()
             for d in range(112) if (start + timedelta(days=d)).weekday() < 5]
    assert len(slots) == 80
    return start.isoformat(), end.isoformat(), (end + timedelta(days=14)).isoformat(), slots


def empty_state(cfg):
    """Read structural counts only. Never decode histories or result rows."""
    plan = json.loads(checked(Path(cfg['comparison_plan']), cfg['comparison_plan_sha256']))
    binding = json.loads(Path(cfg['result_binding']).read_bytes())
    if binding['plan_sha256'] != cfg['comparison_plan_sha256']:
        raise ValueError('result_plan_mismatch')
    authority = json.loads(checked(Path(binding['authority']), binding['authority_sha256']))
    campaign = Campaign(cfg['campaign_root'])
    if digest(campaign.programme) != cfg['programme_authority_sha256']:
        raise ValueError('programme_authority_mismatch')
    ledger_path = campaign.root/'ledger.json'
    ledger = json.loads(ledger_path.read_bytes())
    source_path = Path(cfg['source_state'])
    source = json.loads(source_path.read_bytes())
    counts = {'slots': len(list((Path(cfg['state_root'])/'slots').glob('*')))}
    for name, path in [('membership', Path(plan['programme_root'])), ('prediction', Path(cfg['prediction_root']))]:
        counts[name + '_files'] = sum(p.is_file() for p in path.rglob('*'))
    root = Path(authority['runtime']['state_root'])
    queue = root/'queue.sqlite3'
    if queue.exists():
        with sqlite3.connect(queue.as_uri()+'?mode=ro', uri=True) as db:
            for table in ('jobs', 'requests', 'events'):
                counts['queue_'+table] = db.execute('SELECT count(*) FROM '+table).fetchone()[0]
    result_db = Path(authority['result_database'])
    if result_db.exists():
        with sqlite3.connect(result_db.as_uri()+'?mode=ro', uri=True) as db:
            for table in ('autonomous_official_result_evidence_races', 'autonomous_official_result_evidence_runners'):
                counts[table] = db.execute('SELECT count(*) FROM '+table).fetchone()[0]
    counts['result_evidence_files'] = sum(p.is_file() for folder in ('attempts', 'requests', 'closure')
                                         for p in (root/folder).rglob('*'))
    current = {'capture_attempts': len(ledger['attempts']), 'logical_requests': ledger['logical_requests'],
               'live_seconds': math.ceil(sum(r['charged_seconds'] for r in ledger['launches'].values()))}
    if (any(counts.values()) or current != campaign.programme['initial_counters']
            or ledger.get('persistent_request_usage') or ledger.get('source_holds')
            or any(not r.get('closed_at') for r in ledger['launches'].values())
            or source['phase'] != 'OPEN' or source['active'] is not None
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != cfg['source_baseline']['state_sha256']
            or Path(cfg['lock_path']).exists()):
        raise ValueError('programme_consumed_held_or_changed')
    return {'counts': counts, 'campaign_counters': current,
            'ledger_sha256': hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
            'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'programme_sha256': digest(campaign.programme), 'schedule_sha256': digest(cfg),
            'outcomes_accessed': False}


def prepare(*, previous_control, output, start, source_commit, approval_reference, reservation_review):
    now = datetime.now(timezone.utc)
    starts, ends, closure, slots = calendar(start)
    if datetime.fromisoformat(starts) <= now:
        raise ValueError('future_start_required')
    if not approval_reference.strip():
        raise ValueError('explicit_amendment_reference_required')
    root = Path(__file__).resolve().parents[1]
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != source_commit
            or subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=root, text=True).strip()):
        raise ValueError('source_identity_mismatch')
    old = {name: json.loads((previous_control/(name+'.APPROVED.json')).read_bytes())
           for name in ('plan', 'allocation', 'programme-authority', 'result-authority', 'result-binding', 'schedule')}
    cfg = old['schedule']
    from race_collection.persistent_storage import check_mount
    check_mount(cfg['storage_mount'], output)
    proof = empty_state(cfg)
    review = json.loads(reservation_review.read_bytes())
    if review.get('status') != 'RECHECKED_NO_NEW_RESERVATIONS' or not review.get('deferrals_preserved'):
        raise ValueError('current_reservation_review_required')
    values = deepcopy(old)
    values['plan'].update(starts_at=starts, ends_at=ends,
        activated_at=now.isoformat(), schedule_amendment_reference=approval_reference,
        reservation_review_sha256=hashlib.sha256(reservation_review.read_bytes()).hexdigest())
    # Original approval and scientific rules stay bound; the new reference covers
    # only dates/allocation extension, not new training or evaluation authority.
    values['allocation'].update(starts_at=starts, ends_at=ends, schedule_amendment_reference=approval_reference,
        reservation_review_sha256=values['plan']['reservation_review_sha256'])
    values['programme-authority'].update(starts_at=starts, expires_at=closure)
    amendment = {'schema_version': 'programme_schedule_amendment_v1',
        'authority_reference': approval_reference, 'issued_at': now.isoformat(),
        'prior_programme_sha256': proof['programme_sha256'], 'empty_state_sha256': digest(proof),
        'programme': values['programme-authority']}
    # Fresh binding stores are allowed only with the zero-consumption proof.
    # Old stores, including their original identities and all health records, stay.
    programme_root = previous_control.parent
    runtime = values['result-authority']['runtime']
    runtime.update(state_root=str(programme_root/'results-october1'), expires_at=closure)
    values['result-authority'].update(result_database=str(Path(runtime['state_root'])/'official-results.sqlite3'),
        issued_at=now.isoformat(), schedule_amendment_reference=approval_reference)
    cfg = values['schedule']
    cfg.update(slots=slots, state_root=str(programme_root/'sessions-october1'), source_commit=source_commit,
        schedule_amendment_reference=approval_reference)
    if Path(cfg['state_root']).exists() or Path(runtime['state_root']).exists():
        raise ValueError('replacement_store_already_exists')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    def put(name, value):
        path = output/(name+'.json')
        with path.open('xb') as stream:
            stream.write(encoded(value)); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o400)
        return str(path), digest(value)
    for name in ('allocation', 'plan', 'programme-authority'):
        put(name+'.APPROVED', values[name])
    plan_path = str(output/'plan.APPROVED.json'); plan_hash = digest(values['plan'])
    values['result-authority']['plan_sha256'] = plan_hash
    authority_path, authority_hash = put('result-authority.APPROVED', values['result-authority'])
    binding = {'plan': plan_path, 'plan_sha256': plan_hash, 'authority': authority_path, 'authority_sha256': authority_hash}
    binding_path, _ = put('result-binding.APPROVED', binding)
    cfg.update(comparison_plan=plan_path, comparison_plan_sha256=plan_hash, result_binding=binding_path,
        programme_authority_sha256=digest(values['programme-authority']))
    put('schedule.APPROVED', cfg)
    put('empty-state', proof); put('campaign-amendment', amendment)
    put('reservation-review', review)
    receipt = {'status': 'AUTHORIZED_AMENDMENT_PREPARED_NOT_INSTALLED', 'at': now.isoformat(),
        'authority_reference': approval_reference, 'source_commit': source_commit,
        'previous_control': str(previous_control), 'starts_at': starts, 'ends_at': ends,
        'closure_at': closure, 'first_session': slots[0], 'last_session': slots[-1], 'sessions': len(slots),
        'superseded_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in previous_control.glob('*.APPROVED.json')},
        'replacement_files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.json')},
        'provider_requests': 0, 'scientific_observations': 0}
    put('amendment-receipt', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('previous-control', 'output', 'reservation-review'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('start', 'source-commit', 'approval-reference'):
        p.add_argument('--'+name, required=True)
    print(json.dumps(prepare(**vars(p.parse_args())), sort_keys=True))
