"""Verify explicit, bounded session continuation without rewriting a failed slot."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib, json
import math
from race_collection.live_freshness_contract import digest

def read(path):
    return json.loads(Path(path).read_bytes())

def require(condition):
    if not condition:
        raise ValueError('continuation_proof_invalid')

def checked(ref):
    raw = Path(ref['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('continuation_evidence_changed')
    return json.loads(raw)


def prior_continuation_plans(authority):
    """Retain the explicitly resolved UI-restart failure in the same allowance."""
    records = authority.get('prior_continuations', [])
    require(len(records) <= 1)
    plans = []
    for record in records:
        previous = checked(record['authority'])
        plan = checked(record['plan'])
        terminal = checked(record['terminal'])
        restored = checked(record['restored'])
        resolution = checked(record['resolution'])
        ack = checked(record['r3_ack'])
        require(previous['continuation_plan'] == record['plan'])
        for name in ('original_plan', 'comparison_plan_sha256', 'campaign_root', 'prediction_root', 'source_lease_sha256'):
            require(previous[name] == authority[name])
        require(plan['commit'] == previous['source_commit'])
        require(terminal['status'] == 'RESTORATION_HELD' and terminal['plan_sha256'] == record['plan']['sha256'])
        require(terminal['authority_sha256'] == record['authority']['sha256'])
        require(restored['status'] in {'RESTORED', 'RESTORED_COLLECTOR_TRIGGERS_HELD'} and restored['sportsbet_hold'] is False)
        require(resolution['status'] == 'RESTORATION_COMPLETED_AFTER_EXPLICIT_UI_REPLACEMENT')
        require(resolution['original_terminal_sha256'] == record['terminal']['sha256'])
        require(resolution['restored_sha256'] == record['restored']['sha256'])
        require(resolution['plan_sha256'] == record['plan']['sha256'])
        require(resolution['ack_sha256'] == record['r3_ack']['sha256'])
        require(resolution['collection_resumed'] is False and resolution['outcomes_released'] is False)
        require(restored['r3_replacement_acknowledgement']['sha256'] == record['r3_ack']['sha256'])
        require(ack['plan_sha256'] == record['plan']['sha256'] and ack['collection_resume_allowed'] is False)
        require(read(Path(record['plan']['path']).parent/'failure.json')['reason'] == 'installed_r3_changed')
        plans.append((record['plan'], plan))
    return plans

def completed_continuation(cfg, root):
    """Return authenticated continuation metadata, or None while held/incomplete."""
    try:
        ref = cfg['first_session_continuation']
        authority = checked(ref)
        first = Path(root) / 'slots/001'
        original = first / (cfg['programme_id'] + '-001')
        require(authority['schema_version'] == 'scientific_session_continuation_v1')
        require(authority['status'] == 'AUTHORIZED_CONTINUATION' and authority['authority_reference'])
        require(authority['original_slot'] == '001' and authority['comparison_plan_sha256'] == cfg['comparison_plan_sha256'])
        require(authority['campaign_root'] == cfg['campaign_root'] and authority['prediction_root'] == cfg['prediction_root'])
        require(authority['original_failure_preserved'] is True)
        require(all((authority[k] is False for k in ('consumed_races_retriable', 'new_source_allocation', 'outcome_values_accessed', 'metrics_computed'))))
        expected = {'original_admission': first / 'admission.json', 'original_terminal': first / 'terminal.json', 'original_plan': original / 'plan.json', 'original_restored': original / 'restored.json'}
        evidence = {}
        for (name, path) in expected.items():
            require(Path(authority[name]['path']) == path)
            evidence[name] = checked(authority[name])
        require(evidence['original_terminal']['status'] == 'FAILED_RESTORED')
        require(evidence['original_restored']['sportsbet_hold'] is False)
        plan = checked(authority['continuation_plan'])
        package = Path(authority['continuation_plan']['path']).parent
        require(package.parent == Path(ref['path']).parent)
        require(plan['commit'] == authority['source_commit'] and plan['frozen_comparison']['sha256'] == cfg['comparison_plan_sha256'])
        require(plan['campaign_root'] == cfg['campaign_root'] and plan['prediction_root'] == cfg['prediction_root'])
        require(type(authority['prior_prediction_requests']) is int and authority['prior_prediction_requests'] >= 0)
        require(type(plan['max_logical_requests']) is int)
        require(plan['max_logical_requests'] == authority['continuation_request_cap'])
        require(0 < plan['max_logical_requests'] <= 16000 - authority['prior_prediction_requests'])
        require(authority['prediction_request_total_limit'] == 16000)
        for name in ('starts_at', 'ends_at', 'cleanup_seconds'):
            require(plan[name] == authority[name])
        start = datetime.fromisoformat(plan['starts_at'])
        end = datetime.fromisoformat(plan['ends_at'])
        require(datetime.fromisoformat(authority['issued_at']) < start < end)
        require(start.astimezone(ZoneInfo('Australia/Melbourne')).date() == datetime.fromisoformat(cfg['slots'][0]).date())
        require(digest(authority['source_lease']) == authority['source_lease_sha256'])
        lease = authority['source_lease']
        receipt = read(first / 'source-lease.json')
        require(receipt['slot'] == '1' and receipt['operation_start'] == lease['operation_start'])
        require(receipt['before_sha256'] == lease['prior_state_sha256'])
        require(lease['reference'] == cfg['authority_reference'] + ':slot:1')
        matches = [item for item in read(cfg['source_state'])['diagnostic_authorizations']
                   if item['reference'] == lease['reference']]
        require(len(matches) == 1 and digest(matches[0]) == authority['source_lease_sha256'])
        require(authority['source_operations_total_limit'] == authority['source_lease']['max_operations'] == 192)
        require(end.timestamp() + plan['cleanup_seconds'] < authority['source_lease']['expires_at'])
        require(authority['total_charge_ceiling_seconds'] == 7260)
        require(math.isfinite(authority['prior_charged_seconds']) and authority['prior_charged_seconds'] >= 0)
        require(type(plan['cleanup_seconds']) is int and plan['cleanup_seconds'] >= 0)
        require(authority['prior_charged_seconds'] + (end - start).total_seconds() + plan['cleanup_seconds'] <= 7260)
        started = read(package / 'started.json')
        terminal = read(package.parent / 'terminal.json')
        require(started['approval_id'] == authority['authority_reference'] and started['plan_sha256'] == digest(plan))
        require(terminal['status'] == 'COMPLETED' and terminal['returncode'] == 0 and (terminal['outcomes_released'] is False))
        require(terminal['authority_sha256'] == digest(authority) and terminal['plan_sha256'] == digest(plan))
        require(not (package / 'failure.json').exists())
        measured = read(package / 'measurement.json')
        restored = read(package / 'restored.json')
        require(type(measured['logical_requests']) is int and measured['logical_requests'] >= 0)
        require(measured['status'] == 'REHEARSAL_MEASURED_NOT_RELEASED' and measured['logical_requests'] <= plan['max_logical_requests'])
        require(restored['status'] in {'RESTORED', 'RESTORED_COLLECTOR_TRIGGERS_HELD'} and restored['sportsbet_hold'] is False)
        launches = read(Path(cfg['campaign_root']) / 'ledger.json')['launches']
        old = launches[evidence['original_plan']['rehearsal_id']]
        new = launches[plan['rehearsal_id']]
        prior = prior_continuation_plans(authority)
        previous = [old] + [launches[p['rehearsal_id']] for _, p in prior]
        require(len({evidence['original_plan']['rehearsal_id'], plan['rehearsal_id'], *(p['rehearsal_id'] for _, p in prior)}) == 2 + len(prior))
        require(all(math.isfinite(item['charged_seconds']) and item['charged_seconds'] >= 0 and item['closed_at'] for item in [*previous, new]))
        require(old['charged_seconds'] == authority.get('original_charged_seconds', authority['prior_charged_seconds']))
        require(sum(item['charged_seconds'] for item in previous) == authority['prior_charged_seconds'])
        require(authority['prior_charged_seconds'] + new['charged_seconds'] <= 7260)
        return {'plan': str(package / 'plan.json'), 'authority_sha256': ref['sha256'], 'terminal_sha256': hashlib.sha256((package.parent / 'terminal.json').read_bytes()).hexdigest(), 'original_terminal_sha256': authority['original_terminal']['sha256'], 'original_failure_preserved': True}
    except (KeyError, ValueError, TypeError, OSError):
        return None
