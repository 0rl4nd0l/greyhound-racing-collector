"""Structural persistent-worker monitoring; never read result databases or bodies."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from race_collection.live_phase_checkpoint import atomic_json

SCHEDULE_OK = {'NO_SLOT_DUE', 'ADMISSIONS_PAUSED', 'ADMISSION_ENDPOINT_REACHED', 'SESSION_COMPLETED', 'SESSION_RUNNING'}
RESULT_OK = {'CYCLE_COMPLETE', 'COLLECTOR_LOCK_BUSY', 'CAMPAIGN_OWNER_BUSY', 'CLOSURE_SEALED'}


def stamp(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('timestamp_without_timezone')
    return value.astimezone(timezone.utc)


def read(path):
    return json.loads(Path(path).read_bytes())


def active(unit):
    return subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '--value'],
                                   text=True, timeout=5).strip()


def evaluate(schedule, results, *, now, schedule_active):
    """Use only health status, timestamps and aggregate queue counters."""
    alerts = []
    for label, value, accepted, minutes in (
            ('schedule', schedule, SCHEDULE_OK, 15), ('results', results, RESULT_OK, 45)):
        if value is None:
            alerts.append(label + ':health_missing')
            continue
        state = value.get('status')
        if state not in accepted:
            alerts.append(label + ':worker_hold_or_failure')
        if label == 'schedule' and state == 'SESSION_RUNNING':
            minutes = 135  # 10-minute preparation + 90-minute session + bounded cleanup
            if schedule_active != 'active':
                alerts.append('schedule:running_without_service')
        age = (now - stamp(value['at'])).total_seconds()
        if age < -60 or (age > minutes * 60 and state != 'CLOSURE_SEALED'):
            alerts.append(label + ':health_stale_or_future')
    if results:
        if results.get('oldest_due') and now - stamp(results['oldest_due']) > timedelta(days=1):
            alerts.append('results:overdue_more_than_24h')
        counts = results.get('counts', {})
        if any(counts.get(state, 0) for state in ('QUARANTINED', 'ATTEMPTS_EXHAUSTED', 'DEADLINE_UNRESOLVED')):
            alerts.append('results:unresolved_terminal_members')
    return alerts


def inspect(schedule_config, result_binding, *, now=None):
    now = now or datetime.now(timezone.utc)
    cfg, binding = read(schedule_config), read(result_binding)
    if cfg.get('status') != 'AUTHORIZED_PERSISTENT_SCHEDULE':
        raise ValueError('schedule_not_authorized')
    authority_bytes = Path(binding['authority']).read_bytes()
    if hashlib.sha256(authority_bytes).hexdigest() != binding['authority_sha256']:
        raise ValueError('result_authority_changed')
    runtime = json.loads(authority_bytes)['runtime']
    if Path(cfg['result_binding']) != result_binding or runtime['storage_mount'] != cfg['storage_mount']:
        raise ValueError('monitor_binding_mismatch')
    from race_collection.persistent_storage import check_mount
    check_mount(cfg['storage_mount'], Path(cfg['state_root']))
    check_mount(cfg['storage_mount'], Path(runtime['state_root']))
    health = {}
    for name, root in (('schedule', cfg['state_root']), ('results', runtime['state_root'])):
        path = Path(root) / 'health.json'
        health[name] = read(path) if path.exists() else None
    states = {name: active('greyhound-comparison-' + name.removesuffix('_timer') + suffix)
              for name, suffix in (('schedule', '.service'), ('results', '.service'), ('health_timer', '.timer'), ('schedule_timer', '.timer'), ('results_timer', '.timer'))}
    alerts = evaluate(health['schedule'], health['results'], now=now, schedule_active=states['schedule'])
    for label in ('schedule_timer', 'results_timer', 'health_timer'):
        if states[label] != 'active':
            alerts.append(label + ':inactive')
    gate = read(cfg['source_state'])
    ledger = read(Path(cfg['campaign_root']) / 'ledger.json')
    if gate['phase'] != 'OPEN' or gate.get('not_before', 0) > now.timestamp():
        alerts.append('source:hold')
    if ledger.get('source_holds'):
        alerts.append('campaign:source_hold')
    usage = shutil.disk_usage(cfg['storage_mount']['path'])
    free = usage.free
    from scripts.comparison_status import programme
    overview = programme(cfg, health['results'], now=now)
    if any(n for state,n in overview['sessions'].items() if state not in {'COMPLETED', 'IN_PROGRESS'}):
        alerts.append('schedule:retained_failed_or_missed_slot')
    if states['results'] == 'failed':
        alerts.append('results:service_failed')
    overview['campaign_capture_attempts_since_activation'] = len(ledger['attempts']) - read(Path(cfg['campaign_root'])/'persistent-programme-authority.json')['initial_counters']['capture_attempts']
    if free < 10 * 2**30:
        alerts.append('storage:prediction_admission_floor')
    return {'schema_version': 'comparison_monitor_health_v1', 'at': now.isoformat(),
            'status': 'ALERT' if alerts else 'HEALTHY', 'alerts': alerts,
            'worker_status': {k: v.get('status') if v else None for k, v in health.items()},
            'units': states, 'volume_free_bytes': free, 'volume_used_bytes': usage.used,
            'source_phase': gate['phase'], 'programme': overview, 'outcomes_released': False}, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule-config', type=Path, required=True)
    parser.add_argument('--result-binding', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--notification-config', type=Path)
    parser.add_argument('--human', action='store_true', help='Concise outcome-blind status view')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        value, cfg = inspect(args.schedule_config, args.result_binding)
        if args.output:
            from race_collection.persistent_storage import check_mount
            check_mount(cfg['storage_mount'], args.output)
            args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            from scripts.comparison_notifications import deliver
            try:
                delivery = deliver(value, args.notification_config, args.output.parent/'notification-state.json')
            except Exception:
                delivery = 'DELIVERY_CONFIGURATION_FAILED'
            value['programme']['notifications'] = delivery
            if delivery in {'DELIVERY_FAILED', 'DELIVERY_CONFIGURATION_FAILED'}:
                value['alerts'].append('notifications:delivery_failed')
                value['status'] = 'ALERT'
            atomic_json(args.output, value)
    except Exception as exc:
        # Invalid configuration/missing mount must not create fallback directories.
        value = {'status': 'ALERT', 'at': datetime.now(timezone.utc).isoformat(),
                 'alerts': ['monitor_check_failed'], 'failure_class': type(exc).__name__,
                 'outcomes_released': False}
    from scripts.comparison_status import render
    print(render(value) if args.human else json.dumps(value, sort_keys=True))
    return 0 if value['status'] == 'HEALTHY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
