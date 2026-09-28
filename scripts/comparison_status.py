"""Outcome-blind projections for the existing persistent monitor."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(path.read_bytes())


def programme(cfg, result_health, *, now):
    root = Path(cfg['state_root'])
    slot_counts = Counter()
    windows = Counter()
    samples = unavailable = 0
    timer_gaps = 0
    latest = None
    for claim in sorted((root/'slots').glob('*')):
        terminal = claim/'terminal.json'
        slot_counts[read(terminal)['status'] if terminal.exists() else 'IN_PROGRESS'] += 1
        package = claim/(cfg['programme_id']+'-'+claim.name)
        measurement = package/'measurement.json'
        progress = package/'progress.json'
        path = measurement if measurement.exists() else progress
        if path.exists():
            value = read(path)
            for key in ('eligible_observed_windows', 'attempted_windows', 'missed_observed_windows', 'excluded_observations', 'pending_at_end'):
                windows[key] += len(value.get('windows', {}).get(key, []))
            timer_gaps += sum(r['status'] in {'NO_TRIGGER_OBSERVED', 'TRIGGER_WITHOUT_OBSERVED_START'}
                              for r in value.get('timer_accounting', {}).get('odds_calendar_slots', []))
            samples += value.get('sample_count', 0)
            unavailable += value.get('unavailable_samples_including_warmup', 0)
            latest = {k: value.get(k) for k in ('last_sample_at', 'source_age_seconds', 'index_status', 'maximum_conservative_source_age')}
    next_slot = next((s for i,s in enumerate(cfg['slots'], 1)
                      if not (root/'slots'/f'{i:03d}').exists() and datetime.fromisoformat(s)>now), None)
    predictions = Counter()
    for path in (Path(cfg['prediction_root'])/'races').glob('*/terminal.json'):
        predictions[read(path).get('status', 'UNKNOWN')] += 1
    counts = (result_health or {}).get('counts', {})
    jump = (result_health or {}).get('oldest_outstanding_jump')
    due = (result_health or {}).get('oldest_due')
    result = {'next_session': next_slot, 'installed_release': cfg['source_commit'],
        'sessions': dict(slot_counts), 'first_session_gate_verified': (root/'canary.json').exists(),
        'observed_opportunities': windows['eligible_observed_windows'],
        'attempted_captures': windows['attempted_windows'], 'missed_observed_windows': windows['missed_observed_windows'],
        'verified_predictions': predictions['PREDICTION_READY'], 'prediction_status_counts': dict(predictions),
        'input_freshness': latest or {'index_status': 'NOT_OBSERVED'},
        'unavailable_observation_samples': unavailable, 'observation_samples': samples,
        'unverified_timer_intervals': timer_gaps,
        'unavailable_intervals': 'Retained sample/refresh outage records; gaps outside scheduled observation are unassessed',
        'outstanding_results': sum(n for state,n in counts.items() if state!='CLOSED'),
        'result_status_counts': counts,
        'oldest_outstanding_seconds': max(0, (now-datetime.fromisoformat(jump)).total_seconds()) if jump else None,
        'oldest_overdue_seconds': max(0, (now-datetime.fromisoformat(due)).total_seconds()) if due else None,
        'notifications': 'LOCAL_ONLY_NO_DESTINATION'}
    return result


def render(value):
    """One concise view, only allowlisted structural fields."""
    p = value.get('programme', {})
    slot = p.get('next_session')
    local = datetime.fromisoformat(slot).astimezone(ZoneInfo('Australia/Melbourne')).strftime('%a %d %b %Y %H:%M %Z') if slot else 'none'
    fresh = p.get('input_freshness', {})
    return '\n'.join([
        f"{value['status']} | Next session: {local} | Release: {p.get('installed_release', 'unknown')}",
        'Workers: '+json.dumps(value.get('worker_status', {}), sort_keys=True)+' | Units: '+json.dumps(value.get('units', {}), sort_keys=True),
        f"Inputs: {fresh.get('index_status')} | Last source age: {fresh.get('source_age_seconds')}s | Unverified minute intervals: {p.get('unverified_timer_intervals', 0)} | Unavailable samples: {p.get('unavailable_observation_samples', 0)}/{p.get('observation_samples', 0)}",
        f"Opportunities/windows: {p.get('observed_opportunities', 0)} | Capture attempts: {p.get('attempted_captures', 0)} | Verified predictions: {p.get('verified_predictions', 0)}",
        f"Outstanding results: {p.get('outstanding_results', 0)} | Oldest since jump: {p.get('oldest_outstanding_seconds')}s | First-session gate: {p.get('first_session_gate_verified')}",
        f"Source: {value.get('source_phase')} | Free disk: {value.get('volume_free_bytes', 0)/2**30:.1f} GiB | Used volume: {value.get('volume_used_bytes', 0)/2**30:.1f} GiB",
        'Intervention: '+(', '.join(value.get('alerts', [])) or 'none')+' | Notifications: '+p.get('notifications', 'LOCAL_ONLY_NO_DESTINATION'),
    ])
