"""A completed metadata exclusion permits another scheduled census, not a forecast."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re


def classify_metadata_exclusion(evidence, run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_+.-]+', run_id):
        return None
    try:
        evidence = Path(evidence)
        root = evidence / ('shadow_autopilot_daemonization_v1_' + run_id)
        checkpoint = json.loads((root / 'phase-checkpoint.json').read_bytes())
        phases = checkpoint['phases']
        if checkpoint['cycle_id'] != run_id or not phases or phases[-1]['kind'] != 'refresh':
            return None
        for number, phase in enumerate(phases):
            if (phase.get('number', number) != number or phase['status'] != 'COMPLETE'
                    or phase['budget_exceeded'] or phase['kind'] not in {'refresh', 'capture'}):
                return None
            raw = (root / f'phase-{number}-result.json').read_bytes()
            if hashlib.sha256(raw).hexdigest() != phase['result_sha256']:
                return None
            result = json.loads(raw)
            if number < len(phases)-1 and (result.get('status') != 'PASS'
                    or result.get('collection_phase') != phase['kind']):
                return None
        if result.get('collection_phase') != 'refresh' or result.get('final_verdict') != 'COLLECTION_PHASE_BLOCKED':
            return None
        name = 'odds_capture_refresh_report.json' if run_id.endswith('_odds_capture') else 'refresh_prejump_report.json'
        path = evidence / f'shadow_autopilot_v1_{run_id}_phase_{number}' / name
        publication = result.get('current_race_index_publish', {})
        if (result.get('status') != 'FAIL' or result.get('output_dir') != str(path.parent)
                or publication.get('schema_version') != 'collector_current_race_index_publish_v2'
                or publication.get('status') != 'REJECTED'
                or publication.get('reason') != 'CURRENT_INDEX_SOURCE_INVALID'
                or publication.get('failure_detail') != {'reason': 'refresh_not_accepted_success'}
                or publication.get('source_refresh_report_path') != str(path)
                or publication.get('run_id') != run_id
                or publication.get('index_path') != str(evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json')):
            return None
        raw = path.read_bytes()
        report = json.loads(raw)
        from scripts.refresh_prejump_upcoming import complete_unavailable_metadata_selection
        if not complete_unavailable_metadata_selection(report):
            return None
        return dict(schema_version='completed_metadata_exclusions_v1',
            disposition='COMPLETED_METADATA_EXCLUSIONS', run_id=run_id,
            failed_phase_number=number, phase_result_sha256=phase['result_sha256'],
            refresh_sha256=hashlib.sha256(raw).hexdigest(),
            selected_count=report['selected_count'], excluded_count=report['selected_count'],
            request_retries_added=0, current_index_published=False)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def verify_metadata_exclusion(reference, evidence, allocation_sha):
    from race_collection.persistent_authority import checked
    value = checked(reference)
    expected = classify_metadata_exclusion(evidence, value.get('run_id'))
    observed = datetime.fromisoformat(value['observed_at'])
    if (expected is None or any(value.get(k) != v for k, v in expected.items())
            or value.get('allocation_sha256') != allocation_sha
            or observed.utcoffset() is None):
        raise ValueError('persistent_metadata_exclusion_unverified')
    return value


def metadata_exclusion_pending(plan, reference, allocation_sha, current):
    if reference is None:
        return False
    value = verify_metadata_exclusion(reference, plan['evidence_root'], allocation_sha)
    observed = datetime.fromisoformat(value['observed_at'])
    if observed > current:
        raise ValueError('persistent_metadata_exclusion_future')
    from race_collection.synchronous_manual_capture import bounded_current_race_index, CaptureOneRejected
    evidence = Path(plan['evidence_root'])
    try:
        view = bounded_current_race_index(current_time=current, timeout_seconds=5,
            index_path=evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json',
            evidence_root=evidence, max_age_seconds=270, return_verified_view=True)
        generated = datetime.fromisoformat(view.source_generated_at)
        return not (observed < generated <= current and (current-generated).total_seconds() < 270)
    except CaptureOneRejected as exc:
        if exc.code in {'CURRENT_INDEX_UNAVAILABLE', 'CURRENT_INDEX_STALE', 'DISCOVERY_TIMEOUT'}:
            return True
        raise
