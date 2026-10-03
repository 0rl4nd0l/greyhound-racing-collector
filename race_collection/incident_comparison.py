"""Non-evaluative incident admission using the frozen native comparison path.

This is an explicit reuse authority, never an amendment of study membership.
"""
import hashlib
import json
from pathlib import Path

from race_collection.incident_engineering import load_incident_authority


STATUS = 'AUTHORIZED_ENGINEERING'


def validate_incident_plan(plan):
    if plan.get('persistent_allocation') is not None:
        from race_collection.persistent_comparison import validate_persistent_plan
        return validate_persistent_plan(plan)
    from src.predictor.future_comparison import checked, stamp
    authority = load_incident_authority(plan['incident_authority'])
    slots = {row['id']: row for row in authority['slots']}
    slot = slots[plan['incident_slot']]
    reference = authority['study_plan']
    study = json.loads(checked(Path(reference['path']), reference['sha256']))
    unchanged = ('schema_version', 'candidate_registry', 'decision_seconds_before_jump',
                 'quote_lead_seconds', 'denied_history_intervals', 'history_policy',
                 'machine_history_authority_reference', 'fixed_closure_days',
                 'missing_result_policy')
    if (plan['status'] != STATUS or study['status'] != 'AUTHORIZED'
            or any(plan.get(key) != study.get(key) for key in unchanged)
            or plan['candidate_registry'] != authority['candidate_registry']
            or stamp(plan['starts_at']) != stamp(slot['starts_at'])
            or stamp(plan['ends_at']) != stamp(slot['cleanup_by'])
            or stamp(plan['activated_at']) < stamp(authority['issued_at'])
            or plan['authority_reference'] != authority['authority_reference']
            or plan.get('performance_evaluation') is not False
            or plan.get('study_enrolment') is not False
            or plan.get('race_ids') is not None):
        raise ValueError('incident_comparison_authority_mismatch')
    expected = Path(authority['state_root']) / 'admission' / plan['incident_slot']
    if (Path(plan['programme_root']) != expected
            or plan['prediction_output_roots'] != [str(Path(authority['prediction_root']) / 'bundles')]):
        raise ValueError('incident_comparison_root_mismatch')
    return authority


def assert_incident_race_allowed(plan, race_id):
    if plan['status'] != STATUS:
        return
    from src.predictor.future_comparison import checked
    authority = validate_incident_plan(plan)
    reference = authority['study_plan']
    study = json.loads(checked(Path(reference['path']), reference['sha256']))
    key = hashlib.sha256(race_id.encode()).hexdigest()
    # A consumed scientific admission, even a failed one, remains scientific.
    claim = Path(study['programme_root']) / reference['sha256'] / 'attempts' / key
    if claim.exists():
        raise ValueError('incident_race_already_study_admitted')


def result_deadline(plan):
    from datetime import timedelta
    from src.predictor.future_comparison import stamp
    if plan['status'] == STATUS:
        return stamp(validate_incident_plan(plan)['result_deadline'])
    return stamp(plan['ends_at']) + timedelta(days=14)
