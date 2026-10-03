"""Reuse frozen four-model admission under separate standing engineering scope."""
import json
from pathlib import Path

from race_collection.persistent_authority import load_persistent_allocation, stamp


def validate_persistent_plan(plan):
    from src.predictor.future_comparison import checked
    allocation = load_persistent_allocation(plan['persistent_allocation'])
    reference = allocation['study_plan']
    study = json.loads(checked(Path(reference['path']), reference['sha256']))
    unchanged = ('schema_version', 'candidate_registry', 'decision_seconds_before_jump',
                 'quote_lead_seconds', 'denied_history_intervals', 'history_policy',
                 'machine_history_authority_reference', 'fixed_closure_days', 'missing_result_policy')
    if (plan.get('status') != 'AUTHORIZED_ENGINEERING'
            or study.get('status') != 'AUTHORIZED'
            or any(plan.get(k) != study.get(k) for k in unchanged)
            or plan['candidate_registry'] != allocation['candidate_registry']
            or stamp(plan['starts_at']) != stamp(allocation['starts_at'])
            or stamp(plan['ends_at']) != stamp(allocation['cleanup_by'])
            or stamp(plan['activated_at']) < stamp(allocation['issued_at'])
            or plan['authority_reference'] != allocation['authority_reference']
            or plan.get('performance_evaluation') is not False
            or plan.get('study_enrolment') is not False
            or plan.get('race_ids') is not None
            or any(plan.get(k) is not None for k in ('incident_authority', 'incident_slot', 'development_authority'))
            or Path(plan['programme_root']) != Path(allocation['state_root']) / 'admission'
            or plan['prediction_output_roots'] != [str(Path(allocation['prediction_root']) / 'bundles')]):
        raise ValueError('persistent_comparison_authority_mismatch')
    # This is a disclosure of absent result authority, never permission to fetch.
    # Existing historical result workers keep their independently bound deadline.
    return {**allocation, 'result_deadline': allocation['ends_at'], 'result_access': False}
