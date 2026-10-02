"""Fresh machine-only processing of original engineering result evidence.

The old plan, authority, queue binding and closure remain unchanged. This path
has no transport or admission operation; network capacity requires a separately
installed allocation and is never inferred from a future ceiling in a receipt.
"""
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from src.predictor.future_comparison import checked, stamp


def reference_json(reference):
    path = Path(reference['path'])
    if not path.is_absolute() or path.resolve() != path:
        raise ValueError('renewal_reference_not_exact')
    return json.loads(checked(path, reference['sha256']))


def load_renewal(reference, approval, *, now):
    value = reference_json(reference)
    issued, expires = stamp(value['issued_at']), stamp(value['expires_at'])
    melbourne = ZoneInfo('Australia/Melbourne')
    if (value.get('schema_version') != 'engineering_retained_result_renewal_v1'
            or value.get('status') != 'AUTHORIZED_RENEWED_RETAINED_PROCESSING'
            or not approval or value.get('authority_reference') != approval or not value.get('owner')
            or issued.astimezone(melbourne).date().isoformat() != '2026-10-02'
            or not issued <= now < expires <= stamp('2026-10-04T12:00:00+11:00')
            or value.get('network_requests_allowed') is not False
            or type(value.get('max_additional_requests')) is not int or value['max_additional_requests'] != 0
            or value.get('human_outcome_access') is not False
            or value.get('performance_evaluation') is not False or value.get('study_enrolment') is not False
            or value.get('preserve_attempts') is not True):
        raise ValueError('renewed_result_scope_not_authorized')
    binding = reference_json(value['original_binding'])
    from src.predictor.comparison_result_runtime import load_runtime
    # This existing mode checks expired metadata for closure; it does not read
    # results or grant decoding. Only the NEW receipt above authorizes decoding.
    plan, original_authority, cfg = load_runtime(binding, now=now, allow_closure=True)
    if (plan['status'] != 'AUTHORIZED_ENGINEERING'
            or value['authority_reference'] == original_authority['authority_reference']
            or issued <= stamp(original_authority['issued_at'])):
        raise ValueError('renewal_requires_distinct_engineering_authority')
    rows = value['jobs']
    if not isinstance(rows, list) or not 1 <= len(rows) <= 20:
        raise ValueError('renewal_population_invalid')
    seen_jobs, seen_races = set(), set()
    for row in rows:
        if row['job_id'] in seen_jobs or row['race_id'] in seen_races:
            raise ValueError('renewal_population_duplicate')
        seen_jobs.add(row['job_id']); seen_races.add(row['race_id'])
        key = hashlib.sha256(row['race_id'].encode()).hexdigest()
        claim = Path(plan['programme_root']) / binding['plan_sha256'] / 'attempts' / key
        if (Path(row['admission']['path']) != claim/'admission.json'
                or Path(row['completion']['path']) != claim/'completion.json'):
            raise ValueError('renewal_original_membership_path_changed')
        admission = reference_json(row['admission'])
        reference_json(row['completion'])
        if (admission['job_id'] != row['job_id'] or admission['race']['race_id'] != row['race_id']):
            raise ValueError('renewal_original_membership_changed')
    return value, binding, plan, original_authority, cfg
