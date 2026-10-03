"""Invented standing daily operation authority. Never touches retained runtime."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def make_persistent(root):
    root = Path(root).resolve()
    registry = put(root/'registry.json', {'models': ['frozen-one', 'frozen-two', 'frozen-three', 'frozen-four']})
    study = put(root/'study.json', {'status': 'AUTHORIZED', 'candidate_registry': registry,
        'programme_root': str(root/'science'), 'prediction_output_roots': [str(root/'science-predictions')]})
    caps = dict(max_python_requests=3, max_browser_navigations=2, max_capture_attempts=2,
                max_source_operations=400, max_result_requests=0)
    standing = dict(schema_version='collector_persistent_operation_authority_v1',
        status='AUTHORIZED_PERSISTENT_ENGINEERING', operation_id='SYNTHETIC_DAILY', campaign_id='SYNTHETIC',
        authority_reference='user:synthetic-standing-operation', issued_at='2026-10-03T08:00:00+10:00',
        starts_at='2026-10-03T09:00:00+10:00', timezone='Australia/Melbourne',
        state_root=str(root/'runtime'), prediction_root=str(root/'predictions'),
        protected_roots=[str(root/'science'), str(root/'weekend')], study_plan=study,
        candidate_registry=registry, daily_caps=caps, engineering_only=True, study_enrolment=False,
        performance_evaluation=False, human_outcome_access=False, local_request_caps_are_provider_permission=False)
    standing_ref = put(root/'standing.json', standing)
    allocation = dict(schema_version='collector_persistent_daily_allocation_v1',
        status='AUTHORIZED_PERSISTENT_DAILY_ENGINEERING', standing_authority=standing_ref,
        allocation_id=standing_ref['sha256']+':2026-10-03', racing_date='2026-10-03',
        issued_at='2026-10-03T08:30:00+10:00', starts_at='2026-10-03T09:00:00+10:00',
        ends_at='2026-10-04T01:00:00+10:00', cleanup_by='2026-10-04T01:30:00+10:00',
        state_root=str(root/'runtime/days/2026-10-03'), prediction_root=str(root/'predictions/days/2026-10-03'),
        caps=caps.copy())
    ref = put(root/'allocation.json', allocation)
    return standing, standing_ref, allocation, ref
