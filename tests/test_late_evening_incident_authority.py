"""The new evening instruction cannot revive or enlarge an earlier run."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest
from race_collection.incident_engineering import load_incident_authority, validate_incident_lease
from tests.test_incident_engineering import put
from tests.test_single_window_incident_authority import single


@pytest.fixture
def late(single):
    value, ref = single
    amendment = Path(__file__).resolve().parents[1] / 'docs/operations/late_validation_authority_20261002.json'
    import hashlib
    value.update(schema_version='collector_incident_engineering_authority_20261002_late_v1',
                 authority_reference='user:20261002-remaining-evening-races',
                 issued_at='2026-10-02T21:55:00+10:00',
                 late_window_amendment={'path':str(amendment),'sha256':hashlib.sha256(amendment.read_bytes()).hexdigest()},
                 collection_stop_at='2026-10-02T23:40:00+10:00',
                 cleanup_deadline='2026-10-03T00:10:00+10:00',
                 slots=[{'id':'001','starts_at':'2026-10-02T22:00:00+10:00',
                         'ends_at':'2026-10-02T23:30:00+10:00','cleanup_by':'2026-10-03T00:00:00+10:00'}])
    return value, put(Path(ref['path']),value)


def test_new_evening_window_and_deadline_bound_lease(late):
    value, ref = late
    assert load_incident_authority(ref) == value
    start=datetime.fromisoformat(value['slots'][0]['starts_at']).timestamp()
    row=dict(incident_authority=ref,incident_authority_sha256=ref['sha256'],
             incident_id=value['incident_id'],incident_slot='001',incident_kind='prediction',
             reference=value['authority_reference']+':slot:001',prior_phase='OPEN',
             authorized_at=start-60,expires_at=start+5400,max_operations=192)
    assert validate_incident_lease(row)==value
    row['expires_at']+=1
    with pytest.raises(ValueError):validate_incident_lease(row)


@pytest.mark.parametrize('defect',['missing_amendment','changed_amendment','old_profile',
    'old_reference','early_issue','late_stop','late_cleanup','two_slots','more_requests','study'])
def test_late_window_cannot_expand_or_reinterpret_prior_authority(late,defect):
    value, ref=late
    if defect=='missing_amendment':del value['late_window_amendment']
    elif defect=='changed_amendment':value['late_window_amendment']['sha256']='0'*64
    elif defect=='old_profile':value['schema_version']='collector_incident_engineering_authority_20261002_v2'
    elif defect=='old_reference':value['authority_reference']='previous-user-scope'
    elif defect=='early_issue':value['issued_at']='2026-10-02T21:00:00+10:00'
    elif defect=='late_stop':value['collection_stop_at']='2026-10-03T00:00:00+10:00'
    elif defect=='late_cleanup':value['cleanup_deadline']='2026-10-03T00:30:00+10:00'
    elif defect=='two_slots':value['slots'].append(deepcopy(value['slots'][0]))
    elif defect=='more_requests':value['max_python_requests_per_window']+=1
    else:value['study_enrolment']=True
    with pytest.raises(ValueError):load_incident_authority(put(Path(ref['path']),value))


def test_fresh_correction_keeps_prior_authority_valid(late):
    import json, hashlib
    value, ref = late
    assert load_incident_authority(ref) == value
    amendment = Path(__file__).resolve().parents[1] / 'docs/operations/late_validation_preparation_correction_20261002.json'
    renewed = json.loads(amendment.read_bytes())
    value = deepcopy(value)
    value.update(authority_reference=renewed['authority_reference'], issued_at='2026-10-02T22:19:00+10:00',
        collection_stop_at=renewed['collection_stop_at'], cleanup_deadline=renewed['cleanup_deadline'],
        late_window_amendment={'path':str(amendment),'sha256':hashlib.sha256(amendment.read_bytes()).hexdigest()},
        slots=[{'id':'001','starts_at':'2026-10-02T22:20:00+10:00','ends_at':'2026-10-02T23:50:00+10:00','cleanup_by':'2026-10-03T00:21:00+10:00'}])
    new_ref = put(Path(ref['path']).with_name('corrected.json'),value)
    assert load_incident_authority(new_ref) == value
    assert load_incident_authority(ref)['authority_reference'] != value['authority_reference']
