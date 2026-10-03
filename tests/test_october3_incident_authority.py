"""Fresh October 3 receipt without reviving either October 1 or October 2."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pytest

from race_collection.incident_engineering import load_incident_authority, incident_usage, validate_incident_lease
from tests.fixtures.incident_engineering_case import make_incident
from tests.test_incident_engineering import october2_authority, put


@pytest.fixture
def october3(tmp_path):
    value, ref = october2_authority(make_incident(tmp_path))
    value.update(schema_version='collector_incident_engineering_authority_20261003_v1',
        authority_reference='user:20261003-recovery-continuation',
        incident_id='invented-october3-recovery', issued_at='2026-10-03T12:40:00+10:00',
        collection_stop_at='2026-10-03T21:00:00+10:00', cleanup_deadline='2026-10-03T21:30:00+10:00',
        local_request_caps_are_provider_permission=False,
        slots=[dict(id='001', starts_at='2026-10-03T13:00:00+10:00',
                    ends_at='2026-10-03T14:30:00+10:00', cleanup_by='2026-10-03T15:01:00+10:00')])
    return value, put(Path(ref['path']), value)


def test_fresh_single_window_does_not_require_expired_day_amendments(october3):
    value, ref = october3
    assert 'live_first_amendment' not in value
    assert 'late_window_amendment' not in value
    assert load_incident_authority(ref) == value


@pytest.mark.parametrize('defect', ['old_approval', 'old_issued_day', 'old_slot_day', 'wrong_slot',
    'second_slot', 'short_session', 'late_collection', 'late_cleanup', 'late_results',
    'wrong_result_offset', 'python_cap', 'browser_cap', 'source_cap', 'result_cap',
    'study_membership', 'provider_permission', 'issued_after_start', 'october2_schema'])
def test_october3_cannot_extend_scope_or_reuse_prior_day(october3, defect):
    value, ref = october3; slot=value['slots'][0]
    if defect=='old_approval':value['authority_reference']='user:20261002-live-first-resume-03'
    elif defect=='old_issued_day':value['issued_at']='2026-10-02T12:40:00+10:00'
    elif defect=='old_slot_day':slot.update(starts_at='2026-10-02T13:00:00+10:00',ends_at='2026-10-02T14:30:00+10:00',cleanup_by='2026-10-02T15:01:00+10:00')
    elif defect=='wrong_slot':slot['id']='002'
    elif defect=='second_slot':value['slots'].append(deepcopy(slot))
    elif defect=='short_session':slot['ends_at']='2026-10-03T14:00:00+10:00'
    elif defect=='late_collection':value['collection_stop_at']='2026-10-03T21:01:00+10:00'
    elif defect=='late_cleanup':value['cleanup_deadline']='2026-10-03T21:31:00+10:00'
    elif defect=='late_results':value['result_deadline']='2026-10-04T12:00:01+11:00'
    elif defect=='wrong_result_offset':value['result_deadline']='2026-10-04T12:00:00+10:00'
    elif defect=='python_cap':value['max_python_requests_per_window']+=1
    elif defect=='browser_cap':value['max_browser_navigations_per_window']+=1
    elif defect=='source_cap':value['max_source_operations_per_window']+=1
    elif defect=='result_cap':value['max_result_requests_per_window']+=1
    elif defect=='study_membership':value['study_enrolment']=True
    elif defect=='provider_permission':value['local_request_caps_are_provider_permission']=True
    elif defect=='issued_after_start':value['issued_at']=slot['starts_at']
    else:value['schema_version']='collector_incident_engineering_authority_20261002_v1'
    with pytest.raises(ValueError,match='invalid_incident_authority'):
        load_incident_authority(put(Path(ref['path']),value))


def test_october3_leases_have_separate_collection_and_result_expiry(october3):
    value, ref=october3; start=datetime.fromisoformat(value['slots'][0]['starts_at']).timestamp()
    row=dict(incident_authority=ref,incident_authority_sha256=ref['sha256'],incident_id=value['incident_id'],
        incident_slot='001',incident_kind='prediction',prior_phase='OPEN',authorized_at=start-600,
        expires_at=start+5400,max_operations=192,reference=value['authority_reference']+':slot:001')
    assert validate_incident_lease(row)==value
    row['expires_at']+=1
    with pytest.raises(ValueError):validate_incident_lease(row)
    row.update(incident_kind='results',reference=value['authority_reference']+':slot:001:results',
        max_operations=96,expires_at=datetime.fromisoformat(value['result_deadline']).timestamp())
    assert validate_incident_lease(row)==value


def test_october3_uses_new_counter_bucket_preserving_all_historical_charges(october3,tmp_path):
    new,newref=october3
    old1,ref1=make_incident(tmp_path/'oct1')
    old2,ref2=october2_authority(make_incident(tmp_path/'oct2'))
    ledger={'incident_request_usage':{}}
    for value,ref in [(old1,ref1),(old2,ref2),(new,newref)]:
        ledger['incident_request_usage'][ref['sha256']+':001']=dict(incident_authority=ref,
            incident_authority_sha256=ref['sha256'],incident_id=value['incident_id'],incident_slot='001',
            counts={'prediction':192,'results':1})
    before=deepcopy(ledger)
    assert incident_usage(ledger,'SYNTHETIC')['logical_requests']==579
    assert incident_usage(ledger,'SYNTHETIC',authority_sha256=newref['sha256'])['logical_requests']==193
    assert ledger==before
