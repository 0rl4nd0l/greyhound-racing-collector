"""Reviewed all-excluded correction preserves failed native evidence and grants."""
import json
from pathlib import Path

import pytest

from race_collection import persistent_native as native
from race_collection.metadata_exclusion import classify_metadata_exclusion
from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend
from tests.test_empty_eligible_refresh import report_fixture
from tests.fixtures.persistent_operation_case import put


@pytest.fixture
def metadata_recovery(terminal_recovery):
    recovery,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    evidence=Path(old['plan']['evidence_root'])
    service=native.checked(review['service_terminal'])
    # Full refresh file name is selected natively for this synthetic run id.
    report=report_fixture(evidence);report['status']='METADATA_COVERAGE_INCOMPLETE'
    for row in report['sidecar_metadata_coverage']['races']:
        row['native_identity_evidence_status']='verified'
    old_report_path=Path(review['refresh_report']['path'])
    report_ref=put(old_report_path.with_name('refresh_prejump_report.json'),report)
    phase=native.checked(review['phase_result'])
    phase['current_race_index_publish'].update(
        schema_version='collector_current_race_index_publish_v2',
        failure_detail={'reason':'refresh_not_accepted_success'},
        run_id=service['run_id'],
        index_path=str(evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json'),
        source_refresh_report_path=report_ref['path'])
    review['phase_result']=put(Path(review['phase_result']['path']),phase)
    checkpoint=native.checked(review['checkpoint'])
    checkpoint['phases'][0]['result_sha256']=review['phase_result']['sha256']
    review['checkpoint']=put(Path(review['checkpoint']['path']),checkpoint)
    review['refresh_report']=report_ref
    classified=classify_metadata_exclusion(evidence,service['run_id'])
    assert classified is not None
    review.update(schema_version='persistent_reviewed_metadata_exclusion_v1',
        disposition='PROSPECTIVE_ALL_SELECTED_METADATA_EXCLUSION_CORRECTION',
        classified_metadata_exclusion=put(Path(old['output'])/'classified-metadata.json',classified))
    selection=native.checked(cfg['recovery_selection'])
    cleanup=native.checked(selection['cleanup'])
    cleanup['halt']=put(Path(old['output'])/'HALT.json',{'reason':'persistent_native_scope_stopped'})
    selection['cleanup']=put(Path(selection['cleanup']['path']),cleanup)
    review['cleanup']=selection['cleanup']
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    return recovery,review


def test_reviewed_metadata_recovery_reuses_grant_and_preserves_failed_publication(metadata_recovery):
    recovery,review=metadata_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=native.checked(cfg['recovery_selection'])
    paths=[Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state']),
        Path(old['output'])/'HALT.json',Path(selection['prior_stop']['path']),
        *(Path(review[k]['path']) for k in ['service_terminal','service_lifecycle','checkpoint','phase_result','refresh_report'])]
    before={p:p.read_bytes() for p in paths}
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert new['plan']['prediction_root']==old['plan']['prediction_root']
    assert {p:p.read_bytes() for p in paths}==before
    assert not (Path(old['output'])/'day-closed.json').exists()
    assert not list((Path(new['output'])/'refresh-deferrals').glob('*.json'))
    assert native.prepare_day(cfg,standing,'2026-10-03',now)==new
    assert len(calls)==2


@pytest.mark.parametrize('change',['classifier_missing','wrong_classified_count','wrong_classified_hash',
    'new_retry','index_published','denial','budget','unknown_metadata','native_identity',
    'wrong_source','wrong_cleanup','wrong_disposition','wrong_stop','unreaped','changed_terminal'])
def test_metadata_recovery_rejects_unverified_exception(metadata_recovery,change):
    recovery,review=metadata_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=native.checked(cfg['recovery_selection'])
    if change=='classifier_missing':review.pop('classified_metadata_exclusion')
    elif change in ('wrong_classified_count','wrong_classified_hash','new_retry','index_published'):
        value=native.checked(review['classified_metadata_exclusion'])
        field,replacement={'wrong_classified_count':('excluded_count',0),'wrong_classified_hash':('refresh_sha256','0'*64),'new_retry':('request_retries_added',1),'index_published':('current_index_published',True)}[change]
        value[field]=replacement;review['classified_metadata_exclusion']=put(Path(review['classified_metadata_exclusion']['path']),value)
    elif change in ('denial','unknown_metadata','native_identity'):
        value=native.checked(review['refresh_report'])
        if change=='denial':value['downloads'][0]['result']['source_http_status']=429
        elif change=='unknown_metadata':value['sidecar_metadata_coverage']['races'][0]['safe_track_condition_present']=True
        else:value['sidecar_metadata_coverage']['races'][0]['native_identity_evidence_status']='unverified'
        review['refresh_report']=put(Path(review['refresh_report']['path']),value)
    elif change=='budget':
        value=native.checked(review['checkpoint']);value['phases'][0]['budget_exceeded']=True
        review['checkpoint']=put(Path(review['checkpoint']['path']),value)
    elif change=='wrong_source':review['source_commit']='f'*40
    elif change=='wrong_cleanup':review['cleanup']={'path':'/wrong','sha256':'0'*64}
    elif change=='wrong_disposition':review['disposition']='IGNORE_ANY_FAILURE'
    elif change=='wrong_stop':selection['prior_stop']=put(Path(selection['prior_stop']['path']),{'reason':'OTHER_FAILURE'})
    elif change=='unreaped':
        value=native.checked(review['service_lifecycle']);value['children_reaped']=False
        review['service_lifecycle']=put(Path(review['service_lifecycle']['path']),value)
    elif change=='changed_terminal':
        value=native.checked(review['service_terminal']);value['allocation_sha256']='0'*64
        review['service_terminal']=put(Path(review['service_terminal']['path']),value)
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    ledger=Path(cfg['campaign_root'])/'ledger.json';before=ledger.read_bytes()
    with pytest.raises((ValueError,KeyError)):
        native.prepare_day(cfg,standing,'2026-10-03',now)
    assert ledger.read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('status,lane,accepted', [
    ('NEEDS_MORE_AUTOMATION','full',True),
    ('NEEDS_MORE_AUTOMATION','odds',False),
    ('UNKNOWN','full',False),
])
def test_exact_full_lane_metadata_terminal_status(metadata_recovery,status,lane,accepted):
    recovery,review=metadata_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=native.checked(cfg['recovery_selection'])
    service=native.checked(review['service_terminal']);service['status']=status
    review['service_terminal']=put(Path(review['service_terminal']['path']),service)
    state=native.checked(selection['baseline']['prior_owner_state'])
    state['dispatches'][0]['lane']=lane
    selection['baseline']['prior_owner_state']=put(Path(selection['baseline']['prior_owner_state']['path']),state)
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    halt=json.loads((Path(old['output'])/'HALT.json').read_bytes());stop=native.checked(selection['prior_stop'])
    if accepted:
        assert native._reviewed_refresh_failure(selection,old,halt,stop) is None
    else:
        with pytest.raises(ValueError,match='terminal_unverified'):
            native._reviewed_refresh_failure(selection,old,halt,stop)
