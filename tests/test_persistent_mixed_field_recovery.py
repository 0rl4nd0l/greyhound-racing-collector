"""Explicit mixed-field recovery preserves the failed publication and allocation."""
from pathlib import Path
import pytest
from race_collection import persistent_native as native
from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend
from tests.test_mixed_completed_field_exclusions import mixed_case
from tests.fixtures.persistent_operation_case import put

@pytest.fixture
def mixed_recovery(terminal_recovery, tmp_path):
    case, review = terminal_recovery
    cfg, standing, old, now, item, terminal, calls = case
    report, worker, page = mixed_case(tmp_path/'mixed')
    report.update(status='ACQUISITION_INCOMPLETE', reason='unisolated_selected_race_acquisition_failure')
    review['refresh_report'] = put(Path(review['refresh_report']['path']), report)
    service = native.checked(review['service_terminal'])
    evidence = Path(old['plan']['evidence_root'])
    phase = native.checked(review['phase_result'])
    phase['current_race_index_publish'].update(schema_version='collector_current_race_index_publish_v2',
        failure_detail={'reason':'refresh_not_accepted_success'}, run_id=service['run_id'],
        index_path=str(evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json'))
    review['phase_result'] = put(Path(review['phase_result']['path']), phase)
    checkpoint = native.checked(review['checkpoint'])
    checkpoint['phases'][0]['result_sha256'] = review['phase_result']['sha256']
    review['checkpoint'] = put(Path(review['checkpoint']['path']), checkpoint)
    proof = dict(schema_version='verified_mixed_field_exclusions_v1', disposition='VERIFIED_MIXED_FIELD_EXCLUSIONS',
        run_id=service['run_id'], failed_phase_number=0, phase_result_sha256=review['phase_result']['sha256'],
        refresh_sha256=review['refresh_report']['sha256'], selected_count=2, eligible_count=1, excluded_count=1,
        request_retries_added=0, current_index_published=False)
    review.update(schema_version='persistent_reviewed_mixed_field_exclusion_v1',
        disposition='PROSPECTIVE_MIXED_FIELD_EXCLUSION_CORRECTION',
        classified_mixed_field_exclusion=put(Path(old['output'])/'mixed-proof.json',proof))
    selection = native.checked(cfg['recovery_selection'])
    selection['reviewed_failure'] = put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection'] = put(Path(cfg['recovery_selection']['path']),selection)
    return case, review


def test_mixed_recovery_preserves_old_failure_allocation_and_counters(mixed_recovery):
    case, review = mixed_recovery
    cfg, standing, old, now, item, terminal, calls = case
    selection = native.checked(cfg['recovery_selection'])
    paths = [Path(cfg['campaign_root'])/'ledger.json', Path(cfg['source_state']), Path(old['output'])/'HALT.json',
        Path(selection['prior_stop']['path']), *(Path(review[k]['path']) for k in
        ('service_terminal','service_lifecycle','checkpoint','phase_result','refresh_report'))]
    before = {p:p.read_bytes() for p in paths}
    new = native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref'] == old['allocation_ref']
    assert new['plan']['prediction_root'] == old['plan']['prediction_root']
    assert before == {p:p.read_bytes() for p in paths}
    assert not (Path(old['output'])/'day-closed.json').exists()
    assert not list((Path(new['output'])/'refresh-deferrals').glob('*.json'))
    assert native.prepare_day(cfg,standing,'2026-10-03',now) == new
    assert len(calls) == 2


@pytest.mark.parametrize('defect', ['missing_proof','counts','retry','published','report_success','denial',
    'unknown','missing_candidate','accepted_csv','excluded_page','phase_success','publication_detail',
    'publication_path','unreaped','wrong_allocation','wrong_cleanup','wrong_source','wrong_stop','budget'])
def test_mixed_recovery_requires_entire_original_failure_proof(mixed_recovery,defect):
    case, review = mixed_recovery
    cfg, standing, old, now, item, terminal, calls = case
    selection = native.checked(cfg['recovery_selection'])
    if defect=='missing_proof': review.pop('classified_mixed_field_exclusion')
    elif defect in ('counts','retry','published'):
        proof=native.checked(review['classified_mixed_field_exclusion'])
        k,v={'counts':('excluded_count',0),'retry':('request_retries_added',1),'published':('current_index_published',True)}[defect]
        proof[k]=v;review['classified_mixed_field_exclusion']=put(Path(review['classified_mixed_field_exclusion']['path']),proof)
    elif defect in ('report_success','denial','unknown','missing_candidate','accepted_csv','excluded_page'):
        report=native.checked(review['refresh_report'])
        if defect=='report_success':report['status']='SUCCESS'
        elif defect=='denial':report['downloads'][0]['result']['source_http_status']=403
        elif defect=='unknown':report['downloads'][0]['result']['error']='unknown'
        elif defect=='missing_candidate':report['downloads'].pop()
        elif defect=='accepted_csv':Path(report['sidecar_metadata_coverage']['races'][1]['csv_path']).write_text('changed')
        else:
            pages=list(Path(report['upcoming_dir']).rglob('*.race-page.receipt.json'))
            assert pages;pages[0].chmod(0o600);pages[0].write_text('{}')
        review['refresh_report']=put(Path(review['refresh_report']['path']),report)
    elif defect in ('phase_success','publication_detail','publication_path'):
        phase=native.checked(review['phase_result'])
        if defect=='phase_success':phase['status']='PASS'
        elif defect=='publication_detail':phase['current_race_index_publish']['failure_detail']={'reason':'unknown'}
        else:phase['current_race_index_publish']['index_path']='/foreign'
        review['phase_result']=put(Path(review['phase_result']['path']),phase)
        cp=native.checked(review['checkpoint']);cp['phases'][0]['result_sha256']=review['phase_result']['sha256']
        review['checkpoint']=put(Path(review['checkpoint']['path']),cp)
    elif defect=='unreaped':
        doc=native.checked(review['service_lifecycle']);doc['children_reaped']=False
        review['service_lifecycle']=put(Path(review['service_lifecycle']['path']),doc)
    elif defect=='wrong_allocation':
        doc=native.checked(review['service_terminal']);doc['allocation_sha256']='0'*64
        review['service_terminal']=put(Path(review['service_terminal']['path']),doc)
    elif defect=='wrong_cleanup':review['cleanup']={'path':'/wrong','sha256':'0'*64}
    elif defect=='wrong_source':review['source_commit']='0'*40
    elif defect=='wrong_stop':selection['prior_stop']=put(Path(selection['prior_stop']['path']),{'reason':'OTHER'})
    else:
        doc=native.checked(review['checkpoint']);doc['phases'][0]['budget_exceeded']=True
        review['checkpoint']=put(Path(review['checkpoint']['path']),doc)
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    ledger=Path(cfg['campaign_root'])/'ledger.json';before=ledger.read_bytes()
    with pytest.raises((ValueError,KeyError)):
        native.prepare_day(cfg,standing,'2026-10-03',now)
    assert ledger.read_bytes()==before and len(calls)==1
