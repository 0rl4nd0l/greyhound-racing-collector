"""A retention repair admits only future work, never the old failed index."""
import json
from pathlib import Path

import pytest
from race_collection import persistent_native as native
from tests.fixtures.persistent_operation_case import put
from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend


@pytest.fixture
def identity_recovery(terminal_recovery):
    case, review = terminal_recovery
    cfg, standing, old, now, item, terminal, calls = case
    upcoming = Path(review['refresh_report']['path']).parent/'odds_capture_refreshed_upcoming'
    url = 'https://www.thedogs.com.au/racing/example/2026-10-03/1/race?trial=false'
    race = dict(race_id='Race 1 - EX - 2026-10-03', race_url=url, jump='2026-10-03T12:04:00+10:00')
    alignment = dict(native_identity_status='unavailable', native_identity_reasons=[
        'native_identity_evidence_rejected:expected_native_runner_set_mismatch'])
    report = dict(status='ACQUISITION_INCOMPLETE', reason='unisolated_selected_race_acquisition_failure',
        upcoming_dir=str(upcoming), selected_count=1,
        selected_races=[dict(race_id=race['race_id'], race_url=url, jump_datetime=race['jump'])],
        downloads=[dict(race_url=url,success=False,result=dict(success=False,
            error='Downloaded CSV failed canonical final runner-set alignment gate',normalization=dict(
                normalization_failure_reason='final_runner_set_not_aligned:canonical_participant_missing_from_source_csv',
                canonical_runner_alignment=alignment)))])
    review['refresh_report'] = put(Path(review['refresh_report']['path']), report)
    service = native.checked(review['service_terminal'])
    phase = native.checked(review['phase_result'])
    phase['current_race_index_publish'].update(schema_version='collector_current_race_index_publish_v2',
        failure_detail={'reason':'refresh_not_accepted_success'},run_id=service['run_id'],
        index_path=str(Path(old['plan']['evidence_root'])/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json'))
    review['phase_result']=put(Path(review['phase_result']['path']),phase)
    cp=native.checked(review['checkpoint']);cp['phases'][0]['result_sha256']=review['phase_result']['sha256']
    review['checkpoint']=put(Path(review['checkpoint']['path']),cp)
    timing=upcoming/'workers/0/refresh-request-timing.jsonl';timing.parent.mkdir(parents=True)
    timing.write_text('\n'.join(json.dumps(dict(schema_version='refresh_request_timing_v1',kind='request',
        event=event,id=1,pid=100,method='GET',endpoint=url.split('?')[0]+'/odds',
        utc='2026-10-03T02:03:00+00:00',**extra)) for event,extra in [('start',{}),('end',{'status_code':200})]))
    diagnosis=dict(schema_version='native_identity_failure_diagnosis_v1',
        status='CLASSIFIED_RETENTION_GAP_NO_SAFE_RECLASSIFICATION',source_commit='f'*40,
        invocation_id=service['invocation_id'],dispatch_returncode=2,
        refresh_status=report['status'],refresh_reason=report['reason'],
        refs={'refresh_report':review['refresh_report'],'request_timing':native._ref(timing)},
        failed_candidates=[dict(race_id=race['race_id'],jump=race['jump'],**alignment)],
        transport_evidence=dict(all_completed_status200=True,odds_html_requested=True,odds_api_requested=False))
    review.update(schema_version='persistent_reviewed_native_identity_retention_gap_v1',
        disposition='PROSPECTIVE_NATIVE_IDENTITY_RETENTION_CORRECTION',
        diagnosis=put(Path(old['output'])/'diagnosis.json',diagnosis))
    proof=dict(schema_version='persistent_native_identity_retention_gap_proof_v1',
        source_commit=cfg['source_commit'],original_source_commit='f'*40,diagnosis=review['diagnosis'],
        invocation_id=service['invocation_id'],refresh_report=review['refresh_report'],failed_phase_number=0,
        affected_races=[race],worker_request_timings=[dict(race_url=url,request_timing=native._ref(timing))],
        missing_odds_body=True,mismatch_worker_odds_api_requested=False,current_index_published=False,
        request_retries_added=0,old_failure_disposition='FAILED_UNRESOLVED')
    review['retention_gap_proof']=put(Path(old['output'])/'retention-proof.json',proof)
    update_review(cfg,review)
    return case,review


def update_review(cfg,review):
    selection=native.checked(cfg['recovery_selection'])
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)


def test_reviewed_identity_retention_recovery_preserves_failure_and_consumption(identity_recovery):
    case,review=identity_recovery
    cfg,standing,old,now,item,terminal,calls=case
    paths=[Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state']),Path(old['output'])/'HALT.json',
        *(Path(review[k]['path']) for k in ('service_terminal','service_lifecycle','checkpoint','phase_result','refresh_report'))]
    before={p:p.read_bytes() for p in paths}
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert before=={p:p.read_bytes() for p in paths}
    assert not list((Path(new['output'])/'refresh-deferrals').glob('*.json'))
    assert len(calls)==2


@pytest.mark.parametrize('defect',['proof_tamper','diagnosis_tamper','denial','api_request','shared_failure',
    'future_jump','failed_publication','wrong_source','body_present','missing_worker','wrong_race','unreaped'])
def test_identity_recovery_rejects_incomplete_or_changed_proof(identity_recovery,defect):
    case,review=identity_recovery
    cfg,standing,old,now,item,terminal,calls=case
    proof=native.checked(review['retention_gap_proof'])
    if defect in ('proof_tamper','diagnosis_tamper'):
        key='retention_gap_proof' if defect=='proof_tamper' else 'diagnosis'
        p=Path(review[key]['path']);p.write_bytes(p.read_bytes()+b' ')
    elif defect in ('denial','api_request'):
        ref=proof['worker_request_timings'][0]['request_timing'];p=Path(ref['path'])
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        if defect=='denial':rows[-1]['status_code']=403
        else:
            for row in rows:row['endpoint']='https://www.thedogs.com.au/api/odds'
        p.write_text('\n'.join(map(json.dumps,rows)))
        proof['worker_request_timings'][0]['request_timing']=native._ref(p)
    elif defect in ('shared_failure','body_present','future_jump'):
        r=native._checked_recovery_refresh_report(review['refresh_report'])
        if defect=='shared_failure':r['downloads'][0]['result']['error']='shared_transport_failure'
        elif defect=='body_present':r['downloads'][0]['result']['normalization']['native_identity_evidence']={'odds_page_http':{}}
        else:r['selected_races'][0]['jump_datetime']='2026-10-03T12:06:00+10:00'
        review['refresh_report']=put(Path(review['refresh_report']['path']),r)
        proof['refresh_report']=review['refresh_report']
        diagnosis=native.checked(review['diagnosis']);diagnosis['refs']['refresh_report']=review['refresh_report']
        if defect=='future_jump':
            proof['affected_races'][0]['jump']=r['selected_races'][0]['jump_datetime']
            diagnosis['failed_candidates'][0]['jump']=r['selected_races'][0]['jump_datetime']
        review['diagnosis']=put(Path(review['diagnosis']['path']),diagnosis);proof['diagnosis']=review['diagnosis']
    elif defect=='failed_publication':
        phase=native.checked(review['phase_result']);phase['current_race_index_publish']['status']='PUBLISHED'
        review['phase_result']=put(Path(review['phase_result']['path']),phase)
        cp=native.checked(review['checkpoint']);cp['phases'][0]['result_sha256']=review['phase_result']['sha256']
        review['checkpoint']=put(Path(review['checkpoint']['path']),cp)
    elif defect=='wrong_source':proof['source_commit']='a'*40
    elif defect=='missing_worker':proof['worker_request_timings']=[]
    elif defect=='wrong_race':proof['affected_races'][0]['race_id']='foreign'
    else:
        life=native.checked(review['service_lifecycle']);life['children_reaped']=False
        review['service_lifecycle']=put(Path(review['service_lifecycle']['path']),life)
    if defect not in ('proof_tamper','diagnosis_tamper'):
        review['retention_gap_proof']=put(Path(review['retention_gap_proof']['path']),proof)
    update_review(cfg,review)
    ledger=Path(cfg['campaign_root'])/'ledger.json';before=ledger.read_bytes()
    with pytest.raises((ValueError,KeyError)):
        native.prepare_day(cfg,standing,'2026-10-03',now)
    assert ledger.read_bytes()==before and len(calls)==1


@pytest.mark.parametrize('size,tamper,accepted',[(306255,False,True),(306255,True,False),(4*1024*1024+1,False,False)])
def test_daily_owner_evidence_has_separate_finite_bound(identity_recovery,size,tamper,accepted):
    case,review=identity_recovery
    cfg,standing,old,now,item,terminal,calls=case
    selection=native.checked(cfg['recovery_selection'])
    ref=selection['baseline']['prior_owner_state'];state=native.checked(ref)
    state['padding']='x'*size
    selection['baseline']['prior_owner_state']=put(Path(ref['path']),state)
    if tamper:Path(ref['path']).write_bytes(Path(ref['path']).read_bytes()+b' ')
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    if accepted:
        assert native.prepare_day(cfg,standing,'2026-10-03',now)['allocation_ref']==old['allocation_ref']
    else:
        with pytest.raises(ValueError,match='persistent_recovery_owner_'):
            native.prepare_day(cfg,standing,'2026-10-03',now)
        assert len(calls)==1
