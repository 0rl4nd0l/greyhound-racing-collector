"""Known local identity rejection continues cadence, never admits the race."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import pytest

from scripts import refresh_prejump_upcoming as refresh
from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence
from tests.test_persistent_metadata_exclusion import fixture, classify
from tests.test_empty_eligible_refresh import publish
from tests.test_runner_completeness import _race_page, _runner_row
from utils.runner_completeness import extract_canonical_runner_set_from_html


def local_identity_case(tmp_path):
    output,plan,rid,root,path,report=fixture(tmp_path)
    row=report['sidecar_metadata_coverage']['races'][0]
    candidate=report['selected_races'][0]
    csv=Path(row['csv_path']);sidecar=Path(row['sidecar_path'])
    observed=datetime.fromisoformat(report['generated_at'])
    body=_race_page(_runner_row(1,'Alpha',runner_id='159001'),
                    _runner_row(2,'Beta',runner_id='159002')).replace('>R4<','>R5<').encode()
    canonical=extract_canonical_runner_set_from_html(body.decode(),source_url=candidate['race_url'],
        expected_race_number=5,extraction_timestamp=observed.isoformat())
    page=persist_primary_race_page_evidence(artifact_root=csv.parent,race_discovery_key=candidate['race_id'],
        response=TimedResponse(requested_url=candidate['race_url'],final_url=candidate['race_url'],
            request_start_utc=observed-timedelta(seconds=1),request_end_utc=observed,status_code=200,
            headers={'content-type':'text/html'},body=body),canonical_runner_set=canonical)
    alignment=dict(status='aligned',canonical_runner_set_status='available',native_identity_status='unavailable',
        source_native_race_id=None,native_identity_reasons=['native_identity_evidence_rejected:scratched_runner_has_active_price'],
        canonical_source_url=candidate['race_url'],canonical_runner_count=2,prediction_runner_count=2)
    meta=json.loads(sidecar.read_bytes())
    meta.update(canonical_runner_alignment=alignment,content_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        content_length=csv.stat().st_size,primary_race_page_evidence=page,race_url=candidate['race_url'])
    sidecar.write_text(json.dumps(meta))
    row.update(native_identity_evidence_status='not_required_direct_source_identity',native_identity_evidence_reason=None,
        source_native_race_id=None,source_native_runner_ids=['159001','159002'],safe_expert_form_present=True,
        safe_track_condition_present=True,safe_all_weather_track_expert_form_present=True,expert_form_rejected_reasons=[])
    report['upcoming_dir']=str(csv.parent)
    report['downloads'][0]['result'].update(filepath=str(csv),normalization={
        'normalization_status':'verified','canonical_runner_alignment':alignment})
    _,report['current_index_metadata_selection']=refresh.current_index_metadata_selection(
        [candidate],report['sidecar_metadata_coverage'],source_generated_at=report['generated_at'])
    path.write_text(json.dumps(report))
    return plan,rid,path,report,sidecar


def test_exact_local_identity_rejection_remains_unpublished_and_rejected(tmp_path):
    plan,rid,path,report,sidecar=local_identity_case(tmp_path)
    original=path.read_bytes()
    result=classify(plan['evidence_root'],rid)
    assert result and result['disposition']=='COMPLETED_METADATA_EXCLUSIONS'
    assert result['selected_count']==result['excluded_count']==1
    assert result['current_index_published'] is False
    assert not refresh.complete_empty_metadata_selection(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'local-native')['status']=='REJECTED'
    assert path.read_bytes()==original
    assert report['current_index_metadata_selection']['exclusions'][0]['missing_safe_metadata']==['native_source_identity']


@pytest.mark.parametrize('defect',['unknown','denial','budget','partial','changed_csv','changed_sidecar',
    'changed_page','changed_receipt','wrong_race','native_present','unknown_runner','late_page','symlink','unsafe_path'])
def test_unknown_shared_or_unproven_identity_rejection_stays_terminal(tmp_path,defect):
    plan,rid,path,report,sidecar=local_identity_case(tmp_path)
    row=report['sidecar_metadata_coverage']['races'][0]
    meta=json.loads(sidecar.read_bytes())
    normal=report['downloads'][0]['result']['normalization']
    if defect=='unknown':normal['canonical_runner_alignment']['native_identity_reasons']=['unknown']
    elif defect=='denial':report['downloads'][0]['result']['source_http_status']=429
    elif defect=='budget':report['status']='REFRESH_BUDGET_EXCEEDED'
    elif defect=='partial':report['downloads'][0]['success']=False
    elif defect=='changed_csv':Path(row['csv_path']).write_bytes(b'changed')
    elif defect=='changed_sidecar':meta['content_sha256']='0'*64
    elif defect=='changed_page':
        page=sidecar.parent/meta['primary_race_page_evidence']['raw_path'];page.chmod(0o600);page.write_bytes(b'changed')
    elif defect=='changed_receipt':meta['primary_race_page_evidence']['receipt_sha256']='0'*64
    elif defect=='wrong_race':report['selected_races'][0]['race_number']=6
    elif defect=='native_present':row['source_native_race_id']='123'
    elif defect=='unknown_runner':row['source_native_runner_ids']=['159001',None]
    elif defect=='late_page':report['selected_races'][0]['jump_datetime']='2026-07-19T12:54:00+10:00'
    elif defect=='symlink':
        csv=Path(row['csv_path']);target=csv.with_suffix('.target');csv.rename(target);csv.symlink_to(target)
    elif defect=='unsafe_path':meta['primary_race_page_evidence']['raw_path']='../outside.html'
    sidecar.write_text(json.dumps(meta));path.write_text(json.dumps(report))
    assert classify(plan['evidence_root'],rid) is None


def test_real_terminal_classification_keeps_owner_unavailable_until_new_index(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from race_collection import persistent_collector as owner, synchronous_manual_capture as capture
    from race_collection.metadata_exclusion import metadata_exclusion_pending
    from tests.test_persistent_dispatch_verdict import evidence
    from race_collection.live_freshness_contract import create_once
    plan,rid,path,report,sidecar=local_identity_case(tmp_path)
    record,runtime=evidence(Path(plan['evidence_root']),action='LIVE_PHASE_FAILED',status='FAILED',code=2)
    terminal=runtime/'service-terminals'/('a'*32+'.json')
    value=json.loads(terminal.read_bytes());value.update(run_id=rid,
        output_dir=str(Path(plan['evidence_root'])/('shadow_autopilot_daemonization_v1_'+rid)),
        final_verdict='NEEDS_MORE_AUTOMATION')
    terminal.write_text(json.dumps(value))
    output=tmp_path/'owner';retained=output/'metadata-exclusions'/(rid+'.json')
    observed=datetime.fromisoformat(report['generated_at'])
    create_once(retained,{**classify(plan['evidence_root'],rid),'allocation_sha256':'b'*64,
        'observed_at':observed.isoformat()})
    verdict=owner.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)
    assert verdict['disposition']=='METADATA_EXCLUDED'
    source=[observed-timedelta(seconds=1)]
    monkeypatch.setattr(capture,'bounded_current_race_index',lambda **kw:SimpleNamespace(source_generated_at=source[0].isoformat()))
    reference=owner.reference(retained)
    assert metadata_exclusion_pending(plan,reference,'b'*64,observed+timedelta(seconds=20))
    source[0]=observed+timedelta(seconds=10)
    assert not metadata_exclusion_pending(plan,reference,'b'*64,observed+timedelta(seconds=20))
    assert not (output/'HALT.json').exists()
    retained.unlink()
    with pytest.raises(ValueError,match='terminal_failure'):
        owner.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)
