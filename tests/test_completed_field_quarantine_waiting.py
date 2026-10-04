"""Complete received field exclusions wait without certifying a usable index."""
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import pytest

from scripts import refresh_prejump_upcoming as refresh
from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence
from tests.test_persistent_metadata_exclusion import fixture, classify
from tests.test_semantic_runner_quarantine import undercovered_report
from tests.test_runner_completeness import _race_page, _runner_row
from tests.test_empty_eligible_refresh import publish
from utils.runner_completeness import extract_canonical_runner_set_from_html, align_csv_text_to_canonical_final_runner_set


def field_case(tmp_path):
    output, plan, rid, phase, path, report = fixture(tmp_path)
    worker = Path(plan['evidence_root'])/'worker';worker.mkdir()
    raw_dir = worker/'raw_exports';raw_dir.mkdir()
    local = undercovered_report(worker)
    candidate = local['selected_races'][1]
    download = local['downloads'][1]
    result = download['result'];normal = result['normalization']
    raw = Path(result['raw_export_path']);moved=raw_dir/'raw.csv';raw.rename(moved)
    result['raw_export_path']=normal['raw_export_path']=str(moved)
    observed=datetime.fromisoformat(report['generated_at'])
    body=_race_page(*[_runner_row(i,f'Dog {i}',runner_id=str(159000+i)) for i in range(1,5)]).replace('>R4<','>R6<').encode()
    canonical=extract_canonical_runner_set_from_html(body.decode(),source_url=candidate['race_url'],
        expected_race_number=6,extraction_timestamp=observed.isoformat())
    ref=persist_primary_race_page_evidence(artifact_root=worker,race_discovery_key=candidate['race_id'],
        response=TimedResponse(requested_url=candidate['race_url'],final_url=candidate['race_url'],
            request_start_utc=observed-timedelta(seconds=1),request_end_utc=observed,status_code=200,
            headers={'content-type':'text/html'},body=body),canonical_runner_set=canonical)
    canonical.update(native_identity_status='unavailable',source_native_race_id=None,
        native_identity_reasons=['native_identity_evidence_rejected:scratched_runner_has_active_price'])
    _, normal['canonical_runner_alignment']=align_csv_text_to_canonical_final_runner_set(
        moved.read_text(),canonical,source=normal['accepted_csv_path'])
    normal['normalization_timestamp']=observed.isoformat()
    report.update(upcoming_dir=str(worker),selected_count=1,accepted_csv_count=0,sidecar_count=0,
        quarantine_count=1,selected_races=[candidate],downloads=[download],reason='no_selected_race_csv_sidecars')
    report['sidecar_metadata_coverage']['races']=[local['sidecar_metadata_coverage']['races'][1]]
    report['current_index_races'],report['current_index_metadata_selection']=refresh.current_index_metadata_selection(
        report['selected_races'],report['sidecar_metadata_coverage'],source_generated_at=report['generated_at'])
    path.write_text(json.dumps(report))
    return plan,rid,path,report,worker,ref


def test_exact_received_field_exclusion_is_waiting_only(tmp_path):
    plan,rid,path,report,worker,ref=field_case(tmp_path)
    before=path.read_bytes()
    result=classify(plan['evidence_root'],rid)
    assert result and result['selected_count']==result['excluded_count']==1
    assert result['current_index_published'] is False
    assert not refresh.complete_empty_metadata_selection(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'excluded')['status']=='REJECTED'
    assert path.read_bytes()==before


@pytest.mark.parametrize('defect',['unknown','denial','budget','partial','csv','quarantine','page','receipt',
    'wrong_race','late','future_capture','missing_page','ambiguous_page','symlink','path','canonical','schema','shared'])
def test_unproved_field_exclusion_is_terminal(tmp_path,defect):
    plan,rid,path,report,worker,ref=field_case(tmp_path)
    result=report['downloads'][0]['result'];normal=result['normalization']
    if defect=='unknown':normal['canonical_runner_alignment']['native_identity_reasons']=['unknown']
    elif defect=='denial':result['source_http_status']=403
    elif defect=='budget':report['status']='REFRESH_BUDGET_EXCEEDED'
    elif defect=='partial':report['downloads']=[]
    elif defect in ('csv','quarantine'):Path(result['raw_export_path' if defect=='csv' else 'quarantine_path']).write_bytes(b'changed')
    elif defect=='page':
        (worker/ref['raw_path']).chmod(0o600);(worker/ref['raw_path']).write_bytes(b'changed')
    elif defect=='receipt':
        (worker/ref['receipt_path']).chmod(0o600);(worker/ref['receipt_path']).write_text('{}')
    elif defect=='wrong_race':report['selected_races'][0]['race_number']=9
    elif defect=='late':report['selected_races'][0]['jump_datetime']=report['generated_at']
    elif defect=='future_capture':normal['normalization_timestamp']='2026-07-18T00:00:00+00:00'
    elif defect=='missing_page':(worker/ref['raw_path']).unlink()
    elif defect=='ambiguous_page':(worker/ref['receipt_path']).with_name('other.race-page.receipt.json').write_bytes((worker/ref['receipt_path']).read_bytes())
    elif defect=='symlink':
        p=worker/ref['raw_path'];p.rename(p.with_suffix('.original'));p.symlink_to(p.with_suffix('.original'))
    elif defect=='path':result['raw_export_path']=normal['raw_export_path']=str(worker/'..'/'raw.csv')
    elif defect=='canonical':normal['canonical_runner_alignment']['missing_canonical_participants'][0]['box_number']=8
    elif defect=='schema':normal['normalization_verification']['schema_status']='rejected'
    elif defect=='shared':report['shared_sportsbet_snapshot']['status']='FAILED'
    path.write_text(json.dumps(report))
    assert classify(plan['evidence_root'],rid) is None


def test_real_terminal_owner_holds_until_a_later_verified_fresh_index(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from race_collection import persistent_collector as owner,synchronous_manual_capture as capture
    from race_collection.metadata_exclusion import metadata_exclusion_pending
    from tests.test_persistent_dispatch_verdict import evidence
    from race_collection.live_freshness_contract import create_once
    plan,rid,path,report,worker,ref=field_case(tmp_path)
    record,runtime=evidence(Path(plan['evidence_root']),action='LIVE_PHASE_FAILED',status='FAILED',code=2)
    terminal=runtime/'service-terminals'/('a'*32+'.json')
    value=json.loads(terminal.read_bytes());value.update(run_id=rid,
        output_dir=str(Path(plan['evidence_root'])/('shadow_autopilot_daemonization_v1_'+rid)),final_verdict='NEEDS_MORE_AUTOMATION')
    terminal.write_text(json.dumps(value))
    output=tmp_path/'owner';retained=output/'metadata-exclusions'/(rid+'.json')
    observed=datetime.fromisoformat(report['generated_at'])
    create_once(retained,{**classify(plan['evidence_root'],rid),'allocation_sha256':'b'*64,'observed_at':observed.isoformat()})
    assert owner.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)['disposition']=='METADATA_EXCLUDED'
    source=[observed-timedelta(seconds=1)]
    monkeypatch.setattr(capture,'bounded_current_race_index',lambda **kw:SimpleNamespace(source_generated_at=source[0].isoformat()))
    assert metadata_exclusion_pending(plan,owner.reference(retained),'b'*64,observed+timedelta(seconds=20))
    source[0]=observed+timedelta(seconds=10)
    assert not metadata_exclusion_pending(plan,owner.reference(retained),'b'*64,observed+timedelta(seconds=20))
    assert not (output/'HALT.json').exists()
    retained.unlink()
    with pytest.raises(ValueError,match='terminal_failure'):
        owner.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)
