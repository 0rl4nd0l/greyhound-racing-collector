"""Missing target metadata and rejected identity compose only with exact field proof."""
import csv
import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from scripts import refresh_prejump_upcoming as refresh
from scripts.capture_thedogs_market_history import TimedResponse, persist_primary_race_page_evidence
from tests.test_mixed_completed_field_exclusions import mixed_case
from tests.test_empty_eligible_refresh import publish
from tests.test_runner_completeness import _race_page, _runner_row
from utils.csv_metadata import THEDOGS_EXPERT_FORM_COLUMNS
from utils.runner_completeness import (extract_canonical_runner_set_from_html,
    align_csv_text_to_canonical_final_runner_set, analyze_csv_text_runner_completeness)
from tests.test_persistent_mixed_field_recovery import mixed_recovery
from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend


def composed_case(tmp_path):
    report, worker, old_ref = mixed_case(tmp_path)
    candidate = report['selected_races'][0]
    result = report['downloads'][0]['result']; normal = result['normalization']
    observed = datetime.fromisoformat(report['generated_at'])
    for key in ('raw_path', 'receipt_path'):
        (worker / old_ref[key]).unlink()
    body = _race_page(*[_runner_row(i, f'Dog {i}', runner_id=str(159000+i))
        for i in range(1,6)]).replace('>R4<', '>R6<').encode()
    canonical = extract_canonical_runner_set_from_html(body.decode(), source_url=candidate['race_url'],
        expected_race_number=6, extraction_timestamp=observed.isoformat())
    ref = persist_primary_race_page_evidence(artifact_root=worker, race_discovery_key=candidate['race_id'],
        response=TimedResponse(requested_url=candidate['race_url'], final_url=candidate['race_url'],
            request_start_utc=observed-timedelta(seconds=1), request_end_utc=observed, status_code=200,
            headers={'content-type':'text/html'}, body=body), canonical_runner_set=canonical)
    canonical.update(native_identity_status='unavailable', source_native_race_id=None,
        native_identity_reasons=['native_identity_evidence_rejected:scratched_runner_has_active_price'])
    stream=io.StringIO(); writer=csv.writer(stream); writer.writerow(THEDOGS_EXPERT_FORM_COLUMNS)
    for i in range(1,6):
        row=['']*len(THEDOGS_EXPERT_FORM_COLUMNS); row[0]=f'{i}. Dog {i}'
        row[THEDOGS_EXPERT_FORM_COLUMNS.index('DATE')]='2026-07-18'; writer.writerow(row)
    raw=stream.getvalue().encode()
    for key in ('raw_export_path', 'quarantine_path'): Path(result[key]).write_bytes(raw)
    aligned, alignment=align_csv_text_to_canonical_final_runner_set(raw.decode(), canonical,
        source=normal['accepted_csv_path'])
    after=analyze_csv_text_runner_completeness(aligned,source=normal['accepted_csv_path']).as_dict()
    for row in after['participants']:
        row.update(scratch_state='ACTIVE',source_native_runner_id=str(159000+row['box_number']))
    result.update(error='Downloaded CSV failed canonical TheDogs normalization gate',
        runner_completeness=analyze_csv_text_runner_completeness(raw.decode(),source='download:'+candidate['race_url']).as_dict())
    normal.update(raw_content_length=len(raw),raw_content_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_runner_alignment=alignment,runner_completeness_after_canonical_alignment=after,
        source_native_race_id=None,normalization_failure_reason='target_metadata_not_verified:missing_target_grade')
    normal['normalization_verification'].update(runner_set_status='COMPLETE',target_metadata_status='missing',
        target_metadata_failure_reason='missing_target_grade',race_time_source='canonical_race_url')
    return report,worker,ref


def test_proved_composed_exclusion_preserves_field_and_publishes_only_other_race(tmp_path):
    report,worker,ref=composed_case(tmp_path)
    before=json.dumps(report['downloads'][0],sort_keys=True)
    assert refresh.complete_mixed_field_exclusions(report)
    assert not refresh.has_unisolated_refresh_failure(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'composed')['race_count']==1
    assert json.dumps(report['downloads'][0],sort_keys=True)==before
    report.update(status='ACQUISITION_INCOMPLETE',reason='unisolated_selected_race_acquisition_failure')
    assert refresh.complete_mixed_field_exclusions(report)
    assert publish(tmp_path,tmp_path/'old-state.json',report,'old')['status']=='REJECTED'


@pytest.mark.parametrize('defect',['page','missing_page','late','identity','field','after_count','after_runner',
    'raw','quarantine','schema','target_identity','unknown_metadata','403','429','502','accepted_input'])
def test_composed_exclusion_cannot_bypass_any_original_proof(tmp_path,defect):
    report,worker,ref=composed_case(tmp_path)
    result=report['downloads'][0]['result']; normal=result['normalization']
    if defect=='page':
        p=worker/ref['raw_path'];p.chmod(0o600);p.write_bytes(b'changed')
    elif defect=='missing_page':(worker/ref['raw_path']).unlink()
    elif defect=='late':report['selected_races'][0]['jump_datetime']=report['generated_at']
    elif defect=='identity':normal['canonical_runner_alignment']['native_identity_reasons']=['unknown']
    elif defect=='field':normal['canonical_runner_alignment']['canonical_runner_count']=6
    elif defect=='after_count':normal['runner_completeness_after_canonical_alignment']['runner_count']=6
    elif defect=='after_runner':normal['runner_completeness_after_canonical_alignment']['participants'][0]['source_native_runner_id']='9999'
    elif defect in ('raw','quarantine'):Path(result['raw_export_path' if defect=='raw' else 'quarantine_path']).write_bytes(b'changed')
    elif defect=='schema':normal['normalization_verification']['schema_status']='rejected'
    elif defect=='target_identity':normal['normalization_verification']['capture_race_number']=7
    elif defect=='unknown_metadata':normal['normalization_verification']['target_metadata_failure_reason']='unknown'
    elif defect in ('403','429','502'):result['source_http_status']=int(defect)
    elif defect=='accepted_input':Path(report['sidecar_metadata_coverage']['races'][1]['csv_path']).write_bytes(b'changed')
    assert not refresh.complete_mixed_field_exclusions(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'bad')['status']=='REJECTED'


def test_existing_reviewed_successor_preserves_failed_composed_report_and_allocation(mixed_recovery,tmp_path):
    from race_collection import persistent_native as native
    from tests.fixtures.persistent_operation_case import put
    case,review=mixed_recovery
    cfg,standing,old,now,item,terminal,calls=case
    report,worker,ref=composed_case(tmp_path/'composed')
    report.update(status='ACQUISITION_INCOMPLETE',reason='unisolated_selected_race_acquisition_failure')
    review['refresh_report']=put(Path(review['refresh_report']['path']),report)
    proof=native.checked(review['classified_mixed_field_exclusion'])
    proof['refresh_sha256']=review['refresh_report']['sha256']
    review['classified_mixed_field_exclusion']=put(Path(review['classified_mixed_field_exclusion']['path']),proof)
    selection=native.checked(cfg['recovery_selection'])
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    protected=[Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state']),
        Path(old['output'])/'HALT.json',Path(selection['prior_stop']['path']),
        Path(review['refresh_report']['path']),Path(review['phase_result']['path'])]
    before={p:p.read_bytes() for p in protected}
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert new['plan']['prediction_root']==old['plan']['prediction_root']
    assert {p:p.read_bytes() for p in protected}==before
    assert not (Path(old['output'])/'day-closed.json').exists()
    assert len(calls)==2
