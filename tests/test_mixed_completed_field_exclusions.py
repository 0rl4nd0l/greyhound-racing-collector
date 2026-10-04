"""A complete mixed denominator isolates proved exclusions, never unknown failure."""
import copy
import json
from pathlib import Path
import pytest
from scripts import refresh_prejump_upcoming as refresh
from tests.test_completed_field_quarantine_waiting import field_case
from tests.test_empty_eligible_refresh import report_fixture,publish


def mixed_case(tmp_path):
    plan,rid,path,report,worker,ref=field_case(tmp_path)
    good=report_fixture(worker,eligible=True)
    good['downloads'][0]['result']['filepath']=good['sidecar_metadata_coverage']['races'][0]['csv_path']
    report.update(status='SUCCESS',reason=None,selected_count=2,accepted_csv_count=1,sidecar_count=1,
        raw_export_count=2,quarantine_count=1,artifact_counts={
            'accepted_csv_count':1,'sidecar_count':1,'raw_export_count':2,'quarantine_count':1})
    report['selected_races']+=good['selected_races'];report['downloads']+=good['downloads']
    coverage=report['sidecar_metadata_coverage'];coverage['races']+=good['sidecar_metadata_coverage']['races']
    coverage.update(selected_race_count=2,accepted_selected_csv_count=1)
    report['current_index_races'],report['current_index_metadata_selection']=refresh.current_index_metadata_selection(
        report['selected_races'],coverage,source_generated_at=report['generated_at'])
    report['current_index_race_count']=len(report['current_index_races'])
    return report,worker,ref


def test_native_index_publishes_only_verified_member_with_full_denominator(tmp_path):
    report,worker,ref=mixed_case(tmp_path)
    failed=copy.deepcopy(report['downloads'][0])
    assert refresh.complete_mixed_field_exclusions(report)
    assert not refresh.has_unisolated_refresh_failure(report)
    result=publish(tmp_path,tmp_path/'runtime/state.json',report,'mixed')
    assert result['status']=='PUBLISHED' and result['race_count']==1
    assert report['selected_count']==2 and report['quarantine_count']==1
    assert report['downloads'][0]==failed
    assert report['current_index_races'][0]['race_id']==report['selected_races'][1]['race_id']


def test_old_failed_report_never_becomes_publishable_retroactively(tmp_path):
    report,worker,ref=mixed_case(tmp_path)
    report.update(status='ACQUISITION_INCOMPLETE',reason='unisolated_selected_race_acquisition_failure')
    before=json.dumps(report,sort_keys=True)
    assert refresh.complete_mixed_field_exclusions(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'old')['status']=='REJECTED'
    assert json.dumps(report,sort_keys=True)==before


def test_all_excluded_still_uses_waiting_only_proof(tmp_path):
    plan,rid,path,report,worker,ref=field_case(tmp_path)
    assert refresh.complete_unavailable_metadata_selection(report)
    assert not refresh.complete_mixed_field_exclusions(report)
    assert not refresh.complete_empty_metadata_selection(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'empty')['status']=='REJECTED'


@pytest.mark.parametrize('defect',['denial','budget','unknown','shared','discovery','transport','allowance','mixed_unknown','download_count',
    'selected_count','accepted_count','raw_count','quarantine_count','artifact_count','coverage_count',
    'bad_page','native_reason','put_quarantine_in_index','accepted_csv_tamper','accepted_sidecar_tamper'])
def test_incomplete_or_shared_failure_cannot_publish_subset(tmp_path,defect):
    report,worker,ref=mixed_case(tmp_path)
    failed=report['downloads'][0]['result']
    if defect=='denial':failed['source_http_status']=429
    elif defect=='transport':failed['source_http_status']=503
    elif defect=='allowance':failed['source_failure_category']='REQUEST_CAP_EXHAUSTED'
    elif defect=='mixed_unknown':
        candidate=copy.deepcopy(report['selected_races'][0]);candidate['race_id']='unknown';candidate['race_url']+='-other'
        report['selected_races'].append(candidate)
        report['downloads'].append({'race_url':candidate['race_url'],'success':False,'result':{'success':False,'error':'unknown'}})
        row=copy.deepcopy(report['sidecar_metadata_coverage']['races'][0]);row.update(race_id=candidate['race_id'],race_url=candidate['race_url'])
        report['sidecar_metadata_coverage']['races'].append(row)
        report['selected_count']=3;report['quarantine_count']=2;report['raw_export_count']=3
        report['sidecar_metadata_coverage']['selected_race_count']=3
        report['artifact_counts'].update(raw_export_count=3,quarantine_count=2)
        report['current_index_races'],report['current_index_metadata_selection']=refresh.current_index_metadata_selection(
            report['selected_races'],report['sidecar_metadata_coverage'],source_generated_at=report['generated_at'])
    elif defect=='budget':report.update(status='REFRESH_BUDGET_EXCEEDED',reason='completed_source_acquisition_exceeded_budget')
    elif defect=='unknown':failed['error']='unknown failure'
    elif defect=='shared':report['shared_sportsbet_snapshot']['status']='UNAVAILABLE'
    elif defect=='discovery':report['discovery_failures']=[{'error_type':'ReadTimeout'}]
    elif defect=='download_count':report['downloads'].pop()
    elif defect=='selected_count':report['selected_count']=1
    elif defect=='accepted_count':report['accepted_csv_count']=2
    elif defect=='raw_count':report['raw_export_count']=1
    elif defect=='quarantine_count':report['quarantine_count']=0
    elif defect=='artifact_count':report['artifact_counts']['raw_export_count']=1
    elif defect=='coverage_count':report['sidecar_metadata_coverage']['accepted_selected_csv_count']=2
    elif defect=='bad_page':
        p=worker/ref['raw_path'];p.chmod(0o600);p.write_bytes(b'changed')
    elif defect=='native_reason':failed['normalization']['canonical_runner_alignment']['native_identity_reasons']=['unproved']
    elif defect=='put_quarantine_in_index':report['current_index_races'].append(report['selected_races'][0])
    elif defect in ('accepted_csv_tamper','accepted_sidecar_tamper'):
        row=report['sidecar_metadata_coverage']['races'][1]
        p=Path(row['csv_path' if defect=='accepted_csv_tamper' else 'sidecar_path']);p.write_text('changed')
    assert not refresh.complete_mixed_field_exclusions(report)
    assert publish(tmp_path,tmp_path/'state.json',report,'bad')['status']=='REJECTED'
