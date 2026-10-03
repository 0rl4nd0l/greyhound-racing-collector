"""A completed empty timing selection must not stop a live observation."""
from datetime import datetime, timedelta
import pytest
from tests.test_empty_eligible_refresh import publish, report_fixture
from tests.race_collection.test_synchronous_manual_capture import _write_publication_evidence
from race_collection import synchronous_manual_capture as capture
from scripts.refresh_prejump_upcoming import current_index_metadata_selection


def empty_selection(generated, *, future=False):
    jump = (generated + timedelta(minutes=80 if future else 5)).replace(second=0, microsecond=0)
    bucket = 'future_outside_preferred_window' if future else 'past_or_too_close'
    coverage = dict(schema_version='prejump_sidecar_metadata_coverage_v1',status='NOT_REQUESTED_NO_SELECTED_RACES',reason='no_selected_races',selected_race_count=0,accepted_selected_csv_count=0,races=[])
    rows,selection=current_index_metadata_selection([],coverage,source_generated_at=generated)
    return dict(artifact_counts=dict(accepted_csv_count=0,sidecar_count=0,raw_export_count=0,quarantine_count=0),status='SUCCESS',dry_run=False,generated_at=generated.isoformat(),total_races_found=1,selected_count=0,selected_races=[],downloads=[],accepted_csv_count=0,sidecar_count=0,quarantine_count=0,raw_export_count=0,current_index_race_count=0,current_index_races=rows,current_index_metadata_selection=selection,sidecar_metadata_coverage=coverage,metadata_collection_status='NOT_REQUESTED_NO_SELECTED_RACES',window=dict(min_minutes=6.333333333333333,max_minutes=60),bucket_counts={bucket:1},considered_races=[dict(race_id='Race 1 - TEST - 2026-07-19',race_url='https://www.thedogs.com.au/racing/test/2026-07-19/1',jump_datetime=jump.isoformat(),minutes_to_jump=(jump-generated).total_seconds()/60,date=jump.date().isoformat(),race_time=jump.strftime("%H:%M"),venue="TEST",race_number=1,bucket=bucket,selection_decision=bucket,selected=False)])


@pytest.mark.parametrize('future',[False,True])
def test_native_publisher_and_reader_replace_previous_races_with_complete_empty_selection(tmp_path,future):
    evidence=tmp_path/'evidence';state=evidence/'runtime/odds.json'
    initial=report_fixture(evidence,eligible=True)
    first=publish(evidence,state,initial,'first');assert first['status']=='PUBLISHED'
    _write_publication_evidence(evidence,state,first)
    at=datetime.fromisoformat(initial['generated_at'])+timedelta(seconds=1)
    empty=empty_selection(at,future=future)
    result=publish(evidence,state,empty,'empty')
    assert result['status']=='PUBLISHED', result
    assert result['race_count']==0
    _write_publication_evidence(evidence,state,result)
    view=capture.bounded_current_race_index(current_time=at+timedelta(seconds=1),timeout_seconds=1,index_path=capture.current_race_index_path(state),evidence_root=evidence,max_age_seconds=300,return_verified_view=True)
    assert not view.races and view.source_generated_at==empty['generated_at']
    recovered=publish(evidence,state,report_fixture(evidence,generated=at+timedelta(seconds=2),eligible=True),'recovered')
    assert recovered['status']=='PUBLISHED' and recovered['race_count']==1


@pytest.mark.parametrize('change',['discovery_failed','budget','dry_run','selected','downloads','eligible_time','missing_time','count','bucket','coverage','retry','missing_selection','nan','raw_export','nested_artifact','race_time','duplicate'])
def test_empty_selection_cannot_conceal_incomplete_or_failed_work(tmp_path,change):
    value=empty_selection(datetime.fromisoformat('2026-07-19T12:55:00+10:00'))
    if change=='discovery_failed':value['discovery_failures']=[{'error_type':'ConnectionError'}]
    elif change=='budget':value['status']='REFRESH_BUDGET_EXCEEDED'
    elif change=='dry_run':value['dry_run']=True
    elif change=='selected':value['selected_count']=1
    elif change=='downloads':value['downloads']=[{'success':False}]
    elif change=='eligible_time':value['considered_races'][0]['jump_datetime']='2026-07-19T13:25:00+10:00'
    elif change=='missing_time':value['considered_races'][0]['jump_datetime']=None
    elif change=='count':value['total_races_found']=2
    elif change=='bucket':value['bucket_counts']={}
    elif change=='coverage':value['sidecar_metadata_coverage']['races']=[{}]
    elif change=='retry':value['source_retry_after']='60'
    elif change=='missing_selection':value.pop('current_index_metadata_selection')
    elif change=='nan':value['window']['max_minutes']=float('nan')
    elif change=='raw_export':value['raw_export_count']=1
    elif change=='nested_artifact':value['artifact_counts']['accepted_csv_count']=1
    elif change=='race_time':value['considered_races'][0]['race_time']='13:25'
    elif change=='duplicate':
        value['considered_races']*=2;value['total_races_found']=2;value['bucket_counts']['past_or_too_close']=2
    assert publish(tmp_path/'evidence',tmp_path/'evidence/runtime/odds.json',value,'invalid')['status']=='REJECTED'
