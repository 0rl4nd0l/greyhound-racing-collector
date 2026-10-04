"""Exact failed publisher overlap permits only a reviewed same-grant successor."""
from datetime import timedelta
import os
from pathlib import Path

import pytest

from race_collection import persistent_native as native
from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend
from tests.fixtures.persistent_operation_case import put


@pytest.fixture
def publisher_recovery(terminal_recovery):
    case,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=case
    selection=native.checked(cfg['recovery_selection'])
    output=Path(old['output']);evidence=Path(old['plan']['evidence_root'])
    service=native.checked(review['service_terminal']);cycle=Path(service['output_dir'])
    at=lambda seconds:(now+timedelta(seconds=seconds)).isoformat()
    service.update(status='NEEDS_MORE_AUTOMATION',at=at(-20))
    review['service_terminal']=put(Path(review['service_terminal']['path']),service)
    overlap_id='b'*32;overlap_run='synthetic_overlap_odds_capture'
    other=evidence/('shadow_autopilot_daemonization_v1_'+overlap_run);other.mkdir()
    runtime=evidence/'shadow_autopilot_daemon_runtime'
    review['overlap_terminal']=put(runtime/'service-terminals'/(overlap_id+'.json'),{
        'invocation_id':overlap_id,'run_id':overlap_run,'output_dir':str(other),
        'allocation_sha256':old['allocation_ref']['sha256'],'status':'SKIPPED_LOCK_HELD',
        'runtime_action':'DEFERRED_LOCK_HELD','final_verdict':'PARTIAL_DAEMONIZATION','at':at(-20.4)})
    review['overlap_lifecycle']=put(runtime/'service-lifecycles'/(overlap_id+'.json'),{
        'invocation_id':overlap_id,'status':'COMPLETE','returncode':2,'children_reaped':True,'interrupted':False})
    state=native.checked(selection['baseline']['prior_owner_state'])
    state['dispatches']=[{'invocation_id':service['invocation_id'],'lane':'full','returncode':2,
      'native_disposition':'FAILED_OR_UNVERIFIED','started_at':at(-40),'completed_at':at(-19)},
      {'invocation_id':overlap_id,'lane':'odds','returncode':2,'native_disposition':'DEFERRED',
       'started_at':at(-25),'completed_at':at(-18)}]
    put(output/'persistent-owner-state.json',state)
    stderr=cycle/'phase-0/logs/autopilot_cycle.stderr.txt';stderr.parent.mkdir(parents=True)
    stderr.write_text('Traceback (most recent call last):\n  File "'+str(Path(old['plan']['source_root'])/'race_collection/synchronous_manual_capture.py')+'", in _atomic_replace_canonical\n in _recheck_directory_chain\n raise CaptureOneRejected(reason="publish_root_replaced")\nrace_collection.synchronous_manual_capture.CaptureOneRejected: CURRENT_INDEX_PATH_UNSAFE\n')
    review['stderr']=native._ref(stderr)
    phase={'status':'FAIL','reason':'phase_output_missing','step':{'name':'autopilot_cycle','status':'FAIL',
      'returncode':1,'timed_out':False,'interrupted':False,'stderr_path':str(stderr),
      'started_at':at(-39),'finished_at':at(-20.1)}}
    review['phase_result']=put(Path(review['phase_result']['path']),phase)
    checkpoint=native.checked(review['checkpoint']);checkpoint['phases'][0]['result_sha256']=review['phase_result']['sha256']
    review['checkpoint']=put(Path(review['checkpoint']['path']),checkpoint)
    review['refresh_report']=put(Path(review['refresh_report']['path']).with_name('refresh_prejump_report.json'),{
      'status':'SUCCESS','metadata_collection_status':'READY','selected_count':1,'accepted_csv_count':1,
      'current_index_race_count':1,'shared_sportsbet_snapshot':{'status':'VALIDATED'},
      'downloads':[{'success':True,'result':{'success':True}}]})
    cleanup=native.checked(selection['cleanup']);cleanup['at']=at(-10)
    cleanup['halt']=put(output/'HALT.json',{'reason':'persistent_native_terminal_failure:LIVE_PHASE_FAILED','at':at(-17)})
    selection['cleanup']=put(Path(selection['cleanup']['path']),cleanup)
    ns=int((now+timedelta(seconds=-20.5)).timestamp()*1e9);os.utime(other,ns=(ns,ns));st=other.stat()
    review['overlap_directory_metadata']=put(output/'overlap-directory.json',{
      'path':str(other),'device':st.st_dev,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns,
      'ctime_ns':st.st_ctime_ns,'observed_at':at(-9)})
    review['diagnosis']=put(output/'offline-diagnosis.json',{
      'schema_version':'persistent_publisher_overlap_diagnosis_v1','status':'REPRODUCED_OWNED_SIBLING_CREATION',
      'prospective_fix':'SERIALIZE_NATIVE_PUBLISHERS_UNTIL_REAPED','strict_atomic_guard_unchanged':True,
      'failure_code':'CURRENT_INDEX_PATH_UNSAFE','failure_reason':'publish_root_replaced',
      'owner_regression_red_then_green':True,'source_commit':cfg['source_commit']})
    review.update(schema_version='persistent_reviewed_publisher_overlap_failure_v1',
      disposition='PROSPECTIVE_SERIALIZED_NATIVE_PUBLISHERS_CORRECTION',cleanup=selection['cleanup'],
      live_directory_identity_history='NOT_RETAINED',correlation_only=True)
    selection['baseline']=native.recovery_baseline(old,cfg)
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    return case,review


def test_reviewed_overlap_successor_preserves_failure_and_same_allocation(publisher_recovery):
    case,review=publisher_recovery;cfg,standing,old,now,item,terminal,calls=case
    selection=native.checked(cfg['recovery_selection'])
    paths=[Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state']),Path(old['output'])/'HALT.json',
      Path(selection['prior_stop']['path']),*(Path(review[k]['path']) for k in ['service_terminal','service_lifecycle',
      'overlap_terminal','overlap_lifecycle','checkpoint','phase_result','stderr','refresh_report'])]
    before={p:p.read_bytes() for p in paths}
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert new['plan']['prediction_root']==old['plan']['prediction_root']
    assert all(p.read_bytes()==raw for p,raw in before.items())
    assert not (Path(old['output'])/'day-closed.json').exists()
    assert not list((Path(new['output'])/'refresh-deferrals').glob('*.json'))
    assert native.prepare_day(cfg,standing,'2026-10-03',now)==new
    assert len(calls)==2


@pytest.mark.parametrize('change',['wrong_source','wrong_stop','missing_review','other_reason','stderr_tamper',
 'unreaped','foreign_terminal','wrong_allocation','budget','phase_timeout','phase_success','report_denial',
 'report_incomplete','no_overlap','wrong_directory','unreviewed_fix','guard_weakened','wrong_cleanup'])
def test_overlap_recovery_rejects_unbound_or_different_failure(publisher_recovery,change):
    case,review=publisher_recovery;cfg,standing,old,now,item,terminal,calls=case
    selection=native.checked(cfg['recovery_selection'])
    if change=='wrong_source':review['source_commit']='f'*40
    elif change=='wrong_stop':selection['prior_stop']=put(Path(selection['prior_stop']['path']),{'reason':'OTHER'})
    elif change=='missing_review':selection.pop('reviewed_failure')
    elif change in ['other_reason','stderr_tamper']:
        path=Path(review['stderr']['path']);path.write_text(path.read_text().replace('publish_root_replaced','other_failure'))
        if change=='other_reason':review['stderr']=native._ref(path)
    elif change=='wrong_cleanup':review['cleanup']={'path':'/wrong','sha256':'0'*64}
    else:
        key={'unreaped':'overlap_lifecycle','foreign_terminal':'overlap_terminal','wrong_allocation':'service_terminal',
          'budget':'checkpoint','phase_timeout':'phase_result','phase_success':'phase_result',
          'report_denial':'refresh_report','report_incomplete':'refresh_report','no_overlap':'overlap_terminal',
          'wrong_directory':'overlap_directory_metadata','unreviewed_fix':'diagnosis','guard_weakened':'diagnosis'}[change]
        value=native.checked(review[key])
        if change=='unreaped':value['children_reaped']=False
        elif change=='foreign_terminal':value['invocation_id']='c'*32
        elif change=='wrong_allocation':value['allocation_sha256']='0'*64
        elif change=='budget':value['phases'][0]['budget_exceeded']=True
        elif change=='phase_timeout':value['step']['timed_out']=True
        elif change=='phase_success':value['status']='PASS'
        elif change=='report_denial':value['downloads'][0]['result']['source_http_status']=429
        elif change=='report_incomplete':value['accepted_csv_count']=0
        elif change=='no_overlap':value['at']=(now+timedelta(seconds=1)).isoformat()
        elif change=='wrong_directory':value['inode']+=1
        elif change=='unreviewed_fix':value['prospective_fix']='IGNORE_UNSAFE'
        elif change=='guard_weakened':value['strict_atomic_guard_unchanged']=False
        review[key]=put(Path(review[key]['path']),value)
        if key=='phase_result':
            ck=native.checked(review['checkpoint']);ck['phases'][0]['result_sha256']=review[key]['sha256']
            review['checkpoint']=put(Path(review['checkpoint']['path']),ck)
    if change!='missing_review':selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    ledger=Path(cfg['campaign_root'])/'ledger.json';before=ledger.read_bytes()
    with pytest.raises((ValueError,KeyError)):native.prepare_day(cfg,standing,'2026-10-03',now)
    assert ledger.read_bytes()==before and len(calls)==1
