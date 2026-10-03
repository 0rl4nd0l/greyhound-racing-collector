"""Explicit recovery preserves a failed refresh; it never reclassifies it."""
import json
from pathlib import Path

import pytest

from race_collection import persistent_native as native
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend
from tests.fixtures.persistent_operation_case import put


@pytest.fixture
def terminal_recovery(recovery):
    cfg,standing,old,now,item,terminal,calls=recovery
    output=Path(old['output']); evidence=Path(old['plan']['evidence_root'])
    runtime=evidence/'shadow_autopilot_daemon_runtime'
    invocation='a'*32; run='synthetic_failed_refresh'
    cycle=evidence/('shadow_autopilot_daemonization_v1_'+run)
    phase_output=evidence/('shadow_autopilot_v1_'+run+'_phase_0')
    report=put(phase_output/'odds_capture_refresh_report.json',{
        'status':'ACQUISITION_INCOMPLETE','reason':'unisolated_selected_race_acquisition_failure',
        'downloads':[{'success':False,'result':{'success':False,'error':'discovery_canonical_jump_changed'}}]})
    phase=put(cycle/'phase-0-result.json',{'status':'FAIL','collection_phase':'refresh',
        'final_verdict':'COLLECTION_PHASE_BLOCKED','run_id':run+'_phase_0','output_dir':str(phase_output),
        'current_race_index_publish':{'status':'REJECTED','reason':'CURRENT_INDEX_SOURCE_INVALID',
            'source_refresh_report_path':report['path']}})
    checkpoint=put(cycle/'phase-checkpoint.json',{'schema_version':'collector_live_phase_checkpoint_v1',
        'cycle_id':run,'output_dir':str(cycle),'status':'LIVE_PHASE_FAILED',
        'phases':[{'number':0,'kind':'refresh','status':'COMPLETE','budget_exceeded':False,
            'result_path':phase['path'],'result_sha256':phase['sha256']}]})
    service=put(runtime/'service-terminals'/(invocation+'.json'),{'invocation_id':invocation,
        'allocation_sha256':old['allocation_ref']['sha256'],'status':'FAILED',
        'runtime_action':'LIVE_PHASE_FAILED','final_verdict':'NEEDS_MORE_AUTOMATION',
        'run_id':run,'output_dir':str(cycle)})
    lifecycle=put(runtime/'service-lifecycles'/(invocation+'.json'),{
        'invocation_id':invocation,'status':'COMPLETE','returncode':2,'children_reaped':True})
    state=json.loads((output/'persistent-owner-state.json').read_bytes())
    state['dispatches']=[{'invocation_id':invocation,'lane':'odds','returncode':2,
        'native_disposition':'FAILED_OR_UNVERIFIED'}]
    put(output/'persistent-owner-state.json',state)
    selection=json.loads(Path(cfg['recovery_selection']['path']).read_bytes())
    cleanup=json.loads(Path(selection['cleanup']['path']).read_bytes())
    cleanup['halt']=put(output/'HALT.json',{'reason':'persistent_native_terminal_failure:LIVE_PHASE_FAILED'})
    selection['prior_stop']=put(Path(selection['prior_stop']['path']),{'reason':'LIVE_PHASE_FAILED'})
    selection['baseline']=native.recovery_baseline(old,cfg)
    review={'schema_version':'persistent_reviewed_refresh_failure_v1',
        'disposition':'PROSPECTIVE_CORRECTION_OLD_FAILURE_UNRESOLVED',
        'authority_reference':'SYNTHETIC_EXPLICIT_REVIEW','source_commit':cfg['source_commit'],
        'service_terminal':service,'service_lifecycle':lifecycle,'checkpoint':checkpoint,
        'phase_result':phase,'refresh_report':report}
    selection['cleanup']=put(Path(selection['cleanup']['path']),cleanup)
    review['cleanup']=selection['cleanup']
    selection['reviewed_failure']=put(output/'reviewed-failure.json',review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    return recovery,review


def test_explicit_refresh_recovery_preserves_failed_evidence_grant_and_counters(terminal_recovery):
    recovery,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    preserved=[Path(cfg['campaign_root'])/'ledger.json',Path(cfg['source_state']),
        Path(old['output'])/'HALT.json',*[Path(review[k]['path']) for k in
          ('service_terminal','service_lifecycle','checkpoint','phase_result','refresh_report')]]
    before={p:p.read_bytes() for p in preserved}
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    assert new['allocation_ref']==old['allocation_ref']
    assert new['plan']['prediction_root']==old['plan']['prediction_root']
    assert {p:p.read_bytes() for p in preserved}==before
    assert native.prepare_day(cfg,standing,'2026-10-03',now)==new
    assert len(calls)==2


@pytest.mark.parametrize('change',[
    'missing_review','wrong_source','wrong_cleanup','wrong_disposition',
    'terminal_tampered','terminal_allocation','lifecycle_not_reaped',
    'checkpoint_cycle','checkpoint_budget','phase_success','report_success',
    'foreign_checkpoint','other_failure_signature','stop_pair',
])
def test_refresh_recovery_rejects_missing_or_mismatched_proof(terminal_recovery,change):
    recovery,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=json.loads(Path(cfg['recovery_selection']['path']).read_bytes())
    if change=='missing_review':selection.pop('reviewed_failure')
    elif change=='stop_pair':
        selection['prior_stop']=put(Path(selection['prior_stop']['path']),{'reason':'OTHER_FAILURE'})
    else:
        if change=='wrong_source':review['source_commit']='d'*40
        elif change=='wrong_cleanup':review['cleanup']={'path':'/foreign','sha256':'a'*64}
        elif change=='wrong_disposition':review['disposition']='OLD_FAILURE_ACCEPTED'
        elif change=='foreign_checkpoint':
            review['checkpoint']=put(Path(old['output'])/'foreign.json',native.checked(review['checkpoint']))
        else:
            key,field,value={
                'terminal_tampered':('service_terminal','status','READY'),
                'terminal_allocation':('service_terminal','allocation_sha256','a'*64),
                'lifecycle_not_reaped':('service_lifecycle','children_reaped',False),
                'checkpoint_cycle':('checkpoint','cycle_id','foreign'),
                'checkpoint_budget':('checkpoint',None,None),
                'phase_success':('phase_result','status','PASS'),
                'report_success':('refresh_report','status','SUCCESS'),
                'other_failure_signature':('refresh_report','downloads',[
                    {'success':False,'result':{'success':False,'error':'source_denied'}}]),
            }[change]
            value_to_change=native.checked(review[key])
            if change=='checkpoint_budget':value_to_change['phases'][0]['budget_exceeded']=True
            else:value_to_change[field]=value
            changed=put(Path(review[key]['path']),value_to_change)
            # Rehashed invalid evidence must still fail semantic checks.
            if change!='terminal_tampered':review[key]=changed
        selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    ledger=Path(cfg['campaign_root'])/'ledger.json';before=ledger.read_bytes()
    with pytest.raises((ValueError,KeyError)):
        native.prepare_day(cfg,standing,'2026-10-03',now)
    assert ledger.read_bytes()==before
    assert len(calls)==1


@pytest.mark.parametrize('size,accepted',[(554033,True),(4*1024*1024+1,False)])
def test_refresh_report_has_separate_finite_evidence_limit(terminal_recovery,size,accepted):
    recovery,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    report=native.checked(review['refresh_report']);report['padding']='x'*size
    review['refresh_report']=put(Path(review['refresh_report']['path']),report)
    selection=native.checked(cfg['recovery_selection'])
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    if accepted:
        assert native.prepare_day(cfg,standing,'2026-10-03',now)['allocation_ref']==old['allocation_ref']
    else:
        with pytest.raises(ValueError,match='report_reference_unsafe'):
            native.prepare_day(cfg,standing,'2026-10-03',now)
        assert len(calls)==1


def test_refresh_report_changed_bytes_reject_before_preparation(terminal_recovery):
    recovery,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    path=Path(review['refresh_report']['path'])
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='report_changed'):
        native.prepare_day(cfg,standing,'2026-10-03',now)
    assert len(calls)==1
