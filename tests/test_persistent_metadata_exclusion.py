"""Completed but unusable race metadata remains excluded without stopping collection."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from race_collection import persistent_collector as owner
from scripts import refresh_prejump_upcoming as refresh
from tests.test_empty_eligible_refresh import report_fixture, publish
from tests.test_scheduled_refresh_outage import outage
from tests.test_persistent_dispatch_verdict import evidence


def fixture(tmp_path):
    output,plan,rid,root,path=outage(tmp_path)
    report=report_fixture(Path(plan['evidence_root']))
    row=report['sidecar_metadata_coverage']['races'][0]
    row.update(safe_expert_form_present=False, expert_form_rejected_reasons=[
        'expert_form_metadata_captured_at_not_before_jump','expert_form_runner_metadata_missing'],
        native_identity_evidence_status='verified')
    report.update(status='METADATA_COVERAGE_INCOMPLETE', reason='missing_safe_track_condition_expert_form',
                  shared_sportsbet_snapshot={'status':'VALIDATED','payload_sha256':'c'*64})
    _,report['current_index_metadata_selection']=refresh.current_index_metadata_selection(
        report['selected_races'],report['sidecar_metadata_coverage'],source_generated_at=report['generated_at'])
    path.write_text(json.dumps(report))
    result=root/'phase-0-result.json'
    result.write_text(json.dumps(failed_phase(path, Path(plan['evidence_root']), rid)))
    cp=json.loads((root/'phase-checkpoint.json').read_bytes())
    cp['phases'][0]['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
    (root/'phase-checkpoint.json').write_text(json.dumps(cp))
    return output,plan,rid,root,path,report


def failed_phase(path,evidence_root,rid):
    return {'status':'FAIL','collection_phase':'refresh','final_verdict':'COLLECTION_PHASE_BLOCKED',
        'output_dir':str(path.parent),'current_race_index_publish':{
            'schema_version':'collector_current_race_index_publish_v2','status':'REJECTED',
            'reason':'CURRENT_INDEX_SOURCE_INVALID','failure_detail':{'reason':'refresh_not_accepted_success'},
            'source_refresh_report_path':str(path),'run_id':rid,
            'index_path':str(evidence_root/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json')}}


def classify(evidence,rid):
    try:
        from race_collection.metadata_exclusion import classify_metadata_exclusion
    except ImportError:
        return None
    return classify_metadata_exclusion(evidence,rid)


def test_all_acquired_but_expert_unusable_is_deferred_without_index(tmp_path):
    output,plan,rid,root,path,report=fixture(tmp_path)
    before=path.read_bytes()
    result=classify(plan['evidence_root'],rid)
    assert result and result['disposition']=='COMPLETED_METADATA_EXCLUSIONS'
    assert result['selected_count']==result['excluded_count']==1
    assert result['request_retries_added']==0
    assert not refresh.complete_empty_metadata_selection(report)
    assert publish(tmp_path,tmp_path/'runtime/state.json',report,'rejected')['status']=='REJECTED'
    assert path.read_bytes()==before

@pytest.mark.parametrize('defect',['denial','budget','unknown_expert','expert_source','missing_race','failed_download',
                                  'shared_snapshot','native_identity','runner_time','count','phase_hash','prior_failure'])
def test_shared_or_incomplete_failure_is_never_deferred(tmp_path,defect):
    output,plan,rid,root,path,report=fixture(tmp_path)
    row=report['sidecar_metadata_coverage']['races'][0]
    if defect=='denial':report['downloads'][0]['result']['source_http_status']=429
    elif defect=='budget':report['status']='REFRESH_BUDGET_EXCEEDED'
    elif defect=='unknown_expert':row['expert_form_rejected_reasons']=['unknown']
    elif defect=='expert_source':row['expert_form_rejected_reasons']=['expert_form_source_url_not_allowed']
    elif defect=='missing_race':report['downloads']=[]
    elif defect=='failed_download':report['downloads'][0]['success']=False
    elif defect=='shared_snapshot':report['shared_sportsbet_snapshot']['status']='FAILED'
    elif defect=='native_identity':row['native_identity_evidence_status']='rejected'
    elif defect=='runner_time':row['runner_source_observed_at']=None
    elif defect=='count':report['accepted_csv_count']=0
    elif defect=='phase_hash':(root/'phase-0-result.json').write_text('{}')
    elif defect=='prior_failure':
        cp=json.loads((root/'phase-checkpoint.json').read_bytes());cp['phases'][0]['budget_exceeded']=True
        (root/'phase-checkpoint.json').write_text(json.dumps(cp))
    path.write_text(json.dumps(report))
    assert classify(plan['evidence_root'],rid) is None


def retained_owner_fixture(tmp_path):
    output,plan,rid,root,path,report=fixture(tmp_path)
    record,runtime=evidence(Path(plan['evidence_root']),action='LIVE_PHASE_FAILED',status='FAILED',code=2)
    terminal=runtime/'service-terminals'/('a'*32+'.json')
    value=json.loads(terminal.read_bytes());value.update(run_id=rid,output_dir=str(root),final_verdict='NEEDS_MORE_AUTOMATION')
    terminal.write_text(json.dumps(value))
    retained=output/'metadata-exclusions'/(rid+'.json');retained.parent.mkdir(parents=True)
    retained.write_text(json.dumps({**classify(plan['evidence_root'],rid),
        'allocation_sha256':'b'*64,'observed_at':'2026-07-19T03:00:00+00:00'}))
    return output,plan,record,retained


def test_owner_recognizes_verified_exclusion_without_success_or_retry(tmp_path):
    output,plan,record,retained=retained_owner_fixture(tmp_path)
    verdict=owner.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)
    assert verdict['disposition']=='METADATA_EXCLUDED'
    retained.unlink()
    with pytest.raises(ValueError,match='terminal_failure'):
        owner.classify_native_dispatch(plan['evidence_root'],record,'b'*64,output=output)


@pytest.mark.parametrize('case',['old_fresh','new_stale','missing','unsafe','future','fresh_after'])
def test_exclusion_blocks_admission_until_new_strict_fresh_index(tmp_path,monkeypatch,case):
    from race_collection.metadata_exclusion import metadata_exclusion_pending
    from race_collection import synchronous_manual_capture as capture
    output,plan,record,retained=retained_owner_fixture(tmp_path)
    current=datetime.fromisoformat('2026-07-19T03:10:00+00:00')
    if case=='old_fresh':
        value=json.loads(retained.read_bytes());value['observed_at']='2026-07-19T03:09:00+00:00'
        retained.write_text(json.dumps(value))
    def reader(**kwargs):
        if case in ('missing','unsafe'):
            raise capture.CaptureOneRejected('CURRENT_INDEX_UNAVAILABLE' if case=='missing' else 'CURRENT_INDEX_PATH_UNSAFE')
        source={'old_fresh':'2026-07-19T03:08:00+00:00',
                'future':'2026-07-19T03:11:00+00:00',
                'new_stale':'2026-07-19T03:01:00+00:00','fresh_after':'2026-07-19T03:09:00+00:00'}[case]
        return SimpleNamespace(source_generated_at=source)
    monkeypatch.setattr(capture,'bounded_current_race_index',reader)
    if case=='unsafe':
        with pytest.raises(capture.CaptureOneRejected):
            metadata_exclusion_pending(plan,owner.reference(retained),'b'*64,current)
    else:
        assert metadata_exclusion_pending(plan,owner.reference(retained),'b'*64,current)==(case!='fresh_after')


@pytest.mark.parametrize('failure',['metadata','denial','unknown','budget','allowance'])
def test_actual_native_cycle_keeps_failed_refresh_and_only_defers_proven_metadata(tmp_path,monkeypatch,failure):
    from datetime import timedelta
    from scripts import shadow_autopilot_daemon as daemon
    from race_collection import live_freshness_contract as contract
    from utils.sportsbet_access import SportsbetAccess
    evidence_root=tmp_path/'native-evidence';state=evidence_root/'runtime/state.json';lock=evidence_root/'runtime/collector.lock'
    output=tmp_path/'package';output.mkdir();now=datetime.now(timezone.utc)
    stopped=[]
    def admit(*args,seconds,**kwargs):
        if failure=='allowance' and seconds==0:
            raise ValueError('logical_request_allowance_exhausted')
    scope=SimpleNamespace(start=now-timedelta(minutes=1),end=now+timedelta(hours=1),session=output,
        value={'operational_predictions':True,'persistent_allocation':{'sha256':'b'*64}},
        admit=admit,check_paths=lambda **k:None,stop=lambda why:stopped.append(why))
    monkeypatch.setattr(contract.FreshnessContract,'load',lambda *a:scope)
    monkeypatch.setattr(contract,'AttemptAllowance',lambda s:SimpleNamespace(available=lambda:True))
    monkeypatch.setattr(SportsbetAccess,'read',lambda s:{'phase':'HOLD' if failure=='denial' else 'OPEN','access_basis':{'status':'permitted'}})
    def command(*,name,command,output_dir,**kwargs):
        phase_id=command[command.index('--run-id')+1]
        phase=evidence_root/('shadow_autopilot_v1_'+phase_id);phase.mkdir(parents=True)
        report=report_fixture(phase)
        row=report['sidecar_metadata_coverage']['races'][0]
        row.update(safe_expert_form_present=False,native_identity_evidence_status='verified',
                   expert_form_rejected_reasons=['unknown'] if failure=='unknown' else ['expert_form_metadata_captured_at_not_before_jump'])
        report.update(status='REFRESH_BUDGET_EXCEEDED' if failure=='budget' else 'METADATA_COVERAGE_INCOMPLETE',
                      reason='missing_safe_track_condition_expert_form',shared_sportsbet_snapshot={'status':'VALIDATED','payload_sha256':'c'*64})
        _,report['current_index_metadata_selection']=refresh.current_index_metadata_selection(report['selected_races'],report['sidecar_metadata_coverage'],source_generated_at=report['generated_at'])
        (phase/'odds_capture_refresh_report.json').write_text(json.dumps(report))
        log=output_dir/'logs'/f'{name}.stdout.txt';log.parent.mkdir(parents=True)
        log.write_text(json.dumps(failed_phase(phase/'odds_capture_refresh_report.json',evidence_root,phase_id.rsplit('_phase_',1)[0])))
        return {'name':name,'returncode':2}
    monkeypatch.setattr(daemon,'run_command',command)
    args=daemon.parse_args(['run-odds-capture-once','--live-freshness','--live-freshness-profile','bounded80-v1',
        '--live-freshness-contract',str(output/'contract.json'),'--evidence-root',str(evidence_root),
        '--state-path',str(state),'--lock-path',str(lock),'--db',str(tmp_path/'never-opened.db')])
    result=daemon.run_odds_capture_once(args)
    assert result['runtime_action']=='LIVE_PHASE_FAILED'
    assert not lock.exists()
    receipts=list((output/'metadata-exclusions').glob('*.json'))
    assert bool(receipts)==(failure=='metadata')
    assert bool(stopped)==(failure!='metadata')
    assert not (tmp_path/'never-opened.db').exists()


@pytest.mark.parametrize('change',['published','unsafe','wrong_report','wrong_index'])
def test_unrelated_publication_failure_stays_terminal(tmp_path,change):
    output,plan,rid,root,path,report=fixture(tmp_path)
    result=root/'phase-0-result.json';value=json.loads(result.read_bytes());pub=value['current_race_index_publish']
    if change=='published':pub['status']='PUBLISHED'
    elif change=='unsafe':pub['reason']='CURRENT_INDEX_PATH_UNSAFE'
    elif change=='wrong_report':pub['source_refresh_report_path']='/other/report.json'
    elif change=='wrong_index':pub['index_path']='/other/index.json'
    result.write_text(json.dumps(value));cp=json.loads((root/'phase-checkpoint.json').read_bytes())
    cp['phases'][0]['result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
    (root/'phase-checkpoint.json').write_text(json.dumps(cp))
    assert classify(plan['evidence_root'],rid) is None


def test_actual_missing_index_stays_pending(tmp_path):
    from race_collection.metadata_exclusion import metadata_exclusion_pending
    output,plan,record,retained=retained_owner_fixture(tmp_path)
    assert metadata_exclusion_pending(plan,owner.reference(retained),'b'*64,datetime.fromisoformat('2026-07-19T03:10:00+00:00'))


def test_actual_owner_poll_holds_then_allows_only_new_fresh_index(tmp_path,monkeypatch):
    from race_collection import synchronous_manual_capture as capture
    output,plan,record,retained=retained_owner_fixture(tmp_path)
    current=datetime.fromisoformat('2026-07-19T03:01:00+00:00')
    monkeypatch.setattr(owner,'now',lambda:current)
    generated=['2026-07-19T03:00:00+00:00']
    monkeypatch.setattr(capture,'bounded_current_race_index',lambda **k:SimpleNamespace(source_generated_at=generated[0]))
    monkeypatch.setattr(owner,'verify_captures',lambda *a:None)
    admissions=[]
    daily=object.__new__(owner.DailyOwner)
    daily.children={};daily.output=output;daily.plan=plan;daily.scope=SimpleNamespace(session=output)
    daily.state={'metadata_exclusion':owner.reference(retained),'refresh_failures':[]}
    daily.prepared={'allocation_ref':{'sha256':'b'*64}}
    daily.save=lambda:None
    daily.predictions=SimpleNamespace(tick=lambda **k:admissions.append(k['allow_dispatch']))
    daily.poll()
    assert admissions==[False]
    assert daily.state['forecast_admission_reason']=='NO_QUALIFIED_METADATA'
    generated[0]='2026-07-19T03:00:30+00:00'
    daily.poll()
    assert admissions==[False,True]
    assert daily.state['forecast_admission_ready'] is True
