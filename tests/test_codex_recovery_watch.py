"""The public watcher/worker interface never touches services in these tests."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from race_collection import codex_recovery_watch as watch


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def service(**changes):
    return {**dict(ActiveState='inactive',SubState='dead',MainPID='0',Result='success',
        ExecMainStatus='0',InvocationID='a'*32,WorkingDirectory='/tmp',ExecStart='collector',ControlGroup='/collector'),**changes}


class Host:
    def __init__(self):
        self.collector=service(ActiveState='failed',Result='exit-code',ExecMainStatus='78')
        self.worker=service()
        self.starts=0;self.agents=[];self.exit_code=0;self.failure=None;self.after=None
    def show(self,cfg,unit):
        return dict(self.collector if unit==watch.COLLECTOR_UNIT else self.worker)
    def start(self,cfg):
        self.starts+=1
        if self.failure=='start':raise subprocess.CalledProcessError(1,['fake'])
        self.worker=service(ActiveState='activating')
    def agent(self,cfg,directory,prompt,mode,working):
        self.agents.append(dict(mode=mode,prompt=prompt,working=working))
        if self.failure=='agent':raise OSError('sensitive failure must not be echoed')
        (directory/'last-message.txt').write_text('CODEX_RECOVERY_EXERCISE_OK')
        events=[{'type':'thread.started'},{'type':'item.completed','item':{'type':'command_execution','exit_code':0}},{'type':'turn.completed'}]
        (directory/'events.private.jsonl').write_text('\n'.join(map(json.dumps,events)))
        if self.after:self.after(cfg)
        return self.exit_code


@pytest.fixture
def case(tmp_path):
    runbook=tmp_path/'runbook.md';runbook.write_text('Exact approved runbook')
    cfg=dict(schema_version='codex_recovery_watch_v1',enabled=True,state_root=str(tmp_path/'watch'),
        runtime_root=str(tmp_path/'runtime'),collector_unit=watch.COLLECTOR_UNIT,recovery_unit=watch.RECOVERY_UNIT,
        codex_binary='/usr/bin/true',systemctl_binary='/usr/bin/true',runbook_path=str(runbook),
        runbook_sha256=hashlib.sha256(runbook.read_bytes()).hexdigest(),exercise_working_directory=str(tmp_path),
        debounce_seconds=30,health_max_age_seconds=120,codex_model='gpt-6.1-sol',codex_reasoning_effort='high')
    root=Path(cfg['runtime_root']);package=root/'days/today/native'
    write(root/'current-day.json',{'output':str(package)})
    write(root/'health.json',{'status':'HOLD','output':str(package),'at':datetime.now(timezone.utc).isoformat()})
    return cfg,Host(),package


def trigger(cfg,host):
    assert watch.check(cfg,host,clock=lambda:0)['status']=='DEBOUNCING'
    return watch.check(cfg,host,clock=lambda:30)


@pytest.mark.parametrize('change',[{'ActiveState':'active'},{'ActiveState':'activating'},
    {'ActiveState':'deactivating'},{'MainPID':'42'}])
def test_live_or_remaining_main_pid_never_launches(case,change):
    cfg,host,package=case;host.collector.update(change)
    assert watch.check(cfg,host)['status']=='RUNNING'
    assert host.starts==0


@pytest.mark.parametrize('status',['PAUSED','DAY_ENDED'])
def test_clean_expected_stop_does_not_launch(case,status):
    cfg,host,package=case;host.collector=service()
    write(Path(cfg['runtime_root'])/'health.json',{'status':status,'output':str(package)})
    assert watch.check(cfg,host)['status']=='EXPECTED_STOP'
    assert host.starts==0


def test_stale_pause_cannot_mask_new_fault(case):
    cfg,host,package=case
    write(Path(cfg['runtime_root'])/'health.json',{'status':'PAUSED','output':str(package)})
    assert trigger(cfg,host)['action_mode']=='repair'


def test_debounce_transient_reset_and_stable_fault(case):
    cfg,host,package=case
    assert watch.check(cfg,host,clock=lambda:0)['status']=='DEBOUNCING'
    assert watch.check(cfg,host,clock=lambda:29)['status']=='DEBOUNCING'
    host.collector['ActiveState']='active'
    assert watch.check(cfg,host,clock=lambda:30)['status']=='RUNNING'
    host.collector['ActiveState']='failed'
    assert watch.check(cfg,host,clock=lambda:31)['status']=='DEBOUNCING'
    assert watch.check(cfg,host,clock=lambda:61)['status']=='SPAWN_REQUESTED'
    assert host.starts==1


def test_incident_deduplicates_across_health_changes_and_failed_runner(case):
    cfg,host,package=case
    result=trigger(cfg,host);host.worker=service()
    write(Path(cfg['runtime_root'])/'health.json',{'status':'HOLD','reason':'new heartbeat','output':str(package)})
    again=watch.check(cfg,host,clock=lambda:90)
    assert again['incident_id']==result['incident_id'] and again['incident_status']=='RUNNER_LOST'
    assert host.starts==1
    assert watch.check(cfg,host,clock=lambda:120)['incident_status']=='RUNNER_LOST'


@pytest.mark.parametrize('health',['missing','malformed'])
def test_missing_or_malformed_health_unknown_stop_only_diagnoses(case,health):
    cfg,host,package=case;host.collector=service()
    p=Path(cfg['runtime_root'])/'health.json'
    if health=='missing':p.unlink()
    else:p.write_text('not json')
    assert trigger(cfg,host)['action_mode']=='diagnostic_only'
    assert watch.run_incident(cfg,host)['status']=='DIAGNOSED_ONLY'
    assert host.agents[0]['mode']=='diagnostic_only'


def test_matching_halt_allows_repair_but_foreign_halt_does_not(case):
    cfg,host,package=case;host.collector=service()
    write(package/'HALT.json',{'reason':'exact failure'})
    assert watch._mode(watch.snapshot(cfg,host))=='repair'
    write(Path(cfg['runtime_root'])/'health.json',{'status':'HOLD','output':'/foreign'})
    assert watch._mode(watch.snapshot(cfg,host))=='diagnostic_only'


@pytest.mark.parametrize('which',['disabled','hold'])
def test_maintenance_interlock_before_check_and_before_worker(case,which):
    cfg,host,package=case
    trigger(cfg,host)
    if which=='disabled':cfg['enabled']=False
    else:write(Path(cfg['state_root'])/'operator-hold.json',{'reason':'maintenance'})
    assert watch.check(cfg,host)['status']=='SUPPRESSED'
    assert watch.run_incident(cfg,host)['status']=='SUPPRESSED'
    assert not host.agents


@pytest.mark.parametrize('name,expected',[('watch.lock','WATCH_BUSY'),('recovery.lock','RECOVERY_BUSY')])
def test_parallel_lock_does_not_spawn(case,name,expected):
    cfg,host,package=case
    if name=='recovery.lock':watch.check(cfg,host,clock=lambda:0)
    with watch._lock(Path(cfg['state_root'])/name):
        assert watch.check(cfg,host,clock=lambda:30)['status']==expected
    assert host.starts==0


@pytest.mark.parametrize('failure,status',[('start','SPAWN_FAILED'),('agent','RUNNER_FAILED')])
def test_failure_is_terminal_and_never_retried(case,failure,status):
    cfg,host,package=case;host.failure=failure
    result=trigger(cfg,host)
    if failure=='agent':result=watch.run_incident(cfg,host)
    assert result['status']==status
    host.worker=service()
    assert watch.check(cfg,host,clock=lambda:100)['incident_status']==status
    assert host.starts==1
    assert 'sensitive' not in json.dumps(result)


def test_agent_zero_exit_is_not_recovery_evidence(case):
    cfg,host,package=case;trigger(cfg,host)
    assert watch.run_incident(cfg,host)['status']=='RECOVERY_UNVERIFIED'
    assert watch.run_incident(cfg,host)['status']=='INCIDENT_ALREADY_CONSUMED'


def test_actual_current_collection_and_fresh_health_required(case):
    cfg,host,package=case;trigger(cfg,host)
    def after(cfg):
        host.collector.update(ActiveState='active',SubState='running',MainPID='999')
        write(Path(cfg['runtime_root'])/'health.json',{'status':'ACTIVE_COLLECTION',
            'output':str(package),'at':datetime.now(timezone.utc).isoformat(),'forecast_admission_ready':True})
    host.after=after
    result=watch.run_incident(cfg,host)
    assert result['status']=='COLLECTOR_RUNNING_OBSERVED'
    assert result['collector_running_observed'] is True


@pytest.mark.parametrize('change',['identity','healthy','config'])
def test_revalidate_immediately_before_agent(case,change):
    cfg,host,package=case;trigger(cfg,host)
    if change=='identity':host.collector['InvocationID']='b'*32
    elif change=='healthy':host.collector['ActiveState']='active'
    else:cfg['codex_model']='different'
    assert watch.run_incident(cfg,host)['status'] in ('SUPPRESSED','COLLECTOR_ALREADY_RUNNING')
    assert not host.agents


def test_worker_crash_after_collector_restart_is_recorded(case):
    cfg,host,package=case;trigger(cfg,host)
    pointer=json.loads((Path(cfg['state_root'])/'incident.json').read_text())
    watch._status(Path(pointer['directory']),'RUNNING')
    host.worker=service();host.collector['ActiveState']='active'
    assert watch.check(cfg,host)['status']=='RUNNING'
    assert json.loads((Path(pointer['directory'])/'status.json').read_text())['status']=='RUNNER_LOST'


def test_exercise_uses_same_worker_read_only_outside_incidents(case):
    cfg,host,package=case
    result=watch.run_incident(cfg,host,exercise=True)
    assert result['status']=='EXERCISE_COMPLETE' and host.starts==0
    assert Path(result['exercise_directory']).parent==Path(cfg['state_root'])/'exercises'
    assert host.agents[0]['mode']=='exercise'
    assert not (Path(cfg['state_root'])/'incidents').exists()


def test_actual_command_is_pinned_readonly_or_repair_without_real_execution(case,monkeypatch):
    cfg,host,package=case
    commands=[]
    def fake(command,**kwargs):
        commands.append((command,kwargs));return subprocess.CompletedProcess(command,0)
    monkeypatch.setattr(subprocess,'run',fake)
    for mode in ('repair','diagnostic_only','exercise'):
        directory=Path(cfg['state_root'])/mode;directory.mkdir(parents=True)
        assert watch.Host().agent(cfg,directory,'private prompt',mode,str(package))==0
        command,kwargs=commands[-1]
        assert command[:3]==[cfg['codex_binary'],'exec','--ignore-user-config']
        assert command[command.index('--sandbox')+1]==('danger-full-access' if mode=='repair' else 'read-only')
        assert '--model' in command and 'model_reasoning_effort="high"' in command
        assert kwargs['input']==b'private prompt' and 'timeout' not in kwargs
        assert (directory/'events.private.jsonl').stat().st_mode & 0o777==0o600


def test_configuration_exact_runbook_and_runtime_isolation(case,tmp_path):
    cfg,host,package=case;p=tmp_path/'config.json';write(p,cfg)
    assert watch.load_config(p)==cfg
    Path(cfg['runbook_path']).write_text('changed')
    with pytest.raises(ValueError,match='runbook_changed'):watch.load_config(p)
    cfg['state_root']=cfg['runtime_root'];write(p,cfg)
    with pytest.raises(ValueError,match='overlap'):watch.load_config(p)


def test_outside_runtime_pointer_is_not_followed(case):
    cfg,host,package=case
    write(Path(cfg['runtime_root'])/'current-day.json',{'output':'/etc'})
    observation=watch.snapshot(cfg,host)
    assert observation['package'] is None and observation['halt'] is None


def test_exercise_nonzero_exit_preserved_without_last_message(case):
    cfg,host,package=case
    def rejected(*args):return 2
    host.agent=rejected
    result=watch.run_incident(cfg,host,exercise=True)
    assert result['status']=='AGENT_FAILED' and result['agent_exit_code']==2
    assert json.loads((Path(result['exercise_directory'])/'agent-exit.json').read_text())['exit_code']==2


def test_old_package_pause_cannot_hide_current_unknown_stop(case):
    cfg,host,package=case;host.collector=service()
    write(Path(cfg['runtime_root'])/'health.json',{'status':'PAUSED','output':'/old/package'})
    assert trigger(cfg,host)['action_mode']=='diagnostic_only'


def test_incomplete_incident_after_crash_is_preserved_not_relaunched(case):
    cfg,host,package=case
    identity=watch._identity(watch.snapshot(cfg,host))
    directory=Path(cfg['state_root'])/'incidents'/identity
    write(directory/'incident.json',{'incident_id':identity})
    result=watch.check(cfg,host)
    assert result['incident_status']=='RUNNER_LOST' and host.starts==0


def test_second_worker_cannot_enter_while_recovery_lock_held(case):
    cfg,host,package=case;trigger(cfg,host)
    with watch._lock(Path(cfg['state_root'])/'recovery.lock'):
        assert watch.run_incident(cfg,host)['status']=='RECOVERY_BUSY'
    assert not host.agents


def test_generic_failed_service_without_native_evidence_only_diagnoses(case):
    cfg,host,package=case;host.collector['ExecMainStatus']='9'
    assert trigger(cfg,host)['action_mode']=='diagnostic_only'


@pytest.mark.parametrize('change',['resumed','hold','new_fault'])
def test_rechecks_current_stop_and_hold_immediately_before_service_start(case,change):
    cfg,host,package=case
    original=host.show;reads=0
    def changing_show(cfg,unit):
        nonlocal reads
        if unit==watch.COLLECTOR_UNIT:
            reads+=1
            if reads==3:
                if change=='resumed':host.collector['ActiveState']='active'
                elif change=='hold':write(Path(cfg['state_root'])/'operator-hold.json',{'reason':'maintenance'})
                else:host.collector['InvocationID']='b'*32
        return original(cfg,unit)
    host.show=changing_show
    assert trigger(cfg,host)['status']=='SUPPRESSED'
    assert host.starts==0
