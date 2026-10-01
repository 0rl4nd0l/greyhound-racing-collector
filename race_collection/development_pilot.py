"""Finite, selected-only development session owner using the existing collector.

No timer, provider access or experiment is enabled by importing this module.
Configuration, authority and exported source are pinned by the installed service.
"""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from zoneinfo import ZoneInfo

from race_collection.development_examples import checked, read, put, race_key, stamp
from race_collection.development_source_authority import DATES, load_development_authority
from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest

ZONE = ZoneInfo('Australia/Melbourne')


def utc_now():
    return datetime.now(timezone.utc)


def reference(path):
    path = Path(path).resolve(strict=True)
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def selected_population(session, allocation_sha256):
    """Authenticate fixed first-six membership before any WIN reservation."""
    from race_collection.development_examples import selected_population as select, SELECTION_POLICY
    path = Path(session) / 'population.json'
    if Path(str(path)+'.failure.json').exists():
        raise ValueError('development_population_failed')
    population = json.loads(path.read_bytes())
    completion = json.loads(Path(str(path)+'.completion.json').read_bytes())
    day = Path(session).name
    intended, selected = select(population['observed_races'], day)
    freeze = stamp(population['frozen_at'])
    if (population.get('schema_version') != 'development_population_freeze_v1'
            or population.get('synthetic') is not False
            or population.get('allocation_sha256') != allocation_sha256
            or population.get('local_date') != day or day not in DATES
            or population.get('selection_policy') != SELECTION_POLICY
            or population.get('intended') != intended or population.get('selected_race_ids') != selected
            or completion.get('status') != 'POPULATION_FROZEN'
            or completion.get('population_sha256') != reference(path)['sha256']
            or freeze.astimezone(ZONE).strftime('%Y-%m-%dT%H:%M') != day+'T12:50'
            or stamp(completion['completed_at']).astimezone(ZONE).strftime('%Y-%m-%dT%H:%M') != day+'T12:50'
            or not 0 <= (freeze-stamp(population['source_observed_at'])).total_seconds() <= 300):
        raise ValueError('development_population_not_frozen')
    return [r for r in intended if r['race_id'] in selected]


def load_config(path, expected):
    value = read({'path':str(Path(path).absolute()), 'sha256':expected})
    if any(os.environ.get(name) for name in ('SYNTHETIC_DEVELOPMENT_CLOCK','FRESHNESS_FABRICATED_SOURCE','GREYHOUND_SHARED_SNAPSHOT_FIXTURE')):
        raise ValueError('fixture_environment_forbidden_for_authorized_runtime')
    authority = load_development_authority(value['pilot_campaign_authority'])
    allocation = read(value['allocation'])
    if (value.get('schema_version') != 'development_pilot_runtime_v1'
            or value.get('status') != 'AUTHORIZED'
            or value.get('authority_reference') != authority['authority_reference']
            or value['allocation']['sha256'] != authority['allocation_sha256']
            or allocation.get('status') != 'AUTHORIZED' or allocation.get('dates') != DATES
            or value.get('session_dates') != DATES
            or value.get('state_root') != authority['state_root']
            or value.get('prediction_root') != authority['prediction_root']
            or value.get('result_closure_at') != authority['result_closure_at']
            or value.get('max_result_operations') != 72
            or value.get('max_result_transport_requests') != 720
            or value.get('max_result_checks_per_race') != 3):
        raise ValueError('invalid_development_runtime_configuration')
    for key in ('state_root', 'prediction_root', 'campaign_root', 'source_root', 'python',
                'source_state', 'lock_path', 'history_database', 'installed_dir'):
        candidate=Path(value[key])
        if not candidate.is_absolute() or candidate.resolve() != candidate:
            raise ValueError('development_runtime_path_unsafe')
    if Path(value['source_root']) != Path(__file__).resolve().parents[1]:
        raise ValueError('development_runtime_source_root_changed')
    from scripts.run_freshness_rehearsal import verify_source_package
    # An installed archive is verified by its original source identity. A Git
    # release is verified by its committed manifest exported by the owner.
    verify_source_package(value['source_root'], value['source_identity_sha256'])
    identity=json.loads((Path(value['source_root'])/'SOURCE_IDENTITY.json').read_bytes())
    if identity['commit'] != value['source_commit']:
        raise ValueError('development_runtime_source_commit_changed')
    if str(Path(sys.executable)) != value['python']:
        raise ValueError('development_runtime_interpreter_changed')
    from race_collection.freshness_campaign import Campaign
    campaign=Campaign(value['campaign_root'], development_authority=value['pilot_campaign_authority'])
    if campaign.development != authority:
        raise ValueError('development_campaign_authority_changed')
    if set(value.get('session_packages',{})) != set(DATES):
        raise ValueError('development_session_packages_incomplete')
    for day, ref in value['session_packages'].items():
        plan=read(ref)
        if (plan.get('development_authority') != value['pilot_campaign_authority']
                or plan.get('prediction_root') != value['prediction_root']
                or plan.get('commit') != value['source_commit']):
            raise ValueError('development_session_package_changed')
    read(value['reconciliation_roots'])
    read(value['study_schedule'])
    return value


def status(config, now=None):
    current=(now or utc_now()).astimezone(ZONE)
    day=current.date().isoformat()
    root=Path(config['state_root'])
    sessions=[]
    for date in DATES:
        path=root/'sessions'/date/'terminal.json'
        sessions.append({'date':date, 'status':json.loads(path.read_bytes())['status'] if path.exists() else 'UNCONSUMED'})
    due=day in DATES and current.replace(hour=12,minute=40,second=0,microsecond=0) <= current < current.replace(hour=14,minute=40,second=0,microsecond=0)
    return {'schema_version':'development_pilot_status_v1', 'status':'SLOT_DUE' if due else 'NO_SLOT_DUE',
            'observed_at':current.isoformat(), 'sessions':sessions,
            'provider_operations_performed':0, 'synthetic':config.get('status')=='SYNTHETIC_FIXTURE'}


@contextmanager
def owner(config):
    root=Path(config['campaign_root'])
    with (root/'owner.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def require_study_priority(config, start, end):
    schedule=read(config['study_schedule'])
    # Identity/time metadata only. Frozen results and forecasts are never read.
    rows=schedule.get('slots', schedule.get('sessions'))
    if not isinstance(rows,list):
        raise ValueError('study_schedule_schema_unknown')
    for row in rows:
        if isinstance(row,str):
            begin=stamp(row)-timedelta(minutes=10)
            finish=stamp(row)+timedelta(minutes=schedule['session_minutes'])
            if begin<end and start<finish+timedelta(minutes=10):
                raise ValueError('study_schedule_has_priority')
            continue
        begin=row.get('prepares_at') or row.get('prepare_at') or row.get('starts_at')
        finish=row.get('ends_at')
        if not begin or not finish:
            raise ValueError('study_schedule_slot_schema_unknown')
        if stamp(begin)<end and start<stamp(finish)+timedelta(minutes=10):
            raise ValueError('study_schedule_has_priority')


def require_study_health(config, now):
    """Read scheduling/queue health counts only, never target results."""
    schedule=read(config['study_schedule'])
    if config.get('status')=='SYNTHETIC_FIXTURE':
        # Synthetic tests supply their own explicit health references too.
        schedule_health=read(config['synthetic_study_health'])
        result_health=read(config['synthetic_result_health'])
    else:
        study_root=Path(schedule['state_root'])
        canary=json.loads((study_root/'canary.json').read_bytes())
        if (canary.get('status')!='CANARY_STRUCTURALLY_VERIFIED'
                or canary.get('plan_sha256')!=schedule['comparison_plan_sha256']
                or canary.get('verified_predictions',0)<1 or canary.get('closed_results',0)<1):
            raise ValueError('study_first_session_gate_has_priority')
        schedule_health=json.loads((study_root/'health.json').read_bytes())
        binding=json.loads(Path(schedule['result_binding']).read_bytes())
        authority=read({'path':binding['authority'],'sha256':binding['authority_sha256']})
        result_health=json.loads((Path(authority['runtime']['state_root'])/'health.json').read_bytes())
    if (schedule_health.get('status') not in {'NO_SLOT_DUE','SESSION_COMPLETED'}
            or not 0 <= (now-stamp(schedule_health['at'])).total_seconds() <= 2700
            or result_health.get('status') not in {'CYCLE_COMPLETE','COLLECTOR_LOCK_BUSY','CAMPAIGN_OWNER_BUSY','SOURCE_OPERATION_BUSY'}
            or not 0 <= (now-stamp(result_health['at'])).total_seconds() <= 2700
            or result_health.get('counts',{}).get('RUNNING',0)
            or result_health.get('oldest_due') and stamp(result_health['oldest_due'])<=now):
        raise ValueError('study_recovery_or_results_have_priority')


class Collector:
    """Narrow live adapter. The same scheduling state machine uses a synthetic adapter in tests."""
    def __init__(self, config):
        self.config=config
        self.plan=None
        self.scope=None
        self.owned=None
        self.children_reaped=True
        self.lease_started=False

    @contextmanager
    def lock(self, session, operation):
        from race_collection.synchronous_manual_capture import acquire_collector_lock_no_steal, release_owned_collector_lock
        owner_handle=(Path(self.config['campaign_root'])/'owner.lock').open('a')
        fcntl.flock(owner_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock=acquire_collector_lock_no_steal(Path(self.config['lock_path']),
            run_id='development-'+session.name+'-'+operation, output_dir=session,
            phase='development_'+operation, acquisition_policy='approved_development_selected_only_v1')
        self.owned=lock
        try:
            yield lock
        finally:
            if self.children_reaped:
                release_owned_collector_lock(lock)
                self.owned=None
            owner_handle.close()

    def prepare(self, session, start):
        from race_collection.live_freshness_contract import verify_source_package
        from scripts.check_freshness_runtime import verify_runtime
        from race_collection.freshness_attempt_reconciliation import reconcile
        from scripts.run_freshness_rehearsal import execution_contract
        from race_collection.freshness_campaign import Campaign
        from utils.sportsbet_access import SportsbetAccess
        config=self.config
        require_study_health(config,utc_now())
        # Exactly one source lease per date, before preparation/discovery. All
        # child source operations must carry its validated authority tag.
        gate=SportsbetAccess(Path(config['source_state']))
        source=gate.read()
        baseline=config['source_baseline']
        if (source.get('phase') != 'OPEN' or source.get('active') is not None
                or digest(source.get('access_basis')) != baseline['access_basis_sha256']
                or digest(source.get('denials', [])) != baseline['denials_sha256']
                or source.get('recovery_attempts') != baseline['recovery_attempts']
                or digest(source.get('operating_policy')) != baseline['operating_policy_sha256']
                or utc_now().timestamp() < source.get('not_before',0)):
            raise ValueError('development_source_not_approved_open_baseline')
        with self.lock(session, 'prepare') as lock:
            self.plan=read(config['session_packages'][session.name])
            origin=Path(config['session_packages'][session.name]['path']).parent
            put(session/'package/plan.json',self.plan)
            for name,expected in [('runtime-identity.json',self.plan['runtime_sha256']),('operational-retention.json',self.plan['operational_predictions']['retention_config_sha256'])]:
                raw=(origin/name).read_bytes()
                if hashlib.sha256(raw).hexdigest()!=expected:
                    raise ValueError('development_package_support_changed')
                target=session/'package'/name
                with target.open('xb') as stream:
                    stream.write(raw);stream.flush();os.fsync(stream.fileno())
            verify_source_package(self.plan['source_root'],self.plan['source_identity_sha256'])
            verify_runtime(self.plan)
            if self.plan['starts_at'] != start.isoformat():
                raise ValueError('development_package_start_changed')
            accounting=reconcile(roots=read(config['reconciliation_roots']),db_path=Path(self.plan['db_path']),
                source_date=session.name,lock_path=Path(config['lock_path']),owner_run_id='development-'+session.name+'-prepare')
            contract=execution_contract(self.plan,accounting)
            put(session/'package/contract.json',contract)
            self.scope=FreshnessContract(contract)
            from race_collection.live_execution import configure_profile_execution
            configure_profile_execution(session/'package/contract.json')
            AttemptAllowance(self.scope).initialize(accounting)
            campaign=Campaign.from_scope(contract)
            campaign.begin(self.plan['rehearsal_id'],now=utc_now(),deadline=start+timedelta(seconds=7200))
            self.lease_started=True
            gate.authorize_diagnostic(reference=config['authority_reference']+':slot:'+session.name,
                expected_sha256=hashlib.sha256(gate.path.read_bytes()).hexdigest(),
                expires_at=(start+timedelta(seconds=7200)).timestamp(),max_operations=48,
                rationale='Approved separate selected-only development pilot',
                development_authority=config['pilot_campaign_authority'],development_slot=session.name)
        return self.plan

    def command(self, kind, run_id, item=None):
        p=self.plan
        command=[p['python'],str(Path(p['source_root'])/'scripts/shadow_autopilot_v1.py'),
            '--run-id',run_id,'--evidence-root',p['evidence_root'],'--collector-lock-path',p['lock_path'],
            '--current-race-index-state-path',str(Path(p['evidence_root'])/'shadow_autopilot_daemon_runtime/odds_capture_state.json'),
            '--db',p['db_path'],'--skip-shadow-run','--require-safe-refresh-metadata',
            '--refresh-command-mode','python','--current-time',utc_now().astimezone(ZONE).isoformat(),
            '--collection-phase',kind,'--step-timeout-seconds','90',
            '--live-freshness-profile','bounded80-v1','--live-freshness-contract',str(Path(self.config['state_root'])/'sessions'/self.scope.value['source_date']/'package/contract.json')]
        if kind=='refresh':
            command += ['--days-ahead','0','--refresh-limit','6','--min-minutes','6','--max-minutes','100']
        else:
            command += ['--skip-refresh','--skip-primary-refresh','--input-dir',item['input_dir'],
                '--autonomous-odds-capture-limit','1','--enable-autonomous-odds-capture',
                '--execute-autonomous-odds-capture','--allow-auto-scrape-odds',
                '--live-capture-reservation',item['reservation_path']]
        return command

    def phase(self, session, kind, name, item=None):
        from scripts.shadow_autopilot_daemon import run_command
        config=self.config
        os.environ['GREYHOUND_DEVELOPMENT_AUTHORITY_SHA256']=config['pilot_campaign_authority']['sha256']
        os.environ['GREYHOUND_SPORTSBET_ACCESS_STATE']=config['source_state']
        with self.lock(session,name):
            require_study_health(config,utc_now())
            self.scope.admit(utc_now(),seconds=90 if kind=='refresh' else 155)
            if self.scope.campaign:
                with self.scope.campaign.ledger() as ledger:
                    if ledger.get('source_holds'):raise ValueError('campaign_source_hold')
            self.children_reaped=False
            result=run_command(name=name,command=self.command(kind,'development-'+session.name+'-'+name,item),
                output_dir=session/name,timeout_seconds=90 if kind=='refresh' else 155,
                cwd=Path(self.plan['source_root']),wait_for_descendants=True)
            self.children_reaped=True
            try:
                report=json.loads((session/name/'logs'/(name+'.stdout.txt')).read_bytes())
            except (OSError,ValueError):
                report={'status':'FAIL','reason':'phase_output_missing'}
            report['step']=result
            if kind=='capture':
                from race_collection.operational_prediction import classify_unready_capture
                unready=classify_unready_capture(item['reservation_path'],report,Path(self.plan['evidence_root']),Path(self.plan['source_root']))
                if unready:report['operational_capture_outcome']=unready
                AttemptAllowance(self.scope).finish(item['reservation_path'],report)
            if result['status']!='PASS' or report.get('status')!='PASS':
                raise ValueError('development_'+kind+'_failed')
            if kind=='capture' and report.get('autonomous_live_odds_capture_status',{}).get('status')!='AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED':
                raise ValueError('development_capture_unqualified')
        return report

    def refresh(self,session,name):
        return self.phase(session,'refresh',name)

    def freeze(self,session):
        from race_collection.development_examples import freeze_population
        evidence=Path(self.plan['evidence_root'])
        return freeze_population(evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json',
            evidence,self.config['allocation']['path'],self.config['allocation']['sha256'],session/'population.json')

    def task(self,selected):
        from race_collection.synchronous_manual_capture import bounded_current_race_index, runner_set_sha256
        from scripts.autonomous_live_odds_capture import build_capture_plan
        evidence=Path(self.plan['evidence_root'])
        view=bounded_current_race_index(current_time=utc_now(),timeout_seconds=5,
            index_path=evidence/'shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json',
            evidence_root=evidence,max_age_seconds=300,return_verified_view=True)
        rows=[r for r in view.races if r['race_id']==selected['race_id']]
        if len(rows)!=1:raise ValueError('selected_race_unavailable')
        row=rows[0]
        if (row['race_url']!=selected['url'] or row['jump_datetime']!=selected['jump_at']
                or row['source_native_race_id']!=selected['source_native_race_id']
                or row['runners']!=selected['runners']):
            raise ValueError('selected_identity_or_field_changed')
        refresh=json.loads((evidence/view.source_refresh_report_path).read_bytes())
        coverage=refresh['sidecar_metadata_coverage']['races']
        match=[r for r in coverage if r.get('race_url')==row['race_url'] and r.get('csv_path')]
        if len(match)!=1:raise ValueError('selected_metadata_unavailable')
        directory=Path(match[0]['csv_path']).parent.resolve()
        directory.relative_to(evidence)
        if len(list(directory.glob('*.csv')))!=1:raise ValueError('selected_input_directory_not_single_race')
        plan=build_capture_plan([directory],current_time=utc_now().astimezone(ZONE))
        native=plan['races']
        if len(native)!=1 or native[0].get('status')!='READY_TO_CAPTURE' or native[0].get('capture_window_minutes')!=10:
            raise ValueError('selected_t10_window_not_qualified')
        task={'kind':'capture','race_id':row['race_id'],'race_id_aliases':row.get('race_id_aliases',[row['race_id']]),
              'input_dir':str(directory),'packet_sha256':view.packet_sha256,'capture_window_minutes':10,
              'development_active_runners':row['runners'],
              'capture_runner_set_sha256':runner_set_sha256(native[0]['expected_runners']),
              'input_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
                             if p.is_file() and (p.suffix=='.csv' or p.name.endswith('.csv.metadata.json'))},
              'race_identity':{key:row[key] for key in ('race_id','race_url','jump_datetime','source_native_race_id','runner_set_sha256')}}
        AttemptAllowance.check_window(task,now=utc_now(),required_seconds=50)
        if (stamp(selected['jump_at'])-utc_now()).total_seconds()<300:
            raise ValueError('insufficient_prediction_margin')
        return task

    def consumed(self,session,race_id):
        if self.scope is None:return False
        with self.scope.campaign.ledger() as ledger:
            return any(r['race_id']==race_id and r.get('development_authority_sha256')==self.config['pilot_campaign_authority']['sha256'] for r in ledger['attempts'])

    def capture(self,session,selected):
        task=self.task(selected)
        allowance=AttemptAllowance(self.scope)
        claim=allowance.reserve(task,now=utc_now())
        put(session/'attempts'/(hashlib.sha256(selected['race_id'].encode()).hexdigest()+'.json'),
            {'race_id':selected['race_id'],'claim':str(claim),'status':'CONSUMED','at':utc_now().isoformat()})
        task['reservation_path']=str(claim)
        self.phase(session,'capture','capture-'+hashlib.sha256(selected['race_id'].encode()).hexdigest()[:12],task)
        return claim

    def predict(self,session,claim):
        from scripts.shadow_autopilot_daemon import run_command
        self.children_reaped=False
        outcome=run_command(name='prediction-'+Path(claim).parent.name,
            command=[self.plan['python'],'-B','-m','race_collection.operational_prediction',
                     str(session/'package/plan.json'),str(claim)],
            output_dir=session/'prediction-workers',timeout_seconds=210,
            cwd=Path(self.plan['source_root']),wait_for_descendants=True)
        self.children_reaped=True
        if outcome['status']!='PASS':raise ValueError('development_prediction_failed')
        item=json.loads(Path(claim).read_bytes())['item']
        terminal=Path(self.config['prediction_root'])/'races'/hashlib.sha256(item['race_id'].encode()).hexdigest()/'terminal.json'
        value=json.loads(terminal.read_bytes())
        if value['status']!='PREDICTION_READY':raise ValueError('development_prediction_not_verified')
        return value,terminal

    def seal(self,session,selected,accounting,verification):
        from race_collection.development_examples import seal
        race_id=selected['race_id']; key=hashlib.sha256(race_id.encode()).hexdigest()
        directory=session/'examples'/key
        directory.mkdir(parents=True,mode=0o700,exist_ok=True)
        root=Path(self.config['prediction_root'])/'bundles'
        index=json.loads((root/'prediction_bundle_index_v1.json').read_bytes())
        entries=[r for r in index['entries'] if r.get('job_id')==verification[0]['job_id']]
        if len(entries)!=1:raise ValueError('development_prediction_index_missing')
        put(directory/'opportunities.json',accounting)
        access={'schema_version':'development_access_v1','status':'AUTHORIZED_DEVELOPMENT','enabled':True,
            'allocation':self.config['allocation'],'population':reference(session/'population.json'),
            'population_completion':reference(session/'population.json.completion.json'),
            'opportunities':reference(directory/'opportunities.json'),
            'members':{race_id:{'race_key':race_key(race_id),'jump_at':selected['jump_at'],
                'bundle_root':str(root),'entry':entries[0],'verification':reference(verification[1]),
                'history_access':'machine_only_pre_target','target_label_access':'separate_result_authority'}}}
        put(directory/'access.json',access)
        access_ref=reference(directory/'access.json')
        seal(access_ref['path'],access_ref['sha256'],race_id,directory/'example')
        output=directory/'example'
        ready={'schema_version':'development_pilot_capture_ready_v1','race_id':race_id,'race_key':race_key(race_id),
            'jump_at':selected['jump_at'],'access_path':access_ref['path'],'access_sha256':access_ref['sha256'],
            'example_dir':str(output),'pre_result_sha256':reference(output/'pre_result.json')['sha256'],
            'completion_sha256':reference(output/'completion.json')['sha256'],'job_id':verification[0]['job_id'],
            'prediction_entry':entries[0],'published_at':utc_now().isoformat(),
            'allocation_sha256':self.config['allocation']['sha256']}
        put(Path(self.config['state_root'])/'ready'/(key+'.json'),ready)
        return ready

    def close(self,session):
        if not self.children_reaped or self.owned is not None:
            raise ValueError('development_child_lifetime_unknown_lease_retained')
        if self.scope is not None and self.lease_started:
            from race_collection.operational_prediction import require_completed_lifetimes
            require_completed_lifetimes(session/'package',self.scope.campaign)
            self.scope.campaign.close(self.plan['rehearsal_id'],now=utc_now())


def run_session(config, *, clock=None, sleep=None, collector=None):
    """One literal date. Restart never retries a consumed session or substitutes a race."""
    synthetic=config.get('status')=='SYNTHETIC_FIXTURE'
    if not synthetic and any(x is not None for x in (clock,sleep,collector)):
        raise ValueError('synthetic_adapter_forbidden_for_authorized_runtime')
    if synthetic and collector is None:
        raise ValueError('synthetic_runtime_requires_offline_adapter')
    clock=clock or utc_now; sleep=sleep or time.sleep
    os.umask(0o077)
    current=clock().astimezone(ZONE);day=current.date().isoformat()
    if day not in DATES:return status(config,current)
    start=current.replace(hour=12,minute=40,second=0,microsecond=0)
    if current<start:return status(config,current)
    root=Path(config['state_root']);session=root/'sessions'/day
    session.mkdir(parents=True,exist_ok=True,mode=0o700)
    if session.stat().st_mode & 0o077:raise ValueError('development_output_not_private')
    if (session/'terminal.json').exists():return json.loads((session/'terminal.json').read_bytes())
    if (session/'started.json').exists():
        # A clean terminal handles a restart. Otherwise no acquisition resumes;
        # the unknown prior process lifetime retains its already charged lease.
        return {'status':'INTERRUPTED_REQUIRES_RECONCILIATION','session':day,'provider_operations_performed':0,'synthetic':synthetic}
    if current>=start+timedelta(minutes=1):
        result={'status':'SKIPPED_LATE_PREPARATION','session':day,'at':current.isoformat(),'synthetic':synthetic}
        put(session/'terminal.json',result);return result
    require_study_priority(config,start,start+timedelta(seconds=7200))
    collector=collector or Collector(config)
    with (session/'session.lock').open('a') as session_mutex:
        fcntl.flock(session_mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(session/'started.json',{'status':'STARTED','session':day,'at':current.isoformat(),'pid':os.getpid(),'synthetic':synthetic})
        result={'status':'FAILED','session':day,'synthetic':synthetic}
        accounting=None
        def wait_until(target):
            while clock()<target:sleep(min(30,max(0,(target-clock()).total_seconds())))
        cancelled=False
        def interrupted(signum,frame):
            nonlocal cancelled
            cancelled=True
            raise InterruptedError('development_session_interrupted')
        old={sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            collector.prepare(session,start)
            wait_until(start+timedelta(minutes=8))
            collector.refresh(session,'preparation-refresh')
            wait_until(start+timedelta(minutes=10))
            collector.freeze(session)
            selected=selected_population(session,config['allocation']['sha256'])
            population=json.loads((session/'population.json').read_bytes())
            accounting={'schema_version':'development_opportunities_v1','complete':True,
                'population_sha256':reference(session/'population.json')['sha256'],
                'opportunities':[{'race_id':r['race_id'],'race_key':r['race_key'],'qualified':False,
                    'attempt_consumed':False,'disposition':'PENDING' if r['race_id'] in population['selected_race_ids'] else 'EXCLUDED',
                    'reason':'selected_before_WIN_qualification' if r['race_id'] in population['selected_race_ids'] else 'fixed_first_six_quota'}
                    for r in population['intended']]}
            for number,race in enumerate(selected):
                row=next(r for r in accounting['opportunities'] if r['race_id']==race['race_id'])
                jump=stamp(race['jump_at'])
                wait_until(max(start+timedelta(minutes=20),jump-timedelta(minutes=11)))
                try:
                    if clock()>jump-timedelta(minutes=5):raise ValueError('selected_window_missed_no_substitution')
                    collector.refresh(session,'race-refresh-'+str(number))
                    wait_until(jump-timedelta(minutes=10))
                    claim=collector.capture(session,race)
                    row.update(attempt_consumed=True,qualified=True,disposition='ATTEMPTED',reason='exact_selected_reservation')
                    verification=collector.predict(session,claim)
                    row.update(disposition='VERIFIED_FORECAST',reason='complete_qualified_receipt')
                    collector.seal(session,race,json.loads(json.dumps(accounting)),verification)
                    row['example_status']='SEALED'
                except Exception as exc:
                    attempted=session/'attempts'/(hashlib.sha256(race['race_id'].encode()).hexdigest()+'.json')
                    spent=collector.consumed(session,race['race_id'])
                    if row['disposition']=='VERIFIED_FORECAST':
                        row.update(reason='verified_forecast_example_seal_failed',example_status='FAILED')
                    else:
                        row.update(attempt_consumed=spent,disposition='FAILED' if spent else 'EXCLUDED',reason=type(exc).__name__+':'+str(exc)[:120])
                    put(session/'race-dispositions'/(str(number)+'.json'),row)
                    # Global source/campaign STOP prevents all following work;
                    # ordinary per-race qualification failure does not substitute.
                    if cancelled:raise InterruptedError('development_session_interrupted')
                    if not collector.children_reaped:raise
                    if collector.scope:
                        if (collector.scope.session/'STOP.json').exists():raise
                        with collector.scope.campaign.ledger() as ledger:
                            if ledger.get('source_holds'):raise
            result.update(status='COMPLETE',intended=len(accounting['opportunities']),selected=len(selected),
                attempts=sum(r['attempt_consumed'] for r in accounting['opportunities']),
                verified_forecasts=sum(r['disposition']=='VERIFIED_FORECAST' for r in accounting['opportunities']),
                examples_sealed=sum(r.get('example_status')=='SEALED' for r in accounting['opportunities']))
        except BaseException as exc:
            result.update(status='ABORTED' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'FAILED',
                reason=type(exc).__name__+':'+str(exc)[:160])
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)
            if accounting is not None:
                for row in accounting['opportunities']:
                    if row['disposition']=='PENDING':row.update(disposition='EXCLUDED',reason='session_ended_before_attempt')
                put(session/'opportunities.final.json',accounting)
            try:collector.close(session)
            except Exception as exc:result.update(status='HELD_FOR_RECONCILIATION',cleanup_reason=type(exc).__name__)
            result['completed_at']=clock().isoformat()
            put(session/'terminal.json',result)
        return result
