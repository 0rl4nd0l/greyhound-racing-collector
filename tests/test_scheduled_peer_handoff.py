"""Generated full/odds wrappers and native observer across a real lock handoff."""
import hashlib
import json
import os
import signal
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from tests.test_freshness_campaign import make_campaign
from tests.test_refresh_shared_sportsbet_snapshot import fixture, access

ROOT = Path(__file__).resolve().parents[1]


def test_second_full_service_waits_for_actual_odds_child_then_completes(tmp_path, monkeypatch):
    from scripts import prepare_freshness_rehearsal as packaging
    from scripts.check_freshness_service import service_command
    from race_collection.live_freshness_contract import AttemptAllowance, FreshnessContract, digest
    from race_collection.live_phase_checkpoint import native_publication_lock
    from race_collection.freshness_rehearsal import native_observation
    from race_collection.synchronous_manual_capture import current_race_index_path
    from src.operator_ui.live_adapters import InstalledUnits

    gate = access(tmp_path)
    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(gate))
    http = fixture(tmp_path, 1)
    installed = tmp_path / 'installed'
    installed.mkdir()
    for name in (*packaging.UNITS, 'greyhound-operator-ui-r3.service'):
        (installed / name).write_text('synthetic original ' + name)
    db = tmp_path / 'empty-history.sqlite'
    sqlite3.connect(db).close()
    campaign = make_campaign(tmp_path / 'campaign')
    stamp = datetime.now(ZoneInfo('Australia/Melbourne'))
    package = tmp_path / 'package'
    packaging.prepare(output=package, start=stamp-timedelta(seconds=5), python=Path(sys.executable),
        db=db, lock=tmp_path / 'collector.lock', reconciliation_roots={}, installed_dir=installed,
        campaign_root=campaign.root, operational_predictions=True, observation_minutes=60)
    plan = json.loads((package / 'plan.json').read_bytes())
    accounting = dict(schema_version='freshness_attempt_reconciliation_v1', complete=True,
                      consumed=[], sources=[{'sha256': 'a'*64}])
    keys = ('profile','rehearsal_id','starts_at','ends_at','lock_path','evidence_root','db_path',
            'cleanup_seconds','max_capture_attempts','max_logical_requests','source_identity_sha256',
            'runtime_sha256','campaign_root','campaign_authorization_sha256','operational_predictions')
    contract = {key: plan[key] for key in keys}
    contract.update(schema_version='freshness_rehearsal_contract_v1',source_date=stamp.date().isoformat(),
                    reconciliation_sha256=digest(accounting))
    (package / 'contract.json').write_text(json.dumps(contract))
    scope = FreshnessContract(contract)
    AttemptAllowance(scope).initialize(accounting)
    campaign.begin(plan['rehearsal_id'],now=stamp,deadline=stamp+timedelta(minutes=100))

    # Delay only a fabricated HTTP response. Real wrappers, children, collector
    # locks, reports, state and observer remain unpatched production code.
    transport = tmp_path / 'transport'
    transport.mkdir()
    source = ROOT / 'tests/fixtures/shared_snapshot_transport'
    (transport / 'sitecustomize.py').write_bytes((source / 'sitecustomize.py').read_bytes())
    barrier = '''        if os.environ.get("SYNTHETIC_ODDS_BARRIER") and "/racing/" in url.path:
            import time
            Path(os.environ["SYNTHETIC_ODDS_BARRIER"]).touch()
            deadline = time.monotonic() + 25
            while not Path(os.environ["SYNTHETIC_ODDS_RELEASE"]).exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("synthetic_response_barrier_expired")
                time.sleep(0.02)
'''
    code = (source / 'snapshot_transport.py').read_text()
    assert code.count('        key = url.netloc + url.path') == 1
    (transport / 'snapshot_transport.py').write_text(code.replace('        key = url.netloc + url.path', barrier+'        key = url.netloc + url.path'))
    marker, release = tmp_path / 'odds-entered', tmp_path / 'odds-release'
    processes, logs = [], []
    paused_full_pid = None
    evidence = Path(plan['evidence_root'])
    runtime = evidence / 'shadow_autopilot_daemon_runtime'
    launcher = 'from scripts.check_freshness_service import deny_network; import os,sys; deny_network(); os.execv(sys.argv[1],sys.argv[1:])'

    def start(lane, run_id, invocation, *, pause=False):
        unit = 'shadow-autopilot.service' if lane == 'full' else 'shadow-autopilot-odds-capture.service'
        command, cwd, env = service_command(package / 'units' / unit)
        env.update(PYTHONPATH=os.pathsep.join((str(transport),str(package/'source'))),
                   GREYHOUND_SHARED_SNAPSHOT_FIXTURE=str(http),LIVE_ENHANCE_LIMIT='0',INVOCATION_ID=invocation)
        if pause:
            env.update(SYNTHETIC_ODDS_BARRIER=str(marker),SYNTHETIC_ODDS_RELEASE=str(release))
        log = (tmp_path / (run_id+'.log')).open('w')
        logs.append(log)
        child = subprocess.Popen([sys.executable,'-c',launcher,*command,'--run-id',run_id],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        processes.append(child)
        return child

    def report(run_id, lane):
        name = 'daemon_run_report.json' if lane=='full' else 'odds_capture_only_daemon_report.json'
        return evidence / ('shadow_autopilot_daemonization_v1_'+run_id) / name

    def wait_for(predicate, *, seconds=20):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if predicate():return
            time.sleep(0.02)
        raise AssertionError('synthetic service boundary was not reached')

    try:
        first=start('full','fixture_first_full','1'*32)
        assert first.wait(timeout=100)==0
        assert json.loads(report('fixture_first_full','full').read_bytes())['runtime_action']=='LIVE_COLLECTION_COMPLETE'
        seed=start('odds','fixture_seed_odds_capture','2'*32)
        assert seed.wait(timeout=100)==0
        assert json.loads(report('fixture_seed_odds_capture','odds').read_bytes())['runtime_action']=='LIVE_COLLECTION_COMPLETE'
        peer=start('odds','fixture_peer_odds_capture','3'*32,pause=True)
        wait_for(marker.exists)
        full=start('full','fixture_second_full','4'*32)
        full_report=report('fixture_second_full','full')
        def waiting():
            if not full_report.exists():return False
            return json.loads(full_report.read_bytes()).get('status')=='WAITING_LOCK_HELD'
        wait_for(waiting)
        assert peer.poll() is None and full.poll() is None
        peer_report=report('fixture_peer_odds_capture','odds')
        owner=json.loads(peer_report.read_bytes())['lock']
        assert owner['pid'] != peer.pid  # Service MainPID is its real wrapper.
        assert owner['run_id']=='fixture_peer_odds_capture'
        assert json.loads((runtime/'state.json').read_bytes())['last_run_id']=='fixture_first_full'
        unit_map=dict(zip(('full_service','full_timer','odds_service','odds_timer'),packaging.UNITS))
        raw={key:(package/'units'/name).read_bytes() for key,name in unit_map.items()}
        hashes={key:hashlib.sha256(value).hexdigest() for key,value in raw.items()}
        full_invocation, odds_invocation = '4'*32, '3'*32
        def observe(active, peer_active=None, *, peer_failed=False):
            peer_active = active if peer_active is None else peer_active
            current=datetime.now(ZoneInfo('Australia/Melbourne'))
            args={**raw,**{key+'_sha256':value for key,value in hashes.items()},
                  'observed_at':current,'working_directory':plan['source_root'],
                  'full_unit_name':'shadow-autopilot.service','odds_unit_name':'shadow-autopilot-odds-capture.service',
                  'full_active_state':'activating' if active else 'inactive','full_sub_state':'start' if active else 'dead',
                  'full_exec_main_pid':full.pid if active else 0,
                  'odds_active_state':'activating' if peer_active else ('failed' if peer_failed else 'inactive'),
                  'odds_sub_state':'start' if peer_active else ('failed' if peer_failed else 'dead'),
                  'odds_exec_main_pid':peer.pid if peer_active else 0,
                  'full_service_invocation_id':full_invocation,
                  'odds_service_invocation_id':odds_invocation}
            state=json.loads((runtime/'odds_capture_state.json').read_bytes())
            paths={'full_state':runtime/'state.json','odds_state':runtime/'odds_capture_state.json',
                   'full_report':full_report,'odds_report':peer_report,
                   'odds_refresh':Path(state['autopilot_output_dir'])/'odds_capture_refresh_report.json'}
            with native_publication_lock(evidence, exclusive=False):
                current=datetime.now(ZoneInfo('Australia/Melbourne'))
                args['observed_at']=current
                return native_observation(now=current,paths=paths,units=InstalledUnits(**args),
                    evidence_root=evidence,index_path=current_race_index_path(runtime/'odds_capture_state.json'),
                    authority={**plan,'unit_sha256':hashes},output=tmp_path/'observations')
        observed=observe(True)
        full_lane=next(lane for lane in observed['lanes'] if lane['lane']=='FULL_DAEMON')
        assert full_lane['status']=='WAITING_FOR_PEER', observed
        assert observed['collector_status']=='AVAILABLE/FRESH', observed
        # Stop only this synthetic child between lock polls. Its actual wrapper
        # remains alive; the peer completes naturally. This makes the real
        # released-peer / not-yet-acquired gap deterministic without code hooks.
        paused_full_pid=json.loads(full_report.read_bytes())['timing']['process_pid']
        assert paused_full_pid != full.pid
        os.kill(paused_full_pid, signal.SIGSTOP)
        release.touch()
        assert peer.wait(timeout=100)==0
        gap=observe(True, peer_active=False)
        assert next(lane for lane in gap['lanes'] if lane['lane']=='FULL_DAEMON')['status']=='WAITING_FOR_PEER', gap
        assert next(lane for lane in gap['lanes'] if lane['lane']=='ODDS_ONLY')['status']=='RECEIPT_READY', gap
        assert gap['collector_status']=='AVAILABLE/FRESH', gap
        os.kill(paused_full_pid, signal.SIGCONT)
        paused_full_pid=None
        assert full.wait(timeout=100)==0
        completed=json.loads(full_report.read_bytes())
        assert completed['runtime_action']=='LIVE_COLLECTION_COMPLETE'
        assert completed['timing']['lock_wait_seconds']>0
        assert json.loads((runtime/'state.json').read_bytes())['last_run_id']=='fixture_second_full'
        assert observe(False)['collector_status']=='AVAILABLE/FRESH'
        # Exercise the reverse service seam too: odds defers to a full daemon
        # whose lock-owning child differs from its service wrapper MainPID.
        marker.unlink()
        release.unlink()
        full_invocation, odds_invocation = '5'*32, '6'*32
        full=start('full','fixture_third_full',full_invocation,pause=True)
        full_report=report('fixture_third_full','full')
        wait_for(marker.exists)
        peer=start('odds','fixture_deferred_odds_capture',odds_invocation)
        peer_report=report('fixture_deferred_odds_capture','odds')
        assert peer.wait(timeout=100)==2
        deferred=json.loads(peer_report.read_bytes())
        assert deferred['runtime_action']=='DEFERRED_LOCK_HELD'
        active_full=json.loads(full_report.read_bytes())
        assert active_full['lock']['pid'] != full.pid
        assert deferred['deferred_lock_owner']['pid']==active_full['timing']['process_pid']
        reverse=observe(True,peer_active=False,peer_failed=True)
        assert next(lane for lane in reverse['lanes'] if lane['lane']=='ODDS_ONLY')['status']=='WAITING_FOR_PEER', reverse
        assert reverse['collector_status']=='AVAILABLE/FRESH', reverse
        release.touch()
        assert full.wait(timeout=100)==0
        assert json.loads(full_report.read_bytes())['runtime_action']=='LIVE_COLLECTION_COMPLETE'
        assert observe(False,peer_failed=True)['collector_status']=='AVAILABLE/FRESH'
        assert not (scope.session/'STOP.json').exists()
    finally:
        if paused_full_pid is not None:
            os.kill(paused_full_pid, signal.SIGCONT)
        release.touch()
        for child in processes:
            if child.poll() is None:
                child.wait(timeout=100)
        for log in logs:log.close()
