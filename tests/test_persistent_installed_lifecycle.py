"""Exact CLI process lifecycle under kernel network denial and fixture transport.

The clean temporary checkout keeps the real installed config/source loader. The
native-package builder has its own real-contract tests; this harness supplies
those prepared packages at that seam and replaces only provider discovery/time.
"""
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import pytest

from race_collection import persistent_native as native
from race_collection.live_phase_checkpoint import atomic_json
from race_collection.persistent_authority import stamp
from tests.test_persistent_native import backend
from utils.sportsbet_access import SportsbetAccess

ROOT=Path(__file__).resolve().parents[1]


def wait_for(predicate, process=None, timeout=20):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():return
        if process is not None and process.poll() is not None:
            raise AssertionError('CLI exited before expected fixture transition')
        time.sleep(.05)
    raise AssertionError('fixture lifecycle transition timed out')


def read(path):
    try:return json.loads(Path(path).read_bytes())
    except (FileNotFoundError,json.JSONDecodeError):return {}


def clean_checkout(path):
    path.mkdir()
    # Include pending coordinator integration, while keeping fixture Git/source
    # identity real rather than mocking git or bypassing load_config().
    candidates=list(ROOT.glob('*.py'))
    for directory in ('race_collection','scripts','utils','src','config','configs','accuracy_program'):
        candidates.extend((ROOT/directory).rglob('*.py'))
    for source in candidates:
        if '__pycache__' in source.parts:continue
        target=path/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
    subprocess.run(['git','init','-q',str(path)],check=True)
    subprocess.run(['git','-C',str(path),'add','.'],check=True)
    subprocess.run(['git','-C',str(path),'-c','user.name=Lifecycle Fixture','-c','user.email=fixture@invalid',
                    '-c','core.hooksPath=/dev/null','commit','-qm','Synthetic isolated source fixture'],check=True)
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


BOOTSTRAP = r'''
import os, json, socket, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
F=Path(os.environ['PERSISTENT_LIFECYCLE_FIXTURE'])
def now():return datetime.fromisoformat((F/'clock.txt').read_text()).astimezone(timezone.utc)
def denied(*args,**kwargs):raise RuntimeError('fixture_network_denied')
socket.create_connection=denied
# Kernel unshare-net is the primary network denial; this also blocks accidental
# application socket connection attempts instead of letting them time out.
socket.socket.connect=denied
import race_collection.persistent_collector as owner
import race_collection.persistent_native as native
import race_collection.freshness_campaign as campaign
import race_collection.operational_prediction as prediction
from race_collection.live_freshness_contract import FreshnessContract, create_once
from race_collection.daily_race_inventory import write_daily_inventory
from utils.sportsbet_access import SportsbetAccess
class Clock(datetime):
    @classmethod
    def now(cls,tz=None):return now().astimezone(tz or timezone.utc)
campaign.datetime=Clock
owner.now=now
prediction.now=now
original_init=SportsbetAccess.__init__
def source_init(self,path=None,*,clock=None):
    original_init(self,path,clock=clock or (lambda:now().timestamp()))
SportsbetAccess.__init__=source_init
def prepared(cfg,standing_ref,day,current):
    return json.loads((F/('prepared-'+day+'.json')).read_bytes())
native.prepare_day=prepared
def discover(contract_path,inventory_path,receipt_path):
    scope=FreshnessContract.load(contract_path)
    scope.admit(now(),seconds=0)
    scope.campaign.request()
    create_once(F/('child-'+str(os.getpid())+'.json'),{'pid':os.getpid(),'source_date':scope.value['source_date']})
    while not (F/'release-child').exists():time.sleep(.05)
    jump=now()+timedelta(hours=2)
    day=scope.value['source_date']
    ref=write_daily_inventory(inventory_path,source_date=day,observed_at=now(),races=[{
        'url':'https://www.thedogs.com.au/racing/fixture/'+day+'/1/example',
        'date':day,'venue':'FIXTURE','race_number':1,
        'scheduled_jump_datetime':jump.isoformat(),'race_time':jump.strftime('%I:%M %p')}])
    create_once(receipt_path,{'inventory':ref,'completed_at':now().isoformat()})
owner.discover=discover
if '--discover' not in __import__('sys').argv:
    (F/'owner-pid.txt').write_text(str(os.getpid()))
'''


def test_actual_cli_pause_restart_and_day_rollover_preserve_budgets(backend,tmp_path,monkeypatch):
    cfg,standing,standing_ref,calls=backend
    first=native.prepare_day(cfg,standing_ref,'2026-10-03',stamp('2026-10-03T12:00:00+10:00'))
    second=native.prepare_day(cfg,standing_ref,'2026-10-04',stamp('2026-10-04T01:22:00+10:00'))
    fixture=tmp_path/'harness';fixture.mkdir()
    for day,value in [('2026-10-03',first),('2026-10-04',second)]:
        (fixture/('prepared-'+day+'.json')).write_text(json.dumps(value))
    checkout=tmp_path/'checkout';commit=clean_checkout(checkout)
    cfg={**cfg,'status':'AUTHORIZED_PERSISTENT_COLLECTOR','standing_authority':standing_ref,
         'campaign_id':standing['campaign_id'],'source_commit':commit}
    config=fixture/'config.json';config.write_text(json.dumps(cfg,sort_keys=True))
    config_sha=hashlib.sha256(config.read_bytes()).hexdigest()
    (fixture/'sitecustomize.py').write_text(BOOTSTRAP)
    (fixture/'clock.txt').write_text('2026-10-03T12:03:00+10:00')
    SportsbetAccess(cfg['source_state']).initialize(access_basis={'status':'permitted','reference':'SYNTHETIC_NO_PROVIDER'})
    command=['bwrap','--unshare-net','--bind','/','/','--dev-bind','/dev','/dev','--proc','/proc',
             '--chdir',str(checkout),sys.executable,'-B','-m','scripts.run_persistent_collector',
             '--config',str(config),'--config-sha256',config_sha]
    env={**os.environ,'PERSISTENT_LIFECYCLE_FIXTURE':str(fixture),'PYTHONDONTWRITEBYTECODE':'1',
         'PYTHONPATH':str(fixture)+os.pathsep+str(checkout),'GREYHOUND_SPORTSBET_ACCESS_STATE':cfg['source_state']}
    for key in ('GREYHOUND_INCIDENT_AUTHORITY_SHA256','GREYHOUND_INCIDENT_SLOT','GREYHOUND_PERSISTENT_ALLOCATION_SHA256'):
        env.pop(key,None)
    processes=[];logs=[]
    def start():
        (fixture/'owner-pid.txt').unlink(missing_ok=True)
        log=(fixture/('cli-'+str(len(processes))+'.log')).open('wb');logs.append(log)
        process=subprocess.Popen(command,cwd=checkout,env=env,stdout=log,stderr=log)
        processes.append(process)
        wait_for(lambda:(fixture/'owner-pid.txt').exists(),process)
        return process
    def stop(process):
        os.kill(int((fixture/'owner-pid.txt').read_text()),signal.SIGTERM)
        code=process.wait(timeout=15)
        assert code==0, '\n'.join(p.read_text() for p in fixture.glob('cli-*.log'))
    def usage():return read(Path(cfg['campaign_root'])/'ledger.json')
    def source():return read(cfg['source_state'])
    try:
        process=start()
        wait_for(lambda:len(list(fixture.glob('child-*.json')))==1,process)
        # Stop the actual owner while its actual discovery child is in flight.
        os.kill(int((fixture/'owner-pid.txt').read_text()),signal.SIGTERM)
        time.sleep(.15)
        assert process.poll() is None
        with (Path(cfg['campaign_root'])/'owner.lock').open('a') as lock:
            with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (fixture/'release-child').touch()
        assert process.wait(timeout=15)==0, '\n'.join(p.read_text() for p in fixture.glob('cli-*.log'))
        assert usage()['logical_requests']==1 and len(source()['diagnostic_authorizations'])==1
        first_state=read(Path(first['output'])/'persistent-owner-state.json')
        assert len(first_state['dispatches'])==1 and first_state['dispatches'][0]['returncode']==0
        assert first_state['inventory'] is not None
        charge=usage()['launches'][first['plan']['rehearsal_id']]['charged_seconds']
        process=start()
        wait_for(lambda:read(Path(first['output'])/'persistent-health.json').get('status')=='WAITING_FOR_RACE',process)
        stop(process)
        assert usage()['logical_requests']==1 and len(source()['diagnostic_authorizations'])==1
        assert len(list(fixture.glob('child-*.json')))==1
        assert usage()['launches'][first['plan']['rehearsal_id']]['charged_seconds']==charge
        # Expired old source date must drain and close without reactivation.
        (fixture/'clock.txt').write_text((stamp(first['plan']['ends_at'])+timedelta(seconds=1)).isoformat())
        process=start()
        pointer=Path(standing['state_root'])/'current-day.json'
        wait_for(lambda:read(pointer).get('racing_date')=='2026-10-04',process)
        stop(process)
        assert read(Path(first['output'])/'day-closed.json')['status']=='DRAINED'
        assert len(source()['diagnostic_authorizations'])==1 and usage()['logical_requests']==1
        # Start the genuinely separate day, preserving all previous charges.
        (fixture/'clock.txt').write_text((stamp(second['plan']['starts_at'])+timedelta(seconds=1)).isoformat())
        process=start()
        wait_for(lambda:read(Path(second['output'])/'persistent-health.json').get('status')=='WAITING_FOR_RACE',process)
        stop(process)
        assert usage()['logical_requests']==2 and len(source()['diagnostic_authorizations'])==2
        assert len(usage()['launches'])==2 and len(list(fixture.glob('child-*.json')))==2
        counts=[r['counts']['python'] for r in usage()['persistent_operation_request_usage'].values()]
        assert sorted(counts)==[1,1]
        assert source()['active'] is None and source()['phase']=='OPEN'
        assert not Path(cfg['lock_path']).exists()
    finally:
        (fixture/'release-child').touch()
        for process in processes:
            if process.poll() is None:
                try:os.kill(int((fixture/'owner-pid.txt').read_text()),signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        for log in logs:log.close()
