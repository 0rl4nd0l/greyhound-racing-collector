"""Install an explicitly approved, zero-consumption amendment while quiescent.

Failures leave timers stopped and all evidence intact. This is never a collector
or provider entrypoint. It must be invoked manually under installation authority.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess

from race_collection.live_freshness_contract import encoded, digest
from scripts.amend_comparison_schedule import empty_state
from scripts.check_comparison_deployment import inspect, show
from scripts.run_comparison_schedule import load_config
from src.predictor.comparison_result_runtime import load_runtime
from src.predictor.future_comparison import checked


def install(package, manifest_sha256, control, receipt_sha256, evidence):
    receipt=json.loads(checked(control/'amendment-receipt.json',receipt_sha256))
    manifest=json.loads(checked(package/'deployment.json',manifest_sha256))
    if receipt['status']!='AUTHORIZED_AMENDMENT_PREPARED_NOT_INSTALLED':
        raise ValueError('amendment_not_authorized')
    for name,h in receipt['replacement_files'].items():checked(control/name,h)
    old_control=Path(receipt['previous_control'])
    for name,h in receipt['superseded_files'].items():checked(old_control/name,h)
    cfg,_=load_config(control/'schedule.APPROVED.json')
    if cfg['source_commit']!=receipt['source_commit'] or manifest['source_commit']!=receipt['source_commit']:
        raise ValueError('source_binding_mismatch')
    load_runtime(json.loads((control/'result-binding.APPROVED.json').read_bytes()),now=datetime.now(timezone.utc))
    if datetime.fromisoformat(receipt['starts_at'])<=datetime.now(timezone.utc):raise ValueError('amendment_no_longer_prospective')
    old=json.loads((old_control/'schedule.APPROVED.json').read_bytes())
    proof=json.loads((control/'empty-state.json').read_bytes())
    if empty_state(old)!=proof:raise ValueError('empty_state_changed')
    check=inspect(package,manifest_sha256,preflight=True)
    allowed={'new_unit_absent:'+n for n in manifest['unit_sha256']}
    if set(check['findings'])-allowed:raise ValueError('preflight_failed')
    names=list(manifest['unit_sha256'])
    for name in names:
        if name.endswith('.service'):
            state=show(name)
            if state['ActiveState'] not in {'inactive','failed'} or state['MainPID']!='0':
                raise ValueError('worker_not_quiescent')
    evidence.mkdir(parents=True,exist_ok=False,mode=0o700)
    def put(path,raw):
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        path.chmod(0o400)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    put(evidence/'preflight.json',encoded(check))
    unit_dir=Path(cfg['installed_dir']);backup=evidence/'superseded-units';backup.mkdir()
    for name in names:put(backup/name,(unit_dir/name).read_bytes())
    timers=[n for n in names if n.endswith('.timer')]
    subprocess.run(['systemctl','--user','stop',*timers],check=True)
    # Idle timer services may have started since initial inspection. Stop and
    # wait through their existing grace; never delete a lock or kill children.
    subprocess.run(['systemctl','--user','stop',*[n for n in names if n.endswith('.service')]],check=True)
    with ExitStack() as stack:
        locks=[Path(cfg['campaign_root'])/'owner.lock',Path(old['state_root'])/'scheduler.lock']
        oldbinding=json.loads((old_control/'result-binding.APPROVED.json').read_bytes())
        oldauthority=json.loads(Path(oldbinding['authority']).read_bytes())
        locks.append(Path(oldauthority['runtime']['state_root'])/'worker.lock')
        for path in locks:
            f=stack.enter_context(path.open('rb'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if empty_state(old)!=proof:raise ValueError('state_changed_during_quiescence')
        amendment=json.loads((control/'campaign-amendment.json').read_bytes())
        if amendment['empty_state_sha256']!=digest(proof):raise ValueError('empty_state_proof_mismatch')
        folder=Path(cfg['campaign_root'])/'programme-schedule-amendments';folder.mkdir(exist_ok=True)
        destination=folder/f'{len(list(folder.glob("*.json")))+1:04d}.json'
        put(destination,encoded(amendment))
        from race_collection.freshness_campaign import Campaign
        if digest(Campaign(cfg['campaign_root']).programme)!=cfg['programme_authority_sha256']:
            raise ValueError('effective_authority_mismatch')
        for name,h in manifest['unit_sha256'].items():
            raw=checked(package/'units'/name,h)
            if (unit_dir/name).read_bytes()!=(backup/name).read_bytes():raise ValueError('installed_unit_changed')
            temporary=unit_dir/(name+'.october1.tmp')
            with temporary.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            temporary.chmod(0o644);os.replace(temporary,unit_dir/name)
        fd=os.open(unit_dir,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    check=inspect(package,manifest_sha256,preflight=True,installed=True)
    put(evidence/'installed-preflight.json',encoded(check))
    if check['status']!='CHECKS_PASS':raise ValueError('installed_preflight_failed_timers_left_stopped')
    subprocess.run(['systemctl','--user','enable','--now',*timers],check=True)
    result={'status':'AMENDMENT_INSTALLED_ARMED','at':datetime.now(timezone.utc).isoformat(),
        'release':cfg['source_commit'],'control':str(control),'manifest_sha256':manifest_sha256,
        'amendment_receipt_sha256':receipt_sha256,'campaign_amendment':str(destination),
        'first_session':cfg['slots'][0],'scientific_observations':0,'provider_requests':0}
    put(evidence/'installation.json',encoded(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','control','evidence'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('manifest-sha256','receipt-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(install(**vars(a)),sort_keys=True))
