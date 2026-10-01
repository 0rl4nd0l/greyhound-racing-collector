"""Invented restoration identity; no collector, network or protected data."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from race_collection.live_freshness_contract import digest
from race_collection.restoration_ack import verify_r3_replacement


def fixture(tmp_path):
    from scripts import run_freshness_rehearsal as run
    output=tmp_path/'output';output.mkdir();installed=tmp_path/'units';installed.mkdir()
    backup_dir=output/'backup';backup_dir.mkdir()
    hashes={}
    for name in (*run.UNITS,'greyhound-operator-ui-r3.service'):
        raw=('SYNTHETIC '+name).encode()
        (installed/name).write_bytes(raw);(backup_dir/name).write_bytes(raw)
        hashes[name]=hashlib.sha256(raw).hexdigest()
    backup={'r3_pid':'1','hashes':hashes,'modes':dict.fromkeys(hashes,0o600),
            'active':dict.fromkeys(run.TIMERS,False),'enabled':dict.fromkeys(run.TIMERS,'disabled')}
    (output/'restoration.json').write_text(json.dumps(backup))
    (output/'failure.json').write_text(json.dumps({'reason':'installed_r3_changed'}))
    plan={'installed_dir':str(installed),'rehearsal_id':'INVENTED','cleanup_seconds':2,'lock_path':str(tmp_path/'collector.lock')}
    pid=str(os.getpid());state={'MainPID':pid,'ActiveState':'active','SubState':'running','InvocationID':'synthetic'}
    class Control:
        def show(self,unit):return state if unit=='greyhound-operator-ui-r3.service' else {'ActiveState':'inactive'}
        def idle(self):return True
        def command(self,*args):return 'disabled' if args[0]=='is-enabled' else ''
    value={'schema_version':'r3_restoration_ack_v1','authority_reference':'SYNTHETIC_AUTHORITY',
      'plan_sha256':digest(plan),'restoration_sha256':hashlib.sha256((output/'restoration.json').read_bytes()).hexdigest(),
      'old_pid':'1','new_pid':pid,'invocation_id':'synthetic','collection_resume_allowed':False,
      'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
      'process_start_ticks':Path('/proc',pid,'stat').read_text().rsplit(')',1)[1].split()[19],
      'unit_sha256':hashes['greyhound-operator-ui-r3.service'],
      'pinned_files':{str(installed/'greyhound-operator-ui-r3.service'):hashes['greyhound-operator-ui-r3.service']}}
    path=output/'ack.json';path.write_text(json.dumps(value))
    reference=(path,hashlib.sha256(path.read_bytes()).hexdigest(),'SYNTHETIC_AUTHORITY')
    return output,plan,backup,Control(),reference,value,state


@pytest.mark.parametrize('damage',['none','digest','pid','invocation','start','unit','failure','resumption'])
def test_exact_identity_required(tmp_path,damage):
    output,plan,backup,control,ref,value,state=fixture(tmp_path)
    if damage=='digest':ref=(ref[0],'0'*64,ref[2])
    elif damage=='pid':state['MainPID']='0'
    elif damage=='invocation':state['InvocationID']='different'
    elif damage=='failure':(output/'failure.json').write_text('{"reason":"other_failure"}')
    elif damage in {'start','unit','resumption'}:
        value[{'start':'process_start_ticks','unit':'unit_sha256','resumption':'collection_resume_allowed'}[damage]]=True if damage=='resumption' else 'changed'
        ref[0].write_text(json.dumps(value));ref=(ref[0],hashlib.sha256(ref[0].read_bytes()).hexdigest(),ref[2])
    if damage=='none':assert verify_r3_replacement(ref,output,plan,backup,control)['new_pid']==state['MainPID']
    else:
        with pytest.raises(ValueError,match='ack_invalid'):verify_r3_replacement(ref,output,plan,backup,control)


def test_native_restore_preserves_old_snapshot_and_records_explicit_replacement(tmp_path):
    from scripts.run_freshness_rehearsal import restore
    output,plan,backup,control,ref,value,state=fixture(tmp_path)
    before=(output/'restoration.json').read_bytes()
    with pytest.raises(ValueError,match='r3_process_changed'):restore(output,plan,control)
    assert not (output/'restored.json').exists()
    restore(output,plan,control,r3_replacement=ref)
    assert (output/'restoration.json').read_bytes()==before
    assert json.loads((output/'restored.json').read_bytes())['r3_replacement_acknowledgement']['sha256']==ref[1]
