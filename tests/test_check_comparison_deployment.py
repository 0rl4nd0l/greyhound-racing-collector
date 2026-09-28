"""A deployment check must not change accounting or authorize source recovery."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import check_comparison_deployment as check


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def case(tmp_path, monkeypatch):
    package = tmp_path/'package'; package.mkdir()
    source = tmp_path/'source'; source.mkdir()
    python = tmp_path/'python'; python.write_bytes(b'synthetic interpreter')
    (package/'source.tar').write_bytes(b'synthetic archive')
    gate = write(tmp_path/'gate.json', {'active':None,'phase':'OPEN','operations':[], 'denials':[]})
    campaign=tmp_path/'campaign'
    ledger=write(campaign/'ledger.json',{'launches':{},'source_holds':[], 'attempts':[], 'logical_requests':0})
    (campaign/'owner.lock').touch()
    binding=write(tmp_path/'r3-binding.json',{'synthetic':True})
    names=(*check.COLLECTORS,*check.TIMERS,'greyhound-operator-ui-r3.service')
    baseline_units={}
    for name in names:
        path=tmp_path/'installed'/name; path.parent.mkdir(exist_ok=True);path.write_text('original '+name)
        baseline_units[name]={'path':str(path),'sha256':check.sha(path)}
    baseline=write(package/'baseline.json',{'units':baseline_units,'r3_binding':str(binding),
        'r3_binding_sha256':check.sha(binding),'source':{'path':str(gate)},'collector_lock':str(tmp_path/'absent.lock')})
    unit=package/'units/greyhound-comparison-schedule.service';unit.parent.mkdir();unit.write_text('staged unit')
    fs={'target':str(tmp_path),'uuid':'synthetic-volume','fstype':'ext4'}
    manifest=write(package/'deployment.json',{'schema_version':'persistent_comparison_deployment_v1',
        'source_archive_sha256':check.sha(package/'source.tar'),'baseline_sha256':check.sha(baseline),
        'python':str(python),'python_sha256':check.sha(python),'source_root':str(source),'source_commit':'a'*40,
        'filesystem':fs,'unit_sha256':{unit.name:check.sha(unit)},'campaign_root':str(campaign),
        'schedule_config':str(tmp_path/'schedule.APPROVED.json'),'result_binding':str(tmp_path/'binding.APPROVED.json')})
    def command(args, **kwargs):
        if args[:2]==['findmnt','--json']:return json.dumps({'filesystems':[fs]})
        if args[:3]==['git','rev-parse','HEAD']:return 'a'*40+'\n'
        if args[:2]==['git','status']:return ''
        raise AssertionError('unexpected external command '+repr(args))
    monkeypatch.setattr(check.subprocess,'check_output',command)
    monkeypatch.setattr(Path,'is_mount',lambda self:True)
    monkeypatch.setattr(check.shutil,'disk_usage',lambda path:SimpleNamespace(free=200*2**30))
    monkeypatch.setattr(check,'show',lambda name:{'ActiveState':'inactive','SubState':'dead','MainPID':'0',
        'ControlPID':'0','ControlGroup':'','DropInPaths':'','UnitFileState':'disabled',
        'FragmentPath':baseline_units.get(name,{}).get('path','')})
    return package,check.sha(manifest),gate,ledger,unit


def test_preflight_is_read_only_and_missing_approval_is_explicit(tmp_path,monkeypatch):
    package,identity,gate,ledger,_=case(tmp_path,monkeypatch)
    before=gate.read_bytes(),ledger.read_bytes()
    result=check.inspect(package,identity,preflight=True)
    assert result['status']=='CHECKS_PASS'
    assert result['authority']=={'schedule_config':{'present':False},'result_binding':{'present':False}}
    assert result['live_actions'] is False and result['outcomes_released'] is False
    assert (gate.read_bytes(),ledger.read_bytes())==before


@pytest.mark.parametrize('failure',['source_hold','unmounted','unit_mutation','space'])
def test_preflight_rejects_material_runtime_or_package_failure(tmp_path,monkeypatch,failure):
    package,identity,gate,ledger,unit=case(tmp_path,monkeypatch)
    if failure=='source_hold':
        value=json.loads(gate.read_bytes());value['phase']='STOP';write(gate,value)
    elif failure=='unmounted':monkeypatch.setattr(Path,'is_mount',lambda self:False)
    elif failure=='unit_mutation':unit.write_text('unexpected code')
    elif failure=='space':monkeypatch.setattr(check.shutil,'disk_usage',lambda path:SimpleNamespace(free=99*2**30))
    before=gate.read_bytes(),ledger.read_bytes()
    result=check.inspect(package,identity,preflight=True)
    assert result['status']=='CHECKS_FAILED' and result['findings']
    assert (gate.read_bytes(),ledger.read_bytes())==before
