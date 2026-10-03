"""Approved producer selection keeps code identity independent of current status."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from src.operator_ui import native_verification as nv


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value).encode();path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def generations(tmp_path):
    entries=[];receipts=[]
    for name in ('old','new'):
        output=tmp_path/name/'native-test';source=output/'source'
        source.mkdir(parents=True)
        (source/'producer.py').write_text(name)
        commit=('a' if name=='old' else 'b')*40;tree='c'*40
        identity={'commit':commit,'tree':tree,'files':{'producer.py':hashlib.sha256((source/'producer.py').read_bytes()).hexdigest()}}
        write(source/'SOURCE_IDENTITY.json',identity)
        fields={'source_root':str(source),'commit':commit,'tree':tree,
                'source_identity_sha256':nv.digest(identity)}
        ref=write(output/'plan.json',fields)
        entry={'plan':ref,**fields,'configuration_sha256':'d'*64}
        entries.append(entry);receipts.append({'plan':ref,'configuration_sha256':'d'*64})
    return {'producer_packages':entries},receipts,tmp_path


def test_selects_exact_approved_source_for_two_generations(generations):
    binding,receipts,root=generations
    for name,receipt in zip(('old','new'),receipts):
        value=nv.select_package(binding,receipt,root/name/'native-test')
        assert Path(value['source_root'])==root/name/'native-test/source'
        assert (Path(value['source_root'])/'producer.py').read_text()==name


@pytest.mark.parametrize('change',['unknown','commit','source','configuration','code','extra_module','symlink_directory'])
def test_rejects_unknown_or_mismatched_producer_before_execution(generations,change):
    binding,receipts,root=generations;receipt=receipts[1]
    entry=binding['producer_packages'][1]
    if change=='unknown':binding['producer_packages']=binding['producer_packages'][:1]
    elif change=='configuration':receipt['configuration_sha256']='e'*64
    elif change=='code':(Path(entry['source_root'])/'producer.py').write_text('tampered')
    elif change=='extra_module':(Path(entry['source_root'])/'unapproved.py').write_text('raise RuntimeError')
    elif change=='symlink_directory':
        outside=root/'outside';outside.mkdir()
        (outside/'unapproved.py').write_text('raise RuntimeError')
        (Path(entry['source_root'])/'unapproved').symlink_to(outside,target_is_directory=True)
    else:
        plan=json.loads(Path(receipt['plan']['path']).read_bytes())
        plan['commit' if change=='commit' else 'source_root']='f'*40 if change=='commit' else str(root/'unapproved')
        new_ref=write(Path(receipt['plan']['path']),plan)
        receipt['plan']=new_ref;entry['plan']=new_ref
    with pytest.raises(ValueError):nv.select_package(binding,receipt,root/'new/native-test')


def test_readonly_replay_imports_each_exact_producer(generations):
    binding,receipts,root=generations
    attempts=[]
    for name,receipt,entry in zip(('old','new'),receipts,binding['producer_packages']):
        source=Path(entry['source_root']);output=source.parent
        future=source/'src/predictor/future_comparison.py'
        future.parent.mkdir(parents=True)
        future.write_text('''import json
def load_plan(path,sha):return json.loads(path.read_bytes()),None
def verify_comparison(*args,**kwargs):
    records={model:{'status':'SEALED','model_sha256':GENERATION,'predictions':[{'box_number':1,'identity':'dog1','dog_name':'Dog 1','probability':.6},{'box_number':2,'identity':'dog2','dog_name':'Dog 2','probability':.4}]} for model in ('production','market','residual_box','residual_half')}
    return {'engineering_evidence':True,'future_race_evidence':False,'eligible_common_race':True,'records':records,'completion':{'published_complete_at':'2026-10-03T05:00:00+00:00'}}
'''.replace('GENERATION',repr(name)))
        persistent=source/'race_collection/persistent_comparison.py'
        persistent.parent.mkdir(parents=True)
        persistent.write_text("def validate_persistent_plan(plan):return plan['allocation']\n")
        identity=json.loads((source/'SOURCE_IDENTITY.json').read_bytes())
        for path in (future,persistent):identity['files'][path.relative_to(source).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        write(source/'SOURCE_IDENTITY.json',identity)
        entry['source_identity_sha256']=nv.digest(identity)
        native={key:entry[key] for key in ('source_root','commit','tree','source_identity_sha256')}
        receipt['plan']=entry['plan']=write(output/'plan.json',native)
        plan={'programme_root':str(output/'admissions'),'allocation':{'prediction_root':str(output/'predictions'),'max_capture_attempts':255}}
        comparison=write(output/'comparison.json',plan);receipt['comparison']=comparison
        evidence=output/'collector/evidence'
        native.update(frozen_comparison=comparison,evidence_root=str(evidence))
        receipt['plan']=entry['plan']=write(output/'plan.json',native)
        admission=output/'admissions'/comparison['sha256']/'attempts/test/admission.json'
        write(admission,{'job_id':name,'race':{'race_id':name,'jump_timestamp':'2026-10-03T05:10:00+00:00'}})
        bundle=output/'predictions/bundles/test'
        phase=evidence/('shadow_autopilot_v1_'+name+'_phase_0')
        refresh=write(phase/'refresh_prejump_report.json',{'fixture':name})
        publication=write(phase/'current_race_index_publish.json',{'run_id':name,'packet_sha256':'f'*64,'source_refresh_report_sha256':refresh['sha256'],'source_refresh_report_path':refresh['path']})
        request_ref=write(bundle/'request.json',{'job_id':name,'operational_index_provenance':{'run_id':name,'report_sha256':publication['sha256'],'source_refresh_sha256':refresh['sha256'],'packet_sha256':'f'*64}})
        impl_ref=write(bundle/'features/sealed/implementation_file_manifest.json',{'implementation_files':['producer.py'],'implementation_file_hashes':{'producer.py':identity['files']['producer.py']}})
        manifest=write(bundle/'bundle_manifest.json',{'files':{'request.json':{'sha256':request_ref['sha256']},'features/sealed/implementation_file_manifest.json':{'sha256':impl_ref['sha256']}}})
        write(admission.with_name('completion.json'),{'bundle_entry':{'directory':'test','manifest_sha256':manifest['sha256']}})
        request={'binding':{**binding,'python':sys.executable},'receipt':receipt,'output':str(output),'comparison':comparison}
        script="import json,sys,types;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from src.operator_ui.native_verification import replay;f=types.ModuleType('src.predictor.future_comparison');f.load_plan=lambda path,sha:(json.loads(path.read_bytes()),None);sys.modules[f.__name__]=f;p=types.ModuleType('race_collection.persistent_comparison');p.validate_persistent_plan=lambda plan:plan['allocation'];sys.modules[p.__name__]=p;r=json.load(sys.stdin);r['output']=Path(r['output']);print(json.dumps(replay(**r)))"
        result=subprocess.run(['bwrap','--die-with-parent','--unshare-pid','--ro-bind','/','/','--dev','/dev','--unshare-net',sys.executable,'-B','-c',script,str(Path(__file__).resolve().parents[2])],input=json.dumps(request),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
        value,limited,used=json.loads(result.stdout)
        assert used==1 and limited is False and value['forecast_errors']==[]
        assert value['forecasts'][0]['models']['production']==name
        assert value['forecasts'][0]['producer']==entry
        attempts.append((admission,comparison,plan['allocation']))
    # A successor can share the ledger while the sealed attempt keeps its producer.
    old_admission,shared,allocation=attempts[0]
    new_entry=binding['producer_packages'][1]
    native=json.loads(Path(new_entry['plan']['path']).read_bytes());native['frozen_comparison']=shared
    new_entry['plan']=write(Path(new_entry['plan']['path']),native)
    assert nv.attempt_package(binding,shared,old_admission,allocation)==binding['producer_packages'][0]
    # Missing or changed retained publication cannot be replaced by a compatible checkout.
    old_source=Path(binding['producer_packages'][0]['source_root'])
    report=old_source.parent/'collector/evidence/shadow_autopilot_v1_old_phase_0/current_race_index_publish.json'
    write(report,{'changed':True})
    with pytest.raises(ValueError):nv.attempt_package(binding,shared,old_admission,allocation)


def test_failed_producer_group_keeps_independently_verified_group(generations,monkeypatch):
    binding,_,root=generations
    plan={'programme_root':str(root/'shared'),'allocation':{'max_capture_attempts':255}}
    comparison=write(root/'shared-comparison.json',plan)
    for name in ('old','new'):
        admission=root/'shared'/comparison['sha256']/'attempts'/name/'admission.json'
        write(admission,{});write(admission.with_name('completion.json'),{})
    future=types.ModuleType('src.predictor.future_comparison')
    future.load_plan=lambda path,sha:(plan,None)
    persistent=types.ModuleType('race_collection.persistent_comparison')
    persistent.validate_persistent_plan=lambda value:value['allocation']
    monkeypatch.setitem(sys.modules,future.__name__,future)
    monkeypatch.setitem(sys.modules,persistent.__name__,persistent)
    monkeypatch.setattr(nv,'select_package',lambda *args:binding['producer_packages'][0])
    monkeypatch.setattr(nv,'attempt_package',lambda b,c,path,a:binding['producer_packages'][0 if path.parent.name=='old' else 1])
    def replay_group(b,entry,*args,**kwargs):
        if entry==binding['producer_packages'][1]:raise subprocess.TimeoutExpired('replay',1)
        return {'forecasts':[{'job_id':'old'}],'failed_forecasts':[],'forecast_errors':[]},False,1
    monkeypatch.setattr(nv,'replay_group',replay_group)
    value,limited,used=nv.replay(binding,{},root,comparison)
    assert value['forecasts']==[{'job_id':'old'}]
    assert len(value['forecast_errors'])==1 and used==2 and limited is False
