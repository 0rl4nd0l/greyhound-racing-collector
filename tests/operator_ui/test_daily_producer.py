"""Daily producer continuation is an authority-bound read-only display policy."""
import hashlib
import json
from pathlib import Path
from datetime import date,timedelta

import pytest

from src.operator_ui import native_verification as nv


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value).encode();path.write_bytes(raw)
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def daily_policy(tmp_path):
    state=tmp_path/'runtime'
    registry=write(tmp_path/'registry.json',{'frozen':True});study=write(tmp_path/'study.json',{'frozen':True})
    authority=write(tmp_path/'standing.json',{'state_root':str(state),'prediction_root':str(tmp_path/'predictions'),
        'candidate_registry':registry,'study_plan':study,'engineering_only':True,'human_outcome_access':False})
    cfg={'status':'AUTHORIZED_PERSISTENT_COLLECTOR','source_commit':'a'*40,'python':str(tmp_path/'python'),
        'standing_authority':authority,'campaign_root':str(tmp_path/'campaign')}
    config=write(tmp_path/'collector.json',cfg)
    identity={'commit':'a'*40,'tree':'b'*40,'files':{'producer.py':hashlib.sha256(b'producer').hexdigest()}}
    policy={'config':config,'standing_authority':authority,'commit':'a'*40,'tree':'b'*40,'source_identity_sha256':nv.digest(identity)}
    binding={'python':cfg['python'],'producer_packages':[],'daily_producer':policy,
        'config':config['path'],'config_sha256':config['sha256']}
    def package(day):
        output=state/'days'/day/('native-'+day+'-'+authority['sha256'][:12]);source=output/'source'
        source.mkdir(parents=True);(source/'producer.py').write_bytes(b'producer');write(source/'SOURCE_IDENTITY.json',identity)
        prediction=tmp_path/'predictions/days'/day
        allocation=write(output.parent/'allocation.json',{'racing_date':day,'standing_authority':authority,
            'state_root':str(output.parent),'prediction_root':str(prediction)})
        comparison=write(output.parent/'comparison.json',{'candidate_registry':registry,'persistent_allocation':allocation})
        native={'source_root':str(source),'commit':'a'*40,'tree':'b'*40,'source_identity_sha256':nv.digest(identity),
                'racing_date':day,'python':cfg['python'],'frozen_comparison':comparison,'evidence_root':str(output/'collector/evidence'),
                'persistent_allocation':allocation,'prediction_root':str(prediction),'campaign_root':cfg['campaign_root']}
        receipt={'schema_version':'persistent_native_preparation_v1','status':'PREPARED_NOT_STARTED',
            'plan':write(output/'plan.json',native),'racing_date':day,'output':str(output),'standing_authority':authority,
            'configuration_sha256':nv.digest(cfg),'comparison':comparison,'allocation':allocation}
        write(output.parent/'native-prepared.json',receipt)
        return output,receipt
    return binding,package,tmp_path


def test_unchanged_approved_producer_continues_on_later_days_without_accumulating(daily_policy):
    binding,package,_=daily_policy
    for day in ((date(2026,10,5)+timedelta(days=index)).isoformat() for index in range(40)):
        output,receipt=package(day)
        expanded=nv.expand_daily(binding,receipt,output)
        assert len(expanded['producer_packages'])==1
        assert expanded['producer_packages'][0]['plan']==receipt['plan']
        assert nv.select_package(expanded,receipt,output)['commit']=='a'*40
        assert binding['producer_packages']==[]


@pytest.mark.parametrize('change',['config','source','authority','python','model_scope','comparison_path','native_path','code','recovery','recovery_selection','model','allocation'])
def test_daily_continuation_rejects_changed_identity_or_scope(daily_policy,change):
    binding,package,root=daily_policy;output,receipt=package('2026-10-05')
    native=json.loads(Path(receipt['plan']['path']).read_bytes())
    if change=='config':receipt['configuration_sha256']='c'*64
    elif change=='source':native['commit']='c'*40
    elif change=='authority':receipt['standing_authority']={'path':str(root/'other.json'),'sha256':'c'*64}
    elif change=='python':native['python']=str(root/'other-python')
    elif change=='model_scope':
        config=json.loads(Path(binding['daily_producer']['config']['path']).read_bytes());config['status']='OTHER_SCOPE'
        binding['daily_producer']['config']=write(Path(binding['daily_producer']['config']['path']),config);receipt['configuration_sha256']=nv.digest(config)
    elif change=='comparison_path':native['frozen_comparison']={'path':str(root/'outside.json'),'sha256':'c'*64};receipt['comparison']=native['frozen_comparison']
    elif change=='native_path':receipt['plan']=write(root/'elsewhere/plan.json',native)
    elif change=='code':(output/'source/producer.py').write_bytes(b'changed')
    elif change=='recovery':output=output.parent/'recoveries/unapproved/native-recovery'
    elif change=='recovery_selection':receipt['recovery_selection']={'path':str(root/'unknown.json'),'sha256':'c'*64}
    elif change=='model':
        comparison=json.loads(Path(receipt['comparison']['path']).read_bytes());comparison['candidate_registry']['sha256']='c'*64
        receipt['comparison']=write(Path(receipt['comparison']['path']),comparison);native['frozen_comparison']=receipt['comparison']
    elif change=='allocation':native['prediction_root']=str(root/'outside')
    if change not in ('native_path','code'):receipt['plan']=write(Path(receipt['plan']['path']),native)
    original_output=Path(receipt['output']);write(original_output.parent/'native-prepared.json',receipt)
    with pytest.raises(ValueError):nv.expand_daily(binding,receipt,output)


def test_legacy_binding_has_no_automatic_producer_authority(daily_policy):
    binding,package,_=daily_policy;binding.pop('daily_producer');output,receipt=package('2026-10-05')
    assert nv.expand_daily(binding,receipt,output)==binding


def test_damaged_unrelated_history_does_not_block_independently_approved_day(daily_policy):
    binding,package,root=daily_policy;output,receipt=package('2026-10-05')
    approved=nv.expand_daily(binding,receipt,output)['producer_packages'][0]
    bad={**approved,'plan':write(root/'old-plan.json',{'frozen_comparison':{'path':str(root/'old-comparison.json'),'sha256':'d'*64}})}
    Path(bad['plan']['path']).write_text('changed')
    binding['producer_packages']=[bad]
    expanded=nv.expand_daily(binding,receipt,output)
    assert expanded['producer_packages']==[approved]
    assert expanded['_daily_approval_errors']==[{'reason':'Retained producer approval could not be verified.'}]
    assert binding['producer_packages']==[bad]
