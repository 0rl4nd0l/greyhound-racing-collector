import copy
import json
import pytest
from scripts import project_prerace_field_live_results_v6 as p
from scripts import project_acquired_live_results as base
from tests.test_project_acquired_live_results import original,URL


def proof():
    return {'dropped_participants':[{'box_number':3,'dog_name':'Inactive Dog'}], 'remapped_participants':[]}


def rows():
    return [{'box_number':1,'dog_name':'Alpha','finish_position':1,'status':None},
            {'box_number':2,'dog_name':'Beta','finish_position':2,'status':None},
            {'box_number':3,'dog_name':'Inactive Dog','finish_position':None,'status':'SCR'}]


def wrapper(tmp_path):
    src=original();src['score_timestamp']='2026-07-17T12:00:00+10:00'
    metadata={'canonical_runner_alignment':{**proof(),'status':'aligned','canonical_runner_set_status':'available','canonical_source_url':URL,'canonical_runner_count':2,'prediction_runner_count':2},
              'prejump_shadow_metadata':{'status':'PASS','metadata_is_leakage_safe':True,'metadata_captured_at':'2026-07-17T11:59:00+10:00','source_url':URL,'runner_box_name_list':src['predictions']}}
    path=tmp_path/'sidecar.json';path.write_text(json.dumps(metadata))
    return {'original':src,'append_proof':{'native_prediction':{'input_hashes':{'sidecar_sha256':base.digest(path.read_bytes())}},'sealed_at':'2026-07-17T12:00:01+10:00','artifacts':{'sidecar':{'path':str(path),'sha256':base.digest(path.read_bytes())}}},
            'win_field':{'status':'EXACT_COMPLETE_WIN_FIELD','capture_at':'2026-07-17T11:59:00+10:00','available_at':'2026-07-17T11:59:01+10:00'}}


def test_presealed_excluded_scratch_does_not_change_original_field():
    active,excluded,mapping=p.derive_original_field_rows(original(),rows(),proof())
    validated,_,_=base.validate_official_field(original(),active)
    assert len(validated)==2 and excluded==[rows()[2]] and mapping==[]


def test_missing_exclusion_proof_cannot_drop_extra_runner():
    active,excluded,_=p.derive_original_field_rows(original(),rows(),{'dropped_participants':[],'remapped_participants':[]})
    assert not excluded
    with pytest.raises(ValueError,match='outside_forecast_field'):
        base.validate_official_field(original(),active)


def test_original_forecasted_runner_scratch_stays_rejected():
    official=rows()[:2];official[1].update(finish_position=None,status='L/SCR')
    contradictory=proof();contradictory['dropped_participants'].append({'box_number':2,'dog_name':'Beta'})
    with pytest.raises(ValueError,match='forecast_runner_scratched:2'):
        p.derive_original_field_rows(original(),official,contradictory)


def test_only_explicit_scratch_can_be_excluded():
    official=rows();official[2]['status']=None
    active,excluded,_=p.derive_original_field_rows(original(),official,proof());assert not excluded
    with pytest.raises(ValueError,match='outside_forecast_field'):
        base.validate_official_field(original(),active)


def test_presealed_reserve_mapping_preserves_unknown_finish():
    official=[rows()[0],{'box_number':2,'dog_name':'Inactive Dog','finish_position':None,'status':'SCR'},
              {'box_number':10,'dog_name':'Beta (from box 2)','finish_position':None,'status':None}]
    evidence={'dropped_participants':[{'box_number':2,'dog_name':'Inactive Dog'}],
              'remapped_participants':[{'dog_name':'Beta','final_box_number':2,'original_box_number':10,'source_box_number':10}]}
    active,excluded,mapping=p.derive_original_field_rows(original(),official,evidence)
    assert active[1]['box_number']==2 and active[1]['dog_name']=='Beta'
    assert active[1]['finish_position'] is None and active[1]['status'] is None
    with pytest.raises(ValueError,match='terminal_status_missing:2'):
        base.validate_official_field(original(),active)


def test_reserve_requires_both_presealed_mapping_and_source_note():
    official=[rows()[0],{'box_number':10,'dog_name':'Beta (from box 2)','finish_position':2,'status':None}]
    with pytest.raises(ValueError,match='pre_race_reserve_continuity_unproved'):
        p.derive_original_field_rows(original(),official,proof())


def test_authenticates_prejump_sidecar_field_and_chronology(tmp_path):
    data=wrapper(tmp_path);actual=p.verify_prerace_field_proof(data,URL)
    assert actual['dropped_participants']==proof()['dropped_participants']
    data['original']['score_timestamp']='2026-07-17T11:58:00+10:00'
    with pytest.raises(ValueError,match='pre_race_field_chronology'):
        p.verify_prerace_field_proof(data,URL)


def test_sidecar_hash_tampering_fails_before_body(tmp_path):
    data=wrapper(tmp_path);data['append_proof']['artifacts']['sidecar']['sha256']='0'*64
    with pytest.raises(ValueError,match='pre_race_sidecar_hash'):
        p.verify_prerace_field_proof(data,URL)

from scripts import integrate_acquired_live_results as integration
from tests.test_project_acquired_live_results import item, markup
from tests.test_project_structured_live_results_v4 import structured, NAME_REF
from tests.test_project_policy_live_results_v3 import ready, policy, POLICY_REF
from tests.test_project_navigation_live_results_v5 import navigation, NAV_REF
FIELD_REF = {'path':'/synthetic/field-policy.json','sha256':'6'*64}


def official_body(status='2nd', reserve=False):
    result=[(1,structured('Alpha',123),'1st'),(10 if reserve else 2,structured('Beta',456,box_note='(from box 2)' if reserve else ''),status),
            (3,structured('Inactive Dog',789),'SCR')]
    return markup(result)


def resign_sidecar(data, mutate):
    from pathlib import Path
    ref=data['append_proof']['artifacts']['sidecar'];path=Path(ref['path'])
    value=json.loads(path.read_bytes());mutate(value);path.write_text(json.dumps(value))
    ref['sha256']=base.digest(path.read_bytes())
    data['append_proof']['native_prediction']['input_hashes']['sidecar_sha256']=ref['sha256']


@pytest.mark.parametrize('identity',['canonical','title','navigation'])
def test_end_to_end_identity_field_and_original_refusals_preserved(tmp_path,identity):
    data=wrapper(tmp_path);raw=official_body()
    if identity=='title':
        raw=raw.replace(f'<link rel="canonical" href="{URL}">'.encode(),b'<title>thedogs - Healesville 17 July 2026 Race 7</title>')
    elif identity=='navigation':
        raw=navigation(raw)
    obs=ready(item(tmp_path,raw))
    projected=integration.project_observation(obs,data['original'],URL,tmp_path/'derived',policy(),POLICY_REF,{},NAME_REF,{},NAV_REF,{},FIELD_REF,data)
    assert projected['pre_race_field_policy_fallback']['status']=='QUALIFIED'
    race=projected['races'][0]
    assert len(race['native_official_rows'])==3
    assert len(race['pre_race_field_projection']['excluded_source_rows'])==1
    assert [r['dog_name'] for r in projected['runners']]==['Alpha','Beta']
    assert race['response_provenance']['runner_name_policy_reference'] is None
    assert race['response_provenance']['source_name_extraction_policy_reference']==NAME_REF
    if identity!='canonical':
        assert projected['projection_policy_fallback']['status']=='REJECTED'
    if identity=='navigation':
        assert projected['structural_identity_policy_fallback']['status']=='REJECTED'


@pytest.mark.parametrize('terminal',['DNF','FELL','DISQ','-','T'])
def test_promoted_original_identity_keeps_terminal_gate(tmp_path,terminal):
    data=wrapper(tmp_path)
    resign_sidecar(data,lambda value:value['canonical_runner_alignment'].update(remapped_participants=[{'dog_name':'Beta','source_box_number':10,'original_box_number':10,'final_box_number':2}]))
    arguments=(item(tmp_path,official_body(terminal,True)),data,URL,tmp_path/'derived',policy(),POLICY_REF,NAME_REF,NAV_REF,FIELD_REF)
    if terminal in {'-','T'}:
        with pytest.raises(ValueError,match='terminal_status_missing:2'):
            p.project_one(*arguments)
        assert (tmp_path/'derived/parsed_official_response.json').exists()
    else:
        result=p.project_one(*arguments)
        assert result['runners'][1]['finish_position'] is None
        assert result['runners'][1]['original_source_box']==10
        assert result['runners'][1]['terminal_status']==terminal


def test_original_scratch_never_opens_field_fallback(tmp_path):
    data=wrapper(tmp_path);obs=ready(item(tmp_path,official_body('SCR')))
    with pytest.raises(ValueError,match='forecast_runner_scratched:2'):
        integration.project_observation(obs,data['original'],URL,tmp_path/'derived',policy(),POLICY_REF,{},NAME_REF,{},NAV_REF,{},FIELD_REF,data)
    assert 'pre_race_field_policy_fallback' not in obs


@pytest.mark.parametrize('mutation',[
 lambda v:v['prejump_shadow_metadata'].update(metadata_is_leakage_safe=False),
 lambda v:v['canonical_runner_alignment'].update(missing_canonical_participants=['Beta']),
 lambda v:v['prejump_shadow_metadata'].update(runner_box_name_list=[{'box_number':1,'dog_name':'Alpha'}]),
 lambda v:v['canonical_runner_alignment'].update(canonical_source_url=URL.replace('/7/','/8/')),
])
def test_invalid_preseal_proof_cannot_qualify(tmp_path,mutation):
    data=wrapper(tmp_path);resign_sidecar(data,mutation)
    with pytest.raises(ValueError,match='pre_race_field_'):
        p.project_one(item(tmp_path,official_body()),data,URL,tmp_path/'derived',policy(),POLICY_REF,NAME_REF,NAV_REF,FIELD_REF)


def test_unavailable_or_late_complete_market_cannot_qualify(tmp_path):
    data=wrapper(tmp_path);data['win_field']['available_at']='2026-07-17T12:01:00+10:00'
    with pytest.raises(ValueError,match='chronology_or_complete_market'):
        p.verify_prerace_field_proof(data,URL)
