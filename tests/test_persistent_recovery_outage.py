"""Reviewed typed-outage recovery carries the consumed failed cycle forward."""
import json
from pathlib import Path

import pytest

from tests.test_persistent_recovery_terminal import terminal_recovery
from tests.test_persistent_recovery import recovery
from tests.test_persistent_native import backend
from tests.fixtures.persistent_operation_case import put
from race_collection import persistent_native as native
from race_collection.live_freshness_contract import classify_refresh_outage


@pytest.fixture
def outage_recovery(terminal_recovery):
    recovery,review=terminal_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=native.checked(cfg['recovery_selection'])
    report_path=Path(review['refresh_report']['path']).with_name('refresh_prejump_report.json')
    candidates=[{'race_url':f'https://www.thedogs.com.au/synthetic/{i}','race_id':f'synthetic-{i}'} for i in range(2)]
    report={'status':'ACQUISITION_INCOMPLETE','reason':'unisolated_selected_race_acquisition_failure',
        'dry_run':False,'selected_count':2,'selected_races':candidates,
        'accepted_csv_count':0,'sidecar_count':0,'shared_sportsbet_snapshot':{'status':'VALIDATED','payload_sha256':'a'*64},
        'downloads':[{'race_url':c['race_url'],'success':False,'result':{'success':False,'source_http_status':502,'error':'Source HTTP status 502'}} for c in candidates],
        'sidecar_metadata_coverage':{'races':[{**c,'weather_track_rejected_reasons':['accepted_csv_missing']} for c in candidates]},
        'current_index_races':[]}
    review['refresh_report']=put(report_path,report)
    phase=native.checked(review['phase_result'])
    phase['current_race_index_publish']['source_refresh_report_path']=str(report_path)
    review['phase_result']=put(Path(review['phase_result']['path']),phase)
    cp=native.checked(review['checkpoint']);cp['phases'][0]['result_sha256']=review['phase_result']['sha256']
    review['checkpoint']=put(Path(review['checkpoint']['path']),cp)
    service=native.checked(review['service_terminal']);service['at']=now.isoformat()
    review['service_terminal']=put(Path(review['service_terminal']['path']),service)
    classified=classify_refresh_outage(old['plan']['evidence_root'],service['run_id'])
    assert classified and classified['upstream_statuses']==[502,502]
    review['classified_outage']=put(Path(old['output'])/'classified-outage.json',classified)
    review['disposition']='PROSPECTIVE_TYPED_UPSTREAM_OUTAGE_RECOVERY'
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)
    return recovery,review


def test_reviewed_two502_responses_consume_one_failed_cycle(outage_recovery):
    recovery,review=outage_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    ledger=Path(cfg['campaign_root'])/'ledger.json';before=ledger.read_bytes()
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    paths=list((Path(new['output'])/'refresh-deferrals').glob('*.json'))
    assert len(paths)==1
    record=json.loads(paths[0].read_bytes())
    assert record['failed_cycle_count']==1 and record['upstream_statuses']==[502,502]
    assert record['source_evidence_root']==old['plan']['evidence_root']
    assert record['inherited_review']==cfg['recovery_selection'] or record['inherited_review']==native.checked(cfg['recovery_selection'])['reviewed_failure']
    state=json.loads((Path(new['output'])/'persistent-owner-state.json').read_bytes())
    assert state['refresh_failures']==[native._ref(paths[0])]
    assert ledger.read_bytes()==before
    assert native.prepare_day(cfg,standing,'2026-10-03',now)==new
    assert len(list((Path(new['output'])/'refresh-deferrals').glob('*.json')))==1


def rebind(outage_recovery,selection,review):
    cfg=outage_recovery[0][0]
    selection['reviewed_failure']=put(Path(selection['reviewed_failure']['path']),review)
    cfg['recovery_selection']=put(Path(cfg['recovery_selection']['path']),selection)


@pytest.mark.parametrize('field,value',[
    ('maximum_failed_cycles',3),('request_retries_added',1),('upstream_statuses',[429]),
    ('refresh_sha256','a'*64),('phase_result_sha256','b'*64),
])
def test_typed_outage_review_cannot_substitute_classification(outage_recovery,field,value):
    recovery,review=outage_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=native.checked(cfg['recovery_selection'])
    classified=native.checked(review['classified_outage']);classified[field]=value
    review['classified_outage']=put(Path(review['classified_outage']['path']),classified)
    rebind(outage_recovery,selection,review)
    with pytest.raises(ValueError,match='typed_outage_unverified'):
        native.prepare_day(cfg,standing,'2026-10-03',now)
    assert len(calls)==1


def test_typed_outage_accepts_exact_scope_stopped_pair(outage_recovery):
    recovery,review=outage_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    selection=native.checked(cfg['recovery_selection']);cleanup=native.checked(selection['cleanup'])
    cleanup['halt']=put(Path(cleanup['halt']['path']),{'reason':'persistent_native_scope_stopped'})
    selection['cleanup']=put(Path(selection['cleanup']['path']),cleanup);review['cleanup']=selection['cleanup']
    rebind(outage_recovery,selection,review)
    assert native.prepare_day(cfg,standing,'2026-10-03',now)['allocation_ref']==old['allocation_ref']


@pytest.mark.parametrize('prior_count',[1,2])
def test_prior_cycles_are_preserved_and_third_cycle_cannot_be_forgiven(outage_recovery,tmp_path,prior_count):
    from tests.test_scheduled_refresh_outage import outage
    recovery,review=outage_recovery
    cfg,standing,old,now,item,terminal,calls=recovery
    originals=[]
    for i in range(prior_count):
        _,p,run,_,_=outage(tmp_path/f'earlier-{i}',run_id=f'000{i}_odds_capture')
        classified=classify_refresh_outage(p['evidence_root'],run)
        record={**classified,'observed_at':now.isoformat(),'failed_cycle_count':i+1,
            'source_evidence_root':p['evidence_root'],'allocation_sha256':old['allocation_ref']['sha256']}
        ref=put(Path(old['output'])/'refresh-deferrals'/(run+'.json'),record)
        originals.append((ref,Path(ref['path']).read_bytes()))
    selection=native.checked(cfg['recovery_selection']);selection['baseline']=native.recovery_baseline(old,cfg)
    rebind(outage_recovery,selection,review)
    if prior_count==2:
        with pytest.raises(ValueError,match='failed_refresh_budget_exhausted'):
            native.prepare_day(cfg,standing,'2026-10-03',now)
        assert len(calls)==1
        return
    new=native.prepare_day(cfg,standing,'2026-10-03',now)
    directory=Path(new['output'])/'refresh-deferrals'
    assert len(list(directory.glob('*.json')))==2
    for ref,raw in originals:
        copied=json.loads((directory/Path(ref['path']).name).read_bytes())
        assert copied['inherited_from']==ref
        assert Path(copied['inherited_bytes']['path']).read_bytes()==raw
        assert Path(ref['path']).read_bytes()==raw
    states=[json.loads(p.read_bytes())['failed_cycle_count'] for p in directory.glob('*.json')]
    assert sorted(states)==[1,2]
