"""Clearly synthetic controls around the actual exported collector fixture."""
from datetime import datetime, timedelta
import json
from pathlib import Path

from race_collection.development_examples import canonical, digest, put, race_key


def ref(path):
    return {'path': str(path.absolute()), 'sha256': digest(path.read_bytes())}


def prepare_access(seed, control):
    control.mkdir(parents=True)
    root = seed / 'campaign/operational-predictions'
    index = json.loads((root/'bundles/prediction_bundle_index_v1.json').read_bytes())
    entry, = index['entries']
    result = json.loads((root/'bundles'/entry['directory']/'result.json').read_bytes())
    race_id = result['race']['race_id']
    jump = datetime.fromisoformat(result['race']['jump_timestamp'])
    terminal = root/'races'/digest(race_id.encode())/'terminal.json'
    put(control/'reservations.json', {'schema_version': 'development_reservation_registry_v1', 'sources': []})
    put(control/'allocation.json', {'schema_version': 'development_allocation_v1',
        'status': 'SYNTHETIC_FIXTURE', 'authority_reference': 'synthetic_fixture_only',
        'allocation_id': 'synthetic_development', 'mode': 'single_snapshot',
        'operational_owner': 'primary_orchestrator', 'dates': [jump.date().isoformat()],
        'starts_at': (jump-timedelta(days=1)).isoformat(), 'ends_at': (jump+timedelta(days=1)).isoformat(),
        'max_capture_attempts': 2, 'max_attempts_per_date': 2,
        'reservation_registry': ref(control/'reservations.json'), 'denied_history_intervals': []})
    put(control/'opportunities.json', {'schema_version':'development_opportunities_v1', 'complete':True,
        'opportunities': [
            {'race_id':race_id, 'race_key':race_key(race_id), 'qualified':True, 'attempt_consumed':True,
             'disposition':'VERIFIED_FORECAST','reason':'complete_qualified_receipt'},
            {'race_id':race_id.replace('Race 9', 'Race 10'), 'race_key':race_key(race_id.replace('Race 9','Race 10')),
             'qualified':False,'attempt_consumed':False,'disposition':'EXCLUDED','reason':'required_WIN_field_unavailable'},
            {'race_id':race_id.replace('Race 9', 'Race 11'), 'race_key':race_key(race_id.replace('Race 9','Race 11')),
             'qualified':True,'attempt_consumed':True,'disposition':'FAILED','reason':'synthetic_consumed_failure'}]})
    put(control/'access.json', {'schema_version':'development_access_v1','status':'SYNTHETIC_FIXTURE', 'enabled':True,
        'allocation':ref(control/'allocation.json'),'opportunities':ref(control/'opportunities.json'),
        'members':{race_id:{'race_key':race_key(race_id),'jump_at':jump.isoformat(), 'bundle_root':str(root/'bundles'),
            'entry':entry,'verification':ref(terminal),'history_access':'machine_only_pre_target',
            'target_label_access':'separate_result_authority'}}})
    return race_id, control/'access.json', ref(control/'access.json')['sha256']


def prepare_result(control, output, race_id, *, disposition='OFFICIAL'):
    packet = json.loads((output/'pre_result.json').read_bytes())
    observed=(datetime.fromisoformat(packet['times']['scheduled_jump_at'])+timedelta(minutes=5)).isoformat()
    rr=packet['runners']
    race={**{k:packet['race'][k] for k in ('race_id','race_date','race_number','venue')},
        'source':'thedogs_official','status':'resulted','source_url':packet['race']['url'],
        'captured_at':observed,'start_datetime':packet['times']['scheduled_jump_at'],
        'winner_box':rr[0]['box_number'],'winner_name':rr[0]['display_name'],
        'position_count':len(rr),'participant_count':len(rr),'box_order':[r['box_number'] for r in rr]}
    rows=[{**{k:race[k] for k in ('source','source_url','race_id','race_date','race_number','venue','captured_at')},
        'box_number':r['box_number'],'dog_name':r['display_name'],'finish_position':n,'is_winner':n==1,
        'source_native_runner_id':r.get('source_native_runner_id')}
        for n,r in enumerate(rr,1)]
    evidence={'race_rows':[race],'runner_rows':rows}
    result = {'schema_version':'development_official_result_v1','synthetic':True,
        'race_id':race_id,'race_key':race_key(race_id),'disposition':disposition,
        'observed_at':observed,
        'official_source':'synthetic_official_result_fixture','source_evidence_sha256':digest(canonical(evidence)),
        'official_evidence':evidence if disposition=='OFFICIAL' else None,
        'finishers':{r['identity']:n for n,r in enumerate(packet['runners'],1)} if disposition=='OFFICIAL' else {},
        'reason':'synthetic_fixture'}
    put(control/'official-result.json', result)
    put(control/'result-authority.json', {'schema_version':'development_result_authority_v1',
        'status':'SYNTHETIC_FIXTURE','allocation_id':'synthetic_development','authority_reference':'synthetic_fixture_only',
        'members':{race_id:{'race_key':race_key(race_id),'pre_result_sha256':digest((output/'pre_result.json').read_bytes()),
                            'result':ref(control/'official-result.json')}}})
    return control/'result-authority.json', ref(control/'result-authority.json')['sha256']
