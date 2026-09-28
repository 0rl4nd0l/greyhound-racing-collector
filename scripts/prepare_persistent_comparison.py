"""Prepare inactive finite execution authorities; no runtime writes or providers."""
import argparse
from datetime import datetime,timedelta
import hashlib
import json
import math
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import digest
from scripts.prepare_frozen_comparison_activation import prepare as prepare_science
from src.predictor.on_demand import canonical_bytes


def prepare(output,programme_root,starts_at,source_commit,campaign_root,source_state,python,reconciliation_roots):
    start=datetime.fromisoformat(starts_at).astimezone(ZoneInfo('Australia/Melbourne'))
    campaign=Campaign(campaign_root)
    ledger=json.loads((campaign_root/'ledger.json').read_bytes())
    source=json.loads(source_state.read_bytes())
    if campaign.programme is not None:raise ValueError('programme_already_bound')
    reservation=Path(__file__).resolve().parents[1]/'docs/research/future_comparison_20260928_evidence/reservation_review.json'
    prediction_root=programme_root/'operational-predictions'
    prepare_science(starts_at=start.isoformat(),programme_root=programme_root,
        prediction_output_root=prediction_root/'bundles',output=output,
        reservation_sha256=hashlib.sha256(reservation.read_bytes()).hexdigest())
    read=lambda name:json.loads((output/name).read_bytes())
    put=lambda name,value:(output/name).write_bytes(canonical_bytes(value))
    plan=read('plan.prepared.json');end=datetime.fromisoformat(plan['ends_at']);closure=end+timedelta(days=14)
    mount={'path':'/mnt/tenn-nvme2','uuid':'c7b07087-52a2-4afc-8293-c8eba3f48a4d'}
    programme={'schema_version':'collector_persistent_programme_v1','status':'PREPARED_NOT_AUTHORIZED',
        'programme_id':'four-way-'+start.date().isoformat(),'campaign_id':campaign.value['campaign_id'],
        'prior_effective_authorization_sha256':digest(campaign.value),'authority_reference':None,
        'starts_at':plan['starts_at'],'expires_at':closure.isoformat(),
        'max_capture_attempts':1128,'max_logical_requests':1400000,'max_live_seconds':624000,
        'prediction_root':str(prediction_root),
        'initial_counters':{'capture_attempts':len(ledger['attempts']),'logical_requests':ledger['logical_requests'],
            'live_seconds':math.ceil(sum(r['charged_seconds'] for r in ledger['launches'].values()))}}
    put('programme-authority.prepared.json',programme)
    authority=read('result-authority.prepared.json')
    result_root=programme_root/'results'
    authority.update(owner='greyhound-comparison-results.service',result_database=str(result_root/'official-results.sqlite3'),
        runtime={'schema_version':'comparison_result_runtime_v1','state_root':str(result_root),
          'job_store':str(prediction_root/'jobs.sqlite3'),'prediction_bundles':str(prediction_root/'bundles'),
          'campaign_root':str(campaign_root),'lock_path':'/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemon_runtime/shadow_autopilot.lock',
          'source_state':str(source_state),'storage_mount':mount,'expires_at':closure.isoformat(),
          'max_races':1000,'max_requests':24000,'max_attempts_per_race':24,'races_per_cycle':8,'max_storage_bytes':32*2**30})
    put('result-authority.prepared.json',authority)
    slots=[]
    for day in range(112):
        date=start+timedelta(days=day)
        slot=date.replace(hour=13,minute=0,second=0,microsecond=0)
        if slot.weekday()<5 and start<=slot<end:slots.append(slot.isoformat())
    if len(slots)!=80:raise ValueError('allocation_requires_80_weekday_slots')
    schedule={'schema_version':'comparison_schedule_v1','status':'PREPARED_NOT_AUTHORIZED','authority_reference':None,
        'programme_id':programme['programme_id'],'state_root':str(programme_root/'sessions'),'storage_mount':mount,
        'source_commit':source_commit,'python':str(python),'slots':slots,'session_minutes':90,'grace_seconds':300,
        'source_operations_per_session':192,'max_source_operations':15360,
        'prediction_root':str(prediction_root),'campaign_root':str(campaign_root),'source_state':str(source_state),
        'programme_authority_sha256':None,'comparison_plan':str(output/'plan.APPROVED.json'),'comparison_plan_sha256':None,
        'result_binding':str(output/'result-binding.APPROVED.json'),
        'source_baseline':{'state_sha256':hashlib.sha256(source_state.read_bytes()).hexdigest(),
            'operation_count':len(source.get('operations',[])),'denials_sha256':digest(source['denials']),
            'recovery_attempts':source['recovery_attempts'],'access_basis_sha256':digest(source['access_basis']),
            'operating_policy_sha256':digest(source.get('operating_policy'))},
        'history_database':'/mnt/tenn-nvme2/tenn/offloaded-home/l4nd0/greyhound_racing_collector/greyhound_racing_data.db',
        'lock_path':authority['runtime']['lock_path'],'reconciliation_roots':str(reconciliation_roots),
        'reconciliation_roots_sha256':hashlib.sha256(reconciliation_roots.read_bytes()).hexdigest(),
        'installed_dir':str(Path.home()/'.config/systemd/user')}
    put('schedule.prepared.json',schedule)
    manifest={'schema_version':'persistent_prepared_packet_v1','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob('*.prepared.json'))}}
    (output/'prepared-manifest.json').write_bytes(canonical_bytes(manifest))
    return {'prepared_manifest_sha256':hashlib.sha256((output/'prepared-manifest.json').read_bytes()).hexdigest(),'status':'PREPARED_NOT_AUTHORIZED','slots':len(slots),'starts_at':start.isoformat(),'ends_at':end.isoformat(),
            'closure_at':closure.isoformat(),'runtime_writes':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--programme-root',type=Path,required=True)
    p.add_argument('--starts-at',required=True);p.add_argument('--source-commit',required=True)
    p.add_argument('--campaign-root',type=Path,default=Path.home()/'greyhound-collector-campaign-20260923')
    p.add_argument('--reconciliation-roots',type=Path,required=True)
    p.add_argument('--source-state',type=Path,default=Path.home()/'.local/state/greyhound/sportsbet-access.json')
    p.add_argument('--python',type=Path,default=Path(sys.executable));a=p.parse_args()
    print(json.dumps(prepare(a.output.absolute(),a.programme_root,a.starts_at,a.source_commit,a.campaign_root,a.source_state,a.python,a.reconciliation_roots)))
