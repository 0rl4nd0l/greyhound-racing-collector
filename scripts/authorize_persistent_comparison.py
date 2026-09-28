"""Materialize the reviewed packet ONLY after explicit consolidated approval.

Does not install units, write the shared campaign, request providers or launch a
study. These approval-bound files are a separate, reviewable deployment step.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import digest
from race_collection.persistent_storage import check_mount
from src.predictor.on_demand import canonical_bytes

ROOT=Path(__file__).resolve().parents[1]


def materialize(prepared,output,*,approval_reference,allocation_reference,history_reference,result_reference):
    if not all(isinstance(r,str) and r.strip() for r in (approval_reference,allocation_reference,history_reference,result_reference)):
        raise ValueError('explicit_approval_references_required')
    values={name:json.loads((prepared/(name+'.prepared.json')).read_bytes())
            for name in ('plan','programme-authority','result-authority','schedule')}
    if any(v['status']!='PREPARED_NOT_AUTHORIZED' for v in values.values()):raise ValueError('prepared_only')
    plan=values['plan'];cfg=values['schedule'];programme=values['programme-authority'];authority=values['result-authority']
    now=datetime.now(timezone.utc)
    start=datetime.fromisoformat(plan['starts_at'])
    if start<=now:raise ValueError('future_allocation_required_no_backdating')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=cfg['source_commit']:
        raise ValueError('reviewed_source_changed')
    check_mount(cfg['storage_mount'],output)
    campaign=Campaign(cfg['campaign_root'])
    if campaign.programme or digest(campaign.value)!=programme['prior_effective_authorization_sha256']:
        raise ValueError('campaign_authority_changed')
    ledger=json.loads((campaign.root/'ledger.json').read_bytes())
    import math
    counters={'capture_attempts':len(ledger['attempts']),'logical_requests':ledger['logical_requests'],
              'live_seconds':math.ceil(sum(r['charged_seconds'] for r in ledger['launches'].values()))}
    if counters!=programme['initial_counters'] or ledger.get('source_holds') or any(not r.get('closed_at') for r in ledger['launches'].values()):
        raise ValueError('campaign_consumption_changed_or_held_prepare_again')
    source_path=Path(cfg['source_state']);source=json.loads(source_path.read_bytes())
    if (hashlib.sha256(source_path.read_bytes()).hexdigest()!=cfg['source_baseline']['state_sha256']
            or source['phase']!='OPEN' or source['active'] is not None):
        raise ValueError('source_state_changed_or_held_prepare_again')
    output.mkdir(parents=True,exist_ok=False,mode=0o700)
    def put(name,value):
        path=output/(name+'.APPROVED.json')
        with path.open('xb') as stream:stream.write(canonical_bytes(value))
        path.chmod(0o400)
        return str(path),hashlib.sha256(path.read_bytes()).hexdigest()
    allocation={'status':'AUTHORIZED_EXCLUSIVE_ALLOCATION','authority_reference':allocation_reference,
        'starts_at':plan['starts_at'],'ends_at':plan['ends_at'],'reservation_review_sha256':plan['reservation_review_sha256'],
        'deferred_proposals':['docs/forward_overround_successor_protocol.md','September16 residual successor proposal'],
        'retrospective_membership':False}
    put('allocation',allocation)
    plan.update(status='AUTHORIZED',authority_reference=approval_reference,activated_at=now.isoformat(),
        exclusive_population_allocation_reference=allocation_reference,machine_history_authority_reference=history_reference)
    plan_path,plan_hash=put('plan',plan)
    programme.update(status='AUTHORIZED_PERSISTENT_PROGRAMME',authority_reference=approval_reference)
    _,programme_hash=put('programme-authority',programme)
    authority.update(status='AUTHORIZED_MACHINE_RESULT_RETENTION',authority_reference=result_reference,
        plan_sha256=plan_hash,issued_at=now.isoformat(),source_budget_reference=approval_reference)
    authority_path,authority_hash=put('result-authority',authority)
    binding={'plan':plan_path,'plan_sha256':plan_hash,'authority':authority_path,'authority_sha256':authority_hash}
    binding_path,_=put('result-binding',binding)
    cfg.update(status='AUTHORIZED_PERSISTENT_SCHEDULE',authority_reference=approval_reference,
        programme_authority_sha256=programme_hash,comparison_plan=plan_path,comparison_plan_sha256=plan_hash,result_binding=binding_path)
    put('schedule',cfg)
    put('approval-receipt',{'status':'APPROVAL_FILES_MATERIALIZED_NOT_DEPLOYED','authority_reference':approval_reference,
        'at':now.isoformat(),'prepared_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared.glob('*.prepared.json')},
        'provider_requests':0,'services_changed':False})
    return {'status':'APPROVAL_FILES_MATERIALIZED_NOT_DEPLOYED','output':str(output)}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for name in ('approval-reference','allocation-reference','history-reference','result-reference'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(materialize(**vars(a))))
