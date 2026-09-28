"""Prepare an inactive allocation packet. Never creates runtime/study state."""
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse,hashlib,json
from src.predictor.on_demand import canonical_bytes


def prepare(*, starts_at, programme_root, prediction_output_root, output, reservation_sha256):
    output=output.absolute()
    start=datetime.fromisoformat(starts_at)
    if start.tzinfo is None:raise ValueError('timezone_required')
    start=start.astimezone(ZoneInfo('Australia/Melbourne'))
    if start.date().isoformat()<'2026-10-01' or start<=datetime.now(start.tzinfo):raise ValueError('future_unreserved_allocation_required')
    if not prediction_output_root.is_absolute():raise ValueError('absolute_existing_prediction_output_required')
    if not programme_root.is_absolute() or programme_root.exists():raise ValueError('new_absolute_programme_root_required')
    if len(reservation_sha256)!=64 or any(c not in '0123456789abcdef' for c in reservation_sha256):raise ValueError('reservation_review_hash_required')
    registry=Path(__file__).resolve().parents[1]/'artifacts/research_comparison/frozen_20260924/registry.json'
    plan={'schema_version':'frozen_four_way_comparison_plan_v2','status':'PREPARED_NOT_AUTHORIZED',
        'authority_reference':None,'exclusive_population_allocation_reference':None,'machine_history_authority_reference':None,
        'reservation_review_sha256':reservation_sha256,'activated_at':None,'starts_at':start.isoformat(),'ends_at':(start+timedelta(days=112)).isoformat(),
        'programme_root':str(programme_root/'comparison'),'prediction_output_roots':[str(prediction_output_root)],
        'candidate_registry':{'path':str(registry),'sha256':hashlib.sha256(registry.read_bytes()).hexdigest()},
        'history_policy':'strictly_earlier_machine_features_only','denied_history_intervals':[],
        'decision_seconds_before_jump':120,'quote_lead_seconds':[120,600],'fixed_closure_days':14,
        'missing_result_policy':'bounded_paired_losses_v1','no_interim_metrics':True,'production_promotion':False,'betting':False,
        'scheduled_sessions_per_week':5,'session_minutes':90,'fixed_calendar_days':112,'family_contrasts':4,
        'bootstrap_replicates':20000,'bootstrap_seed':20260924}
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.prepared.json').write_bytes(canonical_bytes(plan))
    authority={'status':'PREPARED_NOT_AUTHORIZED','plan_sha256':None,'owner':None,'authority_reference':None,'issued_at':None,'human_outcome_access':False,
        'result_database':str(programme_root/'official-results.sqlite3'),'source_budget_reference':None,'retention_policy':'first existing cycle after T+15m, next-day repair, fixed D+14 closure; no metrics',
        'failure_policy':'stop on source gate denial; record outage, never bypass operational ownership'}
    (output/'result-authority.prepared.json').write_bytes(canonical_bytes(authority))
    (output/'result-binding.prepared.json').write_bytes(canonical_bytes({'plan':str(output/'plan.APPROVED.json'),'plan_sha256':None,
        'authority':str(output/'result-authority.APPROVED.json'),'authority_sha256':None}))
    return {'status':'PREPARED_NOT_AUTHORIZED','plan_sha256':hashlib.sha256((output/'plan.prepared.json').read_bytes()).hexdigest(),'starts_at':plan['starts_at'],'ends_at':plan['ends_at'],'runtime_writes':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--starts-at',required=True);p.add_argument('--programme-root',type=Path,required=True);p.add_argument('--prediction-output-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reservation-sha256',required=True);a=p.parse_args()
    print(json.dumps(prepare(starts_at=a.starts_at,programme_root=a.programme_root,prediction_output_root=a.prediction_output_root,output=a.output,reservation_sha256=a.reservation_sha256)))
