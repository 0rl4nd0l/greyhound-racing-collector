"""Opt-in study member authorization for the existing result collector.

No transport, scheduler or result read. The default R3 operational exclusion
remains. A result-retention authority is distinct from prediction activation.
"""
from datetime import timedelta
import hashlib,json
from pathlib import Path
from src.predictor.future_comparison import checked,load_plan,stamp,verify_comparison


def result_scope(binding, *, now, prediction_bundles, result_database):
    plan,raw=load_plan(Path(binding['plan']),binding['plan_sha256'])
    authority=json.loads(checked(Path(binding['authority']),binding['authority_sha256']))
    if (plan['status']!='AUTHORIZED' or authority.get('status')!='AUTHORIZED_MACHINE_RESULT_RETENTION'
            or authority.get('plan_sha256')!=binding['plan_sha256'] or not authority.get('owner')
            or Path(authority.get('result_database','/UNBOUND')).absolute()!=result_database.absolute()
            or not authority.get('authority_reference') or not authority.get('source_budget_reference') or authority.get('human_outcome_access') is not False
            or not stamp(authority['issued_at'])<=stamp(plan['activated_at'])<=now
            or now>stamp(plan['ends_at'])+timedelta(days=14)
            or str(prediction_bundles.resolve()) not in [str(Path(p).resolve()) for p in plan['prediction_output_roots']]):
        raise ValueError('comparison_result_scope_not_authorized')
    return plan


def authorize_job(job, binding, plan, *, now, prediction_bundles):
    jump=stamp(job.input.jump_timestamp)
    if not stamp(plan['starts_at'])<=jump<stamp(plan['ends_at']) or now<jump+timedelta(minutes=15):
        raise ValueError('comparison_result_not_due_or_outside_population')
    key=hashlib.sha256(job.input.race_id.encode()).hexdigest()
    claim=Path(plan['programme_root'])/binding['plan_sha256']/'attempts'/key
    if not (claim/'admission.json').is_file() or not (claim/'completion.json').is_file():
        raise ValueError('comparison_result_membership_or_completion_missing')
    admission=json.loads((claim/'admission.json').read_bytes())
    if admission['job_id']!=job.job_id or admission['race']['race_id']!=job.input.race_id:
        raise ValueError('comparison_result_job_not_admitted')
    verified=verify_comparison(prediction_bundles,claim/'admission.json',expected_plan_sha256=binding['plan_sha256'])
    if not verified['future_race_evidence']:
        raise ValueError('comparison_result_not_common_sealed')
