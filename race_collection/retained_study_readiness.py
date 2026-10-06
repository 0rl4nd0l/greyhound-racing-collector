"""Prospective study gate using separate, retained engineering readiness dimensions.

Only metadata and opaque hashes are read. No result database, label, forecast
replay, expired authority revival or retrospective scientific admission occurs.
"""
from datetime import timedelta
import json
from pathlib import Path
import subprocess

from race_collection.incident_acceptance import file_reference
from race_collection.live_freshness_contract import create_once,digest
from src.predictor.future_comparison import stamp

ROOT=Path(__file__).resolve().parents[1]
ALLOWED_SOURCE_CHANGES={
    'scripts/run_comparison_schedule.py','scripts/prepare_freshness_rehearsal.py',
    'race_collection/retained_study_readiness.py','race_collection/scientific_request_amendment.py',
    'race_collection/freshness_campaign.py','utils/sportsbet_access.py',
    'race_collection/retained_study_observer.py','scripts/run_retained_study_observer.py',
    'race_collection/retained_study_reservations.py',
}


def checked(ref):
    path=Path(ref['path'])
    if path.stat().st_size>4*1024*1024:raise ValueError('study_metadata_oversized')
    file_reference(path,ref['sha256'])
    return json.loads(path.read_bytes())


def source_delta(producing,target):
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=target
            or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()):
        raise ValueError('study_readiness_installed_source_changed')
    return subprocess.check_output(['git','diff','--name-only',producing,target],cwd=ROOT,text=True).splitlines()


def validate_amendment(cfg,now):
    amendment=checked(cfg['study_amendment']);old=checked(amendment['predecessor_config'])
    user=checked(amendment['user_authorization'])
    issued=stamp(amendment['issued_at']);effective=stamp(amendment['effective_at'])
    if (amendment['schema_version']!='prospective_study_amendment_v1'
            or amendment['status']!='AUTHORIZED_PROSPECTIVE_STUDY_AMENDMENT'
            or amendment['operation_mode']!='OUTCOME_BLIND_RETAINED_EVIDENCE_OBSERVER'
            or not amendment['authority_reference']
            or amendment['campaign_root']!=cfg['campaign_root']
            or amendment['target_config_sha256']!=digest({k:v for k,v in cfg.items() if k!='study_amendment'})
            or user['schema_version']!='study_recovery_user_authorization_v1'
            or user['status']!='AUTHORIZED_IMPLEMENTATION_AND_COORDINATED_ACTIVATION'
            or user['predecessor_schedule']!=amendment['predecessor_config']
            or user['current_release_live_acceptance']!=amendment['readiness']['integrity']
            or user['protected_outcomes_decoded'] is not False
            or not stamp(user['recorded_at'])<=issued<=now or issued>=effective):
        raise ValueError('study_amendment_invalid')
    allowed={'source_commit','slots','source_operations_per_session','max_source_operations','study_amendment','retained_study_protocol','development_reservations'}
    if {k:v for k,v in old.items() if k not in allowed}!={k:v for k,v in cfg.items() if k not in allowed}:
        raise ValueError('study_immutable_configuration_changed')
    historical=[(i+1,s) for i,s in enumerate(old['slots']) if stamp(s)<effective]
    if [(r['index'],r['slot']) for r in amendment['historical_slots']]!=historical:
        raise ValueError('study_historical_slot_inventory_incomplete')
    if cfg['slots'][:len(historical)]!=[s for _,s in historical]:
        raise ValueError('study_historical_slots_changed')
    if any(stamp(s)-stamp(amendment['issued_at'])<timedelta(minutes=10)
           for s in cfg['slots'][len(historical):]):
        raise ValueError('study_future_slot_not_prospective')
    root=Path(cfg['state_root'])
    for row in amendment['historical_slots']:
        claim=root/'slots'/f"{row['index']:03d}"
        files=row['files']
        if not files:
            if claim.exists():raise ValueError('study_absent_historical_slot_changed')
            continue
        paths={Path(r['path']) for r in files}
        if not {claim/'admission.json',claim/'terminal.json'}<=paths or any(p.parent!=claim for p in paths):
            raise ValueError('study_historical_slot_proof_incomplete')
        for ref in files:file_reference(ref['path'],ref['sha256'])
    return amendment,old


def bind_configuration(cfg,root,now):
    """Preserve original identity; a new checked identity never replaces it."""
    amendment,old=validate_amendment(cfg,now)
    prior=json.loads((root/'config-identity.json').read_bytes())
    if prior!={'sha256':digest(old)}:raise ValueError('study_predecessor_identity_changed')
    path=root/'config-amendments'/(cfg['study_amendment']['sha256']+'.json')
    receipt={'schema_version':'study_config_amendment_identity_v1','prior_config_sha256':digest(old),
             'sha256':digest(cfg),'amendment':cfg['study_amendment']}
    if path.exists():
        if checked(file_reference(path))!=receipt:raise ValueError('study_successor_identity_changed')
    else:
        path.parent.mkdir(parents=True,exist_ok=True);create_once(path,receipt)
    return amendment


def verify_retained_readiness(cfg,now):
    amendment,_=validate_amendment(cfg,now)
    if now<stamp(amendment['effective_at']):return None
    refs=amendment['readiness'];integrity=checked(refs['integrity'])
    metadata=checked(integrity['metadata_audit']);freeze=checked(integrity['independent_freeze_audit'])
    n=integrity['verified_races']
    if (integrity['status']!='PASSED_90_MINUTE_COLLECTION_TO_PREDICTION_WITH_EXCLUSIONS'
            or integrity['seconds']!=5400 or type(n) is not int or n<1
            or integrity['verified_forecasts']!=4*n or integrity['models_changed'] is not False
            or (stamp(integrity['window_end'])-stamp(integrity['window_start'])).total_seconds()!=5400
            or not stamp(integrity['window_end'])<=stamp(amendment['issued_at'])<=now
            or any(metadata[k]!=integrity[k] for k in ('source_commit','configuration_sha256','seconds','window_start','window_end','verified_races','verified_forecasts'))
            or metadata['halt'] is not None or metadata['native_publisher_overlap_pairs']
            or metadata['observer_record_coverage']['spans_window'] is not True
            or freeze['status']!='CONTROL_FREEZE_AND_SERIALIZATION_CLEAR_WITH_DECLARED_AUXILIARY_GAPS'
            or freeze['all_sample_violations'] or freeze['native_pointer_matches'] is not True
            or not freeze['pin_matches'] or any(v is not True for v in freeze['pin_matches'].values())):
        raise ValueError('study_current_release_integrity_unverified')
    audits=metadata['forecast_audits']
    if len(audits)!=n or len({r['job_id'] for r in audits})!=n or len({r['race_id'] for r in audits})!=n:
        raise ValueError('study_forecast_witness_membership_changed')
    for row in audits:
        file_reference(row['path'],row['sha256'])
        if (row['status']!='VERIFIED_PREJUMP' or row['native_engineering_evidence'] is not True
                or row['completion_status']!='COMPLETE_BEFORE_CUTOFF'
                or row['candidate_statuses']!={k:'SEALED' for k in ('production','market','residual_box','residual_half')}
                or not stamp(row['published_complete_at'])<=stamp(row['checked_at'])<stamp(row['jump_at'])
                or len(row['metadata'])!=3):
            raise ValueError('study_prejump_witness_invalid')
        for ref in row['metadata']:file_reference(ref['path'],ref['sha256'])
    compatibility=checked(refs['compatibility'])
    changed=source_delta(integrity['source_commit'],cfg['source_commit'])
    permitted=lambda p:p in ALLOWED_SOURCE_CHANGES or (p.startswith(('tests/','docs/')) and p!='tests/test_run_shadow_non_tgr_rf_evaluation.py')
    if (compatibility['status']!='REVIEWED_READINESS_COMPATIBLE_SUCCESSOR'
            or compatibility['producing_source_commit']!=integrity['source_commit']
            or compatibility['target_source_commit']!=cfg['source_commit']
            or sorted(compatibility['changed_paths'])!=sorted(changed)
            or any(not permitted(p) for p in changed)
            or any(compatibility[k] is not False for k in ('models_changed','feature_generators_changed','frozen_membership_changed'))):
        raise ValueError('study_readiness_source_incompatible')
    accounting=checked(refs['closure_accounting']);completion=checked(refs['closure_completion'])
    states=accounting['states'];count=accounting['cohort_races'];projection=completion['result_projection']
    if (accounting['schema_version']!='independent_sealed_private_closure_accounting_v1'
            or accounting['status']!='PASS_METADATA_ONLY' or accounting['pending']!=0
            or not states.get('CLOSED',0)>0 or set(states)-{'CLOSED','CLOSED_NON_FINISH','QUARANTINED','ATTEMPTS_EXHAUSTED'}
            or sum(states.values())!=count or accounting['frozen_forecasts']!=4*count
            or accounting['result_request_consumed']>accounting['result_request_allowance']
            or accounting['outside_cohort_charged_races']!=0 or accounting['raw_outcomes_released'] is not False
            or accounting['quarantines_terminal_no_automatic_retry'] is not True
            or completion['status']!='PRIVATE_CLOSURE_SEALED_COLLECTION_RESUMED'
            or completion['outcomes_released'] is not False or completion['performance_evaluation'] is not False
            or completion['pending']!=0 or projection['queue_health']['status']!='CLOSURE_SEALED'
            or projection['queue_health']['counts']!=states or projection['queue_health']['outcomes_released'] is not False
            or projection['final_cohort']['eligible_races']!=count
            or projection['final_cohort']['max_requests']!=accounting['result_request_allowance']
            or projection['closure']['status']!='RESULT_CLOSURE_SEALED_NOT_EVALUATED'
            or projection['closure']['target_values_decoded'] is not False
            or projection['closure']['result_authority_sha256']!=accounting['evidence']['authority']['sha256']
            or stamp(projection['closure']['sealed_at'])!=stamp(accounting['sealed_at'])
            or stamp(accounting['sealed_at'])>stamp(amendment['issued_at'])):
        raise ValueError('study_historical_closure_unverified')
    # Deliberately omit mutable global ledger and all databases: the sealed
    # accounting witness preserves its historical snapshot; closure stays opaque.
    for key in ('closure','authority','cohort','binding'):
        ref=accounting['evidence'][key];file_reference(ref['path'],ref['sha256'])
    return {'status':'RETAINED_ENGINEERING_READINESS_DIMENSIONS_VERIFIED',
        'amendment':cfg['study_amendment'],'integrity':refs['integrity'],
        'historical_closure':refs['closure_accounting'],'compatibility':refs['compatibility'],
        'verified_forecasts':4*n,'verified_races':n,'historical_closed_results':states['CLOSED'],
        'historical_quarantines':states.get('QUARANTINED',0),'same_cohort_end_to_end_closed':False,
        'outcomes_released':False,'study_enrolment':False}
