"""Invented finite incident authority builder; no installed state access."""
import hashlib
import json
from pathlib import Path

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def make_incident(tmp_path, *, study_plan=None, study_allocation=None, weekend_allocation=None, weekend_authority=None, candidate_registry=None):
    registry=candidate_registry or put(tmp_path/'registry.json', {'frozen':True})
    study=study_plan or put(tmp_path/'study.json', {'status':'AUTHORIZED','programme_root':str(tmp_path/'study-members'),
        'candidate_registry':registry,'prediction_output_roots':[str(tmp_path/'study-predictions')]})
    allocation=study_allocation or put(tmp_path/'allocation.json', {'status':'AUTHORIZED_EXCLUSIVE_ALLOCATION'})
    study_results=put(tmp_path/'study-result-authority.json', {'status':'AUTHORIZED_MACHINE_RESULT_RETENTION',
        'plan_sha256':study['sha256'], 'result_database':str(tmp_path/'study-results/official.sqlite3')})
    weekend=weekend_allocation or put(tmp_path/'weekend.json', {'status':'AUTHORIZED','state_root':str(tmp_path/'weekend-state')})
    pilot=weekend_authority or put(tmp_path/'weekend-authority.json',{'allocation_sha256':weekend['sha256'],
        'state_root':str(tmp_path/'weekend-state'),'prediction_root':str(tmp_path/'weekend-predictions')})
    value={'schema_version':'collector_incident_engineering_authority_v1','status':'AUTHORIZED_INCIDENT_ENGINEERING',
        'incident_id':'invented-incident','authority_reference':'SYNTHETIC_ONLY','campaign_id':'SYNTHETIC',
        'issued_at':'2026-10-01T16:00:00+10:00','slots':[
            {'id':'001','starts_at':'2026-10-01T16:10:00+10:00','ends_at':'2026-10-01T17:40:00+10:00','cleanup_by':'2026-10-01T18:11:00+10:00'},
            {'id':'002','starts_at':'2026-10-01T18:15:00+10:00','ends_at':'2026-10-01T19:45:00+10:00','cleanup_by':'2026-10-01T20:16:00+10:00'}],
        'collection_stop_at':'2026-10-01T21:00:00+10:00','cleanup_deadline':'2026-10-01T21:30:00+10:00',
        'result_deadline':'2026-10-02T12:00:00+10:00','max_capture_attempts_per_window':24,
        'max_prediction_logical_requests_per_window':16000,'max_source_operations_per_window':192,
        'max_result_requests_per_window':72,'max_result_operations_per_window':72,
        'study_plan':study,'study_result_authority':study_results,'study_allocation':allocation,'weekend_allocation':weekend,'weekend_authority':pilot,'candidate_registry':registry,
        'state_root':str(tmp_path/'incident-state'),'prediction_root':str(tmp_path/'incident-predictions'),
        'result_root':str(tmp_path/'incident-results'),'engineering_only':True,'study_enrolment':False,
        'performance_evaluation':False,'human_outcome_access':False,
        'reservation_disposition':'AUTHORIZED_NON_EVALUATIVE_REUSE_EXCLUDING_ADMITTED_STUDY_IDENTITIES'}
    ref=put(tmp_path/'incident.json',value)
    return value,ref
