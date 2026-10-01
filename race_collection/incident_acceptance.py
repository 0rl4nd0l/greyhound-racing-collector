"""Derive readiness from a completed native engineering window and private joins."""
import hashlib
import json
from pathlib import Path
import sqlite3

from race_collection.live_freshness_contract import digest
from race_collection.incident_engineering import checked
from src.predictor.future_comparison import stamp, verify_comparison


def verified_incident_acceptance(study_cfg, reference, now):
    """Return structural evidence only. Failed windows and missing results stay held."""
    try:
        cfg = checked(reference)
        if (cfg['status'] != 'AUTHORIZED_INCIDENT_SCHEDULE'
                or cfg['source_commit'] != study_cfg['source_commit']
                or cfg['campaign_root'] != study_cfg['campaign_root']):
            return None
        from race_collection.incident_comparison import validate_incident_plan
        from src.predictor.future_comparison import load_plan
        plan, _ = load_plan(Path(cfg['comparison_plan']), cfg['comparison_plan_sha256'])
        authority = validate_incident_plan(plan)
        if authority['study_plan'] != {'path':study_cfg['comparison_plan'], 'sha256':study_cfg['comparison_plan_sha256']}:
            return None
        slot = next(row for row in authority['slots'] if row['id'] == cfg['incident_slot'])
        if cfg['slots'] != [slot['starts_at']] or now < stamp(slot['ends_at']):
            return None
        claim = Path(cfg['state_root']) / 'slots/001'
        package = claim / (cfg['programme_id']+'-001')
        read = lambda path: json.loads(path.read_bytes())
        terminal = read(claim/'terminal.json')
        prepared = read(package/'plan.json')
        started = read(package/'started.json')
        measured = read(package/'measurement.json')
        restored = read(package/'restored.json')
        if (terminal['status'] != 'COMPLETED' or (package/'failure.json').exists()
                or started['plan_sha256'] != digest(prepared)
                or prepared['commit'] != cfg['source_commit']
                or prepared['incident_authority'] != cfg['incident_authority']
                or prepared['incident_slot'] != cfg['incident_slot']
                or prepared['frozen_comparison'] != {'path':cfg['comparison_plan'], 'sha256':cfg['comparison_plan_sha256']}
                or prepared['starts_at'] != slot['starts_at'] or prepared['ends_at'] != slot['ends_at']
                or measured['status'] != 'REHEARSAL_MEASURED_NOT_RELEASED'
                or measured['completed_cycles']['full'] < 3 or measured['completed_cycles']['odds'] < 6
                or measured['capture_count'] < 3 or measured['maximum_conservative_source_age'] >= 270
                or measured['logical_requests'] > 16000
                or restored['sportsbet_hold'] is not False
                or restored['status'] not in {'RESTORED', 'RESTORED_COLLECTOR_TRIGGERS_HELD'}):
            return None
        ledger = read(Path(cfg['campaign_root'])/'ledger.json')
        launch = ledger['launches'][prepared['rehearsal_id']]
        if (not launch['closed_at'] or launch['incident_authority_sha256'] != cfg['incident_authority']['sha256']
                or launch['incident_slot'] != cfg['incident_slot']
                or launch['charged_seconds'] > 7260
                or any(not row.get('closed_at') for row in ledger['launches'].values())):
            return None
        from src.predictor.comparison_result_runtime import load_runtime
        binding = read(Path(cfg['result_binding']))
        _, result_authority, result_cfg = load_runtime(binding, now=now, allow_closure=True)
        if binding['plan_sha256'] != cfg['comparison_plan_sha256']:
            return None
        from src.operator_ui.job_store import JobStore
        from src.operator_ui.r3_api import build_verified_bundle_reader
        from src.predictor.comparison_results import ComparisonResultSource
        store = JobStore(Path(result_cfg['job_store']), readonly=True)
        jobs = {job.job_id:job for job in store.recorded_jobs()}
        bundles = Path(result_cfg['prediction_bundles'])
        reader = build_verified_bundle_reader(bundles, store)
        results = ComparisonResultSource(Path(result_authority['result_database']))
        verified = closed = 0
        identities = set()
        job_ids = set()
        with sqlite3.connect((Path(result_cfg['state_root'])/'queue.sqlite3').as_uri()+'?mode=ro', uri=True) as queue:
            for admission in (Path(plan['programme_root'])/binding['plan_sha256']/'attempts').glob('*/admission.json'):
                admitted = read(admission)
                race_id = admitted['race']['race_id']
                job_id = admitted['job_id']
                if (admission.parent.name != hashlib.sha256(race_id.encode()).hexdigest()
                        or race_id in identities or job_id in job_ids):
                    return None
                identities.add(race_id)
                job_ids.add(job_id)
                value = verify_comparison(bundles, admission, expected_plan_sha256=binding['plan_sha256'])
                if not value.get('engineering_evidence') or value['future_race_evidence']:
                    continue
                verified += 1
                job = jobs[job_id]
                if job.input.race_id != race_id:
                    return None
                if (queue.execute("SELECT 1 FROM jobs WHERE job=? AND state='CLOSED'",(job.job_id,)).fetchone()
                        and results.read(job, reader(job), now=now)['state'] == 'RESULT_AVAILABLE'):
                    closed += 1
        if closed < 3:
            return None
        return {'status':'NATIVE_ENGINEERING_CHAIN_VERIFIED', 'schedule':reference,
                'authority_sha256':cfg['incident_authority']['sha256'], 'source_commit':cfg['source_commit'],
                'terminal_sha256':hashlib.sha256((claim/'terminal.json').read_bytes()).hexdigest(),
                'verified_predictions':verified, 'closed_results':closed, 'outcomes_released':False,
                'study_enrolment':False, 'original_failure_preserved':True}
    except (OSError, ValueError, KeyError, TypeError, StopIteration, sqlite3.Error):
        return None
