"""External boundary proof against immutable integration 869fca1c."""
import hashlib
import json
from pathlib import Path
from tests.test_short_operational_observation import prepared
from scripts.run_comparison_schedule import prepare_session
from scripts.run_freshness_rehearsal import execution_contract
from race_collection.live_freshness_contract import FreshnessContract
from race_collection.freshness_campaign import Campaign


def test_actual_scheduler_prepare_session_to_execution_contract(prepared,monkeypatch):
    from race_collection.freshness_campaign import Campaign
    campaign=Campaign(prepared['campaign_root'])
    prediction=prepared['output'].parent/'prospective-predictions'
    campaign.programme={'prediction_root':str(prediction)}
    roots={k:[str(prepared['output'].parent)] for k in ('scheduled_progress','scheduled_reports','phase_checkpoints','prior_rehearsals','manual_claims','manual_attempts')}
    roots_path=prepared['output'].parent/'roots.json';roots_path.write_text(json.dumps(roots))
    # Approval validation is covered by packaged real-plan tests. Here isolate
    # the actual scheduler's typed/preparation/contract handoff, with no systemd.
    comparison=prepared['output'].parent/'invented-plan.json';comparison.write_text('{}')
    monkeypatch.setattr('src.predictor.future_comparison.load_plan',lambda *a:({},b'{}'))
    cfg={'python':str(prepared['python']),'history_database':str(prepared['db']),
         'lock_path':str(prepared['lock']),'installed_dir':str(prepared['installed_dir']),
         'campaign_root':str(prepared['campaign_root']),'comparison_plan':str(comparison),
         'prediction_root':str(prediction),'state_root':str(prepared['output'].parent/'sessions'),
         'reconciliation_roots':str(roots_path),'reconciliation_roots_sha256':hashlib.sha256(roots_path.read_bytes()).hexdigest()}
    prepare_session(cfg,prepared['output'],prepared['start'])
    plan=json.loads((prepared['output']/'plan.json').read_bytes())
    assert plan['reconciliation_roots']==roots
    scope=FreshnessContract(execution_contract(plan,{'source_date':'2026-09-24'}))
    assert scope.value['prediction_root']==str(prediction)
    assert scope.value['db_path']==str(prediction/'capture.sqlite3')
    assert scope.value['max_logical_requests']==16000
