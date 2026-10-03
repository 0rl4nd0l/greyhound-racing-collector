"""Independent review of actual daily scope and frozen-comparison validation."""
from datetime import timedelta
import copy
import json
from pathlib import Path
import pytest

from race_collection.live_freshness_contract import FreshnessContract, digest
from race_collection.freshness_campaign import Campaign
from race_collection.persistent_authority import stamp
from src.predictor.future_comparison import load_plan
from tests.fixtures.persistent_operation_case import put
from tests.test_persistent_native import backend, run


def test_actual_contract_accepts_26h_total_scope_with_exact_cleanup(backend):
    prepared=run(backend);contract=copy.deepcopy(prepared['contract'])
    allocation=json.loads(Path(prepared['allocation_ref']['path']).read_bytes())
    start=stamp('2026-10-03T10:00:00+10:00');cleanup=start+timedelta(hours=26);end=cleanup-timedelta(seconds=1860)
    allocation.update(issued_at=(start-timedelta(minutes=1)).isoformat(),starts_at=start.isoformat(),
                      ends_at=end.isoformat(),cleanup_by=cleanup.isoformat())
    root=Path(prepared['output']);ref=put(root/'review-26h-allocation.json',allocation)
    comparison=json.loads(Path(contract['frozen_comparison']['path']).read_bytes())
    comparison.update(persistent_allocation=ref,activated_at=allocation['issued_at'],starts_at=start.isoformat(),ends_at=cleanup.isoformat())
    comparison_ref=put(root/'review-26h-comparison.json',comparison)
    contract.update(persistent_allocation=ref,frozen_comparison=comparison_ref,starts_at=start.isoformat(),ends_at=end.isoformat())
    campaign=Campaign(contract['campaign_root'],persistent_allocation=ref)
    contract['campaign_authorization_sha256']=digest(campaign.value)
    loaded,_=load_plan(Path(comparison_ref['path']),comparison_ref['sha256'])
    assert loaded['status']=='AUTHORIZED_ENGINEERING'
    scope=FreshnessContract(contract)
    assert (scope.end-scope.start).total_seconds()==26*3600-1860
    assert scope.campaign.persistent['max_live_seconds']==26*3600


@pytest.mark.parametrize('flag',['result_access','research_activation'])
def test_persistent_contract_rejects_unauthorized_result_or_research_flags(backend,flag):
    prepared=run(backend);contract=copy.deepcopy(prepared['contract'])
    contract['operational_predictions'][flag]=True
    with pytest.raises(ValueError,match='persistent_.*(result|research|scope|prediction|authority)'):
        FreshnessContract(contract)


def test_actual_frozen_comparison_rejects_model_policy_or_scope_change(backend):
    prepared=run(backend);binding=prepared['contract']['frozen_comparison']
    comparison=json.loads(Path(binding['path']).read_bytes())
    comparison['quote_lead_seconds']=[120,300]
    changed=put(Path(prepared['output'])/'review-changed-comparison.json',comparison)
    with pytest.raises(ValueError):load_plan(Path(changed['path']),changed['sha256'])
