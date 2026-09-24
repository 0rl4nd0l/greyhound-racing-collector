"""Synthetic comparison failures and real retained worker/subprocess verification."""
from datetime import date
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src.predictor.comparison_candidates import card_features, projected_dates
from src.predictor.future_comparison import verify_comparison, load_plan
from src.predictor.on_demand import canonical_bytes, sha256_file
from tests.fixtures.frozen_comparison_case import prepare, execute, ROOT


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    root=tmp_path_factory.mktemp("comparison")
    case=prepare(root);case["db"].unlink()
    baseline=execute(case,comparison=False,output_name="baseline")
    compared=execute(case)
    assert baseline["phase"]==compared["phase"]=="PREDICTION_READY",(baseline,compared)
    return case


def bundle(case, name="predictions"):
    root=case["root"]/name
    entry=json.loads((root/"prediction_bundle_index_v1.json").read_bytes())["entries"][0]
    return root/entry["directory"]


def admission(case):
    return next((case["root"]/"comparison-programme").glob("*/attempts/*/admission.json"))


def test_real_subprocess_retained_replay_and_default_off(executed):
    case=executed
    value=verify_comparison(case["root"]/"predictions",admission(case),expected_plan_sha256=sha256_file(case["comparison"]))
    assert value["eligible_common_race"] and not value["future_race_evidence"]
    assert all(r["status"]=="SEALED" for r in value["records"].values())
    assert not (bundle(case,"baseline")/"comparison").exists()
    a=json.loads((bundle(case,"baseline")/"result.json").read_bytes())
    b=json.loads((bundle(case)/"result.json").read_bytes())
    assert a["prediction"]==b["prediction"]
    registry=json.loads((ROOT/"artifacts/research_comparison/frozen_20260924/registry.json").read_bytes())
    for path,digest in registry["production"].items(): assert sha256_file(ROOT/path)==digest
    inputs=[r["input_identity"] for r in value["records"].values()]
    assert all(v==inputs[0] for v in inputs)


def test_consumed_comparison_cannot_replace_sealed_predictions(executed):
    case=executed;old={p:p.read_bytes() for p in admission(case).parent.rglob("*") if p.is_file()}
    result=execute(case,output_name="duplicate")
    assert result["phase"]=="PREDICTION_READY"
    summary=json.loads((bundle(case,"duplicate")/"comparison/summary.json").read_bytes())
    assert summary["shared_failure"]=="comparison_attempt_already_consumed"
    assert all(v=="FAILED" for v in summary["models"].values())
    assert all(p.read_bytes()==raw for p,raw in old.items())


def test_individual_candidate_failure_preserved_without_substitution(executed):
    case=dict(executed);root=case["root"]/"broken-model";root.mkdir()
    origin=ROOT/"artifacts/research_comparison/frozen_20260924"
    for name in ("registry.json","residual_box.json","residual_half.json"):
        shutil.copyfile(origin/name,root/name)
    (root/"residual_box.json").write_text("corrupt synthetic candidate copy")
    plan=json.loads(case["comparison"].read_bytes());plan["candidate_registry"]["path"]=str(root/"registry.json")
    plan["programme_root"]=str(root/"programme")
    case["comparison"]=root/"plan.json";case["comparison"].write_bytes(canonical_bytes(plan))
    result=execute(case,output_name="candidate-failure")
    assert result["phase"]=="PREDICTION_READY"
    p=next((root/"programme").glob("*/attempts/*/admission.json"))
    value=verify_comparison(case["root"]/"candidate-failure",p)
    assert not value["eligible_common_race"]
    assert value["records"]["residual_box"]["failure"]=="comparison_file_hash_changed"
    assert value["records"]["residual_half"]["status"]=="SEALED"
    assert value["records"]["residual_box"]["predictions"] is None


def test_subprocess_rejects_late_comparison_without_changing_production(tmp_path):
    case=prepare(tmp_path,late=True);case["db"].unlink()
    result=execute(case)
    assert result["phase"]=="PREDICTION_READY",result
    summary=json.loads((bundle(case)/"comparison/summary.json").read_bytes())
    assert summary["shared_failure"]=="comparison_late_admission"
    assert all(v=="FAILED" for v in summary["models"].values())


def test_subprocess_rejects_changed_retained_inputs_and_preserves_failure(executed):
    case=dict(executed);root=case["root"]/"bad-retention";shutil.copytree(case["retained_root"].parent,root)
    case["retained_root"]=root/"bundle"
    manifest=json.loads((case["retained_root"]/"manifest.json").read_bytes())
    source=case["retained_root"]/manifest["files"]["normalized_form"]["path"]
    source.write_text("Dog Name|PLC|DATE\n1. Alpha Fast|1|2026-07-09\n")
    plan=json.loads(case["comparison"].read_bytes());plan["programme_root"]=str(root/"programme")
    case["comparison"]=root/"plan.json";case["comparison"].write_bytes(canonical_bytes(plan))
    result=execute(case,output_name="retention-failure")
    assert result["phase"]!="PREDICTION_READY"
    summary=json.loads((bundle(case,"retention-failure")/"comparison/summary.json").read_bytes())
    assert summary["shared_failure"]=="RETAINED_INPUT_INVALID"
    assert all(v=="FAILED" for v in summary["models"].values())


def test_outcome_values_not_decoded_when_history_date_reserved(monkeypatch):
    from src.predictor import comparison_candidates as module
    monkeypatch.setattr(module.canonical,"parse_card_target_roster_bytes",lambda *a,**k:pytest.fail("full card must not decode"))
    raw=b'Dog Name,PLC,DATE\n"1. Invented, Dog",\xff,2026-08-01\n'
    assert projected_dates(raw)==[date(2026,8,1)]
    with pytest.raises(ValueError,match="protected_history_date_before_outcome_decode"):
        card_features(raw,{},"Race 1 - SAN - 2026-11-01",[],captured_at=date(2026,10,31),denied_history_intervals=[("2026-07-15","2026-10-31")])


def test_card_complete_field_and_availability_rejected():
    from datetime import datetime,timezone
    captured=datetime(2026,7,9,tzinfo=timezone.utc)
    with pytest.raises(ValueError,match="candidate_card_runner_mismatch"):
        card_features(b"Dog Name|DATE\n1. Alpha|2026-07-08\n",{},"Race 1 - SAN - 2026-07-10",
            [{"box_number":1,"display_name":"Alpha"},{"box_number":2,"display_name":"Bravo"}],captured_at=captured)
    with pytest.raises(ValueError,match="history_not_available"):
        card_features(b"Dog Name|DATE\n1. Alpha|2026-07-10\n",{},"Race 1 - SAN - 2026-07-10",[],captured_at=captured)


def test_prepared_plan_is_not_activation(tmp_path):
    p=tmp_path/"plan.json";p.write_bytes(canonical_bytes({"schema_version":"frozen_four_way_comparison_plan_v1","status":"PREPARED_NOT_AUTHORIZED"}))
    with pytest.raises(ValueError,match="comparison_not_activated"): load_plan(p,sha256_file(p))


def test_prior_prediction_bytes_cannot_change(executed):
    # Copy the bundle so the successful original and completion remain intact.
    case=executed;root=case["root"]/"tamper";shutil.copytree(case["root"]/"predictions",root)
    entry=json.loads((root/"prediction_bundle_index_v1.json").read_bytes())["entries"][0]
    path=root/entry["directory"]/"comparison/residual_box.json";raw=path.read_bytes();path.write_bytes(raw+b" ")
    with pytest.raises(Exception,match="PREDICTION_BUNDLE_CHANGED"):
        verify_comparison(root,admission(case))


def test_worker_preflight_failure_consumed_before_subprocess(tmp_path):
    case=prepare(tmp_path);case['db'].unlink()
    case['index'].write_text('changed synthetic current index')
    with pytest.raises(Exception): execute(case)
    claim=next((case['root']/'comparison-programme').glob('*/attempts/*'))
    assert (claim/'dispatch.json').exists()
    assert json.loads((claim/'worker_failure.json').read_bytes())['predictions_missing']
    assert not (claim/'admission.json').exists()


def test_terminal_evaluator_gates_before_outcome_reader(tmp_path,monkeypatch):
    from scripts.evaluate_frozen_comparison import evaluate
    from src.operator_ui import journal_results
    monkeypatch.setattr(journal_results,'OfficialResultSource',lambda *a:pytest.fail('must not open outcomes'))
    p=tmp_path/'plan.json';p.write_bytes(canonical_bytes({'schema_version':'frozen_four_way_comparison_plan_v1','status':'PREPARED_NOT_AUTHORIZED'}))
    with pytest.raises(ValueError,match='comparison_not_activated'):
        evaluate(p,sha256_file(p),tmp_path/'no-authority','none',tmp_path/'no-results',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_fixed_terminal_metrics_have_race_denominators_and_four_contrasts():
    from scripts.evaluate_frozen_comparison import summarize
    from src.predictor.future_comparison import MODELS
    races=[{'date':f'2027-01-{d:02}','venue':'invented','boxes':[1,2], 'winner':0,
            'probabilities':{m:[.6,.4] for m in MODELS}} for d in range(1,5)]
    result=summarize(races,replicates=200)
    assert result['races']==result['dates']==4 and len(result['paired'])==4
    assert result['metrics']['market']['brier']==pytest.approx(.32)
    assert result['metrics']['market']['log_loss']==pytest.approx(-__import__('math').log(.6))
    assert sum(b['runners'] for b in result['calibration']['market'])==8
    assert all(v['log_loss']==0 and v['brier']==0 and not v['both_upper_bounds_below_zero'] for v in result['paired'].values())


def test_campaign_preparer_cannot_activate_prepared_plan(tmp_path):
    from scripts.prepare_freshness_rehearsal import prepare as prepare_campaign
    p=tmp_path/'plan.json';p.write_bytes(canonical_bytes({'schema_version':'frozen_four_way_comparison_plan_v1','status':'PREPARED_NOT_AUTHORIZED'}))
    with pytest.raises(ValueError,match='comparison_not_activated'):
        prepare_campaign(output=tmp_path/'package',start=None,python=None,db=None,lock=None,reconciliation_roots=None,installed_dir=None,
            operational_predictions=True,comparison_plan=p)
    assert not (tmp_path/'package').exists()


def test_existing_supervisor_records_opportunities_without_prediction(tmp_path):
    from race_collection.operational_prediction import Supervisor
    case=prepare(tmp_path)
    binding={'path':str(case['comparison']),'sha256':sha256_file(case['comparison'])}
    supervisor=Supervisor(tmp_path/'supervisor',{'frozen_comparison':binding,'evidence_root':str(case['evidence'])},None)
    supervisor.observe_comparison_schedule()
    root=tmp_path/'comparison-programme'/binding['sha256']
    opportunities=list((root/'opportunities').glob('*.json'))
    assert len(opportunities)==1 and not (root/'attempts').exists()
    before=opportunities[0].read_bytes()
    supervisor.observe_comparison_schedule()
    assert opportunities[0].read_bytes()==before


@pytest.mark.parametrize('endpoint_future',[True,False])
def test_endpoint_and_outcome_authority_precede_any_result_access(tmp_path,monkeypatch,endpoint_future):
    from datetime import datetime,timedelta,timezone
    from scripts.evaluate_frozen_comparison import evaluate
    from src.operator_ui import journal_results
    monkeypatch.setattr(journal_results,'OfficialResultSource',lambda *a:pytest.fail('result access must remain closed'))
    plan=json.loads((ROOT/'docs/research/future_comparison_evidence/prepared_plan.json').read_bytes())
    now=datetime.now(timezone.utc)
    plan.update(status='AUTHORIZED',authority_reference='synthetic gate test',exclusive_population_allocation_reference='synthetic allocation',machine_history_authority_reference='synthetic history',
        activated_at=(now-timedelta(days=100)).isoformat(),starts_at=(now-timedelta(days=99)).isoformat(),ends_at=(now+timedelta(days=1) if endpoint_future else now-timedelta(days=15)).isoformat(),programme_root=str(tmp_path/'programme'))
    p=tmp_path/'plan.json';p.write_bytes(canonical_bytes(plan))
    authority=tmp_path/'authority.json';authority.write_bytes(canonical_bytes({'status':'NOT_AUTHORIZED'}))
    expected='fixed_result_closure_endpoint_not_reached' if endpoint_future else 'outcome_authority_missing'
    with pytest.raises(ValueError,match=expected):
        evaluate(p,sha256_file(p),authority,sha256_file(authority),tmp_path/'never-open-results',tmp_path/'out')
    assert not (tmp_path/'programme').exists() and not (tmp_path/'out').exists()
