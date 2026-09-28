from datetime import datetime,timedelta
import json
from pathlib import Path
import pytest
from src.predictor.on_demand import canonical_bytes,sha256_file
from src.predictor.comparison_result_scope import result_scope
from tests.fixtures.frozen_comparison_case import prepare,execute,use_v2


def test_actual_v2_worker_and_authenticated_result_nomination(tmp_path):
    from scripts.r3_official_result_candidates import r3_prediction_candidates
    # Entirely fabricated race/card/odds/result-store. Authorization strings are
    # test fixtures; none refer to an actual scientific population.
    case=use_v2(prepare(tmp_path),status='AUTHORIZED');case['db'].unlink()
    assert execute(case)['phase']=='PREDICTION_READY'
    plan=json.loads(case['comparison'].read_bytes())
    authority=tmp_path/'result-authority.json'
    authority.write_bytes(canonical_bytes({'status':'AUTHORIZED_MACHINE_RESULT_RETENTION','plan_sha256':sha256_file(case['comparison']),
        'result_database':str(tmp_path/'NO_REAL_RESULTS.db'),'owner':'SYNTHETIC_TEST','authority_reference':'SYNTHETIC_TEST','source_budget_reference':'SYNTHETIC_NO_NETWORK','human_outcome_access':False,'issued_at':plan['activated_at']}))
    binding={'plan':str(case['comparison']),'plan_sha256':sha256_file(case['comparison']),'authority':str(authority),'authority_sha256':sha256_file(authority)}
    args=dict(job_store_path=tmp_path/'predictions-jobs.db',prediction_bundles=tmp_path/'predictions',
        result_database=tmp_path/'NO_REAL_RESULTS.db',target_date=case['jump'].date().isoformat(),current_time=case['jump']+timedelta(minutes=20),race_ids=[],output_dir=tmp_path/'out')
    default,skipped,_=r3_prediction_candidates(**args)
    assert not default and skipped[0]['reason']=='OPERATIONAL_RESULT_ACCESS_FORBIDDEN'
    candidates,skipped,report=r3_prediction_candidates(**args,comparison_result_binding=binding)
    assert len(candidates)==1,(skipped,report)
    assert candidates[0].race_id==case['race_id']
    # Tampering with approved result-retention authority stops before DB reads.
    authority.write_text('{}')
    with pytest.raises(ValueError,match='hash_changed'):r3_prediction_candidates(**args,comparison_result_binding=binding)


def test_result_scope_requires_real_activation_and_owner_before_results(tmp_path,monkeypatch):
    import src.predictor.comparison_result_scope as module
    monkeypatch.setattr(module,'load_plan',lambda *a:({'status':'PREPARED_NOT_AUTHORIZED'},b''))
    authority=tmp_path/'authority.json';authority.write_text('{}')
    with pytest.raises(ValueError,match='not_authorized'):
        result_scope({'plan':str(tmp_path/'unused'),'plan_sha256':'a'*64,'authority':str(authority),'authority_sha256':sha256_file(authority)},now=datetime.now(),prediction_bundles=tmp_path,result_database=tmp_path/"unused")


def test_existing_artifacts_cannot_bypass_comparison_membership(tmp_path):
    from scripts.autonomous_official_result_capture import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--comparison-result-binding',str(tmp_path/'binding'),
            '--r3-job-store',str(tmp_path/'jobs'),'--r3-prediction-bundles',str(tmp_path/'bundles'),
            '--existing-race-rows-jsonl',str(tmp_path/'FORBIDDEN'),'--existing-runner-rows-jsonl',str(tmp_path/'FORBIDDEN2')])
