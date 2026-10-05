"""Fabricated races at the controlled development derivation boundary."""
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.predictor.market_form_residual import FEATURES, load_frozen_model, score_race
from src.predictor.controlled_adjustment_pair import derive_controlled_pair, SCORER_SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]

def digest(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()).hexdigest()


def fabricated():
    frozen = load_frozen_model(ROOT/'artifacts/frozen_models/market_form_residual_v1/model.json', ROOT/'artifacts/frozen_models/market_form_residual_v1/manifest.json')
    rid = 'fabricated-control-race'
    rows = []
    for i, (name, odds) in enumerate([('ALPHA', 2.), ('BETA', 3.), ('GAMMA', 6.)], 1):
        rows.append(dict(race_id=rid, runner_id=f'{rid}|box:{i}|dog:{name}', box_number=i, dog_name=name, strict_win_odds=odds,
                         features={k: (None if j==i else float(i*2+j)) for j,k in enumerate(FEATURES)},
                         feature_source_sha256='1'*64, odds_source_sha256='2'*64,
                         feature_freeze_timestamp='2026-01-01T11:50:00+00:00', odds_capture_timestamp='2026-01-01T11:51:00+00:00'))
    ids=sorted(r['runner_id'] for r in rows)
    provenance=dict(race_id=rid, expected_runner_ids=ids, runner_set_sha256=hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest(), jump_timestamp='2026-01-01T12:00:00+00:00', score_timestamp='2026-01-01T11:52:00+00:00')
    record=score_race(frozen, rows, provenance)
    binding=dict(schema_version='controlled_adjustment_common_binding_v1', race_id=rid, original_record_sha256=digest(record),
                 feature_source_sha256='1'*64, odds_source_sha256='2'*64, history_snapshot_sha256='3'*64,
                 retained_manifest_sha256='4'*64, original_sealed_at='2026-01-01T11:53:00+00:00', original_source_commit='a'*40)
    plan=dict(schema_version='controlled_adjustment_plan_v1', parent_family='market_form_residual_v1',
              model_sha256=frozen.model_sha256, manifest_sha256=frozen.manifest_sha256, effective_state_sha256=frozen.effective_state_sha256,
              race_id=rid, common_binding_sha256=digest(binding), strengths=[1.0, .5], runtime_sha256='6'*64, scorer_source_sha256=SCORER_SOURCE_SHA256)
    return frozen,record,binding,plan


def test_default_off_rejects_before_examining_input():
    with pytest.raises(ValueError, match='controlled_derivation_disabled'):
        derive_controlled_pair(None, None, None, None, derived_at=datetime.now(timezone.utc))


def test_pair_changes_only_strength_and_preserves_original_record():
    frozen, record, binding, plan = fabricated()
    before = copy.deepcopy(record)
    output = derive_controlled_pair(frozen, record, binding, plan, derived_at=datetime(2026, 10, 4, tzinfo=timezone.utc), execute=True)
    assert record == before
    assert output['status'] == 'DEVELOPMENT_DERIVATION_NOT_AN_ORIGINAL_PREJUMP_FORECAST'
    full, half = output['candidates']
    assert [full['strength'], half['strength']] == [1., .5]
    assert full['common_input_sha256'] == half['common_input_sha256']
    assert full['parent_effective_state_sha256'] == half['parent_effective_state_sha256'] == frozen.effective_state_sha256
    assert math.fsum(full['probabilities']) == pytest.approx(1)
    assert math.fsum(half['probabilities']) == pytest.approx(1)
    market = output['market_probabilities']
    for i in [1, 2]:
        full_shift=math.log(full['probabilities'][i]/full['probabilities'][0])-math.log(market[i]/market[0])
        half_shift=math.log(half['probabilities'][i]/half['probabilities'][0])-math.log(market[i]/market[0])
        assert full_shift == pytest.approx(2*half_shift, abs=1e-12)
        assert abs(full_shift) <= .7+1e-12
    assert output['derived_at'] == '2026-10-04T00:00:00+00:00'
    assert output['original_score_timestamp'] == record['score_timestamp']
    assert output['scientific_membership_created'] is False


def test_original_seal_inside_last_two_minutes_is_ineligible():
    f, r, b, p = fabricated()
    b['original_sealed_at']='2026-01-01T11:59:00+00:00';p['common_binding_sha256']=digest(b)
    with pytest.raises(ValueError, match='controlled_seal_timing_invalid'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


def test_quote_before_ten_minute_window_is_ineligible():
    f,r,b,p=fabricated()
    for row in r['inputs']['runners']:row['odds_capture_timestamp']='2026-01-01T11:49:59+00:00'
    r=score_race(f,r['inputs']['runners'],r['inputs']['provenance']);b['original_record_sha256']=digest(r);p['common_binding_sha256']=digest(b)
    with pytest.raises(ValueError,match='controlled_quote_window_invalid'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


@pytest.mark.parametrize('field', ['history_snapshot_sha256','retained_manifest_sha256','feature_source_sha256','odds_source_sha256','original_record_sha256'])
def test_changed_common_binding_fails_without_replay(field):
    f,r,b,p=fabricated();b[field]='9'*64
    with pytest.raises(ValueError,match='controlled_common_binding_mismatch'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


def test_rehashed_altered_prediction_does_not_pass_native_replay():
    f,r,b,p=fabricated();r['predictions'][0]['full_probability']=.999
    b['original_record_sha256']=digest(r);p['common_binding_sha256']=digest(b)
    with pytest.raises(ValueError,match='controlled_original_record_replay_mismatch'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


@pytest.mark.parametrize('field,value',[('parent_family','residual_half'),('model_sha256','9'*64),('strengths',[1.,.7]),('strengths',[True,.5])])
def test_different_parent_or_search_strength_is_rejected(field,value):
    f,r,b,p=fabricated();p[field]=value
    with pytest.raises(ValueError,match='controlled_parent_or_strength_changed'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


def test_mutated_effective_scoring_policy_is_rejected_by_native_scorer():
    from dataclasses import replace
    from src.predictor.market_form_residual import ResidualContractError
    f,r,b,p=fabricated();f=replace(f,full_strength=.75)
    with pytest.raises(ResidualContractError):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


def test_outcome_field_is_not_accepted_even_with_changed_reference_hashes():
    from src.predictor.market_form_residual import ResidualContractError
    f,r,b,p=fabricated();r['inputs']['runners'][0]['winner']=True
    b['original_record_sha256']=digest(r);p['common_binding_sha256']=digest(b)
    with pytest.raises(ResidualContractError,match='contains_outcome'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime.now(timezone.utc),execute=True)


def test_derivation_cannot_be_backdated_before_original_seal():
    f,r,b,p=fabricated()
    with pytest.raises(ValueError,match='controlled_seal_timing_invalid'):
        derive_controlled_pair(f,r,b,p,derived_at=datetime(2026,1,1,11,52,tzinfo=timezone.utc),execute=True)


def test_repeated_derivation_is_deterministic_and_not_half_of_probability():
    f,r,b,p=fabricated();now=datetime(2026,10,4,tzinfo=timezone.utc)
    a=derive_controlled_pair(f,r,b,p,derived_at=now,execute=True)
    assert derive_controlled_pair(f,r,b,p,derived_at=now,execute=True)==a
    full,half=a['candidates']
    assert half['probabilities'] != [x*.5 for x in full['probabilities']]
    assert full['candidate_id'] != 'residual_box' and half['candidate_id'] != 'residual_half'
