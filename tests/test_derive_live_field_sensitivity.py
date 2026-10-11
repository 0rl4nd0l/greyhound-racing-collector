import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from derive_live_field_sensitivity import run


def put(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def inputs(tmp_path):
    completed = tmp_path / 'completed'
    completed.mkdir()
    forecasts, originals = [], []
    for race in ['A', 'B', 'C']:
        runners = [{'runner_id': race + '|box:1|dog:ALPHA', 'box': 1,
                    'probability': .8, 'decimal_odds': 2.},
                   {'runner_id': race + '|box:2|dog:BETA', 'box': 2,
                    'probability': .2, 'decimal_odds': 2.}]
        row = {'race_id': race, 'prediction_id': race, 'model_version': 'frozen:full',
               'date': '2026-07-19', 'predicted_at': 't1', 'jump_at': 't2', 'runners': runners}
        forecasts.append(row)
        originals.append({'original': {'race_id': race, 'record_key': race,
            'model_sha256': 'frozen', 'score_timestamp': 't1', 'jump_timestamp': 't2',
            'predictions': [{'runner_id': r['runner_id'], 'box_number': r['box'],
                             'full_probability': r['probability'], 'strict_win_odds': 2.}
                            for r in runners]}})
    forecast_ref = put(completed / 'forecast-dataset.json', {'records': forecasts})
    plan_ref = put(completed / 'forecast-selector-plan.FROZEN.json',
                   {'dataset_sha256': forecast_ref['sha256']})
    census = {'dataset_sha256': forecast_ref['sha256'], 'plan_sha256': plan_ref['sha256'],
        'dispositions': [{'race_id': 'A', 'selected': True}, {'race_id': 'B', 'selected': False},
                         {'race_id': 'C', 'selected': True}],
        'comparisons': [{'selector': 'confidence', 'target_coverage': .5,
                         'selected_race_ids': ['A', 'C'], 'market_confidence_race_ids': ['B', 'C']}]}
    census_ref = put(completed / 'forecast-selector-dispositions.FROZEN.json', census)
    originals_ref = put(tmp_path / 'authenticated-originals.json', originals)
    access_ref = put(tmp_path / 'access-boundary.json', {'scope': 'synthetic nonreserved'})
    membership_ref = put(tmp_path / 'residual-membership.json', {'scope': 'synthetic nonreserved'})
    population = put(tmp_path / 'population.json', {'authority': 'exact nonreserved test population',
        'frozen_references': [forecast_ref, plan_ref, census_ref, originals_ref, access_ref, membership_ref],
        'records': [{k: r[k] for k in ['race_id', 'prediction_id', 'model_version', 'date']}
                    for r in forecasts]})
    known = copy.deepcopy(forecasts[0])
    known['official_runner_identity_observed'] = True
    for i, r in enumerate(known['runners']):
        r['winner'] = int(i == 0)
    known_ref = put(completed / 'strict-dataset.json', {'records': [known]})
    legacy_ref = put(completed / 'strict-missing-outcome-sensitivity.json',
                     {'all_210': {'conditional_mean_difference_bounds': 'obsolete fixture'}})
    manifest = put(completed / 'manifest.json', {'artifacts': {
        Path(r['path']).name: r['sha256'] for r in [forecast_ref, plan_ref, census_ref, known_ref, legacy_ref]}})
    dispositions = put(tmp_path / 'dispositions.json', [
        {'race_id': 'A', 'disposition': 'STRICT'}, {'race_id': 'B', 'disposition': 'MISSING'},
        {'race_id': 'C', 'disposition': 'FIELD_CHANGED'}])
    witness = put(tmp_path / 'witness.json', [{'race_id': 'C',
        'forecast_field': [{'box': 2, 'name_token': 'BETA'}],
        'source_rows': [{'dog_anchor_text': ['Beta'], 'rug_ids': ['rug_2'], 'status': 'L/SCR'}]}])
    exclusion = {'race_id': 'C', 'prediction_id': 'C', 'runner_id': 'C|box:2|dog:BETA',
        'box': 2, 'official_name': 'Beta', 'official_status': 'L/SCR', 'witness_reference': witness}
    reviewer = put(tmp_path / 'review.json', {'status': 'PASS_REVIEWED_ORIGINAL_FIELD_EXCLUSIONS',
        'actual_field_change_verified': True,
        'records': [exclusion]})
    exclusions = put(tmp_path / 'exclusions.json', {'schema': 'reviewed_original_field_exclusions_v1',
        'population_reference': population, 'review_reference': reviewer, 'records': [exclusion]})
    return dict(population_reference=population, exclusions_reference=exclusions,
        completed_manifest_reference=manifest, dispositions_reference=dispositions,
        output_dir=tmp_path / 'output')


def test_verified_field_change_invalidates_all_population_but_preserves_every_record(inputs):
    result = run(**inputs)
    assert result['original_population']['races'] == 3
    assert result['original_population']['status'] == 'UNCHANGED_FIELD_ASSUMPTION_CONTRADICTED'
    assert result['original_population']['bounds_applicability'] == 'NOT_APPLICABLE'
    assert result['original_population']['legacy_sensitivity_reference']['sha256']
    assert result['original_population']['conditional_mean_difference_bounds'] is None
    scope = result['potentially_eligible_population']
    assert (scope['races'], scope['known_outcomes'], scope['missing_outcomes']) == (2, 1, 1)
    assert scope['conditional_mean_difference_bounds']['delta_log_loss'] == pytest.approx(
        [-0.4700036292457356, 0.2231435513142097])
    assert scope['conditional_mean_difference_bounds']['delta_brier'] == pytest.approx([-.42, .18])
    assert scope['conditional_mean_difference_bounds']['delta_top_credit'] == [0., .5]
    group = result['selected_groups'][0]
    assert group['original_selected_race_ids'] == ['A', 'C']
    assert group['potentially_eligible_selected_race_ids'] == ['A']
    assert group['original_market_confidence_race_ids'] == ['B', 'C']
    assert group['potentially_eligible_market_confidence_race_ids'] == ['B']
    assert group['selected_scope_counts'] == {'original': 2, 'field_excluded': 1,
        'potentially_eligible': 1, 'known_outcomes': 1, 'missing_outcomes': 0}
    out = inputs['output_dir']
    assert (out / 'integration-dispositions.PRESERVED.json').read_bytes() == Path(
        inputs['dispositions_reference']['path']).read_bytes()
    assert len(json.loads((out / 'scope-dispositions.json').read_text())['records']) == 3


def change_known(inputs, mutate):
    manifest_path = Path(inputs['completed_manifest_reference']['path'])
    source = manifest_path.parent / 'strict-dataset.json'
    payload = json.loads(source.read_text())
    mutate(payload['records'])
    reference = put(source, payload)
    manifest = json.loads(manifest_path.read_text())
    manifest['artifacts']['strict-dataset.json'] = reference['sha256']
    inputs['completed_manifest_reference'] = put(manifest_path, manifest)


def change_exclusions(inputs, mutate, approved=False):
    path = Path(inputs['exclusions_reference']['path'])
    value = json.loads(path.read_text())
    mutate(value['records'])
    if approved:
        review_path = Path(value['review_reference']['path'])
        review = json.loads(review_path.read_text())
        review['records'] = value['records']
        value['review_reference'] = put(review_path, review)
    inputs['exclusions_reference'] = put(path, value)


def test_hash_mismatch_fails_before_any_label_decoding(inputs):
    Path(inputs['population_reference']['path']).write_text('{}')
    (Path(inputs['completed_manifest_reference']['path']).parent / 'strict-dataset.json').unlink()
    with pytest.raises(ValueError, match='reference_hash_mismatch'):
        run(**inputs)
    assert not inputs['output_dir'].exists()


def test_unreviewed_exclusion_is_rejected_before_labels(inputs):
    change_exclusions(inputs, lambda rows: rows[0].update(official_status='SCR'))
    (Path(inputs['completed_manifest_reference']['path']).parent / 'strict-dataset.json').unlink()
    with pytest.raises(ValueError, match='not_exactly_reviewer_approved'):
        run(**inputs)


def test_nonfinisher_cannot_be_excluded_as_field_change(inputs):
    change_exclusions(inputs, lambda rows: rows[0].update(official_status='DNF'), approved=True)
    with pytest.raises(ValueError, match='not_original_runner_scratch'):
        run(**inputs)


def test_duplicate_known_runner_is_rejected(inputs):
    change_known(inputs, lambda rows: rows[0]['runners'].append(copy.deepcopy(rows[0]['runners'][1])))
    with pytest.raises(ValueError, match='duplicate_known_runner'):
        run(**inputs)


def test_diagnostic_label_cannot_be_used_as_known(inputs):
    change_known(inputs, lambda rows: rows[0].update(official_runner_identity_observed=False))
    with pytest.raises(ValueError, match='known_outcome_not_strict'):
        run(**inputs)


def test_excluded_race_cannot_also_have_strict_known_outcome(inputs):
    def add_excluded(rows):
        row = copy.deepcopy(rows[0])
        row.update(race_id='C', prediction_id='C')
        for runner in row['runners']:
            runner['runner_id'] = 'C' + runner['runner_id'][1:]
        rows.append(row)
    change_known(inputs, add_excluded)
    with pytest.raises(ValueError, match='strict_outcome_conflicts_with_verified_exclusion'):
        run(**inputs)


def test_changed_known_probability_is_rejected(inputs):
    change_known(inputs, lambda rows: rows[0]['runners'][0].update(probability=.9))
    with pytest.raises(ValueError, match='original_prediction_or_price_changed'):
        run(**inputs)


def test_existing_output_is_never_overwritten(inputs):
    run(**inputs)
    original = (inputs['output_dir'] / 'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='output_directory_must_be_new_or_empty'):
        run(**inputs)
    assert (inputs['output_dir'] / 'manifest.json').read_bytes() == original


def test_reviewer_must_explicitly_verify_actual_field_change(inputs):
    path = Path(inputs['exclusions_reference']['path'])
    value = json.loads(path.read_text())
    review_path = Path(value['review_reference']['path'])
    review = json.loads(review_path.read_text())
    review['actual_field_change_verified'] = False
    value['review_reference'] = put(review_path, review)
    inputs['exclusions_reference'] = put(path, value)
    with pytest.raises(ValueError, match='not_exactly_reviewer_approved'):
        run(**inputs)


def test_known_outcomes_must_use_original_primary_prediction(inputs):
    change_known(inputs, lambda rows: rows[0].update(prediction_id='later-authentic-forecast'))
    with pytest.raises(ValueError, match='known_outcome_primary_forecast_changed'):
        run(**inputs)


def test_scratch_status_requires_exact_witness(inputs):
    change_exclusions(inputs, lambda rows: rows[0].update(official_status='SCR'), approved=True)
    with pytest.raises(ValueError, match='exclusion_official_status_witness_mismatch'):
        run(**inputs)


def test_no_known_labels_uses_all_possible_original_winners(inputs):
    change_known(inputs, lambda rows: rows.clear())
    result = run(**inputs)
    scope = result['potentially_eligible_population']
    assert (scope['known_outcomes'], scope['missing_outcomes']) == (0, 2)
    assert scope['conditional_mean_difference_bounds']['delta_log_loss'] == pytest.approx(
        [-0.4700036292457356, 0.9162907318741551])
    assert scope['conditional_mean_difference_bounds']['delta_brier'] == pytest.approx([-.42, .78])
