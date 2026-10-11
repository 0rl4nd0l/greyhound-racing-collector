#!/usr/bin/env python3
"""Derived conditional bounds after independently verified original-field exclusions.

This utility never changes forecasts, selections, benchmark qualifications or
existing reports. Inputs require external SHA256 pins; known labels come only
from the accepted strict dataset. Bounds reuse the frozen benchmark calculation.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from rerun_frozen_live_reconciliation import paired_bounds, verify_originals, write


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference):
    raw = Path(reference['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('reference_hash_mismatch:' + reference['path'])
    return json.loads(raw)


def unique(rows, key, error):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(error)
    return result


def run(*, population_reference, exclusions_reference, completed_manifest_reference,
        dispositions_reference, output_dir):
    # Membership and access controls precede any forecast or outcome decoding.
    population = checked(population_reference)
    if not population.get('authority') or not population.get('records'):
        raise ValueError('population_authority_or_membership_missing')
    members = unique(population['records'], lambda r: r['race_id'], 'duplicate_population')
    frozen = population['frozen_references']

    def frozen_ref(*names):
        matches = [r for r in frozen if Path(r['path']).name in names]
        if len(matches) != 1:
            raise ValueError('frozen_reference_missing_or_ambiguous:' + names[0])
        return matches[0]

    checked(frozen_ref('access-boundary.json'))
    checked(frozen_ref('residual-membership.json'))
    exclusions = checked(exclusions_reference)
    if (exclusions.get('schema') != 'reviewed_original_field_exclusions_v1'
            or exclusions.get('population_reference') != population_reference):
        raise ValueError('exclusion_population_or_schema_mismatch')
    review = checked(exclusions['review_reference'])
    if (review.get('status') != 'PASS_REVIEWED_ORIGINAL_FIELD_EXCLUSIONS'
            or review.get('actual_field_change_verified') is not True
            or exclusions.get('records') != review.get('records') or not review.get('records')):
        raise ValueError('exclusions_not_exactly_reviewer_approved')
    excluded = unique(exclusions['records'], lambda r: r['race_id'], 'duplicate_exclusion')
    if not set(excluded) <= set(members):
        raise ValueError('exclusion_outside_authorized_population')
    dispositions = checked(dispositions_reference)
    by_disposition = unique(dispositions, lambda r: r['race_id'], 'duplicate_disposition')
    if set(by_disposition) != set(members):
        raise ValueError('disposition_population_mismatch')
    manifest = checked(completed_manifest_reference)
    completed = Path(completed_manifest_reference['path']).parent

    def completed_ref(name):
        return {'path': str(completed / name), 'sha256': manifest['artifacts'][name]}

    forecast_ref = frozen_ref('forecast-dataset.json')
    plan_ref = frozen_ref('forecast-selector-plan.json', 'forecast-selector-plan.FROZEN.json')
    census_ref = frozen_ref('forecast-selector-results.json', 'forecast-selector-dispositions.FROZEN.json')
    for name, original in [('forecast-dataset.json', forecast_ref),
                           ('forecast-selector-plan.FROZEN.json', plan_ref),
                           ('forecast-selector-dispositions.FROZEN.json', census_ref)]:
        if completed_ref(name)['sha256'] != original['sha256']:
            raise ValueError('frozen_completed_input_changed:' + name)
        checked(completed_ref(name))
    forecasts = checked(forecast_ref)['records']
    by_forecast = unique(forecasts, lambda r: r['race_id'], 'duplicate_forecast')
    if set(by_forecast) != set(members):
        raise ValueError('forecast_population_mismatch')
    for race, forecast in by_forecast.items():
        if any(forecast[key] != members[race][key]
               for key in ('prediction_id', 'model_version', 'date')):
            raise ValueError('frozen_primary_identity_changed')
    originals = checked(frozen_ref('authenticated-originals.json'))
    verify_originals(forecasts, originals)
    plan, census = checked(plan_ref), checked(census_ref)
    if (plan['dataset_sha256'] != forecast_ref['sha256']
            or census['dataset_sha256'] != forecast_ref['sha256']
            or census['plan_sha256'] != plan_ref['sha256']):
        raise ValueError('frozen_selector_input_binding_changed')
    for race, record in excluded.items():
        forecast = by_forecast[race]
        if record['prediction_id'] != forecast['prediction_id']:
            raise ValueError('exclusion_forecast_identity_changed')
        runner = [r for r in forecast['runners'] if r['runner_id'] == record['runner_id']
                  and r['box'] == record['box']]
        token = re.sub('[^A-Z0-9]', '', record['official_name'].upper())
        if (len(runner) != 1 or not record['runner_id'].endswith('|dog:' + token)
                or record['official_status'] not in {'SCR', 'L/SCR'}):
            raise ValueError('exclusion_not_original_runner_scratch')
        witness = checked(record['witness_reference'])
        rows = [r for r in witness if r['race_id'] == race]
        if len(rows) != 1 or {'box': record['box'], 'name_token': token} not in rows[0]['forecast_field']:
            raise ValueError('exclusion_original_field_witness_mismatch')
        source_matches = [r for r in rows[0]['source_rows']
            if r['dog_anchor_text'] == [record['official_name']]
            and r['rug_ids'] == ['rug_' + str(record['box'])]
            and r['status'] == record['official_status']]
        if len(source_matches) != 1:
            raise ValueError('exclusion_official_status_witness_mismatch')

    # Only after membership, access, original identities and reviewer approval
    # are authenticated do we decode accepted evaluation labels.
    known = checked(completed_ref('strict-dataset.json'))['records']
    unique(known, lambda r: (r['race_id'], r['model_version']), 'duplicate_known_outcome')
    if any(r['race_id'] not in members for r in known):
        raise ValueError('known_outcome_outside_population')
    if any(r.get('official_runner_identity_observed') is not True for r in known):
        raise ValueError('known_outcome_not_strict')
    if any(r['race_id'] in excluded for r in known):
        raise ValueError('strict_outcome_conflicts_with_verified_exclusion')
    if any(r['prediction_id'] != members[r['race_id']]['prediction_id'] for r in known):
        raise ValueError('known_outcome_primary_forecast_changed')
    for row in known:
        unique(row['runners'], lambda r: r['runner_id'], 'duplicate_known_runner')
    verify_originals(known, originals)
    full = [r for r in known if r['model_version'] == by_forecast[r['race_id']]['model_version']]
    for row in full:
        if {r['winner'] for r in row['runners']} - {0, 1} or sum(r['winner'] for r in row['runners']) != 1:
            raise ValueError('known_outcome_not_unique_binary_winner')
    potential = [r for r in forecasts if r['race_id'] not in excluded]
    legacy_ref = completed_ref('strict-missing-outcome-sensitivity.json')
    checked(legacy_ref)

    def bounds(rows):
        value = paired_bounds(rows, full)
        value['assumption'] = ('Every remaining race must eventually prove its original field '
            'unchanged with one winner. Known labels are accepted strict outcomes only. '
            'These conditional identification bounds are not confidence intervals '
            'and do not qualify missing fields.')
        return value

    report = {
        'schema': 'derived_original_field_conditional_sensitivity_v1',
        'original_population': {'races': len(forecasts),
            'status': 'UNCHANGED_FIELD_ASSUMPTION_CONTRADICTED',
            'bounds_applicability': 'NOT_APPLICABLE',
            'legacy_sensitivity_reference': legacy_ref,
            'conditional_mean_difference_bounds': None,
            'verified_original_field_exclusions': list(excluded)},
        'potentially_eligible_population': bounds(potential),
        'known_outcome_source': 'ACCEPTED_STRICT_RECORDS_ONLY_NO_DIAGNOSTIC_LABELS',
        'remaining_scope_is_conditional_not_qualification': True,
        'selected_groups': [],
        'limitations': ['Remaining races must prove unchanged original fields and one winner.',
            'Bounds are conditional identification ranges, not confidence intervals.',
            'Original selections and passes are preserved; excluded selections are not replaced.',
            'No profitability or predictive-advantage inference is made.'],
        'model_fits': 0, 'calibration_fits': 0, 'selector_threshold_changes': 0,
    }
    for group in census['comparisons']:
        derived = {'selector': group['selector'], 'target_coverage': group['target_coverage']}
        for role, field in [('selected', 'selected_race_ids'),
                            ('market_confidence', 'market_confidence_race_ids')]:
            ids = group[field]
            if len(set(ids)) != len(ids) or not set(ids) <= set(members):
                raise ValueError('selector_membership_changed')
            retained = [race for race in ids if race not in excluded]
            derived['original_' + field] = ids
            derived['potentially_eligible_' + field] = retained
            derived['excluded_' + field] = [race for race in ids if race in excluded]
            derived[role] = bounds([by_forecast[race] for race in retained])
            derived[role + '_scope_counts'] = dict(original=len(ids),
                field_excluded=len(ids)-len(retained), potentially_eligible=len(retained),
                known_outcomes=derived[role]['known_outcomes'],
                missing_outcomes=derived[role]['missing_outcomes'])
        report['selected_groups'].append(derived)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output_directory_must_be_new_or_empty')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'integration-dispositions.PRESERVED.json').write_bytes(Path(dispositions_reference['path']).read_bytes())
    (output / 'forecast-selector-dispositions.PRESERVED.json').write_bytes(Path(census_ref['path']).read_bytes())
    write(output / 'scope-dispositions.json', {'records': [dict(
        race_id=r['race_id'], prediction_id=r['prediction_id'],
        original_disposition=by_disposition[r['race_id']],
        potentially_eligible=r['race_id'] not in excluded,
        exclusion=excluded.get(r['race_id'])) for r in forecasts]})
    report['input_references'] = {'population': population_reference,
        'exclusions': exclusions_reference, 'completed_manifest': completed_manifest_reference,
        'dispositions': dispositions_reference}
    report['implementation_references'] = [
        {'path': str(Path(__file__).resolve()), 'sha256': digest(__file__)},
        {'path': str(Path(__file__).with_name('rerun_frozen_live_reconciliation.py').resolve()),
         'sha256': digest(Path(__file__).with_name('rerun_frozen_live_reconciliation.py'))}]
    write(output / 'conditional-sensitivity.json', report)
    write(output / 'manifest.json', {'artifacts': {p.name: digest(p)
        for p in sorted(output.glob('*.json')) if p.name != 'manifest.json'}})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('population', 'exclusions', 'completed-manifest', 'dispositions'):
        parser.add_argument('--' + name, required=True, type=Path)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    kwargs = {name + '_reference': {'path': str(getattr(args, name)),
              'sha256': getattr(args, name + '_sha256')}
              for name in ('population', 'exclusions', 'completed_manifest', 'dispositions')}
    report = run(**kwargs, output_dir=args.output_dir)
    print(json.dumps({'original_races': report['original_population']['races'],
        'potentially_eligible_races': report['potentially_eligible_population']['races']}))


if __name__ == '__main__':
    main()
