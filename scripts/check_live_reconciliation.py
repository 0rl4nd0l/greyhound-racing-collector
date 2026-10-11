#!/usr/bin/env python3
"""Independent continuation checks; imports independent arithmetic, never a scorer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path

try:
    from .check_live_benchmark_independently import (
        audit_integrated_sources, check_membership, check_primary_selector_closure,
        check_selectors, compare_scorecard, compute_race,
    )
except ImportError:
    from check_live_benchmark_independently import (
        audit_integrated_sources, check_membership, check_primary_selector_closure,
        check_selectors, compare_scorecard, compute_race,
    )


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_forecast_continuity(before, after):
    """Outcome availability cannot edit, omit, add or replace a primary forecast."""
    for rows in (before, after):
        if len({r['race_id'] for r in rows}) != len(rows):
            raise ValueError('duplicate primary forecast')
    if before != after:
        raise ValueError('original forecast population changed')
    return {'status': 'PASS', 'unchanged_forecasts': len(before)}


def check_original_forecasts(forecasts, membership_path):
    """Reauthenticate all primary p, odds, fields and original prediction identity."""
    membership = read(membership_path)
    allowed = {r['record_key']: r for r in membership['records']
               if r['access_disposition'] == 'AUTHORIZED_NONRESERVED_RETAINED_OPERATIONAL'}
    raw = Path(membership['source']).read_bytes().splitlines()
    primary = {}
    for member in allowed.values():
        original = json.loads(raw[member['line'] - 1])
        key = (original['score_timestamp'], original['record_key'])
        race = original['race_id']
        if race not in primary or key > primary[race][0]:
            primary[race] = key, original
    if set(primary) != {r['race_id'] for r in forecasts}:
        raise ValueError('primary race set differs from admitted originals')
    for forecast in forecasts:
        original = primary[forecast['race_id']][1]
        if forecast['prediction_id'] != original['record_key']:
            raise ValueError('primary forecast horizon changed')
        if forecast['model_version'] != original['model_sha256'] + ':full':
            raise ValueError('original model changed')
        expected = [{'runner_id': r['runner_id'], 'box': r['box_number'],
                     'probability': r['full_probability'], 'decimal_odds': r['strict_win_odds']}
                    for r in original['predictions']]
        if forecast['runners'] != expected:
            raise ValueError('original probability odds or field changed')
        if (forecast['predicted_at'] != original['score_timestamp'] or
                forecast['jump_at'] != original['jump_timestamp']):
            raise ValueError('original timestamps changed')
        times = [datetime.fromisoformat(forecast[key].replace('Z', '+00:00'))
                 for key in ('quote_at', 'predicted_at', 'sealed_at', 'jump_at')]
        if any(t.tzinfo is None for t in times) or not times[0] <= times[1] <= times[2] < times[3]:
            raise ValueError('forecast chronology invalid')
    return {'status': 'PASS', 'original_forecasts_checked': len(forecasts),
            'original_probabilities_odds_fields_models_timestamps': True}


def check_attempt_units(coverage, forecasts):
    """Derive unique-race outcomes independently from nonreserved attempt rows."""
    groups = defaultdict(list)
    for attempt in coverage['attempts']:
        if not attempt['reserved']:
            groups[attempt['canonical_race_key']].append(attempt)
    successes = {key for key, rows in groups.items() if any(r['status'] == 'APPENDED' for r in rows)}
    failures = set(groups) - successes
    successful_ids = {r['race_id'] for key in successes for r in groups[key] if r['status'] == 'APPENDED'}
    if successful_ids != {r['race_id'] for r in forecasts}:
        raise ValueError('attempt successes differ from original forecast population')
    claimed = coverage['summary']
    for key, value in [('nonreserved_unique_races', len(groups)),
                       ('nonreserved_races_with_append', len(successes)),
                       ('nonreserved_without_append', len(failures))]:
        if claimed[key] != value:
            raise ValueError('unique-race denominator mismatch')
    combinations = Counter()
    for key in failures:
        combinations[tuple(sorted({r.get('failure_reason') or 'UNRESOLVED' for r in groups[key]}))] += 1
    return {'status': 'PASS', 'attempted_unique_races': len(groups),
            'successful_unique_races': len(successes), 'unsuccessful_unique_races': len(failures),
            'nonreserved_attempt_rows': sum(map(len, groups.values())),
            'failure_reason_combinations': [{'reasons': list(k), 'unique_races': n}
                                            for k, n in sorted(combinations.items())]}


def check_frozen_selection(original_plan, current_plan, original_selection, current_selection):
    # Require exact retained artifacts, including the original dataset binding.
    if original_plan != current_plan:
        raise ValueError('frozen selector plan changed')
    if original_selection != current_selection:
        raise ValueError('frozen selections or pass dispositions changed')
    return {'status': 'PASS', 'comparisons': len(current_selection['comparisons']),
            'dispositions': len(current_selection['dispositions'])}


def check_failure_artifacts(coverage, failure_root):
    attempts = read(failure_root / 'attempt-accounting.json')
    races = read(failure_root / 'race-accounting.json')
    corrections = read(failure_root / 'derived-grade-corrections.json')
    summary = read(failure_root / 'failure-summary.json')
    expected = [r for r in coverage['attempts'] if not r['reserved']]
    if len(attempts) != len(expected):
        raise ValueError('failure attempt denominator changed')
    for original, derived in zip(expected, attempts):
        if any(derived.get(k) != v for k, v in original.items()):
            raise ValueError('original failure disposition changed')
    groups = defaultdict(list)
    for attempt in attempts:
        groups[attempt['canonical_race_key']].append(attempt)
    if len(races) != len(groups) or {r['canonical_race_key'] for r in races} != set(groups):
        raise ValueError('failure unique race accounting changed')
    counts = Counter()
    for race in races:
        items = groups[race['canonical_race_key']]
        successful = any(r['status'] == 'APPENDED' for r in items)
        if race['forecast_retained'] != successful or race['attempt_records'] != len(items):
            raise ValueError('failure race status mismatch')
        reasons = sorted({r['failure_reason'] for r in items if r.get('failure_reason')})
        expected_reason = ('FORECAST_RETAINED' if successful else reasons[0]
                           if len(reasons) == 1 else 'MULTIPLE_CAUSES_REQUIRES_REVIEW')
        if race['primary_disposition'] != expected_reason:
            raise ValueError('failure reason mismatch')
        if not successful:
            counts[expected_reason] += 1
    if dict(counts) != summary['primary_failure_race_counts']:
        raise ValueError('ranked failure cause mismatch')
    correction_keys = {(r['canonical_race_key'], r['attempt_reference']) for r in corrections}
    if len(correction_keys) != len(corrections):
        raise ValueError('duplicate derived correction')
    for correction in corrections:
        if correction['original_records_modified'] or correction['new_forecast_generated']:
            raise ValueError('offline validation represented as new forecast')
        candidates = [r for r in groups[correction['canonical_race_key']]
                      if r['path'] == correction['attempt_reference']]
        if len(candidates) != 1 or candidates[0]['failure_reason'] != 'feature_row_target_grade_mismatch':
            raise ValueError('correction not bound to original grade failure')
        if candidates[0]['original_validation'] != {'status': 'BLOCKED', 'reason': 'feature_row_target_grade_mismatch'}:
            raise ValueError('original failure not reproduced')
        if candidates[0]['repaired_validation']['status'] != 'VALIDATED':
            raise ValueError('correction did not pass complete feature validator')
    refs = read(failure_root / 'source-references.json')
    for reference in refs:
        if digest(reference['path']) != reference['sha256']:
            raise ValueError('failure source changed')
    return {'status': 'PASS', 'source_hashes_verified': len(refs),
            'primary_failure_race_counts': dict(counts), 'grade_validation_corrections': len(corrections),
            'grade_validation_unique_races': len({r['canonical_race_key'] for r in corrections}),
            'grade_validation_chronology_valid_unique_races': len({r['canonical_race_key'] for r in corrections
                                                                  if r['feature_to_score_chronology_valid']}),
            'forecast_recoveries': 0}


def check_reconciliation(forecasts, result_root, strict_records):
    records = read(result_root / 'reconciliation.json')['records']
    projected = read(result_root / 'archive-result-projection.json')['records']
    summary = read(result_root / 'summary.json')
    corrections = read(result_root / 'correction-trail.json')
    originals = {r['race_id']: r for r in forecasts}
    if len(records) != len(originals) or {r['race_id'] for r in records} != set(originals):
        raise ValueError('reconciliation omitted or duplicated original forecast')
    counts = Counter(r['current_disposition'] for r in records)
    if dict(counts) != summary['current_dispositions']:
        raise ValueError('reconciliation disposition counts disagree')
    joined = {r['race_id']: r for r in projected}
    if len(joined) != len(projected):
        raise ValueError('duplicate projected result')
    for record in records:
        original = originals[record['race_id']]
        if record['prediction_id'] != original['prediction_id'] or record['native_record_id'] != original['prediction_id']:
            raise ValueError('reconciliation original identity replaced')
        expected_field = [(r['box'], r['runner_id']) for r in original['runners']]
        actual_field = [(r['box'], r['runner_id']) for r in record['original_runner_field']]
        if actual_field != expected_field:
            raise ValueError('reconciliation original runner field replaced')
        if not record['remaining_gaps'] and record['current_disposition'] in (
                'RETAINED_EVIDENCE_INCOMPLETE_OR_CONTRADICTORY',
                'REQUIRES_SPECIFIC_FRESH_OFFICIAL_RESULT_ACQUISITION'):
            raise ValueError('unresolved record lacks concrete gap')
    for record in strict_records:
        source = joined[record['race_id']]
        if source.get('official_runner_identity_observed') is not True:
            raise ValueError('unobserved official runner identity promoted to strict')
    audit_path = result_root / 'correction-trail-audit.json'
    if audit_path.exists():
        audit = read(audit_path)
        historical_path = Path(audit['preserved_original'])
        if digest(historical_path) != audit['sha256']:
            raise ValueError('original duplicate correction history was not preserved')
        history = read(historical_path)
        unique = {json.dumps(r, sort_keys=True, separators=(',', ':')) for r in history}
        current = {json.dumps(r, sort_keys=True, separators=(',', ':')) for r in corrections}
        if unique != current or len(history)-len(unique) != audit['duplicate_append_entries']:
            raise ValueError('correction replay deduplication changed actual events')
    if len(corrections) != summary['recovered_archive_joins']:
        raise ValueError('recovered join count mismatch')
    if len({r['race_id'] for r in corrections}) != len(corrections):
        raise ValueError('duplicate correction event')
    for correction in corrections:
        source = joined[correction['race_id']]
        expected = [(r['box_number'], r['finish_position']) for r in source['runners']]
        if sorted(map(tuple, correction['preserved_finishes'])) != sorted(expected):
            raise ValueError('correction invented finishing positions')
    references = read(result_root / 'archive-references.json')['artifacts']
    for reference in references:
        if digest(reference['path']) != reference['sha256']:
            raise ValueError('native result archive hash changed')
    # Read only exact already-allowlisted line bodies; reserved lines remain bytes.
    cache = {}
    native_rows = read(result_root / 'allowlisted-native-jsonl-records.json')
    for retained in native_rows:
        path = retained['path']
        if path not in cache:
            cache[path] = Path(path).read_bytes().splitlines()
        if json.loads(cache[path][retained['line'] - 1]) != retained['row']:
            raise ValueError('allowlisted native projection differs from source')
        race_id = retained['row'].get('race_id')
        if race_id is not None and race_id not in originals:
            raise ValueError('native projection includes out-of-population race')
    for projection in projected:
        for kind, filename in (('races', 'official_result_races.jsonl'),
                               ('runners', 'official_result_runners.jsonl')):
            for row in projection[kind]:
                source = str(Path(row['source_artifact_dir']) / filename)
                native = json.loads(row['row_json'])
                if source not in cache or row['row_json'].encode() not in cache[source]:
                    # JSON whitespace is not identity; compare the exact admitted decoded row.
                    matches = [r for r in native_rows if r['path'] == source and r['row'] == native]
                    if not matches:
                        raise ValueError('projected result row lacks allowlisted native witness')
                for key, value in native.items():
                    if key in row and row[key] != value:
                        raise ValueError('projected result column differs from native row')
    return {'status': 'PASS', 'original_forecasts': len(records),
            'current_dispositions': dict(counts), 'recovered_archive_joins': len(corrections),
            'source_hashes_verified': len(references), 'native_jsonl_rows_checked': len(native_rows),
            'strict_races': len({r['race_id'] for r in strict_records})}


def check_missing_outcome_bounds(forecasts, outcome_records, census, claimed):
    """Enumerate each possible winning runner using independent proper-loss arithmetic."""
    import math
    outcomes = {(r['race_id'], r['model_version']): r for r in outcome_records}

    def compare(rows, observed):
        ranges = {'delta_log_loss': [0., 0.], 'delta_brier': [0., 0.], 'delta_top_credit': [0., 0.]}
        known = 0
        for record in rows:
            outcome = outcomes.get((record['race_id'], record['model_version']))
            candidates = [r['runner_id'] for r in record['runners']]
            if outcome is not None:
                candidates = [r['runner_id'] for r in outcome['runners'] if r['winner']]
                known += 1
            possibilities = []
            for winner in candidates:
                runners = [{**r, 'model_probability': r['probability'], 'winner': int(r['runner_id'] == winner)}
                           for r in record['runners']]
                possibilities.append(compute_race(runners)['difference'])
            for key, metric in [('delta_log_loss', 'log_loss'), ('delta_brier', 'brier'),
                                ('delta_top_credit', 'top_accuracy')]:
                ranges[key][0] += min(r[metric] for r in possibilities)
                ranges[key][1] += max(r[metric] for r in possibilities)
        if observed['races'] != len(rows) or observed['known_outcomes'] != known or observed['missing_outcomes'] != len(rows)-known:
            raise ValueError('missing-outcome sensitivity denominator mismatch')
        for metric, values in ranges.items():
            expected = [v/len(rows) for v in values] if rows else None
            actual = observed['conditional_mean_difference_bounds'][metric]
            if expected is None:
                if actual is not None:
                    raise ValueError('empty sensitivity has invented estimate')
            elif any(not math.isclose(a, e, rel_tol=1e-10, abs_tol=1e-12) for a, e in zip(actual, expected)):
                raise ValueError('missing-outcome bound calculation mismatch')

    compare(forecasts, claimed['all_210'])
    if len(claimed['selected_groups']) != len(census['comparisons']):
        raise ValueError('missing sensitivity selector group')
    for group, frozen in zip(claimed['selected_groups'], census['comparisons']):
        if group['selector'] != frozen['selector'] or group['target_coverage'] != frozen['target_coverage']:
            raise ValueError('sensitivity selector identity mismatch')
        for output_name, key in [('selected', 'selected_race_ids'), ('market_confidence', 'market_confidence_race_ids')]:
            ids = set(frozen[key])
            compare([r for r in forecasts if r['race_id'] in ids], group[output_name])
    return {'status': 'PASS', 'possible_winners_independently_enumerated': True,
            'groups_checked': 1+2*len(census['comparisons'])}


def run(baseline, completed):
    membership = baseline / 'inventory/residual-membership.json'
    boundary = baseline / 'inventory/access-boundary.json'
    result = {'membership': check_membership(membership, boundary)}
    before = read(baseline / 'completed/forecast-dataset.json')['records']
    after_path = completed / 'forecast-dataset.json'
    after = (read(after_path)['records'] if after_path.exists() else
             read(completed.parent / 'qualified/forecast-records.json'))
    result['forecast_continuity'] = check_forecast_continuity(before, after)
    result['original_forecasts'] = check_original_forecasts(after, membership)
    result['attempt_units'] = check_attempt_units(read(baseline / 'inventory/july-operational-coverage.json'), after)
    plan_path = completed / 'forecast-selector-plan.FROZEN.json'
    selection_path = completed / 'forecast-selector-dispositions.FROZEN.json'
    if not plan_path.exists():
        plan_path = completed / 'forecast-selector-plan.json'
        selection_path = completed / 'forecast-selector-results.json'
    plan = read(plan_path)
    selection = read(selection_path)
    result['frozen_selection'] = check_frozen_selection(
        read(baseline / 'completed/forecast-selector-plan.json'), plan,
        read(baseline / 'completed/forecast-selector-results.json'), selection)
    result['independent_selector_reconstruction'] = check_selectors(after, plan, selection)
    result['scorecards'] = {}
    for kind in ('strict', 'diagnostic'):
        path = completed / f'{kind}-dataset.json'
        dataset = read(path)
        scorecard = read(completed / f'{kind}-scorecard.json')
        if scorecard['dataset_sha256'] != digest(path):
            raise ValueError('scorecard dataset binding mismatch')
        diagnostic = kind == 'diagnostic'
        records = dataset['records']
        selector_path = completed / f'{kind}-selector-evaluation.json'
        if not selector_path.exists():
            selector_path = completed / f'primary-{kind}-selector-evaluation.json'
        result['scorecards'][kind] = {
            'metrics': compare_scorecard(records, scorecard, diagnostic),
            'sources': audit_integrated_sources(records, membership),
            'primary_selectors': check_primary_selector_closure(
                records, selection, read(selector_path), diagnostic),
        }
        bounds_path = completed / f'{kind}-missing-outcome-sensitivity.json'
        if bounds_path.exists():
            result['scorecards'][kind]['missing_outcome_sensitivity'] = check_missing_outcome_bounds(
                after, records, selection, read(bounds_path))
    result['status'] = 'PASS'
    result['checked_artifacts'] = {str(path): digest(path) for path in sorted(completed.glob('*.json'))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--completed', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--failure-root', type=Path)
    parser.add_argument('--result-root', type=Path)
    args = parser.parse_args()
    result = run(args.baseline_root, args.completed)
    if args.failure_root:
        result['failure_audit'] = check_failure_artifacts(
            read(args.baseline_root / 'inventory/july-operational-coverage.json'), args.failure_root)
    if args.result_root:
        result['reconciliation'] = check_reconciliation(
            read(args.baseline_root / 'completed/forecast-dataset.json')['records'], args.result_root,
            read(args.completed / 'strict-dataset.json')['records'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
