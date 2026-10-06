"""Pure, prespecified retrospective comparison of one sectional candidate.

Root authenticates the pinned source/label population before constructing this
packet. This module never reads files, acquires data, changes models, or grants
authority. All returned predictions and losses belong in private output.
"""
from collections import Counter
from datetime import date, datetime
import hashlib
import json
import math
import random
import re


POLICY = 'PAST_SECTIONAL_GRID_TRAIN_VALIDATION_GATE_RETROSPECTIVE_V1'
BETAS = (0.0, 0.1, 0.25, 0.5, 1.0)
SPLITS = {'training': ['2026-10-01'], 'validation': ['2026-10-02'],
          'evaluation': ['2026-10-03']}
ELIGIBLE = {'FULL_ORDER_WIN_ELIGIBLE', 'KNOWN_NONFINISH_WIN_ELIGIBLE'}
METHODS = ('market', 'baseline', 'baseline_speed')
MIN_BOOTSTRAP_DATES, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 5, 2000, 20261006


class EvaluationRejected(ValueError):
    """Invalid shared input; safe reason codes never include source values."""


def _require(condition, reason):
    if not condition:
        raise EvaluationRejected(reason)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _reference(value):
    _require(isinstance(value, dict) and set(value) == {'path', 'sha256'}
             and isinstance(value['path'], str) and value['path'].startswith('/')
             and isinstance(value['sha256'], str)
             and re.fullmatch('[a-f0-9]{64}', value['sha256']) is not None,
             'SOURCE_REFERENCE_INVALID')


def frozen_protocol(members, *, membership_reference, label_authority_reference,
                    feature_policy_reference, baseline_reproduction_reference,
                    split_dates=None, baseline_reproduction_tolerance=0.0):
    """Record this exact outcome-independent design before root opens labels.

Members are identity/date metadata only. Root binds this document's bytes in a
fresh execution receipt; reference syntax here does not authenticate files.
"""
    splits = SPLITS if split_dates is None else split_dates
    _require(isinstance(splits, dict) and set(splits) == set(SPLITS), 'SPLITS_INVALID')
    for days in splits.values():
        _require(isinstance(days, list) and bool(days) and days == sorted(set(days)), 'SPLITS_INVALID')
        for day in days:
            try:
                _require(isinstance(day, str) and re.fullmatch(r'\d{4}-\d{2}-\d{2}', day) is not None
                    and date.fromisoformat(day).isoformat() == day, 'SPLITS_INVALID')
            except (TypeError, ValueError):
                raise EvaluationRejected('SPLITS_INVALID') from None
    _require(max(splits['training']) < min(splits['validation'])
        and max(splits['validation']) < min(splits['evaluation']), 'SPLITS_NOT_CHRONOLOGICAL')
    _require(type(baseline_reproduction_tolerance) in (int, float)
        and math.isfinite(baseline_reproduction_tolerance)
        and 0 <= baseline_reproduction_tolerance <= 1e-12, 'REPRODUCTION_TOLERANCE_INVALID')
    _require(isinstance(members, list) and bool(members), 'MEMBERSHIP_INVALID')
    for member in members:
        _require(isinstance(member, dict) and set(member) == {'race_id', 'race_date'}
                 and isinstance(member['race_id'], str) and bool(member['race_id'])
                 and member['race_date'] in {d for ds in splits.values() for d in ds},
                 'MEMBERSHIP_INVALID')
    _require(len({m['race_id'] for m in members}) == len(members), 'MEMBERSHIP_DUPLICATE')
    refs = {'membership': membership_reference, 'label_authority': label_authority_reference,
            'feature_policy': feature_policy_reference,
            'baseline_reproduction': baseline_reproduction_reference}
    for reference in refs.values():
        _reference(reference)
    return {'schema_version': 'sectional_speed_evaluation_protocol_v1', 'policy': POLICY,
        'members': sorted(members, key=lambda m: (m['race_date'], m['race_id'])),
        'references': refs, 'splits': {role: list(days) for role, days in splits.items()},
        'coefficient_grid': list(BETAS),
        'baseline_reproduction_max_absolute_tolerance': baseline_reproduction_tolerance,
        'training_selection': 'MINIMUM_MEAN_LOGLOSS_SMALLEST_BETA_EXACT_TIE',
        'validation_gate': 'TRAINED_BETA_ONLY_IF_STRICTLY_LOWER_LOGLOSS_THAN_ZERO',
        'refit_after_validation': False, 'missing_direct_adjustment': 0.0,
        'probability_rule': 'NORMALIZE_BASELINE_TIMES_EXP_BETA_SHRUNK_SPEED',
        'all_unsupported_rule': 'EXACT_BASELINE_COPY',
        'primary_metric': 'MEAN_RACE_LOGLOSS_NEGATIVE_SUM_TARGET_LOG_PROBABILITY',
        'secondary_metric': 'MEAN_RACE_MULTICLASS_BRIER_SUM_SQUARED_ERROR',
        'paired_change_direction': 'CANDIDATE_MINUS_COMPARATOR_NEGATIVE_IS_BETTER',
        'uncertainty': {'unit': 'RACING_DATE_CLUSTER', 'minimum_dates': MIN_BOOTSTRAP_DATES,
            'draws': BOOTSTRAP_DRAWS, 'seed': BOOTSTRAP_SEED, 'interval': 'PERCENTILE_95',
            'fewer_dates': 'NOT_ESTIMABLE_NO_RACE_INDEPENDENCE_SUBSTITUTE'},
        'online_rule': 'FIXED_PAST_ONLY_UNLABELLED_BENCHMARK_UPDATES_BEFORE_EACH_TARGET_CUTOFF',
        'prior_exposure': 'PREVIOUSLY_INSPECTED_DEVELOPMENT_NOT_FRESH_HOLDOUT',
        'confirmatory': False, 'promotion': False, 'new_provider_requests': 0}


def normalized_market(odds):
    """Reproduce future_comparison's inverse-odds normalization."""
    _require(isinstance(odds, list) and len(odds) >= 2 and all(
        type(v) in (int, float) and math.isfinite(v) and v > 1 for v in odds),
        'MARKET_ODDS_INVALID')
    inverse = [1.0 / v for v in odds]
    total = math.fsum(inverse)
    return [v / total for v in inverse]


def _simplex(values, size, *, outcome=False):
    _require(isinstance(values, list) and len(values) == size and all(
        type(v) in (int, float) and math.isfinite(v)
        and (0 <= v <= 1 if outcome else 0 < v < 1) for v in values)
        and abs(math.fsum(values) - 1) <= 1e-12, 'OUTCOME_INVALID' if outcome else 'PROBABILITY_INVALID')
    if outcome:
        positive = [v for v in values if v > 0]
        _require(all(abs(v - 1.0 / len(positive)) <= 1e-12 for v in positive), 'OUTCOME_INVALID')


def adjusted_probabilities(baseline, estimates, supported, beta):
    """Unsupported runners have zero direct adjustment, but share normalization."""
    _require(isinstance(baseline, list) and isinstance(estimates, list)
             and isinstance(supported, list), 'SPEED_FEATURE_INVALID')
    _simplex(baseline, len(baseline))
    _require(len(estimates) == len(supported) == len(baseline)
        and all(type(v) is bool for v in supported)
        and all(type(v) in (int, float) and math.isfinite(v) and -3 <= v <= 3 for v in estimates)
        and all(ok or v == 0 for ok, v in zip(supported, estimates)), 'SPEED_FEATURE_INVALID')
    _require(type(beta) in (int, float) and beta in BETAS, 'COEFFICIENT_NOT_PRESPECIFIED')
    if beta == 0 or not any(supported):
        return list(baseline)
    adjustments = [beta * value if ok else 0.0 for value, ok in zip(estimates, supported)]
    if len(set(adjustments)) == 1:
        return list(baseline)
    maximum = max(adjustments)
    weights = [p * math.exp(a - maximum) for p, a in zip(baseline, adjustments)]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]
    _simplex(probabilities, len(baseline))
    return probabilities


def _losses(probabilities, outcome):
    return {'log_loss': -math.fsum(y * math.log(p) for p, y in zip(probabilities, outcome)),
            'brier': math.fsum((p - y) ** 2 for p, y in zip(probabilities, outcome))}


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _validate(records, protocol):
    _require(isinstance(protocol, dict), 'PROTOCOL_INVALID')
    try:
        expected = frozen_protocol(protocol['members'],
            **{key + '_reference': value for key, value in protocol['references'].items()},
            split_dates=protocol['splits'],
            baseline_reproduction_tolerance=protocol['baseline_reproduction_max_absolute_tolerance'])
    except (KeyError, TypeError):
        raise EvaluationRejected('PROTOCOL_INVALID') from None
    _require(protocol == expected, 'PROTOCOL_CHANGED')
    _require(isinstance(records, list) and len(records) == len(protocol['members']), 'DENOMINATOR_CHANGED')
    members = {m['race_id']: m['race_date'] for m in protocol['members']}
    _require(all(isinstance(r, dict) for r in records), 'RECORD_INVALID')
    _require(len({r.get('race_id') for r in records}) == len(records)
        and {r.get('race_id') for r in records} == set(members), 'DENOMINATOR_CHANGED')
    fields = {'race_id', 'race_date', 'cutoff', 'runner_ids', 'market_odds',
        'stored_market_probabilities', 'stored_baseline_probabilities',
        'reproduced_baseline_probabilities', 'speed_estimates', 'speed_supported',
        'label_status', 'outcome', 'source_bindings'}
    for row in records:
        _require(set(row) == fields and row['race_date'] == members[row['race_id']], 'RECORD_INVALID')
        try:
            cutoff = datetime.fromisoformat(row['cutoff'].replace('Z', '+00:00'))
            _require(cutoff.utcoffset() is not None and date.fromisoformat(row['race_date']) <= cutoff.date(),
                'CUTOFF_INVALID')
        except (ValueError, TypeError, AttributeError):
            raise EvaluationRejected('CUTOFF_INVALID') from None
        ids = row['runner_ids']
        _require(isinstance(ids, list) and len(ids) >= 2 and all(isinstance(v, str) and v for v in ids)
            and len(set(ids)) == len(ids), 'RUNNER_IDENTITY_INVALID')
        for key in ('stored_market_probabilities', 'stored_baseline_probabilities', 'reproduced_baseline_probabilities'):
            _simplex(row[key], len(ids))
        # Root authenticates the replay source; any floating reduction tolerance
        # is fixed in the protocol before labels and never exceeds 1e-12.
        _require(max(abs(a - b) for a, b in zip(row['stored_baseline_probabilities'],
            row['reproduced_baseline_probabilities'])) <= protocol['baseline_reproduction_max_absolute_tolerance'],
            'BASELINE_REPRODUCTION_FAILED')
        market = normalized_market(row['market_odds'])
        _require(len(market) == len(ids) and all(abs(a - b) <= 1e-12 for a, b in
            zip(market, row['stored_market_probabilities'])), 'MARKET_REPRODUCTION_FAILED')
        adjusted_probabilities(row['stored_baseline_probabilities'], row['speed_estimates'], row['speed_supported'], 0)
        bindings = row['source_bindings']
        _require(isinstance(bindings, dict) and set(bindings) == {'forecast', 'reproduction', 'features', 'label'},
            'SOURCE_BINDINGS_INVALID')
        for key in ('forecast', 'reproduction', 'features'):
            _reference(bindings[key])
        _require(isinstance(row['label_status'], str) and bool(row['label_status']), 'LABEL_STATUS_INVALID')
        if row['label_status'] in ELIGIBLE:
            _reference(bindings['label'])
            _simplex(row['outcome'], len(ids), outcome=True)
        else:
            _require(row['outcome'] is None and bindings['label'] is None, 'EXCLUDED_LABEL_MUST_BE_UNREAD')


def _uncertainty(rows, comparator):
    groups = {}
    for row in rows:
        groups.setdefault(row['race_date'], []).append(row)
    result = {'unit': 'RACING_DATE_CLUSTER', 'date_count': len(groups), 'race_count': len(rows),
        'status': 'INSUFFICIENT_DATE_CLUSTERS', 'log_loss_interval_95': None, 'brier_interval_95': None}
    if len(groups) < MIN_BOOTSTRAP_DATES:
        return result
    rng = random.Random(BOOTSTRAP_SEED)
    days = sorted(groups)
    draws = {'log_loss': [], 'brier': []}
    for _ in range(BOOTSTRAP_DRAWS):
        sample = [row for _ in days for row in groups[rng.choice(days)]]
        for metric in draws:
            draws[metric].append(_mean([r['losses']['baseline_speed'][metric] - r['losses'][comparator][metric]
                                        for r in sample]))
    for metric, values in draws.items():
        values.sort()
        result[metric + '_interval_95'] = [values[int(0.025 * (len(values) - 1))],
                                          values[int(0.975 * (len(values) - 1))]]
    result.update(status='DESCRIPTIVE_CLUSTER_BOOTSTRAP', draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)
    return result


def _report(rows):
    metrics = {method: {metric: _mean([r['losses'][method][metric] for r in rows])
        for metric in ('log_loss', 'brier')} for method in METHODS}
    changes = {}
    for comparator in ('baseline', 'market'):
        changes['baseline_speed_minus_' + comparator] = {
            metric: _mean([r['losses']['baseline_speed'][metric] - r['losses'][comparator][metric] for r in rows])
            for metric in ('log_loss', 'brier')}
    changes['baseline_minus_market'] = {metric: _mean([
        r['losses']['baseline'][metric] - r['losses']['market'][metric] for r in rows])
        for metric in ('log_loss', 'brier')}
    return {'race_count': len(rows), 'date_count': len({r['race_date'] for r in rows}),
        'metrics': metrics, 'paired_changes': changes,
        'uncertainty': {key: _uncertainty(rows, key) for key in ('baseline', 'market')}}


def evaluate_experiment(records, protocol, *, progress_sink=None):
    """Fit one finite grid, validate once, evaluate once; retain every member.

No supported-only selection enters fitting or the principal comparison. A
quarantined label must be absent, not merely ignored after loading its value.
"""
    # The orchestrator supplies a durable private sink. Start events precede
    # each consumed fit/prediction; completed events preserve their values even
    # when a later trial, report or independent oracle fails.
    def emit(kind, **payload):
        if progress_sink is not None:
            progress_sink({'kind': kind, **payload})

    emit('INPUT_VALIDATION_STARTED')
    _validate(records, protocol)
    ordered = sorted(records, key=lambda r: (r['race_date'], r['cutoff'], r['race_id']))
    eligible = [r for r in ordered if r['label_status'] in ELIGIBLE]
    splits = protocol['splits']
    partitions = {role: [r for r in eligible if r['race_date'] in days] for role, days in splits.items()}
    accounting = {'population_races': len(records), 'eligible_races': len(eligible),
        'excluded_races': len(records) - len(eligible),
        'label_status_counts': dict(Counter(r['label_status'] for r in records)),
        'split_counts': {role: len(rows) for role, rows in partitions.items()},
        'runner_appearances': sum(len(r['runner_ids']) for r in records),
        'supported_runner_appearances': sum(sum(r['speed_supported']) for r in records),
        'eligible_races_with_speed': sum(any(r['speed_supported']) for r in eligible),
        'eligible_races_with_exact_baseline_fallback': sum(not any(r['speed_supported']) for r in eligible),
        'supplied_baseline_vectors_checked_races': len(records), 'market_reproduced_races': len(records),
        'maximum_supplied_baseline_vector_absolute_error': max(abs(a - b) for row in records
            for a, b in zip(row['stored_baseline_probabilities'], row['reproduced_baseline_probabilities']))}
    emit('INPUT_VALIDATED', accounting=accounting,
         eligible_race_ids=[r['race_id'] for r in eligible])
    base = {'schema_version': 'sectional_speed_evaluation_v1', 'policy': POLICY,
        'protocol_sha256': _hash(protocol), 'input_packet_sha256': _hash(records),
        'accounting': accounting, 'recommendation': 'RETAIN_AS_EXPLORATORY',
        'confirmatory': False, 'model_promotion': False,
        'caveats': ['PREVIOUSLY_EXPOSED_RETROSPECTIVE_POPULATION',
            'REPEATED_RUNNERS_AND_HISTORIES_ARE_NOT_INDEPENDENT_REPLICATIONS',
            'CROSS_CONTEXT_TRANSFER_IS_AN_EXPLORATORY_ASSUMPTION',
            'UNSUPPORTED_DIRECT_ADJUSTMENT_ZERO_CAN_STILL_CHANGE_NORMALIZED_PROBABILITY']}
    evaluation_days = len({r['race_date'] for r in partitions['evaluation']})
    if evaluation_days < 10:
        base['caveats'].append('FEWER_THAN_TEN_EVALUATION_DATES_LIMIT_CLUSTER_UNCERTAINTY')
    if not all(partitions.values()):
        emit('EVALUATION_BLOCKED', reason='EMPTY_CHRONOLOGICAL_SPLIT', accounting=accounting)
        return {**base, 'status': 'BLOCKED_EMPTY_CHRONOLOGICAL_SPLIT', 'fit_trials': [],
            'records': [{'race_id': r['race_id'], 'race_date': r['race_date'], 'label_status': r['label_status']}
                        for r in ordered], 'selected_beta': None}
    def score(rows, beta):
        return _mean([_losses(adjusted_probabilities(r['stored_baseline_probabilities'], r['speed_estimates'],
            r['speed_supported'], beta), r['outcome'])['log_loss'] for r in rows])
    trials = []
    for beta in BETAS:
        training_ids = [r['race_id'] for r in partitions['training']]
        emit('FIT_TRIAL_STARTED', beta=beta, training_race_ids=training_ids)
        trial = {'beta': beta, 'training_log_loss': score(partitions['training'], beta),
                 'training_race_ids': training_ids}
        emit('FIT_TRIAL_COMPLETED', trial=trial)
        trials.append(trial)
    trained = min(trials, key=lambda t: (t['training_log_loss'], t['beta']))['beta']
    emit('TRAINING_SELECTION_COMPLETED', trained_beta=trained)
    validation_ids = [r['race_id'] for r in partitions['validation']]
    emit('VALIDATION_STARTED', trained_beta=trained, validation_race_ids=validation_ids)
    validation_baseline = score(partitions['validation'], 0.0)
    emit('VALIDATION_BASELINE_SCORED', beta=0.0, log_loss=validation_baseline)
    validation_candidate = score(partitions['validation'], trained)
    emit('VALIDATION_CANDIDATE_SCORED', beta=trained, log_loss=validation_candidate)
    accepted = trained != 0 and validation_candidate < validation_baseline
    selected = trained if accepted else 0.0
    validation = {'baseline_log_loss': validation_baseline, 'trained_candidate_log_loss': validation_candidate,
        'accepted_trained_candidate': accepted, 'validation_race_ids': validation_ids}
    emit('VALIDATION_COMPLETED', selected_beta=selected, validation=validation)
    predictions = []
    for row in ordered:
        emit('PREDICTION_STARTED', race_id=row['race_id'], coefficient=selected)
        probabilities = {'market': normalized_market(row['market_odds']),
            'baseline': list(row['stored_baseline_probabilities']),
            'baseline_speed': adjusted_probabilities(row['stored_baseline_probabilities'],
                row['speed_estimates'], row['speed_supported'], selected)}
        prediction = {**row, 'split': next(role for role, days in splits.items() if row['race_date'] in days),
            'probabilities': probabilities, 'coefficient': selected,
            'direct_logit_adjustments': [selected * v if supported else 0.0
                for v, supported in zip(row['speed_estimates'], row['speed_supported'])],
            'losses': {method: _losses(p, row['outcome']) for method, p in probabilities.items()}
                      if row['label_status'] in ELIGIBLE else None}
        emit('PREDICTION_COMPLETED', record=prediction)
        predictions.append(prediction)
    scored = [r for r in predictions if r['losses'] is not None]
    evaluated = [r for r in scored if r['split'] == 'evaluation']
    emit('REPORTS_STARTED', predicted_race_ids=[r['race_id'] for r in predictions])
    result = {**base, 'status': 'COMPLETE_RETROSPECTIVE_EXPLORATORY_EVALUATION',
        'fit_trials': trials, 'trained_beta': trained, 'selected_beta': selected,
        'validation': validation,
        'records': predictions, 'principal_evaluation': _report(evaluated),
        'supported_only_secondary_evaluation': {**_report([r for r in evaluated if any(r['speed_supported'])]),
            'selection_limitation': 'NONRANDOM_HISTORY_COVERAGE_NOT_PRINCIPAL_POPULATION'},
        'date_reports': {day: {'role': next(role for role, ds in splits.items() if day in ds),
            **_report([r for r in scored if r['race_date'] == day])} for day in sorted({r['race_date'] for r in ordered})},
        'pooled_descriptive_only': _report(scored),
        'evaluation_leave_one_date_out_fixed_beta': {day: _report([
            r for r in evaluated if r['race_date'] != day])
            for day in sorted({r['race_date'] for r in evaluated})}}
    emit('REPORTS_COMPLETED', result_sha256=_hash(result))
    return result
