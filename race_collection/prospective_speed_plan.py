"""Pure prospective admission and one-look analysis for the frozen candidate.

The caller authenticates references, preserves immutable population/evaluation
claims, and admits result bytes. This module cannot acquire or authorise data.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import re
from statistics import stdev
from zoneinfo import ZoneInfo

from race_collection.development_examples import selected_population
from race_collection.sectional_speed_evaluation import adjusted_probabilities

ZONE = ZoneInfo('Australia/Melbourne')
DATES = ['2026-10-10', '2026-10-11']
CANDIDATE_COMMIT = 'af55ae9322d08d0f255f767a14e385c794205dfe'
ALLOCATION_SHA256 = 'e877d79118f7bf60832c8b1ee18c6ba0e72d5de1fdf2d551d18f68500cec7c8e'
METHODS = ('market', 'baseline', 'baseline_speed')
VERIFIED_LABELS = {'FULL_ORDER_WIN_ELIGIBLE', 'KNOWN_NONFINISH_WIN_ELIGIBLE'}


class PlanRejected(ValueError):
    """Safe finite rejection reason; no private record content."""


def _require(condition, reason):
    if not condition:
        raise PlanRejected(reason)


def _stamp(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        _require(result.utcoffset() is not None, 'TIMESTAMP_INVALID')
        return result
    except (ValueError, AttributeError, TypeError):
        raise PlanRejected('TIMESTAMP_INVALID') from None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _reference(value):
    _require(isinstance(value, dict) and set(value) == {'path', 'sha256'}
        and isinstance(value['path'], str) and value['path'].startswith('/')
        and isinstance(value['sha256'], str)
        and re.fullmatch('[a-f0-9]{64}', value['sha256']) is not None, 'REFERENCE_INVALID')


def precision_diagnostic(date_changes, effect=0.0023720881499839233):
    """Planning approximation from already exposed aggregate dates, not power proof.

Equal-date normal approximations cannot promise future race-weighted precision.
Six legacy dates and changed feature coverage make the estimate especially weak.
"""
    _require(isinstance(date_changes, list) and len(date_changes) >= 2 and all(
        type(v) in (int, float) and math.isfinite(v) for v in date_changes)
        and type(effect) in (int, float) and math.isfinite(effect) and effect > 0,
        'PRECISION_INPUT_INVALID')
    spread = stdev(date_changes)
    return {'source_date_count': len(date_changes), 'date_mean_change_sd': spread,
        'reference_absolute_effect': effect,
        'approximate_dates_for_95_percent_half_width_equal_effect':
            math.ceil((1.96 * spread / effect) ** 2),
        'approximate_dates_for_80_percent_power_two_sided_5_percent':
            math.ceil(((1.96 + 0.8416212335729143) * spread / effect) ** 2),
        'two_date_normal_half_width_for_scale_only': 1.96 * spread / math.sqrt(2),
        'available_future_dates': 2,
        'supports_precise_test_of_exploratory_effect': False,
        'limitations': ['SIX_PREVIOUSLY_INSPECTED_DATES', 'DATE_DEPENDENCE_NOT_KNOWN',
            'EQUAL_DATE_APPROXIMATION_NOT_RACE_WEIGHTED_POWER',
            'COVERAGE_AND_POPULATION_CAN_CHANGE_VARIANCE',
            'NORMAL_APPROXIMATION_UNRELIABLE_AT_TWO_DATES']}


def build_plan(*, frozen_at, candidate_reference, authority_references, precision):
    """Fix all rules before the first eligible collection; does not grant authority."""
    _require(_stamp(frozen_at) < _stamp('2026-10-10T12:50:00+11:00'), 'PLAN_TOO_LATE')
    _reference(candidate_reference)
    _require(isinstance(authority_references, dict) and set(authority_references) == {
        'allocation', 'historical_amendment', 'exclusive_amendment', 'current_user',
        'result_runtime', 'reservation_registry'}, 'AUTHORITY_REFERENCES_INVALID')
    for reference in authority_references.values():
        _reference(reference)
    _require(authority_references['allocation']['sha256'] == ALLOCATION_SHA256,
             'ALLOCATION_CHANGED')
    _require(isinstance(precision, dict) and
        precision.get('supports_precise_test_of_exploratory_effect') is False,
        'PRECISION_LIMITATION_MISSING')
    return {'schema_version': 'prospective_sectional_plan_v1',
        'frozen_at': frozen_at, 'candidate_commit': CANDIDATE_COMMIT,
        'candidate': candidate_reference, 'authority': authority_references,
        'dates': list(DATES), 'timezone': 'Australia/Melbourne',
        'allocation_id': 'development-single-snapshot-20261003-v1',
        'population': {'freeze_local_time': '12:50', 'max_index_age_seconds': 300,
            'selection_policy': 'first_six_1310_1420_melbourne_before_WIN_qualification_v1',
            'maximum_per_date': 6, 'maximum_total': 12,
            'no_replacement_after_failure': True, 'protect_existing_study_members': True},
        'horizon_basis': 'ALL_REMAINING_DATES_AND_CAPACITY_OF_EXISTING_DEVELOPMENT_ALLOCATION',
        'purpose': 'FRESH_PROSPECTIVE_FEASIBILITY_COVERAGE_AND_DESCRIPTIVE_PREDICTION',
        'precision': precision, 'beta': 0.1, 'fit_or_tune': False,
        'methods': list(METHODS),
        'baseline': 'EXACT_SAVED_PERIOD1_BASE16_DEVELOPMENT_MODEL_NOT_PRODUCTION',
        'cutoff_rule': 'ONE_EXISTING_QUALIFIED_WIN_SNAPSHOT_SAME_ROSTER_AND_INFORMATION_CUTOFF',
        'history_rule': 'FIXED_ALGORITHM_AS_OF_CUTOFF_NO_LATER_CAPTURE_OR_CORRECTION',
        'online_updates': 'EARLIER_AVAILABLE_RETAINED_HISTORY_WITH_ORIGINAL_SOURCE_TIMESTAMPS',
        'missing_history': 'ZERO_DIRECT_ADJUSTMENT_FULL_FIELD_NORMALIZATION',
        'all_unsupported': 'EXACT_BASELINE_COPY',
        'result_requests_stop_at': '2026-10-25T12:00:00+11:00',
        'evaluation_at': '2026-10-25T12:05:00+11:00',
        'evaluation_looks': 1, 'interim_comparative_performance': False,
        'outcome_admission': sorted(VERIFIED_LABELS),
        'dead_heat': 'EQUAL_MASS_OVER_OFFICIALLY_VERIFIED_JOINT_WINNERS_NO_INFERRED_TIES',
        'primary': 'PAIRED_MEAN_RACE_LOG_LOSS_SPEED_MINUS_BASELINE',
        'secondary': 'PAIRED_MEAN_RACE_MULTICLASS_BRIER_SUM',
        'uncertainty': 'DATE_LEVEL_DESCRIPTIVE_ONLY_NO_INTERVAL_WITH_FEWER_THAN_FIVE_DATES',
        'influence': 'LEAVE_ONE_DATE_OUT_FIXED_CANDIDATE_NO_REFIT',
        'supported_only': 'SECONDARY_NONRANDOM_COVERAGE_SUBSET',
        'missing_result': 'NO_IMPUTATION_REPORT_COUNT_AND_REASON_COMPLETE_CASE_COMPARISON',
        'forecast_failure': 'RETAIN_IN_FULL_POPULATION_NO_RETROSPECTIVE_REPLACEMENT',
        'source_requests_added': 0, 'result_requests_added': 0,
        'scientific_membership_changes': False, 'production_changes': False,
        'extend_based_on_results': False, 'model_promotion': False}


def _validate_plan(plan):
    try:
        expected = build_plan(frozen_at=plan['frozen_at'], candidate_reference=plan['candidate'],
                              authority_references=plan['authority'], precision=plan['precision'])
    except (KeyError, TypeError):
        raise PlanRejected('PLAN_INVALID') from None
    _require(plan == expected, 'PLAN_CHANGED')


def select_population(plan, races, *, local_date, frozen_at, source_observed_at,
                      index_complete, protected_membership_reference, protected_race_ids=()):
    """Apply the existing first-six rule before qualification, keeping every row."""
    _validate_plan(plan)
    # The caller authenticates this immutable identity-only reservation snapshot.
    _reference(protected_membership_reference)
    now = _stamp(frozen_at).astimezone(ZONE)
    observed = _stamp(source_observed_at)
    _require(local_date in DATES and now.date().isoformat() == local_date
        and now.strftime('%H:%M') == '12:50', 'FREEZE_NOT_DUE')
    _require(_stamp(plan['frozen_at']) < now, 'PLAN_NOT_FROZEN_BEFORE_COLLECTION')
    _require(index_complete is True and 0 <= (now - observed).total_seconds() <= 300,
             'INDEX_NOT_COMPLETE_AND_FRESH')
    _require(isinstance(races, list) and all(isinstance(r, dict) for r in races),
             'POPULATION_INVALID')
    _require(len({r['race_id'] for r in races}) == len(races), 'POPULATION_DUPLICATE')
    intended, selected = selected_population(races, local_date)
    protected = set(protected_race_ids)
    # Already admitted scientific members cannot be reassigned by a later freeze.
    admitted = [race_id for race_id in selected if race_id not in protected]
    intended_ids = {r['race_id'] for r in intended}
    dispositions = []
    for row in races:
        race_id = row['race_id']
        reason = ('PROTECTED_STUDY_MEMBER' if race_id in selected and race_id in protected
            else 'SELECTED' if race_id in admitted
            else 'BEYOND_FIRST_SIX' if race_id in intended_ids else 'OUTSIDE_SELECTION_WINDOW')
        dispositions.append({'race_id': race_id, 'jump_at': row['jump_at'], 'disposition': reason})
    return {'schema_version': 'prospective_sectional_population_v1',
        'plan_sha256': _digest(plan), 'local_date': local_date, 'frozen_at': frozen_at,
        'source_observed_at': source_observed_at, 'index_complete': True,
        'observed_races_sha256': _digest(races), 'observed_races': races,
        'first_six_race_ids': selected, 'selected_race_ids': admitted,
        'protected_membership_reference': protected_membership_reference,
        'protected_race_ids': sorted(protected), 'dispositions': dispositions}


def _validate_population(plan, population):
    try:
        expected = select_population(plan, population['observed_races'],
            local_date=population['local_date'], frozen_at=population['frozen_at'],
            source_observed_at=population['source_observed_at'],
            index_complete=population['index_complete'],
            protected_membership_reference=population['protected_membership_reference'],
            protected_race_ids=population['protected_race_ids'])
    except (KeyError, TypeError):
        raise PlanRejected('POPULATION_INVALID') from None
    _require(population == expected, 'POPULATION_CHANGED')


def forecast_admission(plan, population, race_id, *, cutoff, sealed_at,
                       input_available_at, prior_attempt=False):
    """Metadata gate; source bindings and durable first-attempt claim are caller-owned."""
    _validate_population(plan, population)
    _require(race_id in population['selected_race_ids'], 'RACE_NOT_ALLOCATED')
    _require(prior_attempt is False, 'ATTEMPT_ALREADY_CONSUMED')
    row = next(r for r in population['observed_races'] if r['race_id'] == race_id)
    jump = _stamp(row['jump_at'])
    decision, seal, available = map(_stamp, (cutoff, sealed_at, input_available_at))
    _require(_stamp(population['frozen_at']) <= available <= decision <= seal < jump,
             'FORECAST_NOT_PREJUMP_OR_INPUT_AFTER_CUTOFF')
    lead = (jump - decision).total_seconds()
    _require(120 <= lead <= 600, 'PRICE_SNAPSHOT_OUTSIDE_FROZEN_2_TO_10_MINUTES')
    return {'status': 'ELIGIBLE', 'race_id': race_id, 'race_date': population['local_date'],
        'jump_at': row['jump_at'], 'cutoff': cutoff, 'sealed_at': sealed_at,
        'plan_sha256': _digest(plan), 'population_sha256': _digest(population)}


def evaluation_gate(plan, *, now, collection_terminal, closure_terminal,
                    prior_evaluation_claim=False):
    _validate_plan(plan)
    if prior_evaluation_claim:
        return 'EVALUATION_ALREADY_CLAIMED'
    if _stamp(now) < _stamp(plan['evaluation_at']):
        return 'WAIT_FOR_FIXED_EVALUATION_TIME'
    if collection_terminal is not True:
        return 'WAIT_FOR_COLLECTION_ACCOUNTING'
    if closure_terminal is not True:
        return 'WAIT_FOR_TERMINAL_RESULT_DISPOSITIONS'
    return 'EVALUATION_DUE'


def _simplex(values, count, *, outcome=False):
    _require(isinstance(values, list) and len(values) == count and count >= 2
        and all(type(v) in (int, float) and math.isfinite(v)
            and (0 <= v <= 1 if outcome else 0 < v < 1) for v in values)
        and abs(math.fsum(values) - 1) <= 1e-12, 'OUTCOME_INVALID' if outcome else 'PROBABILITY_INVALID')
    if outcome:
        positive = [v for v in values if v]
        _require(all(abs(v - 1 / len(positive)) <= 1e-12 for v in positive), 'OUTCOME_INVALID')


def _report(rows):
    def mean(values):
        return math.fsum(values) / len(values) if values else None
    metrics = {method: {key: mean([r['losses'][method][key] for r in rows])
        for key in ('log_loss', 'brier')} for method in METHODS}
    changes = {comparator: {key: mean([r['losses']['baseline_speed'][key]
        - r['losses'][comparator][key] for r in rows]) for key in ('log_loss', 'brier')}
        for comparator in ('baseline', 'market')}
    return {'race_count': len(rows), 'date_count': len({r['race_date'] for r in rows}),
        'metrics': metrics, 'speed_minus': changes,
        'uncertainty': {'status': 'INSUFFICIENT_DATE_CLUSTERS', 'interval_95': None,
            'reason': 'FIXED_HORIZON_HAS_AT_MOST_TWO_DATES_NO_RACE_INDEPENDENCE_SUBSTITUTE'}}


def evaluate_closed_population(plan, populations, records, *, now, date_accounting,
                               collection_terminal, closure_terminal,
                               prior_evaluation_claim=False):
    """One fixed analysis; caller durably claims once before opening admitted labels.

Rows cover every selected member, including failures. Successful rows contain
the exact sealed probability vectors; no post-jump forecast is manufactured.
Result identity verification is required before supplying an eligible outcome.
"""
    gate = evaluation_gate(plan, now=now, collection_terminal=collection_terminal,
        closure_terminal=closure_terminal, prior_evaluation_claim=prior_evaluation_claim)
    _require(gate == 'EVALUATION_DUE', gate)
    _require(isinstance(populations, list) and len(populations) <= 2, 'POPULATIONS_INVALID')
    _require(len({p['local_date'] for p in populations}) == len(populations), 'POPULATIONS_DUPLICATE')
    _require(isinstance(date_accounting, list) and len(date_accounting) == len(DATES)
        and {r['local_date'] for r in date_accounting} == set(DATES), 'DATE_ACCOUNTING_INCOMPLETE')
    populations_by_date = {p['local_date']: p for p in populations}
    failed_statuses = {'INDEX_MISSING', 'INDEX_STALE', 'INDEX_INCOMPLETE',
                       'FREEZE_INTERRUPTED', 'SOURCE_OR_AUTHORITY_UNAVAILABLE'}
    for day in date_accounting:
        if day['status'] == 'POPULATION_FROZEN':
            _require(day['local_date'] in populations_by_date and
                day.get('population_sha256') == _digest(populations_by_date[day['local_date']]),
                'DATE_POPULATION_CHANGED')
        else:
            _require(day['status'] in failed_statuses and day['local_date'] not in populations_by_date
                and isinstance(day.get('reason'), str) and bool(day['reason'])
                and day.get('population_sha256') is None, 'DATE_DISPOSITION_INVALID')
    members = {}
    for population in populations:
        _validate_population(plan, population)
        for race_id in population['selected_race_ids']:
            _require(race_id not in members, 'POPULATION_DUPLICATE')
            members[race_id] = population
    _require(isinstance(records, list) and len(records) == len(members)
        and {r['race_id'] for r in records} == set(members), 'EVALUATION_DENOMINATOR_CHANGED')
    scored, dispositions, supported, appearances = [], [], 0, 0
    for row in records:
        population = members[row['race_id']]
        _require(row['race_date'] == population['local_date'], 'RACE_DATE_CHANGED')
        disposition = {'race_id': row['race_id'], 'race_date': row['race_date'],
            'forecast_status': row['forecast_status'], 'label_status': row['label_status']}
        dispositions.append(disposition)
        if row['forecast_status'] != 'SEALED_PREJUMP':
            _require(row.get('outcome') is None and row.get('probabilities') is None,
                     'FAILED_FORECAST_VALUES_MUST_BE_ABSENT')
            continue
        forecast_admission(plan, population, row['race_id'], cutoff=row['cutoff'],
            sealed_at=row['sealed_at'], input_available_at=row['input_available_at'])
        ids = row['runner_ids']
        _require(isinstance(ids, list) and len(ids) >= 2 and len(set(ids)) == len(ids)
            and all(isinstance(v, str) and v for v in ids), 'RUNNER_IDENTITY_INVALID')
        probabilities = row['probabilities']
        _require(isinstance(probabilities, dict) and set(probabilities) == set(METHODS),
                 'METHODS_CHANGED')
        for vector in probabilities.values():
            _simplex(vector, len(ids))
        expected = adjusted_probabilities(probabilities['baseline'], row['speed_estimates'],
                                           row['speed_supported'], 0.1)
        _require(expected == probabilities['baseline_speed'], 'FROZEN_ADJUSTMENT_CHANGED')
        supported += sum(row['speed_supported'])
        appearances += len(ids)
        if row['label_status'] not in VERIFIED_LABELS:
            _require(row.get('outcome') is None, 'UNVERIFIED_TARGET_MUST_BE_ABSENT')
            continue
        _require(row.get('result_identity_verified') is True, 'RESULT_IDENTITY_NOT_VERIFIED')
        _simplex(row['outcome'], len(ids), outcome=True)
        losses = {method: {'log_loss': -math.fsum(y * math.log(p) for p, y in zip(vector, row['outcome'])),
            'brier': math.fsum((p - y) ** 2 for p, y in zip(vector, row['outcome']))}
            for method, vector in probabilities.items()}
        scored.append({'race_id': row['race_id'], 'race_date': row['race_date'],
            'has_speed': any(row['speed_supported']), 'losses': losses})
    return {'schema_version': 'prospective_sectional_evaluation_v1',
        'status': 'COMPLETE_DESCRIPTIVE' if scored else 'COMPLETE_NO_SCORABLE_RESULTS',
        'plan_sha256': _digest(plan), 'input_sha256': _digest(records), 'beta': 0.1,
        'date_accounting': date_accounting,
        'evaluation_looks': 1, 'fit_trials': [], 'promotion': False,
        'accounting': {'selected_races': len(members), 'scored_races': len(scored),
            'forecast_status_counts': dict(Counter(r['forecast_status'] for r in records)),
            'label_status_counts': dict(Counter(r['label_status'] for r in records)),
            'supported_runner_appearances': supported, 'sealed_runner_appearances': appearances,
            'scored_races_with_speed': sum(r['has_speed'] for r in scored),
            'scored_races_neutral_fallback': sum(not r['has_speed'] for r in scored)},
        'dispositions': dispositions, 'loss_records': scored,
        'principal_complete_case_comparison': _report(scored),
        'supported_only_secondary': _report([r for r in scored if r['has_speed']]),
        'date_results': {day: _report([r for r in scored if r['race_date'] == day]) for day in DATES},
        'leave_one_date_out': {day: _report([r for r in scored if r['race_date'] != day]) for day in DATES},
        'limitations': ['AT_MOST_TWO_DATES_NOT_PRECISE_CONFIRMATION_OF_SMALL_EFFECT',
            'MISSING_RESULTS_AND_FORECAST_FAILURES_NOT_IMPUTED_CAN_BIAS_COMPLETE_CASE_COMPARISON',
            'SUPPORTED_SUBSET_NONRANDOM', 'NO_RESULT_BASED_HORIZON_EXTENSION']}
