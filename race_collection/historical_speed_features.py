"""Pure first-sectional summaries under explicit user-supplied semantics.

The caller authenticates the retained card, race and exact roster. This module
does not acquire evidence, certify provider meanings, fit models or authorize use.
"""
from collections import Counter, defaultdict
from datetime import date, datetime
import math
import re
from statistics import median


class FeatureRejected(ValueError):
    """Invalid shared input, reported without source values."""


def _require(condition, reason):
    if not condition:
        raise FeatureRejected(reason)


def _shape(value, required, optional=()):
    _require(isinstance(value, dict) and set(required) <= set(value)
             and set(value) <= set(required) | set(optional), 'SCHEMA_INVALID')


def _instant(value):
    _require(isinstance(value, str), 'TIMESTAMP_INVALID')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise FeatureRejected('TIMESTAMP_INVALID') from None
    _require(result.utcoffset() is not None, 'TIMESTAMP_INVALID')
    return result


def _validate(packet):
    _shape(packet, {'target', 'captured_at', 'roster', 'histories'})
    target = packet['target']
    _shape(target, {'race_id', 'date', 'source_track', 'distance_m', 'cutoff'},
           {'layout_id', 'layout_era'})
    _require(all(isinstance(target[key], str) and target[key].strip()
                 for key in ('race_id', 'source_track')) and _date(target['date']) is not None
             and type(target['distance_m']) is int and target['distance_m'] > 0, 'TARGET_CONTEXT_INVALID')
    roster, histories = packet['roster'], packet['histories']
    _require(isinstance(roster, list) and bool(roster)
             and all(isinstance(runner, str) and runner.strip() == runner and bool(runner)
                     for runner in roster), 'ROSTER_INVALID')
    _require(len(roster) == len(set(roster)) and isinstance(histories, dict)
             and set(histories) <= set(roster), 'ROSTER_INVALID')
    for rows in histories.values():
        _require(isinstance(rows, list), 'SCHEMA_INVALID')
        for row in rows:
            _shape(row, {'date', 'source_track', 'distance_m', 'first_sectional'},
                   {'layout_id', 'layout_era', 'observation_fingerprint'})
            if 'observation_fingerprint' in row:
                _require(isinstance(row['observation_fingerprint'], str)
                         and re.fullmatch('[a-f0-9]{64}', row['observation_fingerprint']) is not None,
                         'SCHEMA_INVALID')
    for item in [target, *(row for rows in histories.values() for row in rows)]:
        _require(all(item.get(key) is None or isinstance(item[key], str)
                     for key in ('layout_id', 'layout_era')), 'SCHEMA_INVALID')
    captured, cutoff = _instant(packet['captured_at']), _instant(target['cutoff'])
    _require(captured < cutoff, 'CAPTURE_NOT_BEFORE_CUTOFF')
    _require(_date(target['date']) <= cutoff.date(), 'TARGET_DATE_AFTER_CUTOFF')
    return {**target, 'source_track': target['source_track'].strip()}, captured.astimezone(cutoff.tzinfo).date()


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value.strip()):
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        return None


def _sectional(value):
    if value is None or (isinstance(value, str) and value.strip().upper() in
                        {'', '-', '--', '—', '–', 'N/A', 'NA', 'NULL', 'NONE'}):
        return None, 'MISSING_FIRST_SECTIONAL'
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None, 'INVALID_FIRST_SECTIONAL'
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None, 'INVALID_FIRST_SECTIONAL'
    return ((number, None) if math.isfinite(number) and number > 0
            else (None, 'INVALID_FIRST_SECTIONAL'))


def _layout(value):
    return value.strip() if isinstance(value, str) and value.strip().upper() not in {
        '', 'UNKNOWN', 'UNQUALIFIED', 'N/A', '-'} else None


def _layout_conflict(target, selected):
    return any(len({value for item in [target, *selected]
                    if (value := _layout(item.get(key))) is not None}) > 1
               for key in ('layout_id', 'layout_era'))


def _field_median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    low, high = ordered[middle - 1], ordered[middle]
    return low + (high - low) / 2


def _runner(runner_id, rows, target, target_date, capture_date):
    exclusions = Counter()
    candidates = []
    groups = defaultdict(list)
    for row in rows:
        prior = _date(row['date'])
        if prior is None:
            exclusions['INVALID_HISTORY_DATE'] += 1
            continue
        if prior >= target_date:
            exclusions['NOT_PRIOR_DATE'] += 1
            continue
        if prior > capture_date:
            exclusions['HISTORY_AFTER_CAPTURE_DATE'] += 1
            continue
        groups[prior].append(row)
    for prior, group in groups.items():
        signatures = {
            (type(row['source_track']).__name__, str(row['source_track']).strip(),
             type(row['distance_m']).__name__, str(row['distance_m']),
             *_sectional(row['first_sectional']),
             _layout(row.get('layout_id')), _layout(row.get('layout_era')),
             row.get('observation_fingerprint'))
            for row in group
        }
        if len(signatures) != 1:
            exclusions['CONFLICTING_PRIOR_DATE'] += len(group)
            continue
        if len(group) > 1:
            exclusions['DUPLICATE_OBSERVATION'] += len(group) - 1
        row = group[0]
        if (not isinstance(row['source_track'], str)
                or type(row['distance_m']) is not int or row['distance_m'] <= 0):
            exclusions['INVALID_HISTORY_CONTEXT'] += 1
            continue
        if (row['source_track'].strip() != target['source_track']
                or row['distance_m'] != target['distance_m']):
            exclusions['CONTEXT_MISMATCH'] += 1
            continue
        if _layout_conflict(target, [row]):
            exclusions['KNOWN_LAYOUT_MISMATCH'] += 1
            continue
        value, reason = _sectional(row['first_sectional'])
        if reason:
            exclusions[reason] += 1
            continue
        candidates.append((prior, value, row))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:3] if len(candidates) >= 3 else []
    layout_conflict = _layout_conflict(target, [item[2] for item in selected])
    if layout_conflict:
        exclusions['KNOWN_LAYOUT_CONFLICT'] += len(selected)
        selected = []
    center = median(item[1] for item in selected) if selected else None
    result = {
        'runner_id': runner_id,
        'status': ('KNOWN_LAYOUT_CONFLICT' if layout_conflict else 'SUPPORTED' if selected
                   else 'INSUFFICIENT_COMPARABLE_HISTORY' if rows else 'NO_RETAINED_HISTORY'),
        'usable_prior_dates': len(candidates),
        'selected_dates': [item[0].isoformat() for item in selected],
        'selected_ages_days': [(target_date - item[0]).days for item in selected],
        'median_first_sectional': center,
        'mad_first_sectional': median(abs(item[1] - center) for item in selected) if selected else None,
        'field_median_gap': None, 'exclusions': dict(exclusions),
    }
    return result, [item[2] for item in selected]


def build_speed_features(packet):
    """Construct private summaries for every roster runner; never impute history."""
    target, capture_date = _validate(packet)
    target_date = _date(target['date'])
    built = [_runner(runner_id, packet['histories'].get(runner_id, []), target, target_date, capture_date)
             for runner_id in packet['roster']]
    runners = [item[0] for item in built]
    selected = [row for _, rows in built for row in rows]
    supported = sum(runner['status'] == 'SUPPORTED' for runner in runners)
    complete = supported == len(runners)
    field_blocker = None if complete else 'INCOMPLETE_ROSTER_SUPPORT'
    if complete and _layout_conflict(target, selected):
        complete, field_blocker = False, 'KNOWN_LAYOUT_CONFLICT'
    field_center = _field_median(runner['median_first_sectional'] for runner in runners) if complete else None
    if complete:
        for runner in runners:
            runner['field_median_gap'] = runner['median_first_sectional'] - field_center
    return {
        'schema_version': 'historical_first_sectional_v1',
        'status': 'FULL_FIELD_SUPPORTED' if complete else 'INCOMPLETE_FIELD',
        'race_id': target['race_id'], 'runner_count': len(runners),
        'supported_runner_count': supported, 'full_field_supported': complete,
        'field_median_first_sectional': field_center,
        'field_blocker': field_blocker,
        'assumptions': ['USER_SUPPLIED_RUNNER_FIRST_SECTIONAL_MEANING',
                        'SOURCE_TIME_UNIT_ASSUMED_SECONDS', 'EXACT_RAW_TRACK_DISTANCE_CONTEXT',
                        'UNKNOWN_LAYOUT_OR_ERA_ASSUMED_STABLE'],
        'runners': runners,
    }
