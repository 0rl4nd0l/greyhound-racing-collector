"""Pure, past-only sectional candidate under recorded source-meaning assumptions.

The adapter proves runner identities, aliases, event proxies and their availability.
This module supplies no acquisition, label access, fitting or authorization.
"""
from collections import Counter, defaultdict
from datetime import date, datetime
import hashlib
import json
import math
import re
from statistics import median


MIN_BENCHMARK_RUNNERS = 5
MAX_RUNNER_OBSERVATIONS = 5
MAD_SCALE = 1.4826
SHRINKAGE_PRIOR_COUNT = 3
ZSCORE_LIMIT = 3.0


class CandidateRejected(ValueError):
    """Shared packet invalid; reasons contain no private source values."""


def _require(condition, reason):
    if not condition:
        raise CandidateRejected(reason)


def _instant(value):
    _require(isinstance(value, str), 'TIMESTAMP_INVALID')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        raise CandidateRejected('TIMESTAMP_INVALID') from None
    _require(result.utcoffset() is not None, 'TIMESTAMP_INVALID')
    return result


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _text(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _layout(value):
    if value is None or isinstance(value, str) and value in {'', 'UNKNOWN', 'UNQUALIFIED', 'N/A', '-'}:
        return None
    _require(_text(value), 'LAYOUT_INVALID')
    return value


def _number(value):
    if value is None or isinstance(value, str) and value.strip().upper() in {
        '', '-', '--', '—', '–', 'N/A', 'NA', 'NULL', 'NONE'
    }:
        return None, 'MISSING_FIRST_SECTIONAL'
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None, 'INVALID_FIRST_SECTIONAL'
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None, 'INVALID_FIRST_SECTIONAL'
    return ((result, None) if math.isfinite(result) and result > 0
            else (None, 'INVALID_FIRST_SECTIONAL'))


def _median(values):
    """Positive source values can overflow the usual even-count midpoint."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else ordered[mid - 1] / 2 + ordered[mid] / 2


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _validate(packet):
    _require(isinstance(packet, dict) and set(packet) == {'target', 'roster', 'observations'}, 'SCHEMA_INVALID')
    target = packet['target']
    _require(isinstance(target, dict) and {'race_id', 'date', 'cutoff'} <= set(target)
             and _text(target['race_id']) and _date(target['date']) is not None, 'TARGET_INVALID')
    cutoff = _instant(target['cutoff'])
    _require(_date(target['date']) <= cutoff.date(), 'TARGET_DATE_AFTER_CUTOFF')
    roster = packet['roster']
    _require(isinstance(roster, list) and bool(roster), 'ROSTER_INVALID')
    for item in roster:
        required = {'runner_id', 'identity_id', 'identity_available_at'}
        optional = {'identity_status', 'box_number', 'strict_runner_id'}
        _require(isinstance(item, dict) and required <= set(item) <= required | optional
                 and _text(item['runner_id']), 'ROSTER_INVALID')
        _require(item['identity_id'] is None or _text(item['identity_id']), 'ROSTER_INVALID')
        _require((item['identity_id'] is None) == (item['identity_available_at'] is None), 'ROSTER_INVALID')
        if item['identity_available_at'] is not None:
            _instant(item['identity_available_at'])
    _require(len({item['runner_id'] for item in roster}) == len(roster), 'DUPLICATE_ROSTER_RUNNER')
    identities = [item['identity_id'] for item in roster if item['identity_id'] is not None]
    _require(len(set(identities)) == len(identities), 'DUPLICATE_ROSTER_IDENTITY')
    _require(isinstance(packet['observations'], list), 'SCHEMA_INVALID')
    for row in packet['observations']:
        required = {'observation_id', 'event_id', 'runner_identity_id', 'date', 'available_at',
                    'source_track', 'canonical_track', 'distance_m', 'first_sectional',
                    'observation_fingerprint', 'source_bindings'}
        optional = {'layout_id', 'layout_era', 'event_identity_kind', 'alias_evidence'}
        _require(isinstance(row, dict) and required <= set(row) <= required | optional, 'SCHEMA_INVALID')
        _require(all(_text(row[key]) for key in ('observation_id', 'event_id', 'runner_identity_id',
                    'source_track', 'canonical_track')), 'OBSERVATION_IDENTITY_INVALID')
        _require(type(row['distance_m']) is int and row['distance_m'] > 0, 'HISTORY_CONTEXT_INVALID')
        _require(isinstance(row['observation_fingerprint'], str)
                 and re.fullmatch('[a-f0-9]{64}', row['observation_fingerprint']) is not None,
                 'OBSERVATION_FINGERPRINT_INVALID')
        _require(isinstance(row['source_bindings'], list) and bool(row['source_bindings'])
                 and all(isinstance(binding, dict) and bool(binding) for binding in row['source_bindings']),
                 'SOURCE_BINDINGS_INVALID')
        if row.get('alias_evidence') is not None:
            _require(_text(row['alias_evidence']), 'ALIAS_EVIDENCE_INVALID')
        if row['canonical_track'] != row['source_track']:
            _require(_text(row.get('alias_evidence')), 'UNVERIFIED_ALIAS')
        if 'event_identity_kind' in row:
            _require(row['event_identity_kind'] in {'SOURCE_EVENT', 'RUNNER_DATE_CONTEXT_PROXY'},
                     'EVENT_IDENTITY_KIND_INVALID')
        _instant(row['available_at'])
        _layout(row.get('layout_id'))
        _layout(row.get('layout_era'))
    return target, cutoff, _date(target['date'])


def _population(rows, cutoff, target_date):
    """Filter first, then reconcile copies; future contradictory evidence stays future."""
    exclusions = Counter()
    groups = defaultdict(list)
    for row in rows:
        available = _instant(row['available_at'])
        prior = _date(row['date'])
        if available >= cutoff:
            exclusions['EVIDENCE_NOT_BEFORE_CUTOFF'] += 1
        elif prior is None:
            exclusions['INVALID_HISTORY_DATE'] += 1
        elif prior >= target_date:
            exclusions['NOT_PRIOR_DATE'] += 1
        elif prior > available.astimezone(cutoff.tzinfo).date():
            exclusions['HISTORY_AFTER_EVIDENCE_DATE'] += 1
        else:
            groups[row['runner_identity_id'], row['date']].append(row)
    event_dates = defaultdict(set)
    for copies in groups.values():
        for row in copies:
            event_dates[row['runner_identity_id'], row['event_id']].add(row['date'])
    population = []
    for (identity, prior), copies in sorted(groups.items()):
        if any(len(event_dates[identity, row['event_id']]) != 1 for row in copies):
            exclusions['CONFLICTING_EVENT_DATES'] += len(copies)
            continue
        signatures = set()
        for row in copies:
            value, reason = _number(row['first_sectional'])
            context = (row['canonical_track'], row['distance_m'], _layout(row.get('layout_id')),
                       _layout(row.get('layout_era')))
            signatures.add((row['event_id'], context, value, reason, row['observation_fingerprint']))
        if len(signatures) != 1:
            exclusions['CONFLICTING_RUNNER_DATE'] += len(copies)
            continue
        event, context, value, reason, fingerprint = next(iter(signatures))
        if reason:
            exclusions[reason] += len(copies)
            continue
        exclusions['DUPLICATE_COPY'] += len(copies) - 1
        bindings = {json.dumps(binding, sort_keys=True, separators=(',', ':'), allow_nan=False): binding
                    for row in copies for binding in row['source_bindings']}
        population.append({
            'observation_key': _digest([identity, prior, event]), 'runner_identity_id': identity,
            'event_id': event, 'event_identity_kinds': sorted({row.get('event_identity_kind', 'SOURCE_EVENT') for row in copies}),
            'date': prior, 'available_at': min(copies, key=lambda row: _instant(row['available_at']))['available_at'],
            'context': list(context), 'first_sectional': value, 'observation_fingerprint': fingerprint,
            'observation_ids': sorted({row['observation_id'] for row in copies}),
            'source_bindings': [bindings[key] for key in sorted(bindings)],
            'source_tracks': sorted({row['source_track'] for row in copies}),
            'alias_evidence': sorted({row['alias_evidence'] for row in copies if row.get('alias_evidence')}),
        })
    return population, {key: value for key, value in sorted(exclusions.items()) if value}


def _benchmark(identity, context, population):
    by_runner = defaultdict(list)
    for row in population:
        if row['runner_identity_id'] != identity and row['context'] == context:
            by_runner[row['runner_identity_id']].append(row)
    members = []
    for other, rows in sorted(by_runner.items()):
        members.append({'runner_identity_id': other,
                        'median_first_sectional': _median([row['first_sectional'] for row in rows]),
                        'observations': rows})
    enough = len(members) >= MIN_BENCHMARK_RUNNERS
    center = _median([member['median_first_sectional'] for member in members]) if enough else None
    mad = _median([abs(member['median_first_sectional'] - center) for member in members]) if enough else None
    spread = mad * MAD_SCALE if enough else None
    supported = enough and spread > 0 and math.isfinite(spread)
    return {
        'benchmark_id': _digest([identity, context]), 'excluded_runner_identity_id': identity,
        'context': context, 'status': ('SUPPORTED' if supported else 'INSUFFICIENT_OTHER_RUNNERS'
                                      if not enough else 'ZERO_OR_NONFINITE_SPREAD'),
        'other_runner_count': len(members), 'observation_count': sum(len(rows) for rows in by_runner.values()),
        'centre': center, 'mad': mad, 'spread': spread if spread is None or math.isfinite(spread) else None,
        'members': members,
    }


def build_sectional_candidate(packet):
    """Construct auditable speed estimates; zero means no *direct* adjustment.

    Packet identity and alias claims must already be authenticated by the caller.
    Availability must include the latest dependency needed to establish those claims.
    The output contains private historical cells and must stay in private storage.
    """
    target, cutoff, target_date = _validate(packet)
    population, exclusions = _population(packet['observations'], cutoff, target_date)
    runners, catalogue = [], {}
    for roster in packet['roster']:
        identity = roster['identity_id']
        identity_ready = identity is not None and _instant(roster['identity_available_at']) < cutoff
        own = [row for row in population if row['runner_identity_id'] == identity] if identity_ready else []
        own.sort(key=lambda row: (row['date'], row['event_id']), reverse=True)
        candidates, rejected = [], Counter()
        for row in own:
            key = _digest([identity, row['context']])
            if key not in catalogue:
                catalogue[key] = _benchmark(identity, row['context'], population)
            bench = catalogue[key]
            if bench['status'] != 'SUPPORTED':
                rejected[bench['status']] += 1
                continue
            raw = (bench['centre'] - row['first_sectional']) / bench['spread']
            clipped = max(-ZSCORE_LIMIT, min(ZSCORE_LIMIT, raw))
            candidates.append({**row, 'benchmark_id': key, 'z_clipped': clipped,
                               'z_was_clipped': not math.isfinite(raw) or raw != clipped})
        selected = candidates[:MAX_RUNNER_OBSERVATIONS]
        count = len(selected)
        weight = count / (count + SHRINKAGE_PRIOR_COUNT)
        typical = median(row['z_clipped'] for row in selected) if selected else 0.0
        runners.append({
            **roster, 'status': ('SUPPORTED' if count else 'NO_VERIFIED_IDENTITY' if identity is None
                                else 'IDENTITY_NOT_BEFORE_CUTOFF' if not identity_ready
                                else 'NO_SUPPORTED_HISTORIES'),
            'retained_prior_observations': len(own), 'usable_history_count': len(candidates),
            'supported_history_count': count, 'shrinkage_weight': weight,
            'unshrunk_relative_sectional': typical, 'speed_estimate': typical * weight,
            'selected_observations': selected, 'history_exclusions': dict(sorted(rejected.items())),
        })
    supported = sum(row['status'] == 'SUPPORTED' for row in runners)
    return {
        'schema_version': 'sectional_speed_candidate_v1', 'race_id': target['race_id'],
        'target_date': target['date'], 'cutoff': target['cutoff'],
        'runner_count': len(runners), 'supported_runner_count': supported,
        'status': 'SOME_SPEED_SUPPORT' if supported else 'BASELINE_ONLY',
        'parameters': {'minimum_other_runners': MIN_BENCHMARK_RUNNERS,
                       'maximum_runner_observations': MAX_RUNNER_OBSERVATIONS,
                       'mad_scale': MAD_SCALE, 'shrinkage_prior_count': SHRINKAGE_PRIOR_COUNT,
                       'zscore_limit': ZSCORE_LIMIT},
        'assumptions': ['USER_SUPPLIED_INDIVIDUAL_FIRST_SECTIONAL_MEANING',
                        'SOURCE_TIME_UNIT_ASSUMED_SECONDS',
                        'CROSS_CONTEXT_STANDARDIZED_SECTIONAL_TRANSFER_EXPLORATORY',
                        'UNKNOWN_LAYOUT_AND_ERA_STABLE_WITHIN_LITERAL_CONTEXT',
                        'DATE_ONLY_HISTORY_REQUIRES_STRICTLY_PRIOR_DATE',
                        'BENCHMARKS_UPDATE_FROM_ALL_AUTHENTICATED_PAST_AT_TARGET_CUTOFF'],
        'population_observation_count': len(population),
        'population_identity_count': len({row['runner_identity_id'] for row in population}),
        'population_exclusions': exclusions, 'runners': runners,
        'benchmarks': [catalogue[key] for key in sorted(catalogue)],
    }
