"""Independent raw-cell and arithmetic checks for a fixed sectional sample.

No candidate-module functions are imported. The supplied reader authenticates
and bounds original source reads. This oracle checks arithmetic and cell
bindings; the original adapter still owns native identity/alias proof.
"""
from collections import defaultdict
import csv
from datetime import date, datetime
import hashlib
import io
import json
import math
import re
from statistics import median


FIELDS = ('DATE', 'TRACK', 'DIST', 'TIME', 'WIN', 'BON', '1 SEC', 'PIR')
CATEGORIES = ('supported', 'sparse_one', 'sparse_two', 'missing', 'conflict', 'alias')
PARAMETERS = {'minimum_other_runners': 5, 'maximum_runner_observations': 5,
              'mad_scale': 1.4826, 'shrinkage_prior_count': 3, 'zscore_limit': 3.0}


class OracleRejected(ValueError):
    """Safe check category, containing no source values."""


def _assert(condition, category):
    if not condition:
        raise OracleRejected(category)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _time(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    _assert(parsed.utcoffset() is not None, 'ORACLE_TIMESTAMP_UNAWARE')
    return parsed


def _value(value):
    if value is None or isinstance(value, str) and value.strip().upper() in {
        '', '-', '--', '—', '–', 'N/A', 'NA', 'NULL', 'NONE'}:
        return None, 'missing'
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None, 'invalid'
    return (result, None) if not isinstance(value, bool) and math.isfinite(result) and result > 0 else (None, 'invalid')


def _layout(value):
    return None if value in (None, '', 'UNKNOWN', 'UNQUALIFIED', 'N/A', '-') else value


def _context(row):
    return (row['canonical_track'], row['distance_m'], _layout(row.get('layout_id')),
            _layout(row.get('layout_era')))


def _prior_rows(packet):
    cutoff = _time(packet['target']['cutoff'])
    target = date.fromisoformat(packet['target']['date'])
    rows = []
    for row in packet['observations']:
        available = _time(row['available_at'])
        if available >= cutoff:
            continue
        try:
            historical = date.fromisoformat(row['date'])
        except ValueError:
            continue
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', row['date']):
            continue
        if historical >= target or historical > available.astimezone(cutoff.tzinfo).date():
            continue
        rows.append(row)
    return rows


def _pool(rows):
    """Reconcile runner dates after temporal filtering, independently of output."""
    grouped, event_dates = defaultdict(list), defaultdict(set)
    for row in rows:
        grouped[row['runner_identity_id'], row['date']].append(row)
        event_dates[row['runner_identity_id'], row['event_id']].add(row['date'])
    pool, conflicts = [], set()
    for (identity, day), copies in grouped.items():
        signatures = {(r['event_id'], _context(r), *_value(r['first_sectional']),
                       r['observation_fingerprint']) for r in copies}
        if len(signatures) != 1 or any(len(event_dates[identity, r['event_id']]) > 1 for r in copies):
            conflicts.add(identity)
            continue
        event, context, value, reason, fingerprint = next(iter(signatures))
        if reason:
            continue
        bindings = {_hash(binding): binding for row in copies for binding in row['source_bindings']}
        pool.append({'identity': identity, 'date': day, 'event': event, 'context': context,
            'value': value, 'fingerprint': fingerprint, 'key': _hash([identity, day, event]),
            'ids': sorted({r['observation_id'] for r in copies}),
            'available': min(copies, key=lambda row: _time(row['available_at']))['available_at'],
            'bindings': bindings})
    return pool, conflicts


class _Cells:
    def __init__(self, reader):
        self.reader = reader
        self.blocks = {}
        self.reads = self.bytes = self.checked_bindings = 0

    def check(self, row):
        for binding in row['source_bindings']:
            reference = binding['accepted_csv']
            key = (reference['path'], reference['sha256'])
            if key not in self.blocks:
                data = self.reader.read(reference) if hasattr(self.reader, 'read') else self.reader(reference)
                _assert(hashlib.sha256(data).hexdigest() == reference['sha256'], 'ORACLE_SOURCE_HASH')
                text = data.decode('utf-8-sig', errors='strict')
                delimiter = '|' if text.splitlines()[0].count('|') > text.splitlines()[0].count(',') else ','
                parsed = csv.DictReader(io.StringIO(text), delimiter=delimiter)
                _assert({'Dog Name', *FIELDS} <= set(parsed.fieldnames or []), 'ORACLE_SOURCE_COLUMNS')
                blocks, token = defaultdict(list), None
                for record in parsed:
                    label = record['Dog Name'].strip().strip('"')
                    if label:
                        label = re.sub(r'^\s*\d+\.\s*', '', label)
                        token = re.sub('[^A-Z0-9]', '', label.upper())
                    _assert(token is not None, 'ORACLE_SOURCE_BLOCK')
                    blocks[token].append({field: record[field].strip() for field in FIELDS})
                self.blocks[key] = blocks
                self.reads += 1
                self.bytes += len(data)
            try:
                original = self.blocks[key][binding['block_token']][binding['block_row_index']]
            except (KeyError, IndexError, TypeError):
                raise OracleRejected('ORACLE_SOURCE_CELL_LOCATION') from None
            _assert(type(binding['block_row_index']) is int and binding['block_row_index'] >= 0,
                    'ORACLE_SOURCE_CELL_LOCATION')
            metres = re.fullmatch(r'\s*([1-9][0-9]*)\s*m?\s*', original['DIST'])
            _assert(original['DATE'] == row['date'] and original['TRACK'] == row['source_track']
                    and metres is not None and int(metres.group(1)) == row['distance_m']
                    and original['1 SEC'] == str(row['first_sectional']).strip()
                    and _hash(original) == row['observation_fingerprint'], 'ORACLE_SOURCE_CELL_MISMATCH')
            _assert(_time(binding['available_by']) <= _time(row['available_at'])
                    and _time(binding.get('identity_available_by', binding['available_by'])) <= _time(row['available_at']),
                    'ORACLE_SOURCE_AVAILABILITY')
            self.checked_bindings += 1


def _near(left, right, category):
    _assert(isinstance(left, (float, int)) and not isinstance(left, bool)
            and math.isfinite(left) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), category)


def _observation(actual, expected):
    _assert(actual['observation_key'] == expected['key']
            and actual['runner_identity_id'] == expected['identity']
            and actual['date'] == expected['date'] and actual['event_id'] == expected['event']
            and tuple(actual['context']) == expected['context']
            and actual['observation_fingerprint'] == expected['fingerprint']
            and actual['observation_ids'] == expected['ids']
            and actual['available_at'] == expected['available']
            and {_hash(b) for b in actual['source_bindings']} == set(expected['bindings']),
            'ORACLE_OBSERVATION_MEMBERSHIP')
    _near(actual['first_sectional'], expected['value'], 'ORACLE_OBSERVATION_VALUE')


def _check_runner(packet, output, runner_id, pool):
    roster = next(r for r in packet['roster'] if r['runner_id'] == runner_id)
    produced = next(r for r in output['runners'] if r['runner_id'] == runner_id)
    identity = roster['identity_id']
    ready = identity is not None and _time(roster['identity_available_at']) < _time(packet['target']['cutoff'])
    own = sorted((r for r in pool if ready and r['identity'] == identity),
                 key=lambda r: (r['date'], r['event']), reverse=True)
    supported = []
    checked_contexts = {}
    catalog = {b['benchmark_id']: b for b in output['benchmarks']}
    for observation in own:
        context = observation['context']
        if context not in checked_contexts:
            other = defaultdict(list)
            for peer in pool:
                if peer['identity'] != identity and peer['context'] == context:
                    other[peer['identity']].append(peer)
            centres = {who: median(r['value'] for r in records) for who, records in other.items()}
            enough = len(centres) >= 5
            centre = median(centres.values()) if enough else None
            mad = median(abs(value-centre) for value in centres.values()) if enough else None
            spread = 1.4826 * mad if enough else None
            available = enough and spread > 0 and math.isfinite(spread)
            benchmark_id = _hash([identity, list(context)])
            _assert(benchmark_id in catalog, 'ORACLE_BENCHMARK_MISSING')
            actual = catalog[benchmark_id]
            expected_status = 'SUPPORTED' if available else ('INSUFFICIENT_OTHER_RUNNERS' if not enough else 'ZERO_OR_NONFINITE_SPREAD')
            _assert(actual['excluded_runner_identity_id'] == identity and tuple(actual['context']) == context
                and actual['status'] == expected_status and actual['other_runner_count'] == len(other)
                and actual['observation_count'] == sum(len(r) for r in other.values())
                and len(actual['members']) == len(other), 'ORACLE_BENCHMARK_MEMBERSHIP')
            member_map = {m['runner_identity_id']: m for m in actual['members']}
            _assert(set(member_map) == set(other), 'ORACLE_BENCHMARK_MEMBERSHIP')
            for who, histories in other.items():
                member = member_map[who]
                _near(member['median_first_sectional'], centres[who], 'ORACLE_RUNNER_CENTRE')
                empirical = {r['observation_key']: r for r in member['observations']}
                _assert(len(member['observations']) == len(histories) and set(empirical) == {r['key'] for r in histories},
                        'ORACLE_BENCHMARK_OBSERVATIONS')
                for history in histories:
                    _observation(empirical[history['key']], history)
            for name, expected in (('centre', centre), ('mad', mad), ('spread', spread)):
                if expected is None:
                    _assert(actual[name] is None, 'ORACLE_BENCHMARK_ARITHMETIC')
                else:
                    _near(actual[name], expected, 'ORACLE_BENCHMARK_ARITHMETIC')
            checked_contexts[context] = (available, centre, spread)
        usable, centre, spread = checked_contexts[context]
        if usable:
            z = min(3.0, max(-3.0, (centre-observation['value']) / spread))
            supported.append((observation, z))
    selected = supported[:5]
    expected_status = ('SUPPORTED' if selected else 'NO_VERIFIED_IDENTITY' if identity is None
        else 'IDENTITY_NOT_BEFORE_CUTOFF' if not ready else 'NO_SUPPORTED_HISTORIES')
    _assert(produced['status'] == expected_status and produced['identity_id'] == identity
        and produced['identity_available_at'] == roster['identity_available_at'], 'ORACLE_RUNNER_DISPOSITION')
    _assert(produced['retained_prior_observations'] == len(own)
        and produced['usable_history_count'] == len(supported)
        and produced['supported_history_count'] == len(selected)
        and len(produced['selected_observations']) == len(selected), 'ORACLE_HISTORY_COUNTS')
    for actual, (expected, z) in zip(produced['selected_observations'], selected):
        _observation(actual, expected)
        _near(actual['z_clipped'], z, 'ORACLE_STANDARDIZATION')
    weight = len(selected) / (len(selected)+3)
    estimate = median(z for _, z in selected) if selected else 0.0
    _near(produced['shrinkage_weight'], weight, 'ORACLE_SHRINKAGE')
    _near(produced['unshrunk_relative_sectional'], estimate, 'ORACLE_RUNNER_ESTIMATE')
    _near(produced['speed_estimate'], estimate*weight, 'ORACLE_RUNNER_ESTIMATE')
    return len(selected), len(checked_contexts)


def _sample_support_count(packet, roster, pool):
    """Select sample categories from inputs, never produced support claims."""
    identity = roster['identity_id']
    if identity is None or _time(roster['identity_available_at']) >= _time(packet['target']['cutoff']):
        return 0
    supported = 0
    contexts = {}
    for observation in pool:
        if observation['identity'] != identity:
            continue
        context = observation['context']
        if context not in contexts:
            others = defaultdict(list)
            for row in pool:
                if row['identity'] != identity and row['context'] == context:
                    others[row['identity']].append(row['value'])
            centres = [median(values) for values in others.values()]
            if len(centres) < 5:
                contexts[context] = False
            else:
                centre = median(centres)
                spread = 1.4826 * median(abs(value-centre) for value in centres)
                contexts[context] = spread > 0 and math.isfinite(spread)
        supported += contexts[context]
    return min(supported, 5)


def verify_sample(packets, outputs, reader):
    """Verify up to six min-hash category representatives against raw cells.

    ``outputs`` is the matching list of ``build_sectional_candidate`` results.
    Sampling uses only feature-support and provenance categories, never labels,
    probabilities, effect sizes or outcomes. Absent categories stay absent.
    """
    targets = {p['target']['race_id']: p for p in packets}
    actuals = {o['race_id']: o for o in outputs}
    _assert(len(targets) == len(packets) == len(actuals) == len(outputs)
            and targets.keys() == actuals.keys(), 'ORACLE_POPULATION_MISMATCH')
    candidates = defaultdict(list)
    for race_id, packet in targets.items():
        output = actuals[race_id]
        _assert(output['parameters'] == PARAMETERS, 'ORACLE_PARAMETERS_CHANGED')
        _assert(len(output['runners']) == len(packet['roster'])
            and {r['runner_id'] for r in output['runners']} == {r['runner_id'] for r in packet['roster']},
            'ORACLE_ROSTER_MISMATCH')
        prior = _prior_rows(packet)
        pool, conflicting = _pool(prior)
        for roster in packet['roster']:
            identity = roster['identity_id']
            own = [r for r in prior if r['runner_identity_id'] == identity]
            count = _sample_support_count(packet, roster, pool)
            flags = {'supported': count > 0,
                     'sparse_one': count == 1,
                     'sparse_two': count == 2,
                     'missing': count == 0 or any(_value(r['first_sectional'])[1] for r in own),
                     'conflict': identity in conflicting,
                     'alias': any(r['canonical_track'] != r['source_track'] for r in own)}
            for category, present in flags.items():
                if present:
                    candidates[category].append((_hash([race_id, roster['runner_id']]), race_id, roster['runner_id']))
    selections = {category: min(candidates[category])[1:] for category in CATEGORIES if candidates[category]}
    cells = _Cells(reader)
    checked, race_pools, benchmark_count = [], {}, 0
    for race_id, runner_id in sorted(set(selections.values())):
        if race_id not in race_pools:
            prior = _prior_rows(targets[race_id])
            for row in prior:
                cells.check(row)
            race_pools[race_id], _ = _pool(prior)
        count, contexts = _check_runner(targets[race_id], actuals[race_id], runner_id, race_pools[race_id])
        benchmark_count += contexts
        checked.append({'race_id': race_id, 'runner_id': runner_id, 'selected_count': count,
                        'categories': [c for c, pair in selections.items() if pair == (race_id, runner_id)]})
    return {'schema_version': 'sectional_speed_independent_raw_cell_oracle_v1', 'status': 'VERIFIED',
        'sampling': 'FIXED_CATEGORY_MIN_SHA256_RACE_AND_RUNNER', 'maximum_runners': 6,
        'checked_runners': checked, 'observed_categories': sorted(selections),
        'absent_categories': [c for c in CATEGORIES if c not in selections],
        'benchmark_contexts_recomputed': benchmark_count, 'raw_csv_reads': cells.reads,
        'raw_csv_bytes': cells.bytes, 'raw_binding_checks': cells.checked_bindings,
        'provider_requests': 0, 'result_payloads_opened': 0,
        'limits': ['Identity and alias proof remain authenticated adapter responsibilities',
                   'Representative sample rather than every runner arithmetic',
                   'No predictive value or probability adjustment conclusion']}


def verify_adjustment(baseline, estimates, coefficient, probabilities):
    """Optional independent whole-field softmax adjustment check."""
    _assert(len(baseline) == len(estimates) == len(probabilities) and len(baseline) > 1,
            'ORACLE_PROBABILITY_SHAPE')
    _assert(all(math.isfinite(p) and p > 0 for p in baseline)
        and math.isclose(sum(baseline), 1, abs_tol=1e-12)
        and math.isfinite(coefficient) and all(math.isfinite(z) for z in estimates),
        'ORACLE_PROBABILITY_INPUT')
    if coefficient == 0 or not any(estimates):
        _assert(probabilities == baseline, 'ORACLE_NEUTRAL_NOT_EXACT')
        return True
    logits = [math.log(p)+coefficient*z for p, z in zip(baseline, estimates)]
    high = max(logits)
    weights = [math.exp(value-high) for value in logits]
    total = sum(weights)
    for actual, expected in zip(probabilities, (weight/total for weight in weights)):
        _near(actual, expected, 'ORACLE_PROBABILITY_ADJUSTMENT')
    return True
