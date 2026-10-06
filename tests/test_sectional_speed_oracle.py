"""The oracle rereads fabricated raw cells rather than trusting feature output."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from race_collection import sectional_speed_oracle as oracle
from race_collection.sectional_speed_candidate import build_sectional_candidate
from tests.test_retained_card_timing_coverage import card


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fabricated(tmp_path):
    definitions = {
        1: [('2026-09-20', 'A', '5.1'), ('2026-09-19', 'A', '5.2')],
        2: [('2026-09-20', 'A', '5.4')],
        3: [('2026-09-20', 'A', '5.6'), ('2026-09-19', 'A', '5.7'), ('2026-09-18', 'A', '5.8')],
        4: [('2026-09-20', 'A_LONG', '6.0')],
        5: [('2026-09-20', 'A', '-')],
        6: [('2026-09-20', 'A', '6.4'), ('2026-09-20', 'A', '6.5')],
        7: [('2026-09-20', 'A', '6.7')],
        8: [('2026-09-20', 'A', '6.9')],
        9: [('2026-09-20', 'A', '7.1')],
    }
    source_rows, info = [], []
    for dog, history in definitions.items():
        for index, (day, track, split) in enumerate(history):
            row = {field: '' for field in oracle.FIELDS}
            row.update({'Dog Name': f'{dog}. Dog{dog}' if index == 0 else '',
                'DATE': day, 'TRACK': track, 'DIST': '400', '1 SEC': split,
                'TIME': '22.0', 'WIN': '21.0', 'BON': '20.0', 'PIR': '1234'})
            source_rows.append(row)
            info.append((dog, index, row))
    data = card(source_rows)
    path = tmp_path/'source.csv'
    path.write_bytes(data)
    reference = {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}
    observations = []
    for dog, index, row in info:
        projection = {field: row[field] for field in oracle.FIELDS}
        observation = {'observation_id': f'copy-{dog}-{index}',
            'event_id': digest([row['DATE'], 'A', 400]), 'event_identity_kind': 'RUNNER_DATE_CONTEXT_PROXY',
            'runner_identity_id': f'dog-{dog}', 'date': row['DATE'],
            'available_at': '2026-09-30T09:00:00+10:00', 'source_track': row['TRACK'],
            'canonical_track': 'A', 'distance_m': 400, 'first_sectional': row['1 SEC'],
            'observation_fingerprint': digest(projection),
            'source_bindings': [{'accepted_csv': reference, 'block_token': f'DOG{dog}',
                'block_row_index': index, 'box_number': dog,
                'available_by': '2026-09-30T09:00:00+10:00'}]}
        if row['TRACK'] != 'A':
            observation['alias_evidence'] = 'fabricated-exact-native-venue-proof'
        observations.append(observation)
    # Repeated retained copies of the same performance must not enlarge support.
    observations.append(copy.deepcopy(observations[0]))
    packet = {'target': {'race_id': 'fabricated-race', 'date': '2026-10-01',
        'cutoff': '2026-10-01T17:58:00+10:00'},
        'roster': [{'runner_id': f'entry-{dog}', 'identity_id': f'dog-{dog}',
                    'identity_available_at': '2026-09-30T09:00:00+10:00'} for dog in definitions],
        'observations': observations}
    return packet, build_sectional_candidate(packet)


def read(reference):
    return Path(reference['path']).read_bytes()


def test_recomputes_every_observed_category_from_original_cells(tmp_path):
    packet, result = fabricated(tmp_path)
    verified = oracle.verify_sample([packet], [result], read)
    assert verified['status'] == 'VERIFIED'
    assert set(verified['observed_categories']) == set(oracle.CATEGORIES)
    assert verified['absent_categories'] == []
    assert len(verified['checked_runners']) <= 6
    assert verified['raw_csv_reads'] == 1
    assert verified['raw_binding_checks'] == len(packet['observations'])
    assert verified['benchmark_contexts_recomputed'] > 0
    assert oracle.verify_sample([packet], [result], read) == verified


@pytest.mark.parametrize('change', ['estimate', 'benchmark', 'selected', 'support_claim'])
def test_rejects_arithmetic_and_membership_defects(tmp_path, change):
    packet, result = fabricated(tmp_path)
    if change == 'estimate':
        for runner in result['runners']:
            runner['speed_estimate'] += .2
    elif change == 'benchmark':
        for benchmark in result['benchmarks']:
            if benchmark['centre'] is not None:
                benchmark['centre'] += .2
    elif change == 'selected':
        for runner in result['runners']:
            for observation in runner['selected_observations']:
                observation['observation_ids'] = []
    else:
        for runner in result['runners']:
            runner['supported_history_count'] = 0
    with pytest.raises(oracle.OracleRejected):
        oracle.verify_sample([packet], [result], read)


def test_source_hash_and_location_checked_independently(tmp_path):
    packet, result = fabricated(tmp_path)
    Path(packet['observations'][0]['source_bindings'][0]['accepted_csv']['path']).write_text('changed')
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_HASH'):
        oracle.verify_sample([packet], [result], read)


def test_bound_cell_mismatch_is_not_hidden_by_recomputed_output(tmp_path):
    packet, _ = fabricated(tmp_path)
    packet['observations'][0]['first_sectional'] = '5.9'
    result = build_sectional_candidate(packet)
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_CELL_MISMATCH'):
        oracle.verify_sample([packet], [result], read)


def test_future_conflicting_copy_excluded_before_raw_reads_and_conflict_check(tmp_path):
    packet, _ = fabricated(tmp_path)
    packet['observations'] = [r for r in packet['observations'] if r['runner_identity_id'] != 'dog-6']
    later = copy.deepcopy(packet['observations'][0])
    later['available_at'] = '2026-10-02T09:00:00+10:00'
    later['first_sectional'] = '9.9'
    later['source_bindings'][0]['accepted_csv']['path'] = '/must/not/be/read'
    packet['observations'].append(later)
    result = build_sectional_candidate(packet)
    verified = oracle.verify_sample([packet], [result], read)
    assert 'conflict' in verified['absent_categories']
    assert verified['raw_csv_reads'] == 1


def test_neutral_probability_fallback_is_exact_and_unsupported_runner_is_renormalized():
    assert oracle.verify_adjustment([.5, .5], [0., 0.], .5, [.5, .5])
    assert oracle.verify_adjustment([.5, .5], [1., 0.], 0., [.5, .5])
    assert oracle.verify_adjustment([.5, .5], [1., 0.], 1.,
        [0.7310585786300049, 0.2689414213699951])
    with pytest.raises(oracle.OracleRejected, match='ORACLE_PROBABILITY_ADJUSTMENT'):
        oracle.verify_adjustment([.5, .5], [1., 0.], 1., [.5, .5])
    with pytest.raises(oracle.OracleRejected, match='ORACLE_NEUTRAL_NOT_EXACT'):
        oracle.verify_adjustment([.5, .5], [0., 0.], 1., [.5000000000000001, .4999999999999999])
