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


def fabricated(tmp_path, *, missing_sectional='-'):
    definitions = {
        1: [('2026-09-20', 'A', '5.1'), ('2026-09-19', 'A', '5.2')],
        2: [('2026-09-20', 'A', '5.4')],
        3: [('2026-09-20', 'A', '5.6'), ('2026-09-19', 'A', '5.7'), ('2026-09-18', 'A', '5.8')],
        4: [('2026-09-20', 'A_LONG', '6.0')],
        5: [('2026-09-20', 'A', missing_sectional)],
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
        event = digest([row['DATE'], row['TRACK'], 400])
        observation = {'observation_id': digest([f'dog-{dog}', event, digest(projection)]),
            'event_id': event, 'event_identity_kind': 'RUNNER_DATE_CONTEXT_PROXY',
            'runner_identity_id': f'dog-{dog}', 'date': row['DATE'],
            'available_at': '2026-09-30T09:00:00+10:00', 'source_track': row['TRACK'],
            'canonical_track': 'A', 'distance_m': 400, 'first_sectional': row['1 SEC'],
            'observation_fingerprint': digest(projection),
            'source_bindings': [{'source_race_id': 'fabricated-race', 'accepted_csv': reference, 'block_token': f'DOG{dog}',
                'block_row_index': index, 'box_number': dog,
                'available_by': '2026-09-30T09:00:00+10:00',
                'identity_available_by': '2026-09-30T09:00:00+10:00'}]}
        if row['TRACK'] != 'A':
            observation['alias_evidence'] = 'fabricated-exact-native-venue-proof'
        observations.append(observation)
    # Repeated retained copies of the same performance must not enlarge support.
    observations.append(copy.deepcopy(observations[0]))
    packet = {'target': {'race_id': 'fabricated-race', 'date': '2026-10-01',
        'cutoff': '2026-10-01T17:58:00+10:00'},
        'roster': [{'runner_id': f'entry-{dog}', 'identity_id': f'dog-{dog}',
                    'box_number': dog,
                    'identity_available_at': '2026-09-30T09:00:00+10:00'} for dog in definitions],
        'observations': observations}
    packet['target']['observation_pool_scope'] = 'CARD_LOCAL'
    packet['target']['source_card'] = {'race_id': 'fabricated-race', 'racing_date': '2026-10-01',
        'accepted_csv': reference, 'available_at': '2026-09-30T09:00:00+10:00',
        'aliases': {'A_LONG': {'canonical_track': 'A', 'evidence': 'fabricated-exact-native-venue-proof'}},
        'binding_native_runner_id': False,
        'binding_base': {'source_race_id': 'fabricated-race', 'accepted_csv': reference,
            'available_by': '2026-09-30T09:00:00+10:00',
            'identity_available_by': '2026-09-30T09:00:00+10:00'},
        'roster': [{**r, 'block_token': f'DOG{r["box_number"]}'} for r in packet['roster']]}
    return packet, build_sectional_candidate(packet)


def read(reference):
    return Path(reference['path']).read_bytes()


def test_recomputes_every_observed_category_from_original_cells(tmp_path):
    packet, result = fabricated(tmp_path)
    verified = oracle.verify_sample([packet], [result], read)
    assert verified['status'] == 'VERIFIED'
    assert set(verified['observed_categories']) == set(oracle.CATEGORIES)
    assert verified['absent_categories'] == []
    assert len(verified['checked_runners']) <= 7
    assert verified['raw_csv_reads'] == 1
    assert verified['raw_binding_checks'] == len(packet['observations'])-1
    assert verified['complete_source_cards_enumerated'] == 1
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
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_COPY_CONFLICT'):
        oracle.verify_sample([packet], [result], read)


def test_future_conflicting_copy_excluded_before_raw_reads_and_conflict_check(tmp_path):
    packet, _ = fabricated(tmp_path)
    later = copy.deepcopy(packet['observations'][0])
    later['available_at'] = '2026-10-02T09:00:00+10:00'
    later['first_sectional'] = '9.9'
    later['source_bindings'][0]['accepted_csv']['path'] = '/must/not/be/read'
    packet['observations'].append(later)
    result = build_sectional_candidate(packet)
    verified = oracle.verify_sample([packet], [result], read)
    assert 'conflict' in verified['observed_categories']  # Only the original raw conflict.
    assert verified['raw_csv_reads'] == 1


@pytest.mark.parametrize('omission', ['newest', 'peer', 'missing_cell'])
def test_adapter_omissions_cannot_pass_after_features_recomputed(tmp_path, omission):
    packet, _ = fabricated(tmp_path)
    if omission == 'newest':
        packet['observations'] = [r for r in packet['observations'] if not (
            r['runner_identity_id'] == 'dog-1' and r['date'] == '2026-09-20')]
    elif omission == 'peer':
        packet['observations'] = [r for r in packet['observations'] if r['runner_identity_id'] != 'dog-9']
    else:
        packet['observations'] = [r for r in packet['observations'] if r['runner_identity_id'] != 'dog-5']
    result = build_sectional_candidate(packet)
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_ENUMERATION_MISMATCH'):
        oracle.verify_sample([packet], [result], read)


def test_missing_cell_category_does_not_mean_merely_unsupported(tmp_path):
    packet, result = fabricated(tmp_path, missing_sectional='7.5')
    verified = oracle.verify_sample([packet], [result], read)
    assert 'unsupported' in verified['observed_categories']
    assert 'missing' in verified['absent_categories']


def test_inventory_is_required_even_when_observation_bank_empty(tmp_path):
    packet, _ = fabricated(tmp_path)
    packet['observations'] = []
    result = build_sectional_candidate(packet)
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_ENUMERATION_MISMATCH'):
        oracle.verify_sample([packet], [result], read)
    del packet['target']['source_card']
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_INVENTORY_REQUIRED'):
        oracle.verify_sample([packet], [result], read)


def test_actual_october_adapter_shape_global_pool_and_whole_source_omission(tmp_path):
    from race_collection.speed_candidate_inputs import SnapshotReader, construct_member
    from tests.test_speed_candidate_inputs import case

    packets, observations = [], []
    for index in (1, 2):
        member, original = case(tmp_path, index=index)
        packet, rows, _ = construct_member(SnapshotReader(), member, original)
        packets.append(packet)
        observations.extend(rows)
    packets = [{**p, 'observations': observations} for p in packets]
    outputs = [build_sectional_candidate(p) for p in packets]
    verified = oracle.verify_sample(packets, outputs, read)
    assert verified['complete_source_cards_enumerated'] == 2
    assert verified['eligible_source_row_copies_enumerated'] == 6
    assert verified['packet_source_enumerations_verified'] == 2
    assert verified['raw_csv_reads'] == 2
    omitted = [row for row in observations if row['source_bindings'][0]['source_race_id']
               != packets[1]['target']['race_id']]
    partial = [{**p, 'observations': omitted} for p in packets]
    with pytest.raises(oracle.OracleRejected, match='ORACLE_SOURCE_ENUMERATION_MISMATCH'):
        oracle.verify_sample(partial, [build_sectional_candidate(p) for p in partial], read)


def test_neutral_probability_fallback_is_exact_and_unsupported_runner_is_renormalized():
    assert oracle.verify_adjustment([.5, .5], [0., 0.], .5, [.5, .5])
    assert oracle.verify_adjustment([.5, .5], [1., 0.], 0., [.5, .5])
    assert oracle.verify_adjustment([.5, .5], [1., 0.], 1.,
        [0.7310585786300049, 0.2689414213699951])
    with pytest.raises(oracle.OracleRejected, match='ORACLE_PROBABILITY_ADJUSTMENT'):
        oracle.verify_adjustment([.5, .5], [1., 0.], 1., [.5, .5])
    with pytest.raises(oracle.OracleRejected, match='ORACLE_NEUTRAL_NOT_EXACT'):
        oracle.verify_adjustment([.5, .5], [0., 0.], 1., [.5000000000000001, .4999999999999999])
