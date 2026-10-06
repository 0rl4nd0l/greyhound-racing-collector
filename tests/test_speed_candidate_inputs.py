"""Fabricated source bytes; no retained real timing or result access."""
import hashlib
import json

import pytest

from race_collection import speed_candidate_inputs as inputs
from tests.test_retained_card_timing_coverage import card, put
from tests.test_runner_completeness import _runner_row


def case(tmp_path, *, index=1, profile='101', conflict=False, track='GUNN', page_entry='201',
         dom_race_ids=('500',), sealed_race_id='500'):
    base = tmp_path / str(index)
    rows = [{'Dog Name': '1. Alpha', 'DATE': '2026-09-27', 'TRACK': track,
        'DIST': '340', '1 SEC': '6.2', 'TIME': '20.1'},
        {'Dog Name': '', 'DATE': '2026-09-26', 'TRACK': track, 'DIST': '340', '1 SEC': '-'},
        {'Dog Name': '2. Beta', 'DATE': '2026-09-25', 'TRACK': track,
         'DIST': '340', '1 SEC': '6.4', 'TIME': '20.2'}]
    markup = '<main>' + ''.join(f'<span data-race-id="{value}"></span>' for value in dom_race_ids)
    markup += '<table>' + _runner_row(1, 'Alpha', runner_id=page_entry,
        dog_id=profile) + _runner_row(2, 'Beta', runner_id='202', dog_id='102') + '</table></main>'
    if conflict:
        markup = markup.replace('data-dog-id="101"', 'data-dog-id="101"><i data-dog-id="999"></i')
    member = {'race_id': f'Race {index} - GUNN - 2026-10-01', 'source_race_date': '2026-10-01',
        'venue': 'GUNN', 'target_distance_raw': '340', 'target_runner_slots': 2,
        'runner_set_sha256': hashlib.sha256(b'roster').hexdigest(),
        'jump_at': '2026-10-01T20:00:00+10:00', 'csv_header': inputs.strict.HEADER}
    member['accepted_csv'] = put(base/'source/card.csv', card(rows))
    member['raw_export'] = put(base/'raw.csv', card(rows))
    member['primary_page'] = put(base/'page.html', markup.encode())
    url = f'https://www.thedogs.com.au/racing/gunnedah/2026-10-01/{index}'
    receipt = {'body_sha256': member['primary_page']['sha256'], 'status_code': 200,
        'race_discovery_key': member['race_id'], 'requested_url': url,
        'capture_timestamp': '2026-10-01T19:54:00+10:00'}
    member['primary_receipt'] = put(base/'receipt.json', receipt)
    sidecar = {'runner_completeness_after_canonical_alignment': {'runner_count': 2,
        'participants': [{'box_number': 1, 'dog_name': 'Alpha', 'source_native_runner_id': '201'},
                         {'box_number': 2, 'dog_name': 'Beta', 'source_native_runner_id': '202'}]},
        'target_distance': '340', 'content_sha256': member['accepted_csv']['sha256'],
        'raw_content_sha256': member['raw_export']['sha256'], 'race_url': url,
        'source_native_race_id': sealed_race_id, 'primary_race_page_evidence': {
            'body_sha256': member['primary_page']['sha256'],
            'receipt_sha256': member['primary_receipt']['sha256']}}
    member['sidecar'] = put(base/'source/card.csv.metadata.json', sidecar)
    member['bundle_manifest'] = put(base/'bundle.json', {'files': {
        'source/card.csv': member['accepted_csv'], 'source/card.csv.metadata.json': member['sidecar']}})
    member['admission'] = put(base/'admission.json', {'race': {'race_id': member['race_id'],
        'race_date': member['source_race_date'], 'venue': 'GUNN'},
        'admitted_at': '2026-10-01T19:55:00+10:00', 'decision_at': '2026-10-01T19:58:30+10:00'})
    original = {**member, 'original_admitted_at': '2026-10-01T19:55:00+10:00',
        'original_published_complete_at': '2026-10-01T19:55:01+10:00'}
    return member, original


def test_verified_same_row_bridge_binds_history_and_never_projects_outcomes(tmp_path):
    member, original = case(tmp_path)
    reader = inputs.SnapshotReader()
    packet, observations, audit = inputs.construct_member(reader, member, original)
    assert [(r['runner_id'], r['identity_id']) for r in packet['roster']] == [
        ('201', 'thedogs:dog:101'), ('202', 'thedogs:dog:102')]
    assert observations[0]['available_at'] == original['original_published_complete_at']
    assert observations[0]['source_bindings'][0]['accepted_csv'] == member['accepted_csv']
    assert observations[0]['source_bindings'][0]['block_row_index'] == 0
    assert observations[0]['source_bindings'][0]['block_token'] == 'ALPHA'
    assert observations[0]['first_sectional'] == '6.2'
    assert observations[1]['first_sectional'] == '-'
    assert audit['verified_profile_count'] == 2
    assert reader.reader.reads == 7  # Unique input graph, no repeated CSV/HTML reads.
    encoded = json.dumps([packet, observations, audit])
    assert 'SENSITIVE_PLACEMENT' not in encoded and 'SENSITIVE_ODDS' not in encoded
    assert '20.1' not in encoded


@pytest.mark.parametrize('profile,conflict,category', [
    (None, False, 'PROFILE_ID_MISSING'), ('101', True, 'PROFILE_ID_CONFLICT')])
def test_unknown_or_conflicting_profile_retains_neutral_runner_without_name_join(tmp_path, profile, conflict, category):
    member, original = case(tmp_path, profile=profile, conflict=conflict)
    packet, observations, audit = inputs.construct_member(inputs.SnapshotReader(), member, original)
    assert packet['roster'][0]['identity_id'] is None
    assert packet['roster'][0]['identity_available_at'] is None
    assert audit['profile_dispositions'][category] == 1
    assert len(observations) == 1
    assert observations[0]['runner_identity_id'] == 'thedogs:dog:102'


def test_different_native_entry_cannot_be_bound_by_name_and_box(tmp_path):
    member, original = case(tmp_path, page_entry='777')
    with pytest.raises(ValueError, match='TARGET_NATIVE_ROSTER_MISMATCH'):
        inputs.construct_member(inputs.SnapshotReader(), member, original)


def test_inactive_reserves_without_entry_ids_do_not_conflict_with_active_field():
    markup = '<main data-race-id="500"><table>' + ''.join([
        _runner_row(1, 'Alpha', runner_id='201', dog_id='101'),
        _runner_row(2, 'Beta', runner_id='202', dog_id='102'),
        _runner_row(9, 'Reserve One', runner_id=''),
        _runner_row(10, 'Reserve Two', runner_id='')]) + '</table></main>'
    sidecar = {'source_native_race_id': '500', 'race_url': 'https://www.thedogs.com.au/racing/gunnedah/2026-10-01/1',
        'runner_completeness_after_canonical_alignment': {'participants': [
            {'box_number': 1, 'dog_name': 'Alpha', 'source_native_runner_id': '201'},
            {'box_number': 2, 'dog_name': 'Beta', 'source_native_runner_id': '202'}]}}
    profiles, method = inputs._profiles(markup.encode(), sidecar, [(1, 'ALPHA'), (2, 'BETA')],
        {'capture_timestamp': '2026-10-01T19:54:00+10:00'})
    assert profiles[(1, 'ALPHA')][1] == 'thedogs:dog:101'
    assert profiles[(2, 'BETA')][1] == 'thedogs:dog:102'
    assert method == 'MATCHING_NATIVE_HTML_AND_SEALED_RACE_ID'


def test_missing_html_race_id_uses_complete_sealed_page_receipt_binding(tmp_path):
    member, original = case(tmp_path, dom_race_ids=())
    packet, observations, audit = inputs.construct_member(inputs.SnapshotReader(), member, original)
    assert [r['identity_id'] for r in packet['roster']] == ['thedogs:dog:101', 'thedogs:dog:102']
    assert len(observations) == 3
    assert audit['race_identity_method'] == 'SEALED_NATIVE_RACE_AND_BOUND_PREJUMP_PAGE_RECEIPT'


@pytest.mark.parametrize('dom_ids', [('',), ('501',), ('not-numeric',), ('500', '501')])
def test_present_blank_invalid_conflicting_or_mismatching_dom_race_id_rejects(tmp_path, dom_ids):
    member, original = case(tmp_path, dom_race_ids=dom_ids)
    with pytest.raises(ValueError, match='TARGET_NATIVE_RACE_MISMATCH'):
        inputs.construct_member(inputs.SnapshotReader(), member, original)


@pytest.mark.parametrize('sealed', ['', '0', 'not-numeric'])
def test_missing_dom_race_id_does_not_make_invalid_sealed_race_id_usable(tmp_path, sealed):
    member, original = case(tmp_path, dom_race_ids=(), sealed_race_id=sealed)
    with pytest.raises(ValueError, match='TARGET_SEALED_NATIVE_RACE_INVALID'):
        inputs.construct_member(inputs.SnapshotReader(), member, original)


def test_missing_dom_race_id_never_relaxes_native_runner_field(tmp_path):
    member, original = case(tmp_path, dom_race_ids=(), page_entry='777')
    with pytest.raises(ValueError, match='TARGET_NATIVE_ROSTER_MISMATCH'):
        inputs.construct_member(inputs.SnapshotReader(), member, original)


@pytest.mark.parametrize('change', ['missing_binding', 'wrong_race', 'wrong_url', 'wrong_body', 'late_receipt'])
def test_missing_dom_race_id_requires_exact_bound_receipt(tmp_path, change):
    from pathlib import Path
    member, original = case(tmp_path, dom_race_ids=())
    sidecar = json.loads(Path(member['sidecar']['path']).read_bytes())
    receipt = json.loads(Path(member['primary_receipt']['path']).read_bytes())
    binding = {'race_id': member['race_id'], 'jump_at': member['jump_at']}
    if change == 'missing_binding':
        binding = None
    elif change == 'wrong_race':
        receipt['race_discovery_key'] = 'other-race'
    elif change == 'wrong_url':
        receipt['requested_url'] = 'https://www.thedogs.com.au/racing/other'
    elif change == 'wrong_body':
        receipt['body_sha256'] = 'a'*64
    else:
        receipt['capture_timestamp'] = member['jump_at']
    with pytest.raises(ValueError, match='TARGET_RACE_RECEIPT_BINDING_'):
        inputs._profiles(Path(member['primary_page']['path']).read_bytes(), sidecar,
            [(1, 'ALPHA'), (2, 'BETA')], receipt, race_binding=binding)


def test_unverified_track_alias_stays_distinct_and_is_audited(tmp_path):
    member, original = case(tmp_path, track='GUNNEDAH')
    _, observations, audit = inputs.construct_member(inputs.SnapshotReader(), member, original)
    assert all(o['canonical_track'] == 'GUNNEDAH' for o in observations)
    assert audit['row_dispositions']['RAW_TARGET_CONTEXT_MISMATCH'] == 3
    assert audit['raw_contexts'] == [{'track': 'GUNNEDAH', 'distance_m': 340, 'rows': 3}]


def test_changed_source_hash_is_rejected_before_construction(tmp_path):
    member, original = case(tmp_path)
    from pathlib import Path
    Path(member['primary_page']['path']).write_text('tampered')
    with pytest.raises(ValueError, match='INPUT_HASH'):
        inputs.construct_member(inputs.SnapshotReader(), member, original)


def test_later_identity_proof_cannot_be_used_for_earlier_cutoff(tmp_path):
    member, original = case(tmp_path)
    original['original_published_complete_at'] = '2026-10-01T19:59:00+10:00'
    with pytest.raises(ValueError, match='FEATURE_CAPTURE_NOT_BEFORE_CUTOFF'):
        inputs.construct_member(inputs.SnapshotReader(), member, original)


def test_utc_decision_uses_authenticated_source_jump_calendar_offset(tmp_path):
    member, original = case(tmp_path)
    from pathlib import Path
    admission = json.loads(Path(member['admission']['path']).read_bytes())
    admission['decision_at'] = '2026-10-01T09:58:30+00:00'
    member['admission'] = put(tmp_path/'utc-admission.json', admission)
    original['admission'] = member['admission']
    packet, _, _ = inputs.construct_member(inputs.SnapshotReader(), member, original)
    assert packet['target']['cutoff'] == '2026-10-01T19:58:30+10:00'


def test_fixed_population_loader_preserves_copies_for_availability_aware_dedup(tmp_path, monkeypatch):
    pairs = [case(tmp_path, index=index) for index in (1, 2)]
    originals = [p[1] for p in pairs]
    membership = put(tmp_path/'membership.json', {'members': originals})
    manifest = put(tmp_path/'manifest.json', {
        'schema_version': 'retained_speed_breadth_metadata_manifest_v1',
        'selection': 'ALL_82_FIXED_MEMBERS_IRRESPECTIVE_OF_RESULT_STATUS',
        'membership': membership, 'members': [p[0] for p in pairs]})
    reference = {k: manifest[k] for k in ('path', 'sha256')}
    monkeypatch.setattr(inputs.strict, 'MANIFEST_PATH', reference['path'])
    monkeypatch.setattr(inputs.strict, 'MANIFEST_SHA', reference['sha256'])
    monkeypatch.setattr(inputs.strict, 'MEMBERSHIP_SHA', membership['sha256'])
    monkeypatch.setattr(inputs.strict, 'MEMBERS', 2)
    monkeypatch.setattr(inputs.strict, 'RUNNER_SLOTS', 4)
    result = inputs.load_inputs(reference)
    assert len(result['observations']) == 6
    assert len({o['observation_id'] for o in result['observations']}) == 3
    assert result['audit']['unique_verified_dog_profiles'] == 2
    assert result['audit']['runner_appearances'] == 4
    assert result['audit']['verified_aliases'] == []
    assert result['audit']['result_payloads_opened'] == 0


def test_manifest_not_in_exact_scope_is_rejected_before_read():
    with pytest.raises(ValueError, match='FEATURE_MANIFEST_NOT_FIXED'):
        inputs.load_inputs({'path': '/not/authorized', 'sha256': 'a'*64})
