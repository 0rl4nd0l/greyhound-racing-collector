"""Fabricated retained cards only; never open the private fixed cohort."""
import json
import hashlib
from pathlib import Path

import pytest

from race_collection import retained_speed_features as speed
from tests.test_retained_card_timing_coverage import card, put


def test_cli_is_default_off_without_opening_supplied_manifest(capsys):
    from scripts.derive_retained_speed_features import main

    assert main(['--manifest', '/does/not/exist']) == 0
    assert json.loads(capsys.readouterr().out) == {
        'status': 'DEFAULT_OFF', 'provider_requests': 0, 'result_requests': 0}


@pytest.fixture
def retained_case(tmp_path, monkeypatch):
    """Actual authentication/parser with fabricated 82-card / 583-runner graph."""
    def prepare(change=None):
        members, originals = [], []
        for index in range(82):
            count = 7 if index < 73 else 8
            base = tmp_path/f'card{index}'
            rows = []
            for box in range(1, count+1):
                for start, day in enumerate(('2026-09-28', '2026-09-27', '2026-09-26', '2026-09-25')):
                    rows.append({'Dog Name': f'{box}. Dog{box}' if start == 0 else '',
                        'DATE': day, 'TRACK': 'GUNN', 'DIST': '340',
                        '1 SEC': str(6+(box-1)*.1+start*.2), 'TIME': 'PRIVATE_UNUSED_TOTAL'})
            if change is not None:
                change(index, rows)
            m = {'race_id': f'Race {index+1} - GUNN - 2026-10-01',
                'source_race_date': '2026-10-01', 'venue': 'GUNN',
                'target_distance_raw': '340m', 'target_runner_slots': count,
                'runner_set_sha256': hashlib.sha256(f'roster-{index}'.encode()).hexdigest(),
                'jump_at': '2026-10-01T20:00:00+10:00', 'csv_header': speed.HEADER}
            m['accepted_csv'] = put(base/'source/card.csv', card(rows))
            m['raw_export'] = put(base/'raw.csv', card(rows))
            m['primary_page'] = put(base/'page.html', b'<html>fabricated prior card</html>')
            receipt = {'body_sha256': m['primary_page']['sha256'], 'status_code': 200,
                'race_discovery_key': m['race_id'], 'requested_url': f'https://source/race/{index}',
                'capture_timestamp': '2026-10-01T19:54:00+10:00'}
            m['primary_receipt'] = put(base/'receipt.json', receipt)
            sidecar = {'runner_completeness_after_canonical_alignment': {'runner_count': count,
                'participants': [{'box_number': box, 'dog_name': f'Dog{box}'} for box in range(1, count+1)]},
                'target_distance': '340m', 'content_sha256': m['accepted_csv']['sha256'],
                'raw_content_sha256': m['raw_export']['sha256'], 'race_url': receipt['requested_url'],
                'primary_race_page_evidence': {'body_sha256': m['primary_page']['sha256'],
                    'receipt_sha256': m['primary_receipt']['sha256']}}
            m['sidecar'] = put(base/'source/card.csv.metadata.json', sidecar)
            m['bundle_manifest'] = put(base/'bundle_manifest.json', {'files': {
                'source/card.csv': m['accepted_csv'], 'source/card.csv.metadata.json': m['sidecar']}})
            m['admission'] = put(base/'admission.json', {'race': {'race_id': m['race_id'],
                'race_date': m['source_race_date'], 'venue': m['venue']},
                'admitted_at': '2026-10-01T19:55:00+10:00', 'decision_at': '2026-10-01T19:58:30+10:00'})
            originals.append({**m, 'original_admitted_at': '2026-10-01T19:55:00+10:00',
                'original_published_complete_at': '2026-10-01T19:55:01+10:00',
                'result_status': 'QUARANTINED' if index < 8 else 'UNKNOWN'})
            members.append(m)
        membership = put(tmp_path/'membership.json', {'members': originals})
        manifest = {'schema_version': 'retained_speed_breadth_metadata_manifest_v1',
            'selection': 'ALL_82_FIXED_MEMBERS_IRRESPECTIVE_OF_RESULT_STATUS',
            'membership': membership, 'members': members}
        manifest_ref = put(tmp_path/'manifest.json', manifest)
        ref = {k: manifest_ref[k] for k in ('path', 'sha256')}
        # Substitute only exact fixture identity pins, never authentication/parser.
        monkeypatch.setattr(speed, 'MANIFEST_PATH', ref['path'])
        monkeypatch.setattr(speed, 'MANIFEST_SHA', ref['sha256'])
        monkeypatch.setattr(speed, 'MEMBERSHIP_SHA', membership['sha256'])
        return ref, manifest, originals
    return prepare


def test_constructs_private_latest_three_features_and_value_free_complete_summary(retained_case, tmp_path, capsys):
    ref, manifest, originals = retained_case()
    from scripts.derive_retained_speed_features import main
    assert main(['--execute', '--manifest', ref['path'], '--manifest-sha256', ref['sha256'],
                 '--output', str(tmp_path/'out')]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['cards'] == 82 and result['target_runner_slots'] == 583
    assert result['supported_runner_slots'] == 583 and result['full_field_supported_cards'] == 82
    private = json.loads((tmp_path/'out/features.private.json').read_bytes())
    first = private['records'][0]
    runner = first['features']['runners'][0]
    assert runner['selected_dates'] == ['2026-09-28', '2026-09-27', '2026-09-26']
    assert runner['median_first_sectional'] == pytest.approx(6.2)
    assert runner['mad_first_sectional'] == pytest.approx(.2)
    assert runner['field_median_gap'] == pytest.approx(-.3)
    assert first['cutoff'] == '2026-10-01T19:58:30+10:00'  # Recorded, no invented T-120 requirement.
    assert first['retained_available_by'] == originals[0]['original_published_complete_at']
    assert first['retained_available_by'] != '2026-10-01T19:54:00+10:00'  # Page receipt is not CSV capture.
    summary = (tmp_path/'out/summary.json').read_text()
    for secret in ('median_first_sectional', 'mad_first_sectional', 'field_median_gap',
                   'Dog1', 'PRIVATE_UNUSED_TOTAL', 'SENSITIVE_PLACEMENT', 'SENSITIVE_ODDS'):
        assert secret not in summary
    public = json.loads(summary)
    assert len(public['records']) == 82 and public['reads'] == 904
    assert (tmp_path/'out/features.private.json').stat().st_mode & 0o777 == 0o600
    assert (tmp_path/'out/features.private.json').stat().st_nlink == 1
    assert (tmp_path/'out').stat().st_mode & 0o777 == 0o700
    assert all(r['race_id'] == m['race_id'] for r, m in zip(public['records'], manifest['members']))
    with pytest.raises(FileExistsError):
        speed.run(ref, tmp_path/'out')


def test_incomplete_field_keeps_individual_features_and_whole_denominator(retained_case, tmp_path):
    def missing(index, rows):
        if index == 0:
            rows[0]['1 SEC'] = '-'
            rows[1]['1 SEC'] = 'nan'
    ref, _, _ = retained_case(missing)
    result = speed.run(ref, tmp_path/'out')
    assert result['target_runner_slots'] == 583 and result['supported_runner_slots'] == 582
    assert result['full_field_supported_cards'] == 81
    private = json.loads((tmp_path/'out/features.private.json').read_bytes())
    first = private['records'][0]['features']
    assert first['runners'][0]['median_first_sectional'] is None
    assert first['runners'][1]['median_first_sectional'] == pytest.approx(6.3)
    assert all(r['field_median_gap'] is None for r in first['runners'])
    assert first['field_blocker'] == 'INCOMPLETE_ROSTER_SUPPORT'


@pytest.mark.parametrize('defect', ['tampered_card', 'read_limit', 'deadline', 'output_limit'])
def test_shared_failure_never_publishes_success(retained_case, tmp_path, monkeypatch, defect):
    ref, manifest, _ = retained_case()
    if defect == 'tampered_card':
        Path(manifest['members'][1]['accepted_csv']['path']).write_bytes(b'PRIVATE_BAD_CARD')
    elif defect == 'read_limit':
        monkeypatch.setattr(speed.coverage, 'MAX_READS', 15)
    elif defect == 'deadline':
        monkeypatch.setattr(speed.coverage, 'MAX_SECONDS', -1)
    else:
        monkeypatch.setattr(speed.coverage, 'MAX_OUTPUT', 1)
    with pytest.raises(ValueError):
        speed.run(ref, tmp_path/'out')
    assert not (tmp_path/'out/summary.json').exists()
    failed = json.loads((tmp_path/'out/FAILED.json').read_bytes())
    assert failed['status'] == 'FAILED_NO_SUCCESSFUL_SPEED_FEATURES'
    assert 'PRIVATE_BAD_CARD' not in json.dumps(failed)
    if defect in ('tampered_card', 'read_limit'):
        assert failed['completed_race_ids'] == [manifest['members'][0]['race_id']]
        assert failed['failed_race_id'] == manifest['members'][1]['race_id']
        assert failed['unattempted_race_ids'] == [m['race_id'] for m in manifest['members'][2:]]
    with pytest.raises(FileExistsError):
        speed.run(ref, tmp_path/'out')


def test_post_cutoff_capture_fails_before_features(retained_case, tmp_path):
    ref, manifest, originals = retained_case()
    original = originals[0]
    original['original_published_complete_at'] = '2026-10-01T19:59:00+10:00'
    with pytest.raises(ValueError, match='FEATURE_CAPTURE_NOT_BEFORE_CUTOFF'):
        speed.construct_member(speed.coverage.Reader(), manifest['members'][0], original)


def test_output_deadline_failure_leaves_no_success_summary(retained_case, tmp_path, monkeypatch):
    ref, _, _ = retained_case()
    original = speed.os.fsync
    def expire(fd):
        original(fd)
        monkeypatch.setattr(speed.coverage, 'MAX_SECONDS', -1)
    monkeypatch.setattr(speed.os, 'fsync', expire)
    with pytest.raises(ValueError, match='WALL_LIMIT'):
        speed.run(ref, tmp_path/'out')
    assert not (tmp_path/'out/summary.json').exists()
    assert (tmp_path/'out/FAILED.json').exists()


def test_conflicting_source_observations_exclude_date_before_recent_three_selection(retained_case, tmp_path):
    def conflict(index, rows):
        if index == 0:
            rows.insert(1, {**rows[0], 'Dog Name': '', 'TIME': 'DIFFERENT_HISTORICAL_TOTAL'})
    ref, _, _ = retained_case(conflict)
    result = speed.run(ref, tmp_path/'out')
    assert result['supported_runner_slots'] == 583
    private = json.loads((tmp_path/'out/features.private.json').read_bytes())
    runner = private['records'][0]['features']['runners'][0]
    assert runner['selected_dates'] == ['2026-09-27', '2026-09-26', '2026-09-25']
    assert runner['median_first_sectional'] == pytest.approx(6.4)
    assert sum(runner['exclusions'].values()) == 2
