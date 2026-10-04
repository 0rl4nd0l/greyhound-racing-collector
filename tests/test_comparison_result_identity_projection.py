"""Fabricated sealed inputs and official markup; no providers or real results."""
from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest

from scripts import autonomous_official_result_capture as capture
from scripts import ingest_results_for_date as ingest
from src.predictor.comparison_runner_identity import frozen_result_participants
from tests.test_comparison_reserve_continuity import official, official_row, sealed_case
from tests import test_capture_thedogs_market_history as fixture


def candidate_for(sealed_case):
    root, job, bundle, _ = sealed_case
    participants = frozen_result_participants(job, bundle, root)
    candidate = SimpleNamespace(participants=participants, race_id='invented',
        venue='Invented', race_number=1, race_date=fixture.JUMP.date().isoformat(),
        race_time=None, start_datetime=fixture.JUMP.isoformat(),
        participant_source='verified_r3_prediction', canonical_thedogs_url=fixture.RACE_URL)
    return candidate


def test_live_dryrun_and_artifact_preserve_verified_bridge(sealed_case, monkeypatch):
    candidate = candidate_for(sealed_case)
    result = official(candidate)
    before = deepcopy((candidate, result))
    monkeypatch.setattr(ingest, 'winner_odds_for_box', lambda *args: None)
    report = ingest.write_result(None, candidate, result, [result], dry_run=True)
    rows = capture.build_artifact_rows({'ingested': [report]}, generated_at=fixture.JUMP+timedelta(minutes=20))
    assert [r['source_native_runner_id'] for r in rows['runner_rows']] == ['101', '104']
    assert [r['source_native_dog_id'] for r in rows['runner_rows']] == ['501', '504']
    assert rows['race_rows'][0]['native_identity_status'] == 'IDENTITY_VERIFIED'
    assert rows['runner_rows'][1]['native_identity_proof']['prejump']['original_box_number'] == 9
    assert candidate == before[0] and result == before[1]


def test_retained_reconciliation_preserves_the_same_bridge(sealed_case, monkeypatch):
    from scripts import reconcile_comparison_result_identity as reconcile
    root, job, bundle, _ = sealed_case
    job.input.race_id = 'invented'
    bundle.result['race'].update(race_id='invented', venue='Invented', race_date=fixture.JUMP.date().isoformat())
    markup = '<table class="race-runners--result">' + (
        official_row(1, 'Alpha', 501, '1st') + official_row(2, 'Beta', 502, 'SCR') +
        official_row(9, 'Reserve (from box 2)', 504, '2nd')) + '</table>'
    class Deadline:
        def check_deadline(self):
            return fixture.JUMP+timedelta(minutes=30)
        def call(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)
    # This test targets the producer/serializer seam, not the independent reader.
    monkeypatch.setattr(reconcile.ComparisonResultSource, '_validate', lambda *args: None)
    rows = reconcile.evidence_rows(job, bundle, markup.encode(), fixture.RACE_URL,
        fixture.JUMP+timedelta(minutes=20), fixture.JUMP+timedelta(minutes=30),
        deadline=Deadline(), prediction_bundles=root)
    assert [r['source_native_runner_id'] for r in rows['runner_rows']] == ['101', '104']
    assert all(r['native_identity_proof']['official_markup_sha256'] for r in rows['runner_rows'])


@pytest.mark.parametrize('mutation', [
    'forecast_only', 'missing_prejump_profile', 'missing_prejump_proof',
    'changed_prejump_proof', 'missing_official_profile', 'changed_official_profile',
    'duplicate_entry', 'duplicate_profile', 'duplicate_box', 'profile_conflict',
    'incomplete_source_rows', 'missing_markup_proof', 'bad_markup_proof',
    'wrong_result_url', 'mixed_prejump_evidence', 'unproved_reserve',
    'unknown_extra', 'changed_name', 'incomplete_result', 'other_source',
])
def test_unproven_identity_never_gets_forecast_ids(sealed_case, mutation):
    candidate = candidate_for(sealed_case)
    result = official(candidate)
    first = candidate.participants[0]
    if mutation == 'forecast_only':
        for member in candidate.participants:
            member.pop('source_native_dog_id')
            member.pop('native_identity_proof')
    elif mutation == 'missing_prejump_profile': first.pop('source_native_dog_id')
    elif mutation == 'missing_prejump_proof': first.pop('native_identity_proof')
    elif mutation == 'changed_prejump_proof': first['native_identity_proof']['source_native_runner_id'] = '999'
    elif mutation == 'missing_official_profile': result.dog_ids_by_box.pop(1)
    elif mutation == 'changed_official_profile': result.dog_ids_by_box[1] = '999'
    elif mutation == 'duplicate_entry': candidate.participants[1]['source_native_runner_id'] = '101'
    elif mutation == 'duplicate_profile': candidate.participants[1]['source_native_dog_id'] = '501'
    elif mutation == 'duplicate_box': candidate.participants[1]['box_number'] = 1
    elif mutation == 'profile_conflict': result.runner_profile_identity_conflict = True
    elif mutation == 'incomplete_source_rows': result.runner_identity_rows_complete = False
    elif mutation == 'missing_markup_proof': result.official_markup_sha256 = None
    elif mutation == 'bad_markup_proof': result.official_markup_sha256 = 'not-a-hash'
    elif mutation == 'wrong_result_url': result.source_url += '/different'
    elif mutation == 'mixed_prejump_evidence': first['native_identity_proof']['metadata_sha256'] = 'f'*64
    elif mutation == 'unproved_reserve': candidate.participants[1].pop('original_box_number')
    elif mutation == 'unknown_extra': result.dog_names_by_box[8] = 'Unexpected'
    elif mutation == 'changed_name': result.dog_names_by_box[1] = 'Different'
    elif mutation == 'incomplete_result': result.positions_by_box.pop(2)
    elif mutation == 'other_source': result.source = 'invented-other-source'
    before = deepcopy((candidate, result))
    projected = capture.comparison_native_identity_projection(candidate, result)
    assert projected['native_identity_status'] == 'IDENTITY_INCOMPLETE'
    assert projected['native_identity_reason']
    assert all('source_native_runner_id' not in row and 'source_native_dog_id' not in row
               and 'native_identity_proof' not in row for row in projected['positions'])
    assert candidate == before[0] and result == before[1]


def test_legacy_name_box_closure_keeps_explicit_incomplete_status(sealed_case, monkeypatch):
    candidate = candidate_for(sealed_case)
    result = official(candidate)
    for member in candidate.participants:
        member.pop('native_identity_proof')
    monkeypatch.setattr(ingest, 'winner_odds_for_box', lambda *args: None)
    report = ingest.write_result(None, candidate, result, [result], dry_run=True)
    rows = capture.build_artifact_rows({'ingested': [report]}, generated_at=fixture.JUMP+timedelta(minutes=20))
    assert rows['race_rows'][0]['native_identity_status'] == 'IDENTITY_INCOMPLETE'
    assert all(row['native_identity_status'] == 'IDENTITY_INCOMPLETE' and
               'source_native_runner_id' not in row for row in rows['runner_rows'])


def test_plain_serialized_native_ids_cannot_create_a_bridge(sealed_case):
    item = {'source': 'thedogs_official', 'status': 'resulted',
            'positions': [{'box_number': 1, 'dog_name': 'Alpha', 'finish_position': 1,
                           'source_native_runner_id': '101', 'source_native_dog_id': '501'}]}
    rows = capture.build_artifact_rows({'ingested': [item]}, generated_at=fixture.JUMP)
    assert rows['runner_rows'][0]['native_identity_status'] == 'IDENTITY_INCOMPLETE'
    assert 'source_native_runner_id' not in rows['runner_rows'][0]


@pytest.mark.parametrize('mutation', ['missing_proof', 'changed_entry', 'changed_profile',
                                    'changed_proof', 'changed_url'])
def test_modified_serialized_bridge_is_explicitly_incomplete(sealed_case, monkeypatch, mutation):
    candidate = candidate_for(sealed_case)
    result = official(candidate)
    monkeypatch.setattr(ingest, 'winner_odds_for_box', lambda *args: None)
    report = ingest.write_result(None, candidate, result, [result], dry_run=True)
    position = report['positions'][0]
    if mutation == 'missing_proof': position.pop('native_identity_proof')
    elif mutation == 'changed_entry': position['source_native_runner_id'] = '999'
    elif mutation == 'changed_profile': position['source_native_dog_id'] = '999'
    elif mutation == 'changed_proof': position['native_identity_proof']['official_markup_sha256'] = ''
    elif mutation == 'changed_url': report['source_url'] += '/different'
    rows = capture.build_artifact_rows({'ingested': [report]}, generated_at=fixture.JUMP)
    assert rows['race_rows'][0]['native_identity_status'] == 'IDENTITY_INCOMPLETE'
    assert rows['race_rows'][0]['native_identity_reason'] == 'SERIALIZED_BRIDGE_PROOF_MISSING_OR_CHANGED'
    assert all('source_native_runner_id' not in row for row in rows['runner_rows'])
