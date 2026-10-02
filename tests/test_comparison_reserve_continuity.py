"""A reserve must be the same dog sealed before jump, not just the same box."""
from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import ingest_results_for_date as ingest
from scripts.autonomous_official_result_capture import comparison_runner_identity_error
from tests import test_capture_thedogs_market_history as fixture


@pytest.fixture
def sealed_case(tmp_path, request):
    observed = fixture.JUMP - timedelta(minutes=20)
    body = fixture.replacement_source_html(from_box_text="(into box 2)")
    body = body.replace(b'<td><sprite-svg', b'<td class="race-runners__box"><sprite-svg')
    for entry, dog in ((101, 501), (102, 502), (104, 504)):
        body = body.replace(f'data-runner-id="{entry}"'.encode(),
                            f'data-runner-id="{entry}" data-dog-id="{dog}"'.encode())
    if getattr(request, 'param', None) == 'missing_first_profile':
        body = body.replace(b' data-dog-id="501"', b'')
    evidence = fixture.subject.capture_native_identity_from_retained_race_page(
        session=fixture.FakeSession(server_time=observed, source_body=body,
                                    api_body=fixture.replacement_api_payload()),
        race_page=replace(fixture.retained_primary_race_page(observed), body=body),
        expected_active_runner_boxes=[("101", 1), ("104", 2)],
        expected_jump_utc=fixture.JUMP, current_time=observed + timedelta(seconds=1),
        clock=fixture.FakeClock(observed + timedelta(milliseconds=100)))
    metadata = {"native_identity_evidence": evidence, "source_native_race_id": "9001",
                "metadata_captured_at": evidence["odds_api_http"]["request_end_utc"]}
    relative = "source/invented.csv.metadata.json"
    root = tmp_path / "bundles"
    path = root / "sealed" / relative
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(metadata))
    bundle = SimpleNamespace(directory="sealed", result={"race": {
        "url": fixture.RACE_URL, "race_number": 1}}, manifest={"files": {relative: {
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}})
    job = SimpleNamespace(input=SimpleNamespace(jump_timestamp=fixture.JUMP.isoformat(),
        ordered_runners=[{"box": 1, "name": "Alpha", "source_native_runner_id": "101"},
                         {"box": 2, "name": "Reserve", "source_native_runner_id": "104"}]))
    return root, job, bundle, path


def official_row(box, name, dog, position):
    return (f'<tr class="race-runner"><td class="race-runners__finish-position">{position}</td>'
            f'<td class="race-runners__box"><sprite-svg name="rug_{box}"></sprite-svg></td>'
            f'<td class="race-runners__name"><a href="/dogs/{dog}/invented" '
            f'data-dog-id="{dog}">{name}</a></td></tr>')


def official(candidate, *, dog=504, note="(from box 2)", status="2nd"):
    markup = '<table class="race-runners--result">' + (
        official_row(1, "Alpha", 501, "1st") + official_row(2, "Beta", 502, "SCR") +
        official_row(9, "Reserve " + note, dog, status)) + '</table>'
    return ingest.TheDogsResultFetcher(None)._result_from_html(candidate, fixture.RACE_URL, markup)


def test_sealed_entry_to_dog_bridge_accepts_only_same_promoted_reserve(sealed_case):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, _ = sealed_case
    participants = frozen_result_participants(job, bundle, root)
    assert participants[1]['source_native_runner_id'] == '104'
    assert participants[1]['source_native_dog_id'] == '504'
    assert participants[1]['original_box_number'] == 9
    candidate = SimpleNamespace(participants=participants)
    assert comparison_runner_identity_error(candidate, official(candidate)) is None
    assert comparison_runner_identity_error(candidate, official(candidate, dog=999)) is not None


@pytest.mark.parametrize('missing', ['source_native_dog_id', 'original_box_number'])
def test_reserve_without_sealed_continuity_stays_quarantined(sealed_case, missing):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, _ = sealed_case
    participants = frozen_result_participants(job, bundle, root)
    participants[1].pop(missing)
    candidate = SimpleNamespace(participants=participants)
    assert comparison_runner_identity_error(candidate, official(candidate)) is not None


def test_changed_sealed_metadata_never_becomes_identity_evidence(sealed_case):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, path = sealed_case
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError):
        frozen_result_participants(job, bundle, root)


def test_profile_identity_does_not_make_incomplete_positions_complete(sealed_case):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, _ = sealed_case
    candidate = SimpleNamespace(participants=frozen_result_participants(job, bundle, root))
    selected = official(candidate, status='DNF')
    assert set(selected.positions_by_box) != {1, 2}


@pytest.mark.parametrize('mutation', ['entry', 'box', 'name', 'jump', 'source'])
def test_bridge_rejects_different_frozen_identity_or_provenance(sealed_case, mutation):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, _ = sealed_case
    if mutation == 'entry':
        job.input.ordered_runners[1]['source_native_runner_id'] = '504'
    elif mutation == 'box':
        job.input.ordered_runners[1]['box'] = 3
    elif mutation == 'name':
        job.input.ordered_runners[1]['name'] = 'Different Dog'
    elif mutation == 'jump':
        job.input.jump_timestamp = (fixture.JUMP + timedelta(minutes=1)).isoformat()
    else:
        bundle.result['race']['url'] = fixture.RACE_URL.replace('/1/', '/2/')
    with pytest.raises(ValueError):
        frozen_result_participants(job, bundle, root)


@pytest.mark.parametrize('mutation', ['duplicate_profile', 'missing_note', 'wrong_box_note'])
def test_official_ambiguity_cannot_use_a_valid_prejump_bridge(sealed_case, mutation):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, _ = sealed_case
    candidate = SimpleNamespace(participants=frozen_result_participants(job, bundle, root))
    kwargs = {'dog': 501} if mutation == 'duplicate_profile' else {
        'note': '' if mutation == 'missing_note' else '(from box 1)'}
    assert comparison_runner_identity_error(candidate, official(candidate, **kwargs)) is not None


def test_explicit_official_profile_conflict_never_downgrades_to_legacy_name_match():
    candidate = SimpleNamespace(participants=[{'box_number': 1, 'dog_name': 'Alpha',
                                              'source_native_dog_id': '501'}])
    markup = '<table class="race-runners--result">' + official_row(1, 'Alpha', 501, '1st') + '</table>'
    markup = markup.replace('data-dog-id="501"', 'data-dog-id="999"')
    result = ingest.TheDogsResultFetcher(None)._result_from_html(candidate, fixture.RACE_URL, markup)
    assert comparison_runner_identity_error(candidate, result) is not None


@pytest.mark.parametrize('sealed_case', ['missing_first_profile'], indirect=True)
def test_one_missing_legacy_profile_does_not_discard_other_proven_identities(sealed_case):
    from src.predictor.comparison_runner_identity import frozen_result_participants
    root, job, bundle, _ = sealed_case
    participants = frozen_result_participants(job, bundle, root)
    assert 'source_native_dog_id' not in participants[0]
    assert participants[1]['source_native_dog_id'] == '504'
    candidate = SimpleNamespace(participants=participants)
    assert comparison_runner_identity_error(candidate, official(candidate, dog=999)) is not None
