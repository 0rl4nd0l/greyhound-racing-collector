from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import scripts.predict_market_form_residual as manual
from scripts.predict_market_form_residual import (
    ManualPredictionError,
    _canonical_target_grade,
    _trusted_sportsbet_url,
    _trusted_thedogs_url,
    build_parser,
    score_from_artifacts,
)
from src.predictor.market_form_residual import FEATURES


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/frozen_models/market_form_residual_v1"
RACE_ID = "Race 2 - SAN - 2026-07-16"
MELBOURNE = ZoneInfo("Australia/Melbourne")
SCORE_TIME = datetime(2026, 7, 16, 18, 52, tzinfo=MELBOURNE)
SOURCE_URL = "https://www.thedogs.com.au/racing/sandown/2026-07-16/2/test"

RUNNERS = (
    (1, "Alpha Fast", "ALPHAFAST", 2.5),
    (2, "Beta Bale", "BETABALE", 3.2),
    (4, "Gamma Rule", "GAMMARULE", 5.0),
)

GOLDEN_FEATURES = {
    1: dict(
        zip(
            FEATURES,
            (2, 7, 2.0, 1.0, 0.5, 1.0, 1.25, 0.5, 1.0, 2.0, 2, 0.5, 2, 0.5, 2, 0.5),
        )
    ),
    2: dict(
        zip(
            FEATURES,
            (
                8,
                8,
                3.0,
                2.0,
                0.125,
                0.375,
                2.0,
                0.125,
                0.375,
                3.5,
                4,
                0.25,
                6,
                1 / 6,
                8,
                0.125,
            ),
        )
    ),
    4: dict(
        zip(
            FEATURES,
            (
                0,
                None,
                None,
                None,
                0.0,
                0.0,
                None,
                0.0,
                0.0,
                None,
                0,
                0.0,
                0,
                0.0,
                0,
                0.0,
            ),
        )
    ),
}


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_bytes())


def _seal_packet(paths: dict[str, Path], rows: list[dict], manifest: dict) -> None:
    _write_json(paths["feature_rows"], rows)
    _write_json(paths["feature_manifest"], manifest)
    artifacts = {}
    for key in ("feature_rows", "feature_manifest"):
        raw = paths[key].read_bytes()
        artifacts[str(paths[key].resolve())] = {
            "bytes": len(raw),
            "sha256": _sha256(raw),
        }
    _write_json(
        paths["implementation_manifest"],
        {
            "schema_version": "shadow_implementation_file_manifest_v1",
            "git_branch": "codex/residual-handoff-under-test",
            "git_head": "feedfacefeed",
            "implementation_files": manual.FEATURE_GENERATOR_FILES,
            "implementation_file_hashes": {
                relative: _sha256((ROOT / relative).read_bytes())
                for relative in manual.FEATURE_GENERATOR_FILES
            },
            "output_dir": str(paths["feature_rows"].parent.resolve()),
            "artifact_files": artifacts,
        },
    )


def _feature_row(box: int, name: str) -> dict:
    return {
        "race_id": RACE_ID,
        "box_number": box,
        "dog_name": name,
        "race_date": "2026-07-16",
        "race_number": 2,
        "venue": "SAN",
        "source_csv": "filled-by-fixture",
        "metadata_is_leakage_safe": True,
        "target_metadata_from_sidecar": True,
        "target_metadata_rejected_sources": [],
        "target_metadata_source_url": SOURCE_URL,
        "target_distance_source_is_safe": 1,
        "target_grade_provenance_safe": 1,
        "target_distance_missing": 0,
        "target_grade_missing": 0,
        "target_distance_safe": 515.0,
        "target_grade_safe": "Grade 5",
        "same_distance_same_grade_history_cutoff": "strictly_before_target_race",
        "same_distance_same_grade_history_cutoff_basis": (
            "race_date_less_than_target_race_date"
        ),
        "same_distance_same_grade_target_race_rows_used": 0,
        "same_distance_same_grade_post_outcome_rows_used": 0,
        "same_distance_same_grade_post_outcome_fields_used": [],
        **GOLDEN_FEATURES[box],
    }


def _write_fixture(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    form_csv = tmp_path / f"{RACE_ID}.csv"
    form_csv.write_text(
        "Dog Name|PLC|DATE\n"
        "1. Alpha Fast|1|2026-07-09\n"
        "2. Beta Bale|2|2026-07-08\n"
        "4. Gamma Rule|3|2026-07-07\n",
        encoding="utf-8",
    )
    form_raw = form_csv.read_bytes()
    sidecar = form_csv.with_name(form_csv.name + ".metadata.json")
    participants = [
        {"box_number": box, "dog_name": name} for box, name, _, _ in RUNNERS
    ]
    _write_json(
        sidecar,
        {
            "schema_version": "form_guide_download_provenance_v1",
            "filename": form_csv.name,
            "content_length": len(form_raw),
            "content_sha256": _sha256(form_raw),
            "metadata_is_leakage_safe": True,
            "race_url": SOURCE_URL,
            "canonical_runner_alignment": {"status": "aligned"},
            "runner_completeness": {
                "status": "COMPLETE",
                "runner_count": 3,
                "participants": participants,
            },
            "prejump_shadow_metadata": {
                "status": "PASS",
                "metadata_is_leakage_safe": True,
                "metadata_captured_at": "2026-07-16T18:40:00+10:00",
                "race_date": "2026-07-16",
                "race_number": 2,
                "venue": "SAN",
                "distance": "515m",
                "grade": "Grade 5",
                "jump_datetime": "2026-07-16T18:58:00+10:00",
                "source_url": SOURCE_URL,
                "canonical_final_runner_alignment": {"status": "aligned"},
                "runner_box_name_list": participants,
            },
        },
    )

    packet = tmp_path / "shadow_score_live"
    paths = {
        "form_csv": form_csv,
        "sidecar": sidecar,
        "feature_rows": packet / "shadow_feature_rows.json",
        "feature_manifest": packet / "shadow_manifest.json",
        "implementation_manifest": packet / "implementation_file_manifest.json",
        "capture": tmp_path / "autonomous_live_odds_capture_report.json",
    }
    rows = [_feature_row(box, name) for box, name, _, _ in RUNNERS]
    for row in rows:
        row["source_csv"] = str(form_csv.resolve())
    feature_manifest = {
        "schema_version": "shadow_live_scoring_manifest_v1",
        "generated_at": "2026-07-16T18:46:00+10:00",
        "feature_freeze_timestamp": "2026-07-16T18:45:00+10:00",
        "output_mode": "shadow_only",
        "betting_output": False,
        "ev_output": False,
        "odds_used_for_shadow_scoring": False,
        "production_prediction_write": False,
        "registry_mutation": False,
        "tgr_enabled": False,
        "feature_rows": str(paths["feature_rows"].resolve()),
        "input_files": [str(form_csv.resolve())],
    }
    _seal_packet(paths, rows, feature_manifest)

    accepted_rows = [
        {
            "box_number": box,
            "dog_name": name,
            "identity": identity,
            "odds_decimal": odds,
            "sportsbet_box_source": "runner_text",
        }
        for box, name, identity, odds in RUNNERS
    ]
    _write_json(
        paths["capture"],
        {
            "schema_version": "autonomous_live_odds_capture_report_v1",
            "attempts": [
                {
                    "schema_version": "autonomous_live_odds_capture_attempt_v1",
                    "race_id": RACE_ID,
                    "status": "APPENDED",
                    "fetch_time": "2026-07-16T18:50:00+10:00",
                    "append_time": "2026-07-16T18:51:00+10:00",
                    "reasons": [],
                    "validation": {
                        "schema_version": (
                            "autonomous_live_odds_capture_validation_v1"
                        ),
                        "status": "PASS",
                        "source_url": (
                            "https://www.sportsbet.com.au/greyhound-racing/"
                            "australia-nz/sandown-park/race-2-123"
                        ),
                        "source_race_number": 2,
                        "source_race_date": "2026-07-16",
                        "source_venue": "SAN",
                        "accepted_rows": accepted_rows,
                        "accepted_row_count": 3,
                        "rejected_rows": [],
                        "expected_runner_count": 3,
                        "active_expected_runner_count": 3,
                        "scratched_expected_runner_count": 0,
                        "scratched_expected_runners": [],
                        "scratched_expected_runners_with_odds": [],
                        "missing_expected_runners": [],
                        "extra_unexpected_runners": [],
                        "failure_root_cause": None,
                        "reasons": [],
                    },
                }
            ],
        },
    )
    return paths


def _reseal(paths: dict[str, Path], mutate) -> None:
    rows = _json(paths["feature_rows"])
    manifest = _json(paths["feature_manifest"])
    assert isinstance(rows, list) and isinstance(manifest, dict)
    mutate(rows, manifest)
    _seal_packet(paths, rows, manifest)


def _score_paths(paths: dict[str, Path], **overrides):
    arguments = {
        "race_id": RACE_ID,
        "form_csv_path": paths["form_csv"],
        "sidecar_path": paths["sidecar"],
        "feature_rows_path": paths["feature_rows"],
        "feature_manifest_path": paths["feature_manifest"],
        "implementation_manifest_path": paths["implementation_manifest"],
        "capture_path": paths["capture"],
        "model_path": ARTIFACT_DIR / "model.json",
        "manifest_path": ARTIFACT_DIR / "manifest.json",
        "score_timestamp": SCORE_TIME,
    }
    arguments.update(overrides)
    return score_from_artifacts(**arguments)


def _score(tmp_path: Path):
    return _score_paths(_write_fixture(tmp_path))


def test_scores_exact_packet_deterministically(tmp_path):
    paths = _write_fixture(tmp_path)
    first = _score_paths(paths)
    second = _score_paths(paths)

    assert first == second
    assert first["schema_version"] == "manual_market_form_residual_prediction_v2"
    assert first["status"] == "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION"
    assert first["source_contract"] == {
        "feature_source": "exact_hash_bound_system_shadow_feature_rows",
        "feature_reconstruction_performed": False,
        "database_access": False,
        "network_access": False,
    }
    assert first["activation"] is False
    assert first["persisted"] is False
    assert first["outcomes_present"] is False
    assert len(first["predictions"]) == 3
    for key in ("market", "half", "full"):
        assert first["probability_sums"][key] == pytest.approx(1.0)


def test_explicit_shadow_output_appends_once_and_replays_idempotently(tmp_path):
    paths = _write_fixture(tmp_path / "fixture")
    shadow_output = tmp_path / "market_form_residual_shadow_predictions_v3.jsonl"

    first = _score_paths(paths, shadow_output_path=shadow_output)
    original = shadow_output.read_bytes()
    second = _score_paths(paths, shadow_output_path=shadow_output)

    assert first["persisted"] is True
    assert first["persistence_status"] == "APPENDED"
    assert second["persisted"] is True
    assert second["persistence_status"] == "EXACT_REPLAY"
    assert shadow_output.read_bytes() == original
    records = [json.loads(line) for line in original.splitlines()]
    assert len(records) == 1
    assert records[0]["race_id"] == RACE_ID
    assert records[0]["outcomes_present"] is False
    assert records[0]["activation"] is False


def test_scores_when_strict_odds_capture_precedes_feature_generation(tmp_path):
    paths = _write_fixture(tmp_path)
    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    capture["attempts"][0]["fetch_time"] = "2026-07-16T18:42:00+10:00"
    capture["attempts"][0]["append_time"] = "2026-07-16T18:43:00+10:00"
    _write_json(paths["capture"], capture)

    output = _score_paths(paths)

    assert output["odds_append_timestamp"] == "2026-07-16T18:43:00+10:00"
    assert output["feature_manifest_generated_at"] == "2026-07-16T18:46:00+10:00"


def test_uses_exact_top_level_golden_features(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path)
    real_score_race = manual.score_race
    observed = {}

    def spy(frozen, runners, provenance):
        observed.update({int(row["box_number"]): row["features"] for row in runners})
        return real_score_race(frozen, runners, provenance)

    monkeypatch.setattr(manual, "score_race", spy)
    _score_paths(paths)

    assert observed == GOLDEN_FEATURES


def test_reads_each_mutable_input_once_and_hashes_same_bytes(tmp_path, monkeypatch):
    paths = _write_fixture(tmp_path)
    watched = {path.resolve() for path in paths.values()}
    counts = {path: 0 for path in watched}
    original = Path.read_bytes

    def counted(path):
        resolved = path.resolve()
        if resolved in counts:
            counts[resolved] += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", counted)
    output = _score_paths(paths)

    assert counts == {path: 1 for path in watched}
    for key, path_key in (
        ("form_csv_sha256", "form_csv"),
        ("sidecar_sha256", "sidecar"),
        ("feature_rows_sha256", "feature_rows"),
        ("feature_manifest_sha256", "feature_manifest"),
        ("implementation_manifest_sha256", "implementation_manifest"),
        ("capture_artifact_sha256", "capture"),
    ):
        assert output["input_hashes"][key] == _sha256(original(paths[path_key]))


def test_rejects_packet_hash_mismatch(tmp_path):
    paths = _write_fixture(tmp_path)
    paths["feature_rows"].write_bytes(paths["feature_rows"].read_bytes() + b" ")

    with pytest.raises(
        ManualPredictionError, match="feature_rows_manifest_hash_mismatch"
    ):
        _score_paths(paths)


def test_rejects_form_csv_not_bound_by_sidecar(tmp_path):
    paths = _write_fixture(tmp_path)
    paths["form_csv"].write_bytes(paths["form_csv"].read_bytes() + b"tampered\n")

    with pytest.raises(ManualPredictionError, match="form_csv_sidecar_hash_mismatch"):
        _score_paths(paths)


def test_rejects_date_with_trailing_content(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["race_date"] = "2026-07-16garbage"
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_race_date_invalid"):
        _score_paths(paths)


def test_rejects_box_outside_greyhound_range_even_when_sources_agree(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["runner_box_name_list"][0]["box_number"] = 9
    sidecar["runner_completeness"]["participants"][0]["box_number"] = 9
    _write_json(paths["sidecar"], sidecar)

    _reseal(paths, lambda rows, _: rows[0].__setitem__("box_number", 9))
    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    capture["attempts"][0]["validation"]["accepted_rows"][0]["box_number"] = 9
    _write_json(paths["capture"], capture)

    with pytest.raises(ManualPredictionError, match="sidecar_runner_box_invalid"):
        _score_paths(paths)


@pytest.mark.parametrize("feature", [FEATURES[0], FEATURES[-1]])
def test_rejects_missing_feature(tmp_path, feature):
    paths = _write_fixture(tmp_path)
    _reseal(paths, lambda rows, _: rows[0].pop(feature))

    with pytest.raises(ManualPredictionError, match=f"feature_value_missing:{feature}"):
        _score_paths(paths)


def test_rejects_boolean_feature_value(tmp_path):
    paths = _write_fixture(tmp_path)
    _reseal(paths, lambda rows, _: rows[0].__setitem__(FEATURES[0], True))

    with pytest.raises(
        ManualPredictionError, match=f"feature_value:{FEATURES[0]}_invalid"
    ):
        _score_paths(paths)


def test_rejects_nested_or_extra_feature_bundle(tmp_path):
    paths = _write_fixture(tmp_path)
    _reseal(paths, lambda rows, _: rows[0].__setitem__("features", {"extra": 1}))

    with pytest.raises(ManualPredictionError, match="feature.*top_level.*exact"):
        _score_paths(paths)


@pytest.mark.parametrize(
    "outcome_key",
    [
        "is_winner",
        "is_placer",
        "official_finish_position",
        "db_finish_position",
        "scraped_finish_position",
        "target_finish_position",
        "result_position",
        "official_position",
        "actual_winner",
        "official_winner",
        "result_status",
        "results_status",
        "official_result_status",
    ],
)
def test_rejects_supported_outcome_fields_in_feature_rows(tmp_path, outcome_key):
    paths = _write_fixture(tmp_path)
    _reseal(paths, lambda rows, _: rows[0].__setitem__(outcome_key, 1))

    with pytest.raises(
        ManualPredictionError, match="feature_rows_invalid_or_contains_outcome"
    ):
        _score_paths(paths)


@pytest.mark.parametrize(
    ("artifact_key", "error"),
    [
        ("sidecar", "sidecar_contains_outcome_field"),
        ("capture", "capture_contains_outcome_field"),
    ],
)
def test_rejects_is_winner_across_input_artifacts(tmp_path, artifact_key, error):
    paths = _write_fixture(tmp_path)
    artifact = _json(paths[artifact_key])
    assert isinstance(artifact, dict)
    artifact["is_winner"] = 1
    _write_json(paths[artifact_key], artifact)

    with pytest.raises(ManualPredictionError, match=error):
        _score_paths(paths)


def test_rejects_outcome_field_in_feature_manifest(tmp_path):
    paths = _write_fixture(tmp_path)
    _reseal(paths, lambda _, manifest: manifest.__setitem__("winner", "Alpha Fast"))

    with pytest.raises(
        ManualPredictionError, match="feature_manifest_contains_outcome_field"
    ):
        _score_paths(paths)


def test_rejects_outcome_field_in_implementation_manifest(tmp_path):
    paths = _write_fixture(tmp_path)
    manifest = _json(paths["implementation_manifest"])
    manifest["result"] = "post-race"
    _write_json(paths["implementation_manifest"], manifest)

    with pytest.raises(
        ManualPredictionError, match="implementation_manifest_contains_outcome_field"
    ):
        _score_paths(paths)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda rows, manifest: rows[0].__setitem__(
                "source_csv", "/tmp/not-the-form.csv"
            ),
            "feature_row_source_csv_mismatch",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__("dog_name", "Wrong Dog"),
            "feature_runner_set_mismatch_or_duplicate",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__("box_number", 8),
            "feature_runner_set_mismatch_or_duplicate",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__(
                "metadata_is_leakage_safe", False
            ),
            "feature_row_metadata_not_safe",
        ),
        (
            lambda rows, manifest: rows[0].__setitem__(
                "same_distance_same_grade_post_outcome_rows_used", 1
            ),
            "feature_post_outcome_rows_used",
        ),
        (
            lambda rows, manifest: manifest.__setitem__("betting_output", True),
            "feature_manifest_betting_output_mismatch",
        ),
        (
            lambda rows, manifest: manifest.__setitem__(
                "production_prediction_write", True
            ),
            "feature_manifest_production_prediction_write_mismatch",
        ),
    ],
)
def test_rejects_unsafe_or_mismatched_packet(tmp_path, mutation, error):
    paths = _write_fixture(tmp_path)
    _reseal(paths, mutation)

    with pytest.raises(ManualPredictionError, match=error):
        _score_paths(paths)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.thedogs.com.au/racing/sandown/2026-07-16/2/result",
        "https://www.thedogs.com.au/racing/sandown/2026-07-16/2/results",
        "https://www.thedogs.com.au/racing/sandown/2026-07-16/2/test?tab=dividend",
        "https://www.thedogs.com.au/racing/sandown/2026-07-16/2/test?view=payout",
    ],
)
def test_rejects_post_race_thedogs_urls(url):
    assert _trusted_thedogs_url(url) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://thedogs.com.au/racing/sandown/2026-07-16/2/test",
        "https://evil.thedogs.com.au/racing/sandown/2026-07-16/2/test",
        "https://user@www.thedogs.com.au/racing/sandown/2026-07-16/2/test",
    ],
)
def test_rejects_non_exact_thedogs_authority(url):
    assert _trusted_thedogs_url(url) is False


def test_sportsbet_binding_accepts_hyphen_venue_and_rejects_authority_tricks():
    race_id = "Race 2 - SANDOWN-PARK - 2026-07-16"
    valid = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
        "sandown-park/race-2-123"
    )
    assert _trusted_sportsbet_url(valid, race_id, "SANDOWN-PARK") is True
    assert (
        _trusted_sportsbet_url(
            valid.replace("www.sportsbet.com.au", "evil.sportsbet.com.au"),
            race_id,
            "SANDOWN-PARK",
        )
        is False
    )
    assert (
        _trusted_sportsbet_url(
            valid.replace("www.sportsbet.com.au", "user@www.sportsbet.com.au"),
            race_id,
            "SANDOWN-PARK",
        )
        is False
    )


def test_grade_aliases_are_finite_and_do_not_parse_sponsor_substrings():
    assert _canonical_target_grade("5th Grade") == "GRADE5"
    assert _canonical_target_grade("Grade 5") == "GRADE5"
    assert _canonical_target_grade("6th") == "GRADE6"
    assert _canonical_target_grade("Grade 6") == "GRADE6"
    assert _canonical_target_grade("Sponsor Grade 5 Trophy") is None


def test_scores_strict_ordinal_grade_alias(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    sidecar["prejump_shadow_metadata"]["grade"] = "5th Grade"
    _write_json(paths["sidecar"], sidecar)

    assert _score_paths(paths)["race_id"] == RACE_ID


def test_rejects_sponsor_grade_substring(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    sidecar["prejump_shadow_metadata"]["grade"] = "Sponsor Grade 5 Trophy"
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(
        ManualPredictionError, match="target_grade_not_exact_supported_alias"
    ):
        _score_paths(paths)


def test_rejects_result_url_in_bound_sidecar(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["source_url"] = SOURCE_URL + "/result"
    sidecar["race_url"] = SOURCE_URL + "/result"
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="sidecar_source_url_not_trusted"):
        _score_paths(paths)


def test_rejects_contradictory_sidecar_source_url_alias(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["race_url"] = SOURCE_URL + "/result"
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(
        ManualPredictionError, match="sidecar_source_url_alias_mismatch"
    ):
        _score_paths(paths)


def test_rejects_wrong_feature_generator_head(tmp_path):
    paths = _write_fixture(tmp_path)
    manifest = _json(paths["implementation_manifest"])
    assert isinstance(manifest, dict)
    manifest["git_branch"] = "codex/greyhound-resource-isolation-20260716"
    manifest["git_head"] = "deadbeefdead"
    manifest.pop("implementation_file_hashes")
    _write_json(paths["implementation_manifest"], manifest)

    with pytest.raises(ManualPredictionError, match="feature_generator_head_mismatch"):
        _score_paths(paths)


def test_rejects_packet_impersonating_legacy_generator_identity(tmp_path):
    paths = _write_fixture(tmp_path)
    manifest = _json(paths["implementation_manifest"])
    assert isinstance(manifest, dict)
    manifest["git_branch"] = "codex/greyhound-resource-isolation-20260716"
    manifest["git_head"] = "aa35fa70fc49"
    manifest.pop("implementation_file_hashes")
    _write_json(paths["implementation_manifest"], manifest)

    with pytest.raises(
        ManualPredictionError, match="feature_generator_legacy_packet_hash_mismatch"
    ):
        _score_paths(paths)


def _bind_current_generator_hashes(paths: dict[str, Path]) -> dict:
    manifest = _json(paths["implementation_manifest"])
    assert isinstance(manifest, dict)
    manifest["git_branch"] = "codex/residual-handoff-under-test"
    manifest["git_head"] = "feedfacefeed"
    manifest["implementation_file_hashes"] = {
        relative: _sha256((ROOT / relative).read_bytes())
        for relative in manifest["implementation_files"]
    }
    _write_json(paths["implementation_manifest"], manifest)
    return manifest


def test_accepts_new_packet_bound_to_current_generator_hashes(tmp_path):
    paths = _write_fixture(tmp_path)
    _bind_current_generator_hashes(paths)

    output = _score_paths(paths)

    assert output["status"] == "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION"


def test_rejects_new_packet_with_generator_hash_mismatch(tmp_path):
    paths = _write_fixture(tmp_path)
    manifest = _bind_current_generator_hashes(paths)
    manifest["implementation_file_hashes"][manifest["implementation_files"][0]] = (
        "0" * 64
    )
    _write_json(paths["implementation_manifest"], manifest)

    with pytest.raises(
        ManualPredictionError, match="feature_generator_implementation_hash_mismatch"
    ):
        _score_paths(paths)


@pytest.mark.parametrize(
    ("append_value", "score_time", "error"),
    [
        (None, SCORE_TIME, "capture_append_timestamp_missing"),
        (
            "2026-07-16T18:49:00+10:00",
            SCORE_TIME,
            "capture_append_before_fetch",
        ),
        (
            "2026-07-16T18:51:00+10:00",
            datetime(2026, 7, 16, 18, 58, tzinfo=MELBOURNE),
            "manual_score_not_prejump",
        ),
    ],
)
def test_rejects_missing_reversed_or_postjump_timestamps(
    tmp_path, append_value, score_time, error
):
    paths = _write_fixture(tmp_path)
    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    attempt = capture["attempts"][0]
    if append_value is None:
        attempt.pop("append_time")
    else:
        attempt["append_time"] = append_value
    _write_json(paths["capture"], capture)

    with pytest.raises(ManualPredictionError, match=error):
        _score_paths(paths, score_timestamp=score_time)


def test_rejects_ambiguous_accepted_attempts(tmp_path):
    paths = _write_fixture(tmp_path)
    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    capture["attempts"].append(copy.deepcopy(capture["attempts"][0]))
    _write_json(paths["capture"], capture)

    with pytest.raises(
        ManualPredictionError, match="accepted_capture_attempt_ambiguous"
    ):
        _score_paths(paths)


def test_scores_supported_scratched_runner(tmp_path):
    paths = _write_fixture(tmp_path)
    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    validation = capture["attempts"][0]["validation"]
    validation["accepted_rows"] = [
        row for row in validation["accepted_rows"] if row["box_number"] != 2
    ]
    validation["accepted_row_count"] = 2
    validation["active_expected_runner_count"] = 2
    validation["scratched_expected_runner_count"] = 1
    validation["scratched_expected_runners"] = [
        {"box_number": 2, "dog_name": "Beta Bale"}
    ]
    _write_json(paths["capture"], capture)

    output = _score_paths(paths)

    assert {row["box"] for row in output["predictions"]} == {1, 4}
    for key in ("market", "half", "full"):
        assert output["probability_sums"][key] == pytest.approx(1.0)


def test_runner_reordering_is_semantically_equivalent(tmp_path):
    paths = _write_fixture(tmp_path)
    first = _score_paths(paths)
    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    capture["attempts"][0]["validation"]["accepted_rows"].reverse()
    _write_json(paths["capture"], capture)
    _reseal(paths, lambda rows, _: rows.reverse())

    second = _score_paths(paths)

    assert first["predictions"] == second["predictions"]
    assert first["probability_sums"] == second["probability_sums"]
    assert first["runner_set_sha256"] == second["runner_set_sha256"]


def test_cli_has_no_model_or_manifest_override():
    destinations = {action.dest for action in build_parser()._actions}

    assert "model" not in destinations
    assert "model_path" not in destinations
    assert "manifest" not in destinations
    assert "manifest_path" not in destinations


def _retime_for_cli(paths: dict[str, Path]) -> str:
    now = datetime.now(MELBOURNE).replace(microsecond=0)
    metadata = now - timedelta(minutes=6)
    freeze = now - timedelta(minutes=5)
    generated = now - timedelta(minutes=4)
    fetch = now - timedelta(minutes=3)
    append = now - timedelta(minutes=2)
    jump = now + timedelta(hours=1)
    race_id = f"Race 2 - SAN - {jump.date().isoformat()}"

    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    shadow = sidecar["prejump_shadow_metadata"]
    shadow["race_date"] = jump.date().isoformat()
    shadow["jump_datetime"] = jump.isoformat()
    shadow["metadata_captured_at"] = metadata.isoformat()
    retimed_source_url = SOURCE_URL.replace("2026-07-16", jump.date().isoformat())
    shadow["source_url"] = retimed_source_url
    sidecar["race_url"] = retimed_source_url
    _write_json(paths["sidecar"], sidecar)

    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    attempt = capture["attempts"][0]
    attempt["race_id"] = race_id
    attempt["fetch_time"] = fetch.isoformat()
    attempt["append_time"] = append.isoformat()
    attempt["validation"]["source_race_date"] = jump.date().isoformat()
    _write_json(paths["capture"], capture)

    def retime(rows, manifest):
        manifest["feature_freeze_timestamp"] = freeze.isoformat()
        manifest["generated_at"] = generated.isoformat()
        for row in rows:
            row["race_id"] = race_id
            row["race_date"] = jump.date().isoformat()
            row["target_metadata_source_url"] = retimed_source_url

    _reseal(paths, retime)
    return race_id


def test_cli_prints_canonical_json_and_writes_nothing(tmp_path):
    paths = _write_fixture(tmp_path)
    race_id = _retime_for_cli(paths)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/predict_market_form_residual.py"),
            "--race-id",
            race_id,
            "--form-csv",
            str(paths["form_csv"]),
            "--feature-rows",
            str(paths["feature_rows"]),
            "--capture",
            str(paths["capture"]),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert completed.stderr == b""
    payload = json.loads(completed.stdout)
    assert completed.stdout == _canonical_bytes(payload)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_cli_explicit_shadow_output_persists_v3_record(tmp_path):
    paths = _write_fixture(tmp_path / "fixture")
    race_id = _retime_for_cli(paths)
    shadow_output = tmp_path / "market_form_residual_shadow_predictions_v3.jsonl"
    command = [
        sys.executable,
        str(ROOT / "scripts/predict_market_form_residual.py"),
        "--race-id",
        race_id,
        "--form-csv",
        str(paths["form_csv"]),
        "--feature-rows",
        str(paths["feature_rows"]),
        "--capture",
        str(paths["capture"]),
        "--append-shadow-output",
        str(shadow_output),
    ]

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert completed.stderr == b""
    output = json.loads(completed.stdout)
    assert completed.stdout == _canonical_bytes(output)
    assert output["persisted"] is True
    assert output["persistence_status"] == "APPENDED"
    assert output["shadow_output_path"] == str(shadow_output.resolve())

    replayed = subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert replayed.returncode == 0, replayed.stderr.decode("utf-8")
    assert replayed.stderr == b""
    replayed_output = json.loads(replayed.stdout)
    assert replayed.stdout == _canonical_bytes(replayed_output)
    assert replayed_output["persistence_status"] == "EXACT_REPLAY"
    assert replayed_output["record_key"] == output["record_key"]
    assert replayed_output["score_timestamp"] == output["score_timestamp"]

    records = [json.loads(line) for line in shadow_output.read_bytes().splitlines()]
    assert len(records) == 1
    assert records[0]["schema_version"] == "market_form_residual_shadow_record_v3"
    assert records[0]["race_id"] == race_id
    assert records[0]["outcomes_present"] is False
    assert records[0]["activation"] is False


def test_cli_explicit_shadow_output_rejects_changed_input_as_conflict(tmp_path):
    paths = _write_fixture(tmp_path / "fixture")
    race_id = _retime_for_cli(paths)
    shadow_output = tmp_path / "market_form_residual_shadow_predictions_v3.jsonl"
    command = [
        sys.executable,
        str(ROOT / "scripts/predict_market_form_residual.py"),
        "--race-id",
        race_id,
        "--form-csv",
        str(paths["form_csv"]),
        "--feature-rows",
        str(paths["feature_rows"]),
        "--capture",
        str(paths["capture"]),
        "--append-shadow-output",
        str(shadow_output),
    ]

    first = subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert first.returncode == 0, first.stderr.decode("utf-8")
    original = shadow_output.read_bytes()

    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    capture["attempts"][0]["validation"]["accepted_rows"][0]["odds_decimal"] = 2.6
    _write_json(paths["capture"], capture)

    conflicting = subprocess.run(
        command,
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )
    assert conflicting.returncode == 2
    assert conflicting.stdout == b""
    assert json.loads(conflicting.stderr) == {
        "status": "BLOCKED_MANUAL_PREDICTION",
        "reason": "conflicting_shadow_duplicate",
    }
    assert shadow_output.read_bytes() == original


def test_race_first_cli_discovers_exact_packet_with_one_query_and_writes_nothing(
    tmp_path,
):
    paths = _write_fixture(tmp_path)
    race_id = _retime_for_cli(paths)
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/predict_market_form_residual.py"),
            "--race",
            "sundown r2",
            "--evidence-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert completed.stderr == b""
    payload = json.loads(completed.stdout)
    assert payload["race_id"] == race_id
    assert completed.stdout == _canonical_bytes(payload)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_race_first_discovery_fails_closed_on_equal_latest_capture_reports(tmp_path):
    paths = _write_fixture(tmp_path)
    duplicate = tmp_path / "duplicate/autonomous_live_odds_capture_report.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes(paths["capture"].read_bytes())

    with pytest.raises(ManualPredictionError, match="race_capture_report_ambiguous"):
        manual.discover_race_artifacts(
            race_query="sandown r2",
            evidence_roots=[tmp_path],
            score_timestamp=SCORE_TIME,
        )


def test_race_first_discovery_does_not_treat_generic_url_path_as_venue(tmp_path):
    _write_fixture(tmp_path)

    with pytest.raises(ManualPredictionError, match="race_feature_packet_not_found"):
        manual.discover_race_artifacts(
            race_query="racing r2",
            evidence_roots=[tmp_path],
            score_timestamp=SCORE_TIME,
        )


def test_race_first_discovery_resolves_venue_before_race_date(tmp_path):
    warrnambool = _write_fixture(tmp_path / "warrnambool")
    warragul = _write_fixture(tmp_path / "warragul")

    def retarget(paths, *, race_id, venue, url):
        def mutate(rows, _manifest):
            for row in rows:
                row["race_id"] = race_id
                row["race_date"] = race_id.rsplit(" - ", 1)[1]
                row["venue"] = venue
                row["target_metadata_source_url"] = url

        _reseal(paths, mutate)

    retarget(
        warrnambool,
        race_id="Race 2 - WAR - 2026-07-16",
        venue="WAR",
        url="https://www.thedogs.com.au/racing/warrnambool/2026-07-16/2/test",
    )
    retarget(
        warragul,
        race_id="Race 2 - WARG - 2026-07-17",
        venue="WARG",
        url="https://www.thedogs.com.au/racing/warragul/2026-07-17/2/test",
    )

    with pytest.raises(ManualPredictionError, match="race_query_ambiguous"):
        manual.discover_race_artifacts(
            race_query="warr r2",
            evidence_roots=[tmp_path],
            score_timestamp=SCORE_TIME,
        )


def test_race_first_discovery_rejects_capture_symlink_escape(tmp_path):
    evidence_root = tmp_path / "evidence"
    paths = _write_fixture(evidence_root)
    outside_capture = tmp_path / "outside_capture.json"
    outside_capture.write_bytes(paths["capture"].read_bytes())
    paths["capture"].unlink()
    paths["capture"].symlink_to(outside_capture)

    with pytest.raises(
        ManualPredictionError, match="discovery_path_outside_evidence_root"
    ):
        manual.discover_race_artifacts(
            race_query="sandown r2",
            evidence_roots=[evidence_root],
            score_timestamp=SCORE_TIME,
        )


def test_race_first_discovery_rejects_unrelated_outcome_capture(tmp_path):
    _write_fixture(tmp_path)
    unrelated = tmp_path / "unrelated/autonomous_live_odds_capture_report.json"
    _write_json(
        unrelated,
        {
            "schema_version": "autonomous_live_odds_capture_report_v1",
            "winner": "Must Not Be Read",
            "attempts": [
                {
                    "race_id": "Race 9 - OTHER - 2026-07-16",
                    "status": "APPENDED",
                }
            ],
        },
    )

    with pytest.raises(
        ManualPredictionError, match="discovery_capture_contains_outcome"
    ):
        manual.discover_race_artifacts(
            race_query="sandown r2",
            evidence_roots=[tmp_path],
            score_timestamp=SCORE_TIME,
        )


def test_race_first_discovery_does_not_fallback_past_invalid_target_capture(
    tmp_path,
):
    _write_fixture(tmp_path)
    invalid = tmp_path / "later/autonomous_live_odds_capture_report.json"
    invalid.parent.mkdir()
    _write_json(
        invalid,
        {
            "schema_version": "unexpected_capture_schema",
            "attempts": [
                {
                    "race_id": RACE_ID,
                    "status": "APPENDED",
                    "validation": {"status": "PASS"},
                }
            ],
        },
    )

    with pytest.raises(ManualPredictionError, match="race_capture_report_invalid"):
        manual.discover_race_artifacts(
            race_query="sandown r2",
            evidence_roots=[tmp_path],
            score_timestamp=SCORE_TIME,
        )
