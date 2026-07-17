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

    with pytest.raises(ManualPredictionError, match="sidecar_source_url_alias_mismatch"):
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
    _write_json(paths["sidecar"], sidecar)

    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    attempt = capture["attempts"][0]
    attempt["race_id"] = race_id
    attempt["fetch_time"] = fetch.isoformat()
    attempt["append_time"] = append.isoformat()
    _write_json(paths["capture"], capture)

    def retime(rows, manifest):
        manifest["feature_freeze_timestamp"] = freeze.isoformat()
        manifest["generated_at"] = generated.isoformat()
        for row in rows:
            row["race_id"] = race_id
            row["race_date"] = jump.date().isoformat()

    _reseal(paths, retime)
    return race_id


def _retarget_fixture_contract(
    paths: dict[str, Path],
    *,
    venue: str,
    venue_slug: str,
    sidecar_grade: str,
    feature_grade: str,
) -> str:
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    shadow = sidecar["prejump_shadow_metadata"]
    race_number = int(shadow["race_number"])
    race_date = str(shadow["race_date"])
    race_id = f"Race {race_number} - {venue} - {race_date}"
    thedogs_url = (
        f"https://www.thedogs.com.au/racing/{venue_slug}/"
        f"{race_date}/{race_number}/test"
    )
    shadow["venue"] = venue
    shadow["grade"] = sidecar_grade
    shadow["source_url"] = thedogs_url
    sidecar["race_url"] = thedogs_url
    old_form_csv = paths["form_csv"]
    old_sidecar = paths["sidecar"]
    form_csv = old_form_csv.with_name(f"{race_id}.csv")
    sidecar_path = form_csv.with_name(form_csv.name + ".metadata.json")
    if form_csv != old_form_csv:
        old_form_csv.rename(form_csv)
        old_sidecar.rename(sidecar_path)
        paths["form_csv"] = form_csv
        paths["sidecar"] = sidecar_path
    sidecar["filename"] = form_csv.name
    _write_json(paths["sidecar"], sidecar)

    capture = _json(paths["capture"])
    assert isinstance(capture, dict)
    attempt = capture["attempts"][0]
    attempt["race_id"] = race_id
    attempt["validation"]["source_url"] = (
        "https://www.sportsbet.com.au/greyhound-racing/"
        f"australia-nz/{venue_slug}/race-{race_number}-123"
    )
    _write_json(paths["capture"], capture)

    def retarget(rows, manifest):
        manifest["input_files"] = [str(form_csv.resolve())]
        for row in rows:
            row["race_id"] = race_id
            row["race_date"] = race_date
            row["race_number"] = race_number
            row["venue"] = venue
            row["target_metadata_source_url"] = thedogs_url
            row["target_grade_safe"] = feature_grade
            row["source_csv"] = str(form_csv.resolve())

    _reseal(paths, retarget)
    return race_id


def _early_residual_index_payload(
    run_id: str, races: list[dict[str, object]]
) -> dict[str, object]:
    race_count = len(races)
    return {
        "activation": False,
        "appended_count": race_count,
        "blocked_count": 0,
        "exact_replay_count": 0,
        "lock_release_preceded_stage_completion": False,
        "outcomes_read": False,
        "race_count": race_count,
        "races": [
            {"race_id": race.get("race_id"), "status": "APPENDED"}
            for race in races
        ],
        "schema_version": "early_residual_shadow_prediction_status_v1",
        "status": "PASS",
        "plan": {
            "activation": False,
            "blockers": [],
            "outcomes_read": False,
            "production_db_access": "sqlite_mode_ro_feature_history_only",
            "race_count": race_count,
            "races": races,
            "run_id": run_id,
            "schema_version": "early_residual_shadow_prediction_plan_v1",
            "status": "READY",
        },
    }


def _embedded_index_prediction(race_id: str) -> dict[str, object]:
    return {
        "activation": False,
        "outcomes_present": False,
        "persisted": True,
        "persistence_status": "APPENDED",
        "race_id": race_id,
        "schema_version": "manual_market_form_residual_prediction_v2",
        "source_contract": {
            "database_access": False,
            "feature_reconstruction_performed": False,
            "feature_source": "exact_hash_bound_system_shadow_feature_rows",
            "network_access": False,
        },
        "status": "MANUAL_PREJUMP_FROZEN_RESIDUAL_PREDICTION",
        "variants": {"full_strength": 1.0, "half_strength": 0.5},
    }


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


def test_race_first_uses_current_indexed_system_evidence_root_by_default(
    tmp_path, monkeypatch, capfd
):
    live_root = tmp_path / "retained_evidence"
    paths = _write_fixture(live_root / "sealed_packet")
    race_id = _retime_for_cli(paths)
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / (
        f"shadow_autopilot_daemonization_v1_{run_id}"
    )
    status_dir.mkdir(parents=True)
    _write_json(
        status_dir / "early_residual_shadow_status.json",
        _early_residual_index_payload(
            run_id,
            [
                {
                    "race_id": race_id,
                    "form_csv_path": str(paths["form_csv"]),
                    "sidecar_path": str(paths["sidecar"]),
                    "feature_output_dir": str(paths["feature_rows"].parent),
                    "capture_path": str(paths["capture"]),
                }
            ],
        ),
    )
    local_root = tmp_path / "checkout_evidence"
    local_root.mkdir()
    monkeypatch.setattr(manual, "DEFAULT_EVIDENCE_ROOT", local_root)
    monkeypatch.setattr(
        manual, "DEFAULT_RETAINED_EVIDENCE_ROOTS", (live_root,)
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }

    returncode = manual.main(["--race", "sundown r2"])

    captured = capfd.readouterr()
    assert returncode == 0, captured.err
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["race_id"] == race_id
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_current_system_evidence_index_rejects_path_escape(tmp_path):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / (
        f"shadow_autopilot_daemonization_v1_{run_id}"
    )
    status_dir.mkdir(parents=True)
    escaped = tmp_path / "outside"
    _write_json(
        status_dir / "early_residual_shadow_status.json",
        _early_residual_index_payload(
            run_id,
            [
                {
                    "race_id": RACE_ID,
                    "form_csv_path": str(escaped / "race.csv"),
                    "sidecar_path": str(escaped / "race.csv.metadata.json"),
                    "feature_output_dir": str(escaped / "features"),
                    "capture_path": str(escaped / "capture.json"),
                }
            ],
        ),
    )

    with pytest.raises(
        ManualPredictionError, match="early_residual_status_index_path_escape"
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


def test_current_system_evidence_index_rejects_symlinked_external_index(tmp_path):
    live_root = tmp_path / "retained_evidence"
    live_root.mkdir()
    packet = _write_fixture(live_root / "sealed_packet")
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    external_status_dir = tmp_path / "external_status"
    external_status_dir.mkdir()
    _write_json(
        external_status_dir / "early_residual_shadow_status.json",
        _early_residual_index_payload(
            run_id,
            [
                {
                    "race_id": RACE_ID,
                    "form_csv_path": str(packet["form_csv"]),
                    "sidecar_path": str(packet["sidecar"]),
                    "feature_output_dir": str(packet["feature_rows"].parent),
                    "capture_path": str(packet["capture"]),
                }
            ],
        ),
    )
    (live_root / f"shadow_autopilot_daemonization_v1_{run_id}").symlink_to(
        external_status_dir,
        target_is_directory=True,
    )

    with pytest.raises(
        ManualPredictionError, match="early_residual_status_index_path_escape"
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


@pytest.mark.parametrize(
    ("status_schema", "plan_schema", "plan_run_id"),
    [
        ("wrong", "early_residual_shadow_prediction_plan_v1", "CURRENT"),
        ("early_residual_shadow_prediction_status_v1", "wrong", "CURRENT"),
        (
            "early_residual_shadow_prediction_status_v1",
            "early_residual_shadow_prediction_plan_v1",
            "20200101T000000+1000_odds_capture",
        ),
    ],
)
def test_current_system_evidence_index_requires_exact_schema_and_run_binding(
    status_schema, plan_schema, plan_run_id, tmp_path
):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    payload = _early_residual_index_payload(run_id, [])
    payload["schema_version"] = status_schema
    payload["plan"]["schema_version"] = plan_schema
    payload["plan"]["run_id"] = (
        run_id if plan_run_id == "CURRENT" else plan_run_id
    )
    _write_json(
        status_dir / "early_residual_shadow_status.json",
        payload,
    )

    with pytest.raises(
        ManualPredictionError,
        match="early_residual_status_index_(schema|run_id)_mismatch",
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


@pytest.mark.parametrize(
    "unsafe_case",
    [
        "plural_outcomes",
        "plural_results",
        "prefixed_outcomes",
        "prefixed_results",
        "camel_outcomes",
        "winner_details",
        "unsafe_outcomes_marker",
        "actual_win_by_box",
        "winning_dog",
        "first_place_dog",
        "positions",
        "plan_blockers",
        "read_write_db",
        "early_lock_release",
    ],
)
def test_current_system_evidence_index_rejects_unsafe_or_nonfinal_status(
    unsafe_case, tmp_path
):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    payload = _early_residual_index_payload(run_id, [])
    if unsafe_case == "plural_outcomes":
        payload["outcomes"] = []
    elif unsafe_case == "plural_results":
        payload["results"] = []
    elif unsafe_case == "prefixed_outcomes":
        payload["race_outcomes"] = []
    elif unsafe_case == "prefixed_results":
        payload["official_results"] = []
    elif unsafe_case == "camel_outcomes":
        payload["raceOutcomes"] = []
    elif unsafe_case == "winner_details":
        payload["winner_details"] = {}
    elif unsafe_case == "unsafe_outcomes_marker":
        payload["outcomes_present"] = True
    elif unsafe_case == "actual_win_by_box":
        payload["actualWinByBox"] = {}
    elif unsafe_case == "winning_dog":
        payload["winningDog"] = "synthetic"
    elif unsafe_case == "first_place_dog":
        payload["firstPlaceDog"] = "synthetic"
    elif unsafe_case == "positions":
        payload["positions"] = []
    elif unsafe_case == "plan_blockers":
        payload["plan"]["blockers"] = ["synthetic_blocker"]
    elif unsafe_case == "read_write_db":
        payload["plan"]["production_db_access"] = "sqlite_mode_rw"
    elif unsafe_case == "early_lock_release":
        payload["lock_release_preceded_stage_completion"] = True
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    with pytest.raises(
        ManualPredictionError,
        match="early_residual_status_index_(contains_outcome|unknown_field|unsafe)",
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


@pytest.mark.parametrize(
    "unknown_location",
    ["top", "plan", "plan_race", "status_race", "prediction"],
)
def test_current_system_evidence_index_rejects_unknown_nested_fields(
    unknown_location, tmp_path
):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    race = {
        "race_id": RACE_ID,
        "form_csv_path": str(live_root / "race.csv"),
        "sidecar_path": str(live_root / "race.csv.metadata.json"),
        "feature_output_dir": str(live_root / "features"),
        "capture_path": str(live_root / "capture.json"),
    }
    payload = _early_residual_index_payload(run_id, [race])
    if unknown_location == "top":
        target = payload
    elif unknown_location == "plan":
        target = payload["plan"]
    elif unknown_location == "plan_race":
        target = payload["plan"]["races"][0]
    elif unknown_location == "status_race":
        target = payload["races"][0]
    else:
        payload["races"][0]["prediction"] = {}
        target = payload["races"][0]["prediction"]
    target["winningDog"] = "synthetic"
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    with pytest.raises(
        ManualPredictionError, match="early_residual_status_index_unknown_field"
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


@pytest.mark.parametrize(
    "unsafe_nested_prediction",
    [
        "activation",
        "database_access",
        "network_access",
        "feature_reconstruction",
        "race_id",
        "strength",
    ],
)
def test_current_system_evidence_index_rejects_unsafe_nested_prediction(
    unsafe_nested_prediction, tmp_path
):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    race = {
        "race_id": RACE_ID,
        "form_csv_path": str(live_root / "race.csv"),
        "sidecar_path": str(live_root / "race.csv.metadata.json"),
        "feature_output_dir": str(live_root / "features"),
        "capture_path": str(live_root / "capture.json"),
    }
    payload = _early_residual_index_payload(run_id, [race])
    prediction = _embedded_index_prediction(RACE_ID)
    payload["races"][0]["prediction"] = prediction
    if unsafe_nested_prediction == "activation":
        prediction["activation"] = True
    elif unsafe_nested_prediction == "database_access":
        prediction["source_contract"]["database_access"] = True
    elif unsafe_nested_prediction == "network_access":
        prediction["source_contract"]["network_access"] = True
    elif unsafe_nested_prediction == "feature_reconstruction":
        prediction["source_contract"]["feature_reconstruction_performed"] = True
    elif unsafe_nested_prediction == "race_id":
        prediction["race_id"] = "Race 3 - SAN - 2026-07-16"
    else:
        prediction["variants"]["full_strength"] = 2.0
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    with pytest.raises(
        ManualPredictionError, match="early_residual_status_index_unsafe"
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


@pytest.mark.parametrize("non_authority", ["plan_blocked", "missing_status", "skipped"])
def test_current_system_evidence_index_skips_non_authoritative_status(
    non_authority, tmp_path
):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    payload = _early_residual_index_payload(run_id, [])
    if non_authority == "plan_blocked":
        payload["plan"]["status"] = "BLOCKED"
    elif non_authority == "missing_status":
        payload.pop("status")
    else:
        payload["status"] = "SKIPPED_NO_NEW_CAPTURE"
        payload["plan"]["status"] = "SKIPPED_NO_NEW_CAPTURE"
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    assert manual._indexed_evidence_roots(live_root, score_timestamp=now) == []


def test_current_system_evidence_index_rejects_outcome_key_on_skipped_status(tmp_path):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    payload = _early_residual_index_payload(run_id, [])
    payload["status"] = "SKIPPED_NO_NEW_CAPTURE"
    payload["plan"]["status"] = "SKIPPED_NO_NEW_CAPTURE"
    payload["race_outcomes"] = []
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    with pytest.raises(
        ManualPredictionError, match="early_residual_status_index_contains_outcome"
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


def test_current_system_evidence_index_skips_empty_index_before_usable_index(tmp_path):
    live_root = tmp_path / "retained_evidence"
    paths = _write_fixture(live_root / "sealed_packet")
    now = datetime.now(tz=MELBOURNE).replace(microsecond=0)
    skipped_time = now - timedelta(minutes=1)
    skipped_run_id = f"{skipped_time.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    skipped_dir = live_root / (
        f"shadow_autopilot_daemonization_v1_{skipped_run_id}"
    )
    skipped_dir.mkdir(parents=True)
    skipped = _early_residual_index_payload(skipped_run_id, [])
    skipped["status"] = "SKIPPED_NO_NEW_CAPTURE"
    skipped["plan"]["status"] = "SKIPPED_NO_NEW_CAPTURE"
    _write_json(skipped_dir / "early_residual_shadow_status.json", skipped)

    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    ready_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    ready_dir.mkdir(parents=True)
    ready = _early_residual_index_payload(
        run_id,
        [
            {
                "race_id": RACE_ID,
                "form_csv_path": str(paths["form_csv"]),
                "sidecar_path": str(paths["sidecar"]),
                "feature_output_dir": str(paths["feature_rows"].parent),
                "capture_path": str(paths["capture"]),
            }
        ],
    )
    _write_json(ready_dir / "early_residual_shadow_status.json", ready)

    roots = manual._indexed_evidence_roots(live_root, score_timestamp=now)

    assert set(roots) == {
        paths["form_csv"].parent.resolve(),
        paths["feature_rows"].parent.resolve(),
        paths["capture"].parent.resolve(),
    }


def test_current_system_evidence_index_accepts_finalized_blocked_status_with_packet(
    tmp_path,
):
    live_root = tmp_path / "retained_evidence"
    paths = _write_fixture(live_root / "sealed_packet")
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    missing = live_root / "blocked_packet"
    races = [
        {
            "race_id": RACE_ID,
            "form_csv_path": str(paths["form_csv"]),
            "sidecar_path": str(paths["sidecar"]),
            "feature_output_dir": str(paths["feature_rows"].parent),
            "capture_path": str(paths["capture"]),
        },
        {
            "race_id": "Race 3 - SAN - 2026-07-16",
            "form_csv_path": str(missing / "race.csv"),
            "sidecar_path": str(missing / "race.csv.metadata.json"),
            "feature_output_dir": str(missing / "features"),
            "capture_path": str(missing / "capture.json"),
        },
    ]
    payload = _early_residual_index_payload(run_id, races)
    payload["status"] = "BLOCKED"
    payload["appended_count"] = 1
    payload["blocked_count"] = 1
    payload["races"] = [
        {"race_id": races[0]["race_id"], "status": "APPENDED"},
        {"race_id": races[1]["race_id"], "status": "BLOCKED"},
    ]
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    roots = manual._indexed_evidence_roots(live_root, score_timestamp=now)

    assert set(roots) == {
        paths["form_csv"].parent.resolve(),
        paths["feature_rows"].parent.resolve(),
        paths["capture"].parent.resolve(),
    }


@pytest.mark.parametrize("identity_case", ["mismatch", "duplicate"])
def test_current_system_evidence_index_binds_unique_status_race_ids(
    identity_case, tmp_path
):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    status_dir = live_root / f"shadow_autopilot_daemonization_v1_{run_id}"
    status_dir.mkdir(parents=True)
    races = [
        {
            "race_id": RACE_ID,
            "form_csv_path": str(live_root / "one.csv"),
            "sidecar_path": str(live_root / "one.csv.metadata.json"),
            "feature_output_dir": str(live_root / "one_features"),
            "capture_path": str(live_root / "one_capture.json"),
        },
        {
            "race_id": "Race 3 - SAN - 2026-07-16",
            "form_csv_path": str(live_root / "two.csv"),
            "sidecar_path": str(live_root / "two.csv.metadata.json"),
            "feature_output_dir": str(live_root / "two_features"),
            "capture_path": str(live_root / "two_capture.json"),
        },
    ]
    payload = _early_residual_index_payload(run_id, races)
    if identity_case == "mismatch":
        payload["races"][1]["race_id"] = "Race 4 - SAN - 2026-07-16"
    else:
        payload["races"][1]["race_id"] = RACE_ID
    _write_json(status_dir / "early_residual_shadow_status.json", payload)

    with pytest.raises(
        ManualPredictionError, match="early_residual_status_index_races_invalid"
    ):
        manual._indexed_evidence_roots(live_root, score_timestamp=now)


def test_stale_system_evidence_index_is_not_discovery_authority(tmp_path):
    live_root = tmp_path / "retained_evidence"
    now = datetime.now(tz=MELBOURNE)
    stale = now - timedelta(hours=37)
    status_dir = live_root / (
        "shadow_autopilot_daemonization_v1_"
        f"{stale.strftime('%Y%m%dT%H%M%S%z')}_odds_capture"
    )
    status_dir.mkdir(parents=True)
    _write_json(
        status_dir / "early_residual_shadow_status.json",
        {"winner": "synthetic-outcome-that-must-not-be-read"},
    )

    assert manual._indexed_evidence_roots(live_root, score_timestamp=now) == []


def test_race_first_cli_accepts_canonical_hyphenated_venue(tmp_path):
    paths = _write_fixture(tmp_path)
    baseline_race_id = _retime_for_cli(paths)
    baseline = _score_paths(
        paths,
        race_id=baseline_race_id,
        score_timestamp=datetime.now(MELBOURNE),
    )
    race_id = _retarget_fixture_contract(
        paths,
        venue="LADBROKES-Q1-LAKESIDE",
        venue_slug="ladbrokes-q1-lakeside",
        sidecar_grade="Grade 5",
        feature_grade="Grade 5",
    )
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
            "q1 lakeside r2",
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
    assert payload["predictions"] == baseline["predictions"]
    assert paths["form_csv"].name == f"{race_id}.csv"
    assert paths["sidecar"].name == f"{race_id}.csv.metadata.json"
    assert completed.stdout == _canonical_bytes(payload)
    for key in ("market", "half", "full"):
        assert payload["probability_sums"][key] == pytest.approx(1.0)
    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert after == before


def test_explicit_scorer_accepts_restricted_grade_alias_without_prediction_change(
    tmp_path,
):
    paths = _write_fixture(tmp_path)
    baseline = _score_paths(paths)
    race_id = _retarget_fixture_contract(
        paths,
        venue="SAN",
        venue_slug="sandown-park",
        sidecar_grade="Restricted",
        feature_grade="Restricted Win",
    )

    aliased = _score_paths(paths, race_id=race_id)

    assert aliased["predictions"] == baseline["predictions"]
    assert aliased["probability_sums"] == baseline["probability_sums"]


@pytest.mark.parametrize(
    ("sidecar_grade", "feature_grade"),
    [
        ("RW", "R/W"),
        ("NP", "N/P"),
        ("FFA", "Free For All"),
        ("INVITATIONAL", "Invitation"),
        ("Restricted Win Final", "Restricted Win"),
        ("4/5", "Mixed 4/5"),
        ("Tier 3 - Grade 5", "Grade 5"),
        ("5th Grade", "Grade 5"),
    ],
)
def test_accepts_finite_generator_grade_alias_equivalence(
    sidecar_grade, feature_grade, tmp_path
):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["grade"] = sidecar_grade
    _write_json(paths["sidecar"], sidecar)

    def set_feature_grade(rows, _manifest):
        for row in rows:
            row["target_grade_safe"] = feature_grade

    _reseal(paths, set_feature_grade)

    payload = _score_paths(paths)
    for key in ("market", "half", "full"):
        assert payload["probability_sums"][key] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "grade",
    [
        "Mystery Grade",
        "Restricted nonsense",
        "Mystery Grade 5",
        "Grade 5 garbage",
        "not Maiden at all",
        "PM 390m",
        "Other 515m",
        "1th Grade",
        "Grade 999",
        "G999",
        "P999",
        "Group 999",
        "99-100 Win",
        "M999",
        "NG999-999",
        "NG14",
        "M",
        "1st/8th Grade",
        "8th/1st Grade",
        "8th/8th Grade",
        "1st/1st Grade",
        "2nd/7th Grade",
    ],
)
def test_rejects_unknown_sidecar_grade(grade, tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["grade"] = grade
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_grade_invalid"):
        _score_paths(paths)


def test_rejects_unknown_feature_grade(tmp_path):
    paths = _write_fixture(tmp_path)

    def add_unknown_grade(rows, _manifest):
        for row in rows:
            row["target_grade_safe"] = "Grade 5 garbage"

    _reseal(paths, add_unknown_grade)

    with pytest.raises(ManualPredictionError, match="feature_row_target_grade_invalid"):
        _score_paths(paths)


@pytest.mark.parametrize(
    "grade",
    [
        "5",
        "FFA",
        "NP",
        "INVITATIONAL",
        "Restricted Win Heat",
        "Restricted Win Final",
        "Tier 3 - Grade 5",
        "Mixed 4/5",
        "Mixed 5/6",
        "BT8",
        "J/M",
        "I",
        "TG1-4W",
        "TG1-6W",
        "TG5+W",
        "MI4/5MA",
        "5/M",
    ],
)
def test_accepts_exact_generator_grade_contract(grade):
    assert manual._canonical_target_grade(grade) is not None


@pytest.mark.parametrize(
    "grade",
    [
        "3rd/4th Grade",
        "4th Grade",
        "4th/5th Grade",
        "5th Grade",
        "5th/6th Grade",
        "6th Grade",
        "Best 8",
        "Free For All",
        "Grade 4",
        "Grade 5",
        "Grade 6",
        "Grade 7",
        "Invitation",
        "M1/M2/M3",
        "M3",
        "M5",
        "Maiden",
        "Masters",
        "Mixed",
        "N/P",
        "NG1-4",
        "Open",
        "Other",
        "P5",
        "R/W",
        "Restricted",
    ],
)
def test_accepts_current_system_grade_vocabulary(grade):
    assert manual._canonical_target_grade(grade) is not None


@pytest.mark.parametrize("grade", ["Restricted/Win", "Restricted-Win"])
def test_rejects_undeclared_restricted_aliases(grade):
    assert manual._canonical_target_grade(grade) is None


@pytest.mark.parametrize(
    "grade",
    [
        "M",
        "1st/8th Grade",
        "8th/1st Grade",
        "8th/8th Grade",
        "1st/1st Grade",
        "2nd/7th Grade",
    ],
)
def test_rejects_ambiguous_or_undeclared_ordinal_grade_aliases(grade):
    assert manual._canonical_target_grade(grade) is None


def test_finite_grade_alias_contract_is_immutable():
    with pytest.raises(TypeError):
        manual.GRADE_ALIASES["GRADE 999"] = "GRADE 5"


@pytest.mark.parametrize("grade", [["Grade 5"], {"grade": "Grade 5"}, 5])
def test_rejects_non_string_sidecar_grade(grade, tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["grade"] = grade
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_grade_invalid"):
        _score_paths(paths)


def test_rejects_falsey_non_string_primary_grade_instead_of_falling_back(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["grade"] = False
    sidecar["race_info"] = {"grade": "Grade 5"}
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_grade_invalid"):
        _score_paths(paths)


def test_rejects_conflicting_sidecar_grade_aliases(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["race_info"] = {"grade": "Grade 4"}
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_grade_alias_mismatch"):
        _score_paths(paths)


@pytest.mark.parametrize("grade", [["Grade 5"], {"grade": "Grade 5"}, 5])
def test_rejects_non_string_feature_grade(grade, tmp_path):
    paths = _write_fixture(tmp_path)

    def set_feature_grade(rows, _manifest):
        for row in rows:
            row["target_grade_safe"] = grade

    _reseal(paths, set_feature_grade)

    with pytest.raises(ManualPredictionError, match="feature_row_target_grade_invalid"):
        _score_paths(paths)


@pytest.mark.parametrize(
    ("sidecar_grade", "feature_grade"),
    [
        ("Group 1", "Group 2"),
        ("Grade 4", "Grade 5"),
        ("Mixed 4/5", "Mixed 5/6"),
        ("4th/5th Grade", "Mixed 4/5"),
    ],
)
def test_rejects_distinct_known_grade_contracts(
    sidecar_grade, feature_grade, tmp_path
):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["grade"] = sidecar_grade
    _write_json(paths["sidecar"], sidecar)

    def set_feature_grade(rows, _manifest):
        for row in rows:
            row["target_grade_safe"] = feature_grade

    _reseal(paths, set_feature_grade)

    with pytest.raises(ManualPredictionError, match="feature_row_target_grade_mismatch"):
        _score_paths(paths)


def test_rejects_genuinely_different_known_grades(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["grade"] = "Restricted"
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="feature_row_target_grade_mismatch"):
        _score_paths(paths)


@pytest.mark.parametrize(
    "venue",
    [
        "-SAN",
        "SAN-",
        "SAN--PARK",
        "SAN/PARK",
        "SAN PARK",
        "SAN\nPARK",
    ],
)
def test_rejects_malformed_canonical_venue(venue, tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["venue"] = venue
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_venue_invalid"):
        _score_paths(paths)


def test_rejects_falsey_non_string_primary_venue_instead_of_falling_back(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["prejump_shadow_metadata"]["venue"] = False
    sidecar["race_info"] = {"venue": "SAN"}
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_venue_invalid"):
        _score_paths(paths)


def test_rejects_conflicting_sidecar_venue_aliases(tmp_path):
    paths = _write_fixture(tmp_path)
    sidecar = _json(paths["sidecar"])
    assert isinstance(sidecar, dict)
    sidecar["race_info"] = {"venue": "HEA"}
    _write_json(paths["sidecar"], sidecar)

    with pytest.raises(ManualPredictionError, match="target_venue_alias_mismatch"):
        _score_paths(paths)


@pytest.mark.parametrize(
    ("url", "race_id", "expected"),
    [
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/race-7-123",
            "Race 7 - LADBROKES-Q1-LAKESIDE - 2026-07-17",
            True,
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/race-8-123",
            "Race 7 - LADBROKES-Q1-LAKESIDE - 2026-07-17",
            False,
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/race-7-123",
            "Race 7 - LADBROKES--Q1 - 2026-07-17",
            False,
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/race-7-123",
            "Race 7 - -LADBROKES - 2026-07-17",
            False,
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/race-7-123",
            "Race 7 - LADBROKES- - 2026-07-17",
            False,
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/results/race-7-123",
            "Race 7 - LADBROKES-Q1-LAKESIDE - 2026-07-17",
            False,
        ),
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
            "ladbrokes-q1-lakeside/dividends/race-7-123",
            "Race 7 - LADBROKES-Q1-LAKESIDE - 2026-07-17",
            False,
        ),
    ],
)
def test_sportsbet_url_validator_binds_hyphenated_venue_race_id(
    url, race_id, expected
):
    assert manual._trusted_sportsbet_url(url, race_id) is expected


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
