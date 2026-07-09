import json
from pathlib import Path

from scripts import build_weather_track_feature_utility_root_cause_packet as packet


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _race_csv(root: Path, name: str = "Race 1 - TEST - 2026-06-08.csv") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Dog Name|BOX\n1. Alpha Runner|1\n", encoding="utf-8")
    return path


def _safe_sidecar_payload(**overrides):
    payload = {
        "metadata_is_leakage_safe": True,
        "metadata_captured_at": "2026-06-08T00:30:00Z",
        "metadata_source_url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
        "race_url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
        "weather": "Overcast",
        "track_condition": "Soft",
        "weather_track_metadata_source": "canonical_pre_race_page",
        "weather_track_metadata_is_leakage_safe": True,
        "runner_count": 8,
        "race_info": {
            "date": "2026-06-08",
            "venue": "TEST",
            "race_number": "1",
            "race_time": "11:15 AM",
            "url": "https://www.thedogs.com.au/racing/test/2026-06-08/1/test?trial=false",
        },
    }
    payload.update(overrides)
    return payload


def _write_sidecar(csv_path: Path, payload: dict) -> Path:
    return _write_json(Path(f"{csv_path}.metadata.json"), payload)


def test_safe_sidecar_rows_are_accepted_only_with_explicit_safe_flag(tmp_path):
    accepted_csv = _race_csv(tmp_path / "accepted")
    _write_sidecar(accepted_csv, _safe_sidecar_payload())
    unsafe_csv = _race_csv(tmp_path / "unsafe")
    _write_sidecar(
        unsafe_csv,
        _safe_sidecar_payload(weather_track_metadata_is_leakage_safe=False),
    )

    rows = packet.discover_weather_track_source_rows([tmp_path])
    by_path = {row["csv_path"]: row for row in rows}

    assert by_path[packet.relpath(accepted_csv)]["status"] == "ACCEPTED"
    assert by_path[packet.relpath(accepted_csv)]["both_weather_track_present"] is True
    assert by_path[packet.relpath(unsafe_csv)]["status"] == "REJECTED"
    assert "weather_track_metadata_is_leakage_safe_not_true" in by_path[
        packet.relpath(unsafe_csv)
    ]["rejected_reasons"]


def test_post_result_url_sources_are_rejected(tmp_path):
    csv_path = _race_csv(tmp_path)
    result_url = "https://www.thedogs.com.au/racing/test/2026-06-08/1/results"
    _write_sidecar(
        csv_path,
        _safe_sidecar_payload(
            metadata_source_url=result_url,
            race_url=result_url,
            race_info={
                "date": "2026-06-08",
                "venue": "TEST",
                "race_number": "1",
                "race_time": "11:15 AM",
                "url": result_url,
            },
        ),
    )

    [row] = packet.discover_weather_track_source_rows([tmp_path])

    assert row["status"] == "REJECTED"
    assert "source_url_looks_post_result" in row["rejected_reasons"]


def test_placeholder_defaults_without_source_proof_are_rejected(tmp_path):
    csv_path = _race_csv(tmp_path)
    _write_sidecar(
        csv_path,
        _safe_sidecar_payload(
            weather="Fine",
            track_condition="Good",
            metadata_is_leakage_safe=False,
            weather_track_metadata_is_leakage_safe=False,
        ),
    )

    [row] = packet.discover_weather_track_source_rows([tmp_path])

    assert row["status"] == "REJECTED"
    assert row["weather_track_metadata_is_leakage_safe"] is False


def test_collection_after_race_time_is_rejected(tmp_path):
    csv_path = _race_csv(tmp_path)
    _write_sidecar(
        csv_path,
        _safe_sidecar_payload(metadata_captured_at="2026-06-08T12:30:00+10:00"),
    )

    [row] = packet.discover_weather_track_source_rows([tmp_path])

    assert row["status"] == "REJECTED"
    assert "metadata_captured_at_not_before_jump" in row["rejected_reasons"]


def test_zero_byte_db_is_data_missing_not_failure(tmp_path):
    db_path = tmp_path / "greyhound.sqlite"
    db_path.write_bytes(b"")

    status = packet.inspect_optional_db(db_path)

    assert status["status"] == "DATA_MISSING"
    assert status["reason"] == "db_zero_bytes"


def test_low_weather_track_coverage_skips_ablation(tmp_path):
    csv_path = _race_csv(tmp_path / "artifacts")
    _write_sidecar(csv_path, _safe_sidecar_payload())
    feature_path = tmp_path / "artifacts/shadow_evaluation_test/shadow_feature_rows.json"
    _write_json(
        feature_path,
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-08",
                "dog_name": "Alpha Runner",
                "metadata_is_leakage_safe": True,
                "weather_track_metadata_from_sidecar": True,
                "weather_track_metadata_source": "canonical_pre_race_page",
                "weather": "Overcast",
                "track_condition": "Soft",
                "weather_source_backed": True,
                "track_condition_source_backed": True,
            }
        ],
    )

    report, _ledgers = packet.build_packet(
        artifact_roots=[tmp_path / "artifacts"],
        db_path=tmp_path / "missing.sqlite",
    )

    assert report["decision"]["final_status"] == packet.FINAL_SOURCE_REPAIR
    assert report["decision"]["ablation_status"] == "NOT_RUN_SOURCE_COVERAGE_LOW"


def test_run_packet_writes_report_only_and_preserves_protected_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    protected_db = tmp_path / "greyhound_racing_data.db"
    protected_db.write_text("do-not-change", encoding="utf-8")
    monkeypatch.setattr(packet, "DEFAULT_PROTECTED_PATHS", (protected_db,))

    artifacts = tmp_path / "artifacts"
    csv_path = _race_csv(artifacts)
    _write_sidecar(csv_path, _safe_sidecar_payload())
    _write_json(
        artifacts / "shadow_evaluation_test/shadow_feature_rows.json",
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-08",
                "dog_name": "Alpha Runner",
                "metadata_is_leakage_safe": True,
                "weather_track_metadata_from_sidecar": True,
                "weather_track_metadata_source": "canonical_pre_race_page",
                "weather": "Overcast",
                "track_condition": "Soft",
                "weather_source_backed": True,
                "track_condition_source_backed": True,
            }
        ],
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "weather_track_feature_utility_root_cause_test_report_only"
    )

    result = packet.run_packet(
        output_dir=output_dir,
        artifact_roots=[artifacts],
        db_path=tmp_path / "missing.sqlite",
    )

    assert result["final_status"] == packet.FINAL_SOURCE_REPAIR
    assert result["protected_paths_unchanged"] is True
    assert protected_db.read_text(encoding="utf-8") == "do-not-change"
    assert (output_dir / "weather_track_feature_utility_root_cause_report.json").exists()
    assert (output_dir / "weather_track_source_coverage.csv").exists()
    assert (output_dir / "BOARD_DECISION.json").exists()
    assert (output_dir / "output_manifest.json").exists()
