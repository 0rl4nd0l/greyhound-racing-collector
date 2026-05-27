import json
from datetime import date
from pathlib import Path

import pytest

from scripts import validate_upcoming_races
from scripts.capture_prediction_snapshot import _candidate_files
from utils.csv_metadata import (
    build_csv_download_provenance_payload,
    normalize_verified_thedogs_export_content,
    verify_canonical_sidecar_target_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
REAL_COMMA_EXPORT = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_live_batch/quarantine/20260527T092141Z_non_pipe_delimited_Race 13 - BAL - 2026-05-27.csv"
)
REAL_COMMA_SIDECAR = Path(f"{REAL_COMMA_EXPORT}.metadata.json")
ACCEPTED_NAME = "Race 13 - BAL - 2026-05-27.csv"


def _require_real_export_fixture() -> None:
    if not REAL_COMMA_EXPORT.exists() or not REAL_COMMA_SIDECAR.exists():
        pytest.skip("real TheDogs comma export fixture is not present")


def _real_content_and_sidecar() -> tuple[str, dict]:
    _require_real_export_fixture()
    return (
        REAL_COMMA_EXPORT.read_text(encoding="utf-8"),
        json.loads(REAL_COMMA_SIDECAR.read_text(encoding="utf-8")),
    )


def _normalise_fixture(tmp_path: Path) -> tuple[Path, dict, dict]:
    content, sidecar = _real_content_and_sidecar()
    accepted = tmp_path / ACCEPTED_NAME
    raw = tmp_path / "raw_exports" / ACCEPTED_NAME
    raw.parent.mkdir()
    raw.write_text(content, encoding="utf-8")

    result = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=accepted,
        raw_export_path=raw,
        sidecar_payload=sidecar,
        runner_completeness=sidecar["runner_completeness"],
    )
    assert result["normalization_status"] == "verified", result
    accepted.write_text(result["normalized_content"], encoding="utf-8")

    normalization_metadata = {
        key: value for key, value in result.items() if key != "normalized_content"
    }
    final_sidecar = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url=sidecar["race_url"],
        csv_info={"type": "GET", "url": sidecar["resolved_csv_url"]},
        content=result["normalized_content"],
        completeness=sidecar["runner_completeness"],
        race_info={
            **sidecar["race_info"],
            "target_distance": sidecar["target_distance"],
            "target_distance_source": sidecar["target_distance_source"],
            "target_grade": sidecar["target_grade"],
            "target_grade_source": sidecar["target_grade_source"],
            "metadata_is_leakage_safe": sidecar["metadata_is_leakage_safe"],
        },
        source="test_real_thedogs_export",
        normalization=normalization_metadata,
        filename=ACCEPTED_NAME,
        allow_generic_fields=False,
    )
    Path(f"{accepted}.metadata.json").write_text(
        json.dumps(final_sidecar, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return accepted, result, final_sidecar


def test_real_thedogs_comma_export_normalizes_to_pipe_with_provenance(tmp_path):
    accepted, result, sidecar = _normalise_fixture(tmp_path)

    first_line = accepted.read_text(encoding="utf-8").splitlines()[0]
    assert result["original_delimiter"] == ","
    assert result["normalized_delimiter"] == "|"
    assert result["normalization_source"] == "canonical_thedogs_export"
    assert result["normalization_action"] == "converted_to_pipe"
    assert "|" in first_line
    assert first_line.count("|") > first_line.count(",")
    assert sidecar["raw_export_path"].endswith(f"raw_exports/{ACCEPTED_NAME}")
    assert sidecar["accepted_csv_path"].endswith(ACCEPTED_NAME)
    assert sidecar["normalization_status"] == "verified"
    assert sidecar["normalization_verification"]["target_metadata_status"] == "verified"

    verified = verify_canonical_sidecar_target_metadata(accepted, race_number=13)
    assert verified["target_metadata_status"] == "verified"


def test_normalized_csv_passes_upcoming_validator(tmp_path, monkeypatch):
    accepted, _result, _sidecar = _normalise_fixture(tmp_path)

    class FixedDate(date):
        @classmethod
        def today(cls):
            return date(2026, 5, 27)

    monkeypatch.setattr(validate_upcoming_races, "date", FixedDate)
    assert validate_upcoming_races.validate_file(accepted, strict_future=False) == []
    assert validate_upcoming_races.main(["--dir", str(tmp_path)]) == 0


def test_bom_prefixed_export_normalizes_to_validator_accepted_pipe_csv(
    tmp_path, monkeypatch
):
    content, sidecar = _real_content_and_sidecar()
    accepted = tmp_path / ACCEPTED_NAME
    raw = tmp_path / "raw_exports" / ACCEPTED_NAME
    raw.parent.mkdir()
    raw.write_text("\ufeff" + content, encoding="utf-8")

    result = normalize_verified_thedogs_export_content(
        "\ufeff" + content,
        accepted_csv_path=accepted,
        raw_export_path=raw,
        sidecar_payload=sidecar,
        runner_completeness=sidecar["runner_completeness"],
    )
    assert result["normalization_status"] == "verified", result
    accepted.write_text(result["normalized_content"], encoding="utf-8")

    class FixedDate(date):
        @classmethod
        def today(cls):
            return date(2026, 5, 27)

    monkeypatch.setattr(validate_upcoming_races, "date", FixedDate)
    assert accepted.read_text(encoding="utf-8").startswith("Dog Name|Sex|PLC|BOX")
    assert validate_upcoming_races.validate_file(accepted, strict_future=False) == []


def test_upcoming_validator_rejects_arbitrary_non_csv_but_allows_sidecars(
    tmp_path, monkeypatch
):
    _accepted, _result, _sidecar = _normalise_fixture(tmp_path)

    class FixedDate(date):
        @classmethod
        def today(cls):
            return date(2026, 5, 27)

    monkeypatch.setattr(validate_upcoming_races, "date", FixedDate)
    assert validate_upcoming_races.main(["--dir", str(tmp_path)]) == 0

    (tmp_path / "unexpected.json").write_text("{}", encoding="utf-8")
    assert validate_upcoming_races.main(["--dir", str(tmp_path)]) == 1


def test_malformed_real_export_copy_is_rejected(tmp_path):
    content, sidecar = _real_content_and_sidecar()
    malformed = content.replace("Dog Name,Sex,PLC,BOX", "Dog Name,Sex,BOX", 1)

    result = normalize_verified_thedogs_export_content(
        malformed,
        accepted_csv_path=tmp_path / ACCEPTED_NAME,
        raw_export_path=tmp_path / "raw_exports" / ACCEPTED_NAME,
        sidecar_payload=sidecar,
        runner_completeness=sidecar["runner_completeness"],
    )

    assert result["normalization_status"] == "rejected"
    assert "column_count_mismatch" in result["normalization_failure_reason"]


def test_missing_target_grade_fails_closed_before_normalization(tmp_path):
    content, sidecar = _real_content_and_sidecar()
    unsafe_sidecar = dict(sidecar)
    unsafe_sidecar["target_grade"] = None

    result = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=tmp_path / ACCEPTED_NAME,
        raw_export_path=tmp_path / "raw_exports" / ACCEPTED_NAME,
        sidecar_payload=unsafe_sidecar,
        runner_completeness=sidecar["runner_completeness"],
    )

    assert result["normalization_status"] == "rejected"
    assert "target_metadata_not_verified:missing_target_grade" in result[
        "normalization_failure_reason"
    ]


def test_nested_race_info_time_metadata_is_accepted(tmp_path):
    content, sidecar = _real_content_and_sidecar()
    assert "race_time_source" not in sidecar
    assert sidecar["race_info"]["race_time_source"] == "canonical_race_url"

    result = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=tmp_path / ACCEPTED_NAME,
        raw_export_path=tmp_path / "raw_exports" / ACCEPTED_NAME,
        sidecar_payload=sidecar,
        runner_completeness=sidecar["runner_completeness"],
    )

    assert result["normalization_status"] == "verified"
    assert result["normalization_verification"]["race_time_source"] == "canonical_race_url"
    assert result["normalization_verification"]["race_time_mapping_status"] == "exact_url_match"


def test_target_date_history_row_is_rejected_as_temporal_leakage(tmp_path):
    content, sidecar = _real_content_and_sidecar()
    leaking = content.replace(",2026-05-20,", ",2026-05-27,", 1)

    result = normalize_verified_thedogs_export_content(
        leaking,
        accepted_csv_path=tmp_path / ACCEPTED_NAME,
        raw_export_path=tmp_path / "raw_exports" / ACCEPTED_NAME,
        sidecar_payload=sidecar,
        runner_completeness=sidecar["runner_completeness"],
    )

    assert result["normalization_status"] == "rejected"
    assert "non_historical_date" in result["normalization_failure_reason"]


def test_capture_candidates_ignore_raw_exports_directory(tmp_path):
    accepted, _result, _sidecar = _normalise_fixture(tmp_path)
    raw = tmp_path / "raw_exports" / ACCEPTED_NAME

    candidates = _candidate_files([], str(tmp_path))

    assert candidates == [accepted]
    assert raw.exists()
    assert raw not in candidates
