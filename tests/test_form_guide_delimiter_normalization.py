import json
from datetime import date
from pathlib import Path

import pytest

from scripts import validate_upcoming_races
from scripts.capture_prediction_snapshot import _candidate_files
from utils.csv_metadata import (
    THEDOGS_EXPERT_FORM_COLUMNS,
    build_csv_download_provenance_payload,
    build_safe_target_metadata_payload,
    load_safe_weather_track_metadata,
    normalize_verified_thedogs_export_content,
    verify_canonical_sidecar_target_metadata,
)
from utils.runner_completeness import analyze_csv_text_runner_completeness


ROOT = Path(__file__).resolve().parents[1]
REAL_COMMA_EXPORT = (
    ROOT
    / "artifacts/full_evidence_orchestration_20260525/post_target_metadata_fix_live_batch/quarantine/20260527T092141Z_non_pipe_delimited_Race 13 - BAL - 2026-05-27.csv"
)
REAL_COMMA_SIDECAR = Path(f"{REAL_COMMA_EXPORT}.metadata.json")
ACCEPTED_NAME = "Race 13 - BAL - 2026-05-27.csv"
SYNTHETIC_RACE_URL = (
    "https://www.thedogs.com.au/racing/test/2026-05-29/1/test-race?trial=false"
)


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


def _synthetic_form_export(runners: list[tuple[int, str]], *, delimiter: str = ",") -> str:
    rows = [list(THEDOGS_EXPERT_FORM_COLUMNS)]
    for box_number, dog_name in runners:
        rows.append(
            [
                f"{box_number}. {dog_name}",
                "D",
                "1",
                str(box_number),
                "30.0",
                "400",
                "2026-05-01",
                "TEST",
                "M",
                "22.10",
                "2.00",
                "22.00",
                "5.00",
                "1.00",
                "111",
                "1",
                "$2.00",
            ]
        )
    return "\n".join(delimiter.join(row) for row in rows) + "\n"


def _synthetic_sidecar() -> dict:
    return {
        "race_url": SYNTHETIC_RACE_URL,
        "race_info": {
            "race_number": 1,
            "race_time_mapping_status": "exact_url_match",
            "race_time_source": "canonical_race_url",
            "url": SYNTHETIC_RACE_URL,
        },
        "target_distance": "400m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade": "Maiden",
        "target_grade_source": "canonical_pre_race_page",
        "metadata_is_leakage_safe": True,
    }


def _canonical_runner_set(runners: list[tuple[int, str]]) -> dict:
    participants = [
        {"box_number": box_number, "dog_name": dog_name}
        for box_number, dog_name in runners
    ]
    return {
        "schema_version": "canonical_pre_race_runner_set_v1",
        "canonical_runner_set_status": "available",
        "final_runner_source": "canonical_pre_race_page",
        "final_runner_source_url": SYNTHETIC_RACE_URL,
        "final_runner_boxes": [box_number for box_number, _dog_name in runners],
        "final_runner_names": [dog_name for _box_number, dog_name in runners],
        "final_runner_participants": participants,
        "scratched_boxes": [],
        "scratched_participants": [],
        "reserve_boxes": [],
        "vacant_boxes": [],
        "race_number": 1,
        "expected_race_number": 1,
        "extraction_timestamp": "2026-05-29T07:00:00Z",
        "ambiguous_reasons": [],
    }


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


def test_provenance_payload_writes_flat_prejump_shadow_metadata_block(tmp_path):
    runners = [
        (1, "Alpha Runner"),
        (2, "Bravo Runner"),
        (3, "Charlie Runner"),
        (4, "Delta Runner"),
    ]
    content = _synthetic_form_export(runners)
    accepted = tmp_path / "Race 1 - TEST - 2026-05-29.csv"
    raw = tmp_path / "raw_exports" / accepted.name
    raw.parent.mkdir()
    raw.write_text(content, encoding="utf-8")
    sidecar = _synthetic_sidecar()
    sidecar["race_info"].update(
        {
            "date": "2026-05-29",
            "venue": "TEST",
            "race_time": "11:15 AM",
        }
    )
    completeness = analyze_csv_text_runner_completeness(content, source="synthetic")

    result = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=accepted,
        raw_export_path=raw,
        sidecar_payload=sidecar,
        runner_completeness=completeness.as_dict(),
        canonical_runner_set=_canonical_runner_set(runners),
    )
    assert result["normalization_status"] == "verified", result

    final_sidecar = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url=SYNTHETIC_RACE_URL,
        csv_info={"type": "GET", "url": f"{SYNTHETIC_RACE_URL}/export-expert-form"},
        content=result["normalized_content"],
        completeness=analyze_csv_text_runner_completeness(
            result["normalized_content"],
            source="synthetic-normalized",
        ),
        race_info={
            **sidecar["race_info"],
            "target_distance": sidecar["target_distance"],
            "target_distance_source": sidecar["target_distance_source"],
            "target_grade": sidecar["target_grade"],
            "target_grade_source": sidecar["target_grade_source"],
            "metadata_is_leakage_safe": sidecar["metadata_is_leakage_safe"],
        },
        normalization={key: value for key, value in result.items() if key != "normalized_content"},
        filename=accepted.name,
        allow_generic_fields=False,
    )

    shadow_metadata = final_sidecar["prejump_shadow_metadata"]
    assert shadow_metadata["status"] == "PASS"
    assert shadow_metadata["race_date"] == "2026-05-29"
    assert shadow_metadata["venue"] == "TEST"
    assert shadow_metadata["race_number"] == 1
    assert shadow_metadata["jump_time"] == "11:15 AM"
    assert shadow_metadata["metadata_captured_at"]
    assert shadow_metadata["target_distance_safe"] == "400m"
    assert shadow_metadata["target_grade_safe"] == "Maiden"
    assert shadow_metadata["source_url"] == SYNTHETIC_RACE_URL
    assert shadow_metadata["runner_box_name_list"] == [
        {"box_number": 1, "dog_name": "Alpha Runner"},
        {"box_number": 2, "dog_name": "Bravo Runner"},
        {"box_number": 3, "dog_name": "Charlie Runner"},
        {"box_number": 4, "dog_name": "Delta Runner"},
    ]
    assert shadow_metadata["canonical_final_runner_alignment"]["status"] == "aligned"


def test_provenance_payload_preserves_safe_weather_track_fields(tmp_path):
    content = _synthetic_form_export([(1, "Alpha Dog")])
    sidecar = _synthetic_sidecar()
    race_info = {
        **sidecar["race_info"],
        "date": "2026-05-29",
        "venue": "TEST",
        "race_time": "11:15 AM",
        "track_condition": "Slow",
        "weather_condition": "Showers",
        "target_distance": sidecar["target_distance"],
        "target_distance_source": sidecar["target_distance_source"],
        "target_grade": sidecar["target_grade"],
        "target_grade_source": sidecar["target_grade_source"],
    }

    payload = build_csv_download_provenance_payload(
        filepath=tmp_path / "Race 1 - TEST - 2026-05-29.csv",
        race_url=SYNTHETIC_RACE_URL,
        csv_info={"type": "GET", "url": "https://example.test/form.csv"},
        content=content,
        completeness={"status": "COMPLETE"},
        race_info=race_info,
    )

    assert payload["race_info"]["track_condition"] == "Slow"
    assert payload["race_info"]["weather_condition"] == "Showers"
    assert payload["track_condition"] == "Slow"
    assert payload["weather"] == "Showers"
    assert payload["weather_condition"] == "Showers"
    assert payload["weather_track_metadata_is_leakage_safe"] is True


def test_provenance_payload_rejects_weather_track_placeholders_and_result_like_source(
    tmp_path,
):
    payload = build_csv_download_provenance_payload(
        filepath=tmp_path / "Race 1 - TEST - 2026-05-29.csv",
        race_url="https://www.thedogs.com.au/results/test/2026-05-29/1/test-race",
        csv_info={"type": "GET", "url": "https://example.test/form.csv"},
        content=_synthetic_form_export([(1, "Alpha Dog")]),
        completeness={"status": "COMPLETE"},
        race_info={
            "race_number": 1,
            "track_condition": "Unknown",
            "weather": "N/A",
            "winner_track_condition": "Fast",
            "odds_weather": "Fine",
            "target_distance": "400m",
            "target_grade": "Maiden",
        },
    )

    assert payload["track_condition"] is None
    assert payload["weather"] is None
    assert payload["weather_track_metadata_is_leakage_safe"] is False
    assert "winner_track_condition" not in payload
    assert "odds_weather" not in payload


def test_prejump_shadow_metadata_fails_closed_for_non_thedogs_race_url(tmp_path):
    runners = [
        (1, "Alpha Runner"),
        (2, "Bravo Runner"),
        (3, "Charlie Runner"),
        (4, "Delta Runner"),
    ]
    content = _synthetic_form_export(runners)
    accepted = tmp_path / "Race 1 - TEST - 2026-05-29.csv"
    sidecar = _synthetic_sidecar()
    sidecar["race_info"].update(
        {"date": "2026-05-29", "venue": "TEST", "race_time": "11:15 AM"}
    )
    normalization = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=accepted,
        raw_export_path=tmp_path / "raw_exports" / accepted.name,
        sidecar_payload=sidecar,
        runner_completeness=analyze_csv_text_runner_completeness(
            content,
            source="synthetic",
        ).as_dict(),
        canonical_runner_set=_canonical_runner_set(runners),
    )

    final_sidecar = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url="https://example.com/racing/test/2026-05-29/1/result",
        csv_info={"type": "GET", "url": "https://example.com/export-expert-form"},
        content=normalization["normalized_content"],
        completeness=analyze_csv_text_runner_completeness(
            normalization["normalized_content"],
            source="synthetic-normalized",
        ),
        race_info={
            **sidecar["race_info"],
            "target_distance": sidecar["target_distance"],
            "target_distance_source": sidecar["target_distance_source"],
            "target_grade": sidecar["target_grade"],
            "target_grade_source": sidecar["target_grade_source"],
            "metadata_is_leakage_safe": sidecar["metadata_is_leakage_safe"],
        },
        normalization={key: value for key, value in normalization.items() if key != "normalized_content"},
        filename=accepted.name,
        allow_generic_fields=False,
    )

    shadow_metadata = final_sidecar["prejump_shadow_metadata"]
    assert shadow_metadata["status"] == "FAIL"
    assert "source_url_not_thedogs" in shadow_metadata["fail_reasons"]


@pytest.mark.parametrize(
    ("source_url", "race_info_override", "expected_safe"),
    [
        (SYNTHETIC_RACE_URL, {}, True),
        (None, {}, False),
        ("https://example.com/racing/test/2026-05-29/1/test-race", {}, False),
        (
            "https://www.thedogs.com.au/racing/test/2026-05-29/1/results",
            {},
            False,
        ),
        (SYNTHETIC_RACE_URL, {"target_grade": None, "grade": None}, False),
        (SYNTHETIC_RACE_URL, {"target_distance": None, "distance": None}, False),
    ],
)
def test_safe_target_metadata_payload_requires_prejump_thedogs_source_and_complete_fields(
    source_url,
    race_info_override,
    expected_safe,
):
    race_info = {
        "target_distance": "400m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade": "Maiden",
        "target_grade_source": "canonical_pre_race_page",
    }
    race_info.update(race_info_override)

    payload = build_safe_target_metadata_payload(
        race_info,
        source_url=source_url,
        allow_generic_fields=False,
    )

    assert payload["metadata_is_leakage_safe"] is expected_safe


def test_provenance_payload_writes_only_safe_weather_track_metadata(tmp_path):
    accepted = tmp_path / "Race 1 - TEST - 2026-05-29.csv"
    race_info = {
        "date": "2026-05-29",
        "venue": "TEST",
        "race_number": 1,
        "race_time": "11:15 AM",
        "url": SYNTHETIC_RACE_URL,
        "target_distance": "400m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade": "Maiden",
        "target_grade_source": "canonical_pre_race_page",
        "track_condition": "Soft",
        "weather_condition": "Overcast",
    }

    payload = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url=SYNTHETIC_RACE_URL,
        csv_info={"type": "GET", "url": f"{SYNTHETIC_RACE_URL}/export-expert-form"},
        content="Dog Name|BOX\n1. Alpha Runner|1\n",
        completeness={"participants": [{"box_number": 1, "dog_name": "Alpha Runner"}]},
        race_info=race_info,
        allow_generic_fields=False,
    )

    assert payload["track_condition"] == "Soft"
    assert payload["weather"] == "Overcast"
    assert payload["weather_condition"] == "Overcast"
    assert payload["weather_track_metadata_source"] == "canonical_pre_race_page"
    assert payload["weather_track_metadata_is_leakage_safe"] is True
    assert payload["race_info"]["track_condition"] == "Soft"
    assert payload["race_info"]["weather_condition"] == "Overcast"

    unsafe_payload = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url="https://www.thedogs.com.au/racing/test/2026-05-29/1/results",
        csv_info={"type": "GET", "url": f"{SYNTHETIC_RACE_URL}/export-expert-form"},
        content="Dog Name|BOX\n1. Alpha Runner|1\n",
        completeness={"participants": [{"box_number": 1, "dog_name": "Alpha Runner"}]},
        race_info={**race_info, "track_condition": "0.0", "weather_condition": "20.0"},
        allow_generic_fields=False,
    )

    assert unsafe_payload["track_condition"] is None
    assert unsafe_payload["weather"] is None
    assert unsafe_payload["weather_track_metadata_is_leakage_safe"] is False
    assert "source_url_looks_post_result" in unsafe_payload[
        "rejected_weather_track_metadata_sources"
    ]


def test_provenance_payload_accepts_combined_sportsbet_track_and_forecast_weather(tmp_path):
    accepted = tmp_path / "Race 9 - SAL - 2099-06-17.csv"
    race_url = "https://www.thedogs.com.au/racing/sale/2099-06-17/9/test?trial=false"
    sportsbet_url = (
        "https://www.sportsbet.com.au/apigw/sportsbook-racing/"
        "Sportsbook/Racing/NextEvents?racingFilters=GH_DOMESTIC"
    )
    weather_url = "https://api.open-meteo.com/v1/forecast?latitude=-38.10&longitude=147.07"
    race_info = {
        "date": "2099-06-17",
        "venue": "SAL",
        "race_number": 9,
        "race_time": "1:57 PM",
        "url": race_url,
        "target_distance": "435m",
        "target_distance_source": "canonical_pre_race_page",
        "target_grade": "Grade 5",
        "target_grade_source": "canonical_pre_race_page",
        "track_condition": "Good",
        "weather_condition": "Overcast",
        "weather_track_metadata_source": (
            "sportsbet_pre_race_page+open_meteo_forecast_api"
        ),
        "weather_track_metadata_source_url": {
            "sportsbet_pre_race_page": sportsbet_url,
            "open_meteo_forecast_api": weather_url,
        },
        "weather_track_metadata_is_leakage_safe": True,
    }

    payload = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url=race_url,
        csv_info={"type": "GET", "url": f"{race_url}/export-expert-form"},
        content="Dog Name|BOX\n1. Alpha Runner|1\n",
        completeness={"participants": [{"box_number": 1, "dog_name": "Alpha Runner"}]},
        race_info=race_info,
        allow_generic_fields=False,
    )

    accepted.write_text("Dog Name|BOX\n1. Alpha Runner|1\n", encoding="utf-8")
    accepted.with_name(accepted.name + ".metadata.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    safe = load_safe_weather_track_metadata(accepted)

    assert payload["weather_track_metadata_is_leakage_safe"] is True
    assert payload["weather_track_metadata_source"] == (
        "sportsbet_pre_race_page+open_meteo_forecast_api"
    )
    assert safe["track_condition"] == "Good"
    assert safe["weather"] == "Overcast"
    assert safe["weather_track_metadata_source_url"] == {
        "sportsbet_pre_race_page": sportsbet_url,
        "open_meteo_forecast_api": weather_url,
    }


def test_prejump_shadow_metadata_fails_closed_for_unsafe_canonical_runner_url(tmp_path):
    runners = [
        (1, "Alpha Runner"),
        (2, "Bravo Runner"),
        (3, "Charlie Runner"),
        (4, "Delta Runner"),
    ]
    content = _synthetic_form_export(runners)
    accepted = tmp_path / "Race 1 - TEST - 2026-05-29.csv"
    sidecar = _synthetic_sidecar()
    sidecar["race_info"].update(
        {"date": "2026-05-29", "venue": "TEST", "race_time": "11:15 AM"}
    )
    normalization = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=accepted,
        raw_export_path=tmp_path / "raw_exports" / accepted.name,
        sidecar_payload=sidecar,
        runner_completeness=analyze_csv_text_runner_completeness(
            content,
            source="synthetic",
        ).as_dict(),
        canonical_runner_set=_canonical_runner_set(runners),
    )
    normalization["canonical_runner_alignment"]["canonical_source_url"] = (
        "https://www.thedogs.com.au/racing/test/2026-05-29/1/results"
    )

    final_sidecar = build_csv_download_provenance_payload(
        filepath=accepted,
        race_url=SYNTHETIC_RACE_URL,
        csv_info={"type": "GET", "url": f"{SYNTHETIC_RACE_URL}/export-expert-form"},
        content=normalization["normalized_content"],
        completeness=analyze_csv_text_runner_completeness(
            normalization["normalized_content"],
            source="synthetic-normalized",
        ),
        race_info={
            **sidecar["race_info"],
            "target_distance": sidecar["target_distance"],
            "target_distance_source": sidecar["target_distance_source"],
            "target_grade": sidecar["target_grade"],
            "target_grade_source": sidecar["target_grade_source"],
            "metadata_is_leakage_safe": sidecar["metadata_is_leakage_safe"],
        },
        normalization={key: value for key, value in normalization.items() if key != "normalized_content"},
        filename=accepted.name,
        allow_generic_fields=False,
    )

    shadow_metadata = final_sidecar["prejump_shadow_metadata"]
    assert shadow_metadata["status"] == "FAIL"
    assert "canonical_runner_source_url_looks_post_result" in shadow_metadata[
        "fail_reasons"
    ]


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


def test_canonical_final_runner_missing_from_source_rejects_normalization(tmp_path):
    content = _synthetic_form_export(
        [
            (1, "Alpha Runner"),
            (2, "Bravo Runner"),
            (3, "Charlie Runner"),
            (4, "Delta Runner"),
        ]
    )
    runner_completeness = analyze_csv_text_runner_completeness(content).as_dict()

    result = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=tmp_path / "Race 1 - TST - 2026-05-29.csv",
        raw_export_path=tmp_path / "raw_exports/Race 1 - TST - 2026-05-29.csv",
        sidecar_payload=_synthetic_sidecar(),
        runner_completeness=runner_completeness,
        canonical_runner_set=_canonical_runner_set(
            [
                (1, "Alpha Runner"),
                (2, "Bravo Runner"),
                (3, "Charlie Runner"),
                (4, "Delta Runner"),
                (5, "Echo Runner"),
            ]
        ),
    )

    assert result["normalization_status"] == "rejected"
    assert (
        "final_runner_set_not_aligned:canonical_participant_missing_from_source_csv"
        in result["normalization_failure_reason"]
    )
    assert result["normalization_verification"][
        "canonical_runner_alignment_status"
    ] == "not_aligned"
    assert result["canonical_runner_alignment"]["missing_canonical_participants"] == [
        {
            "box_number": 5,
            "dog_name": "Echo Runner",
            "original_box_number": None,
        }
    ]
    assert result["normalized_content"] is None


def test_canonical_final_runner_alignment_normalizes_final_starters(tmp_path):
    content = _synthetic_form_export(
        [
            (1, "Alpha Runner"),
            (2, "Bravo Runner"),
            (3, "Charlie Runner"),
            (4, "Scratched Runner"),
            (9, "Reserve Runner"),
        ]
    )
    runner_completeness = analyze_csv_text_runner_completeness(content).as_dict()
    canonical = _canonical_runner_set(
        [
            (1, "Alpha Runner"),
            (2, "Bravo Runner"),
            (3, "Charlie Runner"),
            (4, "Reserve Runner"),
        ]
    )
    canonical["final_runner_participants"][-1]["original_box_number"] = 9
    canonical["scratched_boxes"] = [4]
    canonical["scratched_participants"] = [
        {"box_number": 4, "dog_name": "Scratched Runner"}
    ]
    canonical["reserve_boxes"] = [9]

    result = normalize_verified_thedogs_export_content(
        content,
        accepted_csv_path=tmp_path / "Race 1 - TST - 2026-05-29.csv",
        raw_export_path=tmp_path / "raw_exports/Race 1 - TST - 2026-05-29.csv",
        sidecar_payload=_synthetic_sidecar(),
        runner_completeness=runner_completeness,
        canonical_runner_set=canonical,
    )

    assert result["normalization_status"] == "verified", result
    assert result["raw_content_sha256"]
    assert result["normalization_verification"][
        "canonical_runner_alignment_status"
    ] == "aligned"
    assert result["normalization_verification"]["runner_set_status"] == "COMPLETE"
    assert "4. Reserve Runner" in result["normalized_content"]
    assert "4. Scratched Runner" not in result["normalized_content"]
    assert "9. Reserve Runner" not in result["normalized_content"]
    assert result["canonical_runner_alignment"]["remapped_participants"] == [
        {
            "dog_name": "Reserve Runner",
            "source_box_number": 9,
            "final_box_number": 4,
            "original_box_number": 9,
        }
    ]
