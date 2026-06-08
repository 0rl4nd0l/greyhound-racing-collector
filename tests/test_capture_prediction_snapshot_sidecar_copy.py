import json

from scripts.capture_prediction_snapshot import _copy_metadata_sidecar_for_prediction_input
from utils.csv_metadata import load_safe_sidecar_target_metadata


SAFE_THEDOGS_RACE_URL = "https://www.thedogs.com.au/racing/test/2026-06-03/1/test-race"


def test_aligned_prediction_input_receives_original_metadata_sidecar(tmp_path):
    source_csv = tmp_path / "Race 1 - TEST - 2026-06-03.csv"
    source_csv.write_text("Dog Name|BOX\n1. Alpha|1\n", encoding="utf-8")
    source_sidecar = tmp_path / "Race 1 - TEST - 2026-06-03.csv.metadata.json"
    source_sidecar.write_text(
        json.dumps(
            {
                "race_url": SAFE_THEDOGS_RACE_URL,
                "metadata_is_leakage_safe": True,
                "target_distance": "525m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
            }
        ),
        encoding="utf-8",
    )
    prediction_csv = tmp_path / "aligned" / source_csv.name
    prediction_csv.parent.mkdir()
    prediction_csv.write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")

    result = _copy_metadata_sidecar_for_prediction_input(
        source_csv=source_csv,
        prediction_csv=prediction_csv,
    )

    assert result["status"] == "copied"
    copied = prediction_csv.with_suffix(prediction_csv.suffix + ".metadata.json")
    assert copied.exists()
    safe = load_safe_sidecar_target_metadata(prediction_csv)
    assert safe["target_distance"] == "525m"
    assert safe["target_grade"] == "Grade 5"
    assert safe["metadata_is_leakage_safe"] is True


def test_sidecar_copy_reports_missing_source_sidecar_without_writing(tmp_path):
    source_csv = tmp_path / "Race 1 - TEST - 2026-06-03.csv"
    source_csv.write_text("Dog Name|BOX\n1. Alpha|1\n", encoding="utf-8")
    prediction_csv = tmp_path / "aligned" / source_csv.name
    prediction_csv.parent.mkdir()
    prediction_csv.write_text(source_csv.read_text(encoding="utf-8"), encoding="utf-8")

    result = _copy_metadata_sidecar_for_prediction_input(
        source_csv=source_csv,
        prediction_csv=prediction_csv,
    )

    assert result["status"] == "source_sidecar_missing"
    assert not prediction_csv.with_suffix(prediction_csv.suffix + ".metadata.json").exists()


def test_safe_sidecar_loader_rejects_non_thedogs_target_metadata_url(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-03.csv"
    csv_path.write_text("Dog Name|BOX\n1. Alpha|1\n", encoding="utf-8")
    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "race_url": "https://example.com/racing/test/2026-06-03/1/test-race",
                "metadata_is_leakage_safe": True,
                "target_distance": "525m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
            }
        ),
        encoding="utf-8",
    )

    safe = load_safe_sidecar_target_metadata(csv_path)

    assert safe["target_distance"] is None
    assert safe["target_grade"] is None
    assert safe["metadata_is_leakage_safe"] is False
    assert "source_url_not_thedogs" in safe["rejected_metadata_sources"]


def test_safe_sidecar_loader_rejects_failed_prejump_shadow_contract(tmp_path):
    csv_path = tmp_path / "Race 1 - TEST - 2026-06-03.csv"
    csv_path.write_text("Dog Name|BOX\n1. Alpha|1\n", encoding="utf-8")
    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")
    sidecar_path.write_text(
        json.dumps(
            {
                "race_url": SAFE_THEDOGS_RACE_URL,
                "metadata_is_leakage_safe": True,
                "target_distance": "525m",
                "target_distance_source": "canonical_pre_race_page",
                "target_grade": "Grade 5",
                "target_grade_source": "canonical_pre_race_page",
                "prejump_shadow_metadata": {
                    "status": "FAIL",
                    "fail_reasons": ["canonical_runner_alignment_missing"],
                    "metadata_is_leakage_safe": True,
                    "source_url": SAFE_THEDOGS_RACE_URL,
                },
            }
        ),
        encoding="utf-8",
    )

    safe = load_safe_sidecar_target_metadata(csv_path)

    assert safe["target_distance"] is None
    assert safe["target_grade"] is None
    assert safe["metadata_is_leakage_safe"] is False
    assert (
        "prejump_shadow_metadata_failed:canonical_runner_alignment_missing"
        in safe["rejected_metadata_sources"]
    )
