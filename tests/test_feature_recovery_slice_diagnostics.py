import json

import pytest

from scripts.run_feature_recovery_execution_v1 import (
    REPAIRED_SLICE_DIMENSIONS,
    build_repaired_slice_population_diagnostics,
    safe_output_dir,
    slice_population_csv_rows,
)


def test_stage2_slice_population_diagnostics_reports_context_buckets_and_family_rates():
    features = [
        "target_distance_safe",
        "target_distance_source_is_safe",
        "target_grade_safe",
        "target_grade_provenance_safe",
        "prior_start_count",
        "recent_avg_speed_mps_5",
        "safe_field_strength",
        "target_box_band_prior_start_count",
        "venue_box_band_start_count",
        "distance_box_band_start_count",
        "starts_same_distance",
        "same_distance_same_grade_best_time",
        "same_grade_start_count",
        "recent_avg_sectional_1st_5",
    ]
    schema = {
        "feature_families": {
            "safe_target_context": [
                "target_distance_safe",
                "target_distance_source_is_safe",
                "target_grade_safe",
                "target_grade_provenance_safe",
            ],
            "repaired_history_reconstruction": [
                "prior_start_count",
                "recent_avg_speed_mps_5",
            ],
            "class_and_field_strength": ["safe_field_strength"],
            "draw_adjusted_history": [
                "target_box_band_prior_start_count",
                "venue_box_band_start_count",
                "distance_box_band_start_count",
            ],
            "same_distance": [
                "starts_same_distance",
                "same_distance_same_grade_best_time",
            ],
            "same_grade_and_grade_transition": ["same_grade_start_count"],
            "sectional_metrics": ["recent_avg_sectional_1st_5"],
        }
    }
    rows = [
        {
            "race_id": "race-1",
            "snapshot_instance_id": "race-1",
            "venue": "SHEP",
            "field_size": 8,
            "box_number": 1,
            "target_distance_safe": 390,
            "target_distance_band_sprint": 1,
            "target_grade_safe": "Grade 5",
            "target_distance_source_is_safe": 1,
            "target_grade_provenance_safe": 1,
            "prior_start_count": 12,
            "recent_avg_speed_mps_5": 17.2,
            "safe_field_strength": 56,
            "target_box_band_prior_start_count": 4,
            "venue_box_band_start_count": 2,
            "distance_box_band_start_count": 3,
            "starts_same_distance": 5,
            "same_distance_same_grade_best_time": 22.14,
            "same_grade_start_count": 7,
            "recent_avg_sectional_1st_5": 6.61,
        },
        {
            "race_id": "race-1",
            "snapshot_instance_id": "race-1",
            "venue": "SHEP",
            "field_size": 8,
            "box_number": 8,
            "target_distance_safe": 390,
            "target_distance_band_sprint": 1,
            "target_grade_safe": "Grade 5",
            "target_distance_source_is_safe": 1,
            "target_grade_provenance_safe": 1,
            "prior_start_count": 3,
            "recent_avg_speed_mps_5": "",
            "safe_field_strength": 56,
            "target_box_band_prior_start_count": "",
            "venue_box_band_start_count": "",
            "distance_box_band_start_count": 1,
            "starts_same_distance": 1,
            "same_distance_same_grade_best_time": "",
            "same_grade_start_count": 2,
            "recent_avg_sectional_1st_5": "",
        },
        {
            "race_id": "race-2",
            "snapshot_instance_id": "race-2",
            "venue": "BAL",
            "field_size": 6,
            "box_number": 3,
            "target_distance_safe": "",
            "target_grade_safe": "",
            "prior_start_count": 0,
            "safe_field_strength": "",
            "starts_same_distance": "",
        },
    ]

    report = build_repaired_slice_population_diagnostics(rows, features, schema)

    assert report["schema_version"] == "stage2_repaired_slice_population_diagnostics_v1"
    assert report["row_count"] == 3
    assert report["race_count"] == 2
    assert set(report["dimensions"]) == set(REPAIRED_SLICE_DIMENSIONS)
    assert report["dimensions"]["target_distance_band"]["missing_row_count"] == 1

    venue_bucket = report["dimensions"]["venue"]["buckets"]["SHEP"]
    assert venue_bucket["row_count"] == 2
    assert venue_bucket["race_count"] == 1
    assert venue_bucket["family_population"]["safe_target_context"][
        "avg_present_pct"
    ] == pytest.approx(1.0)
    assert venue_bucket["family_population"]["draw_adjusted_history"][
        "key_feature_present_pct"
    ]["venue_box_band_start_count"] == pytest.approx(0.5)

    assert "sprint|inside" in report["dimensions"]["distance_box_band"]["buckets"]
    missing_target = report["dimensions"]["target_grade"]["buckets"]["DATA_MISSING"]
    assert missing_target["family_population"]["safe_target_context"][
        "key_feature_present_rows"
    ]["target_grade_safe"] == 0

    csv_rows = slice_population_csv_rows(report)
    assert {
        "dimension",
        "bucket",
        "family",
        "row_count",
        "race_count",
        "avg_present_pct",
        "key_feature_present_pct",
    }.issubset(csv_rows[0])
    key_rates = json.loads(
        next(
            item
            for item in csv_rows
            if item["dimension"] == "venue"
            and item["bucket"] == "SHEP"
            and item["family"] == "draw_adjusted_history"
        )["key_feature_present_pct"]
    )
    assert key_rates["distance_box_band_start_count"] == pytest.approx(1.0)


def test_feature_recovery_output_dir_guard_rejects_artifact_symlink_escape(
    tmp_path, monkeypatch
):
    import scripts.run_feature_recovery_execution_v1 as recovery

    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}_outside"
    outside.mkdir()
    link = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "feature_recovery_execution_v1_symlink_report_only"
    )
    link.parent.mkdir(parents=True)
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        safe_output_dir(link)
