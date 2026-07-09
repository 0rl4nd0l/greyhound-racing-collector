import json
from pathlib import Path

import pytest

from scripts.analyze_no_box_dog_form_feature_coverage import (
    analyze_feature_coverage,
    feature_family,
    write_outputs,
)


def _packet() -> dict:
    return {
        "schema_version": "no_box_actual_win_dog_form_feature_join_v1",
        "status": "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY",
        "report_only": True,
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
        },
        "summary": {
            "history_db_fill_policy": "no_outcome_proxy_fields",
            "label_proxy_audit": {"status": "PASS"},
        },
    }


def _row(race: int, dog: str, actual_win: int, speed: float | None) -> dict:
    return {
        "race_id": f"R{race:02d}",
        "identity_key": f"2025-01-{race:02d}|TEST|R{race:02d}",
        "race_date": f"2025-01-{race:02d}",
        "venue": "TEST",
        "race_number": race,
        "dog_name_key": dog.lower(),
        "dog_name": dog,
        "actual_win": actual_win,
        "candidate_kind": "partial_field",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": False,
        "race_grouped_actual_win_ranking_allowed": False,
        "target_source": "official_winner_name_metadata_confirmed",
        "label_scope": "actual_win_only",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
        "feature_join_status": "MATCHED",
        "feature_recent_win_rate_5": speed,
        "feature_recent_place_rate_5": speed,
        "feature_days_since_last_start": 7 if speed is not None else None,
        "feature_starts_same_venue": 2 if speed is not None else None,
        "feature_recent_finish_mean_3": 2.0 if speed is not None else None,
    }


def _rows() -> list[dict]:
    rows = []
    for race in range(1, 7):
        if race <= 3:
            rows.extend(
                [
                    _row(race, f"Fast {race}", 1, 0.9),
                    _row(race, f"Slow {race}", 0, 0.2),
                ]
            )
        else:
            rows.extend(
                [
                    _row(race, f"Fast {race}", 1, None),
                    _row(race, f"Slow {race}", 0, 0.4),
                ]
            )
    return rows


def _predictions() -> list[dict]:
    predictions = []
    for row in _rows()[4:]:
        prediction = dict(row)
        race = int(str(row["race_id"]).removeprefix("R"))
        if race <= 3:
            prediction["predicted_rank"] = 1 if row["actual_win"] else 2
        else:
            prediction["predicted_rank"] = 2 if row["actual_win"] else 1
        prediction["window_id"] = "window_01"
        prediction["score"] = 1.0 / prediction["predicted_rank"]
        predictions.append(prediction)
    return predictions


def test_feature_family_buckets_named_dog_form_surfaces():
    assert feature_family("feature_recent_win_rate_5") == "recent_win_place"
    assert feature_family("feature_win_rate_same_distance") == "same_distance"
    assert feature_family("feature_starts_same_venue") == "venue_history"
    assert feature_family("feature_grade_change_indicator") == "grade_movement"
    assert feature_family("feature_days_since_last_start") == "recency"
    assert feature_family("feature_recent_finish_mean_3") == "finish_trend_excluded"
    assert feature_family("feature_recent_avg_time_5") == "time_trend"


def test_feature_coverage_reports_family_miss_correlation_and_no_writes(
    tmp_path: Path, monkeypatch
):
    import scripts.analyze_no_box_dog_form_feature_coverage as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    rolling_report = {
        "validation": {
            "usable_feature_columns": [
                "feature_recent_win_rate_5",
                "feature_recent_place_rate_5",
                "feature_days_since_last_start",
                "feature_starts_same_venue",
            ]
        },
        "aggregate_metrics": {
            "top1_accuracy": 0.5,
            "top3_hit_rate": 1.0,
            "mean_winner_rank": 1.5,
        },
    }
    report, family_rows, column_rows, miss_examples = analyze_feature_coverage(
        feature_join_packet=_packet(),
        feature_rows=_rows(),
        rolling_report=rolling_report,
        rolling_predictions=_predictions(),
        expected_races=6,
    )

    assert report["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_COMPLETE"
    assert report["writes_performed"]["db_write"] is False
    assert report["writes_performed"]["label_write"] is False
    assert report["safe_to_write_now"] is False
    assert report["summary"]["race_count"] == 6
    assert report["summary"]["rolling_evaluated_races"] == 4
    assert report["summary"]["rolling_top1_misses"] == 3
    assert "actual_win_race_count_below_50" in report["blockers"]
    recent = next(row for row in family_rows if row["family"] == "recent_win_place")
    assert recent["ranker_usable_feature_count"] == 2
    assert recent["winner_usable_feature_share_miss_minus_hit"] < 0
    finish = next(row for row in family_rows if row["family"] == "finish_trend_excluded")
    assert finish["ranker_usable_feature_count"] == 0
    assert "finish_trend_excluded" in report["summary"]["families_without_current_ranker_usable_features"]
    assert any(row["feature"] == "feature_recent_win_rate_5" for row in column_rows)
    assert miss_examples

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/coverage"
    write_outputs(output_dir, report, family_rows, column_rows, miss_examples)
    written = json.loads((output_dir / "dog_form_feature_coverage_report.json").read_text())
    assert written["schema_version"] == "no_box_dog_form_feature_coverage_audit_v1"
    assert (output_dir / "dog_form_feature_family_coverage.csv").exists()
    assert (output_dir / "dog_form_feature_column_coverage.csv").exists()
    assert (output_dir / "dog_form_top1_miss_feature_examples.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_coverage"
    )
    write_outputs(relative_output_dir, report, family_rows, column_rows, miss_examples)
    assert (tmp_path / relative_output_dir / "dog_form_feature_coverage_report.json").exists()
    assert not (cwd / relative_output_dir / "dog_form_feature_coverage_report.json").exists()


def test_feature_coverage_fails_closed_outside_artifacts(tmp_path: Path, monkeypatch):
    import scripts.analyze_no_box_dog_form_feature_coverage as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    report, family_rows, column_rows, miss_examples = analyze_feature_coverage(
        feature_join_packet=_packet(),
        feature_rows=_rows(),
        rolling_predictions=[],
        expected_races=6,
    )

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", report, family_rows, column_rows, miss_examples)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/coverage",
            report,
            family_rows,
            column_rows,
            miss_examples,
        )


def test_feature_coverage_fails_contract_on_box_feature():
    rows = _rows()
    rows[0]["feature_box_number"] = 1

    report, _, _, _ = analyze_feature_coverage(
        feature_join_packet=_packet(),
        feature_rows=rows,
        rolling_predictions=[],
        expected_races=6,
    )

    assert report["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_COVERAGE_AUDIT_FAILED_CONTRACT"
    assert report["validation"]["status"] == "FAIL"
    assert any(
        "forbidden_numeric_features_present:feature_box_number:box_feature" in item
        for item in report["validation"]["failures"]
    )
