import csv
import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_no_box_downstream_diagnostics as diagnostics
from scripts.build_no_box_downstream_diagnostics import (
    _assert_output_dir_safe,
    build_outputs,
)


def _artifact_output_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "full_evidence_orchestration_20260525" / "no-box-diagnostics"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sqlite_fixture(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            "create table race_metadata (race_id text, distance real, grade text, "
            "winner_source text, results_status text)"
        )
        conn.execute(
            "create table dog_race_data (race_id text, dog_name text, box_number integer, "
            "data_source text)"
        )
        conn.execute(
            "insert into race_metadata values ('RACE_1', 515, 'Grade 5', 'fixture', 'resulted')"
        )
        conn.executemany(
            "insert into dog_race_data values (?, ?, ?, ?)",
            [
                ("RACE_1", "Alpha", 1, "fixture"),
                ("RACE_1", "Bravo", 8, "fixture"),
            ],
        )


def test_no_box_downstream_diagnostics_writes_report_only_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)
    predictions_csv = tmp_path / "predictions.csv"
    feature_rows_csv = tmp_path / "features.csv"
    rolling_report_json = tmp_path / "rolling_report.json"
    feature_join_json = tmp_path / "feature_join.json"
    db_path = tmp_path / "metadata.sqlite"
    output_root = _artifact_output_root(tmp_path)

    _write_csv(
        predictions_csv,
        [
            {
                "window_id": "w1",
                "race_id": "RACE_1",
                "race_date": "2026-01-01",
                "venue": "TEST",
                "race_number": "4",
                "dog_name": "Alpha",
                "dog_name_key": "alpha",
                "actual_win": "0",
                "predicted_rank": "1",
                "score": "0.8",
                "field_scope": "no_box",
                "feature_count": "6",
            },
            {
                "window_id": "w1",
                "race_id": "RACE_1",
                "race_date": "2026-01-01",
                "venue": "TEST",
                "race_number": "4",
                "dog_name": "Bravo",
                "dog_name_key": "bravo",
                "actual_win": "1",
                "predicted_rank": "2",
                "score": "0.2",
                "field_scope": "no_box",
                "feature_count": "6",
            },
        ],
    )
    _write_csv(
        feature_rows_csv,
        [
            {
                "race_id": "RACE_1",
                "dog_name": "Alpha",
                "dog_name_key": "alpha",
                "feature_recent_score": "1.2",
                "feature_starts_same_distance": "3",
                "feature_grade_delta": "0",
                "feature_days_since_last_start": "7",
                "global_prior_history_count": "2",
                "global_prior_history_values_filled": "4",
            },
            {
                "race_id": "RACE_1",
                "dog_name": "Bravo",
                "dog_name_key": "bravo",
                "feature_recent_score": "0.5",
                "feature_starts_same_distance": "2",
                "feature_grade_delta": "1",
                "feature_days_since_last_start": "10",
                "global_prior_history_count": "1",
                "global_prior_history_values_filled": "4",
            },
        ],
    )
    _write_json(
        rolling_report_json,
        {
            "validation": {"status": "PASS", "usable_feature_count": 6},
            "aggregate_metrics": {
                "expected_random_top1": 0.5,
                "expected_random_top3": 1.0,
            },
        },
    )
    _write_json(feature_join_json, {"status": "fixture"})
    _sqlite_fixture(db_path)

    result = build_outputs(
        predictions_csv=predictions_csv,
        feature_rows_csv=feature_rows_csv,
        rolling_report_json=rolling_report_json,
        feature_join_json=feature_join_json,
        stratified_csv=None,
        stratified_json=None,
        db_path=db_path,
        output_root=output_root,
        tag="fixture",
    )

    assert result["status"] == "REPORT_ONLY_DOWNSTREAM_DIAGNOSTICS_COMPLETE"
    assert result["aggregate"]["evaluated_races"] == 1
    assert result["db_summary"]["quick_check"] == "ok"
    failure_report = (
        output_root
        / "no_box_pairwise_rolling_failure_surface_fixture"
        / "failure_surface_report.json"
    )
    payload = json.loads(failure_report.read_text(encoding="utf-8"))
    assert all(value is False for value in payload["writes_performed"].values())
    assert (
        output_root
        / "stratified_error_priority_digest_fixture"
        / "stratified_error_priority_digest_report.json"
    ).exists()


def test_no_box_diagnostics_rejects_absolute_output_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path / "repo")

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        _assert_output_dir_safe(
            tmp_path / "outside" / "artifacts" / "full_evidence_orchestration_20260525"
        )


def test_no_box_diagnostics_rejects_in_repo_non_artifact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        _assert_output_dir_safe(tmp_path / "reports" / "no-box-diagnostics")
