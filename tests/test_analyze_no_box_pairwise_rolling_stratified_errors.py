import json
import sqlite3
from pathlib import Path

import pytest

from scripts.analyze_no_box_pairwise_rolling_stratified_errors import (
    build_stratified_error_analysis,
    main,
    write_outputs,
)


def _prediction_row(
    *,
    race_id: str,
    window_id: str,
    dog: str,
    actual_win: int,
    rank: int,
    venue: str = "TEST",
) -> dict:
    return {
        "model": "report_local_pairwise_logistic_ranker",
        "window_id": window_id,
        "race_id": race_id,
        "race_date": "2025-01-01",
        "venue": venue,
        "race_number": 1,
        "dog_name_key": dog,
        "dog_name": dog.title(),
        "predicted_rank": rank,
        "actual_win": actual_win,
        "candidate_kind": "partial_field",
        "field_scope": "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": False,
        "feature_join_status": "MATCHED",
        "history_feature_join_status": "MATCHED",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
    }


def _rows() -> list[dict]:
    return [
        _prediction_row(race_id="R1", window_id="window_01", dog="fast dog", actual_win=1, rank=1),
        _prediction_row(race_id="R1", window_id="window_01", dog="slow dog", actual_win=0, rank=2),
        _prediction_row(race_id="R2", window_id="window_01", dog="wide dog", actual_win=1, rank=4),
        _prediction_row(race_id="R2", window_id="window_01", dog="other dog", actual_win=0, rank=1),
    ]


def _db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table race_metadata (
            race_id text,
            distance text,
            grade text,
            winner_name text,
            winner_source text,
            results_status text
        )
        """
    )
    conn.execute(
        """
        create table dog_race_data (
            race_id text,
            dog_name text,
            box_number integer,
            data_source text
        )
        """
    )
    conn.executemany(
        "insert into race_metadata values (?,?,?,?,?,?)",
        [
            ("R1", "395", "5", "Fast Dog", "thedogs_official", "complete"),
            ("R2", "650", "5", "Wide Dog", None, "complete"),
        ],
    )
    conn.executemany(
        "insert into dog_race_data values (?,?,?,?)",
        [
            ("R1", "Fast Dog", 1, "thedogs_official"),
            ("R1", "Slow Dog", 8, None),
            ("R2", "Wide Dog", 8, "current_db"),
            ("R2", "Other Dog", 2, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def test_rolling_stratified_analysis_enriches_distance_box_and_source(tmp_path: Path):
    report = {"status": "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED"}
    analysis = build_stratified_error_analysis(
        rolling_report=report,
        prediction_rows=_rows(),
        db_path=_db(tmp_path / "greyhound.db"),
    )

    assert analysis["status"] == "REPORT_ONLY_PAIRWISE_ROLLING_STRATIFIED_ERROR_ANALYSIS"
    assert analysis["writes_performed"]["db_write"] is False
    assert analysis["db_enrichment"]["quick_check"] == "ok"
    model = analysis["models"]["report_local_pairwise_logistic_ranker"]
    assert model["overall"]["top1_accuracy"] == 0.5
    assert model["dimensions"]["distance_bucket"]["sprint_lt_400"]["top1_accuracy"] == 1.0
    assert model["dimensions"]["winner_box_bucket"]["outside_6_plus"]["top1_miss_count"] == 1
    assert model["dimensions"]["source_bucket"]["current_db"]["top1_miss_count"] == 1
    assert analysis["missing_dimension_counts"] == {
        "distance": 0,
        "source_bucket": 0,
        "winner_box": 0,
    }


def test_rolling_stratified_analysis_fails_closed_on_prediction_box_field():
    rows = _rows()
    rows[0]["box_number"] = 1
    analysis = build_stratified_error_analysis(
        rolling_report={},
        prediction_rows=rows,
    )

    assert analysis["status"] == (
        "REPORT_ONLY_PAIRWISE_ROLLING_STRATIFIED_ERROR_ANALYSIS_WITH_FAILURES"
    )
    assert any("forbidden_prediction_row_fields" in item for item in analysis["failures"])


def test_rolling_stratified_cli_writes_outputs(tmp_path: Path, monkeypatch):
    import scripts.analyze_no_box_pairwise_rolling_stratified_errors as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    rows_path = tmp_path / "predictions.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    report_path = tmp_path / "rolling_report.json"
    report_path.write_text(json.dumps({"status": "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED"}))
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/stratified"

    status = main(
        [
            "--rolling-report",
            str(report_path),
            "--predictions-jsonl",
            str(rows_path),
            "--db",
            str(_db(tmp_path / "greyhound.db")),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    assert (output_dir / "no_box_pairwise_rolling_stratified_error_analysis.json").exists()
    assert (output_dir / "no_box_pairwise_rolling_stratified_error_analysis.csv").exists()
    assert (output_dir / "SUMMARY.md").exists()


def test_rolling_stratified_output_guard_fails_closed(tmp_path: Path, monkeypatch):
    import scripts.analyze_no_box_pairwise_rolling_stratified_errors as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    analysis = {
        "status": "REPORT_ONLY_PAIRWISE_ROLLING_STRATIFIED_ERROR_ANALYSIS",
        "summary": {},
        "missing_dimension_counts": {},
        "csv_rows": [],
    }

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        write_outputs(tmp_path / "outside", analysis)
    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        write_outputs(
            tmp_path.parent
            / "outside"
            / "artifacts/full_evidence_orchestration_20260525/stratified",
            analysis,
        )

    cwd = tmp_path / "caller_cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    relative_output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/relative_stratified"
    )
    write_outputs(relative_output_dir, analysis)
    assert (
        tmp_path
        / relative_output_dir
        / "no_box_pairwise_rolling_stratified_error_analysis.json"
    ).exists()
    assert not (
        cwd
        / relative_output_dir
        / "no_box_pairwise_rolling_stratified_error_analysis.json"
    ).exists()
