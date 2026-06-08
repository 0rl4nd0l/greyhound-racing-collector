import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from accuracy_program.odds_coverage import analyze_odds_coverage, normalize_dog_name
from accuracy_program.snapshots import assert_no_result_fields, build_prediction_snapshot
from scripts.evaluate_prediction_snapshots import evaluate_snapshots
from sportsbet_odds_integrator import SportsbetOddsIntegrator


def _build_odds_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE live_odds (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            race_time TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            odds_decimal REAL,
            odds_fractional TEXT,
            market_type TEXT,
            source TEXT,
            timestamp TEXT,
            is_current INTEGER,
            topN INTEGER
        );
        CREATE TABLE odds_history (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_clean_name TEXT,
            odds_decimal REAL,
            odds_change REAL,
            timestamp TEXT,
            source TEXT
        );
        CREATE TABLE race_metadata (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            start_datetime TEXT
        );
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER
        );
        """
    )
    conn.executemany(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, start_datetime)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            ("R1", "Wentworth Park", 1, "2026-05-24", "2026-05-24T10:30:00"),
            ("R2", "Sandown", 2, "2026-05-24", "2026-05-24T10:45:00"),
            ("R4", "Geelong", 4, "2026-05-24", "2026-05-24T11:30:00"),
        ],
    )
    conn.executemany(
        """
        INSERT INTO dog_race_data
            (race_id, dog_name, dog_clean_name, box_number)
        VALUES (?, ?, ?, ?)
        """,
        [
            ("R1", "Alpha Runner", "Alpha Runner", 1),
            ("R1", "Beta Runner", "Beta Runner", 2),
            ("R2", "Gamma Runner", "Gamma Runner", 3),
            ("R2", "Gamma Runner", "Gamma Runner", 3),
            ("R4", "Delta Runner", "Delta Runner", 4),
        ],
    )
    conn.executemany(
        """
        INSERT INTO live_odds
            (id, race_id, venue, race_number, race_date, dog_name,
             dog_clean_name, box_number, odds_decimal, market_type, source,
             timestamp, is_current)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                1,
                "R1",
                "Wentworth Park",
                1,
                "2026-05-24",
                "1. Alpha-Runner",
                "1. Alpha-Runner",
                1,
                3.0,
                "win",
                "sportsbet",
                "2026-05-24T09:45:00",
                1,
            ),
            (
                2,
                "R1",
                "Wentworth Park",
                1,
                "2026-05-24",
                "2. Beta Runner",
                "2. Beta Runner",
                2,
                4.0,
                "win",
                "sportsbet",
                "2026-05-24T08:00:00",
                1,
            ),
            (
                3,
                "R2",
                "Sandown",
                2,
                "2026-05-24",
                "Gamma Runner",
                "Gamma Runner",
                3,
                2.5,
                "win",
                "sportsbet",
                "2026-05-24T09:50:00",
                1,
            ),
            (
                4,
                "BAD_RACE_ID",
                "Geelong",
                4,
                "2026-05-24",
                "Delta Runner",
                "Delta Runner",
                4,
                5.0,
                "win",
                "sportsbet",
                "2026-05-24T09:55:00",
                1,
            ),
        ],
    )
    conn.execute(
        """
        INSERT INTO odds_history
            (race_id, dog_clean_name, odds_decimal, odds_change, timestamp, source)
        VALUES ('R1', 'Alpha Runner', 3.2, 0.2, '2026-05-24T09:40:00', 'sportsbet')
        """
    )
    conn.commit()
    conn.close()


def test_odds_coverage_reports_normalized_identity_ambiguity_and_timestamp_quality(tmp_path):
    db_path = tmp_path / "odds.db"
    _build_odds_db(db_path)

    report = analyze_odds_coverage(
        db_path,
        stale_after_hours=1.0,
        now=datetime.fromisoformat("2026-05-24T10:20:00"),
    )

    assert normalize_dog_name("1. Alpha-Runner") == "ALPHARUNNER"
    assert report["counts"]["live_odds_rows"] == 4
    assert report["counts"]["dog_level_win_odds_rows"] == 4
    assert report["counts"]["odds_history_dog_level_rows"] == 1
    assert report["match_counts"]["race_id_name_matches"] == 3
    assert report["match_counts"]["race_id_box_name_matches"] == 3
    assert report["safe_match_counts"]["safe_direct_identity_matches"] == 2
    assert report["safe_match_counts"]["ambiguous_strict_identity_rows"] == 1
    assert report["stale_late_risks"]["stale_current_win_rows"] == 1
    assert report["timestamp_quality"]["live_odds_current_win"]["stale_rows"] == 1
    assert report["source_provenance"]["live_odds"] == [{"source": "sportsbet", "rows": 4}]

    missing = {row["missing_reason"]: row["rows"] for row in report["missing_reasons"]}
    assert missing["ambiguous_race_id_box_name"] == 1
    assert missing["no_race_metadata_race_id"] == 1
    mismatch_counts = {
        row["mismatch_type"]: row["rows"] for row in report["venue_date_race_mismatches"]["counts"]
    }
    assert mismatch_counts["venue_date_race_resolves_different_race_id"] == 1


def test_prediction_snapshot_carries_odds_timestamp_provenance_and_readiness():
    snapshot = build_prediction_snapshot(
        {
            "race_id": "R1",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha Runner",
                    "box_number": 1,
                    "win_prob_norm": 0.4,
                    "predicted_rank": 1,
                    "odds_win": 3.0,
                    "odds_timestamp": "2026-05-24T09:55:00",
                    "odds_source": "sportsbet",
                    "odds_source_table": "live_odds",
                    "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/r1",
                    "odds_race_id": "R1",
                    "odds_dog_name": "Alpha Runner",
                    "odds_box_number": 1,
                    "odds_match_method": "race_id_box_name",
                    "odds_match_confidence": 1.0,
                    "ev_win": 0.2,
                },
                {
                    "dog_clean_name": "Beta Runner",
                    "box_number": 2,
                    "win_prob_norm": 0.6,
                    "predicted_rank": 2,
                    "ev_win": None,
                    "quality_flags": ["missing_live_odds"],
                },
            ],
        },
        source_file_path="Race 1 - WPK - 2026-05-24.csv",
        lifecycle={
            "status": "upcoming_not_jumped",
            "jump_datetime": "2026-05-24T10:30:00",
        },
        prediction_timestamp="2026-05-24T10:00:00",
    )

    odds_snapshot = snapshot["predictions"][0]["odds_snapshot"]
    assert odds_snapshot["odds_age_seconds_at_prediction"] == pytest.approx(300.0)
    assert odds_snapshot["odds_captured_before_prediction"] is True
    assert odds_snapshot["odds_captured_before_jump"] is True
    assert odds_snapshot["odds_stale_at_prediction"] is False
    assert odds_snapshot["odds_provenance"] == {
        "source": "sportsbet",
        "source_url": "https://www.sportsbet.com.au/greyhound-racing/r1",
        "source_table": "live_odds",
        "odds_race_id": "R1",
        "odds_dog_name": "Alpha Runner",
        "odds_box_number": 1,
        "match_method": "race_id_box_name",
        "match_confidence": 1.0,
    }
    assert snapshot["predictions"][0]["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert snapshot["snapshot_readiness"]["counts"]["missing_live_odds_count"] == 1
    assert snapshot["snapshot_readiness"]["status"] == "READY"
    assert_no_result_fields(snapshot)


def test_non_box_feature_quality_audit_writes_reports_without_db_writes(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    snapshot_dir = tmp_path / "snapshots"
    output_dir = tmp_path / "audit_out"
    snapshot_dir.mkdir()
    metadata_path = tmp_path / "metadata.json"
    db_path = tmp_path / "audit.db"
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": "fixture-model",
                "feature_names": [
                    "venue",
                    "grade",
                    "distance",
                    "field_size",
                    "box_number",
                    "historical_avg_position",
                    "historical_win_rate",
                    "target_distance",
                ],
            }
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "fixture.json").write_text(
        json.dumps(
            {
                "race_id": "Race 1 - TEST - 2026-05-26",
                "race_date": "2026-05-26",
                "venue": "TEST",
                "race_number": 1,
                "predictions": [
                    {
                        "dog_name": "Alpha Runner",
                        "box_number": 1,
                        "predicted_rank": 1,
                        "win_prob_norm": 0.6,
                        "distance_source": "default_missing_target",
                        "grade_source": "default_missing_target",
                        "metadata_is_leakage_safe": False,
                        "history_source": "embedded_csv_form_history",
                        "db_result_history_count": 0,
                    },
                    {
                        "dog_name": "Beta Runner",
                        "box_number": 2,
                        "predicted_rank": 2,
                        "win_prob_norm": 0.4,
                        "distance_source": "default_missing_target",
                        "grade_source": "default_missing_target",
                        "metadata_is_leakage_safe": False,
                        "history_source": "no_usable_history",
                        "db_result_history_count": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_non_box_feature_quality.py",
            "--snapshots",
            str(snapshot_dir),
            "--output-dir",
            str(output_dir),
            "--model-metadata",
            str(metadata_path),
            "--db",
            str(db_path),
            "--no-reconstruct",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    assert not db_path.exists()
    assert (output_dir / "live_feature_missingness.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "report.md").exists()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["runner_rows"] == 2
    assert summary["model_id"] == "fixture-model"
    assert "Production predictions changed" in (
        output_dir / "report.md"
    ).read_text(encoding="utf-8")
    assert "runner_rows" in result.stdout


def test_append_only_pre_jump_odds_capture_preserves_rows_and_source_url(tmp_path):
    db_path = tmp_path / "append_only_odds.db"
    integrator = SportsbetOddsIntegrator(db_path=str(db_path))
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE dog_race_data (
                id INTEGER PRIMARY KEY,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO dog_race_data
                (race_id, dog_name, dog_clean_name, box_number)
            VALUES ('R1', 'Alpha Runner', 'Alpha Runner', 1)
            """
        )
        conn.commit()

    race_info = {
        "race_id": "R1",
        "preserve_race_id": True,
        "venue": "Wentworth Park",
        "race_number": 1,
        "race_date": "2026-05-24",
        "race_time": "10:30",
        "venue_url": "https://www.sportsbet.com.au/greyhound-racing/wpk-r1",
    }
    odds_row = {
        "dog_name": "Alpha Runner",
        "dog_clean_name": "Alpha Runner",
        "box_number": 1,
        "odds_decimal": 3.0,
        "odds_fractional": "3.00",
    }

    first = integrator.append_pre_jump_odds_snapshot(
        race_info,
        [odds_row],
        capture_timestamp="2026-05-24T09:55:00",
    )
    second = integrator.append_pre_jump_odds_snapshot(
        race_info,
        [{**odds_row, "odds_decimal": 2.8}],
        capture_timestamp="2026-05-24T09:56:00",
    )

    assert first["status"] == "SUCCESS"
    assert second["status"] == "SUCCESS"
    rejected = integrator.append_pre_jump_odds_snapshot(
        {
            **race_info,
            "race_id": "R2",
            "venue_url": "https://www.sportsbet.com.au/greyhound-racing/results/wpk-r1",
        },
        [odds_row],
    )
    assert rejected["status"] == "REJECTED"
    assert "post_race_source_url_rejected" in rejected["warnings"]
    for unsafe_url in (
        "https://www.sportsbet.com.au/greyhound-racing/starting-price/wpk-r1",
        "https://www.sportsbet.com.au/greyhound-racing/wpk-r1?market=sp",
    ):
        unsafe = integrator.append_pre_jump_odds_snapshot(
            {**race_info, "race_id": "R2", "venue_url": unsafe_url},
            [odds_row],
        )
        assert unsafe["status"] == "REJECTED"
        assert "post_race_source_url_rejected" in unsafe["warnings"]
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT odds_decimal, source_url, capture_mode, timestamp, is_current
            FROM live_odds
            WHERE race_id = 'R1'
            ORDER BY id
            """
        ).fetchall()

    assert rows == [
        (
            3.0,
            "https://www.sportsbet.com.au/greyhound-racing/wpk-r1",
            "manual_pre_jump_snapshot",
            "2026-05-24T09:55:00",
            1,
        ),
        (
            2.8,
            "https://www.sportsbet.com.au/greyhound-racing/wpk-r1",
            "manual_pre_jump_snapshot",
            "2026-05-24T09:56:00",
            1,
        ),
    ]

    report = analyze_odds_coverage(
        db_path,
        now=datetime.fromisoformat("2026-05-24T10:00:00"),
    )
    assert report["source_url_quality"]["source_url_column_present"] is True
    assert report["source_url_quality"]["rows_with_source_url"] == 2
    assert report["source_url_quality"]["rows_missing_source_url"] == 0


def test_prediction_market_context_carries_db_odds_provenance_into_snapshot():
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_real_prediction_pipeline_v4_for_odds_snapshot_test",
        repo_root / "prediction_pipeline_v4.py",
    )
    pipeline_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pipeline_module)

    predictions = [
        {
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "win_prob_norm": 0.4,
            "predicted_rank": 1,
        }
    ]
    market_odds = {"Alpha Runner": 3.0}
    market_records = {
        "Alpha Runner": {
            "id": 10,
            "race_id": "R1",
            "dog_clean_name": "Alpha Runner",
            "box_number": 1,
            "odds_decimal": 3.0,
            "market_type": "win",
            "source": "sportsbet",
            "timestamp": "2026-05-24T09:55:00",
            "source_url": "https://www.sportsbet.com.au/greyhound-racing/wpk-r1",
            "odds_level": "dog",
        }
    }

    pipeline_module._annotate_market_context(predictions, market_odds, market_records)
    snapshot = build_prediction_snapshot(
        {
            "race_id": "R1",
            "model_version": "model-v1",
            "predictions": predictions,
        },
        lifecycle={
            "status": "upcoming_not_jumped",
            "jump_datetime": "2026-05-24T10:30:00",
        },
        prediction_timestamp="2026-05-24T10:00:00",
    )

    runner = snapshot["predictions"][0]
    assert runner["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert runner["ev_win"] == pytest.approx(0.2)
    assert runner["odds_snapshot"]["odds_provenance"]["source_url"] == (
        "https://www.sportsbet.com.au/greyhound-racing/wpk-r1"
    )
    assert_no_result_fields(snapshot)


def test_alt_race_odds_merge_preserves_snapshot_race_win_odds():
    import importlib.util
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "_real_prediction_pipeline_v4_for_alt_odds_merge_test",
        repo_root / "prediction_pipeline_v4.py",
    )
    pipeline_module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pipeline_module)

    win_odds, win_records, place_odds, resolved_markets = (
        pipeline_module._merge_missing_market_odds(
            {"Alpha Runner": 3.0},
            {"Alpha Runner": {"race_id": "Race 7 - HOR - 2026-05-26"}},
            {},
            {"Alpha Runner": 4.0},
            {"Alpha Runner": {"race_id": "HOR_2026-05-26_7"}},
            {"Alpha Runner": 1.6},
        )
    )

    assert win_odds == {"Alpha Runner": 3.0}
    assert win_records == {"Alpha Runner": {"race_id": "Race 7 - HOR - 2026-05-26"}}
    assert place_odds == {"Alpha Runner": 1.6}
    assert resolved_markets == ["place"]


def test_snapshot_evaluation_reports_missing_frozen_corpus(tmp_path):
    report = evaluate_snapshots(
        str(tmp_path / "labels.db"),
        [str(tmp_path / "prediction_snapshots")],
    )

    assert report["status"] == "DATA_MISSING"
    assert report["reason"] == "no_snapshot_files_found"
    readiness = report["snapshot_corpus_readiness"]
    assert readiness["status"] == "DATA_MISSING"
    assert readiness["reason"] == "no_frozen_pre_jump_snapshot_files_found"
    assert "result_free" in readiness["durable_pre_jump_snapshot_requirements"]


def test_snapshot_evaluation_links_results_and_scores_valid_pre_jump_odds_only(tmp_path):
    db_path = tmp_path / "labels.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            distance TEXT,
            results_status TEXT,
            winner_source TEXT,
            data_quality_note TEXT,
            winner_name TEXT
        );
        CREATE TABLE dog_race_data (
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            data_source TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, results_status, winner_name)
        VALUES
            ('Race 1 - WPK - 2026-05-24', 'WPK', 1, '2026-05-24', 'complete', 'Alpha')
        """
    )
    conn.executemany(
        """
        INSERT INTO dog_race_data
            (race_id, dog_name, dog_clean_name, box_number, finish_position, data_source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("Race 1 - WPK - 2026-05-24", "Alpha", "Alpha", 1, 1, "official"),
            ("Race 1 - WPK - 2026-05-24", "Bravo", "Bravo", 2, 2, "official"),
            ("Race 1 - WPK - 2026-05-24", "Charlie", "Charlie", 3, 3, "official"),
            ("Race 1 - WPK - 2026-05-24", "Delta", "Delta", 4, 4, "official"),
        ],
    )
    conn.commit()
    conn.close()

    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 1 - WPK - 2026-05-24",
            "model_version": "model-v1",
            "predictions": [
                {
                    "dog_clean_name": "Alpha",
                    "box_number": 1,
                    "win_prob_norm": 0.6,
                    "predicted_rank": 1,
                    "odds_win": 2.0,
                    "odds_timestamp": "2026-05-24T09:55:00",
                    "odds_source": "sportsbet",
                    "odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/wpk-r1",
                    "odds_race_id": "Race 1 - WPK - 2026-05-24",
                    "odds_dog_name": "Alpha",
                    "odds_box_number": 1,
                    "odds_match_method": "race_id_box_name",
                    "odds_match_confidence": 1.0,
                    "history_source": "db_and_embedded_csv_history",
                    "history_match_status": "matched_identity_with_pre_target_results",
                    "db_history_match_status": "matched_identity_with_pre_target_results",
                    "runner_inclusion_reason": "model_scored",
                    "distance_source": "target_column:Race Distance",
                    "grade_source": "target_column:Race Grade",
                    "metadata_is_leakage_safe": True,
                },
                {
                    "dog_clean_name": "Bravo",
                    "box_number": 2,
                    "win_prob_norm": 0.2,
                    "predicted_rank": 2,
                    "history_source": "embedded_csv_form_history",
                    "history_match_status": "embedded_history_only",
                    "db_history_match_status": "matched_identity_no_result_rows",
                    "runner_inclusion_reason": "model_scored_low_confidence_retained",
                    "distance_source": "default_missing_target",
                    "grade_source": "default_missing_target",
                    "metadata_is_leakage_safe": False,
                },
                {
                    "dog_clean_name": "Charlie",
                    "box_number": 3,
                    "win_prob_norm": 0.15,
                    "predicted_rank": 3,
                    "history_source": "no_usable_history",
                    "history_match_status": "no_matching_identity",
                    "db_history_match_status": "no_matching_identity",
                    "runner_inclusion_reason": "model_scored",
                    "distance_source": "default_missing_target",
                    "grade_source": "default_missing_target",
                    "metadata_is_leakage_safe": False,
                },
                {
                    "dog_clean_name": "Delta",
                    "box_number": 4,
                    "win_prob_norm": 0.05,
                    "predicted_rank": 4,
                    "history_source": "db_result_history",
                    "history_match_status": "matched_identity_with_pre_target_results",
                    "db_history_match_status": "matched_identity_with_pre_target_results",
                    "runner_inclusion_reason": "model_scored",
                    "distance_source": "target_column:Race Distance",
                    "grade_source": "target_column:Race Grade",
                    "metadata_is_leakage_safe": True,
                },
            ],
        },
        source_file_path="Race 1 - WPK - 2026-05-24.csv",
        lifecycle={
            "status": "upcoming_not_jumped",
            "race_date": "2026-05-24",
            "venue": "WPK",
            "race_number": 1,
            "jump_datetime": "2026-05-24T10:30:00",
        },
        prediction_timestamp="2026-05-24T10:00:00",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = evaluate_snapshots(str(db_path), [str(snapshot_path)])

    assert report["status"] == "SUCCESS"
    assert report["runner_rows_scored"] == 4
    assert report["metrics_by_arm"]["model_only"]["top1"] == pytest.approx(1.0)
    assert report["metrics_by_arm"]["model_only"]["winner_ranks"] == [1]
    assert report["ev_roi_coverage"]["status"] == "DATA_MISSING"
    assert report["ev_roi_coverage"]["reason"] == "partial_pre_jump_dog_level_odds"
    provenance = report["snapshot_provenance_report"]
    assert provenance["history_source_distribution"] == {
        "db_and_embedded_csv_history": 1,
        "embedded_csv_form_history": 1,
        "no_usable_history": 1,
        "db_result_history": 1,
    }
    assert (
        provenance["runner_inclusion_reason_distribution"]["model_scored_low_confidence_retained"]
        == 1
    )
    assert provenance["odds_match_status_distribution"] == {
        "valid_pre_jump_dog_odds": 1,
        "no_odds_row": 3,
    }
    assert provenance["odds_exclusion_reason_distribution"]["no_odds_row"] == 3
    assert provenance["target_distance_present_races"] == 1


def test_snapshot_evaluator_excludes_incomplete_runner_sets_even_if_labels_exist(tmp_path):
    db_path = tmp_path / "labels.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            results_status TEXT,
            winner_name TEXT
        );
        CREATE TABLE dog_race_data (
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            data_source TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO race_metadata VALUES ('Race 1 - SHEP - 2026-05-25', 'SHEP', 1, '2026-05-25', 'complete', 'Shima Lexie')"
    )
    conn.executemany(
        "INSERT INTO dog_race_data VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("Race 1 - SHEP - 2026-05-25", "Shima Lexie", "Shima Lexie", 2, 1, "official"),
            ("Race 1 - SHEP - 2026-05-25", "Sekiro", "Sekiro", 4, 2, "official"),
        ],
    )
    conn.commit()
    conn.close()

    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 1 - SHEP - 2026-05-25",
            "model_version": "model-v1",
            "predictions": [
                {"dog_clean_name": "Shima Lexie", "box_number": 2, "win_prob_norm": 0.6},
                {"dog_clean_name": "Sekiro", "box_number": 4, "win_prob_norm": 0.4},
            ],
        },
        lifecycle={
            "status": "upcoming_not_jumped",
            "race_date": "2026-05-25",
            "venue": "SHEP",
            "race_number": 1,
        },
        prediction_timestamp="2026-05-24T21:38:53",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = evaluate_snapshots(str(db_path), [str(snapshot_path)])

    assert report["status"] == "DATA_MISSING"
    assert report["label_quality_counts"] == {"snapshot_not_ready": 1}
    assert report["snapshot_corpus_readiness"]["status"] == "NOT_READY"


def test_snapshot_evaluator_does_not_join_wrong_same_date_race_number_venue(tmp_path):
    db_path = tmp_path / "labels.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            results_status TEXT,
            winner_name TEXT
        );
        CREATE TABLE dog_race_data (
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            data_source TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO race_metadata VALUES ('Race 1 - WAR - 2026-05-25', 'WAR', 1, '2026-05-25', 'complete', 'Alpha')"
    )
    conn.commit()
    conn.close()

    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 1 - WPK - 2026-05-25",
            "model_version": "model-v1",
            "predictions": [
                {"dog_clean_name": "Alpha", "box_number": 1, "win_prob_norm": 0.4},
                {"dog_clean_name": "Bravo", "box_number": 2, "win_prob_norm": 0.3},
                {"dog_clean_name": "Charlie", "box_number": 3, "win_prob_norm": 0.2},
                {"dog_clean_name": "Delta", "box_number": 4, "win_prob_norm": 0.1},
            ],
        },
        lifecycle={
            "status": "upcoming_not_jumped",
            "race_date": "2026-05-25",
            "venue": "WPK",
            "race_number": 1,
        },
        source_file_path="Race 1 - WPK - 2026-05-25.csv",
        prediction_timestamp="2026-05-24T21:38:53",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = evaluate_snapshots(str(db_path), [str(snapshot_path)])

    assert report["status"] == "DATA_MISSING"
    assert report["label_quality_counts"] == {"missing_race_metadata": 1}


def test_snapshot_evaluator_scores_partial_sportsbet_winner_only_labels(tmp_path):
    db_path = tmp_path / "labels.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            results_status TEXT,
            winner_name TEXT
        );
        CREATE TABLE dog_race_data (
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            data_source TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO race_metadata VALUES ('Race 1 - WPK - 2026-05-25', 'WPK', 1, '2026-05-25', 'partial_sportsbet_results', 'Alpha')"
    )
    conn.executemany(
        "INSERT INTO dog_race_data VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("Race 1 - WPK - 2026-05-25", "Alpha", "Alpha", 1, 1, "sportsbet"),
            ("Race 1 - WPK - 2026-05-25", "Bravo", "Bravo", 2, 2, "sportsbet"),
            ("Race 1 - WPK - 2026-05-25", "Charlie", "Charlie", 3, None, "sportsbet"),
            ("Race 1 - WPK - 2026-05-25", "Delta", "Delta", 4, None, "sportsbet"),
        ],
    )
    conn.commit()
    conn.close()

    snapshot = build_prediction_snapshot(
        {
            "race_id": "Race 1 - WPK - 2026-05-25",
            "model_version": "model-v1",
            "predictions": [
                {"dog_clean_name": "Alpha", "box_number": 1, "win_prob_norm": 0.4},
                {"dog_clean_name": "Bravo", "box_number": 2, "win_prob_norm": 0.3},
                {"dog_clean_name": "Charlie", "box_number": 3, "win_prob_norm": 0.2},
                {"dog_clean_name": "Delta", "box_number": 4, "win_prob_norm": 0.1},
            ],
        },
        lifecycle={
            "status": "upcoming_not_jumped",
            "race_date": "2026-05-25",
            "venue": "WPK",
            "race_number": 1,
        },
        source_file_path="Race 1 - WPK - 2026-05-25.csv",
        prediction_timestamp="2026-05-24T21:38:53",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    report = evaluate_snapshots(str(db_path), [str(snapshot_path)])

    assert report["status"] == "SUCCESS"
    assert report["runner_rows_scored"] == 4
    assert report["label_quality_counts"] == {"partial_sportsbet_winner_only": 1}
    assert report["metrics_by_arm"]["model_only"]["top1"] == pytest.approx(1.0)
    assert report["metrics_by_arm"]["model_only"]["winner_ranks"] == [1]
    diagnostics = report["failure_mode_diagnostics"]
    assert diagnostics["winner_rank_by_race"] == {"Race 1 - WPK - 2026-05-25": 1}
    assert diagnostics["complete_vs_partial_labels"]["partial"]["races"] == 1
    assert diagnostics["label_quality_breakdown"]["partial_sportsbet_winner_only"][
        "top1"
    ] == pytest.approx(1.0)
    assert diagnostics["distance_breakdown"] == {
        "status": "DATA_MISSING",
        "reason": "no_distance_metadata",
        "races_missing_distance": 1,
    }
