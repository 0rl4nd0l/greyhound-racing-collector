import sqlite3
from datetime import datetime

import pytest

from accuracy_program.odds_coverage import analyze_odds_coverage, normalize_dog_name
from accuracy_program.snapshots import build_prediction_snapshot
from scripts.evaluate_prediction_snapshots import evaluate_snapshots


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
    assert report["source_provenance"]["live_odds"] == [
        {"source": "sportsbet", "rows": 4}
    ]

    missing = {row["missing_reason"]: row["rows"] for row in report["missing_reasons"]}
    assert missing["ambiguous_race_id_box_name"] == 1
    assert missing["no_race_metadata_race_id"] == 1
    mismatch_counts = {
        row["mismatch_type"]: row["rows"]
        for row in report["venue_date_race_mismatches"]["counts"]
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
        "source_table": "live_odds",
    }
    assert snapshot["snapshot_readiness"]["counts"]["missing_live_odds_count"] == 1
    assert snapshot["snapshot_readiness"]["status"] == "READY"
    assert "finish_position" not in str(snapshot)


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
