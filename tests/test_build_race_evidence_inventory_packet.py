import csv
import json
import sqlite3
from pathlib import Path

from scripts import build_race_evidence_inventory_packet as packet


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _create_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE autonomous_official_result_evidence_races (
                id INTEGER PRIMARY KEY,
                race_id TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue TEXT,
                race_number INTEGER
            );
            CREATE TABLE autonomous_official_result_evidence_runners (
                id INTEGER PRIMARY KEY,
                race_id TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue TEXT,
                race_number INTEGER,
                box_number INTEGER NOT NULL,
                dog_name TEXT,
                finish_position INTEGER,
                is_winner INTEGER
            );
            CREATE TABLE live_odds (
                id INTEGER PRIMARY KEY,
                race_id TEXT,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_time TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                odds_fractional TEXT,
                market_type TEXT DEFAULT 'win',
                source TEXT DEFAULT 'sportsbet',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT TRUE,
                topN INTEGER,
                source_url TEXT,
                capture_timestamp TEXT,
                capture_mode TEXT,
                odds_level TEXT,
                sportsbet_box_source TEXT,
                sportsbet_list_position INTEGER,
                sportsbet_raw_runner_text TEXT
            );
            """
        )


def test_race_evidence_inventory_reports_joinable_and_gap_races(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    monkeypatch.setattr(
        packet,
        "DEFAULT_EVIDENCE_ROOT",
        tmp_path / "artifacts/full_evidence_orchestration_20260525",
    )
    db_path = tmp_path / "greyhound_racing_data.db"
    _create_db(db_path)
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"

    shadow_dir = artifact_root / "daily_race_ingest_shadow_test"
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "box": 1,
                "dog_name": "Alpha",
                "predicted_rank": 1,
                "shadow_rf_calibrated_probability": 0.7,
            },
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "box": 2,
                "dog_name": "Bravo",
                "predicted_rank": 2,
                "shadow_rf_calibrated_probability": 0.3,
            },
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "box": 1,
                "dog_name": "Charlie",
                "predicted_rank": 1,
                "shadow_rf_calibrated_probability": 1.0,
            },
            {
                "race_id": "Race 3 - WPK - 2026-06-10",
                "box": 1,
                "dog_name": "Delta",
                "predicted_rank": 1,
                "shadow_rf_calibrated_probability": 1.0,
            },
            {
                "race_id": "Race 4 - WPK - 2026-06-10",
                "box": 1,
                "dog_name": "Echo",
                "predicted_rank": 1,
                "shadow_rf_calibrated_probability": 0.6,
            },
            {
                "race_id": "Race 4 - WPK - 2026-06-10",
                "box": 2,
                "dog_name": "Foxtrot",
                "predicted_rank": 2,
                "shadow_rf_calibrated_probability": 0.4,
            },
            {
                "race_id": "Race 5 - WPK - 2026-06-10",
                "box": 1,
                "dog_name": "Golf",
                "predicted_rank": 1,
                "shadow_rf_calibrated_probability": 1.0,
            },
        ],
    )
    official_dir = artifact_root / "autonomous_official_result_capture_test"
    _write_jsonl(
        official_dir / "official_result_races.jsonl",
        [
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": 2,
            }
        ],
    )
    _write_jsonl(
        official_dir / "official_result_runners.jsonl",
        [
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": 2,
                "box_number": 1,
            }
        ],
    )

    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO autonomous_official_result_evidence_races
              (race_id, race_date, venue, race_number)
            VALUES (?, '2026-06-10', 'WPK', ?)
            """,
            [
                ("Race 1 - WPK - 2026-06-10", 1),
                ("Race 4 - WPK - 2026-06-10", 4),
                ("Race 5 - WPK - 2026-06-10", 5),
            ],
        )
        conn.executemany(
            """
            INSERT INTO autonomous_official_result_evidence_runners
              (race_id, race_date, venue, race_number, box_number, dog_name, finish_position, is_winner)
            VALUES (?, '2026-06-10', 'WPK', ?, ?, ?, ?, ?)
            """,
            [
                ("Race 1 - WPK - 2026-06-10", 1, 1, "Alpha", 1, 1),
                ("Race 1 - WPK - 2026-06-10", 1, 2, "Bravo", 2, 0),
                ("Race 4 - WPK - 2026-06-10", 4, 1, "Echo", 1, 1),
                ("Race 5 - WPK - 2026-06-10", 5, 1, "Golf", 1, 1),
            ],
        )
        conn.executemany(
            """
            INSERT INTO live_odds
              (race_id, venue, race_number, race_date, dog_name, box_number,
               odds_decimal, market_type, source, source_url, capture_timestamp,
               capture_mode, odds_level, sportsbet_box_source)
            VALUES
              ('Race 1 - WPK - 2026-06-10', 'WPK', 1, '2026-06-10', ?, ?,
               2.5, 'win', 'sportsbet', 'https://www.sportsbet.com.au/test',
               '2026-06-10T13:59:00+10:00', 't2', 'dog', 'explicit_dom')
            """,
            [("Alpha", 1), ("Bravo", 2)],
        )

    output_dir = artifact_root / "race_evidence_inventory_test_report_only"
    report = packet.build_packet(
        artifact_roots=[artifact_root],
        db_path=db_path,
        output_dir=output_dir,
    )

    assert report["final_status"] == "RACE_EVIDENCE_INVENTORY_READY_FOR_EVALUATION"
    assert report["recommended_decision"] == "RUN_POST_BACKLOG_UNIFIED_EVALUATION"
    assert report["summary_counts"]["shadow_races_complete_official_and_strict_odds"] == 1
    assert report["scorecard_metrics"]["evaluation_race_count"] == 1
    assert report["scorecard_metrics"]["model_top1_accuracy"] == 1.0
    assert report["scorecard_metrics"]["market_top1_accuracy"] == 1.0
    assert report["summary_counts"]["action_counts"] == {
        "append_official_result_evidence_backlog": 1,
        "capture_official_result": 1,
        "collect_future_strict_prejump_odds": 1,
        "ready_for_unified_evidence_evaluation": 1,
        "repair_official_result_runner_set_or_identity_join": 1,
    }
    assert report["scorecard_metrics"]["skipped_race_reason_counts"] == {
        "official_result_incomplete_for_shadow_boxes": 3,
        "strict_odds_incomplete_for_shadow_boxes": 1,
    }
    assert report["scorecard_metrics"]["skipped_race_action_counts"] == {
        "append_official_result_evidence_backlog": 1,
        "capture_official_result": 1,
        "collect_future_strict_prejump_odds": 1,
        "repair_official_result_runner_set_or_identity_join": 1,
    }
    assert report["scorecard_metrics"]["official_result_gap_action_counts"] == {
        "append_official_result_evidence_backlog": 1,
        "capture_official_result": 1,
        "repair_official_result_runner_set_or_identity_join": 1,
    }
    assert report["scorecard_metrics"]["strict_odds_gap_action_counts"] == {
        "collect_future_strict_prejump_odds": 1,
    }
    assert report["no_write_guarantees"]["db_write"] is False
    summary_text = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "## Scorecard Gap Action Counts" in summary_text
    assert '"capture_official_result": 1' in summary_text
    assert '"repair_official_result_runner_set_or_identity_join": 1' in summary_text

    rows = list(csv.DictReader((output_dir / "race_evidence_inventory.csv").open()))
    by_race = {row["race_id"]: row for row in rows}
    assert by_race["Race 1 - WPK - 2026-06-10"]["recommended_next_action"] == (
        "ready_for_unified_evidence_evaluation"
    )
    assert by_race["Race 2 - WPK - 2026-06-10"]["recommended_next_action"] == (
        "append_official_result_evidence_backlog"
    )
    assert by_race["Race 3 - WPK - 2026-06-10"]["recommended_next_action"] == (
        "capture_official_result"
    )
    assert by_race["Race 4 - WPK - 2026-06-10"]["recommended_next_action"] == (
        "repair_official_result_runner_set_or_identity_join"
    )
    assert by_race["Race 5 - WPK - 2026-06-10"]["recommended_next_action"] == (
        "collect_future_strict_prejump_odds"
    )


def test_race_evidence_inventory_zero_byte_db_is_data_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    artifact_root.mkdir(parents=True)
    db_path = tmp_path / "zero.sqlite"
    db_path.write_bytes(b"")

    report = packet.build_packet(
        artifact_roots=[artifact_root],
        db_path=db_path,
        output_dir=artifact_root / "race_evidence_inventory_zero_db_report_only",
    )

    assert report["final_status"] == "DATA_MISSING"
    assert report["db_summary"]["db_status"]["reason"] == "db_zero_byte"
