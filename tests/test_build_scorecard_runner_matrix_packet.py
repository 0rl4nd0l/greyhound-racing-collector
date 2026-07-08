import csv
import json
import sqlite3
from pathlib import Path

import pytest

from scripts import build_scorecard_runner_matrix_packet as packet


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _clean_official_exclusion_reasons(rows: list[dict[str, object]]) -> list[str]:
    reasons: list[str] = []
    if any(row.get("label_quality") != "official_or_complete_result" for row in rows):
        reasons.append("label_quality_not_official_or_complete")
    if any(row.get("result_detail_quality") != "finish_position" for row in rows):
        reasons.append("result_detail_not_full_finish_position")
    if any(row.get("finish_position") is None for row in rows):
        reasons.append("finish_position_missing")
    winner_count = sum(int(row.get("actual_win") or 0) for row in rows)
    if winner_count != 1:
        reasons.append(f"winner_count_not_one:{winner_count}")
    return reasons


def _create_live_odds_db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE autonomous_official_result_evidence_races (
                id INTEGER PRIMARY KEY,
                race_id TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue TEXT,
                race_number INTEGER,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                status TEXT NOT NULL,
                winner_name TEXT,
                winner_box INTEGER,
                position_count INTEGER NOT NULL,
                participant_count INTEGER,
                box_order_json TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL
            );
            CREATE TABLE autonomous_official_result_evidence_runners (
                id INTEGER PRIMARY KEY,
                race_id TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue TEXT,
                race_number INTEGER,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                box_number INTEGER NOT NULL,
                dog_name TEXT NOT NULL,
                finish_position INTEGER NOT NULL,
                is_winner INTEGER NOT NULL,
                captured_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL
            );
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
                topN INTEGER,
                source_url TEXT,
                capture_timestamp TEXT,
                capture_mode TEXT,
                odds_level TEXT,
                sportsbet_box_source TEXT,
                sportsbet_list_position INTEGER,
                sportsbet_raw_runner_text TEXT
            );
            """)
        conn.execute("""
            INSERT INTO autonomous_official_result_evidence_races
              (race_id, race_date, venue, race_number, source, source_url, status,
               winner_name, winner_box, position_count, participant_count,
               box_order_json, captured_at, inserted_at)
            VALUES
              ('Race 1 - WPK - 2026-06-10', '2026-06-10', 'WPK', 1,
               'thedogs_official', 'https://www.thedogs.com.au/racing/test',
               'resulted', 'Alpha', 1, 3, 3, '[1,2,3]',
               '2026-06-10T18:30:00+10:00', '2026-06-10T18:31:00+10:00')
            """)
        conn.executemany(
            """
            INSERT INTO autonomous_official_result_evidence_runners
              (race_id, race_date, venue, race_number, source, source_url,
               box_number, dog_name, finish_position, is_winner, captured_at, inserted_at)
            VALUES
              ('Race 1 - WPK - 2026-06-10', '2026-06-10', 'WPK', 1,
               'thedogs_official', 'https://www.thedogs.com.au/racing/test',
               ?, ?, ?, ?, '2026-06-10T18:30:00+10:00', '2026-06-10T18:31:00+10:00')
            """,
            [(1, "Alpha", 1, 1), (2, "Bravo", 2, 0), (3, "Charlie", 3, 0)],
        )
        conn.executemany(
            """
            INSERT INTO live_odds
              (race_id, venue, race_number, race_date, race_time, dog_name,
               dog_clean_name, box_number, odds_decimal, odds_fractional,
               market_type, source, timestamp, is_current, source_url,
               capture_timestamp, capture_mode, odds_level, sportsbet_box_source,
               sportsbet_list_position, sportsbet_raw_runner_text)
            VALUES
              ('Race 1 - WPK - 2026-06-10', 'WPK', 1, '2026-06-10', '18:00',
               ?, ?, ?, ?, ?, 'win', 'sportsbet',
               '2026-06-10T17:58:00+10:00', 1,
               'https://www.sportsbet.com.au/greyhound-racing/test',
               '2026-06-10T17:58:00+10:00', 'autonomous_prejump_t2m',
               'dog', 'runner_text', ?, 'runner')
            """,
            [
                ("Alpha", "ALPHA", 1, 2.0, "2.00", 1),
                ("Bravo", "BRAVO", 2, 4.0, "4.00", 2),
                ("Charlie", "CHARLIE", 3, 4.0, "4.00", 3),
            ],
        )


def test_scorecard_runner_matrix_reproduces_scorecard_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)
    db_path = tmp_path / "greyhound.db"
    _create_live_odds_db(db_path)
    source_dir = tmp_path / "shadow_run"
    source_dir.mkdir()
    prediction_path = source_dir / "shadow_predictions.jsonl"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "box": 1,
            "dog_name": "Alpha",
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.5,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "box": 2,
            "dog_name": "Bravo",
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.25,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "box": 3,
            "dog_name": "Charlie",
            "predicted_rank": 3,
            "shadow_rf_calibrated_probability": 0.25,
        },
    ]
    prediction_path.write_text(
        "\n".join(json.dumps(row) for row in predictions) + "\n",
        encoding="utf-8",
    )
    (source_dir / "shadow_feature_rows.json").write_text(
        json.dumps(
            [
                {
                    "race_id": "Race 1 - WPK - 2026-06-10",
                    "box_number": 1,
                    "weather": "Fine",
                    "track_condition": "Good",
                    "prior_start_count": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    scorecard_csv = tmp_path / "scorecard.csv"
    _write_csv(
        scorecard_csv,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "WPK",
                "race_number": "1",
                "runner_count": "3",
                "winner_box": "1",
                "winner_dog_name": "Alpha",
                "model_winner_rank": "1",
                "model_top1_correct": "True",
                "model_top3_correct": "True",
                "model_winner_probability": "0.5",
                "model_logloss": str(-__import__("math").log(0.5)),
                "market_winner_rank": "1",
                "market_top1_correct": "True",
                "market_top3_correct": "True",
                "market_winner_probability": "0.5",
                "market_logloss": str(-__import__("math").log(0.5)),
                "model_top_box": "1",
                "market_top_box": "1",
                "winner_prediction_source_path": str(prediction_path),
                "winner_prediction_raw_probability": "0.5",
            }
        ],
    )

    report = packet.build_packet(
        scorecard_csv=scorecard_csv,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/scorecard_runner_matrix_test_report_only",
    )

    assert report["final_status"] == "SCORECARD_RUNNER_MATRIX_READY"
    assert report["metric_reproduction_ok"] is True
    assert report["matrix_race_count"] == 1
    assert report["matrix_runner_row_count"] == 3
    assert report["feature_join_counts"] == {
        "DATA_MISSING_FEATURE_ROW": 2,
        "JOINED_FEATURE_ROW": 1,
    }
    assert report["official_result_join_counts"] == {"JOINED_OFFICIAL_RESULT": 3}
    assert report["no_write_guarantees"]["db_write"] is False
    assert (tmp_path / report["matrix_jsonl"]).exists()
    assert (tmp_path / report["matrix_csv"]).exists()
    rows = packet.load_jsonl(tmp_path / report["matrix_jsonl"])
    assert _clean_official_exclusion_reasons(rows) == []
    assert [row["finish_position"] for row in sorted(rows, key=lambda row: row["box_number"])] == [
        1,
        2,
        3,
    ]


def test_scorecard_runner_matrix_output_dir_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_scorecard_runner_matrix"):
        packet.assert_output_dir_safe(
            tmp_path / "artifacts/full_evidence_orchestration_20260525/wrong_report_only"
        )
