import json
import sqlite3
from datetime import datetime
from pathlib import Path

from scripts import build_unified_evidence_dataset as dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "greyhound.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                race_time TEXT,
                start_datetime TEXT,
                url TEXT,
                winner_source TEXT,
                results_status TEXT
            );
            CREATE TABLE dog_race_data (
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                data_source TEXT
            );
            CREATE TABLE live_odds (
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
            """
        )
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, venue, race_number, race_date, race_time, start_datetime, url, winner_source, results_status)
            VALUES (?, 'WPK', 1, '2026-06-10', '15:00', '2026-06-10T15:00:00+10:00',
                    'https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results',
                    'thedogs_official', 'resulted')
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )
        conn.executemany(
            """
            INSERT INTO dog_race_data
                (race_id, dog_name, dog_clean_name, box_number, finish_position, data_source)
            VALUES (?, ?, ?, ?, ?, 'thedogs_official')
            """,
            [
                ("Race 1 - WPK - 2026-06-10", "Alpha Runner", "ALPHA RUNNER", 1, 1),
                ("Race 1 - WPK - 2026-06-10", "Bravo Runner", "BRAVO RUNNER", 2, 2),
            ],
        )
        conn.execute(
            """
            INSERT INTO live_odds
                (race_id, venue, race_number, race_date, race_time, dog_name, dog_clean_name,
                 box_number, odds_decimal, odds_fractional, market_type, source, timestamp,
                 is_current, topN, source_url, capture_timestamp, capture_mode, odds_level,
                 sportsbet_box_source, sportsbet_list_position, sportsbet_raw_runner_text)
            VALUES (?, 'WPK', 1, '2026-06-10', '15:00', 'Alpha Runner', 'ALPHA RUNNER',
                    1, 2.8, '9/5', 'win', 'sportsbet', '2026-06-10T14:30:00+10:00',
                    1, NULL, 'https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-1',
                    '2026-06-10T14:30:00+10:00', 'autonomous_prejump_t30m', 'dog',
                    'runner_text', NULL, '1. Alpha Runner')
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )
    return db_path


def _remove_official_result_rows(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE race_metadata SET winner_source = 'pending', results_status = 'pending'"
        )
        conn.execute("DELETE FROM dog_race_data")


def _insert_official_result_evidence_rows(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE autonomous_official_result_evidence_races (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue TEXT,
                race_number INTEGER,
                race_time TEXT,
                start_datetime TEXT,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                status TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                source_artifact_dir TEXT NOT NULL
            );
            CREATE TABLE autonomous_official_result_evidence_runners (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                inserted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_artifact_dir TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO autonomous_official_result_evidence_races
                (race_id, race_date, venue, race_number, race_time, start_datetime,
                 source, source_url, status, captured_at, source_artifact_dir)
            VALUES (?, '2026-06-10', 'WPK', 1, '15:00',
                    '2026-06-10T15:00:00+10:00', 'thedogs_official',
                    'https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results',
                    'resulted', '2026-06-10T15:20:00+10:00',
                    'artifacts/results/evidence')
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )
        conn.executemany(
            """
            INSERT INTO autonomous_official_result_evidence_runners
                (race_id, race_date, venue, race_number, source, source_url,
                 box_number, dog_name, finish_position, is_winner, captured_at,
                 source_artifact_dir)
            VALUES (?, '2026-06-10', 'WPK', 1, 'thedogs_official',
                    'https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results',
                    ?, ?, ?, ?, '2026-06-10T15:20:00+10:00',
                    'artifacts/results/evidence')
            """,
            [
                ("Race 1 - WPK - 2026-06-10", 1, "Alpha Runner", 1, 1),
                ("Race 1 - WPK - 2026-06-10", 2, "Bravo Runner", 2, 0),
            ],
        )


def test_build_dataset_joins_predictions_official_results_and_strict_odds(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
            "model_version": "baseline",
            "calibration_method": "power_gamma_2.4",
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Bravo Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.22,
            "model_version": "baseline",
            "calibration_method": "power_gamma_2.4",
        },
    ]
    stage2 = [
        {
            **predictions[0],
            "schema_version": "stage2_shadow_prediction_v1",
            "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            "stage2_forward_shadow_status": "STAGE2_FORWARD_SHADOW_COLLECTING",
            "shadow_rf_calibrated_probability": 0.34,
        },
        {
            **predictions[1],
            "schema_version": "stage2_shadow_prediction_v1",
            "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            "stage2_forward_shadow_status": "STAGE2_FORWARD_SHADOW_COLLECTING",
            "shadow_rf_calibrated_probability": 0.2,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(shadow_dir / "stage2_shadow_predictions.jsonl", stage2)
    db_path = _make_db(tmp_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["final_status"] == "UNIFIED_EVIDENCE_DATASET_BUILT"
    assert report["row_count"] == 2
    assert report["rows_with_official_results"] == 2
    assert report["rows_with_stage2_predictions"] == 2
    assert report["rows_with_strict_prejump_odds"] == 1
    assert report["label_evaluation_eligible_rows"] == 2
    assert report["stage2_evaluation_eligible_rows"] == 2
    assert report["odds_evaluation_eligible_rows"] == 1
    assert report["unified_evidence_eligible_rows"] == 1

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = next(row for row in rows if row["dog_name"] == "Alpha Runner")
    bravo = next(row for row in rows if row["dog_name"] == "Bravo Runner")
    assert alpha["finish_position"] == 1
    assert alpha["is_winner"] is True
    assert alpha["strict_prejump_odds_available"] is True
    assert alpha["odds_by_capture_bucket"]["t30"]["odds_decimal"] == 2.8
    assert alpha["unified_evidence_eligible"] is True
    assert bravo["label_evaluation_eligible"] is True
    assert bravo["strict_prejump_odds_available"] is False
    assert "strict_prejump_odds_missing" in bravo["excluded_from_unified_reason"]


def test_build_dataset_uses_valid_shadow_odds_snapshot_artifact_without_db_odds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Bravo Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.22,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **row,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
            for row in predictions
        ],
    )
    db_path = _make_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM live_odds")
    odds_path = tmp_path / "shadow_odds_snapshot.jsonl"
    valid_odds_row = {
        "schema_version": "shadow_odds_snapshot_runner_v1",
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "odds_match_status": "valid_pre_jump_dog_odds",
        "odds_provenance_status": "complete",
        "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        "ev_win": None,
        "race_context": {
            "venue": "WPK",
            "race_number": 1,
            "race_date": "2026-06-10",
            "race_time": "15:00",
        },
        "odds_snapshot": {
            "market_odds_win": 2.8,
            "market_type": "win",
            "odds_timestamp": "2026-06-10T14:30:00+10:00",
            "odds_level": "dog",
            "odds_captured_before_feature_freeze": True,
            "odds_captured_before_jump": True,
            "odds_captured_before_prediction": True,
            "odds_provenance": {
                "source": "sportsbet",
                "source_table": "live_odds",
                "source_url": "https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-1",
                "capture_mode": "autonomous_prejump_t30m",
                "sportsbet_box_source": "runner_text",
                "sportsbet_list_position": 1,
                "sportsbet_raw_runner_text": "1. Alpha Runner",
                "odds_box_number": 1,
                "odds_dog_name": "ALPHA RUNNER",
                "odds_race_id": "Race 1 - WPK - 2026-06-10",
            },
        },
    }
    invalid_odds_row = {
        **valid_odds_row,
        "dog_name": "Bravo Runner",
        "box": 2,
        "odds_match_status": "NO_VALID_ODDS",
    }
    _write_jsonl(odds_path, [valid_odds_row, invalid_odds_row])

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_artifact_odds",
        odds_jsonl_paths=[odds_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["rows_with_strict_prejump_odds"] == 1
    assert report["rows_with_artifact_shadow_odds"] == 1
    assert report["artifact_odds_rows_seen"] == 2
    assert report["artifact_odds_rows_accepted"] == 1
    assert report["artifact_odds_rows_rejected"] == 1
    assert report["artifact_odds_audits"][0]["rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 1
    }
    assert report["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 1
    }
    summary = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_artifact_odds/SUMMARY.md"
    ).read_text(encoding="utf-8")
    assert (
        "- Artifact odds rejection reasons: `{'odds_match_status_not_valid_pre_jump_dog_odds': 1}`"
        in summary
    )
    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_artifact_odds/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = next(row for row in rows if row["dog_name"] == "Alpha Runner")
    bravo = next(row for row in rows if row["dog_name"] == "Bravo Runner")
    assert alpha["strict_prejump_odds_available"] is True
    assert alpha["odds_by_capture_bucket"]["t30"]["odds_decimal"] == 2.8
    assert alpha["odds_by_capture_bucket"]["t30"]["source_artifact_path"].endswith(
        "shadow_odds_snapshot.jsonl"
    )
    assert bravo["strict_prejump_odds_available"] is False


def test_build_dataset_reports_artifact_candidates_superseded_by_db_bucket(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Bravo Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.22,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **row,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
            for row in predictions
        ],
    )
    db_path = _make_db(tmp_path)
    odds_path = tmp_path / "shadow_odds_snapshot.jsonl"
    _write_jsonl(
        odds_path,
        [
            {
                "schema_version": "shadow_odds_snapshot_runner_v1",
                "race_id": "Race 1 - WPK - 2026-06-10",
                "dog_name": "Alpha Runner",
                "box": 1,
                "odds_match_status": "valid_pre_jump_dog_odds",
                "odds_provenance_status": "complete",
                "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
                "ev_win": None,
                "race_context": {
                    "venue": "WPK",
                    "race_number": 1,
                    "race_date": "2026-06-10",
                    "race_time": "15:00",
                },
                "odds_snapshot": {
                    "market_odds_win": 3.1,
                    "market_type": "win",
                    "odds_timestamp": "2026-06-10T14:20:00+10:00",
                    "odds_level": "dog",
                    "odds_captured_before_feature_freeze": True,
                    "odds_captured_before_jump": True,
                    "odds_captured_before_prediction": True,
                    "odds_provenance": {
                        "source": "sportsbet",
                        "source_table": "live_odds",
                        "source_url": "https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-1",
                        "capture_mode": "autonomous_prejump_t30m",
                        "sportsbet_box_source": "runner_text",
                        "sportsbet_raw_runner_text": "1. Alpha Runner",
                        "odds_box_number": 1,
                        "odds_dog_name": "ALPHA RUNNER",
                        "odds_race_id": "Race 1 - WPK - 2026-06-10",
                    },
                },
            }
        ],
    )

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_artifact_candidate",
        odds_jsonl_paths=[odds_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["artifact_odds_rows_accepted"] == 1
    assert report["rows_with_artifact_shadow_odds"] == 0
    assert report["rows_with_artifact_shadow_odds_candidates"] == 1
    assert report["artifact_shadow_odds_candidate_count"] == 1
    assert report["artifact_shadow_odds_selected_bucket_count"] == 0
    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_artifact_candidate/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = next(row for row in rows if row["dog_name"] == "Alpha Runner")
    assert alpha["artifact_shadow_odds_available"] is True
    assert alpha["artifact_shadow_odds_candidate_count"] == 1
    assert alpha["artifact_shadow_odds_selected_bucket_count"] == 0
    assert alpha["odds_by_capture_bucket"]["t30"]["source_artifact_path"] is None
    assert alpha["odds_by_capture_bucket"]["t30"]["odds_decimal"] == 2.8


def test_join_eligibility_packet_filters_dataset_report_only_without_db_write(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Bravo Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.22,
        },
        {
            "race_id": "Race 2 - WPK - 2026-06-10",
            "dog_name": "Charlie Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.4,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **row,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
            for row in predictions
        ],
    )
    packet_path = tmp_path / "join_eligibility.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "live_odds_backlog_join_eligibility_packet_v1",
                "diagnostic_only": True,
                "join_authorized": False,
                "db_write_performed": False,
                "races": [
                    {
                        "race_id": "Race 1 - WPK - 2026-06-10",
                        "eligibility_status": "JOIN_ELIGIBLE_REPORT_ONLY",
                        "blockers": [],
                        "join_authorized": False,
                        "db_write_performed": False,
                    },
                    {
                        "race_id": "Race 2 - WPK - 2026-06-10",
                        "eligibility_status": "JOIN_ELIGIBILITY_BLOCKED",
                        "blockers": ["official_result_missing"],
                        "join_authorized": False,
                        "db_write_performed": False,
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    db_path = _make_db(tmp_path)
    before_hash = dataset.sha256_file(db_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_join_eligibility",
        join_eligibility_packet_paths=[packet_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert dataset.sha256_file(db_path) == before_hash
    assert report["row_count"] == 2
    assert report["race_count"] == 1
    assert report["join_eligibility_packet_rows_seen"] == 2
    assert report["join_eligibility_packet_accepted_races"] == 1
    assert report["join_eligibility_packet_accepted_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["join_eligibility_packet_accepted_races_present_in_shadow_run"] == 1
    assert (
        report["join_eligibility_packet_accepted_race_ids_missing_from_shadow_run"]
        == []
    )
    assert report["join_eligibility_packet_rejected_races"] == 1
    assert report["join_eligibility_packet_audits"][0]["accepted_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["join_eligibility_packet_audits"][0][
        "rejection_reason_counts"
    ] == {"eligibility_status_not_report_only": 1}

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_join_eligibility/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert {row["race_id"] for row in rows} == {"Race 1 - WPK - 2026-06-10"}
    assert all(row["race_id"] != "Race 2 - WPK - 2026-06-10" for row in rows)
    assert any(row["unified_evidence_eligible"] for row in rows)


def test_build_dataset_uses_nested_stage2_predictions_when_root_file_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "shadow_score_live/stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
                "shadow_rf_calibrated_probability": 0.37,
            }
        ],
    )
    db_path = _make_db(tmp_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_nested_stage2",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["stage2_predictions_path"].endswith(
        "daily/shadow_score_live/stage2_shadow_predictions.jsonl"
    )
    assert report["stage2_predictions_path_source"] == "shadow_score_live_fallback"
    assert report["stage2_prediction_rows"] == 1
    assert report["rows_with_stage2_predictions"] == 1
    assert report["unified_evidence_eligible_rows"] == 1

    row = json.loads(
        (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_nested_stage2/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert row["stage2_challenger_key"] == "shadow_calibrated_rf_power_gamma_2_4"
    assert row["stage2_shadow_probability"] == 0.37


def test_build_dataset_prefers_root_stage2_predictions_over_nested_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "root_stage2",
                "shadow_rf_calibrated_probability": 0.4,
            }
        ],
    )
    _write_jsonl(
        shadow_dir / "shadow_score_live/stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "nested_stage2",
                "shadow_rf_calibrated_probability": 0.2,
            }
        ],
    )
    db_path = _make_db(tmp_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_root_stage2",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["stage2_predictions_path"].endswith("daily/stage2_shadow_predictions.jsonl")
    assert report["stage2_predictions_path_source"] == "shadow_run_root"
    row = json.loads(
        (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_root_stage2/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert row["stage2_challenger_key"] == "root_stage2"
    assert row["stage2_shadow_probability"] == 0.4


def test_build_dataset_uses_nested_stage2_when_root_file_is_empty(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    (shadow_dir / "stage2_shadow_predictions.jsonl").write_text("", encoding="utf-8")
    _write_jsonl(
        shadow_dir / "shadow_score_live/stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "nested_stage2",
                "shadow_rf_calibrated_probability": 0.38,
            }
        ],
    )
    db_path = _make_db(tmp_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_empty_root_stage2",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["stage2_predictions_path"].endswith(
        "daily/shadow_score_live/stage2_shadow_predictions.jsonl"
    )
    assert report["stage2_predictions_path_source"] == "shadow_score_live_fallback"
    row = json.loads(
        (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_empty_root_stage2/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert row["stage2_challenger_key"] == "nested_stage2"
    assert row["stage2_shadow_probability"] == 0.38


def test_build_dataset_joins_apostrophe_variants_by_exact_box(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    race_id = "Race 6 - CASO - 2026-06-11"
    prediction = {
        "race_id": race_id,
        "dog_name": "Chads Girl",
        "box": 7,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.42,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
        ],
    )
    db_path = tmp_path / "greyhound.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                race_time TEXT,
                start_datetime TEXT,
                url TEXT,
                winner_source TEXT,
                results_status TEXT
            );
            CREATE TABLE dog_race_data (
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                data_source TEXT
            );
            CREATE TABLE live_odds (
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
            """
        )
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, venue, race_number, race_date, race_time, start_datetime, url, winner_source, results_status)
            VALUES (?, 'CASO', 6, '2026-06-11', '16:45', '2026-06-11T16:45:00+10:00',
                    'https://www.thedogs.com.au/racing/casino/2026-06-11/6/results',
                    'thedogs_official', 'resulted')
            """,
            (race_id,),
        )
        conn.execute(
            """
            INSERT INTO dog_race_data
                (race_id, dog_name, dog_clean_name, box_number, finish_position, data_source)
            VALUES (?, 'Chad''s Girl', 'CHADS GIRL', 7, 1, 'thedogs_official')
            """,
            (race_id,),
        )
        conn.execute(
            """
            INSERT INTO live_odds
                (race_id, venue, race_number, race_date, race_time, dog_name, dog_clean_name,
                 box_number, odds_decimal, odds_fractional, market_type, source, timestamp,
                 is_current, topN, source_url, capture_timestamp, capture_mode, odds_level,
                 sportsbet_box_source, sportsbet_list_position, sportsbet_raw_runner_text)
            VALUES (?, 'CASO', 6, '2026-06-11', '16:45', 'Chad''s Girl', 'CHADS GIRL',
                    7, 1.55, '11/20', 'win', 'sportsbet', '2026-06-11T16:31:00+10:00',
                    1, NULL, 'https://www.sportsbet.com.au/greyhound-racing/australia-nz/casino/race-6',
                    '2026-06-11T16:31:00+10:00', 'autonomous_prejump_t30m', 'dog',
                    'runner_text', 7, '7. Chad''s Girl (7)')
            """,
            (race_id,),
        )

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_apostrophe",
        generated_at=datetime.fromisoformat("2026-06-11T17:00:00+10:00"),
    )

    assert report["rows_with_official_results"] == 1
    assert report["rows_with_strict_prejump_odds"] == 1
    assert report["unified_evidence_eligible_rows"] == 1
    row = json.loads(
        (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_apostrophe/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8")
    )
    assert row["dog_identity"] == "CHADSGIRL"
    assert row["strict_prejump_odds_available"] is True
    assert row["odds_by_capture_bucket"]["t30"]["odds_decimal"] == 1.55
    assert row["finish_position"] == 1


def test_joined_shadow_predictions_exact_rows_supply_official_results_without_db_write(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Bravo Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.22,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **row,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
            for row in predictions
        ],
    )
    joined_path = tmp_path / "join" / "joined_shadow_predictions.jsonl"
    _write_jsonl(
        joined_path,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "dog_name": "Alpha Runner",
                "official_dog_name": "Alpha Runner",
                "box": 1,
                "finish_position": 1,
                "is_winner": True,
                "identity_match_status": "exact_box_and_normalized_name",
                "result_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results",
            },
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "dog_name": "Bravo Runner",
                "official_dog_name": "Bravo Runner",
                "box": 2,
                "finish_position": 2,
                "is_winner": False,
                "identity_match_status": "exact_box_and_normalized_name",
                "result_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results",
            },
        ],
    )
    db_path = _make_db(tmp_path)
    _remove_official_result_rows(db_path)
    before_hash = dataset.sha256_file(db_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test",
        joined_shadow_prediction_paths=[joined_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert dataset.sha256_file(db_path) == before_hash
    assert report["rows_with_official_results"] == 2
    assert report["label_evaluation_eligible_rows"] == 2
    assert report["stage2_evaluation_eligible_rows"] == 2
    assert report["unified_evidence_eligible_rows"] == 1
    assert report["joined_shadow_prediction_rows_seen"] == 2
    assert report["joined_shadow_prediction_rows_accepted"] == 2
    assert report["joined_shadow_prediction_rows_rejected"] == 0

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = next(row for row in rows if row["dog_name"] == "Alpha Runner")
    assert alpha["finish_position"] == 1
    assert alpha["is_winner"] is True
    assert alpha["official_result_source"] == "forward_shadow_exact_join_artifact"
    assert alpha["official_result_data_source"] == "forward_shadow_exact_join_artifact"
    assert alpha["unified_evidence_eligible"] is True


def test_db_official_result_evidence_supplies_results_without_label_write(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        },
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Bravo Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.22,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **row,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
            for row in predictions
        ],
    )
    db_path = _make_db(tmp_path)
    _remove_official_result_rows(db_path)
    _insert_official_result_evidence_rows(db_path)
    before_hash = dataset.sha256_file(db_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_evidence_db",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert dataset.sha256_file(db_path) == before_hash
    assert report["rows_with_official_results"] == 2
    assert report["rows_with_official_result_evidence_db"] == 2
    assert report["label_evaluation_eligible_rows"] == 2
    assert report["stage2_evaluation_eligible_rows"] == 2
    assert report["unified_evidence_eligible_rows"] == 1
    assert report["official_result_evidence_db_audit"]["rows_seen"] == 2
    assert report["official_result_evidence_db_audit"]["accepted_rows"] == 2
    assert report["official_result_evidence_db_audit"]["rejected_rows"] == 0
    assert report["official_result_evidence_db_audit"]["requested_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["official_result_evidence_db_audit"]["race_ids_with_rows"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["official_result_evidence_db_audit"]["missing_race_ids"] == []
    assert report["official_result_coverage"] == {
        "source": "unified_evidence_dataset",
        "requested_race_count": 1,
        "requested_race_count_source": (
            "official_result_evidence_db_audit_requested_race_ids"
        ),
        "requested_race_ids": ["Race 1 - WPK - 2026-06-10"],
        "races_with_rows_count": 1,
        "missing_race_count": 0,
        "missing_race_ids": [],
        "races_with_rows": ["Race 1 - WPK - 2026-06-10"],
        "runner_path_count": 0,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 0,
    }

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM dog_race_data").fetchone()[0] == 0
        assert conn.execute(
            "SELECT DISTINCT winner_source FROM race_metadata"
        ).fetchone()[0] == "pending"

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_evidence_db/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = next(row for row in rows if row["dog_name"] == "Alpha Runner")
    bravo = next(row for row in rows if row["dog_name"] == "Bravo Runner")
    assert alpha["finish_position"] == 1
    assert alpha["is_winner"] is True
    assert alpha["official_result_data_source"] == "official_result_evidence_db"
    assert alpha["unified_evidence_eligible"] is True
    assert bravo["finish_position"] == 2
    assert bravo["official_result_data_source"] == "official_result_evidence_db"


def test_db_official_result_evidence_audit_lists_exact_missing_race_ids(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    predictions = [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        },
        {
            "race_id": "Race 2 - WPK - 2026-06-10",
            "dog_name": "Charlie Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.28,
        },
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", predictions)
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **row,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
            for row in predictions
        ],
    )
    db_path = _make_db(tmp_path)
    _remove_official_result_rows(db_path)
    _insert_official_result_evidence_rows(db_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_evidence_db_missing",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    audit = report["official_result_evidence_db_audit"]
    assert audit["requested_race_ids"] == [
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-10",
    ]
    assert audit["race_ids_with_rows"] == ["Race 1 - WPK - 2026-06-10"]
    assert audit["missing_race_ids"] == ["Race 2 - WPK - 2026-06-10"]
    assert report["rows_with_official_result_evidence_db"] == 1
    assert report["rows_with_official_results"] == 1

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_evidence_db_missing/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    missing = next(row for row in rows if row["race_id"] == "Race 2 - WPK - 2026-06-10")
    assert missing["official_result_available"] is False
    assert "official_result_missing" in missing["excluded_from_unified_reason"]


def test_joined_shadow_predictions_reject_non_exact_identity_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
        ],
    )
    joined_path = tmp_path / "join" / "joined_shadow_predictions.jsonl"
    _write_jsonl(
        joined_path,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "dog_name": "Alpha Runner",
                "official_dog_name": "Different Runner",
                "box": 1,
                "finish_position": 1,
                "is_winner": True,
                "identity_match_status": "dog_name_mismatch_after_exact_badge_stripping",
                "result_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results",
            },
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "dog_name": "Alpha Runner",
                "official_dog_name": "Alpha Runner",
                "box": 1,
                "finish_position": 1,
                "is_winner": False,
                "identity_match_status": "exact_box_and_normalized_name",
                "result_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results",
            }
        ],
    )
    db_path = _make_db(tmp_path)
    _remove_official_result_rows(db_path)

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test",
        joined_shadow_prediction_paths=[joined_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["rows_with_official_results"] == 0
    assert report["joined_shadow_prediction_rows_seen"] == 2
    assert report["joined_shadow_prediction_rows_accepted"] == 0
    assert report["joined_shadow_prediction_rows_rejected"] == 2
    assert report["joined_shadow_prediction_audits"][0]["rejection_reason_counts"] == {
        "finish_position_winner_flag_conflict": 1,
        "identity_match_not_exact_box_and_normalized_name": 1,
    }
    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["official_result_available"] is False
    assert "official_result_missing" in rows[0]["excluded_from_unified_reason"]


def test_race_gap_prioritization_uses_source_lineage_not_raw_db_counts(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"

    def prediction(race_id: str, dog_name: str) -> dict:
        return {
            "race_id": race_id,
            "dog_name": dog_name,
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.31,
        }

    ready = "Race 1 - WPK - 2026-06-10"
    official_missing = "Race 2 - WPK - 2026-06-10"
    odds_missing = "Race 3 - WPK - 2026-06-10"
    stage2_missing = "Race 4 - WPK - 2026-06-10"
    identity_mismatch = "Race 5 - WPK - 2026-06-10"
    source_set_missing = "Race 6 - WPK - 2026-06-10"
    other_gate = "Race 7 - WPK - 2026-06-10"
    dog_by_race = {
        ready: "Ready Runner",
        official_missing: "Official Missing Runner",
        odds_missing: "Odds Missing Runner",
        stage2_missing: "Stage Two Missing Runner",
        identity_mismatch: "Identity Mismatch Runner",
        source_set_missing: "Source Missing Runner",
        other_gate: "Other Gate Runner",
    }

    primary_predictions = [
        prediction(ready, dog_by_race[ready]),
        prediction(official_missing, dog_by_race[official_missing]),
        prediction(odds_missing, dog_by_race[odds_missing]),
        prediction(stage2_missing, dog_by_race[stage2_missing]),
        prediction(identity_mismatch, dog_by_race[identity_mismatch]),
    ]
    stage2_predictions = [
        {
            **prediction(race_id, dog_by_race[race_id]),
            "schema_version": "stage2_shadow_prediction_v1",
            "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
        }
        for race_id in [
            ready,
            official_missing,
            odds_missing,
            identity_mismatch,
            other_gate,
        ]
    ]
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", primary_predictions)
    _write_jsonl(shadow_dir / "stage2_shadow_predictions.jsonl", stage2_predictions)

    runner_path = tmp_path / "official_result_runners.jsonl"
    _write_jsonl(
        runner_path,
        [
            {
                "race_id": race_id,
                "box_number": 1,
                "dog_name": dog_by_race[race_id],
                "finish_position": 1,
                "source": "thedogs_official",
                "source_url": (
                    "https://www.thedogs.com.au/racing/"
                    f"wentworth-park/2026-06-10/{index}/results"
                ),
            }
            for index, race_id in enumerate(
                [ready, odds_missing, stage2_missing, other_gate],
                start=1,
            )
        ],
    )

    joined_path = tmp_path / "joined_shadow_predictions.jsonl"
    _write_jsonl(
        joined_path,
        [
            {
                "race_id": identity_mismatch,
                "dog_name": dog_by_race[identity_mismatch],
                "official_dog_name": "Different Runner",
                "box": 1,
                "finish_position": 1,
                "is_winner": True,
                "identity_match_status": "dog_name_mismatch_after_exact_badge_stripping",
                "result_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/5/results",
            }
        ],
    )

    odds_path = tmp_path / "shadow_odds_snapshot.jsonl"
    _write_jsonl(
        odds_path,
        [
            {
                "schema_version": "shadow_odds_snapshot_runner_v1",
                "race_id": race_id,
                "dog_name": dog_by_race[race_id],
                "box": 1,
                "odds_match_status": "valid_pre_jump_dog_odds",
                "odds_provenance_status": "complete",
                "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
                "ev_win": None,
                "race_context": {
                    "venue": "WPK",
                    "race_number": int(race_id.split()[1]),
                    "race_date": "2026-06-10",
                    "race_time": "15:00",
                },
                "odds_snapshot": {
                    "market_odds_win": 2.8,
                    "market_type": "win",
                    "odds_timestamp": "2026-06-10T14:30:00+10:00",
                    "odds_level": "dog",
                    "odds_captured_before_feature_freeze": True,
                    "odds_captured_before_jump": True,
                    "odds_captured_before_prediction": True,
                    "odds_provenance": {
                        "source": "sportsbet",
                        "source_table": "live_odds",
                        "source_url": (
                            "https://www.sportsbet.com.au/greyhound-racing/"
                            "australia-nz/wentworth-park/race-1"
                        ),
                        "capture_mode": "autonomous_prejump_t30m",
                        "sportsbet_box_source": "runner_text",
                        "sportsbet_raw_runner_text": f"1. {dog_by_race[race_id]}",
                        "odds_box_number": 1,
                        "odds_dog_name": dog_by_race[race_id].upper(),
                        "odds_race_id": race_id,
                    },
                },
            }
            for race_id in [
                ready,
                official_missing,
                stage2_missing,
                identity_mismatch,
                other_gate,
            ]
        ],
    )

    packet_path = tmp_path / "join_eligibility.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": "live_odds_backlog_join_eligibility_packet_v1",
                "diagnostic_only": True,
                "join_authorized": False,
                "db_write_performed": False,
                "races": [
                    {
                        "race_id": race_id,
                        "eligibility_status": "JOIN_ELIGIBLE_REPORT_ONLY",
                        "blockers": [],
                        "join_authorized": False,
                        "db_write_performed": False,
                    }
                    for race_id in dog_by_race
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    db_path = _make_db(tmp_path)
    _remove_official_result_rows(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM live_odds")

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_gap_priority",
        official_result_runner_paths=[runner_path],
        joined_shadow_prediction_paths=[joined_path],
        join_eligibility_packet_paths=[packet_path],
        odds_jsonl_paths=[odds_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    gap = report["race_gap_prioritization"]
    assert gap["source_race_id_source"] == "join_eligibility_packet_accepted_race_ids"
    assert gap["raw_db_count_basis"] is False
    assert gap["source_race_count"] == 7
    assert gap["dataset_race_count"] == 6
    assert gap["source_set_missing_race_count"] == 1
    assert gap["primary_gap_class_counts"] == {
        "identity_mismatch": 1,
        "official_result_missing": 1,
        "other_gate": 1,
        "source_set_missing": 1,
        "stage2_missing": 1,
        "strict_prejump_odds_missing": 1,
    }
    assert gap["gap_class_counts"] == {
        "identity_mismatch": 1,
        "official_result_missing": 2,
        "other_gate": 1,
        "source_set_missing": 1,
        "stage2_missing": 1,
        "strict_prejump_odds_missing": 1,
    }
    assert gap["top_gap_race_ids"] == [
        source_set_missing,
        identity_mismatch,
        official_missing,
        odds_missing,
        stage2_missing,
        other_gate,
    ]
    top_by_race = {row["race_id"]: row for row in gap["top_gap_races"]}
    assert top_by_race[source_set_missing]["primary_gap_class"] == "source_set_missing"
    assert top_by_race[source_set_missing]["source_set_present"] is False
    assert top_by_race[identity_mismatch]["gap_classes"] == [
        "identity_mismatch",
        "official_result_missing",
    ]
    assert top_by_race[identity_mismatch]["identity_mismatch_reasons"] == [
        "identity_match_not_exact_box_and_normalized_name"
    ]
    assert top_by_race[official_missing]["primary_gap_class"] == "official_result_missing"
    assert top_by_race[odds_missing]["primary_gap_class"] == "strict_prejump_odds_missing"
    assert top_by_race[stage2_missing]["primary_gap_class"] == "stage2_missing"
    assert top_by_race[other_gate]["primary_gap_class"] == "other_gate"

    summary = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_gap_priority/SUMMARY.md"
    ).read_text(encoding="utf-8")
    assert "- Race gap raw DB count basis: `False`" in summary
    assert f"- Race gap top race IDs: `{gap['top_gap_race_ids']}`" in summary


def test_valid_strict_odds_do_not_inherit_stale_candidate_exclusions(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
        ],
    )
    db_path = _make_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO live_odds
                (race_id, venue, race_number, race_date, race_time, dog_name, dog_clean_name,
                 box_number, odds_decimal, odds_fractional, market_type, source, timestamp,
                 is_current, topN, source_url, capture_timestamp, capture_mode, odds_level,
                 sportsbet_box_source, sportsbet_list_position, sportsbet_raw_runner_text)
            VALUES (?, 'WPK', 1, '2026-06-10', '15:00', 'Alpha Runner', 'ALPHA RUNNER',
                    1, NULL, NULL, NULL, 'sportsbet', '2026-06-10T14:00:00+10:00',
                    0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = rows[0]
    assert report["unified_evidence_eligible_rows"] == 1
    assert "odds_source_url_missing" in alpha["odds_exclusion_reasons"]
    assert "odds_source_url_missing" not in alpha["excluded_from_unified_reason"]
    assert "odds_decimal_invalid" not in report["exclusion_reason_counts"]
    assert alpha["rejected_live_odds_candidate_count"] == 1
    rejected = alpha["rejected_live_odds_candidates"][0]
    assert rejected["source"] == "sportsbet"
    assert rejected["source_url"] is None
    assert rejected["capture_timestamp"] is None
    assert rejected["sportsbet_box_source"] is None
    assert rejected["rejection_reasons"] == [
        "odds_capture_timestamp_missing",
        "odds_decimal_invalid",
        "odds_level_missing",
        "odds_market_not_win",
        "odds_source_url_missing",
        "unsupported_sportsbet_box_source:missing",
    ]
    assert report["rejected_live_odds_candidate_count"] == 1
    assert report["rows_with_rejected_live_odds_candidates"] == 1
    assert report["rejected_live_odds_candidate_reason_counts"] == {
        "odds_capture_timestamp_missing": 1,
        "odds_decimal_invalid": 1,
        "odds_level_missing": 1,
        "odds_market_not_win": 1,
        "odds_source_url_missing": 1,
        "unsupported_sportsbet_box_source:missing": 1,
    }
    assert report["rejected_live_odds_candidate_samples"] == [
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "dog_name": "Alpha Runner",
            "box_number": 1,
            "rejection_reasons": rejected["rejection_reasons"],
            "source": "sportsbet",
            "source_url": None,
            "capture_timestamp": None,
            "capture_mode": None,
            "sportsbet_box_source": None,
        }
    ]
    summary_text = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_test/SUMMARY.md"
    ).read_text(encoding="utf-8")
    assert "- Rejected live odds candidates: `1`" in summary_text
    assert '"odds_decimal_invalid": 1' in summary_text


def test_db_live_odds_captured_after_prediction_and_jump_are_quarantined(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
        "prediction_timestamp": "2026-06-10T15:10:00+10:00",
        "feature_freeze_timestamp": "2026-06-10T15:05:00+10:00",
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
        ],
    )
    db_path = _make_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE race_metadata
            SET race_time = '15:20',
                start_datetime = '2026-06-10T15:20:00+10:00'
            WHERE race_id = ?
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )
        conn.execute(
            """
            UPDATE live_odds
            SET race_time = '15:20',
                timestamp = '2026-06-10T15:25:00+10:00',
                capture_timestamp = '2026-06-10T15:25:00+10:00',
                capture_mode = 'autonomous_prejump_t2m'
            WHERE race_id = ? AND dog_clean_name = 'ALPHA RUNNER'
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )

    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_late_db_odds",
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    rows = [
        json.loads(line)
        for line in (
            tmp_path
            / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_late_db_odds/unified_evidence_dataset.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    alpha = rows[0]
    assert report["rows_with_strict_prejump_odds"] == 0
    assert report["odds_evaluation_eligible_rows"] == 0
    assert alpha["strict_prejump_odds_available"] is False
    assert alpha["rejected_live_odds_candidate_count"] == 1
    assert alpha["rejected_live_odds_candidates"][0]["rejection_reasons"] == [
        "odds_capture_not_before_feature_freeze",
        "odds_capture_not_before_jump",
        "odds_capture_not_before_prediction",
    ]
    assert alpha["odds_exclusion_reasons"] == {
        "odds_capture_not_before_feature_freeze": 1,
        "odds_capture_not_before_jump": 1,
        "odds_capture_not_before_prediction": 1,
    }
    assert "strict_prejump_odds_missing" in alpha["excluded_from_unified_reason"]
    assert report["rejected_live_odds_candidate_reason_counts"] == {
        "odds_capture_not_before_feature_freeze": 1,
        "odds_capture_not_before_jump": 1,
        "odds_capture_not_before_prediction": 1,
    }


def test_summary_surfaces_compact_official_result_coverage_without_path_list(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    shadow_dir = tmp_path / "daily"
    prediction = {
        "race_id": "Race 1 - WPK - 2026-06-10",
        "dog_name": "Alpha Runner",
        "box": 1,
        "predicted_rank": 1,
        "shadow_rf_calibrated_probability": 0.31,
    }
    _write_jsonl(shadow_dir / "shadow_predictions.jsonl", [prediction])
    _write_jsonl(
        shadow_dir / "stage2_shadow_predictions.jsonl",
        [
            {
                **prediction,
                "schema_version": "stage2_shadow_prediction_v1",
                "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
            }
        ],
    )
    runner_path = tmp_path / "official_result_runners.jsonl"
    _write_jsonl(
        runner_path,
        [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "box_number": 1,
                "dog_name": "Alpha Runner",
                "finish_position": 1,
                "source": "thedogs_official",
                "source_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results",
            }
        ],
    )
    db_path = _make_db(tmp_path)
    _remove_official_result_rows(db_path)

    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/unified_evidence_dataset_official_coverage"
    )
    report = dataset.build_dataset(
        shadow_run_dir=shadow_dir,
        db_path=db_path,
        output_dir=output_dir,
        official_result_runner_paths=[runner_path],
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert report["official_result_runner_paths"] == ["official_result_runners.jsonl"]
    assert report["official_result_coverage"] == {
        "source": "unified_evidence_dataset",
        "requested_race_count": 1,
        "requested_race_count_source": (
            "official_result_evidence_db_audit_requested_race_ids"
        ),
        "requested_race_ids": ["Race 1 - WPK - 2026-06-10"],
        "races_with_rows_count": 0,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 1 - WPK - 2026-06-10"],
        "races_with_rows": [],
        "runner_path_count": 1,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 0,
    }
    summary_text = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "- Official-result coverage requested races: `1`" in summary_text
    assert (
        "- Official-result coverage requested race count source: "
        "`official_result_evidence_db_audit_requested_race_ids`"
    ) in summary_text
    assert "- Official-result coverage races with rows: `0`" in summary_text
    assert "- Official-result coverage missing races: `1`" in summary_text
    assert "- Official-result missing exclusions: `0`" in summary_text
    assert "- Official-result runner path count: `1`" in summary_text
    assert (
        "- Official-result runner paths source field: "
        "`official_result_runner_paths`"
    ) in summary_text
    assert "Official-result runner paths:" not in summary_text


def test_dataset_output_guard_rejects_unscoped_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "ROOT", tmp_path)
    try:
        dataset.assert_output_dir_safe(tmp_path / "not_artifacts")
    except ValueError as exc:
        assert "output_dir_must_be_unified" in str(exc)
    else:
        raise AssertionError("expected unsafe output path to be rejected")
