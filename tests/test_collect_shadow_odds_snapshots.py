import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from scripts import collect_shadow_odds_snapshots as odds


def _write_shadow_run(
    path: Path,
    *,
    race_id: str = "Race 1 - TEST - 2026-06-09",
    manifest_generated_at: str = "2026-06-09T00:05:00+10:00",
    prediction_timestamp: str | None = None,
    feature_freeze_timestamp: str | None = None,
    jump_time: str = "11:35 AM",
) -> None:
    path.mkdir(parents=True)
    (path / "shadow_predictions.jsonl").write_text(
        json.dumps(
            {
                "race_id": race_id,
                "dog_name": "Alpha Runner",
                "box": 1,
                "predicted_rank": 1,
                "shadow_rf_calibrated_probability": 0.42,
                "output_mode": "shadow_only",
                "tgr_enabled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {"generated_at": manifest_generated_at}
    score_live_manifest = {}
    if prediction_timestamp is not None:
        score_live_manifest["prediction_timestamp"] = prediction_timestamp
    if feature_freeze_timestamp is not None:
        score_live_manifest["feature_freeze_timestamp"] = feature_freeze_timestamp
    if score_live_manifest:
        manifest["score_live_manifest"] = score_live_manifest
    (path / "shadow_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "prejump_metadata_report.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "race_date": "2026-06-09",
                        "venue": "TEST",
                        "race_number": 1,
                        "jump_time": jump_time,
                        "source_url": "https://www.thedogs.com.au/racing/test/2026-06-09/1/test?trial=false",
                        "runner_count": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _build_db(
    path: Path,
    *,
    duplicate_odds: bool = False,
    repeated_capture_windows: bool = False,
    include_odds: bool = True,
    odds_timestamp: str = "2026-06-09T00:01:00+10:00",
    odds_source: str = "sportsbet",
    odds_source_url: str = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/test/race-1"
    ),
    market_type: str = "win",
) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE race_metadata (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            winner_source TEXT
        );
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            data_source TEXT
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
          (race_id, venue, race_number, race_date, winner_source)
        VALUES ('Race 1 - TEST - 2026-06-09', 'TEST', 1, '2026-06-09', 'thedogs_official')
        """
    )
    conn.execute(
        """
        INSERT INTO dog_race_data
          (race_id, dog_name, dog_clean_name, box_number, data_source)
        VALUES ('Race 1 - TEST - 2026-06-09', 'Alpha Runner', 'Alpha Runner', 1, 'thedogs_official')
        """
    )
    if include_odds:
        rows = [
            (
                "Race 1 - TEST - 2026-06-09",
                "TEST",
                1,
                "2026-06-09",
                "Alpha Runner",
                "Alpha Runner",
                1,
                3.2,
                market_type,
                odds_source,
                odds_timestamp,
                1,
                odds_source_url,
                odds_timestamp,
                "autonomous_prejump_t60m",
                "dog",
                "runner_text",
                1,
                "1. Alpha Runner",
            )
        ]
        if duplicate_odds:
            rows.append((*rows[0][:-12], 3.4, *rows[0][-11:]))
        if repeated_capture_windows:
            rows.append(
                (
                    "Race 1 - TEST - 2026-06-09",
                    "TEST",
                    1,
                    "2026-06-09",
                    "Alpha Runner",
                    "Alpha Runner",
                    1,
                    3.4,
                    market_type,
                    odds_source,
                    "2026-06-09T00:03:00+10:00",
                    1,
                    odds_source_url,
                    "2026-06-09T00:03:00+10:00",
                    "autonomous_prejump_t30m",
                    "dog",
                    "runner_text",
                    1,
                    "1. Alpha Runner",
                )
            )
        conn.executemany(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, dog_name, dog_clean_name,
                box_number, odds_decimal, market_type, source, timestamp, is_current,
                source_url, capture_timestamp, capture_mode, odds_level,
                sportsbet_box_source, sportsbet_list_position, sportsbet_raw_runner_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.commit()
    conn.close()


def _run(
    tmp_path,
    monkeypatch,
    *,
    include_odds=True,
    duplicate_odds=False,
    repeated_capture_windows=False,
    odds_timestamp: str = "2026-06-09T00:01:00+10:00",
    odds_source: str = "sportsbet",
    odds_source_url: str = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/test/race-1"
    ),
    market_type: str = "win",
    manifest_generated_at: str = "2026-06-09T00:05:00+10:00",
    prediction_timestamp: str | None = None,
    feature_freeze_timestamp: str | None = None,
    jump_time: str = "11:35 AM",
):
    monkeypatch.setattr(odds, "ROOT", tmp_path)
    monkeypatch.setattr(odds, "EXPECTED_OFFICIAL_RACES", 1)
    monkeypatch.setattr(odds, "EXPECTED_OFFICIAL_DOG_ROWS", 1)
    db_path = tmp_path / "greyhound_racing_data.db"
    monkeypatch.setattr(odds, "PROTECTED_PATHS", (db_path,))
    shadow_run = tmp_path / "shadow_run"
    _write_shadow_run(
        shadow_run,
        manifest_generated_at=manifest_generated_at,
        prediction_timestamp=prediction_timestamp,
        feature_freeze_timestamp=feature_freeze_timestamp,
        jump_time=jump_time,
    )
    _build_db(
        db_path,
        include_odds=include_odds,
        duplicate_odds=duplicate_odds,
        repeated_capture_windows=repeated_capture_windows,
        odds_timestamp=odds_timestamp,
        odds_source=odds_source,
        odds_source_url=odds_source_url,
        market_type=market_type,
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_test"
    )

    report = odds.collect_shadow_odds_snapshot(
        shadow_run_dir=shadow_run,
        db_path=db_path,
        output_dir=output_dir,
        current_time=datetime.fromisoformat("2026-06-09T00:10:00+10:00"),
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "shadow_odds_snapshot.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    return report, rows, output_dir


def _race_coverage(output_dir: Path) -> dict:
    return json.loads((output_dir / "shadow_odds_race_coverage.json").read_text(encoding="utf-8"))


def test_parse_jump_datetime_uses_iso_context_before_display_time():
    parsed = odds.parse_jump_datetime(
        {
            "race_date": "2026-06-09",
            "jump_datetime": "2026-06-09T12:05:00+10:00",
            "jump_time": "11:35 AM",
        }
    )

    assert parsed is not None
    assert parsed.isoformat() == "2026-06-09T12:05:00+10:00"


def test_manifest_timestamp_prefers_nested_score_live_timestamps():
    manifest = {
        "generated_at": "2026-06-09T00:20:00+10:00",
        "score_live_manifest": {
            "prediction_timestamp": "2026-06-09T00:10:00+10:00",
            "feature_freeze_timestamp": "2026-06-09T00:05:00+10:00",
        },
    }

    parsed = odds.manifest_timestamp(
        manifest,
        (
            ("score_live_manifest", "prediction_timestamp"),
            ("generated_at",),
        ),
    )

    assert parsed is not None
    assert parsed.isoformat() == "2026-06-09T00:10:00+10:00"
    parsed_with_source, source = odds.manifest_timestamp_with_source(
        manifest,
        (
            ("score_live_manifest", "prediction_timestamp"),
            ("generated_at",),
        ),
    )
    assert parsed_with_source == parsed
    assert source == "score_live_manifest.prediction_timestamp"


def test_collect_shadow_odds_snapshot_marks_exact_dog_box_odds_eligible(tmp_path, monkeypatch):
    report, rows, output_dir = _run(tmp_path, monkeypatch)
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["prediction_rows"] == 1
    assert report["odds_candidate_rows"] == 1
    assert report["valid_pre_jump_dog_odds_rows"] == 1
    assert report["races_with_complete_odds_candidate_coverage"] == 1
    assert report["races_with_complete_valid_prejump_odds"] == 1
    assert report["races_with_missing_odds_rows"] == 0
    assert report["races_with_post_jump_odds_rows"] == 0
    assert report["ev_output_rows"] == 0
    assert report["protected_paths_unchanged"] is True
    assert report["odds_research_readiness"]["status"] == (
        "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
    )
    assert report["effective_prediction_timestamp"] == "2026-06-09T00:05:00+10:00"
    assert report["effective_prediction_timestamp_source"] == "generated_at"
    assert report["effective_feature_freeze_timestamp"] is None
    assert report["effective_feature_freeze_timestamp_source"] is None
    assert report["odds_research_readiness"]["blocker_counts"] == {}
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "REPORT_ONLY_REVIEW_ODDS_CALIBRATION_NO_EV_ACTION"
    )
    gate = report["odds_research_readiness"]["ev_research_gate"]
    assert gate["status"] == "READY_FOR_REPORT_ONLY_ODDS_REVIEW_NO_EV_OUTPUT"
    assert gate["ev_output_allowed"] is False
    assert gate["betting_action_allowed"] is False
    research_gate = report["odds_research_gate"]
    assert research_gate["status"] == odds.ODDS_RESEARCH_BLOCKED_PROVENANCE
    assert research_gate["complete_valid_prejump_odds_races"] == 1
    assert research_gate["minimum_complete_valid_prejump_odds_races"] == 100
    assert research_gate["blocker_counts"] == {
        "complete_valid_prejump_odds_races_below_min": 99
    }
    assert research_gate["source_url_coverage_pct"] == 100.0
    assert research_gate["odds_used_for_shadow_scoring"] is False
    assert research_gate["ev_diagnostics_report_only_allowed"] is False
    assert report["odds_augmented_challenger"]["final_status"] == (
        odds.ODDS_AUGMENTED_MODEL_BLOCKED
    )
    assert report["report_only_ev_diagnostics"]["status"] == (
        "EV_DIAGNOSTICS_BLOCKED_ODDS_RESEARCH_GATE"
    )
    approved = report["approved_odds_augmented_predictions"]
    assert approved["candidate_key"] == odds.APPROVED_ODDS_AUGMENTED_CANDIDATE_KEY
    assert approved["status"] == "APPROVED_BLEND_READY"
    assert approved["ready_race_count"] == 1
    assert approved["blocked_race_count"] == 0
    approved_rows = [
        json.loads(line)
        for line in (output_dir / "approved_odds_augmented_predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert approved_rows[0]["candidate_key"] == "stage2_market_blend_70"
    assert approved_rows[0]["approved_blend_probability"] == pytest.approx(1.0)
    assert approved_rows[0]["approved_blend_rank"] == 1
    assert approved_rows[0]["production_prediction_write"] is False
    assert approved_rows[0]["ev_output"] is False
    assert approved_rows[0]["betting_action"] is False
    assert (output_dir / "approved_odds_augmented_prediction_report.json").exists()
    policy = report["odds_research_readiness"]["odds_research_gate_policy"]
    assert policy["source_requirements"]["trusted_source_required"] is True
    assert policy["timing_requirements"]["captured_before_prediction_required"] is True
    assert policy["identity_requirements"]["dog_name_match_required"] is True
    assert policy["coverage_requirements"][
        "complete_valid_prejump_odds_required_for_odds_research_ready"
    ] is True
    assert policy["ev_policy"]["ev_output_allowed"] is False
    assert rows[0]["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert rows[0]["odds_effective_prediction_timestamp"] == (
        "2026-06-09T00:05:00+10:00"
    )
    assert rows[0]["odds_effective_prediction_timestamp_source"] == "generated_at"
    assert rows[0]["odds_effective_feature_freeze_timestamp"] is None
    assert rows[0]["odds_effective_feature_freeze_timestamp_source"] is None
    assert rows[0]["is_ev_eligible"] is True
    assert rows[0]["ev_win"] is None
    assert rows[0]["ev_calculation_status"] == "DISABLED_REPORT_ONLY_NO_EV_OUTPUT"
    assert rows[0]["odds_snapshot"]["odds_captured_before_jump"] is True
    assert rows[0]["odds_snapshot"]["odds_age_minutes_at_jump"] == pytest.approx(694.0)
    assert rows[0]["odds_snapshot"]["odds_stale_at_prediction"] is False
    provenance = rows[0]["odds_snapshot"]["odds_provenance"]
    assert provenance["sportsbet_raw_runner_text"] == "1. Alpha Runner"
    assert provenance["capture_mode"] == "autonomous_prejump_t60m"
    assert (output_dir / "shadow_odds_snapshot.csv").exists()
    assert (output_dir / "shadow_odds_race_coverage.json").exists()
    assert (output_dir / "odds_research_gate_report.json").exists()
    assert (output_dir / "odds_augmented_challenger_report.json").exists()
    assert (output_dir / "report_only_ev_diagnostics.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert race_coverage["races_with_complete_valid_prejump_odds"] == 1
    race = race_coverage["races"][0]
    assert race["odds_coverage_status"] == "COMPLETE_VALID_PREJUMP_ODDS"
    assert race["predicted_runner_count"] == 1
    assert race["runner_rows_with_odds_candidates"] == 1
    assert race["valid_pre_jump_dog_odds_rows"] == 1
    assert race["missing_odds_rows"] == 0
    assert race["post_jump_odds_rows"] == 0
    assert race["complete_valid_prejump_odds"] is True
    assert race["odds_analysis_status"] == "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
    assert race["odds_analysis_blockers"] == []
    assert race["ev_calculation_status"] == "DISABLED_REPORT_ONLY_NO_EV_OUTPUT"
    assert race["odds_source_url_count"] == 1
    assert race["selected_capture_mode_distribution"] == {
        "autonomous_prejump_t60m": 1
    }
    assert race["raw_capture_mode_distribution"] == {
        "autonomous_prejump_t60m": 1
    }
    assert race["valid_capture_mode_distribution"] == {
        "autonomous_prejump_t60m": 1
    }
    assert race["raw_missing_complete_expected_prejump_capture_modes"] == [
        "autonomous_prejump_t30m",
        "autonomous_prejump_t10m",
        "autonomous_prejump_t2m",
    ]
    assert race["selected_valid_capture_mode_distribution"] == {
        "autonomous_prejump_t60m": 1
    }
    assert race["selected_expected_prejump_capture_modes_missing"] == [
        "autonomous_prejump_t30m",
        "autonomous_prejump_t10m",
        "autonomous_prejump_t2m",
    ]


def test_odds_research_gate_ready_requires_100_complete_valid_source_url_races():
    predictions = []
    rows = []
    race_reports = []
    for index in range(100):
        race_id = f"Race {index + 1} - TEST - 2026-06-09"
        predictions.append({"race_id": race_id, "dog_name": "Alpha Runner", "box": 1})
        rows.append(
            {
                "race_id": race_id,
                "dog_name": "Alpha Runner",
                "box": 1,
                "odds_candidate_count": 1,
                "odds_match_status": "valid_pre_jump_dog_odds",
                "shadow_rf_calibrated_probability": 0.42,
                "odds_snapshot": {
                    "market_odds_win": 3.0,
                    "odds_provenance": {
                        "source_url": f"https://www.sportsbet.com.au/race-{index + 1}",
                    },
                },
            }
        )
        race_reports.append(
            {
                "race_id": race_id,
                "complete_valid_prejump_odds": True,
                "odds_analysis_status": "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED",
                "odds_analysis_blockers": [],
            }
        )
    race_coverage = {
        "race_count": 100,
        "races": race_reports,
        "races_with_complete_valid_prejump_odds": 100,
    }
    readiness = {
        "status": "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED",
        "blocker_counts": {},
    }

    gate = odds.odds_research_gate_report(
        predictions=predictions,
        rows=rows,
        race_coverage=race_coverage,
        odds_research_readiness=readiness,
        collection_status=odds.FINAL_COLLECTED,
        generated_at=datetime.fromisoformat("2026-06-09T00:10:00+10:00"),
    )
    ev = odds.report_only_ev_diagnostics(gate=gate, rows=rows)

    assert gate["status"] == odds.ODDS_RESEARCH_READY_REPORT_ONLY
    assert gate["blocker_counts"] == {}
    assert gate["source_url_coverage_pct"] == 100.0
    assert gate["odds_model_input_report_only_allowed"] is True
    assert gate["odds_used_for_shadow_scoring"] is False
    assert gate["betting_action_allowed"] is False
    assert ev["status"] == "EV_DIAGNOSTICS_REPORT_ONLY"
    assert ev["ev_rows"] == 100
    assert ev["positive_ev_rows"] == 100
    assert ev["betting_action_allowed"] is False
    assert ev["ev_can_override_accuracy_gate"] is False


def test_collect_shadow_odds_snapshot_records_missing_odds_without_ev(tmp_path, monkeypatch):
    report, rows, output_dir = _run(tmp_path, monkeypatch, include_odds=False)
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_NO_MATCHES
    assert report["odds_candidate_rows"] == 0
    assert report["races_with_complete_valid_prejump_odds"] == 0
    assert report["races_with_missing_odds_rows"] == 1
    assert rows[0]["odds_match_status"] == "no_odds_row"
    assert rows[0]["is_ev_eligible"] is False
    assert rows[0]["ev_win"] is None
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "missing_odds_rows": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "COLLECT_EXACT_PREJUMP_DOG_ODDS_FOR_ALL_RUNNERS"
    )
    gate = report["odds_research_readiness"]["ev_research_gate"]
    assert gate["status"] == "BLOCKED_REPORT_ONLY_NO_EV_OUTPUT"
    assert gate["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "missing_odds_rows": 1,
    }
    assert gate["ev_output_allowed"] is False
    race = race_coverage["races"][0]
    assert race["odds_coverage_status"] == "NO_ODDS_COVERAGE"
    assert race["missing_odds_rows"] == 1
    assert race["complete_odds_candidate_coverage"] is False
    assert race["odds_analysis_status"] == "ODDS_ANALYSIS_BLOCKED"
    assert race["odds_analysis_blockers"] == [
        "missing_odds_rows",
        "incomplete_valid_prejump_odds",
    ]
    assert report["odds_research_gate"]["incomplete_valid_prejump_odds_races"] == [
        {
            "race_id": "Race 1 - TEST - 2026-06-09",
            "odds_coverage_status": "NO_ODDS_COVERAGE",
            "valid_pre_jump_dog_odds_rows": 0,
            "predicted_runner_count": 1,
            "selected_capture_mode_distribution": {},
            "selected_valid_capture_mode_distribution": {},
            "raw_capture_mode_distribution": {},
            "valid_capture_mode_distribution": {},
            "raw_missing_complete_expected_prejump_capture_modes": [
                "autonomous_prejump_t60m",
                "autonomous_prejump_t30m",
                "autonomous_prejump_t10m",
                "autonomous_prejump_t2m",
            ],
            "valid_missing_complete_expected_prejump_capture_modes": [
                "autonomous_prejump_t60m",
                "autonomous_prejump_t30m",
                "autonomous_prejump_t10m",
                "autonomous_prejump_t2m",
            ],
            "selected_expected_prejump_capture_modes_missing": [
                "autonomous_prejump_t60m",
                "autonomous_prejump_t30m",
                "autonomous_prejump_t10m",
                "autonomous_prejump_t2m",
            ],
            "selected_valid_expected_prejump_capture_modes_missing": [
                "autonomous_prejump_t60m",
                "autonomous_prejump_t30m",
                "autonomous_prejump_t10m",
                "autonomous_prejump_t2m",
            ],
            "odds_analysis_blockers": [
                "missing_odds_rows",
                "incomplete_valid_prejump_odds",
            ],
        }
    ]
    approved = report["approved_odds_augmented_predictions"]
    assert approved["status"] == "APPROVED_BLEND_BLOCKED"
    assert approved["ready_race_count"] == 0
    assert approved["blocked_race_count"] == 1
    assert approved["prediction_rows"] == 0
    assert approved["race_reports"][0]["blockers"] == [
        "race_not_complete_valid_prejump_odds",
        "market_odds_missing_or_invalid",
        "market_probability_normalization_failed",
    ]
    assert (
        output_dir / "approved_odds_augmented_predictions.jsonl"
    ).read_text(encoding="utf-8") == ""


def test_approved_odds_augmented_blend_ranks_with_reviewed_formula(tmp_path):
    rows = [
        {
            "race_id": "Race 1 - TEST - 2026-06-09",
            "dog_name": "Model Pick",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.8,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_snapshot": {
                "market_odds_win": 4.0,
                "odds_timestamp": "2026-06-09T00:01:00+10:00",
                "odds_provenance": {"source_url": "https://sportsbet.test/r1"},
            },
        },
        {
            "race_id": "Race 1 - TEST - 2026-06-09",
            "dog_name": "Market Pick",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.2,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_snapshot": {
                "market_odds_win": 2.0,
                "odds_timestamp": "2026-06-09T00:01:00+10:00",
                "odds_provenance": {"source_url": "https://sportsbet.test/r1"},
            },
        },
    ]

    report, predictions = odds.approved_blend_prediction_report(
        rows,
        output_dir=tmp_path,
    )

    assert report["status"] == "APPROVED_BLEND_READY"
    assert report["ready_race_count"] == 1
    assert report["blocked_race_count"] == 0
    by_dog = {row["dog_name"]: row for row in predictions}
    assert by_dog["Model Pick"]["stage2_shadow_probability"] == pytest.approx(0.8)
    assert by_dog["Model Pick"]["market_implied_probability"] == pytest.approx(1 / 3)
    assert by_dog["Model Pick"]["approved_blend_probability"] == pytest.approx(
        0.30 * 0.8 + 0.70 * (1 / 3)
    )
    assert by_dog["Model Pick"]["approved_blend_rank"] == 2
    assert by_dog["Market Pick"]["approved_blend_probability"] == pytest.approx(
        0.30 * 0.2 + 0.70 * (2 / 3)
    )
    assert by_dog["Market Pick"]["approved_blend_rank"] == 1
    assert all(row["production_prediction_write"] is False for row in predictions)


def test_trusted_inactive_runners_do_not_require_odds_or_enter_blend(tmp_path):
    race_id = "Race 1 - TEST - 2026-06-09"
    predictions = [
        {"race_id": race_id, "dog_name": "Model Pick", "box": 1},
        {"race_id": race_id, "dog_name": "Scratched Runner", "box": 2},
        {"race_id": race_id, "dog_name": "Non Starter", "box": 3},
        {"race_id": race_id, "dog_name": "Market Pick", "box": 4},
    ]
    rows = [
        {
            "race_id": race_id,
            "dog_name": "Model Pick",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.8,
            "odds_candidate_count": 1,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_snapshot": {
                "market_odds_win": 4.0,
                "odds_timestamp": "2026-06-09T00:01:00+10:00",
                "odds_provenance": {"source_url": "https://sportsbet.test/r1"},
            },
        },
        {
            "race_id": race_id,
            "dog_name": "Scratched Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.99,
            "runner_status": "scratched",
            "runner_status_trusted": True,
            "runner_status_source": "sportsbet",
            "odds_candidate_count": 1,
            "odds_match_status": "trusted_scratched_runner",
            "odds_exclusion_reason": "trusted_scratched_runner",
            "odds_snapshot": {
                "runner_status": "scratched",
                "runner_status_trusted": True,
                "runner_status_source": "sportsbet",
            },
        },
        {
            "race_id": race_id,
            "dog_name": "Non Starter",
            "box": 3,
            "predicted_rank": 3,
            "shadow_rf_calibrated_probability": 0.5,
            "runner_status": "non_starter",
            "runner_status_trusted": True,
            "runner_status_source": "sportsbet",
            "odds_candidate_count": 1,
            "odds_match_status": "trusted_non_starter_runner",
            "odds_exclusion_reason": "trusted_non_starter_runner",
            "odds_snapshot": {
                "runner_status": "non_starter",
                "runner_status_trusted": True,
                "runner_status_source": "sportsbet",
            },
        },
        {
            "race_id": race_id,
            "dog_name": "Market Pick",
            "box": 4,
            "predicted_rank": 4,
            "shadow_rf_calibrated_probability": 0.2,
            "odds_candidate_count": 1,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_snapshot": {
                "market_odds_win": 2.0,
                "odds_timestamp": "2026-06-09T00:01:00+10:00",
                "odds_provenance": {"source_url": "https://sportsbet.test/r1"},
            },
        },
    ]

    coverage = odds.race_odds_coverage_report(
        predictions=predictions,
        rows=rows,
        contexts={},
        collection_status=odds.FINAL_COLLECTED,
    )
    race = coverage["races"][0]
    report, blend_rows = odds.approved_blend_prediction_report(rows, output_dir=tmp_path)

    assert coverage["races_with_complete_valid_prejump_odds"] == 1
    assert coverage["trusted_scratched_runner_rows"] == 1
    assert coverage["trusted_non_starter_runner_rows"] == 1
    assert race["predicted_runner_count"] == 4
    assert race["active_predicted_runner_count"] == 2
    assert race["valid_pre_jump_dog_odds_rows"] == 2
    assert race["missing_odds_rows"] == 0
    assert race["complete_valid_prejump_odds"] is True
    assert race["odds_analysis_blockers"] == []
    assert report["status"] == "APPROVED_BLEND_READY"
    assert report["prediction_rows"] == 2
    by_dog = {row["dog_name"]: row for row in blend_rows}
    assert set(by_dog) == {"Model Pick", "Market Pick"}
    assert by_dog["Market Pick"]["approved_blend_rank"] == 1
    assert by_dog["Model Pick"]["approved_blend_rank"] == 2


def test_trusted_inactive_runner_with_odds_price_blocks_review(tmp_path):
    race_id = "Race 1 - TEST - 2026-06-09"
    predictions = [
        {"race_id": race_id, "dog_name": "Active Runner", "box": 1},
        {"race_id": race_id, "dog_name": "Scratched Runner", "box": 2},
    ]
    rows = [
        {
            "race_id": race_id,
            "dog_name": "Active Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.6,
            "odds_candidate_count": 1,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_snapshot": {
                "market_odds_win": 2.0,
                "odds_timestamp": "2026-06-09T00:01:00+10:00",
                "odds_provenance": {"source_url": "https://sportsbet.test/r1"},
            },
        },
        {
            "race_id": race_id,
            "dog_name": "Scratched Runner",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.4,
            "runner_status": "scratched",
            "runner_status_trusted": True,
            "runner_status_source": "sportsbet",
            "odds_candidate_count": 1,
            "odds_match_status": "trusted_scratched_runner",
            "odds_exclusion_reason": "trusted_scratched_runner",
            "odds_snapshot": {
                "market_odds_win": 5.0,
                "runner_status": "scratched",
                "runner_status_trusted": True,
                "runner_status_source": "sportsbet",
            },
        },
    ]

    coverage = odds.race_odds_coverage_report(
        predictions=predictions,
        rows=rows,
        contexts={},
        collection_status=odds.FINAL_COLLECTED,
    )
    race = coverage["races"][0]
    report, blend_rows = odds.approved_blend_prediction_report(rows, output_dir=tmp_path)

    assert coverage["races_with_complete_valid_prejump_odds"] == 0
    assert coverage["trusted_inactive_runner_price_conflict_rows"] == 1
    assert race["odds_coverage_status"] == "TRUSTED_INACTIVE_RUNNER_ODDS_CONFLICT"
    assert race["complete_valid_prejump_odds"] is False
    assert race["odds_analysis_blockers"] == [
        "trusted_inactive_runner_has_odds_price"
    ]
    assert report["status"] == "APPROVED_BLEND_BLOCKED"
    assert report["prediction_rows"] == 0
    assert report["race_reports"][0]["trusted_inactive_runner_price_conflict_rows"] == 1
    assert (
        "trusted_inactive_runner_has_odds_price"
        in report["race_reports"][0]["blockers"]
    )
    assert blend_rows == []


def test_untrusted_inactive_status_still_requires_valid_odds(tmp_path):
    race_id = "Race 1 - TEST - 2026-06-09"
    predictions = [
        {"race_id": race_id, "dog_name": "Priced Runner", "box": 1},
        {"race_id": race_id, "dog_name": "Untrusted Scratch", "box": 2},
    ]
    rows = [
        {
            "race_id": race_id,
            "dog_name": "Priced Runner",
            "box": 1,
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.4,
            "odds_candidate_count": 1,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_snapshot": {
                "market_odds_win": 3.0,
                "odds_timestamp": "2026-06-09T00:01:00+10:00",
                "odds_provenance": {"source_url": "https://sportsbet.test/r1"},
            },
        },
        {
            "race_id": race_id,
            "dog_name": "Untrusted Scratch",
            "box": 2,
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.6,
            "runner_status": "scratched",
            "runner_status_trusted": False,
            "odds_candidate_count": 0,
            "odds_match_status": "no_odds_row",
            "odds_exclusion_reason": "no_odds_row",
            "odds_snapshot": {
                "runner_status": "scratched",
                "runner_status_trusted": False,
            },
        },
    ]

    coverage = odds.race_odds_coverage_report(
        predictions=predictions,
        rows=rows,
        contexts={},
        collection_status=odds.FINAL_COLLECTED,
    )
    race = coverage["races"][0]
    report, blend_rows = odds.approved_blend_prediction_report(rows, output_dir=tmp_path)

    assert coverage["races_with_complete_valid_prejump_odds"] == 0
    assert coverage["trusted_inactive_runner_rows"] == 0
    assert race["active_predicted_runner_count"] == 2
    assert race["missing_odds_rows"] == 1
    assert race["complete_valid_prejump_odds"] is False
    assert race["odds_analysis_blockers"] == [
        "missing_odds_rows",
        "incomplete_valid_prejump_odds",
    ]
    assert report["status"] == "APPROVED_BLEND_BLOCKED"
    assert report["prediction_rows"] == 0
    assert blend_rows == []
    assert "race_not_complete_valid_prejump_odds" in report["race_reports"][0]["blockers"]
    assert "market_odds_missing_or_invalid" in report["race_reports"][0]["blockers"]


def test_collect_shadow_odds_snapshot_rejects_duplicate_odds_rows(tmp_path, monkeypatch):
    report, rows, output_dir = _run(tmp_path, monkeypatch, duplicate_odds=True)
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_complete_odds_candidate_coverage"] == 1
    assert report["races_with_complete_valid_prejump_odds"] == 0
    assert report["races_with_duplicate_odds_rows"] == 1
    assert rows[0]["odds_candidate_count"] == 2
    assert rows[0]["odds_match_status"] == "duplicate_odds_rows"
    assert rows[0]["is_ev_eligible"] is False
    assert rows[0]["ev_win"] is None
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "duplicate_odds_rows": 1,
        "incomplete_valid_prejump_odds": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "FIX_ODDS_DEDUPLICATION_OR_IDENTITY_PROVENANCE"
    )
    race = race_coverage["races"][0]
    assert race["odds_coverage_status"] == "COMPLETE_CANDIDATE_COVERAGE_WITH_REJECTIONS"
    assert race["total_odds_candidate_count"] == 2
    assert race["duplicate_odds_rows"] == 1
    assert race["complete_valid_prejump_odds"] is False
    assert race["odds_analysis_blockers"] == [
        "duplicate_odds_rows",
        "incomplete_valid_prejump_odds",
    ]


def test_collect_shadow_odds_snapshot_accepts_distinct_capture_windows(
    tmp_path,
    monkeypatch,
):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        repeated_capture_windows=True,
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["odds_candidate_rows"] == 1
    assert report["valid_pre_jump_dog_odds_rows"] == 1
    assert report["races_with_complete_valid_prejump_odds"] == 1
    assert report["races_with_duplicate_odds_rows"] == 0
    assert rows[0]["odds_candidate_count"] == 1
    assert rows[0]["odds_raw_candidate_count"] == 2
    assert rows[0]["odds_duplicate_candidate_count"] == 0
    assert rows[0]["odds_ignored_candidate_count"] == 1
    assert rows[0]["odds_selection_status"] == "selected_latest_valid_prejump_capture"
    assert rows[0]["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert rows[0]["odds_snapshot"]["market_odds_win"] == 3.4
    assert rows[0]["odds_snapshot"]["odds_provenance"]["capture_mode"] == (
        "autonomous_prejump_t30m"
    )
    assert rows[0]["odds_snapshot"]["odds_provenance"]["candidate_count"] == 1
    assert rows[0]["odds_snapshot"]["odds_provenance"]["duplicate_count"] == 1
    race = race_coverage["races"][0]
    assert race["total_odds_candidate_count"] == 1
    assert race["duplicate_odds_rows"] == 0
    assert race["complete_valid_prejump_odds"] is True
    assert race["odds_analysis_blockers"] == []
    assert race["selected_capture_mode_distribution"] == {
        "autonomous_prejump_t30m": 1
    }
    assert race["raw_capture_mode_distribution"] == {
        "autonomous_prejump_t30m": 1,
        "autonomous_prejump_t60m": 1,
    }
    assert race["valid_capture_mode_distribution"] == {
        "autonomous_prejump_t30m": 1,
        "autonomous_prejump_t60m": 1,
    }
    assert race["raw_missing_complete_expected_prejump_capture_modes"] == [
        "autonomous_prejump_t10m",
        "autonomous_prejump_t2m",
    ]
    assert race["selected_valid_capture_mode_distribution"] == {
        "autonomous_prejump_t30m": 1
    }
    assert race["selected_expected_prejump_capture_modes_missing"] == [
        "autonomous_prejump_t60m",
        "autonomous_prejump_t10m",
        "autonomous_prejump_t2m",
    ]


def test_collect_shadow_odds_snapshot_blocks_post_prediction_odds(tmp_path, monkeypatch):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_timestamp="2026-06-09T00:06:00+10:00",
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_post_prediction_odds_rows"] == 1
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "timestamp_after_prediction": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "CAPTURE_ODDS_BEFORE_SHADOW_PREDICTION_AND_FEATURE_FREEZE"
    )
    assert rows[0]["odds_match_status"] == "timestamp_after_prediction"
    assert rows[0]["odds_exclusion_reason"] == "timestamp_after_prediction"
    race = race_coverage["races"][0]
    assert race["post_prediction_odds_rows"] == 1
    assert race["post_jump_odds_rows"] == 0
    assert race["odds_analysis_blockers"] == [
        "timestamp_after_prediction",
        "incomplete_valid_prejump_odds",
    ]
    assert (output_dir / "shadow_odds_research_readiness.json").exists()


def test_odds_readiness_recommends_shadow_rerun_when_raw_windows_complete_after_prediction():
    race_id = "Race 1 - TEST - 2026-06-09"
    readiness = odds.odds_research_readiness_report(
        predictions=[{"race_id": race_id, "dog_name": "Alpha Runner", "box": 1}],
        race_coverage={
            "race_count": 1,
            "races_with_complete_valid_prejump_odds": 0,
            "races": [
                {
                    "race_id": race_id,
                    "odds_coverage_status": "COMPLETE_CANDIDATE_COVERAGE_WITH_REJECTIONS",
                    "predicted_runner_count": 1,
                    "valid_pre_jump_dog_odds_rows": 0,
                    "post_prediction_odds_rows": 4,
                    "complete_valid_prejump_odds": False,
                    "raw_capture_mode_distribution": {
                        "autonomous_prejump_t60m": 1,
                        "autonomous_prejump_t30m": 1,
                        "autonomous_prejump_t10m": 1,
                        "autonomous_prejump_t2m": 1,
                    },
                    "valid_capture_mode_distribution": {},
                    "raw_complete_expected_prejump_capture_modes": [
                        "autonomous_prejump_t60m",
                        "autonomous_prejump_t30m",
                        "autonomous_prejump_t10m",
                        "autonomous_prejump_t2m",
                    ],
                    "raw_missing_complete_expected_prejump_capture_modes": [],
                    "valid_complete_expected_prejump_capture_modes": [],
                    "valid_missing_complete_expected_prejump_capture_modes": [
                        "autonomous_prejump_t60m",
                        "autonomous_prejump_t30m",
                        "autonomous_prejump_t10m",
                        "autonomous_prejump_t2m",
                    ],
                    "odds_analysis_blockers": [
                        "timestamp_after_prediction",
                        "incomplete_valid_prejump_odds",
                    ],
                }
            ],
        },
        collection_status=odds.FINAL_COLLECTED,
        odds_source_report={"live_odds_table_available": True},
    )

    assert readiness["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert readiness["odds_research_next_action"] == (
        "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
    )
    assert readiness["timing_aligned_prediction_rerun_required"] is True
    assert readiness["timing_aligned_prediction_rerun_race_count"] == 1
    assert readiness["timing_aligned_prediction_rerun_reason_counts"] == {
        "raw_expected_prejump_windows_complete_but_after_prediction": 1
    }
    assert readiness["timing_aligned_prediction_rerun_races"] == [
        {
            "race_id": race_id,
            "reason": "raw_expected_prejump_windows_complete_but_after_prediction",
            "raw_capture_mode_distribution": {
                "autonomous_prejump_t60m": 1,
                "autonomous_prejump_t30m": 1,
                "autonomous_prejump_t10m": 1,
                "autonomous_prejump_t2m": 1,
            },
            "valid_capture_mode_distribution": {},
            "raw_complete_expected_prejump_capture_modes": [
                "autonomous_prejump_t60m",
                "autonomous_prejump_t30m",
                "autonomous_prejump_t10m",
                "autonomous_prejump_t2m",
            ],
            "valid_complete_expected_prejump_capture_modes": [],
            "post_prediction_odds_rows": 4,
            "predicted_runner_count": 1,
            "valid_pre_jump_dog_odds_rows": 0,
        }
    ]


def test_collect_shadow_odds_snapshot_blocks_post_jump_odds_when_jump_context_available(
    tmp_path,
    monkeypatch,
):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_timestamp="2026-06-09T00:06:00+10:00",
        manifest_generated_at="2026-06-09T00:10:00+10:00",
        jump_time="12:05 AM",
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_post_prediction_odds_rows"] == 0
    assert report["races_with_post_jump_odds_rows"] == 1
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "timestamp_after_jump": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "CAPTURE_ODDS_BEFORE_RACE_JUMP"
    )
    assert rows[0]["odds_match_status"] == "timestamp_after_jump"
    assert rows[0]["odds_exclusion_reason"] == "timestamp_after_jump"
    assert rows[0]["odds_snapshot"]["odds_captured_before_prediction"] is True
    assert rows[0]["odds_snapshot"]["odds_captured_before_jump"] is False
    race = race_coverage["races"][0]
    assert race["post_prediction_odds_rows"] == 0
    assert race["post_jump_odds_rows"] == 1
    assert race["odds_analysis_blockers"] == [
        "timestamp_after_jump",
        "incomplete_valid_prejump_odds",
    ]


def test_collect_shadow_odds_snapshot_blocks_after_feature_freeze_odds_when_available(
    tmp_path,
    monkeypatch,
):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_timestamp="2026-06-09T00:06:00+10:00",
        manifest_generated_at="2026-06-09T00:20:00+10:00",
        prediction_timestamp="2026-06-09T00:10:00+10:00",
        feature_freeze_timestamp="2026-06-09T00:05:00+10:00",
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_post_prediction_odds_rows"] == 0
    assert report["races_with_post_feature_freeze_odds_rows"] == 1
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "timestamp_after_feature_freeze": 1,
    }
    assert rows[0]["odds_match_status"] == "timestamp_after_feature_freeze"
    assert rows[0]["odds_exclusion_reason"] == "timestamp_after_feature_freeze"
    assert rows[0]["odds_effective_prediction_timestamp"] == (
        "2026-06-09T00:10:00+10:00"
    )
    assert rows[0]["odds_effective_prediction_timestamp_source"] == (
        "score_live_manifest.prediction_timestamp"
    )
    assert rows[0]["odds_effective_feature_freeze_timestamp"] == (
        "2026-06-09T00:05:00+10:00"
    )
    assert rows[0]["odds_effective_feature_freeze_timestamp_source"] == (
        "score_live_manifest.feature_freeze_timestamp"
    )
    assert rows[0]["odds_snapshot"]["odds_captured_before_prediction"] is True
    assert rows[0]["odds_snapshot"]["odds_captured_before_feature_freeze"] is False
    assert rows[0]["odds_snapshot"]["odds_age_minutes_at_feature_freeze"] == pytest.approx(
        -1.0
    )
    assert rows[0]["odds_snapshot"]["odds_captured_before_jump"] is True
    race = race_coverage["races"][0]
    assert race["post_prediction_odds_rows"] == 0
    assert race["post_feature_freeze_odds_rows"] == 1
    assert race["post_jump_odds_rows"] == 0


def test_collect_shadow_odds_snapshot_blocks_stale_odds(tmp_path, monkeypatch):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_timestamp="2026-06-08T23:00:00+10:00",
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_stale_odds_rows"] == 1
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "stale_beyond_ttl": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "REFRESH_ODDS_WITHIN_TTL_BEFORE_FEATURE_FREEZE"
    )
    assert rows[0]["odds_match_status"] == "stale_beyond_ttl"
    assert rows[0]["odds_exclusion_reason"] == "stale_beyond_ttl"
    assert rows[0]["odds_snapshot"]["odds_stale_at_prediction"] is True
    assert rows[0]["is_ev_eligible"] is False
    assert rows[0]["ev_win"] is None
    race = race_coverage["races"][0]
    assert race["stale_odds_rows"] == 1
    assert race["odds_analysis_blockers"] == [
        "stale_beyond_ttl",
        "incomplete_valid_prejump_odds",
    ]


def test_collect_shadow_odds_snapshot_surfaces_source_provenance_blockers(
    tmp_path,
    monkeypatch,
):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_source_url="",
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_missing_source_url_rows"] == 1
    assert report["odds_research_readiness"]["status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "missing_source_url": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "FIX_ODDS_SOURCE_PROVENANCE"
    )
    assert rows[0]["odds_match_status"] == "missing_source_url"
    assert rows[0]["odds_exclusion_reason"] == "missing_source_url"
    assert rows[0]["ev_win"] is None
    race = race_coverage["races"][0]
    assert race["missing_source_url_rows"] == 1
    assert race["odds_analysis_blockers"] == [
        "missing_source_url",
        "incomplete_valid_prejump_odds",
    ]


def test_collect_shadow_odds_snapshot_surfaces_untrusted_source_blocker(
    tmp_path,
    monkeypatch,
):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_source="unknownbook",
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_untrusted_source_rows"] == 1
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "untrusted_source": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "FIX_ODDS_SOURCE_PROVENANCE"
    )
    assert rows[0]["odds_match_status"] == "untrusted_source"
    assert rows[0]["ev_win"] is None
    assert race_coverage["races"][0]["untrusted_source_rows"] == 1


def test_collect_shadow_odds_snapshot_surfaces_post_result_source_url_blocker(
    tmp_path,
    monkeypatch,
):
    report, rows, output_dir = _run(
        tmp_path,
        monkeypatch,
        odds_source_url=(
            "https://www.sportsbet.com.au/greyhound-racing/results/test/race-1"
        ),
    )
    race_coverage = _race_coverage(output_dir)

    assert report["final_status"] == odds.FINAL_COLLECTED
    assert report["races_with_post_race_or_sp_only_rows"] == 1
    assert report["odds_research_readiness"]["blocker_counts"] == {
        "incomplete_valid_prejump_odds": 1,
        "post_race_or_sp_only": 1,
    }
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "FIX_ODDS_SOURCE_PROVENANCE"
    )
    assert rows[0]["odds_match_status"] == "post_race_or_sp_only"
    assert rows[0]["ev_win"] is None
    assert race_coverage["races"][0]["post_race_or_sp_only_rows"] == 1


def test_collect_shadow_odds_snapshot_rejects_non_positive_stale_ttl(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(odds, "ROOT", tmp_path)
    monkeypatch.setattr(odds, "EXPECTED_OFFICIAL_RACES", 1)
    monkeypatch.setattr(odds, "EXPECTED_OFFICIAL_DOG_ROWS", 1)
    db_path = tmp_path / "greyhound_racing_data.db"
    monkeypatch.setattr(odds, "PROTECTED_PATHS", (db_path,))
    shadow_run = tmp_path / "shadow_run"
    _write_shadow_run(shadow_run)
    _build_db(db_path)

    with pytest.raises(ValueError, match="stale_odds_after_minutes_must_be_positive"):
        odds.collect_shadow_odds_snapshot(
            shadow_run_dir=shadow_run,
            db_path=db_path,
            output_dir=(
                tmp_path
                / "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_test"
            ),
            current_time=datetime.fromisoformat("2026-06-09T00:10:00+10:00"),
            stale_odds_after_minutes=0,
        )
