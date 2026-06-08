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
                "dog",
                "runner_text",
                1,
                "1. Alpha Runner",
            )
        ]
        if duplicate_odds:
            rows.append((*rows[0][:-10], 3.4, *rows[0][-9:]))
        conn.executemany(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, dog_name, dog_clean_name,
                box_number, odds_decimal, market_type, source, timestamp, is_current,
                source_url, odds_level, sportsbet_box_source, sportsbet_list_position,
                sportsbet_raw_runner_text
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    assert report["odds_research_readiness"]["blocker_counts"] == {}
    assert report["odds_research_readiness"]["odds_research_next_action"] == (
        "REPORT_ONLY_REVIEW_ODDS_CALIBRATION_NO_EV_ACTION"
    )
    gate = report["odds_research_readiness"]["ev_research_gate"]
    assert gate["status"] == "READY_FOR_REPORT_ONLY_ODDS_REVIEW_NO_EV_OUTPUT"
    assert gate["ev_output_allowed"] is False
    assert gate["betting_action_allowed"] is False
    policy = report["odds_research_readiness"]["odds_research_gate_policy"]
    assert policy["source_requirements"]["trusted_source_required"] is True
    assert policy["timing_requirements"]["captured_before_prediction_required"] is True
    assert policy["identity_requirements"]["dog_name_match_required"] is True
    assert policy["coverage_requirements"][
        "complete_valid_prejump_odds_required_for_odds_research_ready"
    ] is True
    assert policy["ev_policy"]["ev_output_allowed"] is False
    assert rows[0]["odds_match_status"] == "valid_pre_jump_dog_odds"
    assert rows[0]["is_ev_eligible"] is True
    assert rows[0]["ev_win"] is None
    assert rows[0]["ev_calculation_status"] == "DISABLED_REPORT_ONLY_NO_EV_OUTPUT"
    assert rows[0]["odds_snapshot"]["odds_captured_before_jump"] is True
    assert rows[0]["odds_snapshot"]["odds_stale_at_prediction"] is False
    assert (output_dir / "shadow_odds_snapshot.csv").exists()
    assert (output_dir / "shadow_odds_race_coverage.json").exists()
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
    assert rows[0]["odds_snapshot"]["odds_captured_before_prediction"] is True
    assert rows[0]["odds_snapshot"]["odds_captured_before_feature_freeze"] is False
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
