import os
import json
import sqlite3
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from scripts import autonomous_official_result_capture as capture


def _write_shadow_source_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Dog Name,Box",
                "1. Alpha,1",
                "2. Bravo,2",
                "3. Charlie,3",
                "4. Delta,4",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_shadow_run(
    tmp_path: Path,
    *,
    source_csv: Path,
    race_id: str = "Race 1 - WPK - 2026-06-10",
    race_time_minutes: int = 14 * 60,
    dirname: str = "shadow_run",
) -> Path:
    shadow_run_dir = tmp_path / dirname
    shadow_run_dir.mkdir()
    identity = capture.parse_race_identity(race_id)
    predictions = [
        {"race_id": race_id, "box": 1, "dog_name": "Alpha", "stage2_probability": 0.4},
        {"race_id": race_id, "box": 2, "dog_name": "Bravo", "stage2_probability": 0.3},
        {"race_id": race_id, "box": 3, "dog_name": "Charlie", "stage2_probability": 0.2},
        {"race_id": race_id, "box": 4, "dog_name": "Delta", "stage2_probability": 0.1},
    ]
    (shadow_run_dir / "stage2_shadow_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions),
        encoding="utf-8",
    )
    feature_rows = [
        {
            "race_id": race_id,
            "race_date": identity["race_date"],
            "race_number": identity["race_number"],
            "race_time_minutes_since_midnight": race_time_minutes,
            "venue": identity["venue"],
            "box_number": row["box"],
            "dog_name": row["dog_name"],
            "source_csv": str(source_csv),
            "target_metadata_source_url": (
                "https://www.thedogs.com.au/racing/wentworth-park/"
                f"{identity['race_date']}/{identity['race_number']}/test-race?trial=false"
            ),
        }
        for row in predictions
    ]
    (shadow_run_dir / "shadow_feature_rows.json").write_text(
        json.dumps(feature_rows),
        encoding="utf-8",
    )
    return shadow_run_dir


def test_parse_args_defaults_to_wide_live_odds_backlog_scan():
    args = capture.parse_args(["--shadow-run-dir", "shadow"])

    assert args.backlog_limit == capture.DEFAULT_BACKLOG_LIMIT
    assert args.backlog_limit == 128
    assert args.backlog_shadow_run_limit == capture.DEFAULT_BACKLOG_SHADOW_RUN_LIMIT
    assert args.backlog_shadow_run_limit == 200
    assert args.backlog_lookback_days == capture.DEFAULT_BACKLOG_LOOKBACK_DAYS
    assert args.backlog_lookback_days == 2


def test_flush_official_result_progress_writes_active_candidate(tmp_path):
    output_dir = tmp_path / "capture"

    capture.flush_official_result_progress(
        output_dir,
        candidates=[object(), object()],
        progress_rows=[
            {
                "race_id": "Race 1 - WPK - 2026-06-12",
                "status": "INGESTED_DRY_RUN",
            }
        ],
        active_row={
            "race_id": "Race 2 - WPK - 2026-06-12",
            "status": "FETCH_IN_PROGRESS",
        },
    )

    progress = json.loads(
        (output_dir / "autonomous_official_result_capture_progress.json").read_text(
            encoding="utf-8"
        )
    )
    attempts = [
        json.loads(line)
        for line in (
            output_dir / "autonomous_official_result_capture_attempts.progress.jsonl"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert progress["candidate_count"] == 2
    assert progress["completed_count"] == 1
    assert progress["active_candidate"]["race_id"] == "Race 2 - WPK - 2026-06-12"
    assert progress["status_counts"] == {
        "FETCH_IN_PROGRESS": 1,
        "INGESTED_DRY_RUN": 1,
    }
    assert progress["no_write_guarantees"]["db_write"] is False
    assert progress["no_write_guarantees"]["label_write"] is False
    assert [row["status"] for row in attempts] == [
        "INGESTED_DRY_RUN",
        "FETCH_IN_PROGRESS",
    ]


def test_ingest_dry_run_command_is_fetch_only_and_output_scoped():
    command = capture.ingest_dry_run_command(
        db_path=Path("greyhound_racing_data.db"),
        target_date="2026-06-10",
        upcoming_dir=Path("upcoming"),
        snapshot_dir=Path("artifacts/prediction_snapshots"),
        output_path=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x/dry_run.json"
        ),
        race_ids=["Race 1 - WPK - 2026-06-10"],
        require_ready_snapshot=True,
    )

    assert "scripts/ingest_results_for_date.py" in command[1]
    assert "--dry-run" in command
    assert "--write-labels-approved" not in command
    assert "--require-ready-snapshot" in command
    assert command[-2:] == ["--race-id", "Race 1 - WPK - 2026-06-10"]


def test_build_artifact_rows_keeps_official_rows_and_quarantines_unsafe():
    generated_at = datetime.fromisoformat("2026-06-10T14:00:00+10:00")
    report = {
        "scope": {
            "date": "2026-06-10",
            "db_path": "/db.sqlite",
        },
        "ingested": [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "venue": "WPK",
                "race_number": 1,
                "race_date": "2026-06-10",
                "race_time": "14:30",
                "source": "thedogs_official",
                "source_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/results",
                "status": "resulted",
                "winner_name": "Alpha",
                "winner_box": 1,
                "box_order": [1, 2],
                "participant_source": "snapshot",
                "positions": [
                    {"box_number": 1, "finish_position": 1, "dog_name": "Alpha"},
                    {"box_number": 2, "finish_position": 2, "dog_name": "Bravo"},
                ],
                "participants": [
                    {"box_number": 1, "dog_name": "Alpha"},
                    {"box_number": 2, "dog_name": "Bravo"},
                ],
            },
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "source": "sportsbet_results_top4",
                "status": "partial_sportsbet_results",
                "box_order": [3, 4],
            },
        ],
        "failed": [
            {
                "race_id": "Race 3 - WPK - 2026-06-10",
                "errors": ["result_boxes_not_in_participants:8"],
            }
        ],
        "skipped": [
            {
                "race_id": "Race 4 - WPK - 2026-06-10",
                "reason": "race_not_jumped:upcoming_not_jumped",
            }
        ],
    }

    rows = capture.build_artifact_rows(report, generated_at=generated_at)

    assert len(rows["race_rows"]) == 1
    assert rows["race_rows"][0]["source"] == "thedogs_official"
    assert rows["race_rows"][0]["position_count"] == 2
    assert len(rows["runner_rows"]) == 2
    assert rows["runner_rows"][0]["is_winner"] is True
    assert len(rows["quarantine_rows"]) == 3
    assert {row["race_id"] for row in rows["quarantine_rows"]} == {
        "Race 2 - WPK - 2026-06-10",
        "Race 3 - WPK - 2026-06-10",
        "Race 4 - WPK - 2026-06-10",
    }


def test_capture_report_status_reflects_official_rows():
    generated_at = datetime.fromisoformat("2026-06-10T14:00:00+10:00")

    report = capture.build_capture_report(
        generated_at=generated_at,
        dry_run_command=["python", "scripts/ingest_results_for_date.py"],
        dry_run_returncode=0,
        ingest_report={
            "status": "SUCCESS",
            "candidate_count": 1,
            "ingested_count": 1,
            "shadow_run_candidate_source_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "shadow_run_candidate_source_report.json"
            ),
            "live_odds_backlog": {
                "enabled": True,
                "backlog_lookback_days": 2,
                "target_dates": ["2026-06-10", "2026-06-09"],
                "discovered_race_ids": [
                    "Race 1 - WPK - 2026-06-10",
                    "Race 2 - WPK - 2026-06-09",
                ],
                "candidate_race_ids": ["Race 1 - WPK - 2026-06-10"],
                "unresolved_race_ids": ["Race 2 - WPK - 2026-06-09"],
                "unresolved_races": [
                    {
                        "race_id": "Race 2 - WPK - 2026-06-09",
                        "race_date": "2026-06-09",
                        "latest_capture": "2026-06-09T13:30:00+10:00",
                        "reason": "no_matching_shadow_run_candidate_found",
                        "parsed_identity": {
                            "race_number": 2,
                            "venue": "WPK",
                            "race_date": "2026-06-09",
                        },
                        "shadow_run_report_count": 5,
                        "shadow_run_skip_reasons": [],
                        "source": "source_backed_live_odds_without_official_results",
                    }
                ],
                    "unresolved_reason_counts": {
                        "no_matching_shadow_run_candidate_found": 1
                    },
                    "unresolved_recovery_action_counts": {
                        "inspect_shadow_run_candidate_coverage": 1
                    },
                    "unresolved_alias_status_counts": {
                        "NO_EXACT_SHADOW_ARTIFACT_MATCH": 1
                    },
                    "retryable_exact_shadow_match_race_count": 0,
                    "no_exact_shadow_match_race_count": 1,
                    "retryable_exact_shadow_match_race_ids": [],
                    "no_exact_shadow_match_race_ids": ["Race 2 - WPK - 2026-06-09"],
                },
            },
        artifact_rows={
            "race_rows": [{"race_id": "Race 1"}],
            "runner_rows": [{"race_id": "Race 1", "box_number": 1}],
            "quarantine_rows": [],
        },
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
    )

    assert report["final_status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    assert report["official_result_race_rows"] == 1
    assert report["official_result_runner_rows"] == 1
    assert report["live_odds_backlog_discovered_race_ids"] == [
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-09",
    ]
    assert report["live_odds_backlog_candidate_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["live_odds_backlog_unresolved_race_ids"] == [
        "Race 2 - WPK - 2026-06-09"
    ]
    assert report["live_odds_backlog_unresolved_races"][0]["reason"] == (
        "no_matching_shadow_run_candidate_found"
    )
    assert report["live_odds_backlog_unresolved_reason_counts"] == {
        "no_matching_shadow_run_candidate_found": 1
    }
    assert report["live_odds_backlog_unresolved_recovery_action_counts"] == {
        "inspect_shadow_run_candidate_coverage": 1
    }
    assert report["live_odds_backlog_unresolved_alias_status_counts"] == {
        "NO_EXACT_SHADOW_ARTIFACT_MATCH": 1
    }
    assert report["live_odds_backlog_retryable_exact_shadow_match_race_count"] == 0
    assert report["live_odds_backlog_no_exact_shadow_match_race_count"] == 1
    assert report["live_odds_backlog_retryable_exact_shadow_match_race_ids"] == []
    assert report["live_odds_backlog_no_exact_shadow_match_race_ids"] == [
        "Race 2 - WPK - 2026-06-09"
    ]
    assert report["live_odds_backlog_recovery_queue_path"].endswith(
        "live_odds_backlog_recovery_queue.json"
    )
    assert report["shadow_run_candidate_source_report"].endswith(
        "shadow_run_candidate_source_report.json"
    )
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["label_write"] is False


def test_capture_report_surfaces_quarantine_error_sources():
    report = capture.build_capture_report(
        generated_at=datetime.fromisoformat("2026-06-14T00:04:00+10:00"),
        dry_run_command=["python", "scripts/ingest_results_for_date.py"],
        dry_run_returncode=0,
        ingest_report={
            "status": "FAILED",
            "candidate_count": 1,
            "failed_count": 1,
            "shadow_run_candidate_source_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "shadow_run_candidate_source_report.json"
            ),
            "live_odds_backlog": {"enabled": False},
        },
        artifact_rows={
            "race_rows": [],
            "runner_rows": [],
            "quarantine_rows": [
                {
                    "race_id": "Race 12 - GRDN - 2026-06-13",
                    "reason": "ingest_failed_or_unsafe_match",
                    "item": {
                        "race_id": "Race 12 - GRDN - 2026-06-13",
                        "participant_source": "shadow_run_predictions",
                        "participant_count": 6,
                        "participant_boxes": [1, 2, 3, 4, 5, 6],
                        "participants": [
                            {"box_number": 1, "dog_name": "Alpha"},
                            {"box_number": 2, "dog_name": "Bravo"},
                            {"box_number": 3, "dog_name": "Charlie"},
                            {"box_number": 4, "dog_name": "Delta"},
                            {"box_number": 5, "dog_name": "Echo"},
                            {"box_number": 6, "dog_name": "Foxtrot"},
                        ],
                        "errors": [
                            "result_boxes_not_in_participants:9; "
                            "fallback:result_boxes_not_in_participants:9"
                        ],
                        "attempted_sources": [
                            {
                                "source": "thedogs_official",
                                "status": "resulted",
                                "source_url": "https://www.thedogs.com.au/racing/the-gardens/2026-06-13/12/red-tv-pathways?trial=false",
                                "raw_order": [6, 5, 7, 9, 2, 1, 4],
                                "terminal_statuses": [
                                    {"box_number": 3, "status": "SCR"},
                                    {"box_number": 10, "status": "SCR"},
                                ],
                            },
                            {
                                "source": "sportsbet_results_top4",
                                "status": "partial_sportsbet_results",
                                "source_url": "https://www.sportsbet.com.au/results/2026-06-13/racing/greyhound-racing-4/the-gardens-63",
                                "raw_order": [6, 5, 7, 9],
                            },
                        ],
                    },
                }
            ],
        },
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
    )

    assert report["final_status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_QUARANTINED"
    assert report["quarantined_race_ids"] == ["Race 12 - GRDN - 2026-06-13"]
    assert report["quarantine_reason_counts"] == {
        "ingest_failed_or_unsafe_match": 1
    }
    assert report["quarantine_error_counts"] == {
        "result_boxes_not_in_participants:9; "
        "fallback:result_boxes_not_in_participants:9": 1
    }
    assert report["quarantine_attempted_source_counts"] == {
        "sportsbet_results_top4": 1,
        "thedogs_official": 1,
    }
    assert report["quarantine_result_boxes_not_in_participants_counts"] == {"9": 2}
    sample = report["quarantine_samples"][0]
    assert sample["race_id"] == "Race 12 - GRDN - 2026-06-13"
    assert sample["attempted_sources"][0]["raw_order"] == [6, 5, 7, 9, 2, 1, 4]
    assert sample["attempted_sources"][0]["terminal_statuses"] == [
        {"box_number": 3, "status": "SCR"},
        {"box_number": 10, "status": "SCR"},
    ]
    mismatch_sample = report["quarantine_runner_set_mismatch_samples"][0]
    assert mismatch_sample["race_id"] == "Race 12 - GRDN - 2026-06-13"
    assert mismatch_sample["participant_source"] == "shadow_run_predictions"
    assert mismatch_sample["participant_count"] == 6
    assert mismatch_sample["participant_boxes"] == [1, 2, 3, 4, 5, 6]
    assert mismatch_sample["participants"][0] == {
        "box_number": 1,
        "dog_name": "Alpha",
    }
    assert mismatch_sample["result_boxes_not_in_participants"] == [9]
    assert mismatch_sample["result_boxes_in_participants"] == [1, 2, 4, 5, 6]
    assert mismatch_sample["attempted_source_box_sets"][0] == {
        "source": "thedogs_official",
        "status": "resulted",
        "source_url": (
            "https://www.thedogs.com.au/racing/the-gardens/2026-06-13/12/"
            "red-tv-pathways?trial=false"
        ),
        "result_boxes": [6, 5, 7, 9, 2, 1, 4],
        "dog_names_by_box": {},
        "terminal_status_boxes": [3, 10],
        "terminal_statuses": [
            {"box_number": 3, "status": "SCR"},
            {"box_number": 10, "status": "SCR"},
        ],
    }


def test_quarantine_summary_excludes_browser_sentinel_from_race_ids():
    summary = capture.summarize_quarantine_rows(
        [
            {
                "race_id": "__browser__",
                "reason": "browser_unavailable:ModuleNotFoundError",
                "item": {"errors": ["browser_unavailable"]},
            },
            {
                "race_id": "Race 8 - TAREE - 2026-06-13",
                "reason": "ingest_failed_or_unsafe_match",
                "item": {"errors": ["result_boxes_not_in_participants:9,10"]},
            },
        ]
    )

    assert summary["race_ids"] == ["Race 8 - TAREE - 2026-06-13"]
    assert summary["reason_counts"] == {
        "browser_unavailable:ModuleNotFoundError": 1,
        "ingest_failed_or_unsafe_match": 1,
    }
    assert summary["result_boxes_not_in_participants_counts"] == {"10": 1, "9": 1}


def test_capture_report_surfaces_awaiting_jump_recheck_plan():
    report = capture.build_capture_report(
        generated_at=datetime.fromisoformat("2026-06-13T22:31:00+10:00"),
        dry_run_command=["python", "scripts/autonomous_official_result_capture.py"],
        dry_run_returncode=0,
        ingest_report={
            "status": "DATA_MISSING",
            "candidate_count": 0,
            "ingested_count": 0,
            "skipped_count": 2,
            "skipped": [
                {
                    "race_id": "Race 8 - CANN - 2026-06-13",
                    "race_time": "23:21",
                    "jump_datetime": "2026-06-13T23:21:00+10:00",
                    "reason": "race_not_jumped:upcoming_not_jumped",
                },
                {
                    "race_id": "Race 12 - GRDN - 2026-06-13",
                    "race_time": "22:55",
                    "jump_datetime": "2026-06-13T22:55:00+10:00",
                    "reason": "race_not_jumped:upcoming_not_jumped",
                },
            ],
            "shadow_run_candidate_source_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "shadow_run_candidate_source_report.json"
            ),
            "live_odds_backlog": {"enabled": False},
        },
        artifact_rows={
            "race_rows": [],
            "runner_rows": [],
            "quarantine_rows": [
                {"race_id": "Race 8 - CANN - 2026-06-13"},
                {"race_id": "Race 12 - GRDN - 2026-06-13"},
            ],
        },
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
    )

    assert report["final_status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_AWAITING_JUMP"
    assert report["skipped_reason_counts"] == {
        "race_not_jumped:upcoming_not_jumped": 2
    }
    assert report["awaiting_jump_race_count"] == 2
    assert report["awaiting_jump_race_ids"] == [
        "Race 12 - GRDN - 2026-06-13",
        "Race 8 - CANN - 2026-06-13",
    ]
    assert (
        report["awaiting_jump_next_recheck_after_local"]
        == "2026-06-13T22:55:00+10:00"
    )
    assert report["awaiting_jump_races"][0]["race_id"] == (
        "Race 12 - GRDN - 2026-06-13"
    )
    assert report["no_write_guarantees"]["label_write"] is False


def test_live_odds_backlog_recovery_queue_groups_races_without_joining():
    capture_report = {
        "generated_at": "2026-06-10T14:00:00+10:00",
        "output_dir": (
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
        "live_odds_backlog_enabled": True,
        "live_odds_backlog_unresolved_race_count": 2,
        "live_odds_backlog_unresolved_reason_counts": {
            "shadow_run_candidate_rejected": 1,
            "no_matching_shadow_run_candidate_found": 1,
        },
        "live_odds_backlog_unresolved_recovery_action_counts": {
            "validate_runner_set_then_alias_join": 1,
            "inspect_shadow_run_candidate_coverage": 1,
        },
        "live_odds_backlog_unresolved_alias_status_counts": {
            "EXACT_SHADOW_ARTIFACT_MATCH_FOUND": 1,
            "NO_EXACT_SHADOW_ARTIFACT_MATCH": 1,
        },
        "live_odds_backlog_retryable_exact_shadow_match_race_count": 1,
        "live_odds_backlog_no_exact_shadow_match_race_count": 1,
        "live_odds_backlog_unresolved_races": [
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "source": "source_backed_live_odds_without_official_results",
                "reason": "shadow_run_candidate_rejected",
                "recovery_action": "validate_runner_set_then_alias_join",
                "alias_reconciliation_status": "EXACT_SHADOW_ARTIFACT_MATCH_FOUND",
                "latest_capture": "2026-06-10T13:50:00+10:00",
                "live_odds_source_url": "https://www.sportsbet.com.au/greyhound-racing/test/race-1",
                "live_odds_venue": "WPK",
                "live_odds_race_number": 1,
                "race_date": "2026-06-10",
                "candidate_shadow_race_id_match_count": 2,
            },
            {
                "race_id": "Race 2 - WPK - 2026-06-10",
                "reason": "no_matching_shadow_run_candidate_found",
                "recovery_action": "inspect_shadow_run_candidate_coverage",
                "alias_reconciliation_status": "NO_EXACT_SHADOW_ARTIFACT_MATCH",
            },
        ],
    }

    queue = capture.build_live_odds_backlog_recovery_queue(
        capture_report=capture_report
    )

    assert queue["schema_version"] == "live_odds_backlog_recovery_queue_v1"
    assert queue["diagnostic_only"] is True
    assert queue["join_acceptance_changed"] is False
    assert queue["db_write_performed"] is False
    assert queue["promotion_or_registry_mutation"] is False
    assert queue["no_write_guarantees"]["db_write"] is False
    assert queue["queue_count"] == 2
    assert [item["race_id"] for item in queue["items"]] == [
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-10",
    ]
    assert queue["items"][0]["authorized_action"] == (
        "diagnostic_recheck_official_result_evidence_only"
    )
    assert queue["items"][0]["db_write_performed"] is False
    assert queue["items"][0]["join_acceptance_changed"] is False
    assert queue["items"][1]["authorized_action"] == "diagnostic_review_only"
    assert queue["queues"]["retryable_exact_shadow_match"]["race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert queue["queues"]["no_exact_shadow_match"]["race_ids"] == [
        "Race 2 - WPK - 2026-06-10"
    ]
    assert queue["queues"]["by_recovery_action"][
        "validate_runner_set_then_alias_join"
    ]["race_count"] == 1
    assert queue["queues"]["retryable_exact_shadow_match"][
        "authorized_action"
    ] == "diagnostic_review_only"
    assert queue["queues"]["awaiting_official_result_evidence"][
        "authorized_action"
    ] == "diagnostic_recheck_official_result_evidence_only"
    assert queue["queues"]["awaiting_official_result_evidence"]["race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    recheck_plan = queue["queues"]["awaiting_official_result_evidence"][
        "recheck_plan"
    ]
    assert recheck_plan["diagnostic_only"] is True
    assert recheck_plan["join_acceptance_changed"] is False
    assert recheck_plan["db_write_performed"] is False
    assert recheck_plan["authorized_action"] == (
        "diagnostic_recheck_official_result_evidence_only"
    )
    assert recheck_plan["race_count"] == 1
    assert recheck_plan["recheck_ready_race_count"] == 1
    assert recheck_plan["races"][0]["race_id"] == "Race 1 - WPK - 2026-06-10"
    assert recheck_plan["races"][0]["minutes_since_latest_live_odds_capture"] == 10.0
    assert recheck_plan["races"][0]["official_result_recheck_ready"] is True


def _write_live_odds_runner_set_db(
    db_path: Path,
    *,
    race_id: str,
    captured_at: str,
    runners: list[tuple[int, str]],
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE live_odds (
                race_id TEXT,
                dog_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                timestamp TEXT,
                capture_timestamp TEXT,
                source_url TEXT,
                sportsbet_box_source TEXT,
                sportsbet_raw_runner_text TEXT
            )
            """
        )
        for box_number, dog_name in runners:
            conn.execute(
                """
                INSERT INTO live_odds (
                    race_id,
                    dog_name,
                    box_number,
                    odds_decimal,
                    timestamp,
                    capture_timestamp,
                    source_url,
                    sportsbet_box_source,
                    sportsbet_raw_runner_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    dog_name,
                    box_number,
                    2.5,
                    captured_at,
                    captured_at,
                    "https://www.sportsbet.com.au/greyhound-racing/test/race-1",
                    "runner_text",
                    f"{box_number}. {dog_name} ({box_number})",
                ),
            )


def _write_official_result_evidence_db(
    db_path: Path,
    *,
    race_id: str,
    start_datetime: str,
    runners: list[tuple[int, str, int]],
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS autonomous_official_result_evidence_races (
                id INTEGER PRIMARY KEY,
                race_id TEXT NOT NULL,
                race_date TEXT NOT NULL,
                venue TEXT,
                race_number INTEGER,
                race_time TEXT,
                start_datetime TEXT,
                source TEXT NOT NULL,
                source_url TEXT NOT NULL,
                status TEXT NOT NULL,
                winner_name TEXT,
                winner_box INTEGER,
                position_count INTEGER NOT NULL,
                participant_count INTEGER,
                box_order_json TEXT NOT NULL,
                participant_source TEXT,
                captured_at TEXT NOT NULL,
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source_artifact_dir TEXT NOT NULL,
                row_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS autonomous_official_result_evidence_runners (
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
                inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
                source_artifact_dir TEXT NOT NULL,
                row_json TEXT NOT NULL
            )
            """
        )
        identity = capture.parse_race_identity(race_id)
        source_url = (
            "https://www.thedogs.com.au/racing/test-venue/2026-06-10/1/test"
        )
        winner = next(row for row in runners if row[2] == 1)
        conn.execute(
            """
            INSERT INTO autonomous_official_result_evidence_races (
                race_id,
                race_date,
                venue,
                race_number,
                race_time,
                start_datetime,
                source,
                source_url,
                status,
                winner_name,
                winner_box,
                position_count,
                participant_count,
                box_order_json,
                participant_source,
                captured_at,
                source_artifact_dir,
                row_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                race_id,
                identity["race_date"],
                identity["venue"],
                identity["race_number"],
                "14:10",
                start_datetime,
                "thedogs_official",
                source_url,
                "resulted",
                winner[1],
                winner[0],
                len(runners),
                len(runners),
                json.dumps([row[0] for row in sorted(runners)]),
                "shadow_run_predictions",
                "2026-06-10T14:20:00+10:00",
                "artifacts/test_official_result_capture",
                "{}",
            ),
        )
        for box_number, dog_name, finish_position in runners:
            conn.execute(
                """
                INSERT INTO autonomous_official_result_evidence_runners (
                    race_id,
                    race_date,
                    venue,
                    race_number,
                    source,
                    source_url,
                    box_number,
                    dog_name,
                    finish_position,
                    is_winner,
                    captured_at,
                    source_artifact_dir,
                    row_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    identity["race_date"],
                    identity["venue"],
                    identity["race_number"],
                    "thedogs_official",
                    source_url,
                    box_number,
                    dog_name,
                    finish_position,
                    1 if finish_position == 1 else 0,
                    "2026-06-10T14:20:00+10:00",
                    "artifacts/test_official_result_capture",
                    "{}",
                ),
            )


def _runner_set_recovery_queue(
    *,
    race_id: str,
    captured_at: str,
    shadow_run_dir: Path,
) -> dict[str, object]:
    return {
        "schema_version": "live_odds_backlog_recovery_queue_v1",
        "generated_at": "2026-06-10T14:00:00+10:00",
        "source_capture_report": (
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x/"
            "autonomous_official_result_capture_report.json"
        ),
        "queues": {
            "retryable_exact_shadow_match": {
                "race_count": 1,
                "race_ids": [race_id],
                "races": [
                    {
                        "race_id": race_id,
                        "canonical_live_odds_race_id": race_id,
                        "latest_capture": captured_at,
                        "live_odds_source_url": (
                            "https://www.sportsbet.com.au/greyhound-racing/test/race-1"
                        ),
                        "candidate_shadow_race_id_matches": [
                            {
                                "race_id": race_id,
                                "shadow_run_dir": str(shadow_run_dir),
                                "artifact_sources": [
                                    "shadow_feature_rows",
                                    "shadow_predictions",
                                ],
                            }
                        ],
                    }
                ],
            }
        },
    }


def test_live_odds_backlog_runner_set_validation_exact_match_is_diagnostic_only(
    tmp_path,
):
    race_id = "Race 1 - WPK - 2026-06-10"
    captured_at = "2026-06-10T14:00:00+10:00"
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv, race_id=race_id)
    db_path = tmp_path / "odds.sqlite"
    _write_live_odds_runner_set_db(
        db_path,
        race_id=race_id,
        captured_at=captured_at,
        runners=[
            (1, "Alpha"),
            (2, "Bravo"),
            (3, "Charlie"),
            (4, "Delta"),
        ],
    )

    report = capture.build_live_odds_backlog_runner_set_validation(
        recovery_queue=_runner_set_recovery_queue(
            race_id=race_id,
            captured_at=captured_at,
            shadow_run_dir=shadow_run_dir,
        ),
        db_path=db_path,
    )

    assert report["schema_version"] == "live_odds_backlog_runner_set_validation_v1"
    assert report["diagnostic_only"] is True
    assert report["join_authorized"] is False
    assert report["db_write_performed"] is False
    assert report["retryable_race_count"] == 1
    assert report["exact_runner_set_match_race_count"] == 1
    assert report["blocked_race_count"] == 0
    validation = report["validations"][0]
    assert validation["validation_status"] == "RUNNER_SET_EXACT_MATCH_DIAGNOSTIC_ONLY"
    assert validation["join_authorized"] is False
    assert validation["match_validations"][0]["exact_runner_set_match"] is True
    assert validation["match_validations"][0]["missing_from_shadow"] == []
    assert validation["match_validations"][0]["missing_from_live_odds"] == []


def test_live_odds_backlog_runner_set_validation_blocks_mismatch(tmp_path):
    race_id = "Race 1 - WPK - 2026-06-10"
    captured_at = "2026-06-10T14:00:00+10:00"
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv, race_id=race_id)
    db_path = tmp_path / "odds.sqlite"
    _write_live_odds_runner_set_db(
        db_path,
        race_id=race_id,
        captured_at=captured_at,
        runners=[
            (1, "Alpha"),
            (2, "Bravo"),
            (3, "Charlie"),
            (4, "Echo"),
        ],
    )

    report = capture.build_live_odds_backlog_runner_set_validation(
        recovery_queue=_runner_set_recovery_queue(
            race_id=race_id,
            captured_at=captured_at,
            shadow_run_dir=shadow_run_dir,
        ),
        db_path=db_path,
    )

    assert report["exact_runner_set_match_race_count"] == 0
    assert report["blocked_race_count"] == 1
    validation = report["validations"][0]
    assert validation["validation_status"] == "RUNNER_SET_VALIDATION_BLOCKED"
    assert validation["join_authorized"] is False
    match = validation["match_validations"][0]
    assert match["exact_runner_set_match"] is False
    assert match["missing_from_shadow"] == [[4, "ECHO"]]
    assert match["missing_from_live_odds"] == [[4, "DELTA"]]


def test_live_odds_backlog_join_eligibility_all_gates_report_only(tmp_path):
    race_id = "Race 1 - WPK - 2026-06-10"
    captured_at = "2026-06-10T14:00:00+10:00"
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv, race_id=race_id)
    db_path = tmp_path / "odds.sqlite"
    runners = [
        (1, "Alpha"),
        (2, "Bravo"),
        (3, "Charlie"),
        (4, "Delta"),
    ]
    _write_live_odds_runner_set_db(
        db_path,
        race_id=race_id,
        captured_at=captured_at,
        runners=runners,
    )
    _write_official_result_evidence_db(
        db_path,
        race_id=race_id,
        start_datetime="2026-06-10T14:10:00+10:00",
        runners=[
            (1, "Alpha", 1),
            (2, "Bravo", 2),
            (3, "Charlie", 3),
            (4, "Delta", 4),
        ],
    )
    runner_set = capture.build_live_odds_backlog_runner_set_validation(
        recovery_queue=_runner_set_recovery_queue(
            race_id=race_id,
            captured_at=captured_at,
            shadow_run_dir=shadow_run_dir,
        ),
        db_path=db_path,
    )
    runner_set["generated_at"] = "2026-06-10T14:10:00+10:00"

    packet = capture.build_live_odds_backlog_join_eligibility_packet(
        runner_set_validation=runner_set,
        db_path=db_path,
    )

    assert packet["schema_version"] == "live_odds_backlog_join_eligibility_packet_v1"
    assert packet["diagnostic_only"] is True
    assert packet["join_authorized"] is False
    assert packet["db_write_performed"] is False
    assert packet["evaluated_race_count"] == 1
    assert packet["eligible_report_only_race_count"] == 1
    assert packet["blocked_race_count"] == 0
    row = packet["races"][0]
    assert row["eligibility_status"] == "JOIN_ELIGIBLE_REPORT_ONLY"
    assert row["blockers"] == []
    assert row["join_authorized"] is False
    assert row["db_write_performed"] is False
    assert row["gates"]["prejump_timing_verified"] is True
    assert row["gates"]["official_result_runner_set_exact_live_odds_match"] is True


def test_live_odds_backlog_join_eligibility_blocks_missing_official_result(tmp_path):
    race_id = "Race 1 - WPK - 2026-06-10"
    captured_at = "2026-06-10T14:00:00+10:00"
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv, race_id=race_id)
    db_path = tmp_path / "odds.sqlite"
    _write_live_odds_runner_set_db(
        db_path,
        race_id=race_id,
        captured_at=captured_at,
        runners=[
            (1, "Alpha"),
            (2, "Bravo"),
            (3, "Charlie"),
            (4, "Delta"),
        ],
    )
    runner_set = capture.build_live_odds_backlog_runner_set_validation(
        recovery_queue=_runner_set_recovery_queue(
            race_id=race_id,
            captured_at=captured_at,
            shadow_run_dir=shadow_run_dir,
        ),
        db_path=db_path,
    )
    runner_set["generated_at"] = "2026-06-10T14:10:00+10:00"

    packet = capture.build_live_odds_backlog_join_eligibility_packet(
        runner_set_validation=runner_set,
        db_path=db_path,
    )

    assert packet["eligible_report_only_race_count"] == 0
    assert packet["blocked_race_count"] == 1
    assert packet["awaiting_official_result_evidence_race_count"] == 1
    assert packet["awaiting_official_result_evidence_race_ids"] == [race_id]
    row = packet["races"][0]
    assert row["eligibility_status"] == "JOIN_ELIGIBILITY_BLOCKED"
    assert row["blocker_category"] == "awaiting_official_result_evidence"
    assert row["awaiting_official_result_evidence"] is True
    assert (
        row["next_authorized_action"]
        == "diagnostic_recheck_official_result_evidence_only"
    )
    assert "official_result_race_row_present" in row["blockers"]
    assert "official_result_runner_rows_present" in row["blockers"]
    assert row["join_authorized"] is False
    assert row["db_write_performed"] is False
    recheck_plan = packet["awaiting_official_result_recheck_plan"]
    assert (
        recheck_plan["schema_version"]
        == "join_eligibility_awaiting_official_result_recheck_plan_v1"
    )
    assert recheck_plan["diagnostic_only"] is True
    assert recheck_plan["join_acceptance_changed"] is False
    assert recheck_plan["join_authorized"] is False
    assert recheck_plan["db_write_performed"] is False
    assert recheck_plan["authorized_action"] == (
        "diagnostic_recheck_official_result_evidence_only"
    )
    assert recheck_plan["minimum_minutes_since_latest_live_odds_capture_for_recheck"] == 5.0
    assert recheck_plan["race_count"] == 1
    assert recheck_plan["race_ids"] == [race_id]
    assert recheck_plan["races"][0]["race_id"] == race_id
    assert recheck_plan["races"][0]["latest_live_odds_capture"] == captured_at
    assert recheck_plan["races"][0]["official_result_recheck_ready"] is True
    assert recheck_plan["races"][0]["join_authorized"] is False
    assert recheck_plan["races"][0]["db_write_performed"] is False


def test_join_eligibility_report_fields_surface_recheck_plan():
    report = {}
    recheck_plan = {
        "schema_version": "join_eligibility_awaiting_official_result_recheck_plan_v1",
        "diagnostic_only": True,
        "join_acceptance_changed": False,
        "join_authorized": False,
        "db_write_performed": False,
        "recheck_ready_race_count": 2,
        "race_count": 7,
        "race_ids": [
            "Race 2 - GRDN - 2026-06-13",
            "Race 3 - WPK - 2026-06-13",
        ],
    }

    capture.add_live_odds_backlog_join_eligibility_report_fields(
        report,
        {
            "evaluated_race_count": 7,
            "eligible_report_only_race_count": 0,
            "blocked_race_count": 7,
            "blocker_counts": {
                "official_result_runner_set_exact_live_odds_match": 4,
                "prejump_timing_verified": 3,
            },
            "diagnostic_only": True,
            "join_authorized": False,
            "db_write_performed": False,
            "awaiting_official_result_recheck_plan": recheck_plan,
        },
    )

    assert report["live_odds_backlog_join_eligibility_evaluated_race_count"] == 7
    assert report["live_odds_backlog_join_eligibility_eligible_report_only_race_count"] == 0
    assert report["live_odds_backlog_join_eligibility_blocked_race_count"] == 7
    assert report["live_odds_backlog_join_eligibility_blocker_counts"] == {
        "official_result_runner_set_exact_live_odds_match": 4,
        "prejump_timing_verified": 3,
    }
    assert report["live_odds_backlog_join_eligibility_diagnostic_only"] is True
    assert report["live_odds_backlog_join_eligibility_join_authorized"] is False
    assert report["live_odds_backlog_join_eligibility_db_write_performed"] is False
    assert (
        report[
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
        ]
        == 2
    )
    assert (
        report[
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_plan"
        ]
        == recheck_plan
    )


def test_shadow_run_candidates_require_source_csv_match_and_jumped(tmp_path):
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv)

    candidates, skipped, source_report = capture.shadow_run_candidates(
        shadow_run_dir=shadow_run_dir,
        target_date="2026-06-10",
        current_time=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        race_ids=[],
        output_dir=tmp_path / "out",
    )

    assert skipped == []
    assert source_report["candidate_count"] == 1
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.race_id == "Race 1 - WPK - 2026-06-10"
    assert candidate.participant_source == "shadow_run_predictions"
    assert candidate.csv_path == source_csv
    assert candidate.runner_completeness["status"] == "COMPLETE"
    assert candidate.canonical_thedogs_url.endswith("/test-race?trial=false")


def test_shadow_run_candidates_use_score_live_feature_rows_from_manifest(tmp_path):
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv)
    root_feature_rows = shadow_run_dir / "shadow_feature_rows.json"
    score_dir = shadow_run_dir / "shadow_score_live"
    score_dir.mkdir()
    score_feature_rows = score_dir / "shadow_feature_rows.json"
    score_feature_rows.write_text(root_feature_rows.read_text(encoding="utf-8"), encoding="utf-8")
    root_feature_rows.unlink()
    (shadow_run_dir / "shadow_manifest.json").write_text(
        json.dumps(
            {
                "score_live_manifest": {
                    "feature_rows": str(score_feature_rows),
                }
            }
        ),
        encoding="utf-8",
    )

    candidates, skipped, source_report = capture.shadow_run_candidates(
        shadow_run_dir=shadow_run_dir,
        target_date="2026-06-10",
        current_time=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        race_ids=[],
        output_dir=tmp_path / "out",
    )

    assert skipped == []
    assert source_report["candidate_count"] == 1
    assert len(candidates) == 1
    assert candidates[0].csv_path == source_csv


def test_shadow_run_candidates_resolve_runtime_repo_relative_source_csv(tmp_path):
    runtime_root = tmp_path / "runtime_checkout"
    artifact_root = runtime_root / "artifacts/full_evidence_orchestration_20260525"
    artifact_root.mkdir(parents=True)
    run_name = "daily_race_ingest_shadow_20260610T140000+1000_daemon_autopilot"
    source_rel = (
        Path("artifacts/full_evidence_orchestration_20260525")
        / run_name
        / "eligible_inputs/source_0001/Race 1 - WPK - 2026-06-10.csv"
    )
    source_csv = runtime_root / source_rel
    shadow_run_dir = _write_shadow_run(
        artifact_root,
        source_csv=source_rel,
        dirname=run_name,
    )
    source_csv.parent.mkdir(parents=True)
    _write_shadow_source_csv(source_csv)

    candidates, skipped, source_report = capture.shadow_run_candidates(
        shadow_run_dir=shadow_run_dir,
        target_date="2026-06-10",
        current_time=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        race_ids=[],
        output_dir=tmp_path / "out",
    )

    assert skipped == []
    assert source_report["candidate_count"] == 1
    assert len(candidates) == 1
    assert candidates[0].csv_path == source_csv.resolve()


def test_shadow_run_candidates_skip_missing_source_csv(tmp_path):
    shadow_run_dir = _write_shadow_run(
        tmp_path,
        source_csv=tmp_path / "missing.csv",
    )

    candidates, skipped, source_report = capture.shadow_run_candidates(
        shadow_run_dir=shadow_run_dir,
        target_date="2026-06-10",
        current_time=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        race_ids=[],
        output_dir=tmp_path / "out",
    )

    assert candidates == []
    assert source_report["candidate_count"] == 0
    assert skipped[0]["reason"] == "shadow_run_source_csv_missing"


def test_shadow_run_official_dry_run_uses_official_fetcher_without_db_writes(
    tmp_path,
    monkeypatch,
):
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass

    class FakeTheDogsResultFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, candidate):
            return capture.ingest.SourceResult(
                source="thedogs_official",
                status="resulted",
                source_url=f"{candidate.canonical_thedogs_url}/results",
                positions_by_box={1: 1, 2: 2, 3: 3, 4: 4},
                raw_order=[1, 2, 3, 4],
            )

    monkeypatch.setattr(
        capture.ingest,
        "TheDogsResultFetcher",
        FakeTheDogsResultFetcher,
    )
    monkeypatch.setattr(
        capture.ingest,
        "optional_browser_driver",
        lambda headless=True: (None, None, "browser_unavailable"),
    )

    report, returncode = capture.run_shadow_run_official_dry_run(
        db_path=db_path,
        shadow_run_dir=shadow_run_dir,
        target_date="2026-06-10",
        current_time=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        output_dir=tmp_path / "out",
        race_ids=[],
    )

    assert returncode == 0
    assert report["dry_run"] is True
    assert report["scope"]["candidate_source"] == "shadow_run_predictions"
    assert report["candidate_count"] == 1
    assert report["ingested_count"] == 1
    assert report["ingested"][0]["source"] == "thedogs_official"
    assert report["ingested"][0]["participant_source"] == "shadow_run_predictions"
    assert report["clean_for_label_write"] is False


def test_shadow_run_official_dry_run_failed_validation_keeps_participant_context(
    tmp_path,
    monkeypatch,
):
    source_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(tmp_path, source_csv=source_csv)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass

    class FakeTheDogsResultFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, candidate):
            return capture.ingest.SourceResult(
                source="thedogs_official",
                status="resulted",
                source_url=f"{candidate.canonical_thedogs_url}/results",
                positions_by_box={1: 1, 2: 2, 9: 3, 4: 4},
                raw_order=[1, 2, 9, 4],
                dog_names_by_box={
                    1: "Alpha",
                    2: "Bravo",
                    4: "Delta",
                    9: "Reserve Runner",
                },
            )

    monkeypatch.setattr(
        capture.ingest,
        "TheDogsResultFetcher",
        FakeTheDogsResultFetcher,
    )
    monkeypatch.setattr(
        capture.ingest,
        "optional_browser_driver",
        lambda headless=True: (None, None, "browser_unavailable"),
    )

    report, returncode = capture.run_shadow_run_official_dry_run(
        db_path=db_path,
        shadow_run_dir=shadow_run_dir,
        target_date="2026-06-10",
        current_time=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        output_dir=tmp_path / "out",
        race_ids=[],
    )

    assert returncode == 0
    assert report["failed_count"] == 1
    failed = report["failed"][0]
    assert failed["errors"] == ["result_boxes_not_in_participants:9"]
    assert failed["participant_source"] == "shadow_run_predictions"
    assert failed["participant_count"] == 4
    assert failed["participant_boxes"] == [1, 2, 3, 4]
    assert failed["participants"] == [
        {"box_number": 1, "dog_name": "Alpha"},
        {"box_number": 2, "dog_name": "Bravo"},
        {"box_number": 3, "dog_name": "Charlie"},
        {"box_number": 4, "dog_name": "Delta"},
    ]
    assert failed["attempted_sources"][0]["dog_names_by_box"] == {
        "1": "Alpha",
        "2": "Bravo",
        "4": "Delta",
        "9": "Reserve Runner",
    }
    assert failed["attempted_sources"][0]["positions"] == [
        {"box_number": 1, "finish_position": 1, "dog_name": "Alpha"},
        {"box_number": 2, "finish_position": 2, "dog_name": "Bravo"},
        {"box_number": 9, "finish_position": 3, "dog_name": "Reserve Runner"},
        {"box_number": 4, "finish_position": 4, "dog_name": "Delta"},
    ]


def test_shadow_run_official_dry_run_retries_source_backed_live_odds_backlog(
    tmp_path,
    monkeypatch,
):
    current_csv = tmp_path / "Race 2 - WPK - 2026-06-11.csv"
    backlog_csv = tmp_path / "Race 1 - WPK - 2026-06-10.csv"
    _write_shadow_source_csv(current_csv)
    _write_shadow_source_csv(backlog_csv)
    current_shadow_run = _write_shadow_run(
        tmp_path,
        source_csv=current_csv,
        race_id="Race 2 - WPK - 2026-06-11",
        race_time_minutes=16 * 60,
        dirname="shadow_current",
    )
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    evidence_root.mkdir(parents=True)
    _write_shadow_run(
        evidence_root,
        source_csv=backlog_csv,
        race_id="Race 1 - WPK - 2026-06-10",
        race_time_minutes=14 * 60,
        dirname="daily_race_ingest_shadow_20260610T140000_backlog",
    )
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                winner_source TEXT
            );
            CREATE TABLE live_odds (
                race_id TEXT,
                race_date TEXT,
                source_url TEXT,
                capture_timestamp TEXT,
                timestamp TEXT,
                market_type TEXT,
                odds_decimal REAL,
                odds_level TEXT,
                sportsbet_box_source TEXT
            );
            """
        )
        conn.execute(
            """
            INSERT INTO live_odds
                (race_id, race_date, source_url, capture_timestamp, timestamp,
                 market_type, odds_decimal, odds_level, sportsbet_box_source)
            VALUES (?, '2026-06-10',
                    'https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-1',
                    '2026-06-10T13:40:00+10:00',
                    '2026-06-10T13:40:00+10:00',
                    'win', 2.8, 'dog', 'runner_text')
            """,
            ("Race 1 - WPK - 2026-06-10",),
        )

    class FakeTheDogsResultFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, candidate):
            return capture.ingest.SourceResult(
                source="thedogs_official",
                status="resulted",
                source_url=f"{candidate.canonical_thedogs_url}/results",
                positions_by_box={1: 1, 2: 2, 3: 3, 4: 4},
                raw_order=[1, 2, 3, 4],
            )

    monkeypatch.setattr(
        capture.ingest,
        "TheDogsResultFetcher",
        FakeTheDogsResultFetcher,
    )
    monkeypatch.setattr(
        capture.ingest,
        "optional_browser_driver",
        lambda headless=True: (None, None, "browser_unavailable"),
    )

    report, returncode = capture.run_shadow_run_official_dry_run(
        db_path=db_path,
        shadow_run_dir=current_shadow_run,
        target_date="2026-06-11",
        current_time=datetime.fromisoformat("2026-06-11T17:00:00+10:00"),
        output_dir=tmp_path / "out",
        race_ids=[],
        include_live_odds_backlog=True,
        backlog_evidence_root=evidence_root,
        backlog_limit=10,
        backlog_shadow_run_limit=10,
        backlog_lookback_days=1,
    )

    assert returncode == 0
    assert set(report["candidate_race_ids"]) == {
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-11",
    }
    assert report["live_odds_backlog"]["target_dates"] == [
        "2026-06-11",
        "2026-06-10",
    ]
    assert report["live_odds_backlog"]["discovered_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["live_odds_backlog"]["candidate_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert report["live_odds_backlog"]["unresolved_race_ids"] == []
    assert report["ingested_count"] == 2
    assert {row["race_id"] for row in report["ingested"]} == {
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-11",
    }


def test_shadow_run_official_dry_run_retains_shadow_backlog_candidates_beyond_limit(
    tmp_path,
    monkeypatch,
):
    current_csv = tmp_path / "Race 99 - WPK - 2026-06-29.csv"
    retained_csv = tmp_path / "Race 10 - WAR - 2026-06-29.csv"
    retained_race_id = "Race 10 - WAR - 2026-06-29"
    _write_shadow_source_csv(current_csv)
    _write_shadow_source_csv(retained_csv)
    current_shadow_run = _write_shadow_run(
        tmp_path,
        source_csv=current_csv,
        race_id="Race 99 - WPK - 2026-06-29",
        race_time_minutes=20 * 60,
        dirname="shadow_current",
    )
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    evidence_root.mkdir(parents=True)
    _write_shadow_run(
        evidence_root,
        source_csv=retained_csv,
        race_id=retained_race_id,
        race_time_minutes=17 * 60,
        dirname="daily_race_ingest_shadow_20260629T170200_retained",
    )
    newer_unrelated_csv = tmp_path / "Race 1 - WPK - 2026-06-30.csv"
    _write_shadow_source_csv(newer_unrelated_csv)
    _write_shadow_run(
        evidence_root,
        source_csv=newer_unrelated_csv,
        race_id="Race 1 - WPK - 2026-06-30",
        race_time_minutes=12 * 60,
        dirname="daily_race_ingest_shadow_20260630T120000_newer",
    )
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                winner_source TEXT
            );
            CREATE TABLE live_odds (
                race_id TEXT,
                race_date TEXT,
                source_url TEXT,
                capture_timestamp TEXT,
                timestamp TEXT,
                market_type TEXT,
                odds_decimal REAL,
                odds_level TEXT,
                sportsbet_box_source TEXT
            );
            """
        )
        rows = [
            (
                f"Race {index} - FILLER{index:02d} - 2026-06-29",
                "2026-06-29",
                f"2026-06-29T17:{index:02d}:00+10:00",
                f"2026-06-29T17:{index:02d}:00+10:00",
            )
            for index in range(1, 34)
        ]
        rows.append(
            (
                retained_race_id,
                "2026-06-29",
                "2026-06-29T17:34:00+10:00",
                "2026-06-29T17:34:00+10:00",
            )
        )
        conn.executemany(
            """
            INSERT INTO live_odds
                (race_id, race_date, source_url, capture_timestamp, timestamp,
                 market_type, odds_decimal, odds_level, sportsbet_box_source)
            VALUES (?, ?, 'https://www.sportsbet.com.au/greyhound-racing/test/race',
                    ?, ?, 'win', 2.8, 'dog', 'runner_text')
            """,
            rows,
        )

    class FakeTheDogsResultFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, candidate):
            return capture.ingest.SourceResult(
                source="thedogs_official",
                status="resulted",
                source_url=f"{candidate.canonical_thedogs_url}/results",
                positions_by_box={1: 1, 2: 2, 3: 3, 4: 4},
                raw_order=[1, 2, 3, 4],
            )

    monkeypatch.setattr(
        capture.ingest,
        "TheDogsResultFetcher",
        FakeTheDogsResultFetcher,
    )
    monkeypatch.setattr(
        capture.ingest,
        "optional_browser_driver",
        lambda headless=True: (None, None, "browser_unavailable"),
    )

    report, returncode = capture.run_shadow_run_official_dry_run(
        db_path=db_path,
        shadow_run_dir=current_shadow_run,
        target_date="2026-06-29",
        current_time=datetime.fromisoformat("2026-06-29T21:00:00+10:00"),
        output_dir=tmp_path / "out",
        race_ids=[],
        include_live_odds_backlog=True,
        backlog_evidence_root=evidence_root,
        backlog_limit=32,
        backlog_shadow_run_limit=1,
        backlog_lookback_days=0,
    )

    assert returncode == 0
    assert retained_race_id in report["candidate_race_ids"]
    retained_entry = next(
        row
        for row in report["live_odds_backlog"]["discovered_races"]
        if row["race_id"] == retained_race_id
    )
    assert retained_entry["backlog_rank"] == 34
    assert retained_entry["retained_beyond_limit"] is True
    assert report["live_odds_backlog"]["retained_beyond_limit_race_ids"] == [
        retained_race_id
    ]
    assert retained_race_id in {row["race_id"] for row in report["ingested"]}


def test_shadow_run_official_dry_run_reports_unresolved_live_odds_backlog_reasons(
    tmp_path,
    monkeypatch,
):
    current_csv = tmp_path / "Race 2 - WPK - 2026-06-11.csv"
    _write_shadow_source_csv(current_csv)
    current_shadow_run = _write_shadow_run(
        tmp_path,
        source_csv=current_csv,
        race_id="Race 2 - WPK - 2026-06-11",
        race_time_minutes=16 * 60,
        dirname="shadow_current",
    )
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    evidence_root.mkdir(parents=True)
    _write_shadow_run(
        evidence_root,
        source_csv=tmp_path / "missing_backlog_source.csv",
        race_id="Race 3 - WPK - 2026-06-10",
        race_time_minutes=14 * 60,
        dirname="daily_race_ingest_shadow_20260610T140000_backlog",
    )
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                winner_source TEXT
            );
            CREATE TABLE live_odds (
                race_id TEXT,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                source_url TEXT,
                capture_timestamp TEXT,
                timestamp TEXT,
                market_type TEXT,
                odds_decimal REAL,
                odds_level TEXT,
                sportsbet_box_source TEXT,
                box_number INTEGER
            );
            """
        )
        for race_id, venue, race_number, box_number, source_url in [
            (
                "Race 3 - WPK - 2026-06-10",
                "Wentworth Park",
                3,
                1,
                "https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-3",
            ),
            (
                "ASCOT PARK_2026-06-10_6",
                "Ascot Park",
                6,
                3,
                "https://www.sportsbet.com.au/greyhound-racing/australia-nz/ascot-park/race-6",
            ),
        ]:
            conn.execute(
                """
                INSERT INTO live_odds
                    (race_id, venue, race_number, race_date, source_url,
                     capture_timestamp, timestamp, market_type, odds_decimal,
                     odds_level, sportsbet_box_source, box_number)
                VALUES (?, ?, ?, '2026-06-10',
                        ?,
                        '2026-06-10T13:40:00+10:00',
                        '2026-06-10T13:40:00+10:00',
                        'win', 2.8, 'dog', 'runner_text', ?)
                """,
                (race_id, venue, race_number, source_url, box_number),
            )
        conn.execute(
            """
            INSERT INTO live_odds
                (race_id, venue, race_number, race_date, source_url,
                 capture_timestamp, timestamp, market_type, odds_decimal,
                 odds_level, sportsbet_box_source, box_number)
            VALUES ('ASCOT PARK_2026-06-10_6', 'Ascot Park', 6, '2026-06-10',
                    'https://www.sportsbet.com.au/greyhound-racing/australia-nz/ascot-park/race-6',
                    '2026-06-10T13:41:00+10:00',
                    '2026-06-10T13:41:00+10:00',
                    'win', 3.2, 'dog', 'runner_text', NULL)
            """
        )

    class FakeTheDogsResultFetcher:
        def __init__(self, *args, **kwargs):
            pass

        def fetch(self, candidate):
            return capture.ingest.SourceResult(
                source="thedogs_official",
                status="resulted",
                source_url=f"{candidate.canonical_thedogs_url}/results",
                positions_by_box={1: 1, 2: 2, 3: 3, 4: 4},
                raw_order=[1, 2, 3, 4],
            )

    monkeypatch.setattr(capture.ingest, "TheDogsResultFetcher", FakeTheDogsResultFetcher)
    monkeypatch.setattr(
        capture.ingest,
        "optional_browser_driver",
        lambda headless=True: (None, None, "browser_unavailable"),
    )

    report, returncode = capture.run_shadow_run_official_dry_run(
        db_path=db_path,
        shadow_run_dir=current_shadow_run,
        target_date="2026-06-11",
        current_time=datetime.fromisoformat("2026-06-11T17:00:00+10:00"),
        output_dir=tmp_path / "out",
        race_ids=[],
        include_live_odds_backlog=True,
        backlog_evidence_root=evidence_root,
        backlog_limit=10,
        backlog_shadow_run_limit=10,
        backlog_lookback_days=1,
    )

    assert returncode == 0
    assert sorted(report["live_odds_backlog"]["unresolved_race_ids"]) == [
        "ASCOT PARK_2026-06-10_6",
        "Race 3 - WPK - 2026-06-10",
    ]
    diagnostics = {
        item["race_id"]: item for item in report["live_odds_backlog"]["unresolved_races"]
    }
    assert diagnostics["ASCOT PARK_2026-06-10_6"]["reason"] == (
        "live_odds_race_id_not_canonical_shadow_race_id"
    )
    assert diagnostics["ASCOT PARK_2026-06-10_6"]["parsed_identity"] == {
        "race_number": None,
        "venue": None,
        "race_date": None,
    }
    assert diagnostics["ASCOT PARK_2026-06-10_6"]["canonical_live_odds_race_id"] == (
        "ASCOT PARK_2026-06-10_6"
    )
    assert "Race 6 - ASCOT PARK - 2026-06-10" in (
        diagnostics["ASCOT PARK_2026-06-10_6"]["candidate_shadow_race_ids"]
    )
    assert diagnostics["ASCOT PARK_2026-06-10_6"]["live_odds_box_count"] == 1
    assert diagnostics["ASCOT PARK_2026-06-10_6"]["live_odds_box_sources"] == [
        "runner_text"
    ]
    assert diagnostics["ASCOT PARK_2026-06-10_6"][
        "alias_reconciliation_status"
    ] == "NO_EXACT_SHADOW_ARTIFACT_MATCH"
    assert diagnostics["ASCOT PARK_2026-06-10_6"][
        "candidate_shadow_race_id_match_count"
    ] == 0
    assert diagnostics["ASCOT PARK_2026-06-10_6"]["recovery_action"] == (
        "recover_shadow_predictions_for_source_identity"
    )
    assert diagnostics["Race 3 - WPK - 2026-06-10"]["reason"] == (
        "shadow_run_candidate_rejected"
    )
    assert diagnostics["Race 3 - WPK - 2026-06-10"]["shadow_run_skip_reasons"] == [
        "shadow_run_source_csv_missing"
    ]
    assert report["live_odds_backlog"]["unresolved_reason_counts"] == {
        "live_odds_race_id_not_canonical_shadow_race_id": 1,
        "shadow_run_candidate_rejected": 1,
    }
    assert report["live_odds_backlog"]["unresolved_recovery_action_counts"] == {
        "recover_shadow_predictions_for_source_identity": 1,
        "validate_runner_set_then_alias_join": 1,
    }
    assert report["live_odds_backlog"]["unresolved_alias_status_counts"] == {
        "EXACT_SHADOW_ARTIFACT_MATCH_FOUND": 1,
        "NO_EXACT_SHADOW_ARTIFACT_MATCH": 1,
    }
    assert report["live_odds_backlog"]["retryable_exact_shadow_match_race_count"] == 1
    assert report["live_odds_backlog"]["no_exact_shadow_match_race_count"] == 1
    assert report["live_odds_backlog"]["retryable_exact_shadow_match_race_ids"] == [
        "Race 3 - WPK - 2026-06-10"
    ]
    assert report["live_odds_backlog"]["no_exact_shadow_match_race_ids"] == [
        "ASCOT PARK_2026-06-10_6"
    ]


def test_unresolved_live_odds_backlog_diagnostics_suggest_shadow_aliases(tmp_path):
    source_csv = tmp_path / "Race 1 - SAL - 2026-06-10.csv"
    _write_shadow_source_csv(source_csv)
    shadow_run_dir = _write_shadow_run(
        tmp_path,
        source_csv=source_csv,
        race_id="Race 1 - SAL - 2026-06-10",
        race_time_minutes=14 * 60,
        dirname="shadow_sal_alias",
    )

    diagnostics = capture.unresolved_live_odds_backlog_diagnostics(
        unresolved_race_ids=["SAL_2026-06-10_1"],
        backlog_entries=[
            {
                "race_id": "SAL_2026-06-10_1",
                "race_date": "2026-06-10",
                "venue": "Sale",
                "race_number": 1,
                "source_url": (
                    "https://www.sportsbet.com.au/greyhound-racing/"
                    "australia-nz/sale/race-1-10572923"
                ),
                "latest_capture": "2026-06-10T11:32:42+10:00",
                "odds_row_count": 7,
                "box_count": 7,
                "sportsbet_box_sources": ["runner_text", "list_position_fallback"],
            }
        ],
        skipped=[],
        shadow_run_report_count=12,
        shadow_run_dirs=[shadow_run_dir],
    )

    row = diagnostics[0]
    assert row["reason"] == "live_odds_race_id_not_canonical_shadow_race_id"
    assert row["canonical_live_odds_race_id"] == "SAL_2026-06-10_1"
    assert "Race 1 - SAL - 2026-06-10" in row["candidate_shadow_race_ids"]
    assert "Race 1 - SALE - 2026-06-10" in row["candidate_shadow_race_ids"]
    assert row["alias_reconciliation_status"] == "EXACT_SHADOW_ARTIFACT_MATCH_FOUND"
    assert row["candidate_shadow_race_id_match_count"] == 1
    assert row["candidate_shadow_race_id_matches"] == [
        {
            "race_id": "Race 1 - SAL - 2026-06-10",
            "shadow_run_dir": str(shadow_run_dir),
            "artifact_sources": ["shadow_feature_rows", "shadow_predictions"],
        }
    ]
    assert row["live_odds_row_count"] == 7
    assert row["live_odds_box_count"] == 7
    assert row["recovery_action"] == "validate_runner_set_then_alias_join"


def test_source_backed_live_odds_backlog_entries_respect_lookback(tmp_path):
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                winner_source TEXT
            );
            CREATE TABLE live_odds (
                race_id TEXT,
                race_date TEXT,
                source_url TEXT,
                capture_timestamp TEXT,
                timestamp TEXT,
                market_type TEXT,
                odds_decimal REAL,
                odds_level TEXT,
                sportsbet_box_source TEXT
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO live_odds
                (race_id, race_date, source_url, capture_timestamp, timestamp,
                 market_type, odds_decimal, odds_level, sportsbet_box_source)
            VALUES (?, ?, 'https://sportsbet.example/race', ?, ?, 'win', 2.4, 'dog', 'runner_text')
            """,
            [
                (
                    "Race 1 - WPK - 2026-06-12",
                    "2026-06-12",
                    "2026-06-12T10:00:00+10:00",
                    "2026-06-12T10:00:00+10:00",
                ),
                (
                    "Race 1 - WPK - 2026-06-11",
                    "2026-06-11",
                    "2026-06-11T10:00:00+10:00",
                    "2026-06-11T10:00:00+10:00",
                ),
                (
                    "Race 1 - WPK - 2026-06-10",
                    "2026-06-10",
                    "2026-06-10T10:00:00+10:00",
                    "2026-06-10T10:00:00+10:00",
                ),
            ],
        )

    entries = capture.source_backed_live_odds_backlog_entries(
        db_path=db_path,
        target_date="2026-06-12",
        limit=10,
        lookback_days=1,
    )

    assert [entry["race_id"] for entry in entries] == [
        "Race 1 - WPK - 2026-06-12",
        "Race 1 - WPK - 2026-06-11",
    ]
    assert [entry["race_date"] for entry in entries] == [
        "2026-06-12",
        "2026-06-11",
    ]


def _official_artifact_rows():
    generated_at = datetime.fromisoformat("2026-06-10T15:00:00+10:00")
    return capture.build_artifact_rows(
        {
            "scope": {
                "date": "2026-06-10",
                "db_path": "/tmp/labels.sqlite",
            },
            "ingested": [
                {
                    "race_id": "Race 1 - WPK - 2026-06-10",
                    "venue": "WPK",
                    "race_number": 1,
                    "race_date": "2026-06-10",
                    "race_time": "14:00",
                    "start_datetime": "2026-06-10T14:00:00+10:00",
                    "source": "thedogs_official",
                    "source_url": (
                        "https://www.thedogs.com.au/racing/"
                        "wentworth-park/2026-06-10/1/test-race?trial=false"
                    ),
                    "status": "resulted",
                    "winner_name": "Alpha",
                    "winner_box": 1,
                    "box_order": [1, 2],
                    "participant_source": "shadow_run_predictions",
                    "positions": [
                        {
                            "box_number": 1,
                            "finish_position": 1,
                            "dog_name": "Alpha",
                        },
                        {
                            "box_number": 2,
                            "finish_position": 2,
                            "dog_name": "Bravo",
                        },
                    ],
                    "participants": [
                        {"box_number": 1, "dog_name": "Alpha"},
                        {"box_number": 2, "dog_name": "Bravo"},
                    ],
                }
            ],
            "failed": [],
            "skipped": [],
        },
        generated_at=generated_at,
    )


def _mixed_official_artifact_rows():
    artifact_rows = _official_artifact_rows()
    blocked_race = deepcopy(artifact_rows["race_rows"][0])
    blocked_race["race_id"] = "Race 2 - WPK - 2026-06-10"
    blocked_race["race_number"] = 2
    blocked_race["source_url"] = (
        "https://www.thedogs.com.au/racing/"
        "wentworth-park/2026-06-10/2/test-race?trial=false"
    )
    blocked_race["winner_name"] = "Charlie"
    blocked_race["winner_box"] = 1
    blocked_runners = []
    for runner in artifact_rows["runner_rows"]:
        blocked_runner = deepcopy(runner)
        blocked_runner["race_id"] = blocked_race["race_id"]
        blocked_runner["race_number"] = 2
        blocked_runner["source_url"] = blocked_race["source_url"]
        blocked_runner["finish_position"] = 1
        blocked_runners.append(blocked_runner)
    artifact_rows["race_rows"].append(blocked_race)
    artifact_rows["runner_rows"].extend(blocked_runners)
    return artifact_rows


def _write_official_artifact_jsonl(tmp_path, artifact_rows):
    race_rows_path = tmp_path / "official_result_races.jsonl"
    runner_rows_path = tmp_path / "official_result_runners.jsonl"
    quarantine_path = tmp_path / "official_result_quarantine.jsonl"
    race_rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in artifact_rows["race_rows"]),
        encoding="utf-8",
    )
    runner_rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in artifact_rows["runner_rows"]),
        encoding="utf-8",
    )
    quarantine_path.write_text("", encoding="utf-8")
    return race_rows_path, runner_rows_path, quarantine_path


def test_official_result_evidence_db_ingest_is_ready_not_executed(tmp_path):
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass

    status = capture.append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=_official_artifact_rows(),
        output_dir=tmp_path / "capture",
        execute=False,
    )

    assert status["status"] == "READY_NOT_EXECUTED"
    assert status["execute"] is False
    assert status["db_write_performed"] is False
    assert status["valid_race_rows"] == 1
    assert status["valid_runner_rows"] == 2
    with sqlite3.connect(db_path) as conn:
        table_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE,),
        ).fetchone()[0]
    assert table_count == 0


def test_existing_official_result_artifact_cli_appends_without_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    race_rows_path, runner_rows_path, quarantine_path = _write_official_artifact_jsonl(
        tmp_path,
        _official_artifact_rows(),
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_existing_artifact_append"
    )

    result = capture.main(
        [
            "--date",
            "2026-06-10",
            "--existing-race-rows-jsonl",
            str(race_rows_path),
            "--existing-runner-rows-jsonl",
            str(runner_rows_path),
            "--existing-quarantine-jsonl",
            str(quarantine_path),
            "--output-dir",
            str(output_dir),
            "--db",
            str(db_path),
            "--execute-db-ingest",
        ]
    )

    assert result == 0
    report = json.loads((output_dir / "autonomous_official_result_capture_report.json").read_text())
    assert report["ingest_report_status"] == "SUCCESS"
    assert report["official_result_evidence_db_ingest"]["status"] == (
        "APPENDED_OFFICIAL_RESULT_EVIDENCE"
    )
    assert report["official_result_evidence_db_ingest"]["inserted_race_rows"] == 1
    assert report["official_result_evidence_db_ingest"]["inserted_runner_rows"] == 2
    assert report["no_write_guarantees"]["label_write"] is False
    with sqlite3.connect(db_path) as conn:
        race_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}"
        ).fetchone()[0]
        runner_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}"
        ).fetchone()[0]
        label_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('race_metadata', 'dog_race_data')
            """
        ).fetchone()[0]
    assert race_count == 1
    assert runner_count == 2
    assert label_count == 0


def test_existing_official_result_artifact_cli_blocks_live_shared_lock(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    race_rows_path, runner_rows_path, quarantine_path = _write_official_artifact_jsonl(
        tmp_path,
        _official_artifact_rows(),
    )
    lock_path = tmp_path / "shadow_autopilot.lock"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "shadow_autopilot_daemon_lock_v1",
                "run_id": "test_live_lock",
                "pid": os.getpid(),
            }
        ),
        encoding="utf-8",
    )
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_existing_artifact_lock_blocked"
    )

    result = capture.main(
        [
            "--date",
            "2026-06-10",
            "--existing-race-rows-jsonl",
            str(race_rows_path),
            "--existing-runner-rows-jsonl",
            str(runner_rows_path),
            "--existing-quarantine-jsonl",
            str(quarantine_path),
            "--output-dir",
            str(output_dir),
            "--db",
            str(db_path),
            "--execute-db-ingest",
            "--lock-path",
            str(lock_path),
            "--require-lock-free",
        ]
    )

    assert result == 0
    report = json.loads((output_dir / "autonomous_official_result_capture_report.json").read_text())
    ingest_status = report["official_result_evidence_db_ingest"]
    assert ingest_status["status"] == "BLOCKED_SHARED_LOCK_HELD"
    assert ingest_status["execute"] is True
    assert ingest_status["db_write_performed"] is False
    assert ingest_status["valid_race_rows"] == 1
    assert ingest_status["valid_runner_rows"] == 2
    assert ingest_status["shared_lock_status"]["status"] == "present_live_pid"
    with sqlite3.connect(db_path) as conn:
        table_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE,),
        ).fetchone()[0]
    assert table_count == 0


def test_official_result_evidence_db_ingest_appends_idempotently(tmp_path):
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_rows = _official_artifact_rows()
    output_dir = tmp_path / "capture"

    first = capture.append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=artifact_rows,
        output_dir=output_dir,
        execute=True,
    )
    second = capture.append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=artifact_rows,
        output_dir=output_dir,
        execute=True,
    )

    assert first["status"] == "APPENDED_OFFICIAL_RESULT_EVIDENCE"
    assert first["db_write_performed"] is True
    assert first["inserted_race_rows"] == 1
    assert first["inserted_runner_rows"] == 2
    assert second["status"] == "NOOP_ALREADY_PRESENT"
    assert second["db_write_performed"] is False
    assert second["inserted_race_rows"] == 0
    assert second["inserted_runner_rows"] == 0
    with sqlite3.connect(db_path) as conn:
        race_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}"
        ).fetchone()[0]
        runner_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}"
        ).fetchone()[0]
        label_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('race_metadata', 'dog_race_data')
            """
        ).fetchone()[0]
    assert race_count == 1
    assert runner_count == 2
    assert label_count == 0


def test_official_result_evidence_db_ingest_appends_valid_rows_with_quarantine(tmp_path):
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass

    status = capture.append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=_mixed_official_artifact_rows(),
        output_dir=tmp_path / "capture",
        execute=True,
    )

    assert status["status"] == "APPENDED_OFFICIAL_RESULT_EVIDENCE_WITH_QUARANTINE"
    assert status["db_write_performed"] is True
    assert status["valid_race_rows"] == 1
    assert status["valid_runner_rows"] == 2
    assert status["blocked_race_rows"] == 1
    assert status["blocked_runner_rows"] == 2
    assert status["inserted_race_rows"] == 1
    assert status["inserted_runner_rows"] == 2
    assert status["blocker_reason_counts"] == {
        "duplicate_finish_positions": 1,
        "finish_positions_not_contiguous": 1,
    }
    with sqlite3.connect(db_path) as conn:
        race_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE}"
        ).fetchone()[0]
        runner_count = conn.execute(
            f"SELECT COUNT(*) FROM {capture.OFFICIAL_RESULT_EVIDENCE_RUNNERS_TABLE}"
        ).fetchone()[0]
        label_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('race_metadata', 'dog_race_data')
            """
        ).fetchone()[0]
    assert race_count == 1
    assert runner_count == 2
    assert label_count == 0


def test_official_result_evidence_db_ingest_blocks_unsafe_rows(tmp_path):
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_rows = _official_artifact_rows()
    artifact_rows["runner_rows"][0]["source_url"] = "https://sportsbet.example/race"

    status = capture.append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=artifact_rows,
        output_dir=tmp_path / "capture",
        execute=True,
    )

    assert status["status"] == "BLOCKED_UNSAFE_OFFICIAL_RESULT_EVIDENCE"
    assert status["db_write_performed"] is False
    assert status["blocked_race_rows"] == 1
    assert status["blocker_reason_counts"] == {"runner_source_url_mismatch": 1}
    with sqlite3.connect(db_path) as conn:
        table_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type = 'table'
              AND name = ?
            """,
            (capture.OFFICIAL_RESULT_EVIDENCE_RACES_TABLE,),
        ).fetchone()[0]
    assert table_count == 0


def test_official_result_evidence_db_ingest_rejects_lookalike_source_url(tmp_path):
    db_path = tmp_path / "labels.sqlite"
    with sqlite3.connect(db_path):
        pass
    artifact_rows = _official_artifact_rows()
    artifact_rows["race_rows"][0]["source_url"] = "https://thedogs.com.au.evil/race"
    for row in artifact_rows["runner_rows"]:
        row["source_url"] = "https://thedogs.com.au.evil/race"

    status = capture.append_official_result_evidence_to_db(
        db_path=db_path,
        artifact_rows=artifact_rows,
        output_dir=tmp_path / "capture",
        execute=True,
    )

    assert status["status"] == "BLOCKED_UNSAFE_OFFICIAL_RESULT_EVIDENCE"
    assert status["db_write_performed"] is False
    assert status["blocker_reason_counts"] == {
        "official_source_url_missing_or_invalid": 1
    }


def test_capture_report_surfaces_official_result_evidence_db_write_scope(tmp_path):
    status = {
        **capture.evidence_db_ingest_not_executed(),
        "execute": True,
        "status": "APPENDED_OFFICIAL_RESULT_EVIDENCE",
        "db_write_performed": True,
        "inserted_race_rows": 1,
        "inserted_runner_rows": 2,
    }

    report = capture.build_capture_report(
        generated_at=datetime.fromisoformat("2026-06-10T15:00:00+10:00"),
        dry_run_command=["python", "scripts/ingest_results_for_date.py"],
        dry_run_returncode=0,
        ingest_report={
            "status": "SUCCESS",
            "candidate_count": 1,
            "ingested_count": 1,
        },
        artifact_rows=_official_artifact_rows(),
        output_dir=tmp_path / "capture",
        evidence_db_ingest=status,
    )

    assert report["official_result_evidence_db_ingest"]["status"] == (
        "APPENDED_OFFICIAL_RESULT_EVIDENCE"
    )
    assert report["no_write_guarantees"]["db_write"] is True
    assert report["no_write_guarantees"]["label_write"] is False
    assert report["no_write_guarantees"]["canonical_result_label_write"] is False
