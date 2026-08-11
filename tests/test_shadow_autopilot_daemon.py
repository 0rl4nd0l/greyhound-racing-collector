import os
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import shadow_autopilot_daemon as daemon
from src.predictor.on_demand import canonical_bytes


def _write_current_index_runner_sources(evidence_root: Path, race_url: str) -> dict:
    form = evidence_root / "upcoming/healesville-r1.csv"
    form.parent.mkdir(parents=True, exist_ok=True)
    form.write_bytes(b"box|dog_name\n1|Alpha\n2|Beta\n")
    sidecar = form.with_name(form.name + ".metadata.json")
    sidecar.write_bytes(canonical_bytes({
        "runner_completeness_after_canonical_alignment": {
            "status": "COMPLETE", "runner_count": 2,
            "participants": [{"box_number": 1, "dog_name": "Alpha", "scratch_state": "ACTIVE"}, {"box_number": 2, "dog_name": "Beta", "scratch_state": "ACTIVE"}],
        },
        "prejump_shadow_metadata": {
        "status": "PASS", "metadata_is_leakage_safe": True,
        "race_date": "2026-06-12", "venue": "HEA", "race_number": 1,
        "source_url": race_url, "metadata_captured_at": "2026-06-12T00:01:01+10:00",
        "runner_box_name_list": [{"box_number": 1, "dog_name": "Alpha"}, {"box_number": 2, "dog_name": "Beta"}],
        "canonical_final_runner_alignment": {"status": "aligned", "canonical_runner_set_status": "available"},
    }}))
    return {"schema_version": "prejump_sidecar_metadata_coverage_v1", "races": [{
        "race_url": race_url, "csv_path": str(form), "sidecar_path": str(sidecar)
    }]}


def test_daemon_default_min_joined_races_matches_review_target():
    args = daemon.parse_args(["run-once"])

    assert daemon.DEFAULT_TARGET_JOINED_RACES == 100
    assert daemon.DEFAULT_MIN_JOINED_RACES == 100
    assert args.target_joined_races == 100
    assert args.min_joined_races == 100
    assert args.enable_autonomous_odds_capture is False
    assert args.execute_autonomous_odds_capture is False
    assert args.allow_auto_scrape_odds is False
    assert args.enable_autonomous_result_capture is False
    assert args.skip_unified_dataset is False
    assert args.shadow_model is None
    assert (
        args.autonomous_odds_capture_limit
        == daemon.DEFAULT_FULL_DAEMON_AUTONOMOUS_ODDS_CAPTURE_LIMIT
    )
    assert daemon.DEFAULT_FULL_DAEMON_AUTONOMOUS_ODDS_CAPTURE_LIMIT == 16
    assert args.refresh_limit == daemon.DEFAULT_FULL_DAEMON_REFRESH_LIMIT == 6
    assert args.result_backlog_limit == daemon.DEFAULT_FULL_DAEMON_RESULT_BACKLOG_LIMIT
    assert (
        args.result_backlog_shadow_run_limit
        == daemon.DEFAULT_FULL_DAEMON_RESULT_BACKLOG_SHADOW_RUN_LIMIT
    )
    assert (
        args.result_backlog_lookback_days
        == daemon.DEFAULT_FULL_DAEMON_RESULT_BACKLOG_LOOKBACK_DAYS
    )


def test_daemon_accepts_autonomous_odds_capture_passthrough_flags():
    args = daemon.parse_args(
        [
            "run-once",
            "--enable-autonomous-odds-capture",
            "--execute-autonomous-odds-capture",
            "--allow-auto-scrape-odds",
        ]
    )

    assert args.enable_autonomous_odds_capture is True
    assert args.execute_autonomous_odds_capture is True
    assert args.allow_auto_scrape_odds is True


def test_daemon_accepts_autonomous_result_capture_passthrough_flag():
    args = daemon.parse_args(
        [
            "run-once",
            "--enable-autonomous-result-capture",
        ]
    )

    assert args.enable_autonomous_result_capture is True


def test_initial_daemon_run_report_marks_long_cycle_running(tmp_path):
    report = daemon.initial_daemon_run_report(
        run_id="20260613T145200+1000",
        generated_at=datetime.fromisoformat("2026-06-13T14:52:00+10:00"),
        current_time="2026-06-13T14:52:00+10:00",
        output_dir=tmp_path / "shadow_autopilot_daemonization_v1_test",
        lock_path=tmp_path / "runtime" / "shadow_autopilot.lock",
        state_path=tmp_path / "runtime" / "state.json",
        odds_capture_state_path=tmp_path / "runtime" / "odds_capture_state.json",
        autonomous_odds_capture_enabled=True,
        autonomous_result_capture_enabled=True,
    )

    assert report["schema_version"] == "shadow_autopilot_daemon_run_v1"
    assert report["status"] == "RUNNING"
    assert report["final_verdict"] == "DAEMON_RUNNING"
    assert report["runtime_action"] == "FULL_DAEMON_IN_PROGRESS"
    assert report["readiness_decision"] == "IN_PROGRESS"
    assert report["current_time"] == "2026-06-13T14:52:00+10:00"
    assert report["lock_path"].endswith("shadow_autopilot.lock")
    assert report["state_path"].endswith("state.json")
    assert report["odds_capture_state_path"].endswith("odds_capture_state.json")
    assert report["autonomous_odds_capture_enabled"] is True
    assert report["autonomous_result_capture_enabled"] is True
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["betting_action"] is False


def test_output_dir_safe_accepts_configured_external_evidence_root(
    tmp_path, monkeypatch
):
    repo_root = tmp_path / "release_repo"
    evidence_root = tmp_path / "runtime_artifacts" / "full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_external"
    repo_root.mkdir()

    monkeypatch.setattr(daemon, "ROOT", repo_root)

    assert (
        daemon.assert_output_dir_safe(output_dir, evidence_root=evidence_root)
        == output_dir.absolute()
    )

    try:
        daemon.assert_output_dir_safe(
            evidence_root / "not_a_daemon_output",
            evidence_root=evidence_root,
        )
    except ValueError as exc:
        assert str(exc).startswith("output_dir_must_be_shadow_autopilot_daemon_artifact")
    else:
        raise AssertionError("external evidence root must still enforce daemon prefix")


def test_daily_shadow_run_from_autopilot_uses_run_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    autopilot_output_dir = tmp_path / "artifacts/shadow_autopilot_v1_x"
    daily_dir = tmp_path / "artifacts/daily_race_ingest_shadow_x"
    daemon.write_json(
        autopilot_output_dir / "run_manifest.json",
        {
            "source_artifacts": {
                "daily_shadow_run_dir": "artifacts/daily_race_ingest_shadow_x"
            }
        },
    )
    daemon.write_json(
        daily_dir / "shadow_manifest.json",
        {"race_count": 1, "prediction_rows": 8},
    )

    resolved_dir, manifest = daemon.daily_shadow_run_from_autopilot(
        autopilot_output_dir
    )

    assert resolved_dir == daily_dir
    assert manifest["prediction_rows"] == 8


def test_timing_aligned_rerun_source_paths_from_autopilot_are_existence_gated(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    autopilot_output_dir = tmp_path / "artifacts/shadow_autopilot_v1_x"
    autopilot_output_dir.mkdir(parents=True)
    plan_path = autopilot_output_dir / "timing_aligned_prediction_rerun_plan.json"
    execution_path = (
        autopilot_output_dir / "timing_aligned_prediction_rerun_execution_status.json"
    )
    plan_path.write_text("{}", encoding="utf-8")
    execution_path.write_text("{}", encoding="utf-8")

    paths = daemon.timing_aligned_rerun_source_paths_from_autopilot(
        autopilot_output_dir
    )
    source_artifacts = daemon.timing_aligned_rerun_source_artifacts_from_autopilot(
        autopilot_output_dir
    )
    missing_paths = daemon.timing_aligned_rerun_source_paths_from_autopilot(
        tmp_path / "artifacts/missing_autopilot"
    )
    missing_source_artifacts = (
        daemon.timing_aligned_rerun_source_artifacts_from_autopilot(
            tmp_path / "artifacts/missing_autopilot"
        )
    )

    assert paths["timing_aligned_rerun_plan"] == plan_path
    assert paths["timing_aligned_rerun_execution_status"] == execution_path
    assert source_artifacts == {
        "timing_aligned_rerun_plan": (
            "artifacts/shadow_autopilot_v1_x/"
            "timing_aligned_prediction_rerun_plan.json"
        ),
        "timing_aligned_rerun_execution_status": (
            "artifacts/shadow_autopilot_v1_x/"
            "timing_aligned_prediction_rerun_execution_status.json"
        ),
    }
    assert missing_paths == {
        "timing_aligned_rerun_plan": None,
        "timing_aligned_rerun_execution_status": None,
    }
    assert missing_source_artifacts == {}


def test_run_once_exception_writes_terminal_daemon_report(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_exception"
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "protected_hashes", lambda: {})
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    generated = {}

    def capture_write_service_files(**kwargs):
        generated.update(kwargs)
        return {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        }

    monkeypatch.setattr(daemon, "write_service_files", capture_write_service_files)
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "lock_path": str(kwargs["lock_path"]),
        },
    )
    monkeypatch.setattr(
        daemon,
        "probe_duplicate_lock",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        daemon,
        "probe_stale_lock_cleanup",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        daemon,
        "simulate_timeout_recovery",
        lambda output_dir: {"status": "PASS"},
    )
    monkeypatch.setattr(
        daemon,
        "release_lock",
        lambda lock_path, run_id: {"status": "RELEASED", "run_id": run_id},
    )

    def failing_run_command(**kwargs):
        assert kwargs["name"] == "autopilot_cycle"
        command = kwargs["command"]
        assert command[command.index("--collector-lock-path") + 1] == str(
            launch_dir / "shared-runtime" / "shadow_autopilot.lock"
        )
        raise RuntimeError("synthetic daemon failure")

    monkeypatch.setattr(daemon, "run_command", failing_run_command)

    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    monkeypatch.chdir(launch_dir)
    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "exception",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--lock-path",
            "shared-runtime/shadow_autopilot.lock",
        ]
    )

    report = daemon.run_once(args)
    written = json.loads((output_dir / "daemon_run_report.json").read_text())

    assert report["final_verdict"] == "PARTIAL_DAEMONIZATION"
    assert report["runtime_action"] == "CHECK_DAEMON_EXCEPTION"
    assert report["exception_type"] == "RuntimeError"
    assert report["exception_message"] == "synthetic daemon failure"
    assert written["runtime_action"] == "CHECK_DAEMON_EXCEPTION"
    assert generated["pause_path"] == (
        launch_dir / "shared-runtime" / "pause-heavy-scheduling"
    )
    assert (output_dir / "final_status.txt").read_text(encoding="utf-8") == (
        "PARTIAL_DAEMONIZATION\n"
    )


def test_run_once_releases_after_primary_when_odds_refresh_due(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_release"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    autopilot_dir = evidence_root / "shadow_autopilot_v1_release_daemon"
    daily_dir = evidence_root / "daily_race_ingest_shadow_release_daemon"
    release_calls = []

    daemon.write_json(
        autopilot_dir / "run_manifest.json",
        {
            "source_artifacts": {
                "daily_shadow_run_dir": str(daily_dir.relative_to(tmp_path))
            }
        },
    )
    daemon.write_json(daily_dir / "shadow_manifest.json", {"prediction_rows": 8})
    daemon.write_json(
        autopilot_dir / "odds_capture_refresh_report.json",
        {
            "status": "SUCCESS",
            "next_preferred_window": {
                "status": "CLOSED",
                "next_window_opens_at": "2026-06-13T20:21:00+10:00",
                "next_window_closes_at": "2026-06-13T20:21:00+10:00",
                "recommended_rerun_after_local": "2026-06-13T21:12:00+10:00",
                "next_race": {
                    "race_id": "Race 6 - TAREE - 2026-06-13",
                    "date": "2026-06-13",
                    "venue": "TAREE",
                    "race_number": 6,
                    "race_time": "20:21",
                    "jump_datetime": "2026-06-13T20:21:00+10:00",
                },
            },
        },
    )
    daemon.write_json(
        autopilot_dir / "autonomous_live_odds_capture_status.json",
        {
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "operator_status": "READY",
            "inserted_live_odds_rows": 0,
            "ready_count": 0,
            "status_counts": {},
            "blocked_attempt_count": 0,
            "blocked_attempts": [],
        },
    )
    daemon.write_json(
        autopilot_dir / "DAILY_STATUS.json",
        {
            "schema_version": "shadow_autopilot_daily_status_v1",
            "generated_at": "2026-06-13T21:12:08+10:00",
            "readiness_decision": "READY_FOR_RELIABILITY_REVIEW",
            "autonomous_live_odds_capture_status": (
                "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
            ),
            "autonomous_live_odds_inserted_rows": 0,
            "autonomous_official_result_capture_status": (
                "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
            ),
            "autonomous_official_result_candidate_count": 25,
            "autonomous_official_result_quarantined_race_ids": [
                "Race 8 - TAREE - 2026-06-13"
            ],
            "autonomous_official_result_quarantine_reason_counts": {
                "ingest_failed_or_unsafe_match": 1
            },
            "autonomous_official_result_quarantine_error_counts": {
                "result_boxes_not_in_participants:9,10": 1
            },
            "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": {
                "10": 1,
                "9": 1,
            },
            "autonomous_official_result_quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "result_boxes_not_in_participants": [9, 10],
                }
            ],
            "unified_evidence_dataset_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_dataset_rows": 80,
            "unified_evidence_eligible_rows": 0,
            "backlog_unified_evidence_eligible_rows": 189,
            "best_aggregate_unified_evidence_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/backlog_unified_evidence_datasets_status.json"
            ),
            "best_aggregate_unified_evidence_status": (
                "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
            ),
            "best_aggregate_unified_evidence_dataset_count": 12,
            "best_aggregate_unified_evidence_row_count": 241,
            "best_aggregate_unified_evidence_eligible_rows": 189,
            "best_aggregate_unified_evidence_artifact_odds_rows_seen": 114,
            "best_aggregate_unified_evidence_artifact_odds_rows_accepted": 14,
            "best_aggregate_unified_evidence_artifact_odds_rows_rejected": 100,
            "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts": {
                "odds_after_jump": 73,
                "runner_set_mismatch": 27,
            },
            "rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "rolling_model_comparison_sample_races": 163,
            "high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "high_accuracy_promotion_pr_gate_status": "BLOCKED",
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
            "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_distance_blockers": [
                "no_candidate_passed_rank_first_accuracy_gate"
            ],
            "prediction_rows_today": 80,
            "races_scored_today": 11,
            "races_with_complete_valid_prejump_odds": 2,
            "races_with_missing_odds_rows": 9,
        },
    )

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "protected_hashes", lambda: {})
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "full_daemon_odds_window_defer_decision",
        lambda odds_state, current_time: {
            "should_defer": False,
            "reason": "odds_capture_action_not_imminent",
        },
    )
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: {
            "run_id": kwargs["run_id"],
            "lock_path": str(kwargs["lock_path"]),
        },
    )
    monkeypatch.setattr(
        daemon,
        "probe_duplicate_lock",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        daemon,
        "probe_stale_lock_cleanup",
        lambda *args, **kwargs: {"status": "PASS"},
    )
    monkeypatch.setattr(
        daemon,
        "simulate_timeout_recovery",
        lambda output_dir: {"status": "PASS"},
    )

    def fake_release(lock_path, run_id):
        release_calls.append((lock_path, run_id))
        return {"status": "RELEASED", "released": True, "run_id": run_id}

    monkeypatch.setattr(daemon, "release_lock", fake_release)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        if name != "autopilot_cycle":
            raise AssertionError(f"post-primary release should not run {name}")
        assert command[command.index("--autonomous-odds-capture-limit") + 1] == "16"
        assert command[command.index("--result-backlog-limit") + 1] == "8"
        assert command[command.index("--result-backlog-shadow-run-limit") + 1] == "16"
        assert command[command.index("--result-backlog-lookback-days") + 1] == "2"
        daemon.write_json(
            output_dir / "logs" / "autopilot_cycle.stdout.txt",
            {"output_dir": str(autopilot_dir.relative_to(tmp_path))},
        )
        return {
            "name": name,
            "returncode": 0,
            "timed_out": False,
            "status": "PASS",
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "release",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T21:12:00+10:00",
            "--db",
            str(db_path),
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)
    written = json.loads((output_dir / "daemon_run_report.json").read_text())
    decision = json.loads(
        (output_dir / "post_primary_odds_capture_release_decision.json").read_text()
    )

    assert report["final_verdict"] == "PARTIAL_DAEMONIZATION"
    assert report["runtime_action"] == "RELEASE_FULL_DAEMON_FOR_ODDS_CAPTURE"
    assert report["readiness_decision"] == "ODDS_CAPTURE_PRIORITY"
    assert report["odds_capture_state_publish_status"] == "PUBLISHED"
    assert report["autopilot_daily_status_path"].endswith("DAILY_STATUS.json")
    assert report["autopilot_daily_readiness_decision"] == (
        "READY_FOR_RELIABILITY_REVIEW"
    )
    assert report["autonomous_official_result_capture_status"] == (
        "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    )
    assert report["autonomous_official_result_quarantined_race_ids"] == [
        "Race 8 - TAREE - 2026-06-13"
    ]
    assert report["autonomous_official_result_quarantine_error_counts"] == {
        "result_boxes_not_in_participants:9,10": 1
    }
    assert report[
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
    ] == {
        "10": 1,
        "9": 1,
    }
    assert report["autonomous_official_result_quarantine_runner_set_mismatch_samples"][
        0
    ]["result_boxes_not_in_participants"] == [9, 10]
    assert report["unified_evidence_dataset_status"] == "UNIFIED_EVIDENCE_DATASET_BUILT"
    assert report["backlog_unified_evidence_eligible_rows"] == 189
    assert report[
        "autopilot_cycle_best_aggregate_unified_evidence_status_path"
    ].endswith("backlog_unified_evidence_datasets_status.json")
    assert report["autopilot_cycle_best_aggregate_unified_evidence_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert report["autopilot_cycle_best_aggregate_unified_evidence_dataset_count"] == 12
    assert report["autopilot_cycle_best_aggregate_unified_evidence_row_count"] == 241
    assert report[
        "autopilot_cycle_best_aggregate_unified_evidence_eligible_rows"
    ] == 189
    assert report[
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_seen"
    ] == 114
    assert report[
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_accepted"
    ] == 14
    assert report[
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_rejected"
    ] == 100
    assert report[
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts"
    ] == {
        "odds_after_jump": 73,
        "runner_set_mismatch": 27,
    }
    assert report["rolling_model_comparison_status"] == (
        "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    )
    assert report["rolling_model_comparison_sample_races"] == 163
    assert report["high_accuracy_refinement_status"] == "BLOCKED_KEEP_BASELINE"
    assert report[
        "autopilot_cycle_reserve_substitution_policy_impact_status"
    ] == "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    assert report[
        "autopilot_cycle_reserve_substitution_policy_impact_ready_candidate_count"
    ] == 4
    assert report[
        "autopilot_cycle_reserve_substitution_policy_impact_mapping_pair_count"
    ] == 5
    assert report[
        "autopilot_cycle_reserve_substitution_policy_impact_potential_runner_rows_blocked"
    ] == 32
    assert (
        report[
            "autopilot_cycle_reserve_substitution_policy_impact_dataset_join_allowed"
        ]
        is False
    )
    assert (
        report[
            "autopilot_cycle_reserve_substitution_policy_impact_official_result_acceptance_allowed"
        ]
        is False
    )
    assert report[
        "autopilot_cycle_reserve_substitution_policy_impact_db_write"
    ] is False
    assert report["promotion_distance_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert report["prediction_rows_today"] == 80
    assert report["races_with_complete_valid_prejump_odds"] == 2
    assert report["odds_capture_next_meaningful_action"] == (
        "REFRESH_UPCOMING_RACE_WINDOW"
    )
    assert report["lock_release"]["released"] is True
    assert release_calls == [(args.lock_path or daemon.DEFAULT_LOCK_PATH, "release")]
    assert decision["should_release"] is True
    assert decision["reason"] == "post_primary_odds_capture_refresh_due_now"
    assert not (output_dir / "automated_join_report.json").exists()
    assert written["runtime_action"] == report["runtime_action"]
    assert written[
        "autopilot_cycle_reserve_substitution_policy_impact_status"
    ] == report["autopilot_cycle_reserve_substitution_policy_impact_status"]
    assert written[
        "autopilot_cycle_best_aggregate_unified_evidence_eligible_rows"
    ] == report["autopilot_cycle_best_aggregate_unified_evidence_eligible_rows"]
    assert odds_state_path.exists()


def test_lock_held_daemon_run_report_surfaces_active_owner(tmp_path):
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    report = daemon.lock_held_daemon_run_report(
        run_id="20260613T150410+1000",
        generated_at=datetime.fromisoformat("2026-06-13T15:04:10+10:00"),
        current_time="2026-06-13T15:04:10+10:00",
        output_dir=tmp_path / "shadow_autopilot_daemonization_v1_lock_held",
        lock_path=tmp_path / "runtime" / "shadow_autopilot.lock",
        lock_details={
            "reason": "active_lock_present",
            "lock_path": "artifacts/runtime/shadow_autopilot.lock",
            "existing_lock": {
                "run_id": "20260613T150400+1000_odds_capture",
                "output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_daemonization_v1_20260613T150400+1000_odds_capture",
                "pid": 443246,
                "hostname": "worker-host",
                "started_at": "2026-06-13T15:04:00.593224+10:00",
            },
        },
        odds_capture_state_path=odds_state_path,
        odds_capture_state={
            "run_id": "20260613T145856+1000_odds_capture",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "status": "READY",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "inserted_live_odds_rows": 15,
            "ready_count": 8,
            "status_counts": {"APPENDED": 2, "SKIPPED_ALREADY_CAPTURED": 6},
            "blocked_attempt_count": 0,
            "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
            "next_meaningful_action_at": "2026-06-13T14:58:56+10:00",
        },
    )

    assert report["schema_version"] == "shadow_autopilot_daemon_run_v1"
    assert report["status"] == "SKIPPED_LOCK_HELD"
    assert report["final_verdict"] == "PARTIAL_DAEMONIZATION"
    assert report["runtime_action"] == "SKIP_LOCK_HELD"
    assert report["readiness_decision"] == "WAIT_FOR_ACTIVE_DAEMON"
    assert report["lock_validation_status"] == "SKIPPED_LOCK_HELD"
    assert report["lock_owner_kind"] == "odds_capture"
    assert report["lock_owner_run_id"] == "20260613T150400+1000_odds_capture"
    assert report["lock_owner_pid"] == 443246
    assert report["lock_reason"] == "active_lock_present"
    assert report["odds_capture_state_path"] == str(odds_state_path)
    assert report["last_odds_capture_run_id"] == "20260613T145856+1000_odds_capture"
    assert report["last_odds_capture_final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["last_odds_capture_status"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    )
    assert report["last_odds_capture_operator_status"] == "READY"
    assert report["last_odds_capture_runtime_action"] == (
        "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    )
    assert report["last_odds_capture_readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert report["last_odds_capture_inserted_live_odds_rows"] == 15
    assert report["last_odds_capture_ready_count"] == 8
    assert report["last_odds_capture_status_counts"] == {
        "APPENDED": 2,
        "SKIPPED_ALREADY_CAPTURED": 6,
    }
    assert report["last_odds_capture_blocked_attempt_count"] == 0
    assert report["last_odds_capture_next_meaningful_action"] == (
        "REFRESH_UPCOMING_RACE_WINDOW"
    )
    assert report["last_odds_capture_next_meaningful_action_at"] == (
        "2026-06-13T14:58:56+10:00"
    )
    assert report["protected_paths_unchanged_or_allowed"] is True
    assert report["no_write_guarantees"]["production_promotion"] is False
    assert report["no_write_guarantees"]["betting_action"] is False


def test_full_daemon_lock_retry_waits_for_odds_capture_owner(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    output_dir = tmp_path / "packet"
    calls = []
    sleeps = []

    def fake_acquire_lock(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise daemon.LockBusy(
                {
                    "reason": "active_lock_present",
                    "existing_lock": {
                        "run_id": "20260613T182954+1000_odds_capture",
                        "output_dir": (
                            "artifacts/full_evidence_orchestration_20260525/"
                            "shadow_autopilot_daemonization_v1_20260613T182954+1000_odds_capture"
                        ),
                        "pid": 888886,
                        "hostname": "worker-host",
                        "started_at": "2026-06-13T18:29:54+10:00",
                    },
                }
            )
        return {
            "schema_version": "shadow_autopilot_daemon_lock_v1",
            "run_id": kwargs["run_id"],
            "pid": 12345,
            "hostname": "worker-host",
            "started_at": "2026-06-13T18:32:06+10:00",
            "output_dir": daemon.relpath(kwargs["output_dir"]),
        }

    monkeypatch.setattr(daemon, "acquire_lock", fake_acquire_lock)

    def fake_sleep(seconds):
        sleeps.append(seconds)
        marker = daemon.read_active_full_daemon_lock_wait_marker(lock_path)
        assert marker is not None
        assert marker["run_id"] == "20260613T183200+1000"
        assert marker["reason"] == "full_daemon_waiting_for_odds_capture_lock_handoff"
        waiting_report = daemon.load_json(output_dir / "daemon_run_report.json")
        assert waiting_report["status"] == "WAITING_LOCK_HELD"
        assert waiting_report["final_verdict"] == "DAEMON_WAITING_FOR_ODDS_CAPTURE_LOCK"
        assert waiting_report["runtime_action"] == "WAIT_FOR_ODDS_CAPTURE_LOCK_HANDOFF"
        assert waiting_report["readiness_decision"] == "ODDS_CAPTURE_IN_PROGRESS"
        assert waiting_report["lock_owner_kind"] == "odds_capture"
        assert waiting_report["lock_owner_run_id"] == (
            "20260613T182954+1000_odds_capture"
        )
        assert waiting_report["lock_retry"]["status"] == (
            "WAITING_FOR_ODDS_CAPTURE_LOCK"
        )
        assert waiting_report["lock_retry"]["attempt_count"] == 1
        assert waiting_report["lock_retry"]["waited_seconds"] == 0.0

    monkeypatch.setattr(daemon.time, "sleep", fake_sleep)

    payload = daemon.acquire_lock_with_odds_capture_retry(
        lock_path=lock_path,
        run_id="20260613T183200+1000",
        stale_after_seconds=3600,
        output_dir=output_dir,
        retry_seconds=15,
        poll_seconds=5,
    )

    assert len(calls) == 2
    assert sleeps == [5.0]
    assert payload["run_id"] == "20260613T183200+1000"
    assert payload["lock_retry"]["status"] == "ACQUIRED_AFTER_ODDS_CAPTURE_WAIT"
    assert payload["lock_retry"]["attempt_count"] == 2
    assert payload["lock_retry"]["waited_seconds"] == 5.0
    assert payload["lock_retry"]["first_lock"]["existing_lock"]["run_id"].endswith(
        "_odds_capture"
    )
    assert not daemon.full_daemon_lock_wait_marker_path(lock_path).exists()


def test_full_daemon_lock_retry_budget_covers_odds_capture_timeout():
    assert daemon.DEFAULT_FULL_DAEMON_ODDS_LOCK_RETRY_SECONDS > (
        daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS
    )


def test_observer_lock_retry_exhaustion_preserves_odds_owner_and_cleans_marker(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    output_dir = tmp_path / "packet"
    lock_path.parent.mkdir(parents=True)
    output_dir.mkdir()
    owner = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": "20260613T182954+1000_odds_capture",
        "pid": os.getpid(),
        "hostname": "worker-host",
        "started_at": "2026-06-13T18:29:54+10:00",
        "output_dir": "/runtime/active-odds",
        "phase": "odds_capture",
    }
    lock_bytes = json.dumps(owner, sort_keys=True).encode()
    lock_path.write_bytes(lock_bytes)
    evidence = {
        "lock_path": str(lock_path),
        "lock_owner_run_id": owner["run_id"],
        "lock_owner_pid": owner["pid"],
        "lock_owner_hostname": owner["hostname"],
        "lock_owner_started_at": owner["started_at"],
        "lock_owner_output_dir": owner["output_dir"],
        "lock_owner_phase": owner["phase"],
        "reason": "existing_lock_present_no_steal",
    }
    attempts = []
    sleeps = []

    def busy_acquire(*args, **kwargs):
        attempts.append((args, kwargs))
        raise daemon.CollectorBusy(evidence)

    def fake_sleep(seconds):
        sleeps.append(seconds)
        marker = daemon.read_active_full_daemon_lock_wait_marker(lock_path)
        assert marker is not None
        assert marker["run_id"] == "20260613T183200+1000"
        assert lock_path.read_bytes() == lock_bytes

    monkeypatch.setattr(daemon, "acquire_collector_lock_no_steal", busy_acquire)
    monkeypatch.setattr(daemon.time, "sleep", fake_sleep)

    with pytest.raises(daemon.CollectorBusy) as exc_info:
        daemon.acquire_observer_lock_with_odds_capture_retry(
            lock_path=lock_path,
            run_id="20260613T183200+1000",
            output_dir=output_dir,
            retry_seconds=10,
            poll_seconds=5,
        )

    retry = exc_info.value.evidence["lock_retry"]
    assert retry["status"] == "GAVE_UP_LOCK_HELD"
    assert retry["attempt_count"] == 3
    assert retry["waited_seconds"] == 10.0
    assert retry["retried_for_odds_capture_lock"] is True
    assert len(attempts) == 3
    assert sleeps == [5.0, 5.0]
    assert lock_path.read_bytes() == lock_bytes
    assert not daemon.full_daemon_lock_wait_marker_path(lock_path).exists()


def test_completed_daemon_run_report_envelope_is_self_describing(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/daemon"

    report = daemon.completed_daemon_run_report_envelope(
        run_id="20260613T160211+1000",
        generated_at=datetime.fromisoformat("2026-06-13T16:09:28+10:00"),
        current_time="2026-06-13T16:02:11+10:00",
        output_dir=output_dir,
        final_verdict="DAEMON_READY",
    )

    assert report["schema_version"] == "shadow_autopilot_daemon_run_v1"
    assert report["run_id"] == "20260613T160211+1000"
    assert report["generated_at"] == "2026-06-13T16:09:28+10:00"
    assert report["current_time"] == "2026-06-13T16:02:11+10:00"
    assert report["output_dir"] == (
        "artifacts/full_evidence_orchestration_20260525/daemon"
    )
    assert report["status"] == "DAEMON_READY"
    assert report["final_verdict"] == "DAEMON_READY"


def test_run_once_defer_writes_startup_output_manifest(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_defer"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "full_daemon_odds_window_defer_decision",
        lambda odds_state, current_time: {
            "should_defer": True,
            "reason": "test_fixed_window_open",
        },
    )
    monkeypatch.setattr(
        daemon,
        "acquire_lock",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("deferred full daemon should not acquire lock")
        ),
    )

    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "defer",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T15:17:11+10:00",
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)

    assert report["final_verdict"] == "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"
    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    assert manifest["schema_version"] == "shadow_autopilot_daemon_output_manifest_v1"
    assert any(path.endswith("daemon_run_report.json") for path in manifest["files"])
    assert any(path.endswith("final_status.txt") for path in manifest["files"])
    assert any(path.endswith("full_daemon_odds_window_defer.json") for path in manifest["files"])


def test_run_once_defer_observes_result_before_odds_priority_return(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_observer_defer"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    corpus_root = tmp_path / "forward-corpus"
    observed = {
        "status": "COMPLETED",
        "attempted_race_ids": ["race-1"],
        "counts": {"observed": 1},
    }
    defer_times = []

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "run_forward_official_result_observer",
        lambda args, run_id: observed | {"run_id": run_id},
    )
    monkeypatch.setattr(
        daemon,
        "full_daemon_odds_window_defer_decision",
        lambda odds_state, current_time: (
            defer_times.append(current_time)
            or {"should_defer": True, "reason": "test_fixed_window_open"}
        ),
    )
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("odds-priority return must remain before the full lock")
        ),
    )

    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "observer_defer",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T15:17:11+10:00",
            "--lock-path",
            str(lock_path),
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            str(corpus_root),
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)

    assert report["final_verdict"] == "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"
    assert defer_times == [datetime.fromisoformat("2026-06-13T15:17:11+10:00")]
    assert report["forward_official_result_observer"] == observed | {
        "run_id": "observer_defer",
        "shared_lock": {
            "lock_path": daemon.relpath(lock_path),
            "phase": "forward_official_result_observer",
            "acquisition_policy": "forward_official_result_observer_no_steal_v1",
            "release": {
                "released": True,
                "reason": "released_by_observer_owner",
            },
        },
    }
    assert report["no_write_guarantees"]["official_result_evidence_write"] is True
    assert json.loads(
        (output_dir / "forward_official_result_observer.json").read_text()
    ) == report["forward_official_result_observer"]


def test_run_once_observer_waits_for_natural_odds_release_then_continues(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_observer_handoff"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    lock_path.parent.mkdir(parents=True)
    owner = {
        "schema_version": "shadow_autopilot_daemon_lock_v1",
        "run_id": "20260613T182954+1000_odds_capture",
        "pid": os.getpid(),
        "hostname": "worker-host",
        "started_at": "2026-06-13T18:29:54+10:00",
        "output_dir": "/runtime/active-odds",
        "phase": "odds_capture",
    }
    lock_bytes = json.dumps(owner, sort_keys=True).encode()
    lock_path.write_bytes(lock_bytes)
    sleeps = []
    observed = {
        "status": "COMPLETED",
        "attempted_race_ids": ["race-1"],
        "counts": {"observed": 1},
    }

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )

    def fake_sleep(seconds):
        sleeps.append(seconds)
        marker = daemon.read_active_full_daemon_lock_wait_marker(lock_path)
        assert marker is not None
        assert marker["run_id"] == "observer_handoff"
        assert marker["reason"] == "full_daemon_waiting_for_odds_capture_lock_handoff"
        waiting_report = json.loads((output_dir / "daemon_run_report.json").read_text())
        assert waiting_report["lock_owner_kind"] == "odds_capture"
        assert waiting_report["lock_retry"]["status"] == (
            "WAITING_FOR_ODDS_CAPTURE_LOCK"
        )
        assert lock_path.read_bytes() == lock_bytes
        lock_path.unlink()

    def observe(args, run_id):
        lock = json.loads(lock_path.read_text())
        assert lock["run_id"] == run_id
        assert lock["phase"] == "forward_official_result_observer"
        return observed | {"run_id": run_id}

    monkeypatch.setattr(daemon.time, "sleep", fake_sleep)
    monkeypatch.setattr(daemon, "run_forward_official_result_observer", observe)
    monkeypatch.setattr(
        daemon,
        "full_daemon_odds_window_defer_decision",
        lambda odds_state, current_time: {
            "should_defer": True,
            "reason": "test_fixed_window_open",
        },
    )
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("odds-priority return must remain before the full lock")
        ),
    )
    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "observer_handoff",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T18:32:06+10:00",
            "--lock-path",
            str(lock_path),
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            str(tmp_path / "forward-corpus"),
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)

    assert report["final_verdict"] == "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"
    assert sleeps == [5.0]
    shared_lock = report["forward_official_result_observer"]["shared_lock"]
    assert shared_lock["lock_retry"]["status"] == (
        "ACQUIRED_AFTER_ODDS_CAPTURE_WAIT"
    )
    assert shared_lock["lock_retry"]["attempt_count"] == 2
    assert shared_lock["release"] == {
        "released": True,
        "reason": "released_by_observer_owner",
    }
    assert not lock_path.exists()
    assert not daemon.full_daemon_lock_wait_marker_path(lock_path).exists()


def test_run_once_observer_failure_stops_before_odds_defer_and_full_lock(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_observer_failed"
    state_path = tmp_path / "runtime" / "state.json"
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    failed = {
        "status": "COMPLETED_WITH_ERRORS",
        "attempted_race_ids": ["race-1"],
    }

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "run_forward_official_result_observer",
        lambda args, run_id: failed | {"run_id": run_id},
    )
    monkeypatch.setattr(
        daemon,
        "full_daemon_odds_window_defer_decision",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("failed observer must stop before odds defer")
        ),
    )
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("failed observer must stop before the full lock")
        ),
    )

    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "observer_failed",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--state-path",
            str(state_path),
            "--lock-path",
            str(lock_path),
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            str(tmp_path / "forward-corpus"),
            "--enable-autonomous-odds-capture",
        ]
    )

    report = daemon.run_once(args)

    assert report["final_verdict"] == "PARTIAL_DAEMONIZATION"
    assert report["lock_validation_status"] == "OBSERVER_LOCK_RELEASED_AFTER_FAILURE"
    assert report["forward_official_result_observer"] == failed | {
        "run_id": "observer_failed",
        "shared_lock": {
            "lock_path": daemon.relpath(lock_path),
            "phase": "forward_official_result_observer",
            "acquisition_policy": "forward_official_result_observer_no_steal_v1",
            "release": {
                "released": True,
                "reason": "released_by_observer_owner",
            },
        },
    }
    assert json.loads(state_path.read_text())[
        "forward_official_result_observer"
    ] == report["forward_official_result_observer"]


def test_live_shared_lock_defers_before_observer_service_or_corpus_mutation(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_lock_busy"
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    corpus_root = tmp_path / "forward-corpus"
    lock_path.parent.mkdir(parents=True)
    corpus_root.mkdir()
    sentinel = corpus_root / "immutable.json"
    sentinel.write_bytes(b'{"immutable":true}\n')
    lock_bytes = json.dumps(
        {
            "schema_version": "shadow_autopilot_daemon_lock_v1",
            "run_id": "active_full_producer",
            "pid": os.getpid(),
            "hostname": "test-host",
            "started_at": "2026-06-13T15:16:00+10:00",
            "output_dir": "/runtime/active-full",
            "phase": "full_daemon",
        },
        sort_keys=True,
    ).encode()
    lock_path.write_bytes(lock_bytes)

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("live shared owner must prevent service mutation")
        ),
    )
    monkeypatch.setattr(
        daemon,
        "run_forward_official_result_observer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live shared owner must prevent observer I/O")
        ),
    )
    monkeypatch.setattr(
        daemon.time,
        "sleep",
        lambda seconds: (_ for _ in ()).throw(
            AssertionError("non-odds owner must not enter bounded handoff")
        ),
    )
    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "lock-busy",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--lock-path",
            str(lock_path),
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            str(corpus_root),
        ]
    )

    report = daemon.run_once(args)

    assert report["status"] == "SKIPPED_LOCK_HELD"
    assert report["runtime_action"] == "DEFER_FORWARD_OBSERVER_SHARED_LOCK_HELD"
    assert report["lock_owner_phase"] == "full_daemon"
    assert report["lock_owner_pid"] == os.getpid()
    assert report["forward_official_result_observer"]["shared_lock"]["lock_retry"][
        "retried_for_odds_capture_lock"
    ] is False
    assert report["forward_official_result_observer"]["attempted_race_ids"] == []
    assert lock_path.read_bytes() == lock_bytes
    assert not daemon.full_daemon_lock_wait_marker_path(lock_path).exists()
    assert list(corpus_root.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b'{"immutable":true}\n'


def test_observer_completion_refreshes_wall_time_before_odds_defer(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_time_crossing"
    lock_path = tmp_path / "runtime" / "shadow_autopilot.lock"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    before = datetime.fromisoformat("2026-06-13T15:09:59+10:00")
    after = datetime.fromisoformat("2026-06-13T15:10:01+10:00")
    wall_times = iter((before, after))

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "wall_clock_now", lambda: next(wall_times))
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "run_forward_official_result_observer",
        lambda args, run_id: {"status": "COMPLETED", "attempted_race_ids": []},
    )

    def decide(_odds_state, current_time):
        assert current_time == after
        return {"should_defer": True, "reason": "crossed_into_defer_horizon"}

    monkeypatch.setattr(daemon, "full_daemon_odds_window_defer_decision", decide)
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("refreshed defer must remain before the full lock")
        ),
    )
    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "time-crossing",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--lock-path",
            str(lock_path),
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            str(tmp_path / "forward-corpus"),
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)

    assert report["final_verdict"] == "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"
    assert report["odds_capture_defer_decision"]["reason"] == (
        "crossed_into_defer_horizon"
    )
    assert report["forward_official_result_observer"]["shared_lock"]["release"][
        "released"
    ] is True
    assert not lock_path.exists()


def test_run_once_defers_before_lock_when_t2_window_recomputed_due(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_t2_defer"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    daemon.write_json(
        odds_state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-12T01:29:00+10:00",
            "window_state_source_updated_at": "2026-06-12T01:29:00+10:00",
            "run_id": "previous_odds",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "odds_capture_refresh_status": "SUCCESS",
            "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
            "next_meaningful_action_at": "2026-06-12T01:29:00+10:00",
            "next_window_opens_at": "2026-06-12T01:35:00+10:00",
            "next_preferred_window": {
                "status": "OPEN_NOW",
                "next_window_opens_at": "2026-06-12T00:35:00+10:00",
                "next_window_closes_at": "2026-06-12T01:37:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "date": "2026-06-12",
                    "venue": "HEA",
                    "race_number": 1,
                    "race_time": "01:37",
                    "jump_datetime": "2026-06-12T01:37:00+10:00",
                },
            },
            "odds_capture_fixed_window_schedule": {
                "generated_at": "2026-06-12T01:29:00+10:00",
                "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
                "next_meaningful_action_at": "2026-06-12T01:29:00+10:00",
                "status_counts": {"PASSED": 4},
            },
        },
    )

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "acquire_lock_with_odds_capture_retry",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("T-2 deferred full daemon should not acquire lock")
        ),
    )

    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "t2_defer",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T01:35:30+10:00",
            "--db",
            str(db_path),
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)
    decision = json.loads((output_dir / "full_daemon_odds_window_defer.json").read_text())

    assert report["final_verdict"] == "DAEMON_DEFERRED_TO_ODDS_CAPTURE_ONLY"
    assert report["runtime_action"] == "DEFER_FULL_DAEMON_FOR_FIXED_WINDOW_ODDS_CAPTURE"
    assert report["readiness_decision"] == "ODDS_CAPTURE_PRIORITY"
    assert decision["should_defer"] is True
    assert decision["due_capture_window_count"] == 1
    assert decision["next_meaningful_action_offset_minutes"] == 2
    assert report["lock_validation_status"] == "NOT_ACQUIRED_DEFERRED"


def test_run_once_lock_held_surfaces_latest_odds_capture_state(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_lock_held"
    odds_state_path = tmp_path / "runtime" / "odds_capture_state.json"
    daemon.write_json(
        odds_state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "run_id": "20260613T175856+1000_odds_capture",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "status": "READY",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "inserted_live_odds_rows": 15,
            "ready_count": 8,
            "status_counts": {"APPENDED": 2, "SKIPPED_ALREADY_CAPTURED": 6},
            "blocked_attempt_count": 0,
            "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
            "next_meaningful_action_at": "2026-06-13T17:58:56+10:00",
        },
    )

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "copy_if_exists", lambda source, dest: None)
    monkeypatch.setattr(
        daemon,
        "write_service_files",
        lambda **kwargs: {
            "status": "SERVICE_FILES_WRITTEN",
            "systemd_deployment_ready": True,
        },
    )
    monkeypatch.setattr(
        daemon,
        "full_daemon_odds_window_defer_decision",
        lambda odds_state, current_time: {
            "should_defer": False,
            "reason": "odds_capture_action_not_imminent",
        },
    )
    monkeypatch.setattr(daemon, "DEFAULT_FULL_DAEMON_ODDS_LOCK_RETRY_SECONDS", 0)

    def busy_lock(**kwargs):
        raise daemon.LockBusy(
            {
                "reason": "active_lock_present",
                "existing_lock": {
                    "run_id": "20260613T180141+1000_odds_capture",
                    "output_dir": (
                        "artifacts/full_evidence_orchestration_20260525/"
                        "shadow_autopilot_daemonization_v1_20260613T180141+1000_odds_capture"
                    ),
                    "pid": 826407,
                    "hostname": "worker-host",
                    "started_at": "2026-06-13T18:01:41+10:00",
                },
            }
        )

    monkeypatch.setattr(daemon, "acquire_lock", busy_lock)

    args = daemon.parse_args(
        [
            "run-once",
            "--run-id",
            "lock_held",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T18:02:06+10:00",
            "--enable-autonomous-odds-capture",
            "--odds-capture-state-path",
            str(odds_state_path),
        ]
    )

    report = daemon.run_once(args)
    written = json.loads((output_dir / "daemon_run_report.json").read_text())

    assert report["status"] == "SKIPPED_LOCK_HELD"
    assert report["lock_owner_kind"] == "odds_capture"
    assert report["lock_owner_run_id"] == "20260613T180141+1000_odds_capture"
    assert report["odds_capture_state_path"] == "runtime/odds_capture_state.json"
    assert report["last_odds_capture_run_id"] == "20260613T175856+1000_odds_capture"
    assert report["last_odds_capture_inserted_live_odds_rows"] == 15
    assert report["last_odds_capture_status_counts"] == {
        "APPENDED": 2,
        "SKIPPED_ALREADY_CAPTURED": 6,
    }
    assert report["last_odds_capture_next_meaningful_action"] == (
        "REFRESH_UPCOMING_RACE_WINDOW"
    )
    assert written["last_odds_capture_run_id"] == report["last_odds_capture_run_id"]


def test_run_command_streams_logs_and_records_timeout(tmp_path):
    result = daemon.run_command(
        name="stream_probe",
        command=[
            sys.executable,
            "-c",
            (
                "import sys, time; "
                "print('stdout-before-timeout'); "
                "print('stderr-before-timeout', file=sys.stderr); "
                "sys.stdout.flush(); sys.stderr.flush(); "
                "time.sleep(2)"
            ),
        ],
        output_dir=tmp_path,
        timeout_seconds=1,
    )

    stdout_path = tmp_path / "logs" / "stream_probe.stdout.txt"
    stderr_path = tmp_path / "logs" / "stream_probe.stderr.txt"
    started_path = tmp_path / "logs" / "stream_probe.started.json"
    running_path = tmp_path / "logs" / "stream_probe.running.json"
    finished_path = tmp_path / "logs" / "stream_probe.finished.json"
    manifest = json.loads((tmp_path / "output_manifest.json").read_text())
    started = json.loads(started_path.read_text())
    running = json.loads(running_path.read_text())
    finished = json.loads(finished_path.read_text())

    assert result["status"] == "FAIL"
    assert result["timed_out"] is True
    assert result["returncode"] != 0
    assert result["timeout_deadline_at"] == started["timeout_deadline_at"]
    assert "stdout-before-timeout" in stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    assert "stderr-before-timeout" in stderr_text
    assert "[TIMEOUT]" in stderr_text
    assert started_path.exists()
    assert running["schema_version"] == "shadow_autopilot_daemon_step_running_v1"
    assert running["status"] == "RUNNING"
    assert isinstance(running["pid"], int)
    assert running["timeout_deadline_at"] == started["timeout_deadline_at"]
    assert finished["schema_version"] == "shadow_autopilot_daemon_step_finished_v1"
    assert finished["status"] == "FAIL"
    assert finished["timed_out"] is True
    assert finished["returncode"] == result["returncode"]
    assert manifest["schema_version"] == "shadow_autopilot_daemon_output_manifest_v1"
    assert any(path.endswith("logs/stream_probe.started.json") for path in manifest["files"])
    assert any(path.endswith("logs/stream_probe.running.json") for path in manifest["files"])
    assert any(path.endswith("logs/stream_probe.finished.json") for path in manifest["files"])
    assert any(path.endswith("logs/stream_probe.stdout.txt") for path in manifest["files"])
    assert any(path.endswith("logs/stream_probe.stderr.txt") for path in manifest["files"])


def test_daemon_accepts_odds_capture_only_command_defaults():
    args = daemon.parse_args(["run-odds-capture-once"])

    assert args.command == "run-odds-capture-once"
    assert args.timeout_seconds == daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMEOUT_SECONDS
    assert args.refresh_limit == daemon.DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT
    assert args.odds_capture_refresh_limit == daemon.DEFAULT_ODDS_CAPTURE_ONLY_REFRESH_LIMIT
    assert args.state_path == daemon.DEFAULT_ODDS_CAPTURE_ONLY_STATE_PATH
    assert args.require_safe_refresh_metadata is True


def test_daemon_can_explicitly_allow_incomplete_refresh_metadata():
    run_args = daemon.parse_args(["run-once", "--allow-incomplete-refresh-metadata"])
    odds_args = daemon.parse_args(
        ["run-odds-capture-once", "--allow-incomplete-refresh-metadata"]
    )

    assert run_args.require_safe_refresh_metadata is False
    assert odds_args.require_safe_refresh_metadata is False


def test_odds_capture_only_ready_accepts_partial_parent_when_odds_appended():
    final_status = daemon.classify_odds_capture_only_final_status(
        step={"returncode": 0},
        autopilot_result={"final_verdict": "PARTIAL_AUTOMATION_READY"},
        odds_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "inserted_live_odds_rows": 15,
        },
        refresh_report={"status": "SUCCESS"},
    )

    assert final_status == "ODDS_CAPTURE_ONLY_READY"


def test_odds_capture_only_ready_accepts_metadata_incomplete_when_windows_handled():
    final_status = daemon.classify_odds_capture_only_final_status(
        step={"returncode": 0},
        autopilot_result={"final_verdict": "PARTIAL_AUTOMATION_READY"},
        odds_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "ready_count": 5,
            "status_counts": {"SKIPPED_ALREADY_CAPTURED": 5, "SKIPPED_NOT_READY": 3},
        },
        refresh_report={"status": "METADATA_COVERAGE_INCOMPLETE"},
    )

    assert final_status == "ODDS_CAPTURE_ONLY_READY"


def test_odds_capture_only_ready_rejects_metadata_incomplete_without_usable_windows():
    final_status = daemon.classify_odds_capture_only_final_status(
        step={"returncode": 0},
        autopilot_result={"final_verdict": "PARTIAL_AUTOMATION_READY"},
        odds_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "ready_count": 0,
            "status_counts": {"SKIPPED_NOT_READY": 3},
        },
        refresh_report={"status": "METADATA_COVERAGE_INCOMPLETE"},
    )

    assert final_status == "ODDS_CAPTURE_ONLY_FAILED"


def test_odds_capture_only_autopilot_command_is_narrow_and_append_only():
    command = daemon.odds_capture_only_autopilot_command(
        run_id="odds_only_autopilot",
        evidence_root=Path("/evidence"),
        lock_path=Path("/runtime/shared.lock"),
        current_time="2026-06-12T09:48:00+10:00",
        db_path=Path("/data/greyhound_racing_data.db"),
        days_ahead=1,
        refresh_limit=8,
        odds_capture_min_minutes=0.0,
        odds_capture_max_minutes=60.0,
        odds_capture_refresh_limit=8,
        timeout_seconds=600,
        state_path=Path("/runtime/odds_capture_state.json"),
        forward_corpus_root=Path("/evidence/forward-corpus"),
    )

    assert "scripts/shadow_autopilot_v1.py" in command[1]
    assert "--enable-autonomous-odds-capture" in command
    assert "--execute-autonomous-odds-capture" in command
    assert "--allow-auto-scrape-odds" in command
    assert "--skip-primary-refresh" in command
    assert "--skip-shadow-run" in command
    assert "--skip-odds-snapshot" in command
    assert "--skip-result-join" in command
    assert "--skip-aggregate" in command
    assert "--skip-status" in command
    assert "--skip-unified-dataset" in command
    assert "--require-safe-refresh-metadata" in command
    assert command[command.index("--collector-lock-path") + 1] == (
        "/runtime/shared.lock"
    )
    assert command[command.index("--current-race-index-state-path") + 1] == (
        "/runtime/odds_capture_state.json"
    )
    assert command[command.index("--forward-corpus-root") + 1] == (
        "/evidence/forward-corpus"
    )
    assert "--enable-autonomous-result-capture" not in command

    permissive_command = daemon.odds_capture_only_autopilot_command(
        run_id="odds_only_autopilot",
        evidence_root=Path("/evidence"),
        lock_path=Path("/runtime/shared.lock"),
        current_time="2026-06-12T09:48:00+10:00",
        db_path=Path("/data/greyhound_racing_data.db"),
        days_ahead=1,
        refresh_limit=8,
        odds_capture_min_minutes=0.0,
        odds_capture_max_minutes=60.0,
        odds_capture_refresh_limit=8,
        timeout_seconds=600,
        require_safe_refresh_metadata=False,
    )
    assert "--require-safe-refresh-metadata" not in permissive_command


def test_gated_challenger_commands_use_report_only_packet_builders():
    evidence_root = Path("/evidence")
    runner_matrix_csv = Path("/evidence/rolling/market_residual_runner_matrix.csv")
    race_predictions_csv = Path("/evidence/residual/cross_validated_race_predictions.csv")

    pre_race_command = daemon.pre_race_gated_challenger_command(
        runner_matrix_csv=runner_matrix_csv,
        output_dir=Path("/evidence/pre_race_gated_challenger_run"),
        evidence_root=evidence_root,
    )
    rank_first_command = daemon.pre_race_gated_challenger_command(
        runner_matrix_csv=runner_matrix_csv,
        output_dir=Path("/evidence/pre_race_rank_first_hypothesis_review_run"),
        evidence_root=evidence_root,
        rank_first_hypotheses_json=Path(
            "/evidence/regime/next_hypotheses.json"
        ),
    )
    time_split_command = daemon.time_split_gated_challenger_command(
        runner_matrix_csv=runner_matrix_csv,
        output_dir=Path("/evidence/time_split_gated_challenger_run"),
    )
    residual_command = daemon.market_residual_challenger_command(
        runner_matrix_csv=runner_matrix_csv,
        output_dir=Path("/evidence/market_residual_challenger_run"),
    )
    regime_command = daemon.market_residual_regime_audit_command(
        runner_matrix_csv=runner_matrix_csv,
        race_predictions_csv=race_predictions_csv,
        output_dir=Path("/evidence/market_residual_regime_audit_run"),
    )
    promotion_distance_command = daemon.promotion_distance_report_command(
        rolling_report=Path("/evidence/rolling/rolling_model_comparison_report.json"),
        pre_race_gated_report=Path(
            "/evidence/pre_race/pre_race_gated_challenger_report.json"
        ),
        high_accuracy_gate=Path("/evidence/high_accuracy/promotion_pr_gate.json"),
        output_dir=Path("/evidence/promotion_distance_report_run"),
        evidence_root=evidence_root,
    )
    watchlist_command = daemon.rank_first_hypothesis_watchlist_command(
        evidence_root=evidence_root,
        output_dir=Path("/evidence/rank_first_hypothesis_watchlist_run"),
    )

    assert "scripts/build_pre_race_gated_challenger_packet.py" in pre_race_command[1]
    assert "--runner-matrix-csv" in pre_race_command
    assert str(runner_matrix_csv) in pre_race_command
    assert pre_race_command[pre_race_command.index("--evidence-root") + 1] == str(
        evidence_root
    )
    assert "--output-dir" in pre_race_command
    assert "scripts/build_pre_race_gated_challenger_packet.py" in rank_first_command[1]
    assert rank_first_command[rank_first_command.index("--evidence-root") + 1] == str(
        evidence_root
    )
    assert "--rank-first-hypotheses-json" in rank_first_command
    assert "/evidence/regime/next_hypotheses.json" in rank_first_command
    assert "scripts/build_time_split_gated_challenger_packet.py" in time_split_command[1]
    assert "--runner-matrix-csv" in time_split_command
    assert str(runner_matrix_csv) in time_split_command
    assert "--output-dir" in time_split_command
    assert "scripts/build_market_residual_challenger_packet.py" in residual_command[1]
    assert "--runner-matrix-csv" in residual_command
    assert str(runner_matrix_csv) in residual_command
    assert "scripts/build_market_residual_regime_audit.py" in regime_command[1]
    assert "--race-predictions-csv" in regime_command
    assert str(race_predictions_csv) in regime_command
    assert "scripts/build_promotion_distance_report.py" in promotion_distance_command[1]
    assert "--rolling-report" in promotion_distance_command
    assert promotion_distance_command[
        promotion_distance_command.index("--evidence-root") + 1
    ] == str(evidence_root)
    assert "--pre-race-gated-report" in promotion_distance_command
    assert "--high-accuracy-gate" in promotion_distance_command
    assert "scripts/build_rank_first_hypothesis_watchlist.py" in watchlist_command[1]
    assert "--evidence-root" in watchlist_command
    assert "/evidence" in watchlist_command


def test_rank_first_hypothesis_watchlist_status_preserves_longitudinal_blockers():
    status = daemon.rank_first_hypothesis_watchlist_status_from_report(
        generated_at=datetime(2026, 6, 13, 0, 0, tzinfo=timezone.utc),
        packet_dir=Path("/tmp/watchlist"),
        report_path=Path("/tmp/watchlist/rank_first_hypothesis_watchlist_report.json"),
        attempted=True,
        returncode=0,
        packet_report={
            "final_status": "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY",
            "packet_count": 3,
            "evaluation_count": 15,
            "candidate_count": 5,
            "directional_ready_candidate_count": 0,
            "minimum_triggered_races_for_directional_read": 10,
            "minimum_distinct_samples_for_directional_read": 2,
            "best_candidate": {
                "candidate_key": (
                    "rank_first_hypothesis_venue_eq_gee__raw_stage2_uncalibrated"
                ),
                "status": "RANK_FIRST_HYPOTHESIS_WAITING_FOR_FRESH_SAMPLE",
                "distinct_sample_signature_count": 1,
                "latest_gate_triggered_race_count": 5,
                "latest_top1_delta_vs_market": 0.016129032258064557,
                "latest_logloss_delta_vs_market": -0.013124313154672329,
                "blockers": [
                    "needs_distinct_future_sample",
                    "triggered_race_count_below_directional_floor",
                ],
            },
            "blockers": ["no_directional_ready_rank_first_hypotheses"],
            "no_write_guarantees": {"production_promotion": False},
        },
    )

    assert status["status"] == "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY"
    assert status["attempted"] is True
    assert status["packet_count"] == 3
    assert status["evaluation_count"] == 15
    assert status["candidate_count"] == 5
    assert status["directional_ready_candidate_count"] == 0
    assert status["best_candidate_key"] == (
        "rank_first_hypothesis_venue_eq_gee__raw_stage2_uncalibrated"
    )
    assert status["best_candidate_status"] == (
        "RANK_FIRST_HYPOTHESIS_WAITING_FOR_FRESH_SAMPLE"
    )
    assert status["best_candidate_distinct_sample_count"] == 1
    assert status["best_candidate_triggered_race_count"] == 5
    assert status["best_candidate_top1_delta_vs_market"] == 0.016129032258064557
    assert status["best_candidate_logloss_delta_vs_market"] == -0.013124313154672329
    assert status["best_candidate_blockers"] == [
        "needs_distinct_future_sample",
        "triggered_race_count_below_directional_floor",
    ]
    assert status["blockers"] == ["no_directional_ready_rank_first_hypotheses"]
    assert status["no_write_guarantees"]["production_promotion"] is False


def test_gated_challenger_status_preserves_gate_blockers_and_metrics():
    generated_at = datetime(2026, 6, 12, 4, 30, tzinfo=timezone.utc)
    status = daemon.gated_challenger_status_from_report(
        generated_at=generated_at,
        packet_kind="time_split_gated_challenger",
        packet_dir=Path("/tmp/time_split_gated_challenger_run"),
        report_path=Path("/tmp/time_split_gated_challenger_run/time_split_gated_challenger_report.json"),
        attempted=True,
        returncode=0,
        packet_report={
            "final_status": "TIME_SPLIT_GATED_CHALLENGER_REVIEW_READY",
            "matrix_row_count": 746,
            "accepted_race_count": 108,
            "minimum_races_for_review": 100,
            "evaluated_split_count": 2,
            "market_metrics_on_time_split_test_races": {
                "top1": 0.4827586206896552,
                "top3": 0.8045977011494253,
                "mean_winner_rank": 2.2873563218390807,
                "brier": 0.6891128250501608,
                "logloss": 1.5024972840300603,
            },
            "time_split_metrics": {
                "race_count": 87,
                "gate_triggered_test_race_count": 57,
                "top1": 0.47126436781609193,
                "top3": 0.8160919540229885,
                "mean_winner_rank": 2.2988505747126435,
                "brier": 0.6941089753554783,
                "logloss": 1.5266169489365502,
            },
            "promotion_gate": {
                "promotion_ready": False,
                "would_clear_metric_gates": False,
                "candidate_minus_market": {"top1": -0.01149425287356326},
                "blockers": [
                    "report_only_time_split_gated_challenger_not_promotion_eligible",
                    "top1_not_above_market",
                ],
            },
            "blockers": [],
            "no_write_guarantees": {"production_promotion": False},
        },
    )

    assert status["status"] == "TIME_SPLIT_GATED_CHALLENGER_REVIEW_READY"
    assert status["packet_kind"] == "time_split_gated_challenger"
    assert status["attempted"] is True
    assert status["returncode"] == 0
    assert status["accepted_race_count"] == 108
    assert status["time_split_test_race_count"] == 87
    assert status["gate_triggered_test_race_count"] == 57
    assert status["market_top1"] == 0.4827586206896552
    assert status["challenger_top1"] == 0.47126436781609193
    assert status["promotion_ready"] is False
    assert status["would_clear_metric_gates"] is False
    assert status["candidate_minus_market"]["top1"] == -0.01149425287356326
    assert status["promotion_blockers"] == [
        "report_only_time_split_gated_challenger_not_promotion_eligible",
        "top1_not_above_market",
    ]
    assert status["no_write_guarantees"]["production_promotion"] is False


def test_gated_challenger_status_reports_explicit_skip_reason():
    status = daemon.gated_challenger_status_from_report(
        generated_at=datetime(2026, 6, 12, 4, 30, tzinfo=timezone.utc),
        packet_kind="pre_race_gated_challenger",
        packet_dir=None,
        report_path=None,
        packet_report=None,
        skipped_reason="rejoin_market_residual_runner_matrix_missing",
    )

    assert status["status"] == "SKIPPED"
    assert status["attempted"] is False
    assert status["skipped_reason"] == "rejoin_market_residual_runner_matrix_missing"
    assert status["promotion_ready"] is False


def test_residual_regime_audit_status_preserves_promotion_blockers():
    status = daemon.residual_regime_audit_status_from_report(
        generated_at=datetime(2026, 6, 12, 4, 30, tzinfo=timezone.utc),
        packet_dir=Path("/tmp/market_residual_regime_audit_run"),
        report_path=Path(
            "/tmp/market_residual_regime_audit_run/market_residual_regime_audit_report.json"
        ),
        attempted=True,
        returncode=0,
        packet_report={
            "final_status": "MARKET_RESIDUAL_REGIME_AUDIT_READY",
            "matrix_row_count": 930,
            "accepted_race_count": 135,
            "minimum_races_for_review": 100,
            "regime_summary_count": 44,
            "rank_first_hypothesis_status": "NO_PRE_RACE_RANK_FIRST_EDGE_FOUND",
            "rank_first_hypothesis_blockers": [
                "no_pre_race_usable_positive_top1_delta"
            ],
            "pre_race_rank_first_help_regime_count": 0,
            "pre_race_logloss_only_help_regime_count": 6,
            "next_hypotheses_json": (
                "artifacts/full_evidence_orchestration_20260525/"
                "market_residual_regime_audit_run/next_hypotheses.json"
            ),
            "promotion_ready": False,
            "promotion_blockers": [
                "report_only_residual_regime_audit_not_promotion_eligible",
                "requires_new_out_of_sample_packet_for_any_pre_race_gate",
            ],
            "overall_metrics": {"market_top1_rate": 0.4148148148148148},
            "no_write_guarantees": {"production_promotion": False},
        },
    )

    assert status["status"] == "MARKET_RESIDUAL_REGIME_AUDIT_READY"
    assert status["attempted"] is True
    assert status["matrix_row_count"] == 930
    assert status["accepted_race_count"] == 135
    assert status["promotion_ready"] is False
    assert status["promotion_blockers"] == [
        "report_only_residual_regime_audit_not_promotion_eligible",
        "requires_new_out_of_sample_packet_for_any_pre_race_gate",
    ]
    assert status["rank_first_hypothesis_status"] == (
        "NO_PRE_RACE_RANK_FIRST_EDGE_FOUND"
    )
    assert status["rank_first_hypothesis_blockers"] == [
        "no_pre_race_usable_positive_top1_delta"
    ]
    assert status["pre_race_rank_first_help_regime_count"] == 0
    assert status["pre_race_logloss_only_help_regime_count"] == 6
    assert status["next_hypotheses_json"].endswith("next_hypotheses.json")
    assert status["overall_metrics"]["market_top1_rate"] == 0.4148148148148148
    assert status["no_write_guarantees"]["production_promotion"] is False


def test_promotion_distance_status_surfaces_distance_to_gate():
    status = daemon.promotion_distance_status_from_report(
        generated_at=datetime(2026, 6, 12, 4, 30, tzinfo=timezone.utc),
        packet_dir=Path("/tmp/promotion_distance_report_run"),
        report_path=Path("/tmp/promotion_distance_report_run/promotion_distance_report.json"),
        attempted=True,
        returncode=0,
        packet_report={
            "final_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_ready": False,
            "blockers": [
                "no_candidate_passed_rank_first_accuracy_gate",
                "best_non_market_top1_margin_below_target",
            ],
            "rolling_sample": {
                "sample_race_count": 135,
                "sample_runner_rows": 930,
                "source_exclusion_reason_counts": {
                    "official_result_missing": 32,
                },
                "source_odds_exclusion_reason_counts": {
                    "strict_prejump_odds_missing": 6,
                },
                "source_official_result_evidence_db_missing_race_ids": [
                    "Race 7 - TAREE - 2026-06-13",
                    "Race 8 - TAREE - 2026-06-13",
                ],
                "source_official_result_evidence_db_requested_race_count": 7,
                "source_official_result_evidence_db_races_with_rows": [
                    "Race 5 - TAREE - 2026-06-13",
                ],
                "source_official_result_runner_paths": [
                    "artifacts/full_evidence_orchestration_20260525/"
                    "autonomous_official_result_capture_x/official_result_runners.jsonl",
                ],
            },
            "official_result_coverage": {
                "source": (
                    "rolling_sample.source_official_result_evidence_db_missing_race_ids"
                ),
                "requested_race_count": 7,
                "requested_race_count_source": (
                    "deduped_requested_or_inferred_race_ids"
                ),
                "legacy_requested_race_count_without_ids": 4125,
                "races_with_rows_count": 1,
                "missing_race_count": 2,
                "missing_exclusion_count": 32,
                "missing_race_ids": [
                    "Race 7 - TAREE - 2026-06-13",
                    "Race 8 - TAREE - 2026-06-13",
                ],
                "races_with_rows": [
                    "Race 5 - TAREE - 2026-06-13",
                ],
                "runner_path_count": 1,
                "runner_paths_source_field": (
                    "rolling_sample.source_official_result_runner_paths"
                ),
            },
            "market_benchmark": {
                "target_top1_margin_vs_market": 0.02,
                "best_non_market_top1_margin_gap": 0.02,
            },
            "predeclared_residual_candidate": {
                "status": "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING",
                "triggered_race_count": 2,
                "minimum_triggered_races_for_directional_read": 10,
                "directional_read_ready": False,
                "candidate_minus_market": {"top1": 0.0},
            },
            "no_write_guarantees": {"production_promotion": False},
        },
    )

    assert status["status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert status["promotion_ready"] is False
    assert status["blockers"] == [
        "no_candidate_passed_rank_first_accuracy_gate",
        "best_non_market_top1_margin_below_target",
    ]
    assert status["sample_race_count"] == 135
    assert status["sample_runner_rows"] == 930
    assert status["source_exclusion_reason_counts"] == {
        "official_result_missing": 32,
    }
    assert status["source_odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 6,
    }
    assert status[
        "source_official_result_evidence_db_missing_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert status["source_official_result_evidence_db_requested_race_count"] == 7
    assert status["source_official_result_evidence_db_races_with_rows"] == [
        "Race 5 - TAREE - 2026-06-13",
    ]
    assert status["source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_x/official_result_runners.jsonl",
    ]
    assert status["official_result_coverage_requested_race_count"] == 7
    assert (
        status["official_result_coverage_requested_race_count_source"]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        status["official_result_coverage_legacy_requested_race_count_without_ids"]
        == 4125
    )
    assert status["official_result_coverage_races_with_rows_count"] == 1
    assert status["official_result_coverage_missing_race_count"] == 2
    assert status["official_result_coverage_missing_exclusion_count"] == 32
    assert status["official_result_runner_path_count"] == 1
    assert status["official_result_runner_paths_source_field"] == (
        "rolling_sample.source_official_result_runner_paths"
    )
    assert status["official_result_coverage"]["missing_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert status["target_top1_margin_vs_market"] == 0.02
    assert status["best_non_market_top1_margin_gap"] == 0.02
    assert status["predeclared_residual_triggered_race_count"] == 2
    assert (
        status[
            "predeclared_residual_minimum_triggered_races_for_directional_read"
        ]
        == 10
    )
    assert status["predeclared_residual_directional_read_ready"] is False
    assert status["predeclared_residual_candidate_minus_market"] == {"top1": 0.0}
    assert status["no_write_guarantees"]["production_promotion"] is False


def test_gated_challenger_status_surfaces_predeclared_residual_candidate():
    status = daemon.gated_challenger_status_from_report(
        generated_at=datetime(2026, 6, 12, 4, 30, tzinfo=timezone.utc),
        packet_kind="pre_race_gated_challenger",
        packet_dir=Path("/tmp/pre_race_gated_challenger_run"),
        report_path=Path("/tmp/pre_race_gated_challenger_run/pre_race_gated_challenger_report.json"),
        attempted=True,
        returncode=0,
        packet_report={
            "final_status": "PRE_RACE_GATED_CHALLENGER_REVIEW_READY",
            "matrix_row_count": 934,
            "accepted_race_count": 137,
            "minimum_races_for_review": 100,
            "evaluated_fold_count": 5,
            "market_metrics": {"top1": 0.45255474452554745},
            "challenger_metrics": {
                "race_count": 137,
                "gate_triggered_test_race_count": 4,
                "top1": 0.44525547445255476,
            },
            "predeclared_residual_candidate": {
                "candidate_key": "market_favourite_gt_4_0__raw_stage2_market_blend_75",
                "status": "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING",
                "triggered_race_count": 3,
                "minimum_triggered_races_for_directional_read": 10,
                "directional_read_ready": False,
                "candidate_minus_market": {"logloss": -0.0015},
                "blockers": [
                    "predeclared_residual_candidate_report_only",
                    "triggered_race_count_below_directional_floor",
                ],
            },
            "rank_first_hypothesis_gate_review": {
                "status": "RANK_FIRST_HYPOTHESIS_REVIEW_READY",
                "candidate_count": 5,
                "evaluated_candidate_count": 5,
                "best_candidate_key": (
                    "rank_first_hypothesis_runner_count_eq_4__raw_stage2_uncalibrated"
                ),
                "minimum_triggered_races_for_directional_read": 10,
                "directional_read_ready": False,
                "best_candidate": {"gate_triggered_race_count": 13},
                "best_candidate_minus_market": {"top1": 0.008064516129032251},
                "blockers": ["rank_first_hypothesis_review_report_only"],
            },
            "promotion_gate": {
                "promotion_ready": False,
                "blockers": [
                    "report_only_pre_race_gated_challenger_not_promotion_eligible",
                ],
            },
        },
    )

    assert status["predeclared_residual_candidate_key"] == (
        "market_favourite_gt_4_0__raw_stage2_market_blend_75"
    )
    assert status["predeclared_residual_candidate_status"] == (
        "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING"
    )
    assert status["predeclared_residual_triggered_race_count"] == 3
    assert (
        status[
            "predeclared_residual_minimum_triggered_races_for_directional_read"
        ]
        == 10
    )
    assert status["predeclared_residual_directional_read_ready"] is False
    assert status["predeclared_residual_candidate_minus_market"] == {
        "logloss": -0.0015
    }
    assert status["predeclared_residual_blockers"] == [
        "predeclared_residual_candidate_report_only",
        "triggered_race_count_below_directional_floor",
    ]
    assert status["rank_first_hypothesis_review_status"] == (
        "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
    )
    assert status["rank_first_hypothesis_candidate_count"] == 5
    assert status["rank_first_hypothesis_evaluated_candidate_count"] == 5
    assert status["rank_first_hypothesis_best_candidate_key"] == (
        "rank_first_hypothesis_runner_count_eq_4__raw_stage2_uncalibrated"
    )
    assert status["rank_first_hypothesis_best_triggered_race_count"] == 13
    assert (
        status["rank_first_hypothesis_minimum_triggered_races_for_directional_read"]
        == 10
    )
    assert status["rank_first_hypothesis_directional_read_ready"] is False
    assert status["rank_first_hypothesis_best_candidate_minus_market"] == {
        "top1": 0.008064516129032251
    }
    assert status["rank_first_hypothesis_blockers"] == [
        "rank_first_hypothesis_review_report_only"
    ]


def test_daemon_accepts_skip_unified_dataset_passthrough_flag():
    args = daemon.parse_args(
        [
            "run-once",
            "--skip-unified-dataset",
        ]
    )

    assert args.skip_unified_dataset is True


def test_daemon_accepts_shadow_model_passthrough_flag():
    args = daemon.parse_args(
        [
            "run-once",
            "--shadow-model",
            "/models/stage2/shadow_randomforest_model.joblib",
        ]
    )

    assert args.shadow_model == Path("/models/stage2/shadow_randomforest_model.joblib")
    assert daemon.shadow_model_cli_args(args.shadow_model) == [
        "--shadow-model",
        "/models/stage2/shadow_randomforest_model.joblib",
    ]


def test_autopilot_cycle_timeout_exceeds_child_step_timeout():
    assert daemon.autopilot_cycle_timeout_seconds(840) == 1680
    assert daemon.autopilot_cycle_timeout_seconds(120) == 420


def _write_prediction_fixture(daily_dir: Path, model_file: Path) -> None:
    score_dir = daily_dir / "shadow_score_live"
    score_dir.mkdir(parents=True)
    daemon.write_json(
        daily_dir / "shadow_manifest.json",
        {
            "schema_version": "daily_shadow_manifest_v1",
            "final_status": "FORWARD_SHADOW_RUN_COMPLETE",
            "input_summary": {"eligible_count": 1},
            "prediction_rows": 2,
            "race_count": 1,
            "shadow_model": str(model_file),
            "shadow_training_allowed": False,
            "calibration_method": "power_gamma_2.4",
            "tgr_enabled": False,
            "registry_mutation": False,
            "production_prediction_overwrite": False,
            "db_writes": False,
            "label_writes": False,
            "score_live_manifest": {
                "model_source": str(model_file),
                "model_version": "shadow_loaded_test_model",
                "calibration_method": "power_gamma_2.4",
                "active_feature_count": 2,
                "schema_feature_count": 3,
                "inactive_features_due_to_train_all_missing": [
                    "same_distance_same_grade_best_time"
                ],
                "tgr_enabled": False,
                "registry_mutation": False,
                "production_prediction_write": False,
            },
        },
    )
    daemon.write_json(
        daily_dir / "shadow_score_live_command.json",
        {
            "schema_version": "daily_shadow_score_live_command_v1",
            "command": [
                "python",
                "scripts/run_shadow_non_tgr_rf_evaluation.py",
                "score-live",
                "--model",
                str(model_file),
            ],
            "cwd": str(daily_dir.parent),
        },
    )
    daemon.write_json(
        score_dir / "active_feature_policy_report.json",
        {
            "schema_version": "loaded_shadow_model_feature_policy_v1",
            "active_feature_count": 2,
            "schema_feature_count": 3,
            "inactive_features_due_to_train_all_missing": [
                "same_distance_same_grade_best_time"
            ],
        },
    )
    daemon.write_json(
        score_dir / "shadow_candidate_definition.json",
        {
            "schema_version": "shadow_candidate_definition_v1",
            "model_family": "RandomForest",
            "tgr_enabled": False,
            "calibration": {
                "method_key": "power_gamma_2.4",
                "formula": "p_i_cal = p_i^2.4 / sum_j(p_j^2.4)",
            },
        },
    )
    daemon.write_jsonl(
        daily_dir / "shadow_predictions.jsonl",
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-08",
                "dog_name": "Fast One",
                "box": 1,
                "shadow_rf_uncalibrated_probability": 0.6,
                "shadow_rf_calibrated_probability": 0.7,
                "predicted_rank": 1,
                "calibration_method": "power_gamma_2.4",
                "model_source": str(model_file),
                "model_version": "shadow_loaded_test_model",
                "tgr_enabled": False,
                "output_mode": "shadow_only",
            },
            {
                "race_id": "Race 1 - TEST - 2026-06-08",
                "dog_name": "Slow Two",
                "box": 2,
                "shadow_rf_uncalibrated_probability": 0.4,
                "shadow_rf_calibrated_probability": 0.3,
                "predicted_rank": 2,
                "calibration_method": "power_gamma_2.4",
                "model_source": str(model_file),
                "model_version": "shadow_loaded_test_model",
                "tgr_enabled": False,
                "output_mode": "shadow_only",
            },
        ],
    )
    daemon.write_json(
        score_dir / "shadow_feature_rows.json",
        [
            {
                "race_id": "Race 1 - TEST - 2026-06-08",
                "dog_name": "Fast One",
                "box_number": 1,
                "speed_rating": 12.5,
            },
            {
                "race_id": "Race 1 - TEST - 2026-06-08",
                "dog_name": "Slow Two",
                "box_number": 2,
                "speed_rating": 9.1,
            },
        ],
    )
    daemon.write_json(
        daily_dir / "probability_sum_report.json",
        {
            "schema_version": "daily_shadow_probability_sum_report_v1",
            "status": "PASS",
            "prediction_rows": 2,
            "race_count": 1,
            "max_abs_error": 0.0,
            "per_race": [
                {
                    "race_id": "Race 1 - TEST - 2026-06-08",
                    "runner_count": 2,
                    "sum": 1.0,
                    "abs_error": 0.0,
                }
            ],
        },
    )


def _write_model_metadata(model_file: Path, *, inactive_features=None) -> None:
    if inactive_features is None:
        inactive_features = ["same_distance_same_grade_best_time"]
    inactive_features = list(inactive_features)
    model_file.write_text("not a real joblib model", encoding="utf-8")
    daemon.write_json(
        model_file.parent / "shadow_model_metadata.json",
        {
            "schema_version": "shadow_model_metadata_v1",
            "artifact_path": str(model_file),
            "artifact_sha256": daemon.sha256_file(model_file),
            "model_family": "RandomForest",
            "calibration_method": "power_gamma_2.4",
            "feature_columns": [
                "box_number",
                "speed_rating",
                "same_distance_same_grade_best_time",
            ],
            "inactive_features_due_to_train_all_missing": inactive_features,
            "feature_count": 3,
        },
    )
    daemon.write_json(
        model_file.parent / "shadow_training_report.json",
        {
            "schema_version": "shadow_training_report_v1",
            "status": "PASS",
            "model_family": "RandomForest",
            "train_races": 10,
            "train_rows": 80,
            "holdout_races": 2,
            "holdout_rows": 16,
            "schema_feature_count": 3,
            "inactive_features_due_to_train_all_missing": inactive_features,
        },
    )


def test_shadow_observability_reports_prediction_provenance_and_feature_audit(tmp_path):
    daily_dir = tmp_path / "daily_race_ingest_shadow_test"
    model_file = tmp_path / "model" / "shadow_randomforest_model.joblib"
    model_file.parent.mkdir()
    daily_dir.mkdir()
    _write_model_metadata(model_file)
    _write_prediction_fixture(daily_dir, model_file)

    report = daemon.build_shadow_observability(
        generated_at=daemon.datetime.fromisoformat("2026-06-08T20:30:00+10:00"),
        run_id="test_run",
        daily_shadow_run_dir=daily_dir,
        daily_manifest=daemon.load_json(daily_dir / "shadow_manifest.json"),
        dashboard={"safe_joined_races": 82, "pending_races": 10, "unsafe_matches": 0},
        readiness={"decision": "NEED_MORE_RESULTS"},
        steps=[],
        protected_validation={"protected_paths_unchanged": True},
    )

    assert report["status"]["status"] == "OBSERVABILITY_READY"
    assert report["status"]["safety_flags"]["training_disabled"] is True
    assert report["status"]["safety_flags"]["tgr_disabled"] is True
    assert report["model_card"]["model_sha256"] == daemon.sha256_file(model_file)
    assert report["model_card"]["feature_policy"]["quarantined_features_active"] == []
    assert report["provenance"]["calibration_method"] == "power_gamma_2.4"
    race = report["race_explanations"]["races"][0]
    assert race["top_pick"]["dog_name"] == "Fast One"
    assert race["runner_explanations"][0]["feature_audit"]["feature_value_status"] == "AVAILABLE"
    assert race["runner_explanations"][0]["feature_audit"]["active_feature_missing_count"] == 0


def test_shadow_observability_handles_no_predictions(tmp_path):
    daily_dir = tmp_path / "daily_race_ingest_shadow_waiting"
    daily_dir.mkdir()
    model_file = tmp_path / "model" / "shadow_randomforest_model.joblib"
    model_file.parent.mkdir()
    _write_model_metadata(model_file)
    daemon.write_jsonl(daily_dir / "shadow_predictions.jsonl", [])
    daemon.write_json(
        daily_dir / "shadow_manifest.json",
        {
            "schema_version": "daily_shadow_manifest_v1",
            "final_status": "WAITING_FOR_UPCOMING_RACES",
            "input_summary": {"eligible_count": 0},
            "prediction_rows": 0,
            "race_count": 0,
            "shadow_model": str(model_file),
            "shadow_training_allowed": False,
            "calibration_method": "power_gamma_2.4",
            "tgr_enabled": False,
        },
    )
    daemon.write_json(
        daily_dir / "probability_sum_report.json",
        {"schema_version": "daily_shadow_probability_sum_report_v1", "status": "PASS", "per_race": []},
    )

    report = daemon.build_shadow_observability(
        generated_at=daemon.datetime.fromisoformat("2026-06-08T20:30:00+10:00"),
        run_id="test_run",
        daily_shadow_run_dir=daily_dir,
        daily_manifest=daemon.load_json(daily_dir / "shadow_manifest.json"),
        dashboard={"safe_joined_races": 82, "pending_races": 10, "unsafe_matches": 0},
        readiness={"decision": "NEED_MORE_RESULTS"},
        steps=[],
        protected_validation={"protected_paths_unchanged": True},
    )

    assert report["status"]["status"] == "NO_PREDICTIONS"
    assert report["status"]["no_prediction_reason"] == "no_eligible_current_or_future_races"
    assert report["race_explanations"]["race_count"] == 0
    assert "No scored races" in report["markdown"]


def test_shadow_observability_fails_closed_when_quarantined_feature_active(tmp_path):
    daily_dir = tmp_path / "daily_race_ingest_shadow_quarantine_fail"
    model_file = tmp_path / "model" / "shadow_randomforest_model.joblib"
    model_file.parent.mkdir()
    daily_dir.mkdir()
    _write_model_metadata(model_file, inactive_features=[])
    _write_prediction_fixture(daily_dir, model_file)
    daemon.write_json(
        daily_dir / "shadow_score_live" / "active_feature_policy_report.json",
        {
            "schema_version": "loaded_shadow_model_feature_policy_v1",
            "active_feature_count": 3,
            "schema_feature_count": 3,
            "inactive_features_due_to_train_all_missing": [],
        },
    )

    report = daemon.build_shadow_observability(
        generated_at=daemon.datetime.fromisoformat("2026-06-08T20:30:00+10:00"),
        run_id="test_run",
        daily_shadow_run_dir=daily_dir,
        daily_manifest=daemon.load_json(daily_dir / "shadow_manifest.json"),
        dashboard={"safe_joined_races": 82, "pending_races": 10, "unsafe_matches": 0},
        readiness={"decision": "NEED_MORE_RESULTS"},
        steps=[],
        protected_validation={"protected_paths_unchanged": True},
    )

    assert report["status"]["status"] == "FAIL_CLOSED_FEATURE_POLICY_VIOLATION"
    assert "same_distance_same_grade_best_time" in report["status"]["safety_flags"]["quarantined_features_active"]


def test_alert_report_flags_observability_safety_failures():
    report = daemon.build_alert_report(
        current_dashboard={"safe_joined_races": 82, "unsafe_matches": 0},
        previous_dashboard={"safe_joined_races": 82, "unsafe_matches": 0},
        automated_join_report={"results": []},
        target_joined_races=100,
        current_observability={
            "status": "PROBABILITY_SUM_FAIL",
            "model_sha256": "new",
            "score_command_text": "python scorer --train-if-missing",
            "probability_sum_status": "FAIL",
            "safety_flags": {
                "training_disabled": False,
                "score_command_trains": True,
                "tgr_disabled": False,
                "quarantined_features_active": ["same_distance_same_grade_best_time"],
            },
        },
        previous_observability={
            "model_sha256": "old",
            "score_command_text": "python scorer --model locked.joblib",
        },
    )

    rules = {alert["rule"] for alert in report["triggered_alerts"]}
    assert "model_hash_changed" in rules
    assert "score_command_changed" in rules
    assert "training_enabled" in rules
    assert "tgr_enabled" in rules
    assert "quarantined_feature_active" in rules
    assert "probability_sum_failed" in rules


def test_alert_report_includes_runner_unsafe_match_samples(tmp_path):
    join_dir = tmp_path / "forward_shadow_result_join_20260609T041048+1000_daemon_rejoin_007"
    join_dir.mkdir()
    daemon.write_json(
        join_dir / "unsafe_result_matches.json",
        {
            "schema_version": "forward_shadow_unsafe_result_matches_v1",
            "unsafe_match_count": 2,
            "unsafe_result_matches": [
                {
                    "race_id": "Race 8 - NOR - 2026-06-08",
                    "status": "UNSAFE_RESULT_MATCH_QUARANTINED",
                    "reason": ["dog_name_mismatch_after_exact_badge_stripping"],
                    "missing_predicted_boxes": [],
                    "disallowed_extra_official_boxes": [6],
                    "name_mismatches": [
                        {
                            "box": 6,
                            "predicted_name": "Fast One",
                            "official_name": "Fast Two",
                        }
                    ],
                    "prejump_runner_alignment": {
                        "canonical_runner_alignment_status": "PASS",
                        "canonical_runner_set_status": "PASS",
                        "canonical_runner_count": 8,
                        "canonical_prediction_runner_count": 8,
                    },
                },
                {
                    "race_id": "Race 9 - NOR - 2026-06-08",
                    "status": "UNSAFE_RESULT_MATCH_QUARANTINED",
                    "reason": ["winner_count_not_exactly_one"],
                    "official_runner_rows": [{"box_number": 1, "dog_name": "Clear Name"}],
                },
            ],
        },
    )

    report = daemon.build_alert_report(
        current_dashboard={"safe_joined_races": 84, "unsafe_matches": 14},
        previous_dashboard={"safe_joined_races": 84, "unsafe_matches": 14},
        automated_join_report={
            "results": [
                {
                    "join_dir": str(join_dir),
                    "metrics": {"unsafe_match_count": 2},
                }
            ]
        },
        target_joined_races=100,
    )

    runner_alert = next(
        alert
        for alert in report["triggered_alerts"]
        if alert["rule"] == "runner_set_mismatch_spike"
    )
    assert runner_alert["runner_related_unsafe_match_count"] == 1
    assert runner_alert["reason_counts"] == {
        "dog_name_mismatch_after_exact_badge_stripping": 1
    }
    assert runner_alert["samples"][0]["race_id"] == "Race 8 - NOR - 2026-06-08"
    assert runner_alert["samples"][0]["disallowed_extra_official_boxes"] == [6]
    assert runner_alert["samples"][0]["name_mismatch_count"] == 1
    assert (
        runner_alert["samples"][0]["prejump_runner_alignment"][
            "canonical_runner_set_status"
        ]
        == "PASS"
    )


def test_duplicate_lock_probe_blocks_active_lock(tmp_path):
    lock_path = tmp_path / "shadow.lock"
    output_dir = tmp_path / "packet"
    output_dir.mkdir()

    daemon.acquire_lock(
        lock_path=lock_path,
        run_id="active",
        stale_after_seconds=3600,
        output_dir=output_dir,
    )
    try:
        report = daemon.probe_duplicate_lock(
            lock_path,
            stale_after_seconds=3600,
            output_dir=output_dir,
        )
    finally:
        daemon.release_lock(lock_path, "active")

    assert report["status"] == "PASS"
    assert report["duplicate_acquire_blocked"] is True


def test_stale_lock_probe_replaces_dead_pid(tmp_path):
    output_dir = tmp_path / "packet"
    output_dir.mkdir()

    report = daemon.probe_stale_lock_cleanup(output_dir)

    assert report["status"] == "PASS"
    assert report["stale_lock_cleaned"] is True


def test_run_odds_capture_once_uses_lock_and_writes_compact_report(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_only"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    launch_dir = tmp_path / "launch"
    launch_dir.mkdir()
    lock_path = launch_dir / "runtime" / "shadow.lock"
    state_path = evidence_root / "shadow_autopilot_daemon_runtime/odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_only_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.chdir(launch_dir)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        assert name == "odds_capture_autopilot_cycle"
        assert "--skip-shadow-run" in command
        assert "--skip-primary-refresh" in command
        assert "--skip-result-join" in command
        assert "--skip-unified-dataset" in command
        assert "--enable-autonomous-odds-capture" in command
        assert "--allow-auto-scrape-odds" in command
        assert "--require-safe-refresh-metadata" in command
        assert command[command.index("--collector-lock-path") + 1] == str(lock_path)
        running_report = json.loads(
            (output_dir / "odds_capture_only_daemon_report.json").read_text(
                encoding="utf-8"
            )
        )
        running_manifest = json.loads(
            (output_dir / "output_manifest.json").read_text(encoding="utf-8")
        )
        assert running_report["final_status"] == "ODDS_CAPTURE_ONLY_RUNNING"
        assert running_report["status"] == "RUNNING"
        assert running_report["runtime_action"] == "ODDS_CAPTURE_ONLY_IN_PROGRESS"
        assert running_report["readiness_decision"] == "IN_PROGRESS"
        assert running_report["autopilot_command"] == list(command)
        assert any(
            path.endswith("odds_capture_only_daemon_report.json")
            for path in running_manifest["files"]
        )
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                "inserted_live_odds_rows": 0,
                "ready_count": 0,
                "final_status": "NO_ELIGIBLE_WINDOWS",
                "operator_status": "READY",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "generated_at": "2026-06-12T00:01:01+10:00",
                "sidecar_metadata_coverage": _write_current_index_runner_sources(
                    evidence_root,
                    "https://www.thedogs.com.au/racing/healesville/2026-06-12/1",
                ),
                "selected_count": 1,
                "selected_races": [
                    {
                        "date": "2026-06-12",
                        "jump_datetime": "2026-06-12T10:48:00+10:00",
                        "race_id": "Race 1 - HEA - 2026-06-12",
                        "race_id_aliases": [
                            "Race 1 - HEA - 2026-06-12",
                            "Race 1 - HEALESVILLE - 2026-06-12",
                        ],
                        "race_number": 1,
                        "race_time": "10:48",
                        "race_url": (
                            "https://www.thedogs.com.au/racing/healesville/"
                            "2026-06-12/1"
                        ),
                        "venue": "HEA",
                    }
                ],
                "next_preferred_window": {
                    "status": "WAITING_FOR_FUTURE_WINDOW",
                    "next_window_opens_at": "2026-06-12T09:48:00+10:00",
                    "recommended_rerun_after_local": "2026-06-12T09:48:00+10:00",
                    "next_race": {
                        "race_id": "Race 1 - HEA - 2026-06-12",
                        "date": "2026-06-12",
                        "venue": "HEA",
                        "race_number": 1,
                        "race_time": "10:48",
                        "jump_datetime": "2026-06-12T10:48:00+10:00",
                    },
                },
            },
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "AUTOPILOT_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_only",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T00:01:00+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            "runtime/shadow.lock",
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["lock_release"]["released"] is True
    assert report["autonomous_live_odds_capture_status"]["status"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    )
    assert report["autonomous_live_odds_capture_status_text"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    )
    assert report["allowed_write_scope"] == "append_only_live_odds_rows_when_validation_passes"
    assert report["current_race_index_publish"]["status"] == "PUBLISHED"
    assert not lock_path.exists()
    current_index = json.loads(
        (
            state_path.parent / "manual_prediction_current_race_index.json"
        ).read_text()
    )
    assert current_index["race_count"] == 1
    assert current_index["races"][0]["race_id"] == "Race 1 - HEA - 2026-06-12"
    written = json.loads((output_dir / "odds_capture_only_daemon_report.json").read_text())
    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    state = json.loads(state_path.read_text())
    assert written["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert written["status"] == "READY"
    assert written["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert written["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert written["inserted_live_odds_rows"] == 0
    assert written["ready_count"] == 0
    assert written["status_counts"] == {}
    assert manifest["schema_version"] == "shadow_autopilot_daemon_output_manifest_v1"
    assert any(
        path.endswith("odds_capture_only_daemon_report.json")
        for path in manifest["files"]
    )
    assert state["odds_capture_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    assert state["status"] == "READY"
    assert state["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert state["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert state["inserted_live_odds_rows"] == 0
    assert state["ready_count"] == 0
    assert state["status_counts"] == {}
    assert state["current_race_index_state"] == {
        "schema_version": "collector_current_race_index_state_v1",
        "updated_at": state["updated_at"],
        "run_id": "odds_only",
        "output_dir": str(output_dir.relative_to(evidence_root)),
        "autopilot_output_dir": str(autopilot_dir.relative_to(evidence_root)),
        "final_status": "ODDS_CAPTURE_ONLY_READY",
        "status": "READY",
    }
    assert state["autonomous_live_odds_capture_final_status"] == "NO_ELIGIBLE_WINDOWS"
    assert state["autonomous_live_odds_capture_operator_status"] == "READY"
    assert state["next_window_opens_at"] == "2026-06-12T09:48:00+10:00"
    assert state["recommended_rerun_after_local"] == "2026-06-12T09:43:00+10:00"
    assert state["source_recommended_rerun_after_local"] == "2026-06-12T09:48:00+10:00"
    assert state["resume_after_local"] == "2026-06-12T09:43:00+10:00"
    assert state["odds_capture_fixed_window_schedule"]["coverage_checked"] is False
    assert state["odds_capture_fixed_window_schedule"]["status_counts"] == {"PENDING": 4}
    assert state["odds_capture_fixed_window_schedule"]["next_pending_offset_minutes"] == 60
    assert state["odds_capture_fixed_window_schedule"]["next_pending_capture_at"] == (
        "2026-06-12T09:48:00+10:00"
    )
    assert state["odds_capture_fixed_window_schedule"][
        "odds_capture_timer_on_calendar"
    ] == daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR
    assert state["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_minute"
    ] == 48
    assert state["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_covered"
    ] is True
    assert state["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_coverage_reason"
    ] == "minute_covered_by_odds_capture_timer_on_calendar"
    assert state["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert state["next_meaningful_action_at"] == "2026-06-12T09:48:00+10:00"


def test_run_odds_capture_once_uses_capture_window_coverage_for_next_action(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_coverage"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_coverage_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        autopilot_dir.mkdir(parents=True)
        coverage_path = autopilot_dir / "autonomous_live_odds_capture_window_coverage.json"
        daemon.write_json(
            coverage_path,
            {
                "schema_version": "autonomous_live_odds_capture_window_coverage_report_v1",
                "status_counts": {"CAPTURED": 3, "PENDING": 1},
                "windows": [
                    {
                        "race_id": "Race 12 - TAREE - 2026-06-13",
                        "race_date": "2026-06-13",
                        "venue": "TAREE",
                        "race_number": 12,
                        "offset_minutes": offset,
                        "capture_mode": f"autonomous_prejump_t{offset}m",
                        "status": "CAPTURED",
                        "reason": "complete_existing_capture",
                        "existing_capture_count": 8,
                        "existing_capture_status": "COMPLETE",
                    }
                    for offset in (60, 30, 10)
                ]
                + [
                    {
                        "race_id": "Race 12 - TAREE - 2026-06-13",
                        "race_date": "2026-06-13",
                        "venue": "TAREE",
                        "race_number": 12,
                        "offset_minutes": 2,
                        "capture_mode": "autonomous_prejump_t2m",
                        "status": "PENDING",
                        "reason": "window_not_open_yet",
                        "existing_capture_count": 0,
                        "existing_capture_status": "NONE",
                    }
                ],
            },
        )
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                "inserted_live_odds_rows": 0,
                "ready_count": 1,
                "capture_window_coverage_report": str(coverage_path),
                "capture_window_coverage_status_counts": {"CAPTURED": 3, "PENDING": 1},
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "next_preferred_window": {
                    "status": "OPEN_NOW",
                    "next_window_opens_at": "2026-06-13T10:05:00+10:00",
                    "next_window_closes_at": "2026-06-13T11:05:00+10:00",
                    "next_race": {
                        "race_id": "Race 12 - TAREE - 2026-06-13",
                        "date": "2026-06-13",
                        "venue": "TAREE",
                        "race_number": "12",
                        "race_time": "11:05 AM",
                        "jump_datetime": "2026-06-13T11:05:00+10:00",
                    },
                },
            },
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "AUTOPILOT_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_coverage",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T11:00:11+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)
    state = json.loads(state_path.read_text())

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["odds_capture_fixed_window_schedule"]["coverage_checked"] is True
    assert report["odds_capture_fixed_window_schedule"]["status_counts"] == {
        "CAPTURED": 3,
        "PENDING": 1,
    }
    assert report["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert report["next_meaningful_action_at"] == "2026-06-13T11:03:00+10:00"
    assert state["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert state["next_meaningful_action_at"] == "2026-06-13T11:03:00+10:00"


def test_run_odds_capture_once_reconciles_all_ready_already_captured(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_reconciled"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_reconciled_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                "inserted_live_odds_rows": 0,
                "ready_count": 7,
                "status_counts": {"SKIPPED_ALREADY_CAPTURED": 7},
                "run_id": "direct_all_captured",
                "final_status": "NO_ELIGIBLE_WINDOWS",
                "operator_status": "READY",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "next_preferred_window": {
                    "status": "OPEN_NOW",
                    "next_window_opens_at": "2026-06-13T16:32:00+10:00",
                    "next_window_closes_at": "2026-06-13T17:32:00+10:00",
                    "next_race": {
                        "race_id": "Race 3 - DUBBO - 2026-06-13",
                        "date": "2026-06-13",
                        "venue": "DUBBO",
                        "race_number": "3",
                        "race_time": "5:32 PM",
                        "jump_datetime": "2026-06-13T17:32:00+10:00",
                    },
                },
            },
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "AUTOPILOT_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_reconciled",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T17:30:47+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)
    state = json.loads(state_path.read_text())

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["status_counts"] == {"SKIPPED_ALREADY_CAPTURED": 7}
    schedule = report["odds_capture_fixed_window_schedule"]
    assert schedule["status_counts"] == {"PASSED": 3, "DUE": 1}
    assert schedule["schedule_reconciled_with_direct_capture"] is True
    assert schedule["schedule_reconciliation_reason"] == (
        "direct_capture_all_ready_races_already_captured"
    )
    assert schedule["pre_reconciliation_next_meaningful_action"] == (
        "RUN_ODDS_CAPTURE_NOW"
    )
    assert schedule["pre_reconciliation_next_meaningful_action_offset_minutes"] == 2
    assert schedule["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert schedule["next_meaningful_action_at"] == "2026-06-13T17:30:47+10:00"
    assert schedule["next_meaningful_action_reason"] == (
        "direct_capture_all_ready_races_already_captured"
    )
    assert report["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert state["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert state["odds_capture_fixed_window_schedule"][
        "pre_reconciliation_next_meaningful_action"
    ] == "RUN_ODDS_CAPTURE_NOW"


def test_run_odds_capture_once_reconciles_all_ready_handled_after_append(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_append_reconciled"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_append_reconciled_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
                "inserted_live_odds_rows": 8,
                "ready_count": 8,
                "status_counts": {"APPENDED": 1, "SKIPPED_ALREADY_CAPTURED": 7},
                "blocked_attempt_count": 0,
                "blocked_attempts": [],
                "run_id": "direct_append_all_handled",
                "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
                "operator_status": "APPENDED",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "next_preferred_window": {
                    "status": "OPEN_NOW",
                    "next_window_opens_at": "2026-06-13T17:02:00+10:00",
                    "next_window_closes_at": "2026-06-13T18:02:00+10:00",
                    "next_race": {
                        "race_id": "Race 5 - QOT - 2026-06-13",
                        "date": "2026-06-13",
                        "venue": "QOT",
                        "race_number": "5",
                        "race_time": "6:02 PM",
                        "jump_datetime": "2026-06-13T18:02:00+10:00",
                    },
                },
            },
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "AUTOPILOT_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_append_reconciled",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T17:53:53+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)
    state = json.loads(state_path.read_text())

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["status"] == "READY"
    assert report["inserted_live_odds_rows"] == 8
    assert report["status_counts"] == {"APPENDED": 1, "SKIPPED_ALREADY_CAPTURED": 7}
    schedule = report["odds_capture_fixed_window_schedule"]
    assert schedule["status_counts"] == {"PASSED": 2, "DUE": 1, "PENDING": 1}
    assert schedule["schedule_reconciled_with_direct_capture"] is True
    assert schedule["schedule_reconciliation_reason"] == (
        "direct_capture_all_ready_races_handled_after_append"
    )
    assert schedule["pre_reconciliation_next_meaningful_action"] == (
        "RUN_ODDS_CAPTURE_NOW"
    )
    assert schedule["pre_reconciliation_next_meaningful_action_offset_minutes"] == 10
    assert schedule["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert schedule["next_meaningful_action_at"] == "2026-06-13T17:53:53+10:00"
    assert report["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert state["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert state["odds_capture_fixed_window_schedule"][
        "schedule_reconciliation_reason"
    ] == "direct_capture_all_ready_races_handled_after_append"


def test_direct_capture_reconciliation_ignores_not_ready_rows():
    already_captured_status = {
        "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
        "inserted_live_odds_rows": 0,
        "ready_count": 1,
        "status_counts": {
            "SKIPPED_ALREADY_CAPTURED": 1,
            "SKIPPED_NOT_READY": 1,
        },
    }
    appended_status = {
        "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
        "inserted_live_odds_rows": 8,
        "ready_count": 1,
        "blocked_attempt_count": 0,
        "blocked_attempts": [],
        "status_counts": {
            "APPENDED": 1,
            "SKIPPED_NOT_READY": 1,
        },
    }

    assert daemon.direct_capture_all_ready_races_already_captured(
        already_captured_status
    )
    assert daemon.direct_capture_all_ready_races_handled_after_append(appended_status)


def test_run_odds_capture_once_waits_from_recent_future_window_state(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_wait"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    state_path.parent.mkdir(parents=True)
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-12T01:00:00+10:00",
            "window_state_source_updated_at": "2026-06-12T01:00:00+10:00",
            "state_source": "full_daemon",
            "source_report_path": "artifacts/full/odds_capture_refresh_report.json",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "odds_capture_refresh_status": "SUCCESS",
            "next_window_opens_at": "2026-06-12T09:48:00+10:00",
            "recommended_rerun_after_local": "2026-06-12T09:48:00+10:00",
            "next_preferred_window": {
                "status": "WAITING_FOR_FUTURE_WINDOW",
                "next_window_opens_at": "2026-06-12T09:48:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "date": "2026-06-12",
                    "venue": "HEA",
                    "race_number": 1,
                    "race_time": "10:48",
                    "jump_datetime": "2026-06-12T10:48:00+10:00",
                },
            },
        },
    )
    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fail_lock(**kwargs):
        raise AssertionError("preflight wait should not acquire the daemon lock")

    def fail_command(**kwargs):
        raise AssertionError("preflight wait should not run the autopilot child")

    monkeypatch.setattr(daemon, "acquire_lock", fail_lock)
    monkeypatch.setattr(daemon, "run_command", fail_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_wait",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T01:10:00+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW"
    assert report["steps"] == []
    assert report["lock"] is None
    assert report["preflight_wait"]["resume_after_local"] == "2026-06-12T09:43:00+10:00"
    assert report["preflight_wait"]["recommended_rerun_after_local"] == (
        "2026-06-12T09:43:00+10:00"
    )
    assert report["preflight_wait"]["source_recommended_rerun_after_local"] == (
        "2026-06-12T09:48:00+10:00"
    )
    assert report["odds_capture_fixed_window_schedule"]["status_counts"] == {"PENDING": 4}
    assert report["odds_capture_fixed_window_schedule"]["next_pending_offset_minutes"] == 60
    assert report["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_minute"
    ] == 48
    assert report["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_covered"
    ] is True
    assert report["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert report["next_meaningful_action_at"] == "2026-06-12T09:48:00+10:00"
    assert report["next_race_id"] == "Race 1 - HEA - 2026-06-12"
    assert report["next_race_date"] == "2026-06-12"
    assert report["next_race_venue"] == "HEA"
    assert report["next_race_number"] == 1
    assert report["next_race_time"] == "10:48"
    assert report["next_race_jump_datetime"] == "2026-06-12T10:48:00+10:00"
    assert report["next_meaningful_action_offset_minutes"] == 60
    assert report["next_meaningful_action_timer_minute"] == 48
    assert report["next_meaningful_action_timer_covered"] is True
    assert report["next_meaningful_action_timer_coverage_reason"] == (
        "minute_covered_by_odds_capture_timer_on_calendar"
    )
    written = json.loads((output_dir / "odds_capture_only_daemon_report.json").read_text())
    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    state = json.loads(state_path.read_text())
    assert written["final_status"] == "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW"
    assert written["status"] == "WAITING"
    assert written["runtime_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert written["readiness_decision"] == "ODDS_CAPTURE_WAITING"
    assert written["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert written["next_race_id"] == report["next_race_id"]
    assert written["next_meaningful_action_offset_minutes"] == 60
    assert written["next_meaningful_action_timer_covered"] is True
    assert manifest["schema_version"] == "shadow_autopilot_daemon_output_manifest_v1"
    assert any(
        path.endswith("odds_capture_only_daemon_report.json")
        for path in manifest["files"]
    )
    assert state["window_state_source_updated_at"] == "2026-06-12T01:00:00+10:00"
    assert state["state_source"] == "full_daemon"
    assert state["source_report_path"] == "artifacts/full/odds_capture_refresh_report.json"
    assert state["recommended_rerun_after_local"] == "2026-06-12T09:43:00+10:00"
    assert state["source_recommended_rerun_after_local"] == "2026-06-12T09:48:00+10:00"
    assert state["resume_after_local"] == "2026-06-12T09:43:00+10:00"
    assert state["odds_capture_fixed_window_schedule"]["status_counts"] == {"PENDING": 4}
    assert state["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_covered"
    ] is True
    assert state["next_race_id"] == "Race 1 - HEA - 2026-06-12"
    assert state["next_race_venue"] == "HEA"
    assert state["next_meaningful_action_offset_minutes"] == 60
    assert state["next_meaningful_action_timer_minute"] == 48
    assert state["next_meaningful_action_timer_covered"] is True
    assert not lock_path.exists()


def test_odds_capture_preflight_wait_ignores_failed_state(tmp_path):
    state_path = tmp_path / "odds_capture_state.json"
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-12T01:00:00+10:00",
            "window_state_source_updated_at": "2026-06-12T01:00:00+10:00",
            "final_status": "ODDS_CAPTURE_ONLY_FAILED",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "odds_capture_refresh_status": "SUCCESS",
            "next_window_opens_at": "2026-06-12T09:48:00+10:00",
        },
    )

    wait = daemon.odds_capture_preflight_wait(
        state_path=state_path,
        now=daemon.datetime.fromisoformat("2026-06-12T01:10:00+10:00"),
    )

    assert wait is None


def test_full_daemon_publishes_fresh_odds_capture_preflight_state(tmp_path):
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    output_dir = tmp_path / "daemon"
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    output_dir.mkdir()
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "odds_capture_refresh_report.json",
        {
            "status": "SUCCESS",
            "next_preferred_window": {
                "status": "WAITING_FOR_FUTURE_WINDOW",
                "next_window_opens_at": "2026-06-12T09:48:00+10:00",
                "recommended_rerun_after_local": "2026-06-12T09:48:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "date": "2026-06-12",
                    "venue": "HEA",
                    "race_number": 1,
                    "race_time": "10:48",
                    "jump_datetime": "2026-06-12T10:48:00+10:00",
                },
            },
        },
    )

    report = daemon.publish_full_daemon_odds_capture_state(
        state_path=state_path,
        generated_at=daemon.datetime.fromisoformat("2026-06-12T03:31:37+10:00"),
        run_id="20260612T033137+1000",
        output_dir=output_dir,
        autopilot_output_dir=autopilot_dir,
        odds_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "inserted_live_odds_rows": 0,
            "ready_count": 6,
            "status_counts": {"SKIPPED_ALREADY_CAPTURED": 6},
            "blocked_attempt_count": 0,
            "blocked_attempts": [],
            "t2_miss_attempt_count": 2,
            "t2_miss_cause_counts": {
                "t2_miss_late_time_gate": 1,
                "t2_miss_validation_failed": 1,
            },
            "t2_miss_examples": [
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "cause": "t2_miss_late_time_gate",
                }
            ],
            "run_id": "direct_full_daemon_odds_capture",
            "final_status": "NO_ELIGIBLE_WINDOWS",
            "operator_status": "READY",
            "capture_window_coverage_status_counts": {"PENDING": 4},
        },
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert report["status"] == "PUBLISHED"
    assert report["inserted_live_odds_rows"] == 0
    assert report["ready_count"] == 6
    assert report["status_counts"] == {"SKIPPED_ALREADY_CAPTURED": 6}
    assert report["blocked_attempt_count"] == 0
    assert report["blocked_attempts"] == []
    assert report["t2_miss_attempt_count"] == 2
    assert report["t2_miss_cause_counts"] == {
        "t2_miss_late_time_gate": 1,
        "t2_miss_validation_failed": 1,
    }
    assert report["t2_miss_examples"][0]["race_id"] == "Race 8 - TAREE - 2026-06-13"
    assert report["autonomous_live_odds_capture_run_id"] == (
        "direct_full_daemon_odds_capture"
    )
    assert report["autonomous_live_odds_capture_final_status"] == (
        "NO_ELIGIBLE_WINDOWS"
    )
    assert report["autonomous_live_odds_capture_operator_status"] == "READY"
    assert report["capture_window_coverage_status_counts"] == {"PENDING": 4}
    assert report["next_window_opens_at"] == "2026-06-12T09:48:00+10:00"
    assert report["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert report["next_meaningful_action_at"] == "2026-06-12T09:48:00+10:00"
    assert state["state_source"] == "full_daemon"
    assert state["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert state["status"] == "READY"
    assert state["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert state["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert state["odds_capture_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    assert state["inserted_live_odds_rows"] == 0
    assert state["ready_count"] == 6
    assert state["status_counts"] == {"SKIPPED_ALREADY_CAPTURED": 6}
    assert state["blocked_attempt_count"] == 0
    assert state["blocked_attempts"] == []
    assert state["t2_miss_attempt_count"] == 2
    assert state["t2_miss_cause_counts"] == {
        "t2_miss_late_time_gate": 1,
        "t2_miss_validation_failed": 1,
    }
    assert state["t2_miss_examples"][0]["cause"] == "t2_miss_late_time_gate"
    assert state["autonomous_live_odds_capture_run_id"] == (
        "direct_full_daemon_odds_capture"
    )
    assert state["autonomous_live_odds_capture_final_status"] == "NO_ELIGIBLE_WINDOWS"
    assert state["autonomous_live_odds_capture_operator_status"] == "READY"
    assert state["capture_window_coverage_status_counts"] == {"PENDING": 4}
    assert state["window_state_source_updated_at"] == "2026-06-12T03:31:37+10:00"
    assert state["next_window_opens_at"] == "2026-06-12T09:48:00+10:00"
    assert state["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert state["next_meaningful_action_at"] == "2026-06-12T09:48:00+10:00"
    assert "schedule_reconciled_with_direct_capture" not in state[
        "odds_capture_fixed_window_schedule"
    ]
    assert state["recommended_rerun_after_local"] == "2026-06-12T09:43:00+10:00"
    assert state["source_recommended_rerun_after_local"] == "2026-06-12T09:48:00+10:00"
    assert state["resume_after_local"] == "2026-06-12T09:43:00+10:00"
    assert state["odds_capture_fixed_window_schedule"]["status_counts"] == {"PENDING": 4}
    assert state["odds_capture_fixed_window_schedule"]["next_pending_offset_minutes"] == 60
    assert state["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_minute"
    ] == 48
    assert state["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_covered"
    ] is True

    wait = daemon.odds_capture_preflight_wait(
        state_path=state_path,
        now=daemon.datetime.fromisoformat("2026-06-12T03:36:37+10:00"),
    )
    assert wait is not None
    assert wait["window_state_source_updated_at"] == "2026-06-12T03:31:37+10:00"
    assert wait["window_state_source"] == "full_daemon"
    assert wait["recommended_rerun_after_local"] == "2026-06-12T09:43:00+10:00"
    assert wait["source_recommended_rerun_after_local"] == "2026-06-12T09:48:00+10:00"
    assert wait["odds_capture_fixed_window_schedule"]["status_counts"] == {"PENDING": 4}
    assert wait["odds_capture_fixed_window_schedule"][
        "next_meaningful_action_timer_covered"
    ] is True


def test_odds_capture_fixed_window_schedule_marks_current_due_window():
    schedule = daemon.odds_capture_fixed_window_schedule(
        {
            "next_race": {
                "race_id": "Race 1 - HEA - 2026-06-12",
                "date": "2026-06-12",
                "venue": "HEA",
                "race_number": 1,
                "race_time": "10:48",
                "jump_datetime": "2026-06-12T10:48:00+10:00",
            }
        },
        current_time=daemon.datetime.fromisoformat("2026-06-12T10:20:00+10:00"),
    )

    assert schedule["coverage_checked"] is False
    assert schedule["status_counts"] == {"PASSED": 1, "DUE": 1, "PENDING": 2}
    assert schedule["next_due_offset_minutes"] == 30
    assert schedule["next_due_capture_at"] == "2026-06-12T10:18:00+10:00"
    assert schedule["next_pending_offset_minutes"] == 10
    assert schedule["next_meaningful_action"] == "RUN_ODDS_CAPTURE_NOW"
    assert schedule["next_meaningful_action_at"] == "2026-06-12T10:20:00+10:00"
    assert schedule["next_meaningful_action_offset_minutes"] == 30
    assert schedule["next_meaningful_action_timer_minute"] == 20
    assert schedule["next_meaningful_action_timer_covered"] is True
    assert schedule["next_meaningful_action_timer_coverage_reason"] == (
        "minute_covered_by_odds_capture_timer_on_calendar"
    )


def test_odds_capture_fixed_window_schedule_uses_captured_window_coverage():
    schedule = daemon.odds_capture_fixed_window_schedule(
        {
            "next_race": {
                "race_id": "Race 12 - TAREE - 2026-06-13",
                "date": "2026-06-13",
                "venue": "TAREE",
                "race_number": "12",
                "race_time": "11:05 AM",
                "jump_datetime": "2026-06-13T11:05:00+10:00",
            }
        },
        current_time=daemon.datetime.fromisoformat("2026-06-13T11:00:11+10:00"),
        capture_window_coverage={
            "schema_version": "autonomous_live_odds_capture_window_coverage_report_v1",
            "status_counts": {"CAPTURED": 3, "PENDING": 1},
            "windows": [
                {
                    "race_id": "Race 12 - TAREE - 2026-06-13",
                    "race_date": "2026-06-13",
                    "venue": "TAREE",
                    "race_number": 12,
                    "offset_minutes": offset,
                    "capture_mode": f"autonomous_prejump_t{offset}m",
                    "status": "CAPTURED",
                    "reason": "complete_existing_capture",
                    "existing_capture_count": 8,
                    "existing_capture_status": "COMPLETE",
                }
                for offset in (60, 30, 10)
            ]
            + [
                {
                    "race_id": "Race 12 - TAREE - 2026-06-13",
                    "race_date": "2026-06-13",
                    "venue": "TAREE",
                    "race_number": 12,
                    "offset_minutes": 2,
                    "capture_mode": "autonomous_prejump_t2m",
                    "status": "PENDING",
                    "reason": "window_not_open_yet",
                    "existing_capture_count": 0,
                    "existing_capture_status": "NONE",
                }
            ],
        },
    )

    assert schedule["coverage_checked"] is True
    assert schedule["capture_window_coverage_available"] is True
    assert schedule["capture_window_coverage_window_count"] == 4
    assert schedule["capture_window_coverage_matched_window_count"] == 4
    assert schedule["capture_window_coverage_match_status"] == "MATCHED_NEXT_RACE"
    assert schedule["status_counts"] == {"CAPTURED": 3, "PENDING": 1}
    assert schedule["next_due_offset_minutes"] is None
    assert schedule["next_pending_offset_minutes"] == 2
    assert schedule["next_meaningful_action"] == "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    assert schedule["next_meaningful_action_at"] == "2026-06-13T11:03:00+10:00"
    assert schedule["next_meaningful_action_offset_minutes"] == 2
    assert schedule["next_meaningful_action_timer_minute"] == 3
    assert schedule["next_meaningful_action_timer_covered"] is True


def test_odds_capture_fixed_window_schedule_surfaces_unmatched_coverage_report():
    schedule = daemon.odds_capture_fixed_window_schedule(
        {
            "next_race": {
                "race_id": "Race 2 - QOT - 2026-06-13",
                "date": "2026-06-13",
                "venue": "QOT",
                "race_number": "2",
                "race_time": "5:09 PM",
                "jump_datetime": "2026-06-13T17:09:00+10:00",
            }
        },
        current_time=daemon.datetime.fromisoformat("2026-06-13T17:06:39+10:00"),
        capture_window_coverage={
            "schema_version": "autonomous_live_odds_capture_window_coverage_report_v1",
            "status_counts": {"CAPTURED": 4},
            "windows": [
                {
                    "race_id": "Race 2 - LADBROKES-Q1-LAKESIDE - 2026-06-13",
                    "race_date": "2026-06-13",
                    "venue": "LADBROKES-Q1-LAKESIDE",
                    "race_number": 2,
                    "offset_minutes": offset,
                    "capture_mode": f"autonomous_prejump_t{offset}m",
                    "status": "CAPTURED",
                    "reason": "complete_existing_capture",
                    "existing_capture_count": 5,
                    "existing_capture_status": "COMPLETE",
                }
                for offset in daemon.ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES
            ],
        },
    )

    assert schedule["capture_window_coverage_available"] is True
    assert schedule["capture_window_coverage_window_count"] == 4
    assert schedule["capture_window_coverage_matched_window_count"] == 0
    assert schedule["capture_window_coverage_match_status"] == (
        "COVERAGE_AVAILABLE_NO_NEXT_RACE_MATCH"
    )
    assert schedule["coverage_checked"] is False
    assert schedule["capture_window_coverage_status_counts"] == {}
    assert schedule["next_meaningful_action"] == "RUN_ODDS_CAPTURE_NOW"
    assert schedule["next_due_offset_minutes"] == 10


def test_odds_capture_fixed_window_schedule_marks_uncovered_timer_minute():
    schedule = daemon.next_meaningful_action_timer_coverage(
        next_action_at="2026-06-12T10:02:00+10:00",
        current_time=daemon.datetime.fromisoformat("2026-06-12T10:20:00+10:00"),
    )

    assert schedule["next_meaningful_action_timer_minute"] == 2
    assert schedule["next_meaningful_action_timer_covered"] is False
    assert schedule["next_meaningful_action_timer_coverage_reason"] == (
        "minute_not_covered_by_odds_capture_timer_on_calendar"
    )


def test_odds_capture_fixed_window_schedule_timestamps_refresh_after_all_windows_captured():
    schedule = daemon.odds_capture_fixed_window_schedule(
        {
            "next_race": {
                "race_id": "Race 2 - GEE - 2026-06-13",
                "date": "2026-06-13",
                "venue": "GEE",
                "race_number": "2",
                "race_time": "5:04 PM",
                "jump_datetime": "2026-06-13T17:04:00+10:00",
            }
        },
        current_time=daemon.datetime.fromisoformat("2026-06-13T17:01:45+10:00"),
        capture_window_coverage={
            "schema_version": "autonomous_live_odds_capture_window_coverage_report_v1",
            "status_counts": {"CAPTURED": 4},
            "windows": [
                {
                    "race_id": "Race 2 - GEE - 2026-06-13",
                    "race_date": "2026-06-13",
                    "venue": "GEE",
                    "race_number": 2,
                    "offset_minutes": offset,
                    "capture_mode": f"autonomous_prejump_t{offset}m",
                    "status": "CAPTURED",
                    "reason": "complete_existing_capture",
                    "existing_capture_count": 8,
                    "existing_capture_status": "COMPLETE",
                }
                for offset in daemon.ODDS_CAPTURE_WINDOW_OFFSETS_MINUTES
            ],
        },
    )

    assert schedule["coverage_checked"] is True
    assert schedule["capture_window_coverage_match_status"] == "MATCHED_NEXT_RACE"
    assert schedule["status_counts"] == {"CAPTURED": 4}
    assert schedule["next_due_offset_minutes"] is None
    assert schedule["next_pending_offset_minutes"] is None
    assert schedule["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert schedule["next_meaningful_action_at"] == "2026-06-13T17:01:45+10:00"
    assert schedule["next_meaningful_action_timer_minute"] == 1
    assert schedule["next_meaningful_action_timer_covered"] is True
    assert schedule["next_meaningful_action_timer_coverage_reason"] == (
        "minute_covered_by_odds_capture_timer_on_calendar"
    )


def test_full_daemon_odds_window_defer_decision_defers_imminent_capture():
    state = {
        "next_meaningful_action": "RUN_ODDS_CAPTURE_NOW",
        "next_meaningful_action_at": "2026-06-13T07:42:11+10:00",
        "next_preferred_window": {
            "next_window_closes_at": "2026-06-13T07:59:00+10:00",
        },
        "odds_capture_fixed_window_schedule": {
            "next_pending_capture_at": "2026-06-13T07:49:00+10:00",
            "status_counts": {"DUE": 1, "PENDING": 2},
        },
    }

    decision = daemon.full_daemon_odds_window_defer_decision(
        state,
        current_time=daemon.datetime.fromisoformat("2026-06-13T07:47:11+10:00"),
    )

    assert decision["should_defer"] is True
    assert decision["reason"] == "odds_capture_window_open_or_imminent"
    assert decision["next_pending_capture_at"] == "2026-06-13T07:49:00+10:00"


def test_full_daemon_odds_window_defer_decision_allows_refresh_action():
    state = {
        "updated_at": "2026-06-13T18:12:48+10:00",
        "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
        "next_meaningful_action_at": "2026-06-13T18:12:48+10:00",
        "next_preferred_window": {
            "status": "OPEN_NOW",
            "next_window_closes_at": "2026-06-13T18:18:00+10:00",
        },
        "odds_capture_fixed_window_schedule": {
            "next_pending_capture_at": "2026-06-13T18:16:00+10:00",
            "status_counts": {"DUE": 1, "PASSED": 2, "PENDING": 1},
            "schedule_reconciled_with_direct_capture": True,
            "schedule_reconciliation_reason": (
                "direct_capture_all_ready_races_handled_after_append"
            ),
        },
    }

    decision = daemon.full_daemon_odds_window_defer_decision(
        state,
        current_time=daemon.datetime.fromisoformat("2026-06-13T18:17:00+10:00"),
    )

    assert decision["should_defer"] is False
    assert decision["reason"] == "odds_capture_refresh_action_requested"
    assert decision["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert decision["due_capture_window_count"] == 1


def test_full_daemon_odds_window_defer_decision_recomputes_near_due_t2_window():
    state = {
        "updated_at": "2026-06-12T01:29:00+10:00",
        "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
        "next_meaningful_action_at": "2026-06-12T01:29:00+10:00",
        "next_preferred_window": {
            "status": "OPEN_NOW",
            "next_window_opens_at": "2026-06-12T00:35:00+10:00",
            "next_window_closes_at": "2026-06-12T01:37:00+10:00",
            "next_race": {
                "race_id": "Race 1 - HEA - 2026-06-12",
                "date": "2026-06-12",
                "venue": "HEA",
                "race_number": 1,
                "race_time": "01:37",
                "jump_datetime": "2026-06-12T01:37:00+10:00",
            },
        },
        "odds_capture_fixed_window_schedule": {
            "generated_at": "2026-06-12T01:29:00+10:00",
            "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
            "next_meaningful_action_at": "2026-06-12T01:29:00+10:00",
            "status_counts": {"PASSED": 4},
        },
    }

    decision = daemon.full_daemon_odds_window_defer_decision(
        state,
        current_time=daemon.datetime.fromisoformat("2026-06-12T01:34:45+10:00"),
    )

    assert decision["should_defer"] is True
    assert decision["reason"] == "odds_capture_window_open_or_imminent"
    assert decision["fixed_window_schedule_source"] == (
        "recomputed_from_next_preferred_window"
    )
    assert decision["next_pending_capture_at"] == "2026-06-12T01:35:00+10:00"
    assert decision["next_pending_offset_minutes"] == 2


def test_full_daemon_odds_window_defer_decision_defers_future_pending_capture_after_refresh():
    state = {
        "updated_at": "2026-06-18T09:24:59+10:00",
        "next_meaningful_action": "REFRESH_UPCOMING_RACE_WINDOW",
        "next_meaningful_action_at": "2026-06-18T09:24:59+10:00",
        "next_preferred_window": {
            "status": "OPEN_NOW",
            "next_window_closes_at": "2026-06-18T10:08:00+10:00",
        },
        "odds_capture_fixed_window_schedule": {
            "next_pending_capture_at": "2026-06-18T09:38:00+10:00",
            "status_counts": {"DUE": 1, "PENDING": 3},
            "schedule_reconciled_with_direct_capture": True,
            "schedule_reconciliation_reason": (
                "direct_capture_all_ready_races_already_captured"
            ),
        },
    }

    decision = daemon.full_daemon_odds_window_defer_decision(
        state,
        current_time=daemon.datetime.fromisoformat("2026-06-18T09:32:00+10:00"),
    )

    assert decision["should_defer"] is True
    assert decision["reason"] == "odds_capture_window_open_or_imminent"
    assert decision["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert decision["next_pending_capture_at"] == "2026-06-18T09:38:00+10:00"


def test_full_daemon_odds_window_defer_decision_ignores_closed_stale_state():
    state = {
        "next_meaningful_action": "RUN_ODDS_CAPTURE_NOW",
        "next_meaningful_action_at": "2026-06-13T07:42:11+10:00",
        "next_preferred_window": {
            "next_window_closes_at": "2026-06-13T07:59:00+10:00",
        },
        "odds_capture_fixed_window_schedule": {
            "next_pending_capture_at": "2026-06-13T07:57:00+10:00",
            "status_counts": {"DUE": 1, "PENDING": 1},
        },
    }

    decision = daemon.full_daemon_odds_window_defer_decision(
        state,
        current_time=daemon.datetime.fromisoformat("2026-06-13T08:02:11+10:00"),
    )

    assert decision["should_defer"] is False
    assert decision["reason"] == "odds_capture_window_closed"


def test_full_daemon_odds_window_defer_decision_defers_fresh_multi_race_open_state():
    state = {
        "updated_at": "2026-06-13T08:00:00+10:00",
        "next_meaningful_action": "RUN_ODDS_CAPTURE_NOW",
        "next_meaningful_action_at": "2026-06-13T07:58:11+10:00",
        "next_preferred_window": {
            "status": "OPEN_NOW",
            "selected_count": 4,
            "next_window_closes_at": "2026-06-13T07:59:00+10:00",
        },
        "odds_capture_fixed_window_schedule": {
            "status_counts": {"DUE": 1, "PENDING": 2},
        },
    }

    decision = daemon.full_daemon_odds_window_defer_decision(
        state,
        current_time=daemon.datetime.fromisoformat("2026-06-13T08:02:00+10:00"),
    )

    assert decision["should_defer"] is True
    assert decision["reason"] == "odds_capture_state_open_with_additional_selected_races"
    assert decision["fresh_open_multi_race_state"] is True


def test_run_odds_capture_once_preserves_window_state_on_lock_busy(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_lock"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    state_path.parent.mkdir(parents=True)
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-12T01:29:00+10:00",
            "window_state_source_updated_at": "2026-06-12T01:29:00+10:00",
            "run_id": "previous_odds",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "odds_capture_refresh_status": "SUCCESS",
            "next_window_opens_at": "2026-06-12T01:35:00+10:00",
            "recommended_rerun_after_local": "2026-06-12T01:35:00+10:00",
            "next_preferred_window": {
                "status": "OPEN_NOW",
                "next_window_opens_at": "2026-06-12T00:35:00+10:00",
                "next_window_closes_at": "2026-06-12T01:37:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "date": "2026-06-12",
                    "venue": "HEA",
                    "race_number": 1,
                    "race_time": "01:37",
                    "jump_datetime": "2026-06-12T01:37:00+10:00",
                },
            },
        },
    )
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_SECONDS", 0)

    def busy_lock(**kwargs):
        raise daemon.LockBusy(
            {
                "existing_lock": {
                    "run_id": "20260613T214711+1000",
                    "output_dir": (
                        "artifacts/full_evidence_orchestration_20260525/"
                        "shadow_autopilot_daemonization_v1_20260613T214711+1000"
                    ),
                    "pid": 1316493,
                    "hostname": "worker-host",
                    "started_at": "2026-06-13T21:47:11+10:00",
                },
                "reason": "active_lock_present",
            }
        )

    monkeypatch.setattr(daemon, "acquire_lock", busy_lock)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_lock",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T01:33:00+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert report["final_status"] == "SKIPPED_LOCK_HELD"
    assert report["lock_owner_kind"] == "full_daemon"
    assert report["lock_owner_run_id"] == "20260613T214711+1000"
    assert report["lock_owner_pid"] == 1316493
    assert report["lock_owner_hostname"] == "worker-host"
    assert report["lock_owner_started_at"] == "2026-06-13T21:47:11+10:00"
    assert report["odds_capture_fixed_window_schedule"]["race_id"] == (
        "Race 1 - HEA - 2026-06-12"
    )
    assert report["odds_capture_fixed_window_schedule"]["jump_datetime"] == (
        "2026-06-12T01:37:00+10:00"
    )
    assert report["next_race_id"] == "Race 1 - HEA - 2026-06-12"
    assert report["next_race_date"] == "2026-06-12"
    assert report["next_race_venue"] == "HEA"
    assert report["next_race_number"] == 1
    assert report["next_race_time"] == "01:37"
    assert report["next_race_jump_datetime"] == "2026-06-12T01:37:00+10:00"
    assert report["next_meaningful_action_offset_minutes"] == 10
    assert report["next_meaningful_action_timer_minute"] == 33
    assert report["next_meaningful_action_timer_covered"] is True
    assert report["odds_capture_fixed_window_schedule"]["status_counts"] == {
        "DUE": 1,
        "PASSED": 2,
        "PENDING": 1,
    }
    assert report["next_meaningful_action"] == "RUN_ODDS_CAPTURE_NOW"
    manifest = json.loads((output_dir / "output_manifest.json").read_text())
    assert manifest["schema_version"] == "shadow_autopilot_daemon_output_manifest_v1"
    assert any(
        path.endswith("odds_capture_only_daemon_report.json")
        for path in manifest["files"]
    )
    state = json.loads(state_path.read_text())
    assert state["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert state["odds_capture_refresh_status"] == "SUCCESS"
    assert state["next_window_opens_at"] == "2026-06-12T01:35:00+10:00"
    assert state["window_state_source_updated_at"] == "2026-06-12T01:29:00+10:00"
    assert state["last_lock_skip"]["run_id"] == "odds_lock"
    assert state["last_lock_skip"]["lock"]["reason"] == "active_lock_present"
    assert state["last_lock_skip"]["lock_owner_kind"] == "full_daemon"
    assert state["last_lock_skip"]["lock_owner_run_id"] == "20260613T214711+1000"


def test_run_odds_capture_once_classifies_t2_lock_held_miss(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_t2_lock"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    state_path.parent.mkdir(parents=True)
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-12T01:34:00+10:00",
            "window_state_source_updated_at": "2026-06-12T01:34:00+10:00",
            "run_id": "previous_odds",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "odds_capture_refresh_status": "SUCCESS",
            "next_window_opens_at": "2026-06-12T01:35:00+10:00",
            "recommended_rerun_after_local": "2026-06-12T01:35:00+10:00",
            "next_preferred_window": {
                "status": "OPEN_NOW",
                "next_window_opens_at": "2026-06-12T00:35:00+10:00",
                "next_window_closes_at": "2026-06-12T01:37:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "date": "2026-06-12",
                    "venue": "HEA",
                    "race_number": 1,
                    "race_time": "01:37",
                    "jump_datetime": "2026-06-12T01:37:00+10:00",
                },
            },
        },
    )
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_SECONDS", 0)

    def busy_lock(**kwargs):
        raise daemon.LockBusy(
            {
                "existing_lock": {
                    "run_id": "20260613T214711+1000",
                    "output_dir": (
                        "artifacts/full_evidence_orchestration_20260525/"
                        "shadow_autopilot_daemonization_v1_20260613T214711+1000"
                    ),
                    "pid": 1316493,
                    "hostname": "worker-host",
                    "started_at": "2026-06-13T21:47:11+10:00",
                },
                "reason": "active_lock_present",
            }
        )

    monkeypatch.setattr(daemon, "acquire_lock", busy_lock)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "t2_lock",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T01:35:30+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)
    state = json.loads(state_path.read_text())

    assert report["final_status"] == "SKIPPED_LOCK_HELD"
    assert report["next_meaningful_action_offset_minutes"] == 2
    assert report["t2_miss_lock_held"] is True
    assert report["t2_miss_cause_counts"] == {"t2_miss_lock_held": 1}
    assert report["t2_lock_skip_race_id"] == "Race 1 - HEA - 2026-06-12"
    assert report["t2_lock_skip_target_capture_at"] == "2026-06-12T01:35:00+10:00"
    assert report["t2_lock_skip_lock_status"] == "SKIPPED_LOCK_HELD"
    assert state["last_lock_skip"]["t2_miss_lock_held"] is True
    assert state["last_lock_skip"]["t2_miss_cause_counts"] == {
        "t2_miss_lock_held": 1
    }


def test_run_odds_capture_once_retries_full_daemon_lock_when_t2_due(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_t2_retry"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    state_path.parent.mkdir(parents=True)
    autopilot_dir = evidence_root / "shadow_autopilot_v1_t2_retry_autopilot"
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-12T01:34:00+10:00",
            "window_state_source_updated_at": "2026-06-12T01:34:00+10:00",
            "run_id": "previous_odds",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "odds_capture_refresh_status": "SUCCESS",
            "next_window_opens_at": "2026-06-12T01:35:00+10:00",
            "recommended_rerun_after_local": "2026-06-12T01:35:00+10:00",
            "next_preferred_window": {
                "status": "OPEN_NOW",
                "next_window_opens_at": "2026-06-12T00:35:00+10:00",
                "next_window_closes_at": "2026-06-12T01:37:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "date": "2026-06-12",
                    "venue": "HEA",
                    "race_number": 1,
                    "race_time": "01:37",
                    "jump_datetime": "2026-06-12T01:37:00+10:00",
                },
            },
        },
    )
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon, "DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_SECONDS", 10)
    monkeypatch.setattr(daemon, "DEFAULT_ODDS_CAPTURE_ONLY_T2_LOCK_RETRY_POLL_SECONDS", 5)
    acquire_calls = []
    sleeps = []

    def retry_then_acquire(**kwargs):
        acquire_calls.append(kwargs)
        if len(acquire_calls) == 1:
            raise daemon.LockBusy(
                {
                    "existing_lock": {
                        "run_id": "20260613T214711+1000",
                        "output_dir": (
                            "artifacts/full_evidence_orchestration_20260525/"
                            "shadow_autopilot_daemonization_v1_20260613T214711+1000"
                        ),
                        "pid": 1316493,
                        "hostname": "worker-host",
                        "started_at": "2026-06-13T21:47:11+10:00",
                    },
                    "reason": "active_lock_present",
                }
            )
        return {
            "schema_version": "shadow_autopilot_daemon_lock_v1",
            "run_id": kwargs["run_id"],
            "pid": 12345,
            "hostname": "worker-host",
            "started_at": "2026-06-12T01:35:35+10:00",
            "output_dir": daemon.relpath(kwargs["output_dir"]),
        }

    def fake_sleep(seconds):
        sleeps.append(seconds)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        assert name == "odds_capture_autopilot_cycle"
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                "inserted_live_odds_rows": 0,
                "ready_count": 0,
                "final_status": "NO_ELIGIBLE_WINDOWS",
                "operator_status": "READY",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "next_preferred_window": {
                    "status": "WAITING_FOR_FUTURE_WINDOW",
                    "next_window_opens_at": "2026-06-12T02:00:00+10:00",
                    "recommended_rerun_after_local": "2026-06-12T02:00:00+10:00",
                    "next_race": {
                        "race_id": "Race 2 - HEA - 2026-06-12",
                        "date": "2026-06-12",
                        "venue": "HEA",
                        "race_number": 2,
                        "race_time": "03:00",
                        "jump_datetime": "2026-06-12T03:00:00+10:00",
                    },
                },
            },
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "AUTOPILOT_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "acquire_lock", retry_then_acquire)
    monkeypatch.setattr(daemon.time, "sleep", fake_sleep)
    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "t2_retry",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T01:35:30+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert len(acquire_calls) == 2
    assert sleeps
    assert sleeps[0] == 5.0
    assert report["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["lock"]["lock_retry"]["status"] == "ACQUIRED_AFTER_T2_DUE_LOCK_WAIT"
    assert report["lock"]["lock_retry"]["retried_for_t2_due_lock"] is True
    assert report["lock"]["lock_retry"]["waited_seconds"] == 5.0
    assert report["pre_lock_odds_capture_fixed_window_schedule"][
        "next_meaningful_action_offset_minutes"
    ] == 2
    retry_report = json.loads((output_dir / "odds_capture_t2_lock_retry.json").read_text())
    assert retry_report["status"] == "WAITING_FOR_FULL_DAEMON_LOCK_DURING_T2"
    assert retry_report["retry_window"]["race_id"] == "Race 1 - HEA - 2026-06-12"


def test_run_odds_capture_once_yields_to_full_daemon_lock_handoff(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_handoff"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    state_path.parent.mkdir(parents=True)
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_odds_capture_only_state_v1",
            "updated_at": "2026-06-13T18:44:21+10:00",
            "run_id": "previous_odds",
            "final_status": "ODDS_CAPTURE_ONLY_READY",
            "odds_capture_refresh_status": "SUCCESS",
            "next_window_opens_at": "2026-06-13T18:48:00+10:00",
            "next_preferred_window": {
                "status": "OPEN_NOW",
                "next_window_opens_at": "2026-06-13T18:00:00+10:00",
                "next_window_closes_at": "2026-06-13T18:50:00+10:00",
                "next_race": {
                    "race_id": "Race 3 - MEADOWS - 2026-06-13",
                    "date": "2026-06-13",
                    "venue": "MEADOWS",
                    "race_number": 3,
                    "race_time": "18:50",
                    "jump_datetime": "2026-06-13T18:50:00+10:00",
                },
            },
        },
    )
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    daemon.write_full_daemon_lock_wait_marker(
        lock_path=lock_path,
        run_id="20260613T184700+1000",
        output_dir=evidence_root / "shadow_autopilot_daemonization_v1_20260613T184700+1000",
    )

    def fail_acquire_lock(**kwargs):
        raise AssertionError("odds-only must not acquire while full daemon is waiting")

    def fail_run_command(**kwargs):
        raise AssertionError("odds-only must not run capture while yielding handoff")

    monkeypatch.setattr(daemon, "acquire_lock", fail_acquire_lock)
    monkeypatch.setattr(daemon, "run_command", fail_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_handoff",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T18:48:00+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert report["final_status"] == "SKIPPED_FULL_DAEMON_LOCK_HANDOFF"
    assert report["status"] == "SKIPPED_FULL_DAEMON_LOCK_HANDOFF"
    assert report["runtime_action"] == "YIELD_LOCK_HANDOFF_TO_FULL_DAEMON"
    assert report["readiness_decision"] == "WAIT_FOR_FULL_DAEMON"
    assert report["lock"]["reason"] == "full_daemon_waiting_for_odds_lock_handoff"
    assert report["lock"]["full_daemon_wait_marker"]["run_id"] == "20260613T184700+1000"
    assert report["steps"] == []
    state = json.loads(state_path.read_text())
    assert state["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert state["last_lock_skip"]["run_id"] == "odds_handoff"
    assert state["last_lock_skip"]["lock"]["reason"] == (
        "full_daemon_waiting_for_odds_lock_handoff"
    )


def test_run_odds_capture_once_handles_validation_block_as_no_write(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_blocked"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_blocked_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED",
                "inserted_live_odds_rows": 0,
                "ready_count": 1,
                "status_counts": {"BLOCKED_VALIDATION_FAILED": 1},
                "blocked_attempt_count": 1,
                "blocked_attempts": [
                    {
                        "race_id": "Race 5 - GEE - 2026-06-13",
                        "capture_window_minutes": 60,
                        "status": "BLOCKED_VALIDATION_FAILED",
                        "reasons": [
                            "sportsbet_missing_expected_runners:5:BENSHIRAZ"
                        ],
                        "validation_status": "FAIL",
                        "validation_expected_runner_count": 8,
                        "validation_accepted_row_count": 4,
                        "validation_missing_expected_runner_count": 4,
                    }
                ],
                "run_id": "direct_validation_block",
                "final_status": "BLOCKED",
                "operator_status": "BLOCKED",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {"status": "SUCCESS"},
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "PARTIAL_AUTOMATION_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_blocked",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T09:48:00+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE"
    assert report["autopilot_result"]["final_verdict"] == "PARTIAL_AUTOMATION_READY"
    assert report["autonomous_live_odds_capture_status"]["status"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    )
    assert report["autonomous_live_odds_capture_status_text"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    )
    assert report["autonomous_live_odds_capture_status"]["inserted_live_odds_rows"] == 0
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["lock_release"]["released"] is True
    written = json.loads((output_dir / "odds_capture_only_daemon_report.json").read_text())
    assert written["status"] == "HANDLED_NO_WRITE"
    assert written["runtime_action"] == "REVIEW_BLOCKED_CAPTURE_NO_DB_WRITE"
    assert written["readiness_decision"] == "CHECK_ODDS_CAPTURE_BLOCKER"
    assert written["inserted_live_odds_rows"] == 0
    assert written["ready_count"] == 1
    assert written["status_counts"] == {"BLOCKED_VALIDATION_FAILED": 1}
    assert written["blocked_attempt_count"] == 1
    assert written["blocked_attempts"][0]["race_id"] == "Race 5 - GEE - 2026-06-13"
    assert written["blocked_attempts"][0]["validation_status"] == "FAIL"
    state = json.loads(state_path.read_text())
    assert state["final_status"] == "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE"
    assert state["status"] == "HANDLED_NO_WRITE"
    assert state["runtime_action"] == "REVIEW_BLOCKED_CAPTURE_NO_DB_WRITE"
    assert state["readiness_decision"] == "CHECK_ODDS_CAPTURE_BLOCKER"
    assert state["odds_capture_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert state["inserted_live_odds_rows"] == 0
    assert state["ready_count"] == 1
    assert state["status_counts"] == {"BLOCKED_VALIDATION_FAILED": 1}
    assert state["blocked_attempt_count"] == 1
    assert state["blocked_attempts"][0]["race_id"] == "Race 5 - GEE - 2026-06-13"
    assert state["blocked_attempts"][0]["validation_missing_expected_runner_count"] == 4
    assert state["autonomous_live_odds_capture_run_id"] == "direct_validation_block"
    assert state["autonomous_live_odds_capture_final_status"] == "BLOCKED"
    assert state["autonomous_live_odds_capture_operator_status"] == "BLOCKED"


def test_run_odds_capture_once_handles_time_gate_block_as_no_write(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_time_gate"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_time_gate_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED",
                "inserted_live_odds_rows": 0,
                "ready_count": 7,
                "status_counts": {
                    "BLOCKED_TIME_GATE_BEFORE_FETCH": 1,
                    "SKIPPED_ALREADY_CAPTURED": 6,
                },
                "run_id": "direct_time_gate_block",
                "final_status": "BLOCKED",
                "operator_status": "BLOCKED",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {"status": "SUCCESS"},
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "PARTIAL_AUTOMATION_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_time_gate",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-12T09:48:00+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE"
    assert report["status"] == "HANDLED_NO_WRITE"
    assert report["runtime_action"] == "REVIEW_BLOCKED_CAPTURE_NO_DB_WRITE"
    assert report["readiness_decision"] == "CHECK_ODDS_CAPTURE_BLOCKER"
    assert report["inserted_live_odds_rows"] == 0
    assert report["ready_count"] == 7
    assert report["status_counts"] == {
        "BLOCKED_TIME_GATE_BEFORE_FETCH": 1,
        "SKIPPED_ALREADY_CAPTURED": 6,
    }
    assert report["no_write_guarantees"]["db_write"] is False
    state = json.loads(state_path.read_text())
    assert state["final_status"] == "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE"
    assert state["status"] == "HANDLED_NO_WRITE"
    assert state["runtime_action"] == "REVIEW_BLOCKED_CAPTURE_NO_DB_WRITE"
    assert state["readiness_decision"] == "CHECK_ODDS_CAPTURE_BLOCKER"
    assert state["odds_capture_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_BLOCKED"
    assert state["inserted_live_odds_rows"] == 0
    assert state["ready_count"] == 7
    assert state["status_counts"] == {
        "BLOCKED_TIME_GATE_BEFORE_FETCH": 1,
        "SKIPPED_ALREADY_CAPTURED": 6,
    }
    assert state["autonomous_live_odds_capture_run_id"] == "direct_time_gate_block"
    assert state["autonomous_live_odds_capture_final_status"] == "BLOCKED"
    assert state["autonomous_live_odds_capture_operator_status"] == "BLOCKED"


def test_run_odds_capture_once_surfaces_appended_with_blocked_attempts(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_odds_mixed"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "runtime" / "shadow.lock"
    state_path = tmp_path / "runtime" / "odds_capture_state.json"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_odds_mixed_autopilot"

    monkeypatch.setattr(daemon, "ROOT", tmp_path)

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        autopilot_dir.mkdir(parents=True)
        daemon.write_json(
            autopilot_dir / "autonomous_live_odds_capture_status.json",
            {
                "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
                "inserted_live_odds_rows": 8,
                "ready_count": 8,
                "status_counts": {
                    "APPENDED": 1,
                    "BLOCKED_TIME_GATE_BEFORE_APPEND": 1,
                    "SKIPPED_ALREADY_CAPTURED": 6,
                },
                "blocked_attempt_count": 1,
                "blocked_attempts": [
                    {
                        "race_id": "Race 4 - LADBROKES-Q1-LAKESIDE - 2026-06-13",
                        "capture_window_minutes": 2,
                        "status": "BLOCKED_TIME_GATE_BEFORE_APPEND",
                        "reasons": ["race_already_jumped"],
                        "validation_status": "PASS",
                    }
                ],
                "run_id": "direct_mixed_append_block",
                "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
                "operator_status": "APPENDED_WITH_BLOCKED_ATTEMPTS",
                "runtime_action": "REVIEW_CAPTURE_BLOCKERS_AFTER_APPEND",
                "readiness_decision": "CONTINUE_ODDS_CAPTURE_WITH_BLOCKER_REVIEW",
            },
        )
        daemon.write_json(
            autopilot_dir / "odds_capture_refresh_report.json",
            {
                "status": "SUCCESS",
                "next_preferred_window": {
                    "status": "OPEN_NOW",
                    "next_window_opens_at": "2026-06-13T16:47:00+10:00",
                    "next_window_closes_at": "2026-06-13T17:47:00+10:00",
                    "next_race": {
                        "race_id": "Race 4 - QOT - 2026-06-13",
                        "date": "2026-06-13",
                        "venue": "QOT",
                        "race_number": "4",
                        "race_time": "5:47 PM",
                        "jump_datetime": "2026-06-13T17:47:00+10:00",
                    },
                },
            },
        )
        daemon.write_json(
            output_dir / "logs" / "odds_capture_autopilot_cycle.stdout.txt",
            {
                "output_dir": str(autopilot_dir),
                "final_verdict": "AUTOPILOT_READY",
            },
        )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    args = daemon.parse_args(
        [
            "run-odds-capture-once",
            "--run-id",
            "odds_mixed",
            "--evidence-root",
            str(evidence_root),
            "--output-dir",
            str(output_dir),
            "--current-time",
            "2026-06-13T17:45:05+10:00",
            "--db",
            str(db_path),
            "--lock-path",
            str(lock_path),
            "--state-path",
            str(state_path),
        ]
    )

    report = daemon.run_odds_capture_once(args)
    state = json.loads(state_path.read_text())

    assert report["final_status"] == "ODDS_CAPTURE_ONLY_READY"
    assert report["status"] == "READY_WITH_BLOCKED_ATTEMPTS"
    assert report["runtime_action"] == "REVIEW_BLOCKED_CAPTURE_AFTER_APPEND"
    assert report["readiness_decision"] == (
        "CONTINUE_ODDS_CAPTURE_WITH_BLOCKER_REVIEW"
    )
    assert report["autonomous_live_odds_capture_status_text"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    )
    assert report["inserted_live_odds_rows"] == 8
    assert report["blocked_attempt_count"] == 1
    assert report["status_counts"] == {
        "APPENDED": 1,
        "BLOCKED_TIME_GATE_BEFORE_APPEND": 1,
        "SKIPPED_ALREADY_CAPTURED": 6,
    }
    assert state["status"] == "READY_WITH_BLOCKED_ATTEMPTS"
    assert state["runtime_action"] == "REVIEW_BLOCKED_CAPTURE_AFTER_APPEND"
    assert state["readiness_decision"] == "CONTINUE_ODDS_CAPTURE_WITH_BLOCKER_REVIEW"
    assert state["autonomous_live_odds_capture_operator_status"] == (
        "APPENDED_WITH_BLOCKED_ATTEMPTS"
    )
    assert state["blocked_attempts"][0]["status"] == "BLOCKED_TIME_GATE_BEFORE_APPEND"
    assert report["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert state["next_meaningful_action"] == "REFRESH_UPCOMING_RACE_WINDOW"
    assert report["odds_capture_fixed_window_schedule"][
        "schedule_reconciliation_reason"
    ] == "direct_capture_time_gate_blockers_already_expired"
    assert report["odds_capture_fixed_window_schedule"][
        "pre_reconciliation_next_meaningful_action"
    ] == "RUN_ODDS_CAPTURE_NOW"


def test_odds_capture_only_main_returns_zero_for_handled_no_write(monkeypatch, capsys):
    monkeypatch.setattr(
        daemon,
        "run_odds_capture_once",
        lambda args: {"final_status": "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE"},
    )

    assert daemon.main(["run-odds-capture-once"]) == 0
    assert "ODDS_CAPTURE_ONLY_HANDLED_NO_WRITE" in capsys.readouterr().out


def test_alert_report_flags_safe_join_increase_and_target():
    report = daemon.build_alert_report(
        current_dashboard={
            "safe_joined_races": 101,
            "unsafe_matches": 2,
            "top1": 0.2,
            "box_1_share": 0.22,
            "calibration": {"slope": 0.7},
        },
        previous_dashboard={
            "safe_joined_races": 99,
            "unsafe_matches": 2,
            "top1": 0.21,
            "box_1_share": 0.21,
            "calibration": {"slope": 0.75},
        },
        automated_join_report={"results": []},
        target_joined_races=100,
    )

    rules = {alert["rule"] for alert in report["triggered_alerts"]}
    assert "safe_joined_increase" in rules
    assert "safe_joined_target_reached" in rules


def test_service_and_timer_define_15_minute_oneshot_cycle():
    service = daemon.service_file_text(
        repo_path=Path("/home/l4nd0/greyhound_racing_collector"),
        timeout_seconds=840,
        python_path=Path("/runtime/.venv/bin/python"),
        evidence_root=Path("/runtime/artifacts/full_evidence_orchestration_20260525"),
        shadow_model=Path("/models/stage2/shadow_randomforest_model.joblib"),
        db_path=Path("/data/greyhound_racing_data.db"),
        lock_path=Path("/runtime/shared-shadow-autopilot.lock"),
        state_path=Path("/runtime/state.json"),
        odds_capture_state_path=Path("/runtime/odds_capture_state.json"),
    )
    timer = daemon.timer_file_text()

    assert "Type=oneshot" in service
    assert "ExecStart=/runtime/.venv/bin/python" in service
    assert "shadow_autopilot_daemon.py run-once" in service
    assert (
        "--evidence-root /runtime/artifacts/full_evidence_orchestration_20260525"
        in service
    )
    assert service.index("--evidence-root") < service.index("--days-ahead")
    assert "--shadow-model /models/stage2/shadow_randomforest_model.joblib" in service
    assert "--db /data/greyhound_racing_data.db" in service
    assert "--lock-path /runtime/shared-shadow-autopilot.lock" in service
    assert "--state-path /runtime/state.json" in service
    assert "--odds-capture-state-path /runtime/odds_capture_state.json" in service
    assert "--rejoin-pending-limit 8" in service
    assert "--refresh-limit 6" in service
    assert "--autonomous-odds-capture-limit 16" in service
    assert "--result-backlog-limit 8" in service
    assert "--result-backlog-shadow-run-limit 16" in service
    assert "--enable-autonomous-odds-capture" in service
    assert "--execute-autonomous-odds-capture" in service
    assert "--allow-auto-scrape-odds" in service
    assert "--enable-autonomous-result-capture" in service
    assert "--require-safe-refresh-metadata" in service
    assert "TimeoutStartSec=3360" in service
    assert "GREYHOUND_ALLOW_TGR=0" in service
    assert "/home/l4nd0/.local/bin" in service
    assert "OnActiveSec=15min" in timer
    assert "OnUnitInactiveSec=15min" in timer
    assert "OnCalendar" not in timer
    assert "AccuracySec=30s" in timer
    assert "Persistent=true" in timer
    assert f"ConditionPathExists=!{daemon.DEFAULT_HEAVY_SCHEDULING_PAUSE_PATH}" in service
    assert "Nice=10" in service
    assert "CPUWeight=20" in service
    assert "IOWeight=20" in service
    assert "IOSchedulingClass=idle" in service


def test_odds_capture_service_and_timer_define_minutely_locked_lane():
    service = daemon.odds_capture_service_file_text(
        repo_path=Path("/home/l4nd0/greyhound_racing_collector"),
        timeout_seconds=600,
        python_path=Path("/runtime/.venv/bin/python"),
        evidence_root=Path("/runtime/artifacts/full_evidence_orchestration_20260525"),
        db_path=Path("/data/greyhound_racing_data.db"),
        lock_path=Path("/runtime/shared-shadow-autopilot.lock"),
        state_path=Path("/runtime/odds_capture_state.json"),
        forward_corpus_root=Path("/runtime/forward-corpus"),
        refresh_limit=8,
    )
    timer = daemon.odds_capture_timer_file_text()

    assert "Type=oneshot" in service
    assert "ExecStart=/runtime/.venv/bin/python" in service
    assert "shadow_autopilot_daemon.py run-odds-capture-once" in service
    assert (
        "--evidence-root /runtime/artifacts/full_evidence_orchestration_20260525"
        in service
    )
    assert service.index("--evidence-root") < service.index("--days-ahead")
    assert "--db /data/greyhound_racing_data.db" in service
    assert "--lock-path /runtime/shared-shadow-autopilot.lock" in service
    assert "--state-path /runtime/odds_capture_state.json" in service
    assert '--forward-corpus-root "/runtime/forward-corpus"' in service
    assert "--odds-capture-refresh-limit 8" in service
    assert "--require-safe-refresh-metadata" in service
    assert "--skip-primary-refresh" in service
    assert "TimeoutStartSec=1200" in service
    assert "GREYHOUND_ALLOW_TGR=0" in service
    assert f"OnCalendar={daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR}" in timer
    assert "OnUnitActiveSec" not in timer
    assert "AccuracySec=15s" in timer
    assert "Unit=shadow-autopilot-odds-capture.service" in timer


def test_write_odds_capture_service_files_preserves_db_and_lock(tmp_path):
    service_dir = tmp_path / "systemd"

    result = daemon.write_odds_capture_service_files(
        service_dir=service_dir,
        repo_path=Path("/home/l4nd0/greyhound_racing_collector"),
        timeout_seconds=600,
        python_path=Path("/runtime/.venv/bin/python"),
        evidence_root=Path("/runtime/artifacts/full_evidence_orchestration_20260525"),
        db_path=Path("/data/greyhound_racing_data.db"),
        lock_path=Path("/runtime/shared-shadow-autopilot.lock"),
        state_path=Path("/runtime/odds_capture_state.json"),
        forward_corpus_root=Path("/runtime/forward-corpus"),
        refresh_limit=8,
    )

    service = (service_dir / daemon.ODDS_CAPTURE_SERVICE_NAME).read_text(encoding="utf-8")
    timer = (service_dir / daemon.ODDS_CAPTURE_TIMER_NAME).read_text(encoding="utf-8")
    assert result["timer_frequency"] == "1min_except_full_daemon"
    assert result["timer_calendar"] == (
        daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR
    )
    assert result["python_path"] == "/runtime/.venv/bin/python"
    assert (
        result["evidence_root"]
        == "/runtime/artifacts/full_evidence_orchestration_20260525"
    )
    assert result["db_path"] == "/data/greyhound_racing_data.db"
    assert result["lock_path"] == "/runtime/shared-shadow-autopilot.lock"
    assert result["forward_corpus_root"] == "/runtime/forward-corpus"
    assert "ExecStart=/runtime/.venv/bin/python" in service
    assert (
        "--evidence-root /runtime/artifacts/full_evidence_orchestration_20260525"
        in service
    )
    assert service.index("--evidence-root") < service.index("--days-ahead")
    assert "--db /data/greyhound_racing_data.db" in service
    assert "--lock-path /runtime/shared-shadow-autopilot.lock" in service
    assert '--forward-corpus-root "/runtime/forward-corpus"' in service
    assert f"OnCalendar={daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR}" in timer
    assert "OnUnitActiveSec" not in timer


def test_odds_capture_timer_reserves_full_daemon_minutes_and_covers_t2_windows():
    odds_minutes = {
        int(value)
        for value in daemon.DEFAULT_ODDS_CAPTURE_ONLY_TIMER_ON_CALENDAR.removeprefix("*:").split(",")
    }
    full_minutes = {2, 17, 32, 47}

    assert full_minutes.isdisjoint(odds_minutes)
    assert {0, 1, 3, 16, 31, 46, 48, 58, 59}.issubset(odds_minutes)
    assert {2, 17, 32, 47}.isdisjoint(odds_minutes)
    for target_minute in range(60):
        jump_minute = (target_minute + 2) % 60
        covered_before_jump = any(
            (target_minute + delta) % 60 in odds_minutes for delta in range(2)
        )
        assert covered_before_jump, (
            f"T-2 target minute {target_minute:02d} before jump minute "
            f"{jump_minute:02d} has no odds-only timer tick before jump"
        )


def test_best_unified_evidence_aggregate_status_path_prefers_larger_ready_backlog(
    tmp_path,
):
    skipped = tmp_path / "skipped.json"
    backlog = tmp_path / "backlog.json"
    rejoin = tmp_path / "rejoin.json"
    daemon.write_json(
        skipped,
        {
            "status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_SKIPPED",
            "unified_evidence_eligible_rows": 9999,
            "row_count": 9999,
        },
    )
    daemon.write_json(
        backlog,
        {
            "status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "unified_evidence_eligible_rows": 3872,
            "row_count": 5108,
        },
    )
    daemon.write_json(
        rejoin,
        {
            "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "unified_evidence_eligible_rows": 15,
            "row_count": 15,
        },
    )

    selected = daemon.best_unified_evidence_aggregate_status_path(
        [None, skipped, rejoin, backlog]
    )

    assert selected == backlog


def test_apply_best_aggregate_unified_evidence_to_daily_status_surfaces_backlog(
    tmp_path,
):
    status_path = tmp_path / "backlog_unified_evidence_datasets_status.json"
    status = {
        "status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
        "dataset_count": 56,
        "failed_dataset_count": 0,
        "row_count": 5108,
        "unified_evidence_eligible_rows": 3872,
        "artifact_odds_rows_seen": 29,
        "artifact_odds_rows_accepted": 12,
        "artifact_odds_rows_rejected": 17,
        "artifact_odds_rejection_reason_counts": {"too_late": 17},
        "rejected_live_odds_candidate_count": 9,
        "rows_with_rejected_live_odds_candidates": 6,
        "rejected_live_odds_candidate_reason_counts": {
            "odds_decimal_invalid": 4,
            "odds_source_url_missing": 5,
        },
        "race_coverage": {
            "gap_action_plan": {
                "sample_blocking_gap_count": 2,
                "action_counts": {
                    "retry_official_result_capture_or_join": 2,
                    "collect_future_strict_prejump_odds": 0,
                },
                "evidence_missing_reason_counts": {
                    "official_result_missing_only": 2,
                    "strict_prejump_odds_missing": 0,
                },
                "top_gap_races": [
                    {
                        "race_id": "Race 7 - TAREE - 2026-06-13",
                        "recommended_action": (
                            "inspect_quarantined_official_result_runner_set"
                        ),
                        "evidence_missing_reason": (
                            "official_result_quarantined_unsafe_match"
                        ),
                        "missing_official_result": True,
                        "official_result_quarantine_reason": (
                            "ingest_failed_or_unsafe_match"
                        ),
                        "official_result_quarantine_errors": [
                            "result_boxes_not_in_participants:9"
                        ],
                        "official_result_quarantine_source_urls": [
                            "https://www.thedogs.com.au/racing/taree/2026-06-13/7"
                        ],
                        "official_result_quarantine_participant_boxes": [
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                        ],
                        "official_result_quarantine_result_boxes_not_in_participants": [
                            9
                        ],
                        "official_result_quarantine_reserve_substitution_diagnostic": {
                            "classification": (
                                "possible_reserve_substitution_manual_review_required"
                            ),
                            "acceptance_status": "not_accepted_report_only",
                            "candidate_reserve_boxes": [9],
                            "scratched_participant_boxes": [1],
                        },
                    },
                    {"race_id": "Race 4 - TAREE - 2026-06-13"},
                ],
            },
            "top_official_result_missing_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "official_result_quarantine_reason": (
                        "ingest_failed_or_unsafe_match"
                    ),
                    "official_result_quarantine_errors": [
                        "result_boxes_not_in_participants:9"
                    ],
                },
                {"race_id": "Race 4 - TAREE - 2026-06-13"},
            ],
        },
    }
    daily_status = {
        "backlog_unified_evidence_status": None,
        "backlog_unified_evidence_dataset_count": 0,
        "backlog_unified_evidence_eligible_rows": 0,
    }

    daemon.apply_best_aggregate_unified_evidence_to_daily_status(
        daily_status,
        best_status_path=status_path,
        best_status=status,
    )

    assert daily_status["best_aggregate_unified_evidence_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert daily_status["best_aggregate_unified_evidence_dataset_count"] == 56
    assert daily_status["best_aggregate_unified_evidence_failed_dataset_count"] == 0
    assert daily_status["best_aggregate_unified_evidence_row_count"] == 5108
    assert daily_status["best_aggregate_unified_evidence_eligible_rows"] == 3872
    assert daily_status[
        "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts"
    ] == {"too_late": 17}
    assert daily_status[
        "best_aggregate_unified_rejected_live_odds_candidate_count"
    ] == 9
    assert daily_status[
        "best_aggregate_unified_rows_with_rejected_live_odds_candidates"
    ] == 6
    assert daily_status[
        "best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert daily_status["best_aggregate_unified_sample_blocking_gap_count"] == 2
    assert daily_status["best_aggregate_unified_gap_action_counts"] == {
        "retry_official_result_capture_or_join": 2,
        "collect_future_strict_prejump_odds": 0,
    }
    assert daily_status[
        "best_aggregate_unified_gap_evidence_missing_reason_counts"
    ] == {
        "official_result_missing_only": 2,
        "strict_prejump_odds_missing": 0,
    }
    assert daily_status["best_aggregate_unified_top_gap_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert daily_status["best_aggregate_unified_top_gap_races"][0] == {
        "race_id": "Race 7 - TAREE - 2026-06-13",
        "recommended_action": "inspect_quarantined_official_result_runner_set",
        "evidence_missing_reason": "official_result_quarantined_unsafe_match",
        "missing_official_result": True,
        "official_result_quarantine_reason": "ingest_failed_or_unsafe_match",
        "official_result_quarantine_errors": ["result_boxes_not_in_participants:9"],
        "official_result_quarantine_source_urls": [
            "https://www.thedogs.com.au/racing/taree/2026-06-13/7"
        ],
        "official_result_quarantine_participant_boxes": [1, 2, 3, 4, 5, 6, 7, 8],
        "official_result_quarantine_result_boxes_not_in_participants": [9],
        "official_result_quarantine_reserve_substitution_diagnostic": {
            "classification": "possible_reserve_substitution_manual_review_required",
            "acceptance_status": "not_accepted_report_only",
            "candidate_reserve_boxes": [9],
            "scratched_participant_boxes": [1],
        },
    }
    assert daily_status[
        "best_aggregate_unified_top_official_result_missing_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert daily_status[
        "best_aggregate_unified_top_official_result_missing_races"
    ][0] == {
        "race_id": "Race 7 - TAREE - 2026-06-13",
        "official_result_quarantine_reason": "ingest_failed_or_unsafe_match",
        "official_result_quarantine_errors": ["result_boxes_not_in_participants:9"],
    }
    assert daily_status["backlog_unified_evidence_source_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert daily_status["backlog_unified_evidence_dataset_count"] == 56
    assert daily_status["backlog_unified_evidence_failed_dataset_count"] == 0
    assert daily_status["backlog_unified_evidence_row_count"] == 5108
    assert daily_status["backlog_unified_evidence_eligible_rows"] == 3872
    assert daily_status["backlog_unified_rejected_live_odds_candidate_count"] == 9
    assert daily_status[
        "backlog_unified_rows_with_rejected_live_odds_candidates"
    ] == 6
    assert daily_status[
        "backlog_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert daily_status["backlog_unified_sample_blocking_gap_count"] == 2
    assert daily_status["backlog_unified_top_gap_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert daily_status["backlog_unified_top_gap_races"] == daily_status[
        "best_aggregate_unified_top_gap_races"
    ]
    assert daily_status[
        "backlog_unified_top_official_result_missing_races"
    ] == daily_status["best_aggregate_unified_top_official_result_missing_races"]
    assert daily_status["backlog_unified_evidence_status_path"].endswith(
        "backlog_unified_evidence_datasets_status.json"
    )


def test_apply_autopilot_cycle_status_to_daily_status_surfaces_primary_cycle_fields(
    tmp_path,
):
    autopilot_status_path = tmp_path / "autopilot" / "DAILY_STATUS.json"
    stale_daemon_status = {
        "unified_evidence_dataset_status": None,
        "unified_evidence_eligible_rows": 0,
        "rolling_model_comparison_status": None,
        "rolling_model_comparison_sample_races": 0,
        "high_accuracy_refinement_status": None,
        "high_accuracy_promotion_pr_gate_status": None,
        "promotion_distance_status": None,
        "promotion_distance_sample_race_count": 0,
        "odds_research_gate_status": None,
        "rejoin_rolling_model_comparison_status": "SKIPPED",
    }
    autopilot_status = {
        "unified_evidence_dataset_status": "UNIFIED_EVIDENCE_DATASET_EMPTY",
        "unified_evidence_eligible_rows": 0,
        "best_aggregate_unified_evidence_status": (
            "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
        ),
        "best_aggregate_unified_evidence_eligible_rows": 189,
        "best_aggregate_unified_sample_blocking_gap_count": 5,
        "best_aggregate_unified_gap_action_counts": {
            "inspect_quarantined_official_result_runner_set": 4,
            "retry_official_result_capture_or_join": 1,
        },
        "best_aggregate_unified_gap_evidence_missing_reason_counts": {
            "official_result_quarantined_unsafe_match": 4,
            "official_result_missing_only": 1,
        },
        "best_aggregate_unified_top_gap_race_ids": [
            "Race 7 - TAREE - 2026-06-13",
            "Race 1 - TAREE - 2026-06-13",
        ],
        "best_aggregate_unified_top_gap_races": [
            {
                "race_id": "Race 7 - TAREE - 2026-06-13",
                "action": "inspect_quarantined_official_result_runner_set",
                "evidence_missing_reason": (
                    "official_result_quarantined_unsafe_match"
                ),
                "official_result_quarantine_errors": [
                    "result_boxes_not_in_participants:9"
                ],
            },
            {"race_id": "Race 1 - TAREE - 2026-06-13"},
        ],
        "best_aggregate_unified_top_official_result_missing_race_ids": [
            "Race 7 - TAREE - 2026-06-13",
            "Race 1 - TAREE - 2026-06-13",
        ],
        "best_aggregate_unified_top_official_result_missing_races": [
            {
                "race_id": "Race 7 - TAREE - 2026-06-13",
                "official_result_quarantine_errors": [
                    "result_boxes_not_in_participants:9"
                ],
            },
            {"race_id": "Race 1 - TAREE - 2026-06-13"},
        ],
        "rolling_model_comparison_status": "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW",
        "rolling_model_comparison_sample_races": 131,
        "rolling_model_comparison_best_candidate": "market_only_implied",
        "rolling_model_comparison_source_rejected_live_odds_candidate_count": 5,
        "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": 4,
        "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": {
            "odds_decimal_invalid": 2,
            "odds_source_url_missing": 3,
        },
        "high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
        "high_accuracy_promotion_pr_gate_status": "BLOCKED",
        "high_accuracy_unified_evidence_eligible_rows": 109,
        "reserve_substitution_preflight_status": (
            "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
        ),
        "reserve_substitution_preflight_candidate_count": 4,
        "reserve_substitution_preflight_ready_for_policy_review_count": 4,
        "reserve_substitution_preflight_blocked_candidate_count": 0,
        "reserve_substitution_preflight_readiness_blocker_counts": {},
        "reserve_substitution_preflight_dataset_join_blocker_counts": {
            "manual_policy_review_required_before_join": 4,
            "official_result_remains_quarantined": 4,
        },
        "reserve_substitution_preflight_ready_race_ids": [
            "Race 7 - TAREE - 2026-06-13",
        ],
        "reserve_substitution_preflight_blocked_race_ids": [],
        "reserve_substitution_preflight_report": (
            "artifacts/full_evidence_orchestration_20260525/"
            "official_result_reserve_substitution_preflight_x/"
            "official_result_reserve_substitution_preflight.json"
        ),
        "reserve_substitution_manual_review_status": (
            "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
        ),
        "reserve_substitution_manual_review_ready_candidate_count": 4,
        "reserve_substitution_manual_review_mapping_pair_count": 5,
        "reserve_substitution_manual_review_dataset_join_allowed": False,
        "reserve_substitution_manual_review_official_result_acceptance_allowed": False,
        "reserve_substitution_manual_review_db_write": False,
        "reserve_substitution_manual_review_blockers": [],
        "reserve_substitution_manual_review_ready_race_ids": [
            "Race 7 - TAREE - 2026-06-13",
        ],
        "reserve_substitution_manual_review_report": (
            "artifacts/full_evidence_orchestration_20260525/"
            "official_result_reserve_substitution_preflight_x/"
            "reserve_substitution_manual_review_packet.json"
        ),
        "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
        "promotion_distance_promotion_ready": False,
        "promotion_distance_blockers": [
            "no_candidate_passed_rank_first_accuracy_gate"
        ],
        "promotion_distance_sample_race_count": 131,
        "promotion_distance_sample_runner_rows": 927,
        "promotion_distance_source_rejected_live_odds_candidate_count": 5,
        "promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
        "promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
            "odds_decimal_invalid": 2,
            "odds_source_url_missing": 3,
        },
        "promotion_distance_source_exclusion_reason_counts": {
            "official_result_missing": 32,
        },
        "promotion_distance_source_odds_exclusion_reason_counts": {
            "strict_prejump_odds_missing": 6,
        },
        "promotion_distance_source_official_result_evidence_db_missing_race_ids": [
            "Race 7 - TAREE - 2026-06-13",
        ],
        "promotion_distance_source_official_result_evidence_db_requested_race_count": 7,
        "promotion_distance_source_official_result_evidence_db_races_with_rows": [
            "Race 5 - TAREE - 2026-06-13",
        ],
        "promotion_distance_source_official_result_runner_paths": [
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x/official_result_runners.jsonl",
        ],
        "promotion_distance_official_result_coverage_requested_race_count": 7,
        "promotion_distance_official_result_coverage_requested_race_count_source": (
            "deduped_requested_or_inferred_race_ids"
        ),
        "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": 4125,
        "promotion_distance_official_result_coverage_races_with_rows_count": 1,
        "promotion_distance_official_result_coverage_missing_race_count": 1,
        "promotion_distance_official_result_coverage_missing_exclusion_count": 32,
        "promotion_distance_official_result_runner_path_count": 1,
        "promotion_distance_official_result_runner_paths_source_field": (
            "rolling_sample.source_official_result_runner_paths"
        ),
        "promotion_distance_best_candidate_key": "market_only_implied",
        "promotion_distance_best_non_market_candidate_key": (
            "stage2_rf_calibrated"
        ),
        "promotion_distance_best_non_market_top1_margin_gap": -0.015,
        "promotion_distance_predeclared_residual_candidate_status": (
            "BELOW_DIRECTIONAL_FLOOR"
        ),
        "promotion_distance_predeclared_residual_triggered_race_count": 2,
        "promotion_distance_report": (
            "artifacts/full_evidence_orchestration_20260525/"
            "promotion_distance_report_x/promotion_distance_report.json"
        ),
        "timing_aligned_rerun_plan": (
            "artifacts/full_evidence_orchestration_20260525/"
            "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
        ),
        "timing_aligned_rerun_execution_status": (
            "artifacts/full_evidence_orchestration_20260525/"
            "shadow_autopilot_v1_x/"
            "timing_aligned_prediction_rerun_execution_status.json"
        ),
        "timing_aligned_prediction_rerun_plan_status": (
            "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
        ),
        "timing_aligned_prediction_rerun_plan_hard_stops": [
            "timing_aligned_rerun_window_already_closed_after_jump"
        ],
        "timing_aligned_prediction_rerun_execution_status": (
            "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
        ),
        "timing_aligned_prediction_rerun_execution_hard_stops": [
            "timing_aligned_rerun_window_already_closed_after_jump"
        ],
        "timing_aligned_prediction_rerun_execution_performed": False,
        "timing_aligned_prediction_rerun_output_dir": (
            "artifacts/full_evidence_orchestration_20260525/"
            "daily_race_ingest_shadow_x_timing_aligned_rerun"
        ),
        "timing_aligned_prediction_rerun_odds_snapshot_dir": (
            "artifacts/full_evidence_orchestration_20260525/"
            "shadow_odds_snapshot_x_timing_aligned_rerun"
        ),
        "timing_aligned_prediction_rerun_odds_snapshot_status": None,
        "timing_aligned_prediction_rerun_returncode": None,
        "odds_research_gate_status": None,
    }

    daemon.apply_autopilot_cycle_status_to_daily_status(
        stale_daemon_status,
        autopilot_daily_status_path=autopilot_status_path,
        autopilot_daily_status=autopilot_status,
    )

    assert stale_daemon_status["autopilot_cycle_daily_status_path"].endswith(
        "autopilot/DAILY_STATUS.json"
    )
    assert stale_daemon_status["unified_evidence_dataset_status"] == (
        "UNIFIED_EVIDENCE_DATASET_EMPTY"
    )
    assert stale_daemon_status["best_aggregate_unified_evidence_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert stale_daemon_status["best_aggregate_unified_evidence_eligible_rows"] == 189
    assert stale_daemon_status["best_aggregate_unified_sample_blocking_gap_count"] == 5
    assert stale_daemon_status["best_aggregate_unified_gap_action_counts"] == {
        "inspect_quarantined_official_result_runner_set": 4,
        "retry_official_result_capture_or_join": 1,
    }
    assert stale_daemon_status[
        "best_aggregate_unified_gap_evidence_missing_reason_counts"
    ] == {
        "official_result_quarantined_unsafe_match": 4,
        "official_result_missing_only": 1,
    }
    assert stale_daemon_status["best_aggregate_unified_top_gap_races"][0] == {
        "race_id": "Race 7 - TAREE - 2026-06-13",
        "action": "inspect_quarantined_official_result_runner_set",
        "evidence_missing_reason": "official_result_quarantined_unsafe_match",
        "official_result_quarantine_errors": ["result_boxes_not_in_participants:9"],
    }
    assert stale_daemon_status[
        "best_aggregate_unified_top_official_result_missing_races"
    ][0] == {
        "race_id": "Race 7 - TAREE - 2026-06-13",
        "official_result_quarantine_errors": ["result_boxes_not_in_participants:9"],
    }
    assert stale_daemon_status["rolling_model_comparison_status"] == (
        "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    )
    assert stale_daemon_status["rolling_model_comparison_sample_races"] == 131
    assert stale_daemon_status["rolling_model_comparison_best_candidate"] == (
        "market_only_implied"
    )
    assert stale_daemon_status[
        "rolling_model_comparison_source_rejected_live_odds_candidate_count"
    ] == 5
    assert stale_daemon_status[
        "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
    ] == 4
    assert stale_daemon_status[
        "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert stale_daemon_status["high_accuracy_refinement_status"] == (
        "BLOCKED_KEEP_BASELINE"
    )
    assert stale_daemon_status["high_accuracy_promotion_pr_gate_status"] == "BLOCKED"
    assert stale_daemon_status["high_accuracy_unified_evidence_eligible_rows"] == 109
    assert stale_daemon_status["reserve_substitution_preflight_status"] == (
        "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
    )
    assert stale_daemon_status[
        "reserve_substitution_preflight_ready_for_policy_review_count"
    ] == 4
    assert stale_daemon_status[
        "reserve_substitution_preflight_dataset_join_blocker_counts"
    ] == {
        "manual_policy_review_required_before_join": 4,
        "official_result_remains_quarantined": 4,
    }
    assert stale_daemon_status["reserve_substitution_preflight_ready_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
    ]
    assert stale_daemon_status["reserve_substitution_manual_review_status"] == (
        "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    )
    assert stale_daemon_status[
        "reserve_substitution_manual_review_ready_candidate_count"
    ] == 4
    assert stale_daemon_status[
        "reserve_substitution_manual_review_mapping_pair_count"
    ] == 5
    assert (
        stale_daemon_status[
            "reserve_substitution_manual_review_dataset_join_allowed"
        ]
        is False
    )
    assert stale_daemon_status["promotion_distance_status"] == (
        "PROMOTION_DISTANCE_BLOCKED"
    )
    assert stale_daemon_status["timing_aligned_prediction_rerun_plan_status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
    )
    assert stale_daemon_status[
        "timing_aligned_prediction_rerun_execution_status"
    ] == "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
    assert stale_daemon_status["timing_aligned_prediction_rerun_execution_hard_stops"] == [
        "timing_aligned_rerun_window_already_closed_after_jump"
    ]
    assert stale_daemon_status["timing_aligned_prediction_rerun_odds_snapshot_dir"].endswith(
        "shadow_odds_snapshot_x_timing_aligned_rerun"
    )
    assert stale_daemon_status["promotion_distance_promotion_ready"] is False
    assert stale_daemon_status["promotion_distance_sample_race_count"] == 131
    assert stale_daemon_status["promotion_distance_sample_runner_rows"] == 927
    assert stale_daemon_status[
        "promotion_distance_source_rejected_live_odds_candidate_count"
    ] == 5
    assert stale_daemon_status[
        "promotion_distance_source_rows_with_rejected_live_odds_candidates"
    ] == 4
    assert stale_daemon_status[
        "promotion_distance_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert stale_daemon_status[
        "promotion_distance_source_exclusion_reason_counts"
    ] == {
        "official_result_missing": 32,
    }
    assert stale_daemon_status[
        "promotion_distance_source_odds_exclusion_reason_counts"
    ] == {
        "strict_prejump_odds_missing": 6,
    }
    assert stale_daemon_status[
        "promotion_distance_source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 7 - TAREE - 2026-06-13"]
    assert (
        stale_daemon_status[
            "promotion_distance_source_official_result_evidence_db_requested_race_count"
        ]
        == 7
    )
    assert stale_daemon_status[
        "promotion_distance_source_official_result_evidence_db_races_with_rows"
    ] == ["Race 5 - TAREE - 2026-06-13"]
    assert stale_daemon_status[
        "promotion_distance_source_official_result_runner_paths"
    ] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_x/official_result_runners.jsonl",
    ]
    assert (
        stale_daemon_status[
            "promotion_distance_official_result_coverage_requested_race_count"
        ]
        == 7
    )
    assert (
        stale_daemon_status[
            "promotion_distance_official_result_coverage_requested_race_count_source"
        ]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        stale_daemon_status[
            "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
        ]
        == 4125
    )
    assert (
        stale_daemon_status[
            "promotion_distance_official_result_coverage_races_with_rows_count"
        ]
        == 1
    )
    assert (
        stale_daemon_status[
            "promotion_distance_official_result_coverage_missing_race_count"
        ]
        == 1
    )
    assert (
        stale_daemon_status[
            "promotion_distance_official_result_coverage_missing_exclusion_count"
        ]
        == 32
    )
    assert stale_daemon_status["promotion_distance_official_result_runner_path_count"] == 1
    assert stale_daemon_status[
        "promotion_distance_official_result_runner_paths_source_field"
    ] == "rolling_sample.source_official_result_runner_paths"
    assert stale_daemon_status["promotion_distance_best_candidate_key"] == (
        "market_only_implied"
    )
    assert stale_daemon_status[
        "promotion_distance_best_non_market_candidate_key"
    ] == "stage2_rf_calibrated"
    assert stale_daemon_status["promotion_distance_blockers"] == [
        "no_candidate_passed_rank_first_accuracy_gate"
    ]
    assert stale_daemon_status["promotion_distance_report"].endswith(
        "promotion_distance_report.json"
    )
    assert stale_daemon_status["timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert stale_daemon_status["timing_aligned_rerun_execution_status"].endswith(
        "timing_aligned_prediction_rerun_execution_status.json"
    )
    assert stale_daemon_status["rejoin_rolling_model_comparison_status"] == "SKIPPED"


def test_autopilot_cycle_state_fields_surface_primary_cycle_gate_context():
    state_fields = daemon.autopilot_cycle_state_fields(
        {
            "autopilot_cycle_daily_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/DAILY_STATUS.json"
            ),
            "unified_evidence_dataset_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_dataset_rows": 927,
            "unified_evidence_dataset_races": 131,
            "unified_evidence_eligible_rows": 856,
            "best_aggregate_unified_evidence_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/backlog_unified_evidence_datasets_status.json"
            ),
            "best_aggregate_unified_evidence_status": (
                "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
            ),
            "best_aggregate_unified_evidence_dataset_count": 56,
            "best_aggregate_unified_evidence_row_count": 5108,
            "best_aggregate_unified_evidence_eligible_rows": 3872,
            "best_aggregate_unified_evidence_artifact_odds_rows_seen": 114,
            "best_aggregate_unified_evidence_artifact_odds_rows_accepted": 14,
            "best_aggregate_unified_evidence_artifact_odds_rows_rejected": 100,
            "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts": {
                "odds_after_jump": 73,
                "runner_set_mismatch": 27,
            },
            "best_aggregate_unified_rejected_live_odds_candidate_count": 9,
            "best_aggregate_unified_rows_with_rejected_live_odds_candidates": 6,
            "best_aggregate_unified_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 4,
                "odds_source_url_missing": 5,
            },
            "best_aggregate_unified_sample_blocking_gap_count": 2,
            "best_aggregate_unified_gap_action_counts": {
                "retry_official_result_capture_or_join": 2,
            },
            "best_aggregate_unified_gap_evidence_missing_reason_counts": {
                "official_result_missing_only": 2,
            },
            "best_aggregate_unified_top_gap_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "best_aggregate_unified_top_gap_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "action": "inspect_quarantined_official_result_runner_set",
                    "evidence_missing_reason": (
                        "official_result_quarantined_unsafe_match"
                    ),
                    "official_result_quarantine_errors": [
                        "result_boxes_not_in_participants:9"
                    ],
                }
            ],
            "best_aggregate_unified_top_official_result_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "best_aggregate_unified_top_official_result_missing_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "official_result_quarantine_errors": [
                        "result_boxes_not_in_participants:9"
                    ],
                }
            ],
            "rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "rolling_model_comparison_sample_races": 131,
            "rolling_model_comparison_sample_runner_rows": 927,
            "rolling_model_comparison_best_candidate": "market_only_implied",
            "rolling_model_comparison_source_rejected_live_odds_candidate_count": 5,
            "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": 4,
            "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "rolling_model_comparison_blockers": [
                "no_candidate_passed_rank_first_accuracy_gate"
            ],
            "high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "high_accuracy_promotion_pr_gate_status": "BLOCKED",
            "high_accuracy_unified_evidence_eligible_rows": 856,
            "timing_aligned_rerun_plan": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
            ),
            "timing_aligned_rerun_execution_status": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/"
                "timing_aligned_prediction_rerun_execution_status.json"
            ),
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
            "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_distance_promotion_ready": False,
            "promotion_distance_blockers": [
                "best_non_market_top1_margin_below_target"
            ],
            "promotion_distance_sample_race_count": 131,
            "promotion_distance_sample_runner_rows": 927,
            "promotion_distance_source_rejected_live_odds_candidate_count": 5,
            "promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
            "promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "promotion_distance_source_exclusion_reason_counts": {
                "official_result_missing": 32,
            },
            "promotion_distance_source_odds_exclusion_reason_counts": {
                "strict_prejump_odds_missing": 6,
            },
            "promotion_distance_source_official_result_evidence_db_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "promotion_distance_source_official_result_evidence_db_requested_race_count": 7,
            "promotion_distance_source_official_result_evidence_db_races_with_rows": [
                "Race 5 - TAREE - 2026-06-13",
            ],
            "promotion_distance_source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/official_result_runners.jsonl",
            ],
            "promotion_distance_official_result_coverage_requested_race_count": 7,
            "promotion_distance_official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": 4125,
            "promotion_distance_official_result_coverage_races_with_rows_count": 1,
            "promotion_distance_official_result_coverage_missing_race_count": 1,
            "promotion_distance_official_result_coverage_missing_exclusion_count": 32,
            "promotion_distance_official_result_runner_path_count": 1,
            "promotion_distance_official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
            "promotion_distance_best_candidate_key": "market_only_implied",
            "promotion_distance_best_non_market_candidate_key": (
                "stage2_rf_calibrated"
            ),
            "promotion_distance_predeclared_residual_candidate_status": (
                "BELOW_DIRECTIONAL_FLOOR"
            ),
            "promotion_distance_predeclared_residual_triggered_race_count": 2,
            "promotion_distance_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "promotion_distance_report_x/promotion_distance_report.json"
            ),
            "timing_aligned_prediction_rerun_plan_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
            ),
            "timing_aligned_prediction_rerun_plan_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
            ),
            "timing_aligned_prediction_rerun_execution_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_performed": False,
            "timing_aligned_prediction_rerun_output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_race_ingest_shadow_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_status": None,
            "timing_aligned_prediction_rerun_returncode": None,
        }
    )

    assert state_fields["last_autopilot_cycle_daily_status_path"].endswith(
        "DAILY_STATUS.json"
    )
    assert state_fields["last_unified_evidence_dataset_status"] == (
        "UNIFIED_EVIDENCE_DATASET_BUILT"
    )
    assert state_fields["last_best_aggregate_unified_evidence_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert state_fields["last_best_aggregate_unified_evidence_eligible_rows"] == 3872
    assert state_fields[
        "last_best_aggregate_unified_rejected_live_odds_candidate_count"
    ] == 9
    assert state_fields[
        "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates"
    ] == 6
    assert state_fields[
        "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert state_fields["last_best_aggregate_unified_sample_blocking_gap_count"] == 2
    assert state_fields["last_best_aggregate_unified_gap_action_counts"] == {
        "retry_official_result_capture_or_join": 2,
    }
    assert state_fields[
        "last_best_aggregate_unified_gap_evidence_missing_reason_counts"
    ] == {"official_result_missing_only": 2}
    assert state_fields["last_best_aggregate_unified_top_gap_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert state_fields["last_best_aggregate_unified_top_gap_races"] == [
        {
            "race_id": "Race 7 - TAREE - 2026-06-13",
            "action": "inspect_quarantined_official_result_runner_set",
            "evidence_missing_reason": "official_result_quarantined_unsafe_match",
            "official_result_quarantine_errors": [
                "result_boxes_not_in_participants:9"
            ],
        }
    ]
    assert state_fields[
        "last_best_aggregate_unified_top_official_result_missing_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert state_fields[
        "last_best_aggregate_unified_top_official_result_missing_races"
    ] == [
        {
            "race_id": "Race 7 - TAREE - 2026-06-13",
            "official_result_quarantine_errors": [
                "result_boxes_not_in_participants:9"
            ],
        }
    ]
    assert state_fields["last_rolling_model_comparison_status"] == (
        "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    )
    assert state_fields[
        "last_rolling_model_comparison_source_rejected_live_odds_candidate_count"
    ] == 5
    assert state_fields[
        "last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
    ] == 4
    assert state_fields[
        "last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert state_fields["last_high_accuracy_refinement_status"] == (
        "BLOCKED_KEEP_BASELINE"
    )
    assert state_fields["last_reserve_substitution_policy_impact_status"] == (
        "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    )
    assert (
        state_fields[
            "last_reserve_substitution_policy_impact_ready_candidate_count"
        ]
        == 4
    )
    assert (
        state_fields["last_reserve_substitution_policy_impact_mapping_pair_count"]
        == 5
    )
    assert (
        state_fields[
            "last_reserve_substitution_policy_impact_potential_runner_rows_blocked"
        ]
        == 32
    )
    assert (
        state_fields[
            "last_reserve_substitution_policy_impact_dataset_join_allowed"
        ]
        is False
    )
    assert (
        state_fields[
            "last_reserve_substitution_policy_impact_official_result_acceptance_allowed"
        ]
        is False
    )
    assert state_fields["last_reserve_substitution_policy_impact_db_write"] is False
    assert state_fields["last_promotion_distance_status"] == (
        "PROMOTION_DISTANCE_BLOCKED"
    )
    assert state_fields["last_promotion_distance_promotion_ready"] is False
    assert state_fields["last_promotion_distance_sample_race_count"] == 131
    assert (
        state_fields[
            "last_promotion_distance_source_rejected_live_odds_candidate_count"
        ]
        == 5
    )
    assert (
        state_fields[
            "last_promotion_distance_source_rows_with_rejected_live_odds_candidates"
        ]
        == 4
    )
    assert state_fields[
        "last_promotion_distance_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert state_fields["last_promotion_distance_best_candidate_key"] == (
        "market_only_implied"
    )
    assert state_fields[
        "last_promotion_distance_best_non_market_candidate_key"
    ] == "stage2_rf_calibrated"
    assert state_fields["last_promotion_distance_blockers"] == [
        "best_non_market_top1_margin_below_target"
    ]
    assert state_fields["last_promotion_distance_report"].endswith(
        "promotion_distance_report.json"
    )
    assert state_fields["last_timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert state_fields["last_timing_aligned_rerun_execution_status"].endswith(
        "timing_aligned_prediction_rerun_execution_status.json"
    )
    assert state_fields["last_timing_aligned_prediction_rerun_plan_status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
    )
    assert state_fields["last_timing_aligned_prediction_rerun_plan_hard_stops"] == [
        "timing_aligned_rerun_window_already_closed_after_jump"
    ]
    assert state_fields["last_timing_aligned_prediction_rerun_execution_status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
    )
    assert state_fields["last_timing_aligned_prediction_rerun_execution_hard_stops"] == [
        "timing_aligned_rerun_window_already_closed_after_jump"
    ]
    assert state_fields["last_timing_aligned_prediction_rerun_execution_performed"] is False
    assert state_fields["last_timing_aligned_prediction_rerun_output_dir"].endswith(
        "daily_race_ingest_shadow_x_timing_aligned_rerun"
    )
    assert state_fields[
        "last_timing_aligned_prediction_rerun_odds_snapshot_dir"
    ].endswith("shadow_odds_snapshot_x_timing_aligned_rerun")
    assert state_fields["last_timing_aligned_prediction_rerun_odds_snapshot_status"] is None
    assert state_fields["last_timing_aligned_prediction_rerun_returncode"] is None


def test_rejoin_unified_state_fields_preserve_rejected_odds_row_context():
    state_fields = daemon.rejoin_unified_state_fields(
        {
            "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "dataset_count": 2,
            "unified_evidence_eligible_rows": 16,
            "rejected_live_odds_candidate_count": 8,
            "rows_with_rejected_live_odds_candidates": 6,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 3,
                "odds_source_url_missing": 5,
            },
            "status_reason": "all_rejoin_unified_evidence_dataset_candidates_skipped",
            "evaluated_dataset_candidate_count": 3,
            "skipped_dataset_count": 3,
            "skip_reason_counts": {
                "joined_shadow_predictions_already_converted": 2,
                "safe_joined_race_count_zero": 1,
            },
            "failure_reason_counts": {},
            "join_eligibility_preview_dataset_count": 1,
            "join_eligibility_preview_unified_eligible_rows": 27,
            "join_eligibility_preview_packet_accepted_races": 4,
            "join_eligibility_preview_packet_present_races": 4,
        }
    )

    assert state_fields["last_rejoin_unified_evidence_status"] == (
        "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert state_fields["last_rejoin_unified_evidence_status_reason"] == (
        "all_rejoin_unified_evidence_dataset_candidates_skipped"
    )
    assert state_fields["last_rejoin_unified_evidence_dataset_count"] == 2
    assert state_fields["last_rejoin_unified_evidence_eligible_rows"] == 16
    assert state_fields["last_rejoin_unified_evaluated_candidate_count"] == 3
    assert state_fields["last_rejoin_unified_skipped_dataset_count"] == 3
    assert state_fields["last_rejoin_unified_skip_reason_counts"] == {
        "joined_shadow_predictions_already_converted": 2,
        "safe_joined_race_count_zero": 1,
    }
    assert state_fields["last_rejoin_unified_failure_reason_counts"] == {}
    assert state_fields[
        "last_rejoin_unified_rejected_live_odds_candidate_count"
    ] == 8
    assert state_fields[
        "last_rejoin_unified_rows_with_rejected_live_odds_candidates"
    ] == 6
    assert state_fields[
        "last_rejoin_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 5,
    }
    assert state_fields["last_join_eligibility_preview_dataset_count"] == 1
    assert state_fields["last_join_eligibility_preview_unified_eligible_rows"] == 27
    assert state_fields[
        "last_join_eligibility_preview_packet_accepted_races"
    ] == 4
    assert state_fields["last_join_eligibility_preview_packet_present_races"] == 4


def test_rejoin_high_accuracy_timing_source_fields_are_prefixed():
    fields = daemon.rejoin_high_accuracy_timing_source_fields(
        {
            "timing_aligned_rerun_plan": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
            ),
            "timing_aligned_rerun_execution_status": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/"
                "timing_aligned_prediction_rerun_execution_status.json"
            ),
            "reserve_substitution_preflight_status": (
                "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
            ),
            "reserve_substitution_preflight_ready_for_policy_review_count": 4,
            "reserve_substitution_preflight_dataset_join_blocker_counts": {
                "manual_policy_review_required_before_join": 4,
                "official_result_remains_quarantined": 4,
            },
            "reserve_substitution_preflight_ready_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "reserve_substitution_preflight_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "official_result_reserve_substitution_preflight_x/"
                "official_result_reserve_substitution_preflight.json"
            ),
            "reserve_substitution_manual_review_status": (
                "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
            ),
            "reserve_substitution_manual_review_ready_candidate_count": 4,
            "reserve_substitution_manual_review_mapping_pair_count": 5,
            "reserve_substitution_manual_review_dataset_join_allowed": False,
            "reserve_substitution_manual_review_official_result_acceptance_allowed": False,
            "reserve_substitution_manual_review_db_write": False,
            "reserve_substitution_manual_review_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "official_result_reserve_substitution_preflight_x/"
                "reserve_substitution_manual_review_packet.json"
            ),
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": 4,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
            "reserve_substitution_policy_impact_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "official_result_reserve_substitution_preflight_x/"
                "reserve_substitution_policy_impact_preview.json"
            ),
        },
        prefix="rejoin_high_accuracy_",
    )

    assert fields["rejoin_high_accuracy_timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert fields[
        "rejoin_high_accuracy_timing_aligned_rerun_execution_status"
    ].endswith("timing_aligned_prediction_rerun_execution_status.json")
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_preflight_status"
    ] == "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_preflight_ready_for_policy_review_count"
    ] == 4
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_preflight_dataset_join_blocker_counts"
    ] == {
        "manual_policy_review_required_before_join": 4,
        "official_result_remains_quarantined": 4,
    }
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_preflight_report"
    ].endswith("official_result_reserve_substitution_preflight.json")
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_manual_review_status"
    ] == "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_manual_review_ready_candidate_count"
    ] == 4
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_manual_review_mapping_pair_count"
    ] == 5
    assert (
        fields[
            "rejoin_high_accuracy_reserve_substitution_manual_review_dataset_join_allowed"
        ]
        is False
    )
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_manual_review_report"
    ].endswith("reserve_substitution_manual_review_packet.json")
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_policy_impact_status"
    ] == "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_policy_impact_ready_candidate_count"
    ] == 4
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_policy_impact_mapping_pair_count"
    ] == 5
    assert (
        fields[
            "rejoin_high_accuracy_reserve_substitution_policy_impact_potential_runner_rows_blocked"
        ]
        == 32
    )
    assert (
        fields[
            "rejoin_high_accuracy_reserve_substitution_policy_impact_dataset_join_allowed"
        ]
        is False
    )
    assert fields[
        "rejoin_high_accuracy_reserve_substitution_policy_impact_report"
    ].endswith("reserve_substitution_policy_impact_preview.json")


def test_annotate_rejoin_skipped_status_preserves_upstream_skip_context():
    rejoin_status = {
        "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_SKIPPED",
        "status_reason": "all_rejoin_unified_evidence_dataset_candidates_skipped",
        "dataset_count": 0,
        "evaluated_dataset_candidate_count": 2,
        "skipped_dataset_count": 2,
        "skip_reason_counts": {"safe_joined_race_count_zero": 2},
        "failure_reason_counts": {},
    }

    skipped_status = daemon.annotate_rejoin_skipped_status(
        {
            "status": "SKIPPED",
            "skipped_reason": "rejoin_unified_evidence_reports_missing",
        },
        rejoin_status,
    )
    built_status = daemon.annotate_rejoin_skipped_status(
        {"status": "ROLLING_MODEL_COMPARISON_BUILT"},
        rejoin_status,
    )

    assert skipped_status["skipped_reason"] == "rejoin_unified_evidence_reports_missing"
    assert skipped_status["rejoin_unified_evidence_status"] == (
        "REJOIN_UNIFIED_EVIDENCE_DATASETS_SKIPPED"
    )
    assert skipped_status["rejoin_unified_evidence_status_reason"] == (
        "all_rejoin_unified_evidence_dataset_candidates_skipped"
    )
    assert skipped_status["rejoin_unified_evidence_dataset_count"] == 0
    assert skipped_status["rejoin_unified_evidence_evaluated_candidate_count"] == 2
    assert skipped_status["rejoin_unified_evidence_skipped_dataset_count"] == 2
    assert skipped_status["rejoin_unified_evidence_skip_reason_counts"] == {
        "safe_joined_race_count_zero": 2
    }
    assert skipped_status["rejoin_unified_evidence_failure_reason_counts"] == {}
    assert built_status == {"status": "ROLLING_MODEL_COMPARISON_BUILT"}


def test_rejoin_unified_operational_diagnostic_fields_preserve_skip_context():
    fields = daemon.rejoin_unified_operational_diagnostic_fields(
        {
            "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_SKIPPED",
            "status_reason": "all_rejoin_unified_evidence_dataset_candidates_skipped",
            "evaluated_dataset_candidate_count": 2,
            "skipped_dataset_count": 2,
            "skip_reason_counts": {"safe_joined_race_count_zero": 2},
            "failure_reason_counts": {},
        }
    )

    assert fields["rejoin_unified_evidence_status_reason"] == (
        "all_rejoin_unified_evidence_dataset_candidates_skipped"
    )
    assert fields["rejoin_unified_evidence_evaluated_candidate_count"] == 2
    assert fields["rejoin_unified_evidence_skipped_dataset_count"] == 2
    assert fields["rejoin_unified_evidence_skip_reason_counts"] == {
        "safe_joined_race_count_zero": 2
    }
    assert fields["rejoin_unified_evidence_failure_reason_counts"] == {}


def test_autopilot_cycle_operational_fields_surface_primary_cycle_gate_context():
    operational_fields = daemon.autopilot_cycle_operational_fields(
        {
            "autopilot_cycle_daily_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/DAILY_STATUS.json"
            ),
            "unified_evidence_dataset_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_dataset_rows": 927,
            "unified_evidence_dataset_races": 131,
            "unified_evidence_eligible_rows": 856,
            "best_aggregate_unified_evidence_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/backlog_unified_evidence_datasets_status.json"
            ),
            "best_aggregate_unified_evidence_status": (
                "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
            ),
            "best_aggregate_unified_evidence_dataset_count": 56,
            "best_aggregate_unified_evidence_row_count": 5108,
            "best_aggregate_unified_evidence_eligible_rows": 3872,
            "best_aggregate_unified_evidence_artifact_odds_rows_seen": 114,
            "best_aggregate_unified_evidence_artifact_odds_rows_accepted": 14,
            "best_aggregate_unified_evidence_artifact_odds_rows_rejected": 100,
            "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts": {
                "odds_after_jump": 73,
                "runner_set_mismatch": 27,
            },
            "best_aggregate_unified_rejected_live_odds_candidate_count": 9,
            "best_aggregate_unified_rows_with_rejected_live_odds_candidates": 6,
            "best_aggregate_unified_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 4,
                "odds_source_url_missing": 5,
            },
            "best_aggregate_unified_sample_blocking_gap_count": 2,
            "best_aggregate_unified_gap_action_counts": {
                "retry_official_result_capture_or_join": 2,
            },
            "best_aggregate_unified_gap_evidence_missing_reason_counts": {
                "official_result_missing_only": 2,
            },
            "best_aggregate_unified_top_gap_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "best_aggregate_unified_top_gap_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "action": "inspect_quarantined_official_result_runner_set",
                    "evidence_missing_reason": (
                        "official_result_quarantined_unsafe_match"
                    ),
                    "official_result_quarantine_errors": [
                        "result_boxes_not_in_participants:9"
                    ],
                }
            ],
            "best_aggregate_unified_top_official_result_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "best_aggregate_unified_top_official_result_missing_races": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "official_result_quarantine_errors": [
                        "result_boxes_not_in_participants:9"
                    ],
                }
            ],
            "rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "rolling_model_comparison_sample_races": 131,
            "rolling_model_comparison_best_candidate": "market_only_implied",
            "rolling_model_comparison_source_rejected_live_odds_candidate_count": 5,
            "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": 4,
            "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "rolling_model_comparison_blockers": [
                "no_candidate_passed_rank_first_accuracy_gate"
            ],
            "high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "high_accuracy_promotion_pr_gate_status": "BLOCKED",
            "timing_aligned_rerun_plan": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
            ),
            "timing_aligned_rerun_execution_status": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/"
                "timing_aligned_prediction_rerun_execution_status.json"
            ),
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
            "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_distance_promotion_ready": False,
            "promotion_distance_blockers": [
                "best_non_market_top1_margin_below_target"
            ],
            "promotion_distance_sample_race_count": 131,
            "promotion_distance_source_rejected_live_odds_candidate_count": 5,
            "promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
            "promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "promotion_distance_source_exclusion_reason_counts": {
                "official_result_missing": 32,
            },
            "promotion_distance_source_odds_exclusion_reason_counts": {
                "strict_prejump_odds_missing": 6,
            },
            "promotion_distance_source_official_result_evidence_db_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "promotion_distance_source_official_result_evidence_db_requested_race_count": 7,
            "promotion_distance_source_official_result_evidence_db_races_with_rows": [
                "Race 5 - TAREE - 2026-06-13",
            ],
            "promotion_distance_source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/official_result_runners.jsonl",
            ],
            "promotion_distance_official_result_coverage_requested_race_count": 7,
            "promotion_distance_official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": 4125,
            "promotion_distance_official_result_coverage_races_with_rows_count": 1,
            "promotion_distance_official_result_coverage_missing_race_count": 1,
            "promotion_distance_official_result_coverage_missing_exclusion_count": 32,
            "promotion_distance_official_result_runner_path_count": 1,
            "promotion_distance_official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
            "promotion_distance_best_candidate_key": "market_only_implied",
            "promotion_distance_best_non_market_candidate_key": (
                "stage2_rf_calibrated"
            ),
            "promotion_distance_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "promotion_distance_report_x/promotion_distance_report.json"
            ),
            "timing_aligned_prediction_rerun_plan_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
            ),
            "timing_aligned_prediction_rerun_plan_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
            ),
            "timing_aligned_prediction_rerun_execution_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_performed": False,
            "timing_aligned_prediction_rerun_output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_race_ingest_shadow_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_status": None,
            "timing_aligned_prediction_rerun_returncode": None,
        }
    )

    assert operational_fields["autopilot_cycle_daily_status_path"].endswith(
        "DAILY_STATUS.json"
    )
    assert operational_fields["autopilot_cycle_unified_evidence_status"] == (
        "UNIFIED_EVIDENCE_DATASET_BUILT"
    )
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_evidence_status_path"
    ].endswith("backlog_unified_evidence_datasets_status.json")
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_evidence_status"
    ] == "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_evidence_dataset_count"
        ]
        == 56
    )
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_evidence_row_count"
        ]
        == 5108
    )
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_evidence_eligible_rows"
        ]
        == 3872
    )
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_seen"
        ]
        == 114
    )
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_accepted"
        ]
        == 14
    )
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rows_rejected"
        ]
        == 100
    )
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts"
    ] == {
        "odds_after_jump": 73,
        "runner_set_mismatch": 27,
    }
    assert (
        operational_fields[
            "autopilot_cycle_best_aggregate_unified_sample_blocking_gap_count"
        ]
        == 2
    )
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_top_gap_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_top_gap_races"
    ] == [
        {
            "race_id": "Race 7 - TAREE - 2026-06-13",
            "action": "inspect_quarantined_official_result_runner_set",
            "evidence_missing_reason": "official_result_quarantined_unsafe_match",
            "official_result_quarantine_errors": [
                "result_boxes_not_in_participants:9"
            ],
        }
    ]
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_top_official_result_missing_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 4 - TAREE - 2026-06-13",
    ]
    assert operational_fields[
        "autopilot_cycle_best_aggregate_unified_top_official_result_missing_races"
    ] == [
        {
            "race_id": "Race 7 - TAREE - 2026-06-13",
            "official_result_quarantine_errors": [
                "result_boxes_not_in_participants:9"
            ],
        }
    ]
    assert operational_fields[
        "autopilot_cycle_rolling_model_comparison_status"
    ] == "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    assert operational_fields[
        "autopilot_cycle_rolling_model_comparison_source_rejected_live_odds_candidate_count"
    ] == 5
    assert operational_fields[
        "autopilot_cycle_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
    ] == 4
    assert operational_fields[
        "autopilot_cycle_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert operational_fields[
        "autopilot_cycle_high_accuracy_refinement_status"
    ] == "BLOCKED_KEEP_BASELINE"
    assert operational_fields[
        "autopilot_cycle_reserve_substitution_policy_impact_status"
    ] == "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    assert (
        operational_fields[
            "autopilot_cycle_reserve_substitution_policy_impact_ready_candidate_count"
        ]
        == 4
    )
    assert (
        operational_fields[
            "autopilot_cycle_reserve_substitution_policy_impact_mapping_pair_count"
        ]
        == 5
    )
    assert (
        operational_fields[
            "autopilot_cycle_reserve_substitution_policy_impact_potential_runner_rows_blocked"
        ]
        == 32
    )
    assert (
        operational_fields[
            "autopilot_cycle_reserve_substitution_policy_impact_dataset_join_allowed"
        ]
        is False
    )
    assert (
        operational_fields[
            "autopilot_cycle_reserve_substitution_policy_impact_official_result_acceptance_allowed"
        ]
        is False
    )
    assert (
        operational_fields[
            "autopilot_cycle_reserve_substitution_policy_impact_db_write"
        ]
        is False
    )
    assert operational_fields[
        "autopilot_cycle_promotion_distance_status"
    ] == "PROMOTION_DISTANCE_BLOCKED"
    assert operational_fields[
        "autopilot_cycle_promotion_distance_promotion_ready"
    ] is False
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_source_rejected_live_odds_candidate_count"
        ]
        == 5
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_source_rows_with_rejected_live_odds_candidates"
        ]
        == 4
    )
    assert operational_fields[
        "autopilot_cycle_promotion_distance_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_coverage_requested_race_count"
        ]
        == 7
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_coverage_requested_race_count_source"
        ]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
        ]
        == 4125
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_coverage_races_with_rows_count"
        ]
        == 1
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_coverage_missing_race_count"
        ]
        == 1
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_coverage_missing_exclusion_count"
        ]
        == 32
    )
    assert (
        operational_fields[
            "autopilot_cycle_promotion_distance_official_result_runner_path_count"
        ]
        == 1
    )
    assert operational_fields[
        "autopilot_cycle_promotion_distance_official_result_runner_paths_source_field"
    ] == "rolling_sample.source_official_result_runner_paths"
    assert operational_fields[
        "autopilot_cycle_promotion_distance_best_non_market_candidate_key"
    ] == "stage2_rf_calibrated"
    assert operational_fields["autopilot_cycle_promotion_distance_blockers"] == [
        "best_non_market_top1_margin_below_target"
    ]
    assert operational_fields["autopilot_cycle_promotion_distance_report"].endswith(
        "promotion_distance_report.json"
    )
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert operational_fields[
        "autopilot_cycle_timing_aligned_rerun_execution_status_path"
    ].endswith("timing_aligned_prediction_rerun_execution_status.json")
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_plan_status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
    )
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_plan_hard_stops"] == [
        "timing_aligned_rerun_window_already_closed_after_jump"
    ]
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_execution_status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
    )
    assert operational_fields[
        "autopilot_cycle_timing_aligned_rerun_execution_hard_stops"
    ] == ["timing_aligned_rerun_window_already_closed_after_jump"]
    assert operational_fields[
        "autopilot_cycle_timing_aligned_rerun_execution_performed"
    ] is False
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_output_dir"].endswith(
        "daily_race_ingest_shadow_x_timing_aligned_rerun"
    )
    assert operational_fields[
        "autopilot_cycle_timing_aligned_rerun_odds_snapshot_dir"
    ].endswith("shadow_odds_snapshot_x_timing_aligned_rerun")
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_odds_snapshot_status"] is None
    assert operational_fields["autopilot_cycle_timing_aligned_rerun_returncode"] is None


def test_autonomous_official_result_operational_fields_prefer_daily_status_counts():
    fields = daemon.autonomous_official_result_operational_fields(
        {
            "autonomous_official_result_candidate_count": 14,
            "autonomous_official_result_race_rows": 14,
            "autonomous_official_result_runner_rows": 99,
            "autonomous_official_result_quarantine_rows": 1,
            "autonomous_official_result_evidence_inserted_rows": 0,
            "autonomous_official_result_evidence_db_ingest_status": (
                "NOOP_ALREADY_PRESENT"
            ),
            "autonomous_official_result_evidence_db_execute": True,
            "autonomous_official_result_evidence_db_write_performed": False,
            "autonomous_official_result_evidence_valid_race_rows": 14,
            "autonomous_official_result_evidence_valid_runner_rows": 99,
            "autonomous_official_result_evidence_blocked_race_rows": 0,
            "autonomous_official_result_evidence_blocked_runner_rows": 0,
            "autonomous_official_result_evidence_inserted_race_rows": 0,
            "autonomous_official_result_evidence_inserted_runner_rows": 0,
            "autonomous_official_result_evidence_blocker_reason_counts": {},
        },
        {
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "official_result_race_rows": 14,
            "official_result_runner_rows": 99,
            "quarantine_rows": 1,
        },
        evidence_inserted_rows=0,
    )

    assert fields["autonomous_official_result_candidate_count"] == 14
    assert fields["autonomous_official_result_race_rows"] == 14
    assert fields["autonomous_official_result_runner_rows"] == 99
    assert fields["autonomous_official_result_quarantine_rows"] == 1
    assert fields["autonomous_official_result_evidence_db_ingest_status"] == (
        "NOOP_ALREADY_PRESENT"
    )
    assert fields["autonomous_official_result_evidence_db_execute"] is True
    assert fields["autonomous_official_result_evidence_db_write_performed"] is False
    assert fields["autonomous_official_result_evidence_valid_race_rows"] == 14
    assert fields["autonomous_official_result_evidence_valid_runner_rows"] == 99
    assert fields["autonomous_official_result_evidence_blocked_race_rows"] == 0
    assert fields["autonomous_official_result_evidence_blocked_runner_rows"] == 0
    assert fields["autonomous_official_result_evidence_inserted_race_rows"] == 0
    assert fields["autonomous_official_result_evidence_inserted_runner_rows"] == 0
    assert fields["autonomous_official_result_evidence_blocker_reason_counts"] == {}


def test_autonomous_official_result_operational_fields_fall_back_to_capture_status():
    fields = daemon.autonomous_official_result_operational_fields(
        {},
        {
            "candidate_count": 8,
            "official_result_race_rows": 8,
            "official_result_runner_rows": 64,
            "quarantine_rows": 2,
            "official_result_evidence_db_ingest_status": "REPORT_ONLY",
            "official_result_evidence_db_execute": False,
            "official_result_evidence_db_write_performed": False,
            "official_result_evidence_valid_race_rows": 6,
            "official_result_evidence_valid_runner_rows": 48,
            "official_result_evidence_blocked_race_rows": 2,
            "official_result_evidence_blocked_runner_rows": 16,
            "official_result_evidence_inserted_race_rows": 0,
            "official_result_evidence_inserted_runner_rows": 0,
            "official_result_evidence_blocker_reason_counts": {
                "runner_set_mismatch": 2,
            },
        },
        evidence_inserted_rows=0,
    )

    assert fields["autonomous_official_result_candidate_count"] == 8
    assert fields["autonomous_official_result_race_rows"] == 8
    assert fields["autonomous_official_result_runner_rows"] == 64
    assert fields["autonomous_official_result_quarantine_rows"] == 2
    assert fields["autonomous_official_result_evidence_db_ingest_status"] == "REPORT_ONLY"
    assert fields["autonomous_official_result_evidence_valid_race_rows"] == 6
    assert fields["autonomous_official_result_evidence_valid_runner_rows"] == 48
    assert fields["autonomous_official_result_evidence_blocked_race_rows"] == 2
    assert fields["autonomous_official_result_evidence_blocked_runner_rows"] == 16
    assert fields["autonomous_official_result_evidence_blocker_reason_counts"] == {
        "runner_set_mismatch": 2,
    }


def test_autopilot_cycle_verification_lines_surface_primary_cycle_gate_context():
    lines = daemon.autopilot_cycle_verification_lines(
        {
            "autopilot_cycle_daily_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/DAILY_STATUS.json"
            ),
            "unified_evidence_dataset_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_dataset_rows": 927,
            "unified_evidence_dataset_races": 131,
            "unified_evidence_eligible_rows": 856,
            "rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "rolling_model_comparison_sample_races": 131,
            "rolling_model_comparison_best_candidate": "market_only_implied",
            "rolling_model_comparison_source_rejected_live_odds_candidate_count": 5,
            "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": 4,
            "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "rolling_model_comparison_blockers": [
                "no_candidate_passed_rank_first_accuracy_gate"
            ],
            "high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "high_accuracy_promotion_pr_gate_status": "BLOCKED",
            "timing_aligned_rerun_plan": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
            ),
            "timing_aligned_rerun_execution_status": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/"
                "timing_aligned_prediction_rerun_execution_status.json"
            ),
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
            "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_distance_promotion_ready": False,
            "promotion_distance_blockers": [
                "best_non_market_top1_margin_below_target"
            ],
            "promotion_distance_sample_race_count": 131,
            "promotion_distance_source_rejected_live_odds_candidate_count": 5,
            "promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
            "promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "promotion_distance_source_exclusion_reason_counts": {
                "official_result_missing": 32,
            },
            "promotion_distance_source_odds_exclusion_reason_counts": {
                "strict_prejump_odds_missing": 6,
            },
            "promotion_distance_source_official_result_evidence_db_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "promotion_distance_source_official_result_evidence_db_requested_race_count": 7,
            "promotion_distance_source_official_result_evidence_db_races_with_rows": [
                "Race 5 - TAREE - 2026-06-13",
            ],
            "promotion_distance_source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/official_result_runners.jsonl",
            ],
            "promotion_distance_official_result_coverage_requested_race_count": 7,
            "promotion_distance_official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": 4125,
            "promotion_distance_official_result_coverage_races_with_rows_count": 1,
            "promotion_distance_official_result_coverage_missing_race_count": 1,
            "promotion_distance_official_result_coverage_missing_exclusion_count": 32,
            "promotion_distance_official_result_runner_path_count": 1,
            "promotion_distance_official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
            "promotion_distance_best_candidate_key": "market_only_implied",
            "promotion_distance_best_non_market_candidate_key": (
                "stage2_rf_calibrated"
            ),
            "promotion_distance_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "promotion_distance_report_x/promotion_distance_report.json"
            ),
            "timing_aligned_prediction_rerun_plan_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
            ),
            "timing_aligned_prediction_rerun_plan_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
            ),
            "timing_aligned_prediction_rerun_execution_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_performed": False,
            "timing_aligned_prediction_rerun_output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_race_ingest_shadow_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_status": None,
            "timing_aligned_prediction_rerun_returncode": None,
        }
    )

    assert (
        "autopilot_cycle_unified_evidence_status=UNIFIED_EVIDENCE_DATASET_BUILT"
        in lines
    )
    assert "autopilot_cycle_unified_evidence_rows=927" in lines
    assert (
        "autopilot_cycle_rolling_model_comparison_status="
        "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    ) in lines
    assert (
        "autopilot_cycle_rolling_model_comparison_source_rejected_live_odds_candidate_count=5"
        in lines
    )
    assert (
        "autopilot_cycle_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates=4"
        in lines
    )
    assert (
        "autopilot_cycle_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts={'odds_decimal_invalid': 2, 'odds_source_url_missing': 3}"
        in lines
    )
    assert (
        "autopilot_cycle_high_accuracy_refinement_status=BLOCKED_KEEP_BASELINE"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_status=PROMOTION_DISTANCE_BLOCKED"
        in lines
    )
    assert "autopilot_cycle_promotion_distance_promotion_ready=False" in lines
    assert (
        "autopilot_cycle_promotion_distance_source_rejected_live_odds_candidate_count=5"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_source_rows_with_rejected_live_odds_candidates=4"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_source_rejected_live_odds_candidate_reason_counts={'odds_decimal_invalid': 2, 'odds_source_url_missing': 3}"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_source_exclusion_reason_counts={'official_result_missing': 32}"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_source_official_result_evidence_db_missing_race_ids=['Race 7 - TAREE - 2026-06-13']"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_coverage_requested_race_count=7"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_coverage_requested_race_count_source=deduped_requested_or_inferred_race_ids"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids=4125"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_coverage_races_with_rows_count=1"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_coverage_missing_race_count=1"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_coverage_missing_exclusion_count=32"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_runner_path_count=1"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_official_result_runner_paths_source_field=rolling_sample.source_official_result_runner_paths"
        in lines
    )
    assert (
        "autopilot_cycle_promotion_distance_best_non_market_candidate_key="
        "stage2_rf_calibrated"
    ) in lines
    assert (
        "autopilot_cycle_promotion_distance_blockers="
        "['best_non_market_top1_margin_below_target']"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_plan_status="
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_plan="
        "artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_execution_status_path="
        "artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/"
        "timing_aligned_prediction_rerun_execution_status.json"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_plan_hard_stops="
        "['timing_aligned_rerun_window_already_closed_after_jump']"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_execution_status="
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_execution_hard_stops="
        "['timing_aligned_rerun_window_already_closed_after_jump']"
    ) in lines
    assert "autopilot_cycle_timing_aligned_rerun_execution_performed=False" in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_output_dir="
        "artifacts/full_evidence_orchestration_20260525/"
        "daily_race_ingest_shadow_x_timing_aligned_rerun"
    ) in lines
    assert (
        "autopilot_cycle_timing_aligned_rerun_odds_snapshot_dir="
        "artifacts/full_evidence_orchestration_20260525/"
        "shadow_odds_snapshot_x_timing_aligned_rerun"
    ) in lines
    assert "autopilot_cycle_timing_aligned_rerun_odds_snapshot_status=None" in lines
    assert "autopilot_cycle_timing_aligned_rerun_returncode=None" in lines


def test_rejoin_promotion_distance_verification_lines_surface_coverage_provenance():
    lines = daemon.rejoin_promotion_distance_verification_lines(
        {
            "status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_ready": False,
            "blockers": ["best_non_market_top1_margin_below_target"],
            "official_result_coverage_requested_race_count": 69,
            "official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "official_result_coverage_legacy_requested_race_count_without_ids": 4125,
            "official_result_coverage_races_with_rows_count": 35,
            "official_result_coverage_missing_race_count": 35,
            "official_result_coverage_missing_exclusion_count": 13404,
            "official_result_runner_path_count": 1,
            "official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
        }
    )

    assert "rejoin_promotion_distance_status=PROMOTION_DISTANCE_BLOCKED" in lines
    assert "rejoin_promotion_distance_promotion_ready=False" in lines
    assert (
        "rejoin_promotion_distance_blockers=['best_non_market_top1_margin_below_target']"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_coverage_requested_race_count=69"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_coverage_requested_race_count_source=deduped_requested_or_inferred_race_ids"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids=4125"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_coverage_races_with_rows_count=35"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_coverage_missing_race_count=35"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_coverage_missing_exclusion_count=13404"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_runner_path_count=1"
        in lines
    )
    assert (
        "rejoin_promotion_distance_official_result_runner_paths_source_field=rolling_sample.source_official_result_runner_paths"
        in lines
    )


def test_write_service_files_preserves_shadow_model_pin(tmp_path):
    service_dir = tmp_path / "systemd"

    result = daemon.write_service_files(
        service_dir=service_dir,
        repo_path=Path("/home/l4nd0/greyhound_racing_collector"),
        timeout_seconds=840,
        python_path=Path("/runtime/.venv/bin/python"),
        evidence_root=Path("/runtime/artifacts/full_evidence_orchestration_20260525"),
        shadow_model=Path("/models/stage2/shadow_randomforest_model.joblib"),
        db_path=Path("/data/greyhound_racing_data.db"),
        lock_path=Path("/runtime/shared-shadow-autopilot.lock"),
        state_path=Path("/runtime/state.json"),
        odds_capture_state_path=Path("/runtime/odds_capture_state.json"),
    )

    service = (service_dir / daemon.SERVICE_NAME).read_text(encoding="utf-8")
    assert result["shadow_model"] == "/models/stage2/shadow_randomforest_model.joblib"
    assert result["python_path"] == "/runtime/.venv/bin/python"
    assert (
        result["evidence_root"]
        == "/runtime/artifacts/full_evidence_orchestration_20260525"
    )
    assert result["db_path"] == "/data/greyhound_racing_data.db"
    assert result["lock_path"] == "/runtime/shared-shadow-autopilot.lock"
    assert result["state_path"] == "/runtime/state.json"
    assert result["odds_capture_state_path"] == "/runtime/odds_capture_state.json"
    assert result["systemd_timeout_start_seconds"] == 3360
    assert "ExecStart=/runtime/.venv/bin/python" in service
    assert (
        "--evidence-root /runtime/artifacts/full_evidence_orchestration_20260525"
        in service
    )
    assert service.index("--evidence-root") < service.index("--days-ahead")
    assert "--shadow-model /models/stage2/shadow_randomforest_model.joblib" in service
    assert "--db /data/greyhound_racing_data.db" in service
    assert "--lock-path /runtime/shared-shadow-autopilot.lock" in service
    assert "--odds-capture-state-path /runtime/odds_capture_state.json" in service
    assert "--enable-autonomous-odds-capture" in service


def test_systemd_deployment_status_reports_active_installed_timer():
    def fake_runner(command, capture_output, text, timeout, check):
        assert command[1] == "--user"
        unit_name = command[3]
        if unit_name == daemon.SERVICE_NAME:
            stdout = "\n".join(
                [
                    "LoadState=loaded",
                    "ActiveState=inactive",
                    "UnitFileState=enabled",
                    "FragmentPath=/etc/systemd/system/shadow-autopilot.service",
                    "ExecStart={ path=/home/l4nd0/greyhound_racing_collector/.venv/bin/python ; argv[]=/home/l4nd0/greyhound_racing_collector/.venv/bin/python /home/l4nd0/greyhound_racing_collector/scripts/shadow_autopilot_daemon.py run-once --enable-autonomous-odds-capture --execute-autonomous-odds-capture --allow-auto-scrape-odds --enable-autonomous-result-capture ; }",
                    "",
                ]
            )
        elif unit_name == daemon.TIMER_NAME:
            stdout = "\n".join(
                [
                    "LoadState=loaded",
                    "ActiveState=active",
                    "UnitFileState=enabled",
                    "FragmentPath=/etc/systemd/system/shadow-autopilot.timer",
                    "",
                ]
            )
        else:
            stdout = "LoadState=not-found\nActiveState=inactive\nUnitFileState=\nFragmentPath=\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    status = daemon.systemd_deployment_status(
        systemctl_path="/bin/systemctl",
        runner=fake_runner,
    )

    assert status["scope"] == "user"
    assert status["deployment_status"] == "INSTALLED_AND_ACTIVE"
    assert status["deployment_ready"] is True
    assert status["service_installed"] is True
    assert status["timer_installed"] is True
    assert status["timer_enabled"] is True
    assert status["timer_active"] is True
    assert status["service_unit"]["status"] == "LOADED_ENABLED_INACTIVE"
    assert status["timer_unit"]["status"] == "ACTIVE"
    assert status["no_write_guarantees"]["db_write"] is False


def test_systemd_deployment_status_fails_closed_when_active_timer_command_is_stale():
    def fake_runner(command, capture_output, text, timeout, check):
        assert command[1] == "--user"
        unit_name = command[3]
        if unit_name == daemon.SERVICE_NAME:
            stdout = "\n".join(
                [
                    "LoadState=loaded",
                    "ActiveState=inactive",
                    "UnitFileState=static",
                    "FragmentPath=/etc/systemd/system/shadow-autopilot.service",
                    "ExecStart={ path=/home/l4nd0/greyhound_racing_collector/.venv/bin/python ; argv[]=/home/l4nd0/greyhound_racing_collector/.venv/bin/python /home/l4nd0/greyhound_racing_collector/scripts/shadow_autopilot_daemon.py run-once --days-ahead 1 --refresh-limit 16 ; }",
                    "",
                ]
            )
        elif unit_name == daemon.TIMER_NAME:
            stdout = "\n".join(
                [
                    "LoadState=loaded",
                    "ActiveState=active",
                    "UnitFileState=enabled",
                    "FragmentPath=/etc/systemd/system/shadow-autopilot.timer",
                    "",
                ]
            )
        else:
            stdout = "LoadState=not-found\nActiveState=inactive\nUnitFileState=\nFragmentPath=\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    status = daemon.systemd_deployment_status(
        systemctl_path="/bin/systemctl",
        runner=fake_runner,
        expected_service_exec_fragments=[
            "--enable-autonomous-odds-capture",
            "--shadow-model",
            "/models/stage2/shadow_randomforest_model.joblib",
        ],
    )

    assert status["scope"] == "user"
    assert status["deployment_status"] == "INSTALLED_COMMAND_MISMATCH"
    assert status["deployment_ready"] is False
    assert status["service_installed"] is True
    assert status["timer_installed"] is True
    assert status["timer_enabled"] is True
    assert status["timer_active"] is True
    assert status["service_command_matches_expected"] is False
    assert status["missing_service_exec_fragments"] == [
        "--enable-autonomous-odds-capture",
        "--shadow-model",
        "/models/stage2/shadow_randomforest_model.joblib",
    ]


def test_systemd_deployment_status_fails_closed_when_timer_missing():
    def fake_runner(command, capture_output, text, timeout, check):
        assert command[1] == "--user"
        unit_name = command[3]
        if unit_name == daemon.SERVICE_NAME:
            stdout = "LoadState=loaded\nActiveState=inactive\nUnitFileState=enabled\nFragmentPath=/etc/systemd/system/shadow-autopilot.service\n"
        else:
            stdout = "LoadState=not-found\nActiveState=inactive\nUnitFileState=\nFragmentPath=\n"
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    status = daemon.systemd_deployment_status(
        systemctl_path="/bin/systemctl",
        runner=fake_runner,
    )

    assert status["deployment_status"] == "INSTALLED_NOT_ACTIVE"
    assert status["deployment_ready"] is False
    assert status["service_installed"] is True
    assert status["timer_installed"] is False
    assert status["timer_active"] is False


def test_final_verdict_reports_daemon_ready_when_service_deployed():
    verdict = daemon.final_verdict(
        protected_paths_unchanged=True,
        required_outputs_present=True,
        service_files_present=True,
        lock_ok=True,
        operational_ok=True,
        service_installed=True,
    )

    assert verdict == "DAEMON_READY"


def test_daemon_official_result_status_surfaces_evidence_insert_counts(tmp_path):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "autonomous_official_result_capture_status.json",
        {
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "attempted": True,
            "candidate_count": 12,
            "official_result_race_rows": 9,
            "official_result_runner_rows": 63,
            "quarantine_rows": 1,
            "official_result_evidence_db_ingest_status": (
                "APPENDED_OFFICIAL_RESULT_EVIDENCE_WITH_QUARANTINE"
            ),
            "official_result_evidence_db_execute": True,
            "official_result_evidence_db_write_performed": True,
            "official_result_evidence_valid_race_rows": 8,
            "official_result_evidence_valid_runner_rows": 56,
            "official_result_evidence_blocked_race_rows": 1,
            "official_result_evidence_blocked_runner_rows": 7,
            "official_result_evidence_inserted_race_rows": 8,
            "official_result_evidence_inserted_runner_rows": 56,
            "official_result_evidence_blocker_reason_counts": {
                "runner_set_mismatch_quarantined": 1
            },
        },
    )

    status = daemon.autonomous_official_result_capture_status_from_autopilot(
        autopilot_dir
    )

    assert status["official_result_evidence_db_ingest_status"] == (
        "APPENDED_OFFICIAL_RESULT_EVIDENCE_WITH_QUARANTINE"
    )
    assert status["official_result_evidence_db_execute"] is True
    assert status["official_result_evidence_db_write_performed"] is True
    assert status["official_result_evidence_valid_race_rows"] == 8
    assert status["official_result_evidence_valid_runner_rows"] == 56
    assert status["official_result_evidence_blocked_race_rows"] == 1
    assert status["official_result_evidence_blocked_runner_rows"] == 7
    assert status["official_result_evidence_inserted_race_rows"] == 8
    assert status["official_result_evidence_inserted_runner_rows"] == 56
    assert status["official_result_evidence_blocker_reason_counts"] == {
        "runner_set_mismatch_quarantined": 1
    }


def test_daemon_readiness_requires_more_results_below_target():
    readiness = daemon.daemon_readiness(
        generated_at=daemon.datetime.fromisoformat("2026-06-08T20:30:00+10:00"),
        dashboard={
            "safe_joined_races": 54,
            "pending_races": 274,
            "unsafe_matches": 9,
            "calibration": {"status": "computed"},
            "probability_sum_status": {"status": "PASS"},
            "quarantined_features": ["same_distance_same_grade_best_time"],
            "box_1_share": 0.22,
        },
        target_joined_races=100,
    )

    assert readiness["decision"] == "NEED_MORE_RESULTS"
    assert "insufficient_forward_shadow_joined_races" in readiness["outstanding_blockers"]
    assert readiness["promotion_allowed"] is False


def test_daemon_readiness_surfaces_feature_activation_gate_blocker():
    readiness = daemon.daemon_readiness(
        generated_at=daemon.datetime.fromisoformat("2026-06-08T20:30:00+10:00"),
        dashboard={
            "safe_joined_races": 100,
            "pending_races": 0,
            "unsafe_matches": 0,
            "calibration": {"status": "computed"},
            "probability_sum_status": {"status": "PASS"},
            "quarantined_features": ["same_distance_same_grade_best_time"],
            "feature_activation_gate": {
                "status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
                "kept_quarantined_features": ["same_distance_same_grade_best_time"],
                "activation_allowed_features": [],
            },
        },
        target_joined_races=100,
    )

    assert readiness["feature_activation_gate_status"] == "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
    assert readiness["kept_quarantined_features"] == ["same_distance_same_grade_best_time"]
    assert "feature_activation_gate_not_passed" in readiness["outstanding_blockers"]
    assert readiness["promotion_allowed"] is False


def test_feature_activation_gate_status_from_autopilot_packet(tmp_path):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_20260608T220000_daemon"
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "feature_activation_gate_status.json",
        {
            "schema_version": "shadow_autopilot_feature_activation_gate_status_v1",
            "status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
            "output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_feature_activation_gate_run",
            "provenance_audit": "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_run/feature_activation_provenance_audit.json",
            "activation_allowed_features": [],
            "kept_quarantined_features": [
                "same_distance_same_grade_best_time",
                "same_distance_same_grade_avg_time",
            ],
            "inputs": {"parity_report": "train_eval_feature_parity_report.json"},
            "no_write_guarantees": {"training": False, "db_write": False},
        },
    )

    status = daemon.feature_activation_gate_status_from_autopilot(autopilot_dir)

    assert status is not None
    assert status["status"] == "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
    assert status["activation_allowed_features"] == []
    assert status["kept_quarantined_features"] == [
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    ]
    assert status["no_write_guarantees"]["training"] is False


def test_shadow_odds_snapshot_status_from_autopilot_packet(tmp_path):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_20260609T005503_daemon"
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "shadow_odds_snapshot_status.json",
        {
            "schema_version": "shadow_autopilot_odds_snapshot_status_v1",
            "status": "SKIPPED",
            "final_status": "SKIPPED",
            "collection_attempted": False,
            "skipped_reason": "no_shadow_predictions",
            "prediction_rows": 0,
            "odds_candidate_rows": 0,
            "valid_pre_jump_dog_odds_rows": 0,
            "races_with_complete_valid_prejump_odds": 0,
            "races_with_missing_odds_rows": 0,
            "races_with_post_feature_freeze_odds_rows": 0,
            "odds_research_gate_status": "ODDS_RESEARCH_BLOCKED_PROVENANCE",
            "odds_research_gate_report_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x/odds_research_gate_report.json"
            ),
            "odds_research_gate_complete_valid_prejump_odds_races": 2,
            "odds_research_gate_minimum_complete_valid_prejump_odds_races": 100,
            "odds_research_gate_source_url_coverage_pct": 100.0,
            "odds_research_gate_source_url_rows_missing": 0,
            "odds_research_gate_blocker_counts": {
                "complete_valid_prejump_odds_races_below_min": 98
            },
            "odds_research_next_action": (
                "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
            ),
            "timing_aligned_prediction_rerun_required": True,
            "timing_aligned_prediction_rerun_race_count": 2,
            "timing_aligned_prediction_rerun_race_ids": [
                "Race 10 - CANN - 2026-06-13",
                "Race 8 - CANN - 2026-06-13",
            ],
            "timing_aligned_prediction_rerun_reason_counts": {
                "raw_expected_prejump_windows_complete_but_after_prediction": 2
            },
            "ev_output_rows": 0,
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            "no_write_guarantees": {"db_write": False, "betting_or_ev_action": False},
        },
    )

    status = daemon.shadow_odds_snapshot_status_from_autopilot(autopilot_dir)

    assert status is not None
    assert status["status"] == "SKIPPED"
    assert status["skipped_reason"] == "no_shadow_predictions"
    assert status["prediction_rows"] == 0
    assert status["valid_pre_jump_dog_odds_rows"] == 0
    assert status["races_with_complete_valid_prejump_odds"] == 0
    assert status["races_with_missing_odds_rows"] == 0
    assert status["races_with_post_feature_freeze_odds_rows"] == 0
    assert status["odds_research_gate_status"] == "ODDS_RESEARCH_BLOCKED_PROVENANCE"
    assert status["odds_research_gate_report_path"].endswith(
        "odds_research_gate_report.json"
    )
    assert status["odds_research_gate_complete_valid_prejump_odds_races"] == 2
    assert status["odds_research_gate_minimum_complete_valid_prejump_odds_races"] == 100
    assert status["odds_research_gate_source_url_coverage_pct"] == 100.0
    assert status["odds_research_gate_source_url_rows_missing"] == 0
    assert status["odds_research_gate_blocker_counts"] == {
        "complete_valid_prejump_odds_races_below_min": 98
    }
    assert status["odds_research_next_action"] == (
        "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
    )
    assert status["timing_aligned_prediction_rerun_required"] is True
    assert status["timing_aligned_prediction_rerun_race_count"] == 2
    assert status["timing_aligned_prediction_rerun_race_ids"] == [
        "Race 10 - CANN - 2026-06-13",
        "Race 8 - CANN - 2026-06-13",
    ]
    assert status["timing_aligned_prediction_rerun_reason_counts"] == {
        "raw_expected_prejump_windows_complete_but_after_prediction": 2
    }
    assert status["ev_output_rows"] == 0
    assert status["ev_calculation_status"] == "DISABLED_REPORT_ONLY_NO_EV_OUTPUT"
    assert status["no_write_guarantees"]["db_write"] is False


def test_shadow_odds_snapshot_status_missing_autopilot_packet_is_explicit():
    status = daemon.shadow_odds_snapshot_status_from_autopilot(None)

    assert status is not None
    assert status["status"] == "MISSING_AUTOPILOT_OUTPUT"
    assert status["skipped_reason"] == "autopilot_output_missing"
    assert status["ev_output_rows"] == 0


def test_next_prejump_refresh_window_from_autopilot_packet(tmp_path):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_20260608T235427_daemon"
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "refresh_prejump_report.json",
        {
            "status": "SUCCESS",
            "generated_at": "2026-06-08T23:54:29+10:00",
            "total_races_found": 24,
            "selected_count": 0,
            "selected_races": [],
            "next_preferred_window": {
                "status": "WAITING_FOR_FUTURE_WINDOW",
                "reason": "next_race_not_yet_inside_preferred_window",
                "recommended_rerun_after_local": "2026-06-09T08:55:00+10:00",
                "next_window_opens_at": "2026-06-09T08:55:00+10:00",
                "next_window_closes_at": "2026-06-09T11:15:00+10:00",
                "minutes_until_window_opens": 540.5,
                "minutes_until_window_closes": 680.5,
                "next_race": {
                    "race_id": "Race 1 - AP_K - 2026-06-09",
                    "date": "2026-06-09",
                    "venue": "AP_K",
                    "race_number": "1",
                    "race_time": "11:35 AM",
                    "jump_datetime": "2026-06-09T11:35:00+10:00",
                    "minutes_to_jump": 700.5,
                    "bucket": "future_outside_preferred_window",
                    "selected": False,
                    "race_url": "https://www.thedogs.com.au/racing/angle-park/2026-06-09/1/example",
                },
            },
        },
    )

    status = daemon.next_prejump_refresh_window_from_autopilot(autopilot_dir)

    assert status is not None
    assert status["status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert status["recommended_rerun_after_local"] == "2026-06-09T08:55:00+10:00"
    assert status["selected_count"] == 0
    assert status["total_races_found"] == 24
    assert status["next_race"]["race_id"] == "Race 1 - AP_K - 2026-06-09"
    assert status["next_race"]["race_url"].startswith("https://www.thedogs.com.au/")
    assert status["no_write_guarantees"]["db_write"] is False


def test_daemon_runtime_state_packet_is_written_from_persisted_state(tmp_path):
    output_dir = tmp_path / "shadow_autopilot_daemonization_v1_test"
    daily_dir = tmp_path / "daily_race_ingest_shadow_test"
    state_path = tmp_path / "runtime" / "state.json"
    output_dir.mkdir()
    daily_dir.mkdir()
    (output_dir / "final_status.txt").write_text("DAEMON_READY\n", encoding="utf-8")
    daemon.write_json(
        output_dir / "prediction_provenance_report.json",
        {"daily_shadow_run_dir": str(daily_dir)},
    )
    (daily_dir / "final_status.txt").write_text("WAITING_FOR_UPCOMING_RACES\n", encoding="utf-8")
    daemon.write_json(
        daily_dir / "shadow_manifest.json",
        {
            "final_status": "WAITING_FOR_UPCOMING_RACES",
            "input_classification": {
                "scanned_csv_count": 12,
                "eligible_count": 0,
                "stale_count": 12,
                "malformed_count": 0,
            },
        },
    )
    daemon.write_json(
        daily_dir / "prejump_metadata_report.json",
        {
            "target_metadata_readiness": {
                "status": "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS",
                "target_metadata_capture_status": "WAITING",
                "blocker_counts": {},
            }
        },
    )
    daemon.write_json(
        daily_dir / "same_distance_same_grade_history_provenance.json",
        {
            "status": "NOT_POPULATED",
            "live_input_status": "NO_ELIGIBLE_PREJUMP_RACES",
            "target_race_rows_allowed": 0,
            "post_outcome_rows_allowed": 0,
        },
    )
    daemon.write_json(
        state_path,
        {
            "schema_version": "shadow_autopilot_daemon_state_v1",
            "last_output_dir": str(output_dir),
            "last_verdict": "DAEMON_READY",
            "last_cycle_activity_status": "NO_NEW_PREDICTIONS_OR_SAFE_JOINS",
            "last_safe_joined_races": 84,
            "last_next_prejump_refresh_status": "WAITING_FOR_FUTURE_WINDOW",
            "last_recommended_rerun_after_local": "2026-06-09T08:55:00+10:00",
            "last_next_prejump_race": {
                "race_id": "Race 1 - AP_K - 2026-06-09",
                "jump_datetime": "2026-06-09T11:35:00+10:00",
            },
            "last_autonomous_live_odds_next_window_opens_at": (
                "2026-06-09T10:35:00+10:00"
            ),
            "last_autonomous_live_odds_recommended_rerun_after_local": (
                "2026-06-09T10:30:00+10:00"
            ),
            "last_odds_capture_next_meaningful_action": (
                "WAIT_UNTIL_NEXT_FIXED_WINDOW"
            ),
            "last_odds_capture_next_meaningful_action_at": "2026-06-09T10:35:00+10:00",
            "last_autonomous_live_odds_next_race_id": "Race 1 - AP_K - 2026-06-09",
            "last_autonomous_live_odds_next_prejump_window": {
                "status": "WAITING_FOR_FUTURE_WINDOW",
                "next_window_opens_at": "2026-06-09T10:35:00+10:00",
                "recommended_rerun_after_local": "2026-06-09T10:30:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - AP_K - 2026-06-09",
                    "jump_datetime": "2026-06-09T11:35:00+10:00",
                },
            },
            "last_live_odds_backlog_unresolved_race_count": 11,
            "last_live_odds_backlog_unresolved_recovery_action_counts": {
                "validate_runner_set_then_alias_join": 5
            },
            "last_live_odds_backlog_recovery_queue_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_recovery_queue.json"
            ),
            "last_live_odds_backlog_recovery_queue_diagnostic_only": True,
            "last_live_odds_backlog_recovery_queue_db_write_performed": False,
            "last_live_odds_backlog_awaiting_official_result_evidence_race_count": 5,
            "last_live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
                "diagnostic_recheck_official_result_evidence_only"
            ),
            "last_live_odds_backlog_runner_set_validation_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_runner_set_validation.json"
            ),
            "last_live_odds_backlog_runner_set_validation_exact_match_race_count": 5,
            "last_live_odds_backlog_runner_set_validation_join_authorized": False,
            "last_live_odds_backlog_runner_set_validation_db_write_performed": False,
            "last_live_odds_backlog_join_eligibility_packet_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_join_eligibility_packet.json"
            ),
            "last_live_odds_backlog_join_eligibility_eligible_report_only_race_count": 2,
            "last_live_odds_backlog_join_eligibility_blocked_race_count": 3,
            "last_live_odds_backlog_join_eligibility_join_authorized": False,
            "last_live_odds_backlog_join_eligibility_db_write_performed": False,
            "last_live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 2,
            "last_best_aggregate_unified_evidence_status": (
                "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
            ),
            "last_best_aggregate_unified_evidence_eligible_rows": 3872,
            "last_best_aggregate_unified_rejected_live_odds_candidate_count": 9,
            "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates": 6,
            "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 4,
                "odds_source_url_missing": 5,
            },
            "last_rejoin_unified_evidence_status": (
                "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
            ),
            "last_rejoin_unified_rejected_live_odds_candidate_count": 8,
            "last_rejoin_unified_rows_with_rejected_live_odds_candidates": 6,
            "last_rejoin_unified_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 3,
                "odds_source_url_missing": 5,
            },
        },
    )

    report = daemon.write_daemon_runtime_state_packet(
        output_dir=output_dir,
        state_path=state_path,
        systemd_deployment={
            "deployment_status": "INSTALLED_AND_ACTIVE",
            "deployment_ready": True,
            "timer_enabled": True,
            "timer_active": True,
        },
        target_joined_races=100,
        generated_at=daemon.datetime.fromisoformat("2026-06-09T03:30:00+10:00"),
    )

    assert report["runtime_action"] == "WAIT_UNTIL_RECOMMENDED_REFRESH"
    assert report["safe_joined_races"] == 84
    assert report["safe_joined_races_remaining"] == 16
    assert report["timer"]["service_status"] == "cycle_finalizing"
    assert report["timer"]["timer_status"] == "active"
    assert report["daily_shadow_run"]["final_status"] == "WAITING_FOR_UPCOMING_RACES"
    assert (
        report["daily_shadow_run"]["same_distance_history_provenance"]["status"]
        == "NOT_POPULATED"
    )
    assert report["daemon"]["last_autonomous_live_odds_next_window_opens_at"] == (
        "2026-06-09T10:35:00+10:00"
    )
    assert report["daemon"]["last_autonomous_live_odds_next_race_id"] == (
        "Race 1 - AP_K - 2026-06-09"
    )
    assert report["daemon"]["last_odds_capture_next_meaningful_action"] == (
        "WAIT_UNTIL_NEXT_FIXED_WINDOW"
    )
    assert report["daemon"]["last_odds_capture_next_meaningful_action_at"] == (
        "2026-06-09T10:35:00+10:00"
    )
    assert report["daemon"]["last_live_odds_backlog_unresolved_race_count"] == 11
    assert report["daemon"][
        "last_live_odds_backlog_unresolved_recovery_action_counts"
    ] == {"validate_runner_set_then_alias_join": 5}
    assert report["daemon"]["last_live_odds_backlog_recovery_queue_path"].endswith(
        "live_odds_backlog_recovery_queue.json"
    )
    assert report["daemon"]["last_live_odds_backlog_recovery_queue_diagnostic_only"] is True
    assert report["daemon"]["last_live_odds_backlog_recovery_queue_db_write_performed"] is False
    assert (
        report["daemon"][
            "last_live_odds_backlog_awaiting_official_result_evidence_race_count"
        ]
        == 5
    )
    assert report["daemon"][
        "last_live_odds_backlog_awaiting_official_result_evidence_authorized_action"
    ] == "diagnostic_recheck_official_result_evidence_only"
    assert report["daemon"]["last_live_odds_backlog_runner_set_validation_path"].endswith(
        "live_odds_backlog_runner_set_validation.json"
    )
    assert (
        report["daemon"][
            "last_live_odds_backlog_runner_set_validation_exact_match_race_count"
        ]
        == 5
    )
    assert (
        report["daemon"]["last_live_odds_backlog_runner_set_validation_join_authorized"]
        is False
    )
    assert (
        report["daemon"]["last_live_odds_backlog_runner_set_validation_db_write_performed"]
        is False
    )
    assert report["daemon"]["last_live_odds_backlog_join_eligibility_packet_path"].endswith(
        "live_odds_backlog_join_eligibility_packet.json"
    )
    assert (
        report["daemon"][
            "last_live_odds_backlog_join_eligibility_eligible_report_only_race_count"
        ]
        == 2
    )
    assert report["daemon"]["last_live_odds_backlog_join_eligibility_blocked_race_count"] == 3
    assert report["daemon"]["last_live_odds_backlog_join_eligibility_join_authorized"] is False
    assert (
        report["daemon"]["last_live_odds_backlog_join_eligibility_db_write_performed"]
        is False
    )
    assert (
        report["daemon"][
            "last_live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
        ]
        == 2
    )
    assert report["daemon"]["last_best_aggregate_unified_evidence_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert report["daemon"][
        "last_best_aggregate_unified_evidence_eligible_rows"
    ] == 3872
    assert report["daemon"][
        "last_best_aggregate_unified_rejected_live_odds_candidate_count"
    ] == 9
    assert report["daemon"][
        "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates"
    ] == 6
    assert report["daemon"][
        "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert report["daemon"]["last_rejoin_unified_evidence_status"] == (
        "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert report["daemon"][
        "last_rejoin_unified_rejected_live_odds_candidate_count"
    ] == 8
    assert report["daemon"][
        "last_rejoin_unified_rows_with_rejected_live_odds_candidates"
    ] == 6
    assert report["daemon"][
        "last_rejoin_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 5,
    }
    summary_text = (output_dir / "FORWARD_SHADOW_RUNTIME_STATE.md").read_text(
        encoding="utf-8"
    )
    assert "Live odds backlog unresolved races: `11`" in summary_text
    assert "Live odds backlog recovery queue:" in summary_text
    assert "live_odds_backlog_recovery_queue.json" in summary_text
    assert "Live odds backlog runner-set validation:" in summary_text
    assert "live_odds_backlog_runner_set_validation.json" in summary_text
    assert "Live odds backlog join eligibility packet:" in summary_text
    assert "live_odds_backlog_join_eligibility_packet.json" in summary_text
    assert "Live odds backlog join eligibility report-only races: `2`" in summary_text
    assert "Live odds backlog join eligibility DB write performed: `False`" in summary_text
    assert "Best aggregate unified evidence: `BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT`" in summary_text
    assert "Best aggregate unified rows with rejected live odds candidates: `6`" in summary_text
    assert "Rejoin rejected live odds candidates: `8`" in summary_text
    assert "Rejoin rows with rejected live odds candidates: `6`" in summary_text
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_or_ev_output"] is False
    assert (output_dir / "forward_shadow_runtime_state.json").exists()
    assert (output_dir / "FORWARD_SHADOW_RUNTIME_STATE.md").exists()


def test_final_summary_includes_feature_activation_gate_status():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={
            "safe_joined_races": 82,
            "pending_races": 239,
            "unsafe_matches": 14,
            "probability_sum_status": {"status": "PASS"},
            "odds_capture_next_meaningful_action": "WAIT_UNTIL_NEXT_FIXED_WINDOW",
            "odds_capture_next_meaningful_action_at": "2026-06-12T09:48:00+10:00",
            "autonomous_official_result_candidate_count": 21,
            "autonomous_official_result_race_rows": 21,
            "autonomous_official_result_runner_rows": 150,
            "autonomous_official_result_quarantine_rows": 0,
            "autonomous_official_result_evidence_db_ingest_status": (
                "NOOP_ALREADY_PRESENT"
            ),
            "autonomous_official_result_evidence_db_execute": True,
            "autonomous_official_result_evidence_db_write_performed": False,
            "autonomous_official_result_evidence_valid_race_rows": 21,
            "autonomous_official_result_evidence_valid_runner_rows": 150,
            "autonomous_official_result_evidence_blocked_race_rows": 0,
            "autonomous_official_result_evidence_blocked_runner_rows": 0,
            "autonomous_official_result_evidence_inserted_race_rows": 0,
            "autonomous_official_result_evidence_inserted_runner_rows": 0,
            "autonomous_official_result_evidence_blocker_reason_counts": {},
            "autonomous_official_result_quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "participant_source": "shadow_run_predictions",
                    "participant_boxes": [1, 2, 3, 4, 5, 6, 7, 8],
                    "result_boxes_in_participants": [1, 2, 3, 4, 7, 8],
                    "result_boxes_not_in_participants": [9, 10],
                }
            ],
        },
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        feature_activation_gate={
            "status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
            "kept_quarantined_features": [
                "same_distance_same_grade_best_time",
                "same_distance_same_grade_avg_time",
            ],
        },
    )

    assert "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED" in summary
    assert "same_distance_same_grade_avg_time" in summary
    assert (
        "Autonomous odds next action: `WAIT_UNTIL_NEXT_FIXED_WINDOW` at "
        "`2026-06-12T09:48:00+10:00`"
    ) in summary
    assert (
        "Autonomous official result evidence DB ingest: `NOOP_ALREADY_PRESENT`"
        in summary
    )
    assert "Autonomous official result candidates: `21`" in summary
    assert "Autonomous official result race rows: `21`" in summary
    assert "Autonomous official result runner rows: `150`" in summary
    assert "Autonomous official result quarantine rows: `0`" in summary
    assert "Autonomous official result evidence valid runner rows: `150`" in summary
    assert "Autonomous official result evidence DB write performed: `False`" in summary
    assert "Autonomous official result runner-set mismatch samples:" in summary
    assert "shadow_run_predictions" in summary
    assert "result_boxes_not_in_participants" in summary


def test_final_summary_includes_shadow_odds_race_coverage():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={
            "safe_joined_races": 82,
            "pending_races": 239,
            "unsafe_matches": 14,
            "probability_sum_status": {"status": "PASS"},
        },
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        shadow_odds_snapshot={
            "status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "valid_pre_jump_dog_odds_rows": 8,
            "races_with_complete_valid_prejump_odds": 1,
            "races_with_missing_odds_rows": 0,
            "races_with_post_feature_freeze_odds_rows": 0,
            "ev_output_rows": 0,
        },
    )

    assert "Shadow odds complete valid races: `1`" in summary
    assert "Shadow odds races with missing rows: `0`" in summary
    assert "Shadow odds races after feature freeze: `0`" in summary


def test_final_summary_includes_rejoin_high_accuracy_timing_sources():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={
            "safe_joined_races": 82,
            "pending_races": 0,
            "unsafe_matches": 0,
            "probability_sum_status": {"status": "PASS"},
        },
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 1},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        rejoin_high_accuracy_refinement_status={
            "status": "BLOCKED_KEEP_BASELINE",
            "timing_aligned_rerun_plan": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
            ),
            "timing_aligned_rerun_execution_status": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/"
                "timing_aligned_prediction_rerun_execution_status.json"
            ),
            "reserve_substitution_preflight_status": (
                "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
            ),
            "reserve_substitution_preflight_ready_for_policy_review_count": 4,
            "reserve_substitution_preflight_dataset_join_blocker_counts": {
                "manual_policy_review_required_before_join": 4,
                "official_result_remains_quarantined": 4,
            },
            "reserve_substitution_preflight_ready_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
        },
    )

    assert "Rejoin high-accuracy packet: `BLOCKED_KEEP_BASELINE`" in summary
    assert "Rejoin high-accuracy timing-aligned rerun plan:" in summary
    assert "timing_aligned_prediction_rerun_plan.json" in summary
    assert "Rejoin high-accuracy timing-aligned rerun execution status:" in summary
    assert "timing_aligned_prediction_rerun_execution_status.json" in summary
    assert "Rejoin reserve substitution preflight:" in summary
    assert "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW" in summary
    assert "Rejoin reserve substitution ready for policy review: `4`" in summary
    assert "Rejoin reserve substitution dataset join blockers:" in summary
    assert (
        "Rejoin reserve substitution policy impact: "
        "`RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY`"
    ) in summary
    assert (
        "Rejoin reserve substitution policy impact potential runner rows blocked: `32`"
        in summary
    )
    assert (
        "Rejoin reserve substitution policy impact dataset join allowed: `False`"
        in summary
    )


def test_final_summary_includes_autopilot_cycle_gate_context():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={
            "safe_joined_races": 131,
            "pending_races": 0,
            "unsafe_matches": 0,
            "probability_sum_status": {"status": "PASS"},
        },
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        autopilot_cycle_daily_status={
            "unified_evidence_dataset_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_dataset_rows": 927,
            "unified_evidence_dataset_races": 131,
            "unified_evidence_eligible_rows": 856,
            "best_aggregate_unified_evidence_status_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/backlog_unified_evidence_datasets_status.json"
            ),
            "best_aggregate_unified_evidence_status": (
                "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
            ),
            "best_aggregate_unified_evidence_dataset_count": 56,
            "best_aggregate_unified_evidence_row_count": 5108,
            "best_aggregate_unified_evidence_eligible_rows": 3872,
            "best_aggregate_unified_evidence_artifact_odds_rows_seen": 114,
            "best_aggregate_unified_evidence_artifact_odds_rows_accepted": 14,
            "best_aggregate_unified_evidence_artifact_odds_rows_rejected": 100,
            "best_aggregate_unified_evidence_artifact_odds_rejection_reason_counts": {
                "odds_after_jump": 73,
                "runner_set_mismatch": 27,
            },
            "best_aggregate_unified_rejected_live_odds_candidate_count": 9,
            "best_aggregate_unified_rows_with_rejected_live_odds_candidates": 6,
            "best_aggregate_unified_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 4,
                "odds_source_url_missing": 5,
            },
            "best_aggregate_unified_sample_blocking_gap_count": 2,
            "best_aggregate_unified_gap_action_counts": {
                "retry_official_result_capture_or_join": 2,
            },
            "best_aggregate_unified_gap_evidence_missing_reason_counts": {
                "official_result_missing_only": 2,
            },
            "best_aggregate_unified_top_gap_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "best_aggregate_unified_top_official_result_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
                "Race 4 - TAREE - 2026-06-13",
            ],
            "rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "rolling_model_comparison_sample_races": 131,
            "rolling_model_comparison_best_candidate": "market_only_implied",
            "rolling_model_comparison_source_rejected_live_odds_candidate_count": 5,
            "rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": 4,
            "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "rolling_model_comparison_blockers": [
                "no_candidate_passed_rank_first_accuracy_gate"
            ],
            "high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "high_accuracy_promotion_pr_gate_status": "BLOCKED",
            "timing_aligned_rerun_plan": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
            ),
            "timing_aligned_rerun_execution_status": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_autopilot_v1_x/"
                "timing_aligned_prediction_rerun_execution_status.json"
            ),
            "reserve_substitution_policy_impact_status": (
                "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
            ),
            "reserve_substitution_policy_impact_ready_candidate_count": 4,
            "reserve_substitution_policy_impact_mapping_pair_count": 5,
            "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
            "reserve_substitution_policy_impact_dataset_join_allowed": False,
            "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
            "reserve_substitution_policy_impact_db_write": False,
            "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_distance_promotion_ready": False,
            "promotion_distance_sample_race_count": 131,
            "promotion_distance_source_rejected_live_odds_candidate_count": 5,
            "promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
            "promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "promotion_distance_best_non_market_candidate_key": (
                "stage2_rf_calibrated"
            ),
            "promotion_distance_blockers": [
                "best_non_market_top1_margin_below_target"
            ],
            "timing_aligned_prediction_rerun_plan_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED"
            ),
            "timing_aligned_prediction_rerun_plan_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
            ),
            "timing_aligned_prediction_rerun_execution_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_performed": False,
            "timing_aligned_prediction_rerun_output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_race_ingest_shadow_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_status": None,
        },
    )

    assert (
        "Autopilot cycle unified evidence: `UNIFIED_EVIDENCE_DATASET_BUILT`"
        in summary
    )
    assert "Autopilot cycle unified evidence rows: `927`" in summary
    assert (
        "Best aggregate unified evidence path: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/backlog_unified_evidence_datasets_status.json`"
    ) in summary
    assert (
        "Best aggregate unified evidence: "
        "`BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT`"
    ) in summary
    assert "Best aggregate unified dataset count: `56`" in summary
    assert "Best aggregate unified row count: `5108`" in summary
    assert "Best aggregate unified eligible rows: `3872`" in summary
    assert "Best aggregate unified artifact odds rows seen: `114`" in summary
    assert "Best aggregate unified artifact odds rows accepted: `14`" in summary
    assert "Best aggregate unified artifact odds rows rejected: `100`" in summary
    assert (
        "Best aggregate unified artifact odds rejection reasons: "
        "`{'odds_after_jump': 73, 'runner_set_mismatch': 27}`"
    ) in summary
    assert (
        "Best aggregate unified rejected live odds candidates: `9`"
        in summary
    )
    assert (
        "Best aggregate unified rows with rejected live odds candidates: `6`"
        in summary
    )
    assert (
        "Best aggregate unified rejected live odds candidate reasons: "
        "`{'odds_decimal_invalid': 4, 'odds_source_url_missing': 5}`"
    ) in summary
    assert (
        "Best aggregate unified sample-blocking gap races: `2`"
        in summary
    )
    assert (
        "Best aggregate unified gap actions: "
        "`{'retry_official_result_capture_or_join': 2}`"
    ) in summary
    assert (
        "Best aggregate unified top gap race IDs: "
        "`['Race 7 - TAREE - 2026-06-13', 'Race 4 - TAREE - 2026-06-13']`"
    ) in summary
    assert (
        "Best aggregate unified top official-result-missing race IDs: "
        "`['Race 7 - TAREE - 2026-06-13', 'Race 4 - TAREE - 2026-06-13']`"
    ) in summary
    assert (
        "Autopilot cycle rolling comparison: "
        "`ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW`"
    ) in summary
    assert (
        "Autopilot cycle rolling comparison source rejected live odds candidates: `5`"
        in summary
    )
    assert (
        "Autopilot cycle rolling comparison source rows with rejected live odds candidates: `4`"
        in summary
    )
    assert (
        "Autopilot cycle rolling comparison source rejected live odds candidate reasons: `{'odds_decimal_invalid': 2, 'odds_source_url_missing': 3}`"
        in summary
    )
    assert (
        "Autopilot cycle high-accuracy packet: `BLOCKED_KEEP_BASELINE`"
        in summary
    )
    assert (
        "Autopilot cycle high-accuracy timing-aligned rerun plan: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json`"
    ) in summary
    assert (
        "Autopilot cycle high-accuracy timing-aligned rerun execution status: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/"
        "timing_aligned_prediction_rerun_execution_status.json`"
    ) in summary
    assert (
        "Autopilot cycle reserve substitution policy impact: "
        "`RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY`"
    ) in summary
    assert (
        "Autopilot cycle reserve substitution policy impact potential runner rows blocked: `32`"
        in summary
    )
    assert (
        "Autopilot cycle reserve substitution policy impact dataset join allowed: `False`"
        in summary
    )
    assert (
        "Autopilot cycle promotion distance: `PROMOTION_DISTANCE_BLOCKED`"
        in summary
    )
    assert (
        "Autopilot cycle promotion distance source rejected live odds candidates: `5`"
        in summary
    )
    assert (
        "Autopilot cycle promotion distance source rows with rejected live odds candidates: `4`"
        in summary
    )
    assert (
        "Autopilot cycle promotion distance source rejected live odds candidate reasons: `{'odds_decimal_invalid': 2, 'odds_source_url_missing': 3}`"
        in summary
    )
    assert (
        "Autopilot cycle promotion distance best non-market candidate: "
        "`stage2_rf_calibrated`"
    ) in summary
    assert (
        "Autopilot cycle promotion distance blockers: "
        "`['best_non_market_top1_margin_below_target']`"
    ) in summary
    assert (
        "Autopilot cycle timing-aligned rerun plan: "
        "`TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED`"
    ) in summary
    assert (
        "Autopilot cycle timing-aligned rerun hard stops: "
        "`['timing_aligned_rerun_window_already_closed_after_jump']`"
    ) in summary
    assert (
        "Autopilot cycle timing-aligned rerun execution: "
        "`TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY`"
    ) in summary
    assert (
        "Autopilot cycle timing-aligned rerun execution hard stops: "
        "`['timing_aligned_rerun_window_already_closed_after_jump']`"
    ) in summary
    assert "Autopilot cycle timing-aligned rerun executed: `False`" in summary
    assert (
        "Autopilot cycle timing-aligned rerun output: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "daily_race_ingest_shadow_x_timing_aligned_rerun`"
    ) in summary
    assert (
        "Autopilot cycle timing-aligned rerun odds snapshot dir: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_odds_snapshot_x_timing_aligned_rerun`"
    ) in summary
    assert "Autopilot cycle timing-aligned rerun odds snapshot: `None`" in summary


def test_final_summary_includes_predeclared_residual_candidate_status():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={
            "safe_joined_races": 150,
            "pending_races": 30,
            "unsafe_matches": 28,
            "probability_sum_status": {"status": "PASS"},
        },
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "10min"},
        rejoin_pre_race_gated_challenger_status={
            "status": "PRE_RACE_GATED_CHALLENGER_REVIEW_READY",
            "promotion_ready": False,
            "predeclared_residual_candidate_status": (
                "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING"
            ),
            "predeclared_residual_triggered_race_count": 3,
            "predeclared_residual_minimum_triggered_races_for_directional_read": 10,
            "predeclared_residual_directional_read_ready": False,
        },
        rejoin_rank_first_hypothesis_gated_status={
            "rank_first_hypothesis_review_status": (
                "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
            ),
            "rank_first_hypothesis_candidate_count": 5,
            "rank_first_hypothesis_evaluated_candidate_count": 5,
            "rank_first_hypothesis_best_candidate_key": (
                "rank_first_hypothesis_runner_count_eq_4__raw_stage2_uncalibrated"
            ),
            "rank_first_hypothesis_best_triggered_race_count": 13,
            "rank_first_hypothesis_minimum_triggered_races_for_directional_read": 10,
            "rank_first_hypothesis_directional_read_ready": False,
        },
        rejoin_market_residual_challenger_status={
            "status": "MARKET_RESIDUAL_CHALLENGER_REVIEW_READY",
            "promotion_ready": False,
        },
        rejoin_market_residual_regime_audit_status={
            "status": "MARKET_RESIDUAL_REGIME_AUDIT_READY",
            "promotion_ready": False,
            "rank_first_hypothesis_status": "NO_PRE_RACE_RANK_FIRST_EDGE_FOUND",
            "pre_race_rank_first_help_regime_count": 0,
            "pre_race_logloss_only_help_regime_count": 6,
        },
        rejoin_rank_first_hypothesis_watchlist_status={
            "status": "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY",
            "candidate_count": 5,
            "directional_ready_candidate_count": 0,
            "best_candidate_key": (
                "rank_first_hypothesis_venue_eq_gee__raw_stage2_uncalibrated"
            ),
            "best_candidate_status": (
                "RANK_FIRST_HYPOTHESIS_WAITING_FOR_FRESH_SAMPLE"
            ),
            "best_candidate_distinct_sample_count": 1,
            "minimum_distinct_samples_for_directional_read": 2,
        },
        rejoin_promotion_distance_status={
            "status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_ready": False,
            "blockers": ["best_non_market_top1_margin_below_target"],
            "official_result_coverage_requested_race_count": 69,
            "official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "official_result_coverage_legacy_requested_race_count_without_ids": 4125,
            "official_result_coverage_races_with_rows_count": 35,
            "official_result_coverage_missing_race_count": 35,
            "official_result_coverage_missing_exclusion_count": 13404,
            "official_result_runner_path_count": 1,
            "official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
        },
    )

    assert (
        "Rejoin pre-race predeclared residual candidate: "
        "`PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING`"
    ) in summary
    assert "Rejoin pre-race predeclared residual triggered races: `3` / `10`" in summary
    assert (
        "Rejoin pre-race predeclared residual directional read ready: `False`"
        in summary
    )
    assert (
        "Rejoin rank-first hypothesis gate review: "
        "`RANK_FIRST_HYPOTHESIS_REVIEW_READY`"
    ) in summary
    assert "Rejoin rank-first hypothesis candidates: `5` / `5`" in summary
    assert (
        "Rejoin rank-first hypothesis best candidate: "
        "`rank_first_hypothesis_runner_count_eq_4__raw_stage2_uncalibrated`"
    ) in summary
    assert "Rejoin rank-first hypothesis best triggered races: `13` / `10`" in summary
    assert (
        "Rejoin rank-first hypothesis directional read ready: `False`"
        in summary
    )
    assert (
        "Rejoin market residual challenger: "
        "`MARKET_RESIDUAL_CHALLENGER_REVIEW_READY`"
    ) in summary
    assert (
        "Rejoin market residual regime audit: "
        "`MARKET_RESIDUAL_REGIME_AUDIT_READY`"
    ) in summary
    assert (
        "Rejoin market residual rank-first hypothesis: "
        "`NO_PRE_RACE_RANK_FIRST_EDGE_FOUND`"
    ) in summary
    assert "Rejoin market residual rank-first help regimes: `0`" in summary
    assert "Rejoin market residual logloss-only help regimes: `6`" in summary
    assert (
        "Rejoin rank-first hypothesis watchlist: "
        "`RANK_FIRST_HYPOTHESIS_WATCHLIST_READY`"
    ) in summary
    assert "Rejoin rank-first watchlist candidates: `5`" in summary
    assert "Rejoin rank-first watchlist directional-ready candidates: `0`" in summary
    assert (
        "Rejoin rank-first watchlist best candidate: "
        "`rank_first_hypothesis_venue_eq_gee__raw_stage2_uncalibrated`"
    ) in summary
    assert (
        "Rejoin rank-first watchlist best status: "
        "`RANK_FIRST_HYPOTHESIS_WAITING_FOR_FRESH_SAMPLE`"
    ) in summary
    assert "Rejoin rank-first watchlist best distinct samples: `1` / `2`" in summary
    assert "Rejoin promotion distance: `PROMOTION_DISTANCE_BLOCKED`" in summary
    assert (
        "Rejoin promotion distance official-result coverage requested races: `69`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result requested race count source: "
        "`deduped_requested_or_inferred_race_ids`"
    ) in summary
    assert (
        "Rejoin promotion distance official-result legacy requested race count without IDs: "
        "`4125`"
    ) in summary
    assert (
        "Rejoin promotion distance official-result coverage races with rows: `35`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result coverage missing races: `35`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result missing exclusions: `13404`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result runner path count: `1`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result runner paths source: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in summary
    assert (
        "Rejoin promotion distance blockers: "
        "`['best_non_market_top1_margin_below_target']`"
    ) in summary


def test_final_summary_includes_next_prejump_refresh_window():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 84, "pending_races": 237, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        next_prejump_refresh_window={
            "status": "WAITING_FOR_FUTURE_WINDOW",
            "recommended_rerun_after_local": "2026-06-09T08:55:00+10:00",
            "next_race": {
                "race_id": "Race 1 - AP_K - 2026-06-09",
                "jump_datetime": "2026-06-09T11:35:00+10:00",
            },
        },
    )

    assert "Next pre-jump refresh status: `WAITING_FOR_FUTURE_WINDOW`" in summary
    assert "Recommended rerun after: `2026-06-09T08:55:00+10:00`" in summary
    assert "Race 1 - AP_K - 2026-06-09" in summary


def test_prejump_metadata_status_from_daily_run_reports_no_eligible(tmp_path):
    daily_dir = tmp_path / "daily_shadow"
    daily_dir.mkdir()
    daemon.write_json(
        daily_dir / "prejump_metadata_report.json",
        {
            "schema_version": "daily_shadow_prejump_metadata_report_v1",
            "status": "PASS",
            "eligible_count": 0,
            "eligible_with_verified_prejump_metadata": 0,
            "malformed_prejump_metadata_count": 0,
            "stale_with_prejump_metadata_count": 0,
            "required_fields": ["race_date", "venue", "target_distance"],
            "field_coverage": {
                "race_date": {"eligible_present_rows": 0},
                "venue": {"eligible_present_rows": 0},
                "target_distance": {"eligible_present_rows": 0},
            },
            "unsafe_or_incomplete_metadata": [],
            "rejected_metadata_sources": [],
        },
    )

    status = daemon.prejump_metadata_status_from_daily_run(daily_dir)

    assert status is not None
    assert status["status"] == "NO_ELIGIBLE_PREJUMP_RACES"
    assert status["eligible_count"] == 0
    assert status["eligible_with_verified_prejump_metadata"] == 0
    assert status["missing_required_fields"] == []


def test_prejump_metadata_status_from_daily_run_reports_missing_required_fields(tmp_path):
    daily_dir = tmp_path / "daily_shadow"
    daily_dir.mkdir()
    daemon.write_json(
        daily_dir / "prejump_metadata_report.json",
        {
            "schema_version": "daily_shadow_prejump_metadata_report_v1",
            "status": "PASS",
            "eligible_count": 2,
            "eligible_with_verified_prejump_metadata": 1,
            "malformed_prejump_metadata_count": 1,
            "stale_with_prejump_metadata_count": 0,
            "required_fields": ["race_date", "venue", "target_grade"],
            "field_coverage": {
                "race_date": {"eligible_present_rows": 2},
                "venue": {"eligible_present_rows": 2},
                "target_grade": {"eligible_present_rows": 1},
            },
            "unsafe_or_incomplete_metadata": [
                {
                    "csv_path": "upcoming_races/Race 1 - TEST - 2026-06-08.csv",
                    "missing_fields": ["target_grade"],
                }
            ],
            "rejected_metadata_sources": ["embedded_form_history"],
        },
    )

    status = daemon.prejump_metadata_status_from_daily_run(daily_dir)

    assert status is not None
    assert status["status"] == "PREJUMP_METADATA_PARTIAL"
    assert status["eligible_count"] == 2
    assert status["eligible_with_verified_prejump_metadata"] == 1
    assert status["malformed_prejump_metadata_count"] == 1
    assert status["unsafe_or_incomplete_metadata_count"] == 1
    assert status["missing_required_fields"] == ["target_grade"]
    assert status["rejected_metadata_sources"] == ["embedded_form_history"]


def _write_prejump_metadata_report(
    evidence_root: Path,
    run_name: str,
    *,
    eligible_count: int,
    verified_count: int,
    malformed_count: int = 0,
    field_present_rows: dict[str, int] | None = None,
    rejected_sources: list[str] | None = None,
) -> Path:
    required_fields = ["race_date", "venue", "target_distance", "target_grade"]
    present_rows = field_present_rows or {
        field: eligible_count for field in required_fields
    }
    report_dir = evidence_root / f"daily_race_ingest_shadow_{run_name}"
    report_dir.mkdir(parents=True)
    daemon.write_json(
        report_dir / "prejump_metadata_report.json",
        {
            "schema_version": "daily_shadow_prejump_metadata_report_v1",
            "status": "PASS",
            "eligible_count": eligible_count,
            "eligible_with_verified_prejump_metadata": verified_count,
            "malformed_prejump_metadata_count": malformed_count,
            "stale_with_prejump_metadata_count": 0,
            "required_fields": required_fields,
            "field_coverage": {
                field: {"eligible_present_rows": int(present_rows.get(field, 0))}
                for field in required_fields
            },
            "unsafe_or_incomplete_metadata": [],
            "rejected_metadata_sources": rejected_sources or [],
        },
    )
    return report_dir


def test_prejump_metadata_trend_report_passes_when_recent_eligible_runs_are_verified(tmp_path):
    _write_prejump_metadata_report(
        tmp_path,
        "20260609T010000+1000",
        eligible_count=0,
        verified_count=0,
    )
    _write_prejump_metadata_report(
        tmp_path,
        "20260609T020000+1000",
        eligible_count=2,
        verified_count=2,
    )

    trend = daemon.build_prejump_metadata_trend_report(
        evidence_root=tmp_path,
        output_dir=tmp_path / "daemon_packet",
        generated_at=daemon.datetime.fromisoformat("2026-06-09T02:30:00+10:00"),
    )

    assert trend["status"] == "PREJUMP_METADATA_TREND_PASS"
    assert trend["runs_checked"] == 2
    assert trend["runs_with_eligible_prejump_races"] == 1
    assert trend["runs_with_full_verified_metadata"] == 1
    assert trend["runs_needing_metadata_attention"] == 0
    assert trend["total_eligible_prejump_races"] == 2
    assert trend["total_verified_prejump_metadata_races"] == 2
    assert trend["verified_metadata_rate"] == 1.0
    assert trend["field_totals"]["target_grade"]["missing_rows"] == 0
    assert trend["missing_required_field_counts"] == {}
    assert trend["no_write_guarantees"]["db_write"] is False
    assert (tmp_path / "daemon_packet" / "prejump_metadata_trend_report.json").exists()


def test_prejump_metadata_trend_report_surfaces_missing_required_fields(tmp_path):
    _write_prejump_metadata_report(
        tmp_path,
        "20260609T030000+1000",
        eligible_count=2,
        verified_count=1,
        malformed_count=1,
        field_present_rows={
            "race_date": 2,
            "venue": 2,
            "target_distance": 2,
            "target_grade": 1,
        },
        rejected_sources=["embedded_form_history"],
    )

    trend = daemon.build_prejump_metadata_trend_report(
        evidence_root=tmp_path,
        output_dir=tmp_path / "daemon_packet",
        generated_at=daemon.datetime.fromisoformat("2026-06-09T03:30:00+10:00"),
    )

    assert trend["status"] == "PREJUMP_METADATA_TREND_NEEDS_ATTENTION"
    assert trend["total_eligible_prejump_races"] == 2
    assert trend["total_verified_prejump_metadata_races"] == 1
    assert trend["total_malformed_prejump_metadata_races"] == 1
    assert trend["runs_needing_metadata_attention"] == 1
    assert trend["verified_metadata_rate"] == 0.5
    assert trend["field_totals"]["target_grade"]["present_rows"] == 1
    assert trend["field_totals"]["target_grade"]["missing_rows"] == 1
    assert trend["missing_required_field_counts"] == {"target_grade": 1}
    assert trend["rejected_metadata_source_counts"] == {"embedded_form_history": 1}
    assert trend["runs"][0]["missing_required_fields"] == ["target_grade"]


def test_prejump_metadata_trend_report_does_not_pass_inconsistent_verified_counts(tmp_path):
    _write_prejump_metadata_report(
        tmp_path,
        "20260609T040000+1000",
        eligible_count=2,
        verified_count=2,
        field_present_rows={
            "race_date": 2,
            "venue": 2,
            "target_distance": 2,
            "target_grade": 1,
        },
    )

    trend = daemon.build_prejump_metadata_trend_report(
        evidence_root=tmp_path,
        output_dir=tmp_path / "daemon_packet",
        generated_at=daemon.datetime.fromisoformat("2026-06-09T04:30:00+10:00"),
    )

    assert trend["status"] == "PREJUMP_METADATA_TREND_NEEDS_ATTENTION"
    assert trend["total_verified_prejump_metadata_races"] == 2
    assert trend["runs_with_full_verified_metadata"] == 0
    assert trend["runs_needing_metadata_attention"] == 1
    assert trend["missing_required_field_counts"] == {"target_grade": 1}


def test_final_summary_includes_prejump_metadata_status():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 84, "pending_races": 237, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        prejump_metadata_status={
            "status": "PREJUMP_METADATA_PARTIAL",
            "eligible_count": 2,
            "eligible_with_verified_prejump_metadata": 1,
        },
    )

    assert "Pre-jump metadata: `PREJUMP_METADATA_PARTIAL`" in summary
    assert "Pre-jump metadata verified eligible: `1` / `2`" in summary


def test_final_summary_includes_prejump_metadata_trend():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 84, "pending_races": 237, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        prejump_metadata_trend={
            "status": "PREJUMP_METADATA_TREND_PASS",
            "verified_metadata_rate": 1.0,
        },
    )

    assert "Pre-jump metadata trend: `PREJUMP_METADATA_TREND_PASS`" in summary
    assert "Trend verified metadata rate: `1.0`" in summary


def test_live_odds_capture_packet_from_autopilot_surfaces_approval_artifact(tmp_path):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    autopilot_dir.mkdir()
    packet_path = autopilot_dir / "live_odds_capture_approval_packet.json"
    daemon.write_json(
        packet_path,
        {
            "status": "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS",
            "verified_prejump_race_count": 2,
            "capture_window_offsets_minutes": [60, 30, 10, 2],
            "can_capture_live_odds_now": False,
            "approval_required": True,
            "no_write_guarantees": {"db_write": False},
        },
    )

    packet = daemon.live_odds_capture_packet_from_autopilot(autopilot_dir)

    assert packet["status"] == "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS"
    assert packet["packet_path"].endswith("live_odds_capture_approval_packet.json")
    assert packet["verified_prejump_race_count"] == 2
    assert packet["capture_window_offsets_minutes"] == [60, 30, 10, 2]
    assert packet["can_capture_live_odds_now"] is False
    assert packet["no_write_guarantees"]["db_write"] is False


def test_autonomous_live_odds_capture_status_from_autopilot_surfaces_append_count(
    tmp_path,
):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "autonomous_live_odds_capture_status.json",
        {
            "run_id": "capture_x",
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "operator_status": "APPENDED",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
            "output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_live_odds_capture_capture_x"
            ),
            "attempted": True,
            "execute": True,
            "allow_auto_scrape_odds": True,
            "ready_count": 6,
            "validation_pass_count": 4,
            "inserted_live_odds_rows": 29,
            "status_counts": {"APPENDED": 4},
            "t2_miss_attempt_count": 1,
            "t2_miss_cause_counts": {"t2_miss_late_time_gate": 1},
            "t2_miss_examples": [
                {
                    "race_id": "Race 12 - TAREE - 2026-06-13",
                    "cause": "t2_miss_late_time_gate",
                }
            ],
            "next_prejump_window": {
                "status": "WAITING_FOR_FUTURE_WINDOW",
                "recommended_rerun_after_local": "2026-06-12T08:06:00+10:00",
                "next_window_opens_at": "2026-06-12T08:06:00+10:00",
                "next_race": {
                    "race_id": "Race 1 - HEA - 2026-06-12",
                    "jump_datetime": "2026-06-12T10:48:00+10:00",
                },
            },
            "next_window_opens_at": "2026-06-12T08:06:00+10:00",
            "recommended_rerun_after_local": "2026-06-12T08:06:00+10:00",
            "next_race_id": "Race 1 - HEA - 2026-06-12",
            "no_write_guarantees": {"production_promotion": False},
        },
    )

    status = daemon.autonomous_live_odds_capture_status_from_autopilot(autopilot_dir)

    assert status["status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert status["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert status["operator_status"] == "APPENDED"
    assert status["run_id"] == "capture_x"
    assert status["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert status["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert status["output_dir"].endswith("autonomous_live_odds_capture_capture_x")
    assert status["status_path"].endswith("autonomous_live_odds_capture_status.json")
    assert status["attempted"] is True
    assert status["execute"] is True
    assert status["allow_auto_scrape_odds"] is True
    assert status["ready_count"] == 6
    assert status["validation_pass_count"] == 4
    assert status["inserted_live_odds_rows"] == 29
    assert status["t2_miss_attempt_count"] == 1
    assert status["t2_miss_cause_counts"] == {"t2_miss_late_time_gate": 1}
    assert status["t2_miss_examples"][0]["race_id"] == (
        "Race 12 - TAREE - 2026-06-13"
    )
    assert status["append_only"] is True
    assert status["next_window_opens_at"] == "2026-06-12T08:06:00+10:00"
    assert status["recommended_rerun_after_local"] == "2026-06-12T08:06:00+10:00"
    assert status["next_race_id"] == "Race 1 - HEA - 2026-06-12"
    assert status["next_prejump_window"]["status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert status["no_write_guarantees"]["production_promotion"] is False


def test_t2_odds_capture_surface_fields_use_current_and_last_prefixes():
    autonomous_status = {
        "t2_miss_attempt_count": 2,
        "t2_miss_cause_counts": {
            "t2_miss_late_time_gate": 1,
            "t2_miss_validation_failed": 1,
        },
        "t2_miss_examples": [{"race_id": "Race 8 - TAREE - 2026-06-13"}],
    }
    odds_state = {
        "t2_miss_attempt_count": 0,
        "t2_miss_cause_counts": {"t2_miss_lock_held": 1},
        "t2_miss_lock_held": True,
        "t2_lock_skip_race_id": "Race 7 - TAREE - 2026-06-13",
        "t2_lock_skip_target_capture_at": "2026-06-13T09:35:00+10:00",
        "t2_lock_skip_minutes_to_jump": 1.5,
        "t2_lock_skip_lock_status": "SKIPPED_LOCK_HELD",
    }

    current = daemon.t2_odds_capture_surface_fields(
        autonomous_live_odds_capture_status=autonomous_status,
        odds_capture_state_publish=odds_state,
    )
    last = daemon.t2_odds_capture_surface_fields(
        autonomous_live_odds_capture_status=autonomous_status,
        odds_capture_state_publish=odds_state,
        last=True,
    )

    assert current["autonomous_live_odds_capture_t2_miss_attempt_count"] == 2
    assert current["autonomous_live_odds_capture_t2_miss_cause_counts"] == {
        "t2_miss_late_time_gate": 1,
        "t2_miss_validation_failed": 1,
    }
    assert current["odds_capture_t2_miss_lock_held"] is True
    assert current["odds_capture_t2_miss_cause_counts"] == {"t2_miss_lock_held": 1}
    assert current["odds_capture_t2_lock_skip_race_id"] == (
        "Race 7 - TAREE - 2026-06-13"
    )
    assert last["last_autonomous_live_odds_capture_t2_miss_attempt_count"] == 2
    assert last["last_odds_capture_t2_miss_lock_held"] is True
    assert last["last_odds_capture_t2_lock_skip_target_capture_at"] == (
        "2026-06-13T09:35:00+10:00"
    )


def test_autonomous_official_result_capture_status_from_autopilot_surfaces_counts(
    tmp_path,
):
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    autopilot_dir.mkdir()
    daemon.write_json(
        autopilot_dir / "autonomous_official_result_capture_status.json",
        {
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "output_dir": "artifacts/autonomous_official_result_capture_test",
            "attempted": True,
            "candidate_count": 9,
            "ingested_count": 7,
            "failed_count": 1,
            "skipped_count": 1,
            "skipped_reason_counts": {"race_not_jumped": 4},
            "awaiting_jump_race_count": 2,
            "awaiting_jump_race_ids": [
                "Race 7 - CANN - 2026-06-13",
                "Race 12 - GRDN - 2026-06-13",
            ],
            "awaiting_jump_next_recheck_after_local": "2026-06-13T22:55:00+10:00",
            "awaiting_jump_races": [
                {
                    "race_id": "Race 7 - CANN - 2026-06-13",
                    "reason": "race_not_jumped",
                }
            ],
            "official_result_race_rows": 7,
            "official_result_runner_rows": 43,
            "quarantine_rows": 70,
            "quarantined_race_ids": ["Race 8 - TAREE - 2026-06-13"],
            "quarantine_reason_counts": {"ingest_failed_or_unsafe_match": 1},
            "quarantine_error_counts": {
                "result_boxes_not_in_participants:9,10": 1
            },
            "quarantine_result_boxes_not_in_participants_counts": {
                "10": 1,
                "9": 1,
            },
            "quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "result_boxes_not_in_participants": [9, 10],
                    "attempted_source_box_sets": [
                        {
                            "source": "thedogs_official",
                            "result_boxes": [3, 4, 9, 1, 2, 8, 7, 10],
                        }
                    ],
                }
            ],
            "official_result_evidence_db_ingest_status": "READY_NOT_EXECUTED",
            "official_result_evidence_db_execute": False,
            "official_result_evidence_db_write_performed": False,
            "official_result_evidence_valid_race_rows": 7,
            "official_result_evidence_valid_runner_rows": 43,
            "official_result_evidence_blocked_race_rows": 1,
            "official_result_evidence_blocked_runner_rows": 8,
            "official_result_evidence_inserted_race_rows": 0,
            "official_result_evidence_inserted_runner_rows": 0,
            "official_result_evidence_blocker_reason_counts": {
                "runner_set_mismatch_quarantined": 1,
            },
            "live_odds_backlog_enabled": True,
            "live_odds_backlog_lookback_days": 2,
            "live_odds_backlog_target_dates": ["2026-06-11", "2026-06-10"],
            "live_odds_backlog_discovered_race_count": 11,
            "live_odds_backlog_discovered_race_ids": [
                "Race 1 - BEN - 2026-06-11",
                "Race 2 - BEN - 2026-06-10",
            ],
            "live_odds_backlog_candidate_race_count": 8,
            "live_odds_backlog_candidate_race_ids": [
                "Race 1 - BEN - 2026-06-11"
            ],
            "live_odds_backlog_unresolved_race_count": 3,
            "live_odds_backlog_unresolved_race_ids": [
                "Race 2 - BEN - 2026-06-10"
            ],
            "live_odds_backlog_unresolved_races": [
                {
                    "race_id": "Race 2 - BEN - 2026-06-10",
                    "reason": "no_matching_shadow_run_candidate_found",
                    "race_date": "2026-06-10",
                    "recovery_action": "validate_runner_set_then_alias_join",
                    "alias_reconciliation_status": "EXACT_SHADOW_ARTIFACT_MATCH_FOUND",
                }
            ],
            "live_odds_backlog_unresolved_reason_counts": {
                "no_matching_shadow_run_candidate_found": 1
            },
            "live_odds_backlog_recovery_queue_path": (
                "artifacts/autonomous_official_result_capture_test/"
                "live_odds_backlog_recovery_queue.json"
            ),
            "live_odds_backlog_recovery_queue_diagnostic_only": True,
            "live_odds_backlog_recovery_queue_join_acceptance_changed": False,
            "live_odds_backlog_recovery_queue_db_write_performed": False,
            "live_odds_backlog_awaiting_official_result_evidence_race_count": 1,
            "live_odds_backlog_awaiting_official_result_evidence_race_ids": [
                "Race 2 - BEN - 2026-06-10"
            ],
            "live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
                "diagnostic_recheck_official_result_evidence_only"
            ),
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": 1,
            "live_odds_backlog_runner_set_validation_path": (
                "artifacts/autonomous_official_result_capture_test/"
                "live_odds_backlog_runner_set_validation.json"
            ),
            "live_odds_backlog_runner_set_validation_retryable_race_count": 1,
            "live_odds_backlog_runner_set_validation_exact_match_race_count": 1,
            "live_odds_backlog_runner_set_validation_blocked_race_count": 0,
            "live_odds_backlog_runner_set_validation_diagnostic_only": True,
            "live_odds_backlog_runner_set_validation_join_authorized": False,
            "live_odds_backlog_runner_set_validation_db_write_performed": False,
            "live_odds_backlog_join_eligibility_packet_path": (
                "artifacts/autonomous_official_result_capture_test/"
                "live_odds_backlog_join_eligibility_packet.json"
            ),
            "live_odds_backlog_join_eligibility_evaluated_race_count": 1,
            "live_odds_backlog_join_eligibility_eligible_report_only_race_count": 1,
            "live_odds_backlog_join_eligibility_blocked_race_count": 0,
            "live_odds_backlog_join_eligibility_blocker_counts": {
                "official_result_runner_set_exact_live_odds_match": 1,
            },
            "live_odds_backlog_join_eligibility_diagnostic_only": True,
            "live_odds_backlog_join_eligibility_join_authorized": False,
            "live_odds_backlog_join_eligibility_db_write_performed": False,
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 1,
            "shadow_run_candidate_source_report": (
                "artifacts/autonomous_official_result_capture_test/"
                "shadow_run_candidate_source_report.json"
            ),
            "candidate_source": "shadow_run_predictions",
            "target_date": "2026-06-11",
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
    )

    status = daemon.autonomous_official_result_capture_status_from_autopilot(
        autopilot_dir
    )

    assert status["status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    assert status["status_path"].endswith(
        "autonomous_official_result_capture_status.json"
    )
    assert status["attempted"] is True
    assert status["candidate_count"] == 9
    assert status["ingested_count"] == 7
    assert status["failed_count"] == 1
    assert status["skipped_count"] == 1
    assert status["skipped_reason_counts"] == {"race_not_jumped": 4}
    assert status["awaiting_jump_race_count"] == 2
    assert status["awaiting_jump_race_ids"] == [
        "Race 7 - CANN - 2026-06-13",
        "Race 12 - GRDN - 2026-06-13",
    ]
    assert status["awaiting_jump_next_recheck_after_local"] == (
        "2026-06-13T22:55:00+10:00"
    )
    assert status["awaiting_jump_races"][0]["reason"] == "race_not_jumped"
    assert status["official_result_race_rows"] == 7
    assert status["official_result_runner_rows"] == 43
    assert status["quarantine_rows"] == 70
    assert status["quarantined_race_ids"] == ["Race 8 - TAREE - 2026-06-13"]
    assert status["quarantine_reason_counts"] == {
        "ingest_failed_or_unsafe_match": 1
    }
    assert status["quarantine_error_counts"] == {
        "result_boxes_not_in_participants:9,10": 1
    }
    assert status["quarantine_result_boxes_not_in_participants_counts"] == {
        "10": 1,
        "9": 1,
    }
    assert status["quarantine_runner_set_mismatch_samples"][0][
        "result_boxes_not_in_participants"
    ] == [9, 10]
    assert status["official_result_evidence_db_ingest_status"] == "READY_NOT_EXECUTED"
    assert status["official_result_evidence_db_execute"] is False
    assert status["official_result_evidence_db_write_performed"] is False
    assert status["official_result_evidence_valid_race_rows"] == 7
    assert status["official_result_evidence_valid_runner_rows"] == 43
    assert status["official_result_evidence_blocked_race_rows"] == 1
    assert status["official_result_evidence_blocked_runner_rows"] == 8
    assert status["official_result_evidence_inserted_race_rows"] == 0
    assert status["official_result_evidence_inserted_runner_rows"] == 0
    assert status["official_result_evidence_blocker_reason_counts"] == {
        "runner_set_mismatch_quarantined": 1,
    }
    assert status["live_odds_backlog_enabled"] is True
    assert status["live_odds_backlog_lookback_days"] == 2
    assert status["live_odds_backlog_target_dates"] == ["2026-06-11", "2026-06-10"]
    assert status["live_odds_backlog_discovered_race_count"] == 11
    assert status["live_odds_backlog_discovered_race_ids"] == [
        "Race 1 - BEN - 2026-06-11",
        "Race 2 - BEN - 2026-06-10",
    ]
    assert status["live_odds_backlog_candidate_race_count"] == 8
    assert status["live_odds_backlog_candidate_race_ids"] == [
        "Race 1 - BEN - 2026-06-11"
    ]
    assert status["live_odds_backlog_unresolved_race_count"] == 3
    assert status["live_odds_backlog_unresolved_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert status["live_odds_backlog_unresolved_races"][0]["reason"] == (
        "no_matching_shadow_run_candidate_found"
    )
    assert status["live_odds_backlog_unresolved_reason_counts"] == {
        "no_matching_shadow_run_candidate_found": 1
    }
    assert status["live_odds_backlog_join_eligibility_blocker_counts"] == {
        "official_result_runner_set_exact_live_odds_match": 1,
    }
    backlog = daemon.live_odds_backlog_operational_fields(status)
    assert backlog["live_odds_backlog_unresolved_race_count"] == 3
    assert backlog["live_odds_backlog_unresolved_reason_counts"] == {
        "no_matching_shadow_run_candidate_found": 1
    }
    assert backlog["live_odds_backlog_unresolved_recovery_action_counts"] == {
        "validate_runner_set_then_alias_join": 1
    }
    assert backlog["live_odds_backlog_unresolved_alias_status_counts"] == {
        "EXACT_SHADOW_ARTIFACT_MATCH_FOUND": 1
    }
    assert backlog["live_odds_backlog_retryable_exact_shadow_match_race_count"] == 1
    assert backlog["live_odds_backlog_no_exact_shadow_match_race_count"] == 0
    assert backlog["live_odds_backlog_retryable_exact_shadow_match_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert backlog["live_odds_backlog_no_exact_shadow_match_race_ids"] == []
    assert backlog["live_odds_backlog_awaiting_official_result_evidence_race_count"] == 1
    assert backlog["live_odds_backlog_awaiting_official_result_evidence_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert backlog[
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
    ] == "diagnostic_recheck_official_result_evidence_only"
    assert (
        backlog[
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        ]
        == 1
    )
    assert backlog["live_odds_backlog_recovery_queue_path"].endswith(
        "live_odds_backlog_recovery_queue.json"
    )
    assert backlog["live_odds_backlog_recovery_queue_diagnostic_only"] is True
    assert backlog["live_odds_backlog_recovery_queue_join_acceptance_changed"] is False
    assert backlog["live_odds_backlog_recovery_queue_db_write_performed"] is False
    assert backlog["live_odds_backlog_runner_set_validation_path"].endswith(
        "live_odds_backlog_runner_set_validation.json"
    )
    assert backlog["live_odds_backlog_runner_set_validation_retryable_race_count"] == 1
    assert backlog["live_odds_backlog_runner_set_validation_exact_match_race_count"] == 1
    assert backlog["live_odds_backlog_runner_set_validation_blocked_race_count"] == 0
    assert backlog["live_odds_backlog_runner_set_validation_diagnostic_only"] is True
    assert backlog["live_odds_backlog_runner_set_validation_join_authorized"] is False
    assert backlog["live_odds_backlog_runner_set_validation_db_write_performed"] is False
    assert backlog["live_odds_backlog_join_eligibility_packet_path"].endswith(
        "live_odds_backlog_join_eligibility_packet.json"
    )
    assert backlog["live_odds_backlog_join_eligibility_evaluated_race_count"] == 1
    assert (
        backlog["live_odds_backlog_join_eligibility_eligible_report_only_race_count"]
        == 1
    )
    assert backlog["live_odds_backlog_join_eligibility_blocked_race_count"] == 0
    assert backlog["live_odds_backlog_join_eligibility_blocker_counts"] == {
        "official_result_runner_set_exact_live_odds_match": 1,
    }
    assert backlog["live_odds_backlog_join_eligibility_diagnostic_only"] is True
    assert backlog["live_odds_backlog_join_eligibility_join_authorized"] is False
    assert backlog["live_odds_backlog_join_eligibility_db_write_performed"] is False
    assert (
        backlog[
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
        ]
        == 1
    )
    state_fields = daemon.live_odds_backlog_state_fields(status)
    assert state_fields["last_live_odds_backlog_unresolved_race_count"] == 3
    assert state_fields[
        "last_live_odds_backlog_unresolved_recovery_action_counts"
    ] == {
        "validate_runner_set_then_alias_join": 1
    }
    assert state_fields["last_live_odds_backlog_retryable_exact_shadow_match_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert state_fields["last_live_odds_backlog_recovery_queue_path"].endswith(
        "live_odds_backlog_recovery_queue.json"
    )
    assert state_fields["last_live_odds_backlog_recovery_queue_diagnostic_only"] is True
    assert state_fields["last_live_odds_backlog_recovery_queue_db_write_performed"] is False
    assert (
        state_fields[
            "last_live_odds_backlog_awaiting_official_result_evidence_authorized_action"
        ]
        == "diagnostic_recheck_official_result_evidence_only"
    )
    assert state_fields["last_live_odds_backlog_runner_set_validation_path"].endswith(
        "live_odds_backlog_runner_set_validation.json"
    )
    assert state_fields["last_live_odds_backlog_runner_set_validation_join_authorized"] is False
    assert (
        state_fields["last_live_odds_backlog_runner_set_validation_db_write_performed"]
        is False
    )
    assert state_fields["last_live_odds_backlog_join_eligibility_packet_path"].endswith(
        "live_odds_backlog_join_eligibility_packet.json"
    )
    assert (
        state_fields[
            "last_live_odds_backlog_join_eligibility_eligible_report_only_race_count"
        ]
        == 1
    )
    assert state_fields["last_live_odds_backlog_join_eligibility_blocker_counts"] == {
        "official_result_runner_set_exact_live_odds_match": 1,
    }
    assert state_fields["last_live_odds_backlog_join_eligibility_join_authorized"] is False
    assert (
        state_fields["last_live_odds_backlog_join_eligibility_db_write_performed"]
        is False
    )
    assert (
        state_fields[
            "last_live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
        ]
        == 1
    )
    assert status["shadow_run_candidate_source_report"].endswith(
        "shadow_run_candidate_source_report.json"
    )
    assert status["candidate_source"] == "shadow_run_predictions"
    assert status["target_date"] == "2026-06-11"
    assert status["no_write_guarantees"]["db_write"] is False
    assert status["no_write_guarantees"]["label_write"] is False


def test_autonomous_official_result_capture_status_surfaces_in_progress_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    autopilot_dir = evidence_root / "shadow_autopilot_v1_20260613T201800+1000_daemon"
    capture_dir = (
        evidence_root
        / "autonomous_official_result_capture_20260613T201800+1000_daemon_autopilot"
    )
    autopilot_dir.mkdir(parents=True)
    capture_dir.mkdir(parents=True)
    daemon.write_json(
        capture_dir / "autonomous_official_result_capture_progress.json",
        {
            "schema_version": "autonomous_official_result_capture_progress_v1",
            "candidate_count": 109,
            "completed_count": 48,
            "status_counts": {
                "FAILED_VALIDATION": 8,
                "FETCH_IN_PROGRESS": 1,
                "INGESTED_DRY_RUN": 40,
            },
            "active_candidate": {
                "race_id": "Race 1 - BEN - 2026-06-12",
                "candidate_index": 49,
                "candidate_count": 109,
                "status": "FETCH_IN_PROGRESS",
            },
        },
    )

    status = daemon.autonomous_official_result_capture_status_from_autopilot(
        autopilot_dir
    )

    assert status["status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_IN_PROGRESS"
    assert status["status_path"].endswith(
        "autonomous_official_result_capture_status.json"
    )
    assert status["output_dir"].endswith(
        "autonomous_official_result_capture_20260613T201800+1000_daemon_autopilot"
    )
    assert status["attempted"] is True
    assert status["candidate_count"] == 109
    assert status["progress_candidate_count"] == 109
    assert status["progress_completed_count"] == 48
    assert status["progress_status_counts"] == {
        "FAILED_VALIDATION": 8,
        "FETCH_IN_PROGRESS": 1,
        "INGESTED_DRY_RUN": 40,
    }
    assert status["progress_active_candidate"]["race_id"] == (
        "Race 1 - BEN - 2026-06-12"
    )
    assert status["progress_active_candidate"]["candidate_index"] == 49
    assert status["official_result_evidence_db_write_performed"] is False
    assert status["live_odds_backlog_join_eligibility_join_authorized"] is False
    assert status["no_write_guarantees"]["label_write"] is False


def test_autonomous_official_result_capture_status_falls_back_to_recovery_queue(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    capture_dir = tmp_path / "artifacts/autonomous_official_result_capture_test"
    autopilot_dir.mkdir()
    capture_dir.mkdir(parents=True)
    daemon.write_json(
        autopilot_dir / "autonomous_official_result_capture_status.json",
        {
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "output_dir": daemon.relpath(capture_dir),
            "attempted": True,
            "live_odds_backlog_recovery_queue_path": daemon.relpath(
                capture_dir / "live_odds_backlog_recovery_queue.json"
            ),
        },
    )
    daemon.write_json(
        capture_dir / "live_odds_backlog_recovery_queue.json",
        {
            "schema_version": "live_odds_backlog_recovery_queue_v1",
            "queues": {
                "awaiting_official_result_evidence": {
                    "race_count": 2,
                    "race_ids": [
                        "Race 7 - GEE - 2026-06-12",
                        "Race 8 - GEE - 2026-06-12",
                    ],
                    "authorized_action": (
                        "diagnostic_recheck_official_result_evidence_only"
                    ),
                    "recheck_plan": {
                        "recheck_ready_race_count": 2,
                    },
                }
            },
        },
    )

    status = daemon.autonomous_official_result_capture_status_from_autopilot(
        autopilot_dir
    )

    assert status["live_odds_backlog_awaiting_official_result_evidence_race_count"] == 2
    assert status["live_odds_backlog_awaiting_official_result_evidence_race_ids"] == [
        "Race 7 - GEE - 2026-06-12",
        "Race 8 - GEE - 2026-06-12",
    ]
    assert status[
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
    ] == "diagnostic_recheck_official_result_evidence_only"
    assert (
        status[
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        ]
        == 2
    )


def test_autonomous_official_result_capture_status_fills_missing_recheck_count(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    autopilot_dir = tmp_path / "shadow_autopilot_v1_test"
    capture_dir = tmp_path / "artifacts/autonomous_official_result_capture_test"
    autopilot_dir.mkdir()
    capture_dir.mkdir(parents=True)
    daemon.write_json(
        autopilot_dir / "autonomous_official_result_capture_status.json",
        {
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "output_dir": daemon.relpath(capture_dir),
            "attempted": True,
            "live_odds_backlog_recovery_queue_path": daemon.relpath(
                capture_dir / "live_odds_backlog_recovery_queue.json"
            ),
            "live_odds_backlog_awaiting_official_result_evidence_race_count": 7,
            "live_odds_backlog_awaiting_official_result_evidence_race_ids": [
                "Race 10 - GEE - 2026-06-12"
            ],
        },
    )
    daemon.write_json(
        capture_dir / "live_odds_backlog_recovery_queue.json",
        {
            "schema_version": "live_odds_backlog_recovery_queue_v1",
            "queues": {
                "awaiting_official_result_evidence": {
                    "race_count": 7,
                    "race_ids": ["Race 10 - GEE - 2026-06-12"],
                    "authorized_action": (
                        "diagnostic_recheck_official_result_evidence_only"
                    ),
                    "recheck_plan": {
                        "recheck_ready_race_count": 1,
                    },
                }
            },
        },
    )

    status = daemon.autonomous_official_result_capture_status_from_autopilot(
        autopilot_dir
    )

    assert status["live_odds_backlog_awaiting_official_result_evidence_race_count"] == 7
    assert (
        status[
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        ]
        == 1
    )
    assert status[
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
    ] == "diagnostic_recheck_official_result_evidence_only"


def test_read_only_odds_coverage_report_does_not_mutate_db(tmp_path):
    db_path = tmp_path / "odds.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE live_odds (
                id INTEGER PRIMARY KEY,
                race_id TEXT,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                market_type TEXT,
                source TEXT,
                timestamp TEXT,
                is_current INTEGER,
                source_url TEXT
            );
            CREATE TABLE odds_history (
                id INTEGER PRIMARY KEY,
                race_id TEXT,
                dog_clean_name TEXT,
                odds_decimal REAL,
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
            INSERT INTO race_metadata
                (race_id, venue, race_number, race_date, start_datetime)
            VALUES
                ('R1', 'Sandown', 1, '2026-06-08', '2026-06-08T22:00:00+10:00'),
                ('R2', 'Sandown', 2, '2026-06-08', '2026-06-08T23:00:00+10:00'),
                ('R3', 'Sandown', 3, '2026-06-08', '2026-06-08T23:30:00+10:00');
            INSERT INTO dog_race_data
                (race_id, dog_name, dog_clean_name, box_number)
            VALUES
                ('R1', 'Fast Dog', 'Fast Dog', 1),
                ('R2', 'Old Dog', 'Old Dog', 2),
                ('R3', 'No Prediction Dog', 'No Prediction Dog', 3);
            INSERT INTO live_odds
                (race_id, venue, race_number, race_date, dog_name, dog_clean_name,
                 box_number, odds_decimal, market_type, source, timestamp, is_current,
                 source_url)
            VALUES
                ('R1', 'Sandown', 1, '2026-06-08', '1. Fast Dog', '1. Fast Dog',
                 1, 3.5, 'win', 'sportsbet', '2026-06-08T21:45:00+10:00', 1,
                 'https://www.sportsbet.com.au/greyhound-racing/sandown/race-1'),
                ('R2', 'Sandown', 2, '2026-06-08', '2. Old Dog', '2. Old Dog',
                 2, 4.2, 'win', 'sportsbet', '2026-06-08T12:00:00+10:00', 1,
                 'https://www.sportsbet.com.au/greyhound-racing/sandown/race-1'),
                ('R3', 'Sandown', 3, '2026-06-08', '3. No Prediction Dog', '3. No Prediction Dog',
                 3, 6.0, 'win', 'sportsbet', '2026-06-08T21:46:00+10:00', 1,
                 'https://www.sportsbet.com.au/greyhound-racing/sandown/race-1');
            """
        )

    evidence_root = tmp_path / "evidence"
    prediction_dir = evidence_root / "daily_race_ingest_shadow_20260608T214600+1000"
    prediction_dir.mkdir(parents=True)
    daemon.write_json(
        prediction_dir / "shadow_manifest.json",
        {
            "final_status": "SHADOW_RUN_COMPLETE",
            "input_summary": {"eligible_count": 1},
        },
    )
    prediction_row = {"race_id": "R1", "box": 1, "dog_name": "Fast Dog"}
    daemon.write_jsonl(prediction_dir / "shadow_predictions.jsonl", [prediction_row])
    daemon.write_jsonl(prediction_dir / "stage2_shadow_predictions.jsonl", [prediction_row])

    before_hash = daemon.sha256_file(db_path)
    summary = daemon.build_read_only_odds_coverage_report(
        db_path=db_path,
        output_dir=tmp_path / "packet",
        generated_at=daemon.datetime.fromisoformat("2026-06-08T21:50:00+10:00"),
        evidence_root=evidence_root,
    )
    after_hash = daemon.sha256_file(db_path)

    assert before_hash == after_hash
    assert summary["status"] == "SUCCESS"
    assert summary["odds_capture_performed"] is False
    assert summary["odds_used_for_shadow_scoring"] is False
    assert summary["dog_level_win_odds_rows"] == 3
    assert summary["safe_direct_identity_matches"] == 3
    assert summary["fresh_current_win_rows"] == 2
    assert summary["fresh_safe_direct_identity_matches"] == 2
    assert summary["fresh_safe_direct_identity_match_rate"] == 1.0
    assert summary["fresh_unmatched_rows"] == 0
    assert summary["fresh_odds_prediction_races"] == 2
    assert summary["fresh_odds_races_with_primary_predictions"] == 1
    assert summary["fresh_odds_races_with_stage2_predictions"] == 1
    assert summary["fresh_odds_races_missing_prediction_artifact"] == 1
    assert summary["fresh_odds_runner_keys"] == 2
    assert summary["fresh_odds_runner_keys_with_primary_prediction_match"] == 1
    assert summary["fresh_odds_runner_keys_with_stage2_prediction_match"] == 1
    assert summary["fresh_odds_runner_keys_missing_primary_prediction_match"] == 1
    assert summary["fresh_odds_runner_keys_missing_stage2_prediction_match"] == 1
    assert summary["old_odds_row_audit"]["stale_rows"] == 1
    assert summary["old_odds_row_audit"]["missing_source_url_rows"] == 0
    assert summary["old_odds_row_audit"]["race_id_mismatch_rows"] == 0
    assert summary["old_odds_row_audit"]["dog_name_box_conflict_rows"] == 0
    report = json.loads((tmp_path / "packet" / "odds_coverage_report.json").read_text())
    assert report["fresh_strict_identity_split"]["fresh_current_win_rows"] == 2
    assert report["fresh_strict_identity_split"]["stale_current_win_rows"] == 1
    assert report["fresh_strict_identity_split"]["fresh_safe_direct_identity_matches"] == 2
    assert report["fresh_strict_identity_split"]["stale_safe_direct_identity_matches"] == 1
    prediction_coverage = report["fresh_odds_shadow_prediction_coverage"]
    assert prediction_coverage["fresh_current_win_odds_races"] == 2
    assert prediction_coverage["races_missing_prediction_artifact"] == 1
    assert prediction_coverage["missing_prediction_race_ids_sample"] == ["R3"]
    assert prediction_coverage["fresh_odds_runner_keys"] == 2
    assert prediction_coverage["fresh_odds_runner_keys_with_primary_prediction_match"] == 1
    assert prediction_coverage["fresh_odds_runner_keys_with_stage2_prediction_match"] == 1
    assert (tmp_path / "packet" / "odds_coverage_report.json").exists()


def test_final_summary_includes_odds_coverage_diagnostic():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 83, "pending_races": 238, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 1},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        odds_coverage={
            "status": "SUCCESS",
            "odds_used_for_shadow_scoring": False,
        },
    )

    assert "Odds coverage diagnostic: `SUCCESS`" in summary
    assert "Odds used for shadow scoring: `False`" in summary


def test_final_summary_includes_live_odds_capture_packet():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 84, "pending_races": 237, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        live_odds_capture_packet={
            "status": "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS",
            "verified_prejump_race_count": 2,
            "capture_window_offsets_minutes": [60, 30, 10, 2],
            "can_capture_live_odds_now": False,
        },
    )

    assert "Live odds capture approval: `AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS`" in summary
    assert "Live odds verified races: `2`" in summary
    assert "Live odds capture windows: `[60, 30, 10, 2]`" in summary


def test_final_summary_includes_live_odds_backlog_blockers():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={
            "safe_joined_races": 120,
            "pending_races": 29,
            "unsafe_matches": 20,
            "live_odds_backlog_discovered_race_count": 125,
            "live_odds_backlog_candidate_race_count": 114,
            "live_odds_backlog_unresolved_race_count": 11,
            "live_odds_backlog_unresolved_reason_counts": {
                "shadow_run_candidate_rejected": 5,
                "no_matching_shadow_run_candidate_found": 3,
            },
            "live_odds_backlog_unresolved_recovery_action_counts": {
                "validate_runner_set_then_alias_join": 5,
                "inspect_missing_shadow_candidate": 3,
            },
            "live_odds_backlog_unresolved_alias_status_counts": {
                "EXACT_SHADOW_ARTIFACT_MATCH_FOUND": 5,
                "NO_EXACT_SHADOW_ARTIFACT_MATCH": 6,
            },
            "live_odds_backlog_retryable_exact_shadow_match_race_count": 5,
            "live_odds_backlog_no_exact_shadow_match_race_count": 6,
            "live_odds_backlog_retryable_exact_shadow_match_race_ids": [
                "Race 1 - GRDN - 2026-06-12"
            ],
            "live_odds_backlog_no_exact_shadow_match_race_ids": [
                "ASCOT PARK_2026-06-10_6"
            ],
            "live_odds_backlog_awaiting_official_result_evidence_race_count": 5,
            "live_odds_backlog_awaiting_official_result_evidence_race_ids": [
                "Race 1 - GRDN - 2026-06-12"
            ],
            "live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
                "diagnostic_recheck_official_result_evidence_only"
            ),
            "live_odds_backlog_recovery_queue_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_recovery_queue.json"
            ),
            "live_odds_backlog_recovery_queue_diagnostic_only": True,
            "live_odds_backlog_recovery_queue_join_acceptance_changed": False,
            "live_odds_backlog_recovery_queue_db_write_performed": False,
            "live_odds_backlog_runner_set_validation_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_runner_set_validation.json"
            ),
            "live_odds_backlog_runner_set_validation_retryable_race_count": 5,
            "live_odds_backlog_runner_set_validation_exact_match_race_count": 5,
            "live_odds_backlog_runner_set_validation_blocked_race_count": 0,
            "live_odds_backlog_runner_set_validation_diagnostic_only": True,
            "live_odds_backlog_runner_set_validation_join_authorized": False,
            "live_odds_backlog_runner_set_validation_db_write_performed": False,
            "live_odds_backlog_join_eligibility_packet_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_join_eligibility_packet.json"
            ),
            "live_odds_backlog_join_eligibility_evaluated_race_count": 5,
            "live_odds_backlog_join_eligibility_eligible_report_only_race_count": 2,
            "live_odds_backlog_join_eligibility_blocked_race_count": 3,
            "live_odds_backlog_join_eligibility_blocker_counts": {
                "official_result_runner_set_exact_live_odds_match": 2,
                "prejump_timing_verified": 1,
            },
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 1,
            "live_odds_backlog_join_eligibility_diagnostic_only": True,
            "live_odds_backlog_join_eligibility_join_authorized": False,
            "live_odds_backlog_join_eligibility_db_write_performed": False,
        },
        readiness={"decision": "READY_FOR_RELIABILITY_REVIEW", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 1},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "10min"},
        rejoin_unified_evidence_status={
            "status": "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT",
            "status_reason": "rejoin_unified_evidence_datasets_built",
            "evaluated_dataset_candidate_count": 2,
            "dataset_count": 1,
            "skipped_dataset_count": 1,
            "skip_reason_counts": {"safe_joined_race_count_zero": 1},
            "failure_reason_counts": {},
            "unified_evidence_eligible_rows": 8,
            "rejected_live_odds_candidate_count": 5,
            "rows_with_rejected_live_odds_candidates": 4,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_source_url_missing": 3,
                "odds_decimal_invalid": 2,
            },
            "join_eligibility_preview_dataset_count": 1,
            "join_eligibility_preview_unified_eligible_rows": 27,
            "join_eligibility_preview_packet_accepted_races": 4,
            "join_eligibility_preview_packet_present_races": 4,
            "join_eligibility_preview_missing_race_ids": [],
        },
    )

    assert "Live odds backlog discovered races: `125`" in summary
    assert "Live odds backlog candidate races: `114`" in summary
    assert "Live odds backlog unresolved races: `11`" in summary
    assert "shadow_run_candidate_rejected" in summary
    assert "validate_runner_set_then_alias_join" in summary
    assert "EXACT_SHADOW_ARTIFACT_MATCH_FOUND" in summary
    assert "Live odds backlog retryable exact-shadow matches: `5`" in summary
    assert "Live odds backlog no exact shadow match: `6`" in summary
    assert "Live odds backlog awaiting official-result evidence races: `5`" in summary
    assert "Live odds backlog awaiting official-result authorized action: `diagnostic_recheck_official_result_evidence_only`" in summary
    assert "Race 1 - GRDN - 2026-06-12" in summary
    assert "ASCOT PARK_2026-06-10_6" in summary
    assert "Live odds backlog recovery queue:" in summary
    assert "live_odds_backlog_recovery_queue.json" in summary
    assert "Live odds backlog recovery queue diagnostic only: `True`" in summary
    assert "Live odds backlog recovery queue changed join acceptance: `False`" in summary
    assert "Live odds backlog recovery queue DB write performed: `False`" in summary
    assert "Live odds backlog runner-set validation:" in summary
    assert "live_odds_backlog_runner_set_validation.json" in summary
    assert "Live odds backlog runner-set retryable races: `5`" in summary
    assert "Live odds backlog runner-set exact matches: `5`" in summary
    assert "Live odds backlog runner-set blocked races: `0`" in summary
    assert "Live odds backlog runner-set join authorized: `False`" in summary
    assert "Live odds backlog runner-set DB write performed: `False`" in summary
    assert "Live odds backlog join eligibility packet:" in summary
    assert "live_odds_backlog_join_eligibility_packet.json" in summary
    assert "Live odds backlog join eligibility evaluated races: `5`" in summary
    assert "Live odds backlog join eligibility report-only races: `2`" in summary
    assert "Live odds backlog join eligibility blocked races: `3`" in summary
    assert "Live odds backlog join eligibility blocker counts:" in summary
    assert "official_result_runner_set_exact_live_odds_match" in summary
    assert "prejump_timing_verified" in summary
    assert (
        "Live odds backlog join eligibility awaiting official-result recheck-ready "
        "races: `1`"
    ) in summary
    assert "Live odds backlog join eligibility join authorized: `False`" in summary
    assert "Live odds backlog join eligibility DB write performed: `False`" in summary
    assert "Rejoin unified evidence reason: `rejoin_unified_evidence_datasets_built`" in summary
    assert "Rejoin unified evaluated candidates: `2`" in summary
    assert "Rejoin unified skipped datasets: `1`" in summary
    assert "Rejoin unified skip reasons: `{'safe_joined_race_count_zero': 1}`" in summary
    assert "Rejoin unified failure reasons: `{}`" in summary
    assert "Rejoin rejected live odds candidates: `5`" in summary
    assert "Rejoin rows with rejected live odds candidates: `4`" in summary
    assert (
        "Rejoin rejected live odds candidate reasons: "
        "`{'odds_source_url_missing': 3, 'odds_decimal_invalid': 2}`"
    ) in summary
    assert "Join-eligibility preview datasets: `1`" in summary
    assert "Join-eligibility preview eligible rows: `27`" in summary
    assert "Join-eligibility preview accepted races: `4`" in summary
    assert "Join-eligibility preview present races: `4`" in summary


def test_final_summary_includes_shadow_odds_snapshot_status():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 84, "pending_races": 237, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 0},
        alert_report={"status": "NO_ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        odds_coverage={"status": "SUCCESS", "odds_used_for_shadow_scoring": False},
        shadow_odds_snapshot={
            "status": "SKIPPED",
            "valid_pre_jump_dog_odds_rows": 0,
            "races_with_post_feature_freeze_odds_rows": 0,
            "odds_research_next_action": (
                "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
            ),
            "timing_aligned_prediction_rerun_required": True,
            "timing_aligned_prediction_rerun_race_count": 2,
            "timing_aligned_prediction_rerun_race_ids": [
                "Race 10 - CANN - 2026-06-13",
                "Race 8 - CANN - 2026-06-13",
            ],
            "ev_output_rows": 0,
        },
    )

    assert "Shadow odds snapshot: `SKIPPED`" in summary
    assert "Shadow odds snapshot valid rows: `0`" in summary
    assert "Shadow odds races after feature freeze: `0`" in summary
    assert "Odds research next action: `RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE`" in summary
    assert "Timing-aligned prediction rerun required: `True`" in summary
    assert "Timing-aligned prediction rerun races: `2`" in summary
    assert "Timing-aligned prediction rerun race IDs: `['Race 10 - CANN - 2026-06-13', 'Race 8 - CANN - 2026-06-13']`" in summary
    assert "Shadow odds EV output rows: `0`" in summary


def test_build_rejoin_unified_evidence_datasets_uses_exact_join_artifact(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon.autopilot, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_test"
    shadow_dir = evidence_root / "daily_race_ingest_shadow_test"
    join_dir = evidence_root / "forward_shadow_result_join_test"
    output_dir.mkdir(parents=True)
    shadow_dir.mkdir(parents=True)
    join_dir.mkdir(parents=True)
    (shadow_dir / "shadow_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    (join_dir / "joined_shadow_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    db_path = tmp_path / "greyhound.sqlite"
    db_path.write_text("", encoding="utf-8")
    calls = []

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        calls.append(
            {
                "name": name,
                "command": list(command),
                "output_dir": output_dir,
                "timeout_seconds": timeout_seconds,
                "cwd": cwd,
            }
        )
        dataset_dir = Path(command[command.index("--output-dir") + 1])
        daemon.write_json(
            dataset_dir / "unified_evidence_dataset_report.json",
            {
                "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
                "output_dir": daemon.relpath(dataset_dir),
                "shadow_run_dir": daemon.relpath(shadow_dir),
                "row_count": 8,
                "race_count": 1,
                "rows_with_official_results": 8,
                "rows_with_strict_prejump_odds": 8,
                "unified_evidence_eligible_rows": 8,
            },
        )
        return {"name": name, "command": list(command), "returncode": 0, "status": "PASS"}

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    status, steps, report_paths = daemon.build_rejoin_unified_evidence_datasets(
        run_id="20260611T154107+1000",
        output_dir=output_dir,
        evidence_root=evidence_root,
        db_path=db_path,
        automated_join_report={
            "results": [
                {
                    "candidate": {"shadow_run_dir": daemon.relpath(shadow_dir)},
                    "join_dir": daemon.relpath(join_dir),
                    "metrics": {"safe_joined_race_count": 1},
                },
                {
                    "candidate": {"shadow_run_dir": daemon.relpath(shadow_dir)},
                    "join_dir": daemon.relpath(join_dir),
                    "metrics": {"safe_joined_race_count": 0},
                },
            ]
        },
        generated_at=daemon.datetime.fromisoformat("2026-06-11T15:41:07+10:00"),
        timeout_seconds=120,
    )

    assert status["status"] == "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    assert status["dataset_count"] == 1
    assert status["skipped_dataset_count"] == 1
    assert status["unified_evidence_eligible_rows"] == 8
    assert len(steps) == 1
    assert len(report_paths) == 1
    assert calls[0]["name"] == "rejoin_unified_evidence_dataset_001"
    assert "--joined-shadow-predictions-jsonl" in calls[0]["command"]
    assert str(join_dir / "joined_shadow_predictions.jsonl") in calls[0]["command"]


def test_rejoin_unified_evidence_status_explains_skipped_candidates():
    status = daemon.build_rejoin_unified_evidence_status(
        generated_at=daemon.datetime.fromisoformat("2026-06-13T14:17:11+10:00"),
        reports=[],
        failures=[],
        skipped=[
            {
                "reason": "joined_shadow_predictions_already_converted",
                "safe_joined_race_count": 4,
            },
            {
                "reason": "safe_joined_race_count_zero",
                "safe_joined_race_count": 0,
            },
        ],
    )

    assert status["status"] == "REJOIN_UNIFIED_EVIDENCE_DATASETS_SKIPPED"
    assert status["status_reason"] == (
        "all_rejoin_unified_evidence_dataset_candidates_skipped"
    )
    assert status["evaluated_dataset_candidate_count"] == 2
    assert status["attempted_dataset_count"] == 0
    assert status["skipped_dataset_count"] == 2
    assert status["skip_reason_counts"] == {
        "joined_shadow_predictions_already_converted": 1,
        "safe_joined_race_count_zero": 1,
    }
    assert status["failure_reason_counts"] == {}
    assert status["skipped_safe_joined_race_count"] == 4


def test_rejoin_unified_evidence_status_explains_no_candidates():
    status = daemon.build_rejoin_unified_evidence_status(
        generated_at=daemon.datetime.fromisoformat("2026-06-13T14:17:11+10:00"),
        reports=[],
        failures=[],
        skipped=[],
    )

    assert status["status"] == "REJOIN_UNIFIED_EVIDENCE_DATASETS_SKIPPED"
    assert status["status_reason"] == "no_rejoin_unified_evidence_dataset_candidates"
    assert status["evaluated_dataset_candidate_count"] == 0
    assert status["skip_reason_counts"] == {}
    assert status["failure_reason_counts"] == {}


def test_rejoin_unified_evidence_status_surfaces_join_eligibility_previews():
    status = daemon.build_rejoin_unified_evidence_status(
        generated_at=daemon.datetime.fromisoformat("2026-06-12T16:18:38+10:00"),
        reports=[
            {
                "output_dir": "artifacts/unified_regular",
                "shadow_run_dir": "artifacts/daily",
                "row_count": 8,
                "race_count": 1,
                "rows_with_official_results": 8,
                "rows_with_strict_prejump_odds": 8,
                "rows_with_artifact_shadow_odds": 3,
                "rows_with_artifact_shadow_odds_candidates": 4,
                "artifact_shadow_odds_candidate_count": 4,
                "artifact_shadow_odds_selected_bucket_count": 3,
                "artifact_odds_rows_seen": 8,
                "artifact_odds_rows_accepted": 3,
                "artifact_odds_rows_rejected": 5,
                "artifact_odds_audits": [
                    {
                        "rows_seen": 8,
                        "accepted_rows": 3,
                        "rejected_rows": 5,
                        "rejection_reason_counts": {
                            "odds_match_status_not_valid_pre_jump_dog_odds": 5
                        },
                    }
                ],
                "unified_evidence_eligible_rows": 8,
                "rejected_live_odds_candidate_count": 5,
                "rows_with_rejected_live_odds_candidates": 4,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 2,
                    "odds_source_url_missing": 3,
                },
            },
            {
                "output_dir": "artifacts/unified_regular_2",
                "shadow_run_dir": "artifacts/daily_2",
                "row_count": 4,
                "race_count": 1,
                "rows_with_official_results": 4,
                "rows_with_strict_prejump_odds": 4,
                "rows_with_artifact_shadow_odds": 2,
                "rows_with_artifact_shadow_odds_candidates": 3,
                "artifact_shadow_odds_candidate_count": 3,
                "artifact_shadow_odds_selected_bucket_count": 2,
                "artifact_odds_rows_seen": 4,
                "artifact_odds_rows_accepted": 2,
                "artifact_odds_rows_rejected": 2,
                "artifact_odds_audits": [
                    {
                        "rows_seen": 2,
                        "accepted_rows": 1,
                        "rejected_rows": 1,
                        "rejection_reason_counts": {
                            "odds_match_status_not_valid_pre_jump_dog_odds": 1
                        },
                    },
                    {
                        "rows_seen": 2,
                        "accepted_rows": 1,
                        "rejected_rows": 1,
                        "rejection_reason_counts": {
                            "source_url_missing": 1
                        },
                    },
                ],
                "unified_evidence_eligible_rows": 4,
                "rejected_live_odds_candidate_count": 3,
                "rows_with_rejected_live_odds_candidates": 2,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 1,
                    "unsupported_sportsbet_box_source:missing": 2,
                },
            }
        ],
        join_eligibility_preview_reports=[
            {
                "output_dir": "artifacts/unified_preview_partial",
                "shadow_run_dir": "artifacts/daily_preview_partial",
                "row_count": 13,
                "race_count": 2,
                "unified_evidence_eligible_rows": 13,
                "join_eligibility_packet_paths": [
                    "artifacts/live_odds_backlog_join_eligibility_packet.json"
                ],
                "join_eligibility_packet_accepted_races": 4,
                "join_eligibility_packet_accepted_races_present_in_shadow_run": 2,
                "join_eligibility_packet_rejected_races": 1,
                "join_eligibility_packet_accepted_race_ids_missing_from_shadow_run": [
                    "Race 2 - TEST - 2026-06-12"
                ],
            },
            {
                "output_dir": "artifacts/unified_preview",
                "shadow_run_dir": "artifacts/daily_preview",
                "row_count": 27,
                "race_count": 4,
                "unified_evidence_eligible_rows": 27,
                "join_eligibility_packet_paths": [
                    "artifacts/live_odds_backlog_join_eligibility_packet.json"
                ],
                "join_eligibility_packet_accepted_races": 4,
                "join_eligibility_packet_accepted_races_present_in_shadow_run": 4,
                "join_eligibility_packet_rejected_races": 1,
                "join_eligibility_packet_accepted_race_ids_missing_from_shadow_run": [],
            }
        ],
    )

    assert status["status"] == "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    assert status["dataset_count"] == 2
    assert status["row_count"] == 12
    assert status["race_count"] == 2
    assert status["rows_with_artifact_shadow_odds"] == 5
    assert status["rows_with_artifact_shadow_odds_candidates"] == 7
    assert status["artifact_shadow_odds_candidate_count"] == 7
    assert status["artifact_shadow_odds_selected_bucket_count"] == 5
    assert status["artifact_odds_rows_seen"] == 12
    assert status["artifact_odds_rows_accepted"] == 5
    assert status["artifact_odds_rows_rejected"] == 7
    assert status["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 6,
        "source_url_missing": 1,
    }
    assert status["rejected_live_odds_candidate_count"] == 8
    assert status["rows_with_rejected_live_odds_candidates"] == 6
    assert status["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 3,
        "unsupported_sportsbet_box_source:missing": 2,
    }
    assert status["reports"][0]["artifact_odds_rows_accepted"] == 3
    assert status["reports"][0]["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 5
    }
    assert status["reports"][0]["rejected_live_odds_candidate_count"] == 5
    assert status["reports"][0]["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert status["reports"][1]["rows_with_artifact_shadow_odds"] == 2
    assert status["reports"][1]["rows_with_artifact_shadow_odds_candidates"] == 3
    assert status["reports"][1]["artifact_shadow_odds_candidate_count"] == 3
    assert status["reports"][1]["artifact_shadow_odds_selected_bucket_count"] == 2
    assert status["reports"][1]["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 1,
        "source_url_missing": 1,
    }
    assert status["reports"][1]["rows_with_rejected_live_odds_candidates"] == 2
    assert status["reports"][1]["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 1,
        "unsupported_sportsbet_box_source:missing": 2,
    }
    assert status["join_eligibility_preview_dataset_count"] == 1
    assert status["join_eligibility_preview_unified_eligible_rows"] == 27
    assert status["join_eligibility_preview_packet_accepted_races"] == 4
    assert status["join_eligibility_preview_packet_present_races"] == 4
    assert status["join_eligibility_preview_packet_rejected_races"] == 1
    assert status["join_eligibility_preview_missing_race_ids"] == []
    assert status["join_eligibility_preview_reports"][0]["row_count"] == 27


def test_discovered_join_eligibility_preview_reports(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    preview_dir = evidence_root / "unified_evidence_dataset_preview"
    better_preview_dir = evidence_root / "unified_evidence_dataset_preview_full"
    regular_dir = evidence_root / "unified_evidence_dataset_regular"
    preview_dir.mkdir(parents=True)
    better_preview_dir.mkdir(parents=True)
    regular_dir.mkdir(parents=True)
    packet_path = tmp_path / "artifacts/live_odds_backlog_join_eligibility_packet.json"
    daemon.write_json(
        preview_dir / "unified_evidence_dataset_report.json",
        {
            "output_dir": daemon.relpath(preview_dir),
            "row_count": 27,
            "race_count": 4,
            "unified_evidence_eligible_rows": 27,
            "join_eligibility_packet_paths": [
                "artifacts/live_odds_backlog_join_eligibility_packet.json"
            ],
            "join_eligibility_packet_accepted_races": 4,
            "join_eligibility_packet_accepted_races_present_in_shadow_run": 2,
        },
    )
    daemon.write_json(
        better_preview_dir / "unified_evidence_dataset_report.json",
        {
            "output_dir": daemon.relpath(better_preview_dir),
            "row_count": 31,
            "race_count": 4,
            "unified_evidence_eligible_rows": 31,
            "join_eligibility_packet_paths": [
                str(packet_path)
            ],
            "join_eligibility_packet_accepted_races": 4,
            "join_eligibility_packet_accepted_races_present_in_shadow_run": 4,
        },
    )
    daemon.write_json(
        regular_dir / "unified_evidence_dataset_report.json",
        {
            "output_dir": daemon.relpath(regular_dir),
            "row_count": 8,
            "race_count": 1,
            "unified_evidence_eligible_rows": 8,
        },
    )

    reports = daemon.discovered_join_eligibility_preview_reports(evidence_root)

    assert len(reports) == 1
    assert reports[0]["row_count"] == 31
    assert reports[0]["join_eligibility_packet_accepted_races_present_in_shadow_run"] == 4
    assert reports[0]["report_path"].endswith("unified_evidence_dataset_report.json")


def test_rejoin_unified_evidence_discovers_unconverted_historical_safe_joins(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(daemon, "ROOT", tmp_path)
    monkeypatch.setattr(daemon.autopilot, "ROOT", tmp_path)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_daemonization_v1_test"
    shadow_dir = evidence_root / "daily_race_ingest_shadow_test"
    converted_join_dir = evidence_root / "forward_shadow_result_join_old_daemon_rejoin_001"
    unconverted_join_dir = evidence_root / "forward_shadow_result_join_old_daemon_rejoin_002"
    output_dir.mkdir(parents=True)
    shadow_dir.mkdir(parents=True)
    for join_dir in (converted_join_dir, unconverted_join_dir):
        join_dir.mkdir(parents=True)
        (join_dir / "joined_shadow_predictions.jsonl").write_text("{}\n", encoding="utf-8")
        daemon.write_json(
            join_dir / "shadow_forward_metrics.json",
            {
                "safe_joined_race_count": 1,
                "source_shadow_run": daemon.relpath(shadow_dir),
            },
        )
    (shadow_dir / "shadow_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    existing_dataset_dir = (
        evidence_root / "unified_evidence_dataset_20260610T191153+1000_daemon_rejoin_001"
    )
    daemon.write_json(
        existing_dataset_dir / "unified_evidence_dataset_report.json",
        {
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "joined_shadow_prediction_paths": [
                daemon.relpath(converted_join_dir / "joined_shadow_predictions.jsonl")
            ],
        },
    )
    db_path = tmp_path / "greyhound.sqlite"
    db_path.write_text("", encoding="utf-8")
    calls = []

    def fake_run_command(*, name, command, output_dir, timeout_seconds, cwd=daemon.ROOT):
        calls.append({"name": name, "command": list(command)})
        dataset_dir = Path(command[command.index("--output-dir") + 1])
        daemon.write_json(
            dataset_dir / "unified_evidence_dataset_report.json",
            {
                "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
                "output_dir": daemon.relpath(dataset_dir),
                "shadow_run_dir": daemon.relpath(shadow_dir),
                "row_count": 8,
                "race_count": 1,
                "rows_with_official_results": 8,
                "rows_with_strict_prejump_odds": 4,
                "unified_evidence_eligible_rows": 4,
                "joined_shadow_prediction_paths": [
                    daemon.relpath(unconverted_join_dir / "joined_shadow_predictions.jsonl")
                ],
            },
        )
        return {"name": name, "command": list(command), "returncode": 0, "status": "PASS"}

    monkeypatch.setattr(daemon, "run_command", fake_run_command)

    status, steps, report_paths = daemon.build_rejoin_unified_evidence_datasets(
        run_id="20260611T161131+1000",
        output_dir=output_dir,
        evidence_root=evidence_root,
        db_path=db_path,
        automated_join_report={"results": []},
        generated_at=daemon.datetime.fromisoformat("2026-06-11T16:11:31+10:00"),
        timeout_seconds=120,
    )

    assert status["status"] == "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    assert status["dataset_count"] == 1
    assert status["skipped_dataset_count"] == 0
    assert status["unified_evidence_eligible_rows"] == 4
    assert len(steps) == 1
    assert len(report_paths) == 1
    assert len(calls) == 1
    assert str(unconverted_join_dir / "joined_shadow_predictions.jsonl") in calls[0]["command"]
    assert str(converted_join_dir / "joined_shadow_predictions.jsonl") not in calls[0]["command"]


def test_cycle_activity_reports_result_join_delta_without_new_predictions():
    activity = daemon.build_cycle_activity_summary(
        current_dashboard={"safe_joined_races": 84},
        previous_dashboard={"safe_joined_races": 83},
        daily_status={"races_scored_today": 0},
        observability_status={"status": "NO_PREDICTIONS", "prediction_rows": 0},
    )

    assert activity["status"] == "RESULT_JOINS_ADVANCED_NO_NEW_PREDICTIONS"
    assert activity["safe_joined_delta_this_cycle"] == 1
    assert activity["prediction_rows_this_cycle"] == 0


def test_final_summary_includes_cycle_activity_delta():
    summary = daemon.build_final_summary(
        verdict="DAEMON_READY",
        dashboard={"safe_joined_races": 84, "pending_races": 237, "unsafe_matches": 14},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        automated_join_report={"rejoin_attempt_count": 8, "rejoin_safe_joined_count_sum": 9},
        alert_report={"status": "ALERTS_TRIGGERED", "triggered_alerts": []},
        service_validation={"service_files_present": True, "timer_frequency": "15min"},
        observability_status={"status": "NO_PREDICTIONS", "prediction_rows": 0},
        cycle_activity={
            "status": "RESULT_JOINS_ADVANCED_NO_NEW_PREDICTIONS",
            "safe_joined_delta_this_cycle": 1,
        },
    )

    assert "Cycle activity: `RESULT_JOINS_ADVANCED_NO_NEW_PREDICTIONS`" in summary
    assert "Safe joined delta this cycle: `1`" in summary
    assert "Rejoin safe joined count sum across attempts" in summary


def test_join_index_keeps_newest_metrics_for_same_shadow_run(tmp_path):
    old_dir = tmp_path / "forward_shadow_result_join_20260608T100000_old"
    new_dir = tmp_path / "forward_shadow_result_join_20260608T110000_new"
    old_dir.mkdir()
    new_dir.mkdir()
    source = "artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_same"
    old_metrics = old_dir / "shadow_forward_metrics.json"
    new_metrics = new_dir / "shadow_forward_metrics.json"
    daemon.write_json(
        old_metrics,
        {
            "source_shadow_run": source,
            "pending_race_count": 3,
            "safe_joined_race_count": 1,
        },
    )
    daemon.write_json(
        new_metrics,
        {
            "source_shadow_run": source,
            "pending_race_count": 0,
            "safe_joined_race_count": 4,
        },
    )
    os.utime(old_metrics, (1000, 1000))
    os.utime(new_metrics, (2000, 2000))

    index = daemon.join_index(tmp_path)

    selected = index["daily_race_ingest_shadow_same"]
    assert selected["join_dir"] == new_dir
    assert selected["metrics"]["safe_joined_race_count"] == 4


def test_candidate_shadow_runs_rotates_by_oldest_join_check(tmp_path):
    for index, latest_join_mtime in enumerate([3000, 1000, 2000], start=1):
        shadow_dir = tmp_path / f"daily_race_ingest_shadow_run_{index}"
        shadow_dir.mkdir()
        daemon.write_json(
            shadow_dir / "shadow_manifest.json",
            {
                "final_status": "FORWARD_SHADOW_RUN_COMPLETE",
                "race_count": 3,
            },
        )
        os.utime(shadow_dir, (index * 100, index * 100))

        join_dir = tmp_path / f"forward_shadow_result_join_run_{index}"
        join_dir.mkdir()
        metrics_path = join_dir / "shadow_forward_metrics.json"
        daemon.write_json(
            metrics_path,
            {
                "source_shadow_run": str(shadow_dir),
                "pending_race_count": 1,
                "safe_joined_race_count": 0,
                "unsafe_match_count": 0,
            },
        )
        os.utime(metrics_path, (latest_join_mtime, latest_join_mtime))

    candidates = daemon.candidate_shadow_runs(
        evidence_root=tmp_path,
        pending_limit=2,
        lookback_days=999999,
    )

    assert [row["shadow_run_key"] for row in candidates] == [
        "daily_race_ingest_shadow_run_2",
        "daily_race_ingest_shadow_run_3",
    ]


def test_forward_observer_defaults_off_and_requires_root():
    args = daemon.parse_args(["run-once"])
    assert args.enable_forward_official_result_observer is False
    assert args.forward_corpus_root is None
    assert daemon.run_forward_official_result_observer(args, "run") == {
        "status": "DISABLED",
        "attempted_race_ids": [],
    }
    with pytest.raises(SystemExit):
        daemon.parse_args(["run-once", "--enable-forward-official-result-observer"])


def test_forward_corpus_root_is_required_in_installed_service_identity():
    args = daemon.parse_args(
        [
            "run-once",
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            "/runtime/forward-corpus",
        ]
    )

    fragments = daemon.expected_service_exec_fragments_for_run(args)

    assert fragments[fragments.index("--forward-corpus-root") + 1] == (
        "/runtime/forward-corpus"
    )


def test_forward_observer_opt_in_invokes_owned_cycle(monkeypatch, tmp_path):
    expected = {"status": "COMPLETED", "attempted_race_ids": ["race-1"]}
    monkeypatch.setattr(
        "scripts.observe_forward_official_results.observe_once",
        lambda **kwargs: expected | {"arguments": kwargs},
    )
    args = daemon.parse_args(
        [
            "run-once",
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            str(tmp_path),
            "--timeout-seconds",
            "840",
        ]
    )
    result = daemon.run_forward_official_result_observer(args, "daemon-cycle")
    assert result["status"] == "COMPLETED"
    assert result["arguments"] == {
        "corpus_root": tmp_path,
        "cycle_id": "daemon-cycle",
        "timeout_seconds": 120.0,
    }


def test_full_service_generator_emits_forward_opt_in_only_when_declared(tmp_path):
    lock_path = Path("/runtime/shared-shadow-autopilot.lock")
    default = daemon.service_file_text(repo_path=tmp_path, timeout_seconds=840)
    enabled = daemon.service_file_text(
        repo_path=tmp_path,
        timeout_seconds=840,
        lock_path=lock_path,
        forward_corpus_root=Path("/runtime/forward-corpus"),
    )
    assert "--enable-forward-official-result-observer" not in default
    assert "--forward-corpus-root" not in default
    assert (
        "--enable-forward-official-result-observer "
        '--forward-corpus-root "/runtime/forward-corpus"'
    ) in enabled
    assert "--lock-path /runtime/shared-shadow-autopilot.lock" in enabled
    generated = daemon.write_service_files(
        service_dir=tmp_path / "systemd",
        repo_path=tmp_path,
        timeout_seconds=840,
        lock_path=lock_path,
        forward_corpus_root=Path("/runtime/forward-corpus"),
    )
    assert generated["forward_corpus_root"] == "/runtime/forward-corpus"
    assert generated["lock_path"] == "/runtime/shared-shadow-autopilot.lock"
    assert '--forward-corpus-root "/runtime/forward-corpus"' in (
        tmp_path / "systemd" / daemon.SERVICE_NAME
    ).read_text()
    odds = daemon.odds_capture_service_file_text(
        repo_path=tmp_path,
        timeout_seconds=600,
    )
    assert "--forward-corpus-root" not in odds


@pytest.mark.parametrize("status", ["LOCK_BUSY", "COMPLETED_WITH_ERRORS", "FAILED"])
def test_forward_observer_nonclean_daemon_cli_exit_is_nonzero(monkeypatch, status):
    args = daemon.parse_args(
        [
            "run-once",
            "--enable-forward-official-result-observer",
            "--forward-corpus-root",
            "/corpus",
        ]
    )
    monkeypatch.setattr(daemon, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(
        daemon,
        "run_once",
        lambda _args: {
            "final_verdict": "PARTIAL_DAEMONIZATION",
            "forward_official_result_observer": {
                "status": status,
                "cycle_id": "current-cycle",
            },
        },
    )
    assert daemon.main([]) == 2


def test_forward_observer_failure_replaces_durable_success(tmp_path):
    state_path = tmp_path / "state" / "daemon.json"
    state_path.parent.mkdir()
    state_path.write_text(
        json.dumps(
            {
                "last_run_id": "old-success",
                "forward_official_result_observer": {"status": "COMPLETED"},
            }
        )
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    failed = {"status": "LOCK_BUSY", "cycle_id": "current-cycle"}
    daemon.persist_forward_observer_failure(
        output_dir=output_dir,
        state_path=state_path,
        run_id="current-cycle",
        generated_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
        observer=failed,
    )
    durable = json.loads(state_path.read_text())
    runtime = json.loads((output_dir / "forward_shadow_runtime_state.json").read_text())
    assert durable["last_run_id"] == "current-cycle"
    assert durable["forward_official_result_observer"] == failed
    assert runtime["daemon_run_id"] == "current-cycle"
    assert runtime["forward_official_result_observer"] == failed
    assert "current-cycle" in (output_dir / "FORWARD_SHADOW_RUNTIME_STATE.md").read_text()


@pytest.mark.parametrize(
    ("path", "encoded"),
    [
        ("/corpus/with space", '"/corpus/with space"'),
        ('/corpus/a"b', '"/corpus/a\\"b"'),
        (r"/corpus/a\b", '"/corpus/a\\\\b"'),
        ("/corpus/100%ready", '"/corpus/100%%ready"'),
        ("/corpus/café/犬", '"/corpus/café/犬"'),
    ],
)
def test_full_service_generator_systemd_escapes_corpus_root(tmp_path, path, encoded):
    service = daemon.service_file_text(
        repo_path=tmp_path,
        timeout_seconds=840,
        forward_corpus_root=Path(path),
    )
    assert f"--forward-corpus-root {encoded}" in service


@pytest.mark.parametrize(
    "path",
    [
        "/corpus/new\nline",
        "/corpus/null\0byte",
        "/corpus/tab\tpath",
        "/corpus/next-line\u0085path",
        "/corpus/application-control\u009fpath",
        "/corpus/left-to-right-mark\u200epath",
    ],
)
def test_full_service_generator_rejects_control_corpus_root(tmp_path, path):
    with pytest.raises(ValueError, match="control"):
        daemon.service_file_text(
            repo_path=tmp_path,
            timeout_seconds=840,
            forward_corpus_root=Path(path),
        )
