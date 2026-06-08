import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import forward_shadow_runtime_state as runtime


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_runtime_state_reports_waiting_window_and_daily_child(tmp_path):
    daily_dir = tmp_path / "daily_race_ingest_shadow_test"
    daemon_output = tmp_path / "daemon"
    activation_dir = tmp_path / "shadow_feature_activation_gate_test"
    daily_dir.mkdir()
    daemon_output.mkdir()
    activation_dir.mkdir()
    write_json(
        daemon_output / "prediction_provenance_report.json",
        {"daily_shadow_run_dir": str(daily_dir)},
    )
    write_json(
        daemon_output / "shadow_dashboard.json",
        {
            "feature_activation_gate": {
                "status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
                "output_dir": str(activation_dir),
                "status_path": str(daemon_output / "feature_activation_gate_status.json"),
                "provenance_audit": str(daemon_output / "feature_activation_provenance_audit.json"),
                "activation_allowed_features": [],
                "kept_quarantined_features": [
                    "same_distance_same_grade_best_time",
                    "same_distance_same_grade_avg_time",
                ],
            }
        },
    )
    write_json(
        activation_dir / "feature_activation_gate_report.json",
        {
            "final_status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
            "activation_allowed_features": [],
            "kept_quarantined_features": [
                "same_distance_same_grade_best_time",
                "same_distance_same_grade_avg_time",
            ],
            "fail_reason_summary": {
                "category_counts": {
                    "feature_population_parity": 10,
                    "target_metadata_provenance": 6,
                },
                "reason_counts": {
                    "target_metadata_readiness_not_ready:"
                    "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS": 2
                },
            },
        },
    )
    write_json(
        daemon_output / "odds_coverage_report.json",
        {
            "schema_version": "shadow_daemon_read_only_odds_coverage_report_v1",
            "summary": {
                "status": "SUCCESS",
                "mode": "read_only_coverage_diagnostic",
                "dog_level_win_odds_rows": 316,
                "live_odds_rows": 1954,
                "live_odds_races": 49,
                "odds_history_rows": 757,
                "races_with_dog_level_win_odds": 49,
                "safe_direct_identity_matches": 98,
                "safe_direct_identity_match_rate": 0.310126582278481,
                "source_url_rows_checked": 316,
                "source_url_rows_missing": 316,
                "stale_current_win_rows": 316,
                "stale_after_hours": 6.0,
                "source_provenance": {
                    "live_odds": [{"source": "sportsbet", "rows": 316}]
                },
                "odds_capture_performed": False,
                "odds_used_for_shadow_scoring": False,
                "shadow_model_input": False,
                "db_write": False,
                "ev_action": False,
                "betting_action": False,
            },
        },
    )
    (daemon_output / "final_status.txt").write_text("DAEMON_READY\n", encoding="utf-8")
    (daily_dir / "final_status.txt").write_text("WAITING_FOR_UPCOMING_RACES\n", encoding="utf-8")
    write_json(
        daily_dir / "shadow_manifest.json",
        {
            "input_classification": {
                "scanned_csv_count": 281,
                "eligible_count": 0,
                "stale_count": 281,
                "malformed_count": 0,
            }
        },
    )
    write_json(
        daily_dir / "prejump_metadata_report.json",
        {
            "target_metadata_readiness": {
                "status": "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS",
                "target_metadata_capture_status": "WAITING",
                "blocker_counts": {},
            }
        },
    )
    write_json(
        daily_dir / "same_distance_same_grade_history_provenance.json",
        {
            "status": "NOT_POPULATED",
            "live_input_status": "NO_ELIGIBLE_PREJUMP_RACES",
            "target_race_rows_allowed": 0,
            "post_outcome_rows_allowed": 0,
        },
    )
    daemon_state = tmp_path / "state.json"
    write_json(
        daemon_state,
        {
            "last_output_dir": str(daemon_output),
            "last_verdict": "DAEMON_READY",
            "last_cycle_activity_status": "NO_NEW_PREDICTIONS_OR_SAFE_JOINS",
            "last_safe_joined_races": 84,
            "last_next_prejump_refresh_status": "WAITING_FOR_FUTURE_WINDOW",
            "last_recommended_rerun_after_local": "2026-06-09T08:55:00+10:00",
        },
    )

    report = runtime.build_runtime_state(
        evidence_root=tmp_path,
        daemon_state_path=daemon_state,
        timer={"service_status": "inactive", "timer_status": "active"},
        generated_at=datetime(2026, 6, 9, 3, 5, tzinfo=timezone.utc),
    )

    assert report["runtime_action"] == "WAIT_UNTIL_RECOMMENDED_REFRESH"
    assert report["safe_joined_races"] == 84
    assert report["safe_joined_races_remaining"] == 16
    assert report["daily_shadow_run"]["final_status"] == "WAITING_FOR_UPCOMING_RACES"
    assert report["daily_shadow_run"]["eligible_count"] == 0
    assert report["daily_shadow_run"]["stale_count"] == 281
    assert (
        report["daily_shadow_run"]["target_metadata_readiness"]["status"]
        == "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
    )
    assert report["daily_shadow_run"]["target_metadata_readiness"]["capture_status"] == "WAITING"
    assert (
        report["daily_shadow_run"]["same_distance_history_provenance"]["status"]
        == "NOT_POPULATED"
    )
    assert report["feature_activation_gate"]["status"] == (
        "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
    )
    assert report["feature_activation_gate"]["kept_quarantined_features"] == [
        "same_distance_same_grade_best_time",
        "same_distance_same_grade_avg_time",
    ]
    assert report["feature_activation_gate"]["fail_reason_summary"]["category_counts"] == {
        "feature_population_parity": 10,
        "target_metadata_provenance": 6,
    }
    assert (
        report["feature_activation_gate"]["fail_reason_summary"]["reason_counts"][
            "target_metadata_readiness_not_ready:"
            "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS"
        ]
        == 2
    )
    assert report["shadow_odds_coverage"]["readiness_status"] == (
        "ODDS_COVERAGE_BLOCKED_REPORT_ONLY_EV_DISABLED"
    )
    assert report["shadow_odds_coverage"]["blocker_counts"] == {
        "missing_source_url_rows": 316,
        "stale_current_win_rows": 316,
    }
    assert report["shadow_odds_coverage"]["next_action"] == (
        "CAPTURE_FRESH_DOG_LEVEL_ODDS_WITH_SOURCE_URLS"
    )
    assert report["shadow_odds_coverage"]["odds_used_for_shadow_scoring"] is False
    assert report["shadow_odds_coverage"]["db_write"] is False
    assert report["shadow_odds_coverage"]["ev_action"] is False
    assert report["shadow_odds_coverage"]["betting_action"] is False
    assert report["daemon_output"]["odds_coverage_report"].endswith(
        "daemon/odds_coverage_report.json"
    )
    assert report["no_write_guarantees"]["db_write"] is False
    assert report["no_write_guarantees"]["betting_or_ev_output"] is False


def test_build_runtime_state_reports_missing_odds_coverage_diagnostic(tmp_path):
    daemon_output = tmp_path / "daemon"
    daemon_output.mkdir()
    daemon_state = tmp_path / "state.json"
    write_json(
        daemon_state,
        {
            "last_output_dir": str(daemon_output),
            "last_verdict": "DAEMON_READY",
            "last_next_prejump_refresh_status": "READY_FOR_REFRESH",
            "last_safe_joined_races": 84,
        },
    )

    report = runtime.build_runtime_state(
        evidence_root=tmp_path,
        daemon_state_path=daemon_state,
        timer={"service_status": "inactive", "timer_status": "active"},
        generated_at=datetime(2026, 6, 9, 3, 5, tzinfo=timezone.utc),
    )

    assert report["shadow_odds_coverage"]["readiness_status"] == (
        "ODDS_COVERAGE_BLOCKED_REPORT_ONLY_EV_DISABLED"
    )
    assert report["shadow_odds_coverage"]["blocker_counts"] == {
        "odds_coverage_report_missing": 1,
        "no_dog_level_win_odds_rows": 1,
    }
    assert report["shadow_odds_coverage"]["next_action"] == (
        "WAIT_FOR_DAEMON_ODDS_COVERAGE_DIAGNOSTIC"
    )
    assert report["daemon_output"]["odds_coverage_report"] is None


def test_feature_activation_gate_summary_fails_closed_when_report_missing(tmp_path):
    daemon_output = tmp_path / "daemon"
    activation_dir = tmp_path / "missing_gate_report"
    daemon_output.mkdir()
    activation_dir.mkdir()
    write_json(
        daemon_output / "shadow_dashboard.json",
        {
            "feature_activation_gate": {
                "status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
                "output_dir": str(activation_dir),
                "activation_allowed_features": [],
                "kept_quarantined_features": ["same_distance_same_grade_best_time"],
            }
        },
    )

    summary = runtime.feature_activation_gate_summary(daemon_output)

    assert summary["status"] == "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED"
    assert summary["activation_allowed_features"] == []
    assert summary["kept_quarantined_features"] == ["same_distance_same_grade_best_time"]
    assert summary["fail_reason_summary"] == {}


def test_daily_child_summary_falls_back_to_standalone_classification(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    (daily_dir / "final_status.txt").write_text("WAITING_FOR_UPCOMING_RACES\n", encoding="utf-8")
    write_json(daily_dir / "shadow_manifest.json", {"final_status": "WAITING_FOR_UPCOMING_RACES"})
    write_json(
        daily_dir / "malformed_or_stale_inputs.json",
        {
            "scanned_csv_count": 10,
            "eligible_count": 2,
            "stale_count": 8,
            "malformed_count": 0,
        },
    )

    summary = runtime.daily_child_summary(daily_dir)

    assert summary["scanned_csv_count"] == 10
    assert summary["eligible_count"] == 2
    assert summary["stale_count"] == 8


def test_runtime_action_reports_active_daemon_before_waiting_state():
    action, reasons = runtime.decide_runtime_action(
        daemon_state={
            "last_verdict": "DAEMON_READY",
            "last_next_prejump_refresh_status": "WAITING_FOR_FUTURE_WINDOW",
            "last_safe_joined_races": 84,
        },
        timer={"service_status": "activating"},
        target_joined_races=100,
    )

    assert action == "DAEMON_RUNNING_WAIT_FOR_CYCLE"
    assert reasons == ["shadow_autopilot_service_active"]


def test_runtime_action_continues_collection_when_not_waiting_and_under_target():
    action, reasons = runtime.decide_runtime_action(
        daemon_state={
            "last_verdict": "DAEMON_READY",
            "last_next_prejump_refresh_status": "READY_FOR_REFRESH",
            "last_safe_joined_races": 84,
        },
        timer={"service_status": "inactive"},
        target_joined_races=100,
    )

    assert action == "CONTINUE_FORWARD_SHADOW_COLLECTION"
    assert reasons == ["safe_joined_race_count_below_target"]


def test_runtime_action_allows_report_only_review_after_target():
    action, reasons = runtime.decide_runtime_action(
        daemon_state={
            "last_verdict": "DAEMON_READY",
            "last_next_prejump_refresh_status": "READY_FOR_REFRESH",
            "last_safe_joined_races": 100,
        },
        timer={"service_status": "inactive"},
        target_joined_races=100,
    )

    assert action == "READY_FOR_FORWARD_SHADOW_REVIEW_REPORT_ONLY"
    assert reasons == []
