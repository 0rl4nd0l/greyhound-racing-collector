import json
from datetime import datetime

from scripts import forward_shadow_status_report as status


def _db_report(state="PASS"):
    return {"status": state}


def _metrics(**overrides):
    payload = {
        "safe_joined_race_count": 25,
        "pending_race_count": 0,
        "unsafe_match_count": 0,
        "probability_sum_max_error_joined_races": 0.0,
    }
    payload.update(overrides)
    return payload


def _activation(**overrides):
    payload = {"kept_quarantined_features": []}
    payload.update(overrides)
    return payload


def test_status_collects_more_when_joined_sample_is_too_small():
    final_status, reasons = status.decide_status(
        db_report=_db_report(),
        metrics=_metrics(safe_joined_race_count=4, pending_race_count=12),
        activation=_activation(kept_quarantined_features=["same_distance_same_grade_best_time"]),
        min_joined_races=20,
    )

    assert final_status == "CONTINUE_FORWARD_SHADOW_COLLECTION"
    assert "safe_joined_race_count_below_review_min" in reasons
    assert "pending_official_results_remain" in reasons
    assert "features_remain_quarantined" in reasons


def test_status_blocks_on_db_failure():
    final_status, reasons = status.decide_status(
        db_report=_db_report("FAIL"),
        metrics=_metrics(),
        activation=_activation(),
        min_joined_races=20,
    )

    assert final_status == "BLOCKED_DB_STATE"
    assert reasons == ["db_state_not_pass"]


def test_status_ready_for_report_only_review_when_sample_and_gates_pass():
    final_status, reasons = status.decide_status(
        db_report=_db_report(),
        metrics=_metrics(),
        activation=_activation(),
        min_joined_races=20,
    )

    assert final_status == "READY_FOR_FORWARD_SHADOW_REVIEW_REPORT_ONLY"
    assert reasons == []


def test_status_review_ready_keep_quarantined_when_only_quarantine_blocks():
    final_status, reasons = status.decide_status(
        db_report=_db_report(),
        metrics=_metrics(safe_joined_race_count=120),
        activation=_activation(
            kept_quarantined_features=["same_distance_same_grade_best_time"]
        ),
        min_joined_races=100,
    )

    assert final_status == "FORWARD_REVIEW_READY_KEEP_QUARANTINED"
    assert reasons == ["features_remain_quarantined"]


def test_status_still_collects_more_when_pending_results_remain_with_quarantine():
    final_status, reasons = status.decide_status(
        db_report=_db_report(),
        metrics=_metrics(safe_joined_race_count=120, pending_race_count=4),
        activation=_activation(
            kept_quarantined_features=["same_distance_same_grade_best_time"]
        ),
        min_joined_races=100,
    )

    assert final_status == "CONTINUE_FORWARD_SHADOW_COLLECTION"
    assert "pending_official_results_remain" in reasons
    assert "features_remain_quarantined" in reasons


def test_metric_summary_handles_missing_metrics():
    summary = status.metric_summary(None)

    assert summary["safe_joined_race_count"] == 0
    assert summary["pending_race_count"] == 0
    assert summary["winner_ranks"] == []


def test_artifact_final_status_reads_status_file(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "final_status.txt").write_text("READY\n", encoding="utf-8")

    assert status.artifact_final_status(artifact) == "READY"


def test_status_prefers_aggregate_metrics_over_latest_single_join(tmp_path, monkeypatch):
    aggregate_dir = tmp_path / "forward_shadow_result_aggregate_20260608T123000+1000"
    aggregate_dir.mkdir()
    (aggregate_dir / "aggregate_forward_metrics.json").write_text(
        json.dumps(
            {
                "safe_joined_race_count": 6,
                "pending_race_count": 10,
                "unsafe_match_count": 0,
                "probability_sum_max_error_joined_races": 0.0,
                "winner_ranks": [7, 1, 5, 5, 2, 8],
            }
        ),
        encoding="utf-8",
    )
    single_dir = tmp_path / "forward_shadow_result_join_20260608T124000+1000"
    single_dir.mkdir()
    (single_dir / "shadow_forward_metrics.json").write_text(
        json.dumps(
            {
                "safe_joined_race_count": 1,
                "pending_race_count": 10,
                "unsafe_match_count": 0,
                "probability_sum_max_error_joined_races": 0.0,
                "winner_ranks": [5],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "db_state", lambda _path: {"status": "PASS"})

    report = status.build_status_report(
        evidence_root=tmp_path,
        db_path=tmp_path / "db.sqlite",
        min_joined_races=20,
        generated_at=datetime.fromisoformat("2026-06-08T12:45:00+10:00"),
    )

    assert report["result_metric_source"] == "aggregate_forward_metrics"
    assert report["forward_metrics"]["safe_joined_race_count"] == 6
    assert report["coverage_gap"]["latest_forward_metrics_summary"][
        "safe_joined_race_count"
    ] == 6
    assert report["source_dirs"]["aggregate_result_dir"].endswith(
        "forward_shadow_result_aggregate_20260608T123000+1000"
    )


def test_coverage_summary_uses_selected_forward_metrics_over_stale_audit_metrics():
    summary = status.coverage_summary(
        {
            "latest_forward_metrics_summary": {
                "safe_joined_race_count": 4,
                "pending_race_count": 12,
            },
            "blocked_reasons": ["features_remain_quarantined"],
        },
        selected_metrics={
            "safe_joined_race_count": 7,
            "pending_race_count": 9,
        },
    )

    assert summary["latest_forward_metrics_summary"] == {
        "safe_joined_race_count": 7,
        "pending_race_count": 9,
    }
    assert summary["blocked_reasons"] == ["features_remain_quarantined"]


def test_odds_snapshot_summary_is_report_only_and_ev_disabled():
    summary = status.odds_snapshot_summary(
        {
            "final_status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "prediction_rows": 8,
            "race_count": 1,
            "odds_candidate_rows": 8,
            "valid_pre_jump_dog_odds_rows": 8,
            "races_with_complete_valid_prejump_odds": 1,
            "ev_output_rows": 0,
            "approved_odds_augmented_predictions": {
                "candidate_key": "stage2_market_blend_70",
                "status": "APPROVED_BLEND_READY",
                "ready_race_count": 1,
                "blocked_race_count": 0,
                "prediction_rows": 8,
            },
            "approved_odds_augmented_prediction_report_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x/approved_odds_augmented_prediction_report.json"
            ),
        },
        {
            "status": "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED",
            "blocker_counts": {},
            "odds_research_next_action": "REPORT_ONLY_REVIEW_ODDS_CALIBRATION_NO_EV_ACTION",
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            "odds_used_for_shadow_scoring": False,
        },
    )

    assert summary["odds_analysis_status"] == "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
    assert summary["odds_analysis_blocker_counts"] == {}
    assert summary["ev_output_rows"] == 0
    assert summary["ev_calculation_status"] == "DISABLED_REPORT_ONLY_NO_EV_OUTPUT"
    assert summary["odds_used_for_shadow_scoring"] is False
    assert summary["approved_odds_augmented_candidate_key"] == "stage2_market_blend_70"
    assert summary["approved_odds_augmented_prediction_status"] == "APPROVED_BLEND_READY"
    assert summary["approved_odds_augmented_ready_race_count"] == 1
    assert summary["approved_odds_augmented_blocked_race_count"] == 0
    assert summary["approved_odds_augmented_prediction_rows"] == 8
    assert summary["approved_odds_augmented_prediction_report_path"].endswith(
        "approved_odds_augmented_prediction_report.json"
    )


def test_status_report_includes_latest_shadow_odds_snapshot(tmp_path, monkeypatch):
    odds_dir = tmp_path / "shadow_odds_snapshot_20260609T120000+1000"
    odds_dir.mkdir()
    (odds_dir / "shadow_odds_snapshot_report.json").write_text(
        json.dumps(
            {
                "final_status": "SHADOW_ODDS_SNAPSHOT_NO_MATCHES",
                "prediction_rows": 8,
                "race_count": 1,
                "odds_candidate_rows": 0,
                "valid_pre_jump_dog_odds_rows": 0,
                "races_with_missing_odds_rows": 1,
                "ev_output_rows": 0,
                "approved_odds_augmented_predictions": {
                    "candidate_key": "stage2_market_blend_70",
                    "status": "APPROVED_BLEND_BLOCKED",
                    "ready_race_count": 0,
                    "blocked_race_count": 1,
                    "prediction_rows": 0,
                },
                "approved_odds_augmented_prediction_report_path": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "shadow_odds_snapshot_20260609T120000+1000/"
                    "approved_odds_augmented_prediction_report.json"
                ),
            }
        ),
        encoding="utf-8",
    )
    (odds_dir / "shadow_odds_research_readiness.json").write_text(
        json.dumps(
            {
                "status": "ODDS_ANALYSIS_BLOCKED",
                "blocker_counts": {
                    "missing_odds_rows": 1,
                    "incomplete_valid_prejump_odds": 1,
                },
                "odds_research_next_action": "COLLECT_EXACT_PREJUMP_DOG_ODDS_FOR_ALL_RUNNERS",
                "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
                "odds_used_for_shadow_scoring": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "db_state", lambda _path: {"status": "PASS"})

    report = status.build_status_report(
        evidence_root=tmp_path,
        db_path=tmp_path / "db.sqlite",
        min_joined_races=20,
        generated_at=datetime.fromisoformat("2026-06-09T12:05:00+10:00"),
    )

    assert report["shadow_odds_snapshot"]["final_status"] == "SHADOW_ODDS_SNAPSHOT_NO_MATCHES"
    assert report["shadow_odds_snapshot"]["odds_analysis_status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["shadow_odds_snapshot"]["odds_analysis_blocker_counts"] == {
        "missing_odds_rows": 1,
        "incomplete_valid_prejump_odds": 1,
    }
    assert report["shadow_odds_snapshot"]["ev_output_rows"] == 0
    assert report["shadow_odds_snapshot"]["approved_odds_augmented_candidate_key"] == (
        "stage2_market_blend_70"
    )
    assert report["shadow_odds_snapshot"]["approved_odds_augmented_prediction_status"] == (
        "APPROVED_BLEND_BLOCKED"
    )
    assert report["shadow_odds_snapshot"]["approved_odds_augmented_blocked_race_count"] == 1
    assert report["shadow_odds_snapshot"]["approved_odds_augmented_prediction_rows"] == 0
    assert report["source_dirs"]["odds_snapshot_dir"].endswith(
        "shadow_odds_snapshot_20260609T120000+1000"
    )
    assert report["source_dirs"]["odds_snapshot_source_kind"] == "standalone_snapshot_report"


def test_status_report_prefers_fresher_daemon_odds_status_over_stale_snapshot(
    tmp_path,
    monkeypatch,
):
    odds_dir = tmp_path / "shadow_odds_snapshot_20260609T031900+1000_manual"
    odds_dir.mkdir()
    (odds_dir / "shadow_odds_snapshot_report.json").write_text(
        json.dumps(
            {
                "final_status": "DATA_MISSING_NO_SHADOW_PREDICTIONS",
                "prediction_rows": 0,
                "race_count": 0,
                "ev_output_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    (odds_dir / "shadow_odds_research_readiness.json").write_text(
        json.dumps(
            {
                "status": "ODDS_ANALYSIS_BLOCKED",
                "blocker_counts": {"no_shadow_predictions": 1},
                "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            }
        ),
        encoding="utf-8",
    )
    daemon_dir = tmp_path / "shadow_autopilot_daemonization_v1_20260609T064116+1000"
    daemon_dir.mkdir()
    (daemon_dir / "shadow_odds_snapshot_status.json").write_text(
        json.dumps(
            {
                "schema_version": "shadow_daemon_odds_snapshot_summary_v1",
                "status": "SKIPPED",
                "final_status": "SKIPPED",
                "collection_attempted": False,
                "skipped_reason": "no_shadow_predictions",
                "odds_analysis_blocker_counts": {},
                "prediction_rows": 0,
                "valid_pre_jump_dog_odds_rows": 0,
                "ev_output_rows": 0,
                "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "db_state", lambda _path: {"status": "PASS"})

    report = status.build_status_report(
        evidence_root=tmp_path,
        db_path=tmp_path / "db.sqlite",
        min_joined_races=20,
        generated_at=datetime.fromisoformat("2026-06-09T12:05:00+10:00"),
    )

    assert report["shadow_odds_snapshot"]["final_status"] == "SKIPPED"
    assert report["shadow_odds_snapshot"]["collection_attempted"] is False
    assert report["shadow_odds_snapshot"]["skipped_reason"] == "no_shadow_predictions"
    assert report["shadow_odds_snapshot"]["odds_analysis_status"] == "ODDS_ANALYSIS_BLOCKED"
    assert report["shadow_odds_snapshot"]["odds_analysis_blocker_counts"] == {
        "no_shadow_predictions": 1
    }
    assert report["shadow_odds_snapshot"]["odds_research_next_action"] == (
        "WAIT_FOR_SHADOW_PREDICTIONS"
    )
    assert report["shadow_odds_snapshot"]["ev_output_rows"] == 0
    assert report["source_dirs"]["odds_snapshot_dir"] is None
    assert report["source_dirs"]["odds_status_path"].endswith(
        "shadow_autopilot_daemonization_v1_20260609T064116+1000/shadow_odds_snapshot_status.json"
    )
    assert report["source_dirs"]["odds_snapshot_source_kind"] == "autopilot_or_daemon_status"


def test_status_report_includes_latest_read_only_odds_coverage(tmp_path, monkeypatch):
    daemon_dir = tmp_path / "shadow_autopilot_daemonization_v1_20260609T071121+1000"
    daemon_dir.mkdir()
    (daemon_dir / "odds_coverage_report.json").write_text(
        json.dumps(
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
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "db_state", lambda _path: {"status": "PASS"})

    report = status.build_status_report(
        evidence_root=tmp_path,
        db_path=tmp_path / "db.sqlite",
        min_joined_races=20,
        generated_at=datetime.fromisoformat("2026-06-09T12:05:00+10:00"),
    )

    coverage = report["shadow_odds_coverage"]
    assert coverage["status"] == "SUCCESS"
    assert coverage["mode"] == "read_only_coverage_diagnostic"
    assert coverage["readiness_status"] == "ODDS_COVERAGE_BLOCKED_REPORT_ONLY_EV_DISABLED"
    assert coverage["blocker_counts"] == {
        "missing_source_url_rows": 316,
        "stale_current_win_rows": 316,
    }
    assert coverage["next_action"] == "CAPTURE_FRESH_DOG_LEVEL_ODDS_WITH_SOURCE_URLS"
    assert coverage["dog_level_win_odds_rows"] == 316
    assert coverage["source_url_rows_missing"] == 316
    assert coverage["stale_current_win_rows"] == 316
    assert coverage["safe_direct_identity_match_rate"] == 0.310126582278481
    assert coverage["odds_capture_performed"] is False
    assert coverage["odds_used_for_shadow_scoring"] is False
    assert coverage["db_write"] is False
    assert coverage["ev_action"] is False
    assert coverage["betting_action"] is False
    assert report["source_dirs"]["odds_coverage_report_path"].endswith(
        "shadow_autopilot_daemonization_v1_20260609T071121+1000/odds_coverage_report.json"
    )

    summary = status.build_summary(report)
    assert "Odds coverage readiness: `ODDS_COVERAGE_BLOCKED_REPORT_ONLY_EV_DISABLED`" in summary
    assert "Odds coverage next action: `CAPTURE_FRESH_DOG_LEVEL_ODDS_WITH_SOURCE_URLS`" in summary
    assert "Dog-level odds rows: `316`" in summary
    assert "Odds rows missing source URL: `316`" in summary
    assert "Stale current odds rows: `316`" in summary


def test_status_report_includes_daemon_runtime_wait_window_state(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "shadow_autopilot_daemon_runtime"
    runtime_dir.mkdir()
    (runtime_dir / "state.json").write_text(
        json.dumps(
            {
                "last_verdict": "DAEMON_READY",
                "last_cycle_activity_status": "NO_NEW_PREDICTIONS_OR_SAFE_JOINS",
                "last_next_prejump_refresh_status": "WAITING_FOR_FUTURE_WINDOW",
                "last_next_prejump_race": {
                    "race_id": "Race 1 - AP_K - 2026-06-09",
                    "venue": "AP_K",
                    "race_number": "1",
                    "jump_datetime": "2026-06-09T11:35:00+10:00",
                    "selected": False,
                },
                "last_recommended_rerun_after_local": "2026-06-09T08:55:00+10:00",
                "last_safe_joined_delta": 0,
                "last_safe_joined_races": 84,
                "last_prejump_metadata_status": "NO_ELIGIBLE_PREJUMP_RACES",
                "last_shadow_odds_snapshot_status": "SKIPPED",
                "last_shadow_odds_snapshot_ev_output_rows": 0,
                "last_systemd_deployment_status": "INSTALLED_AND_ACTIVE",
                "updated_at": "2026-06-09T06:45:16+10:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(status, "db_state", lambda _path: {"status": "PASS"})

    report = status.build_status_report(
        evidence_root=tmp_path,
        db_path=tmp_path / "db.sqlite",
        min_joined_races=20,
        generated_at=datetime.fromisoformat("2026-06-09T12:05:00+10:00"),
    )

    daemon = report["daemon_runtime"]
    assert daemon["last_verdict"] == "DAEMON_READY"
    assert daemon["last_next_prejump_refresh_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert daemon["last_next_prejump_race"]["race_id"] == "Race 1 - AP_K - 2026-06-09"
    assert daemon["last_recommended_rerun_after_local"] == "2026-06-09T08:55:00+10:00"
    assert daemon["last_shadow_odds_snapshot_ev_output_rows"] == 0
    assert report["source_dirs"]["daemon_runtime_state_path"].endswith(
        "shadow_autopilot_daemon_runtime/state.json"
    )

    summary = status.build_summary(report)
    assert "Race 1 - AP_K - 2026-06-09" in summary
    assert "2026-06-09T08:55:00+10:00" in summary
