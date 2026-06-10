import json
from datetime import datetime
from pathlib import Path

from scripts import shadow_autopilot_v1 as autopilot


def test_autopilot_default_min_joined_races_matches_review_target():
    args = autopilot.parse_args([])

    assert autopilot.DEFAULT_TARGET_JOINED_RACES == 100
    assert autopilot.DEFAULT_MIN_JOINED_RACES_FOR_STATUS == 100
    assert args.target_joined_races == 100
    assert args.min_joined_races == 100


def test_output_guard_rejects_non_autopilot_paths():
    try:
        autopilot.assert_output_dir_safe(Path("model_registry/shadow_autopilot_v1_bad"))
    except ValueError as exc:
        assert "output_dir_must_be_shadow_autopilot_artifact" in str(exc)
    else:
        raise AssertionError("expected protected output path to be rejected")


def test_latest_challenger_activation_metric_paths_requires_pair(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    partial_dir = evidence_root / "forward_shadow_challenger_calibration_20260609T050000+1000"
    complete_dir = evidence_root / "forward_shadow_challenger_calibration_20260609T060000+1000"
    partial_dir.mkdir(parents=True)
    complete_dir.mkdir(parents=True)
    (partial_dir / "candidate_eval_metrics_for_activation.json").write_text("{}", encoding="utf-8")
    baseline = complete_dir / "baseline_eval_metrics_for_activation.json"
    candidate = complete_dir / "candidate_eval_metrics_for_activation.json"
    baseline.write_text("{}", encoding="utf-8")
    candidate.write_text("{}", encoding="utf-8")

    paths = autopilot.latest_challenger_activation_metric_paths(evidence_root)

    assert paths == {
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
    }


def test_latest_challenger_activation_metric_paths_fails_closed_on_partial_pair(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    partial_dir = evidence_root / "forward_shadow_challenger_calibration_20260609T050000+1000"
    partial_dir.mkdir(parents=True)
    (partial_dir / "candidate_eval_metrics_for_activation.json").write_text("{}", encoding="utf-8")

    paths = autopilot.latest_challenger_activation_metric_paths(evidence_root)

    assert paths == {
        "baseline_metrics": None,
        "candidate_metrics": None,
    }


def test_latest_challenger_activation_metric_paths_rejects_stale_source_sample(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    challenger_dir = evidence_root / "forward_shadow_challenger_calibration_20260609T050000+1000"
    aggregate_dir = evidence_root / "forward_shadow_result_aggregate_20260609T060000+1000"
    challenger_dir.mkdir(parents=True)
    aggregate_dir.mkdir(parents=True)
    baseline = challenger_dir / "baseline_eval_metrics_for_activation.json"
    candidate = challenger_dir / "candidate_eval_metrics_for_activation.json"
    baseline.write_text(json.dumps({"source_safe_exact_joined_race_count": 84}), encoding="utf-8")
    candidate.write_text(json.dumps({"source_safe_exact_joined_race_count": 84}), encoding="utf-8")
    aggregate = aggregate_dir / "aggregate_forward_metrics.json"
    aggregate.write_text(json.dumps({"safe_joined_race_count": 85}), encoding="utf-8")

    paths = autopilot.latest_challenger_activation_metric_paths(
        evidence_root,
        aggregate_metrics_path=aggregate,
    )

    assert paths == {
        "baseline_metrics": None,
        "candidate_metrics": None,
    }


def test_latest_challenger_activation_metric_paths_accepts_current_source_sample(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    challenger_dir = evidence_root / "forward_shadow_challenger_calibration_20260609T050000+1000"
    aggregate_dir = evidence_root / "forward_shadow_result_aggregate_20260609T060000+1000"
    challenger_dir.mkdir(parents=True)
    aggregate_dir.mkdir(parents=True)
    baseline = challenger_dir / "baseline_eval_metrics_for_activation.json"
    candidate = challenger_dir / "candidate_eval_metrics_for_activation.json"
    baseline.write_text(json.dumps({"source_safe_exact_joined_race_count": 85}), encoding="utf-8")
    candidate.write_text(json.dumps({"source_safe_exact_joined_race_count": 85}), encoding="utf-8")
    aggregate = aggregate_dir / "aggregate_forward_metrics.json"
    aggregate.write_text(json.dumps({"safe_joined_race_count": 85}), encoding="utf-8")

    paths = autopilot.latest_challenger_activation_metric_paths(
        evidence_root,
        aggregate_metrics_path=aggregate,
    )

    assert paths == {
        "baseline_metrics": baseline,
        "candidate_metrics": candidate,
    }


def test_dashboard_surfaces_required_shadow_metrics():
    dashboard = autopilot.build_dashboard(
        generated_at=datetime.fromisoformat("2026-06-08T19:30:00+10:00"),
        aggregate_metrics={
            "safe_joined_race_count": 37,
            "pending_race_count": 293,
            "unsafe_match_count": 7,
            "top1": 0.19,
            "top3": 0.51,
            "mean_winner_rank": 3.5,
            "winner_ranks": [1, 4],
            "brier": 0.116,
            "logloss": 1.97,
            "probability_sum_max_error_joined_races": 1e-10,
        },
        join_metrics=None,
        aggregate_calibration={
            "slope_intercept": {
                "status": "computed",
                "slope": 0.7,
                "intercept": -0.2,
                "sample_size": 260,
            }
        },
        aggregate_box_bias={"safe_joined_box_1_top_pick_share": 0.18},
        status_report={"final_status": "CONTINUE_FORWARD_SHADOW_COLLECTION"},
        sources={"aggregate_dir": "artifacts/example"},
        odds_snapshot_status={
            "status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x",
            "prediction_rows": 8,
            "odds_candidate_rows": 8,
            "valid_pre_jump_dog_odds_rows": 8,
            "races_with_complete_valid_prejump_odds": 1,
            "races_with_missing_odds_rows": 0,
            "ev_output_rows": 0,
            "ev_calculation_status": "DISABLED_REPORT_ONLY_NO_EV_OUTPUT",
        },
    )

    assert dashboard["safe_joined_races"] == 37
    assert dashboard["pending_races"] == 293
    assert dashboard["unsafe_matches"] == 7
    assert dashboard["top1"] == 0.19
    assert dashboard["top3"] == 0.51
    assert dashboard["winner_rank"]["mean"] == 3.5
    assert dashboard["brier"] == 0.116
    assert dashboard["logloss"] == 1.97
    assert dashboard["calibration"]["slope"] == 0.7
    assert dashboard["box_1_share"] == 0.18
    assert dashboard["probability_sum_status"]["status"] == "PASS"
    assert dashboard["odds_snapshot"]["status"] == "SHADOW_ODDS_SNAPSHOT_COLLECTED"
    assert dashboard["odds_snapshot"]["valid_pre_jump_dog_odds_rows"] == 8
    assert dashboard["odds_snapshot"]["races_with_complete_valid_prejump_odds"] == 1
    assert dashboard["odds_snapshot"]["races_with_missing_odds_rows"] == 0
    assert dashboard["odds_snapshot"]["ev_output_rows"] == 0


def test_shadow_odds_snapshot_guard_requires_prediction_rows(tmp_path):
    assert autopilot.should_collect_shadow_odds_snapshot(None) == (
        False,
        "daily_shadow_run_missing",
        0,
    )

    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    assert autopilot.should_collect_shadow_odds_snapshot(daily_dir) == (
        False,
        "shadow_predictions_missing",
        0,
    )

    (daily_dir / "shadow_predictions.jsonl").write_text("", encoding="utf-8")
    assert autopilot.should_collect_shadow_odds_snapshot(daily_dir) == (
        False,
        "no_shadow_predictions",
        0,
    )

    (daily_dir / "shadow_predictions.jsonl").write_text(
        '{"race_id":"Race 1 - AP_K - 2026-06-09","dog_name":"Example","box":1}\n',
        encoding="utf-8",
    )
    assert autopilot.should_collect_shadow_odds_snapshot(daily_dir) == (
        True,
        "shadow_predictions_present",
        1,
    )


def test_shadow_odds_snapshot_command_is_report_only_artifact():
    command = autopilot.shadow_odds_snapshot_command(
        daily_dir=Path("artifacts/full_evidence_orchestration_20260525/daily_race_ingest_shadow_x"),
        odds_dir=Path("artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x"),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-09T00:52:00+10:00",
    )

    assert "scripts/collect_shadow_odds_snapshots.py" in command[1]
    assert "--shadow-run-dir" in command
    assert "--output-dir" in command
    assert "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x" in command
    assert "--db" in command
    assert "greyhound_racing_data.db" in command
    assert "--current-time" in command


def test_shadow_odds_snapshot_attempted_missing_report_is_not_skip():
    status = autopilot.build_shadow_odds_snapshot_status(
        generated_at=datetime.fromisoformat("2026-06-09T00:52:00+10:00"),
        odds_dir=Path("artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x"),
        odds_report=None,
        skipped_reason="odds_snapshot_report_missing_returncode_2",
        prediction_rows=8,
        attempted=True,
        status_override="SHADOW_ODDS_SNAPSHOT_FAILED_NO_REPORT",
    )

    assert status["status"] == "SHADOW_ODDS_SNAPSHOT_FAILED_NO_REPORT"
    assert status["collection_attempted"] is True
    assert status["prediction_rows"] == 8
    assert status["races_with_complete_valid_prejump_odds"] == 0
    assert status["races_with_missing_odds_rows"] == 0
    assert status["ev_output_rows"] == 0


def test_shadow_odds_snapshot_status_surfaces_race_level_coverage():
    status = autopilot.build_shadow_odds_snapshot_status(
        generated_at=datetime.fromisoformat("2026-06-09T00:52:00+10:00"),
        odds_dir=Path("artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x"),
        odds_report={
            "final_status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "prediction_rows": 8,
            "race_count": 1,
            "runner_rows": 8,
            "odds_candidate_rows": 8,
            "valid_pre_jump_dog_odds_rows": 8,
            "race_coverage_path": "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x/shadow_odds_race_coverage.json",
            "races_with_any_odds_candidates": 1,
            "races_with_complete_odds_candidate_coverage": 1,
            "races_with_complete_valid_prejump_odds": 1,
            "races_with_missing_odds_rows": 0,
            "races_with_duplicate_odds_rows": 0,
            "races_with_post_prediction_odds_rows": 0,
            "races_with_post_feature_freeze_odds_rows": 1,
            "odds_research_readiness": {
                "status": "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED",
                "blocker_counts": {},
            },
            "ev_output_rows": 0,
        },
    )

    assert status["races_with_any_odds_candidates"] == 1
    assert status["races_with_complete_odds_candidate_coverage"] == 1
    assert status["races_with_complete_valid_prejump_odds"] == 1
    assert status["races_with_missing_odds_rows"] == 0
    assert status["races_with_post_feature_freeze_odds_rows"] == 1
    assert status["race_coverage_path"].endswith("shadow_odds_race_coverage.json")
    assert status["odds_analysis_status"] == "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
    assert status["odds_analysis_blocker_counts"] == {}


def test_live_odds_capture_approval_packet_plans_fixed_windows_for_verified_races(tmp_path):
    daily_dir = tmp_path / "daily"
    upcoming_dir = tmp_path / "upcoming"
    daily_dir.mkdir()
    upcoming_dir.mkdir()
    autopilot.write_json(
        daily_dir / "prejump_metadata_report.json",
        {
            "schema_version": "daily_shadow_prejump_metadata_report_v1",
            "status": "PASS",
            "eligible_count": 1,
            "eligible_with_verified_prejump_metadata": 1,
            "malformed_prejump_metadata_count": 0,
            "required_fields": [
                "race_date",
                "venue",
                "race_number",
                "runner_box_name_list",
                "source_url",
                "canonical_runner_source_url",
                "csv_sidecar_runner_identity",
                "canonical_final_runner_alignment",
            ],
            "target_metadata_readiness": {
                "target_metadata_capture_status": "READY",
                "all_current_future_inputs_verified": True,
            },
            "files": [
                {
                    "bucket": "eligible",
                    "path": "upcoming/Race 1 - WPK - 2026-06-10.csv",
                    "race_date": "2026-06-10",
                    "venue": "WPK",
                    "race_number": 1,
                    "jump_datetime": "2026-06-10T15:00:00+10:00",
                    "source_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/test",
                    "canonical_runner_source_url": "https://www.thedogs.com.au/racing/wentworth-park/2026-06-10/1/test",
                    "runner_count": 8,
                    "sidecar_status": "PASS",
                    "metadata_is_leakage_safe": True,
                    "csv_sidecar_runner_identity_verified": True,
                    "canonical_runner_alignment_verified": True,
                }
            ],
        },
    )

    packet = autopilot.build_live_odds_capture_approval_packet(
        generated_at=datetime.fromisoformat("2026-06-10T14:00:00+10:00"),
        daily_shadow_run_dir=daily_dir,
        upcoming_dir=upcoming_dir,
        db_path=Path("greyhound_racing_data.db"),
        output_path=Path("artifacts/full_evidence_orchestration_20260525/live_odds_capture_report.json"),
        limit=16,
    )

    assert packet["status"] == "AWAITING_EXPLICIT_APPROVAL_READY_FOR_LIVE_ODDS"
    assert packet["approval_required"] is True
    assert packet["can_capture_live_odds_now"] is False
    assert packet["capture_window_offsets_minutes"] == [60, 30, 10, 2]
    assert packet["verified_prejump_race_count"] == 1
    assert packet["races"][0]["canonical_race_identity"] == "Race 1 - WPK - 2026-06-10"
    assert packet["races"][0]["runner_set_validation"]["status"] == "PASS"
    assert [row["offset_minutes"] for row in packet["races"][0]["capture_windows"]] == [
        60,
        30,
        10,
        2,
    ]
    assert packet["races"][0]["capture_windows"][0]["target_capture_at"].startswith(
        "2026-06-10T14:00:00"
    )
    assert "--capture-live-odds" in packet["planned_live_odds_capture_command"]
    assert "--approve-live-odds-capture" not in packet["planned_live_odds_capture_command"]
    assert "--approve-live-odds-capture" in packet["approved_live_odds_capture_command_template"]
    assert "--persist" not in packet["approved_live_odds_capture_command_template"]
    assert packet["write_scope"] == "append_only_live_odds_rows"
    assert packet["no_write_guarantees"]["db_write"] is False
    assert "canonical_race_identity" in packet["required_provenance_fields"]
    assert "sportsbet_source_url" in packet["required_provenance_fields"]
    assert "runner_name_box_match_status" in packet["required_provenance_fields"]


def test_live_odds_capture_approval_packet_fails_closed_without_verified_races(tmp_path):
    daily_dir = tmp_path / "daily"
    upcoming_dir = tmp_path / "upcoming"
    daily_dir.mkdir()
    upcoming_dir.mkdir()
    autopilot.write_json(
        daily_dir / "prejump_metadata_report.json",
        {
            "schema_version": "daily_shadow_prejump_metadata_report_v1",
            "status": "PASS",
            "eligible_count": 1,
            "eligible_with_verified_prejump_metadata": 0,
            "malformed_prejump_metadata_count": 0,
            "target_metadata_readiness": {
                "target_metadata_capture_status": "BLOCKED",
                "all_current_future_inputs_verified": False,
            },
            "files": [
                {
                    "bucket": "eligible",
                    "path": "upcoming/Race 1 - WPK - 2026-06-10.csv",
                    "race_date": "2026-06-10",
                    "venue": "WPK",
                    "race_number": 1,
                    "jump_datetime": "2026-06-10T15:00:00+10:00",
                    "runner_count": 8,
                    "sidecar_status": "PASS",
                    "metadata_is_leakage_safe": True,
                    "csv_sidecar_runner_identity_verified": True,
                    "canonical_runner_alignment_verified": False,
                }
            ],
        },
    )

    packet = autopilot.build_live_odds_capture_approval_packet(
        generated_at=datetime.fromisoformat("2026-06-10T14:00:00+10:00"),
        daily_shadow_run_dir=daily_dir,
        upcoming_dir=upcoming_dir,
        db_path=Path("greyhound_racing_data.db"),
        output_path=Path("artifacts/full_evidence_orchestration_20260525/live_odds_capture_report.json"),
        limit=16,
    )

    assert packet["status"] == "NOT_READY"
    assert packet["can_capture_live_odds_now"] is False
    assert "verified_prejump_race_count_zero" in packet["hard_stops"]
    assert "target_metadata_capture_not_ready" in packet["hard_stops"]
    assert packet["no_write_guarantees"]["db_write"] is False


def _waiting_refresh_report():
    return {
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
    }


def test_next_prejump_refresh_window_from_report_surfaces_operator_timing():
    status = autopilot.next_prejump_refresh_window_from_report(_waiting_refresh_report())

    assert status is not None
    assert status["status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert status["recommended_rerun_after_local"] == "2026-06-09T08:55:00+10:00"
    assert status["next_race"]["race_id"] == "Race 1 - AP_K - 2026-06-09"
    assert status["next_race"]["jump_datetime"] == "2026-06-09T11:35:00+10:00"
    assert status["no_write_guarantees"]["db_write"] is False


def test_dashboard_and_status_markdown_surface_next_prejump_window():
    dashboard = autopilot.build_dashboard(
        generated_at=datetime.fromisoformat("2026-06-08T23:55:00+10:00"),
        aggregate_metrics={"safe_joined_race_count": 84},
        join_metrics=None,
        aggregate_calibration={"slope_intercept": {"status": "computed"}},
        aggregate_box_bias={},
        status_report={"final_status": "CONTINUE_FORWARD_SHADOW_COLLECTION"},
        sources={"refresh_report": "artifacts/example/refresh_prejump_report.json"},
        refresh_report=_waiting_refresh_report(),
    )
    readiness = {"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []}
    daily_status = autopilot.build_daily_status(
        generated_at=datetime.fromisoformat("2026-06-08T23:55:00+10:00"),
        daily_manifest={"race_count": 0, "prediction_rows": 0},
        result_join_status={"latest_join": {"joined_count": 0}},
        dashboard=dashboard,
        timeseries=[],
        readiness=readiness,
    )

    assert dashboard["next_prejump_refresh_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert dashboard["recommended_rerun_after_local"] == "2026-06-09T08:55:00+10:00"
    assert daily_status["next_prejump_refresh_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert daily_status["next_prejump_race"]["race_id"] == "Race 1 - AP_K - 2026-06-09"
    assert "2026-06-09T08:55:00+10:00" in autopilot.daily_status_markdown(daily_status)
    assert "Race 1 - AP_K - 2026-06-09" in autopilot.shadow_status_markdown(dashboard, readiness)
    assert "Race 1 - AP_K - 2026-06-09" in autopilot.summary_markdown(
        final_verdict="AUTOPILOT_READY",
        dashboard=dashboard,
        readiness=readiness,
        result_join_status={"latest_join": {"joined_count": 0}, "cumulative": {"joined_count": 84}},
    )


def test_promotion_readiness_stays_need_more_results_with_quarantined_features():
    dashboard = {
        "safe_joined_races": 37,
        "pending_races": 293,
        "unsafe_matches": 7,
        "calibration": {"status": "computed"},
        "probability_sum_status": {"status": "PASS"},
        "quarantined_features": list(autopilot.WATCHED_QUARANTINED_FEATURES),
        "box_1_share": 0.18,
    }

    readiness = autopilot.build_promotion_readiness(
        generated_at=datetime.fromisoformat("2026-06-08T19:30:00+10:00"),
        dashboard=dashboard,
        target_joined_races=100,
    )

    assert readiness["decision"] == "NEED_MORE_RESULTS"
    assert "insufficient_forward_shadow_joined_races" in readiness["outstanding_blockers"]
    assert "same_distance_same_grade_features_remain_quarantined" in readiness["outstanding_blockers"]
    assert readiness["promotion_allowed"] is False


def test_feature_activation_provenance_audit_uses_only_verified_prejump_metadata():
    target_metadata_readiness = {
        "schema_version": "daily_shadow_target_metadata_readiness_v1",
        "status": "TARGET_METADATA_READY_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS",
        "target_metadata_capture_status": "READY",
        "current_or_future_input_count": 2,
        "eligible_count": 2,
        "verified_eligible_count": 2,
        "malformed_prejump_metadata_count": 0,
        "blocker_counts": {},
        "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
    }
    report = autopilot.build_feature_activation_provenance_audit(
        generated_at=datetime.fromisoformat("2026-06-08T19:30:00+10:00"),
        protected_paths_unchanged=True,
        prejump_metadata_report={
            "status": "PASS",
            "target_metadata_readiness": target_metadata_readiness,
            "field_coverage": {
                "target_distance": {"eligible_present_rows": 2},
                "target_grade": {"eligible_present_rows": 2},
            },
            "files": [
                {
                    "bucket": "eligible",
                    "target_distance_source": "canonical_pre_race_page",
                    "target_grade_source": "canonical_pre_race_page",
                    "rejected_metadata_sources": [],
                    "fail_reasons": [],
                },
                {
                    "bucket": "eligible",
                    "target_distance_source": "canonical_pre_race_page",
                    "target_grade_source": "canonical_pre_race_page",
                    "rejected_metadata_sources": [],
                    "fail_reasons": [],
                },
                {
                    "bucket": "malformed",
                    "target_distance_source": "result_page",
                    "target_grade_source": "embedded_form_history:G",
                    "rejected_metadata_sources": ["result_page"],
                    "fail_reasons": ["target_distance_missing_or_unsafe"],
                },
            ],
        },
    )

    assert report["protected_paths_unchanged"] is True
    assert report["eligible_count"] == 2
    assert report["rejected_source_rows"] == 1
    assert report["target_distance_sources"] == {"canonical_pre_race_page": 2}
    assert report["target_grade_sources"] == {"canonical_pre_race_page": 2}
    assert report["target_metadata_readiness"] == target_metadata_readiness
    assert report["by_feature"]["target_distance_safe"]["present_rows"] == 2
    assert report["by_feature"]["target_grade_safe"]["present_rows"] == 2
    assert report["same_distance_same_grade_history_provenance"]["status"] == "NOT_VERIFIED"
    assert (
        report["same_distance_same_grade_history_provenance"]["required_history_cutoff"]
        == "strictly_before_target_race"
    )
    assert report["no_write_guarantees"]["db_write"] is False


def test_feature_activation_provenance_audit_uses_live_same_distance_history_report():
    same_distance_report = {
        "schema_version": "same_distance_same_grade_history_provenance_v1",
        "status": "PASS",
        "required_source": "prior_dog_history",
        "required_history_cutoff": "strictly_before_target_race",
        "target_race_rows_allowed": 0,
        "post_outcome_rows_allowed": 0,
        "by_feature": {
            "same_distance_same_grade_best_time": {
                "status": "PASS",
                "source": "prior_dog_history",
                "history_cutoff": "strictly_before_target_race",
                "prior_history_rows_used": 2,
                "target_race_rows_used": 0,
                "post_outcome_rows_used": 0,
                "post_outcome_fields_used": [],
            },
            "same_distance_same_grade_avg_time": {
                "status": "PASS",
                "source": "prior_dog_history",
                "history_cutoff": "strictly_before_target_race",
                "prior_history_rows_used": 2,
                "target_race_rows_used": 0,
                "post_outcome_rows_used": 0,
                "post_outcome_fields_used": [],
            },
        },
    }

    report = autopilot.build_feature_activation_provenance_audit(
        generated_at=datetime.fromisoformat("2026-06-08T19:30:00+10:00"),
        protected_paths_unchanged=True,
        prejump_metadata_report={
            "status": "PASS",
            "field_coverage": {
                "target_distance": {"eligible_present_rows": 1},
                "target_grade": {"eligible_present_rows": 1},
            },
            "files": [
                {
                    "bucket": "eligible",
                    "target_distance_source": "canonical_pre_race_page",
                    "target_grade_source": "canonical_pre_race_page",
                    "rejected_metadata_sources": [],
                    "fail_reasons": [],
                }
            ],
        },
        same_distance_history_provenance=same_distance_report,
    )

    assert report["same_distance_same_grade_history_provenance"] == same_distance_report


def test_feature_activation_provenance_audit_preserves_waiting_target_metadata_readiness():
    target_metadata_readiness = {
        "schema_version": "daily_shadow_target_metadata_readiness_v1",
        "status": "TARGET_METADATA_WAITING_FOR_CURRENT_OR_FUTURE_PREJUMP_INPUTS",
        "target_metadata_capture_status": "WAITING",
        "current_or_future_input_count": 0,
        "eligible_count": 0,
        "verified_eligible_count": 0,
        "malformed_prejump_metadata_count": 0,
        "blocker_counts": {},
        "historical_repair_policy": "NO_REPAIR_WITHOUT_PROVENANCE_SAFE_PRE_RACE_SOURCE",
    }

    report = autopilot.build_feature_activation_provenance_audit(
        generated_at=datetime.fromisoformat("2026-06-09T08:18:00+10:00"),
        protected_paths_unchanged=True,
        prejump_metadata_report={
            "status": "PASS",
            "target_metadata_readiness": target_metadata_readiness,
            "field_coverage": {
                "target_distance": {"eligible_present_rows": 0},
                "target_grade": {"eligible_present_rows": 0},
            },
            "files": [],
        },
    )

    assert report["target_metadata_readiness"] == target_metadata_readiness
    assert report["eligible_count"] == 0
    assert report["by_feature"]["target_distance_safe"]["present_rows"] == 0
    assert report["by_feature"]["target_grade_safe"]["present_rows"] == 0


def test_feature_activation_gate_inputs_prefers_live_policy_and_model_parity(tmp_path):
    daily_dir = tmp_path / "daily"
    model_dir = tmp_path / "model"
    score_live_dir = daily_dir / "shadow_score_live"
    score_live_dir.mkdir(parents=True)
    model_dir.mkdir()
    model_path = model_dir / "shadow_randomforest_model.joblib"
    model_path.write_bytes(b"model")
    model_parity = model_dir / "train_eval_feature_parity_report.json"
    model_matrix = model_dir / "shadow_feature_matrix_audit.json"
    live_policy = score_live_dir / "active_feature_policy_report.json"
    live_same_distance = score_live_dir / "same_distance_same_grade_history_provenance.json"
    model_policy = model_dir / "inactive_feature_policy_report.json"
    baseline_metrics = tmp_path / "baseline_metrics.json"
    candidate_metrics = tmp_path / "candidate_metrics.json"
    for path in (
        model_parity,
        model_matrix,
        live_policy,
        live_same_distance,
        model_policy,
        baseline_metrics,
        candidate_metrics,
    ):
        path.write_text("{}", encoding="utf-8")

    inputs = autopilot.feature_activation_gate_inputs(
        daily_dir=daily_dir,
        shadow_model=model_path,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )

    assert inputs["parity_report"] == model_parity
    assert inputs["inactive_policy_report"] == live_policy
    assert inputs["matrix_audit"] == model_matrix
    assert inputs["same_distance_history_provenance"] == live_same_distance
    assert inputs["baseline_metrics"] == baseline_metrics
    assert inputs["candidate_metrics"] == candidate_metrics


def test_feature_activation_gate_inputs_falls_back_to_daily_same_distance_provenance(tmp_path):
    daily_dir = tmp_path / "daily"
    model_dir = tmp_path / "model"
    daily_dir.mkdir()
    model_dir.mkdir()
    shadow_model = model_dir / "shadow_randomforest_model.joblib"
    shadow_model.write_text("placeholder", encoding="utf-8")
    (model_dir / "train_eval_feature_parity_report.json").write_text("{}", encoding="utf-8")
    daily_same_distance = daily_dir / "same_distance_same_grade_history_provenance.json"
    daily_same_distance.write_text("{}", encoding="utf-8")

    inputs = autopilot.feature_activation_gate_inputs(
        daily_dir=daily_dir,
        shadow_model=shadow_model,
    )

    assert inputs["same_distance_history_provenance"] == daily_same_distance


def test_summary_surfaces_feature_activation_gate_status():
    summary = autopilot.summary_markdown(
        final_verdict="AUTOPILOT_READY",
        dashboard={"safe_joined_races": 1},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        result_join_status={"latest_join": {"joined_count": 0}, "cumulative": {"joined_count": 1}},
        activation_gate_status={
            "status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
            "output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_feature_activation_gate_x",
            "activation_allowed_features": [],
            "kept_quarantined_features": ["same_distance_same_grade_best_time"],
        },
    )

    assert "## Feature Activation Gate" in summary
    assert "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED" in summary
    assert "same_distance_same_grade_best_time" in summary


def test_summary_surfaces_shadow_odds_snapshot_status():
    summary = autopilot.summary_markdown(
        final_verdict="AUTOPILOT_READY",
        dashboard={"safe_joined_races": 1},
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        result_join_status={"latest_join": {"joined_count": 0}, "cumulative": {"joined_count": 1}},
        odds_snapshot_status={
            "status": "SHADOW_ODDS_SNAPSHOT_NO_MATCHES",
            "output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x",
            "odds_candidate_rows": 0,
            "valid_pre_jump_dog_odds_rows": 0,
            "races_with_complete_valid_prejump_odds": 0,
            "races_with_missing_odds_rows": 1,
            "ev_output_rows": 0,
        },
    )

    assert "## Odds Snapshot" in summary
    assert "SHADOW_ODDS_SNAPSHOT_NO_MATCHES" in summary
    assert "Races with complete valid pre-jump odds: `0`" in summary
    assert "Races with missing odds rows: `1`" in summary
    assert "EV output rows: `0`" in summary


def test_daily_status_compares_latest_aggregate_against_previous():
    status = autopilot.build_daily_status(
        generated_at=datetime.fromisoformat("2026-06-08T19:30:00+10:00"),
        daily_manifest={"race_count": 2, "prediction_rows": 14},
        result_join_status={"latest_join": {"joined_count": 3}},
        dashboard={"safe_joined_races": 40},
        timeseries=[
            {"safe_joined_race_count": 37, "top1": 0.2, "brier": 0.12},
            {"safe_joined_race_count": 40, "top1": 0.25, "brier": 0.11},
        ],
        readiness={"decision": "NEED_MORE_RESULTS"},
        odds_snapshot_status={
            "status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "odds_candidate_rows": 14,
            "valid_pre_jump_dog_odds_rows": 14,
            "races_with_complete_valid_prejump_odds": 2,
            "races_with_missing_odds_rows": 0,
            "ev_output_rows": 0,
        },
    )

    assert status["races_scored_today"] == 2
    assert status["prediction_rows_today"] == 14
    assert status["results_joined_this_run"] == 3
    assert status["metrics_improved_or_worsened"]["safe_joined_race_count"]["direction"] == "IMPROVED"
    assert status["metrics_improved_or_worsened"]["brier"]["direction"] == "IMPROVED"
    assert status["odds_snapshot_status"] == "SHADOW_ODDS_SNAPSHOT_COLLECTED"
    assert status["valid_pre_jump_dog_odds_rows"] == 14
    assert status["races_with_complete_valid_prejump_odds"] == 2
    assert status["races_with_missing_odds_rows"] == 0
    assert status["ev_output_rows"] == 0


def test_aggregate_timeseries_orders_post_rejoin_daemon_after_pre_rejoin(tmp_path):
    def write_aggregate(name, safe_joined):
        aggregate_dir = tmp_path / name
        aggregate_dir.mkdir()
        autopilot.write_json(
            aggregate_dir / "aggregate_forward_metrics.json",
            {
                "safe_joined_race_count": safe_joined,
                "pending_race_count": 1,
                "unsafe_match_count": 0,
            },
        )
        autopilot.write_json(
            aggregate_dir / "forward_shadow_result_aggregate_report.json",
            {"generated_at": "2026-06-08T22:09:08+10:00"},
        )
        return aggregate_dir

    write_aggregate("forward_shadow_result_aggregate_20260608T220908+1000_daemon_autopilot", 77)
    write_aggregate("forward_shadow_result_aggregate_20260608T220908+1000_daemon", 82)

    timeseries = autopilot.build_aggregate_timeseries(tmp_path)

    assert timeseries[-1]["safe_joined_race_count"] == 82
    status = autopilot.build_daily_status(
        generated_at=datetime.fromisoformat("2026-06-08T22:12:00+10:00"),
        daily_manifest={"race_count": 1, "prediction_rows": 8},
        result_join_status={"latest_join": {"joined_count": 11}},
        dashboard={"safe_joined_races": 82},
        timeseries=timeseries,
        readiness={"decision": "NEED_MORE_RESULTS"},
    )
    assert status["metrics_improved_or_worsened"]["safe_joined_race_count"]["current"] == 82
    assert status["metrics_improved_or_worsened"]["safe_joined_race_count"]["direction"] == "IMPROVED"
    assert status["closer_to_promotion_review"] == "YES_RESULTS_ACCUMULATED"


def test_final_verdict_is_partial_when_result_join_fails():
    verdict = autopilot.final_verdict_for(
        steps=[
            {"name": "daily_shadow_run", "returncode": 0},
            {"name": "result_join", "returncode": 2},
            {"name": "aggregate_results", "returncode": 0},
            {"name": "status_report", "returncode": 0},
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "PARTIAL_AUTOMATION_READY"


def test_final_verdict_keeps_odds_snapshot_observability_non_gating():
    verdict = autopilot.final_verdict_for(
        steps=[
            {"name": "daily_shadow_run", "returncode": 0},
            {"name": "shadow_odds_snapshot", "returncode": 2},
            {"name": "result_join", "returncode": 0},
            {"name": "aggregate_results", "returncode": 0},
            {"name": "status_report", "returncode": 0},
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "AUTOPILOT_READY"


def test_refresh_command_auto_uses_uv_when_parser_deps_missing(monkeypatch):
    monkeypatch.setattr(autopilot, "refresh_dependencies_available", lambda: False)
    monkeypatch.setattr(autopilot.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    command = autopilot.refresh_command_prefix("auto")

    assert command[:2] == ["/usr/bin/uv", "run"]
    assert "requests" in command
    assert "beautifulsoup4" in command
    assert "pandas" in command
    assert command[-1] == "python"


def test_refresh_command_auto_fails_closed_without_parser_deps_or_uv(monkeypatch):
    monkeypatch.setattr(autopilot, "refresh_dependencies_available", lambda: False)
    monkeypatch.setattr(autopilot.shutil, "which", lambda name: None)

    try:
        autopilot.refresh_command_prefix("auto")
    except RuntimeError as exc:
        assert "refresh_dependencies_missing_and_uv_unavailable" in str(exc)
    else:
        raise AssertionError("expected missing refresh dependencies to fail closed")
