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
            "last_autonomous_official_result_capture_status": (
                "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COMPLETE"
            ),
            "last_autonomous_official_result_capture_attempted": True,
            "last_autonomous_official_result_race_rows": 7,
            "last_autonomous_official_result_runner_rows": 56,
            "last_autonomous_official_result_quarantine_rows": 2,
            "last_autonomous_official_result_quarantined_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "last_autonomous_official_result_quarantine_reason_counts": {
                "ingest_failed_or_unsafe_match": 2,
            },
            "last_autonomous_official_result_quarantine_error_counts": {
                "result_boxes_not_in_participants:9": 1,
                "result_boxes_not_in_participants:10": 1,
            },
            "last_autonomous_official_result_quarantine_result_boxes_not_in_participants_counts": {
                "9": 1,
                "10": 1,
            },
            "last_autonomous_official_result_quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 7 - TAREE - 2026-06-13",
                    "participant_source": "shadow_run_predictions",
                    "participant_boxes": [1, 2, 3, 4, 5, 6, 7, 8],
                    "result_boxes": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "result_boxes_not_in_participants": [9],
                }
            ],
            "last_autonomous_official_result_skipped_reason_counts": {
                "race_not_jumped": 1,
            },
            "last_autonomous_official_result_awaiting_jump_race_count": 1,
            "last_autonomous_official_result_awaiting_jump_race_ids": [
                "Race 12 - TAREE - 2026-06-13",
            ],
            "last_autonomous_official_result_awaiting_jump_next_recheck_after_local": (
                "2026-06-13T23:55:00+10:00"
            ),
            "last_autonomous_official_result_evidence_db_ingest_status": "NOT_EXECUTED",
            "last_autonomous_official_result_evidence_db_execute": False,
            "last_autonomous_official_result_evidence_db_write_performed": False,
            "last_autonomous_official_result_evidence_valid_race_rows": 21,
            "last_autonomous_official_result_evidence_valid_runner_rows": 150,
            "last_autonomous_official_result_evidence_blocked_race_rows": 4,
            "last_autonomous_official_result_evidence_blocked_runner_rows": 36,
            "last_autonomous_official_result_evidence_inserted_race_rows": 0,
            "last_autonomous_official_result_evidence_inserted_runner_rows": 0,
            "last_autonomous_official_result_evidence_blocker_reason_counts": {
                "runner_set_mismatch_quarantined": 4,
            },
            "last_autonomous_official_result_evidence_inserted_rows": 63,
            "last_live_odds_backlog_unresolved_race_count": 11,
            "last_live_odds_backlog_unresolved_reason_counts": {
                "shadow_run_candidate_rejected": 5,
                "no_matching_shadow_run_candidate_found": 3,
            },
            "last_live_odds_backlog_unresolved_recovery_action_counts": {
                "validate_runner_set_then_alias_join": 5,
                "inspect_missing_shadow_candidate": 3,
            },
            "last_live_odds_backlog_unresolved_alias_status_counts": {
                "EXACT_SHADOW_ARTIFACT_MATCH_FOUND": 5,
                "NO_EXACT_SHADOW_ARTIFACT_MATCH": 6,
            },
            "last_live_odds_backlog_retryable_exact_shadow_match_race_count": 5,
            "last_live_odds_backlog_no_exact_shadow_match_race_count": 6,
            "last_live_odds_backlog_retryable_exact_shadow_match_race_ids": [
                "Race 1 - GRDN - 2026-06-12"
            ],
            "last_live_odds_backlog_no_exact_shadow_match_race_ids": [
                "ASCOT PARK_2026-06-10_6"
            ],
            "last_live_odds_backlog_recovery_queue_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_recovery_queue.json"
            ),
            "last_live_odds_backlog_recovery_queue_diagnostic_only": True,
            "last_live_odds_backlog_recovery_queue_join_acceptance_changed": False,
            "last_live_odds_backlog_recovery_queue_db_write_performed": False,
            "last_live_odds_backlog_awaiting_official_result_evidence_race_count": 5,
            "last_live_odds_backlog_awaiting_official_result_evidence_race_ids": [
                "Race 1 - GRDN - 2026-06-12"
            ],
            "last_live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
                "diagnostic_recheck_official_result_evidence_only"
            ),
            "last_live_odds_backlog_awaiting_official_result_recheck_ready_race_count": 5,
            "last_live_odds_backlog_runner_set_validation_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_runner_set_validation.json"
            ),
            "last_live_odds_backlog_runner_set_validation_retryable_race_count": 5,
            "last_live_odds_backlog_runner_set_validation_exact_match_race_count": 5,
            "last_live_odds_backlog_runner_set_validation_blocked_race_count": 0,
            "last_live_odds_backlog_runner_set_validation_diagnostic_only": True,
            "last_live_odds_backlog_runner_set_validation_join_authorized": False,
            "last_live_odds_backlog_runner_set_validation_db_write_performed": False,
            "last_live_odds_backlog_join_eligibility_packet_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_join_eligibility_packet.json"
            ),
            "last_live_odds_backlog_join_eligibility_evaluated_race_count": 5,
            "last_live_odds_backlog_join_eligibility_eligible_report_only_race_count": 2,
            "last_live_odds_backlog_join_eligibility_blocked_race_count": 3,
            "last_live_odds_backlog_join_eligibility_diagnostic_only": True,
            "last_live_odds_backlog_join_eligibility_join_authorized": False,
            "last_live_odds_backlog_join_eligibility_db_write_performed": False,
            "last_rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "last_rolling_model_comparison_sample_races": 131,
            "last_rolling_model_comparison_best_candidate": "market_only_implied",
            "last_rolling_model_comparison_source_rejected_live_odds_candidate_count": 5,
            "last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates": 4,
            "last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "last_high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "last_high_accuracy_promotion_pr_gate_status": "BLOCKED",
            "last_high_accuracy_unified_evidence_eligible_rows": 856,
            "last_promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "last_promotion_distance_promotion_ready": False,
            "last_promotion_distance_sample_race_count": 131,
            "last_promotion_distance_sample_runner_rows": 927,
            "last_promotion_distance_best_candidate_key": "market_only_implied",
            "last_promotion_distance_best_non_market_candidate_key": (
                "stage2_rf_calibrated"
            ),
            "last_promotion_distance_blockers": [
                "best_non_market_top1_margin_below_target"
            ],
            "last_promotion_distance_source_rejected_live_odds_candidate_count": 5,
            "last_promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
            "last_promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "last_promotion_distance_source_exclusion_reason_counts": {
                "official_result_missing": 32,
            },
            "last_promotion_distance_source_odds_exclusion_reason_counts": {
                "strict_prejump_odds_missing": 6,
            },
            "last_promotion_distance_source_official_result_evidence_db_missing_race_ids": [
                "Race 7 - TAREE - 2026-06-13",
            ],
            "last_promotion_distance_source_official_result_evidence_db_requested_race_count": 7,
            "last_promotion_distance_source_official_result_evidence_db_races_with_rows": [
                "Race 5 - TAREE - 2026-06-13",
            ],
            "last_promotion_distance_source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/official_result_runners.jsonl",
            ],
            "last_promotion_distance_official_result_coverage_requested_race_count": 7,
            "last_promotion_distance_official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "last_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": 4125,
            "last_promotion_distance_official_result_coverage_races_with_rows_count": 1,
            "last_promotion_distance_official_result_coverage_missing_race_count": 1,
            "last_promotion_distance_official_result_coverage_missing_exclusion_count": 32,
            "last_promotion_distance_official_result_runner_path_count": 1,
            "last_promotion_distance_official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
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
            "last_rejoin_rolling_model_comparison_status": (
                "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
            ),
            "last_rejoin_rolling_model_comparison_sample_races": 132,
            "last_rejoin_high_accuracy_refinement_status": "BLOCKED_KEEP_BASELINE",
            "last_rejoin_pre_race_gated_challenger_status": (
                "PRE_RACE_GATED_CHALLENGER_REVIEW_READY"
            ),
            "last_rejoin_pre_race_gated_challenger_promotion_ready": False,
            "last_rejoin_rank_first_hypothesis_review_status": (
                "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
            ),
            "last_rejoin_rank_first_hypothesis_candidate_count": 6,
            "last_rejoin_rank_first_hypothesis_evaluated_candidate_count": 4,
            "last_rejoin_rank_first_hypothesis_best_candidate_key": (
                "market_residual_blend"
            ),
            "last_rejoin_rank_first_hypothesis_directional_read_ready": False,
            "last_rejoin_time_split_gated_challenger_status": (
                "TIME_SPLIT_GATED_CHALLENGER_BLOCKED"
            ),
            "last_rejoin_time_split_gated_challenger_promotion_ready": False,
            "last_rejoin_market_residual_challenger_status": (
                "MARKET_RESIDUAL_CHALLENGER_REVIEW_READY"
            ),
            "last_rejoin_market_residual_challenger_promotion_ready": False,
            "last_rejoin_market_residual_regime_audit_status": (
                "MARKET_RESIDUAL_REGIME_AUDIT_READY"
            ),
            "last_rejoin_market_residual_regime_audit_promotion_ready": False,
            "last_rejoin_market_residual_rank_first_hypothesis_status": (
                "RANK_FIRST_HYPOTHESIS_UNDERPOWERED"
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_status": (
                "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY"
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_candidate_count": 3,
            "last_rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count": 1,
            "last_rejoin_rank_first_hypothesis_watchlist_best_candidate": (
                "rank_first_blend_001"
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_best_status": (
                "DIRECTIONAL_READ_UNDERPOWERED"
            ),
            "last_rejoin_rank_first_hypothesis_watchlist_best_distinct_samples": 12,
            "last_rejoin_promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
            "last_rejoin_promotion_distance_promotion_ready": False,
            "last_rejoin_promotion_distance_blockers": [
                "rank_first_directional_sample_underpowered"
            ],
            "last_rejoin_promotion_distance_source_exclusion_reason_counts": {
                "official_result_missing": 4,
            },
            "last_rejoin_promotion_distance_source_odds_exclusion_reason_counts": {
                "strict_prejump_odds_missing": 1,
            },
            "last_rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids": [
                "Race 8 - TAREE - 2026-06-13",
            ],
            "last_rejoin_promotion_distance_source_official_result_evidence_db_requested_race_count": 3,
            "last_rejoin_promotion_distance_source_official_result_evidence_db_races_with_rows": [
                "Race 10 - TAREE - 2026-06-13",
            ],
            "last_rejoin_promotion_distance_source_official_result_runner_paths": [
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_rejoin_x/official_result_runners.jsonl",
            ],
            "last_rejoin_promotion_distance_official_result_coverage_requested_race_count": 3,
            "last_rejoin_promotion_distance_official_result_coverage_requested_race_count_source": (
                "deduped_requested_or_inferred_race_ids"
            ),
            "last_rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids": 19,
            "last_rejoin_promotion_distance_official_result_coverage_races_with_rows_count": 1,
            "last_rejoin_promotion_distance_official_result_coverage_missing_race_count": 1,
            "last_rejoin_promotion_distance_official_result_coverage_missing_exclusion_count": 4,
            "last_rejoin_promotion_distance_official_result_runner_path_count": 1,
            "last_rejoin_promotion_distance_official_result_runner_paths_source_field": (
                "rolling_sample.source_official_result_runner_paths"
            ),
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
    assert report["daemon"]["last_autonomous_live_odds_next_window_opens_at"] == (
        "2026-06-09T10:35:00+10:00"
    )
    assert report["daemon"]["last_autonomous_live_odds_recommended_rerun_after_local"] == (
        "2026-06-09T10:30:00+10:00"
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
    assert report["daemon"]["last_autonomous_live_odds_next_prejump_window"][
        "status"
    ] == "WAITING_FOR_FUTURE_WINDOW"
    assert report["daemon"]["last_autonomous_official_result_capture_status"] == (
        "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COMPLETE"
    )
    assert report["daemon"]["last_autonomous_official_result_capture_attempted"] is True
    assert report["daemon"]["last_autonomous_official_result_race_rows"] == 7
    assert report["daemon"]["last_autonomous_official_result_runner_rows"] == 56
    assert report["daemon"]["last_autonomous_official_result_quarantine_rows"] == 2
    assert report["daemon"]["last_autonomous_official_result_quarantined_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
    ]
    assert report["daemon"][
        "last_autonomous_official_result_quarantine_reason_counts"
    ] == {
        "ingest_failed_or_unsafe_match": 2,
    }
    assert report["daemon"][
        "last_autonomous_official_result_quarantine_error_counts"
    ] == {
        "result_boxes_not_in_participants:9": 1,
        "result_boxes_not_in_participants:10": 1,
    }
    assert report["daemon"][
        "last_autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
    ] == {
        "9": 1,
        "10": 1,
    }
    assert report["daemon"][
        "last_autonomous_official_result_quarantine_runner_set_mismatch_samples"
    ][0]["result_boxes_not_in_participants"] == [9]
    assert report["daemon"]["last_autonomous_official_result_skipped_reason_counts"] == {
        "race_not_jumped": 1,
    }
    assert report["daemon"]["last_autonomous_official_result_awaiting_jump_race_count"] == 1
    assert report["daemon"]["last_autonomous_official_result_awaiting_jump_race_ids"] == [
        "Race 12 - TAREE - 2026-06-13",
    ]
    assert (
        report["daemon"][
            "last_autonomous_official_result_awaiting_jump_next_recheck_after_local"
        ]
        == "2026-06-13T23:55:00+10:00"
    )
    assert report["daemon"][
        "last_autonomous_official_result_evidence_db_ingest_status"
    ] == "NOT_EXECUTED"
    assert report["daemon"]["last_autonomous_official_result_evidence_db_execute"] is False
    assert (
        report["daemon"][
            "last_autonomous_official_result_evidence_db_write_performed"
        ]
        is False
    )
    assert report["daemon"]["last_autonomous_official_result_evidence_valid_race_rows"] == 21
    assert report["daemon"]["last_autonomous_official_result_evidence_valid_runner_rows"] == 150
    assert report["daemon"]["last_autonomous_official_result_evidence_blocked_race_rows"] == 4
    assert report["daemon"]["last_autonomous_official_result_evidence_blocked_runner_rows"] == 36
    assert report["daemon"]["last_autonomous_official_result_evidence_inserted_race_rows"] == 0
    assert report["daemon"]["last_autonomous_official_result_evidence_inserted_runner_rows"] == 0
    assert report["daemon"][
        "last_autonomous_official_result_evidence_blocker_reason_counts"
    ] == {
        "runner_set_mismatch_quarantined": 4,
    }
    assert report["daemon"]["last_autonomous_official_result_evidence_inserted_rows"] == 63
    assert report["daemon"]["last_live_odds_backlog_unresolved_race_count"] == 11
    assert report["daemon"]["last_live_odds_backlog_unresolved_reason_counts"] == {
        "shadow_run_candidate_rejected": 5,
        "no_matching_shadow_run_candidate_found": 3,
    }
    assert report["daemon"][
        "last_live_odds_backlog_unresolved_recovery_action_counts"
    ] == {
        "validate_runner_set_then_alias_join": 5,
        "inspect_missing_shadow_candidate": 3,
    }
    assert report["daemon"]["last_live_odds_backlog_unresolved_alias_status_counts"] == {
        "EXACT_SHADOW_ARTIFACT_MATCH_FOUND": 5,
        "NO_EXACT_SHADOW_ARTIFACT_MATCH": 6,
    }
    assert (
        report["daemon"]["last_live_odds_backlog_retryable_exact_shadow_match_race_count"]
        == 5
    )
    assert report["daemon"]["last_live_odds_backlog_no_exact_shadow_match_race_count"] == 6
    assert report["daemon"]["last_live_odds_backlog_retryable_exact_shadow_match_race_ids"] == [
        "Race 1 - GRDN - 2026-06-12"
    ]
    assert report["daemon"]["last_live_odds_backlog_no_exact_shadow_match_race_ids"] == [
        "ASCOT PARK_2026-06-10_6"
    ]
    assert report["daemon"]["last_live_odds_backlog_recovery_queue_path"].endswith(
        "live_odds_backlog_recovery_queue.json"
    )
    assert report["daemon"]["last_live_odds_backlog_recovery_queue_diagnostic_only"] is True
    assert (
        report["daemon"]["last_live_odds_backlog_recovery_queue_join_acceptance_changed"]
        is False
    )
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
    assert (
        report["daemon"][
            "last_live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        ]
        == 5
    )
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
    assert report["daemon"]["last_live_odds_backlog_join_eligibility_evaluated_race_count"] == 5
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
    assert report["daemon"]["last_rolling_model_comparison_status"] == (
        "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    )
    assert report["daemon"]["last_rolling_model_comparison_sample_races"] == 131
    assert report["daemon"]["last_rolling_model_comparison_best_candidate"] == (
        "market_only_implied"
    )
    assert (
        report["daemon"][
            "last_rolling_model_comparison_source_rejected_live_odds_candidate_count"
        ]
        == 5
    )
    assert (
        report["daemon"][
            "last_rolling_model_comparison_source_rows_with_rejected_live_odds_candidates"
        ]
        == 4
    )
    assert report["daemon"][
        "last_rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert report["daemon"]["last_high_accuracy_refinement_status"] == (
        "BLOCKED_KEEP_BASELINE"
    )
    assert report["daemon"]["last_high_accuracy_promotion_pr_gate_status"] == "BLOCKED"
    assert report["daemon"]["last_high_accuracy_unified_evidence_eligible_rows"] == 856
    assert report["daemon"]["last_promotion_distance_status"] == (
        "PROMOTION_DISTANCE_BLOCKED"
    )
    assert report["daemon"]["last_promotion_distance_promotion_ready"] is False
    assert report["daemon"]["last_promotion_distance_sample_race_count"] == 131
    assert report["daemon"]["last_promotion_distance_sample_runner_rows"] == 927
    assert report["daemon"]["last_promotion_distance_best_candidate_key"] == (
        "market_only_implied"
    )
    assert report["daemon"]["last_promotion_distance_best_non_market_candidate_key"] == (
        "stage2_rf_calibrated"
    )
    assert report["daemon"]["last_promotion_distance_blockers"] == [
        "best_non_market_top1_margin_below_target"
    ]
    assert (
        report["daemon"][
            "last_promotion_distance_source_rejected_live_odds_candidate_count"
        ]
        == 5
    )
    assert (
        report["daemon"][
            "last_promotion_distance_source_rows_with_rejected_live_odds_candidates"
        ]
        == 4
    )
    assert report["daemon"][
        "last_promotion_distance_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert report["daemon"]["last_promotion_distance_source_exclusion_reason_counts"] == {
        "official_result_missing": 32,
    }
    assert report["daemon"][
        "last_promotion_distance_source_odds_exclusion_reason_counts"
    ] == {
        "strict_prejump_odds_missing": 6,
    }
    assert report["daemon"][
        "last_promotion_distance_source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 7 - TAREE - 2026-06-13"]
    assert (
        report["daemon"][
            "last_promotion_distance_source_official_result_evidence_db_requested_race_count"
        ]
        == 7
    )
    assert report["daemon"][
        "last_promotion_distance_source_official_result_evidence_db_races_with_rows"
    ] == ["Race 5 - TAREE - 2026-06-13"]
    assert report["daemon"][
        "last_promotion_distance_source_official_result_runner_paths"
    ] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_x/official_result_runners.jsonl",
    ]
    assert (
        report["daemon"][
            "last_promotion_distance_official_result_coverage_requested_race_count"
        ]
        == 7
    )
    assert (
        report["daemon"][
            "last_promotion_distance_official_result_coverage_requested_race_count_source"
        ]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        report["daemon"][
            "last_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
        ]
        == 4125
    )
    assert (
        report["daemon"][
            "last_promotion_distance_official_result_coverage_races_with_rows_count"
        ]
        == 1
    )
    assert (
        report["daemon"][
            "last_promotion_distance_official_result_coverage_missing_race_count"
        ]
        == 1
    )
    assert (
        report["daemon"][
            "last_promotion_distance_official_result_coverage_missing_exclusion_count"
        ]
        == 32
    )
    assert report["daemon"]["last_promotion_distance_official_result_runner_path_count"] == 1
    assert report["daemon"][
        "last_promotion_distance_official_result_runner_paths_source_field"
    ] == "rolling_sample.source_official_result_runner_paths"
    assert report["daemon"]["last_best_aggregate_unified_evidence_status"] == (
        "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert report["daemon"]["last_best_aggregate_unified_evidence_eligible_rows"] == 3872
    assert (
        report["daemon"][
            "last_best_aggregate_unified_rejected_live_odds_candidate_count"
        ]
        == 9
    )
    assert (
        report["daemon"][
            "last_best_aggregate_unified_rows_with_rejected_live_odds_candidates"
        ]
        == 6
    )
    assert report["daemon"][
        "last_best_aggregate_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 4,
        "odds_source_url_missing": 5,
    }
    assert report["daemon"]["last_rejoin_unified_evidence_status"] == (
        "REJOIN_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert (
        report["daemon"]["last_rejoin_unified_rejected_live_odds_candidate_count"]
        == 8
    )
    assert (
        report["daemon"]["last_rejoin_unified_rows_with_rejected_live_odds_candidates"]
        == 6
    )
    assert report["daemon"][
        "last_rejoin_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 5,
    }
    assert report["daemon"]["last_rejoin_rolling_model_comparison_status"] == (
        "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW"
    )
    assert report["daemon"]["last_rejoin_rolling_model_comparison_sample_races"] == 132
    assert report["daemon"]["last_rejoin_high_accuracy_refinement_status"] == (
        "BLOCKED_KEEP_BASELINE"
    )
    assert report["daemon"]["last_rejoin_pre_race_gated_challenger_status"] == (
        "PRE_RACE_GATED_CHALLENGER_REVIEW_READY"
    )
    assert report["daemon"]["last_rejoin_pre_race_gated_challenger_promotion_ready"] is False
    assert report["daemon"]["last_rejoin_rank_first_hypothesis_review_status"] == (
        "RANK_FIRST_HYPOTHESIS_REVIEW_READY"
    )
    assert report["daemon"]["last_rejoin_rank_first_hypothesis_candidate_count"] == 6
    assert (
        report["daemon"]["last_rejoin_rank_first_hypothesis_evaluated_candidate_count"]
        == 4
    )
    assert report["daemon"]["last_rejoin_rank_first_hypothesis_best_candidate_key"] == (
        "market_residual_blend"
    )
    assert (
        report["daemon"]["last_rejoin_rank_first_hypothesis_directional_read_ready"]
        is False
    )
    assert report["daemon"]["last_rejoin_time_split_gated_challenger_status"] == (
        "TIME_SPLIT_GATED_CHALLENGER_BLOCKED"
    )
    assert report["daemon"]["last_rejoin_time_split_gated_challenger_promotion_ready"] is False
    assert report["daemon"]["last_rejoin_market_residual_challenger_status"] == (
        "MARKET_RESIDUAL_CHALLENGER_REVIEW_READY"
    )
    assert report["daemon"]["last_rejoin_market_residual_challenger_promotion_ready"] is False
    assert report["daemon"]["last_rejoin_market_residual_regime_audit_status"] == (
        "MARKET_RESIDUAL_REGIME_AUDIT_READY"
    )
    assert (
        report["daemon"]["last_rejoin_market_residual_regime_audit_promotion_ready"]
        is False
    )
    assert report["daemon"]["last_rejoin_market_residual_rank_first_hypothesis_status"] == (
        "RANK_FIRST_HYPOTHESIS_UNDERPOWERED"
    )
    assert report["daemon"]["last_rejoin_rank_first_hypothesis_watchlist_status"] == (
        "RANK_FIRST_HYPOTHESIS_WATCHLIST_READY"
    )
    assert report["daemon"]["last_rejoin_rank_first_hypothesis_watchlist_candidate_count"] == 3
    assert (
        report["daemon"][
            "last_rejoin_rank_first_hypothesis_watchlist_directional_ready_candidate_count"
        ]
        == 1
    )
    assert report["daemon"][
        "last_rejoin_rank_first_hypothesis_watchlist_best_candidate"
    ] == "rank_first_blend_001"
    assert report["daemon"]["last_rejoin_rank_first_hypothesis_watchlist_best_status"] == (
        "DIRECTIONAL_READ_UNDERPOWERED"
    )
    assert (
        report["daemon"][
            "last_rejoin_rank_first_hypothesis_watchlist_best_distinct_samples"
        ]
        == 12
    )
    assert report["daemon"]["last_rejoin_promotion_distance_status"] == (
        "PROMOTION_DISTANCE_BLOCKED"
    )
    assert report["daemon"]["last_rejoin_promotion_distance_promotion_ready"] is False
    assert report["daemon"]["last_rejoin_promotion_distance_blockers"] == [
        "rank_first_directional_sample_underpowered"
    ]
    assert report["daemon"][
        "last_rejoin_promotion_distance_source_exclusion_reason_counts"
    ] == {
        "official_result_missing": 4,
    }
    assert report["daemon"][
        "last_rejoin_promotion_distance_source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 8 - TAREE - 2026-06-13"]
    assert report["daemon"][
        "last_rejoin_promotion_distance_source_official_result_runner_paths"
    ] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_rejoin_x/official_result_runners.jsonl",
    ]
    assert (
        report["daemon"][
            "last_rejoin_promotion_distance_official_result_coverage_requested_race_count"
        ]
        == 3
    )
    assert (
        report["daemon"][
            "last_rejoin_promotion_distance_official_result_coverage_requested_race_count_source"
        ]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        report["daemon"][
            "last_rejoin_promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
        ]
        == 19
    )
    assert (
        report["daemon"][
            "last_rejoin_promotion_distance_official_result_coverage_races_with_rows_count"
        ]
        == 1
    )
    assert (
        report["daemon"][
            "last_rejoin_promotion_distance_official_result_coverage_missing_race_count"
        ]
        == 1
    )
    assert (
        report["daemon"][
            "last_rejoin_promotion_distance_official_result_coverage_missing_exclusion_count"
        ]
        == 4
    )
    assert (
        report["daemon"]["last_rejoin_promotion_distance_official_result_runner_path_count"]
        == 1
    )
    assert report["daemon"][
        "last_rejoin_promotion_distance_official_result_runner_paths_source_field"
    ] == "rolling_sample.source_official_result_runner_paths"
    summary = runtime.build_summary(report)
    assert "Autonomous odds next window opens: `2026-06-09T10:35:00+10:00`" in summary
    assert (
        "Autonomous odds next action: `WAIT_UNTIL_NEXT_FIXED_WINDOW` at "
        "`2026-06-09T10:35:00+10:00`"
    ) in summary
    assert "Autonomous odds next race: `Race 1 - AP_K - 2026-06-09`" in summary
    assert (
        "Autonomous official result capture: "
        "`AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COMPLETE`"
    ) in summary
    assert "Autonomous official result race rows: `7`" in summary
    assert "Autonomous official result runner rows: `56`" in summary
    assert "Autonomous official result quarantine rows: `2`" in summary
    assert (
        "Autonomous official result quarantined race IDs: "
        "`['Race 7 - TAREE - 2026-06-13']`"
    ) in summary
    assert (
        "Autonomous official result quarantine result boxes not in participants: "
        "`{'9': 1, '10': 1}`"
    ) in summary
    assert "Autonomous official result skipped reasons: `{'race_not_jumped': 1}`" in summary
    assert "Autonomous official result awaiting-jump races: `1`" in summary
    assert (
        "Autonomous official result next recheck: "
        "`2026-06-13T23:55:00+10:00`"
    ) in summary
    assert "Autonomous official result evidence DB ingest: `NOT_EXECUTED`" in summary
    assert (
        "Autonomous official result evidence DB write performed: `False`"
        in summary
    )
    assert "Autonomous official result evidence valid race rows: `21`" in summary
    assert "Autonomous official result evidence blocked race rows: `4`" in summary
    assert (
        "Autonomous official result evidence blocker reasons: "
        "`{'runner_set_mismatch_quarantined': 4}`"
    ) in summary
    assert "Autonomous official result evidence inserted rows: `63`" in summary
    assert "Live odds backlog unresolved races: `11`" in summary
    assert "Live odds backlog recovery actions:" in summary
    assert "validate_runner_set_then_alias_join" in summary
    assert "Live odds backlog retryable exact-shadow matches: `5`" in summary
    assert "Live odds backlog no exact shadow match: `6`" in summary
    assert "Live odds backlog recovery queue:" in summary
    assert "live_odds_backlog_recovery_queue.json" in summary
    assert "Live odds backlog recovery queue diagnostic only: `True`" in summary
    assert "Live odds backlog recovery queue DB write performed: `False`" in summary
    assert "Live odds backlog awaiting official-result evidence races: `5`" in summary
    assert (
        "Live odds backlog awaiting official-result authorized action: "
        "`diagnostic_recheck_official_result_evidence_only`"
    ) in summary
    assert "Live odds backlog runner-set validation:" in summary
    assert "live_odds_backlog_runner_set_validation.json" in summary
    assert "Live odds backlog runner-set join authorized: `False`" in summary
    assert "Live odds backlog runner-set DB write performed: `False`" in summary
    assert "Live odds backlog join eligibility packet:" in summary
    assert "live_odds_backlog_join_eligibility_packet.json" in summary
    assert "Live odds backlog join eligibility report-only races: `2`" in summary
    assert "Live odds backlog join eligibility blocked races: `3`" in summary
    assert "Live odds backlog join eligibility join authorized: `False`" in summary
    assert "Live odds backlog join eligibility DB write performed: `False`" in summary
    assert (
        "Rolling model comparison: `ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW`"
        in summary
    )
    assert "Rolling model comparison sample races: `131`" in summary
    assert "Rolling model comparison best candidate: `market_only_implied`" in summary
    assert (
        "Rolling model comparison source rejected live odds candidates: `5`"
        in summary
    )
    assert "High-accuracy refinement: `BLOCKED_KEEP_BASELINE`" in summary
    assert "High-accuracy PR gate: `BLOCKED`" in summary
    assert "High-accuracy unified eligible rows: `856`" in summary
    assert "Promotion distance: `PROMOTION_DISTANCE_BLOCKED`" in summary
    assert "Promotion distance promotion ready: `False`" in summary
    assert "Promotion distance sample races: `131`" in summary
    assert "Promotion distance best candidate: `market_only_implied`" in summary
    assert (
        "Promotion distance best non-market candidate: `stage2_rf_calibrated`"
        in summary
    )
    assert (
        "Promotion distance blockers: "
        "`['best_non_market_top1_margin_below_target']`"
    ) in summary
    assert "Promotion distance source rejected live odds candidates: `5`" in summary
    assert (
        "Promotion distance source exclusion reasons: "
        "`{'official_result_missing': 32}`"
    ) in summary
    assert (
        "Promotion distance source official-result missing race IDs: "
        "`['Race 7 - TAREE - 2026-06-13']`"
    ) in summary
    assert (
        "Promotion distance official-result coverage requested races: `7`"
        in summary
    )
    assert (
        "Promotion distance official-result requested race count source: "
        "`deduped_requested_or_inferred_race_ids`"
    ) in summary
    assert (
        "Promotion distance official-result legacy requested race count without IDs: "
        "`4125`"
    ) in summary
    assert (
        "Promotion distance official-result coverage races with rows: `1`"
        in summary
    )
    assert (
        "Promotion distance official-result coverage missing races: `1`"
        in summary
    )
    assert (
        "Promotion distance official-result missing exclusions: `32`"
        in summary
    )
    assert "Promotion distance official-result runner path count: `1`" in summary
    assert (
        "Promotion distance official-result runner paths source: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in summary
    assert "Promotion distance source official-result runner paths:" not in summary
    assert (
        "Best aggregate unified evidence: `BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT`"
        in summary
    )
    assert (
        "Best aggregate unified rows with rejected live odds candidates: `6`"
        in summary
    )
    assert "Rejoin rejected live odds candidates: `8`" in summary
    assert "Rejoin rows with rejected live odds candidates: `6`" in summary
    assert (
        "Rejoin promotion distance source official-result missing race IDs: "
        "`['Race 8 - TAREE - 2026-06-13']`"
    ) in summary
    assert (
        "Rejoin promotion distance official-result coverage requested races: `3`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result requested race count source: "
        "`deduped_requested_or_inferred_race_ids`"
    ) in summary
    assert (
        "Rejoin promotion distance official-result legacy requested race count without IDs: "
        "`19`"
    ) in summary
    assert (
        "Rejoin promotion distance official-result coverage races with rows: `1`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result coverage missing races: `1`"
        in summary
    )
    assert (
        "Rejoin promotion distance official-result missing exclusions: `4`"
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
    assert "Rejoin promotion distance source official-result runner paths:" not in summary
    assert (
        "Rejoin rolling comparison: `ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW`"
        in summary
    )
    assert "Rejoin rolling comparison sample races: `132`" in summary
    assert "Rejoin high-accuracy packet: `BLOCKED_KEEP_BASELINE`" in summary
    assert (
        "Rejoin pre-race gated challenger: "
        "`PRE_RACE_GATED_CHALLENGER_REVIEW_READY`"
    ) in summary
    assert "Rejoin pre-race gated challenger promotion ready: `False`" in summary
    assert (
        "Rejoin rank-first hypothesis review: `RANK_FIRST_HYPOTHESIS_REVIEW_READY`"
        in summary
    )
    assert "Rejoin rank-first hypothesis best candidate: `market_residual_blend`" in summary
    assert (
        "Rejoin time-split gated challenger: `TIME_SPLIT_GATED_CHALLENGER_BLOCKED`"
        in summary
    )
    assert (
        "Rejoin market residual challenger: "
        "`MARKET_RESIDUAL_CHALLENGER_REVIEW_READY`"
    ) in summary
    assert (
        "Rejoin market residual regime audit: `MARKET_RESIDUAL_REGIME_AUDIT_READY`"
        in summary
    )
    assert (
        "Rejoin rank-first watchlist best candidate: `rank_first_blend_001`"
        in summary
    )
    assert (
        "Rejoin promotion distance blockers: "
        "`['rank_first_directional_sample_underpowered']`"
    ) in summary
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


def test_build_summary_tolerates_malformed_autonomous_odds_window():
    summary = runtime.build_summary(
        {
            "runtime_action": "WAIT_UNTIL_RECOMMENDED_REFRESH",
            "runtime_action_reasons": ["next_race_outside_preferred_window"],
            "safe_joined_races": 84,
            "target_joined_races": 100,
            "safe_joined_races_remaining": 16,
            "daemon": {
                "last_verdict": "DAEMON_READY",
                "last_cycle_activity_status": "NO_NEW_PREDICTIONS_OR_SAFE_JOINS",
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
                "last_autonomous_live_odds_next_prejump_window": "malformed",
            },
            "daily_shadow_run": {},
            "feature_activation_gate": {},
            "shadow_odds_coverage": {},
        }
    )

    assert "Autonomous odds next window opens: `2026-06-09T10:35:00+10:00`" in summary
    assert "Autonomous odds next race: `None`" in summary


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
