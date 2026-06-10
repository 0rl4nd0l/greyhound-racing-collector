import os
import sqlite3
import subprocess
from pathlib import Path

from scripts import shadow_autopilot_daemon as daemon


def test_daemon_default_min_joined_races_matches_review_target():
    args = daemon.parse_args(["run-once"])

    assert daemon.DEFAULT_TARGET_JOINED_RACES == 100
    assert daemon.DEFAULT_MIN_JOINED_RACES == 100
    assert args.target_joined_races == 100
    assert args.min_joined_races == 100


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
    )
    timer = daemon.timer_file_text()

    assert "Type=oneshot" in service
    assert "shadow_autopilot_daemon.py run-once" in service
    assert "--rejoin-pending-limit 8" in service
    assert "GREYHOUND_ALLOW_TGR=0" in service
    assert "/home/l4nd0/.local/bin" in service
    assert "OnUnitActiveSec=15min" in timer
    assert "Persistent=true" in timer


def test_systemd_deployment_status_reports_active_installed_timer():
    def fake_runner(command, capture_output, text, timeout, check):
        unit_name = command[2]
        if unit_name == daemon.SERVICE_NAME:
            stdout = "\n".join(
                [
                    "LoadState=loaded",
                    "ActiveState=inactive",
                    "UnitFileState=enabled",
                    "FragmentPath=/etc/systemd/system/shadow-autopilot.service",
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

    assert status["deployment_status"] == "INSTALLED_AND_ACTIVE"
    assert status["deployment_ready"] is True
    assert status["service_installed"] is True
    assert status["timer_installed"] is True
    assert status["timer_enabled"] is True
    assert status["timer_active"] is True
    assert status["service_unit"]["status"] == "LOADED_ENABLED_INACTIVE"
    assert status["timer_unit"]["status"] == "ACTIVE"
    assert status["no_write_guarantees"]["db_write"] is False


def test_systemd_deployment_status_fails_closed_when_timer_missing():
    def fake_runner(command, capture_output, text, timeout, check):
        unit_name = command[2]
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
                ('R1', 'Sandown', 1, '2026-06-08', '2026-06-08T22:00:00+10:00');
            INSERT INTO dog_race_data
                (race_id, dog_name, dog_clean_name, box_number)
            VALUES
                ('R1', 'Fast Dog', 'Fast Dog', 1);
            INSERT INTO live_odds
                (race_id, venue, race_number, race_date, dog_name, dog_clean_name,
                 box_number, odds_decimal, market_type, source, timestamp, is_current,
                 source_url)
            VALUES
                ('R1', 'Sandown', 1, '2026-06-08', '1. Fast Dog', '1. Fast Dog',
                 1, 3.5, 'win', 'sportsbet', '2026-06-08T21:45:00+10:00', 1,
                 'https://www.sportsbet.com.au/greyhound-racing/sandown/race-1');
            """
        )

    before_hash = daemon.sha256_file(db_path)
    summary = daemon.build_read_only_odds_coverage_report(
        db_path=db_path,
        output_dir=tmp_path / "packet",
        generated_at=daemon.datetime.fromisoformat("2026-06-08T21:50:00+10:00"),
    )
    after_hash = daemon.sha256_file(db_path)

    assert before_hash == after_hash
    assert summary["status"] == "SUCCESS"
    assert summary["odds_capture_performed"] is False
    assert summary["odds_used_for_shadow_scoring"] is False
    assert summary["dog_level_win_odds_rows"] == 1
    assert summary["safe_direct_identity_matches"] == 1
    assert summary["old_odds_row_audit"]["stale_rows"] == 0
    assert summary["old_odds_row_audit"]["missing_source_url_rows"] == 0
    assert summary["old_odds_row_audit"]["race_id_mismatch_rows"] == 0
    assert summary["old_odds_row_audit"]["dog_name_box_conflict_rows"] == 0
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
            "ev_output_rows": 0,
        },
    )

    assert "Shadow odds snapshot: `SKIPPED`" in summary
    assert "Shadow odds snapshot valid rows: `0`" in summary
    assert "Shadow odds races after feature freeze: `0`" in summary
    assert "Shadow odds EV output rows: `0`" in summary


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
