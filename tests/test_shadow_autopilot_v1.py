import json
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
    canonical_bytes,
    sha256_bytes,
)
from scripts import shadow_autopilot_v1 as autopilot


def test_autopilot_default_min_joined_races_matches_review_target():
    args = autopilot.parse_args([])

    assert autopilot.DEFAULT_TARGET_JOINED_RACES == 100
    assert autopilot.DEFAULT_MIN_JOINED_RACES_FOR_STATUS == 100
    assert autopilot.DEFAULT_ODDS_CAPTURE_MIN_MINUTES == 0.0
    assert autopilot.DEFAULT_ODDS_CAPTURE_MAX_MINUTES == 60.0
    assert autopilot.DEFAULT_ODDS_CAPTURE_REFRESH_LIMIT == 8
    assert autopilot.DEFAULT_HISTORICAL_UNIFIED_EVIDENCE_REPORT_LIMIT == 600
    assert args.target_joined_races == 100
    assert args.min_joined_races == 100
    assert args.odds_capture_min_minutes == 0.0
    assert args.odds_capture_max_minutes == 60.0
    assert args.odds_capture_refresh_limit == 8
    assert args.autonomous_odds_capture_limit is None
    assert args.enable_autonomous_odds_capture is False
    assert args.execute_autonomous_odds_capture is False
    assert args.allow_auto_scrape_odds is False
    assert args.require_safe_refresh_metadata is False
    assert args.enable_autonomous_result_capture is False
    assert args.skip_unified_dataset is False
    assert args.step_timeout_seconds == 840


def test_autopilot_accepts_autonomous_result_capture_flag():
    args = autopilot.parse_args(["--enable-autonomous-result-capture"])

    assert args.enable_autonomous_result_capture is True


def test_autopilot_accepts_step_timeout_seconds():
    args = autopilot.parse_args(["--step-timeout-seconds", "12"])

    assert args.step_timeout_seconds == 12


def test_autopilot_accepts_current_race_index_state_path():
    args = autopilot.parse_args(
        ["--current-race-index-state-path", "/runtime/odds_capture_state.json"]
    )

    assert args.current_race_index_state_path == Path(
        "/runtime/odds_capture_state.json"
    )


def test_current_race_index_is_published_from_completed_refresh(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "evidence"
    output_dir = evidence_root / "run"
    output_dir.mkdir(parents=True)
    state_path = evidence_root / "runtime/odds_capture_state.json"
    source_path = output_dir / "refresh_prejump_report.json"
    observed = {}
    def fake_publish(**kwargs):
        observed.update(kwargs)
        return {
            "schema_version": "collector_current_race_index_publish_v2",
            "status": "PUBLISHED",
            "source_generated_at": "2026-06-12T00:01:01+10:00",
            "run_id": "scheduled-run",
            "packet_sha256": "a" * 64,
            "race_count": 1,
        }

    monkeypatch.setattr(autopilot, "publish_current_race_index", fake_publish)

    result = autopilot.publish_current_race_index_after_refresh(
        state_path=state_path,
        evidence_root=evidence_root,
        output_dir=output_dir,
        run_id="scheduled-run",
        source_refresh_report_path=source_path,
    )

    assert result["status"] == "PUBLISHED"
    assert result["race_count"] == 1
    assert observed == {
        "state_path": state_path,
        "evidence_root": evidence_root,
        "source_refresh_report_path": source_path,
        "run_id": "scheduled-run",
    }
    report_path = output_dir / "current_race_index_publish.json"
    assert report_path.read_bytes() == canonical_bytes(result)
    lifecycle_path = state_path.parent / "manual_prediction_current_race_index.state.json"
    lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    assert lifecycle["run_id"] == "scheduled-run"
    assert lifecycle["packet_sha256"] == "a" * 64
    assert lifecycle["publication_report_path"] == "run/current_race_index_publish.json"
    assert lifecycle["publication_report_sha256"] == sha256_bytes(
        canonical_bytes(result)
    )


def test_step_command_records_timeout_and_logs_output(tmp_path, monkeypatch):
    def fake_run(command, cwd, text, capture_output, check, timeout):
        assert timeout == 3
        raise autopilot.subprocess.TimeoutExpired(
            cmd=command,
            timeout=timeout,
            output="partial stdout",
            stderr="partial stderr",
        )

    monkeypatch.setattr(autopilot.subprocess, "run", fake_run)

    result = autopilot.step_command(
        name="slow_step",
        command=["python", "-c", "while True: pass"],
        output_dir=tmp_path,
        timeout_seconds=3,
    )

    assert result["status"] == "FAIL"
    assert result["returncode"] == -9
    assert result["timed_out"] is True
    assert result["timeout_seconds"] == 3
    assert (tmp_path / "logs/slow_step.stdout.txt").read_text(
        encoding="utf-8"
    ) == "partial stdout"
    stderr = (tmp_path / "logs/slow_step.stderr.txt").read_text(encoding="utf-8")
    assert "partial stderr" in stderr
    assert "exceeded step timeout of 3 seconds" in stderr


def test_materialize_root_stage2_predictions_uses_nested_shadow_score_live(tmp_path):
    output_dir = tmp_path / "out"
    daily_dir = tmp_path / "daily"
    fallback = daily_dir / "shadow_score_live/stage2_shadow_predictions.jsonl"
    fallback.parent.mkdir(parents=True)
    fallback.write_text(
        json.dumps(
            {
                "race_id": "Race 1 - WPK - 2026-06-10",
                "dog_name": "Alpha Runner",
                "box": 1,
                "shadow_rf_calibrated_probability": 0.37,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    status = autopilot.materialize_root_stage2_predictions(
        daily_dir,
        output_dir=output_dir,
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    root = daily_dir / "stage2_shadow_predictions.jsonl"
    assert status["status"] == "STAGE2_SHADOW_PREDICTIONS_ROOT_MATERIALIZED"
    assert status["root_materialized"] is True
    assert status["hashes_match"] is True
    assert status["stage2_prediction_rows"] == 1
    assert root.read_text(encoding="utf-8") == fallback.read_text(encoding="utf-8")
    persisted = json.loads(
        (output_dir / "stage2_shadow_predictions_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["status"] == status["status"]


def test_materialize_root_stage2_predictions_prefers_existing_root(tmp_path):
    output_dir = tmp_path / "out"
    daily_dir = tmp_path / "daily"
    root = daily_dir / "stage2_shadow_predictions.jsonl"
    fallback = daily_dir / "shadow_score_live/stage2_shadow_predictions.jsonl"
    root.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    root.write_text(
        json.dumps({"stage2_challenger_key": "root"}) + "\n",
        encoding="utf-8",
    )
    fallback.write_text(
        json.dumps({"stage2_challenger_key": "nested"}) + "\n",
        encoding="utf-8",
    )

    status = autopilot.materialize_root_stage2_predictions(
        daily_dir,
        output_dir=output_dir,
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert status["status"] == "STAGE2_SHADOW_PREDICTIONS_ROOT_PRESENT"
    assert status["root_materialized"] is False
    assert json.loads(root.read_text(encoding="utf-8"))["stage2_challenger_key"] == "root"


def test_materialize_root_stage2_predictions_replaces_empty_root_with_nested(
    tmp_path,
):
    output_dir = tmp_path / "out"
    daily_dir = tmp_path / "daily"
    root = daily_dir / "stage2_shadow_predictions.jsonl"
    fallback = daily_dir / "shadow_score_live/stage2_shadow_predictions.jsonl"
    root.parent.mkdir(parents=True)
    fallback.parent.mkdir(parents=True)
    root.write_text("", encoding="utf-8")
    fallback.write_text(
        json.dumps({"stage2_challenger_key": "nested"}) + "\n",
        encoding="utf-8",
    )

    status = autopilot.materialize_root_stage2_predictions(
        daily_dir,
        output_dir=output_dir,
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
    )

    assert status["status"] == "STAGE2_SHADOW_PREDICTIONS_EMPTY_ROOT_REPLACED"
    assert status["root_materialized"] is True
    assert status["root_stage2_prediction_rows_before"] == 0
    assert status["fallback_stage2_prediction_rows"] == 1
    assert json.loads(root.read_text(encoding="utf-8"))["stage2_challenger_key"] == (
        "nested"
    )


def test_output_guard_rejects_non_autopilot_paths():
    try:
        autopilot.assert_output_dir_safe(Path("model_registry/shadow_autopilot_v1_bad"))
    except ValueError as exc:
        assert "output_dir_must_be_shadow_autopilot_artifact" in str(exc)
    else:
        raise AssertionError("expected protected output path to be rejected")


def test_output_guard_accepts_configured_external_evidence_root(
    tmp_path, monkeypatch
):
    repo_root = tmp_path / "release_repo"
    evidence_root = tmp_path / "runtime_artifacts" / "full_evidence_orchestration_20260525"
    output_dir = evidence_root / "shadow_autopilot_v1_external"
    repo_root.mkdir()

    monkeypatch.setattr(autopilot, "ROOT", repo_root)

    assert (
        autopilot.assert_output_dir_safe(output_dir, evidence_root=evidence_root)
        == output_dir.absolute()
    )

    try:
        autopilot.assert_output_dir_safe(
            evidence_root / "not_an_autopilot_output",
            evidence_root=evidence_root,
        )
    except ValueError as exc:
        assert str(exc).startswith("output_dir_must_be_shadow_autopilot_artifact")
    else:
        raise AssertionError("external evidence root must still enforce autopilot prefix")


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
        autonomous_official_result_capture_status={
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x"
            ),
            "attempted": True,
            "candidate_count": 2,
            "official_result_race_rows": 1,
            "official_result_runner_rows": 8,
            "quarantine_rows": 1,
            "quarantined_race_ids": ["Race 7 - CANN - 2026-06-13"],
            "quarantine_result_boxes_not_in_participants_counts": {"9": 1},
            "quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 7 - CANN - 2026-06-13",
                    "result_boxes_not_in_participants": [9],
                }
            ],
            "skipped_reason_counts": {"race_not_jumped": 1},
            "awaiting_jump_race_count": 1,
            "awaiting_jump_race_ids": ["Race 7 - CANN - 2026-06-13"],
            "awaiting_jump_next_recheck_after_local": "2026-06-13T22:55:00+10:00",
            "official_result_evidence_db_ingest_status": "NOOP_ALREADY_PRESENT",
            "official_result_evidence_db_execute": True,
            "official_result_evidence_db_write_performed": False,
            "official_result_evidence_valid_race_rows": 21,
            "official_result_evidence_valid_runner_rows": 150,
            "official_result_evidence_blocked_race_rows": 0,
            "official_result_evidence_blocked_runner_rows": 0,
            "official_result_evidence_inserted_race_rows": 0,
            "official_result_evidence_inserted_runner_rows": 0,
            "official_result_evidence_blocker_reason_counts": {},
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
    assert (
        dashboard["autonomous_official_result_capture"]["status"]
        == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    )
    assert dashboard["autonomous_official_result_capture"]["official_result_race_rows"] == 1
    assert dashboard["autonomous_official_result_capture"]["official_result_runner_rows"] == 8
    assert dashboard["autonomous_official_result_capture"][
        "quarantined_race_ids"
    ] == ["Race 7 - CANN - 2026-06-13"]
    assert dashboard["autonomous_official_result_capture"][
        "quarantine_result_boxes_not_in_participants_counts"
    ] == {"9": 1}
    assert dashboard["autonomous_official_result_capture"][
        "quarantine_runner_set_mismatch_samples"
    ][0]["result_boxes_not_in_participants"] == [9]
    assert dashboard["autonomous_official_result_capture"]["skipped_reason_counts"] == {
        "race_not_jumped": 1
    }
    assert dashboard["autonomous_official_result_capture"]["awaiting_jump_race_count"] == 1
    assert dashboard["autonomous_official_result_capture"]["awaiting_jump_race_ids"] == [
        "Race 7 - CANN - 2026-06-13"
    ]
    assert dashboard["autonomous_official_result_capture"][
        "awaiting_jump_next_recheck_after_local"
    ] == "2026-06-13T22:55:00+10:00"


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
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-09T00:52:00+10:00",
    )

    assert "scripts/collect_shadow_odds_snapshots.py" in command[1]
    assert "--evidence-root" in command
    assert "--shadow-run-dir" in command
    assert "--output-dir" in command
    assert "artifacts/full_evidence_orchestration_20260525/shadow_odds_snapshot_x" in command
    assert "--db" in command
    assert "greyhound_racing_data.db" in command
    assert "--current-time" in command


def test_autonomous_live_odds_capture_command_requires_explicit_execute_flags():
    command = autopilot.autonomous_live_odds_capture_command(
        input_dirs=[Path("upcoming_a"), Path("upcoming_b")],
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_live_odds_capture_x"
        ),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-10T14:00:00+10:00",
        limit=16,
        execute=False,
        allow_auto_scrape_odds=False,
    )

    assert any("scripts/autonomous_live_odds_capture.py" in part for part in command)
    assert "--evidence-root" in command
    assert command.count("--input-dir") == 2
    assert "--execute" not in command
    assert "--allow-auto-scrape-odds" not in command

    approved = autopilot.autonomous_live_odds_capture_command(
        input_dirs=[Path("upcoming_a")],
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_live_odds_capture_x"
        ),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-10T14:00:00+10:00",
        limit=16,
        execute=True,
        allow_auto_scrape_odds=True,
    )

    assert "--execute" in approved
    assert "--allow-auto-scrape-odds" in approved

    receipt_enabled = autopilot.autonomous_live_odds_capture_command(
        input_dirs=[Path("upcoming_a")],
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_live_odds_capture_x"
        ),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-10T14:00:00+10:00",
        limit=16,
        execute=True,
        allow_auto_scrape_odds=True,
        collector_receipt_root=Path("collector-requests"),
        collector_run_id="scheduled-run-1",
        forward_corpus_root=Path("forward-corpus"),
        forward_baseline_config=Path("forward-baseline.json"),
    )

    assert receipt_enabled[
        receipt_enabled.index("--collector-receipt-root") + 1
    ] == "collector-requests"
    assert receipt_enabled[
        receipt_enabled.index("--collector-run-id") + 1
    ] == "scheduled-run-1"
    assert receipt_enabled[
        receipt_enabled.index("--forward-corpus-root") + 1
    ] == "forward-corpus"
    assert receipt_enabled[
        receipt_enabled.index("--forward-baseline-config") + 1
    ] == "forward-baseline.json"
    with pytest.raises(ValueError, match="collector_receipt_authority_missing"):
        autopilot.autonomous_live_odds_capture_command(
            input_dirs=[Path("upcoming_a")],
            evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
            capture_dir=Path("autonomous_live_odds_capture_x"),
            db_path=Path("greyhound_racing_data.db"),
            current_time="2026-06-10T14:00:00+10:00",
            limit=16,
            execute=False,
            allow_auto_scrape_odds=False,
            collector_receipt_root=Path("collector-requests"),
            collector_run_id="scheduled-run-1",
        )
    with pytest.raises(
        ValueError, match="forward_corpus_scheduled_receipt_authority_missing"
    ):
        autopilot.autonomous_live_odds_capture_command(
            input_dirs=[Path("upcoming_a")],
            evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
            capture_dir=Path("autonomous_live_odds_capture_x"),
            db_path=Path("greyhound_racing_data.db"),
            current_time="2026-06-10T14:00:00+10:00",
            limit=16,
            execute=True,
            allow_auto_scrape_odds=True,
            forward_corpus_root=Path("forward-corpus"),
            forward_baseline_config=Path("forward-baseline.json"),
        )

    existing_binding = autopilot.autonomous_live_odds_capture_command(
        input_dirs=[Path("upcoming_a")],
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        capture_dir=Path("autonomous_live_odds_capture_x"),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-10T14:00:00+10:00",
        limit=16,
        execute=True,
        allow_auto_scrape_odds=True,
        collector_receipt_root=Path("collector-requests"),
        collector_run_id="scheduled-run-1",
        forward_corpus_root=Path("forward-corpus"),
        forward_current_race_index_path=Path("current-race-index.json"),
    )
    assert existing_binding[
        existing_binding.index("--forward-corpus-root") + 1
    ] == "forward-corpus"
    assert "--forward-baseline-config" not in existing_binding
    assert existing_binding[
        existing_binding.index("--forward-current-race-index-path") + 1
    ] == "current-race-index.json"

    existing_args = autopilot.parse_args(
        ["--forward-corpus-root", "forward-corpus"]
    )
    assert existing_args.forward_corpus_root == Path("forward-corpus")
    assert existing_args.forward_baseline_config is None
    with pytest.raises(SystemExit):
        autopilot.parse_args(
            ["--forward-baseline-config", "forward-baseline.json"]
        )


def test_manual_request_command_is_bound_to_claimed_collector_run():
    command = autopilot.autonomous_live_odds_capture_command(
        input_dirs=[Path("upcoming_a")],
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_live_odds_capture_x"
        ),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-10T14:00:00+10:00",
        limit=1,
        execute=True,
        allow_auto_scrape_odds=True,
        manual_request_root=Path("manual_requests"),
        manual_request_id="a" * 32,
        collector_run_id="scheduled-run-1",
    )

    assert command[-6:] == [
        "--manual-request-root",
        "manual_requests",
        "--manual-request-id",
        "a" * 32,
        "--collector-run-id",
        "scheduled-run-1",
    ]


def test_scheduled_collector_authority_requires_direct_shared_lock_owner(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "shadow_autopilot_daemon_runtime"
    runtime.mkdir()
    lock_path = runtime / "shadow_autopilot.lock"
    autopilot.write_json(
        lock_path,
        {
            "pid": os.getppid(),
            "hostname": socket.gethostname(),
            "run_id": "scheduled-run-1",
        },
    )

    assert autopilot.scheduled_collector_authority(tmp_path)["run_id"] == (
        "scheduled-run-1"
    )
    monkeypatch.setattr(autopilot.os, "getppid", lambda: 999999)
    assert autopilot.scheduled_collector_authority(tmp_path) is None


def test_scheduled_collector_authority_uses_explicit_external_lock(tmp_path):
    evidence_root = tmp_path / "external_evidence"
    explicit_lock = tmp_path / "daemon_runtime" / "shared.lock"
    explicit_lock.parent.mkdir()
    autopilot.write_json(
        explicit_lock,
        {
            "pid": os.getppid(),
            "hostname": socket.gethostname(),
            "run_id": "scheduled-run-external",
        },
    )

    authority = autopilot.scheduled_collector_authority(
        evidence_root,
        lock_path=explicit_lock,
    )

    assert authority is not None
    assert authority["run_id"] == "scheduled-run-external"


def test_scheduled_collector_authority_parses_from_opened_descriptor(
    tmp_path, monkeypatch
):
    lock_path = tmp_path / "shared.lock"
    original = {
        "pid": os.getppid(),
        "hostname": socket.gethostname(),
        "run_id": "opened-descriptor-run",
    }
    autopilot.write_json(lock_path, original)
    real_fstat = os.fstat

    def replace_path_after_open(descriptor):
        metadata = real_fstat(descriptor)
        lock_path.rename(tmp_path / "opened.lock")
        autopilot.write_json(
            lock_path,
            {
                **original,
                "pid": -1,
                "run_id": "replacement-path-run",
            },
        )
        return metadata

    monkeypatch.setattr(autopilot.os, "fstat", replace_path_after_open)

    authority = autopilot.scheduled_collector_authority(
        tmp_path,
        lock_path=lock_path,
    )

    assert authority is not None
    assert authority["run_id"] == "opened-descriptor-run"


def test_scheduled_collector_authority_rejects_unsafe_explicit_locks(tmp_path):
    lock_path = tmp_path / "shared.lock"
    valid = {
        "pid": os.getppid(),
        "hostname": socket.gethostname(),
        "run_id": "scheduled-run-1",
    }

    assert autopilot.scheduled_collector_authority(
        tmp_path, lock_path=lock_path
    ) is None

    lock_path.mkdir()
    assert autopilot.scheduled_collector_authority(
        tmp_path, lock_path=lock_path
    ) is None
    lock_path.rmdir()

    lock_path.write_text("{malformed", encoding="utf-8")
    assert autopilot.scheduled_collector_authority(
        tmp_path, lock_path=lock_path
    ) is None

    for field in ("pid", "hostname", "run_id"):
        payload = dict(valid)
        payload.pop(field)
        autopilot.write_json(lock_path, payload)
        assert autopilot.scheduled_collector_authority(
            tmp_path, lock_path=lock_path
        ) is None

    for field, value in (("pid", -1), ("hostname", "wrong-host")):
        payload = {**valid, field: value}
        autopilot.write_json(lock_path, payload)
        assert autopilot.scheduled_collector_authority(
            tmp_path, lock_path=lock_path
        ) is None

    target = tmp_path / "lock-target"
    autopilot.write_json(target, valid)
    lock_path.unlink()
    lock_path.symlink_to(target)
    assert autopilot.scheduled_collector_authority(
        tmp_path, lock_path=lock_path
    ) is None


def test_manual_request_is_deferred_during_active_capture_boundary(tmp_path):
    now = datetime.fromisoformat("2026-06-10T14:00:00+10:00")
    protocol = ManualPredictionCollectorProtocol(
        tmp_path / autopilot.PROTOCOL_DIRECTORY
    )
    request = protocol.publish_request(
        race={
            "race_id": "Race 1 - WPK - 2026-06-10",
            "url": (
                "https://www.thedogs.com.au/racing/wentworth-park/"
                "2026-06-10/1/example"
            ),
            "venue": "WPK",
            "race_number": 1,
            "race_date": "2026-06-10",
            "jump_timestamp": "2026-06-10T15:00:00+10:00",
        },
        expected_runners=[],
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )

    _, context = autopilot.prepare_manual_collector_request(
        evidence_root=tmp_path,
        collector_run_id="scheduled-run-1",
        current_time=now,
        active_capture=True,
    )

    assert context is None
    assert not protocol.claim_path(str(request["request_id"])).exists()


def test_manual_request_missing_exact_attempt_emits_terminal_response(tmp_path):
    now = datetime.fromisoformat("2026-06-10T14:00:00+10:00")
    protocol = ManualPredictionCollectorProtocol(
        tmp_path / autopilot.PROTOCOL_DIRECTORY
    )
    request = protocol.publish_request(
        race={
            "race_id": "Race 1 - WPK - 2026-06-10",
            "url": (
                "https://www.thedogs.com.au/racing/wentworth-park/"
                "2026-06-10/1/example"
            ),
            "venue": "WPK",
            "race_number": 1,
            "race_date": "2026-06-10",
            "jump_timestamp": "2026-06-10T15:00:00+10:00",
        },
        expected_runners=[],
        created_at=now,
        expires_at=now + timedelta(minutes=10),
    )
    context = protocol.claim_request(
        str(request["request_id"]),
        now=now,
        collector_run_id="scheduled-run-1",
    )
    protocol.begin_attempt(
        context,
        now=now,
        collector_run_id="scheduled-run-1",
    )

    response = autopilot.finalize_manual_collector_request(
        protocol=protocol,
        context=context,
        capture_report={"attempts": []},
        evidence_root=tmp_path,
        db_path=tmp_path / "unused.db",
        current_time=now + timedelta(seconds=1),
    )

    assert response["status"] == "CAPTURE_FAILED"
    assert protocol.read_response(str(request["request_id"])) == response


def test_capture_only_manual_request_seals_response_without_shadow_model(
    tmp_path, monkeypatch
):
    from scripts import predict_race_now

    generated_at = datetime.now().astimezone()
    parent_cycle_time = generated_at - timedelta(seconds=1)
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    lock_path = tmp_path / "daemon_runtime/shadow_autopilot.lock"
    lock_path.parent.mkdir()
    autopilot.write_json(
        lock_path,
        {
            "pid": os.getppid(),
            "hostname": socket.gethostname(),
            "run_id": "scheduled-capture-only",
        },
    )
    protocol = ManualPredictionCollectorProtocol(
        evidence_root / autopilot.PROTOCOL_DIRECTORY
    )
    jump_at = generated_at + timedelta(minutes=30)
    race_date = jump_at.date().isoformat()
    race = {
        "race_id": f"Race 1 - WPK - {race_date}",
        "url": (
            "https://www.thedogs.com.au/racing/wentworth-park/"
            f"{race_date}/1/example"
        ),
        "venue": "WPK",
        "race_number": 1,
        "race_date": race_date,
        "jump_timestamp": jump_at.isoformat(),
    }
    request = protocol.publish_request(
        race=race,
        expected_runners=[],
        created_at=generated_at,
        expires_at=generated_at + timedelta(minutes=10),
    )
    step_names: list[str] = []
    capture_commands: list[list[str]] = []
    handoff: dict[str, object] = {}

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        step_names.append(name)
        if name == "refresh_odds_capture_candidates":
            autopilot.write_json(
                command_value(command, "--output"),
                {"status": "READY", "dry_run": False, "files": []},
            )
        elif name == "autonomous_live_odds_capture":
            capture_commands.append(list(command))
            claimed_protocol = ManualPredictionCollectorProtocol(
                command_value(command, "--manual-request-root")
            )
            context = claimed_protocol.claimed_request(
                command[command.index("--manual-request-id") + 1]
            )
            captured_at = datetime.fromisoformat(
                command[command.index("--current-time") + 1]
            )
            assert captured_at == datetime.fromisoformat(
                str(context.claim["claimed_at"])
            )
            claimed_protocol.begin_attempt(
                context,
                now=captured_at,
                collector_run_id=command[
                    command.index("--collector-run-id") + 1
                ],
            )
            runners = [
                {
                    "dog_name": "Alpha",
                    "dog_clean_name": "Alpha",
                    "box_number": 1,
                    "identity": "ALPHA",
                    "odds_decimal": 2.5,
                },
                {
                    "dog_name": "Beta",
                    "dog_clean_name": "Beta",
                    "box_number": 2,
                    "identity": "BETA",
                    "odds_decimal": 4.0,
                },
            ]
            validation = {
                "schema_version": "autonomous_live_odds_capture_validation_v1",
                "status": "PASS",
                "source_url": "https://www.sportsbet.com.au/example",
                "accepted_rows": runners,
                "accepted_place_rows": runners,
                "reasons": [],
            }
            attempt = {
                "schema_version": "autonomous_live_odds_capture_attempt_v1",
                "race_id": race["race_id"],
                "status": "APPENDED",
                "reasons": [],
                "capture_window_minutes": 10,
                "validation": validation,
            }
            capture_report = {
                "schema_version": "autonomous_live_odds_capture_report_v1",
                "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
                "status": "APPENDED",
                "execute": True,
                "allow_auto_scrape_odds": True,
                "ready_count": 1,
                "validation_pass_count": 1,
                "inserted_live_odds_rows": 4,
                "status_counts": {"APPENDED": 1},
                "attempts": [attempt],
            }
            capture_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                capture_dir / "autonomous_live_odds_capture_report.json",
                capture_report,
            )
            report_raw = canonical_bytes(capture_report)
            form_raw = b"dog_name,box_number\nAlpha,1\nBeta,2\n"
            sidecar_raw = canonical_bytes(
                {"participants": [{"dog_name": "Alpha"}, {"dog_name": "Beta"}]}
            )
            handoff.update(
                {
                    "schema_version": "on_demand_verified_master_packet_v1",
                    "race_id": race["race_id"],
                    "append_timestamp": captured_at.isoformat(),
                    "source_report_sha256": sha256_bytes(report_raw),
                    "source_form_sha256": sha256_bytes(form_raw),
                    "source_sidecar_sha256": sha256_bytes(sidecar_raw),
                    "packet_record_schema_version": (
                        "market_form_residual_shadow_record_v3"
                    ),
                    "packet_record_checksum_sha256": "d" * 64,
                    "packet_effective_state_schema_version": (
                        "market_form_residual_effective_state_v2"
                    ),
                    "packet_effective_state_sha256": "e" * 64,
                    "_report_bytes": report_raw,
                    "_form_bytes": form_raw,
                    "_sidecar_bytes": sidecar_raw,
                }
            )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)
    monkeypatch.setattr(
        predict_race_now,
        "discover_capture_handoff",
        lambda **_: handoff,
    )

    args = autopilot.parse_args(
        [
            "--run-id",
            "capture_only_manual_request",
            "--evidence-root",
            str(evidence_root),
            "--collector-lock-path",
            str(lock_path),
            "--current-time",
            parent_cycle_time.isoformat(),
            "--db",
            str(db_path),
            "--enable-autonomous-odds-capture",
            "--execute-autonomous-odds-capture",
            "--allow-auto-scrape-odds",
            "--skip-primary-refresh",
            "--skip-shadow-run",
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
            "--skip-unified-dataset",
        ]
    )

    autopilot.run_autopilot(args)

    response = protocol.read_response(str(request["request_id"]))
    assert response is not None
    assert response["status"] == "RECEIPT_READY"
    assert protocol.attempt_path(str(request["request_id"])).exists()
    assert protocol.receipt_path(str(request["request_id"])).exists()
    consumed = protocol.consume_response(
        str(request["request_id"]),
        now=datetime.now().astimezone(),
    )
    assert consumed["response"]["status"] == "RECEIPT_READY"
    assert len(list((protocol.root / "responses").glob("*.json"))) == 1
    assert len(list((protocol.root / "consumed").glob("*.json"))) == 1
    assert step_names == [
        "refresh_odds_capture_candidates",
        "autonomous_live_odds_capture",
    ]
    assert len(capture_commands) == 1
    assert command_value(
        capture_commands[0], "--manual-request-root"
    ) == protocol.root
    assert capture_commands[0][
        capture_commands[0].index("--manual-request-id") + 1
    ] == request["request_id"]
    assert capture_commands[0][
        capture_commands[0].index("--collector-run-id") + 1
    ] == "scheduled-capture-only"
    assert datetime.fromisoformat(
        capture_commands[0][capture_commands[0].index("--current-time") + 1]
    ) > parent_cycle_time


def test_full_shadow_run_without_model_still_fails_closed(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})
    args = autopilot.parse_args(
        [
            "--run-id",
            "full_shadow_without_model",
            "--evidence-root",
            str(evidence_root),
            "--db",
            str(db_path),
        ]
    )

    try:
        autopilot.run_autopilot(args)
    except RuntimeError as exc:
        assert "shadow_model_required_for_no_training_autopilot" in str(exc)
    else:
        raise AssertionError("expected a genuine shadow run to require a model")


def test_autonomous_live_odds_capture_status_surfaces_inserted_rows():
    status = autopilot.build_autonomous_live_odds_capture_status(
        generated_at=datetime.fromisoformat("2026-06-10T14:00:00+10:00"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_live_odds_capture_x"
        ),
        capture_report={
            "run_id": "x",
            "output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_live_odds_capture_x"
            ),
            "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "status": "APPENDED",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
            "execute": True,
            "allow_auto_scrape_odds": True,
            "ready_count": 1,
            "validation_pass_count": 1,
            "inserted_live_odds_rows": 7,
            "status_counts": {"APPENDED": 1},
            "blocked_attempt_count": 0,
            "blocked_attempts": [],
            "capture_window_coverage": {
                "race_count": 1,
                "window_count": 4,
                "status_counts": {
                    "CAPTURED": 1,
                    "DUE": 1,
                    "PENDING": 2,
                },
            },
            "no_write_guarantees": {
                "db_write": True,
                "odds_history_write": False,
                "race_metadata_write": False,
            },
        },
        attempted=True,
        returncode=0,
    )

    assert status["status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert status["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    assert status["operator_status"] == "APPENDED"
    assert status["run_id"] == "x"
    assert status["runtime_action"] == "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW"
    assert status["readiness_decision"] == "CONTINUE_ODDS_CAPTURE"
    assert status["output_dir"].endswith("autonomous_live_odds_capture_x")
    assert status["attempted"] is True
    assert status["execute"] is True
    assert status["timed_out"] is False
    assert status["recovered_from_step_failure"] is False
    assert status["ready_count"] == 1
    assert status["inserted_live_odds_rows"] == 7
    assert status["blocked_attempt_count"] == 0
    assert status["blocked_attempts"] == []
    assert status["capture_window_coverage_race_count"] == 1
    assert status["capture_window_coverage_window_count"] == 4
    assert status["capture_window_coverage_status_counts"] == {
        "CAPTURED": 1,
        "DUE": 1,
        "PENDING": 2,
    }
    assert status["capture_window_coverage_report"].endswith(
        "autonomous_live_odds_capture_window_coverage.json"
    )
    assert status["no_write_guarantees"]["db_write"] is True


def test_autonomous_live_odds_capture_status_omits_empty_window_coverage_report():
    status = autopilot.build_autonomous_live_odds_capture_status(
        generated_at=datetime.fromisoformat("2026-06-12T05:36:16+10:00"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_live_odds_capture_x"
        ),
        capture_report={
            "run_id": "x",
            "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
            "status": "READY",
            "runtime_action": "WAIT_FOR_NEXT_ODDS_CAPTURE_WINDOW",
            "readiness_decision": "CONTINUE_ODDS_CAPTURE",
            "execute": True,
            "allow_auto_scrape_odds": True,
            "ready_count": 0,
            "validation_pass_count": 0,
            "inserted_live_odds_rows": 0,
            "status_counts": {},
            "capture_window_coverage": {
                "race_count": 0,
                "window_count": 0,
                "status_counts": {},
            },
            "no_write_guarantees": {
                "db_write": False,
                "odds_history_write": False,
                "race_metadata_write": False,
            },
        },
        odds_capture_refresh_report=_waiting_refresh_report(),
        attempted=True,
        returncode=0,
    )

    assert status["final_status"] == "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    assert status["operator_status"] == "READY"
    assert status["run_id"] == "x"
    assert status["capture_window_coverage_race_count"] == 0
    assert status["capture_window_coverage_window_count"] == 0
    assert status["capture_window_coverage_status_counts"] == {}
    assert status["capture_window_coverage_report"] is None
    assert status["next_window_opens_at"] == "2026-06-09T08:55:00+10:00"
    assert status["recommended_rerun_after_local"] == "2026-06-09T08:55:00+10:00"
    assert status["next_race_id"] == "Race 1 - AP_K - 2026-06-09"
    assert status["next_prejump_window"]["status"] == "WAITING_FOR_FUTURE_WINDOW"


def test_autonomous_official_result_capture_command_is_dry_run_artifact_lane():
    command = autopilot.autonomous_official_result_capture_command(
        target_date="2026-06-10",
        upcoming_dir=Path("upcoming"),
        shadow_run_dir=None,
        snapshot_dir=Path("artifacts/prediction_snapshots"),
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        db_path=Path("greyhound_racing_data.db"),
        race_ids=["Race 1 - WPK - 2026-06-10"],
        require_ready_snapshot=True,
    )

    assert "scripts/autonomous_official_result_capture.py" in command[1]
    assert "--date" in command
    assert "2026-06-10" in command
    assert "--require-ready-snapshot" in command
    assert "--race-id" in command
    assert "--execute-db-ingest" not in command
    assert "--write-labels-approved" not in command


def test_autonomous_official_result_capture_command_can_use_shadow_run_candidates():
    command = autopilot.autonomous_official_result_capture_command(
        target_date="2026-06-10",
        upcoming_dir=None,
        shadow_run_dir=Path("daily_shadow"),
        snapshot_dir=None,
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        db_path=Path("greyhound_racing_data.db"),
        current_time="2026-06-10T15:00:00+10:00",
        require_ready_snapshot=False,
        include_live_odds_backlog=True,
        backlog_evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        backlog_limit=12,
        backlog_shadow_run_limit=20,
        backlog_lookback_days=2,
        execute_db_ingest=True,
    )

    assert "scripts/autonomous_official_result_capture.py" in command[1]
    assert "--shadow-run-dir" in command
    assert "daily_shadow" in command
    assert "--upcoming-dir" not in command
    assert "--require-ready-snapshot" not in command
    assert "--current-time" in command
    assert "--include-live-odds-backlog" in command
    assert "--backlog-evidence-root" in command
    assert "--backlog-limit" in command
    assert "12" in command
    assert "--backlog-shadow-run-limit" in command
    assert "20" in command
    assert "--backlog-lookback-days" in command
    assert "2" in command
    assert "--execute-db-ingest" in command
    assert "--write-labels-approved" not in command


def test_autopilot_defaults_to_wider_official_result_backlog():
    args = autopilot.parse_args([])

    assert args.result_backlog_limit == autopilot.DEFAULT_RESULT_BACKLOG_LIMIT
    assert args.result_backlog_limit == 128
    assert (
        args.result_backlog_shadow_run_limit
        == autopilot.DEFAULT_RESULT_BACKLOG_SHADOW_RUN_LIMIT
    )
    assert args.result_backlog_shadow_run_limit == 200
    assert args.result_backlog_lookback_days == autopilot.DEFAULT_RESULT_BACKLOG_LOOKBACK_DAYS
    assert args.result_backlog_lookback_days == 2


def test_autonomous_official_result_capture_status_surfaces_artifact_counts():
    status = autopilot.build_autonomous_official_result_capture_status(
        generated_at=datetime.fromisoformat("2026-06-10T14:00:00+10:00"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_x"
        ),
        capture_report={
            "final_status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "candidate_count": 3,
            "ingested_count": 2,
            "failed_count": 1,
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
                },
                {
                    "race_id": "Race 12 - GRDN - 2026-06-13",
                    "reason": "race_not_jumped",
                },
            ],
            "official_result_race_rows": 2,
            "official_result_runner_rows": 16,
            "quarantine_rows": 1,
            "quarantined_race_ids": ["Race 12 - GRDN - 2026-06-13"],
            "quarantine_reason_counts": {"ingest_failed_or_unsafe_match": 1},
            "quarantine_error_counts": {"result_boxes_not_in_participants:9": 1},
            "quarantine_result_boxes_not_in_participants_counts": {"9": 1},
            "quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 12 - GRDN - 2026-06-13",
                    "result_boxes_not_in_participants": [9],
                    "attempted_source_box_sets": [
                        {
                            "source": "thedogs_official",
                            "result_boxes": [6, 5, 7, 9, 2, 1, 4],
                            "terminal_status_boxes": [3, 10],
                        }
                    ],
                }
            ],
            "official_result_evidence_db_ingest": {
                "status": "NOT_EXECUTED",
                "execute": False,
                "db_write_performed": False,
                "valid_race_rows": 91,
                "valid_runner_rows": 628,
                "blocked_race_rows": 1,
                "blocked_runner_rows": 5,
                "inserted_race_rows": 0,
                "inserted_runner_rows": 0,
                "blocker_reason_counts": {
                    "duplicate_finish_positions": 1,
                    "finish_positions_not_contiguous": 1,
                },
            },
            "live_odds_backlog_enabled": True,
            "live_odds_backlog_lookback_days": 2,
            "live_odds_backlog_target_dates": ["2026-06-10", "2026-06-09"],
            "live_odds_backlog_discovered_race_count": 3,
            "live_odds_backlog_discovered_race_ids": [
                "Race 1 - WPK - 2026-06-10",
                "Race 2 - WPK - 2026-06-10",
                "Race 3 - WPK - 2026-06-09",
            ],
            "live_odds_backlog_candidate_race_count": 2,
            "live_odds_backlog_candidate_race_ids": [
                "Race 1 - WPK - 2026-06-10",
                "Race 2 - WPK - 2026-06-10",
            ],
            "live_odds_backlog_unresolved_race_count": 1,
            "live_odds_backlog_unresolved_race_ids": [
                "Race 3 - WPK - 2026-06-09"
            ],
            "live_odds_backlog_unresolved_races": [
                {
                    "race_id": "Race 3 - WPK - 2026-06-09",
                    "reason": "no_matching_shadow_run_candidate_found",
                    "race_date": "2026-06-09",
                }
            ],
            "live_odds_backlog_unresolved_reason_counts": {
                "no_matching_shadow_run_candidate_found": 1
            },
            "live_odds_backlog_unresolved_recovery_action_counts": {
                "inspect_shadow_run_candidate_coverage": 1
            },
            "live_odds_backlog_unresolved_alias_status_counts": {
                "NO_EXACT_SHADOW_ARTIFACT_MATCH": 1
            },
            "live_odds_backlog_retryable_exact_shadow_match_race_count": 0,
            "live_odds_backlog_no_exact_shadow_match_race_count": 1,
            "live_odds_backlog_retryable_exact_shadow_match_race_ids": [],
            "live_odds_backlog_no_exact_shadow_match_race_ids": [
                "Race 3 - WPK - 2026-06-09"
            ],
            "live_odds_backlog_awaiting_official_result_evidence_race_count": 1,
            "live_odds_backlog_awaiting_official_result_evidence_race_ids": [
                "Race 1 - WPK - 2026-06-10"
            ],
            "live_odds_backlog_awaiting_official_result_evidence_authorized_action": (
                "diagnostic_recheck_official_result_evidence_only"
            ),
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count": 1,
            "live_odds_backlog_recovery_queue_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_recovery_queue.json"
            ),
            "live_odds_backlog_runner_set_validation_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "live_odds_backlog_runner_set_validation.json"
            ),
            "live_odds_backlog_runner_set_validation_retryable_race_count": 1,
            "live_odds_backlog_runner_set_validation_exact_match_race_count": 1,
            "live_odds_backlog_runner_set_validation_blocked_race_count": 0,
            "live_odds_backlog_runner_set_validation_diagnostic_only": True,
            "live_odds_backlog_runner_set_validation_join_authorized": False,
            "live_odds_backlog_runner_set_validation_db_write_performed": False,
            "live_odds_backlog_join_eligibility_packet_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
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
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count": 2,
            "shadow_run_candidate_source_report": (
                "artifacts/full_evidence_orchestration_20260525/"
                "autonomous_official_result_capture_x/"
                "shadow_run_candidate_source_report.json"
            ),
            "no_write_guarantees": {
                "db_write": False,
                "label_write": False,
                "snapshot_rewrite": False,
                "manifest_rewrite": False,
            },
        },
        attempted=True,
        returncode=0,
    )

    assert status["status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    assert status["attempted"] is True
    assert status["timed_out"] is False
    assert status["candidate_count"] == 3
    assert status["progress_candidate_count"] == 0
    assert status["progress_completed_count"] == 0
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
    assert status["progress_status_counts"] == {}
    assert status["progress_active_candidate"] is None
    assert status["official_result_race_rows"] == 2
    assert status["official_result_runner_rows"] == 16
    assert status["quarantine_rows"] == 1
    assert status["quarantined_race_ids"] == ["Race 12 - GRDN - 2026-06-13"]
    assert status["quarantine_reason_counts"] == {
        "ingest_failed_or_unsafe_match": 1
    }
    assert status["quarantine_error_counts"] == {
        "result_boxes_not_in_participants:9": 1
    }
    assert status["quarantine_result_boxes_not_in_participants_counts"] == {"9": 1}
    assert status["quarantine_runner_set_mismatch_samples"][0][
        "attempted_source_box_sets"
    ][0]["terminal_status_boxes"] == [3, 10]
    assert status["official_result_evidence_db_ingest_status"] == "NOT_EXECUTED"
    assert status["official_result_evidence_db_execute"] is False
    assert status["official_result_evidence_db_write_performed"] is False
    assert status["official_result_evidence_valid_race_rows"] == 91
    assert status["official_result_evidence_valid_runner_rows"] == 628
    assert status["official_result_evidence_blocked_race_rows"] == 1
    assert status["official_result_evidence_blocked_runner_rows"] == 5
    assert status["official_result_evidence_inserted_race_rows"] == 0
    assert status["official_result_evidence_inserted_runner_rows"] == 0
    assert status["official_result_evidence_blocker_reason_counts"] == {
        "duplicate_finish_positions": 1,
        "finish_positions_not_contiguous": 1,
    }
    assert status["official_result_evidence_db_ingest"]["status"] == "NOT_EXECUTED"
    assert status["live_odds_backlog_enabled"] is True
    assert status["live_odds_backlog_lookback_days"] == 2
    assert status["live_odds_backlog_target_dates"] == ["2026-06-10", "2026-06-09"]
    assert status["live_odds_backlog_discovered_race_count"] == 3
    assert status["live_odds_backlog_discovered_race_ids"] == [
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-10",
        "Race 3 - WPK - 2026-06-09",
    ]
    assert status["live_odds_backlog_candidate_race_count"] == 2
    assert status["live_odds_backlog_candidate_race_ids"] == [
        "Race 1 - WPK - 2026-06-10",
        "Race 2 - WPK - 2026-06-10",
    ]
    assert status["live_odds_backlog_unresolved_race_count"] == 1
    assert status["live_odds_backlog_unresolved_race_ids"] == [
        "Race 3 - WPK - 2026-06-09"
    ]
    assert status["live_odds_backlog_unresolved_races"][0]["reason"] == (
        "no_matching_shadow_run_candidate_found"
    )
    assert status["live_odds_backlog_unresolved_reason_counts"] == {
        "no_matching_shadow_run_candidate_found": 1
    }
    assert status["live_odds_backlog_unresolved_recovery_action_counts"] == {
        "inspect_shadow_run_candidate_coverage": 1
    }
    assert status["live_odds_backlog_unresolved_alias_status_counts"] == {
        "NO_EXACT_SHADOW_ARTIFACT_MATCH": 1
    }
    assert status["live_odds_backlog_retryable_exact_shadow_match_race_count"] == 0
    assert status["live_odds_backlog_no_exact_shadow_match_race_count"] == 1
    assert status["live_odds_backlog_retryable_exact_shadow_match_race_ids"] == []
    assert status["live_odds_backlog_no_exact_shadow_match_race_ids"] == [
        "Race 3 - WPK - 2026-06-09"
    ]
    assert status["live_odds_backlog_awaiting_official_result_evidence_race_count"] == 1
    assert status["live_odds_backlog_awaiting_official_result_evidence_race_ids"] == [
        "Race 1 - WPK - 2026-06-10"
    ]
    assert status[
        "live_odds_backlog_awaiting_official_result_evidence_authorized_action"
    ] == "diagnostic_recheck_official_result_evidence_only"
    assert (
        status[
            "live_odds_backlog_awaiting_official_result_recheck_ready_race_count"
        ]
        == 1
    )
    assert status["live_odds_backlog_recovery_queue_path"].endswith(
        "live_odds_backlog_recovery_queue.json"
    )
    assert status["live_odds_backlog_recovery_queue_diagnostic_only"] is True
    assert status["live_odds_backlog_recovery_queue_join_acceptance_changed"] is False
    assert status["live_odds_backlog_recovery_queue_db_write_performed"] is False
    assert status["live_odds_backlog_runner_set_validation_path"].endswith(
        "live_odds_backlog_runner_set_validation.json"
    )
    assert status["live_odds_backlog_runner_set_validation_retryable_race_count"] == 1
    assert status["live_odds_backlog_runner_set_validation_exact_match_race_count"] == 1
    assert status["live_odds_backlog_runner_set_validation_blocked_race_count"] == 0
    assert status["live_odds_backlog_runner_set_validation_diagnostic_only"] is True
    assert status["live_odds_backlog_runner_set_validation_join_authorized"] is False
    assert status["live_odds_backlog_runner_set_validation_db_write_performed"] is False
    assert status["live_odds_backlog_join_eligibility_packet_path"].endswith(
        "live_odds_backlog_join_eligibility_packet.json"
    )
    assert status["live_odds_backlog_join_eligibility_evaluated_race_count"] == 1
    assert (
        status["live_odds_backlog_join_eligibility_eligible_report_only_race_count"]
        == 1
    )
    assert status["live_odds_backlog_join_eligibility_blocked_race_count"] == 0
    assert status["live_odds_backlog_join_eligibility_blocker_counts"] == {
        "official_result_runner_set_exact_live_odds_match": 1,
    }
    assert status["live_odds_backlog_join_eligibility_diagnostic_only"] is True
    assert status["live_odds_backlog_join_eligibility_join_authorized"] is False
    assert status["live_odds_backlog_join_eligibility_db_write_performed"] is False
    assert (
        status[
            "live_odds_backlog_join_eligibility_awaiting_official_result_recheck_ready_race_count"
        ]
        == 2
    )
    assert status["shadow_run_candidate_source_report"].endswith(
        "shadow_run_candidate_source_report.json"
    )
    assert status["no_write_guarantees"]["label_write"] is False


def test_autonomous_official_result_capture_status_surfaces_timeout_progress():
    status = autopilot.build_autonomous_official_result_capture_status(
        generated_at=datetime.fromisoformat("2026-06-13T15:00:00+10:00"),
        capture_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "autonomous_official_result_capture_timeout"
        ),
        capture_report=None,
        progress_report={
            "candidate_count": 128,
            "completed_count": 37,
            "status_counts": {
                "FAILED_VALIDATION": 27,
                "FETCH_IN_PROGRESS": 1,
                "INGESTED_DRY_RUN": 10,
            },
            "active_candidate": {
                "race_id": "Race 2 - MAND - 2026-06-12",
                "candidate_index": 38,
                "candidate_count": 128,
                "status": "FETCH_IN_PROGRESS",
            },
        },
        attempted=True,
        returncode=-15,
        timed_out=True,
    )

    assert status["status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_FAILED_NO_REPORT"
    assert status["attempted"] is True
    assert status["timed_out"] is True
    assert status["returncode"] == -15
    assert status["progress_path"].endswith(
        "autonomous_official_result_capture_progress.json"
    )
    assert status["progress_attempts_path"].endswith(
        "autonomous_official_result_capture_attempts.progress.jsonl"
    )
    assert status["progress_candidate_count"] == 128
    assert status["progress_completed_count"] == 37
    assert status["progress_status_counts"] == {
        "FAILED_VALIDATION": 27,
        "FETCH_IN_PROGRESS": 1,
        "INGESTED_DRY_RUN": 10,
    }
    assert status["progress_active_candidate"]["race_id"] == (
        "Race 2 - MAND - 2026-06-12"
    )
    assert status["progress_active_candidate"]["candidate_index"] == 38
    assert status["no_write_guarantees"]["label_write"] is False


def test_unified_evidence_dataset_command_is_report_only_artifact_lane(tmp_path):
    odds_path = tmp_path / "shadow_odds_snapshot.jsonl"
    result_path = tmp_path / "official_result_runners.jsonl"
    joined_path = tmp_path / "joined_shadow_predictions.jsonl"
    join_eligibility_packet_path = tmp_path / "live_odds_backlog_join_eligibility_packet.json"
    odds_path.write_text("{}", encoding="utf-8")
    result_path.write_text("{}", encoding="utf-8")
    joined_path.write_text("{}", encoding="utf-8")
    join_eligibility_packet_path.write_text("{}", encoding="utf-8")

    command = autopilot.unified_evidence_dataset_command(
        shadow_run_dir=Path("daily_shadow"),
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "unified_evidence_dataset_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        db_path=Path("greyhound_racing_data.db"),
        odds_jsonl_paths=[odds_path],
        official_result_runner_paths=[result_path],
        joined_shadow_prediction_paths=[joined_path],
        join_eligibility_packet_paths=[join_eligibility_packet_path],
    )

    assert "scripts/build_unified_evidence_dataset.py" in command[1]
    assert "--evidence-root" in command
    assert "--shadow-run-dir" in command
    assert "--odds-jsonl" in command
    assert "--official-result-runners-jsonl" in command
    assert "--joined-shadow-predictions-jsonl" in command
    assert "--join-eligibility-packet" in command
    assert "--write-labels-approved" not in command
    assert "--execute" not in command


def test_shadow_odds_snapshot_paths_for_daily_dir_uses_exact_run_id(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    daily_dir = (
        evidence_root
        / "daily_race_ingest_shadow_20260612T233202+1000_daemon_autopilot"
    )
    odds_path = (
        evidence_root
        / "shadow_odds_snapshot_20260612T233202+1000_daemon_autopilot"
        / "shadow_odds_snapshot.jsonl"
    )
    nearby_wrong_path = (
        evidence_root
        / "shadow_odds_snapshot_20260612T233211+1000_daemon_autopilot"
        / "shadow_odds_snapshot.jsonl"
    )
    odds_path.parent.mkdir(parents=True)
    odds_path.write_text("{}", encoding="utf-8")
    nearby_wrong_path.parent.mkdir(parents=True)
    nearby_wrong_path.write_text("{}", encoding="utf-8")

    assert autopilot.shadow_odds_snapshot_paths_for_daily_dir(
        evidence_root=evidence_root,
        daily_dir=daily_dir,
    ) == [odds_path]


def test_unified_evidence_dataset_status_surfaces_eligibility_counts():
    status = autopilot.build_unified_evidence_dataset_status(
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
        dataset_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "unified_evidence_dataset_x"
        ),
        dataset_report={
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "row_count": 10,
            "race_count": 2,
            "rows_with_official_results": 8,
            "rows_with_stage2_predictions": 10,
            "rows_with_strict_prejump_odds": 4,
            "label_evaluation_eligible_rows": 8,
            "stage2_evaluation_eligible_rows": 8,
            "odds_evaluation_eligible_rows": 4,
            "unified_evidence_eligible_rows": 4,
            "exclusion_reason_counts": {"strict_prejump_odds_missing": 6},
            "odds_exclusion_reason_counts": {"odds_source_url_missing": 2},
            "rejected_live_odds_candidate_count": 3,
            "rows_with_rejected_live_odds_candidates": 2,
            "rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 1,
                "unsupported_sportsbet_box_source:missing": 2,
            },
            "official_result_coverage": {
                "source": "unified_evidence_dataset",
                "requested_race_count": 7,
                "requested_race_count_source": (
                    "official_result_evidence_db_audit_requested_race_ids"
                ),
                "requested_race_ids": [
                    "Race 4 - TAREE - 2026-06-13",
                    "Race 5 - TAREE - 2026-06-13",
                ],
                "races_with_rows_count": 2,
                "missing_race_count": 5,
                "missing_exclusion_count": 12,
                "missing_race_ids": ["Race 4 - TAREE - 2026-06-13"],
                "races_with_rows": [
                    "Race 5 - TAREE - 2026-06-13",
                    "Race 6 - TAREE - 2026-06-13",
                ],
                "runner_path_count": 1,
                "runner_paths_source_field": "official_result_runner_paths",
            },
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        attempted=True,
        returncode=0,
    )

    assert status["status"] == "UNIFIED_EVIDENCE_DATASET_BUILT"
    assert status["attempted"] is True
    assert status["row_count"] == 10
    assert status["rows_with_stage2_predictions"] == 10
    assert status["unified_evidence_eligible_rows"] == 4
    assert status["exclusion_reason_counts"]["strict_prejump_odds_missing"] == 6
    assert status["odds_exclusion_reason_counts"] == {"odds_source_url_missing": 2}
    assert status["rejected_live_odds_candidate_count"] == 3
    assert status["rows_with_rejected_live_odds_candidates"] == 2
    assert status["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 1,
        "unsupported_sportsbet_box_source:missing": 2,
    }
    assert status["official_result_coverage"]["source"] == "unified_evidence_dataset"
    assert status["official_result_coverage_requested_race_count"] == 7
    assert (
        status["official_result_coverage_requested_race_count_source"]
        == "official_result_evidence_db_audit_requested_race_ids"
    )
    assert status["official_result_coverage"]["requested_race_ids"] == [
        "Race 4 - TAREE - 2026-06-13",
        "Race 5 - TAREE - 2026-06-13",
    ]
    assert status["official_result_coverage_races_with_rows_count"] == 2
    assert status["official_result_coverage_missing_race_count"] == 5
    assert status["official_result_coverage_missing_exclusion_count"] == 12
    assert status["official_result_runner_path_count"] == 1
    assert (
        status["official_result_runner_paths_source_field"]
        == "official_result_runner_paths"
    )
    assert status["no_write_guarantees"]["label_write"] is False


def test_dashboard_surfaces_unified_official_result_coverage():
    unified_status = autopilot.build_unified_evidence_dataset_status(
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
        dataset_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "unified_evidence_dataset_x"
        ),
        dataset_report={
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "row_count": 85,
            "race_count": 12,
            "rows_with_official_results": 0,
            "rows_with_stage2_predictions": 85,
            "rows_with_strict_prejump_odds": 21,
            "unified_evidence_eligible_rows": 0,
            "official_result_coverage": {
                "source": "unified_evidence_dataset",
                "requested_race_count": 12,
                "requested_race_count_source": (
                    "official_result_evidence_db_audit_requested_race_ids"
                ),
                "races_with_rows_count": 0,
                "missing_race_count": 12,
                "missing_exclusion_count": 85,
                "runner_path_count": 1,
                "runner_paths_source_field": "official_result_runner_paths",
            },
        },
        attempted=True,
        returncode=0,
    )

    dashboard = autopilot.build_dashboard(
        generated_at=datetime.fromisoformat("2026-06-10T18:30:00+10:00"),
        aggregate_metrics={},
        join_metrics=None,
        aggregate_calibration={},
        aggregate_box_bias={},
        status_report={"final_status": "CONTINUE_FORWARD_SHADOW_COLLECTION"},
        sources={},
        unified_evidence_dataset_status=unified_status,
    )

    unified = dashboard["unified_evidence_dataset"]
    assert unified["official_result_coverage"]["source"] == "unified_evidence_dataset"
    assert unified["official_result_coverage_requested_race_count"] == 12
    assert (
        unified["official_result_coverage_requested_race_count_source"]
        == "official_result_evidence_db_audit_requested_race_ids"
    )
    assert unified["official_result_coverage_races_with_rows_count"] == 0
    assert unified["official_result_coverage_missing_race_count"] == 12
    assert unified["official_result_coverage_missing_exclusion_count"] == 85
    assert unified["official_result_runner_path_count"] == 1
    assert (
        unified["official_result_runner_paths_source_field"]
        == "official_result_runner_paths"
    )


def test_backlog_unified_shadow_run_dirs_uses_candidate_backlog_reports(tmp_path):
    result_capture_dir = tmp_path / "autonomous_official_result_capture_x"
    shadow_a = tmp_path / "daily_race_ingest_shadow_a"
    shadow_b = tmp_path / "daily_race_ingest_shadow_b"
    result_capture_dir.mkdir()
    autopilot.write_json(
        result_capture_dir / "shadow_run_candidate_source_report.json",
        {
            "live_odds_backlog": {
                "shadow_run_reports": [
                    {
                        "backlog_shadow_run_dir": str(shadow_a),
                        "candidate_count": 2,
                        "candidate_race_ids": [
                            "Race 1 - BEN - 2026-06-10",
                            "Race 2 - BEN - 2026-06-10",
                        ],
                    },
                    {
                        "backlog_shadow_run_dir": str(shadow_a),
                        "candidate_count": 1,
                        "candidate_race_ids": ["Race 2 - BEN - 2026-06-10"],
                    },
                    {
                        "backlog_shadow_run_dir": str(shadow_b),
                        "candidate_count": 3,
                        "candidate_race_ids": ["Race 3 - BEN - 2026-06-10"],
                    },
                    {
                        "backlog_shadow_run_dir": str(tmp_path / "empty_shadow"),
                        "candidate_count": 0,
                        "candidate_race_ids": ["Race 4 - BEN - 2026-06-10"],
                    },
                ],
            },
        },
    )

    candidates = autopilot.backlog_unified_shadow_run_candidates(result_capture_dir)
    assert candidates[0]["candidate_race_ids"] == [
        "Race 1 - BEN - 2026-06-10",
        "Race 2 - BEN - 2026-06-10",
    ]
    assert candidates[1]["candidate_race_ids"] == ["Race 3 - BEN - 2026-06-10"]
    assert autopilot.backlog_unified_shadow_run_dirs(result_capture_dir) == [
        shadow_a,
        shadow_b,
    ]


def test_filtered_official_result_rows_for_race_ids_stays_inside_candidate_boundary(tmp_path):
    source = tmp_path / "official_result_runners.jsonl"
    filtered = tmp_path / "filtered" / "official_result_runners.jsonl"
    autopilot.write_jsonl(
        source,
        [
            {"race_id": "Race 1 - BEN - 2026-06-10", "dog_name": "A"},
            {"race_id": "Race 2 - BEN - 2026-06-10", "dog_name": "B"},
            {"race_id": "Race 3 - BEN - 2026-06-10", "dog_name": "C"},
        ],
    )

    row_count = autopilot.filtered_official_result_rows_for_race_ids(
        source,
        filtered,
        ["Race 1 - BEN - 2026-06-10", "Race 3 - BEN - 2026-06-10"],
    )

    rows = autopilot.read_jsonl(filtered)
    assert row_count == 2
    assert [row["race_id"] for row in rows] == [
        "Race 1 - BEN - 2026-06-10",
        "Race 3 - BEN - 2026-06-10",
    ]


def test_backlog_official_result_runner_paths_omits_empty_filtered_artifact(tmp_path):
    official_result_path = tmp_path / "official_result_runners_backlog_001.jsonl"
    official_result_path.write_text("", encoding="utf-8")

    assert autopilot.backlog_official_result_runner_paths(
        official_result_path,
        row_count=0,
    ) == []
    assert autopilot.backlog_official_result_runner_paths(
        official_result_path,
        row_count=3,
    ) == [official_result_path]


def test_backlog_unified_evidence_status_sums_reports_and_failures():
    status = autopilot.build_backlog_unified_evidence_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:30:00+10:00"),
        reports=[
            {
                "output_dir": "artifacts/unified_backlog_001",
                "shadow_run_dir": "artifacts/daily_shadow_001",
                "candidate_race_count": 2,
                "candidate_race_ids": [
                    "Race 1 - BEN - 2026-06-10",
                    "Race 2 - BEN - 2026-06-10",
                ],
                "filtered_official_result_runner_rows": 16,
                "official_result_runner_paths": [
                    "artifacts/official_result_runners_backlog_001.jsonl"
                ],
                "filtered_official_result_runners_empty": False,
                "row_count": 16,
                "race_count": 2,
                "rows_with_official_results": 16,
                "rows_with_strict_prejump_odds": 8,
                "rows_with_artifact_shadow_odds": 5,
                "rows_with_artifact_shadow_odds_candidates": 6,
                "artifact_shadow_odds_candidate_count": 6,
                "artifact_shadow_odds_selected_bucket_count": 5,
                "artifact_odds_rows_seen": 16,
                "artifact_odds_rows_accepted": 5,
                "artifact_odds_rows_rejected": 11,
                "artifact_odds_rejection_reason_counts": {
                    "odds_match_status_not_valid_pre_jump_dog_odds": 11,
                },
                "unified_evidence_eligible_rows": 8,
                "exclusion_reason_counts": {
                    "strict_prejump_odds_missing": 8,
                    "official_result_missing": 1,
                },
                "odds_exclusion_reason_counts": {
                    "unsupported_sportsbet_box_source:missing": 3,
                },
                "rejected_live_odds_candidate_count": 4,
                "rows_with_rejected_live_odds_candidates": 3,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 1,
                    "odds_source_url_missing": 3,
                },
                "official_result_coverage": {
                    "requested_race_count": 2,
                    "requested_race_count_source": (
                        "official_result_evidence_db_audit_requested_race_ids"
                    ),
                    "requested_race_ids": [
                        "Race 1 - BEN - 2026-06-10",
                        "Race 2 - BEN - 2026-06-10",
                    ],
                    "races_with_rows_count": 2,
                    "races_with_rows": [
                        "Race 1 - BEN - 2026-06-10",
                        "Race 2 - BEN - 2026-06-10",
                    ],
                    "missing_race_count": 0,
                    "missing_race_ids": [],
                    "runner_path_count": 1,
                    "runner_paths_source_field": "official_result_runner_paths",
                    "missing_exclusion_count": 1,
                },
            },
            {
                "output_dir": "artifacts/unified_backlog_002",
                "shadow_run_dir": "artifacts/daily_shadow_002",
                "official_result_runner_paths": [
                    "artifacts/official_result_runners_backlog_002.jsonl"
                ],
                "row_count": 8,
                "race_count": 1,
                "rows_with_official_results": 8,
                "rows_with_strict_prejump_odds": 7,
                "rows_with_artifact_shadow_odds": 3,
                "rows_with_artifact_shadow_odds_candidates": 4,
                "artifact_shadow_odds_candidate_count": 4,
                "artifact_shadow_odds_selected_bucket_count": 3,
                "artifact_odds_rows_seen": 8,
                "artifact_odds_rows_accepted": 3,
                "artifact_odds_rows_rejected": 5,
                "artifact_odds_audits": [
                    {
                        "rejection_reason_counts": {
                            "odds_match_status_not_valid_pre_jump_dog_odds": 4,
                            "source_url_missing": 1,
                        }
                    }
                ],
                "unified_evidence_eligible_rows": 7,
                "exclusion_reason_counts": {
                    "strict_prejump_odds_missing": 1,
                    "stage2_shadow_prediction_missing": 2,
                },
                "odds_exclusion_reason_counts": {
                    "unsupported_sportsbet_box_source:missing": 2,
                    "odds_source_url_missing": 1,
                },
                "rejected_live_odds_candidate_count": 2,
                "rows_with_rejected_live_odds_candidates": 1,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 2,
                    "unsupported_sportsbet_box_source:missing": 1,
                },
                "official_result_coverage": {
                    "requested_race_count": 2,
                    "requested_race_count_source": (
                        "official_result_evidence_db_audit_requested_race_ids"
                    ),
                    "requested_race_ids": [
                        "Race 2 - BEN - 2026-06-10",
                        "Race 3 - BEN - 2026-06-10",
                    ],
                    "races_with_rows_count": 1,
                    "races_with_rows": ["Race 3 - BEN - 2026-06-10"],
                    "missing_race_count": 1,
                    "missing_race_ids": ["Race 2 - BEN - 2026-06-10"],
                    "runner_path_count": 1,
                    "runner_paths_source_field": "official_result_runner_paths",
                    "missing_exclusion_count": 2,
                },
            },
        ],
        failures=[
            {
                "output_dir": "artifacts/unified_backlog_003",
                "shadow_run_dir": "artifacts/daily_shadow_003",
                "returncode": 2,
            }
        ],
    )

    assert status["status"] == "BACKLOG_UNIFIED_EVIDENCE_DATASETS_PARTIAL"
    assert status["aggregation_scope"] == "per_dataset_totals_not_cross_dataset_deduped"
    assert status["attempted_dataset_count"] == 3
    assert status["dataset_count"] == 2
    assert status["failed_dataset_count"] == 1
    assert status["row_count"] == 24
    assert status["race_count"] == 3
    assert status["rows_with_official_results"] == 24
    assert status["rows_with_strict_prejump_odds"] == 15
    assert status["rows_with_artifact_shadow_odds"] == 8
    assert status["rows_with_artifact_shadow_odds_candidates"] == 10
    assert status["artifact_shadow_odds_candidate_count"] == 10
    assert status["artifact_shadow_odds_selected_bucket_count"] == 8
    assert status["artifact_odds_rows_seen"] == 24
    assert status["artifact_odds_rows_accepted"] == 8
    assert status["artifact_odds_rows_rejected"] == 16
    assert status["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 15,
        "source_url_missing": 1,
    }
    assert status["unified_evidence_eligible_rows"] == 15
    assert status["exclusion_reason_counts"] == {
        "official_result_missing": 1,
        "stage2_shadow_prediction_missing": 2,
        "strict_prejump_odds_missing": 9,
    }
    assert status["odds_exclusion_reason_counts"] == {
        "odds_source_url_missing": 1,
        "unsupported_sportsbet_box_source:missing": 5,
    }
    assert status["rejected_live_odds_candidate_count"] == 6
    assert status["rows_with_rejected_live_odds_candidates"] == 4
    assert status["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 3,
        "unsupported_sportsbet_box_source:missing": 1,
    }
    assert status["official_result_coverage"] == {
        "source": "backlog_unified_evidence_dataset_reports",
        "requested_race_count": 3,
        "requested_race_count_source": (
            "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
        ),
        "requested_race_ids": [
            "Race 1 - BEN - 2026-06-10",
            "Race 2 - BEN - 2026-06-10",
            "Race 3 - BEN - 2026-06-10",
        ],
        "legacy_requested_race_count_without_ids": 0,
        "races_with_rows_count": 3,
        "missing_race_count": 1,
        "missing_race_ids": ["Race 2 - BEN - 2026-06-10"],
        "races_with_rows": [
            "Race 1 - BEN - 2026-06-10",
            "Race 2 - BEN - 2026-06-10",
            "Race 3 - BEN - 2026-06-10",
        ],
        "runner_path_count": 2,
        "runner_paths_source_field": "official_result_runner_paths",
        "missing_exclusion_count": 3,
    }
    assert status["official_result_coverage_requested_race_count"] == 3
    assert status[
        "official_result_coverage_requested_race_count_source"
    ] == (
        "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
    )
    assert status["official_result_evidence_db_missing_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert status["official_result_evidence_db_races_with_rows"] == [
        "Race 1 - BEN - 2026-06-10",
        "Race 2 - BEN - 2026-06-10",
        "Race 3 - BEN - 2026-06-10",
    ]
    assert status["datasets"][0]["candidate_race_count"] == 2
    assert status["datasets"][0]["filtered_official_result_runner_rows"] == 16
    assert status["datasets"][0]["filtered_official_result_runners_empty"] is False
    assert status["datasets"][0]["artifact_odds_rows_accepted"] == 5
    assert status["datasets"][0]["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 11,
    }
    assert status["datasets"][0]["exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 8,
        "official_result_missing": 1,
    }
    assert status["datasets"][0]["rejected_live_odds_candidate_count"] == 4
    assert status["datasets"][0]["rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 1,
        "odds_source_url_missing": 3,
    }
    assert status["datasets"][1]["artifact_odds_rows_seen"] == 8
    assert status["datasets"][1]["artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 4,
        "source_url_missing": 1,
    }
    assert status["failures"][0]["returncode"] == 2
    assert status["no_write_guarantees"]["label_write"] is False


def test_backlog_unified_race_coverage_dedupes_repeated_race_instances(tmp_path):
    dataset_a = tmp_path / "dataset_a.jsonl"
    dataset_b = tmp_path / "dataset_b.jsonl"
    autopilot.write_jsonl(
        dataset_a,
        [
            {
                "race_id": "Race 1 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": True,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": True,
            },
            {
                "race_id": "Race 1 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": True,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": True,
            },
            {
                "race_id": "Race 2 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": False,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": False,
            },
            {
                "race_id": "Race 2 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": False,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": False,
            },
            {
                "race_id": "Race 3 - SAN - 2026-06-11",
                "race_date": "2026-06-11",
                "venue": "SAN",
                "official_result_available": False,
                "strict_prejump_odds_available": False,
                "unified_evidence_eligible": False,
            },
        ],
    )
    autopilot.write_jsonl(
        dataset_b,
        [
            {
                "race_id": "Race 2 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": True,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": True,
            },
            {
                "race_id": "Race 4 - SAN - 2026-06-11",
                "race_date": "2026-06-11",
                "venue": "SAN",
                "official_result_available": False,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": False,
            },
        ],
    )

    status = autopilot.build_backlog_unified_evidence_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:30:00+10:00"),
        reports=[
            {"output_dir": "artifacts/unified_backlog_001", "dataset_jsonl": str(dataset_a)},
            {"output_dir": "artifacts/unified_backlog_002", "dataset_jsonl": str(dataset_b)},
        ],
    )

    coverage = status["race_coverage"]
    assert status["race_coverage_summary"] == coverage
    assert coverage["scope"] == "dataset_race_instances_and_deduped_race_id"
    assert coverage["dataset_race_instance_count"] == 5
    assert coverage["deduped_race_count"] == 4
    assert coverage["race_instances_with_unified_evidence"] == 2
    assert coverage["deduped_races_with_unified_evidence"] == 2
    assert coverage["deduped_races_without_unified_evidence"] == 2
    assert coverage["deduped_races_without_complete_official_result_instance"] == 2
    assert coverage[
        "deduped_races_without_complete_strict_prejump_odds_instance"
    ] == 1
    assert coverage[
        "deduped_races_without_complete_official_result_instance_by_date"
    ] == {"2026-06-11": 2}
    assert coverage["gap_action_plan"]["sample_blocking_gap_count"] == 2
    assert status["gap_action_plan"] == coverage["gap_action_plan"]
    assert status["sample_blocking_gap_count"] == 2
    assert coverage["gap_action_plan"]["action_counts"] == {
        "collect_future_strict_prejump_odds": 1,
        "investigate_join_or_stage2_gap": 0,
        "monitor_non_sample_blocking_completion_gap": 0,
        "retry_official_result_capture_or_join": 1,
    }
    assert status["gap_action_counts"] == coverage["gap_action_plan"]["action_counts"]
    assert coverage["gap_action_plan"]["evidence_missing_reason_counts"] == {
        "join_or_stage2_gap": 0,
        "official_result_missing_only": 1,
        "strict_prejump_odds_missing": 1,
    }
    assert status["evidence_missing_reason_counts"] == (
        coverage["gap_action_plan"]["evidence_missing_reason_counts"]
    )
    assert {
        row["race_id"]: row["recommended_action"]
        for row in coverage["gap_action_plan"]["top_gap_races"]
    } == {
        "Race 3 - SAN - 2026-06-11": "collect_future_strict_prejump_odds",
        "Race 4 - SAN - 2026-06-11": "retry_official_result_capture_or_join",
    }
    assert status["top_gap_race_ids"] == [
        row["race_id"] for row in coverage["gap_action_plan"]["top_gap_races"]
    ]
    assert status["top_gap_races"] == coverage["gap_action_plan"]["top_gap_races"]
    assert coverage["top_official_result_missing_races"][0]["race_id"] == (
        "Race 2 - BEN - 2026-06-10"
    )
    assert status["top_official_result_missing_race_ids"] == [
        row["race_id"] for row in coverage["top_official_result_missing_races"]
    ]
    assert status["top_official_result_missing_races"] == (
        coverage["top_official_result_missing_races"]
    )


def test_official_result_quarantine_context_by_race_preserves_unsafe_match_reason(
    tmp_path,
):
    quarantine_path = tmp_path / "official_result_quarantine.jsonl"
    autopilot.write_jsonl(
        quarantine_path,
        [
            {
                "race_id": "Race 7 - TAREE - 2026-06-13",
                "reason": "ingest_failed_or_unsafe_match",
                "item": {
                    "errors": ["result_boxes_not_in_participants:9"],
                    "attempted_sources": [
                        {
                            "source": "thedogs_official",
                            "source_url": (
                                "https://www.thedogs.com.au/racing/taree/"
                                "2026-06-13/7/test?trial=false"
                            ),
                        }
                    ],
                },
            },
            {
                "race_id": "__browser__",
                "reason": "browser_unavailable:ModuleNotFoundError",
                "item": {"race_id": "__browser__"},
            },
        ],
    )

    context = autopilot.official_result_quarantine_context_by_race(quarantine_path)

    assert set(context) == {"Race 7 - TAREE - 2026-06-13"}
    assert context["Race 7 - TAREE - 2026-06-13"][
        "official_result_quarantine_reason"
    ] == "ingest_failed_or_unsafe_match"
    assert context["Race 7 - TAREE - 2026-06-13"][
        "official_result_quarantine_errors"
    ] == ["result_boxes_not_in_participants:9"]
    assert context["Race 7 - TAREE - 2026-06-13"][
        "official_result_quarantine_source_urls"
    ] == [
        "https://www.thedogs.com.au/racing/taree/2026-06-13/7/test?trial=false"
    ]


def test_backlog_unified_gap_plan_classifies_quarantined_official_result_gap(
    tmp_path,
):
    dataset_path = tmp_path / "dataset.jsonl"
    autopilot.write_jsonl(
        dataset_path,
        [
            {
                "race_id": "Race 7 - TAREE - 2026-06-13",
                "race_date": "2026-06-13",
                "venue": "TAREE",
                "official_result_available": False,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": False,
            }
        ],
    )

    status = autopilot.build_backlog_unified_evidence_status(
        generated_at=datetime.fromisoformat("2026-06-14T02:40:00+10:00"),
        reports=[
            {
                "output_dir": "artifacts/unified_backlog_001",
                "dataset_jsonl": str(dataset_path),
            }
        ],
        official_result_gap_context_by_race={
            "Race 7 - TAREE - 2026-06-13": {
                "official_result_quarantine_reason": "ingest_failed_or_unsafe_match",
                "official_result_quarantine_errors": [
                    "result_boxes_not_in_participants:9"
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
            }
        },
    )

    gap_plan = status["gap_action_plan"]
    assert gap_plan["sample_blocking_gap_count"] == 1
    assert gap_plan["action_counts"][
        "inspect_quarantined_official_result_runner_set"
    ] == 1
    assert gap_plan["action_counts"]["retry_official_result_capture_or_join"] == 0
    assert gap_plan["evidence_missing_reason_counts"][
        "official_result_quarantined_unsafe_match"
    ] == 1
    assert gap_plan["top_gap_races"][0]["recommended_action"] == (
        "inspect_quarantined_official_result_runner_set"
    )
    assert gap_plan["top_gap_races"][0]["official_result_quarantine_errors"] == [
        "result_boxes_not_in_participants:9"
    ]
    assert gap_plan["top_gap_races"][0][
        "official_result_quarantine_participant_boxes"
    ] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert gap_plan["top_gap_races"][0][
        "official_result_quarantine_result_boxes_not_in_participants"
    ] == [9]
    assert gap_plan["top_gap_races"][0][
        "official_result_quarantine_reserve_substitution_diagnostic"
    ] == {
        "classification": "possible_reserve_substitution_manual_review_required",
        "acceptance_status": "not_accepted_report_only",
        "candidate_reserve_boxes": [9],
        "scratched_participant_boxes": [1],
    }
    assert status["race_coverage"]["top_official_result_missing_races"][0][
        "official_result_quarantine_reason"
    ] == "ingest_failed_or_unsafe_match"


def test_official_result_quarantine_context_surfaces_runner_set_diagnostics(tmp_path):
    quarantine_path = tmp_path / "official_result_quarantine.jsonl"
    autopilot.write_jsonl(
        quarantine_path,
        [
            {
                "race_id": "Race 7 - TAREE - 2026-06-13",
                "reason": "ingest_failed_or_unsafe_match",
                "item": {
                    "errors": ["result_boxes_not_in_participants:9"],
                    "participant_source": "shadow_run_predictions",
                    "participant_count": 8,
                    "participant_boxes": [1, 2, 3, 4, 5, 6, 7, 8],
                    "participants": [
                        {"box_number": 1, "dog_name": "Red Rudolph"},
                        {"box_number": 2, "dog_name": "Riverside Levi"},
                    ],
                    "attempted_sources": [
                        {
                            "source": "thedogs_official",
                            "source_url": (
                                "https://www.thedogs.com.au/racing/taree/"
                                "2026-06-13/7/example?trial=false"
                            ),
                            "status": "resulted",
                            "raw_order": [2, 8, 4, 7, 3, 9, 6, 5],
                            "terminal_statuses": [
                                {"box_number": 1, "status": "SCR"},
                                {"box_number": 10, "status": "SCR"},
                            ],
                        }
                    ],
                },
            },
            {
                "race_id": "__browser__",
                "reason": "browser_unavailable:ModuleNotFoundError",
                "item": {},
            },
        ],
    )

    context = autopilot.official_result_quarantine_context_by_race(quarantine_path)

    assert sorted(context) == ["Race 7 - TAREE - 2026-06-13"]
    row = context["Race 7 - TAREE - 2026-06-13"]
    assert row["official_result_quarantine_reason"] == (
        "ingest_failed_or_unsafe_match"
    )
    assert row["official_result_quarantine_errors"] == [
        "result_boxes_not_in_participants:9"
    ]
    assert row["official_result_quarantine_participant_source"] == (
        "shadow_run_predictions"
    )
    assert row["official_result_quarantine_participant_count"] == 8
    assert row["official_result_quarantine_participant_boxes"] == [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert row[
        "official_result_quarantine_result_boxes_not_in_participants"
    ] == [9]
    assert row["official_result_quarantine_result_boxes_in_participants"] == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ]
    assert row["official_result_quarantine_participants"] == [
        {"box_number": 1, "dog_name": "Red Rudolph"},
        {"box_number": 2, "dog_name": "Riverside Levi"},
    ]
    assert row["official_result_quarantine_attempted_source_box_sets"] == [
        {
            "source": "thedogs_official",
            "status": "resulted",
            "source_url": (
                "https://www.thedogs.com.au/racing/taree/"
                "2026-06-13/7/example?trial=false"
            ),
            "result_boxes": [2, 8, 4, 7, 3, 9, 6, 5],
            "dog_names_by_box": {},
            "terminal_status_boxes": [1, 10],
        }
    ]
    assert row["official_result_quarantine_reserve_substitution_diagnostic"] == {
        "classification": "possible_reserve_substitution_manual_review_required",
        "acceptance_status": "not_accepted_report_only",
        "result_boxes_outside_participants": [9],
        "result_boxes_inside_participants": [2, 3, 4, 5, 6, 7, 8],
        "candidate_reserve_boxes": [9],
        "scratched_participant_boxes": [1],
        "terminal_status_boxes": [1, 10],
        "terminal_status_boxes_outside_participants": [10],
    }
def test_dashboard_and_daily_status_surface_backlog_unified_evidence(tmp_path):
    dataset_path = tmp_path / "unified_evidence_dataset.jsonl"
    autopilot.write_jsonl(
        dataset_path,
        [
            {
                "race_id": "Race 1 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": True,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": True,
            },
            {
                "race_id": "Race 2 - BEN - 2026-06-10",
                "race_date": "2026-06-10",
                "venue": "BEN",
                "official_result_available": False,
                "strict_prejump_odds_available": True,
                "unified_evidence_eligible": False,
            },
        ],
    )
    backlog_status = autopilot.build_backlog_unified_evidence_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:30:00+10:00"),
        reports=[
            {
                "output_dir": "artifacts/unified_backlog_001",
                "shadow_run_dir": "artifacts/daily_shadow_001",
                "dataset_jsonl": str(dataset_path),
                "row_count": 16,
                "race_count": 2,
                "rows_with_official_results": 16,
                "rows_with_strict_prejump_odds": 8,
                "rows_with_artifact_shadow_odds": 3,
                "artifact_odds_rows_seen": 16,
                "artifact_odds_rows_accepted": 3,
                "artifact_odds_rows_rejected": 13,
                "artifact_odds_rejection_reason_counts": {
                    "odds_match_status_not_valid_pre_jump_dog_odds": 13,
                },
                "unified_evidence_eligible_rows": 8,
                "exclusion_reason_counts": {"strict_prejump_odds_missing": 8},
                "odds_exclusion_reason_counts": {"odds_source_url_missing": 2},
                "rejected_live_odds_candidate_count": 5,
                "rows_with_rejected_live_odds_candidates": 4,
                "rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 3,
                    "odds_source_url_missing": 2,
                },
                "official_result_coverage": {
                    "requested_race_count": 2,
                    "requested_race_count_source": (
                        "official_result_evidence_db_audit_requested_race_ids"
                    ),
                    "requested_race_ids": [
                        "Race 1 - BEN - 2026-06-10",
                        "Race 2 - BEN - 2026-06-10",
                    ],
                    "races_with_rows_count": 1,
                    "races_with_rows": ["Race 1 - BEN - 2026-06-10"],
                    "missing_race_count": 1,
                    "missing_race_ids": ["Race 2 - BEN - 2026-06-10"],
                    "runner_path_count": 1,
                    "runner_paths_source_field": "official_result_runner_paths",
                    "missing_exclusion_count": 8,
                },
            }
        ],
    )
    high_accuracy_status = {
        "status": "BLOCKED_KEEP_BASELINE",
        "promotion_pr_gate_status": "BLOCKED_KEEP_BASELINE",
        "unified_evidence_eligible_rows": 11,
        "promotion_distance_status": "PROMOTION_DISTANCE_BLOCKED",
        "promotion_distance_promotion_ready": False,
        "promotion_distance_blockers": [
            "no_candidate_passed_rank_first_accuracy_gate"
        ],
        "promotion_distance_sample_race_count": 124,
        "promotion_distance_sample_runner_rows": 856,
        "promotion_distance_source_rejected_live_odds_candidate_count": 5,
        "promotion_distance_source_rows_with_rejected_live_odds_candidates": 4,
        "promotion_distance_source_rejected_live_odds_candidate_reason_counts": {
            "odds_decimal_invalid": 3,
            "odds_source_url_missing": 2,
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
        "promotion_distance_best_non_market_candidate_key": "stage2_rf_calibrated",
        "promotion_distance_best_non_market_top1_margin_gap": 0.02,
        "promotion_distance_predeclared_residual_candidate_status": "BELOW_FLOOR",
        "promotion_distance_predeclared_residual_triggered_race_count": 2,
        "promotion_distance_report": (
            "artifacts/full_evidence_orchestration_20260525/"
            "promotion_distance_report_x/promotion_distance_report.json"
        ),
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
        "reserve_substitution_manual_review_candidate_count": 4,
        "reserve_substitution_manual_review_ready_candidate_count": 4,
        "reserve_substitution_manual_review_blocked_candidate_count": 0,
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
        "reserve_substitution_policy_impact_status": (
            "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
        ),
        "reserve_substitution_policy_impact_candidate_count": 4,
        "reserve_substitution_policy_impact_ready_candidate_count": 4,
        "reserve_substitution_policy_impact_mapping_pair_count": 5,
        "reserve_substitution_policy_impact_potential_runner_rows_blocked": 32,
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count": 4,
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids": [
            "Race 3 - TAREE - 2026-06-13",
            "Race 4 - TAREE - 2026-06-13",
            "Race 7 - TAREE - 2026-06-13",
            "Race 8 - TAREE - 2026-06-13",
        ],
        "reserve_substitution_policy_impact_dataset_join_allowed": False,
        "reserve_substitution_policy_impact_official_result_acceptance_allowed": False,
        "reserve_substitution_policy_impact_db_write": False,
        "reserve_substitution_policy_impact_blockers": [],
        "reserve_substitution_policy_impact_report": (
            "artifacts/full_evidence_orchestration_20260525/"
            "official_result_reserve_substitution_preflight_x/"
            "reserve_substitution_policy_impact_preview.json"
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
    }
    dashboard_unified_status = {
        "unified_evidence_eligible_rows": 3,
    }
    dashboard = autopilot.build_dashboard(
        generated_at=datetime.fromisoformat("2026-06-10T18:30:00+10:00"),
        aggregate_metrics={"safe_joined_race_count": 9},
        join_metrics=None,
        aggregate_calibration={},
        aggregate_box_bias={},
        status_report={"final_status": "CONTINUE_FORWARD_SHADOW_COLLECTION"},
        sources={},
        unified_evidence_dataset_status=dashboard_unified_status,
        backlog_unified_evidence_status=backlog_status,
        rolling_model_comparison_status={
            "status": "ROLLING_MODEL_COMPARISON_COLLECTING",
            "output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "rolling_model_comparison_x"
            ),
            "attempted": True,
            "sample_scope": "unified",
            "dedupe_race_id": True,
            "sample_race_count": 8,
            "sample_runner_rows": 57,
            "minimum_races_for_review": 100,
            "best_candidate_key": "market_only_implied",
            "best_candidate_top1": 0.375,
            "best_candidate_top3": 0.875,
            "source_rejected_live_odds_candidate_count": 5,
            "source_rows_with_rejected_live_odds_candidates": 4,
            "source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 3,
                "odds_source_url_missing": 2,
            },
            "blockers": ["comparison_race_count_below_review_floor"],
        },
        high_accuracy_refinement_status=high_accuracy_status,
    )
    odds_snapshot_status = {
        "status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
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
    }
    unified_status = {
        "status": "UNIFIED_EVIDENCE_DATASET_BUILT",
        "unified_evidence_eligible_rows": 3,
        "artifact_odds_rows_seen": 6,
        "artifact_odds_rows_accepted": 2,
        "artifact_odds_rows_rejected": 4,
        "artifact_odds_rejection_reason_counts": {
            "odds_match_status_not_valid_pre_jump_dog_odds": 4,
        },
        "rejected_live_odds_candidate_count": 2,
        "rows_with_rejected_live_odds_candidates": 1,
        "rejected_live_odds_candidate_reason_counts": {
            "odds_source_url_missing": 2,
        },
            "official_result_coverage": {
                "source": "unified_evidence_dataset",
                "requested_race_count": 7,
                "requested_race_count_source": (
                    "official_result_evidence_db_audit_requested_race_ids"
                ),
                "races_with_rows_count": 2,
            "missing_race_count": 5,
            "missing_exclusion_count": 12,
            "runner_path_count": 1,
            "runner_paths_source_field": "official_result_runner_paths",
        },
        "official_result_coverage_requested_race_count": 7,
        "official_result_coverage_races_with_rows_count": 2,
        "official_result_coverage_missing_race_count": 5,
        "official_result_coverage_missing_exclusion_count": 12,
        "official_result_runner_path_count": 1,
        "official_result_runner_paths_source_field": "official_result_runner_paths",
    }
    daily_status = autopilot.build_daily_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:30:00+10:00"),
        daily_manifest={"race_count": 1, "prediction_rows": 8},
        result_join_status={"latest_join": {"joined_count": 0}},
        dashboard=dashboard,
        timeseries=[],
        readiness={"decision": "NEED_MORE_RESULTS"},
        odds_snapshot_status=odds_snapshot_status,
        unified_evidence_dataset_status=unified_status,
        backlog_unified_evidence_status=backlog_status,
        rolling_model_comparison_status={
            "status": "ROLLING_MODEL_COMPARISON_COLLECTING",
            "sample_race_count": 8,
            "sample_runner_rows": 57,
            "minimum_races_for_review": 100,
            "best_candidate_key": "market_only_implied",
            "best_candidate_top1": 0.375,
            "best_candidate_top3": 0.875,
            "source_rejected_live_odds_candidate_count": 5,
            "source_rows_with_rejected_live_odds_candidates": 4,
            "source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 3,
                "odds_source_url_missing": 2,
            },
            "blockers": ["comparison_race_count_below_review_floor"],
        },
        high_accuracy_refinement_status=high_accuracy_status,
    )
    daily_status.update(
        {
            "timing_aligned_prediction_rerun_execution_status": (
                "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
            ),
            "timing_aligned_prediction_rerun_execution_hard_stops": [
                "timing_aligned_rerun_window_already_closed_after_jump"
            ],
            "timing_aligned_prediction_rerun_execution_performed": False,
            "timing_aligned_prediction_rerun_output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_shadow_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x_timing_aligned_rerun"
            ),
            "timing_aligned_prediction_rerun_odds_snapshot_status": None,
        }
    )

    assert (
        dashboard["backlog_unified_evidence_datasets"]["status"]
        == "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT"
    )
    assert dashboard["backlog_unified_evidence_datasets"]["dataset_count"] == 1
    assert daily_status["backlog_unified_evidence_dataset_count"] == 1
    assert daily_status["odds_research_next_action"] == (
        "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
    )
    assert daily_status["timing_aligned_prediction_rerun_required"] is True
    assert daily_status["timing_aligned_prediction_rerun_race_count"] == 2
    assert daily_status["timing_aligned_prediction_rerun_race_ids"] == [
        "Race 10 - CANN - 2026-06-13",
        "Race 8 - CANN - 2026-06-13",
    ]
    assert daily_status["timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert daily_status["timing_aligned_rerun_execution_status"].endswith(
        "timing_aligned_prediction_rerun_execution_status.json"
    )
    assert daily_status["backlog_unified_evidence_official_result_rows"] == 16
    assert daily_status["backlog_unified_evidence_eligible_rows"] == 8
    assert daily_status["unified_evidence_eligible_rows"] == 3
    assert daily_status["current_cycle_unified_evidence_eligible_rows"] == 3
    assert daily_status["backlog_unified_evidence_eligible_rows_scope"] == (
        "backlog_unified_evidence_datasets"
    )
    assert daily_status["high_accuracy_unified_evidence_eligible_rows"] == 11
    assert daily_status["high_accuracy_unified_evidence_eligible_rows_scope"] == (
        "high_accuracy_refinement_packet"
    )
    assert daily_status["max_observed_unified_evidence_eligible_rows"] == 11
    assert daily_status["unified_evidence_eligible_rows_scope"] == (
        "current_cycle_unified_evidence_dataset"
    )
    assert dashboard["unified_evidence_growth"] == {
        "current_cycle_unified_evidence_eligible_rows": 3,
        "backlog_unified_evidence_eligible_rows": 8,
        "high_accuracy_unified_evidence_eligible_rows": 11,
        "max_observed_unified_evidence_eligible_rows": 11,
        "existing_unified_evidence_eligible_rows_scope": (
            "current_cycle_unified_evidence_dataset"
        ),
    }
    assert daily_status["unified_evidence_artifact_odds_rows_seen"] == 6
    assert daily_status["unified_evidence_artifact_odds_rows_accepted"] == 2
    assert daily_status["unified_evidence_artifact_odds_rows_rejected"] == 4
    assert daily_status[
        "unified_evidence_artifact_odds_rejection_reason_counts"
    ] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 4,
    }
    assert daily_status["backlog_unified_evidence_artifact_odds_rows_seen"] == 16
    assert daily_status["backlog_unified_evidence_artifact_odds_rows_accepted"] == 3
    assert daily_status["backlog_unified_evidence_artifact_odds_rows_rejected"] == 13
    assert daily_status[
        "backlog_unified_evidence_artifact_odds_rejection_reason_counts"
    ] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 13,
    }
    assert daily_status["backlog_unified_deduped_race_count"] == 2
    assert daily_status["backlog_unified_deduped_races_with_evidence"] == 1
    assert daily_status["backlog_unified_deduped_races_without_evidence"] == 1
    assert dashboard["backlog_unified_evidence_datasets"][
        "exclusion_reason_counts"
    ] == {"strict_prejump_odds_missing": 8}
    assert dashboard["backlog_unified_evidence_datasets"][
        "official_result_coverage_requested_race_count"
    ] == 2
    assert dashboard["backlog_unified_evidence_datasets"][
        "official_result_coverage_requested_race_count_source"
    ] == (
        "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
    )
    assert daily_status[
        "backlog_unified_official_result_coverage_requested_race_count"
    ] == 2
    assert daily_status[
        "backlog_unified_official_result_coverage_requested_race_count_source"
    ] == (
        "deduped_backlog_unified_evidence_official_result_coverage_requested_race_ids"
    )
    assert daily_status[
        "backlog_unified_official_result_coverage_races_with_rows_count"
    ] == 1
    assert daily_status[
        "backlog_unified_official_result_coverage_missing_race_count"
    ] == 1
    assert daily_status[
        "backlog_unified_official_result_coverage_missing_exclusion_count"
    ] == 8
    assert dashboard["backlog_unified_evidence_datasets"]["race_coverage"][
        "deduped_races_without_complete_official_result_instance"
    ] == 1
    assert dashboard["backlog_unified_evidence_datasets"]["gap_action_plan"][
        "sample_blocking_gap_count"
    ] == 1
    assert daily_status["backlog_unified_sample_blocking_gap_count"] == 1
    assert daily_status["backlog_unified_top_gap_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert daily_status["backlog_unified_top_gap_races"] == [
        {
            "race_id": "Race 2 - BEN - 2026-06-10",
            "race_date": "2026-06-10",
            "venue": "BEN",
            "recommended_action": "retry_official_result_capture_or_join",
            "evidence_missing_reason": "official_result_missing_only",
            "has_unified_evidence_instance": False,
            "has_complete_official_result_instance": False,
            "has_complete_strict_prejump_odds_instance": True,
        }
    ]
    assert daily_status["backlog_unified_top_official_result_missing_race_ids"] == [
        "Race 2 - BEN - 2026-06-10"
    ]
    assert daily_status["backlog_unified_top_official_result_missing_races"] == [
        {
            "race_id": "Race 2 - BEN - 2026-06-10",
            "race_date": "2026-06-10",
            "venue": "BEN",
        }
    ]
    assert daily_status["backlog_unified_gap_action_counts"] == {
        "collect_future_strict_prejump_odds": 0,
        "investigate_join_or_stage2_gap": 0,
        "monitor_non_sample_blocking_completion_gap": 0,
        "retry_official_result_capture_or_join": 1,
    }
    assert daily_status["backlog_unified_gap_evidence_missing_reason_counts"] == {
        "join_or_stage2_gap": 0,
        "official_result_missing_only": 1,
        "strict_prejump_odds_missing": 0,
    }
    assert daily_status[
        "backlog_unified_evidence_odds_exclusion_reason_counts"
    ] == {"odds_source_url_missing": 2}
    assert daily_status["unified_rejected_live_odds_candidate_count"] == 2
    assert daily_status["unified_rows_with_rejected_live_odds_candidates"] == 1
    assert daily_status["unified_rejected_live_odds_candidate_reason_counts"] == {
        "odds_source_url_missing": 2,
    }
    assert (
        daily_status["unified_evidence_official_result_coverage_requested_race_count"]
        == 7
    )
    assert (
        daily_status[
            "unified_evidence_official_result_coverage_requested_race_count_source"
        ]
        == "official_result_evidence_db_audit_requested_race_ids"
    )
    assert (
        daily_status[
            "unified_evidence_official_result_coverage_races_with_rows_count"
        ]
        == 2
    )
    assert (
        daily_status["unified_evidence_official_result_coverage_missing_race_count"]
        == 5
    )
    assert (
        daily_status[
            "unified_evidence_official_result_coverage_missing_exclusion_count"
        ]
        == 12
    )
    assert daily_status["unified_evidence_official_result_runner_path_count"] == 1
    assert (
        daily_status["unified_evidence_official_result_runner_paths_source_field"]
        == "official_result_runner_paths"
    )
    assert daily_status["backlog_unified_rejected_live_odds_candidate_count"] == 5
    assert daily_status[
        "backlog_unified_rows_with_rejected_live_odds_candidates"
    ] == 4
    assert daily_status[
        "backlog_unified_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 2,
    }
    assert dashboard["backlog_unified_evidence_datasets"][
        "rejected_live_odds_candidate_count"
    ] == 5
    assert dashboard["backlog_unified_evidence_datasets"][
        "rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 2,
    }
    assert dashboard["rolling_model_comparison"]["sample_race_count"] == 8
    assert dashboard["rolling_model_comparison"]["best_candidate_key"] == "market_only_implied"
    assert dashboard["rolling_model_comparison"][
        "source_rejected_live_odds_candidate_count"
    ] == 5
    assert dashboard["rolling_model_comparison"][
        "source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 2,
    }
    assert dashboard["promotion_distance"]["status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert dashboard["promotion_distance"]["sample_race_count"] == 124
    assert dashboard["promotion_distance"][
        "source_rejected_live_odds_candidate_count"
    ] == 5
    assert dashboard["promotion_distance"][
        "source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 2,
    }
    assert dashboard["promotion_distance"]["source_exclusion_reason_counts"] == {
        "official_result_missing": 32,
    }
    assert dashboard["promotion_distance"][
        "source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 7 - TAREE - 2026-06-13"]
    assert dashboard["promotion_distance"]["source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_x/official_result_runners.jsonl",
    ]
    assert dashboard["promotion_distance"][
        "official_result_coverage_requested_race_count"
    ] == 7
    assert dashboard["promotion_distance"][
        "official_result_coverage_requested_race_count_source"
    ] == "deduped_requested_or_inferred_race_ids"
    assert dashboard["promotion_distance"][
        "official_result_coverage_legacy_requested_race_count_without_ids"
    ] == 4125
    assert dashboard["promotion_distance"][
        "official_result_coverage_races_with_rows_count"
    ] == 1
    assert dashboard["promotion_distance"][
        "official_result_coverage_missing_race_count"
    ] == 1
    assert dashboard["promotion_distance"][
        "official_result_coverage_missing_exclusion_count"
    ] == 32
    assert dashboard["promotion_distance"]["official_result_runner_path_count"] == 1
    assert dashboard["promotion_distance"][
        "official_result_runner_paths_source_field"
    ] == "rolling_sample.source_official_result_runner_paths"
    assert dashboard["promotion_distance"]["best_non_market_candidate_key"] == "stage2_rf_calibrated"
    assert daily_status["rolling_model_comparison_sample_races"] == 8
    assert daily_status["rolling_model_comparison_best_candidate"] == "market_only_implied"
    assert daily_status[
        "rolling_model_comparison_source_rejected_live_odds_candidate_count"
    ] == 5
    assert daily_status[
        "rolling_model_comparison_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 2,
    }
    assert daily_status["promotion_distance_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert daily_status["promotion_distance_sample_race_count"] == 124
    assert daily_status[
        "promotion_distance_source_rejected_live_odds_candidate_count"
    ] == 5
    assert daily_status[
        "promotion_distance_source_rejected_live_odds_candidate_reason_counts"
    ] == {
        "odds_decimal_invalid": 3,
        "odds_source_url_missing": 2,
    }
    assert daily_status["promotion_distance_source_exclusion_reason_counts"] == {
        "official_result_missing": 32,
    }
    assert daily_status[
        "promotion_distance_source_official_result_evidence_db_missing_race_ids"
    ] == ["Race 7 - TAREE - 2026-06-13"]
    assert daily_status["promotion_distance_source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_x/official_result_runners.jsonl",
    ]
    assert (
        daily_status[
            "promotion_distance_official_result_coverage_requested_race_count"
        ]
        == 7
    )
    assert (
        daily_status[
            "promotion_distance_official_result_coverage_requested_race_count_source"
        ]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        daily_status[
            "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
        ]
        == 4125
    )
    assert (
        daily_status[
            "promotion_distance_official_result_coverage_races_with_rows_count"
        ]
        == 1
    )
    assert (
        daily_status["promotion_distance_official_result_coverage_missing_race_count"]
        == 1
    )
    assert (
        daily_status[
            "promotion_distance_official_result_coverage_missing_exclusion_count"
        ]
        == 32
    )
    assert daily_status["promotion_distance_official_result_runner_path_count"] == 1
    assert daily_status[
        "promotion_distance_official_result_runner_paths_source_field"
    ] == "rolling_sample.source_official_result_runner_paths"
    assert daily_status["promotion_distance_best_non_market_candidate_key"] == "stage2_rf_calibrated"
    assert dashboard["reserve_substitution_preflight"]["status"] == (
        "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
    )
    assert dashboard["reserve_substitution_preflight"][
        "ready_for_policy_review_count"
    ] == 4
    assert dashboard["reserve_substitution_manual_review"]["status"] == (
        "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    )
    assert dashboard["reserve_substitution_manual_review"]["ready_candidate_count"] == 4
    assert dashboard["reserve_substitution_manual_review"]["mapping_pair_count"] == 5
    assert (
        dashboard["reserve_substitution_manual_review"]["dataset_join_allowed"]
        is False
    )
    assert dashboard["reserve_substitution_policy_impact"]["status"] == (
        "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    )
    assert dashboard["reserve_substitution_policy_impact"]["ready_candidate_count"] == 4
    assert dashboard["reserve_substitution_policy_impact"]["mapping_pair_count"] == 5
    assert (
        dashboard["reserve_substitution_policy_impact"][
            "potential_runner_rows_blocked"
        ]
        == 32
    )
    assert (
        dashboard["reserve_substitution_policy_impact"][
            "dataset_join_allowed"
        ]
        is False
    )
    assert daily_status["reserve_substitution_preflight_status"] == (
        "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
    )
    assert daily_status[
        "reserve_substitution_preflight_dataset_join_blocker_counts"
    ] == {
        "manual_policy_review_required_before_join": 4,
        "official_result_remains_quarantined": 4,
    }
    assert daily_status["reserve_substitution_manual_review_status"] == (
        "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    )
    assert daily_status[
        "reserve_substitution_manual_review_ready_candidate_count"
    ] == 4
    assert daily_status["reserve_substitution_manual_review_mapping_pair_count"] == 5
    assert (
        daily_status["reserve_substitution_manual_review_dataset_join_allowed"]
        is False
    )
    assert daily_status["reserve_substitution_policy_impact_status"] == (
        "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    )
    assert daily_status["reserve_substitution_policy_impact_ready_candidate_count"] == 4
    assert daily_status["reserve_substitution_policy_impact_mapping_pair_count"] == 5
    assert (
        daily_status[
            "reserve_substitution_policy_impact_potential_runner_rows_blocked"
        ]
        == 32
    )
    assert (
        daily_status[
            "reserve_substitution_policy_impact_dataset_join_allowed"
        ]
        is False
    )
    assert "Backlog unified full-evidence rows: `8`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Unified full-evidence rows scope: "
        "`current_cycle_unified_evidence_dataset`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Current-cycle unified full-evidence rows: `3`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Max observed unified full-evidence rows: `11`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Backlog unified full-evidence rows scope: "
        "`backlog_unified_evidence_datasets`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Unified artifact odds rows seen: `6`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Unified artifact odds rows accepted: `2`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Unified artifact odds rows rejected: `4`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Unified artifact odds rejection reasons: "
        "`{'odds_match_status_not_valid_pre_jump_dog_odds': 4}`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Unified official-result coverage requested races: `7`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Unified official-result coverage races with rows: `2`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Unified official-result coverage missing races: `5`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Unified official-result missing exclusions: `12`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Unified official-result runner path count: `1`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Unified official-result runner paths source: "
        "`official_result_runner_paths`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Backlog unified artifact odds rows seen: `16`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified artifact odds rows accepted: `3`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified artifact odds rows rejected: `13`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Backlog unified artifact odds rejection reasons: "
        "`{'odds_match_status_not_valid_pre_jump_dog_odds': 13}`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Unified rows with rejected live odds candidates: `1`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified odds exclusion reasons: `{'odds_source_url_missing': 2}`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified rejected live odds candidates: `5`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified rows with rejected live odds candidates: `4`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified rejected live odds candidate reasons: `{'odds_decimal_invalid': 3, 'odds_source_url_missing': 2}`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified deduped races with evidence: `1`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified sample-blocking gap races: `1`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Backlog unified evidence-missing reasons: `{'join_or_stage2_gap': 0, 'official_result_missing_only': 1, 'strict_prejump_odds_missing': 0}`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Backlog unified top gap race IDs: "
        "`['Race 2 - BEN - 2026-06-10']`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert (
        "Backlog unified top gap races: "
        "`[{'race_id': 'Race 2 - BEN - 2026-06-10'"
    ) in autopilot.daily_status_markdown(daily_status)
    assert (
        "Backlog unified top official-result-missing race IDs: "
        "`['Race 2 - BEN - 2026-06-10']`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert (
        "Backlog unified top official-result-missing races: "
        "`[{'race_id': 'Race 2 - BEN - 2026-06-10'"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Rolling comparison best candidate: `market_only_implied`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Reserve substitution manual review: `RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Reserve substitution manual review dataset join allowed: `False`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Reserve substitution policy impact: `RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Reserve substitution policy impact potential runner rows blocked: `32`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Reserve substitution policy impact dataset join allowed: `False`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Rolling comparison source rejected live odds candidates: `5`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Rolling comparison source rows with rejected live odds candidates: `4`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Rolling comparison source rejected live odds candidate reasons: `{'odds_decimal_invalid': 3, 'odds_source_url_missing': 2}`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Promotion distance: `PROMOTION_DISTANCE_BLOCKED`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Promotion distance sample races: `124`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Promotion distance source rejected live odds candidates: `5`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Promotion distance source rejected live odds candidate reasons: `{'odds_decimal_invalid': 3, 'odds_source_url_missing': 2}`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "High-accuracy unified full-evidence rows: `11`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "High-accuracy unified full-evidence rows scope: "
        "`high_accuracy_refinement_packet`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert (
        "High-accuracy timing-aligned rerun plan: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert (
        "High-accuracy timing-aligned rerun execution status: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/"
        "timing_aligned_prediction_rerun_execution_status.json`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Odds research next action: `RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Timing-aligned prediction rerun required: `True`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert "Timing-aligned prediction rerun races: `2`" in autopilot.daily_status_markdown(
        daily_status
    )
    assert (
        "Timing-aligned prediction rerun execution hard stops: "
        "`['timing_aligned_rerun_window_already_closed_after_jump']`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert (
        "Timing-aligned prediction rerun odds snapshot dir: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_odds_snapshot_x_timing_aligned_rerun`"
    ) in autopilot.daily_status_markdown(daily_status)
    daily_summary = autopilot.daily_status_markdown(daily_status)
    assert (
        "Unified official-result requested race count source: "
        "`official_result_evidence_db_audit_requested_race_ids`"
    ) in daily_summary
    assert "Promotion distance official-result coverage requested races: `7`" in daily_summary
    assert (
        "Promotion distance official-result requested race count source: "
        "`deduped_requested_or_inferred_race_ids`"
    ) in daily_summary
    assert (
        "Promotion distance official-result legacy requested race count without IDs: "
        "`4125`"
    ) in daily_summary
    assert "Promotion distance official-result coverage races with rows: `1`" in daily_summary
    assert "Promotion distance official-result coverage missing races: `1`" in daily_summary
    assert "Promotion distance official-result missing exclusions: `32`" in daily_summary
    assert "Promotion distance official-result runner path count: `1`" in daily_summary
    assert (
        "Promotion distance official-result runner paths source: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in daily_summary
    assert "Promotion distance source official-result runner paths:" not in daily_summary
    summary = autopilot.summary_markdown(
        final_verdict="AUTOPILOT_READY",
        dashboard=dashboard,
        readiness={"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []},
        result_join_status={"latest_join": {"joined_count": 0}, "cumulative": {"joined_count": 9}},
    )
    assert "Backlog Unified Evidence Datasets" in summary
    assert "Artifact odds rows seen: `16`" in summary
    assert "Artifact odds rows accepted: `3`" in summary
    assert "Artifact odds rows rejected: `13`" in summary
    assert (
        "Artifact odds rejection reasons: "
        "`{'odds_match_status_not_valid_pre_jump_dog_odds': 13}`"
    ) in summary
    assert "Sample-blocking gap races: `1`" in summary
    assert (
        "Top gap race IDs: `['Race 2 - BEN - 2026-06-10']`"
    ) in summary
    assert (
        "Top gap races: `[{'race_id': 'Race 2 - BEN - 2026-06-10'"
    ) in summary
    assert (
        "Top official-result-missing race IDs: "
        "`['Race 2 - BEN - 2026-06-10']`"
    ) in summary
    assert (
        "Top official-result-missing races: "
        "`[{'race_id': 'Race 2 - BEN - 2026-06-10'"
    ) in summary
    assert "Rolling Model Comparison" in summary
    assert "Promotion Distance" in summary
    assert summary.count("Source rejected live odds candidates: `5`") == 2
    assert summary.count("Source rows with rejected live odds candidates: `4`") == 2
    assert (
        summary.count(
            "Source rejected live odds candidate reasons: `{'odds_decimal_invalid': 3, 'odds_source_url_missing': 2}`"
        )
        == 2
    )
    assert "Official-result coverage requested races: `7`" in summary
    assert "Official-result coverage races with rows: `1`" in summary
    assert "Official-result coverage missing races: `1`" in summary
    assert "Official-result missing exclusions: `32`" in summary
    assert "Official-result runner path count: `1`" in summary
    assert (
        "Official-result runner paths source: "
        "`rolling_sample.source_official_result_runner_paths`"
    ) in summary
    assert "Source official-result runner paths:" not in summary
    assert "## High-Accuracy Timing Sources" in summary
    assert "## Reserve Substitution Manual Review" in summary
    assert "Dataset join allowed: `False`" in summary
    assert "Packet status: `BLOCKED_KEEP_BASELINE`" in summary
    assert (
        "Timing-aligned rerun plan: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json`"
    ) in summary
    assert (
        "Timing-aligned rerun execution status: "
        "`artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/"
        "timing_aligned_prediction_rerun_execution_status.json`"
    ) in summary


def test_daily_status_blocks_missing_prediction_sample_odds():
    daily_status = autopilot.build_daily_status(
        generated_at=datetime.fromisoformat("2026-06-24T14:32:00+10:00"),
        daily_manifest={"race_count": 13, "prediction_rows": 97},
        result_join_status={"latest_join": {"joined_count": 0}},
        dashboard={},
        timeseries=[],
        readiness={"decision": "READY_FOR_RELIABILITY_REVIEW"},
        odds_snapshot_status={
            "status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "races_with_complete_valid_prejump_odds": 5,
            "races_with_missing_odds_rows": 8,
            "race_coverage_path": "shadow_odds_snapshot_x/race_coverage.json",
        },
        autonomous_live_odds_capture_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "ready_count": 2,
        },
    )

    assert daily_status["prediction_sample_odds_coverage_status"] == (
        "BLOCKED_MISSING_PREJUMP_ODDS"
    )
    assert daily_status["prediction_sample_odds_coverage_blocker"] == (
        "prediction_sample_missing_complete_valid_prejump_odds"
    )
    assert daily_status["prediction_sample_odds_expected_races"] == 13
    assert daily_status["prediction_sample_odds_complete_prejump_races"] == 5
    assert daily_status["prediction_sample_odds_missing_prejump_races"] == 8
    assert daily_status["prediction_sample_odds_coverage_report"] == (
        "shadow_odds_snapshot_x/race_coverage.json"
    )
    assert daily_status["autonomous_live_odds_capture_scope_status"] == (
        "PARTIAL_AUTONOMOUS_ODDS_SCOPE"
    )
    assert daily_status["autonomous_live_odds_capture_scope_gap_races"] == 11


def test_high_accuracy_refinement_packet_command_and_status_are_report_only(tmp_path):
    odds_augmented_report = tmp_path / "rolling_model_comparison_report.json"
    odds_augmented_report.write_text("{}", encoding="utf-8")
    odds_gate_report = tmp_path / "odds_research_gate_report.json"
    odds_gate_report.write_text("{}", encoding="utf-8")
    promotion_distance_report = tmp_path / "promotion_distance_report.json"
    promotion_distance_report.write_text("{}", encoding="utf-8")
    reserve_substitution_preflight = (
        tmp_path / "official_result_reserve_substitution_preflight.json"
    )
    reserve_substitution_preflight.write_text("{}", encoding="utf-8")
    timing_aligned_rerun_plan = tmp_path / "timing_aligned_prediction_rerun_plan.json"
    timing_aligned_rerun_plan.write_text("{}", encoding="utf-8")
    timing_aligned_rerun_execution_status = (
        tmp_path / "timing_aligned_prediction_rerun_execution_status.json"
    )
    timing_aligned_rerun_execution_status.write_text("{}", encoding="utf-8")
    stage2_predictions = tmp_path / "stage2_shadow_predictions.jsonl"
    stage2_predictions.write_text("{}\n", encoding="utf-8")
    command = autopilot.high_accuracy_refinement_packet_command(
        unified_evidence_report=Path("unified_evidence_dataset_report.json"),
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "high_accuracy_refinement_packet_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        stage2_predictions=stage2_predictions,
        odds_augmented_report=odds_augmented_report,
        odds_gate_report=odds_gate_report,
        promotion_distance_report=promotion_distance_report,
        reserve_substitution_preflight=reserve_substitution_preflight,
        timing_aligned_rerun_plan=timing_aligned_rerun_plan,
        timing_aligned_rerun_execution_status=timing_aligned_rerun_execution_status,
    )

    assert "scripts/build_high_accuracy_refinement_packet.py" in command[1]
    assert "--unified-evidence-report" in command
    assert "--stage2-predictions" in command
    assert str(stage2_predictions) in command
    assert "--odds-augmented-report" in command
    assert "--odds-gate-report" in command
    assert str(odds_gate_report) in command
    assert "--promotion-distance-report" in command
    assert str(promotion_distance_report) in command
    assert "--reserve-substitution-preflight" in command
    assert str(reserve_substitution_preflight) in command
    assert "--timing-aligned-rerun-plan" in command
    assert str(timing_aligned_rerun_plan) in command
    assert "--timing-aligned-rerun-execution-status" in command
    assert str(timing_aligned_rerun_execution_status) in command
    assert "--output-dir" in command
    assert "--write-labels-approved" not in command
    assert "--execute" not in command

    status = autopilot.build_high_accuracy_refinement_status(
        generated_at=datetime.fromisoformat("2026-06-10T15:30:00+10:00"),
        packet_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "high_accuracy_refinement_packet_x"
        ),
        packet_report={
            "final_status": "BLOCKED_KEEP_BASELINE",
            "promotion_pr_gate": {
                "status": "BLOCKED",
                "blockers": ["no_candidate_passed_rank_first_accuracy_gate"],
            },
            "unified_evidence_summary": {
                "status": "UNIFIED_EVIDENCE_COLLECTING",
                "unified_evidence_eligible_rows": 0,
                "minimum_eligible_rows_for_review": 100,
            },
            "odds_research_gate_summary": {
                "status": "ODDS_RESEARCH_BLOCKED_PROVENANCE",
                "complete_valid_prejump_odds_races": 2,
            },
            "stages": {
                "non_tgr_model_challenger": {
                    "status": "STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING",
                    "source_status": "PREDICTIONS_COLLECTED_JOINED_METRICS_MISSING",
                    "stage2_prediction_rows": 114,
                    "stage2_predictions_path": (
                        "artifacts/full_evidence_orchestration_20260525/"
                        "daily_race_ingest_shadow_x/stage2_shadow_predictions.jsonl"
                    ),
                    "gate": {
                        "status": "BLOCKED",
                        "blockers": ["stage2_forward_joined_metrics_missing"],
                    },
                },
                "odds_augmented_model": {
                    "status": "ODDS_AUGMENTED_MODEL_BLOCKED",
                    "source_final_status": "ROLLING_MODEL_COMPARISON_COLLECTING",
                    "cumulative_odds_evidence": {
                        "status": "ROLLING_MODEL_COMPARISON_COLLECTING",
                        "sample_scope": "unified",
                        "sample_race_count": 92,
                        "minimum_complete_valid_prejump_odds_races": 100,
                        "sample_floor_met": False,
                        "races_needed_for_review": 8,
                        "ready": False,
                    },
                    "rolling_model_comparison": {
                        "status": "ROLLING_MODEL_COMPARISON_COLLECTING",
                        "sample_scope": "unified",
                        "sample_race_count": 92,
                        "minimum_races_for_review": 100,
                        "sample_floor_met": False,
                        "races_needed_for_review": 8,
                        "candidate_count": 22,
                        "best_candidate_key": "market_only_implied",
                        "best_non_baseline_candidate_key": "market_only_implied",
                        "rank_first_sort": [
                            "market_only_implied",
                            "stage2_uncalibrated_market_blend_75",
                        ],
                    },
                    "gate": {
                        "status": "BLOCKED",
                        "blockers": [
                            "cumulative_odds_evidence_races_below_min",
                        ],
                    },
                }
            },
            "source_artifacts": {
                "odds_research_gate_report": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "shadow_odds_snapshot_x/odds_research_gate_report.json"
                ),
                "promotion_distance_report": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "promotion_distance_report_x/promotion_distance_report.json"
                ),
                "reserve_substitution_preflight": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "reserve_substitution_preflight_x/"
                    "official_result_reserve_substitution_preflight.json"
                ),
                "reserve_substitution_manual_review": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "reserve_substitution_preflight_x/"
                    "reserve_substitution_manual_review_packet.json"
                ),
                "reserve_substitution_policy_impact_preview": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "reserve_substitution_preflight_x/"
                    "reserve_substitution_policy_impact_preview.json"
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
            },
            "promotion_distance_summary": {
                "status": "PROMOTION_DISTANCE_BLOCKED",
                "promotion_ready": False,
                "blockers": ["best_non_market_top1_margin_below_target"],
                "sample_race_count": 124,
                "sample_runner_rows": 856,
                "source_rejected_live_odds_candidate_count": 5,
                "source_rows_with_rejected_live_odds_candidates": 4,
                "source_rejected_live_odds_candidate_reason_counts": {
                    "odds_decimal_invalid": 2,
                    "odds_source_url_missing": 3,
                },
                "source_exclusion_reason_counts": {"official_result_missing": 32},
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
                "best_candidate_key": "market_only_implied",
                "best_non_market_candidate_key": (
                    "stage2_uncalibrated_market_blend_50"
                ),
                "best_non_market_top1_margin_gap": 0.02,
                "predeclared_residual_candidate_status": (
                    "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING"
                ),
                "predeclared_residual_triggered_race_count": 2,
            },
            "reserve_substitution_preflight_summary": {
                "status": "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW",
                "candidate_count": 4,
                "ready_for_policy_review_count": 4,
                "blocked_candidate_count": 0,
                "readiness_blocker_counts": {},
                "dataset_join_blocker_counts": {
                    "manual_policy_review_required_before_join": 4,
                    "official_result_remains_quarantined": 4,
                },
                "ready_race_ids": [
                    "Race 7 - TAREE - 2026-06-13",
                    "Race 8 - TAREE - 2026-06-13",
                ],
                "blocked_race_ids": [],
            },
            "reserve_substitution_manual_review_summary": {
                "status": "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY",
                "candidate_count": 4,
                "ready_candidate_count": 4,
                "blocked_candidate_count": 0,
                "mapping_pair_count": 5,
                "dataset_join_allowed": False,
                "official_result_acceptance_allowed": False,
                "db_write": False,
                "blockers": [],
                "ready_race_ids": [
                    "Race 7 - TAREE - 2026-06-13",
                    "Race 8 - TAREE - 2026-06-13",
                ],
            },
            "reserve_substitution_policy_impact_summary": {
                "status": "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY",
                "candidate_count": 4,
                "ready_candidate_count": 4,
                "mapping_pair_count": 5,
                "potential_official_result_runner_rows_blocked_by_policy": 32,
                "matched_backlog_top_gap_race_count": 2,
                "matched_backlog_top_gap_race_ids": [
                    "Race 7 - TAREE - 2026-06-13",
                    "Race 8 - TAREE - 2026-06-13",
                ],
                "dataset_join_allowed": False,
                "official_result_acceptance_allowed": False,
                "db_write": False,
                "blockers": [],
            },
            "protected_paths_unchanged": True,
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        attempted=True,
        returncode=0,
    )

    assert status["status"] == "BLOCKED_KEEP_BASELINE"
    assert status["promotion_pr_gate_status"] == "BLOCKED"
    assert (
        status["stage2_status"]
        == "STAGE2_PREDICTIONS_COLLECTED_METRICS_MISSING"
    )
    assert (
        status["stage2_source_status"]
        == "PREDICTIONS_COLLECTED_JOINED_METRICS_MISSING"
    )
    assert status["stage2_prediction_rows"] == 114
    assert status["stage2_predictions_path"].endswith("stage2_shadow_predictions.jsonl")
    assert status["stage2_gate_status"] == "BLOCKED"
    assert status["stage2_gate_blockers"] == ["stage2_forward_joined_metrics_missing"]
    assert status["unified_evidence_eligible_rows"] == 0
    assert status["minimum_eligible_rows_for_review"] == 100
    assert status["odds_research_gate_status"] == "ODDS_RESEARCH_BLOCKED_PROVENANCE"
    assert status["odds_research_gate_complete_valid_prejump_odds_races"] == 2
    assert status["odds_augmented_model_status"] == "ODDS_AUGMENTED_MODEL_BLOCKED"
    assert status["odds_augmented_gate_status"] == "BLOCKED"
    assert status["odds_augmented_gate_blockers"] == [
        "cumulative_odds_evidence_races_below_min"
    ]
    assert status["rolling_model_comparison_status"] == (
        "ROLLING_MODEL_COMPARISON_COLLECTING"
    )
    assert status["rolling_model_comparison_sample_race_count"] == 92
    assert status["rolling_model_comparison_minimum_races_for_review"] == 100
    assert status["rolling_model_comparison_sample_floor_met"] is False
    assert status["rolling_model_comparison_races_needed_for_review"] == 8
    assert status["rolling_model_comparison_candidate_count"] == 22
    assert status["rolling_model_comparison_best_candidate_key"] == (
        "market_only_implied"
    )
    assert status["rolling_model_comparison_rank_first_sort"] == [
        "market_only_implied",
        "stage2_uncalibrated_market_blend_75",
    ]
    assert status["promotion_distance_status"] == "PROMOTION_DISTANCE_BLOCKED"
    assert status["promotion_distance_promotion_ready"] is False
    assert status["promotion_distance_blockers"] == [
        "best_non_market_top1_margin_below_target"
    ]
    assert status["promotion_distance_sample_race_count"] == 124
    assert status["promotion_distance_sample_runner_rows"] == 856
    assert status["promotion_distance_source_rejected_live_odds_candidate_count"] == 5
    assert status["promotion_distance_source_rows_with_rejected_live_odds_candidates"] == 4
    assert status["promotion_distance_source_rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }
    assert status["promotion_distance_source_exclusion_reason_counts"] == {
        "official_result_missing": 32,
    }
    assert status["promotion_distance_source_odds_exclusion_reason_counts"] == {
        "strict_prejump_odds_missing": 6,
    }
    assert status[
        "promotion_distance_source_official_result_evidence_db_missing_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert (
        status[
            "promotion_distance_source_official_result_evidence_db_requested_race_count"
        ]
        == 7
    )
    assert status[
        "promotion_distance_source_official_result_evidence_db_races_with_rows"
    ] == ["Race 5 - TAREE - 2026-06-13"]
    assert status["promotion_distance_source_official_result_runner_paths"] == [
        "artifacts/full_evidence_orchestration_20260525/"
        "autonomous_official_result_capture_x/official_result_runners.jsonl",
    ]
    assert (
        status["promotion_distance_official_result_coverage_requested_race_count"]
        == 7
    )
    assert (
        status[
            "promotion_distance_official_result_coverage_requested_race_count_source"
        ]
        == "deduped_requested_or_inferred_race_ids"
    )
    assert (
        status[
            "promotion_distance_official_result_coverage_legacy_requested_race_count_without_ids"
        ]
        == 4125
    )
    assert (
        status["promotion_distance_official_result_coverage_races_with_rows_count"]
        == 1
    )
    assert status["promotion_distance_official_result_coverage_missing_race_count"] == 2
    assert (
        status[
            "promotion_distance_official_result_coverage_missing_exclusion_count"
        ]
        == 32
    )
    assert status["promotion_distance_official_result_runner_path_count"] == 1
    assert status["promotion_distance_official_result_runner_paths_source_field"] == (
        "rolling_sample.source_official_result_runner_paths"
    )
    assert status["promotion_distance_official_result_coverage"]["missing_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert status["promotion_distance_best_candidate_key"] == "market_only_implied"
    assert (
        status["promotion_distance_best_non_market_candidate_key"]
        == "stage2_uncalibrated_market_blend_50"
    )
    assert status["promotion_distance_best_non_market_top1_margin_gap"] == 0.02
    assert (
        status["promotion_distance_predeclared_residual_candidate_status"]
        == "PREDECLARED_RESIDUAL_CANDIDATE_COLLECTING"
    )
    assert status["promotion_distance_predeclared_residual_triggered_race_count"] == 2
    assert status["reserve_substitution_preflight_status"] == (
        "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
    )
    assert status[
        "reserve_substitution_preflight_ready_for_policy_review_count"
    ] == 4
    assert status["reserve_substitution_preflight_blocked_candidate_count"] == 0
    assert status["reserve_substitution_preflight_dataset_join_blocker_counts"] == {
        "manual_policy_review_required_before_join": 4,
        "official_result_remains_quarantined": 4,
    }
    assert status["reserve_substitution_preflight_ready_race_ids"] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert status["reserve_substitution_preflight_report"].endswith(
        "official_result_reserve_substitution_preflight.json"
    )
    assert status["reserve_substitution_manual_review_status"] == (
        "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    )
    assert status["reserve_substitution_manual_review_ready_candidate_count"] == 4
    assert status["reserve_substitution_manual_review_mapping_pair_count"] == 5
    assert status["reserve_substitution_manual_review_dataset_join_allowed"] is False
    assert (
        status[
            "reserve_substitution_manual_review_official_result_acceptance_allowed"
        ]
        is False
    )
    assert status["reserve_substitution_manual_review_db_write"] is False
    assert status["reserve_substitution_manual_review_report"].endswith(
        "reserve_substitution_manual_review_packet.json"
    )
    assert status["reserve_substitution_policy_impact_status"] == (
        "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
    )
    assert status["reserve_substitution_policy_impact_ready_candidate_count"] == 4
    assert status["reserve_substitution_policy_impact_mapping_pair_count"] == 5
    assert (
        status["reserve_substitution_policy_impact_potential_runner_rows_blocked"]
        == 32
    )
    assert (
        status[
            "reserve_substitution_policy_impact_matched_backlog_top_gap_race_count"
        ]
        == 2
    )
    assert status[
        "reserve_substitution_policy_impact_matched_backlog_top_gap_race_ids"
    ] == [
        "Race 7 - TAREE - 2026-06-13",
        "Race 8 - TAREE - 2026-06-13",
    ]
    assert status["reserve_substitution_policy_impact_dataset_join_allowed"] is False
    assert (
        status[
            "reserve_substitution_policy_impact_official_result_acceptance_allowed"
        ]
        is False
    )
    assert status["reserve_substitution_policy_impact_db_write"] is False
    assert status["reserve_substitution_policy_impact_report"].endswith(
        "reserve_substitution_policy_impact_preview.json"
    )
    assert status["odds_research_gate_report"].endswith("odds_research_gate_report.json")
    assert status["promotion_distance_report"].endswith("promotion_distance_report.json")
    assert status["timing_aligned_rerun_plan"].endswith(
        "timing_aligned_prediction_rerun_plan.json"
    )
    assert status["timing_aligned_rerun_execution_status"].endswith(
        "timing_aligned_prediction_rerun_execution_status.json"
    )
    assert status["no_write_guarantees"]["label_write"] is False


def test_high_accuracy_timing_source_verification_lines_include_artifact_paths():
    lines = autopilot.high_accuracy_timing_source_verification_lines(
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
        }
    )

    assert (
        "high_accuracy_timing_aligned_rerun_plan="
        "artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
    ) in lines
    assert (
        "high_accuracy_timing_aligned_rerun_execution_status="
        "artifacts/full_evidence_orchestration_20260525/"
        "shadow_autopilot_v1_x/"
        "timing_aligned_prediction_rerun_execution_status.json"
    ) in lines


def test_autopilot_promotion_distance_commands_are_report_only():
    pre_race_command = autopilot.pre_race_gated_challenger_command(
        runner_matrix_csv=Path("/evidence/rolling/market_residual_runner_matrix.csv"),
        output_dir=Path("/evidence/pre_race_gated_challenger_run"),
        evidence_root=Path("/evidence"),
    )
    promotion_distance_command = autopilot.promotion_distance_report_command(
        rolling_report=Path("/evidence/rolling/rolling_model_comparison_report.json"),
        pre_race_gated_report=Path(
            "/evidence/pre_race/pre_race_gated_challenger_report.json"
        ),
        high_accuracy_gate=Path("/evidence/high_accuracy/promotion_pr_gate.json"),
        output_dir=Path("/evidence/promotion_distance_report_run"),
        evidence_root=Path("/evidence"),
    )

    assert "scripts/build_pre_race_gated_challenger_packet.py" in pre_race_command[1]
    assert "--runner-matrix-csv" in pre_race_command
    assert "/evidence/rolling/market_residual_runner_matrix.csv" in pre_race_command
    assert "scripts/build_promotion_distance_report.py" in promotion_distance_command[1]
    assert "--rolling-report" in promotion_distance_command
    assert "--pre-race-gated-report" in promotion_distance_command
    assert "--high-accuracy-gate" in promotion_distance_command
    assert "--execute" not in pre_race_command
    assert "--execute" not in promotion_distance_command
    assert "--write-labels-approved" not in pre_race_command
    assert "--write-labels-approved" not in promotion_distance_command


def test_high_accuracy_refinement_packet_command_omits_missing_odds_gate_report(tmp_path):
    command = autopilot.high_accuracy_refinement_packet_command(
        unified_evidence_report=Path("unified_evidence_dataset_report.json"),
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "high_accuracy_refinement_packet_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        odds_gate_report=tmp_path / "missing_odds_research_gate_report.json",
    )

    assert "--odds-gate-report" not in command


def test_high_accuracy_refinement_packet_command_includes_backlog_unified_status(tmp_path):
    backlog_status = tmp_path / "backlog_unified_evidence_datasets_status.json"
    backlog_status.write_text("{}", encoding="utf-8")

    command = autopilot.high_accuracy_refinement_packet_command(
        unified_evidence_report=Path("unified_evidence_dataset_report.json"),
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "high_accuracy_refinement_packet_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
        backlog_unified_evidence_status=backlog_status,
    )

    assert "--backlog-unified-evidence-status" in command
    assert str(backlog_status) in command


def test_odds_research_gate_report_path_from_snapshot_status_requires_existing_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    odds_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/shadow_odds_x"
    odds_dir.mkdir(parents=True)
    gate_report = odds_dir / "odds_research_gate_report.json"
    gate_report.write_text("{}", encoding="utf-8")
    stale_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/shadow_odds_stale"
    stale_dir.mkdir(parents=True)
    stale_gate_report = stale_dir / "odds_research_gate_report.json"
    stale_gate_report.write_text("{}", encoding="utf-8")

    path = autopilot.odds_research_gate_report_path_from_snapshot_status(
        {"output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_odds_x"}
    )
    missing = autopilot.odds_research_gate_report_path_from_snapshot_status(
        {"output_dir": "artifacts/full_evidence_orchestration_20260525/missing_shadow_odds"}
    )
    stale = autopilot.odds_research_gate_report_path_from_snapshot_status(
        {
            "output_dir": "artifacts/full_evidence_orchestration_20260525/shadow_odds_x",
            "odds_research_gate_report_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_stale/odds_research_gate_report.json"
            ),
        }
    )

    assert path == gate_report
    assert missing is None
    assert stale is None


def test_rolling_model_comparison_command_and_status_are_report_only(tmp_path):
    report_a = tmp_path / "unified_a.json"
    report_b = tmp_path / "unified_b.json"
    report_a.write_text("{}", encoding="utf-8")
    report_b.write_text("{}", encoding="utf-8")

    command = autopilot.rolling_model_comparison_command(
        unified_evidence_reports=[report_a, report_b],
        output_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "rolling_model_comparison_x"
        ),
        evidence_root=Path("artifacts/full_evidence_orchestration_20260525"),
    )

    assert "scripts/build_rolling_model_comparison_packet.py" in command[1]
    assert "--evidence-root" in command
    assert command.count("--unified-evidence-report") == 2
    assert "--output-dir" in command
    assert "--execute" not in command
    assert "--write-labels-approved" not in command

    status = autopilot.build_rolling_model_comparison_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:50:00+10:00"),
        comparison_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "rolling_model_comparison_x"
        ),
        comparison_report={
            "final_status": "ROLLING_MODEL_COMPARISON_COLLECTING",
            "sample_scope": "unified",
            "dedupe_race_id": True,
            "sample_race_count": 8,
            "sample_runner_rows": 57,
            "minimum_races_for_review": 100,
            "sample_floor_met": False,
            "races_needed_for_review": 92,
            "candidate_count": 22,
            "best_candidate_key": "market_only_implied",
            "best_non_baseline_candidate_key": "market_only_implied",
            "rank_first_sort": [
                "market_only_implied",
                "stage2_uncalibrated_market_blend_75",
            ],
            "candidate_metrics": {
                "top1": 0.375,
                "top3": 0.875,
                "mean_winner_rank": 2.25,
            },
            "source_reports": [
                {
                    "source_index": 0,
                    "rejected_live_odds_candidate_count": 5,
                }
            ],
            "source_rejected_live_odds_candidate_count": 5,
            "source_rows_with_rejected_live_odds_candidates": 4,
            "source_rejected_live_odds_candidate_reason_counts": {
                "odds_decimal_invalid": 2,
                "odds_source_url_missing": 3,
            },
            "blockers": ["comparison_race_count_below_review_floor"],
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        attempted=True,
        returncode=0,
    )

    assert status["status"] == "ROLLING_MODEL_COMPARISON_COLLECTING"
    assert status["sample_race_count"] == 8
    assert status["sample_floor_met"] is False
    assert status["races_needed_for_review"] == 92
    assert status["candidate_count"] == 22
    assert status["best_candidate_key"] == "market_only_implied"
    assert status["rank_first_sort"] == [
        "market_only_implied",
        "stage2_uncalibrated_market_blend_75",
    ]
    assert status["best_candidate_top1"] == 0.375
    assert status["promotion_ready"] is False
    assert status["blockers"] == ["comparison_race_count_below_review_floor"]
    assert status["no_write_guarantees"]["label_write"] is False
    assert status["source_report_count"] == 1
    assert status["source_rejected_live_odds_candidate_count"] == 5
    assert status["source_rows_with_rejected_live_odds_candidates"] == 4
    assert status["source_rejected_live_odds_candidate_reason_counts"] == {
        "odds_decimal_invalid": 2,
        "odds_source_url_missing": 3,
    }

    malformed_status = autopilot.build_rolling_model_comparison_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:50:00+10:00"),
        comparison_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "rolling_model_comparison_x"
        ),
        comparison_report={
            "final_status": "ROLLING_MODEL_COMPARISON_COLLECTING",
            "sample_race_count": 8,
            "minimum_races_for_review": 100,
            "sample_floor_met": "false",
            "rank_first_sort": "market_only_implied",
        },
        attempted=True,
        returncode=0,
    )

    assert malformed_status["sample_floor_met"] is False
    assert malformed_status["rank_first_sort"] == []


def test_high_accuracy_refinement_status_preserves_current_and_backlog_unified_counts():
    status = autopilot.build_high_accuracy_refinement_status(
        generated_at=datetime.fromisoformat("2026-06-10T18:50:00+10:00"),
        packet_dir=Path(
            "artifacts/full_evidence_orchestration_20260525/"
            "high_accuracy_refinement_packet_x"
        ),
        packet_report={
            "final_status": "BLOCKED_KEEP_BASELINE",
            "promotion_pr_gate": {"status": "BLOCKED", "blockers": []},
            "unified_evidence_summary": {
                "status": "UNIFIED_EVIDENCE_COLLECTING",
                "unified_evidence_eligible_rows": 98,
                "rows_with_artifact_shadow_odds": 29,
                "artifact_odds_rows_seen": 116,
                "artifact_odds_rows_accepted": 29,
                "artifact_odds_rows_rejected": 87,
                "artifact_odds_rejection_reason_counts": {
                    "odds_match_status_not_valid_pre_jump_dog_odds": 87
                },
                "minimum_eligible_rows_for_review": 100,
            },
            "backlog_unified_evidence_summary": {
                "status": "BACKLOG_UNIFIED_EVIDENCE_READY_FOR_REVIEW",
                "source_status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
                "dataset_count": 20,
                "failed_dataset_count": 0,
                "unified_evidence_eligible_rows": 415,
                "rows_with_artifact_shadow_odds": 79,
                "artifact_odds_rows_seen": 250,
                "artifact_odds_rows_accepted": 79,
                "artifact_odds_rows_rejected": 171,
                "artifact_odds_rejection_reason_counts": {
                    "odds_match_status_not_valid_pre_jump_dog_odds": 171
                },
                "aggregation_scope": "per_dataset_totals_not_cross_dataset_deduped",
            },
            "source_artifacts": {
                "backlog_unified_evidence_status": (
                    "artifacts/full_evidence_orchestration_20260525/"
                    "shadow_autopilot_v1_x/backlog_unified_evidence_datasets_status.json"
                )
            },
            "no_write_guarantees": {"db_write": False, "label_write": False},
        },
        attempted=True,
        returncode=0,
    )

    assert status["status"] == "BLOCKED_KEEP_BASELINE"
    assert status["unified_evidence_eligible_rows"] == 98
    assert status["unified_evidence_artifact_odds_rows_accepted"] == 29
    assert status["unified_evidence_artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 87
    }
    assert status["backlog_unified_evidence_eligible_rows"] == 415
    assert status["backlog_unified_evidence_artifact_odds_rows_accepted"] == 79
    assert status["backlog_unified_evidence_artifact_odds_rejection_reason_counts"] == {
        "odds_match_status_not_valid_pre_jump_dog_odds": 171
    }
    assert status["best_available_unified_evidence_eligible_rows"] == 415
    assert status["best_available_unified_evidence_scope"] == "backlog"
    assert (
        status["backlog_unified_evidence_aggregation_scope"]
        == "per_dataset_totals_not_cross_dataset_deduped"
    )


def test_historical_unified_evidence_reports_use_automatic_daemon_evidence_only(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"

    def write_report(
        dirname: str,
        eligible_rows: int,
        *,
        final_status: str = "UNIFIED_EVIDENCE_DATASET_BUILT",
    ) -> Path:
        dataset_dir = evidence_root / dirname
        dataset_dir.mkdir(parents=True)
        dataset_path = dataset_dir / "unified_evidence_dataset.jsonl"
        dataset_path.write_text("{}\n", encoding="utf-8")
        report_path = dataset_dir / "unified_evidence_dataset_report.json"
        autopilot.write_json(
            report_path,
            {
                "final_status": final_status,
                "dataset_jsonl": str(dataset_path),
                "unified_evidence_eligible_rows": eligible_rows,
            },
        )
        return report_path

    older = write_report(
        "unified_evidence_dataset_20260610T184100+1000_daemon_autopilot_backlog_001",
        8,
    )
    newer = write_report(
        "unified_evidence_dataset_20260610T185649+1000_daemon_autopilot_backlog_001",
        12,
    )
    rejoin = write_report(
        "unified_evidence_dataset_20260610T191153+1000_daemon_rejoin_007",
        23,
    )
    approved_append = write_report(
        "unified_evidence_dataset_live_db_approved_append_20260701T0630p1000_01",
        31,
    )
    write_report(
        "unified_evidence_dataset_20260610T191153+1000_daemon_rejoin_bridge_validation_007",
        27,
    )
    write_report(
        "unified_evidence_dataset_20260610T185649+1000_daemon_autopilot_backlog_manual_001",
        23,
    )
    write_report("unified_evidence_dataset_20260610T163700_stage2_result_retry", 11)
    write_report(
        "unified_evidence_dataset_20260610T190000+1000_daemon_autopilot_backlog_001",
        0,
    )
    write_report(
        "unified_evidence_dataset_20260610T191500+1000_daemon_autopilot_backlog_001",
        5,
        final_status="UNIFIED_EVIDENCE_DATASET_EMPTY",
    )

    reports = autopilot.historical_unified_evidence_report_paths(evidence_root)

    assert reports == [older, newer, rejoin, approved_append]
    assert autopilot.best_unified_evidence_report_path(reports) == approved_append


def test_historical_unified_evidence_resolves_approved_append_artifact_relative_dataset(
    tmp_path,
):
    worktree_root = tmp_path / "greyhound-autonomous-accuracy-odds-v1"
    evidence_root = worktree_root / "artifacts/full_evidence_orchestration_20260525"
    dataset_dir = (
        evidence_root
        / "unified_evidence_dataset_live_db_approved_append_20260701T0630p1000_01"
    )
    dataset_dir.mkdir(parents=True)
    dataset_path = dataset_dir / "unified_evidence_dataset.jsonl"
    dataset_path.write_text("{}\n", encoding="utf-8")
    report_path = dataset_dir / "unified_evidence_dataset_report.json"
    dataset_jsonl = (
        "artifacts/full_evidence_orchestration_20260525/"
        "unified_evidence_dataset_live_db_approved_append_20260701T0630p1000_01/"
        "unified_evidence_dataset.jsonl"
    )
    autopilot.write_json(
        report_path,
        {
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "dataset_jsonl": dataset_jsonl,
            "unified_evidence_eligible_rows": 70,
        },
    )

    reports = autopilot.historical_unified_evidence_report_paths(evidence_root)

    assert autopilot.unified_report_dataset_path(
        report_path,
        autopilot.load_json(report_path),
    ) == dataset_path
    assert reports == [report_path]


def test_historical_unified_evidence_default_retains_more_than_100_reports(tmp_path):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"

    for index in range(101):
        dataset_dir = (
            evidence_root
            / f"unified_evidence_dataset_20260610T{index:06d}+1000_daemon_autopilot"
        )
        dataset_dir.mkdir(parents=True)
        dataset_path = dataset_dir / "unified_evidence_dataset.jsonl"
        dataset_path.write_text("{}\n", encoding="utf-8")
        autopilot.write_json(
            dataset_dir / "unified_evidence_dataset_report.json",
            {
                "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
                "dataset_jsonl": str(dataset_path),
                "unified_evidence_eligible_rows": 1,
            },
        )

    reports = autopilot.historical_unified_evidence_report_paths(evidence_root)

    assert len(reports) == 101
    assert reports[0].parent.name.endswith("000000+1000_daemon_autopilot")


def test_final_verdict_treats_enabled_autonomous_odds_capture_failure_as_partial():
    verdict = autopilot.final_verdict_for(
        steps=[
            {
                "name": "autonomous_live_odds_capture",
                "returncode": 2,
            }
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "PARTIAL_AUTOMATION_READY"


def test_final_verdict_treats_unified_dataset_failure_as_partial():
    verdict = autopilot.final_verdict_for(
        steps=[
            {
                "name": "unified_evidence_dataset",
                "returncode": 2,
            }
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "PARTIAL_AUTOMATION_READY"


def test_final_verdict_treats_backlog_unified_dataset_failure_as_partial():
    verdict = autopilot.final_verdict_for(
        steps=[
            {
                "name": "backlog_unified_evidence_dataset_001",
                "returncode": 2,
            }
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "PARTIAL_AUTOMATION_READY"


def test_final_verdict_treats_rolling_model_comparison_failure_as_partial():
    verdict = autopilot.final_verdict_for(
        steps=[
            {
                "name": "rolling_model_comparison",
                "returncode": 2,
            }
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "PARTIAL_AUTOMATION_READY"


def test_final_verdict_treats_enabled_autonomous_result_capture_failure_as_partial():
    verdict = autopilot.final_verdict_for(
        steps=[
            {
                "name": "autonomous_official_result_capture",
                "returncode": 2,
            }
        ],
        protected_paths_unchanged=True,
        required_outputs_present=True,
    )

    assert verdict == "PARTIAL_AUTOMATION_READY"


def test_autonomous_live_odds_capture_runs_before_daily_shadow_run(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    step_names: list[str] = []
    commands_by_step: dict[str, list[str]] = {}

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        step_names.append(name)
        commands_by_step[name] = list(command)
        if name == "refresh_prejump_races":
            autopilot.write_json(
                command_value(command, "--output"),
                {
                    "status": "READY",
                    "dry_run": False,
                    "files": [],
                },
            )
        elif name == "refresh_odds_capture_candidates":
            assert command_value(command, "--upcoming-dir") == (
                output_dir / "odds_capture_refreshed_upcoming"
            )
            autopilot.write_json(
                command_value(command, "--output"),
                {
                    "status": "READY",
                    "dry_run": False,
                    "files": [],
                },
            )
        elif name == "autonomous_live_odds_capture":
            capture_dir = command_value(command, "--output-dir")
            assert command_value(command, "--input-dir") == (
                output_dir / "odds_capture_refreshed_upcoming"
            )
            autopilot.write_json(
                capture_dir / "autonomous_live_odds_capture_report.json",
                {
                    "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_COLLECTED",
                    "execute": False,
                    "allow_auto_scrape_odds": False,
                    "ready_count": 1,
                    "validation_pass_count": 1,
                    "inserted_live_odds_rows": 0,
                    "status_counts": {"READY_FOR_CAPTURE": 1},
                },
            )
        elif name == "daily_shadow_run":
            assert (output_dir / "autonomous_live_odds_capture_status.json").exists()
            daily_dir = command_value(command, "--output-dir")
            assert command_value(command, "--input-dir") == (
                output_dir / "refreshed_upcoming"
            )
            autopilot.write_json(
                daily_dir / "shadow_manifest.json",
                {"race_count": 0, "prediction_rows": 0},
            )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)

    args = autopilot.parse_args(
        [
            "--run-id",
            "order_check",
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            "2026-06-11T20:20:00+10:00",
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--enable-autonomous-odds-capture",
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
            "--skip-unified-dataset",
            "--autonomous-odds-capture-limit",
            "2",
            "--require-safe-refresh-metadata",
        ]
    )

    result = autopilot.run_autopilot(args)
    report = json.loads(
        (evidence_root / "shadow_autopilot_v1_order_check" / "shadow_orchestration_report.json")
        .read_text(encoding="utf-8")
    )

    assert result["autonomous_live_odds_capture_status"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_COLLECTED"
    )
    assert step_names[:4] == [
        "refresh_prejump_races",
        "refresh_odds_capture_candidates",
        "autonomous_live_odds_capture",
        "daily_shadow_run",
    ]
    assert [step["name"] for step in report["steps"][:4]] == step_names[:4]
    odds_refresh_command = commands_by_step["refresh_odds_capture_candidates"]
    assert odds_refresh_command[
        odds_refresh_command.index("--min-minutes") + 1
    ] == "0.0"
    assert odds_refresh_command[
        odds_refresh_command.index("--max-minutes") + 1
    ] == "60.0"
    assert odds_refresh_command[
        odds_refresh_command.index("--limit") + 1
    ] == "8"
    autonomous_capture_command = commands_by_step["autonomous_live_odds_capture"]
    assert autonomous_capture_command[
        autonomous_capture_command.index("--limit") + 1
    ] == "2"
    shadow_refresh_command = commands_by_step["refresh_prejump_races"]
    assert shadow_refresh_command[
        shadow_refresh_command.index("--min-minutes") + 1
    ] == "20.0"
    assert shadow_refresh_command[
        shadow_refresh_command.index("--max-minutes") + 1
    ] == "160.0"
    assert shadow_refresh_command[
        shadow_refresh_command.index("--current-time") + 1
    ] == "2026-06-11T20:20:00+10:00"
    assert odds_refresh_command[
        odds_refresh_command.index("--current-time") + 1
    ] == "2026-06-11T20:20:00+10:00"
    assert "--require-safe-metadata" in shadow_refresh_command
    assert "--require-safe-metadata" in odds_refresh_command
    daily_command = commands_by_step["daily_shadow_run"]
    assert daily_command[daily_command.index("--output-parent") + 1] == str(evidence_root)


def test_autonomous_official_result_capture_uses_fresh_step_time(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    commands_by_step: dict[str, list[str]] = {}

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})
    monkeypatch.setattr(
        autopilot,
        "current_step_time_iso",
        lambda: "2026-06-12T18:53:33+10:00",
    )

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        commands_by_step[name] = list(command)
        if name == "daily_shadow_run":
            daily_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                daily_dir / "shadow_manifest.json",
                {"race_count": 1, "prediction_rows": 8},
            )
            (daily_dir / "shadow_predictions.jsonl").write_text(
                json.dumps(
                    {
                        "race_id": "Race 1 - BEN - 2026-06-12",
                        "dog_name": "Example Runner",
                        "box": 1,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        elif name == "autonomous_official_result_capture":
            early_status = json.loads(
                (output_dir / "autonomous_official_result_capture_status.json").read_text(
                    encoding="utf-8"
                )
            )
            assert early_status["status"] == (
                "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_IN_PROGRESS"
            )
            assert early_status["attempted"] is True
            assert early_status["in_progress"] is True
            assert early_status["target_date"] == "2026-06-12"
            assert early_status["candidate_source"] == "shadow_run_predictions"
            capture_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                capture_dir / "autonomous_official_result_capture_report.json",
                {
                    "final_status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
                    "candidate_count": 1,
                    "official_result_race_rows": 1,
                    "official_result_runner_rows": 8,
                    "quarantine_rows": 0,
                    "official_result_evidence_db_ingest": {
                        "status": "APPENDED_OFFICIAL_RESULT_EVIDENCE",
                        "execute": True,
                        "db_write_performed": True,
                        "inserted_race_rows": 1,
                        "inserted_runner_rows": 8,
                    },
                },
            )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)

    args = autopilot.parse_args(
        [
            "--run-id",
            "fresh_result_time",
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            "2026-06-12T18:47:11+10:00",
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--enable-autonomous-result-capture",
            "--skip-refresh",
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
            "--skip-unified-dataset",
        ]
    )

    result = autopilot.run_autopilot(args)

    daily_command = commands_by_step["daily_shadow_run"]
    result_command = commands_by_step["autonomous_official_result_capture"]
    assert daily_command[daily_command.index("--current-time") + 1] == (
        "2026-06-12T18:47:11+10:00"
    )
    assert result_command[result_command.index("--current-time") + 1] == (
        "2026-06-12T18:53:33+10:00"
    )
    assert result_command[result_command.index("--date") + 1] == "2026-06-12"
    assert result["autonomous_official_result_capture_status"] == (
        "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    )
    final_status = json.loads(
        (
            evidence_root
            / "shadow_autopilot_v1_fresh_result_time"
            / "autonomous_official_result_capture_status.json"
        ).read_text(encoding="utf-8")
    )
    assert final_status["status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    assert final_status["in_progress"] is False


def test_high_accuracy_step_uses_daily_dir_stage2_predictions(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    commands_by_step: dict[str, list[str]] = {}

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})
    monkeypatch.setattr(autopilot, "historical_unified_evidence_report_paths", lambda *args, **kwargs: [])

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        commands_by_step[name] = list(command)
        if name == "daily_shadow_run":
            daily_dir = command_value(command, "--output-dir")
            daily_dir.mkdir(parents=True, exist_ok=True)
            autopilot.write_json(
                daily_dir / "shadow_manifest.json",
                {"race_count": 1, "prediction_rows": 8},
            )
            (daily_dir / "shadow_predictions.jsonl").write_text(
                '{"race_id":"Race 1 - BEN - 2026-06-12","dog_name":"A","box":1}\n',
                encoding="utf-8",
            )
            (daily_dir / "stage2_shadow_predictions.jsonl").write_text(
                '{"race_id":"Race 1 - BEN - 2026-06-12","dog_name":"A","box":1}\n',
                encoding="utf-8",
            )
        elif name == "unified_evidence_dataset":
            dataset_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                dataset_dir / "unified_evidence_dataset_report.json",
                {
                    "status": "UNIFIED_EVIDENCE_DATASET_BUILT",
                    "row_count": 8,
                    "race_count": 1,
                    "rows_with_official_results": 8,
                    "rows_with_stage2_predictions": 8,
                    "rows_with_strict_prejump_odds": 0,
                    "unified_evidence_eligible_rows": 8,
                },
            )
        elif name == "rolling_model_comparison":
            rolling_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                rolling_dir / "rolling_model_comparison_report.json",
                {
                    "status": "ROLLING_MODEL_COMPARISON_READY_FOR_REVIEW",
                    "sample_race_count": 1,
                    "candidate_count": 1,
                    "best_candidate_key": "primary_shadow",
                },
            )
            (rolling_dir / "market_residual_runner_matrix.csv").write_text(
                "race_id,dog_name,box\nRace 1 - BEN - 2026-06-12,A,1\n",
                encoding="utf-8",
            )
        elif name == "pre_race_gated_challenger":
            pre_race_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                pre_race_dir / "pre_race_gated_challenger_report.json",
                {"status": "PRE_RACE_GATED_CHALLENGER_BLOCKED"},
            )
        elif name == "official_result_reserve_substitution_preflight":
            preflight_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                preflight_dir / "official_result_reserve_substitution_preflight.json",
                {
                    "final_status": (
                        "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
                    ),
                    "candidate_count": 1,
                    "blocked_candidate_count": 0,
                    "ready_for_policy_review_count": 1,
                    "candidates": [],
                    "no_write_guarantees": {"db_write": False},
                },
            )
            autopilot.write_json(
                preflight_dir / "reserve_substitution_manual_review_packet.json",
                {
                    "final_status": "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY",
                    "candidate_count": 1,
                    "ready_candidate_count": 1,
                    "blocked_candidate_count": 0,
                    "mapping_pair_count": 1,
                    "dataset_join_allowed": False,
                    "official_result_acceptance_allowed": False,
                    "db_write": False,
                    "ready_race_ids": ["Race 1 - BEN - 2026-06-12"],
                    "candidates": [
                        {
                            "race_id": "Race 1 - BEN - 2026-06-12",
                            "mapping_hypothesis": {
                                "pairs": [
                                    {
                                        "scratched_participant_box": 1,
                                        "reserve_box": 9,
                                        "mapping_acceptance_status": "not_accepted",
                                    }
                                ],
                            },
                        }
                    ],
                    "no_write_guarantees": {"db_write": False},
                },
            )
        elif name == "high_accuracy_refinement_packet":
            high_accuracy_dir = command_value(command, "--output-dir")
            preflight_path = Path(
                command[command.index("--reserve-substitution-preflight") + 1]
            )
            manual_review_path = (
                preflight_path.parent / "reserve_substitution_manual_review_packet.json"
            )
            assert manual_review_path.exists()
            autopilot.write_json(
                high_accuracy_dir / "high_accuracy_refinement_packet.json",
                {
                    "final_status": "BLOCKED_KEEP_BASELINE",
                    "reserve_substitution_manual_review_summary": {
                        "status": "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY",
                        "candidate_count": 1,
                        "ready_candidate_count": 1,
                        "blocked_candidate_count": 0,
                        "mapping_pair_count": 1,
                        "dataset_join_allowed": False,
                        "official_result_acceptance_allowed": False,
                        "db_write": False,
                        "blockers": [],
                        "ready_race_ids": ["Race 1 - BEN - 2026-06-12"],
                    },
                    "source_artifacts": {
                        "reserve_substitution_preflight": str(preflight_path),
                        "reserve_substitution_manual_review": str(manual_review_path),
                    },
                },
            )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)

    args = autopilot.parse_args(
        [
            "--run-id",
            "high_accuracy_stage2_path",
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            "2026-06-12T18:47:11+10:00",
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--skip-refresh",
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
        ]
    )

    autopilot.run_autopilot(args)

    high_accuracy_command = commands_by_step["high_accuracy_refinement_packet"]
    assert "--stage2-predictions" in high_accuracy_command
    stage2_predictions = Path(
        high_accuracy_command[high_accuracy_command.index("--stage2-predictions") + 1]
    )
    assert stage2_predictions == (
        evidence_root
        / "daily_race_ingest_shadow_high_accuracy_stage2_path_autopilot"
        / "stage2_shadow_predictions.jsonl"
    )
    assert "official_result_reserve_substitution_preflight" in commands_by_step
    assert "--reserve-substitution-preflight" in high_accuracy_command
    reserve_preflight = Path(
        high_accuracy_command[
            high_accuracy_command.index("--reserve-substitution-preflight") + 1
        ]
    )
    assert reserve_preflight == (
        evidence_root
        / "official_result_reserve_substitution_preflight_high_accuracy_stage2_path_autopilot"
        / "official_result_reserve_substitution_preflight.json"
    )
    high_accuracy_status = json.loads(
        (
            evidence_root
            / "shadow_autopilot_v1_high_accuracy_stage2_path"
            / "high_accuracy_refinement_status.json"
        ).read_text(encoding="utf-8")
    )
    assert high_accuracy_status["reserve_substitution_manual_review_status"] == (
        "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
    )
    assert (
        high_accuracy_status[
            "reserve_substitution_manual_review_ready_candidate_count"
        ]
        == 1
    )
    assert high_accuracy_status[
        "reserve_substitution_manual_review_mapping_pair_count"
    ] == 1
    assert (
        high_accuracy_status[
            "reserve_substitution_manual_review_dataset_join_allowed"
        ]
        is False
    )
    assert high_accuracy_status["reserve_substitution_manual_review_report"].endswith(
        "reserve_substitution_manual_review_packet.json"
    )


def test_autopilot_materializes_nested_stage2_after_daily_shadow_run(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    step_names: list[str] = []

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        step_names.append(name)
        if name == "refresh_prejump_races":
            autopilot.write_json(
                command_value(command, "--output"),
                {"status": "READY", "dry_run": False, "files": []},
            )
        elif name == "daily_shadow_run":
            daily_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                daily_dir / "shadow_manifest.json",
                {"race_count": 1, "prediction_rows": 1},
            )
            (daily_dir / "shadow_predictions.jsonl").write_text(
                json.dumps({"race_id": "Race 1 - WPK - 2026-06-10"}) + "\n",
                encoding="utf-8",
            )
            nested_stage2 = (
                daily_dir / "shadow_score_live/stage2_shadow_predictions.jsonl"
            )
            nested_stage2.parent.mkdir(parents=True)
            nested_stage2.write_text(
                json.dumps(
                    {
                        "race_id": "Race 1 - WPK - 2026-06-10",
                        "dog_name": "Alpha Runner",
                        "box": 1,
                        "stage2_challenger_key": "shadow_calibrated_rf_power_gamma_2_4",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)

    args = autopilot.parse_args(
        [
            "--run-id",
            "nested_stage2",
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            "2026-06-11T20:20:00+10:00",
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
            "--skip-unified-dataset",
        ]
    )

    autopilot.run_autopilot(args)

    output_dir = evidence_root / "shadow_autopilot_v1_nested_stage2"
    daily_dir = evidence_root / "daily_race_ingest_shadow_nested_stage2_autopilot"
    root_stage2 = daily_dir / "stage2_shadow_predictions.jsonl"
    nested_stage2 = daily_dir / "shadow_score_live/stage2_shadow_predictions.jsonl"
    status = json.loads(
        (output_dir / "stage2_shadow_predictions_status.json").read_text(
            encoding="utf-8"
        )
    )
    report = json.loads(
        (output_dir / "shadow_orchestration_report.json").read_text(encoding="utf-8")
    )

    assert step_names == ["refresh_prejump_races", "daily_shadow_run"]
    assert root_stage2.read_text(encoding="utf-8") == nested_stage2.read_text(
        encoding="utf-8"
    )
    assert status["status"] == "STAGE2_SHADOW_PREDICTIONS_ROOT_MATERIALIZED"
    assert status["root_materialized"] is True
    assert [
        step["name"] for step in report["steps"][:3]
    ] == [
        "refresh_prejump_races",
        "daily_shadow_run",
        "stage2_shadow_predictions_first_class",
    ]
    assert report["steps"][2]["stage2_status"] == (
        "STAGE2_SHADOW_PREDICTIONS_ROOT_MATERIALIZED"
    )


def test_skip_primary_refresh_still_runs_odds_capture_refresh(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    historical_unified_dir = (
        evidence_root / "unified_evidence_dataset_prior_daemon_autopilot"
    )
    autopilot.write_json(
        historical_unified_dir / "unified_evidence_dataset_report.json",
        {
            "final_status": "UNIFIED_EVIDENCE_DATASET_BUILT",
            "unified_evidence_eligible_rows": 8,
        },
    )
    (historical_unified_dir / "unified_evidence_dataset.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    step_names: list[str] = []

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        step_names.append(name)
        if name == "refresh_odds_capture_candidates":
            autopilot.write_json(
                command_value(command, "--output"),
                {"status": "READY", "dry_run": False, "files": []},
            )
        elif name == "autonomous_live_odds_capture":
            capture_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                capture_dir / "autonomous_live_odds_capture_report.json",
                {
                    "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS",
                    "execute": True,
                    "allow_auto_scrape_odds": True,
                    "ready_count": 0,
                    "validation_pass_count": 0,
                    "inserted_live_odds_rows": 0,
                    "status_counts": {},
                },
            )
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timeout_seconds": timeout_seconds,
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)
    monkeypatch.setattr(
        autopilot,
        "publish_current_race_index",
        lambda **_kwargs: pytest.fail(
            "odds-only refresh must not replace the shared candidate index"
        ),
    )

    args = autopilot.parse_args(
        [
            "--run-id",
            "odds_only_refresh",
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            "2026-06-12T00:35:00+10:00",
            "--db",
            str(db_path),
            "--current-race-index-state-path",
            str(evidence_root / "runtime/odds_capture_state.json"),
            "--enable-autonomous-odds-capture",
            "--execute-autonomous-odds-capture",
            "--allow-auto-scrape-odds",
            "--skip-primary-refresh",
            "--skip-shadow-run",
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
            "--skip-unified-dataset",
        ]
    )

    result = autopilot.run_autopilot(args)
    output_dir = evidence_root / "shadow_autopilot_v1_odds_only_refresh"
    refresh_report = json.loads(
        (output_dir / "refresh_prejump_report.json").read_text(encoding="utf-8")
    )

    assert "refresh_prejump_races" not in step_names
    assert step_names == [
        "refresh_odds_capture_candidates",
        "autonomous_live_odds_capture",
    ]
    assert refresh_report["status"] == "SKIPPED"
    assert refresh_report["reason"] == "skip_primary_refresh_requested"
    assert result["autonomous_live_odds_capture_status"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_NO_ELIGIBLE_WINDOWS"
    )
    assert (output_dir / "odds_capture_refresh_report.json").exists()
    publication = json.loads(
        (output_dir / "current_race_index_publish.json").read_text(encoding="utf-8")
    )
    assert publication == {
        "schema_version": "collector_current_race_index_publish_v2",
        "status": "SKIPPED",
        "reason": "primary_candidate_refresh_not_run",
    }
    assert (output_dir / "rolling_model_comparison_status.json").exists()
    assert (output_dir / "high_accuracy_refinement_status.json").exists()
    rolling_status = json.loads(
        (output_dir / "rolling_model_comparison_status.json").read_text(
            encoding="utf-8"
        )
    )
    high_accuracy_status = json.loads(
        (output_dir / "high_accuracy_refinement_status.json").read_text(
            encoding="utf-8"
        )
    )
    assert rolling_status["status"] == "SKIPPED"
    assert rolling_status["skipped_reason"] == "skip_unified_dataset_requested"
    assert high_accuracy_status["status"] == "SKIPPED"
    assert high_accuracy_status["skipped_reason"] == "skip_unified_dataset_requested"
    assert not (
        evidence_root / "rolling_model_comparison_odds_only_refresh_autopilot"
    ).exists()
    assert not (
        evidence_root / "high_accuracy_refinement_packet_odds_only_refresh_autopilot"
    ).exists()


def test_skip_primary_refresh_requires_no_shadow_run_or_explicit_input_dir(tmp_path, monkeypatch):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})

    args = autopilot.parse_args(
        [
            "--run-id",
            "bad_skip_primary_refresh",
            "--evidence-root",
            str(evidence_root),
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--skip-primary-refresh",
        ]
    )

    try:
        autopilot.run_autopilot(args)
    except RuntimeError as exc:
        assert "skip_primary_refresh_requires_skip_shadow_run_or_input_dir" in str(exc)
    else:
        raise AssertionError("expected skip-primary-refresh misuse to fail closed")


def test_autonomous_live_odds_capture_recovers_late_report_after_timeout(
    tmp_path, monkeypatch
):
    evidence_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    shadow_model = tmp_path / "shadow_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    db_path = tmp_path / "greyhound_racing_data.db"
    db_path.write_text("db", encoding="utf-8")

    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    monkeypatch.setattr(autopilot, "protected_hashes", lambda: {})

    def command_value(command, flag):
        return Path(command[command.index(flag) + 1])

    def fake_step_command(
        *,
        name,
        command,
        output_dir,
        cwd=autopilot.ROOT,
        timeout_seconds=None,
    ):
        if name == "refresh_prejump_races":
            autopilot.write_json(
                command_value(command, "--output"),
                {"status": "READY", "dry_run": False, "files": []},
            )
        elif name == "refresh_odds_capture_candidates":
            autopilot.write_json(
                command_value(command, "--output"),
                {"status": "READY", "dry_run": False, "files": []},
            )
        elif name == "daily_shadow_run":
            daily_dir = command_value(command, "--output-dir")
            autopilot.write_json(
                daily_dir / "shadow_manifest.json",
                {"race_count": 0, "prediction_rows": 0},
            )
        if name == "autonomous_live_odds_capture":
            return {
                "name": name,
                "command": list(command),
                "returncode": -9,
                "timed_out": True,
                "timeout_seconds": timeout_seconds,
            }
        return {
            "name": name,
            "command": list(command),
            "returncode": 0,
            "timed_out": False,
            "timeout_seconds": timeout_seconds,
        }

    def fake_timeout_grace(path):
        assert path.name == "autonomous_live_odds_capture_report.json"
        return {
            "final_status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED",
            "execute": True,
            "allow_auto_scrape_odds": True,
            "ready_count": 2,
            "validation_pass_count": 1,
            "inserted_live_odds_rows": 6,
            "status_counts": {"APPENDED": 1, "BLOCKED_FETCH_EXCEPTION": 1},
            "no_write_guarantees": {
                "db_write": True,
                "odds_history_write": False,
                "race_metadata_write": False,
            },
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)
    monkeypatch.setattr(autopilot, "load_json_after_timeout_grace", fake_timeout_grace)

    args = autopilot.parse_args(
        [
            "--run-id",
            "timeout_recovery",
            "--evidence-root",
            str(evidence_root),
            "--current-time",
            "2026-06-11T20:20:00+10:00",
            "--db",
            str(db_path),
            "--shadow-model",
            str(shadow_model),
            "--enable-autonomous-odds-capture",
            "--execute-autonomous-odds-capture",
            "--allow-auto-scrape-odds",
            "--skip-odds-snapshot",
            "--skip-result-join",
            "--skip-aggregate",
            "--skip-status",
            "--skip-unified-dataset",
        ]
    )

    result = autopilot.run_autopilot(args)
    status = json.loads(
        (
            evidence_root
            / "shadow_autopilot_v1_timeout_recovery"
            / "autonomous_live_odds_capture_status.json"
        ).read_text(encoding="utf-8")
    )

    assert result["autonomous_live_odds_capture_status"] == (
        "AUTONOMOUS_LIVE_ODDS_CAPTURE_APPENDED"
    )
    assert status["returncode"] == -9
    assert status["timed_out"] is True
    assert status["recovered_from_step_failure"] is True
    assert status["inserted_live_odds_rows"] == 6


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
            "shadow_run_dir": "artifacts/full_evidence_orchestration_20260525/daily_shadow_x",
            "db_path": "greyhound_racing_data.db",
            "effective_prediction_timestamp": "2026-06-13T22:22:36+10:00",
            "effective_prediction_timestamp_source": "prediction_timestamp",
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
                "odds_research_next_action": (
                    "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
                ),
                "timing_aligned_prediction_rerun_required": True,
                "timing_aligned_prediction_rerun_race_count": 1,
                "timing_aligned_prediction_rerun_races": [
                    {"race_id": "Race 8 - CANN - 2026-06-13"}
                ],
                "timing_aligned_prediction_rerun_reason_counts": {
                    "raw_expected_prejump_windows_complete_but_after_prediction": 1
                },
            },
            "odds_research_gate": {
                "status": "ODDS_RESEARCH_BLOCKED_PROVENANCE",
                "complete_valid_prejump_odds_races": 1,
                "minimum_complete_valid_prejump_odds_races": 100,
                "source_url_coverage_pct": 100.0,
                "source_url_rows_missing": 0,
                "blocker_counts": {"complete_valid_prejump_odds_races_below_min": 99},
            },
            "odds_research_gate_report_path": (
                "artifacts/full_evidence_orchestration_20260525/"
                "shadow_odds_snapshot_x/odds_research_gate_report.json"
            ),
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
            "ev_output_rows": 0,
        },
    )

    assert status["races_with_any_odds_candidates"] == 1
    assert status["races_with_complete_odds_candidate_coverage"] == 1
    assert status["races_with_complete_valid_prejump_odds"] == 1
    assert status["source_shadow_run_dir"] == (
        "artifacts/full_evidence_orchestration_20260525/daily_shadow_x"
    )
    assert status["effective_prediction_timestamp"] == "2026-06-13T22:22:36+10:00"
    assert status["races_with_missing_odds_rows"] == 0
    assert status["races_with_post_feature_freeze_odds_rows"] == 1
    assert status["race_coverage_path"].endswith("shadow_odds_race_coverage.json")
    assert status["odds_analysis_status"] == "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED"
    assert status["odds_analysis_blocker_counts"] == {}
    assert status["odds_research_next_action"] == (
        "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
    )
    assert status["timing_aligned_prediction_rerun_required"] is True
    assert status["timing_aligned_prediction_rerun_race_count"] == 1
    assert status["timing_aligned_prediction_rerun_race_ids"] == [
        "Race 8 - CANN - 2026-06-13"
    ]
    assert status["timing_aligned_prediction_rerun_reason_counts"] == {
        "raw_expected_prejump_windows_complete_but_after_prediction": 1
    }
    assert status["odds_research_gate_status"] == "ODDS_RESEARCH_BLOCKED_PROVENANCE"
    assert status["odds_research_gate_report_path"].endswith(
        "odds_research_gate_report.json"
    )
    assert status["odds_research_gate_complete_valid_prejump_odds_races"] == 1
    assert status["odds_research_gate_minimum_complete_valid_prejump_odds_races"] == 100
    assert status["odds_research_gate_source_url_coverage_pct"] == 100.0
    assert status["odds_research_gate_source_url_rows_missing"] == 0
    assert status["odds_research_gate_blocker_counts"] == {
        "complete_valid_prejump_odds_races_below_min": 99
    }
    assert status["approved_odds_augmented_candidate_key"] == "stage2_market_blend_70"
    assert status["approved_odds_augmented_prediction_status"] == "APPROVED_BLEND_READY"
    assert status["approved_odds_augmented_ready_race_count"] == 1
    assert status["approved_odds_augmented_blocked_race_count"] == 0
    assert status["approved_odds_augmented_prediction_rows"] == 8
    assert status["approved_odds_augmented_prediction_report_path"].endswith(
        "approved_odds_augmented_prediction_report.json"
    )


def test_timing_aligned_prediction_rerun_plan_uses_source_shadow_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    source_run = (
        tmp_path / "artifacts/full_evidence_orchestration_20260525/daily_shadow_x"
    )
    input_a = source_run / "eligible_inputs/source_0001"
    input_b = source_run / "eligible_inputs/source_0002"
    input_a.mkdir(parents=True)
    input_b.mkdir(parents=True)
    race_a = input_a / "Race 10 - CANN - 2026-06-13.csv"
    race_b = input_b / "Race 8 - CANN - 2026-06-13.csv"
    race_a.write_text("dog_name,box\nAlpha,1\n", encoding="utf-8")
    race_b.write_text("dog_name,box\nBeta,2\n", encoding="utf-8")
    model_path = (
        tmp_path / "artifacts/full_evidence_orchestration_20260525/model.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    autopilot.write_json(
        source_run / "shadow_manifest.json",
        {
            "stage2_prediction_rows": 13,
            "score_live_manifest": {
                "input_files": [
                    race_a.relative_to(tmp_path).as_posix(),
                    race_b.relative_to(tmp_path).as_posix(),
                ],
                "model_source": model_path.relative_to(tmp_path).as_posix(),
            },
        },
    )
    autopilot.write_json(
        source_run / "prejump_metadata_report.json",
        {
            "files": [
                {
                    "race_date": "2026-06-13",
                    "venue": "CANN",
                    "race_number": 10,
                    "jump_time": "11:52 PM",
                },
                {
                    "race_date": "2026-06-13",
                    "venue": "CANN",
                    "race_number": 8,
                    "jump_time": "11:55 PM",
                },
            ]
        },
    )

    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_x"
    )
    plan = autopilot.build_timing_aligned_prediction_rerun_plan(
        generated_at=datetime.fromisoformat("2026-06-13T23:40:00+10:00"),
        odds_snapshot_status={
            "odds_research_next_action": (
                "RERUN_FORWARD_SHADOW_AFTER_ODDS_CAPTURE_FOR_TIMING_ALIGNED_EVIDENCE"
            ),
            "timing_aligned_prediction_rerun_required": True,
            "timing_aligned_prediction_rerun_race_ids": [
                "Race 10 - CANN - 2026-06-13",
                "Race 8 - CANN - 2026-06-13",
            ],
            "timing_aligned_prediction_rerun_reason_counts": {
                "raw_expected_prejump_windows_complete_but_after_prediction": 2
            },
            "source_shadow_run_dir": source_run.relative_to(tmp_path).as_posix(),
            "effective_prediction_timestamp": "2026-06-13T22:22:36+10:00",
            "effective_prediction_timestamp_source": "prediction_timestamp",
        },
        output_dir=output_dir,
        db_path=tmp_path / "greyhound_racing_data.db",
        shadow_model=model_path,
        score_command_mode="python",
    )

    assert plan["status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_READY_FOR_GUARDED_EXECUTION"
    )
    assert plan["execution_performed"] is False
    assert plan["timing_aligned_prediction_rerun_race_count"] == 2
    assert plan["missing_input_race_ids"] == []
    assert plan["source_stage2_prediction_rows"] == 13
    assert plan["stage2_predictions_required_first_class"] is True
    assert plan["planned_classification_current_time"] == "2026-06-13T23:40:00+10:00"
    assert plan["missing_jump_race_ids"] == []
    assert [race["generated_at_after_jump"] for race in plan["race_jump_contexts"]] == [
        False,
        False,
    ]
    assert plan["matched_input_dirs"] == [
        "artifacts/full_evidence_orchestration_20260525/daily_shadow_x/eligible_inputs/source_0001",
        "artifacts/full_evidence_orchestration_20260525/daily_shadow_x/eligible_inputs/source_0002",
    ]
    command = plan["planned_command"]
    assert "scripts/daily_race_ingest_shadow_orchestrator.py" in command[1]
    assert "--mode" in command
    assert "full-dry-run" in command
    assert "--shadow-model" in command
    assert str(model_path) in command
    assert command[command.index("--output-parent") + 1] == str(output_dir.parent)
    assert str(input_a) in command
    assert str(input_b) in command
    assert plan["hard_stops"] == []


def test_execute_timing_aligned_prediction_rerun_plan_refreshes_odds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/autopilot_x"
    output_dir.mkdir(parents=True)
    rerun_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/daily_rerun_x"
    )
    plan = {
        "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_READY_FOR_GUARDED_EXECUTION",
        "planned_output_dir": rerun_dir.relative_to(tmp_path).as_posix(),
        "planned_command": [
            "python3",
            str(tmp_path / "scripts/daily_race_ingest_shadow_orchestrator.py"),
            "--output-dir",
            str(rerun_dir),
        ],
    }

    def fake_step_command(*, name, command, output_dir, timeout_seconds):
        if name == "timing_aligned_prediction_rerun":
            rerun_dir.mkdir(parents=True)
            (rerun_dir / "shadow_predictions.jsonl").write_text(
                json.dumps({"race_id": "Race 1 - TEST - 2026-06-13"})
                + "\n",
                encoding="utf-8",
            )
            autopilot.write_json(
                rerun_dir / "shadow_manifest.json",
                {"prediction_rows": 1, "stage2_prediction_rows": 1},
            )
            stage2_dir = rerun_dir / "shadow_score_live"
            stage2_dir.mkdir()
            (stage2_dir / "stage2_shadow_predictions.jsonl").write_text(
                json.dumps({"race_id": "Race 1 - TEST - 2026-06-13"})
                + "\n",
                encoding="utf-8",
            )
        elif name == "timing_aligned_shadow_odds_snapshot":
            odds_dir = Path(command[command.index("--output-dir") + 1])
            odds_dir.mkdir(parents=True)
            autopilot.write_json(
                odds_dir / "shadow_odds_snapshot_report.json",
                {
                    "final_status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
                    "prediction_rows": 1,
                    "odds_candidate_rows": 1,
                    "valid_pre_jump_dog_odds_rows": 1,
                    "races_with_complete_valid_prejump_odds": 1,
                    "odds_research_readiness": {
                        "status": "ODDS_ANALYSIS_READY_REPORT_ONLY_EV_DISABLED",
                        "blocker_counts": {},
                    },
                    "odds_research_gate": {
                        "status": "ODDS_RESEARCH_BLOCKED_PROVENANCE",
                        "complete_valid_prejump_odds_races": 1,
                        "minimum_complete_valid_prejump_odds_races": 100,
                    },
                },
            )
        return {
            "name": name,
            "command": command,
            "returncode": 0,
            "status": "PASS",
        }

    monkeypatch.setattr(autopilot, "step_command", fake_step_command)
    steps: list[dict[str, object]] = []

    status = autopilot.execute_timing_aligned_prediction_rerun_plan(
        generated_at=datetime.fromisoformat("2026-06-13T23:40:00+10:00"),
        plan=plan,
        output_dir=output_dir,
        db_path=tmp_path / "greyhound_racing_data.db",
        current_time="2026-06-13T23:40:00+10:00",
        timeout_seconds=30,
        steps=steps,
    )

    assert status["status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_EXECUTED_WITH_ODDS_REFRESH"
    )
    assert status["execution_performed"] is True
    assert status["returncode"] == 0
    assert status["stage2_materialization_status"] == (
        "STAGE2_SHADOW_PREDICTIONS_ROOT_MATERIALIZED"
    )
    assert status["rerun_odds_snapshot_status"] == "SHADOW_ODDS_SNAPSHOT_COLLECTED"
    assert status["rerun_odds_snapshot"]["valid_pre_jump_dog_odds_rows"] == 1
    assert [step["name"] for step in steps] == [
        "timing_aligned_prediction_rerun",
        "timing_aligned_stage2_shadow_predictions_first_class",
        "timing_aligned_shadow_odds_snapshot",
    ]


def test_execute_timing_aligned_prediction_rerun_plan_preserves_blocked_hard_stops(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/autopilot_x"
    output_dir.mkdir(parents=True)

    status = autopilot.execute_timing_aligned_prediction_rerun_plan(
        generated_at=datetime.fromisoformat("2026-06-14T01:10:00+10:00"),
        plan={
            "status": "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_PLAN_BLOCKED",
            "hard_stops": ["timing_aligned_rerun_window_already_closed_after_jump"],
            "planned_output_dir": (
                "artifacts/full_evidence_orchestration_20260525/"
                "daily_race_ingest_shadow_x_timing_aligned_rerun"
            ),
        },
        output_dir=output_dir,
        db_path=tmp_path / "greyhound_racing_data.db",
        current_time="2026-06-14T01:10:00+10:00",
        timeout_seconds=30,
        steps=[],
    )

    assert status["status"] == (
        "TIMING_ALIGNED_FORWARD_SHADOW_RERUN_SKIPPED_PLAN_NOT_READY"
    )
    assert status["execution_performed"] is False
    assert status["plan_hard_stops"] == [
        "timing_aligned_rerun_window_already_closed_after_jump"
    ]
    assert status["hard_stops"] == [
        "timing_aligned_rerun_window_already_closed_after_jump"
    ]
    assert status["rerun_daily_shadow_run_dir"].endswith(
        "daily_race_ingest_shadow_x_timing_aligned_rerun"
    )


def test_timing_aligned_rerun_manifest_phase_outputs_include_artifact_dirs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(autopilot, "ROOT", tmp_path)
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/shadow_autopilot_v1_x"
    )
    rerun_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525/daily_rerun_x"
    )
    odds_dir = (
        "artifacts/full_evidence_orchestration_20260525/"
        "shadow_odds_snapshot_x_timing_aligned_rerun"
    )

    phase_outputs = autopilot.timing_aligned_rerun_manifest_phase_outputs(
        output_dir=output_dir,
        execution_status={
            "rerun_daily_shadow_run_dir": rerun_dir.as_posix(),
            "rerun_odds_snapshot_dir": odds_dir,
        },
    )

    assert phase_outputs == {
        "phase_2b_timing_aligned_prediction_rerun_plan": (
            "artifacts/full_evidence_orchestration_20260525/"
            "shadow_autopilot_v1_x/timing_aligned_prediction_rerun_plan.json"
        ),
        "phase_2b_timing_aligned_prediction_rerun_execution": (
            "artifacts/full_evidence_orchestration_20260525/"
            "shadow_autopilot_v1_x/"
            "timing_aligned_prediction_rerun_execution_status.json"
        ),
        "phase_2b_timing_aligned_prediction_rerun_output_dir": (
            "artifacts/full_evidence_orchestration_20260525/daily_rerun_x"
        ),
        "phase_2b_timing_aligned_prediction_rerun_odds_snapshot_dir": odds_dir,
    }


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
    assert any(
        "scripts/autonomous_live_odds_capture.py" in part
        for part in packet["planned_live_odds_capture_command"]
    )
    assert "--input-dir" in packet["planned_live_odds_capture_command"]
    assert "--execute" not in packet["planned_live_odds_capture_command"]
    assert "--allow-auto-scrape-odds" not in packet["planned_live_odds_capture_command"]
    assert "--execute" in packet["approved_live_odds_capture_command_template"]
    assert "--allow-auto-scrape-odds" in packet["approved_live_odds_capture_command_template"]
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
        autonomous_live_odds_capture_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_READY_NOT_EXECUTED",
            "attempted": True,
            "execute": False,
            "ready_count": 1,
            "inserted_live_odds_rows": 0,
            "capture_window_coverage_status_counts": {
                "DUE": 1,
                "MISSED": 2,
                "PENDING": 1,
            },
            "capture_window_coverage_race_count": 1,
            "capture_window_coverage_window_count": 4,
        },
    )
    readiness = {"decision": "NEED_MORE_RESULTS", "outstanding_blockers": []}
    daily_status = autopilot.build_daily_status(
        generated_at=datetime.fromisoformat("2026-06-08T23:55:00+10:00"),
        daily_manifest={"race_count": 0, "prediction_rows": 0},
        result_join_status={"latest_join": {"joined_count": 0}},
        dashboard=dashboard,
        timeseries=[],
        readiness=readiness,
        autonomous_live_odds_capture_status={
            "status": "AUTONOMOUS_LIVE_ODDS_CAPTURE_READY_NOT_EXECUTED",
            "attempted": True,
            "execute": False,
            "ready_count": 1,
            "inserted_live_odds_rows": 0,
            "capture_window_coverage_status_counts": {
                "DUE": 1,
                "MISSED": 2,
                "PENDING": 1,
            },
            "capture_window_coverage_race_count": 1,
            "capture_window_coverage_window_count": 4,
        },
        autonomous_official_result_capture_status={
            "status": "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED",
            "attempted": True,
            "candidate_count": 1,
            "official_result_race_rows": 1,
            "official_result_runner_rows": 8,
            "quarantine_rows": 0,
            "quarantined_race_ids": ["Race 8 - TAREE - 2026-06-13"],
            "quarantine_result_boxes_not_in_participants_counts": {
                "10": 1,
                "9": 1,
            },
            "quarantine_runner_set_mismatch_samples": [
                {
                    "race_id": "Race 8 - TAREE - 2026-06-13",
                    "result_boxes_not_in_participants": [9, 10],
                }
            ],
            "skipped_reason_counts": {"race_not_jumped": 1},
            "awaiting_jump_race_count": 1,
            "awaiting_jump_race_ids": ["Race 7 - CANN - 2026-06-13"],
            "awaiting_jump_next_recheck_after_local": "2026-06-13T22:55:00+10:00",
            "official_result_evidence_db_ingest_status": "NOOP_ALREADY_PRESENT",
            "official_result_evidence_db_execute": True,
            "official_result_evidence_db_write_performed": False,
            "official_result_evidence_valid_race_rows": 21,
            "official_result_evidence_valid_runner_rows": 150,
            "official_result_evidence_blocked_race_rows": 0,
            "official_result_evidence_blocked_runner_rows": 0,
            "official_result_evidence_inserted_race_rows": 0,
            "official_result_evidence_inserted_runner_rows": 0,
            "official_result_evidence_blocker_reason_counts": {},
        },
    )

    assert dashboard["next_prejump_refresh_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert dashboard["recommended_rerun_after_local"] == "2026-06-09T08:55:00+10:00"
    assert daily_status["next_prejump_refresh_status"] == "WAITING_FOR_FUTURE_WINDOW"
    assert daily_status["autonomous_live_odds_capture_window_coverage_status_counts"] == {
        "DUE": 1,
        "MISSED": 2,
        "PENDING": 1,
    }
    assert daily_status["autonomous_official_result_capture_status"] == "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED"
    assert daily_status["autonomous_official_result_runner_rows"] == 8
    assert daily_status["autonomous_official_result_quarantined_race_ids"] == [
        "Race 8 - TAREE - 2026-06-13"
    ]
    assert daily_status[
        "autonomous_official_result_quarantine_result_boxes_not_in_participants_counts"
    ] == {"10": 1, "9": 1}
    assert daily_status[
        "autonomous_official_result_quarantine_runner_set_mismatch_samples"
    ][0]["result_boxes_not_in_participants"] == [9, 10]
    assert daily_status["autonomous_official_result_skipped_reason_counts"] == {
        "race_not_jumped": 1
    }
    assert daily_status["autonomous_official_result_awaiting_jump_race_count"] == 1
    assert daily_status["autonomous_official_result_awaiting_jump_race_ids"] == [
        "Race 7 - CANN - 2026-06-13"
    ]
    assert daily_status[
        "autonomous_official_result_awaiting_jump_next_recheck_after_local"
    ] == "2026-06-13T22:55:00+10:00"
    assert (
        daily_status["autonomous_official_result_evidence_db_ingest_status"]
        == "NOOP_ALREADY_PRESENT"
    )
    assert daily_status["autonomous_official_result_evidence_db_execute"] is True
    assert (
        daily_status["autonomous_official_result_evidence_db_write_performed"]
        is False
    )
    assert daily_status["autonomous_official_result_evidence_valid_race_rows"] == 21
    assert daily_status["autonomous_official_result_evidence_valid_runner_rows"] == 150
    assert daily_status["autonomous_official_result_evidence_blocked_race_rows"] == 0
    assert daily_status["autonomous_official_result_evidence_blocked_runner_rows"] == 0
    assert daily_status["autonomous_official_result_evidence_inserted_race_rows"] == 0
    assert daily_status["autonomous_official_result_evidence_inserted_runner_rows"] == 0
    assert daily_status["autonomous_official_result_evidence_blocker_reason_counts"] == {}
    assert daily_status["next_prejump_race"]["race_id"] == "Race 1 - AP_K - 2026-06-09"
    assert "2026-06-09T08:55:00+10:00" in autopilot.daily_status_markdown(daily_status)
    assert "Autonomous odds window coverage: `{'DUE': 1, 'MISSED': 2, 'PENDING': 1}`" in autopilot.daily_status_markdown(daily_status)
    assert (
        "Autonomous official result evidence DB ingest: `NOOP_ALREADY_PRESENT`"
        in autopilot.daily_status_markdown(daily_status)
    )
    assert (
        "Autonomous official result evidence valid runner rows: `150`"
        in autopilot.daily_status_markdown(daily_status)
    )
    assert "Autonomous official result awaiting-jump races: `1`" in autopilot.daily_status_markdown(daily_status)
    assert (
        "Autonomous official result quarantine result boxes not in participants: "
        "`{'10': 1, '9': 1}`"
    ) in autopilot.daily_status_markdown(daily_status)
    assert "Autonomous official result next recheck: `2026-06-13T22:55:00+10:00`" in autopilot.daily_status_markdown(daily_status)
    assert "AUTONOMOUS_OFFICIAL_RESULT_CAPTURE_COLLECTED" in autopilot.daily_status_markdown(daily_status)
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


def test_feature_activation_data_availability_status_keeps_quarantined_live_progress(tmp_path):
    candidate_metrics = tmp_path / "candidate_eval_metrics_for_activation.json"
    activation_report = {
        "final_status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
        "kept_quarantined_features": [
            "same_distance_same_grade_best_time",
            "same_distance_same_grade_avg_time",
        ],
        "activation_allowed_features": [],
        "thresholds": {
            "min_train_present_rows": 30,
            "min_train_present_pct": 0.05,
            "min_train_unique_present_values": 5,
            "min_holdout_present_rows": 10,
            "min_holdout_present_pct": 0.05,
            "min_holdout_unique_present_values": 5,
        },
        "fail_reason_summary": {
            "reason_counts": {
                "all_missing_in_train": 2,
                "missing_shadow_metric_comparison": 2,
            }
        },
        "features": [
            {
                "feature": "same_distance_same_grade_best_time",
                "decision": "KEEP_QUARANTINED",
                "fail_reasons": [
                    "all_missing_in_train",
                    "missing_shadow_metric_comparison",
                ],
                "parity": {
                    "train_present_rows": 0,
                    "train_rows": 751,
                    "train_present_pct": 0,
                    "train_unique_present_values": 0,
                    "holdout_present_rows": 10,
                    "holdout_rows": 192,
                    "holdout_present_pct": 0.052083333333333336,
                    "holdout_unique_present_values": 10,
                },
            }
        ],
    }
    same_distance_report = {
        "status": "PASS",
        "feature_rows": 122,
        "required_source": "prior_dog_history",
        "required_history_cutoff": "strictly_before_target_race",
        "target_race_rows_allowed": 0,
        "post_outcome_rows_allowed": 0,
        "by_feature": {
            "same_distance_same_grade_best_time": {
                "status": "PASS",
                "present_rows": 6,
                "prior_history_rows_used": 8,
                "source": "prior_dog_history",
                "history_cutoff": "strictly_before_target_race",
                "target_race_rows_used": 0,
                "post_outcome_rows_used": 0,
            }
        },
    }

    status = autopilot.build_feature_activation_data_availability_status(
        activation_report=activation_report,
        same_distance_history_provenance=same_distance_report,
        inputs={"candidate_metrics": None},
    )

    assert status["status"] == "FEATURE_ACTIVATION_DATA_STILL_MISSING_KEEP_QUARANTINED"
    assert status["candidate_metric_comparison_status"] == "MISSING_OR_STALE"
    assert status["fail_reason_summary"]["reason_counts"]["all_missing_in_train"] == 2
    assert status["same_distance_history"]["feature_rows"] == 122
    assert (
        status["by_feature"]["same_distance_same_grade_best_time"][
            "live_same_distance_history"
        ]["present_rows"]
        == 6
    )
    assert status["by_feature"]["same_distance_same_grade_best_time"]["train_present_rows"] == 0
    assert status["next_data_requirement"]["min_train_present_rows"] == 30
    assert status["candidate_metrics_path"] is None

    status_with_metrics = autopilot.build_feature_activation_data_availability_status(
        activation_report=activation_report,
        same_distance_history_provenance=same_distance_report,
        inputs={"candidate_metrics": candidate_metrics},
    )
    assert status_with_metrics["candidate_metric_comparison_status"] == "AVAILABLE"


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
            "data_availability_status": {
                "status": "FEATURE_ACTIVATION_DATA_STILL_MISSING_KEEP_QUARANTINED",
                "candidate_metric_comparison_status": "MISSING_OR_STALE",
                "fail_reason_summary": {
                    "reason_counts": {"all_missing_in_train": 1}
                },
                "same_distance_history": {
                    "status": "PASS",
                    "feature_rows": 122,
                },
            },
        },
    )

    assert "## Feature Activation Gate" in summary
    assert "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED" in summary
    assert "same_distance_same_grade_best_time" in summary
    assert "FEATURE_ACTIVATION_DATA_STILL_MISSING_KEEP_QUARANTINED" in summary
    assert "Same-distance feature rows: `122`" in summary


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


def test_odds_capture_command_auto_uses_uv_when_browser_deps_missing(monkeypatch):
    monkeypatch.setattr(autopilot, "odds_capture_dependencies_available", lambda: False)
    monkeypatch.setattr(
        autopilot.shutil,
        "which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    command = autopilot.odds_capture_command_prefix("auto")

    assert command[:2] == ["/usr/bin/uv", "run"]
    assert "selenium" in command
    assert "webdriver-manager" in command
    assert command[-1] == "python"


def test_odds_capture_dependency_probe_requires_parser_and_browser_modules(monkeypatch):
    def fake_find_spec(module):
        if module == "pandas":
            return None
        return object()

    monkeypatch.setattr(autopilot.importlib.util, "find_spec", fake_find_spec)

    assert autopilot.odds_capture_dependencies_available() is False


def test_odds_capture_command_auto_fails_closed_without_browser_deps_or_uv(monkeypatch):
    monkeypatch.setattr(autopilot, "odds_capture_dependencies_available", lambda: False)
    monkeypatch.setattr(autopilot.shutil, "which", lambda name: None)

    try:
        autopilot.odds_capture_command_prefix("auto")
    except RuntimeError as exc:
        assert "odds_capture_dependencies_missing_and_uv_unavailable" in str(exc)
    else:
        raise AssertionError("expected missing odds capture dependencies to fail closed")
