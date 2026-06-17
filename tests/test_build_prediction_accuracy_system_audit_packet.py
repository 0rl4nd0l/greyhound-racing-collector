import csv
import json
from pathlib import Path

import pytest

from scripts import build_prediction_accuracy_system_audit_packet as audit


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def _base_inputs(tmp_path: Path, *, include_labels: bool = True) -> dict[str, Path]:
    evidence = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    shadow = evidence / "shadow_evaluation_runtime_repair_test_report_only"
    race_id = "Race 1 - TEST - 2026-06-17"
    predictions = [
        {
            "race_id": race_id,
            "box": 1,
            "dog_name": "Alpha Runner",
            "predicted_rank": 1,
            "shadow_rf_calibrated_probability": 0.6,
            "shadow_rf_uncalibrated_probability": 0.55,
            "calibration_method": "power_gamma_2.4",
            "output_mode": "shadow_only",
            "tgr_enabled": False,
        },
        {
            "race_id": race_id,
            "box": 2,
            "dog_name": "Beta Runner",
            "predicted_rank": 2,
            "shadow_rf_calibrated_probability": 0.4,
            "shadow_rf_uncalibrated_probability": 0.45,
            "calibration_method": "power_gamma_2.4",
            "output_mode": "shadow_only",
            "tgr_enabled": False,
        },
    ]
    features = [
        {
            "race_id": race_id,
            "box_number": 1,
            "dog_name": "Alpha Runner",
            "metadata_is_leakage_safe": True,
            "expert_form_metadata_from_sidecar": True,
            "expert_form_career_starts": 10,
            "expert_form_career_wins": 5,
            "expert_form_win_percent": 50.0,
            "expert_form_place_percent": 70.0,
            "career_win_rate": 0.5,
            "best_time_same_distance": 22.1,
            "track_condition": "Good",
            "weather": "Fine",
            "race_time_minutes_since_midnight": 740,
        },
        {
            "race_id": race_id,
            "box_number": 2,
            "dog_name": "Beta Runner",
            "metadata_is_leakage_safe": True,
            "expert_form_metadata_from_sidecar": True,
            "expert_form_career_starts": 8,
            "expert_form_career_wins": 2,
            "expert_form_win_percent": 25.0,
            "expert_form_place_percent": 50.0,
            "career_win_rate": 0.25,
            "best_time_same_distance": 22.4,
            "track_condition": "Good",
            "weather": "Fine",
            "race_time_minutes_since_midnight": 740,
        },
    ]
    results = [
        {
            "race_id": race_id,
            "box": 1,
            "dog_name": "Alpha Runner",
            "finish_position": 1,
            "is_winner": True,
            "identity_match_status": "exact_box_and_normalized_name",
        },
        {
            "race_id": race_id,
            "box": 2,
            "dog_name": "Beta Runner",
            "finish_position": 2,
            "is_winner": False,
            "identity_match_status": "exact_box_and_normalized_name",
        },
    ]
    odds = [
        {
            "race_id": race_id,
            "box": 1,
            "dog_name": "Alpha Runner",
            "predicted_rank": 1,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_provenance_status": "complete",
            "odds_snapshot": {"market_odds_win": 2.0},
        },
        {
            "race_id": race_id,
            "box": 2,
            "dog_name": "Beta Runner",
            "predicted_rank": 2,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "odds_provenance_status": "complete",
            "odds_snapshot": {"market_odds_win": 4.0},
        },
    ]
    _write_jsonl(shadow / "shadow_predictions.jsonl", predictions)
    _write_json(shadow / "shadow_feature_rows.json", features)
    _write_json(
        shadow / "shadow_manifest.json",
        {
            "active_feature_count": 73,
            "inactive_features_due_to_train_all_missing": [
                "weather",
                "track_condition",
                "same_distance_same_grade_best_time",
            ],
            "calibration_method": "power_gamma_2.4",
            "betting_output": False,
            "ev_output": False,
            "production_prediction_write": False,
            "registry_mutation": False,
            "tgr_enabled": False,
        },
    )
    result_path = _write_jsonl(
        evidence / "forward_shadow_result_join_test/joined_shadow_predictions.jsonl",
        results if include_labels else [],
    )
    odds_path = _write_jsonl(
        evidence / "shadow_odds_snapshot_test/shadow_odds_snapshot.jsonl",
        odds,
    )
    aggregate_path = _write_json(
        evidence / "forward_shadow_result_aggregate_test/forward_shadow_result_aggregate_report.json",
        {
            "final_status": "PARTIAL_AGGREGATE_PENDING_MORE_RESULTS",
            "aggregate_forward_metrics": {
                "safe_joined_race_count": 120,
                "safe_joined_runner_count": 820,
                "pending_race_count": 5,
                "unsafe_match_count": 0,
                "top1": 0.21,
                "top3": 0.55,
                "mean_winner_rank": 3.5,
                "logloss": 1.9,
                "brier": 0.12,
            },
            "aggregate_calibration_review": {
                "brier": 0.12,
                "logloss": 1.9,
                "slope_intercept": {"status": "computed", "slope": 0.4, "intercept": -1.1},
                "reliability_bins": [],
            },
        },
    )
    odds_report_path = _write_json(
        evidence / "shadow_odds_snapshot_test/shadow_odds_snapshot_report.json",
        {
            "final_status": "SHADOW_ODDS_SNAPSHOT_COLLECTED",
            "races_with_complete_valid_prejump_odds": 1,
        },
    )
    promotion_path = _write_json(
        evidence / "promotion_distance_report_test/promotion_distance_report.json",
        {
            "final_status": "PROMOTION_DISTANCE_BLOCKED",
            "promotion_ready": False,
            "blockers": ["no_candidate_passed_rank_first_accuracy_gate"],
            "market_benchmark": {
                "best_candidate_key": "market_only_implied",
                "best_non_market_candidate_key": "stage2_uncalibrated_market_blend_50",
            },
            "rolling_sample": {"sample_race_count": 120, "sample_runner_rows": 820},
        },
    )
    feature_activation_path = _write_json(
        evidence / "shadow_feature_activation_gate_test/feature_activation_gate_report.json",
        {
            "final_status": "FEATURE_ACTIVATION_BLOCKED_KEEP_QUARANTINED",
            "activation_allowed_features": [],
            "kept_quarantined_features": ["same_distance_same_grade_best_time"],
            "features": [
                {
                    "feature": "same_distance_same_grade_best_time",
                    "decision": "KEEP_QUARANTINED",
                    "fail_reasons": ["all_missing_in_train"],
                }
            ],
            "fail_reason_summary": {"reason_counts": {"all_missing_in_train": 1}},
        },
    )
    expert_path = _write_json(
        evidence / "expert_form_schema_trial_ablation_test/expert_form_schema_trial_ablation_report.json",
        {
            "final_status": "KEEP_COLLECTING_ONLY_EXPERT_FORM_ABLATION_FAILED",
            "activation_allowed": False,
            "coverage_summary": {"feature_rows": 2},
            "control_metrics": {"status": "EVALUATED", "top1": 1.0, "top3": 1.0},
            "trial_metrics": {"status": "EVALUATED", "top1": 0.5, "top3": 1.0},
            "market_metrics": {"status": "EVALUATED", "top1": 1.0, "top3": 1.0},
        },
    )
    return {
        "evidence": evidence,
        "shadow": shadow,
        "result": result_path,
        "odds": odds_path,
        "aggregate": aggregate_path,
        "odds_report": odds_report_path,
        "promotion": promotion_path,
        "feature_activation": feature_activation_path,
        "expert": expert_path,
    }


def test_accuracy_system_audit_joins_labels_features_and_strict_odds(tmp_path):
    paths = _base_inputs(tmp_path)

    report, ledgers = audit.build_packet(
        evidence_root=paths["evidence"],
        shadow_run_dir=paths["shadow"],
        result_join_path=paths["result"],
        odds_snapshot_path=paths["odds"],
        aggregate_report_path=paths["aggregate"],
        promotion_distance_report_path=paths["promotion"],
        feature_activation_report_path=paths["feature_activation"],
        expert_form_ablation_report_path=paths["expert"],
        min_meaningful_races=1,
    )

    assert report["decision"]["runtime_status"] == "SHADOW_SCORER_RUNTIME_REPAIRED"
    assert report["decision"]["next_decision"] == "READY_FOR_REPORT_ONLY_MODEL_TOURNAMENT"
    assert report["counts"]["joined_label_rows"] == 2
    assert report["counts"]["joined_valid_prejump_odds_rows"] == 2
    joined = ledgers["joined_runner_evaluation"]
    winner = next(row for row in joined if row["box"] == 1)
    assert winner["is_winner"] is True
    assert winner["label_join_status"] == "JOINED_OFFICIAL_RESULT"
    assert winner["feature_row_join_status"] == "JOINED_FEATURE_ROW"
    assert winner["metadata_is_leakage_safe"] is True
    assert winner["odds_join_status"] == "valid_pre_jump_dog_odds"
    assert winner["market_implied_probability_normalized"] == pytest.approx(2 / 3)

    expert_market = next(
        row
        for row in ledgers["odds_market_baseline_ledger"]
        if row["scope"] == "current_expert_form_sidecar_rows"
    )
    assert expert_market["market_metrics_status"] == "COMPUTED"
    assert expert_market["market_top1"] == 1.0

    feature_families = {
        row["feature_family"]: row for row in ledgers["feature_family_coverage_utility"]
    }
    assert feature_families["weather_track"]["rows_with_any_present"] == 2
    assert "weather" in feature_families["weather_track"]["inactive_train_all_missing_fields"]
    assert feature_families["expert_form"]["coverage_pct"] == 1.0


def test_accuracy_system_audit_fails_closed_when_current_labels_are_missing(tmp_path):
    paths = _base_inputs(tmp_path, include_labels=False)

    report, ledgers = audit.build_packet(
        evidence_root=paths["evidence"],
        shadow_run_dir=paths["shadow"],
        result_join_path=paths["result"],
        odds_snapshot_path=paths["odds"],
        aggregate_report_path=paths["aggregate"],
        promotion_distance_report_path=paths["promotion"],
        feature_activation_report_path=paths["feature_activation"],
        expert_form_ablation_report_path=paths["expert"],
        min_meaningful_races=200,
    )

    assert report["decision"]["runtime_status"] == "SHADOW_SCORER_RUNTIME_REPAIRED"
    assert report["decision"]["next_decision"] == "KEEP_COLLECTING_ONLY"
    assert report["counts"]["joined_label_rows"] == 0
    assert report["counts"]["joined_valid_prejump_odds_rows"] == 2
    missing = {row["metric"]: row for row in ledgers["missingness_ledger"]}
    assert missing["official_label_joined"]["status"] == "DATA_MISSING"
    assert missing["strict_prejump_odds_joined"]["status"] == "PASS"


def test_accuracy_system_audit_run_packet_writes_report_only_and_preserves_protected_paths(
    tmp_path, monkeypatch
):
    paths = _base_inputs(tmp_path)
    protected_db = tmp_path / "greyhound_racing_data.db"
    protected_db.write_text("do-not-change", encoding="utf-8")
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "DEFAULT_PROTECTED_PATHS", (protected_db,))
    output_dir = (
        tmp_path
        / "artifacts/full_evidence_orchestration_20260525"
        / "prediction_accuracy_system_audit_test_report_only"
    )

    result = audit.run_packet(
        evidence_root=paths["evidence"],
        output_dir=output_dir,
        shadow_run_dir=paths["shadow"],
        result_join_path=paths["result"],
        odds_snapshot_path=paths["odds"],
        aggregate_report_path=paths["aggregate"],
        promotion_distance_report_path=paths["promotion"],
        feature_activation_report_path=paths["feature_activation"],
        expert_form_ablation_report_path=paths["expert"],
        min_meaningful_races=1,
    )

    assert result["final_status"] == "READY_FOR_REPORT_ONLY_MODEL_TOURNAMENT"
    assert result["protected_paths_unchanged"] is True
    assert protected_db.read_text(encoding="utf-8") == "do-not-change"
    assert (output_dir / "prediction_accuracy_system_audit_report.json").exists()
    assert (output_dir / "joined_runner_evaluation.csv").exists()
    assert (output_dir / "BOARD_READY_RECOMMENDATION.md").exists()
    assert (output_dir / "output_manifest.json").exists()
    with (output_dir / "joined_runner_evaluation.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["feature_row_join_status"] == "JOINED_FEATURE_ROW"


def test_protected_directory_digest_detects_same_size_content_change(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    protected_dir = tmp_path / "predictions"
    protected_dir.mkdir()
    protected_file = protected_dir / "snapshot.jsonl"
    protected_file.write_text("aaaa\n", encoding="utf-8")

    before = audit.path_state(protected_dir)
    protected_file.write_text("bbbb\n", encoding="utf-8")
    after = audit.path_state(protected_dir)

    assert before["type"] == "directory"
    assert before["file_count"] == 1
    assert before["file_count"] == after["file_count"]
    assert protected_file.stat().st_size == 5
    assert before["listing_sha256"] != after["listing_sha256"]


def test_metric_summary_counts_only_metric_safe_runner_rows():
    rows = [
        {
            "race_id": "Race 1",
            "dog_name": "Alpha",
            "is_winner": True,
            "predicted_rank": 1,
            audit.PROBABILITY_COLUMN: 0.7,
        },
        {
            "race_id": "Race 1",
            "dog_name": "Beta",
            "is_winner": False,
            "predicted_rank": 2,
            audit.PROBABILITY_COLUMN: 0.3,
        },
        {
            "race_id": "Race 2",
            "dog_name": "Gamma",
            "is_winner": None,
            "predicted_rank": 1,
            audit.PROBABILITY_COLUMN: 0.8,
        },
        {
            "race_id": "Race 2",
            "dog_name": "Delta",
            "is_winner": None,
            "predicted_rank": 2,
            audit.PROBABILITY_COLUMN: 0.2,
        },
    ]

    summary = audit.metric_summary(rows, audit.PROBABILITY_COLUMN, "predicted_rank")

    assert summary["status"] == "COMPUTED"
    assert summary["safe_race_count"] == 1
    assert summary["safe_runner_count"] == 2
    assert summary["top1"] == 1.0
    assert summary["skipped_race_reason_counts"] == {"missing_labels": 1}


def test_accuracy_system_audit_rejects_unsafe_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    with pytest.raises(
        ValueError,
        match="output_dir_must_be_prediction_accuracy_system_audit_artifact",
    ):
        audit.assert_output_dir_safe(tmp_path / "artifacts/prediction_snapshots/audit")
