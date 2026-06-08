import json
from pathlib import Path

from scripts.shadow_feature_audit_packet import (
    SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME,
    copy_shadow_feature_audit_reports,
    ensure_same_distance_history_provenance_report,
    feature_activation_gate_input_paths,
    waiting_same_distance_history_provenance_report,
)


def test_waiting_same_distance_history_provenance_report_is_explicit_no_write():
    report = waiting_same_distance_history_provenance_report()

    assert report["status"] == "NOT_POPULATED"
    assert report["live_input_status"] == "NO_ELIGIBLE_PREJUMP_RACES"
    assert report["target_race_rows_allowed"] == 0
    assert report["post_outcome_rows_allowed"] == 0
    assert report["no_write_guarantees"] == {
        "betting_or_ev_output": False,
        "canonical_schema_mutation": False,
        "db_write": False,
        "label_write": False,
        "production_prediction_write": False,
    }
    assert (
        report["by_feature"]["same_distance_same_grade_best_time"]["status"]
        == "NOT_POPULATED"
    )


def test_ensure_same_distance_history_provenance_prefers_score_live_report(tmp_path):
    output_dir = tmp_path / "daily"
    score_dir = tmp_path / "score"
    output_dir.mkdir()
    score_dir.mkdir()
    source = {
        "schema_version": "same_distance_same_grade_history_provenance_v1",
        "status": "PASS",
    }
    (score_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME).write_text(
        json.dumps(source),
        encoding="utf-8",
    )

    target = ensure_same_distance_history_provenance_report(
        output_dir=output_dir,
        score_output_dir=score_dir,
    )

    assert target == output_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
    assert json.loads(target.read_text()) == source


def test_ensure_same_distance_history_provenance_writes_waiting_report(tmp_path):
    output_dir = tmp_path / "daily"
    output_dir.mkdir()

    target = ensure_same_distance_history_provenance_report(
        output_dir=output_dir,
        score_output_dir=None,
    )

    report = json.loads(target.read_text())
    assert report["status"] == "NOT_POPULATED"
    assert report["live_input_status"] == "NO_ELIGIBLE_PREJUMP_RACES"


def test_copy_shadow_feature_audit_reports_uses_daily_packet_names(tmp_path):
    score_dir = tmp_path / "score"
    output_dir = tmp_path / "daily"
    score_dir.mkdir()
    output_dir.mkdir()
    (score_dir / "shadow_feature_population_report.json").write_text(
        "{\"status\":\"PASS\"}",
        encoding="utf-8",
    )
    (score_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME).write_text(
        "{\"status\":\"PASS\"}",
        encoding="utf-8",
    )

    copy_shadow_feature_audit_reports(score_dir, output_dir)

    assert (output_dir / "feature_population_report.json").exists()
    assert (output_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME).exists()


def test_feature_activation_gate_input_paths_prefers_live_then_daily(tmp_path):
    daily_dir = tmp_path / "daily"
    score_live_dir = daily_dir / "shadow_score_live"
    model_dir = tmp_path / "model"
    score_live_dir.mkdir(parents=True)
    model_dir.mkdir()
    shadow_model = model_dir / "shadow_randomforest_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    model_parity = model_dir / "train_eval_feature_parity_report.json"
    live_policy = score_live_dir / "active_feature_policy_report.json"
    live_same_distance = score_live_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
    daily_same_distance = daily_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
    baseline_metrics = tmp_path / "baseline_metrics.json"
    candidate_metrics = tmp_path / "candidate_metrics.json"
    for path in (
        model_parity,
        live_policy,
        live_same_distance,
        daily_same_distance,
        baseline_metrics,
        candidate_metrics,
    ):
        path.write_text("{}", encoding="utf-8")

    inputs = feature_activation_gate_input_paths(
        daily_dir=daily_dir,
        shadow_model=shadow_model,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )

    assert inputs["parity_report"] == model_parity
    assert inputs["inactive_policy_report"] == live_policy
    assert inputs["same_distance_history_provenance"] == live_same_distance
    assert inputs["baseline_metrics"] == baseline_metrics
    assert inputs["candidate_metrics"] == candidate_metrics


def test_feature_activation_gate_input_paths_falls_back_to_daily_same_distance(tmp_path):
    daily_dir = tmp_path / "daily"
    model_dir = tmp_path / "model"
    daily_dir.mkdir()
    model_dir.mkdir()
    shadow_model = model_dir / "shadow_randomforest_model.joblib"
    shadow_model.write_text("model", encoding="utf-8")
    daily_same_distance = daily_dir / SAME_DISTANCE_HISTORY_PROVENANCE_FILENAME
    daily_same_distance.write_text("{}", encoding="utf-8")

    inputs = feature_activation_gate_input_paths(
        daily_dir=daily_dir,
        shadow_model=shadow_model,
    )

    assert inputs["same_distance_history_provenance"] == daily_same_distance


def test_feature_activation_gate_input_paths_does_not_reuse_baseline_as_candidate(tmp_path):
    baseline_metrics = tmp_path / "aggregate_forward_metrics.json"
    baseline_metrics.write_text("{}", encoding="utf-8")

    inputs = feature_activation_gate_input_paths(
        daily_dir=None,
        shadow_model=None,
        baseline_metrics=baseline_metrics,
        candidate_metrics=None,
    )

    assert inputs["baseline_metrics"] == baseline_metrics
    assert inputs["candidate_metrics"] is None
