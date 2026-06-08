import json

import pytest

from scripts.capture_prediction_snapshot import (
    _prediction_preview,
    _report_only_calibration_from_design,
)


def test_prediction_preview_is_ranked_and_result_free():
    snapshot = {
        "predictions": [
            {
                "predicted_rank": 2,
                "box_number": 5,
                "dog_name": "Beta",
                "win_prob_norm": 0.3,
                "finish_position": 1,
            },
            {
                "predicted_rank": 1,
                "box_number": 2,
                "dog_clean_name": "Alpha",
                "win_prob_norm": 0.7,
                "market_odds_win": None,
                "ev_win": None,
            },
        ]
    }

    preview = _prediction_preview(snapshot)

    assert [row["dog_name"] for row in preview] == ["Alpha", "Beta"]
    assert [row["predicted_rank"] for row in preview] == [1, 2]
    assert all("finish_position" not in row for row in preview)
    assert preview[0]["market_odds_win"] is None
    assert preview[0]["ev_win"] is None


def test_prediction_preview_shows_attached_snapshot_odds_when_ev_is_present():
    snapshot = {
        "predictions": [
            {
                "predicted_rank": 1,
                "box_number": 1,
                "dog_name": "Alpha",
                "win_prob_norm": 0.4,
                "odds": 9.0,
                "odds_snapshot": {"market_odds_win": 9.0},
                "odds_match_status": "valid_pre_jump_dog_odds",
                "ev_win": 2.6,
            }
        ]
    }

    preview = _prediction_preview(snapshot)

    assert preview == [
        {
            "predicted_rank": 1,
            "box_number": 1,
            "dog_name": "Alpha",
            "win_prob_norm": 0.4,
            "odds_match_status": "valid_pre_jump_dog_odds",
            "market_odds_win": 9.0,
            "ev_win": 2.6,
            "quality_flags": [],
        }
    ]


def test_report_only_calibration_design_loader_returns_snapshot_spec(tmp_path):
    design = {
        "schema_version": "calibration_layer_design_v1",
        "status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
        "failures": [],
        "runtime_transform_spec": {
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "rank_preserving_when_alpha_positive": True,
            "requires_runner_complete_race_group": True,
        },
        "deployment_control": {
            "promotion_allowed": False,
            "registry_mutation_allowed": False,
            "model_artifact_written": False,
            "production_config_write_allowed": False,
            "betting_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
        },
    }
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")

    spec = _report_only_calibration_from_design(path)

    assert spec == {
        "algorithm": "power_normalize_per_race",
        "alpha": 0.5,
        "input_probability_key": "win_prob_norm",
        "output_probability_key": "calibrated_win_prob_report_only",
        "source_design_path": str(path),
        "source_schema_version": "calibration_layer_design_v1",
        "source_status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
    }


def test_report_only_calibration_design_loader_fails_closed_on_write_controls(
    tmp_path,
):
    design = {
        "schema_version": "calibration_layer_design_v1",
        "status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
        "failures": [],
        "runtime_transform_spec": {
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "rank_preserving_when_alpha_positive": True,
            "requires_runner_complete_race_group": True,
        },
        "deployment_control": {
            "promotion_allowed": True,
            "registry_mutation_allowed": False,
            "model_artifact_written": False,
            "production_config_write_allowed": False,
            "betting_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
        },
    }
    path = tmp_path / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration_promotion_not_blocked"):
        _report_only_calibration_from_design(path)


def test_report_only_calibration_loader_accepts_runtime_config(tmp_path):
    config = {
        "schema_version": "runtime_calibration_config_v1",
        "status": "ACTIVE_REPORT_ONLY",
        "runtime_transform_spec": {
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "rank_preserving_when_alpha_positive": True,
            "requires_runner_complete_race_group": True,
        },
        "runtime_scope": {
            "canonical_probability_key_unchanged": "win_prob_norm",
            "canonical_rank_unchanged": True,
            "report_only": True,
        },
        "deployment_control": {
            "model_artifact_written": False,
            "model_registry_index_mutated": False,
            "best_model_symlinks_mutated": False,
            "label_write": False,
            "betting": False,
            "required_env_var": "APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR",
        },
    }
    path = tmp_path / "runtime_calibration.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    spec = _report_only_calibration_from_design(path)

    assert spec == {
        "algorithm": "power_normalize_per_race",
        "alpha": 0.5,
        "input_probability_key": "win_prob_norm",
        "output_probability_key": "calibrated_win_prob_report_only",
        "source_design_path": str(path),
        "source_schema_version": "runtime_calibration_config_v1",
        "source_status": "ACTIVE_REPORT_ONLY",
    }


def test_report_only_calibration_loader_rejects_runtime_config_registry_mutation(
    tmp_path,
):
    config = {
        "schema_version": "runtime_calibration_config_v1",
        "status": "ACTIVE_REPORT_ONLY",
        "runtime_transform_spec": {
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "rank_preserving_when_alpha_positive": True,
            "requires_runner_complete_race_group": True,
        },
        "runtime_scope": {
            "canonical_probability_key_unchanged": "win_prob_norm",
            "canonical_rank_unchanged": True,
            "report_only": True,
        },
        "deployment_control": {
            "model_artifact_written": False,
            "model_registry_index_mutated": True,
            "best_model_symlinks_mutated": False,
            "label_write": False,
            "betting": False,
            "required_env_var": "APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR",
        },
    }
    path = tmp_path / "runtime_calibration.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="runtime_calibration_registry_index_mutated",
    ):
        _report_only_calibration_from_design(path)
