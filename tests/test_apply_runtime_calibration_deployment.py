import json
from pathlib import Path

from scripts.apply_runtime_calibration_deployment import build_execution


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _design() -> dict:
    return {
        "schema_version": "calibration_layer_design_v1",
        "status": "READY_FOR_OPERATOR_DESIGN_REVIEW",
        "failures": [],
        "runtime_transform_spec": {
            "candidate_arm": "power_calibrated_baseline",
            "algorithm": "power_normalize_per_race",
            "alpha": 0.5,
            "input_probability_key": "win_prob_norm",
            "output_probability_key": "calibrated_win_prob_report_only",
            "rank_preserving_when_alpha_positive": True,
            "uses_labels_at_runtime": False,
            "uses_odds_at_runtime": False,
            "requires_runner_complete_race_group": True,
        },
        "deployment_control": {
            "action_taken": "none",
            "model_artifact_written": False,
            "registry_mutation_allowed": False,
            "production_config_write_allowed": False,
            "promotion_allowed": False,
            "required_gate": "APPROVE_MODEL_PROMOTION",
            "betting_allowed": False,
        },
    }


def _deployment_plan(design_path: Path) -> dict:
    return {
        "schema_version": "calibration_deployment_plan_v1",
        "status": "READY_FOR_SEPARATE_PROMOTION_IMPLEMENTATION_REVIEW",
        "failures": [],
        "source_evidence": {
            "calibration_design": str(design_path),
            "model_review_packet": "model_packet.json",
            "prejump_loop_plan": "loop_plan.json",
        },
        "deployment_controls": {
            "actual_promotion_command_ready": False,
            "promotion_allowed": False,
            "model_artifact_write_allowed": False,
            "registry_mutation_allowed": False,
            "production_config_write_allowed": False,
            "betting_allowed": False,
        },
        "actual_promotion_command": None,
        "writes_performed": {
            "label_write": False,
            "model_artifact_write": False,
            "registry_mutation": False,
            "production_config_write": False,
            "refresh_signal_write": False,
            "betting": False,
        },
        "loop_pass_through": {
            "dry_run_capture": True,
            "approved_persist_capture": True,
            "live_odds_capture": True,
            "persist_same_run": True,
            "live_odds_same_run": True,
        },
    }


def test_runtime_calibration_deployment_dry_run_has_no_writes(tmp_path, monkeypatch):
    monkeypatch.delenv("APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR", raising=False)
    design_path = tmp_path / "design.json"
    plan_path = tmp_path / "deployment_plan.json"
    config_path = tmp_path / "model_registry" / "runtime_calibration.json"
    signal_path = tmp_path / "model_registry" / "refresh_signal.json"
    _write_json(design_path, _design())
    _write_json(plan_path, _deployment_plan(design_path))

    report = build_execution(
        deployment_plan_path=plan_path,
        config_path=config_path,
        refresh_signal_path=signal_path,
        backup_dir=tmp_path / "backups",
    )

    assert report["status"] == "DRY_RUN_READY"
    assert report["failures"] == []
    assert report["writes_performed"] == {
        "runtime_calibration_config": False,
        "refresh_signal": False,
        "model_artifact_write": False,
        "model_registry_index_mutation": False,
        "best_model_symlink_mutation": False,
        "label_write": False,
        "betting": False,
    }
    assert not config_path.exists()
    assert not signal_path.exists()
    assert report["runtime_calibration_config_preview"]["status"] == "ACTIVE_REPORT_ONLY"
    assert report["approved_loop_command_template"][
        report["approved_loop_command_template"].index(
            "--report-only-calibration-design"
        )
        + 1
    ] == str(config_path.resolve())


def test_runtime_calibration_deployment_write_requires_executor_env(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR", raising=False)
    design_path = tmp_path / "design.json"
    plan_path = tmp_path / "deployment_plan.json"
    config_path = tmp_path / "model_registry" / "runtime_calibration.json"
    signal_path = tmp_path / "model_registry" / "refresh_signal.json"
    _write_json(design_path, _design())
    _write_json(plan_path, _deployment_plan(design_path))

    report = build_execution(
        deployment_plan_path=plan_path,
        config_path=config_path,
        refresh_signal_path=signal_path,
        backup_dir=tmp_path / "backups",
        write_approved=True,
    )

    assert report["status"] == "NOT_READY"
    assert "write_requested_without_executor_env_approval" in report["failures"]
    assert not config_path.exists()
    assert not signal_path.exists()


def test_runtime_calibration_deployment_approved_write_is_scoped(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("APPROVE_MODEL_PROMOTION_FOR_SEPARATE_EXECUTOR", "approved")
    design_path = tmp_path / "design.json"
    plan_path = tmp_path / "deployment_plan.json"
    config_path = tmp_path / "model_registry" / "runtime_calibration.json"
    signal_path = tmp_path / "model_registry" / "refresh_signal.json"
    backup_dir = tmp_path / "backups"
    _write_json(design_path, _design())
    _write_json(plan_path, _deployment_plan(design_path))
    _write_json(config_path, {"previous": True})

    report = build_execution(
        deployment_plan_path=plan_path,
        config_path=config_path,
        refresh_signal_path=signal_path,
        backup_dir=backup_dir,
        write_approved=True,
    )

    assert report["status"] == "ACTIVE_REPORT_ONLY"
    assert report["writes_performed"]["runtime_calibration_config"] is True
    assert report["writes_performed"]["refresh_signal"] is True
    assert report["writes_performed"]["model_registry_index_mutation"] is False
    assert report["writes_performed"]["best_model_symlink_mutation"] is False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    signal = json.loads(signal_path.read_text(encoding="utf-8"))
    assert config["schema_version"] == "runtime_calibration_config_v1"
    assert config["runtime_scope"]["report_only"] is True
    assert signal["schema_version"] == "runtime_calibration_refresh_signal_v1"
    backup_path = Path(report["rollback"]["backup_path"])
    assert backup_path.exists()
    assert json.loads(backup_path.read_text(encoding="utf-8")) == {"previous": True}


def test_runtime_calibration_deployment_fails_closed_on_prior_writes(tmp_path):
    design_path = tmp_path / "design.json"
    plan_path = tmp_path / "deployment_plan.json"
    plan = _deployment_plan(design_path)
    plan["writes_performed"]["registry_mutation"] = True
    _write_json(design_path, _design())
    _write_json(plan_path, plan)

    report = build_execution(
        deployment_plan_path=plan_path,
        config_path=tmp_path / "runtime_calibration.json",
        refresh_signal_path=tmp_path / "refresh_signal.json",
        backup_dir=tmp_path / "backups",
    )

    assert report["status"] == "NOT_READY"
    assert "prior_write_performed:registry_mutation" in report["failures"]
