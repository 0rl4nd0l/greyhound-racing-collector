import hashlib
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from flask import Flask

from src.operator_ui.deployment import DeploymentRejected, generate_package
from src.operator_ui.security import load_connected_environment
import src.operator_ui.bootstrap as bootstrap_module


COMMIT = "881a3cee0c7f93dd26f5ece9185052f59c4c1aed"
TREE = "226dae42ebc4deea7ce1c7954e8da74fabd37f7a"


def deployment_inputs(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source"
    repository = Path(__file__).parents[2]
    fixed = (
        "configs/operator_ui/repository-v1.toml",
        "configs/prediction/manual-default.json",
        "scripts/predict_race_now.py",
        "artifacts/frozen_models/market_form_residual_v1/model.json",
        "artifacts/frozen_models/market_form_residual_v1/manifest.json",
        "configs/prediction/schemas/market_form_residual_v1.schema.json",
        "app.py",
    )
    for relative in fixed:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repository / relative).read_bytes())
    evidence = tmp_path / "evidence"
    producer = tmp_path / "producer"
    operations = tmp_path / "operations"
    output = tmp_path / "package"
    for directory in (source, evidence, producer, operations, output):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o700)
    (evidence / "shadow_autopilot_daemon_runtime").mkdir(mode=0o700)
    (evidence / "manual_prediction_collector_requests_v1").mkdir(mode=0o700)
    (evidence / "shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json").write_text("{}")
    (producer / "artifacts/on_demand_prediction_runs").mkdir(parents=True, mode=0o700)
    python = tmp_path / "python"
    python.write_text("python")
    python.chmod(0o700)
    database = tmp_path / "canonical.sqlite3"
    database.write_text("canonical")
    database.chmod(0o400)
    secrets = tmp_path / "operator-ui.secrets"
    secrets.write_text("OPERATOR_UI_SECRET_KEY=not-copied\nOPERATOR_UI_USERNAME=operator\nOPERATOR_UI_PASSWORD_HASH=scrypt:example\n")
    secrets.chmod(0o600)
    return dict(source_root=source, pinned_python=python, evidence_root=evidence,
                producer_root=producer, canonical_db=database,
                operations_root=operations, secrets_file=secrets,
                output_dir=output, source_commit=COMMIT, source_tree=TREE,
                ui_version="operator-ui-v1", profile_id="repository-v1",
                bind_address="127.0.0.1", port=5055)


def git_identity(monkeypatch, *, commit=COMMIT, tree=TREE, dirty=False):
    def run(command, **kwargs):
        if "status" in command:
            output = " M app.py\n" if dirty else ""
        else:
            output = f"{commit}\n{tree}\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("src.operator_ui.deployment.subprocess.run", run)


@pytest.fixture
def real_startup_tmp_path():
    repository = Path(__file__).parents[2]
    with TemporaryDirectory(prefix=".operator-ui-startup-", dir=repository) as temporary:
        fixture_root = Path(temporary)
        fixture_root.chmod(0o700)
        yield fixture_root


def test_default_off_package_binds_identity_hashes_private_service_and_external_secrets(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    result = generate_package(**values)
    binding_path = values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json"
    binding = json.loads(binding_path.read_text())
    assert binding["deployment"] == {"source_commit": COMMIT, "source_tree": TREE,
                                      "ui_version": "operator-ui-v1", "profile_id": "repository-v1"}
    assert set(binding["artifacts"]) == {"prediction_script", "prediction_config", "model_artifact", "model_manifest", "model_schema"}
    assert all(len(value) == 64 for value in binding["artifacts"].values())
    assert binding["profile_sha256"] == hashlib.sha256((values["source_root"] / "configs/operator_ui/repository-v1.toml").read_bytes()).hexdigest()
    environment = (values["output_dir"] / "operator-ui-r3.env").read_text()
    service = (values["output_dir"] / "greyhound-operator-ui-r3.service").read_text()
    assert "OPERATOR_UI_CONNECTED_MODE=0" in environment
    assert "OPERATOR_UI_R3_PROFILE=disabled" in environment
    assert "127.0.0.1" in service and "--port 5055" in service
    assert f"EnvironmentFile={values['secrets_file']}" in service
    assert "not-copied" not in service + environment
    assert result["enabled"] is False


@pytest.mark.parametrize("unsafe", ["public_bind", "symlink_python", "overlap", "missing_index", "weak_secrets"])
def test_generator_rejects_unsafe_missing_or_overlapping_inputs_without_partial_writes(tmp_path, monkeypatch, unsafe):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    if unsafe == "public_bind":
        values["bind_address"] = "0.0.0.0"
    elif unsafe == "symlink_python":
        python = values["pinned_python"]
        target = python.with_name("python-target")
        python.rename(target)
        python.symlink_to(target)
    elif unsafe == "overlap":
        values["operations_root"] = values["evidence_root"]
    elif unsafe == "missing_index":
        (values["evidence_root"] / "shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json").unlink()
    else:
        values["secrets_file"].chmod(0o644)
    with pytest.raises(DeploymentRejected):
        generate_package(**values)
    assert not (values["output_dir"] / "greyhound-operator-ui-r3.service").exists()
    assert not (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").exists()


def test_explicit_enable_changes_only_feature_gate_and_retains_evidence_on_rollback(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    result = generate_package(**values, enabled=True)
    environment = (values["output_dir"] / "operator-ui-r3.env").read_text()
    rollback = (values["output_dir"] / "ROLLBACK.md").read_text()
    assert "OPERATOR_UI_CONNECTED_MODE=1" in environment
    assert "OPERATOR_UI_R3_PROFILE=repository-v1" in environment
    assert "disable" in rollback.lower()
    assert "do not delete" in rollback.lower()
    assert str(values["operations_root"]) in rollback
    assert result["enabled"] is True


@pytest.mark.parametrize("enabled, expected", [(False, False), (True, True)])
def test_real_generated_package_startup_is_disabled_or_bootstraps_with_all_deployment_identity(
    real_startup_tmp_path, monkeypatch, enabled, expected
):
    values = deployment_inputs(real_startup_tmp_path)
    git_identity(monkeypatch)
    generate_package(**values, enabled=enabled)
    generated = dict(
        line.split("=", 1)
        for line in (values["output_dir"] / "operator-ui-r3.env").read_text().splitlines()
        if line
    )
    for name, value in generated.items():
        monkeypatch.setenv(name, value)
    app = Flask(__name__)
    app.config[bootstrap_module.R3_PROFILE_KEY] = generated["OPERATOR_UI_R3_PROFILE"]
    load_connected_environment(app)
    monkeypatch.setattr(bootstrap_module, "_REPOSITORY_ROOT", values["source_root"])
    assert bootstrap_module.configure_r3_startup(app) is expected
    if enabled:
        assert {
            app.config[name]
            for name in (
                "OPERATOR_UI_DEPLOYED_COMMIT",
                "OPERATOR_UI_DEPLOYED_TREE",
                "OPERATOR_UI_DEPLOYED_VERSION",
                "OPERATOR_UI_DEPLOYED_PROFILE",
            )
        } == {COMMIT, TREE, "operator-ui-v1", "repository-v1"}


@pytest.mark.parametrize("identity", ["dirty", "commit", "tree"])
def test_generator_rejects_unclean_or_mismatched_git_identity_before_any_write(tmp_path, monkeypatch, identity):
    values = deployment_inputs(tmp_path)
    git_identity(
        monkeypatch,
        commit="a" * 40 if identity == "commit" else COMMIT,
        tree="b" * 40 if identity == "tree" else TREE,
        dirty=identity == "dirty",
    )
    with pytest.raises(DeploymentRejected, match="Git identity"):
        generate_package(**values)
    assert not (values["output_dir"] / "greyhound-operator-ui-r3.service").exists()
    assert not (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").exists()


@pytest.mark.parametrize("bad", ["path with space", "path%percent", 'path\"quote', "path\nnewline", "path:colon"])
def test_generator_rejects_systemd_ambiguous_authority_paths(tmp_path, monkeypatch, bad):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    unsafe = tmp_path / bad
    unsafe.mkdir()
    unsafe.chmod(0o700)
    values["operations_root"] = unsafe
    with pytest.raises(DeploymentRejected, match="systemd-safe"):
        generate_package(**values)
    assert not (values["output_dir"] / "greyhound-operator-ui-r3.service").exists()
