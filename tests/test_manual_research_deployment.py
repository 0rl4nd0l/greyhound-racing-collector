import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

import pytest

from src.predictor.manual_research_deployment import (
    ManualDeploymentRejected,
    generate_manual_package,
)

PROTECTED_NAMES = (
    "autonomous_browser_profile",
    "autonomous_shared_lock",
    "canonical_database",
    "canonical_history",
    "live_odds",
    "forward_corpus",
    "collector_requests",
    "collector_state",
    "result_evidence",
    "services",
    "timers",
)


def _git_identity(monkeypatch, commit: str, tree: str) -> None:
    def run(command, **kwargs):
        output = "" if "status" in command else f"{commit}\n{tree}\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("src.predictor.manual_research_deployment.subprocess.run", run)


@pytest.fixture
def deployment_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repository = Path(__file__).parents[1]
    source = tmp_path / "source"
    for relative in (
        "ops/systemd/manual-research-api.service.in",
        "src/predictor/manual_research_cli.py",
        "src/predictor/manual_research_worker.py",
        "src/predictor/manual_live_capture.py",
        "src/predictor/manual_live_capture_child.py",
        "configs/prediction/manual-default.json",
        "artifacts/frozen_models/market_form_residual_v1/model.json",
        "artifacts/frozen_models/market_form_residual_v1/manifest.json",
    ):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    source.mkdir(exist_ok=True)
    manual = tmp_path / "manual-operations"
    profile = manual / "browser-profile"
    runs = manual / "runs"
    output = tmp_path / "package"
    for directory in (manual, profile, runs, output):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o700)
    protected_root = tmp_path / "protected"
    protected_root.mkdir()
    protected = {}
    for index, name in enumerate(PROTECTED_NAMES):
        if name in {"canonical_database", "autonomous_shared_lock"}:
            path = protected_root / f"{name}.sentinel"
            path.write_bytes(name.encode())
            path.chmod(0o600)
        else:
            path = protected_root / name
            path.mkdir()
            path.chmod(0o700)
        protected[name] = path
    model = source / "artifacts/frozen_models/market_form_residual_v1/model.json"
    manifest = source / "artifacts/frozen_models/market_form_residual_v1/manifest.json"
    config = source / "configs/prediction/manual-default.json"
    commit = "a" * 40
    tree = "b" * 40
    _git_identity(monkeypatch, commit, tree)
    return {
        "source_root": source,
        "pinned_python": Path(sys.executable).resolve(),
        "manual_root": manual,
        "browser_profile_root": profile,
        "manual_runs_root": runs,
        "manual_lock": manual / "manual-capture.lock",
        "model": model,
        "model_manifest": manifest,
        "config": config,
        "output_dir": output,
        "source_commit": commit,
        "source_tree": tree,
        "protected_paths": protected,
    }


def test_default_off_package_isolated_and_template_bound(deployment_inputs):
    values = deployment_inputs
    autonomous_before = {
        path: path.read_bytes()
        for path in values["source_root"].glob("ops/systemd/*")
        if path.is_file()
    }
    result = generate_manual_package(**values)
    output = values["output_dir"]
    service = (output / "greyhound-manual-research.service").read_text()
    environment = (output / "manual-research.env").read_text()
    binding = json.loads((output / "manual-research.binding.json").read_text())

    assert result["enabled"] is False
    assert "MANUAL_RESEARCH_ENABLED=0" in environment
    assert "MANUAL_RESEARCH_OPERATIONS_ROOT=" + str(values["manual_root"]) in environment
    assert "canonical_database" not in environment
    assert "canonical_database" not in service.split("ExecStart=", 1)[1].splitlines()[0]
    for protected_path in values["protected_paths"].values():
        assert str(protected_path) not in environment
    assert binding["default_enabled"] is False
    assert binding["research_only"] is True
    assert binding["canonical"] is False
    assert binding["phase7_excluded"] is True
    assert set(binding) == {"artifacts", "canonical", "default_enabled", "deployment", "entrypoint", "executable", "live_capture", "manual", "phase7_excluded", "research_only", "schema_version"}
    assert set(binding["artifacts"]) == {"config", "live_capture", "live_capture_child", "model", "model_manifest"}
    assert binding["live_capture"] == {
        "entrypoint": "src.predictor.manual_live_capture:main",
        "child": "src.predictor.manual_live_capture_child:main",
        "entrypoint_sha256": binding["artifacts"]["live_capture"],
        "child_sha256": binding["artifacts"]["live_capture_child"],
    }
    assert "MANUAL_RESEARCH_LIVE_CAPTURE_ENTRYPOINT=src.predictor.manual_live_capture:main" in environment
    assert "MANUAL_RESEARCH_LIVE_CAPTURE_CHILD=src.predictor.manual_live_capture_child:main" in environment
    worker = values["source_root"] / "src/predictor/manual_research_worker.py"
    binding_path = output / "manual-research.binding.json"
    def run_worker(environment=None):
        process = subprocess.Popen(
            [sys.executable, str(worker), "--binding", str(binding_path)],
            env=environment,
        )
        return process.wait()

    assert run_worker() == 0
    enabled_env = dict(os.environ, MANUAL_RESEARCH_ENABLED="1")
    assert run_worker(enabled_env) == 78
    assert "KillMode=control-group" in service
    assert "FinalKillSignal=SIGKILL" in service
    assert "RestrictAddressFamilies=AF_UNIX" in service
    assert "manual_research_worker --binding" in service
    assert "ConditionPathExists=" + str(values["manual_root"] / ".manual-research-enabled") in service
    for path in values["protected_paths"].values():
        assert str(path) in service
    expected_service = Template(
        (values["source_root"] / "ops/systemd/manual-research-api.service.in").read_text()
    ).substitute(
        SOURCE_ROOT=values["source_root"],
        ENVIRONMENT_FILE=output / "manual-research.env",
        PYTHON_EXECUTABLE=values["pinned_python"],
        BINDING_PATH=output / "manual-research.binding.json",
        MANUAL_ROOT=values["manual_root"],
        ENABLE_MARKER=values["manual_root"] / ".manual-research-enabled",
        PROTECTED_PATHS=" ".join(
            str(values["protected_paths"][name]) for name in sorted(values["protected_paths"])
        ),
        TIMEOUT_SECONDS=900,
    )
    assert service == expected_service
    assert autonomous_before == {
        path: path.read_bytes()
        for path in values["source_root"].glob("ops/systemd/*")
        if path.is_file()
    }


def test_generator_rejects_manual_protected_overlap_without_partial_output(deployment_inputs):
    values = dict(deployment_inputs)
    protected = dict(values["protected_paths"])
    protected["canonical_database"] = values["manual_root"]
    values["protected_paths"] = protected
    with pytest.raises(ManualDeploymentRejected, match="overlap"):
        generate_manual_package(**values)
    assert not any(values["output_dir"].iterdir())


def test_generator_rejects_symlinked_model_without_partial_output(deployment_inputs):
    values = dict(deployment_inputs)
    model = values["model"]
    replacement = model.with_name("model-real.json")
    model.rename(replacement)
    model.symlink_to(replacement)
    with pytest.raises(ManualDeploymentRejected, match="symlink"):
        generate_manual_package(**values)
    assert not any(values["output_dir"].iterdir())


def test_generator_rejects_manual_lock_hardlink_to_protected_file(deployment_inputs):
    values = dict(deployment_inputs)
    protected_file = values["protected_paths"]["canonical_database"]
    values["manual_lock"].hardlink_to(protected_file)
    with pytest.raises(ManualDeploymentRejected, match="aliases protected path"):
        generate_manual_package(**values)
    assert not any(values["output_dir"].iterdir())


@pytest.mark.parametrize("field", ["manual_root", "output_dir"])
def test_generator_rejects_systemd_unsafe_path(deployment_inputs, field):
    values = dict(deployment_inputs)
    unsafe = values[field].parent / (values[field].name + " with-space")
    unsafe.mkdir()
    unsafe.chmod(0o700)
    values[field] = unsafe
    with pytest.raises(ManualDeploymentRejected, match="normalized absolute path"):
        generate_manual_package(**values)
    assert not any(deployment_inputs["output_dir"].iterdir())


@pytest.mark.parametrize(
    ("field", "value"),
    [("timeout_seconds", 901), ("margin_seconds", 7201)],
)
def test_generator_rejects_out_of_contract_timing(deployment_inputs, field, value):
    values = dict(deployment_inputs)
    values[field] = value
    with pytest.raises(ManualDeploymentRejected, match="invalid"):
        generate_manual_package(**values)


def test_generator_rejects_source_as_output(deployment_inputs):
    values = dict(deployment_inputs)
    values["output_dir"] = values["source_root"]
    with pytest.raises(ManualDeploymentRejected, match="separate"):
        generate_manual_package(**values)


def test_generator_rejects_incomplete_protected_inventory(deployment_inputs):
    values = dict(deployment_inputs)
    values["protected_paths"] = {"canonical_database": values["protected_paths"]["canonical_database"]}
    with pytest.raises(ManualDeploymentRejected, match="inventory"):
        generate_manual_package(**values)


def test_cli_subcommand_uses_same_generator(monkeypatch, deployment_inputs, capsys):
    from src.operator_ui.deployment import main

    values = deployment_inputs
    arguments = [
        "generate-manual",
        "--source-root", str(values["source_root"]),
        "--pinned-python", str(values["pinned_python"]),
        "--manual-root", str(values["manual_root"]),
        "--browser-profile-root", str(values["browser_profile_root"]),
        "--manual-runs-root", str(values["manual_runs_root"]),
        "--manual-lock", str(values["manual_lock"]),
        "--model", str(values["model"]),
        "--model-manifest", str(values["model_manifest"]),
        "--config", str(values["config"]),
        "--output-dir", str(values["output_dir"]),
        "--source-commit", values["source_commit"],
        "--source-tree", values["source_tree"],
    ]
    for name, path in values["protected_paths"].items():
        arguments.extend(("--protected-path", f"{name}={path}"))
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is False
