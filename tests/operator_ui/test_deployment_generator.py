import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from flask import Flask

from src.operator_ui.deployment import DeploymentRejected, generate_package, main
from src.operator_ui.security import load_connected_environment
import src.operator_ui.bootstrap as bootstrap_module


COMMIT = "881a3cee0c7f93dd26f5ece9185052f59c4c1aed"
TREE = "226dae42ebc4deea7ce1c7954e8da74fabd37f7a"
SECRET_LINES = (
    "OPERATOR_UI_SECRET_KEY=actual-secret_+/=.$-value",
    "OPERATOR_UI_USERNAME=operator@example.test",
    "OPERATOR_UI_PASSWORD_HASH=scrypt:32768:8:1$saltsalt$0123456789abcdef",
)
AUTHORITY_RELATIVES = (
    "configs/operator_ui/repository-v1.toml",
    "scripts/predict_race_now.py",
    "configs/prediction/manual-default.json",
    "artifacts/frozen_models/market_form_residual_v1/model.json",
    "artifacts/frozen_models/market_form_residual_v1/manifest.json",
    "configs/prediction/schemas/market_form_residual_v1.schema.json",
)


@pytest.fixture
def tmp_path():
    """Keep retained-authority fixtures outside foreign-owned /tmp ancestry."""
    repository = Path(__file__).resolve().parents[2]
    with TemporaryDirectory(prefix=".pytest-deployment-", dir=repository) as raw:
        root = Path(raw)
        info = root.stat()
        assert root == root.resolve()
        assert info.st_uid == os.geteuid()
        assert stat.S_IMODE(info.st_mode) == 0o700
        yield root


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
    secrets.write_text("\n".join(SECRET_LINES) + "\n")
    secrets.chmod(0o600)
    live_root = tmp_path / "live"
    live_root.mkdir(mode=0o700)
    json_keys = {
        "full_state", "full_report", "odds_state", "odds_report", "odds_refresh",
        "corpus_report", "corpus_manifest", "deployment_manifest", "model_catalog",
    }
    raw_keys = {
        "corpus_inventory_csv", "corpus_inventory_jsonl", "corpus_scorecard_csv",
        "corpus_scorecard_jsonl", "corpus_report_bytes", "corpus_summary",
        "corpus_final_status", "model_latest_config", "model_latest_schema",
        "model_latest_artifact", "model_latest_manifest", "model_baseline_config",
        "model_baseline_schema",
    }
    schemas = {
        "full_state": "shadow_autopilot_daemon_state_v1",
        "full_report": "shadow_autopilot_daemon_run_v1",
        "odds_state": "shadow_autopilot_odds_capture_only_state_v1",
        "odds_report": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "corpus_report": "race_evidence_inventory_report_v1",
        "corpus_manifest": "race_evidence_inventory_output_manifest_v1",
        "deployment_manifest": "operator_ui_deployment_manifest_v1",
        "model_catalog": "on_demand_prediction_config_catalog_v1",
    }
    sources = {}
    for key in json_keys:
        payload = {} if key == "odds_refresh" else {"schema_version": schemas[key]}
        if key == "odds_report":
            payload["autopilot_output_dir"] = "reports"
        if key not in {"full_state", "corpus_manifest", "model_catalog"}:
            payload["updated_at" if key == "odds_state" else "generated_at"] = "2026-08-03T01:02:03Z"
        target = (live_root / "reports/odds_capture_refresh_report.json"
                  if key == "odds_refresh" else live_root / f"{key}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload))
        sources[key] = str(target)
    raw_sources = {}
    for key in raw_keys:
        target = live_root / f"{key}.raw"; target.write_bytes(key.encode())
        raw_sources[key] = str(target)
    units = {}
    unit_names = {
        "full_timer": "shadow-autopilot.timer",
        "full_service": "shadow-autopilot.service",
        "odds_timer": "shadow-autopilot-odds-capture.timer",
        "odds_service": "shadow-autopilot-odds-capture.service",
    }
    for key, unit_name in unit_names.items():
        target = live_root / unit_name; target.write_text("[Unit]\nDescription=test\n")
        units[key] = str(target)
    authority = live_root / "authority.json"
    authority.write_text(json.dumps({
        "schema_version": "operator_ui_live_authority_v1",
        "observed_at": "2026-08-03T01:02:03Z", "working_directory": str(source),
        "sources": sources, "raw_sources": raw_sources, "units": units,
        "service_status": {
            "full": {"unit_name": "shadow-autopilot.service", "active_state": "inactive", "sub_state": "dead", "exec_main_pid": 0},
            "odds": {"unit_name": "shadow-autopilot-odds-capture.service", "active_state": "active", "sub_state": "waiting", "exec_main_pid": 0},
        },
    }))
    return dict(source_root=source, pinned_python=python, evidence_root=evidence,
                producer_root=producer, canonical_db=database,
                operations_root=operations, secrets_file=secrets,
                output_dir=output, source_commit=COMMIT, source_tree=TREE,
                ui_version="operator-ui-v1", profile_id="repository-v1",
                bind_address="127.0.0.1", port=5055, live_authority=authority)


def generated_targets(values: dict[str, object]) -> tuple[Path, ...]:
    return (
        values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json",
        values["output_dir"] / "operator-ui-r3.env",
        values["output_dir"] / "greyhound-operator-ui-r3.service",
        values["output_dir"] / "ROLLBACK.md",
    )


def replace_during_authority_read(monkeypatch, victim: Path, *, component: bool) -> None:
    identity = (victim.stat().st_dev, victim.stat().st_ino)
    real_read = os.read
    replaced = False

    def read(descriptor, size):
        nonlocal replaced
        info = os.fstat(descriptor)
        if not replaced and (info.st_dev, info.st_ino) == identity:
            replaced = True
            if component:
                parent = victim.parent
                displaced = parent.with_name(parent.name + "-displaced")
                parent.rename(displaced)
                parent.mkdir()
                (parent / victim.name).write_bytes(b"attacker component replacement")
            else:
                displaced = victim.with_name(victim.name + "-displaced")
                victim.rename(displaced)
                victim.write_bytes(b"attacker leaf replacement")
        return real_read(descriptor, size)

    monkeypatch.setattr("src.operator_ui.deployment.os.read", read)


def git_identity(monkeypatch, *, commit=COMMIT, tree=TREE, dirty=False):
    def run(command, **kwargs):
        if "status" in command:
            output = " M app.py\n" if dirty else ""
        else:
            output = f"{commit}\n{tree}\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("src.operator_ui.deployment.subprocess.run", run)


def load_generated_environment(monkeypatch, values):
    generated = dict(
        line.split("=", 1)
        for line in (values["output_dir"] / "operator-ui-r3.env").read_text().splitlines()
        if line
    )
    for name, value in generated.items():
        monkeypatch.setenv(name, value)
    return generated


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
    assert "OPERATOR_UI_LEVEL=1" in environment
    assert "OPERATOR_UI_R3_PROFILE=disabled" in environment
    assert "127.0.0.1" in service and "--port 5055" in service
    assert f"EnvironmentFile={values['secrets_file']}" in service
    assert "PrivateUsers=true" in service.splitlines()
    assert "actual-secret" not in service + environment
    assert result["enabled"] is False


@pytest.mark.parametrize("relative", AUTHORITY_RELATIVES)
@pytest.mark.parametrize("component", [False, True], ids=["leaf", "component"])
def test_generator_rejects_authority_replacement_during_retained_read_without_output(
    tmp_path, monkeypatch, relative, component
):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    replace_during_authority_read(
        monkeypatch, values["source_root"] / relative, component=component
    )

    with pytest.raises(DeploymentRejected, match="authority.*changed|identity"):
        generate_package(**values)

    assert all(not target.exists() for target in generated_targets(values))


def test_generator_allows_unrelated_ancestor_directory_churn(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    real_read = os.read
    churned = False

    def read(descriptor, size):
        nonlocal churned
        if not churned:
            churned = True
            (values["source_root"].parent / "unrelated-activity").write_text("unrelated")
        return real_read(descriptor, size)

    monkeypatch.setattr("src.operator_ui.deployment.os.read", read)

    assert generate_package(**values)["enabled"] is False
    assert all(target.exists() for target in generated_targets(values))


@pytest.mark.parametrize("relative", AUTHORITY_RELATIVES)
def test_generator_rejects_in_place_authority_change_during_retained_read_without_output(
    tmp_path, monkeypatch, relative
):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    victim = values["source_root"] / relative
    identity = (victim.stat().st_dev, victim.stat().st_ino)
    real_read = os.read
    changed = False

    def read(descriptor, size):
        nonlocal changed
        info = os.fstat(descriptor)
        if not changed and (info.st_dev, info.st_ino) == identity:
            changed = True
            victim.write_bytes(b"in-place attacker change")
        return real_read(descriptor, size)

    monkeypatch.setattr("src.operator_ui.deployment.os.read", read)

    with pytest.raises(DeploymentRejected, match="authority.*changed"):
        generate_package(**values)

    assert all(not target.exists() for target in generated_targets(values))


@pytest.mark.parametrize("relative", AUTHORITY_RELATIVES)
def test_authority_reads_are_bounded_and_close_every_descriptor(tmp_path, monkeypatch, relative):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    (values["source_root"] / relative).write_bytes(b"x" * (256 * 1024 + 1))
    descriptors_before = len(tuple(Path("/proc/self/fd").iterdir()))

    with pytest.raises(DeploymentRejected, match="oversized"):
        generate_package(**values)

    assert len(tuple(Path("/proc/self/fd").iterdir())) == descriptors_before
    assert all(not target.exists() for target in generated_targets(values))


@pytest.mark.parametrize(
    "injected",
    [
        "OPERATOR_UI_CONNECTED_MODE=1",
        "OPERATOR_UI_R3_PROFILE=repository-v1",
        "OPERATOR_UI_DEPLOYED_COMMIT=" + "a" * 40,
        "ENABLE_LIVE_SCRAPING=1",
        "UNRELATED_KEY=value",
        SECRET_LINES[0],
        'OPERATOR_UI_USERNAME="operator"',
        "OPERATOR_UI_USERNAME=operator\\ name",
        " OPERATOR_UI_USERNAME=operator",
        "export OPERATOR_UI_USERNAME=operator",
    ],
)
def test_secrets_file_rejects_override_extra_duplicate_or_ambiguous_syntax_before_output(
    tmp_path, monkeypatch, injected
):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    values["secrets_file"].write_text("\n".join((*SECRET_LINES, injected)) + "\n")

    with pytest.raises(DeploymentRejected, match="secrets file"):
        generate_package(**values)

    assert all(not target.exists() for target in generated_targets(values))


def test_secrets_file_accepts_safe_comments_blanks_and_real_secret_forms(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    values["secrets_file"].write_text(
        "# generated out of band\n\n" + "\n".join(SECRET_LINES) + "\n# end\n"
    )

    generate_package(**values)

    assert all(target.is_file() for target in generated_targets(values))


@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("operation", ["fsync", "replace"])
@pytest.mark.parametrize("failure_point", range(1, 5))
def test_reported_publication_failure_restores_the_whole_package_and_cleans_artifacts(
    tmp_path, monkeypatch, preexisting, operation, failure_point
):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    targets = generated_targets(values)
    expected = {}
    if preexisting:
        for index, target in enumerate(targets):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(f"prior-{index}".encode())
            target.chmod(0o640 + index)
            expected[target] = (target.read_bytes(), stat.S_IMODE(target.stat().st_mode))

    real_operation = getattr(os, operation)
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_point:
            raise OSError(f"injected {operation} failure {failure_point}")
        return real_operation(*args, **kwargs)

    monkeypatch.setattr(f"src.operator_ui.deployment.os.{operation}", fail_once)

    with pytest.raises(OSError, match=f"injected {operation} failure {failure_point}"):
        generate_package(**values)

    for target in targets:
        if preexisting:
            assert (target.read_bytes(), stat.S_IMODE(target.stat().st_mode)) == expected[target]
        else:
            assert not target.exists()
    artifact_names = [
        path.name
        for parent in {target.parent for target in targets}
        for path in parent.iterdir()
        if path.name.startswith(".")
    ]
    assert artifact_names == []


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


@pytest.mark.parametrize(
    "root_name",
    ["source_root", "evidence_root", "producer_root", "operations_root", "output_dir"],
)
def test_generator_rejects_secrets_inside_deployment_roots_without_partial_writes(
    tmp_path, monkeypatch, root_name
):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    secrets = values[root_name] / "operator-ui.secrets"
    secrets.write_text(values["secrets_file"].read_text())
    secrets.chmod(0o600)
    values["secrets_file"] = secrets

    with pytest.raises(DeploymentRejected, match="secrets file must be separate"):
        generate_package(**values)

    assert not (values["output_dir"] / "greyhound-operator-ui-r3.service").exists()
    assert not (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").exists()


def test_generator_rejects_canonical_database_as_secrets_without_partial_writes(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    values["canonical_db"].chmod(0o600)
    values["canonical_db"].write_text(values["secrets_file"].read_text())
    values["secrets_file"].unlink()
    values["secrets_file"].hardlink_to(values["canonical_db"])

    with pytest.raises(DeploymentRejected, match="must not be the canonical database"):
        generate_package(**values)

    assert not (values["output_dir"] / "greyhound-operator-ui-r3.service").exists()
    assert not (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").exists()


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o700])
def test_generator_rejects_any_non_0600_secrets_mode_without_partial_writes(
    tmp_path, monkeypatch, mode
):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    values["secrets_file"].chmod(mode)

    with pytest.raises(DeploymentRejected, match="exact mode 0600"):
        generate_package(**values)

    assert not (values["output_dir"] / "greyhound-operator-ui-r3.service").exists()
    assert not (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").exists()


def test_generator_rejects_secrets_owned_by_another_user_without_partial_writes(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path)
    git_identity(monkeypatch)
    monkeypatch.setattr("src.operator_ui.deployment.os.geteuid", lambda: values["secrets_file"].stat().st_uid + 1)

    with pytest.raises(DeploymentRejected, match="owned by the current service user"):
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
    assert "OPERATOR_UI_LEVEL=2" in environment
    assert "OPERATOR_UI_R3_PROFILE=repository-v1" in environment
    assert "disable" in rollback.lower()
    assert "do not delete" in rollback.lower()
    assert str(values["operations_root"]) in rollback
    assert result["enabled"] is True
    binding=json.loads((values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").read_text())
    refresh=binding["live_evidence"]["sources"]["odds_refresh"]
    assert Path(refresh["path"]).relative_to(Path(refresh["allowlisted_root"])).as_posix()=="reports/odds_capture_refresh_report.json"
    assert binding["live_evidence"]["service_status"]["full"]["unit_name"]=="shadow-autopilot.service"
    assert binding["live_evidence"]["service_status"]["odds"]["unit_name"]=="shadow-autopilot-odds-capture.service"


def test_enabled_generator_rejects_missing_or_incomplete_live_authority_without_output(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = values.pop("live_authority")
    with pytest.raises(DeploymentRejected, match="requires live authority"):
        generate_package(**values, enabled=True)
    assert all(not target.exists() for target in generated_targets(values))
    values["live_authority"] = authority
    authority.write_text(json.dumps({"schema_version": "operator_ui_live_authority_v1"}))
    with pytest.raises(DeploymentRejected, match="incomplete"):
        generate_package(**values, enabled=True)
    assert all(not target.exists() for target in generated_targets(values))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_enabled_generator_strictly_decodes_live_authority(tmp_path, monkeypatch, constant):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = values["live_authority"]
    raw = authority.read_text()
    authority.write_text(raw[:-1] + f',"nonfinite":{constant}}}')
    with pytest.raises(DeploymentRejected, match="malformed"):
        generate_package(**values, enabled=True)
    authority.write_text(raw[:-1] + ',"schema_version":"duplicate"}')
    with pytest.raises(DeploymentRejected, match="malformed"):
        generate_package(**values, enabled=True)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_enabled_generator_strictly_decodes_retained_odds_report(tmp_path, monkeypatch, constant):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    odds_report.write_text('{"autopilot_output_dir":"reports","value":' + constant + '}')
    with pytest.raises(DeploymentRejected, match="odds refresh authority is contradictory"):
        generate_package(**values, enabled=True)
    odds_report.write_text('{"autopilot_output_dir":"reports","autopilot_output_dir":"other"}')
    with pytest.raises(DeploymentRejected, match="odds refresh authority is contradictory"):
        generate_package(**values, enabled=True)


def test_enabled_generator_derives_refresh_root_without_rereading_odds_report(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    real_read = __import__("src.operator_ui.deployment", fromlist=["_retained_file_read"])._retained_file_read
    reads = 0

    def retained_read(path, maximum=256 * 1024):
        nonlocal reads
        if Path(path) == odds_report:
            reads += 1
        return real_read(path, maximum)

    monkeypatch.setattr("src.operator_ui.deployment._retained_file_read", retained_read)
    generate_package(**values, enabled=True)
    assert reads == 1


def test_enabled_generator_accepts_exact_absolute_odds_output_dir(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    refresh = Path(authority["sources"]["odds_refresh"])
    odds_report.write_text(json.dumps({
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "autopilot_output_dir": str(refresh.parent),
    }))

    generate_package(**values, enabled=True)

    binding = json.loads(
        (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").read_text()
    )
    assert Path(binding["live_evidence"]["sources"]["odds_refresh"]["allowlisted_root"]) == refresh.parent


def test_enabled_generator_rejects_mismatched_absolute_odds_output_dir_without_output(
    tmp_path, monkeypatch
):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    odds_report.write_text(json.dumps({
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "autopilot_output_dir": str(tmp_path / "different-reports"),
    }))

    with pytest.raises(DeploymentRejected, match="odds refresh authority is contradictory"):
        generate_package(**values, enabled=True)

    assert all(not target.exists() for target in generated_targets(values))


def test_enabled_generator_binds_waiting_cycle_without_refresh_output(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    odds_report.write_text(json.dumps({
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "generated_at": "2026-08-03T01:02:03Z",
        "final_status": "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
        "status": "WAITING",
        "autopilot_output_dir": None,
        "odds_capture_refresh_report": {},
    }))

    generate_package(**values, enabled=True)

    binding = json.loads(
        (values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").read_text()
    )
    refresh = binding["live_evidence"]["sources"]["odds_refresh"]
    assert Path(refresh["allowlisted_root"]) == Path(refresh["path"]).parent
    assert Path(refresh["path"]).name == "odds_capture_refresh_report.json"


@pytest.mark.parametrize(
    "report_update, removed_key",
    [
        ({"schema_version": "wrong"}, None),
        ({}, "schema_version"),
        ({"generated_at": "not-a-timestamp"}, None),
        ({"generated_at": "2026-08-03T01:02:03"}, None),
        ({"generated_at": []}, None),
        ({}, "generated_at"),
        ({"status": "CAPTURED"}, None),
    ],
)
def test_enabled_generator_rejects_unauthenticated_waiting_cycle_without_refresh_output(
    tmp_path, monkeypatch, report_update, removed_key
):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    report = {
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "generated_at": "2026-08-03T01:02:03Z",
        "final_status": "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
        "status": "WAITING",
        "autopilot_output_dir": None,
        "odds_capture_refresh_report": {},
    }
    report.update(report_update)
    if removed_key is not None:
        del report[removed_key]
    odds_report.write_text(json.dumps(report))

    with pytest.raises(DeploymentRejected, match="odds refresh authority is contradictory"):
        generate_package(**values, enabled=True)

    assert all(not target.exists() for target in generated_targets(values))


def test_enabled_generator_rejects_substituted_refresh_authority_filename(
    tmp_path, monkeypatch
):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    odds_report.write_text(json.dumps({
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "generated_at": "2026-08-03T01:02:03Z",
        "final_status": "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
        "status": "WAITING",
        "autopilot_output_dir": None,
        "odds_capture_refresh_report": {},
    }))
    refresh = Path(authority["sources"]["odds_refresh"])
    substituted = refresh.with_name("substituted_refresh_report.json")
    substituted.write_bytes(refresh.read_bytes())
    authority["sources"]["odds_refresh"] = str(substituted)
    values["live_authority"].write_text(json.dumps(authority))

    with pytest.raises(DeploymentRejected, match="odds refresh authority is contradictory"):
        generate_package(**values, enabled=True)

    assert all(not target.exists() for target in generated_targets(values))


@pytest.mark.parametrize(
    "report_update",
    [
        {"final_status": "ODDS_CAPTURE_ONLY_READY"},
        {"odds_capture_refresh_report": {"status": "CAPTURED"}},
    ],
)
def test_enabled_generator_rejects_null_refresh_locator_when_refresh_is_required(
    tmp_path, monkeypatch, report_update
):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    odds_report = Path(authority["sources"]["odds_report"])
    report = {
        "schema_version": "shadow_autopilot_odds_capture_only_daemon_report_v1",
        "generated_at": "2026-08-03T01:02:03Z",
        "final_status": "ODDS_CAPTURE_ONLY_WAITING_FOR_WINDOW",
        "status": "WAITING",
        "autopilot_output_dir": None,
        "odds_capture_refresh_report": {},
    }
    report.update(report_update)
    odds_report.write_text(json.dumps(report))

    with pytest.raises(DeploymentRejected, match="odds refresh authority is contradictory"):
        generate_package(**values, enabled=True)

    assert all(not target.exists() for target in generated_targets(values))


def test_enabled_generator_rejects_duplicate_unit_paths(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    authority["units"]["odds_service"] = authority["units"]["full_service"]
    values["live_authority"].write_text(json.dumps(authority))
    with pytest.raises(DeploymentRejected, match="unit paths must be distinct"):
        generate_package(**values, enabled=True)


def test_enabled_generator_rejects_unit_path_with_wrong_basename(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    wrong = Path(authority["units"]["full_timer"]).with_name("wrong.timer")
    wrong.write_text("[Unit]\nDescription=test\n")
    authority["units"]["full_timer"] = str(wrong)
    values["live_authority"].write_text(json.dumps(authority))
    with pytest.raises(DeploymentRejected, match="unit path is invalid"):
        generate_package(**values, enabled=True)


@pytest.mark.parametrize("enabled, expected", [(False, False), (True, True)])
def test_real_generated_package_startup_is_disabled_or_bootstraps_with_all_deployment_identity(
    real_startup_tmp_path, monkeypatch, enabled, expected
):
    values = deployment_inputs(real_startup_tmp_path)
    git_identity(monkeypatch)
    generate_package(**values, enabled=enabled)
    generated = load_generated_environment(monkeypatch, values)
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


def test_clean_exact_generated_serve_identity_reaches_exec(real_startup_tmp_path, monkeypatch):
    values = deployment_inputs(real_startup_tmp_path)
    git_identity(monkeypatch)
    generate_package(**values, enabled=True)
    load_generated_environment(monkeypatch, values)
    executed = []

    class ExecReached(Exception):
        pass

    def execv(executable, arguments):
        executed.append((executable, arguments))
        raise ExecReached

    monkeypatch.setattr("src.operator_ui.deployment.os.execv", execv)
    with pytest.raises(ExecReached):
        main(["serve", "--source-root", str(values["source_root"]),
              "--host", "127.0.0.1", "--port", "5055"])
    assert executed[0][1][1:] == [str(values["source_root"] / "app.py"),
                                  "--host", "127.0.0.1", "--port", "5055"]


def test_generated_serve_refuses_later_runtime_source_mutation_before_exec(
    real_startup_tmp_path, monkeypatch
):
    values = deployment_inputs(real_startup_tmp_path)
    app_path = values["source_root"] / "app.py"
    generated_app = app_path.read_bytes()

    def run(command, **kwargs):
        if "status" in command:
            output = " M app.py\n" if app_path.read_bytes() != generated_app else ""
        else:
            output = f"{COMMIT}\n{TREE}\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr("src.operator_ui.deployment.subprocess.run", run)
    generate_package(**values, enabled=True)
    load_generated_environment(monkeypatch, values)
    app_path.write_bytes(generated_app + b"\n# changed after generation\n")
    executed = []
    monkeypatch.setattr("src.operator_ui.deployment.os.execv", lambda *args: executed.append(args))

    with pytest.raises(DeploymentRejected, match="Git identity"):
        main(["serve", "--source-root", str(values["source_root"]),
              "--host", "127.0.0.1", "--port", "5055"])
    assert executed == []


@pytest.mark.parametrize("name,value", [
    ("OPERATOR_UI_DEPLOYED_COMMIT", None),
    ("OPERATOR_UI_DEPLOYED_COMMIT", "not-a-commit"),
    ("OPERATOR_UI_DEPLOYED_TREE", None),
    ("OPERATOR_UI_DEPLOYED_TREE", "A" * 40),
])
def test_serve_refuses_missing_or_malformed_deployed_identity_before_exec(
    tmp_path, monkeypatch, name, value
):
    values = deployment_inputs(tmp_path)
    monkeypatch.setenv("OPERATOR_UI_CONNECTED_MODE", "1")
    monkeypatch.setenv("OPERATOR_UI_R3_PROFILE", "repository-v1")
    monkeypatch.setenv("OPERATOR_UI_DEPLOYED_COMMIT", COMMIT)
    monkeypatch.setenv("OPERATOR_UI_DEPLOYED_TREE", TREE)
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    executed = []
    monkeypatch.setattr("src.operator_ui.deployment.os.execv", lambda *args: executed.append(args))

    with pytest.raises(DeploymentRejected, match="deployed source commit/tree identity is invalid"):
        main(["serve", "--source-root", str(values["source_root"]),
              "--host", "127.0.0.1", "--port", "5055"])
    assert executed == []


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


@pytest.mark.parametrize("key,size", [("corpus_inventory_csv",17_015_083),("corpus_inventory_jsonl",22_391_456)])
def test_generator_streams_canonical_sized_inventory_without_byte_retention(tmp_path, monkeypatch, key, size):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    path = Path(authority["raw_sources"][key]); path.write_bytes(b"x" * size)
    assert generate_package(**values, enabled=True)["enabled"] is True
    binding = json.loads((values["source_root"] / "var/operator_ui/generated/repository-v1.binding.json").read_text())
    assert binding["live_evidence"]["raw_sources"][key] == {
        "path": str(path.absolute()), "sha256": hashlib.sha256(b"x" * size).hexdigest(),
        "bytes": size, "authentication": "sha256_size_only_v1",
    }


def test_generator_digest_only_inventory_ceiling_and_timeout_fail_closed(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    path = Path(authority["raw_sources"]["corpus_inventory_csv"])
    path.write_bytes(b"")
    with path.open("r+b") as oversized:
        oversized.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(DeploymentRejected, match="oversized"):
        generate_package(**values, enabled=True)
    path.write_bytes(b"inventory")
    ticks = iter((0.0,31.0))
    monkeypatch.setattr("src.operator_ui.deployment.time.monotonic", lambda: next(ticks,31.0))
    with pytest.raises(DeploymentRejected, match="timed out"):
        generate_package(**values, enabled=True)


def test_generator_digest_only_inventory_mutation_fails_closed(tmp_path, monkeypatch):
    values = deployment_inputs(tmp_path); git_identity(monkeypatch)
    authority = json.loads(values["live_authority"].read_text())
    path = Path(authority["raw_sources"]["corpus_inventory_jsonl"])
    replace_during_authority_read(monkeypatch, path, component=False)
    with pytest.raises(DeploymentRejected, match="authority.*changed|identity"):
        generate_package(**values, enabled=True)
