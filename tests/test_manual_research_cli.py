from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import test_manual_research_scoring as ghu053

from src.predictor.manual_independent_capture import canonical_bytes
from src.predictor.manual_research_cli import (
    ManualResearchAdapterRejected,
    invoke_manual_research_prediction,
    main,
)


def _write(path: Path, value: bytes | dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else canonical_bytes(value))
    return path


def _fixture(tmp_path: Path) -> dict:
    _, _, execution, expected, identity, sealed = ghu053._sealed_fixture(tmp_path / "fixture")
    form_bytes = ghu053._form(sealed)
    config_bytes = canonical_bytes(ghu053._config())
    args_root = tmp_path / "args"
    model_path = _write(args_root / "model.json", ghu053.MODEL_BYTES)
    manifest_path = _write(args_root / "model-manifest.json", ghu053.MODEL_MANIFEST_PATH.read_bytes())
    form_path = _write(args_root / "form.json", form_bytes)
    config_path = _write(args_root / "config.json", config_bytes)
    expectations_path = _write(args_root / "evidence-expectations.json", asdict(expected))
    sealing_identity_path = _write(args_root / "sealing-identity.json", asdict(identity))
    scoring_identity_path = _write(args_root / "scoring-identity.json", asdict(ghu053.SCORE_IDENTITY))
    output_root = tmp_path / "isolated-output"
    output_root.mkdir()
    return {
        "sealed_bundle_dir": sealed.bundle_dir,
        "run_dir": execution.run_dir,
        "evidence_expectations": expectations_path,
        "sealing_identity": sealing_identity_path,
        "embedded_form": form_path,
        "form_sha256": hashlib.sha256(form_bytes).hexdigest(),
        "model": model_path,
        "model_manifest": manifest_path,
        "model_sha256": hashlib.sha256(ghu053.MODEL_BYTES).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "config": config_path,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "scoring_identity": scoring_identity_path,
        "output_root": output_root,
    }


def _cli_args(arguments: dict) -> list[str]:
    flags = {
        "sealed_bundle_dir": "--sealed-bundle-dir",
        "run_dir": "--run-dir",
        "evidence_expectations": "--evidence-expectations",
        "sealing_identity": "--sealing-identity",
        "embedded_form": "--embedded-form",
        "form_sha256": "--form-sha256",
        "model": "--model",
        "model_manifest": "--model-manifest",
        "model_sha256": "--model-sha256",
        "model_manifest_sha256": "--model-manifest-sha256",
        "config": "--config",
        "config_sha256": "--config-sha256",
        "scoring_identity": "--scoring-identity",
        "output_root": "--output-root",
    }
    result: list[str] = []
    for key, flag in flags.items():
        result.extend((flag, str(arguments[key])))
    return result


def test_valid_fixture_invocation_and_exact_replay_are_machine_readable(tmp_path: Path):
    arguments = _fixture(tmp_path)
    first = invoke_manual_research_prediction(**arguments)
    second = invoke_manual_research_prediction(**arguments)
    assert first == second
    assert set(first) == {"schema_version", "status", "bundle_id", "bundle_path", "verification_status"}
    assert first["status"] == "SUCCESS"
    assert first["verification_status"] == "VERIFIED"
    bundle = Path(first["bundle_path"])
    assert bundle.parent == arguments["output_root"]
    assert sorted(path.name for path in bundle.iterdir()) == ["manifest.json", "prediction.json"]
    schema = json.loads(
        (ROOT / "configs/prediction/manual-independent-capture-v1/manual-research-adapter-response.schema.json").read_bytes()
    )
    Draft202012Validator(schema).validate(first)


def test_cli_returns_only_the_verified_envelope(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    arguments = _fixture(tmp_path)
    assert main(_cli_args(arguments)) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "SUCCESS"
    assert response["verification_status"] == "VERIFIED"
    assert set(response) == {"schema_version", "status", "bundle_id", "bundle_path", "verification_status"}


def test_malformed_arguments_and_unsafe_or_missing_paths_fail_closed(tmp_path: Path, capsys):
    assert main([]) == 2
    response = json.loads(capsys.readouterr().out)
    assert response["error_code"] == "ARGUMENTS_INVALID"
    schema = json.loads(
        (ROOT / "configs/prediction/manual-independent-capture-v1/manual-research-adapter-response.schema.json").read_bytes()
    )
    Draft202012Validator(schema).validate(response)
    arguments = _fixture(tmp_path / "paths")
    missing = dict(arguments, embedded_form=tmp_path / "missing-form.json")
    with pytest.raises(ManualResearchAdapterRejected, match="INPUT_MISSING"):
        invoke_manual_research_prediction(**missing)
    symlink = tmp_path / "symlink-form.json"
    symlink.symlink_to(arguments["embedded_form"])
    with pytest.raises(ManualResearchAdapterRejected, match="PATH_UNSAFE"):
        invoke_manual_research_prediction(**dict(arguments, embedded_form=symlink))
    with pytest.raises(ManualResearchAdapterRejected, match="PATH_UNSAFE"):
        invoke_manual_research_prediction(**dict(arguments, output_root=Path("relative-output")))
    _write(arguments["scoring_identity"], {})
    with pytest.raises(ManualResearchAdapterRejected, match="ARGUMENTS_INVALID"):
        invoke_manual_research_prediction(**arguments)


@pytest.mark.parametrize(
    "field,code",
    [
        ("form_sha256", "FORM_HASH_MISMATCH"),
        ("model_sha256", "MODEL_HASH_MISMATCH"),
        ("model_manifest_sha256", "MODEL_MANIFEST_HASH_MISMATCH"),
        ("config_sha256", "CONFIG_HASH_MISMATCH"),
    ],
)
def test_form_model_and_config_drift_fail_closed(tmp_path: Path, field: str, code: str):
    arguments = _fixture(tmp_path)
    with pytest.raises(ManualResearchAdapterRejected, match=code):
        invoke_manual_research_prediction(**dict(arguments, **{field: "0" * 64}))


def test_evidence_identity_and_temporal_form_drift_delegate_fail_closed(tmp_path: Path):
    arguments = _fixture(tmp_path / "identity")
    evidence = json.loads(arguments["evidence_expectations"].read_bytes())
    evidence["race_identity_sha256"] = "0" * 64
    _write(arguments["evidence_expectations"], evidence)
    with pytest.raises(ManualResearchAdapterRejected, match="BUNDLE_CONTRACT_INVALID|IDENTITY_MISMATCH|RACE_IDENTITY"):
        invoke_manual_research_prediction(**arguments)

    arguments = _fixture(tmp_path / "temporal")
    form = json.loads(arguments["embedded_form"].read_bytes())
    form["runners"][0]["history"][0]["event_timestamp"] = "2026-08-05T01:00:00+00:00"
    form_bytes = canonical_bytes(form)
    _write(arguments["embedded_form"], form_bytes)
    arguments["form_sha256"] = hashlib.sha256(form_bytes).hexdigest()
    with pytest.raises(ManualResearchAdapterRejected, match="FORM_HISTORY_AFTER_CUTOFF"):
        invoke_manual_research_prediction(**arguments)


def test_partial_publication_does_not_expose_final_bundle(tmp_path: Path, monkeypatch):
    arguments = _fixture(tmp_path)
    import src.predictor.manual_research_cli as adapter

    original = adapter.score_verified_manual_evidence

    def stop_after_manifest(**kwargs):
        def hook(stage, _path):
            if stage == "manifest_written":
                raise RuntimeError("adapter-stop")

        kwargs["stage_hook"] = hook
        return original(**kwargs)

    monkeypatch.setattr(adapter, "score_verified_manual_evidence", stop_after_manifest)
    with pytest.raises(RuntimeError, match="adapter-stop"):
        invoke_manual_research_prediction(**arguments)
    assert not [
        path
        for path in arguments["output_root"].iterdir()
        if path.is_dir() and len(path.name) == 64
    ]


def test_forbidden_live_canonical_and_result_surfaces_are_not_touched(tmp_path: Path, monkeypatch):
    import sqlite3

    arguments = _fixture(tmp_path)

    def explode(*_args, **_kwargs):
        raise AssertionError("forbidden API touched")

    monkeypatch.setattr(sqlite3, "connect", explode)
    import src.predictor.manual_research_cli as adapter

    assert not any(
        token in adapter.__dict__
        for token in ("sqlite3", "requests", "httpx", "subprocess", "run_capture_one", "live_odds")
    )
    result = invoke_manual_research_prediction(**arguments)
    assert result["verification_status"] == "VERIFIED"


def test_forbidden_upstream_error_names_are_not_exposed(tmp_path: Path, monkeypatch):
    arguments = _fixture(tmp_path)
    import src.predictor.manual_research_cli as adapter

    def forbidden_error(**_kwargs):
        from src.predictor.manual_independent_capture_sealer import (
            ManualEvidenceRejected,
        )

        raise ManualEvidenceRejected("OUTCOME_MATERIAL_FORBIDDEN")

    monkeypatch.setattr(adapter, "score_verified_manual_evidence", forbidden_error)
    with pytest.raises(ManualResearchAdapterRejected, match="UNAUTHORIZED_INPUT"):
        invoke_manual_research_prediction(**arguments)
