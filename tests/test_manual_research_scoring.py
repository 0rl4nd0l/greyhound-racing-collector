from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_manual_independent_capture_sealer as ghu052

from src.predictor.manual_independent_capture import canonical_bytes
from src.predictor.manual_independent_capture_sealer import (
    build_sealing_identity,
    expectations_from_execution,
    seal_manual_capture,
)
from src.predictor.manual_research_scoring import (
    EMBEDDED_FORM_SCHEMA,
    FEATURE_ADAPTER_VERSION,
    ManualResearchScoringRejected,
    ResearchPredictionExpectations,
    build_research_scoring_identity,
    score_verified_manual_evidence,
    verify_research_prediction_bundle,
)
from src.predictor.market_form_residual import load_frozen_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts/frozen_models/market_form_residual_v1/model.json"
MODEL_MANIFEST_PATH = ROOT / "artifacts/frozen_models/market_form_residual_v1/manifest.json"
MODEL_BYTES = MODEL_PATH.read_bytes()
MODEL = load_frozen_model(MODEL_PATH, MODEL_MANIFEST_PATH)
SOURCE_COMMIT = "492500684fb017b29c3af9748b00e9af8505b457"
SOURCE_TREE = "c16f7acff89fd0e85afe28719f3319eb54154bdd"
SCORE_IDENTITY = build_research_scoring_identity(
    repo_root=ROOT, source_commit=SOURCE_COMMIT, source_tree=SOURCE_TREE
)


def _config() -> dict:
    return {
        "schema_version": "manual_research_scoring_config_v1",
        "safety": {
            "research_only": True,
            "canonical": False,
            "phase7_excluded": True,
            "phase7_eligible": False,
            "phase7_exclusion_reason": "manual_research_only_noncanonical",
        },
        "scorer_id": "manual_research_market_form_residual_v1",
        "model_id": "market_form_residual_v1",
        "model_sha256": MODEL.model_sha256,
        "model_manifest_sha256": MODEL.manifest_sha256,
        "feature_adapter_version": FEATURE_ADAPTER_VERSION,
        "ranking": {"primary_probability": "full_probability", "tie_break": "box_ascending"},
        "persistence": "none",
    }


def _sealed_fixture(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = ghu052._config(tmp_path)
    forbidden = ghu052._forbidden(tmp_path)
    selected = ghu052._race()
    original_model_bytes = ghu052.MODEL_BYTES
    ghu052.MODEL_BYTES = MODEL_BYTES
    try:
        execution = ghu052._execute(
            tmp_path, cfg=config, forbidden=forbidden, race=selected, mode="success"
        )
    finally:
        ghu052.MODEL_BYTES = original_model_bytes
    expected = expectations_from_execution(execution)
    identity = build_sealing_identity(
        repo_root=ROOT, source_commit=ghu052.SOURCE_COMMIT, source_tree=ghu052.SOURCE_TREE
    )
    sealed = seal_manual_capture(
        execution,
        config=config,
        forbidden_paths=forbidden,
        expected=expected,
        identity=identity,
        repo_root=ROOT,
    )
    return config, forbidden, execution, expected, identity, sealed


def _form(sealed) -> bytes:
    bundle = sealed.bundle
    cutoff = bundle["timing"]["capture_timestamp"]
    form = {
        "schema_version": EMBEDDED_FORM_SCHEMA,
        "safety": bundle["safety"],
        "source": {
            "source_class": "embedded_research_form_v1",
            "source_timestamp": "2026-08-05T00:59:00+00:00",
        },
        "cutoff_timestamp": cutoff,
        "target": {
            "race_identity_sha256": bundle["race_identity_sha256"],
            "race_id": bundle["race"]["race_id"],
            "race_date": bundle["race"]["race_date"],
            "venue": bundle["race"]["venue"],
            "distance_m": 515.0,
            "grade": "Grade 5",
        },
        "runners": [
            {
                "box_number": 1,
                "display_name": "Alpha Dog",
                "history": [
                    {
                        "prior_race_id": "prior-alpha-1",
                        "event_timestamp": "2026-08-04T00:10:00+00:00",
                        "race_date": "2026-08-04",
                        "venue": "RICH",
                        "distance_m": 515.0,
                        "grade": "Grade 5",
                        "prior_finish": 1,
                        "prior_margin": 0.2,
                    },
                    {
                        "prior_race_id": "prior-alpha-2",
                        "event_timestamp": "2026-08-04T12:10:00+00:00",
                        "race_date": "2026-08-04",
                        "venue": "RICH",
                        "distance_m": 500.0,
                        "grade": "Grade 6",
                        "prior_finish": 3,
                        "prior_margin": 1.0,
                    },
                ],
            },
            {"box_number": 2, "display_name": "Beta Dog", "history": []},
        ],
    }
    return canonical_bytes(form)


def _score(tmp_path: Path, *, form_bytes: bytes | None = None, config: dict | None = None, **kwargs):
    _, _, execution, expected, identity, sealed = _sealed_fixture(tmp_path)
    form_bytes = form_bytes if form_bytes is not None else _form(sealed)
    config = config if config is not None else _config()
    config_bytes = canonical_bytes(config)
    output_root = tmp_path / "research-predictions"
    output_root.mkdir()
    arguments = {
        "sealed_bundle_dir": sealed.bundle_dir,
        "run_dir": execution.run_dir,
        "evidence_expected": expected,
        "evidence_identity": identity,
        "embedded_form_bytes": form_bytes,
        "form_sha256": hashlib.sha256(form_bytes).hexdigest(),
        "config_bytes": config_bytes,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "frozen_model": MODEL,
        "expected_model_sha256": MODEL.model_sha256,
        "expected_model_manifest_sha256": MODEL.manifest_sha256,
        "scoring_identity": SCORE_IDENTITY,
        "output_root": output_root,
    }
    arguments.update(kwargs)
    result = score_verified_manual_evidence(**arguments)
    return result, output_root, execution, expected, identity, sealed, form_bytes, config_bytes


def _prediction_expected(result, *, form_bytes: bytes, config_bytes: bytes) -> ResearchPredictionExpectations:
    row = result.prediction
    return ResearchPredictionExpectations(
        evidence_bundle_id=row["evidence"]["bundle_id"],
        evidence_manifest_sha256=row["evidence"]["manifest_sha256"],
        race_identity_sha256=row["race_identity_sha256"],
        form_sha256=hashlib.sha256(form_bytes).hexdigest(),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        model_sha256=row["model"]["model_sha256"],
        model_manifest_sha256=row["model"]["manifest_sha256"],
        runner_set_sha256=row["runner_set_sha256"],
        odds_sha256=row["evidence"]["odds_sha256"],
        cutoff_timestamp=row["timing"]["sealed_cutoff_timestamp"],
        scheduled_start=row["timing"]["scheduled_start"],
        effective_state_sha256=row["model"]["effective_state_sha256"],
        implementation=SCORE_IDENTITY,
        feature_sha256=row["features"]["sha256"],
    )


def test_verified_fixture_scores_and_replays_byte_identically(tmp_path: Path):
    first, output_root, execution, expected, identity, sealed, form_bytes, config_bytes = _score(tmp_path)
    second = score_verified_manual_evidence(
        sealed_bundle_dir=sealed.bundle_dir,
        run_dir=execution.run_dir,
        evidence_expected=expected,
        evidence_identity=identity,
        embedded_form_bytes=form_bytes,
        form_sha256=hashlib.sha256(form_bytes).hexdigest(),
        config_bytes=config_bytes,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        frozen_model=MODEL,
        expected_model_sha256=MODEL.model_sha256,
        expected_model_manifest_sha256=MODEL.manifest_sha256,
        scoring_identity=SCORE_IDENTITY,
        output_root=output_root,
    )
    assert first.prediction == second.prediction
    assert (first.bundle_dir / "prediction.json").read_bytes() == (second.bundle_dir / "prediction.json").read_bytes()
    assert first.replayed is False
    assert second.replayed is True
    assert first.prediction["safety"] == sealed.bundle["safety"]
    assert all(0.0 <= row[field] <= 1.0 for row in first.prediction["predictions"] for field in ("market_probability", "half_probability", "full_probability"))
    assert [row["rank"] for row in first.prediction["predictions"]] == [1, 2]
    assert not any(key in first.prediction for key in ("ev", "stake", "bet", "outcome", "result"))


def test_versioned_form_prediction_and_manifest_schemas_validate(tmp_path: Path):
    result, _, _, _, _, sealed, form_bytes, _ = _score(tmp_path)
    schema_root = ROOT / "configs/prediction/manual-independent-capture-v1"
    form_schema = json.loads((schema_root / "embedded-form.schema.json").read_bytes())
    prediction_schema = json.loads((schema_root / "research-prediction.schema.json").read_bytes())
    manifest_schema = json.loads((schema_root / "research-prediction-manifest.schema.json").read_bytes())
    Draft202012Validator(form_schema).validate(json.loads(form_bytes))
    Draft202012Validator(prediction_schema).validate(result.prediction)
    Draft202012Validator(manifest_schema).validate(result.manifest)
    assert json.loads(form_bytes)["target"]["race_identity_sha256"] == sealed.bundle["race_identity_sha256"]


def test_verifier_accepts_untampered_result_and_rejects_prediction_tampering(tmp_path: Path):
    result, output_root, _, _, _, _, form_bytes, config_bytes = _score(tmp_path)
    expected = _prediction_expected(result, form_bytes=form_bytes, config_bytes=config_bytes)
    verified = verify_research_prediction_bundle(result.bundle_dir, output_root=output_root, expected=expected)
    assert verified.prediction == result.prediction
    prediction_path = result.bundle_dir / "prediction.json"
    original = prediction_path.read_bytes()
    prediction_path.chmod(0o600)
    value = json.loads(original)
    value["predictions"][0]["full_probability"] = 0.25
    prediction_path.write_bytes(canonical_bytes(value))
    with pytest.raises(ManualResearchScoringRejected, match="MANIFEST_INVALID"):
        verify_research_prediction_bundle(result.bundle_dir, output_root=output_root, expected=expected)
    prediction_path.write_bytes(original)


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda value: value["runners"].pop(), "FORM_RUNNER_SET_INVALID"),
        (lambda value: value["runners"][0]["history"][0].update(event_timestamp="2026-08-05T01:00:00+00:00"), "FORM_HISTORY_AFTER_CUTOFF"),
        (lambda value: value["runners"][0]["history"][1].update(prior_margin=float("nan")), "FORM_NOT_CANONICAL"),
        (lambda value: value["target"].update(race_identity_sha256="0" * 64), "FORM_TARGET_BINDING_MISMATCH"),
        (lambda value: value["runners"][0]["history"][0].update(prior_race_id="Race 1 - RICH - 2026-08-05"), "FORM_HISTORY_INVALID"),
    ],
)
def test_form_provenance_temporal_and_runner_binding_fail_closed(tmp_path: Path, mutator, error):
    _, _, _, _, _, sealed = _sealed_fixture(tmp_path)
    value = json.loads(_form(sealed))
    mutator(value)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True).encode() + b"\n"
    with pytest.raises(ManualResearchScoringRejected, match=error):
        _score(tmp_path / "nested", form_bytes=raw)


def test_extra_runner_and_nonfinite_feature_fail_closed(tmp_path: Path, monkeypatch):
    _, _, execution, expected, identity, sealed = _sealed_fixture(tmp_path / "extra")
    value = json.loads(_form(sealed))
    value["runners"].append({"box_number": 3, "display_name": "Gamma Dog", "history": []})
    form_bytes = canonical_bytes(value)
    with pytest.raises(ManualResearchScoringRejected, match="RUNNER_SET_MISMATCH"):
        _score(tmp_path / "extra-score", form_bytes=form_bytes)

    import src.predictor.manual_research_scoring as module

    original_features = module._features

    def nonfinite_features(*args, **kwargs):
        result = original_features(*args, **kwargs)
        result["career_avg_finish"] = float("nan")
        return result

    monkeypatch.setattr(module, "_features", nonfinite_features)
    config_bytes = canonical_bytes(_config())
    output_root = tmp_path / "feature-output"
    output_root.mkdir()
    with pytest.raises(ManualResearchScoringRejected, match="SCORING_BLOCKED"):
        score_verified_manual_evidence(
            sealed_bundle_dir=sealed.bundle_dir,
            run_dir=execution.run_dir,
            evidence_expected=expected,
            evidence_identity=identity,
            embedded_form_bytes=_form(sealed),
            form_sha256=hashlib.sha256(_form(sealed)).hexdigest(),
            config_bytes=config_bytes,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            frozen_model=MODEL,
            expected_model_sha256=MODEL.model_sha256,
            expected_model_manifest_sha256=MODEL.manifest_sha256,
            scoring_identity=SCORE_IDENTITY,
            output_root=output_root,
        )


def test_nonfinite_sealed_odds_fail_closed(tmp_path: Path, monkeypatch):
    _, _, execution, expected, identity, sealed = _sealed_fixture(tmp_path)
    import src.predictor.manual_research_scoring as module

    odds = {**sealed.normalized_odds, "runners": [dict(row) for row in sealed.normalized_odds["runners"]]}
    odds["runners"][0]["decimal_odds"] = float("nan")
    monkeypatch.setattr(
        module,
        "verify_manual_evidence_bundle",
        lambda *args, **kwargs: replace(sealed, normalized_odds=odds),
    )
    config_bytes = canonical_bytes(_config())
    output_root = tmp_path / "output"
    output_root.mkdir()
    form_bytes = _form(sealed)
    with pytest.raises(ManualResearchScoringRejected, match="SCORING_BLOCKED"):
        score_verified_manual_evidence(
            sealed_bundle_dir=sealed.bundle_dir,
            run_dir=execution.run_dir,
            evidence_expected=expected,
            evidence_identity=identity,
            embedded_form_bytes=form_bytes,
            form_sha256=hashlib.sha256(form_bytes).hexdigest(),
            config_bytes=config_bytes,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            frozen_model=MODEL,
            expected_model_sha256=MODEL.model_sha256,
            expected_model_manifest_sha256=MODEL.manifest_sha256,
            scoring_identity=SCORE_IDENTITY,
            output_root=output_root,
        )


def test_model_and_config_drift_fail_closed(tmp_path: Path):
    with pytest.raises(ManualResearchScoringRejected, match="MODEL_HASH_DRIFT"):
        _score(tmp_path / "model", expected_model_sha256="0" * 64)
    drifted = _config()
    drifted["ranking"]["tie_break"] = "display_name"
    with pytest.raises(ManualResearchScoringRejected, match="CONFIG_HASH_MISMATCH"):
        _score(tmp_path / "config", config=drifted, config_sha256="0" * 64)


def test_missing_form_fails_closed(tmp_path: Path):
    _, _, execution, expected, identity, sealed = _sealed_fixture(tmp_path)
    config_bytes = canonical_bytes(_config())
    output_root = tmp_path / "out"
    output_root.mkdir()
    with pytest.raises(ManualResearchScoringRejected, match="FORM_MISSING"):
        score_verified_manual_evidence(
            sealed_bundle_dir=sealed.bundle_dir, run_dir=execution.run_dir,
            evidence_expected=expected, evidence_identity=identity,
            embedded_form_bytes=b"", form_sha256="0" * 64,
            config_bytes=config_bytes, config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            frozen_model=MODEL, expected_model_sha256=MODEL.model_sha256,
            expected_model_manifest_sha256=MODEL.manifest_sha256,
            scoring_identity=SCORE_IDENTITY, output_root=output_root,
        )


def test_partial_publication_never_exposes_final_bundle(tmp_path: Path):
    with pytest.raises(RuntimeError, match="stop-stage"):
        _score(tmp_path, stage_hook=lambda stage, path: (_ for _ in ()).throw(RuntimeError("stop-stage")) if stage == "manifest_written" else None)
    assert not [
        path for path in (tmp_path / "research-predictions").iterdir()
        if len(path.name) == 64 and path.is_dir()
    ]


def test_output_verifier_rejects_unsafe_and_extra_members(tmp_path: Path):
    result, output_root, _, _, _, _, form_bytes, config_bytes = _score(tmp_path)
    expected = _prediction_expected(result, form_bytes=form_bytes, config_bytes=config_bytes)
    bundle = result.bundle_dir
    prediction = bundle / "prediction.json"
    bundle.chmod(0o700)
    prediction.chmod(0o600)
    (bundle / "extra.json").write_bytes(b"{}")
    with pytest.raises(ManualResearchScoringRejected, match="PARTIAL_OR_EXTRA_OUTPUT"):
        verify_research_prediction_bundle(bundle, output_root=output_root, expected=expected)
    (bundle / "extra.json").unlink()
    prediction.rename(bundle / "prediction.saved")
    (bundle / "prediction.json").symlink_to("prediction.saved")
    with pytest.raises(ManualResearchScoringRejected, match="UNSAFE_OUTPUT_PATH"):
        verify_research_prediction_bundle(bundle, output_root=output_root, expected=expected)


def test_forbidden_persistence_and_result_surfaces_are_not_imported_or_touched(tmp_path: Path, monkeypatch):
    import src.predictor.manual_research_scoring as module

    assert "sqlite3" not in module.__dict__
    assert "outcome" not in module.__dict__
    assert "phase7" not in module.__dict__
    touched = []
    monkeypatch.setattr(module, "verify_manual_evidence_bundle", lambda *args, **kwargs: touched.append(True) or (_ for _ in ()).throw(AssertionError("patched verifier")))
    config = _config()
    with pytest.raises(AssertionError, match="patched verifier"):
        _score(tmp_path, config=config)
    assert touched == [True]
