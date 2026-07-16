from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.predictor.market_form_residual import (
    ResidualContractError,
    append_shadow_record,
    load_frozen_model,
    score_race,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/frozen_models/market_form_residual_v1"


def load_fixture():
    frozen = load_frozen_model(
        ARTIFACT_DIR / "model.json", ARTIFACT_DIR / "manifest.json"
    )
    fixture = frozen.manifest["fixed_fixture"]
    return frozen, copy.deepcopy(fixture["runners"]), copy.deepcopy(fixture["provenance"])


def write_tampered_artifacts(tmp_path, mutate_model=None, mutate_manifest=None):
    model = json.loads((ARTIFACT_DIR / "model.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    if mutate_model is not None:
        mutate_model(model)
    model_bytes = (
        json.dumps(model, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    manifest_bytes = (
        json.dumps(manifest, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    model_path = tmp_path / "model.json"
    manifest_path = tmp_path / "manifest.json"
    model_path.write_bytes(model_bytes)
    manifest_path.write_bytes(manifest_bytes)
    return model_path, manifest_path


def test_loads_hash_bound_frozen_artifact():
    frozen, _, _ = load_fixture()
    assert frozen.model["fit"]["race_count"] == 678
    assert frozen.model["fit"]["runner_count"] == 4752
    assert frozen.model["algorithm"]["strengths"] == {"full": 1.0, "half": 0.5}
    assert frozen.manifest["derivation_contract"]["shared_base_model_count"] == 1


def test_loader_rejects_tampered_model(tmp_path):
    model_path = tmp_path / "model.json"
    manifest_path = tmp_path / "manifest.json"
    model_path.write_bytes((ARTIFACT_DIR / "model.json").read_bytes() + b" ")
    manifest_path.write_bytes((ARTIFACT_DIR / "manifest.json").read_bytes())
    with pytest.raises(ResidualContractError, match="artifact_not_canonical_json"):
        load_frozen_model(model_path, manifest_path)


def test_loader_rejects_noncanonical_manifest(tmp_path):
    model_path = tmp_path / "model.json"
    manifest_path = tmp_path / "manifest.json"
    model_path.write_bytes((ARTIFACT_DIR / "model.json").read_bytes())
    manifest_path.write_bytes((ARTIFACT_DIR / "manifest.json").read_bytes() + b" ")
    with pytest.raises(ResidualContractError, match="artifact_not_canonical_json"):
        load_frozen_model(model_path, manifest_path)


@pytest.mark.parametrize(
    "mutate,error",
    [
        (
            lambda model: model.__setitem__("model_family", "conditional_logit"),
            "model_family_contract_mismatch",
        ),
        (
            lambda model: model["algorithm"].__setitem__("ridge_l2", 0.5),
            "algorithm_contract_mismatch",
        ),
        (
            lambda model: model["algorithm"].__setitem__("residual_cap", 0.5),
            "algorithm_contract_mismatch",
        ),
        (
            lambda model: model["algorithm"].__setitem__("market_offset_refit", True),
            "algorithm_contract_mismatch",
        ),
        (
            lambda model: model["algorithm"]["optimizer_options"].__setitem__(
                "maxiter", 501
            ),
            "algorithm_contract_mismatch",
        ),
        (
            lambda model: model["algorithm"].__setitem__("normalization", "other"),
            "algorithm_contract_mismatch",
        ),
        (
            lambda model: model["algorithm"]["strengths"].__setitem__("half", 0.6),
            "algorithm_contract_mismatch",
        ),
    ],
)
def test_loader_rejects_algorithm_contract_tampering(tmp_path, mutate, error):
    model_path, manifest_path = write_tampered_artifacts(tmp_path, mutate_model=mutate)
    with pytest.raises(ResidualContractError, match=error):
        load_frozen_model(model_path, manifest_path)


def test_fixed_fixture_predictions_are_identical_and_normalized():
    frozen, runners, provenance = load_fixture()
    first = score_race(frozen, runners, provenance)
    second = score_race(frozen, runners, provenance)
    assert first == second
    expected = frozen.manifest["fixed_fixture"]["expected"]
    assert [row["full_probability"] for row in first["predictions"]] == [
        row["probability"] for row in expected["full"]
    ]
    assert [row["half_probability"] for row in first["predictions"]] == [
        row["probability"] for row in expected["half"]
    ]
    assert sum(row["full_probability"] for row in first["predictions"]) == pytest.approx(1.0)
    assert sum(row["half_probability"] for row in first["predictions"]) == pytest.approx(1.0)
    assert sum(row["market_probability"] for row in first["predictions"]) == pytest.approx(1.0)


def test_reordered_runner_input_produces_byte_identical_record():
    frozen, runners, provenance = load_fixture()
    first = score_race(frozen, runners, provenance)
    provenance["expected_runner_ids"].reverse()
    second = score_race(frozen, list(reversed(runners)), provenance)
    assert second == first
    assert json.dumps(second, sort_keys=True) == json.dumps(first, sort_keys=True)


def test_full_and_half_are_derived_from_one_adjustment():
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    market = np.asarray([row["market_probability"] for row in record["predictions"]])
    adjustment = np.asarray([row["residual_adjustment"] for row in record["predictions"]])
    full_scores = np.log(market) + adjustment
    full = np.exp(full_scores - np.max(full_scores))
    full /= full.sum()
    half_scores = np.log(market) + 0.5 * adjustment
    half = np.exp(half_scores - np.max(half_scores))
    half /= half.sum()
    assert np.array_equal(full, [row["full_probability"] for row in record["predictions"]])
    assert np.array_equal(half, [row["half_probability"] for row in record["predictions"]])


def test_missing_values_use_frozen_preprocessing_and_remain_finite():
    frozen, runners, provenance = load_fixture()
    runners[0]["features"]["recent_finish_mean_3"] = None
    del runners[1]["features"]["recent_finish_mean_3"]
    record = score_race(frozen, runners, provenance)
    values = [
        row[key]
        for row in record["predictions"]
        for key in ("market_probability", "residual_adjustment", "full_probability", "half_probability")
    ]
    assert all(np.isfinite(values))
    assert sum(row["full_probability"] for row in record["predictions"]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda rows, provenance: rows.pop(), "race_incomplete_or_runner_set_mismatch"),
        (
            lambda rows, provenance: rows[0]["features"].__setitem__("unknown_feature", 1.0),
            "unexpected_features",
        ),
        (lambda rows, provenance: rows[0].__setitem__("actual_win", 0), "contains_outcome"),
        (lambda rows, provenance: rows[0].__setitem__("strict_win_odds", 1.0), "strict_win_odds_invalid"),
        (
            lambda rows, provenance: rows[0].__setitem__(
                "feature_freeze_timestamp", provenance["jump_timestamp"]
            ),
            "source_timestamp_not_prejump",
        ),
        (
            lambda rows, provenance: provenance.__setitem__(
                "score_timestamp", provenance["jump_timestamp"]
            ),
            "score_timestamp_not_prejump",
        ),
    ],
)
def test_scoring_contract_fails_closed(mutate, error):
    frozen, runners, provenance = load_fixture()
    mutate(runners, provenance)
    with pytest.raises(ResidualContractError, match=error):
        score_race(frozen, runners, provenance)


def test_runner_set_hash_and_identity_are_enforced():
    frozen, runners, provenance = load_fixture()
    provenance["runner_set_sha256"] = "0" * 64
    with pytest.raises(ResidualContractError, match="declared_runner_set_hash_mismatch"):
        score_race(frozen, runners, provenance)

    frozen, runners, provenance = load_fixture()
    runners[0]["runner_id"] += "X"
    with pytest.raises(ResidualContractError, match="runner_identity_mismatch"):
        score_race(frozen, runners, provenance)

    frozen, runners, provenance = load_fixture()
    provenance["expected_runner_ids"] = [1, "1"]
    with pytest.raises(ResidualContractError, match="expected_runner_ids_duplicate"):
        score_race(frozen, runners, provenance)


def test_append_only_writer_is_idempotent_and_rejects_conflict(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "shadow.jsonl"
    assert append_shadow_record(path, record) == "APPENDED"
    original = path.read_bytes()
    assert append_shadow_record(path, record) == "EXACT_REPLAY"
    assert path.read_bytes() == original
    assert hashlib.sha256(original).hexdigest()

    conflicting = copy.deepcopy(record)
    conflicting["predictions"][0]["full_probability"] += 1e-6
    with pytest.raises(ResidualContractError, match="conflicting_shadow_duplicate"):
        append_shadow_record(path, conflicting)
    assert path.read_bytes() == original

    changed_timestamp = copy.deepcopy(record)
    changed_timestamp["score_timestamp"] = "2026-06-11T18:32:15+10:00"
    with pytest.raises(ResidualContractError, match="conflicting_shadow_duplicate"):
        append_shadow_record(path, changed_timestamp)
    assert path.read_bytes() == original


def test_append_only_writer_validates_new_and_existing_record_keys(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    invalid = copy.deepcopy(record)
    invalid["record_key"] = "0" * 64
    with pytest.raises(ResidualContractError, match="shadow_record_key_mismatch"):
        append_shadow_record(tmp_path / "invalid.jsonl", invalid)

    existing = copy.deepcopy(record)
    existing["record_key"] = "0" * 64
    path = tmp_path / "existing.jsonl"
    path.write_bytes(
        (json.dumps(existing, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    with pytest.raises(ResidualContractError, match="existing_shadow_invalid_record"):
        append_shadow_record(path, record)


def test_append_only_writer_rejects_outcomes_and_malformed_history(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    record["result"] = "unknown"
    with pytest.raises(ResidualContractError, match="contains_outcome"):
        append_shadow_record(tmp_path / "outcomes.jsonl", record)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    clean = score_race(frozen, runners, provenance)
    with pytest.raises(ResidualContractError, match="existing_shadow_invalid_json"):
        append_shadow_record(malformed, clean)


def test_artifact_files_are_canonical_json_bytes():
    for path in (ARTIFACT_DIR / "model.json", ARTIFACT_DIR / "manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = (
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        assert path.read_bytes() == expected


def test_loader_reads_and_hashes_each_artifact_from_one_open(monkeypatch):
    watched = {
        (ARTIFACT_DIR / "model.json").resolve(),
        (ARTIFACT_DIR / "manifest.json").resolve(),
    }
    counts = {path: 0 for path in watched}
    original = Path.open

    def counted(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved in counts:
            counts[resolved] += 1
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted)
    load_frozen_model()

    assert counts == {path: 1 for path in watched}
