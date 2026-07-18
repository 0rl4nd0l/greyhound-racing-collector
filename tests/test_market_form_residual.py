from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from src.predictor import market_form_residual as residual_module
from src.predictor.market_form_residual import (
    OUTCOME_FIELDS,
    PREDICTION_FIELDS,
    PROVENANCE_FIELDS,
    RUNNER_FIELDS,
    SHADOW_RECORD_FIELDS,
    ResidualContractError,
    _effective_state_sha256,
    append_shadow_record,
    load_frozen_model,
    score_race,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/frozen_models/market_form_residual_v1"


def canonical_bytes(value):
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def reseal_record(record):
    content = copy.deepcopy(record)
    content.pop("record_key", None)
    content.pop("record_checksum_sha256", None)
    checksum = hashlib.sha256(canonical_bytes(content)).hexdigest()
    record["record_checksum_sha256"] = checksum
    record["record_key"] = hashlib.sha256(
        canonical_bytes(
            {
                "record_checksum_sha256": checksum,
                "schema_version": record["schema_version"],
            }
        )
    ).hexdigest()


def load_fixture():
    frozen = load_frozen_model(
        ARTIFACT_DIR / "model.json", ARTIFACT_DIR / "manifest.json"
    )
    fixture = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))[
        "fixed_fixture"
    ]
    return (
        frozen,
        copy.deepcopy(fixture["runners"]),
        copy.deepcopy(fixture["provenance"]),
    )


def append_record(path, frozen, record, runners, provenance):
    return append_shadow_record(
        path,
        record,
        frozen=frozen,
        runners=runners,
        provenance=provenance,
    )


def shifted_race_inputs(runners, provenance, suffix):
    shifted_runners = copy.deepcopy(runners)
    shifted_provenance = copy.deepcopy(provenance)
    original_race_id = shifted_provenance["race_id"]
    shifted_race_id = f"{original_race_id}-{suffix}"
    shifted_runner_ids = []
    for runner in shifted_runners:
        runner["race_id"] = shifted_race_id
        runner["runner_id"] = (
            shifted_race_id + runner["runner_id"][len(original_race_id) :]
        )
        shifted_runner_ids.append(runner["runner_id"])
    shifted_provenance["race_id"] = shifted_race_id
    shifted_provenance["expected_runner_ids"] = shifted_runner_ids
    shifted_provenance["runner_set_sha256"] = hashlib.sha256(
        ("\n".join(sorted(shifted_runner_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    return shifted_runners, shifted_provenance


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


def test_loaded_score_state_is_deeply_immutable():
    frozen, _, _ = load_fixture()

    with pytest.raises(TypeError):
        frozen.model["beta"][0] += 100.0
    with pytest.raises(TypeError):
        frozen.model["algorithm"]["optimizer_options"]["maxiter"] = 501
    with pytest.raises(TypeError):
        frozen.manifest["derivation_contract"]["half_strength"] = 0.75
    with pytest.raises(FrozenInstanceError):
        frozen.model_sha256 = "0" * 64


@pytest.mark.parametrize(
    "contract",
    [
        RUNNER_FIELDS,
        PROVENANCE_FIELDS,
        PREDICTION_FIELDS,
        SHADOW_RECORD_FIELDS,
        OUTCOME_FIELDS,
    ],
)
def test_v2_schema_contract_collections_are_immutable(contract):
    with pytest.raises(AttributeError):
        contract.add("forged_field")


def test_score_arrays_are_read_only_and_copies_cannot_alias_state():
    frozen, runners, provenance = load_fixture()
    baseline = score_race(frozen, runners, provenance)

    with pytest.raises(ValueError, match="read-only"):
        frozen.beta[0] += 100.0
    with pytest.raises(ValueError):
        frozen.beta.flags.writeable = True

    copied_beta = frozen.beta.copy()
    assert copied_beta.flags.writeable
    assert not np.shares_memory(copied_beta, frozen.beta)
    copied_beta[0] += 100.0
    assert score_race(frozen, runners, provenance) == baseline


def test_recomputed_effective_state_rejects_encapsulated_array_tampering():
    frozen, runners, provenance = load_fixture()
    tampered_beta = frozen.beta.copy()
    tampered_beta[0] += 100.0
    tampered_beta.setflags(write=False)
    object.__setattr__(frozen, "beta", tampered_beta)

    with pytest.raises(ResidualContractError, match="effective_state_sha256_mismatch"):
        score_race(frozen, runners, provenance)


def test_mutation_cannot_be_legitimized_with_a_new_effective_state_key():
    frozen, runners, provenance = load_fixture()
    tampered_beta = frozen.beta.copy()
    tampered_beta[0] += 100.0
    tampered_beta.setflags(write=False)
    object.__setattr__(frozen, "beta", tampered_beta)
    object.__setattr__(
        frozen, "effective_state_sha256", _effective_state_sha256(frozen)
    )

    with pytest.raises(
        ResidualContractError, match="encapsulated_score_state_mismatch"
    ):
        score_race(frozen, runners, provenance)


def test_recomputed_artifact_state_rejects_cached_hash_tampering():
    frozen, runners, provenance = load_fixture()
    object.__setattr__(frozen, "model_sha256", "0" * 64)

    with pytest.raises(ResidualContractError, match="model_state_sha256_mismatch"):
        score_race(frozen, runners, provenance)


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
    assert sum(
        row["full_probability"] for row in first["predictions"]
    ) == pytest.approx(1.0)
    assert sum(
        row["half_probability"] for row in first["predictions"]
    ) == pytest.approx(1.0)
    assert sum(
        row["market_probability"] for row in first["predictions"]
    ) == pytest.approx(1.0)
    assert first["effective_state_sha256"] == frozen.effective_state_sha256
    assert len(first["effective_state_sha256"]) == 64


def test_reordered_runner_input_produces_byte_identical_record(tmp_path):
    frozen, runners, provenance = load_fixture()
    first = score_race(frozen, runners, provenance)
    path = tmp_path / "ordered.jsonl"
    assert append_record(path, frozen, first, runners, provenance) == "APPENDED"
    provenance["expected_runner_ids"].reverse()
    reversed_runners = list(reversed(runners))
    second = score_race(frozen, reversed_runners, provenance)
    assert second == first
    assert json.dumps(second, sort_keys=True) == json.dumps(first, sort_keys=True)
    assert (
        append_record(path, frozen, second, reversed_runners, provenance)
        == "EXACT_REPLAY"
    )


def test_full_and_half_are_derived_from_one_adjustment():
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    market = np.asarray([row["market_probability"] for row in record["predictions"]])
    adjustment = np.asarray(
        [row["residual_adjustment"] for row in record["predictions"]]
    )
    full_scores = np.log(market) + adjustment
    full = np.exp(full_scores - np.max(full_scores))
    full /= full.sum()
    half_scores = np.log(market) + 0.5 * adjustment
    half = np.exp(half_scores - np.max(half_scores))
    half /= half.sum()
    assert np.array_equal(
        full, [row["full_probability"] for row in record["predictions"]]
    )
    assert np.array_equal(
        half, [row["half_probability"] for row in record["predictions"]]
    )


def test_missing_values_use_frozen_preprocessing_and_remain_finite():
    frozen, runners, provenance = load_fixture()
    runners[0]["features"]["recent_finish_mean_3"] = None
    del runners[1]["features"]["recent_finish_mean_3"]
    record = score_race(frozen, runners, provenance)
    values = [
        row[key]
        for row in record["predictions"]
        for key in (
            "market_probability",
            "residual_adjustment",
            "full_probability",
            "half_probability",
        )
    ]
    assert all(np.isfinite(values))
    assert sum(
        row["full_probability"] for row in record["predictions"]
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutate,error",
    [
        (lambda rows, provenance: rows.pop(), "race_incomplete_or_runner_set_mismatch"),
        (
            lambda rows, provenance: rows[0]["features"].__setitem__(
                "unknown_feature", 1.0
            ),
            "unexpected_features",
        ),
        (
            lambda rows, provenance: rows[0].__setitem__("actual_win", 0),
            "contains_outcome",
        ),
        (
            lambda rows, provenance: rows[0].__setitem__("strict_win_odds", 1.0),
            "strict_win_odds_invalid",
        ),
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
    with pytest.raises(
        ResidualContractError, match="declared_runner_set_hash_mismatch"
    ):
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
    assert append_record(path, frozen, record, runners, provenance) == "APPENDED"
    original = path.read_bytes()
    assert append_record(path, frozen, record, runners, provenance) == "EXACT_REPLAY"
    assert path.read_bytes() == original
    assert hashlib.sha256(original).hexdigest()

    conflicting = copy.deepcopy(record)
    conflicting["predictions"][0]["full_probability"] += 1e-6
    with pytest.raises(
        ResidualContractError, match="shadow_record_not_canonical_score"
    ):
        append_record(path, frozen, conflicting, runners, provenance)
    assert path.read_bytes() == original

    changed_provenance = copy.deepcopy(provenance)
    changed_provenance["score_timestamp"] = "2026-06-11T18:32:15+10:00"
    changed_timestamp = score_race(frozen, runners, changed_provenance)
    with pytest.raises(ResidualContractError, match="conflicting_shadow_duplicate"):
        append_record(path, frozen, changed_timestamp, runners, changed_provenance)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("history_kind", "expected_error"),
    [
        ("identical", "duplicate_shadow_history_identity"),
        ("conflicting", "conflicting_shadow_duplicate"),
    ],
)
def test_repeated_history_identity_blocks_unrelated_append_without_byte_change(
    tmp_path, history_kind, expected_error
):
    frozen, runners, provenance = load_fixture()
    first = score_race(frozen, runners, provenance)
    if history_kind == "identical":
        second = copy.deepcopy(first)
    else:
        changed_provenance = copy.deepcopy(provenance)
        changed_provenance["score_timestamp"] = "2026-06-11T18:32:15+10:00"
        second = score_race(frozen, runners, changed_provenance)

    path = tmp_path / f"repeated-{history_kind}.jsonl"
    path.write_bytes(canonical_bytes(first) + canonical_bytes(second))
    before = path.read_bytes()
    unrelated_runners, unrelated_provenance = shifted_race_inputs(
        runners, provenance, history_kind
    )
    unrelated = score_race(frozen, unrelated_runners, unrelated_provenance)

    with pytest.raises(ResidualContractError, match=f"^{expected_error}$"):
        append_record(
            path,
            frozen,
            unrelated,
            unrelated_runners,
            unrelated_provenance,
        )
    assert path.read_bytes() == before


def test_unique_multi_row_history_allows_append_and_exact_replay(tmp_path):
    frozen, runners, provenance = load_fixture()
    first = score_race(frozen, runners, provenance)
    path = tmp_path / "unique-multi-row.jsonl"
    path.write_bytes(canonical_bytes(first))

    unrelated_runners, unrelated_provenance = shifted_race_inputs(
        runners, provenance, "unique"
    )
    unrelated = score_race(frozen, unrelated_runners, unrelated_provenance)
    assert (
        append_record(
            path,
            frozen,
            unrelated,
            unrelated_runners,
            unrelated_provenance,
        )
        == "APPENDED"
    )
    unique_history = path.read_bytes()
    assert unique_history == canonical_bytes(first) + canonical_bytes(unrelated)
    assert append_record(path, frozen, first, runners, provenance) == "EXACT_REPLAY"
    assert path.read_bytes() == unique_history


def test_history_validation_and_append_are_lock_scoped(tmp_path, monkeypatch):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    path = tmp_path / "lock-scoped.jsonl"
    path.write_bytes(canonical_bytes(existing))
    unrelated_runners, unrelated_provenance = shifted_race_inputs(
        runners, provenance, "lock"
    )
    unrelated = score_race(frozen, unrelated_runners, unrelated_provenance)
    state = {"locked": False, "validated": False, "synced": False}
    real_flock = residual_module.fcntl.flock
    real_validate = residual_module._validate_existing_shadow_record
    real_fsync = residual_module.os.fsync

    def tracked_flock(file_descriptor, operation):
        assert operation == residual_module.fcntl.LOCK_EX
        real_flock(file_descriptor, operation)
        state["locked"] = True

    def tracked_validate(*args, **kwargs):
        assert state["locked"]
        state["validated"] = True
        return real_validate(*args, **kwargs)

    def tracked_fsync(file_descriptor):
        assert state["locked"]
        state["synced"] = True
        return real_fsync(file_descriptor)

    monkeypatch.setattr(residual_module.fcntl, "flock", tracked_flock)
    monkeypatch.setattr(
        residual_module, "_validate_existing_shadow_record", tracked_validate
    )
    monkeypatch.setattr(residual_module.os, "fsync", tracked_fsync)

    assert (
        append_record(
            path,
            frozen,
            unrelated,
            unrelated_runners,
            unrelated_provenance,
        )
        == "APPENDED"
    )
    assert state == {"locked": True, "validated": True, "synced": True}


def test_v2_record_stores_complete_inputs_and_binds_full_content():
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)

    assert record["schema_version"] == "market_form_residual_shadow_record_v2"
    assert record["inputs"]["provenance"] == {
        "expected_runner_ids": sorted(provenance["expected_runner_ids"]),
        "jump_timestamp": provenance["jump_timestamp"],
        "race_id": provenance["race_id"],
        "runner_set_sha256": provenance["runner_set_sha256"],
        "score_timestamp": provenance["score_timestamp"],
    }
    assert [row["runner_id"] for row in record["inputs"]["runners"]] == sorted(
        provenance["expected_runner_ids"]
    )
    assert all(
        tuple(row["features"]) == tuple(frozen.feature_order)
        for row in record["inputs"]["runners"]
    )

    expected = copy.deepcopy(record)
    reseal_record(expected)
    assert expected == record


@pytest.mark.parametrize("field", ["record_key", "record_checksum_sha256"])
def test_existing_history_rejects_mismatched_content_digest(tmp_path, field):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    record[field] = "0" * 64
    path = tmp_path / f"mismatched-{field}.jsonl"
    path.write_bytes(canonical_bytes(record))

    with pytest.raises(
        ResidualContractError, match="existing_shadow_checksum_mismatch"
    ):
        append_record(
            path, frozen, score_race(frozen, runners, provenance), runners, provenance
        )


def test_existing_history_rejects_accidental_prediction_edit(tmp_path):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    existing["predictions"][0]["full_probability"] += 1e-6
    path = tmp_path / "accidental-edit.jsonl"
    path.write_bytes(canonical_bytes(existing))

    with pytest.raises(
        ResidualContractError, match="existing_shadow_checksum_mismatch"
    ):
        append_record(
            path, frozen, score_race(frozen, runners, provenance), runners, provenance
        )


def test_existing_history_rejects_resealed_prediction_input_mismatch(tmp_path):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    existing["inputs"]["runners"][0]["features"]["prior_start_count"] = 1.0
    reseal_record(existing)
    path = tmp_path / "prediction-input-mismatch.jsonl"
    path.write_bytes(canonical_bytes(existing))

    with pytest.raises(
        ResidualContractError, match="existing_shadow_prediction_input_mismatch"
    ):
        append_record(
            path, frozen, score_race(frozen, runners, provenance), runners, provenance
        )


@pytest.mark.parametrize("history_shape", ["v1", "mixed", "insufficient_v2"])
def test_legacy_mixed_and_insufficient_history_require_migration(
    tmp_path, history_shape
):
    frozen, runners, provenance = load_fixture()
    current = score_race(frozen, runners, provenance)
    legacy = copy.deepcopy(current)
    legacy["schema_version"] = "market_form_residual_shadow_record_v1"
    reseal_record(legacy)
    insufficient = copy.deepcopy(current)
    insufficient.pop("inputs")
    reseal_record(insufficient)
    rows = {
        "v1": [legacy],
        "mixed": [current, legacy],
        "insufficient_v2": [insufficient],
    }[history_shape]
    path = tmp_path / f"{history_shape}.jsonl"
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))

    with pytest.raises(ResidualContractError, match="^history_migration_required$"):
        append_record(path, frozen, current, runners, provenance)


@pytest.mark.parametrize(
    "mutation",
    ["runner_order", "unsupported_field", "invalid_runner_id", "invalid_variants"],
)
def test_existing_history_rejects_noncanonical_and_unsupported_content(
    tmp_path, mutation
):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    if mutation == "runner_order":
        existing["inputs"]["runners"].reverse()
        expected_error = "existing_shadow_noncanonical_order"
    elif mutation == "unsupported_field":
        existing["unsupported"] = True
        expected_error = "existing_shadow_unsupported_fields"
    elif mutation == "invalid_runner_id":
        existing["inputs"]["runners"][0]["runner_id"] = None
        expected_error = "existing_shadow_noncanonical_order"
    else:
        existing["variants"] = []
        expected_error = "existing_shadow_invalid_record:variants"
    reseal_record(existing)
    path = tmp_path / f"{mutation}.jsonl"
    path.write_bytes(canonical_bytes(existing))

    with pytest.raises(ResidualContractError, match=expected_error):
        append_record(
            path, frozen, score_race(frozen, runners, provenance), runners, provenance
        )


def test_complete_coordinated_rewrite_is_documented_out_of_scope(tmp_path):
    """A host actor can replace a complete canonical row and recompute its digests."""

    frozen, runners, provenance = load_fixture()
    runners[0]["features"]["prior_start_count"] = 1.0
    rewritten = score_race(frozen, runners, provenance)
    path = tmp_path / "coordinated-host-rewrite-out-of-scope.jsonl"
    path.write_bytes(canonical_bytes(rewritten))

    assert append_record(path, frozen, rewritten, runners, provenance) == "EXACT_REPLAY"


def test_writer_rejects_model_mutation_between_score_and_append(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    tampered_beta = frozen.beta.copy()
    tampered_beta[0] += 100.0
    tampered_beta.setflags(write=False)
    object.__setattr__(frozen, "beta", tampered_beta)

    path = tmp_path / "between.jsonl"
    with pytest.raises(ResidualContractError, match="effective_state_sha256_mismatch"):
        append_record(path, frozen, record, runners, provenance)
    assert not path.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_key", "0" * 64),
        ("model_sha256", "0" * 64),
        ("manifest_sha256", "0" * 64),
        ("effective_state_sha256", "0" * 64),
        ("runner_set_sha256", "0" * 64),
        ("record_checksum_sha256", "0" * 64),
    ],
)
def test_writer_rejects_forged_identity_fields(tmp_path, field, value):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    record[field] = value

    path = tmp_path / f"forged-{field}.jsonl"
    with pytest.raises(
        ResidualContractError, match="shadow_record_not_canonical_score"
    ):
        append_record(path, frozen, record, runners, provenance)
    assert not path.exists()


def test_writer_rejects_forged_caller_inputs_even_when_resealed(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    record["inputs"]["runners"][0]["features"]["prior_start_count"] = 1.0
    reseal_record(record)

    path = tmp_path / "forged-inputs.jsonl"
    with pytest.raises(
        ResidualContractError, match="shadow_record_not_canonical_score"
    ):
        append_record(path, frozen, record, runners, provenance)
    assert not path.exists()


def test_append_only_writer_validates_new_and_existing_record_keys(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    invalid = copy.deepcopy(record)
    invalid["record_key"] = "0" * 64
    with pytest.raises(
        ResidualContractError, match="shadow_record_not_canonical_score"
    ):
        append_record(tmp_path / "invalid.jsonl", frozen, invalid, runners, provenance)

    existing = copy.deepcopy(record)
    existing["record_key"] = "0" * 64
    path = tmp_path / "existing.jsonl"
    path.write_bytes(
        (
            json.dumps(existing, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("utf-8")
    )
    with pytest.raises(
        ResidualContractError, match="existing_shadow_checksum_mismatch"
    ):
        append_record(path, frozen, record, runners, provenance)


def test_append_only_writer_rejects_outcomes_and_malformed_history(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    record["result"] = "unknown"
    with pytest.raises(ResidualContractError, match="contains_outcome"):
        append_record(tmp_path / "outcomes.jsonl", frozen, record, runners, provenance)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("not-json\n", encoding="utf-8")
    clean = score_race(frozen, runners, provenance)
    with pytest.raises(ResidualContractError, match="existing_shadow_invalid_json"):
        append_record(malformed, frozen, clean, runners, provenance)

    truncated = tmp_path / "truncated.jsonl"
    truncated.write_bytes(canonical_bytes(clean)[:-7])
    with pytest.raises(ResidualContractError, match="existing_shadow_invalid_json"):
        append_record(truncated, frozen, clean, runners, provenance)

    invalid_utf8 = tmp_path / "invalid-utf8.jsonl"
    invalid_utf8.write_bytes(b"\xff\n")
    with pytest.raises(ResidualContractError, match="existing_shadow_invalid_utf8"):
        append_record(invalid_utf8, frozen, clean, runners, provenance)


def test_append_only_writer_rejects_noncanonical_existing_record(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "noncanonical.jsonl"
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(
        ResidualContractError, match="existing_shadow_not_canonical_json"
    ):
        append_record(path, frozen, record, runners, provenance)


def test_artifact_files_are_canonical_json_bytes():
    for path in (ARTIFACT_DIR / "model.json", ARTIFACT_DIR / "manifest.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = (
            json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
            + "\n"
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
