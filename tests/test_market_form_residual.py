from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path
from threading import Barrier

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


def output_snapshot(path):
    exists = path.exists()
    raw = path.read_bytes() if exists else b""
    return {
        "exists": exists,
        "bytes": raw,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(raw.splitlines()),
    }


def assert_output_snapshot(path, expected):
    actual = output_snapshot(path)
    assert actual["exists"] is expected["exists"]
    assert actual["bytes"] == expected["bytes"]
    assert actual["sha256"] == expected["sha256"]
    assert actual["rows"] == expected["rows"]


def open_descriptors_for_path(path):
    fd_root = Path("/proc/self/fd")
    assert fd_root.is_dir()
    expected_targets = {str(path), f"{path} (deleted)"}
    descriptors = []
    for fd_path in fd_root.iterdir():
        try:
            target = os.readlink(fd_path)
        except FileNotFoundError:
            continue
        if target in expected_targets:
            descriptors.append(int(fd_path.name))
    return sorted(descriptors)


def distinct_record(frozen, runners, provenance, suffix):
    shifted_runners, shifted_provenance = shifted_race_inputs(
        runners, provenance, suffix
    )
    return (
        score_race(frozen, shifted_runners, shifted_provenance),
        shifted_runners,
        shifted_provenance,
    )


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
def test_v3_schema_contract_collections_are_immutable(contract):
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
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text())
    first = score_race(frozen, runners, provenance)
    second = score_race(frozen, runners, provenance)
    assert first == second
    assert [row["full_probability"] for row in first["predictions"]] == [
        0.4865438888325983,
        0.04255251747956415,
        0.2231390938439158,
        0.0802789211704884,
        0.03806490424396211,
        0.12942067442947117,
    ]
    assert [row["half_probability"] for row in first["predictions"]] == [
        0.4874257548450749,
        0.0429230211436064,
        0.2235435358427413,
        0.07831854008896694,
        0.03813389729082056,
        0.1296552507887899,
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
    for variant, field in (("full", "full_probability"), ("half", "half_probability")):
        prior = {
            row["runner_id"]: row["probability"]
            for row in manifest["fixed_fixture"]["expected"][variant]
        }
        current = {row["runner_id"]: row[field] for row in first["predictions"]}
        assert max(abs(current[key] - prior[key]) for key in prior) <= 3e-16
        assert sorted(prior, key=prior.get, reverse=True) == sorted(
            current, key=current.get, reverse=True
        )


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
    lock_path, _ = residual_module._shadow_transaction_paths(path)
    lock_path.touch(mode=0o600)
    lock_inode = lock_path.stat().st_ino
    state = {
        "locked": False,
        "cleaned": False,
        "validated": False,
        "published": False,
        "synced": False,
    }
    real_flock = residual_module.fcntl.flock
    real_remove = residual_module._remove_staged_shadow_file
    real_validate = residual_module._validate_existing_shadow_record
    real_publish = residual_module._publish_staged_shadow_file
    real_fsync = residual_module.os.fsync

    def tracked_flock(file_descriptor, operation):
        assert operation == residual_module.fcntl.LOCK_EX
        real_flock(file_descriptor, operation)
        assert os.fstat(file_descriptor).st_ino == lock_inode
        state["locked"] = True

    def tracked_remove(*args, **kwargs):
        assert state["locked"]
        state["cleaned"] = True
        return real_remove(*args, **kwargs)

    def tracked_validate(*args, **kwargs):
        assert state["locked"]
        state["validated"] = True
        return real_validate(*args, **kwargs)

    def tracked_publish(*args, **kwargs):
        assert state["locked"]
        assert lock_path.stat().st_ino == lock_inode
        state["published"] = True
        return real_publish(*args, **kwargs)

    def tracked_fsync(file_descriptor):
        assert state["locked"]
        state["synced"] = True
        return real_fsync(file_descriptor)

    monkeypatch.setattr(residual_module.fcntl, "flock", tracked_flock)
    monkeypatch.setattr(residual_module, "_remove_staged_shadow_file", tracked_remove)
    monkeypatch.setattr(
        residual_module, "_validate_existing_shadow_record", tracked_validate
    )
    monkeypatch.setattr(residual_module, "_publish_staged_shadow_file", tracked_publish)
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
    assert state == {
        "locked": True,
        "cleaned": True,
        "validated": True,
        "published": True,
        "synced": True,
    }
    assert lock_path.stat().st_ino == lock_inode


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize(
    "fault_point",
    [
        "_write_staged_shadow_bytes",
        "_flush_staged_shadow_file",
        "_fsync_staged_shadow_file",
        "_publish_staged_shadow_file",
    ],
)
def test_precommit_failures_preserve_exact_target_and_retry_cleanly(
    tmp_path, monkeypatch, target_exists, fault_point
):
    frozen, runners, provenance = load_fixture()
    original = score_race(frozen, runners, provenance)
    path = tmp_path / f"precommit-{target_exists}-{fault_point}.jsonl"
    if target_exists:
        assert append_record(path, frozen, original, runners, provenance) == "APPENDED"
        candidate, candidate_runners, candidate_provenance = distinct_record(
            frozen, runners, provenance, fault_point
        )
    else:
        candidate, candidate_runners, candidate_provenance = (
            original,
            runners,
            provenance,
        )
    before = output_snapshot(path)

    def injected_failure(*args, **kwargs):
        raise OSError(f"injected:{fault_point}")

    monkeypatch.setattr(residual_module, fault_point, injected_failure)
    with pytest.raises(ResidualContractError, match="^shadow_output_write_failed:"):
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
    assert_output_snapshot(path, before)
    _, staged_path = residual_module._shadow_transaction_paths(path)
    assert not staged_path.exists()

    monkeypatch.undo()
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "EXACT_REPLAY"
    )


def test_cleanup_failure_preserves_target_and_retry_removes_leftover(
    tmp_path, monkeypatch
):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    path = tmp_path / "cleanup-failure.jsonl"
    assert append_record(path, frozen, existing, runners, provenance) == "APPENDED"
    candidate, candidate_runners, candidate_provenance = distinct_record(
        frozen, runners, provenance, "cleanup"
    )
    before = output_snapshot(path)
    _, staged_path = residual_module._shadow_transaction_paths(path)
    staged_path.write_bytes(b"partial-leftover")

    def injected_cleanup_failure(*args, **kwargs):
        raise OSError("injected:cleanup")

    monkeypatch.setattr(
        residual_module, "_remove_staged_shadow_file", injected_cleanup_failure
    )
    with pytest.raises(ResidualContractError, match="^shadow_output_write_failed:"):
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
    assert_output_snapshot(path, before)
    assert staged_path.read_bytes() == b"partial-leftover"

    monkeypatch.undo()
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    assert not staged_path.exists()
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "EXACT_REPLAY"
    )


@pytest.mark.parametrize(
    ("fault_point", "target_exists"),
    [("fchmod", True), ("fdopen", False)],
)
def test_pretransfer_staged_fd_failures_do_not_leak_and_retry_cleanly(
    tmp_path, monkeypatch, fault_point, target_exists
):
    frozen, runners, provenance = load_fixture()
    original = score_race(frozen, runners, provenance)
    path = tmp_path / f"pretransfer-{fault_point}.jsonl"
    if target_exists:
        assert append_record(path, frozen, original, runners, provenance) == "APPENDED"
        candidate, candidate_runners, candidate_provenance = distinct_record(
            frozen, runners, provenance, fault_point
        )
    else:
        candidate, candidate_runners, candidate_provenance = (
            original,
            runners,
            provenance,
        )
    before = output_snapshot(path)
    _, staged_path = residual_module._shadow_transaction_paths(path)
    real_open = residual_module.os.open
    real_close = residual_module.os.close
    opened_staged_fds = []
    closed_staged_fds = []

    def record_open(open_path, *args, **kwargs):
        fd = real_open(open_path, *args, **kwargs)
        if Path(open_path) == staged_path:
            opened_staged_fds.append(fd)
        return fd

    def record_close(fd):
        if opened_staged_fds and fd == opened_staged_fds[-1]:
            closed_staged_fds.append(fd)
        return real_close(fd)

    def injected_failure(*args, **kwargs):
        raise OSError(f"injected:{fault_point}")

    monkeypatch.setattr(residual_module.os, "open", record_open)
    monkeypatch.setattr(residual_module.os, "close", record_close)
    monkeypatch.setattr(residual_module.os, fault_point, injected_failure)
    for _ in range(8):
        with pytest.raises(ResidualContractError, match="^shadow_output_write_failed:"):
            append_record(
                path,
                frozen,
                candidate,
                candidate_runners,
                candidate_provenance,
            )
        assert_output_snapshot(path, before)
        assert not staged_path.exists()
    assert closed_staged_fds == opened_staged_fds
    assert open_descriptors_for_path(staged_path) == []

    monkeypatch.undo()
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    assert output_snapshot(path)["rows"] == before["rows"] + 1
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "EXACT_REPLAY"
    )


@pytest.mark.parametrize("target_exists", [False, True])
def test_partial_wrapper_construction_retains_raw_staged_fd_ownership(
    tmp_path, monkeypatch, target_exists
):
    frozen, runners, provenance = load_fixture()
    original = score_race(frozen, runners, provenance)
    path = tmp_path / f"partial-wrapper-{target_exists}.jsonl"
    if target_exists:
        assert append_record(path, frozen, original, runners, provenance) == "APPENDED"
        path.chmod(0o640)
        candidate, candidate_runners, candidate_provenance = distinct_record(
            frozen, runners, provenance, "partial-wrapper"
        )
    else:
        candidate, candidate_runners, candidate_provenance = (
            original,
            runners,
            provenance,
        )
    before = output_snapshot(path)
    before_mode = stat.S_IMODE(path.stat().st_mode) if target_exists else None
    _, staged_path = residual_module._shadow_transaction_paths(path)
    unrelated_path = tmp_path / "partial-wrapper-unrelated.txt"
    unrelated_path.write_bytes(b"unrelated")
    real_open = residual_module.os.open
    real_close = residual_module.os.close
    real_fdopen = residual_module.os.fdopen
    staged_fd = None
    unrelated_fd = None
    closefd_values = []
    raw_close_attempts = []
    fd_count_before = len(os.listdir("/proc/self/fd"))

    def tracked_open(open_path, *args, **kwargs):
        nonlocal staged_fd
        fd = real_open(open_path, *args, **kwargs)
        if Path(open_path) == staged_path:
            staged_fd = fd
        return fd

    def partial_fdopen(fd, *args, **kwargs):
        nonlocal unrelated_fd
        closefd = kwargs["closefd"]
        closefd_values.append(closefd)
        try:
            return real_fdopen(
                fd,
                "w",
                encoding="definitely-not-a-codec",
                closefd=closefd,
            )
        except LookupError:
            unrelated_fd = real_open(unrelated_path, os.O_RDONLY)
            raise

    def tracked_close(fd):
        if fd == staged_fd:
            raw_close_attempts.append(fd)
        return real_close(fd)

    monkeypatch.setattr(residual_module.os, "open", tracked_open)
    monkeypatch.setattr(residual_module.os, "fdopen", partial_fdopen)
    monkeypatch.setattr(residual_module.os, "close", tracked_close)
    try:
        with pytest.raises(LookupError, match="unknown encoding"):
            append_record(
                path,
                frozen,
                candidate,
                candidate_runners,
                candidate_provenance,
            )
        assert closefd_values == [False]
        assert staged_fd is not None
        assert raw_close_attempts == [staged_fd]
        assert unrelated_fd is not None
        assert unrelated_fd != staged_fd
        assert os.read(unrelated_fd, len(b"unrelated")) == b"unrelated"
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(staged_fd)
        assert_output_snapshot(path, before)
        if target_exists:
            assert stat.S_IMODE(path.stat().st_mode) == before_mode
        assert not staged_path.exists()
        assert open_descriptors_for_path(staged_path) == []
    finally:
        if unrelated_fd is not None:
            try:
                real_close(unrelated_fd)
            except OSError:
                pass

    monkeypatch.undo()
    assert len(os.listdir("/proc/self/fd")) == fd_count_before
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    assert output_snapshot(path)["bytes"] == before["bytes"] + canonical_bytes(
        candidate
    )
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "EXACT_REPLAY"
    )


@pytest.mark.parametrize("fault_point", ["fchmod", "fdopen", "write", "flush", "fsync"])
def test_primary_staging_fault_precedes_wrapper_and_raw_close_failures(
    tmp_path, monkeypatch, fault_point
):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    path = tmp_path / f"primary-precedence-{fault_point}.jsonl"
    assert append_record(path, frozen, existing, runners, provenance) == "APPENDED"
    path.chmod(0o640)
    candidate, candidate_runners, candidate_provenance = distinct_record(
        frozen, runners, provenance, f"primary-precedence-{fault_point}"
    )
    before = output_snapshot(path)
    _, staged_path = residual_module._shadow_transaction_paths(path)
    unrelated_path = tmp_path / f"primary-precedence-{fault_point}-unrelated.txt"
    unrelated_path.write_bytes(b"unrelated")
    real_open = residual_module.os.open
    real_close = residual_module.os.close
    real_fdopen = residual_module.os.fdopen
    opened_staged_fds = []
    raw_close_attempts = []
    closefd_values = []
    wrapper_close_count = 0
    unrelated_fds = []
    fd_count_before = len(os.listdir("/proc/self/fd"))
    primary_message = f"PRIMARY_{fault_point.upper()}_FAILURE"

    class CloseFailingBorrower:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def write(self, value):
            return self.handle.write(value)

        def flush(self):
            return self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

        def close(self):
            nonlocal wrapper_close_count
            wrapper_close_count += 1
            self.handle.close()
            raise OSError("SECONDARY_WRAPPER_CLOSE_FAILURE")

    def tracked_open(open_path, *args, **kwargs):
        fd = real_open(open_path, *args, **kwargs)
        if Path(open_path) == staged_path:
            opened_staged_fds.append(fd)
        return fd

    def tracked_fdopen(fd, *args, **kwargs):
        closefd = kwargs["closefd"]
        closefd_values.append(closefd)
        if fault_point == "fdopen":
            try:
                return real_fdopen(
                    fd,
                    "w",
                    encoding="definitely-not-a-codec",
                    closefd=closefd,
                )
            except LookupError as exc:
                raise OSError(primary_message) from exc
        return CloseFailingBorrower(real_fdopen(fd, *args, **kwargs))

    def close_then_reuse_and_fail(fd):
        if opened_staged_fds and fd == opened_staged_fds[-1]:
            raw_close_attempts.append(fd)
            try:
                real_close(fd)
            except OSError:
                pass
            else:
                unrelated_fd = real_open(unrelated_path, os.O_RDONLY)
                unrelated_fds.append(unrelated_fd)
            raise OSError("SECONDARY_RAW_CLOSE_FAILURE")
        return real_close(fd)

    def primary_failure(*args, **kwargs):
        raise OSError(primary_message)

    monkeypatch.setattr(residual_module.os, "open", tracked_open)
    monkeypatch.setattr(residual_module.os, "fdopen", tracked_fdopen)
    monkeypatch.setattr(residual_module.os, "close", close_then_reuse_and_fail)
    if fault_point == "fchmod":
        monkeypatch.setattr(residual_module.os, "fchmod", primary_failure)
    elif fault_point != "fdopen":
        helper_by_fault = {
            "write": "_write_staged_shadow_bytes",
            "flush": "_flush_staged_shadow_file",
            "fsync": "_fsync_staged_shadow_file",
        }
        monkeypatch.setattr(
            residual_module, helper_by_fault[fault_point], primary_failure
        )

    try:
        for _ in range(8):
            with pytest.raises(ResidualContractError) as captured:
                append_record(
                    path,
                    frozen,
                    candidate,
                    candidate_runners,
                    candidate_provenance,
                )
            assert str(captured.value).startswith("shadow_output_write_failed:")
            assert str(captured.value.__cause__) == primary_message
            assert_output_snapshot(path, before)
            assert stat.S_IMODE(path.stat().st_mode) == 0o640
            assert not staged_path.exists()
            assert open_descriptors_for_path(staged_path) == []
            if unrelated_fds:
                unrelated_fd = unrelated_fds.pop()
                assert unrelated_fd == opened_staged_fds[-1]
                assert os.read(unrelated_fd, len(b"unrelated")) == b"unrelated"
                real_close(unrelated_fd)
        assert raw_close_attempts == opened_staged_fds
        expected_wrapper_closes = 0 if fault_point in {"fchmod", "fdopen"} else 8
        assert wrapper_close_count == expected_wrapper_closes
        expected_constructions = 0 if fault_point == "fchmod" else 8
        assert closefd_values == [False] * expected_constructions
        assert len(os.listdir("/proc/self/fd")) == fd_count_before
    finally:
        for unrelated_fd in unrelated_fds:
            try:
                real_close(unrelated_fd)
            except OSError:
                pass

    monkeypatch.undo()
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    assert output_snapshot(path)["bytes"] == before["bytes"] + canonical_bytes(
        candidate
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "EXACT_REPLAY"
    )


@pytest.mark.parametrize("target_exists", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_raw_close_only_failure_is_classified_once_and_retry_is_exact(
    tmp_path, monkeypatch, target_exists, cleanup_fails
):
    frozen, runners, provenance = load_fixture()
    original = score_race(frozen, runners, provenance)
    path = tmp_path / f"raw-close-only-{target_exists}-{cleanup_fails}.jsonl"
    old_umask = os.umask(0o027)
    try:
        if target_exists:
            assert (
                append_record(path, frozen, original, runners, provenance) == "APPENDED"
            )
            candidate, candidate_runners, candidate_provenance = distinct_record(
                frozen, runners, provenance, "raw-close-only"
            )
        else:
            candidate, candidate_runners, candidate_provenance = (
                original,
                runners,
                provenance,
            )
        before = output_snapshot(path)
        _, staged_path = residual_module._shadow_transaction_paths(path)
        replacement = before["bytes"] + canonical_bytes(candidate)
        unrelated_path = tmp_path / "raw-close-only-unrelated.txt"
        unrelated_path.write_bytes(b"unrelated")
        real_open = residual_module.os.open
        real_close = residual_module.os.close
        real_fdopen = residual_module.os.fdopen
        real_remove = residual_module._remove_staged_shadow_file
        opened_staged_fds = []
        raw_close_attempts = []
        closefd_values = []
        unrelated_fds = []
        cleanup_calls = 0
        fd_count_before = len(os.listdir("/proc/self/fd"))

        def tracked_open(open_path, *args, **kwargs):
            fd = real_open(open_path, *args, **kwargs)
            if Path(open_path) == staged_path:
                opened_staged_fds.append(fd)
            return fd

        def tracked_fdopen(fd, *args, **kwargs):
            closefd_values.append(kwargs["closefd"])
            return real_fdopen(fd, *args, **kwargs)

        def close_then_reuse_and_fail(fd):
            if opened_staged_fds and fd == opened_staged_fds[-1]:
                raw_close_attempts.append(fd)
                real_close(fd)
                unrelated_fd = real_open(unrelated_path, os.O_RDONLY)
                unrelated_fds.append(unrelated_fd)
                raise OSError("ONLY_RAW_CLOSE_FAILURE")
            return real_close(fd)

        def maybe_retain_stage(staged):
            nonlocal cleanup_calls
            cleanup_calls += 1
            if cleanup_fails and cleanup_calls % 2 == 0:
                raise OSError("FINAL_STAGE_CLEANUP_FAILURE")
            real_remove(staged)

        monkeypatch.setattr(residual_module.os, "open", tracked_open)
        monkeypatch.setattr(residual_module.os, "fdopen", tracked_fdopen)
        monkeypatch.setattr(residual_module.os, "close", close_then_reuse_and_fail)
        monkeypatch.setattr(
            residual_module, "_remove_staged_shadow_file", maybe_retain_stage
        )
        try:
            for _ in range(8):
                with pytest.raises(ResidualContractError) as captured:
                    append_record(
                        path,
                        frozen,
                        candidate,
                        candidate_runners,
                        candidate_provenance,
                    )
                assert str(captured.value).startswith("shadow_output_write_failed:")
                assert str(captured.value.__cause__) == "ONLY_RAW_CLOSE_FAILURE"
                assert raw_close_attempts[-1] == opened_staged_fds[-1]
                assert len(raw_close_attempts) == len(opened_staged_fds)
                assert unrelated_fds[-1] == opened_staged_fds[-1]
                assert os.read(unrelated_fds[-1], len(b"unrelated")) == b"unrelated"
                real_close(unrelated_fds.pop())
                assert_output_snapshot(path, before)
                if cleanup_fails:
                    assert staged_path.read_bytes() == replacement
                    assert stat.S_IMODE(staged_path.stat().st_mode) == 0o640
                else:
                    assert not staged_path.exists()
                assert open_descriptors_for_path(staged_path) == []
            assert closefd_values == [False] * 8
            assert raw_close_attempts == opened_staged_fds
            assert len(os.listdir("/proc/self/fd")) == fd_count_before
        finally:
            for unrelated_fd in unrelated_fds:
                try:
                    real_close(unrelated_fd)
                except OSError:
                    pass

        monkeypatch.undo()
        assert (
            append_record(
                path,
                frozen,
                candidate,
                candidate_runners,
                candidate_provenance,
            )
            == "APPENDED"
        )
        assert not staged_path.exists()
        after = output_snapshot(path)
        assert after["bytes"] == replacement
        assert after["rows"] == before["rows"] + 1
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
        assert (
            append_record(
                path,
                frozen,
                candidate,
                candidate_runners,
                candidate_provenance,
            )
            == "EXACT_REPLAY"
        )
    finally:
        os.umask(old_umask)


def test_staged_wrapper_and_raw_fd_close_before_publication(tmp_path, monkeypatch):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "close-before-publication.jsonl"
    _, staged_path = residual_module._shadow_transaction_paths(path)
    real_open = residual_module.os.open
    real_close = residual_module.os.close
    real_fdopen = residual_module.os.fdopen
    real_publish = residual_module._publish_staged_shadow_file
    staged_fd = None
    raw_closed = False
    closefd_values = []
    events = []

    class TrackedBorrower:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

        def write(self, value):
            return self.handle.write(value)

        def flush(self):
            return self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

        def close(self):
            events.append("wrapper_close")
            self.handle.close()

    def tracked_open(open_path, *args, **kwargs):
        nonlocal staged_fd
        fd = real_open(open_path, *args, **kwargs)
        if Path(open_path) == staged_path:
            staged_fd = fd
        return fd

    def tracked_fdopen(fd, *args, **kwargs):
        closefd_values.append(kwargs["closefd"])
        return TrackedBorrower(real_fdopen(fd, *args, **kwargs))

    def tracked_close(fd):
        nonlocal raw_closed
        if fd == staged_fd and not raw_closed:
            events.append("raw_close")
            raw_closed = True
        return real_close(fd)

    def checked_publish(staged, output):
        assert events == ["wrapper_close", "raw_close"]
        assert staged_fd is not None
        with pytest.raises(OSError, match="Bad file descriptor"):
            os.fstat(staged_fd)
        events.append("publish")
        real_publish(staged, output)

    monkeypatch.setattr(residual_module.os, "open", tracked_open)
    monkeypatch.setattr(residual_module.os, "fdopen", tracked_fdopen)
    monkeypatch.setattr(residual_module.os, "close", tracked_close)
    monkeypatch.setattr(residual_module, "_publish_staged_shadow_file", checked_publish)

    assert append_record(path, frozen, record, runners, provenance) == "APPENDED"
    assert closefd_values == [False]
    assert events == ["wrapper_close", "raw_close", "publish"]
    assert path.read_bytes() == canonical_bytes(record)
    assert not staged_path.exists()
    assert append_record(path, frozen, record, runners, provenance) == "EXACT_REPLAY"


def test_transferred_staged_fd_is_not_closed_again_after_managed_close(
    tmp_path, monkeypatch
):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "transferred-fd.jsonl"
    unrelated_path = tmp_path / "unrelated.txt"
    unrelated_path.write_bytes(b"unrelated")
    _, staged_path = residual_module._shadow_transaction_paths(path)
    real_fdopen = residual_module.os.fdopen
    real_remove = residual_module._remove_staged_shadow_file
    transferred_fd = None
    unrelated_fd = None
    remove_calls = 0

    def record_transfer(fd, *args, **kwargs):
        nonlocal transferred_fd
        transferred_fd = fd
        return real_fdopen(fd, *args, **kwargs)

    def injected_write_failure(*args, **kwargs):
        raise OSError("injected:write-after-transfer")

    def remove_then_reuse_fd(staged):
        nonlocal remove_calls, unrelated_fd
        remove_calls += 1
        real_remove(staged)
        if remove_calls == 2:
            unrelated_fd = os.open(unrelated_path, os.O_RDONLY)

    monkeypatch.setattr(residual_module.os, "fdopen", record_transfer)
    monkeypatch.setattr(
        residual_module, "_write_staged_shadow_bytes", injected_write_failure
    )
    monkeypatch.setattr(
        residual_module, "_remove_staged_shadow_file", remove_then_reuse_fd
    )
    try:
        with pytest.raises(ResidualContractError, match="^shadow_output_write_failed:"):
            append_record(path, frozen, record, runners, provenance)
        assert transferred_fd is not None
        assert unrelated_fd == transferred_fd
        assert os.read(unrelated_fd, len(b"unrelated")) == b"unrelated"
        assert not path.exists()
        assert not staged_path.exists()
    finally:
        if unrelated_fd is not None:
            os.close(unrelated_fd)


@pytest.mark.parametrize("target_exists", [False, True])
def test_directory_fsync_failure_is_postcommit_and_retry_is_exact(
    tmp_path, monkeypatch, target_exists
):
    frozen, runners, provenance = load_fixture()
    original = score_race(frozen, runners, provenance)
    path = tmp_path / f"directory-fsync-{target_exists}.jsonl"
    if target_exists:
        assert append_record(path, frozen, original, runners, provenance) == "APPENDED"
        candidate, candidate_runners, candidate_provenance = distinct_record(
            frozen, runners, provenance, "directory-fsync"
        )
    else:
        candidate, candidate_runners, candidate_provenance = (
            original,
            runners,
            provenance,
        )
    before = output_snapshot(path)

    def injected_directory_fsync_failure(*args, **kwargs):
        raise OSError("injected:directory-fsync")

    monkeypatch.setattr(
        residual_module,
        "_fsync_parent_directory",
        injected_directory_fsync_failure,
    )
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    after = output_snapshot(path)
    assert after["exists"] is True
    assert after["bytes"] == before["bytes"] + canonical_bytes(candidate)
    assert after["sha256"] != before["sha256"]
    assert after["rows"] == before["rows"] + 1
    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "EXACT_REPLAY"
    )


def test_publish_then_raise_is_recognized_as_committed(tmp_path, monkeypatch):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "publish-then-raise.jsonl"
    real_publish = residual_module._publish_staged_shadow_file

    def publish_then_raise(staged, output):
        real_publish(staged, output)
        raise OSError("injected:post-publication")

    monkeypatch.setattr(
        residual_module, "_publish_staged_shadow_file", publish_then_raise
    )
    assert append_record(path, frozen, record, runners, provenance) == "APPENDED"
    assert path.read_bytes() == canonical_bytes(record)
    assert append_record(path, frozen, record, runners, provenance) == "EXACT_REPLAY"


def test_publication_exception_with_unexpected_valid_target_reports_unknown(
    tmp_path, monkeypatch
):
    frozen, runners, provenance = load_fixture()
    intended = score_race(frozen, runners, provenance)
    other, _, _ = distinct_record(frozen, runners, provenance, "uncertain-other")
    path = tmp_path / "publication-unknown.jsonl"

    def publish_unexpected_target_then_raise(staged, output):
        output.write_bytes(canonical_bytes(other))
        raise OSError("injected:uncertain-publication")

    monkeypatch.setattr(
        residual_module,
        "_publish_staged_shadow_file",
        publish_unexpected_target_then_raise,
    )
    assert (
        append_record(path, frozen, intended, runners, provenance)
        == "COMMIT_STATE_UNKNOWN"
    )
    assert path.read_bytes() == canonical_bytes(other)

    monkeypatch.undo()
    assert append_record(path, frozen, intended, runners, provenance) == "APPENDED"
    assert append_record(path, frozen, intended, runners, provenance) == "EXACT_REPLAY"


def test_existing_target_permissions_survive_atomic_publication(tmp_path):
    frozen, runners, provenance = load_fixture()
    existing = score_race(frozen, runners, provenance)
    path = tmp_path / "existing-permissions.jsonl"
    assert append_record(path, frozen, existing, runners, provenance) == "APPENDED"
    path.chmod(0o640)
    candidate, candidate_runners, candidate_provenance = distinct_record(
        frozen, runners, provenance, "existing-mode"
    )

    assert (
        append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )
        == "APPENDED"
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_absent_target_uses_normal_create_permissions(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "absent-permissions.jsonl"
    old_umask = os.umask(0o027)
    try:
        assert append_record(path, frozen, record, runners, provenance) == "APPENDED"
    finally:
        os.umask(old_umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_concurrent_retries_share_stable_sidecar_and_publish_one_row(tmp_path):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "concurrent.jsonl"
    worker_count = 8
    barrier = Barrier(worker_count)

    def append_concurrently():
        barrier.wait()
        return append_record(path, frozen, record, runners, provenance)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(
            executor.map(lambda _: append_concurrently(), range(worker_count))
        )

    assert results.count("APPENDED") == 1
    assert results.count("EXACT_REPLAY") == worker_count - 1
    assert path.read_bytes() == canonical_bytes(record)
    lock_path, staged_path = residual_module._shadow_transaction_paths(path)
    assert lock_path.is_file()
    lock_inode = lock_path.stat().st_ino
    assert not staged_path.exists()
    assert append_record(path, frozen, record, runners, provenance) == "EXACT_REPLAY"
    assert lock_path.stat().st_ino == lock_inode


def test_concurrent_distinct_appends_do_not_lose_a_replacement(tmp_path):
    frozen, runners, provenance = load_fixture()
    first = score_race(frozen, runners, provenance)
    second, second_runners, second_provenance = distinct_record(
        frozen, runners, provenance, "concurrent-distinct"
    )
    path = tmp_path / "concurrent-distinct.jsonl"
    barrier = Barrier(2)

    def append_after_barrier(candidate, candidate_runners, candidate_provenance):
        barrier.wait()
        return append_record(
            path,
            frozen,
            candidate,
            candidate_runners,
            candidate_provenance,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(append_after_barrier, first, runners, provenance),
            executor.submit(
                append_after_barrier,
                second,
                second_runners,
                second_provenance,
            ),
        ]
        results = [future.result() for future in futures]

    assert results == ["APPENDED", "APPENDED"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {row["record_key"] for row in rows} == {
        first["record_key"],
        second["record_key"],
    }
    assert len(rows) == 2
    _, staged_path = residual_module._shadow_transaction_paths(path)
    assert not staged_path.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_sidecar_lock_symlink_is_rejected_without_transaction_mutation(
    tmp_path, target_exists
):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "symlink-lock.jsonl"
    lock_path, staged_path = residual_module._shadow_transaction_paths(path)
    if target_exists:
        path.write_bytes(canonical_bytes(record))
    before = output_snapshot(path)
    staged_path.write_bytes(b"retained-stage")
    lock_path.symlink_to(staged_path.name)

    with pytest.raises(ResidualContractError, match="shadow_output_write_failed"):
        append_record(path, frozen, record, runners, provenance)

    assert lock_path.is_symlink()
    assert os.readlink(lock_path) == staged_path.name
    assert_output_snapshot(path, before)
    assert staged_path.read_bytes() == b"retained-stage"


@pytest.mark.parametrize("target_exists", [False, True])
def test_nonregular_sidecar_lock_is_rejected_without_transaction_mutation(
    tmp_path, target_exists
):
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)
    path = tmp_path / "fifo-lock.jsonl"
    lock_path, staged_path = residual_module._shadow_transaction_paths(path)
    if target_exists:
        path.write_bytes(canonical_bytes(record))
    before = output_snapshot(path)
    staged_path.write_bytes(b"retained-stage")
    os.mkfifo(lock_path)
    fd_count_before = len(os.listdir("/proc/self/fd"))

    with pytest.raises(ResidualContractError, match="shadow_output_write_failed"):
        append_record(path, frozen, record, runners, provenance)

    assert stat.S_ISFIFO(lock_path.stat().st_mode)
    assert_output_snapshot(path, before)
    assert staged_path.read_bytes() == b"retained-stage"
    assert len(os.listdir("/proc/self/fd")) == fd_count_before


def test_v3_record_stores_complete_inputs_and_binds_full_content():
    frozen, runners, provenance = load_fixture()
    record = score_race(frozen, runners, provenance)

    assert record["schema_version"] == "market_form_residual_shadow_record_v3"
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


@pytest.mark.parametrize("history_shape", ["v1", "v2", "mixed", "insufficient_v3"])
def test_legacy_mixed_and_insufficient_history_require_migration(
    tmp_path, history_shape
):
    frozen, runners, provenance = load_fixture()
    current = score_race(frozen, runners, provenance)
    legacy = copy.deepcopy(current)
    legacy["schema_version"] = "market_form_residual_shadow_record_v1"
    reseal_record(legacy)
    prior = copy.deepcopy(current)
    prior["schema_version"] = "market_form_residual_shadow_record_v2"
    reseal_record(prior)
    insufficient = copy.deepcopy(current)
    insufficient.pop("inputs")
    reseal_record(insufficient)
    rows = {
        "v1": [legacy],
        "v2": [prior],
        "mixed": [current, prior],
        "insufficient_v3": [insufficient],
    }[history_shape]
    path = tmp_path / f"{history_shape}.jsonl"
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))
    before = path.read_bytes()
    before_stat = path.stat()

    with pytest.raises(ResidualContractError, match="^history_migration_required$"):
        append_record(path, frozen, current, runners, provenance)

    after_stat = path.stat()
    assert path.read_bytes() == before
    assert after_stat.st_ino == before_stat.st_ino
    assert stat.S_IMODE(after_stat.st_mode) == stat.S_IMODE(before_stat.st_mode)
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


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
