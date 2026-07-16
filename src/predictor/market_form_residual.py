"""Fail-closed loader and shadow scorer for the frozen market-form residual.

This module has no collector, database, registry, service, or activation hook.
It loads one hash-bound base model, derives the frozen full and half variants,
and can append deterministic outcome-free shadow records to an explicit JSONL
path supplied by a future separately authorized caller.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_SCHEMA = "market_form_residual_frozen_model_v1"
MANIFEST_SCHEMA = "market_form_residual_frozen_manifest_v1"
SHADOW_RECORD_SCHEMA = "market_form_residual_shadow_record_v1"
DEFAULT_ARTIFACT_DIR = Path("artifacts/frozen_models/market_form_residual_v1")
FEATURES = (
    "prior_start_count",
    "days_since_last_start",
    "recent_finish_mean_3",
    "recent_finish_best_5",
    "recent_win_rate_5",
    "recent_place_rate_5",
    "recent_avg_margin_5",
    "career_win_rate",
    "career_place_rate",
    "career_avg_finish",
    "starts_same_venue",
    "win_rate_same_venue",
    "starts_same_distance",
    "win_rate_same_distance",
    "same_grade_start_count",
    "same_grade_win_rate",
)
EXPANDED_FEATURES = FEATURES + tuple(f"{name}__missing" for name in FEATURES)
RUNNER_FIELDS = {
    "race_id",
    "runner_id",
    "box_number",
    "dog_name",
    "strict_win_odds",
    "features",
    "feature_source_sha256",
    "odds_source_sha256",
    "feature_freeze_timestamp",
    "odds_capture_timestamp",
}
OUTCOME_FIELDS = {
    "actual_win",
    "finish_position",
    "official_result",
    "outcome",
    "placing",
    "result",
    "winner",
    "winner_name",
    "winner_odds",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FROZEN_ALGORITHM_CONTRACT = {
    "initialization": "all_zero_coefficients",
    "market_offset": "fixed_log_normalized_inverse_decimal_win_odds",
    "market_offset_refit": False,
    "normalization": "softmax(log(market_probability)+strength*capped_residual)",
    "optimizer": "scipy.optimize.minimize:L-BFGS-B",
    "optimizer_options": {
        "ftol": 1e-12,
        "gtol": 1e-8,
        "maxiter": 500,
        "maxls": 50,
    },
    "random_seed": None,
    "randomness": "none",
    "residual_cap": 0.35,
    "ridge_l2": 1.0,
    "strengths": {"full": 1.0, "half": 0.5},
    "within_race_centering": True,
}
FROZEN_DERIVATION_CONTRACT = {
    "full_strength": 1.0,
    "half_strength": 0.5,
    "shared_base_model_count": 1,
    "variants_are_not_separate_models": True,
}


class ResidualContractError(RuntimeError):
    """Raised when artifact, feature, provenance, or output validation fails."""


@dataclass(frozen=True)
class FrozenResidualModel:
    model: Mapping[str, Any]
    manifest: Mapping[str, Any]
    model_sha256: str
    manifest_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
    except (TypeError, ValueError) as exc:
        raise ResidualContractError("record_not_canonical_json") from exc
    return (text + "\n").encode("utf-8")


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidualContractError(f"artifact_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ResidualContractError(f"artifact_not_object:{path}")
    if raw != _canonical_bytes(value):
        raise ResidualContractError(f"artifact_not_canonical_json:{path}")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ResidualContractError(f"invalid_number:{field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ResidualContractError(f"invalid_number:{field}") from exc
    if not math.isfinite(result):
        raise ResidualContractError(f"non_finite_number:{field}")
    return result


def _numeric_vector(value: Any, length: int, field: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise ResidualContractError(f"invalid_vector_shape:{field}")
    return np.asarray(
        [_finite_float(item, f"{field}[{index}]") for index, item in enumerate(value)],
        dtype=float,
    )


def _require_sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if not SHA256_RE.fullmatch(text):
        raise ResidualContractError(f"invalid_sha256:{field}")
    return text


def _parse_timestamp(value: Any, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ResidualContractError(f"invalid_timestamp:{field}") from exc
    if result.utcoffset() is None:
        raise ResidualContractError(f"timezone_missing:{field}")
    return result


def _runner_set_sha256(runner_ids: Sequence[str]) -> str:
    content = "\n".join(sorted(runner_ids)) + "\n"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in OUTCOME_FIELDS or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_outcome(item) for item in value)
    return False


def load_frozen_model(
    model_path: Path | str = DEFAULT_ARTIFACT_DIR / "model.json",
    manifest_path: Path | str = DEFAULT_ARTIFACT_DIR / "manifest.json",
) -> FrozenResidualModel:
    """Load and validate the one frozen base model and its manifest."""

    model_file = Path(model_path)
    manifest_file = Path(manifest_path)
    model = _read_json(model_file)
    manifest = _read_json(manifest_file)
    model_sha256 = _sha256_file(model_file)
    manifest_sha256 = _sha256_file(manifest_file)

    if model.get("schema_version") != MODEL_SCHEMA:
        raise ResidualContractError("model_schema_mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ResidualContractError("manifest_schema_mismatch")
    if manifest.get("status") != "FROZEN_MODEL_READY_AWAITING_ACTIVATION":
        raise ResidualContractError("manifest_status_not_frozen")
    if _require_sha256(manifest.get("model_sha256"), "manifest.model_sha256") != model_sha256:
        raise ResidualContractError("model_sha256_mismatch")
    if manifest.get("model_schema_version") != MODEL_SCHEMA:
        raise ResidualContractError("manifest_model_schema_mismatch")
    fit = model.get("fit")
    if not isinstance(fit, dict):
        raise ResidualContractError("fit_contract_missing")
    fit_population_sha = _require_sha256(
        manifest.get("fit_population_sha256"), "manifest.fit_population_sha256"
    )
    if fit.get("population_sha256") != fit_population_sha:
        raise ResidualContractError("fit_population_sha256_mismatch")
    if fit.get("race_count") != 678 or fit.get("runner_count") != 4752:
        raise ResidualContractError("fit_population_count_mismatch")
    if fit.get("race_date_max_inclusive") != "2026-07-09":
        raise ResidualContractError("fit_population_cutoff_mismatch")

    algorithm = model.get("algorithm")
    preprocessor = model.get("preprocessor")
    feature_contract = model.get("feature_contract")
    activation = model.get("activation")
    if not all(isinstance(value, dict) for value in (algorithm, preprocessor, feature_contract, activation)):
        raise ResidualContractError("model_contract_section_missing")
    if model.get("model_family") != "race_conditional_logit_with_fixed_market_offset":
        raise ResidualContractError("model_family_contract_mismatch")
    if _canonical_bytes(algorithm) != _canonical_bytes(FROZEN_ALGORITHM_CONTRACT):
        raise ResidualContractError("algorithm_contract_mismatch")
    if any(activation.get(field) is not False for field in (
        "activated", "production_pointer_changed", "runtime_changed", "cohort_cutoff_assigned"
    )):
        raise ResidualContractError("artifact_is_activated")
    if tuple(feature_contract.get("feature_order") or ()) != FEATURES:
        raise ResidualContractError("feature_order_mismatch")
    if tuple(feature_contract.get("expanded_feature_order") or ()) != EXPANDED_FEATURES:
        raise ResidualContractError("expanded_feature_order_mismatch")
    if tuple(preprocessor.get("features") or ()) != FEATURES:
        raise ResidualContractError("preprocessor_feature_order_mismatch")
    if tuple(preprocessor.get("expanded_features") or ()) != EXPANDED_FEATURES:
        raise ResidualContractError("preprocessor_expanded_order_mismatch")

    _numeric_vector(model.get("beta"), len(EXPANDED_FEATURES), "beta")
    _numeric_vector(preprocessor.get("medians"), len(FEATURES), "preprocessor.medians")
    _numeric_vector(preprocessor.get("means"), len(EXPANDED_FEATURES), "preprocessor.means")
    scales = _numeric_vector(preprocessor.get("scales"), len(EXPANDED_FEATURES), "preprocessor.scales")
    if np.any(scales <= 0.0):
        raise ResidualContractError("preprocessor_scale_not_positive")
    candidate_hashes = manifest.get("candidate_hashes")
    if not isinstance(candidate_hashes, dict) or not candidate_hashes:
        raise ResidualContractError("candidate_hashes_missing")
    for name, value in candidate_hashes.items():
        _require_sha256(value, f"candidate_hashes.{name}")
    derivation = manifest.get("derivation_contract")
    if (
        not isinstance(derivation, dict)
        or _canonical_bytes(derivation) != _canonical_bytes(FROZEN_DERIVATION_CONTRACT)
    ):
        raise ResidualContractError("shared_base_model_contract_mismatch")
    return FrozenResidualModel(model, manifest, model_sha256, manifest_sha256)


def _shadow_record_identity(record: Mapping[str, Any]) -> dict[str, str]:
    race_id = record.get("race_id")
    if not isinstance(race_id, str) or not race_id:
        raise ResidualContractError("shadow_record_race_id_invalid")
    return {
        "race_id": race_id,
        "runner_set_sha256": _require_sha256(
            record.get("runner_set_sha256"), "runner_set_sha256"
        ),
        "model_sha256": _require_sha256(record.get("model_sha256"), "model_sha256"),
        "manifest_sha256": _require_sha256(
            record.get("manifest_sha256"), "manifest_sha256"
        ),
    }


def _shadow_record_key(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(_shadow_record_identity(record))).hexdigest()


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    total = float(np.sum(exponentials))
    if not math.isfinite(total) or total <= 0.0:
        raise ResidualContractError("softmax_invalid")
    probabilities = exponentials / total
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ResidualContractError("probability_invalid")
    if not math.isclose(float(np.sum(probabilities)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ResidualContractError("probability_not_normalized")
    return probabilities


def score_race(
    frozen: FrozenResidualModel,
    runners: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one complete race and return an outcome-free shadow record."""

    if not isinstance(provenance, Mapping) or _contains_outcome(provenance):
        raise ResidualContractError("provenance_invalid_or_contains_outcome")
    if not isinstance(runners, Sequence) or isinstance(runners, (str, bytes)) or len(runners) < 2:
        raise ResidualContractError("race_runner_count_invalid")
    race_id = str(provenance.get("race_id") or "")
    if not race_id:
        raise ResidualContractError("race_id_missing")
    expected_runner_ids = provenance.get("expected_runner_ids")
    if not isinstance(expected_runner_ids, list) or not expected_runner_ids:
        raise ResidualContractError("expected_runner_ids_missing")
    expected_runner_ids = [str(value) for value in expected_runner_ids]
    if len(expected_runner_ids) != len(set(expected_runner_ids)):
        raise ResidualContractError("expected_runner_ids_duplicate")
    expected_runner_ids = sorted(expected_runner_ids)
    expected_hash = _runner_set_sha256(expected_runner_ids)
    if _require_sha256(provenance.get("runner_set_sha256"), "runner_set_sha256") != expected_hash:
        raise ResidualContractError("declared_runner_set_hash_mismatch")

    jump = _parse_timestamp(provenance.get("jump_timestamp"), "jump_timestamp")
    score_time = _parse_timestamp(provenance.get("score_timestamp"), "score_timestamp")
    if not score_time < jump:
        raise ResidualContractError("score_timestamp_not_prejump")

    seen_ids: set[str] = set()
    seen_boxes: set[int] = set()
    raw_features: list[list[float]] = []
    odds: list[float] = []
    normalized_runners: list[dict[str, Any]] = []
    ordered_runners = sorted(
        runners,
        key=lambda runner: (
            str(runner.get("runner_id") or "") if isinstance(runner, Mapping) else ""
        ),
    )
    for index, runner in enumerate(ordered_runners):
        if not isinstance(runner, Mapping) or _contains_outcome(runner):
            raise ResidualContractError(f"runner_invalid_or_contains_outcome:{index}")
        extra_fields = set(runner) - RUNNER_FIELDS
        missing_fields = RUNNER_FIELDS - set(runner)
        if extra_fields or missing_fields:
            raise ResidualContractError(
                f"runner_field_contract_mismatch:{index}:missing={sorted(missing_fields)}:extra={sorted(extra_fields)}"
            )
        if str(runner["race_id"]) != race_id:
            raise ResidualContractError(f"runner_race_id_mismatch:{index}")
        runner_id = str(runner["runner_id"] or "")
        if not runner_id or runner_id in seen_ids:
            raise ResidualContractError(f"runner_id_invalid_or_duplicate:{index}")
        seen_ids.add(runner_id)
        box = int(_finite_float(runner["box_number"], f"box_number:{runner_id}"))
        if box <= 0 or box in seen_boxes or float(box) != float(runner["box_number"]):
            raise ResidualContractError(f"box_number_invalid_or_duplicate:{runner_id}")
        seen_boxes.add(box)
        dog_name = str(runner["dog_name"] or "").strip()
        dog_token = re.sub(r"[^A-Z0-9]", "", dog_name.upper())
        if not dog_token or runner_id != f"{race_id}|box:{box}|dog:{dog_token}":
            raise ResidualContractError(f"runner_identity_mismatch:{runner_id}")
        decimal_odds = _finite_float(runner["strict_win_odds"], f"strict_win_odds:{runner_id}")
        if decimal_odds <= 1.0:
            raise ResidualContractError(f"strict_win_odds_invalid:{runner_id}")
        feature_hash = _require_sha256(runner["feature_source_sha256"], f"feature_source_sha256:{runner_id}")
        odds_hash = _require_sha256(runner["odds_source_sha256"], f"odds_source_sha256:{runner_id}")
        feature_time = _parse_timestamp(runner["feature_freeze_timestamp"], f"feature_freeze_timestamp:{runner_id}")
        odds_time = _parse_timestamp(runner["odds_capture_timestamp"], f"odds_capture_timestamp:{runner_id}")
        if not feature_time < jump or not odds_time < jump:
            raise ResidualContractError(f"source_timestamp_not_prejump:{runner_id}")
        if score_time < feature_time or score_time < odds_time:
            raise ResidualContractError(f"score_timestamp_before_source:{runner_id}")
        features = runner["features"]
        if not isinstance(features, Mapping):
            raise ResidualContractError(f"features_not_object:{runner_id}")
        extra_features = set(features) - set(FEATURES)
        if extra_features:
            raise ResidualContractError(f"unexpected_features:{runner_id}:{sorted(extra_features)}")
        values: list[float] = []
        for feature in FEATURES:
            value = features.get(feature)
            values.append(float("nan") if value is None else _finite_float(value, f"{runner_id}.{feature}"))
        raw_features.append(values)
        odds.append(decimal_odds)
        normalized_runners.append(
            {
                "race_id": race_id,
                "runner_id": runner_id,
                "box_number": box,
                "dog_name": dog_name,
                "strict_win_odds": decimal_odds,
                "feature_source_sha256": feature_hash,
                "odds_source_sha256": odds_hash,
                "feature_freeze_timestamp": str(runner["feature_freeze_timestamp"]),
                "odds_capture_timestamp": str(runner["odds_capture_timestamp"]),
            }
        )

    actual_runner_ids = sorted(seen_ids)
    if actual_runner_ids != expected_runner_ids:
        raise ResidualContractError("race_incomplete_or_runner_set_mismatch")
    if _runner_set_sha256(actual_runner_ids) != expected_hash:
        raise ResidualContractError("actual_runner_set_hash_mismatch")

    preprocessor = frozen.model["preprocessor"]
    raw = np.asarray(raw_features, dtype=float)
    medians = _numeric_vector(preprocessor["medians"], len(FEATURES), "preprocessor.medians")
    missing = ~np.isfinite(raw)
    imputed = np.where(missing, medians, raw)
    expanded = np.concatenate([imputed, missing.astype(float)], axis=1)
    means = _numeric_vector(preprocessor["means"], len(EXPANDED_FEATURES), "preprocessor.means")
    scales = _numeric_vector(preprocessor["scales"], len(EXPANDED_FEATURES), "preprocessor.scales")
    transformed = (expanded - means) / scales
    transformed -= np.mean(transformed, axis=0)
    beta = _numeric_vector(frozen.model["beta"], len(EXPANDED_FEATURES), "beta")
    cap = _finite_float(frozen.model["algorithm"]["residual_cap"], "residual_cap")
    adjustment = cap * np.tanh((transformed @ beta) / cap)
    implied = 1.0 / np.asarray(odds, dtype=float)
    market = implied / np.sum(implied)
    full = _softmax(np.log(market) + adjustment)
    half = _softmax(np.log(market) + 0.5 * adjustment)

    predictions = []
    for index, runner in enumerate(normalized_runners):
        predictions.append(
            {
                **runner,
                "market_probability": float(market[index]),
                "residual_adjustment": float(adjustment[index]),
                "full_probability": float(full[index]),
                "half_probability": float(half[index]),
            }
        )
    identity = {
        "race_id": race_id,
        "runner_set_sha256": expected_hash,
        "model_sha256": frozen.model_sha256,
        "manifest_sha256": frozen.manifest_sha256,
    }
    record_key = _shadow_record_key(identity)
    return {
        "schema_version": SHADOW_RECORD_SCHEMA,
        "record_key": record_key,
        **identity,
        "score_timestamp": str(provenance["score_timestamp"]),
        "jump_timestamp": str(provenance["jump_timestamp"]),
        "variants": {"full_strength": 1.0, "half_strength": 0.5},
        "predictions": predictions,
        "outcomes_present": False,
        "activation": False,
    }


def append_shadow_record(path: Path | str, record: Mapping[str, Any]) -> str:
    """Append one canonical record, idempotently, without overwriting history."""

    output = Path(path)
    if output.suffix != ".jsonl" or not output.parent.is_dir():
        raise ResidualContractError("shadow_output_path_invalid")
    if not isinstance(record, Mapping) or record.get("schema_version") != SHADOW_RECORD_SCHEMA:
        raise ResidualContractError("shadow_record_schema_mismatch")
    if _contains_outcome(record) or record.get("outcomes_present") is not False:
        raise ResidualContractError("shadow_record_contains_outcome")
    record_key = _require_sha256(record.get("record_key"), "record_key")
    if record_key != _shadow_record_key(record):
        raise ResidualContractError("shadow_record_key_mismatch")
    encoded = _canonical_bytes(record)
    try:
        with output.open("a+", encoding="utf-8", newline="\n") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ResidualContractError(f"existing_shadow_blank_line:{line_number}")
                try:
                    existing = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ResidualContractError(f"existing_shadow_invalid_json:{line_number}") from exc
                if (
                    not isinstance(existing, dict)
                    or existing.get("schema_version") != SHADOW_RECORD_SCHEMA
                    or _contains_outcome(existing)
                    or existing.get("outcomes_present") is not False
                ):
                    raise ResidualContractError(f"existing_shadow_invalid_record:{line_number}")
                try:
                    existing_key = _require_sha256(
                        existing.get("record_key"), "existing.record_key"
                    )
                    if existing_key != _shadow_record_key(existing):
                        raise ResidualContractError("existing_shadow_record_key_mismatch")
                except ResidualContractError as exc:
                    raise ResidualContractError(
                        f"existing_shadow_invalid_record:{line_number}"
                    ) from exc
                if existing_key == record_key:
                    if _canonical_bytes(existing) == encoded:
                        return "EXACT_REPLAY"
                    raise ResidualContractError("conflicting_shadow_duplicate")
            handle.seek(0, os.SEEK_END)
            handle.write(encoded.decode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise ResidualContractError(f"shadow_output_write_failed:{output}") from exc
    return "APPENDED"
