"""Fail-closed loader and shadow scorer for the frozen market-form residual.

This module has no collector, database, registry, service, or activation hook.
It loads one hash-bound base model, derives the frozen full and half variants,
and can append deterministic outcome-free shadow records to an explicit JSONL
path supplied by a future separately authorized caller.

V2 record checksums detect accidental corruption and inconsistent record
construction. They are not authentication: a malicious actor with filesystem
access can replace a complete canonical row and recompute its checksum and
identifier. External signing and key management are deliberately out of scope.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


MODEL_SCHEMA = "market_form_residual_frozen_model_v1"
MANIFEST_SCHEMA = "market_form_residual_frozen_manifest_v1"
SHADOW_RECORD_SCHEMA = "market_form_residual_shadow_record_v2"
EFFECTIVE_STATE_SCHEMA = "market_form_residual_effective_state_v1"
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
RUNNER_FIELDS = frozenset(
    {
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
)
PROVENANCE_FIELDS = frozenset(
    {
        "expected_runner_ids",
        "jump_timestamp",
        "race_id",
        "runner_set_sha256",
        "score_timestamp",
    }
)
PREDICTION_FIELDS = frozenset(
    (RUNNER_FIELDS - {"features"})
    | {
        "market_probability",
        "residual_adjustment",
        "full_probability",
        "half_probability",
    }
)
SHADOW_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "record_key",
        "record_checksum_sha256",
        "race_id",
        "runner_set_sha256",
        "model_sha256",
        "manifest_sha256",
        "effective_state_sha256",
        "score_timestamp",
        "jump_timestamp",
        "variants",
        "inputs",
        "predictions",
        "outcomes_present",
        "activation",
    }
)
OUTCOME_FIELDS = frozenset(
    {
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
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FROZEN_ALGORITHM_CONTRACT = MappingProxyType(
    {
        "initialization": "all_zero_coefficients",
        "market_offset": "fixed_log_normalized_inverse_decimal_win_odds",
        "market_offset_refit": False,
        "normalization": "softmax(log(market_probability)+strength*capped_residual)",
        "optimizer": "scipy.optimize.minimize:L-BFGS-B",
        "optimizer_options": MappingProxyType(
            {
                "ftol": 1e-12,
                "gtol": 1e-8,
                "maxiter": 500,
                "maxls": 50,
            }
        ),
        "random_seed": None,
        "randomness": "none",
        "residual_cap": 0.35,
        "ridge_l2": 1.0,
        "strengths": MappingProxyType({"full": 1.0, "half": 0.5}),
        "within_race_centering": True,
    }
)
FROZEN_DERIVATION_CONTRACT = MappingProxyType(
    {
        "full_strength": 1.0,
        "half_strength": 0.5,
        "shared_base_model_count": 1,
        "variants_are_not_separate_models": True,
    }
)


class ResidualContractError(RuntimeError):
    """Raised when artifact, feature, provenance, or output validation fails."""


@dataclass(frozen=True, eq=False)
class FrozenResidualModel:
    model: Mapping[str, Any]
    manifest: Mapping[str, Any]
    model_sha256: str
    manifest_sha256: str
    effective_state_sha256: str
    beta: np.ndarray = field(repr=False)
    medians: np.ndarray = field(repr=False)
    means: np.ndarray = field(repr=False)
    scales: np.ndarray = field(repr=False)
    feature_order: tuple[str, ...]
    expanded_feature_order: tuple[str, ...]
    residual_cap: float
    full_strength: float
    half_strength: float
    market_offset: str
    normalization: str
    within_race_centering: bool


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"not_json_value:{type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            _json_value(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ResidualContractError("record_not_canonical_json") from exc
    return (text + "\n").encode("utf-8")


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _read_json(path: Path) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResidualContractError(f"artifact_unreadable:{path}") from exc
    if not isinstance(value, dict):
        raise ResidualContractError(f"artifact_not_object:{path}")
    if raw != _canonical_bytes(value):
        raise ResidualContractError(f"artifact_not_canonical_json:{path}")
    return value, hashlib.sha256(raw).hexdigest()


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


def _immutable_vector(value: Any, length: int, field: str) -> np.ndarray:
    vector = np.asarray(_numeric_vector(value, length, field), dtype=np.float64)
    immutable = np.frombuffer(vector.tobytes(order="C"), dtype=np.float64)
    if immutable.flags.writeable:
        raise ResidualContractError(f"immutable_vector_construction_failed:{field}")
    return immutable


def _score_state_payload(frozen: FrozenResidualModel) -> dict[str, Any]:
    return {
        "beta": frozen.beta,
        "medians": frozen.medians,
        "means": frozen.means,
        "scales": frozen.scales,
        "feature_order": frozen.feature_order,
        "expanded_feature_order": frozen.expanded_feature_order,
        "residual_cap": frozen.residual_cap,
        "full_strength": frozen.full_strength,
        "half_strength": frozen.half_strength,
        "market_offset": frozen.market_offset,
        "normalization": frozen.normalization,
        "within_race_centering": frozen.within_race_centering,
    }


def _artifact_score_state_payload(frozen: FrozenResidualModel) -> dict[str, Any]:
    preprocessor = frozen.model["preprocessor"]
    algorithm = frozen.model["algorithm"]
    feature_contract = frozen.model["feature_contract"]
    derivation = frozen.manifest["derivation_contract"]
    return {
        "beta": frozen.model["beta"],
        "medians": preprocessor["medians"],
        "means": preprocessor["means"],
        "scales": preprocessor["scales"],
        "feature_order": feature_contract["feature_order"],
        "expanded_feature_order": feature_contract["expanded_feature_order"],
        "residual_cap": algorithm["residual_cap"],
        "full_strength": derivation["full_strength"],
        "half_strength": derivation["half_strength"],
        "market_offset": algorithm["market_offset"],
        "normalization": algorithm["normalization"],
        "within_race_centering": algorithm["within_race_centering"],
    }


def _effective_state_payload(frozen: FrozenResidualModel) -> dict[str, Any]:
    return {
        "schema_version": EFFECTIVE_STATE_SCHEMA,
        "artifact_state": {
            "model": frozen.model,
            "manifest": frozen.manifest,
        },
        "score_state": _score_state_payload(frozen),
    }


def _effective_state_sha256(frozen: FrozenResidualModel) -> str:
    return hashlib.sha256(
        _canonical_bytes(_effective_state_payload(frozen))
    ).hexdigest()


def _verify_effective_state(frozen: FrozenResidualModel) -> str:
    if not isinstance(frozen, FrozenResidualModel):
        raise ResidualContractError("frozen_model_type_invalid")
    model_state_sha256 = hashlib.sha256(_canonical_bytes(frozen.model)).hexdigest()
    if model_state_sha256 != frozen.model_sha256:
        raise ResidualContractError("model_state_sha256_mismatch")
    manifest_state_sha256 = hashlib.sha256(
        _canonical_bytes(frozen.manifest)
    ).hexdigest()
    if manifest_state_sha256 != frozen.manifest_sha256:
        raise ResidualContractError("manifest_state_sha256_mismatch")
    if frozen.manifest.get("model_sha256") != model_state_sha256:
        raise ResidualContractError("manifest_model_state_sha256_mismatch")
    effective_state_sha256 = _effective_state_sha256(frozen)
    if effective_state_sha256 != frozen.effective_state_sha256:
        raise ResidualContractError("effective_state_sha256_mismatch")
    if _canonical_bytes(_score_state_payload(frozen)) != _canonical_bytes(
        _artifact_score_state_payload(frozen)
    ):
        raise ResidualContractError("encapsulated_score_state_mismatch")
    return effective_state_sha256


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
    model, model_sha256 = _read_json(model_file)
    manifest, manifest_sha256 = _read_json(manifest_file)

    if model.get("schema_version") != MODEL_SCHEMA:
        raise ResidualContractError("model_schema_mismatch")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ResidualContractError("manifest_schema_mismatch")
    if manifest.get("status") != "FROZEN_MODEL_READY_AWAITING_ACTIVATION":
        raise ResidualContractError("manifest_status_not_frozen")
    if (
        _require_sha256(manifest.get("model_sha256"), "manifest.model_sha256")
        != model_sha256
    ):
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
    if not all(
        isinstance(value, dict)
        for value in (algorithm, preprocessor, feature_contract, activation)
    ):
        raise ResidualContractError("model_contract_section_missing")
    if model.get("model_family") != "race_conditional_logit_with_fixed_market_offset":
        raise ResidualContractError("model_family_contract_mismatch")
    if _canonical_bytes(algorithm) != _canonical_bytes(FROZEN_ALGORITHM_CONTRACT):
        raise ResidualContractError("algorithm_contract_mismatch")
    if any(
        activation.get(field) is not False
        for field in (
            "activated",
            "production_pointer_changed",
            "runtime_changed",
            "cohort_cutoff_assigned",
        )
    ):
        raise ResidualContractError("artifact_is_activated")
    if tuple(feature_contract.get("feature_order") or ()) != FEATURES:
        raise ResidualContractError("feature_order_mismatch")
    if tuple(feature_contract.get("expanded_feature_order") or ()) != EXPANDED_FEATURES:
        raise ResidualContractError("expanded_feature_order_mismatch")
    if tuple(preprocessor.get("features") or ()) != FEATURES:
        raise ResidualContractError("preprocessor_feature_order_mismatch")
    if tuple(preprocessor.get("expanded_features") or ()) != EXPANDED_FEATURES:
        raise ResidualContractError("preprocessor_expanded_order_mismatch")

    beta = _immutable_vector(model.get("beta"), len(EXPANDED_FEATURES), "beta")
    medians = _immutable_vector(
        preprocessor.get("medians"), len(FEATURES), "preprocessor.medians"
    )
    means = _immutable_vector(
        preprocessor.get("means"), len(EXPANDED_FEATURES), "preprocessor.means"
    )
    scales = _immutable_vector(
        preprocessor.get("scales"), len(EXPANDED_FEATURES), "preprocessor.scales"
    )
    if np.any(scales <= 0.0):
        raise ResidualContractError("preprocessor_scale_not_positive")
    candidate_hashes = manifest.get("candidate_hashes")
    if not isinstance(candidate_hashes, dict) or not candidate_hashes:
        raise ResidualContractError("candidate_hashes_missing")
    for name, value in candidate_hashes.items():
        _require_sha256(value, f"candidate_hashes.{name}")
    derivation = manifest.get("derivation_contract")
    if not isinstance(derivation, dict) or _canonical_bytes(
        derivation
    ) != _canonical_bytes(FROZEN_DERIVATION_CONTRACT):
        raise ResidualContractError("shared_base_model_contract_mismatch")
    frozen = FrozenResidualModel(
        model=_deep_freeze(model),
        manifest=_deep_freeze(manifest),
        model_sha256=model_sha256,
        manifest_sha256=manifest_sha256,
        effective_state_sha256="",
        beta=beta,
        medians=medians,
        means=means,
        scales=scales,
        feature_order=tuple(feature_contract["feature_order"]),
        expanded_feature_order=tuple(feature_contract["expanded_feature_order"]),
        residual_cap=_finite_float(algorithm["residual_cap"], "residual_cap"),
        full_strength=_finite_float(
            derivation["full_strength"], "derivation.full_strength"
        ),
        half_strength=_finite_float(
            derivation["half_strength"], "derivation.half_strength"
        ),
        market_offset=str(algorithm["market_offset"]),
        normalization=str(algorithm["normalization"]),
        within_race_centering=algorithm["within_race_centering"] is True,
    )
    return replace(frozen, effective_state_sha256=_effective_state_sha256(frozen))


def _shadow_duplicate_identity(record: Mapping[str, Any]) -> dict[str, str]:
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
        "effective_state_sha256": _require_sha256(
            record.get("effective_state_sha256"), "effective_state_sha256"
        ),
    }


def _shadow_record_content(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in record.items()
        if key not in {"record_key", "record_checksum_sha256"}
    }


def _shadow_record_checksum(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(_shadow_record_content(record))).hexdigest()


def _shadow_record_key(record: Mapping[str, Any]) -> str:
    checksum = _shadow_record_checksum(record)
    return hashlib.sha256(
        _canonical_bytes(
            {
                "record_checksum_sha256": checksum,
                "schema_version": record.get("schema_version"),
            }
        )
    ).hexdigest()


def _seal_shadow_record(content: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(content)
    record["record_checksum_sha256"] = _shadow_record_checksum(record)
    record["record_key"] = _shadow_record_key(record)
    return record


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

    effective_state_sha256 = _verify_effective_state(frozen)
    if not isinstance(provenance, Mapping) or _contains_outcome(provenance):
        raise ResidualContractError("provenance_invalid_or_contains_outcome")
    extra_provenance = set(provenance) - PROVENANCE_FIELDS
    missing_provenance = PROVENANCE_FIELDS - set(provenance)
    if extra_provenance or missing_provenance:
        raise ResidualContractError(
            "provenance_field_contract_mismatch:"
            f"missing={sorted(missing_provenance)}:extra={sorted(extra_provenance)}"
        )
    if (
        not isinstance(runners, Sequence)
        or isinstance(runners, (str, bytes))
        or len(runners) < 2
    ):
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
    if (
        _require_sha256(provenance.get("runner_set_sha256"), "runner_set_sha256")
        != expected_hash
    ):
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
        decimal_odds = _finite_float(
            runner["strict_win_odds"], f"strict_win_odds:{runner_id}"
        )
        if decimal_odds <= 1.0:
            raise ResidualContractError(f"strict_win_odds_invalid:{runner_id}")
        feature_hash = _require_sha256(
            runner["feature_source_sha256"], f"feature_source_sha256:{runner_id}"
        )
        odds_hash = _require_sha256(
            runner["odds_source_sha256"], f"odds_source_sha256:{runner_id}"
        )
        feature_time = _parse_timestamp(
            runner["feature_freeze_timestamp"], f"feature_freeze_timestamp:{runner_id}"
        )
        odds_time = _parse_timestamp(
            runner["odds_capture_timestamp"], f"odds_capture_timestamp:{runner_id}"
        )
        if not feature_time < jump or not odds_time < jump:
            raise ResidualContractError(f"source_timestamp_not_prejump:{runner_id}")
        if score_time < feature_time or score_time < odds_time:
            raise ResidualContractError(f"score_timestamp_before_source:{runner_id}")
        features = runner["features"]
        if not isinstance(features, Mapping):
            raise ResidualContractError(f"features_not_object:{runner_id}")
        extra_features = set(features) - set(frozen.feature_order)
        if extra_features:
            raise ResidualContractError(
                f"unexpected_features:{runner_id}:{sorted(extra_features)}"
            )
        values: list[float] = []
        normalized_features: dict[str, float | None] = {}
        for feature in frozen.feature_order:
            value = features.get(feature)
            if value is None:
                values.append(float("nan"))
                normalized_features[feature] = None
            else:
                normalized = _finite_float(value, f"{runner_id}.{feature}")
                values.append(normalized)
                normalized_features[feature] = normalized
        raw_features.append(values)
        odds.append(decimal_odds)
        normalized_runners.append(
            {
                "race_id": race_id,
                "runner_id": runner_id,
                "box_number": box,
                "dog_name": dog_name,
                "strict_win_odds": decimal_odds,
                "features": normalized_features,
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

    raw = np.asarray(raw_features, dtype=float)
    missing = ~np.isfinite(raw)
    imputed = np.where(missing, frozen.medians, raw)
    expanded = np.concatenate([imputed, missing.astype(float)], axis=1)
    transformed = (expanded - frozen.means) / frozen.scales
    if frozen.within_race_centering:
        transformed -= np.mean(transformed, axis=0)
    adjustment = frozen.residual_cap * np.tanh(
        (transformed @ frozen.beta) / frozen.residual_cap
    )
    implied = 1.0 / np.asarray(odds, dtype=float)
    market = implied / np.sum(implied)
    full = _softmax(np.log(market) + frozen.full_strength * adjustment)
    half = _softmax(np.log(market) + frozen.half_strength * adjustment)

    predictions = []
    for index, runner in enumerate(normalized_runners):
        predictions.append(
            {
                **{key: value for key, value in runner.items() if key != "features"},
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
        "effective_state_sha256": effective_state_sha256,
    }
    canonical_provenance = {
        "race_id": race_id,
        "expected_runner_ids": expected_runner_ids,
        "runner_set_sha256": expected_hash,
        "jump_timestamp": str(provenance["jump_timestamp"]),
        "score_timestamp": str(provenance["score_timestamp"]),
    }
    return _seal_shadow_record(
        {
            "schema_version": SHADOW_RECORD_SCHEMA,
            **identity,
            "score_timestamp": str(provenance["score_timestamp"]),
            "jump_timestamp": str(provenance["jump_timestamp"]),
            "variants": {
                "full_strength": frozen.full_strength,
                "half_strength": frozen.half_strength,
            },
            "inputs": {
                "runners": normalized_runners,
                "provenance": canonical_provenance,
            },
            "predictions": predictions,
            "outcomes_present": False,
            "activation": False,
        }
    )


def _history_inputs(record: Mapping[str, Any]) -> tuple[list[Any], Mapping[str, Any]]:
    inputs = record.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ResidualContractError("history_migration_required")
    missing_input_sections = {"runners", "provenance"} - set(inputs)
    if missing_input_sections:
        raise ResidualContractError("history_migration_required")
    if set(inputs) != {"runners", "provenance"}:
        raise ResidualContractError("existing_shadow_unsupported_fields:inputs")
    runners = inputs["runners"]
    provenance = inputs["provenance"]
    if (
        not isinstance(runners, list)
        or not runners
        or not isinstance(provenance, Mapping)
    ):
        raise ResidualContractError("history_migration_required")
    if PROVENANCE_FIELDS - set(provenance):
        raise ResidualContractError("history_migration_required")
    if set(provenance) != PROVENANCE_FIELDS:
        raise ResidualContractError("existing_shadow_unsupported_fields:provenance")
    for runner in runners:
        if not isinstance(runner, Mapping) or RUNNER_FIELDS - set(runner):
            raise ResidualContractError("history_migration_required")
        if set(runner) != RUNNER_FIELDS:
            raise ResidualContractError("existing_shadow_unsupported_fields:runner")
        features = runner.get("features")
        if not isinstance(features, Mapping) or set(features) != set(FEATURES):
            if isinstance(features, Mapping) and set(features) - set(FEATURES):
                raise ResidualContractError(
                    "existing_shadow_unsupported_fields:features"
                )
            raise ResidualContractError("history_migration_required")
    return runners, provenance


def _require_canonical_history_order(
    record: Mapping[str, Any],
    runners: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    input_runner_ids = [runner.get("runner_id") for runner in runners]
    predictions = record.get("predictions")
    if not isinstance(predictions, list):
        raise ResidualContractError("existing_shadow_invalid_record:predictions")
    prediction_runner_ids: list[Any] = []
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ResidualContractError("existing_shadow_invalid_record:prediction")
        if set(prediction) != PREDICTION_FIELDS:
            raise ResidualContractError("existing_shadow_unsupported_fields:prediction")
        prediction_runner_ids.append(prediction.get("runner_id"))
    expected_runner_ids = provenance.get("expected_runner_ids")

    def ordered_runner_ids(values: Any) -> bool:
        return (
            isinstance(values, list)
            and bool(values)
            and all(isinstance(value, str) and value for value in values)
            and values == sorted(values)
        )

    if (
        not ordered_runner_ids(input_runner_ids)
        or not ordered_runner_ids(prediction_runner_ids)
        or not ordered_runner_ids(expected_runner_ids)
        or input_runner_ids != prediction_runner_ids
        or input_runner_ids != expected_runner_ids
    ):
        raise ResidualContractError("existing_shadow_noncanonical_order")


def _validate_existing_shadow_record(
    existing: Any, frozen: FrozenResidualModel, line_number: int
) -> tuple[dict[str, str], bytes]:
    if not isinstance(existing, dict):
        raise ResidualContractError(f"existing_shadow_invalid_record:{line_number}")
    if existing.get("schema_version") != SHADOW_RECORD_SCHEMA:
        raise ResidualContractError("history_migration_required")
    runners, provenance = _history_inputs(existing)
    if set(existing) != SHADOW_RECORD_FIELDS:
        raise ResidualContractError("existing_shadow_unsupported_fields:record")
    if _contains_outcome(existing) or existing.get("outcomes_present") is not False:
        raise ResidualContractError(f"existing_shadow_invalid_record:{line_number}")
    if existing.get("activation") is not False:
        raise ResidualContractError(f"existing_shadow_invalid_record:{line_number}")
    variants = existing.get("variants")
    if not isinstance(variants, Mapping):
        raise ResidualContractError("existing_shadow_invalid_record:variants")
    if set(variants) != {"full_strength", "half_strength"}:
        raise ResidualContractError("existing_shadow_unsupported_fields:variants")
    _require_canonical_history_order(existing, runners, provenance)
    try:
        checksum = _require_sha256(
            existing.get("record_checksum_sha256"),
            "existing.record_checksum_sha256",
        )
        record_key = _require_sha256(existing.get("record_key"), "existing.record_key")
    except ResidualContractError as exc:
        raise ResidualContractError("existing_shadow_checksum_mismatch") from exc
    if checksum != _shadow_record_checksum(
        existing
    ) or record_key != _shadow_record_key(existing):
        raise ResidualContractError("existing_shadow_checksum_mismatch")
    try:
        rescored = score_race(frozen, runners, provenance)
    except ResidualContractError as exc:
        raise ResidualContractError(
            f"existing_shadow_invalid_inputs:{line_number}"
        ) from exc
    encoded = _canonical_bytes(rescored)
    if _canonical_bytes(existing) != encoded:
        raise ResidualContractError("existing_shadow_prediction_input_mismatch")
    return _shadow_duplicate_identity(rescored), encoded


def _shadow_transaction_paths(output: Path) -> tuple[Path, Path]:
    """Return persistent-lock and replaceable-stage paths for one output."""

    return (
        output.with_name(f".{output.name}.lock"),
        output.with_name(f".{output.name}.tmp"),
    )


def _remove_staged_shadow_file(staged: Path) -> None:
    staged.unlink(missing_ok=True)


def _write_staged_shadow_bytes(handle: Any, replacement: bytes) -> None:
    handle.write(replacement)


def _flush_staged_shadow_file(handle: Any) -> None:
    handle.flush()


def _fsync_staged_shadow_file(handle: Any) -> None:
    os.fsync(handle.fileno())


def _publish_staged_shadow_file(staged: Path, output: Path) -> None:
    os.replace(staged, output)


def _fsync_parent_directory(output: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(output.parent, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_shadow_target(output: Path) -> tuple[bool, bytes, int | None]:
    try:
        target_stat = output.stat()
        return True, output.read_bytes(), stat.S_IMODE(target_stat.st_mode)
    except FileNotFoundError:
        return False, b"", None


def _validate_shadow_history(
    raw_history: bytes, frozen: FrozenResidualModel
) -> dict[bytes, bytes]:
    try:
        history = raw_history.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResidualContractError("existing_shadow_invalid_utf8") from exc

    history_by_identity: dict[bytes, bytes] = {}
    for line_number, line in enumerate(history.splitlines(keepends=True), start=1):
        if not line.strip():
            raise ResidualContractError(f"existing_shadow_blank_line:{line_number}")
        try:
            existing = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResidualContractError(
                f"existing_shadow_invalid_json:{line_number}"
            ) from exc
        if _canonical_bytes(existing).decode("utf-8") != line:
            raise ResidualContractError(
                f"existing_shadow_not_canonical_json:{line_number}"
            )
        existing_identity, existing_encoded = _validate_existing_shadow_record(
            existing, frozen, line_number
        )
        identity_key = _canonical_bytes(existing_identity)
        prior_encoded = history_by_identity.get(identity_key)
        if prior_encoded is not None:
            if prior_encoded == existing_encoded:
                raise ResidualContractError("duplicate_shadow_history_identity")
            raise ResidualContractError("conflicting_shadow_duplicate")
        history_by_identity[identity_key] = existing_encoded
    return history_by_identity


def append_shadow_record(
    path: Path | str,
    record: Mapping[str, Any],
    *,
    frozen: FrozenResidualModel,
    runners: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> str:
    """Transactionally append one v2 record after validating prior history.

    The embedded SHA-256 values detect corruption and inconsistent construction;
    they do not authenticate a row against coordinated host-level rewriting.
    Repeated prior stable identities fail as ``duplicate_shadow_history_identity``
    when their canonical content matches, or ``conflicting_shadow_duplicate`` when
    it differs. A persistent sidecar inode serializes the complete transaction;
    the target is never used as its own lock because atomic publication replaces
    that inode.

    A fully written, flushed, and fsynced same-directory staged file is atomically
    replaced onto the target. Successful replacement is the commit point.
    ``APPENDED`` means the committed bytes became visible. Parent-directory fsync
    is attempted afterward, but its failure is intentionally not reported as a
    rejected append and ``APPENDED`` does not claim crash durability when that
    fsync could not be obtained. ``COMMIT_STATE_UNKNOWN`` is returned only if a
    publication exception leaves neither the exact original nor intended target
    bytes observable; callers may safely retry under the same sidecar protocol.
    """

    output = Path(path)
    if output.suffix != ".jsonl" or not output.parent.is_dir():
        raise ResidualContractError("shadow_output_path_invalid")
    if (
        not isinstance(record, Mapping)
        or record.get("schema_version") != SHADOW_RECORD_SCHEMA
    ):
        raise ResidualContractError("shadow_record_schema_mismatch")
    if _contains_outcome(record) or record.get("outcomes_present") is not False:
        raise ResidualContractError("shadow_record_contains_outcome")
    _verify_effective_state(frozen)
    expected = score_race(frozen, runners, provenance)
    encoded = _canonical_bytes(expected)
    if _canonical_bytes(record) != encoded:
        raise ResidualContractError("shadow_record_not_canonical_score")
    if expected.get("record_checksum_sha256") != _shadow_record_checksum(
        expected
    ) or expected.get("record_key") != _shadow_record_key(expected):
        raise ResidualContractError("shadow_record_verified_key_mismatch")
    candidate_identity = _shadow_duplicate_identity(expected)
    lock_path, staged_path = _shadow_transaction_paths(output)
    lock_fd: int | None = None
    staged_exists = False
    try:
        lock_flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        lock_fd = os.open(lock_path, lock_flags, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        _remove_staged_shadow_file(staged_path)
        target_existed, original, target_mode = _read_shadow_target(output)
        history_by_identity = _validate_shadow_history(original, frozen)
        candidate_history = history_by_identity.get(
            _canonical_bytes(candidate_identity)
        )
        if candidate_history is not None:
            if candidate_history == encoded:
                return "EXACT_REPLAY"
            raise ResidualContractError("conflicting_shadow_duplicate")

        replacement = original + encoded
        staged_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        staged_fd = os.open(staged_path, staged_flags, 0o666)
        staged_exists = True
        if target_mode is not None:
            os.fchmod(staged_fd, target_mode)
        with os.fdopen(staged_fd, "wb", closefd=True) as staged_handle:
            _write_staged_shadow_bytes(staged_handle, replacement)
            _flush_staged_shadow_file(staged_handle)
            _fsync_staged_shadow_file(staged_handle)

        try:
            _publish_staged_shadow_file(staged_path, output)
        except Exception as exc:
            try:
                current_exists, current, _ = _read_shadow_target(output)
            except OSError:
                return "COMMIT_STATE_UNKNOWN"
            if current_exists and current == replacement:
                staged_exists = False
            elif current_exists == target_existed and current == original:
                raise OSError("shadow publication failed before commit") from exc
            else:
                return "COMMIT_STATE_UNKNOWN"
        else:
            staged_exists = False

        try:
            _fsync_parent_directory(output)
        except Exception:
            pass
        return "APPENDED"
    except OSError as exc:
        raise ResidualContractError(f"shadow_output_write_failed:{output}") from exc
    finally:
        if staged_exists:
            try:
                _remove_staged_shadow_file(staged_path)
            except OSError:
                pass
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
