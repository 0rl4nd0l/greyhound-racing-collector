"""Canonical input and core-output contracts for residual scoring parity.

This module owns the bytes that automatic and manual adapters must agree on.
It deliberately does not load artifacts, read history, persist predictions, or
call a network/service.  ``score_race`` remains the only scoring
implementation; this module validates its inputs and canonicalizes its
outcome-free result.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from src.predictor.market_form_residual import (
    EFFECTIVE_STATE_SCHEMA,
    FEATURES,
    FROZEN_ALGORITHM_CONTRACT,
    FROZEN_DERIVATION_CONTRACT,
    NUMERICAL_CANONICALIZATION_CONTRACT,
)

SCORING_INPUT_SCHEMA = "market_form_residual_scoring_input_v1"
SCORING_CORE_OUTPUT_SCHEMA = "market_form_residual_scoring_core_output_v1"
SCORING_CONFIG_SCHEMA = "market_form_residual_scoring_config_v1"
RANKING_CONTRACT = MappingProxyType(
    {"primary_probability": "full_probability", "tie_break": "box_ascending"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_OUTCOME_WORDS = frozenset(
    {
        "actual_win",
        "actual_result",
        "finish",
        "finish_position",
        "finishes",
        "official_result",
        "outcome",
        "outcomes",
        "placing",
        "result",
        "results",
        "winner",
        "winner_name",
        "winner_odds",
    }
)
_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "race_id",
        "runner_set_sha256",
        "runners",
        "timestamps",
        "identities",
        "scoring_parameters",
        "ranking",
    }
)
_RUNNER_FIELDS = frozenset(
    {
        "runner_id",
        "box_number",
        "dog_name",
        "features",
        "feature_source_sha256",
        "strict_win_odds",
        "odds_source_sha256",
        "feature_freeze_timestamp",
        "odds_capture_timestamp",
    }
)
_TIMESTAMP_FIELDS = frozenset(
    {"cutoff_timestamp", "capture_timestamp", "score_timestamp", "jump_timestamp"}
)
_IDENTITY_FIELDS = frozenset(
    {
        "model_id",
        "model_sha256",
        "manifest_sha256",
        "effective_state_schema",
        "effective_state_sha256",
        "config_sha256",
        "numeric_canonicalization_schema",
        "numeric_canonicalization_sha256",
    }
)


class ScoringParityRejected(ValueError):
    """A fail-closed canonical scoring contract rejection."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ScoringParityRejected(f"json_value_invalid:{type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            _json_value(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScoringParityRejected("canonical_json_invalid") from exc
    return (encoded + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _exact(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ScoringParityRejected(f"{label}_fields_invalid")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ScoringParityRejected(f"{label}_invalid")
    return value


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ScoringParityRejected(f"{label}_nonfinite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ScoringParityRejected(f"{label}_nonfinite") from exc
    if not math.isfinite(number) or (minimum is not None and number <= minimum):
        raise ScoringParityRejected(f"{label}_invalid")
    return number


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ScoringParityRejected(f"{label}_invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ScoringParityRejected(f"{label}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.isoformat() != value:
        raise ScoringParityRejected(f"{label}_invalid")
    return parsed


def _contains_outcome(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in _OUTCOME_WORDS or _contains_outcome(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_outcome(item) for item in value)
    return False


def _runner_set_sha256(runner_ids: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(sorted(runner_ids)) + "\n").encode("utf-8"))


def _dog_token(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", name.upper())


def _canonical_scoring_config() -> dict[str, Any]:
    return {
        "schema_version": SCORING_CONFIG_SCHEMA,
        "feature_order": list(FEATURES),
        "algorithm": _json_value(FROZEN_ALGORITHM_CONTRACT),
        "derivation": _json_value(FROZEN_DERIVATION_CONTRACT),
        "numeric_canonicalization": _json_value(NUMERICAL_CANONICALIZATION_CONTRACT),
        "ranking": dict(RANKING_CONTRACT),
        "effective_state_schema": EFFECTIVE_STATE_SCHEMA,
    }


SCORING_CONFIG = _canonical_scoring_config()
SCORING_CONFIG_SHA256 = sha256_bytes(canonical_bytes(SCORING_CONFIG))
NUMERIC_CANONICALIZATION_SHA256 = sha256_bytes(canonical_bytes(NUMERICAL_CANONICALIZATION_CONTRACT))


@dataclass(frozen=True)
class ScoringInputArtifact:
    document: Mapping[str, Any]
    raw: bytes
    sha256: str

    @property
    def scorer_runners(self) -> list[dict[str, Any]]:
        race_id = self.document["race_id"]
        return [
            {
                "race_id": race_id,
                "runner_id": row["runner_id"],
                "box_number": row["box_number"],
                "dog_name": row["dog_name"],
                "strict_win_odds": row["strict_win_odds"],
                "features": dict(row["features"]),
                "feature_source_sha256": row["feature_source_sha256"],
                "odds_source_sha256": row["odds_source_sha256"],
                "feature_freeze_timestamp": row["feature_freeze_timestamp"],
                "odds_capture_timestamp": row["odds_capture_timestamp"],
            }
            for row in self.document["runners"]
        ]

    @property
    def provenance(self) -> dict[str, Any]:
        timestamps = self.document["timestamps"]
        return {
            "expected_runner_ids": [row["runner_id"] for row in self.document["runners"]],
            "jump_timestamp": timestamps["jump_timestamp"],
            "race_id": self.document["race_id"],
            "runner_set_sha256": self.document["runner_set_sha256"],
            "score_timestamp": timestamps["score_timestamp"],
        }


@dataclass(frozen=True)
class CoreOutputArtifact:
    document: Mapping[str, Any]
    raw: bytes
    sha256: str


def build_scoring_input(
    *,
    race_id: str,
    runner_set_sha256: str,
    runners: Sequence[Mapping[str, Any]],
    cutoff_timestamp: str,
    capture_timestamp: str,
    score_timestamp: str,
    jump_timestamp: str,
    model_sha256: str,
    manifest_sha256: str,
    effective_state_sha256: str,
    config_sha256: str = SCORING_CONFIG_SHA256,
    model_id: str = "market_form_residual_v1",
    scoring_parameters: Mapping[str, Any] | None = None,
) -> ScoringInputArtifact:
    if (
        not isinstance(race_id, str)
        or not race_id.strip()
        or _contains_outcome({"race_id": race_id})
    ):
        raise ScoringParityRejected("race_id_invalid")
    if not isinstance(runners, Sequence) or isinstance(runners, (str, bytes)) or len(runners) < 2:
        raise ScoringParityRejected("runner_set_invalid")
    if not all(isinstance(row, Mapping) for row in runners):
        raise ScoringParityRejected("runner_set_invalid")
    if list(runners) != sorted(runners, key=lambda row: str(row.get("runner_id", ""))):
        raise ScoringParityRejected("runner_order_invalid")
    timestamps = {
        "cutoff_timestamp": cutoff_timestamp,
        "capture_timestamp": capture_timestamp,
        "score_timestamp": score_timestamp,
        "jump_timestamp": jump_timestamp,
    }
    parsed_timestamps = {key: _timestamp(value, key) for key, value in timestamps.items()}
    if not (
        parsed_timestamps["cutoff_timestamp"] <= parsed_timestamps["score_timestamp"]
        and parsed_timestamps["capture_timestamp"]
        <= parsed_timestamps["score_timestamp"]
        < parsed_timestamps["jump_timestamp"]
    ):
        raise ScoringParityRejected("timestamp_order_invalid")
    normalized_runners: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_boxes: set[int] = set()
    for index, supplied in enumerate(runners):
        runner = _exact(supplied, _RUNNER_FIELDS, f"runner[{index}]")
        if _contains_outcome(runner):
            raise ScoringParityRejected(f"runner[{index}]_contains_outcome")
        runner_id = runner["runner_id"]
        if not isinstance(runner_id, str) or not runner_id or runner_id in seen_ids:
            raise ScoringParityRejected(f"runner[{index}]_identity_invalid")
        seen_ids.add(runner_id)
        box_number = runner["box_number"]
        if (
            isinstance(box_number, bool)
            or not isinstance(box_number, int)
            or not 1 <= box_number <= 10
            or box_number in seen_boxes
        ):
            raise ScoringParityRejected(f"runner[{index}]_box_invalid")
        seen_boxes.add(box_number)
        dog_name = runner["dog_name"]
        if not isinstance(dog_name, str) or not dog_name or dog_name != dog_name.strip():
            raise ScoringParityRejected(f"runner[{index}]_name_invalid")
        if (
            not _dog_token(dog_name)
            or runner_id != f"{race_id}|box:{box_number}|dog:{_dog_token(dog_name)}"
        ):
            raise ScoringParityRejected(f"runner[{index}]_identity_mismatch")
        feature_values = runner["features"]
        if not isinstance(feature_values, Mapping) or tuple(feature_values) != FEATURES:
            raise ScoringParityRejected(f"runner[{index}]_feature_order_invalid")
        normalized_features: dict[str, float | None] = {}
        for feature in FEATURES:
            value = feature_values[feature]
            if value is None:
                normalized_features[feature] = None
            else:
                normalized_features[feature] = _finite(
                    value, f"runner[{index}].{feature}", minimum=None
                )
        feature_freeze = _timestamp(
            runner["feature_freeze_timestamp"], f"runner[{index}].feature_freeze_timestamp"
        )
        odds_capture = _timestamp(
            runner["odds_capture_timestamp"], f"runner[{index}].odds_capture_timestamp"
        )
        if (
            feature_freeze > parsed_timestamps["cutoff_timestamp"]
            or odds_capture > parsed_timestamps["capture_timestamp"]
        ):
            raise ScoringParityRejected(f"runner[{index}]_source_timestamp_invalid")
        normalized_runners.append(
            {
                "runner_id": runner_id,
                "box_number": box_number,
                "dog_name": dog_name,
                "features": normalized_features,
                "feature_source_sha256": _sha(
                    runner["feature_source_sha256"], f"runner[{index}].feature_source_sha256"
                ),
                "strict_win_odds": _finite(
                    runner["strict_win_odds"], f"runner[{index}].strict_win_odds", minimum=1.0
                ),
                "odds_source_sha256": _sha(
                    runner["odds_source_sha256"], f"runner[{index}].odds_source_sha256"
                ),
                "feature_freeze_timestamp": feature_freeze.isoformat(),
                "odds_capture_timestamp": odds_capture.isoformat(),
            }
        )
    runner_ids = [row["runner_id"] for row in normalized_runners]
    if runner_ids != sorted(runner_ids):
        raise ScoringParityRejected("runner_order_invalid")
    if _sha(runner_set_sha256, "runner_set_sha256") != _runner_set_sha256(runner_ids):
        raise ScoringParityRejected("runner_set_hash_mismatch")
    if not isinstance(model_id, str) or not model_id:
        raise ScoringParityRejected("model_id_invalid")
    identities = {
        "model_id": model_id,
        "model_sha256": _sha(model_sha256, "model_sha256"),
        "manifest_sha256": _sha(manifest_sha256, "manifest_sha256"),
        "effective_state_schema": EFFECTIVE_STATE_SCHEMA,
        "effective_state_sha256": _sha(effective_state_sha256, "effective_state_sha256"),
        "config_sha256": _sha(config_sha256, "config_sha256"),
        "numeric_canonicalization_schema": NUMERICAL_CANONICALIZATION_CONTRACT["schema_version"],
        "numeric_canonicalization_sha256": NUMERIC_CANONICALIZATION_SHA256,
    }
    parameters = dict(
        scoring_parameters
        or {
            "full_strength": 1.0,
            "half_strength": 0.5,
            "residual_cap": 0.35,
            "within_race_centering": True,
            "market_offset": "fixed_log_normalized_inverse_decimal_win_odds",
            "normalization": "softmax(log(market_probability)+strength*capped_residual)",
        }
    )
    if _contains_outcome(parameters):
        raise ScoringParityRejected("scoring_parameters_contains_outcome")
    if parameters != {
        "full_strength": 1.0,
        "half_strength": 0.5,
        "residual_cap": 0.35,
        "within_race_centering": True,
        "market_offset": "fixed_log_normalized_inverse_decimal_win_odds",
        "normalization": "softmax(log(market_probability)+strength*capped_residual)",
    }:
        raise ScoringParityRejected("scoring_parameters_invalid")
    document = {
        "schema_version": SCORING_INPUT_SCHEMA,
        "race_id": race_id,
        "runner_set_sha256": runner_set_sha256,
        "runners": normalized_runners,
        "timestamps": timestamps,
        "identities": identities,
        "scoring_parameters": parameters,
        "ranking": dict(RANKING_CONTRACT),
    }
    raw = canonical_bytes(document)
    return ScoringInputArtifact(document, raw, sha256_bytes(raw))


def build_core_output(
    scoring_input: ScoringInputArtifact, score_record: Mapping[str, Any]
) -> CoreOutputArtifact:
    if not isinstance(score_record, Mapping) or _contains_outcome(score_record):
        raise ScoringParityRejected("core_output_invalid_or_contains_outcome")
    if score_record.get("outcomes_present") is not False:
        raise ScoringParityRejected("core_output_outcome_marker_invalid")
    expected_identities = scoring_input.document["identities"]
    if any(
        score_record.get(record_key) != expected_value
        for record_key, expected_value in (
            ("race_id", scoring_input.document["race_id"]),
            ("runner_set_sha256", scoring_input.document["runner_set_sha256"]),
            ("model_sha256", expected_identities["model_sha256"]),
            ("manifest_sha256", expected_identities["manifest_sha256"]),
            ("effective_state_sha256", expected_identities["effective_state_sha256"]),
        )
    ):
        raise ScoringParityRejected("core_output_identity_mismatch")
    predictions = score_record.get("predictions")
    if not isinstance(predictions, list):
        raise ScoringParityRejected("core_output_predictions_invalid")
    expected_by_id = {row["runner_id"]: row for row in scoring_input.document["runners"]}
    actual_by_id: dict[str, Mapping[str, Any]] = {}
    for row in predictions:
        if not isinstance(row, Mapping) or not isinstance(row.get("runner_id"), str):
            raise ScoringParityRejected("core_output_runner_set_invalid")
        if row["runner_id"] in actual_by_id:
            raise ScoringParityRejected("core_output_runner_set_invalid")
        actual_by_id[row["runner_id"]] = row
    if set(actual_by_id) != set(expected_by_id) or len(actual_by_id) != len(predictions):
        raise ScoringParityRejected("core_output_runner_set_invalid")
    canonical_predictions: list[dict[str, Any]] = []
    for runner in scoring_input.document["runners"]:
        row = actual_by_id[runner["runner_id"]]
        canonical_predictions.append(
            {
                "runner_id": runner["runner_id"],
                "box_number": runner["box_number"],
                "dog_name": runner["dog_name"],
                "strict_win_odds": runner["strict_win_odds"],
                "market_probability": _finite(row.get("market_probability"), "market_probability"),
                "residual_adjustment": _finite(
                    row.get("residual_adjustment"), "residual_adjustment"
                ),
                "full_probability": _finite(row.get("full_probability"), "full_probability"),
                "half_probability": _finite(row.get("half_probability"), "half_probability"),
            }
        )
    ranked = sorted(
        canonical_predictions,
        key=lambda row: (-row["full_probability"], row["box_number"]),
    )
    if any(
        not 0.0 <= row[field] <= 1.0
        for row in canonical_predictions
        for field in ("market_probability", "full_probability", "half_probability")
    ):
        raise ScoringParityRejected("core_output_probability_invalid")
    ranking = [
        {
            "rank": rank,
            "runner_id": row["runner_id"],
            "box_number": row["box_number"],
        }
        for rank, row in enumerate(ranked, start=1)
    ]
    document = {
        "schema_version": SCORING_CORE_OUTPUT_SCHEMA,
        "scoring_input_sha256": scoring_input.sha256,
        "race_id": scoring_input.document["race_id"],
        "runner_set_sha256": scoring_input.document["runner_set_sha256"],
        "identities": dict(expected_identities),
        "ranking": dict(RANKING_CONTRACT),
        "predictions": canonical_predictions,
        "ranks": ranking,
    }
    raw = canonical_bytes(document)
    return CoreOutputArtifact(document, raw, sha256_bytes(raw))


def parity_binding(
    scoring_input: ScoringInputArtifact, core_output: CoreOutputArtifact
) -> dict[str, str]:
    return {
        "input_schema_version": SCORING_INPUT_SCHEMA,
        "input_sha256": scoring_input.sha256,
        "core_output_schema_version": SCORING_CORE_OUTPUT_SCHEMA,
        "core_output_sha256": core_output.sha256,
        "config_sha256": scoring_input.document["identities"]["config_sha256"],
        "numeric_canonicalization_sha256": scoring_input.document["identities"][
            "numeric_canonicalization_sha256"
        ],
    }


def first_difference_diagnostic(expected: bytes, actual: bytes) -> dict[str, Any]:
    if expected == actual:
        return {"equal": True}
    limit = min(len(expected), len(actual))
    offset = next(
        (index for index in range(limit) if expected[index] != actual[index]),
        limit,
    )
    return {
        "equal": False,
        "offset": offset,
        "expected_length": len(expected),
        "actual_length": len(actual),
        "expected_byte": expected[offset] if offset < len(expected) else None,
        "actual_byte": actual[offset] if offset < len(actual) else None,
        "expected_context": expected[max(0, offset - 24) : offset + 24].decode("utf-8", "replace"),
        "actual_context": actual[max(0, offset - 24) : offset + 24].decode("utf-8", "replace"),
    }


def compare_parity(
    left_input: ScoringInputArtifact,
    right_input: ScoringInputArtifact,
    left_core: CoreOutputArtifact,
    right_core: CoreOutputArtifact,
) -> dict[str, Any]:
    return {
        "input_hash_equal": left_input.sha256 == right_input.sha256,
        "core_output_hash_equal": left_core.sha256 == right_core.sha256,
        "input_first_difference": first_difference_diagnostic(left_input.raw, right_input.raw),
        "core_output_first_difference": first_difference_diagnostic(left_core.raw, right_core.raw),
    }


__all__ = [
    "SCORING_CONFIG",
    "SCORING_CONFIG_SHA256",
    "SCORING_CORE_OUTPUT_SCHEMA",
    "SCORING_INPUT_SCHEMA",
    "CoreOutputArtifact",
    "ScoringInputArtifact",
    "ScoringParityRejected",
    "build_core_output",
    "build_scoring_input",
    "canonical_bytes",
    "compare_parity",
    "first_difference_diagnostic",
    "parity_binding",
    "sha256_bytes",
]
