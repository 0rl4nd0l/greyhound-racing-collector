"""Pure versioned feature derivation from immutable Sealed Race Evidence bytes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .domain import ArtifactChecksum, EvidenceField
from .model_bundle import SUPPORTED_FEATURE_CONTRACT

class FeatureQuarantine(ValueError):
    """Sealed evidence cannot satisfy the exact bundle feature contract."""


@dataclass(frozen=True, slots=True)
class FeatureContract:
    version: str
    schema_checksum: ArtifactChecksum
    missingness_policy_checksum: ArtifactChecksum
    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    runner_ids: tuple[str, ...]
    columns: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    checksum: ArtifactChecksum


@dataclass(frozen=True, slots=True)
class DerivationReport:
    version: str
    evidence_checksum: ArtifactChecksum
    output_checksum: ArtifactChecksum
    explicit_missing: Mapping[str, tuple[str, ...]]
    inapplicable: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class DerivationResult:
    matrix: FeatureMatrix
    contract: FeatureContract
    report: DerivationReport


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _checksum(content: bytes) -> ArtifactChecksum:
    return ArtifactChecksum(f"sha256:{hashlib.sha256(content).hexdigest()}")


def derive_features(
    sealed_evidence: bytes,
    *,
    expected_evidence_checksum: ArtifactChecksum,
    schema_bytes: bytes,
    expected_schema_checksum: ArtifactChecksum,
    missingness_policy_bytes: bytes,
    expected_missingness_checksum: ArtifactChecksum,
) -> DerivationResult:
    """Derive without IO, mutation, repair, defaults, or mutable source access."""
    if _checksum(sealed_evidence) != expected_evidence_checksum:
        raise FeatureQuarantine("sealed evidence checksum mismatch")
    if _checksum(schema_bytes) != expected_schema_checksum:
        raise FeatureQuarantine("feature schema checksum mismatch")
    if _checksum(missingness_policy_bytes) != expected_missingness_checksum:
        raise FeatureQuarantine("missingness policy checksum mismatch")
    try:
        evidence, schema, policy = map(
            json.loads, (sealed_evidence, schema_bytes, missingness_policy_bytes)
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FeatureQuarantine("derivation input is not valid JSON") from error
    if not all(type(value) is dict for value in (evidence, schema, policy)):
        raise FeatureQuarantine("derivation inputs must be JSON objects")
    if schema.get("contract_version") != SUPPORTED_FEATURE_CONTRACT:
        raise FeatureQuarantine("feature contract version mismatch")
    if evidence.get("schema_version") != schema.get("evidence_schema_version"):
        raise FeatureQuarantine("sealed evidence schema mismatch")
    if evidence.get("normalization_version") != schema.get("normalization_version"):
        raise FeatureQuarantine("sealed evidence normalization mismatch")
    # Phase 2's closed EvidenceField registry carries runner_set and separate
    # field mappings; no parallel top-level runner envelope is accepted.
    normalized_fields = evidence.get("fields")
    if type(normalized_fields) is not dict:
        raise FeatureQuarantine("sealed evidence fields must be an object")
    runners = normalized_fields.get("runner_set")
    runner_features = normalized_fields.get("runner_features")
    identities = normalized_fields.get("runner_identity", {})
    fields = schema.get("fields")
    if (
        type(runners) is not list
        or not runners
        or type(identities) is not dict
        or type(runner_features) is not dict
        or set(runner_features) != set(runners)
        or type(fields) is not list
        or not fields
    ):
        raise FeatureQuarantine("Phase 2 runner_set/identity fields are required")
    columns = tuple(item.get("name") for item in fields if type(item) is dict)
    if len(columns) != len(fields) or any(type(name) is not str or not name for name in columns):
        raise FeatureQuarantine("feature schema fields are invalid")
    if len(set(columns)) != len(columns):
        raise FeatureQuarantine("feature schema contains duplicate field names")
    semantics = {"identity-critical", "forecast-required", "optional", "inapplicable"}
    if any(item.get("semantics") not in semantics for item in fields):
        raise FeatureQuarantine("feature schema contains invalid semantics")
    if any(
        type(item.get("source_field")) is not str
        or item["source_field"] != EvidenceField.RUNNER_FEATURES.value
        for item in fields
    ):
        raise FeatureQuarantine("per-runner feature source must be runner_features")
    for item in fields:
        if item["semantics"] == "inapplicable":
            encoded = item.get("encoded_value")
            if (
                type(encoded) not in (int, float)
                or isinstance(encoded, bool)
                or not math.isfinite(encoded)
            ):
                raise FeatureQuarantine(
                    "inapplicable fields require a declared finite numeric encoding"
                )
    imputation = policy.get("imputation", {})
    optional = {item["name"] for item in fields if item["semantics"] == "optional"}
    if (
        type(imputation) is not dict
        or set(imputation) != optional
        or any(
            type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value)
            for value in imputation.values()
        )
    ):
        raise FeatureQuarantine("bundle imputation must exactly cover optional finite values")
    rows: list[tuple[float, ...]] = []
    ids: list[str] = []
    explicit_missing: dict[str, tuple[str, ...]] = {}
    inapplicable: dict[str, tuple[str, ...]] = {}
    for runner_id in runners:
        if (
            type(runner_id) is not str
            or not runner_id
            or runner_id in ids
            or identities.get(runner_id) != "authoritative"
        ):
            raise FeatureQuarantine("runner identity is missing, ambiguous, or duplicated")
        values: list[float] = []
        missing: list[str] = []
        absent: list[str] = []
        for item in fields:
            name, semantics = item["name"], item.get("semantics")
            source_field = item.get("source_field", "runner_features")
            if source_field != "runner_features":
                raise FeatureQuarantine("per-runner features must come from runner_features")
            feature_values = runner_features[runner_id]
            if type(feature_values) is not dict:
                raise FeatureQuarantine("runner_features values must be objects")
            value = feature_values.get(name)
            if type(value) is dict and value == {"inapplicable": True}:
                if semantics != "inapplicable":
                    raise FeatureQuarantine(f"{name} is unexpectedly inapplicable")
                absent.append(name)
                values.append(float(item["encoded_value"]))
            elif type(value) is dict and value == {"missing": True}:
                if semantics != "optional":
                    raise FeatureQuarantine(f"required feature is missing: {name}")
                missing.append(name)
                values.append(float(imputation[name]))
            elif semantics == "inapplicable":
                raise FeatureQuarantine(f"{name} must be explicitly inapplicable")
            elif (
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise FeatureQuarantine(f"feature is not numeric: {name}")
            else:
                values.append(float(value))
        ids.append(runner_id)
        rows.append(tuple(values))
        explicit_missing[runner_id] = tuple(missing)
        inapplicable[runner_id] = tuple(absent)
    output = {"runner_ids": ids, "columns": columns, "rows": rows}
    output_checksum = _checksum(_canonical(output))
    contract = FeatureContract(
        SUPPORTED_FEATURE_CONTRACT, expected_schema_checksum, expected_missingness_checksum, columns
    )
    matrix = FeatureMatrix(tuple(ids), columns, tuple(rows), output_checksum)
    report = DerivationReport(
        SUPPORTED_FEATURE_CONTRACT,
        expected_evidence_checksum,
        output_checksum,
        explicit_missing,
        inapplicable,
    )
    return DerivationResult(matrix, contract, report)
