"""Pure versioned feature derivation from immutable Sealed Race Evidence bytes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Mapping

from .domain import ArtifactChecksum, EvidenceField
from .model_bundle import SUPPORTED_FEATURE_CONTRACT


class FeatureQuarantine(ValueError):
    """Sealed evidence cannot satisfy the exact bundle feature contract."""


FEATURE_AVAILABILITY_MANIFEST_VERSION = "feature-availability-manifest-v1"

RESULT_DERIVED_INPUT_NAMES = frozenset(
    {
        "finish_order",
        "finish_position",
        "official_finishing_order",
        "official_order",
        "outcome",
        "outcome_derived",
        "place",
        "placing",
        "position",
        "post_jump_odds",
        "post_jump_price",
        "post_jump_prices",
        "post_race_weather",
        "result",
        "result_order",
        "winner",
    }
)


class FeatureAvailabilityStatus(str, Enum):
    READY_NOW = "READY_NOW"
    DEVELOPMENT_ONLY = "DEVELOPMENT_ONLY"
    FORWARD_CAPTURE = "FORWARD_CAPTURE"
    EXCLUDED = "EXCLUDED"


class FeatureBlockingReason(str, Enum):
    LATE_RECEIPT = "LATE_RECEIPT"
    NAME_ONLY_IDENTITY = "NAME_ONLY_IDENTITY"
    NATIVE_ID_MISSING = "NATIVE_ID_MISSING"
    INCOMPLETE_COVERAGE = "INCOMPLETE_COVERAGE"
    PROVIDER_PUBLICATION_TIME_MISSING = "PROVIDER_PUBLICATION_TIME_MISSING"
    SOURCE_AUTHORIZATION_REQUIRED = "SOURCE_AUTHORIZATION_REQUIRED"
    SOURCE_SCHEMA_UNSUPPORTED = "SOURCE_SCHEMA_UNSUPPORTED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    RIGHTS_BASIS_MISSING = "RIGHTS_BASIS_MISSING"
    UNJUSTIFIED_MODEL_ROLE = "UNJUSTIFIED_MODEL_ROLE"
    NOT_REQUESTED_BY_BASELINE = "NOT_REQUESTED_BY_BASELINE"
    RAW_EVIDENCE_MISSING = "RAW_EVIDENCE_MISSING"
    NORMALIZED_EVIDENCE_MISSING = "NORMALIZED_EVIDENCE_MISSING"
    PROVIDER_TIME_AFTER_RECEIPT = "PROVIDER_TIME_AFTER_RECEIPT"


@dataclass(frozen=True, slots=True)
class FeatureAvailability:
    feature: str
    family: str
    semantics: str
    status: FeatureAvailabilityStatus
    source_name: str | None
    source_schema_version: str | None
    source_native_race_id: str | None
    source_native_runner_ids: tuple[str, ...]
    raw_checksum: ArtifactChecksum | None
    normalized_checksum: ArtifactChecksum | None
    provider_published_at: str | None
    collector_received_at: str | None
    feature_frozen_at: str
    completeness: str
    whole_race_coverage: bool
    derivation_version: str
    blocking_reasons: tuple[FeatureBlockingReason, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "family": self.family,
            "semantics": self.semantics,
            "status": self.status.value,
            "source_name": self.source_name,
            "source_schema_version": self.source_schema_version,
            "source_native_race_id": self.source_native_race_id,
            "source_native_runner_ids": list(self.source_native_runner_ids),
            "raw_checksum": str(self.raw_checksum) if self.raw_checksum else None,
            "normalized_checksum": (
                str(self.normalized_checksum) if self.normalized_checksum else None
            ),
            "provider_published_at": self.provider_published_at,
            "collector_received_at": self.collector_received_at,
            "feature_frozen_at": self.feature_frozen_at,
            "completeness": self.completeness,
            "whole_race_coverage": self.whole_race_coverage,
            "derivation_version": self.derivation_version,
            "blocking_reasons": [reason.value for reason in self.blocking_reasons],
        }


@dataclass(frozen=True, slots=True)
class FeatureAvailabilityManifest:
    version: str
    race_id: str
    evidence_checksum: ArtifactChecksum
    entries: tuple[FeatureAvailability, ...]
    checksum: ArtifactChecksum

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "race_id": self.race_id,
            "evidence_checksum": str(self.evidence_checksum),
            "entries": [entry.as_dict() for entry in self.entries],
        }


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
    availability_manifest_checksum: ArtifactChecksum | None = None
    availability_statuses: Mapping[str, str] = dataclass_field(default_factory=dict)
    blocking_reasons: Mapping[str, tuple[str, ...]] = dataclass_field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DerivationResult:
    matrix: FeatureMatrix
    contract: FeatureContract
    report: DerivationReport
    availability_manifest: FeatureAvailabilityManifest | None = None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _checksum(content: bytes) -> ArtifactChecksum:
    return ArtifactChecksum(f"sha256:{hashlib.sha256(content).hexdigest()}")


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise FeatureQuarantine(f"{name} must be known nonblank text")
    return value


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _timestamp(value: Any, name: str) -> tuple[datetime, str]:
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise FeatureQuarantine(f"{name} must be a timezone-aware timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FeatureQuarantine(f"{name} must be a timezone-aware timestamp")
    return parsed, parsed.isoformat(timespec="microseconds")


def _optional_timestamp(value: Any, name: str) -> tuple[datetime | None, str | None]:
    if value is None:
        return None, None
    return _timestamp(value, name)


def _artifact_checksum(value: Any, name: str) -> ArtifactChecksum | None:
    if value is None:
        return None
    try:
        return ArtifactChecksum(_text(value, name))
    except ValueError as error:
        raise FeatureQuarantine(f"{name} must be a SHA-256 checksum") from error


def _reject_result_derived_input(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is str and key.casefold() in RESULT_DERIVED_INPUT_NAMES:
                raise FeatureQuarantine(
                    "Feature Contract contains result or post-jump input"
                )
            _reject_result_derived_input(nested)
    elif type(value) is list:
        for nested in value:
            _reject_result_derived_input(nested)


def _availability_manifest(
    *,
    evidence: Mapping[str, Any],
    schema: Mapping[str, Any],
    fields: list[Mapping[str, Any]],
    runner_ids: tuple[str, ...],
    evidence_checksum: ArtifactChecksum,
    expected_feature_cutoff_at: datetime | None,
    raw_evidence_reader: Callable[[ArtifactChecksum], bytes] | None,
) -> FeatureAvailabilityManifest | None:
    version = schema.get("availability_manifest_version")
    declared = evidence.get("fields", {}).get("feature_availability")
    if version is None:
        if declared is not None:
            raise FeatureQuarantine(
                "feature availability requires an explicit Feature Contract version"
            )
        return None
    if version != FEATURE_AVAILABILITY_MANIFEST_VERSION:
        raise FeatureQuarantine("feature availability manifest version mismatch")
    if set(evidence) != {
        "schema_version",
        "normalization_version",
        "race_id",
        "fields",
        "field_provenance",
        "freeze",
    }:
        raise FeatureQuarantine("manifest-aware Sealed Race Evidence envelope is invalid")
    allowed_evidence_fields = {
        field.value for field in EvidenceField if field is not EvidenceField.RESULT_ORDER
    } | {"feature_availability"}
    if not set(evidence["fields"]).issubset(allowed_evidence_fields):
        raise FeatureQuarantine(
            "manifest-aware evidence contains a field outside the pre-jump registry"
        )
    freeze = evidence.get("freeze")
    if type(freeze) is not dict:
        raise FeatureQuarantine("sealed evidence freeze envelope is missing")
    frozen_at, frozen_text = _timestamp(freeze.get("at"), "feature freeze")
    if (
        expected_feature_cutoff_at is None
        or expected_feature_cutoff_at.tzinfo is None
        or expected_feature_cutoff_at.utcoffset() is None
        or frozen_at != expected_feature_cutoff_at
    ):
        raise FeatureQuarantine("feature cutoff disagrees with immutable authority")
    race_id = _text(evidence.get("race_id"), "sealed race identity")
    candidates = schema.get("candidate_features", [])
    source_contracts = schema.get("source_contracts")
    if type(source_contracts) is not dict or not source_contracts:
        raise FeatureQuarantine("manifest-aware Feature Contract needs source contracts")
    for source_name, contract in source_contracts.items():
        if (
            type(source_name) is not str
            or not source_name
            or type(contract) is not dict
            or set(contract) != {"schema_versions", "provider_publication_time_exposed"}
            or type(contract["schema_versions"]) is not list
            or not contract["schema_versions"]
            or any(type(value) is not str or not value for value in contract["schema_versions"])
            or len(set(contract["schema_versions"])) != len(contract["schema_versions"])
            or type(contract["provider_publication_time_exposed"]) is not bool
        ):
            raise FeatureQuarantine("source contract declaration is invalid")
    if type(candidates) is not list:
        raise FeatureQuarantine("candidate feature declarations must be a list")
    declared_features = [*fields, *candidates]
    if any(
        type(item) is not dict
        or set(item) != {"name", "family", "semantics"}
        or item.get("semantics") not in {"optional", "inapplicable"}
        for item in candidates
    ):
        raise FeatureQuarantine("candidate feature declaration is invalid")
    names = [item.get("name") for item in declared_features]
    if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
        raise FeatureQuarantine("declared feature identities are invalid or duplicated")
    if type(declared) is not dict or set(declared) != set(names):
        raise FeatureQuarantine("feature availability must exactly cover declared features")

    entries: list[FeatureAvailability] = []
    exact_keys = {
        "status",
        "source_name",
        "source_schema_version",
        "source_native_race_id",
        "source_native_runner_ids",
        "provider_published_at",
        "collector_received_at",
        "completeness",
        "whole_race_coverage",
        "derivation_version",
        "blocking_reasons",
    }
    normalized_fields = evidence["fields"]
    provenance = evidence.get("field_provenance")
    if type(provenance) is not list:
        raise FeatureQuarantine("sealed evidence provenance must be a list")
    raw_bindings: dict[str, ArtifactChecksum] = {}
    for binding in provenance:
        if (
            type(binding) is dict
            and binding.get("field") == EvidenceField.RUNNER_FEATURES.value
            and binding.get("value") == normalized_fields.get("runner_features")
            and type(binding.get("source")) is str
        ):
            checksum = _artifact_checksum(
                binding.get("artifact_checksum"), "runner feature raw checksum"
            )
            if checksum is None:
                continue
            prior = raw_bindings.setdefault(binding["source"], checksum)
            if prior != checksum:
                raise FeatureQuarantine("runner feature raw source binding conflicts")
    verified_raw_bindings: dict[str, ArtifactChecksum] = {}
    if raw_evidence_reader is not None:
        for source_name, checksum in raw_bindings.items():
            try:
                raw_content = raw_evidence_reader(checksum)
            except (KeyError, OSError, ValueError):
                continue
            if type(raw_content) is not bytes or _checksum(raw_content) != checksum:
                raise FeatureQuarantine("raw evidence checksum mismatch")
            verified_raw_bindings[source_name] = checksum

    active_names = {item["name"] for item in fields}
    policy_blockers = {
        FeatureBlockingReason.SOURCE_AUTHORIZATION_REQUIRED,
        FeatureBlockingReason.RIGHTS_BASIS_MISSING,
        FeatureBlockingReason.UNJUSTIFIED_MODEL_ROLE,
        FeatureBlockingReason.NOT_REQUESTED_BY_BASELINE,
    }
    development_blockers = {
        FeatureBlockingReason.LATE_RECEIPT,
        FeatureBlockingReason.NAME_ONLY_IDENTITY,
        FeatureBlockingReason.NATIVE_ID_MISSING,
        FeatureBlockingReason.INCOMPLETE_COVERAGE,
        FeatureBlockingReason.PROVIDER_PUBLICATION_TIME_MISSING,
        FeatureBlockingReason.SOURCE_SCHEMA_UNSUPPORTED,
        FeatureBlockingReason.RAW_EVIDENCE_MISSING,
        FeatureBlockingReason.NORMALIZED_EVIDENCE_MISSING,
        FeatureBlockingReason.PROVIDER_TIME_AFTER_RECEIPT,
    }
    for item in declared_features:
        name = item["name"]
        semantics = item["semantics"]
        family = item.get("family", "feature-contract")
        _text(family, f"{name} feature family")
        metadata = declared[name]
        if type(metadata) is not dict or set(metadata) != exact_keys:
            raise FeatureQuarantine(f"{name} availability envelope is invalid")
        try:
            status = FeatureAvailabilityStatus(metadata["status"])
            reasons = tuple(
                FeatureBlockingReason(reason) for reason in metadata["blocking_reasons"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise FeatureQuarantine(f"{name} availability classification is invalid") from error
        if len(set(reasons)) != len(reasons):
            raise FeatureQuarantine(f"{name} blocking reasons are duplicated")
        native_runner_ids = metadata["source_native_runner_ids"]
        if type(native_runner_ids) is not list or any(
            type(runner_id) is not str or not runner_id for runner_id in native_runner_ids
        ):
            raise FeatureQuarantine(f"{name} source-native runner identities are invalid")
        native_runner_ids = tuple(native_runner_ids)
        if len(set(native_runner_ids)) != len(native_runner_ids):
            raise FeatureQuarantine(f"{name} source-native runner identities are duplicated")
        provider_at, provider_text = _optional_timestamp(
            metadata["provider_published_at"], f"{name} provider publication"
        )
        receipt_at, receipt_text = _optional_timestamp(
            metadata["collector_received_at"], f"{name} collector receipt"
        )
        completeness = _text(metadata["completeness"], f"{name} completeness")
        if completeness not in {"COMPLETE", "INCOMPLETE", "UNKNOWN"}:
            raise FeatureQuarantine(f"{name} completeness is invalid")
        whole_race = metadata["whole_race_coverage"]
        if type(whole_race) is not bool:
            raise FeatureQuarantine(f"{name} whole-race coverage must be boolean")
        source_name = _optional_text(metadata["source_name"], f"{name} source name")
        source_schema = _optional_text(
            metadata["source_schema_version"], f"{name} source schema version"
        )
        source_race = _optional_text(
            metadata["source_native_race_id"], f"{name} source-native race identity"
        )
        raw_checksum = (
            verified_raw_bindings.get(source_name) if source_name is not None else None
        )
        source_contract = source_contracts.get(source_name) if source_name is not None else None
        publication_exposed = (
            source_contract["provider_publication_time_exposed"]
            if source_contract is not None
            else False
        )
        objective: set[FeatureBlockingReason] = set()
        if source_name is None:
            objective.add(FeatureBlockingReason.SOURCE_UNAVAILABLE)
        else:
            if (
                source_schema is None
                or source_contract is None
                or source_schema not in source_contract["schema_versions"]
            ):
                objective.add(FeatureBlockingReason.SOURCE_SCHEMA_UNSUPPORTED)
            if source_race is None or native_runner_ids != runner_ids:
                objective.add(FeatureBlockingReason.NATIVE_ID_MISSING)
            if raw_checksum is None:
                objective.add(FeatureBlockingReason.RAW_EVIDENCE_MISSING)
            if receipt_at is None:
                objective.add(FeatureBlockingReason.SOURCE_UNAVAILABLE)
            elif receipt_at >= frozen_at:
                objective.add(FeatureBlockingReason.LATE_RECEIPT)
            if publication_exposed and provider_at is None:
                objective.add(FeatureBlockingReason.PROVIDER_PUBLICATION_TIME_MISSING)
            if provider_at is not None and receipt_at is not None and provider_at > receipt_at:
                objective.add(FeatureBlockingReason.PROVIDER_TIME_AFTER_RECEIPT)
        if completeness != "COMPLETE" or not whole_race:
            objective.add(FeatureBlockingReason.INCOMPLETE_COVERAGE)
        normalized_checksum = evidence_checksum if name in active_names else None
        if normalized_checksum is None:
            objective.add(FeatureBlockingReason.NORMALIZED_EVIDENCE_MISSING)
        stated = set(reasons)
        if not objective.issubset(stated) or any(
            reason not in objective and reason not in policy_blockers for reason in stated
        ):
            raise FeatureQuarantine(f"{name} blockers disagree with sealed evidence")
        if status is FeatureAvailabilityStatus.READY_NOW:
            valid_status = name in active_names and not stated
        elif status is FeatureAvailabilityStatus.DEVELOPMENT_ONLY:
            valid_status = bool(stated) and stated.issubset(development_blockers)
        elif status is FeatureAvailabilityStatus.FORWARD_CAPTURE:
            valid_status = bool(
                stated
                & {
                    FeatureBlockingReason.SOURCE_AUTHORIZATION_REQUIRED,
                    FeatureBlockingReason.SOURCE_UNAVAILABLE,
                }
            )
        else:
            valid_status = bool(
                name not in active_names
                and stated
                & {
                    FeatureBlockingReason.RIGHTS_BASIS_MISSING,
                    FeatureBlockingReason.UNJUSTIFIED_MODEL_ROLE,
                    FeatureBlockingReason.NOT_REQUESTED_BY_BASELINE,
                    FeatureBlockingReason.SOURCE_SCHEMA_UNSUPPORTED,
                }
            )
        if not valid_status:
            raise FeatureQuarantine(f"{name} status and blockers are inconsistent")
        entries.append(
            FeatureAvailability(
                feature=name,
                family=family,
                semantics=semantics,
                status=status,
                source_name=source_name,
                source_schema_version=source_schema,
                source_native_race_id=source_race,
                source_native_runner_ids=native_runner_ids,
                raw_checksum=raw_checksum,
                normalized_checksum=normalized_checksum,
                provider_published_at=provider_text,
                collector_received_at=receipt_text,
                feature_frozen_at=frozen_text,
                completeness=completeness,
                whole_race_coverage=whole_race,
                derivation_version=_text(
                    metadata["derivation_version"], f"{name} derivation version"
                ),
                blocking_reasons=reasons,
            )
        )
    document = {
        "version": FEATURE_AVAILABILITY_MANIFEST_VERSION,
        "race_id": race_id,
        "evidence_checksum": str(evidence_checksum),
        "entries": [entry.as_dict() for entry in entries],
    }
    return FeatureAvailabilityManifest(
        FEATURE_AVAILABILITY_MANIFEST_VERSION,
        race_id,
        evidence_checksum,
        tuple(entries),
        _checksum(_canonical(document)),
    )


def derive_features(
    sealed_evidence: bytes,
    *,
    expected_evidence_checksum: ArtifactChecksum,
    schema_bytes: bytes,
    expected_schema_checksum: ArtifactChecksum,
    missingness_policy_bytes: bytes,
    expected_missingness_checksum: ArtifactChecksum,
    expected_feature_cutoff_at: datetime | None = None,
    raw_evidence_reader: Callable[[ArtifactChecksum], bytes] | None = None,
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
    _reject_result_derived_input(evidence)
    _reject_result_derived_input(schema)
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
    if any(
        type(runner_id) is not str
        or not runner_id
        or runners.count(runner_id) != 1
        or identities.get(runner_id) != "authoritative"
        for runner_id in runners
    ):
        raise FeatureQuarantine("runner identity is missing, ambiguous, or duplicated")
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
    manifest = _availability_manifest(
        evidence=evidence,
        schema=schema,
        fields=fields,
        runner_ids=tuple(runners),
        evidence_checksum=expected_evidence_checksum,
        expected_feature_cutoff_at=expected_feature_cutoff_at,
        raw_evidence_reader=raw_evidence_reader,
    )
    availability = (
        {entry.feature: entry for entry in manifest.entries} if manifest is not None else {}
    )
    rows: list[tuple[float, ...]] = []
    ids: list[str] = []
    explicit_missing: dict[str, tuple[str, ...]] = {}
    inapplicable: dict[str, tuple[str, ...]] = {}
    for runner_id in runners:
        values: list[float] = []
        missing: list[str] = []
        absent: list[str] = []
        for item in fields:
            name, semantics = item["name"], item.get("semantics")
            source_field = item.get("source_field", "runner_features")
            if source_field != "runner_features":
                raise FeatureQuarantine("per-runner features must come from runner_features")
            feature_values = runner_features[runner_id]
            if type(feature_values) is not dict or set(feature_values) != set(columns):
                raise FeatureQuarantine("runner_features must exactly match the Feature Contract")
            value = feature_values.get(name)
            status = availability.get(name)
            if status is not None and status.status is not FeatureAvailabilityStatus.READY_NOW:
                if semantics in {"identity-critical", "forecast-required"}:
                    raise FeatureQuarantine(
                        f"required feature is not READY_NOW: {name} ({status.status.value})"
                    )
                if semantics == "optional":
                    value = {"missing": True}
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
        manifest.checksum if manifest is not None else None,
        (
            {entry.feature: entry.status.value for entry in manifest.entries}
            if manifest is not None
            else {}
        ),
        {
            entry.feature: tuple(reason.value for reason in entry.blocking_reasons)
            for entry in (manifest.entries if manifest is not None else ())
        },
    )
    return DerivationResult(matrix, contract, report, manifest)
