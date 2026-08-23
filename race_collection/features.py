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
RESULT_FREE_FEATURE_NAMES = frozenset(
    {
        "speed",
        "form",
        "days_since_run",
        "novice",
        "box_number",
        "canonical_race_identity",
        "canonical_runner_identity",
        "race_card_context",
        "form_context",
        "speed_context",
        "venue",
        "distance",
        "recent_workload",
        "prior_official_weight",
        "pir_running_position",
        "typed_trial_state",
        "steward_state",
        "steward_veterinary_state",
        "lifecycle_age",
        "sportsbet_win_market",
    }
)

READY_FEATURE_EVIDENCE_FIELDS = {
    "canonical_race_identity": EvidenceField.RACE_IDENTITY.value,
    "canonical_runner_identity": EvidenceField.RUNNER_IDENTITY.value,
    "venue": EvidenceField.VENUE.value,
    "distance": EvidenceField.DISTANCE.value,
}

RESULT_FREE_SOURCE_METADATA_SHAPES = {
    "meeting metadata": (frozenset({"venue"}), frozenset({"meeting_code", "venue"})),
    "race metadata": (
        frozenset({"race_number"}),
        frozenset({"race_number", "distance_metres"}),
        frozenset({"race_number", "distance_metres", "race_time"}),
    ),
}


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


def _result_free_schema_error() -> FeatureQuarantine:
    return FeatureQuarantine("result-free evidence schema is invalid")


def required_ready_evidence_fields(fields: Mapping[str, Any]) -> frozenset[str]:
    """Return exact evidence fields required by READY_NOW manifest declarations."""
    availability = fields.get("feature_availability")
    if type(availability) is not dict:
        return frozenset()
    return frozenset(
        evidence_field
        for feature, evidence_field in READY_FEATURE_EVIDENCE_FIELDS.items()
        if type(availability.get(feature)) is dict
        and availability[feature].get("status") == FeatureAvailabilityStatus.READY_NOW.value
    )


def validate_result_free_source_metadata(value: Any, name: str) -> None:
    """Own the exact positive schema shared by capture and source admission."""
    shapes = RESULT_FREE_SOURCE_METADATA_SHAPES.get(name)
    if shapes is None or type(value) is not dict or frozenset(value) not in shapes:
        raise ValueError(f"{name} is outside the result-free positive schema")
    if name == "meeting metadata":
        if type(value["venue"]) is not str or not value["venue"].strip():
            raise ValueError("meeting venue must be known nonblank text")
        if "meeting_code" in value and (
            type(value["meeting_code"]) is not str or not value["meeting_code"].strip()
        ):
            raise ValueError("meeting code must be known nonblank text")
        return
    if type(value["race_number"]) is not int or value["race_number"] <= 0:
        raise ValueError("race number is invalid")
    if "distance_metres" in value and (
        type(value["distance_metres"]) is not int or value["distance_metres"] <= 0
    ):
        raise ValueError("race distance is invalid")
    if "race_time" in value and (
        type(value["race_time"]) is not str or not value["race_time"].strip()
    ):
        raise ValueError("race time must be known nonblank text")


def validate_feature_availability_manifest_document(value: Any) -> tuple[str, ...]:
    """Validate the exact immutable availability-manifest projection schema."""
    entry_fields = {
        "feature",
        "family",
        "semantics",
        "status",
        "source_name",
        "source_schema_version",
        "source_native_race_id",
        "source_native_runner_ids",
        "raw_checksum",
        "normalized_checksum",
        "provider_published_at",
        "collector_received_at",
        "feature_frozen_at",
        "completeness",
        "whole_race_coverage",
        "derivation_version",
        "blocking_reasons",
    }
    if (
        type(value) is not dict
        or set(value) != {"version", "race_id", "evidence_checksum", "entries"}
        or value.get("version") != FEATURE_AVAILABILITY_MANIFEST_VERSION
        or type(value.get("entries")) is not list
        or not value["entries"]
    ):
        raise ValueError("feature availability manifest schema is invalid")
    _text(value["race_id"], "manifest race identity")
    if _artifact_checksum(value["evidence_checksum"], "manifest evidence checksum") is None:
        raise ValueError("feature availability manifest schema is invalid")
    names: list[str] = []
    for entry in value["entries"]:
        if type(entry) is not dict or set(entry) != entry_fields:
            raise ValueError("feature availability manifest entry schema is invalid")
        feature = entry["feature"]
        if feature not in RESULT_FREE_FEATURE_NAMES:
            raise ValueError("feature availability manifest entry schema is invalid")
        _text(entry["family"], "manifest feature family")
        if entry["semantics"] not in {
            "identity-critical",
            "forecast-required",
            "optional",
            "inapplicable",
        }:
            raise ValueError("feature availability manifest entry schema is invalid")
        try:
            status = FeatureAvailabilityStatus(entry["status"])
            reasons = tuple(FeatureBlockingReason(reason) for reason in entry["blocking_reasons"])
        except (TypeError, ValueError) as error:
            raise ValueError("feature availability manifest entry schema is invalid") from error
        runner_ids = entry["source_native_runner_ids"]
        if (
            type(runner_ids) is not list
            or any(type(runner_id) is not str or not runner_id.strip() for runner_id in runner_ids)
            or len(set(runner_ids)) != len(runner_ids)
            or type(entry["blocking_reasons"]) is not list
            or len(set(reasons)) != len(reasons)
        ):
            raise ValueError("feature availability manifest entry schema is invalid")
        source_name = _optional_text(entry["source_name"], "manifest source name")
        source_schema = _optional_text(
            entry["source_schema_version"], "manifest source schema"
        )
        source_race = _optional_text(
            entry["source_native_race_id"], "manifest source-native race identity"
        )
        raw_checksum = _artifact_checksum(entry["raw_checksum"], "manifest raw checksum")
        normalized_checksum = _artifact_checksum(
            entry["normalized_checksum"], "manifest normalized checksum"
        )
        _optional_timestamp(entry["provider_published_at"], "manifest provider publication")
        _, receipt = _optional_timestamp(
            entry["collector_received_at"], "manifest collector receipt"
        )
        _timestamp(entry["feature_frozen_at"], "manifest feature freeze")
        _text(entry["derivation_version"], "manifest derivation version")
        if (
            entry["completeness"] not in {"COMPLETE", "INCOMPLETE", "UNKNOWN"}
            or type(entry["whole_race_coverage"]) is not bool
            or (
                status is FeatureAvailabilityStatus.READY_NOW
                and (
                    source_name is None
                    or source_schema is None
                    or source_race is None
                    or not runner_ids
                    or raw_checksum is None
                    or normalized_checksum is None
                    or receipt is None
                    or entry["completeness"] != "COMPLETE"
                    or entry["whole_race_coverage"] is not True
                    or reasons
                )
            )
            or (status is not FeatureAvailabilityStatus.READY_NOW and not reasons)
        ):
            raise ValueError("feature availability manifest entry schema is invalid")
        names.append(feature)
    if len(set(names)) != len(names):
        raise ValueError("feature availability manifest entries are duplicated")
    return tuple(names)


def _result_free_provenance_binding(
    item: Any, fields: Mapping[str, Any]
) -> tuple[str, str, ArtifactChecksum]:
    if type(item) is not dict or set(item) != {
        "field",
        "authority",
        "critical",
        "value",
        "source",
        "artifact_checksum",
    }:
        raise ValueError("result-free evidence provenance binding is invalid")
    try:
        field = EvidenceField(item["field"])
        checksum = ArtifactChecksum(item["artifact_checksum"])
    except (TypeError, ValueError) as error:
        raise ValueError("result-free evidence provenance binding is invalid") from error
    if (
        field is EvidenceField.RESULT_ORDER
        or field.value not in fields
        or type(item["critical"]) is not bool
        or item["critical"] is not field.critical
        or item["value"] != fields[field.value]
    ):
        raise ValueError("result-free evidence provenance binding is invalid")
    _text(item["authority"], "result-free evidence authority")
    source = _text(item["source"], "result-free evidence source")
    return field.value, source, checksum


def validate_result_free_provenance_bindings(
    fields: Mapping[str, Any],
    provenance: Any,
    *,
    raw_checksum: ArtifactChecksum,
    source_name: str,
) -> None:
    """Validate the one raw-source binding contract used by both admission paths."""
    required = {
        "runner_set",
        "runner_identity",
        "runner_features",
        *required_ready_evidence_fields(fields),
    }
    if type(provenance) is not list or not provenance:
        raise ValueError("result-free evidence provenance is incomplete")
    bound: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for item in provenance:
        field_name, source, checksum = _result_free_provenance_binding(item, fields)
        identity = (field_name, source)
        if identity in seen:
            raise ValueError("result-free evidence provenance binding is duplicated")
        seen.add(identity)
        if field_name in required and source == source_name and checksum == raw_checksum:
            bound.add(field_name)
    if bound != required:
        raise ValueError(
            "result-free evidence provenance is not bound to the preserved raw source bytes"
        )


def _positive_evidence_value(field_name: str, value: Any, runners: tuple[str, ...]) -> None:
    text_fields = {
        EvidenceField.IDENTITY.value,
        EvidenceField.RACE_IDENTITY.value,
        EvidenceField.VENUE.value,
        EvidenceField.GRADE.value,
    }
    timestamp_fields = {
        EvidenceField.SCHEDULED_JUMP.value,
        EvidenceField.ACTUAL_JUMP.value,
        EvidenceField.JUMP_TIME.value,
    }
    integer_fields = {
        EvidenceField.RACE_NUMBER.value,
        EvidenceField.FIELD_SIZE.value,
    }
    if field_name in text_fields:
        _text(value, field_name)
    elif field_name in timestamp_fields:
        _timestamp(value, field_name)
    elif field_name in integer_fields:
        if type(value) is not int or value <= 0:
            raise _result_free_schema_error()
    elif field_name == EvidenceField.DISTANCE.value:
        if (
            type(value) not in (int, float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise _result_free_schema_error()
    elif field_name == EvidenceField.RUNNER_SET.value:
        if type(value) is not list or tuple(value) != runners:
            raise _result_free_schema_error()
    elif field_name == EvidenceField.RUNNER_IDENTITY.value:
        if (
            type(value) is not dict
            or set(value) != set(runners)
            or any(value[runner] != "authoritative" for runner in runners)
        ):
            raise _result_free_schema_error()
    elif field_name == EvidenceField.RUNNER_FEATURES.value:
        if type(value) is not dict or set(value) != set(runners):
            raise _result_free_schema_error()
        for runner_values in value.values():
            if type(runner_values) is not dict or any(
                type(name) is not str
                or not name
                or not (
                    (
                        type(item) in (int, float)
                        and not isinstance(item, bool)
                        and math.isfinite(item)
                    )
                    or item == {"missing": True}
                    or item == {"inapplicable": True}
                )
                for name, item in runner_values.items()
            ):
                raise _result_free_schema_error()
    elif field_name in {EvidenceField.BOX.value}:
        if (
            type(value) is not dict
            or set(value) != set(runners)
            or any(type(item) is not int or item <= 0 for item in value.values())
        ):
            raise _result_free_schema_error()
    else:
        raise _result_free_schema_error()


def _validate_result_free_evidence(evidence: Mapping[str, Any]) -> None:
    historical_keys = {
        "schema_version",
        "normalization_version",
        "race_id",
        "historical_capture",
        "fields",
    }
    if set(evidence) == historical_keys:
        capture = evidence.get("historical_capture")
        if type(capture) is not dict or set(capture) != {
            "source",
            "source_record_id",
            "observed_at",
            "scheduled_jump_at",
            "identity_authority",
            "reconstructed",
        }:
            raise _result_free_schema_error()
        _text(capture["source"], "historical source")
        _text(capture["source_record_id"], "historical source record")
        _timestamp(capture["observed_at"], "historical observation")
        _timestamp(capture["scheduled_jump_at"], "historical scheduled jump")
        if (
            capture["identity_authority"] != "source-native"
            or capture["reconstructed"] is not False
        ):
            raise _result_free_schema_error()
        fields = evidence.get("fields")
        if type(fields) is not dict or set(fields) != {
            "runner_set",
            "runner_identity",
            "runner_features",
        }:
            raise _result_free_schema_error()
        raw_runners = fields.get("runner_set")
        if (
            type(raw_runners) is not list
            or not raw_runners
            or any(type(value) is not str or not value for value in raw_runners)
            or len(set(raw_runners)) != len(raw_runners)
        ):
            raise _result_free_schema_error()
        runners = tuple(raw_runners)
        for field_name, value in fields.items():
            _positive_evidence_value(field_name, value, runners)
        return
    if set(evidence) != {
        "schema_version",
        "normalization_version",
        "race_id",
        "fields",
        "field_provenance",
        "freeze",
    }:
        raise _result_free_schema_error()
    fields = evidence.get("fields")
    if type(fields) is not dict:
        raise _result_free_schema_error()
    result_free_fields = {
        field.value for field in EvidenceField if field is not EvidenceField.RESULT_ORDER
    }
    if not {"runner_set", "runner_identity", "runner_features"}.issubset(fields) or not set(
        fields
    ).issubset(result_free_fields | {"feature_availability"}):
        raise _result_free_schema_error()
    raw_runners = fields.get("runner_set")
    if (
        type(raw_runners) is not list
        or not raw_runners
        or any(type(value) is not str or not value for value in raw_runners)
        or len(set(raw_runners)) != len(raw_runners)
    ):
        raise _result_free_schema_error()
    runners = tuple(raw_runners)
    for field_name, value in fields.items():
        if field_name != "feature_availability":
            _positive_evidence_value(field_name, value, runners)
    freeze = evidence.get("freeze")
    if type(freeze) is not dict or set(freeze) != {"at", "authority", "odds_checksum"}:
        raise _result_free_schema_error()
    _timestamp(freeze["at"], "feature freeze")
    _text(freeze["authority"], "feature freeze authority")
    if _artifact_checksum(freeze["odds_checksum"], "feature freeze checksum") is None:
        raise _result_free_schema_error()
    provenance = evidence.get("field_provenance")
    if type(provenance) is not list:
        raise _result_free_schema_error()
    seen: set[tuple[str, str]] = set()
    for binding in provenance:
        try:
            field_name, source, _checksum_value = _result_free_provenance_binding(
                binding, fields
            )
        except ValueError as error:
            raise _result_free_schema_error() from error
        identity = (field_name, source)
        if identity in seen:
            raise _result_free_schema_error()
        seen.add(identity)
    bound_fields = {field_name for field_name, _source in seen}
    if not required_ready_evidence_fields(fields).issubset(bound_fields):
        raise FeatureQuarantine("READY_NOW evidence provenance is incomplete")


def _validate_feature_contract_schema(schema: Mapping[str, Any]) -> None:
    base_keys = {
        "bundle_id",
        "contract_version",
        "evidence_schema_version",
        "normalization_version",
        "fields",
    }
    manifest_core_keys = base_keys | {
        "availability_manifest_version",
        "source_contracts",
    }
    manifest_candidate_keys = manifest_core_keys | {"candidate_features"}
    if set(schema) not in (base_keys, manifest_core_keys, manifest_candidate_keys):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    if any(
        type(schema.get(key)) is not str or not schema[key].strip()
        for key in (
            "bundle_id",
            "contract_version",
            "evidence_schema_version",
            "normalization_version",
        )
    ):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    if "availability_manifest_version" in schema and (
        schema["availability_manifest_version"] != FEATURE_AVAILABILITY_MANIFEST_VERSION
    ):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    source_contracts = schema.get("source_contracts", {})
    if type(source_contracts) is not dict or any(
        type(source_name) is not str
        or not source_name.strip()
        or type(contract) is not dict
        or set(contract) != {"schema_versions", "provider_publication_time_exposed"}
        or type(contract["schema_versions"]) is not list
        or not contract["schema_versions"]
        or any(
            type(version) is not str or not version.strip()
            for version in contract["schema_versions"]
        )
        or len(set(contract["schema_versions"])) != len(contract["schema_versions"])
        or type(contract["provider_publication_time_exposed"]) is not bool
        for source_name, contract in source_contracts.items()
    ):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    fields = schema.get("fields")
    if type(fields) is not list or not fields:
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    for item in fields:
        if type(item) is not dict:
            raise FeatureQuarantine("result-free Feature Contract schema is invalid")
        required = {"name", "source_field", "semantics"}
        allowed = required | {"family"}
        if item.get("semantics") == "inapplicable":
            allowed.add("encoded_value")
        if not required.issubset(item) or not set(item).issubset(allowed):
            raise FeatureQuarantine("result-free Feature Contract schema is invalid")
        if item.get("name") not in RESULT_FREE_FEATURE_NAMES:
            raise FeatureQuarantine("result-free Feature Contract schema is invalid")
        if (
            type(item.get("source_field")) is not str
            or not item["source_field"].strip()
            or item.get("semantics")
            not in {"identity-critical", "forecast-required", "optional", "inapplicable"}
            or (
                "family" in item
                and (type(item["family"]) is not str or not item["family"].strip())
            )
        ):
            raise FeatureQuarantine("result-free Feature Contract schema is invalid")
        if "encoded_value" in item and (
            type(item["encoded_value"]) not in (int, float)
            or isinstance(item["encoded_value"], bool)
            or not math.isfinite(item["encoded_value"])
        ):
            raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    if len({item["name"] for item in fields}) != len(fields):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    candidates = schema.get("candidate_features", [])
    if type(candidates) is not list or any(
        type(item) is not dict
        or set(item) != {"name", "family", "semantics"}
        or item.get("name") not in RESULT_FREE_FEATURE_NAMES
        or type(item.get("family")) is not str
        or not item["family"].strip()
        or item.get("semantics") not in {"optional", "inapplicable"}
        for item in candidates
    ):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")
    declared_names = [item["name"] for item in fields] + [item["name"] for item in candidates]
    if len(set(declared_names)) != len(declared_names):
        raise FeatureQuarantine("result-free Feature Contract schema is invalid")


def _validate_missingness_policy(policy: Mapping[str, Any]) -> None:
    if set(policy) != {"bundle_id", "feature_contract_version", "imputation"}:
        raise FeatureQuarantine("result-free missingness policy schema is invalid")
    imputation = policy.get("imputation")
    if (
        type(policy.get("bundle_id")) is not str
        or not policy["bundle_id"].strip()
        or type(policy.get("feature_contract_version")) is not str
        or not policy["feature_contract_version"].strip()
        or type(imputation) is not dict
        or any(
            type(name) is not str
            or not name.strip()
            or name not in RESULT_FREE_FEATURE_NAMES
            or type(value) not in (int, float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            for name, value in imputation.items()
        )
    ):
        raise FeatureQuarantine("result-free missingness policy schema is invalid")


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
    raw_bindings: dict[tuple[str, str], ArtifactChecksum] = {}
    for binding in provenance:
        if (
            type(binding) is dict
            and type(binding.get("field")) is str
            and type(binding.get("source")) is str
        ):
            checksum = _artifact_checksum(
                binding.get("artifact_checksum"), "result-free raw checksum"
            )
            if checksum is None:
                continue
            binding_key = (binding["field"], binding["source"])
            prior = raw_bindings.setdefault(binding_key, checksum)
            if prior != checksum:
                raise FeatureQuarantine("result-free raw source binding conflicts")
    verified_raw_bindings: dict[tuple[str, str], ArtifactChecksum] = {}
    if raw_evidence_reader is not None:
        for binding_key, checksum in raw_bindings.items():
            try:
                raw_content = raw_evidence_reader(checksum)
            except (KeyError, OSError, ValueError):
                continue
            if type(raw_content) is not bytes or _checksum(raw_content) != checksum:
                raise FeatureQuarantine("raw evidence checksum mismatch")
            verified_raw_bindings[binding_key] = checksum

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
    feature_evidence_fields = {item["name"]: item["source_field"] for item in fields}
    feature_evidence_fields.update(READY_FEATURE_EVIDENCE_FIELDS)
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
            verified_raw_bindings.get((feature_evidence_fields.get(name), source_name))
            if source_name is not None
            else None
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
        normalized_checksum = (
            evidence_checksum
            if name in active_names
            or READY_FEATURE_EVIDENCE_FIELDS.get(name) in normalized_fields
            else None
        )
        if normalized_checksum is None:
            objective.add(FeatureBlockingReason.NORMALIZED_EVIDENCE_MISSING)
        stated = set(reasons)
        if not objective.issubset(stated) or any(
            reason not in objective and reason not in policy_blockers for reason in stated
        ):
            raise FeatureQuarantine(f"{name} blockers disagree with sealed evidence")
        if status is FeatureAvailabilityStatus.READY_NOW:
            valid_status = normalized_checksum is not None and not stated
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
    validate_feature_availability_manifest_document(document)
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
    _validate_result_free_evidence(evidence)
    _validate_feature_contract_schema(schema)
    _validate_missingness_policy(policy)
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
