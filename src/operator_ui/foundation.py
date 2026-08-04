"""Finite, fail-closed evidence primitives for the Operator UI.

This module deliberately knows nothing about collectors, predictions, HTTP,
authentication, or the runtime.  Server configuration owns every filesystem
locator and every claim.  Callers may select only a symbolic key.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import quote


class EvidenceStatus(str, Enum):
    AVAILABLE_FRESH = "AVAILABLE/FRESH"
    STALE = "STALE"
    UNAVAILABLE_DATA_MISSING = "UNAVAILABLE/DATA_MISSING"
    INVALID_INTEGRITY_FAILED = "INVALID/INTEGRITY_FAILED"
    DIVERGENT = "DIVERGENT"


class Availability(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNREADABLE = "unreadable"
    ERROR = "error"


class Integrity(str, Enum):
    VALID = "valid"
    FAILED = "failed"
    UNKNOWN = "unknown"


class Freshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class HistoricalClaim(str, Enum):
    RUN = "historical_run"
    SLICE = "historical_slice"


class JsonSerializationPolicy(str, Enum):
    COMPACT_CANONICAL = "compact_canonical"
    PRODUCER_PRETTY_SORTED = "producer_pretty_sorted"


class TimestampSyntax(str, Enum):
    CANONICAL_TERMINAL_Z = "canonical_terminal_z"
    AWARE_ISO8601 = "aware_iso8601"


_HISTORICAL_CLAIMS: Mapping[HistoricalClaim, str] = MappingProxyType(
    {
        HistoricalClaim.RUN: (
            "Verified result of the identified historical run only."
        ),
        HistoricalClaim.SLICE: (
            "Verified result of the identified historical slice only."
        ),
    }
)


def status_for(
    availability: Availability,
    integrity: Integrity,
    freshness: Freshness,
    *,
    conflict: bool = False,
) -> EvidenceStatus:
    """Map the four evidence axes to the exhaustive contract vocabulary."""
    if integrity is Integrity.FAILED:
        return EvidenceStatus.INVALID_INTEGRITY_FAILED
    if availability is not Availability.PRESENT:
        return EvidenceStatus.UNAVAILABLE_DATA_MISSING
    if integrity is not Integrity.VALID:
        return EvidenceStatus.INVALID_INTEGRITY_FAILED
    if conflict:
        return EvidenceStatus.DIVERGENT
    if freshness is Freshness.STALE:
        return EvidenceStatus.STALE
    if freshness is not Freshness.FRESH:
        return EvidenceStatus.UNAVAILABLE_DATA_MISSING
    return EvidenceStatus.AVAILABLE_FRESH


@dataclass(frozen=True)
class Policy:
    name: str
    mode: str
    maximum_age_seconds: float | None


# Exhaustive names from CONTRACTS.md section 3.  Only fixed-age policies are
# evaluated here; adapter-owned dynamic/aggregate semantics fail closed.
_POLICIES = (
    Policy("P-DEPLOY-60", "fixed", 60),
    Policy("P-COLLECTOR-FULL-DYNAMIC", "adapter", None),
    Policy("P-COLLECTOR-ODDS-DYNAMIC", "adapter", None),
    Policy("P-COLLECTOR-AGGREGATE", "adapter", None),
    Policy("P-UPCOMING-300-PREJUMP", "adapter", 300),
    Policy("P-CURRENT-INDEX-1200", "adapter", 1200),
    Policy("P-CATALOG-60", "fixed", 60),
    Policy("P-BUNDLE-LIST-60", "fixed", 60),
    Policy("P-JOB-5-DEADLINE", "adapter", 5),
    Policy("P-REPORT-24H", "fixed", 86400),
    Policy("P-OPS-5", "adapter", 5),
    Policy("P-IMMUTABLE-HISTORICAL", "historical", None),
)
POLICIES: Mapping[str, Policy] = MappingProxyType(
    {policy.name: policy for policy in _POLICIES}
)


@dataclass(frozen=True)
class ReferenceHash:
    name: str
    json_field: str
    expected_sha256: str


@dataclass(frozen=True)
class JsonSource:
    schema_field: str | None
    schema_value: str | None
    top_level_fields: tuple[str, ...]
    time_field: str | None
    time_role: str = "generated_at"
    reference_hashes: tuple[ReferenceHash, ...] = ()
    identity_fields: tuple[str, ...] = ()
    max_depth: int = 12
    max_items: int = 1024
    max_string_bytes: int = 4096
    serialization_policy: JsonSerializationPolicy = (
        JsonSerializationPolicy.COMPACT_CANONICAL
    )
    timestamp_syntax: TimestampSyntax = TimestampSyntax.CANONICAL_TERMINAL_Z
    authority_observed_at: str | None = None

    def __post_init__(self) -> None:
        try:
            policy = JsonSerializationPolicy(self.serialization_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported JSON serialization policy") from exc
        object.__setattr__(self, "serialization_policy", policy)
        try:
            timestamp_syntax = TimestampSyntax(self.timestamp_syntax)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported timestamp syntax") from exc
        object.__setattr__(self, "timestamp_syntax", timestamp_syntax)
        if self.authority_observed_at is not None:
            _parse_utc(self.authority_observed_at, timestamp_syntax)


@dataclass(frozen=True)
class SourceConfig:
    locator: Path
    allowlisted_root: Path
    source_kind: str
    source_identity: str
    source_locator: str
    policy: str
    supported_claim: str | HistoricalClaim
    json: JsonSource
    max_bytes: int = 1_048_576
    max_envelope_bytes: int = 32_768
    expected_sha256: str | None = None


@dataclass(frozen=True)
class RawSourceConfig:
    """One fixed, bounded non-JSON file observed by symbolic server key."""

    locator: Path
    allowlisted_root: Path
    source_kind: str
    source_identity: str
    source_locator: str
    policy: str
    supported_claim: str
    max_bytes: int = 16_777_216
    max_envelope_bytes: int = 32_768
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    digest_only: bool = False


@dataclass(frozen=True)
class ReadOnlyDatabase:
    locator: Path
    allowlisted_root: Path


_ENVELOPE_TOKEN = object()


@dataclass(frozen=True, init=False)
class EvidenceEnvelope:
    source_kind: str
    source_identity: str
    content_sha256: str | None
    source_locator: str
    source_at: str | None
    generated_at: str | None
    observed_at: str | None
    server_observed_at: str
    age_seconds: float | None
    freshness_policy: str
    availability: str
    schema_integrity: str
    reference_hashes: tuple[tuple[str, str], ...]
    evidence_identity: tuple[tuple[str, str], ...] | None
    status: str
    supported_claim: str

    def __new__(cls, token: object = None) -> EvidenceEnvelope:
        if token is not _ENVELOPE_TOKEN:
            raise TypeError("evidence envelopes are created by server observations")
        return super().__new__(cls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "source_identity": self.source_identity,
            "content_sha256": self.content_sha256,
            "source_locator": self.source_locator,
            "source_at": self.source_at,
            "generated_at": self.generated_at,
            "observed_at": self.observed_at,
            "server_observed_at": self.server_observed_at,
            "age_seconds": self.age_seconds,
            "freshness_policy": self.freshness_policy,
            "availability": self.availability,
            "schema_integrity": self.schema_integrity,
            "reference_hashes": dict(self.reference_hashes),
            "evidence_identity": (
                dict(self.evidence_identity)
                if self.evidence_identity is not None
                else None
            ),
            "status": self.status,
            "supported_claim": self.supported_claim,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _new_envelope(**values: Any) -> EvidenceEnvelope:
    envelope = EvidenceEnvelope(_ENVELOPE_TOKEN)
    for field, value in values.items():
        object.__setattr__(envelope, field, value)
    return envelope


class _InvalidEvidence(ValueError):
    pass


class _OversizeEvidence(_InvalidEvidence):
    pass


class _PathChanged(_InvalidEvidence):
    pass


@dataclass(frozen=True)
class _PathIdentity:
    path: Path
    identity: tuple[int, int, int] | None


@dataclass(frozen=True)
class _BoundPath:
    root: Path
    locator: Path
    components: tuple[_PathIdentity, ...]


@dataclass(frozen=True)
class _OpenedBoundFile:
    descriptors: tuple[int, ...]
    opened: os.stat_result

    @property
    def root_descriptor(self) -> int:
        return self.descriptors[0]

    @property
    def descriptor(self) -> int:
        return self.descriptors[-1]


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
    )


def _bind_path(locator: Path, root: Path) -> _BoundPath:
    paths = [root]
    current = root
    for part in locator.relative_to(root).parts:
        current /= part
        paths.append(current)
    components: list[_PathIdentity] = []
    missing = False
    for path in paths:
        if missing:
            components.append(_PathIdentity(path, None))
            continue
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            missing = True
            components.append(_PathIdentity(path, None))
        else:
            components.append(_PathIdentity(path, _stat_identity(metadata)))
    return _BoundPath(root, locator, tuple(components))


def _verify_path_binding(binding: _BoundPath) -> None:
    for component in binding.components:
        try:
            metadata = os.lstat(component.path)
        except FileNotFoundError:
            if component.identity is not None:
                continue
        else:
            if (
                component.identity is None
                or _stat_identity(metadata) != component.identity
            ):
                raise _PathChanged(
                    "configured path identity changed after construction"
                )


def _require_bounded_string(value: str, name: str, maximum: int = 512) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{name} must be a non-empty bounded string")


def _validate_hash(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _canonical_root(root: Path) -> Path:
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("allowlisted root must be absolute")
    canonical = root.resolve(strict=True)
    if canonical != root or not canonical.is_dir() or canonical.is_symlink():
        raise ValueError("allowlisted root must be a canonical directory")
    return canonical


def _canonical_locator(locator: Path, root: Path) -> Path:
    if not isinstance(locator, Path) or not locator.is_absolute():
        raise ValueError("locator must be absolute")
    resolved = locator.resolve(strict=False)
    if resolved != locator:
        raise ValueError("locator must be canonical and contain no symlink")
    try:
        locator.relative_to(root)
    except ValueError as exc:
        raise ValueError("locator is outside its allowlisted root") from exc
    current = root
    for part in locator.relative_to(root).parts:
        current = current / part
        try:
            if current.is_symlink():
                raise ValueError("locator traverses a symlink")
        except OSError as exc:
            raise ValueError("locator cannot be validated") from exc
    return locator


def _validate_source(config: SourceConfig) -> SourceConfig:
    root = _canonical_root(config.allowlisted_root)
    locator = _canonical_locator(config.locator, root)
    for name, value in (
        ("source_kind", config.source_kind),
        ("source_identity", config.source_identity),
        ("source_locator", config.source_locator),
        ("policy", config.policy),
    ):
        _require_bounded_string(value, name)
    if config.source_locator.startswith("/") or "://" in config.source_locator:
        raise ValueError("source_locator must be symbolic, not a path or URI")
    if config.policy not in POLICIES:
        raise ValueError("unsupported freshness policy")
    policy = POLICIES[config.policy]
    if policy.mode == "historical":
        if not isinstance(config.supported_claim, HistoricalClaim):
            raise ValueError(
                "historical policy requires a finite run- or slice-bound claim"
            )
        supported_claim: str | HistoricalClaim = _HISTORICAL_CLAIMS[
            config.supported_claim
        ]
    else:
        if isinstance(config.supported_claim, HistoricalClaim) or not isinstance(
            config.supported_claim, str
        ):
            raise ValueError(
                "structured historical claims require historical policy"
            )
        _require_bounded_string(config.supported_claim, "supported_claim")
        supported_claim = config.supported_claim
    if (
        not isinstance(config.max_bytes, int)
        or config.max_bytes <= 0
        or not isinstance(config.max_envelope_bytes, int)
        or config.max_envelope_bytes <= 0
    ):
        raise ValueError("limits must be finite positive integers")
    schema = config.json
    if schema.time_role not in {"source_at", "generated_at", "observed_at"}:
        raise ValueError("unsupported time role")
    if (schema.schema_field is None) != (schema.schema_value is None):
        raise ValueError("schema field and value must both be present or absent")
    observed_without_producer_time = (
        schema.time_field is None
        and schema.schema_value in {
            "race_evidence_inventory_output_manifest_v1",
            "on_demand_prediction_config_catalog_v1",
        }
    )
    if schema.authority_observed_at is not None and (
        schema.time_field is not None
        or schema.schema_value != "on_demand_prediction_config_catalog_v1"
    ):
        raise ValueError("authority observation time is unsupported for this source")
    if (schema.schema_field is None or schema.time_field is None) and policy.mode != "adapter" and not observed_without_producer_time:
        raise ValueError("only adapter-owned or approved observation-bound evidence may omit producer time")
    for value in (
        *((schema.schema_field, schema.schema_value) if schema.schema_field else ()),
        *((schema.time_field,) if schema.time_field else ()),
        *schema.top_level_fields,
        *schema.identity_fields,
    ):
        _require_bounded_string(value, "schema value")
    if (
        not isinstance(schema.max_depth, int)
        or schema.max_depth <= 0
        or not isinstance(schema.max_items, int)
        or schema.max_items <= 0
        or not isinstance(schema.max_string_bytes, int)
        or schema.max_string_bytes <= 0
    ):
        raise ValueError("JSON limits must be finite positive integers")
    if (
        len(set(schema.top_level_fields)) != len(schema.top_level_fields)
        or len(schema.top_level_fields) > schema.max_items
        or len(schema.reference_hashes) > schema.max_items
        or len(schema.identity_fields) > schema.max_items
        or len({item.name for item in schema.reference_hashes})
        != len(schema.reference_hashes)
        or len({item.json_field for item in schema.reference_hashes})
        != len(schema.reference_hashes)
        or len(set(schema.identity_fields)) != len(schema.identity_fields)
        or (schema.schema_field is not None and schema.schema_field not in schema.top_level_fields)
        or (schema.time_field is not None and schema.time_field not in schema.top_level_fields)
        or not set(schema.identity_fields).issubset(schema.top_level_fields)
    ):
        raise ValueError("top-level schema is not exact")
    if config.expected_sha256 is not None:
        _validate_hash(config.expected_sha256, "expected_sha256")
    for reference in schema.reference_hashes:
        _require_bounded_string(reference.name, "reference name")
        _require_bounded_string(reference.json_field, "reference field")
        _validate_hash(reference.expected_sha256, "reference hash")
        if reference.json_field not in schema.top_level_fields:
            raise ValueError("reference field is absent from top-level schema")
    return SourceConfig(
        locator=locator,
        allowlisted_root=root,
        source_kind=config.source_kind,
        source_identity=config.source_identity,
        source_locator=config.source_locator,
        policy=config.policy,
        supported_claim=supported_claim,
        json=config.json,
        max_bytes=config.max_bytes,
        max_envelope_bytes=config.max_envelope_bytes,
        expected_sha256=config.expected_sha256,
    )


def _reject_constant(value: str) -> None:
    raise _InvalidEvidence(f"non-finite JSON number: {value}")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidEvidence("duplicate JSON object key")
        result[key] = value
    return result


def _bound_json(
    value: Any, *, depth: int, schema: JsonSource, count: list[int]
) -> None:
    if depth > schema.max_depth:
        raise _OversizeEvidence("JSON nesting exceeds configured maximum")
    count[0] += 1
    if count[0] > schema.max_items:
        raise _OversizeEvidence("JSON items exceed configured maximum")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > schema.max_string_bytes:
            raise _OversizeEvidence("JSON string exceeds configured maximum")
    elif isinstance(value, dict):
        for key, child in value.items():
            if len(key.encode("utf-8")) > schema.max_string_bytes:
                raise _OversizeEvidence("JSON key exceeds configured maximum")
            _bound_json(child, depth=depth + 1, schema=schema, count=count)
    elif isinstance(value, list):
        for child in value:
            _bound_json(child, depth=depth + 1, schema=schema, count=count)
    elif isinstance(value, float) and not math.isfinite(value):
        raise _InvalidEvidence("non-finite JSON number")


_AWARE_ISO8601 = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_MAX_TIMESTAMP_BYTES = 40


def _parse_utc(
    value: Any,
    syntax: TimestampSyntax = TimestampSyntax.CANONICAL_TERMINAL_Z,
) -> datetime:
    if not isinstance(value, str):
        raise _InvalidEvidence("timestamp must be an explicit UTC string")
    if len(value.encode("utf-8")) > _MAX_TIMESTAMP_BYTES:
        raise _InvalidEvidence("timestamp is unbounded")
    if syntax is TimestampSyntax.CANONICAL_TERMINAL_Z and not value.endswith("Z"):
        raise _InvalidEvidence("timestamp must use the canonical terminal-Z UTC form")
    if syntax is TimestampSyntax.AWARE_ISO8601 and _AWARE_ISO8601.fullmatch(value) is None:
        raise _InvalidEvidence("timestamp must be a timezone-aware ISO-8601 instant")
    try:
        parsed = datetime.fromisoformat(
            f"{value[:-1]}+00:00" if value.endswith("Z") else value
        )
    except ValueError as exc:
        raise _InvalidEvidence("timestamp is malformed") from exc
    if parsed.tzinfo is None:
        raise _InvalidEvidence("timestamp must include a timezone")
    if syntax is TimestampSyntax.CANONICAL_TERMINAL_Z and parsed.utcoffset() != timedelta(0):
        raise _InvalidEvidence("timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("server clock must return an aware UTC time")
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _freshness(policy: Policy, age: float) -> Freshness:
    if not math.isfinite(age) or age < 0:
        return Freshness.UNKNOWN
    if policy.mode == "fixed":
        assert policy.maximum_age_seconds is not None
        return (
            Freshness.FRESH
            if age <= policy.maximum_age_seconds
            else Freshness.STALE
        )
    # Historical claims have no wall-clock expiry. Their explicitly historical
    # supported claim, not freshness, prevents any inference of current health.
    if policy.mode == "historical":
        return Freshness.FRESH
    return Freshness.UNKNOWN


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _descriptor_path(descriptor: int) -> Path:
    return Path(os.readlink(f"/proc/self/fd/{descriptor}"))


def _open_bound_file(binding: _BoundPath) -> _OpenedBoundFile:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptors: list[int] = []
    try:
        for index, component in enumerate(binding.components[:-1]):
            descriptor = (
                os.open(component.path, directory_flags)
                if index == 0
                else os.open(
                    component.path.name,
                    directory_flags,
                    dir_fd=descriptors[-1],
                )
            )
            descriptors.append(descriptor)
            opened_component = os.fstat(descriptor)
            if (
                component.identity is None
                or _stat_identity(opened_component) != component.identity
                or _descriptor_path(descriptor) != component.path
            ):
                raise _PathChanged(
                    "configured path identity changed while opening"
                )
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_component = binding.components[-1]
        descriptor = os.open(
            file_component.path.name,
            file_flags,
            dir_fd=descriptors[-1],
        )
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        if (
            file_component.identity is None
            or _stat_identity(opened) != file_component.identity
            or not stat.S_ISREG(opened.st_mode)
            or _descriptor_path(descriptor) != file_component.path
        ):
            raise _PathChanged("source escaped its configured path or root")
        return _OpenedBoundFile(tuple(descriptors), opened)
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _verify_bound_file(
    binding: _BoundPath,
    opened_file: _OpenedBoundFile,
) -> os.stat_result:
    for component, descriptor in zip(
        binding.components, opened_file.descriptors, strict=True
    ):
        after_descriptor = os.fstat(descriptor)
        try:
            after_path = os.lstat(component.path)
        except FileNotFoundError as exc:
            raise _PathChanged("source changed during observation") from exc
        if (
            component.identity is None
            or _stat_identity(after_descriptor) != component.identity
            or _stat_identity(after_path) != component.identity
            or _descriptor_path(descriptor) != component.path
        ):
            raise _PathChanged("source changed during observation")
    after_descriptor = os.fstat(opened_file.descriptor)
    if not _same_file(opened_file.opened, after_descriptor):
        raise _PathChanged("source changed during observation")
    return after_descriptor


def _read_regular_file(binding: _BoundPath, maximum: int) -> bytes:
    path = binding.locator
    _verify_path_binding(binding)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode):
        raise _InvalidEvidence("source is not a regular file")
    if before.st_size > maximum:
        raise _OversizeEvidence("source exceeds configured maximum")
    opened_file = _open_bound_file(binding)
    descriptor = opened_file.descriptor
    opened = opened_file.opened
    try:
        if not _same_file(before, opened):
            raise _PathChanged("source changed while opening")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise _OversizeEvidence("source grew beyond configured maximum")
        after_descriptor = _verify_bound_file(binding, opened_file)
        if (
            opened.st_size != after_descriptor.st_size
            or total != after_descriptor.st_size
        ):
            raise _PathChanged("source changed during observation")
        return b"".join(chunks)
    finally:
        for opened_descriptor in reversed(opened_file.descriptors):
            os.close(opened_descriptor)


def _digest_regular_file(binding: _BoundPath, maximum: int) -> tuple[str, int]:
    """Authenticate a bounded file using fixed chunks and a fixed deadline."""
    started = time.monotonic()
    path = binding.locator
    _verify_path_binding(binding)
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode): raise _InvalidEvidence("source is not a regular file")
    if before.st_size > maximum: raise _OversizeEvidence("source exceeds configured maximum")
    opened_file = _open_bound_file(binding); descriptor = opened_file.descriptor
    try:
        if not _same_file(before, opened_file.opened): raise _PathChanged("source changed while opening")
        digest = hashlib.sha256(); total = 0
        while True:
            if time.monotonic() - started > 30.0: raise _InvalidEvidence("source authentication timed out")
            chunk = os.read(descriptor, 65_536)
            if not chunk: break
            total += len(chunk)
            if total > maximum: raise _OversizeEvidence("source grew beyond configured maximum")
            digest.update(chunk)
        if time.monotonic() - started > 30.0: raise _InvalidEvidence("source authentication timed out")
        after = _verify_bound_file(binding, opened_file)
        if opened_file.opened.st_size != after.st_size or total != after.st_size:
            raise _PathChanged("source changed during observation")
        return digest.hexdigest(), total
    finally:
        for opened_descriptor in reversed(opened_file.descriptors): os.close(opened_descriptor)


class OperatorEvidenceReader:
    """Observe configured JSON evidence using symbolic keys only."""

    def __init__(
        self,
        sources: Mapping[str, SourceConfig],
        *,
        raw_sources: Mapping[str, RawSourceConfig] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not sources:
            raise ValueError("at least one source is required")
        validated: dict[str, SourceConfig] = {}
        bindings: dict[str, _BoundPath] = {}
        for key, config in sources.items():
            _require_bounded_string(key, "source key", 128)
            validated[key] = _validate_source(config)
            bindings[key] = _bind_path(
                validated[key].locator, validated[key].allowlisted_root
            )
        self._sources = MappingProxyType(validated)
        self._bindings = MappingProxyType(bindings)
        raw_validated: dict[str, RawSourceConfig] = {}
        raw_bindings: dict[str, _BoundPath] = {}
        for key, config in (raw_sources or {}).items():
            _require_bounded_string(key, "raw source key", 128)
            if key in validated:
                raise ValueError("source keys must be unique")
            root = _canonical_root(config.allowlisted_root)
            locator = _canonical_locator(config.locator, root)
            for value, name in (
                (config.source_kind, "source_kind"),
                (config.source_identity, "source_identity"),
                (config.source_locator, "source_locator"),
                (config.supported_claim, "supported_claim"),
            ):
                _require_bounded_string(value, name)
            if config.source_locator.startswith("/") or "://" in config.source_locator:
                raise ValueError("source_locator must be symbolic")
            if config.policy not in POLICIES:
                raise ValueError("unsupported freshness policy")
            if type(config.max_bytes) is not int or config.max_bytes <= 0:
                raise ValueError("raw source limit must be positive")
            if type(config.digest_only) is not bool:
                raise ValueError("raw source digest-only policy must be boolean")
            if config.digest_only:
                if config.max_bytes != 64 * 1024 * 1024 or type(config.expected_bytes) is not int or config.expected_bytes < 0 or config.expected_bytes > config.max_bytes:
                    raise ValueError("digest-only source policy is invalid")
            elif config.expected_bytes is not None:
                raise ValueError("byte count is supported only for digest-only sources")
            if config.expected_sha256 is not None:
                _validate_hash(config.expected_sha256, "expected_sha256")
            normalized = RawSourceConfig(
                locator=locator, allowlisted_root=root,
                source_kind=config.source_kind, source_identity=config.source_identity,
                source_locator=config.source_locator, policy=config.policy,
                supported_claim=config.supported_claim, max_bytes=config.max_bytes,
                max_envelope_bytes=config.max_envelope_bytes,
                expected_sha256=config.expected_sha256,
                expected_bytes=config.expected_bytes, digest_only=config.digest_only,
            )
            raw_validated[key] = normalized
            raw_bindings[key] = _bind_path(locator, root)
        self._raw_sources = MappingProxyType(raw_validated)
        self._raw_bindings = MappingProxyType(raw_bindings)
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def policies(self) -> Mapping[str, Policy]:
        return POLICIES

    def read(self, source_key: str) -> EvidenceEnvelope:
        envelope, _ = self.read_payload(source_key)
        return envelope

    def read_payload(
        self, source_key: str
    ) -> tuple[EvidenceEnvelope, Mapping[str, Any] | None]:
        """Observe one configured source and return verified JSON when usable.

        The symbolic-key-only boundary is identical to :meth:`read`.  Payload
        bytes are returned only after the complete bounded canonical read; a
        failed observation never leaks partially parsed data.
        """
        try:
            config = self._sources[source_key]
        except (KeyError, TypeError) as exc:
            raise KeyError("unknown evidence source key") from exc

        observed = self._clock()
        observed_text = _utc_text(observed)
        try:
            raw = _read_regular_file(
                self._bindings[source_key], config.max_bytes
            )
        except FileNotFoundError:
            return self._failure(
                config, observed_text, Availability.MISSING, Integrity.UNKNOWN
            ), None
        except PermissionError:
            return self._failure(
                config, observed_text, Availability.UNREADABLE, Integrity.UNKNOWN
            ), None
        except _InvalidEvidence:
            return self._failure(
                config, observed_text, Availability.PRESENT, Integrity.FAILED
            ), None
        except OSError:
            return self._failure(
                config, observed_text, Availability.ERROR, Integrity.FAILED
            ), None

        digest = hashlib.sha256(raw).hexdigest()
        conflict = (
            config.expected_sha256 is not None
            and digest != config.expected_sha256
        )
        try:
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_pairs,
                parse_constant=_reject_constant,
            )
            if not isinstance(payload, dict):
                raise _InvalidEvidence("top-level JSON must be an object")
            _bound_json(payload, depth=1, schema=config.json, count=[0])
            if set(payload) != set(config.json.top_level_fields):
                raise _InvalidEvidence("unexpected top-level schema")
            if (
                config.json.schema_field is not None
                and payload.get(config.json.schema_field) != config.json.schema_value
            ):
                raise _InvalidEvidence("schema identity mismatch")
            if (
                config.json.serialization_policy
                is JsonSerializationPolicy.COMPACT_CANONICAL
            ):
                canonical = json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            else:
                canonical = (
                    json.dumps(payload, indent=2, sort_keys=True, default=str)
                    + "\n"
                ).encode("utf-8")
            if canonical != raw:
                raise _InvalidEvidence("JSON bytes are not canonical")
            if config.json.time_field is None and config.json.authority_observed_at is None:
                age = math.nan
                freshness = Freshness.UNKNOWN
            else:
                evidence_time = _parse_utc(
                    payload[config.json.time_field] if config.json.time_field is not None
                    else config.json.authority_observed_at,
                    config.json.timestamp_syntax,
                )
                age = (observed.astimezone(timezone.utc) - evidence_time).total_seconds()
                freshness = _freshness(POLICIES[config.policy], age)
            references: list[tuple[str, str]] = []
            for reference in config.json.reference_hashes:
                actual = payload.get(reference.json_field)
                if not isinstance(actual, str):
                    raise _InvalidEvidence("reference hash is missing")
                _validate_hash(actual, "observed reference hash")
                references.append((reference.name, actual))
                conflict = conflict or actual != reference.expected_sha256
            identity: list[tuple[str, str]] = []
            for field in config.json.identity_fields:
                value = payload.get(field)
                if not isinstance(value, str) or not value:
                    raise _InvalidEvidence("evidence identity is invalid")
                if len(value.encode("utf-8")) > config.json.max_string_bytes:
                    raise _OversizeEvidence("evidence identity is unbounded")
                identity.append((field, value))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _InvalidEvidence,
            ValueError,
        ):
            return self._failure(
                config,
                observed_text,
                Availability.PRESENT,
                Integrity.FAILED,
                content_hash=digest,
            ), None

        role_times: dict[str, str | None] = {
            "source_at": None,
            "generated_at": None,
            "observed_at": None,
        }
        if config.json.time_field is not None:
            role_times[config.json.time_role] = payload[config.json.time_field]
        elif config.json.authority_observed_at is not None:
            role_times["observed_at"] = config.json.authority_observed_at
        integrity = Integrity.VALID
        status = status_for(
            Availability.PRESENT, integrity, freshness, conflict=conflict
        )
        envelope = _new_envelope(
            source_kind=config.source_kind,
            source_identity=config.source_identity,
            content_sha256=digest,
            source_locator=config.source_locator,
            source_at=role_times["source_at"],
            generated_at=role_times["generated_at"],
            observed_at=role_times["observed_at"],
            server_observed_at=observed_text,
            age_seconds=age if math.isfinite(age) and age >= 0 else None,
            freshness_policy=config.policy,
            availability=Availability.PRESENT.value,
            schema_integrity=integrity.value,
            reference_hashes=tuple(sorted(references)),
            evidence_identity=tuple(identity) or None,
            status=status.value,
            supported_claim=config.supported_claim,
        )
        self._check_envelope_size(envelope, config.max_envelope_bytes)
        return envelope, MappingProxyType(payload)

    def read_raw(self, source_key: str) -> tuple[EvidenceEnvelope, bytes | None]:
        """Read one configured regular file without interpreting its bytes."""
        try:
            config = self._raw_sources[source_key]
            binding = self._raw_bindings[source_key]
        except (KeyError, TypeError) as exc:
            raise KeyError("unknown raw evidence source key") from exc
        observed = self._clock()
        observed_text = _utc_text(observed)
        try:
            if config.digest_only:
                digest, byte_count = _digest_regular_file(binding, config.max_bytes)
                raw = None
            else:
                raw = _read_regular_file(binding, config.max_bytes)
                digest, byte_count = hashlib.sha256(raw).hexdigest(), len(raw)
        except FileNotFoundError:
            return self._failure(config, observed_text, Availability.MISSING, Integrity.UNKNOWN), None
        except PermissionError:
            return self._failure(config, observed_text, Availability.UNREADABLE, Integrity.UNKNOWN), None
        except (_InvalidEvidence, OSError):
            return self._failure(config, observed_text, Availability.ERROR, Integrity.FAILED), None
        conflict = ((config.expected_sha256 is not None and digest != config.expected_sha256) or (config.expected_bytes is not None and byte_count != config.expected_bytes))
        envelope = _new_envelope(
            source_kind=config.source_kind, source_identity=config.source_identity,
            content_sha256=digest, source_locator=config.source_locator,
            source_at=None, generated_at=None, observed_at=observed_text,
            server_observed_at=observed_text, age_seconds=0.0,
            freshness_policy=config.policy, availability=Availability.PRESENT.value,
            schema_integrity=Integrity.VALID.value, reference_hashes=(),
            evidence_identity=None,
            status=(EvidenceStatus.DIVERGENT if conflict else EvidenceStatus.AVAILABLE_FRESH).value,
            supported_claim=config.supported_claim,
        )
        self._check_envelope_size(envelope, config.max_envelope_bytes)
        return envelope, raw

    def read_raw_authenticated(self, source_key: str) -> tuple[EvidenceEnvelope, bytes | None, int | None]:
        """Return authenticated bytes when retained, plus the verified byte count."""
        envelope, raw = self.read_raw(source_key)
        if envelope.content_sha256 is None: return envelope, raw, None
        config = self._raw_sources[source_key]
        return envelope, raw, config.expected_bytes if config.digest_only and envelope.status != EvidenceStatus.DIVERGENT.value else len(raw) if raw is not None else None

    def read_verified_payload(
        self, source_key: str, expected_source_locator: str
    ) -> tuple[EvidenceEnvelope, Mapping[str, Any] | None]:
        """Read a fixed server source only when its public locator is expected.

        The expected locator is a producer-authored safe relative identity,
        compared only to the server-owned configured path relative to its
        fixed root.  The public symbolic ``source_locator`` remains unchanged.
        """
        try:
            config = self._sources[source_key]
        except (KeyError, TypeError) as exc:
            raise KeyError("unknown evidence source key") from exc
        _require_bounded_string(
            expected_source_locator, "expected source locator", 4096
        )
        if (
            expected_source_locator.startswith("/")
            or "\\" in expected_source_locator
            or any(
                part in {"", ".", ".."}
                for part in expected_source_locator.split("/")
            )
        ):
            raise _PathChanged("expected producer locator is unsafe")
        try:
            configured_relative = config.locator.relative_to(
                config.allowlisted_root
            ).as_posix()
        except ValueError as exc:
            raise _PathChanged("configured source is outside its fixed root") from exc
        if configured_relative != expected_source_locator:
            raise _PathChanged("configured source locator does not match producer identity")
        return self.read_payload(source_key)

    @staticmethod
    def _check_envelope_size(
        envelope: EvidenceEnvelope, maximum: int
    ) -> None:
        if len(envelope.to_json().encode("utf-8")) > maximum:
            raise _OversizeEvidence("serialized envelope exceeds configured maximum")

    def _failure(
        self,
        config: SourceConfig,
        observed_text: str,
        availability: Availability,
        integrity: Integrity,
        *,
        content_hash: str | None = None,
    ) -> EvidenceEnvelope:
        envelope = _new_envelope(
            source_kind=config.source_kind,
            source_identity=config.source_identity,
            content_sha256=content_hash,
            source_locator=config.source_locator,
            source_at=None,
            generated_at=None,
            observed_at=None,
            server_observed_at=observed_text,
            age_seconds=None,
            freshness_policy=config.policy,
            availability=availability.value,
            schema_integrity=integrity.value,
            reference_hashes=(),
            evidence_identity=None,
            status=status_for(
                availability, integrity, Freshness.UNKNOWN
            ).value,
            supported_claim=config.supported_claim,
        )
        self._check_envelope_size(envelope, config.max_envelope_bytes)
        return envelope


_SQLITE_WRITE_ACTIONS = frozenset(
    action
    for action in (
        getattr(sqlite3, "SQLITE_CREATE_INDEX", None),
        getattr(sqlite3, "SQLITE_CREATE_TABLE", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_INDEX", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TABLE", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_CREATE_TEMP_VIEW", None),
        getattr(sqlite3, "SQLITE_CREATE_TRIGGER", None),
        getattr(sqlite3, "SQLITE_CREATE_VIEW", None),
        getattr(sqlite3, "SQLITE_ATTACH", None),
        getattr(sqlite3, "SQLITE_DELETE", None),
        getattr(sqlite3, "SQLITE_DETACH", None),
        getattr(sqlite3, "SQLITE_DROP_INDEX", None),
        getattr(sqlite3, "SQLITE_DROP_TABLE", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_INDEX", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TABLE", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_DROP_TEMP_VIEW", None),
        getattr(sqlite3, "SQLITE_DROP_TRIGGER", None),
        getattr(sqlite3, "SQLITE_DROP_VIEW", None),
        getattr(sqlite3, "SQLITE_INSERT", None),
        getattr(sqlite3, "SQLITE_REINDEX", None),
        getattr(sqlite3, "SQLITE_TRANSACTION", None),
        getattr(sqlite3, "SQLITE_UPDATE", None),
    )
    if action is not None
)


def _sqlite_authorizer(
    action: int,
    arg1: str | None,
    arg2: str | None,
    _database: str | None,
    _trigger: str | None,
) -> int:
    if action in _SQLITE_WRITE_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA:
        readonly = {"database_list", "foreign_keys", "query_only", "table_info"}
        if arg1 not in readonly or (
            arg1 in {"foreign_keys", "query_only"} and arg2 is not None
        ):
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class ReadOnlySqlite:
    """Open existing allowlisted SQLite databases by symbolic key."""

    def __init__(self, databases: Mapping[str, ReadOnlyDatabase]) -> None:
        if not databases:
            raise ValueError("at least one database is required")
        validated: dict[str, ReadOnlyDatabase] = {}
        bindings: dict[str, _BoundPath] = {}
        for key, database in databases.items():
            _require_bounded_string(key, "database key", 128)
            root = _canonical_root(database.allowlisted_root)
            locator = _canonical_locator(database.locator, root)
            if not locator.exists():
                raise ValueError("database must already exist")
            metadata = os.lstat(locator)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("database must be a regular file")
            validated[key] = ReadOnlyDatabase(locator, root)
            bindings[key] = _bind_path(locator, root)
        self._databases = MappingProxyType(validated)
        self._bindings = MappingProxyType(bindings)

    def connect(self, database_key: str) -> sqlite3.Connection:
        try:
            database = self._databases[database_key]
        except (KeyError, TypeError) as exc:
            raise KeyError("unknown database key") from exc
        try:
            _verify_path_binding(self._bindings[database_key])
        except _PathChanged as exc:
            raise ValueError("database path changed after construction") from exc
        before = os.lstat(database.locator)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("database is no longer a regular file")
        binding = self._bindings[database_key]
        try:
            opened_file = _open_bound_file(binding)
        except _PathChanged as exc:
            raise ValueError("database path changed while connecting") from exc
        descriptor = opened_file.descriptor
        opened = opened_file.opened
        try:
            if not _same_file(before, opened):
                raise ValueError("database path changed while connecting")
            descriptor_path = f"/proc/self/fd/{descriptor}"
            uri = f"file:{quote(descriptor_path, safe='/')}?mode=ro"
            connection = sqlite3.connect(uri, uri=True)
            try:
                _verify_bound_file(binding, opened_file)
                main_path = connection.execute("PRAGMA database_list").fetchone()[2]
                if not _same_file(opened, os.stat(main_path)):
                    raise ValueError("database identity changed while connecting")
                connection.execute("PRAGMA query_only=ON")
                enabled = connection.execute("PRAGMA query_only").fetchone()
                if enabled != (1,):
                    raise sqlite3.OperationalError(
                        "query_only could not be enabled"
                    )
                connection.set_authorizer(_sqlite_authorizer)
                return connection
            except Exception:
                connection.close()
                raise
        finally:
            for opened_descriptor in reversed(opened_file.descriptors):
                os.close(opened_descriptor)
