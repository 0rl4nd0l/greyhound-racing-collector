from __future__ import annotations

import inspect
import hashlib
import json
import os
import sqlite3
import subprocess
import threading
import webbrowser
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from scripts.shadow_autopilot_daemon import write_json as producer_write_json

from src.operator_ui import HistoricalClaim
from src.operator_ui import foundation
from src.operator_ui.foundation import (
    Availability,
    EvidenceEnvelope,
    EvidenceStatus,
    Freshness,
    Integrity,
    JsonSerializationPolicy,
    JsonSource,
    OperatorEvidenceReader,
    ReadOnlyDatabase,
    ReadOnlySqlite,
    RawSourceConfig,
    ReferenceHash,
    SourceConfig,
    TimestampSyntax,
    status_for,
)


def digest_only_reader(root: Path, raw_path: Path, *, digest: str, size: int) -> OperatorEvidenceReader:
    json_path = root / "source.json"; write_payload(json_path)
    raw = RawSourceConfig(
        raw_path, root, "fixed_file", "inventory", "fixture.inventory",
        "P-REPORT-24H", "Exact authenticated identity.", max_bytes=64*1024*1024,
        expected_sha256=digest, expected_bytes=size, digest_only=True,
    )
    return OperatorEvidenceReader({"source": source_config(root, json_path)}, raw_sources={"inventory": raw}, clock=lambda: NOW)


NOW = datetime(2026, 7, 31, 2, 0, tzinfo=timezone.utc)
REFERENCE_HASH = "a" * 64
EXPECTED_FIELDS = (
    "generated_at",
    "race_id",
    "reference_sha256",
    "schema",
    "value",
)


def canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def valid_payload(*, generated_at: str = "2026-07-31T01:59:00Z") -> dict:
    return {
        "schema": "operator_fixture_v1",
        "generated_at": generated_at,
        "race_id": "race-7",
        "reference_sha256": REFERENCE_HASH,
        "value": {"count": 3},
    }


def write_payload(path: Path, payload: object | None = None) -> bytes:
    raw = canonical_bytes(valid_payload() if payload is None else payload)
    path.write_bytes(raw)
    return raw


def test_digest_only_raw_source_streams_identity_without_returning_bytes(tmp_path):
    raw_path = tmp_path / "inventory.csv"; raw_path.write_bytes(b"inventory" * 8192)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    envelope, raw, byte_count = digest_only_reader(tmp_path, raw_path, digest=digest, size=raw_path.stat().st_size).read_raw_authenticated("inventory")
    assert envelope.status == EvidenceStatus.AVAILABLE_FRESH.value
    assert envelope.content_sha256 == digest and byte_count == raw_path.stat().st_size
    assert raw is None


@pytest.mark.parametrize("mismatch", ["hash", "bytes"])
def test_digest_only_raw_source_identity_mismatch_is_divergent(tmp_path, mismatch):
    raw_path = tmp_path / "inventory.jsonl"; raw_path.write_bytes(b"{}\n")
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    reader = digest_only_reader(tmp_path, raw_path, digest="0"*64 if mismatch=="hash" else digest, size=4 if mismatch=="bytes" else 3)
    envelope, raw, byte_count = reader.read_raw_authenticated("inventory")
    assert envelope.status == EvidenceStatus.DIVERGENT.value
    assert raw is None and byte_count is None


def test_digest_only_raw_source_timeout_fails_closed(tmp_path, monkeypatch):
    raw_path = tmp_path / "inventory.csv"; raw_path.write_bytes(b"inventory")
    reader = digest_only_reader(tmp_path, raw_path, digest=hashlib.sha256(b"inventory").hexdigest(), size=9)
    ticks = iter((0.0,31.0))
    monkeypatch.setattr(foundation.time, "monotonic", lambda: next(ticks,31.0))
    envelope, raw, byte_count = reader.read_raw_authenticated("inventory")
    assert envelope.status == EvidenceStatus.INVALID_INTEGRITY_FAILED.value
    assert raw is None and byte_count is None


def source_config(root: Path, path: Path, **changes) -> SourceConfig:
    config = SourceConfig(
        locator=path,
        allowlisted_root=root,
        source_kind="test_fixture",
        source_identity="fixture-v1",
        source_locator="fixture.primary",
        policy="P-DEPLOY-60",
        supported_claim="Exact fixture count at the displayed observation.",
        json=JsonSource(
            schema_field="schema",
            schema_value="operator_fixture_v1",
            top_level_fields=EXPECTED_FIELDS,
            time_field="generated_at",
            reference_hashes=(
                ReferenceHash(
                    "fixture_reference", "reference_sha256", REFERENCE_HASH
                ),
            ),
            identity_fields=("race_id",),
            max_depth=6,
            max_items=32,
            max_string_bytes=128,
        ),
        max_bytes=4096,
        max_envelope_bytes=4096,
    )
    return replace(config, **changes)


def reader(config: SourceConfig) -> OperatorEvidenceReader:
    return OperatorEvidenceReader({"fixture": config}, clock=lambda: NOW)


def test_valid_envelope_is_deterministic_finite_immutable_and_serializable(
    tmp_path,
):
    raw = write_payload(tmp_path / "fixture.json")
    observed = reader(
        source_config(tmp_path, tmp_path / "fixture.json")
    ).read("fixture")

    assert observed.status == "AVAILABLE/FRESH"
    assert observed.content_sha256 == foundation.hashlib.sha256(raw).hexdigest()
    assert observed.source_locator == "fixture.primary"
    assert not observed.source_locator.startswith("/")
    assert observed.generated_at == "2026-07-31T01:59:00Z"
    assert observed.server_observed_at == "2026-07-31T02:00:00.000000Z"
    assert observed.age_seconds == 60
    assert observed.reference_hashes == (("fixture_reference", REFERENCE_HASH),)
    assert observed.evidence_identity == (("race_id", "race-7"),)
    assert observed.supported_claim == (
        "Exact fixture count at the displayed observation."
    )
    assert json.loads(observed.to_json())["status"] == "AVAILABLE/FRESH"
    assert len(observed.to_json().encode()) < 4096
    with pytest.raises(FrozenInstanceError):
        observed.status = "STALE"
    with pytest.raises(TypeError):
        EvidenceEnvelope()


@pytest.mark.parametrize(
    ("availability", "integrity", "freshness", "conflict", "expected"),
    [
        (
            Availability.PRESENT,
            Integrity.VALID,
            Freshness.FRESH,
            False,
            EvidenceStatus.AVAILABLE_FRESH,
        ),
        (
            Availability.PRESENT,
            Integrity.VALID,
            Freshness.STALE,
            False,
            EvidenceStatus.STALE,
        ),
        (
            Availability.MISSING,
            Integrity.UNKNOWN,
            Freshness.UNKNOWN,
            False,
            EvidenceStatus.UNAVAILABLE_DATA_MISSING,
        ),
        (
            Availability.UNREADABLE,
            Integrity.UNKNOWN,
            Freshness.UNKNOWN,
            False,
            EvidenceStatus.UNAVAILABLE_DATA_MISSING,
        ),
        (
            Availability.ERROR,
            Integrity.FAILED,
            Freshness.UNKNOWN,
            False,
            EvidenceStatus.INVALID_INTEGRITY_FAILED,
        ),
        (
            Availability.PRESENT,
            Integrity.FAILED,
            Freshness.FRESH,
            False,
            EvidenceStatus.INVALID_INTEGRITY_FAILED,
        ),
        (
            Availability.PRESENT,
            Integrity.VALID,
            Freshness.FRESH,
            True,
            EvidenceStatus.DIVERGENT,
        ),
    ],
)
def test_exact_status_mapping(
    availability, integrity, freshness, conflict, expected
):
    assert (
        status_for(
            availability, integrity, freshness, conflict=conflict
        )
        is expected
    )


def test_fixed_freshness_boundary_stale_and_future_fail_closed(tmp_path):
    path = tmp_path / "fixture.json"
    write_payload(path)
    configured = source_config(tmp_path, path)

    assert reader(configured).read("fixture").status == "AVAILABLE/FRESH"

    write_payload(path, valid_payload(generated_at="2026-07-31T01:58:59Z"))
    assert reader(configured).read("fixture").status == "STALE"

    write_payload(path, valid_payload(generated_at="2026-07-31T02:00:01Z"))
    future = reader(configured).read("fixture")
    assert future.status == "UNAVAILABLE/DATA_MISSING"
    assert future.age_seconds is None

    payload = valid_payload(generated_at="not-a-time")
    write_payload(path, payload)
    assert reader(configured).read("fixture").status == (
        "INVALID/INTEGRITY_FAILED"
    )


@pytest.mark.parametrize(
    "generated_at",
    (
        "2026-07-31T01:59:00+00:00",
        "2026-07-31T01:59:00+0000",
        "2026-07-31T01:59:00z",
    ),
)
def test_reader_rejects_utc_timestamp_without_canonical_terminal_z(
    tmp_path, generated_at
):
    path = tmp_path / "fixture.json"
    write_payload(path, valid_payload(generated_at=generated_at))

    observed = reader(source_config(tmp_path, path)).read("fixture")

    assert observed.status == "INVALID/INTEGRITY_FAILED"
    assert observed.age_seconds is None


@pytest.mark.parametrize(
    ("generated_at", "expected_age"),
    (
        ("2026-07-31T01:59:00+00:00", 60),
        ("2026-07-31T11:59:00+10:00", 60),
    ),
)
def test_native_producer_timestamp_normalizes_aware_offsets(
    tmp_path, generated_at, expected_age
):
    path = tmp_path / "fixture.json"
    write_payload(path, valid_payload(generated_at=generated_at))
    configured = source_config(tmp_path, path)
    configured = replace(
        configured,
        json=replace(
            configured.json, timestamp_syntax=TimestampSyntax.AWARE_ISO8601
        ),
    )

    observed = reader(configured).read("fixture")

    assert observed.status == "AVAILABLE/FRESH"
    assert observed.age_seconds == expected_age
    assert observed.content_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "generated_at",
    (
        "2026-07-31T01:59:00",
        "2026-07-31T01:59:00+0000",
        "2026-07-31T01:59:00z",
        "2026-07-31T01:59:00+00:00junk",
        "x" * 129,
        True,
        None,
    ),
)
def test_native_producer_timestamp_rejects_invalid_or_unbounded_values(
    tmp_path, generated_at
):
    path = tmp_path / "fixture.json"
    write_payload(path, valid_payload(generated_at=generated_at))
    configured = source_config(tmp_path, path)
    configured = replace(
        configured,
        json=replace(
            configured.json, timestamp_syntax=TimestampSyntax.AWARE_ISO8601
        ),
    )

    observed = reader(configured).read("fixture")

    assert observed.status == "INVALID/INTEGRITY_FAILED"
    assert observed.age_seconds is None


def test_adapter_owned_exact_payload_may_omit_producer_schema_identity(tmp_path):
    payload = {
        "generated_at": "2026-07-31T01:59:00Z",
        "status": "SUCCESS",
    }
    path = tmp_path / "producer-policy.json"
    path.write_bytes(canonical_bytes(payload))
    configured = SourceConfig(
        locator=path,
        allowlisted_root=tmp_path,
        source_kind="producer_policy_observation",
        source_identity="shadow_autopilot_refresh_report",
        source_locator="collector.odds_refresh",
        policy="P-COLLECTOR-ODDS-DYNAMIC",
        supported_claim="Exact bounded adapter-owned policy observation.",
        json=JsonSource(
            schema_field=None,
            schema_value=None,
            top_level_fields=("generated_at", "status"),
            time_field=None,
        ),
    )
    envelope, observed = OperatorEvidenceReader(
        {"policy": configured}, clock=lambda: NOW
    ).read_payload("policy")
    assert envelope.schema_integrity == "valid"
    assert envelope.status == "UNAVAILABLE/DATA_MISSING"
    assert envelope.age_seconds is None
    assert dict(observed or {}) == payload

    with pytest.raises(ValueError, match="only adapter-owned"):
        OperatorEvidenceReader(
            {"policy": replace(configured, policy="P-DEPLOY-60")},
            clock=lambda: NOW,
        )


def test_missing_unreadable_and_oversize_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "fixture.json"
    configured = source_config(tmp_path, path)
    assert reader(configured).read("fixture").status == (
        "UNAVAILABLE/DATA_MISSING"
    )

    write_payload(path)

    def denied(*_args, **_kwargs):
        raise PermissionError("fixture denial")

    monkeypatch.setattr(foundation.os, "open", denied)
    unreadable = reader(configured).read("fixture")
    assert unreadable.availability == "unreadable"
    assert unreadable.status == "UNAVAILABLE/DATA_MISSING"
    monkeypatch.undo()

    oversize = replace(configured, max_bytes=8)
    invalid = reader(oversize).read("fixture")
    assert invalid.status == "INVALID/INTEGRITY_FAILED"
    assert invalid.availability == "present"
    assert invalid.schema_integrity == "failed"
    assert invalid.age_seconds is None


@pytest.mark.parametrize(
    "raw",
    [
        b"{",
        b'{"generated_at":"2026-07-31T01:59:00Z","race_id":"race-7",'
        b'"reference_sha256":"'
        + REFERENCE_HASH.encode()
        + b'","schema":"operator_fixture_v1","schema":"duplicate","value":{}}',
        b'{"generated_at":"2026-07-31T01:59:00Z","race_id":"race-7",'
        b'"reference_sha256":"'
        + REFERENCE_HASH.encode()
        + b'","schema":"operator_fixture_v1","value":NaN}',
        b'{"generated_at":"2026-07-31T01:59:00Z","race_id":"race-7",'
        b'"reference_sha256":"'
        + REFERENCE_HASH.encode()
        + b'","schema":"operator_fixture_v1","value":Infinity}',
    ],
)
def test_malformed_duplicate_and_nonfinite_json_is_invalid(tmp_path, raw):
    path = tmp_path / "fixture.json"
    path.write_bytes(raw)
    observed = reader(source_config(tmp_path, path)).read("fixture")
    assert observed.status == "INVALID/INTEGRITY_FAILED"
    assert observed.content_sha256 == foundation.hashlib.sha256(raw).hexdigest()


def test_json_serialization_policy_is_finite_and_defaults_to_compact():
    source = source_config(Path("/unused"), Path("/unused/fixture.json")).json
    assert (
        source.serialization_policy
        is JsonSerializationPolicy.COMPACT_CANONICAL
    )

    explicit = replace(source, serialization_policy="producer_pretty_sorted")
    assert (
        explicit.serialization_policy
        is JsonSerializationPolicy.PRODUCER_PRETTY_SORTED
    )

    with pytest.raises(ValueError, match="serialization policy"):
        replace(source, serialization_policy="unknown")


@pytest.mark.parametrize(
    "producer_writer",
    [
        pytest.param(producer_write_json, id="shadow-autopilot-daemon"),
        pytest.param(
            __import__(
                "scripts.build_race_evidence_inventory_packet",
                fromlist=["write_json"],
            ).write_json,
            id="race-evidence-inventory",
        ),
    ],
)
def test_pretty_sorted_producer_requires_exact_explicit_policy(
    tmp_path, producer_writer
):
    path = tmp_path / "fixture.json"
    configured = source_config(tmp_path, path)
    producer_writer(path, valid_payload())
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\n  " in raw
    assert reader(configured).read("fixture").status == "INVALID/INTEGRITY_FAILED"

    configured = replace(
        configured,
        json=replace(
            configured.json,
            serialization_policy=JsonSerializationPolicy.PRODUCER_PRETTY_SORTED,
        ),
    )
    observed = reader(configured).read("fixture")
    assert observed.status == "AVAILABLE/FRESH"
    assert observed.content_sha256 == hashlib.sha256(raw).hexdigest()

    path.write_bytes(raw.replace(b'  "generated_at"', b' "generated_at"', 1))
    assert reader(configured).read("fixture").status == "INVALID/INTEGRITY_FAILED"


def test_unexpected_and_wrong_schema_are_invalid(tmp_path):
    path = tmp_path / "fixture.json"
    configured = source_config(tmp_path, path)

    extra = valid_payload()
    extra["unexpected"] = True
    write_payload(path, extra)
    assert reader(configured).read("fixture").status == (
        "INVALID/INTEGRITY_FAILED"
    )

    wrong = valid_payload()
    wrong["schema"] = "other"
    write_payload(path, wrong)
    assert reader(configured).read("fixture").status == (
        "INVALID/INTEGRITY_FAILED"
    )


def test_expected_content_and_reference_hash_conflicts_are_divergent(tmp_path):
    path = tmp_path / "fixture.json"
    raw = write_payload(path)
    configured = source_config(tmp_path, path)

    content_conflict = replace(configured, expected_sha256="b" * 64)
    assert reader(content_conflict).read("fixture").status == "DIVERGENT"

    payload = valid_payload()
    payload["reference_sha256"] = "c" * 64
    write_payload(path, payload)
    assert reader(configured).read("fixture").status == "DIVERGENT"

    matching = replace(
        configured,
        expected_sha256=foundation.hashlib.sha256(raw).hexdigest(),
    )
    write_payload(path)
    assert reader(matching).read("fixture").status == "AVAILABLE/FRESH"


def test_unknown_key_and_public_methods_accept_no_path_or_time_control(tmp_path):
    path = tmp_path / "fixture.json"
    write_payload(path)
    evidence = reader(source_config(tmp_path, path))

    with pytest.raises(KeyError, match="unknown evidence source key"):
        evidence.read("/tmp/arbitrary.json")
    assert tuple(inspect.signature(evidence.read).parameters) == ("source_key",)


def test_root_noncanonical_and_symlink_locators_are_rejected(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(ValueError, match="outside"):
        reader(source_config(tmp_path, outside))

    real = tmp_path / "real.json"
    write_payload(real)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="canonical|symlink"):
        reader(source_config(tmp_path, link))

    relative = replace(source_config(tmp_path, real), locator=Path("real.json"))
    with pytest.raises(ValueError, match="absolute"):
        reader(relative)


def test_path_replacement_during_read_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "fixture.json"
    replacement = tmp_path / "replacement.json"
    write_payload(path)
    write_payload(replacement)
    configured = source_config(tmp_path, path)
    real_lstat = foundation.os.lstat
    calls = 0

    def replaced_lstat(candidate):
        nonlocal calls
        if Path(candidate) == path:
            calls += 1
            if calls >= 2:
                return real_lstat(replacement)
        return real_lstat(candidate)

    monkeypatch.setattr(foundation.os, "lstat", replaced_lstat)
    observed = reader(configured).read("fixture")
    assert observed.status == "INVALID/INTEGRITY_FAILED"
    assert observed.schema_integrity == "failed"


def test_parent_replacement_during_read_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    path = parent / "fixture.json"
    write_payload(path)
    write_payload(outside / "fixture.json")
    evidence = reader(source_config(root, path))

    parent.rename(root / "original-parent")
    parent.symlink_to(outside, target_is_directory=True)

    observed = evidence.read("fixture")
    assert observed.status == "INVALID/INTEGRITY_FAILED"
    assert observed.availability == "present"
    assert observed.schema_integrity == "failed"


def test_same_path_ordinary_root_replacement_after_construction_is_rejected(
    tmp_path,
):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "fixture.json"
    write_payload(path)
    evidence = reader(source_config(root, path))

    root.rename(tmp_path / "original-root")
    root.mkdir()
    write_payload(root / "fixture.json")

    observed = evidence.read("fixture")
    assert observed.status == "INVALID/INTEGRITY_FAILED"


def test_same_path_nested_parent_replacement_after_construction_is_rejected(
    tmp_path,
):
    root = tmp_path / "root"
    parent = root / "one" / "two"
    parent.mkdir(parents=True)
    path = parent / "fixture.json"
    write_payload(path)
    evidence = reader(source_config(root, path))

    parent.rename(root / "one" / "original-two")
    parent.mkdir()
    write_payload(parent / "fixture.json")

    assert evidence.read("fixture").status == "INVALID/INTEGRITY_FAILED"


def test_same_path_ordinary_file_replacement_after_construction_is_rejected(
    tmp_path,
):
    path = tmp_path / "fixture.json"
    replacement = tmp_path / "replacement.json"
    write_payload(path)
    write_payload(replacement)
    evidence = reader(source_config(tmp_path, path))

    replacement.replace(path)

    assert evidence.read("fixture").status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize("replacement_target", ["root", "parent", "file"])
def test_ordinary_replacement_after_binding_verification_before_open_is_rejected(
    tmp_path, monkeypatch, replacement_target
):
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    path = parent / "fixture.json"
    write_payload(path)
    evidence = reader(source_config(root, path))
    real_open = foundation.os.open
    replaced = False

    def replacing_open(candidate, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            if replacement_target == "root":
                root.rename(tmp_path / "original-root")
                parent.mkdir(parents=True)
                write_payload(path)
            elif replacement_target == "parent":
                parent.rename(root / "original-parent")
                parent.mkdir()
                write_payload(path)
            else:
                replacement = parent / "replacement.json"
                write_payload(replacement)
                replacement.replace(path)
        return real_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(foundation.os, "open", replacing_open)

    assert evidence.read("fixture").status == "INVALID/INTEGRITY_FAILED"


@pytest.mark.parametrize(
    "schema_change,payload_change",
    [
        ({"max_depth": 2}, {"value": {"nested": {"too": "deep"}}}),
        ({"max_items": 5}, {"value": [1, 2, 3, 4, 5, 6]}),
        ({"max_string_bytes": 4}, {"value": "too-long"}),
    ],
)
def test_json_nesting_item_and_string_bounds(
    tmp_path, schema_change, payload_change
):
    path = tmp_path / "fixture.json"
    payload = valid_payload()
    payload.update(payload_change)
    write_payload(path, payload)
    configured = source_config(tmp_path, path)
    configured = replace(
        configured, json=replace(configured.json, **schema_change)
    )
    assert reader(configured).read("fixture").status == (
        "INVALID/INTEGRITY_FAILED"
    )


def test_serialized_envelope_bound_is_enforced(tmp_path):
    path = tmp_path / "fixture.json"
    write_payload(path)
    configured = replace(
        source_config(tmp_path, path), max_envelope_bytes=32
    )
    with pytest.raises(ValueError, match="serialized envelope"):
        reader(configured).read("fixture")


def test_historical_policy_has_no_expiry_but_proves_no_current_health(
    tmp_path,
):
    path = tmp_path / "fixture.json"
    write_payload(path, valid_payload(generated_at="2020-01-01T00:00:00Z"))
    evidence = reader(
        replace(
            source_config(tmp_path, path),
            policy="P-IMMUTABLE-HISTORICAL",
            supported_claim=HistoricalClaim.RUN,
        )
    )

    assert set(evidence.policies) == {
        "P-DEPLOY-60",
        "P-COLLECTOR-FULL-DYNAMIC",
        "P-COLLECTOR-ODDS-DYNAMIC",
        "P-COLLECTOR-AGGREGATE",
        "P-UPCOMING-300-PREJUMP",
        "P-CURRENT-INDEX-1200",
        "P-CATALOG-60",
        "P-BUNDLE-LIST-60",
        "P-JOB-5-DEADLINE",
        "P-REPORT-24H",
        "P-OPS-5",
        "P-IMMUTABLE-HISTORICAL",
    }
    historical = evidence.read("fixture")
    assert historical.status == "AVAILABLE/FRESH"
    assert historical.age_seconds > 0
    assert historical.supported_claim == (
        "Verified result of the identified historical run only."
    )

    write_payload(path, valid_payload(generated_at="2026-07-31T02:00:01Z"))
    future = evidence.read("fixture")
    assert future.status == "UNAVAILABLE/DATA_MISSING"
    assert future.age_seconds is None


@pytest.mark.parametrize(
    "claim",
    [
        "arbitrary trusted prose",
        "Current system is healthy and promotion-ready.",
        "Representative present-quality evidence.",
        "Verified result of the identified historical run only.",
    ],
)
def test_historical_policy_rejects_unstructured_claims(tmp_path, claim):
    path = tmp_path / "fixture.json"
    write_payload(path)
    with pytest.raises(ValueError, match="finite run- or slice-bound"):
        reader(
            replace(
                source_config(tmp_path, path),
                policy="P-IMMUTABLE-HISTORICAL",
                supported_claim=claim,
            )
        )


def test_structured_historical_claim_rejects_nonhistorical_policy(tmp_path):
    path = tmp_path / "fixture.json"
    write_payload(path)
    with pytest.raises(ValueError, match="require historical policy"):
        reader(
            replace(
                source_config(tmp_path, path),
                supported_claim=HistoricalClaim.RUN,
            )
        )


@pytest.mark.parametrize(
    ("claim", "expected"),
    [
        (
            HistoricalClaim.RUN,
            "Verified result of the identified historical run only.",
        ),
        (
            HistoricalClaim.SLICE,
            "Verified result of the identified historical slice only.",
        ),
    ],
)
def test_historical_claims_have_fixed_narrow_rendering_and_displayed_age(
    tmp_path, claim, expected
):
    path = tmp_path / "fixture.json"
    write_payload(path, valid_payload(generated_at="2020-01-01T00:00:00Z"))
    historical = reader(
        replace(
            source_config(tmp_path, path),
            policy="P-IMMUTABLE-HISTORICAL",
            supported_claim=claim,
        )
    ).read("fixture")

    assert historical.status == "AVAILABLE/FRESH"
    assert historical.age_seconds == (
        NOW - datetime(2020, 1, 1, tzinfo=timezone.utc)
    ).total_seconds()
    assert historical.supported_claim == expected


@pytest.mark.parametrize(
    "policy",
    [
        "P-COLLECTOR-FULL-DYNAMIC",
        "P-COLLECTOR-ODDS-DYNAMIC",
        "P-COLLECTOR-AGGREGATE",
        "P-UPCOMING-300-PREJUMP",
        "P-CURRENT-INDEX-1200",
        "P-JOB-5-DEADLINE",
        "P-OPS-5",
    ],
)
def test_adapter_owned_policy_semantics_fail_closed(tmp_path, policy):
    path = tmp_path / "fixture.json"
    write_payload(path)
    configured = replace(source_config(tmp_path, path), policy=policy)

    assert reader(configured).read("fixture").status == (
        "UNAVAILABLE/DATA_MISSING"
    )


def make_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE evidence (id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO evidence(value) VALUES ('kept')")
    connection.commit()
    connection.close()


def sqlite_helper(root: Path, path: Path) -> ReadOnlySqlite:
    return ReadOnlySqlite(
        {"analytics": ReadOnlyDatabase(locator=path, allowlisted_root=root)}
    )


def test_sqlite_symbolic_key_reads_and_rejects_unknown_or_arbitrary_path(
    tmp_path,
):
    path = tmp_path / "evidence.db"
    make_database(path)
    helper = sqlite_helper(tmp_path, path)
    connection = helper.connect("analytics")
    try:
        assert connection.execute("SELECT value FROM evidence").fetchone() == (
            "kept",
        )
    finally:
        connection.close()

    with pytest.raises(KeyError, match="unknown database key"):
        helper.connect(str(path))
    assert tuple(inspect.signature(helper.connect).parameters) == ("database_key",)


def test_sqlite_root_symlink_missing_and_noncanonical_rejection(tmp_path):
    path = tmp_path / "evidence.db"
    make_database(path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.db"
    make_database(outside)

    with pytest.raises(ValueError, match="outside"):
        sqlite_helper(tmp_path, outside)
    with pytest.raises(ValueError, match="already exist"):
        sqlite_helper(tmp_path, tmp_path / "missing.db")

    link = tmp_path / "linked.db"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="canonical|symlink"):
        sqlite_helper(tmp_path, link)

    with pytest.raises(ValueError, match="absolute"):
        sqlite_helper(tmp_path, Path("evidence.db"))


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TABLE blocked (id INTEGER)",
        "INSERT INTO evidence(value) VALUES ('blocked')",
        "UPDATE evidence SET value='blocked'",
        "DELETE FROM evidence",
        "PRAGMA user_version=7",
        "PRAGMA journal_mode=WAL",
        "PRAGMA query_only=OFF",
    ],
)
def test_sqlite_ddl_dml_and_writable_pragmas_are_denied(tmp_path, statement):
    path = tmp_path / "evidence.db"
    make_database(path)
    connection = sqlite_helper(tmp_path, path).connect("analytics")
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute(statement)
        assert connection.execute("SELECT value FROM evidence").fetchone() == (
            "kept",
        )
        assert connection.execute("PRAGMA query_only").fetchone() == (1,)
    finally:
        connection.close()

    check = sqlite3.connect(path)
    try:
        assert check.execute("SELECT value FROM evidence").fetchall() == [("kept",)]
        assert check.execute("PRAGMA user_version").fetchone() == (0,)
    finally:
        check.close()


def test_sqlite_path_replacement_at_connect_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "evidence.db"
    replacement = tmp_path / "replacement.db"
    make_database(path)
    make_database(replacement)
    helper = sqlite_helper(tmp_path, path)
    real_lstat = foundation.os.lstat
    calls = 0

    def replaced_lstat(candidate):
        nonlocal calls
        if Path(candidate) == path:
            calls += 1
            if calls >= 2:
                return real_lstat(replacement)
        return real_lstat(candidate)

    monkeypatch.setattr(foundation.os, "lstat", replaced_lstat)
    with pytest.raises(ValueError, match="changed"):
        helper.connect("analytics")


def test_sqlite_parent_replacement_cannot_escape_root(tmp_path):
    root = tmp_path / "root"
    parent = root / "parent"
    outside = tmp_path / "outside"
    parent.mkdir(parents=True)
    outside.mkdir()
    path = parent / "evidence.db"
    make_database(path)
    make_database(outside / "evidence.db")
    helper = sqlite_helper(root, path)

    parent.rename(root / "original-parent")
    parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escaped|changed"):
        helper.connect("analytics")


@pytest.mark.parametrize("replacement_target", ["root", "parent", "database"])
def test_sqlite_same_path_ordinary_replacement_after_construction_is_rejected(
    tmp_path, replacement_target
):
    root = tmp_path / "root"
    parent = root / "parent"
    parent.mkdir(parents=True)
    path = parent / "evidence.db"
    make_database(path)
    helper = sqlite_helper(root, path)

    if replacement_target == "root":
        root.rename(tmp_path / "original-root")
        parent.mkdir(parents=True)
        make_database(path)
    elif replacement_target == "parent":
        parent.rename(root / "original-parent")
        parent.mkdir()
        make_database(path)
    else:
        replacement = parent / "replacement.db"
        make_database(replacement)
        replacement.replace(path)

    with pytest.raises(ValueError, match="changed"):
        helper.connect("analytics")


def test_sqlite_replacement_after_binding_verification_during_connect_is_rejected(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "evidence.db"
    make_database(path)
    helper = sqlite_helper(root, path)
    real_open = foundation.os.open
    replaced = False

    def replacing_open(candidate, flags, *args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            replacement = root / "replacement.db"
            make_database(replacement)
            replacement.replace(path)
        return real_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(foundation.os, "open", replacing_open)

    with pytest.raises(ValueError, match="changed"):
        helper.connect("analytics")


def test_sqlite_attach_execution_and_detach_authorizer_are_denied(tmp_path):
    path = tmp_path / "evidence.db"
    make_database(path)
    connection = sqlite_helper(tmp_path, path).connect("analytics")
    try:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("ATTACH DATABASE '/tmp/x.db' AS x")
    finally:
        connection.close()

    assert (
        foundation._sqlite_authorizer(
            sqlite3.SQLITE_DETACH, "attached", None, None, None
        )
        == sqlite3.SQLITE_DENY
    )


def test_foundation_uses_no_shell_subprocess_lock_or_browser_collaborator(
    tmp_path, monkeypatch
):
    path = tmp_path / "fixture.json"
    write_payload(path)
    calls: list[str] = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("forbidden collaborator used")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(threading, "Lock", forbidden)
    monkeypatch.setattr(webbrowser, "open", forbidden)

    assert reader(source_config(tmp_path, path)).read("fixture").status == (
        "AVAILABLE/FRESH"
    )
    assert calls == []
