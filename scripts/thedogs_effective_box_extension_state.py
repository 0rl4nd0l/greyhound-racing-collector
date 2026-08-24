#!/usr/bin/env python3
"""Persist raw TheDogs discovery before parsing a resumable schedule."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.capture_thedogs_market_history import (
    CaptureError,
    canonical_json_bytes,
    exact_odds_identity,
    iso_utc,
    parse_timestamp,
    sha256_bytes,
    utc_now,
)

CLAIM_SCHEMA_VERSION = "thedogs_extension_discovery_request_claim_v1"
RESPONSE_SCHEMA_VERSION = "thedogs_extension_discovery_response_v1"
STATE_SCHEMA_VERSION = "thedogs_extension_discovery_state_v2"
CLAIM_NAME = "discovery_request_claim.json"
RESPONSE_NAME = "discovery_response.json"
STATE_NAME = "discovery_state.json"


class DiscoveryStateError(CaptureError):
    """Raised when discovery cannot be acquired or resumed without ambiguity."""


@dataclass(frozen=True)
class DiscoveryAcquisition:
    """The exact response from one discovery request, before schedule parsing."""

    requested_url: str
    final_url: str
    request_start_utc: datetime
    request_end_utc: datetime
    status_code: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class DiscoveryBundle:
    """Validated immutable response and schedule state."""

    response: dict[str, Any]
    state: dict[str, Any]
    resumed: bool
    network_requests: int


def _serialized_json(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def _write_immutable_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_serialized_json(payload))
        handle.flush()
        os.fsync(handle.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiscoveryStateError(f"{label}_invalid") from exc
    if not isinstance(payload, dict):
        raise DiscoveryStateError(f"{label}_invalid")
    return payload


def _contract_hash(contract: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(contract))


def _strict_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DiscoveryStateError(f"{field}_timezone_required")
    return value.astimezone(timezone.utc)


def _decode_body(response_state: Mapping[str, Any]) -> bytes:
    response = response_state.get("response")
    if not isinstance(response, Mapping):
        raise DiscoveryStateError("discovery_response_invalid")
    parse_timestamp(response_state.get("frozen_at_utc"), field="discovery_frozen_at_utc")
    if response_state.get("request_attempts") != 1:
        raise DiscoveryStateError("discovery_request_attempts_invalid")
    if response_state.get("request_retries") != 0:
        raise DiscoveryStateError("discovery_request_retries_invalid")
    encoded = response.get("body_base64")
    if not isinstance(encoded, str) or not encoded:
        raise DiscoveryStateError("discovery_raw_body_missing")
    try:
        body = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise DiscoveryStateError("discovery_raw_body_invalid") from exc
    if len(body) != response.get("body_bytes"):
        raise DiscoveryStateError("discovery_raw_body_size_mismatch")
    if sha256_bytes(body) != response.get("body_sha256"):
        raise DiscoveryStateError("discovery_raw_body_hash_mismatch")
    return body


def _validated_candidates(values: Any, *, cohort_date: str) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise DiscoveryStateError("discovery_candidates_required")
    candidates: list[dict[str, Any]] = []
    race_urls: set[str] = set()
    odds_urls: set[str] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise DiscoveryStateError("discovery_candidate_invalid")
        race_url = str(value.get("race_url") or "").strip()
        odds_url = str(value.get("odds_url") or "").strip()
        identity = exact_odds_identity(odds_url)
        if identity["race_url"] != race_url:
            raise DiscoveryStateError("discovery_candidate_race_url_mismatch")
        if identity["race_date"] != cohort_date:
            raise DiscoveryStateError("discovery_candidate_cohort_date_mismatch")
        jump = parse_timestamp(value.get("jump_timestamp"), field="jump_timestamp")
        if race_url in race_urls or odds_url in odds_urls:
            raise DiscoveryStateError("discovery_candidate_duplicate")
        race_urls.add(race_url)
        odds_urls.add(odds_url)
        candidates.append(
            {
                "race_url": race_url,
                "odds_url": odds_url,
                "jump_timestamp": iso_utc(jump),
            }
        )
    return candidates


def _validate_claim(claim: Mapping[str, Any], *, contract: Mapping[str, Any]) -> None:
    if claim.get("schema_version") != CLAIM_SCHEMA_VERSION:
        raise DiscoveryStateError("discovery_request_claim_schema_invalid")
    if (
        claim.get("contract_sha256") != _contract_hash(contract)
        or claim.get("contract") != contract
    ):
        raise DiscoveryStateError("discovery_contract_mismatch")
    parse_timestamp(claim.get("request_authorized_at_utc"), field="request_authorized_at_utc")


def _validate_response_state(
    response_state: dict[str, Any],
    *,
    claim: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if response_state.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise DiscoveryStateError("discovery_response_schema_invalid")
    if (
        response_state.get("contract_sha256") != _contract_hash(contract)
        or response_state.get("contract") != contract
    ):
        raise DiscoveryStateError("discovery_contract_mismatch")
    if response_state.get("request_claim_sha256") != sha256_bytes(canonical_json_bytes(claim)):
        raise DiscoveryStateError("discovery_request_claim_hash_mismatch")
    response = response_state.get("response")
    if not isinstance(response, Mapping):
        raise DiscoveryStateError("discovery_response_invalid")
    discovery_url = str(contract.get("discovery_url") or "")
    if response.get("requested_url") != discovery_url or response.get("final_url") != discovery_url:
        raise DiscoveryStateError("discovery_response_url_mismatch")
    if response.get("status_code") != 200:
        raise DiscoveryStateError("discovery_response_status_invalid")
    request_start = parse_timestamp(
        response.get("request_start_utc"), field="discovery_request_start_utc"
    )
    request_end = parse_timestamp(
        response.get("request_end_utc"), field="discovery_request_end_utc"
    )
    if request_end < request_start:
        raise DiscoveryStateError("discovery_request_interval_invalid")
    _decode_body(response_state)
    core = {key: value for key, value in response_state.items() if key != "response_core_sha256"}
    if response_state.get("response_core_sha256") != sha256_bytes(canonical_json_bytes(core)):
        raise DiscoveryStateError("discovery_response_core_hash_mismatch")
    return response_state


def _validate_schedule_state(
    state: dict[str, Any],
    *,
    claim: Mapping[str, Any],
    contract: Mapping[str, Any],
    response_state: Mapping[str, Any],
) -> dict[str, Any]:
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise DiscoveryStateError("discovery_state_schema_invalid")
    parse_timestamp(state.get("frozen_at_utc"), field="discovery_state_frozen_at_utc")
    if (
        state.get("contract_sha256") != _contract_hash(contract)
        or state.get("contract") != contract
    ):
        raise DiscoveryStateError("discovery_contract_mismatch")
    if state.get("request_claim_sha256") != sha256_bytes(canonical_json_bytes(claim)):
        raise DiscoveryStateError("discovery_request_claim_hash_mismatch")
    if state.get("response_core_sha256") != response_state.get("response_core_sha256"):
        raise DiscoveryStateError("discovery_schedule_response_hash_mismatch")
    candidates = _validated_candidates(
        state.get("candidates"), cohort_date=str(contract.get("cohort_date") or "")
    )
    if state.get("candidate_count") != len(candidates):
        raise DiscoveryStateError("discovery_candidate_count_mismatch")
    if state.get("candidates_sha256") != sha256_bytes(canonical_json_bytes(candidates)):
        raise DiscoveryStateError("discovery_candidates_hash_mismatch")
    lead_minutes = int(contract.get("preflight_lead_minutes") or 0)
    if lead_minutes <= 0:
        raise DiscoveryStateError("preflight_lead_minutes_invalid")
    expected_preflight = min(
        parse_timestamp(row["jump_timestamp"], field="jump_timestamp") for row in candidates
    ) - timedelta(minutes=lead_minutes)
    if state.get("preflight_at_utc") != iso_utc(expected_preflight):
        raise DiscoveryStateError("discovery_preflight_time_mismatch")
    core = {key: value for key, value in state.items() if key != "state_core_sha256"}
    if state.get("state_core_sha256") != sha256_bytes(canonical_json_bytes(core)):
        raise DiscoveryStateError("discovery_state_core_hash_mismatch")
    return state


def _new_response_state(
    acquired: DiscoveryAcquisition,
    *,
    claim: Mapping[str, Any],
    contract: Mapping[str, Any],
    frozen_at: datetime,
) -> dict[str, Any]:
    request_start = _strict_utc(acquired.request_start_utc, field="discovery_request_start_utc")
    request_end = _strict_utc(acquired.request_end_utc, field="discovery_request_end_utc")
    body = bytes(acquired.body)
    response_state: dict[str, Any] = {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "frozen_at_utc": iso_utc(_strict_utc(frozen_at, field="discovery_frozen_at_utc")),
        "contract": contract,
        "contract_sha256": _contract_hash(contract),
        "request_claim_sha256": sha256_bytes(canonical_json_bytes(claim)),
        "request_attempts": 1,
        "request_retries": 0,
        "response": {
            "requested_url": acquired.requested_url,
            "final_url": acquired.final_url,
            "request_start_utc": iso_utc(request_start),
            "request_end_utc": iso_utc(request_end),
            "status_code": acquired.status_code,
            "headers": {
                str(key).lower(): str(value) for key, value in sorted(acquired.headers.items())
            },
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_bytes": len(body),
            "body_sha256": sha256_bytes(body),
        },
    }
    response_state["response_core_sha256"] = sha256_bytes(canonical_json_bytes(response_state))
    return response_state


def load_or_acquire_discovery(
    output_dir: Path,
    *,
    contract: Mapping[str, Any],
    acquire: Callable[[], DiscoveryAcquisition],
    parse_candidates: Callable[[bytes], Sequence[Mapping[str, Any]]],
    clock: Callable[[], datetime] = utc_now,
) -> DiscoveryBundle:
    """Persist one raw response, then parse or replay its schedule offline.

    The request claim is written before acquisition. The raw response is written
    and fsynced before ``parse_candidates`` is called. Therefore parser failure
    or interruption retains exact response evidence, and restart calls only the
    parser with the frozen bytes. A claim without a response remains ambiguous
    and fails closed without acquisition.
    """

    frozen_contract = json.loads(canonical_json_bytes(contract).decode("utf-8"))
    if not isinstance(frozen_contract, dict):
        raise DiscoveryStateError("discovery_contract_invalid")
    claim_path = output_dir / CLAIM_NAME
    response_path = output_dir / RESPONSE_NAME
    state_path = output_dir / STATE_NAME

    if state_path.exists() and not response_path.exists():
        raise DiscoveryStateError("discovery_response_missing")
    if (response_path.exists() or state_path.exists()) and not claim_path.exists():
        raise DiscoveryStateError("discovery_request_claim_missing")

    claim_was_existing = claim_path.exists()
    response_was_existing = response_path.exists()
    if claim_was_existing:
        claim = _read_json_object(claim_path, label="discovery_request_claim")
        _validate_claim(claim, contract=frozen_contract)
    else:
        claim = {
            "schema_version": CLAIM_SCHEMA_VERSION,
            "request_authorized_at_utc": iso_utc(
                _strict_utc(clock(), field="request_authorized_at_utc")
            ),
            "contract": frozen_contract,
            "contract_sha256": _contract_hash(frozen_contract),
            "maximum_discovery_requests": 1,
            "retry_allowed": False,
        }
        _write_immutable_json(claim_path, claim)

    if response_was_existing:
        response_state = _validate_response_state(
            _read_json_object(response_path, label="discovery_response"),
            claim=claim,
            contract=frozen_contract,
        )
        network_requests = 0
    else:
        if state_path.exists():
            raise DiscoveryStateError("discovery_response_missing")
        if claim_was_existing:
            # An earlier invocation may have sent its request before interruption.
            raise DiscoveryStateError("discovery_request_outcome_ambiguous")
        acquired = acquire()
        response_state = _new_response_state(
            acquired,
            claim=claim,
            contract=frozen_contract,
            frozen_at=clock(),
        )
        _write_immutable_json(response_path, response_state)
        response_state = _validate_response_state(
            _read_json_object(response_path, label="discovery_response"),
            claim=claim,
            contract=frozen_contract,
        )
        network_requests = 1

    if state_path.exists():
        state = _validate_schedule_state(
            _read_json_object(state_path, label="discovery_state"),
            claim=claim,
            contract=frozen_contract,
            response_state=response_state,
        )
        return DiscoveryBundle(
            response=response_state,
            state=state,
            resumed=True,
            network_requests=0,
        )

    candidates = _validated_candidates(
        list(parse_candidates(_decode_body(response_state))),
        cohort_date=str(frozen_contract.get("cohort_date") or ""),
    )
    lead_minutes = int(frozen_contract.get("preflight_lead_minutes") or 0)
    if lead_minutes <= 0:
        raise DiscoveryStateError("preflight_lead_minutes_invalid")
    preflight_at = min(
        parse_timestamp(row["jump_timestamp"], field="jump_timestamp") for row in candidates
    ) - timedelta(minutes=lead_minutes)
    state: dict[str, Any] = {
        "schema_version": STATE_SCHEMA_VERSION,
        "frozen_at_utc": iso_utc(_strict_utc(clock(), field="discovery_state_frozen_at_utc")),
        "contract": frozen_contract,
        "contract_sha256": _contract_hash(frozen_contract),
        "request_claim_sha256": sha256_bytes(canonical_json_bytes(claim)),
        "response_core_sha256": response_state["response_core_sha256"],
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidates_sha256": sha256_bytes(canonical_json_bytes(candidates)),
        "preflight_at_utc": iso_utc(preflight_at),
    }
    state["state_core_sha256"] = sha256_bytes(canonical_json_bytes(state))
    _write_immutable_json(state_path, state)
    state = _validate_schedule_state(
        _read_json_object(state_path, label="discovery_state"),
        claim=claim,
        contract=frozen_contract,
        response_state=response_state,
    )
    return DiscoveryBundle(
        response=response_state,
        state=state,
        resumed=response_was_existing,
        network_requests=network_requests,
    )
