"""Exact read-only receipt readiness shared by predictor and operator admission."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
    runner_set_sha256,
)
from src.predictor.on_demand import PredictionBlocked, receipt_from_handoff
from utils.race_identity_equivalence import (
    configured_venue_identity,
    race_id_parts,
    race_identity_equivalent,
)


@dataclass(frozen=True, slots=True)
class ExactReceiptReady:
    handoff: Mapping[str, Any]
    receipt: Mapping[str, Any]
    protocol_chain: Mapping[str, str]
    protocol_members: Mapping[str, bytes]


def _expected_runner_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    expected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
        box = row.get("box_number", row.get("box"))
        name = row.get("display_name", row.get("dog_name", row.get("name")))
        identity = row.get("identity")
        expected.append(
            {"box_number": box, "dog_name": name, "identity": identity}
        )
    try:
        result = runner_set_sha256(expected)
    except ProtocolRejected as exc:
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS") from exc
    if result is None:
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    return result


def _sportsbet_source_matches(source_url: Any, race_id: str) -> bool:
    try:
        parsed = urlparse(str(source_url or "").strip())
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower()
        not in {"sportsbet.com.au", "www.sportsbet.com.au"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return False
    match = re.fullmatch(
        r"/(?:betting/)?greyhound-racing/(?:australia-nz/)?([^/]+)/race-([0-9]{1,2})-[0-9]+/?",
        parsed.path,
        flags=re.IGNORECASE,
    )
    caller = race_id_parts(race_id)
    if match is None or caller is None:
        return False
    number, venue, _ = caller
    source_venue = configured_venue_identity(match.group(1).upper())
    return bool(
        int(match.group(2)) == number
        and source_venue is not None
        and source_venue == configured_venue_identity(venue)
    )


def _snapshot(
    protocol: ManualPredictionCollectorProtocol, handoff: Mapping[str, Any]
) -> tuple[Mapping[str, str], Mapping[str, bytes], Mapping[str, Any]]:
    public = {
        str(key): value
        for key, value in handoff.items()
        if not str(key).startswith("_")
    }
    try:
        if handoff.get("_scheduled_exact_receipt") is True:
            chain, members, artifacts = protocol.snapshot_collector_exact_handoff(
                public
            )
            return (
                chain,
                members,
                {
                    **handoff,
                    "_report_bytes": artifacts["report"],
                    "_form_bytes": artifacts["form"],
                    "_sidecar_bytes": artifacts["sidecar"],
                },
            )
        chain, members = protocol.snapshot_authenticated_handoff(public)
        return chain, members, dict(handoff)
    except ProtocolRejected as exc:
        raise PredictionBlocked(
            "COLLECTOR_PROTOCOL_INVALID", reason=exc.code
        ) from exc


def _validate_request_member(
    members: Mapping[str, bytes],
    *,
    race_id: str,
    race_url: str,
    jump_timestamp: str,
    expected_runner_hash: str,
) -> None:
    raw = members.get("request")
    if raw is None:
        return
    try:
        request = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionBlocked("COLLECTOR_PROTOCOL_INVALID") from exc
    race = request.get("race") if isinstance(request, Mapping) else None
    if (
        not isinstance(race, Mapping)
        or not race_identity_equivalent(
            race_id, race.get("race_id"), source_url=race.get("url")
        )
        or race.get("url") != race_url
        or race.get("jump_timestamp") != jump_timestamp
        or request.get("expected_runner_set_sha256") != expected_runner_hash
    ):
        raise PredictionBlocked("RECEIPT_INVALID")


def discover_exact_receipt_ready(
    *,
    protocol: ManualPredictionCollectorProtocol,
    race_id: str,
    race_url: str,
    jump: datetime,
    expected_runners: Sequence[Mapping[str, Any]],
    current_time: datetime,
    receipt_max_age_seconds: int,
    minimum_prejump_margin_seconds: float,
    completion_clock: Callable[[], datetime] | None = None,
    margin_phase: str = "reuse_validation_and_scoring",
) -> ExactReceiptReady:
    """Discover and validate one exact receipt before any one-shot allocation."""

    if (
        current_time.tzinfo is None
        or current_time.utcoffset() is None
        or jump.tzinfo is None
        or jump.utcoffset() is None
    ):
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    if not race_identity_equivalent(race_id, race_id, source_url=race_url):
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
    expected_runner_hash = _expected_runner_hash(expected_runners)
    try:
        manual = protocol.discover_exact_handoff(
            race_id=race_id,
            current_time=current_time,
            max_age_seconds=receipt_max_age_seconds,
        )
        scheduled = protocol.discover_collector_exact_handoff(
            race_id=race_id,
            current_time=current_time,
            max_age_seconds=receipt_max_age_seconds,
        )
    except ProtocolRejected as exc:
        raise PredictionBlocked(
            "COLLECTOR_PROTOCOL_INVALID", reason=exc.code
        ) from exc
    candidates = [value for value in (manual, scheduled) if value is not None]
    if not candidates:
        raise PredictionBlocked("RECEIPT_UNAVAILABLE")
    remaining = (jump - current_time).total_seconds()
    if remaining <= minimum_prejump_margin_seconds:
        raise PredictionBlocked(
            "INSUFFICIENT_PREJUMP_MARGIN",
            phase=margin_phase,
            remaining_seconds=remaining,
            required_seconds=minimum_prejump_margin_seconds,
        )
    selected = max(
        candidates,
        key=lambda value: datetime.fromisoformat(str(value["append_timestamp"])),
    )
    chain, members, selected = _snapshot(protocol, selected)
    if selected.get("race_id") != race_id:
        raise PredictionBlocked("RECEIPT_INVALID")
    receipt, _, _, _ = receipt_from_handoff(
        selected,
        current_time=current_time,
        max_age_seconds=receipt_max_age_seconds,
    )
    if receipt.get("runner_set_sha256") != expected_runner_hash:
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    if not _sportsbet_source_matches(receipt.get("source_url"), race_id):
        raise PredictionBlocked("RECEIPT_INVALID")
    if selected.get("schema_version") == "on_demand_verified_collector_capture_v2":
        source_race = selected.get("race")
        if (
            not isinstance(source_race, Mapping)
            or not race_identity_equivalent(
                race_id, source_race.get("race_id"), source_url=source_race.get("url")
            )
            or source_race.get("url") != race_url
            or source_race.get("jump_timestamp") != jump.isoformat()
            or selected.get("runner_set_sha256") != expected_runner_hash
        ):
            raise PredictionBlocked("RECEIPT_INVALID")
    _validate_request_member(
        members,
        race_id=race_id,
        race_url=race_url,
        jump_timestamp=jump.isoformat(),
        expected_runner_hash=expected_runner_hash,
    )
    completed_at = completion_clock() if completion_clock is not None else current_time
    if completed_at.tzinfo is None or completed_at.utcoffset() is None:
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    receipt, _, _, _ = receipt_from_handoff(
        selected,
        current_time=completed_at,
        max_age_seconds=receipt_max_age_seconds,
    )
    completed_remaining = (jump - completed_at).total_seconds()
    if completed_remaining <= minimum_prejump_margin_seconds:
        raise PredictionBlocked(
            "INSUFFICIENT_PREJUMP_MARGIN",
            phase=margin_phase,
            remaining_seconds=completed_remaining,
            required_seconds=minimum_prejump_margin_seconds,
        )
    return ExactReceiptReady(selected, receipt, chain, members)


__all__ = ["ExactReceiptReady", "discover_exact_receipt_ready"]
