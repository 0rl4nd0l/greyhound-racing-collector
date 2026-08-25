"""Read-only exact-receipt admission for forward prediction journal races.

This module performs one bounded observation of already-published receipt state.
It neither acquires evidence nor owns polling.  The supplied invocation callback
is the sole job/output/attempt boundary and is called only for fully READY races.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
    runner_set_sha256,
)
from scripts.predict_market_form_residual import (
    _configured_venue_identity,
    _race_identity_equivalent,
)
from scripts.predict_race_now import (
    _parse_race_jump_datetime,
    _request_expected_runners,
    _request_race,
    _selected_protocol_chain,
    stable_race_id,
)
from src.predictor.on_demand import PredictionBlocked, receipt_from_handoff

READY = "READY"
PENDING_RECEIPT = "PENDING_RECEIPT"
PREFLIGHT_EXCLUDED = "PREFLIGHT_EXCLUDED"
ALREADY_RECORDED = "ALREADY_RECORDED"

_MAX_INDEX_RACES = 64


@dataclass(frozen=True, slots=True)
class ReceiptAdmission:
    race_id: str
    jump_timestamp: str | None
    state: str
    reason: str | None


def _sportsbet_source_matches_race(source_url: Any, race_id: str) -> bool:
    """Require one exact supported Sportsbet venue/race URL identity."""

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
        r"/greyhound-racing/(?:australia-nz/)?([^/]+)/race-([0-9]{1,2})-[0-9]+/?",
        parsed.path,
        flags=re.IGNORECASE,
    )
    race_match = re.fullmatch(
        r"Race ([0-9]{1,2}) - (.+) - [0-9]{4}-[0-9]{2}-[0-9]{2}",
        race_id,
    )
    if match is None or race_match is None:
        return False
    return bool(
        int(match.group(2)) == int(race_match.group(1))
        and _configured_venue_identity(match.group(1).upper()) is not None
        and _configured_venue_identity(match.group(1).upper())
        == _configured_venue_identity(race_match.group(2).upper())
    )


def validate_exact_receipt_for_race(
    race: Mapping[str, Any],
    handoff: Mapping[str, Any],
    *,
    current_time: datetime,
    receipt_max_age_seconds: int,
) -> Mapping[str, Any]:
    """Apply predictor receipt validation plus exact index-race binding."""

    if current_time.tzinfo is None or current_time.utcoffset() is None:
        raise PredictionBlocked("CURRENT_TIME_TIMEZONE_MISSING")
    race_id = str(stable_race_id(race) or "")
    jump = _parse_race_jump_datetime(race, now=current_time)
    if not race_id or jump is None:
        raise PredictionBlocked("EXACT_RACE_IDENTITY_UNAVAILABLE")
    target_race = _request_race(race, race_id=race_id, jump=jump)
    target_runners = _request_expected_runners(race)
    if not target_runners:
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    if handoff.get("race_id") != race_id:
        raise PredictionBlocked("RECEIPT_INVALID")

    receipt, _, _, _ = receipt_from_handoff(
        handoff,
        current_time=current_time,
        max_age_seconds=receipt_max_age_seconds,
    )
    expected_runner_hash = runner_set_sha256(
        [
            {
                "box_number": row["box_number"],
                "dog_name": row["display_name"],
                "identity": row["identity"],
            }
            for row in target_runners
        ]
    )
    if (
        expected_runner_hash is None
        or receipt.get("runner_set_sha256") != expected_runner_hash
    ):
        raise PredictionBlocked("RUNNER_SET_AMBIGUOUS")
    if not _sportsbet_source_matches_race(receipt.get("source_url"), race_id):
        raise PredictionBlocked("RECEIPT_INVALID")

    if handoff.get("schema_version") == "on_demand_verified_collector_capture_v2":
        source_race = handoff.get("race")
        if not isinstance(source_race, Mapping):
            raise PredictionBlocked("RECEIPT_INVALID")
        if (
            not _race_identity_equivalent(
                race_id,
                source_race.get("race_id"),
                source_url=source_race.get("url"),
            )
            or source_race.get("jump_timestamp") != target_race["jump_timestamp"]
            or handoff.get("runner_set_sha256") != expected_runner_hash
        ):
            raise PredictionBlocked("RECEIPT_INVALID")
    return receipt


def _preflight_one(
    race: Mapping[str, Any],
    *,
    protocol: ManualPredictionCollectorProtocol,
    current_time: datetime,
    receipt_max_age_seconds: int,
    minimum_prejump_margin_seconds: float,
) -> ReceiptAdmission:
    race_id = str(stable_race_id(race) or "")
    jump = _parse_race_jump_datetime(race, now=current_time)
    if not race_id or jump is None:
        return ReceiptAdmission(
            race_id, None, PREFLIGHT_EXCLUDED, "EXACT_RACE_IDENTITY_UNAVAILABLE"
        )
    jump_text = jump.isoformat()
    remaining = (jump - current_time).total_seconds()
    if remaining <= minimum_prejump_margin_seconds:
        return ReceiptAdmission(
            race_id, jump_text, PREFLIGHT_EXCLUDED, "INSUFFICIENT_PREJUMP_MARGIN"
        )
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
        candidates = [value for value in (manual, scheduled) if value is not None]
        if not candidates:
            return ReceiptAdmission(
                race_id, jump_text, PENDING_RECEIPT, "RECEIPT_UNAVAILABLE"
            )
        selected = max(
            candidates,
            key=lambda value: datetime.fromisoformat(str(value["append_timestamp"])),
        )
        # Snapshot validation is read-only and is also performed by the predictor.
        _, _, selected = _selected_protocol_chain(protocol, selected)
        validate_exact_receipt_for_race(
            race,
            selected,
            current_time=current_time,
            receipt_max_age_seconds=receipt_max_age_seconds,
        )
    except ProtocolRejected as exc:
        return ReceiptAdmission(race_id, jump_text, PREFLIGHT_EXCLUDED, exc.code)
    except PredictionBlocked as exc:
        return ReceiptAdmission(race_id, jump_text, PREFLIGHT_EXCLUDED, exc.code)
    return ReceiptAdmission(race_id, jump_text, READY, None)


def observe_receipt_ready_races(
    races: Sequence[Mapping[str, Any]],
    *,
    protocol: ManualPredictionCollectorProtocol,
    current_time: datetime,
    receipt_max_age_seconds: int,
    minimum_prejump_margin_seconds: float,
    recorded_race_ids: AbstractSet[str],
    invoke_ready: Callable[[Mapping[str, Any]], Any],
) -> tuple[ReceiptAdmission, ...]:
    """Observe once, then invoke only independently READY unrecorded races."""

    if (
        current_time.tzinfo is None
        or current_time.utcoffset() is None
        or type(receipt_max_age_seconds) is not int
        or receipt_max_age_seconds <= 0
        or isinstance(minimum_prejump_margin_seconds, bool)
        or not isinstance(minimum_prejump_margin_seconds, (int, float))
        or minimum_prejump_margin_seconds <= 0
        or len(races) > _MAX_INDEX_RACES
    ):
        raise ValueError("invalid bounded forward-journal observation")
    race_ids = [str(stable_race_id(race) or "") for race in races]
    duplicates = {
        race_id for race_id, count in Counter(race_ids).items() if race_id and count > 1
    }
    admissions: list[ReceiptAdmission] = []
    ready: list[tuple[datetime, str, Mapping[str, Any]]] = []
    for race, race_id in zip(races, race_ids, strict=True):
        jump = _parse_race_jump_datetime(race, now=current_time)
        if race_id in recorded_race_ids:
            admissions.append(
                ReceiptAdmission(
                    race_id,
                    jump.isoformat() if jump is not None else None,
                    ALREADY_RECORDED,
                    "NO_RETRY_OR_BACKFILL",
                )
            )
            continue
        if not race_id or race_id in duplicates:
            admissions.append(
                ReceiptAdmission(
                    race_id,
                    jump.isoformat() if jump is not None else None,
                    PREFLIGHT_EXCLUDED,
                    "RACE_ID_MISSING_OR_AMBIGUOUS",
                )
            )
            continue
        admission = _preflight_one(
            race,
            protocol=protocol,
            current_time=current_time,
            receipt_max_age_seconds=receipt_max_age_seconds,
            minimum_prejump_margin_seconds=minimum_prejump_margin_seconds,
        )
        admissions.append(admission)
        if admission.state == READY and jump is not None:
            ready.append((jump, race_id, race))
    for _, _, race in sorted(ready, key=lambda value: (value[0], value[1])):
        invoke_ready(race)
    return tuple(admissions)


__all__ = [
    "ALREADY_RECORDED",
    "PENDING_RECEIPT",
    "PREFLIGHT_EXCLUDED",
    "READY",
    "ReceiptAdmission",
    "observe_receipt_ready_races",
    "validate_exact_receipt_for_race",
]
