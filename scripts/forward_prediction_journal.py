"""Receipt-ready outer admission for the durable forward prediction journal."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
)
from race_collection.synchronous_manual_capture import LatencyBudget
from src.predictor.on_demand import PredictionBlocked
from src.predictor.receipt_preflight import discover_exact_receipt_ready

READY = "READY"
PENDING_RECEIPT = "PENDING_RECEIPT"
PREFLIGHT_EXCLUDED = "PREFLIGHT_EXCLUDED"
ALREADY_RECORDED = "ALREADY_RECORDED"
_MAX_INDEX_RACES = 64


@dataclass(frozen=True, slots=True)
class ReceiptPreflightPolicy:
    receipt_max_age_seconds: int
    minimum_prejump_margin_seconds: float

    @classmethod
    def from_prediction_config(
        cls, config: Mapping[str, Any]
    ) -> ReceiptPreflightPolicy:
        try:
            bundle = config["bundle"]
            maximum_age = bundle["receipt_max_age_seconds"]
            budget = LatencyBudget.from_config(bundle["latency_budget"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid receipt preflight config") from exc
        if type(maximum_age) is not int or maximum_age <= 0:
            raise ValueError("invalid receipt preflight config")
        return cls(maximum_age, budget.reuse_margin_seconds)


@dataclass(frozen=True, slots=True)
class ReceiptAdmission:
    race_id: str
    jump_timestamp: str | None
    state: str
    reason: str | None


def _jump(race: Mapping[str, Any]) -> datetime | None:
    value = race.get("jump_datetime", race.get("jump_timestamp"))
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def preflight_race_receipt(
    race: Mapping[str, Any],
    *,
    protocol: ManualPredictionCollectorProtocol,
    current_time: datetime,
    policy: ReceiptPreflightPolicy,
    completion_clock: Callable[[], datetime] | None = None,
) -> ReceiptAdmission:
    """Return one race-local state without allocating a job or output path."""

    race_id = race.get("race_id")
    jump = _jump(race)
    if not isinstance(race_id, str) or not race_id or jump is None:
        return ReceiptAdmission(
            str(race_id or ""),
            jump.isoformat() if jump is not None else None,
            PREFLIGHT_EXCLUDED,
            "EXACT_RACE_IDENTITY_UNAVAILABLE",
        )
    race_url = race.get("race_url", race.get("url"))
    runners = race.get("runners", race.get("participants"))
    if (
        current_time.tzinfo is None
        or current_time.utcoffset() is None
        or not isinstance(race_url, str)
        or isinstance(runners, (str, bytes, bytearray))
        or not isinstance(runners, Sequence)
    ):
        return ReceiptAdmission(
            race_id,
            jump.isoformat(),
            PREFLIGHT_EXCLUDED,
            "EXACT_RACE_IDENTITY_UNAVAILABLE",
        )
    remaining = (jump - current_time).total_seconds()
    if remaining <= policy.minimum_prejump_margin_seconds:
        return ReceiptAdmission(
            race_id,
            jump.isoformat(),
            PREFLIGHT_EXCLUDED,
            "INSUFFICIENT_PREJUMP_MARGIN",
        )
    try:
        discover_exact_receipt_ready(
            protocol=protocol,
            race_id=race_id,
            race_url=race_url,
            jump=jump,
            expected_runners=runners,
            current_time=current_time,
            receipt_max_age_seconds=policy.receipt_max_age_seconds,
            minimum_prejump_margin_seconds=policy.minimum_prejump_margin_seconds,
            completion_clock=completion_clock,
        )
    except PredictionBlocked as exc:
        state = (
            PENDING_RECEIPT
            if exc.code == "RECEIPT_UNAVAILABLE"
            else PREFLIGHT_EXCLUDED
        )
        return ReceiptAdmission(race_id, jump.isoformat(), state, exc.code)
    return ReceiptAdmission(race_id, jump.isoformat(), READY, None)


def observe_receipt_ready_races(
    races: Sequence[Mapping[str, Any]],
    *,
    protocol: ManualPredictionCollectorProtocol,
    policy: ReceiptPreflightPolicy,
    clock: Callable[[], datetime],
    recorded_race_ids: AbstractSet[str],
    invoke_ready: Callable[[Mapping[str, Any]], Any],
) -> tuple[ReceiptAdmission, ...]:
    """Observe once and revalidate each READY race at its allocation boundary."""

    if len(races) > _MAX_INDEX_RACES:
        raise ValueError("forward-journal index is unbounded")
    race_ids = [str(race.get("race_id") or "") for race in races]
    duplicates = {
        race_id for race_id, count in Counter(race_ids).items() if race_id and count > 1
    }
    admissions: list[ReceiptAdmission] = []
    ready: list[tuple[datetime, str, int, Mapping[str, Any]]] = []
    for race, race_id in zip(races, race_ids, strict=True):
        jump = _jump(race)
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
        admission = preflight_race_receipt(
            race,
            protocol=protocol,
            current_time=clock(),
            policy=policy,
            completion_clock=clock,
        )
        offset = len(admissions)
        admissions.append(admission)
        if admission.state == READY and jump is not None:
            ready.append((jump, race_id, offset, race))
    for _, _, offset, race in sorted(ready, key=lambda value: (value[0], value[1])):
        admission = preflight_race_receipt(
            race,
            protocol=protocol,
            current_time=clock(),
            policy=policy,
            completion_clock=clock,
        )
        admissions[offset] = admission
        if admission.state == READY:
            invoke_ready(race)
    return tuple(admissions)


__all__ = [
    "ALREADY_RECORDED",
    "PENDING_RECEIPT",
    "PREFLIGHT_EXCLUDED",
    "READY",
    "ReceiptAdmission",
    "ReceiptPreflightPolicy",
    "observe_receipt_ready_races",
    "preflight_race_receipt",
]
