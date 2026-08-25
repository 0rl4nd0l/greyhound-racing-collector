"""Strict structured equivalence for evidence-proven greyhound race aliases."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from config.venue_mapping import VENUE_MAPPING, normalize_venue
from utils.csv_metadata import (
    canonical_thedogs_race_identity,
    canonical_thedogs_venue_identity,
)

VENUE_CODE_PATTERN = r"[A-Z0-9_]+(?:-[A-Z0-9_]+)*"
NON_RACING_VENUE_IDENTITIES = frozenset({"UNKNOWN", "TEST_VEN", "RACE"})


def race_id_parts(race_id: Any) -> tuple[int, str, date] | None:
    match = re.fullmatch(
        r"Race\s+([0-9]{1,2})\s+-\s+(.+?)\s+-\s+([0-9]{4}-[0-9]{2}-[0-9]{2})",
        str(race_id or "").strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        race_date = date.fromisoformat(match.group(3))
    except ValueError:
        return None
    return int(match.group(1)), match.group(2).strip().upper(), race_date


def configured_venue_identity(value: Any) -> str | None:
    """Resolve only venue spellings proved by the checked-in canonical map."""

    if not isinstance(value, str):
        return None
    raw = value.strip().upper()
    if not raw or re.fullmatch(VENUE_CODE_PATTERN, raw) is None:
        return None
    candidates = {
        raw,
        raw.replace("-", " "),
        raw.replace("-", "_"),
        raw.replace("_", " "),
        raw.replace("_", "-"),
    }
    normalized = {
        normalize_venue(candidate)
        for candidate in candidates
        if candidate in VENUE_MAPPING
    }
    if len(normalized) != 1:
        return None
    configured = next(iter(normalized))
    canonical = canonical_thedogs_venue_identity(configured)
    if (
        configured in NON_RACING_VENUE_IDENTITIES
        or canonical in NON_RACING_VENUE_IDENTITIES
    ):
        return None
    return canonical


def race_identity_equivalent(
    caller_race_id: Any,
    evidence_race_id: Any,
    *,
    source_url: Any,
) -> bool:
    """Bind caller and sealed-source aliases by exact structured identity."""

    caller = race_id_parts(caller_race_id)
    evidence = race_id_parts(evidence_race_id)
    source = canonical_thedogs_race_identity(source_url)
    if caller is None or evidence is None or source is None:
        return False
    caller_number, caller_venue, caller_date = caller
    evidence_number, evidence_venue, evidence_date = evidence
    venues = (
        configured_venue_identity(caller_venue),
        configured_venue_identity(evidence_venue),
        configured_venue_identity(source["venue_slug"]),
    )
    return bool(
        all(venue is not None for venue in venues)
        and len(set(venues)) == 1
        and caller_date == evidence_date
        and caller_date.isoformat() == source["race_date"]
        and caller_number == evidence_number
        and caller_number == source["race_number"]
    )


__all__ = [
    "configured_venue_identity",
    "race_id_parts",
    "race_identity_equivalent",
]
