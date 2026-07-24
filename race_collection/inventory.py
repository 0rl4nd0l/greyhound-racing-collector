"""Pure adapters for authoritative expected race-programme payloads."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Protocol

from .domain import ProgrammeRaceCandidate


class ProgrammeAdapter(Protocol):
    source: str

    def parse(self, payload: bytes) -> tuple[ProgrammeRaceCandidate, ...]: ...


class JsonProgrammeAdapter:
    """Minimal normalized JSON boundary; fetching remains outside this package."""

    def __init__(self, source: str):
        self.source = source

    def parse(self, payload: bytes) -> tuple[ProgrammeRaceCandidate, ...]:
        document = json.loads(payload)
        races = []
        for item in document["races"]:
            source_id = str(item["source_race_id"])
            races.append(
                ProgrammeRaceCandidate(
                    source=self.source,
                    source_race_id=source_id,
                    venue=str(item["venue"]),
                    race_number=int(item["race_number"]),
                    scheduled_jump=datetime.fromisoformat(item["scheduled_jump"]),
                )
            )
        identities = {(race.venue.casefold(), race.race_number) for race in races}
        if len(identities) != len(races):
            raise ValueError("programme contains duplicate venue/race-number entries")
        return tuple(races)
