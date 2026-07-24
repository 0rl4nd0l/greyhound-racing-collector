"""Deterministic, source-neutral race and dog identity decisions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .domain import DogId, DomainValidationError, IdentityTier, RaceId


def _id(prefix: str, namespace: str, value: str) -> str:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def dog_id_for_registration(authority: str, registration_id: str) -> DogId:
    return DogId(_id("dog", authority.strip().lower(), registration_id.strip()))


def normalize_dog_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()


@dataclass(frozen=True, slots=True)
class DogIdentityDecision:
    tier: IdentityTier
    dog_id: DogId | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.tier, IdentityTier):
            raise DomainValidationError("tier must be an IdentityTier")
        if self.dog_id is not None and not isinstance(self.dog_id, DogId):
            raise DomainValidationError("dog_id must be a DogId or None")
        if self.tier is IdentityTier.AMBIGUOUS:
            if self.dog_id is not None:
                raise DomainValidationError("AMBIGUOUS requires dog_id is None")
        elif self.dog_id is None:
            raise DomainValidationError(f"{self.tier.name} requires a real DogId")


def resolve_dog_identity(
    *,
    source: str,
    registration_authority: str | None = None,
    registration_id: str | None,
    name: str,
    name_candidates: Iterable[DogId] = (),
) -> DogIdentityDecision:
    if registration_id and registration_id.strip():
        if registration_authority is None or not registration_authority.strip():
            raise DomainValidationError(
                "registration_authority is required for an authoritative registration ID"
            )
        return DogIdentityDecision(
            IdentityTier.AUTHORITATIVE,
            dog_id_for_registration(registration_authority, registration_id),
            "authoritative registration alias",
        )
    candidates = tuple(dict.fromkeys(name_candidates))
    if len(candidates) == 1:
        return DogIdentityDecision(
            IdentityTier.HIGH_CONFIDENCE_PROVISIONAL,
            candidates[0],
            "one existing normalized-name candidate",
        )
    if not candidates and normalize_dog_name(name):
        return DogIdentityDecision(
            IdentityTier.HIGH_CONFIDENCE_PROVISIONAL,
            DogId(_id("dog", f"provisional:{source.strip().lower()}", normalize_dog_name(name))),
            "new source-scoped provisional name",
        )
    return DogIdentityDecision(IdentityTier.AMBIGUOUS, None, "multiple identity candidates")
