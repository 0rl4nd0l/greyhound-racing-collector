"""Read dog-profile identity separately from a race-specific runner-entry ID."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit


class TheDogsRunnerIdentityError(ValueError):
    """The retained row does not contain an unambiguous dog-profile identity."""


class TheDogsProfileIdentityMissing(TheDogsRunnerIdentityError):
    """A legacy row contains no profile-identity evidence at all."""


def extract_thedogs_profile_identity(
    row: Any, *, require_profile_link: bool
) -> str:
    """Return the row's numeric ``data-dog-id``, checking any profile links.

    Pre-race rows can omit profile links. Such callers must independently verify
    the retained race page and its same-row race-entry identity. Official-result
    callers require a corroborating profile link. This function does not equate
    a dog-profile ID with a race-specific ``data-runner-id`` or prove a box move.
    """
    values = [
        str(element.get("data-dog-id") or "").strip()
        for element in [row, *row.select("[data-dog-id]")]
        if element.has_attr("data-dog-id")
    ]
    if not values:
        for anchor in row.select('a[href]'):
            try:
                path = urlsplit(str(anchor.get('href') or '')).path.lower()
            except ValueError:
                raise TheDogsRunnerIdentityError('dog_profile_link_invalid') from None
            if path.startswith('/dogs/') and not path.startswith('/dogs/runner/'):
                raise TheDogsRunnerIdentityError('dog_profile_attribute_missing_for_link')
        raise TheDogsProfileIdentityMissing('dog_profile_identity_absent')
    if (
        any(re.fullmatch(r"[1-9][0-9]*", value) is None for value in values)
        or len(set(values)) != 1
    ):
        raise TheDogsRunnerIdentityError("dog_profile_id_missing_invalid_or_ambiguous")
    dog_id = values[0]
    profile_ids: set[str] = set()
    for anchor in row.select("a[href]"):
        href = str(anchor.get("href") or "").strip()
        try:
            parsed = urlsplit(href)
        except ValueError:
            raise TheDogsRunnerIdentityError("dog_profile_link_invalid") from None
        if not parsed.path.lower().startswith("/dogs/"):
            continue
        if parsed.path.lower().startswith("/dogs/runner/"):
            # This is a different ID namespace, independently checked by the
            # native canonical runner-entry parser.
            continue
        match = re.fullmatch(r"/dogs/([1-9][0-9]*)(?:/[^/?#]+)?/?", parsed.path)
        if (
            match is None
            or parsed.query
            or parsed.fragment
            or parsed.scheme not in ("", "https")
            or (
                parsed.netloc
                and parsed.netloc.lower() not in ("thedogs.com.au", "www.thedogs.com.au")
            )
            or (parsed.scheme and not parsed.netloc)
        ):
            raise TheDogsRunnerIdentityError("dog_profile_link_invalid")
        profile_ids.add(match.group(1))
    if profile_ids and profile_ids != {dog_id}:
        raise TheDogsRunnerIdentityError("dog_profile_link_identity_conflict")
    if require_profile_link and not profile_ids:
        raise TheDogsRunnerIdentityError("dog_profile_link_missing")
    return dog_id
