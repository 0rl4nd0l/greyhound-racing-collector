"""Pre-jump Sportsbet metadata collection for upcoming race sidecars.

This module reads only pre-race Sportsbet event metadata. It does not write to
the database and fails closed unless venue, race number, date, and jump time
match the requested race.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlencode, urlparse
from zoneinfo import ZoneInfo

from utils.csv_metadata import normalize_track_condition_text
from utils.http_client import get_shared_session
from utils.prejump_weather import (
    _parse_race_datetime as _parse_prejump_race_datetime,
    venue_weather_location,
)


SPORTSBET_NEXT_EVENTS_ENDPOINT = (
    "https://www.sportsbet.com.au/apigw/sportsbook-racing/"
    "Sportsbook/Racing/NextEvents"
)
SPORTSBET_NEXT_EVENTS_PARAMS = {
    "racingFilters": ",".join(
        [
            "HR_DOMESTIC",
            "HR_INTERNATIONAL",
            "GH_DOMESTIC",
            "GH_INTERNATIONAL",
            "HA_DOMESTIC",
            "HA_INTERNATIONAL",
        ]
    ),
    "groupByFilters": "true",
}
SPORTSBET_NEXT_EVENTS_SOURCE_URL = (
    SPORTSBET_NEXT_EVENTS_ENDPOINT + "?" + urlencode(SPORTSBET_NEXT_EVENTS_PARAMS)
)


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _race_url_venue_slug(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        path = urlparse(text).path
    except Exception:
        path = text
    parts = [part for part in path.strip("/").split("/") if part]
    try:
        racing_idx = parts.index("racing")
    except ValueError:
        return None
    if len(parts) <= racing_idx + 1:
        return None
    return parts[racing_idx + 1]


def _candidate_venue_names(race_info: Mapping[str, Any]) -> set[str]:
    names = {
        str(race_info.get("venue") or "").strip(),
        str(race_info.get("venue_name") or "").strip(),
    }
    slug = _race_url_venue_slug(
        race_info.get("url") or race_info.get("race_url") or race_info.get("source_url")
    )
    if slug:
        names.add(slug)
        names.add(slug.replace("-", " "))
    location = venue_weather_location(
        race_info.get("venue_name") or race_info.get("venue")
    )
    if location is not None:
        names.add(location.venue_name)
        if not location.venue_name.lower().endswith("park"):
            names.add(f"{location.venue_name} Park")
    return {_normalise_name(name) for name in names if name}


def _parse_race_datetime(race_info: Mapping[str, Any], timezone_name: str) -> datetime | None:
    race_date = str(
        race_info.get("date") or race_info.get("race_date") or ""
    ).strip()[:10]
    race_time = (
        str(race_info.get("race_time") or race_info.get("jump_time") or "")
        .strip()
        .upper()
        .replace(".", ":")
    )
    if not race_date or not race_time:
        return None
    return _parse_prejump_race_datetime(
        race_date,
        race_time,
        timezone_name,
        source_timezone=(
            race_info.get("race_time_timezone")
            or race_info.get("jump_time_timezone")
            or race_info.get("display_timezone")
        ),
    )


def _event_datetime(event: Mapping[str, Any], timezone_name: str) -> datetime | None:
    start_time = event.get("startTime")
    try:
        start_epoch = float(start_time)
    except Exception:
        return None
    return datetime.fromtimestamp(start_epoch, timezone.utc).astimezone(
        ZoneInfo(timezone_name)
    )


def _event_is_greyhound(event: Mapping[str, Any]) -> bool:
    event_type = str(event.get("type") or "").strip().lower()
    class_name = str(event.get("className") or "").strip().lower()
    class_id = str(event.get("classId") or "").strip()
    return event_type == "greyhound" or class_id == "4" or "greyhound" in class_name


def _matching_event(
    events: list[Any],
    race_info: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, list[str]]:
    race_number = _parse_int(race_info.get("race_number"))
    if race_number is None:
        return None, ["sportsbet_race_number_missing"]
    location = venue_weather_location(
        race_info.get("venue_name") or race_info.get("venue")
    )
    if location is None:
        return None, ["sportsbet_venue_timezone_unmapped"]
    race_dt = _parse_race_datetime(race_info, location.timezone)
    if race_dt is None:
        return None, ["sportsbet_race_time_unparseable"]

    candidate_venues = _candidate_venue_names(race_info)
    matches: list[tuple[float, Mapping[str, Any]]] = []
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        if not _event_is_greyhound(raw_event):
            continue
        if _parse_int(raw_event.get("raceNumber")) != race_number:
            continue
        event_venue = _normalise_name(raw_event.get("competitionName"))
        if not event_venue or event_venue not in candidate_venues:
            continue
        event_dt = _event_datetime(raw_event, location.timezone)
        if event_dt is None:
            continue
        if event_dt.date() != race_dt.date():
            continue
        delta_seconds = abs((event_dt - race_dt).total_seconds())
        if delta_seconds <= 20 * 60:
            matches.append((delta_seconds, raw_event))

    if not matches:
        return None, ["sportsbet_matching_pre_race_event_not_found"]
    matches.sort(key=lambda item: item[0])
    return matches[0][1], []


def collect_sportsbet_track_metadata(
    race_info: Mapping[str, Any],
    *,
    session: Any = None,
) -> dict[str, Any]:
    """Return source-backed Sportsbet track metadata, or a rejection payload."""

    client = session or get_shared_session()
    response = None
    try:
        response = client.get(
            SPORTSBET_NEXT_EVENTS_ENDPOINT,
            params=SPORTSBET_NEXT_EVENTS_PARAMS,
            headers={
                "Accept": "application/json,text/plain,*/*",
                "Referer": "https://www.sportsbet.com.au/betting/upcoming-sports",
                "User-Agent": "Mozilla/5.0",
            },
            timeout=20,
        )
        response.raise_for_status()
        events = response.json()
    except Exception as exc:
        return {
            "weather_track_metadata_source": "sportsbet_pre_race_page",
            "weather_track_metadata_source_url": SPORTSBET_NEXT_EVENTS_SOURCE_URL,
            "rejected_weather_track_metadata_sources": [
                f"sportsbet_source_fetch_failed:{type(exc).__name__}"
            ],
        }
    finally:
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    if not isinstance(events, list):
        return {
            "weather_track_metadata_source": "sportsbet_pre_race_page",
            "weather_track_metadata_source_url": SPORTSBET_NEXT_EVENTS_SOURCE_URL,
            "rejected_weather_track_metadata_sources": [
                "sportsbet_source_unexpected_payload"
            ],
        }

    event, rejected = _matching_event(events, race_info)
    if event is None:
        return {
            "weather_track_metadata_source": "sportsbet_pre_race_page",
            "weather_track_metadata_source_url": SPORTSBET_NEXT_EVENTS_SOURCE_URL,
            "rejected_weather_track_metadata_sources": rejected,
        }

    track_status = normalize_track_condition_text(event.get("trackStatus"))
    if not track_status:
        return {
            "weather_track_metadata_source": "sportsbet_pre_race_page",
            "weather_track_metadata_source_url": SPORTSBET_NEXT_EVENTS_SOURCE_URL,
            "rejected_weather_track_metadata_sources": [
                "sportsbet_track_status_missing_or_placeholder"
            ],
        }

    return {
        "track_condition": track_status,
        "weather_track_metadata_source": "sportsbet_pre_race_page",
        "weather_track_metadata_source_url": SPORTSBET_NEXT_EVENTS_SOURCE_URL,
        "weather_track_metadata_is_leakage_safe": True,
        "weather_track_metadata_detail": {
            "provider": "sportsbet",
            "event_id": event.get("id"),
            "competition_name": event.get("competitionName"),
            "race_number": event.get("raceNumber"),
            "start_time": event.get("startTime"),
            "distance": event.get("distance"),
            "track_status": event.get("trackStatus"),
        },
    }
