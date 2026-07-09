"""Source-backed Expert Form page metadata extraction.

This module parses the rendered TheDogs expert-form page into sidecar metadata.
It does not write to the database and it does not make these fields model-active.
Callers must keep the pre-jump timing gate intact before treating the payload as
leakage-safe.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from utils.prejump_weather import venue_weather_location

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _texts(element: Any) -> list[str]:
    if element is None:
        return []
    return [
        _clean(text)
        for text in element.get_text("\n", strip=True).split("\n")
        if _clean(text)
    ]


def _safe_int(value: Any) -> int | None:
    text = re.sub(r"[^\d-]+", "", str(value or ""))
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _safe_float(value: Any) -> float | None:
    text = re.sub(r"[^0-9.-]+", "", str(value or ""))
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_record(value: str) -> dict[str, int | None]:
    match = re.search(r"(\d+)\s*:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", value or "")
    if not match:
        return {"starts": None, "wins": None, "seconds": None, "thirds": None}
    return {
        "starts": int(match.group(1)),
        "wins": int(match.group(2)),
        "seconds": int(match.group(3)),
        "thirds": int(match.group(4)),
    }


def _tokens_after(tokens: list[str], label: str) -> str | None:
    try:
        index = tokens.index(label)
    except ValueError:
        return None
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def _parse_greyhound_cell(cell: Any) -> dict[str, Any]:
    tokens = _texts(cell)
    if tokens and tokens[0].upper() == "GREYHOUND":
        tokens = tokens[1:]
    return {
        "colour": tokens[0] if len(tokens) > 0 else None,
        "sex": tokens[1] if len(tokens) > 1 else None,
        "date_of_birth": tokens[2] if len(tokens) > 2 else None,
        "sire": _tokens_after(tokens, "S:"),
        "dam": _tokens_after(tokens, "D:"),
    }


def _parse_career_cell(cell: Any) -> dict[str, Any]:
    text = " ".join(_texts(cell))
    win_match = re.search(r"Win:\s*(\d+(?:\.\d+)?)%\s*/\s*(\d+(?:\.\d+)?)%", text)
    prize_match = re.search(r"P/M:\s*\$?([0-9,]+(?:\.\d+)?)", text)
    return {
        "career": _parse_record(text.split("TD:", 1)[0]),
        "track_distance": _parse_record(text.split("TD:", 1)[1] if "TD:" in text else ""),
        "win_percent": _safe_float(win_match.group(1)) if win_match else None,
        "place_percent": _safe_float(win_match.group(2)) if win_match else None,
        "prize_money": _safe_float(prize_match.group(1)) if prize_match else None,
    }


def _parse_track_distance_cell(cell: Any) -> dict[str, Any]:
    text = " ".join(_texts(cell))
    best_match = re.search(r"Best Time:\s*([0-9.]+)\s+([0-9]{2}/[0-9]{2}/[0-9]{2})", text)
    split_match = re.search(r"Best 1st Split:\s*([0-9.]+)", text)
    return {
        "best_time": _safe_float(best_match.group(1)) if best_match else None,
        "best_time_date": best_match.group(2) if best_match else None,
        "best_first_split": _safe_float(split_match.group(1)) if split_match else None,
    }


def _parse_best_win_times(cell: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in cell.select(".expert-form-cell__content_best__wins"):
        tokens = _texts(item)
        if len(tokens) < 4:
            continue
        rows.append(
            {
                "track": tokens[0],
                "distance": tokens[1],
                "time": _safe_float(tokens[2]),
                "date_time": tokens[3],
            }
        )
    return rows


def _parse_distance_wins(element: Any) -> dict[str, int | None]:
    tokens = _texts(element)
    buckets = ["<400", "400+", "500+", "600+", "700+"]
    values = tokens[-5:] if len(tokens) >= 10 else []
    if len(values) != 5:
        return {}
    return {bucket: _safe_int(value) for bucket, value in zip(buckets, values)}


def _parse_box_history(element: Any) -> dict[str, dict[str, int | None]]:
    tokens = _texts(element)
    out = {str(box): {"starts": None, "wins": None, "places": None} for box in range(1, 9)}
    for label, field in (("Starts", "starts"), ("Wins", "wins"), ("Places", "places")):
        if label not in tokens:
            continue
        start = tokens.index(label) + 1
        values = tokens[start : start + 8]
        if len(values) != 8:
            continue
        for box, value in enumerate(values, start=1):
            out[str(box)][field] = _safe_int(value)
    return out


def _parse_runner_block(block: Any) -> dict[str, Any] | None:
    name_tokens = _texts(block.select_one(".expert-form-runner__details__dog__name"))
    if not name_tokens:
        return None
    dog_name = name_tokens[0]
    grade = name_tokens[1].strip("()") if len(name_tokens) > 1 else None
    cells = block.select(".expert-form-cell")
    histories = block.select(".box-history")
    details = _parse_greyhound_cell(cells[0]) if len(cells) >= 1 else {}
    career = _parse_career_cell(cells[1]) if len(cells) >= 2 else {}
    track_distance = _parse_track_distance_cell(cells[2]) if len(cells) >= 3 else {}
    best_wins = _parse_best_win_times(cells[3]) if len(cells) >= 4 else []
    return {
        "dog_name": dog_name,
        "grade": grade,
        "trainer": {
            "name": (_texts(block.select_one(".trainer-name")) or [None])[0],
            "district": (_texts(block.select_one(".trainer-district")) or [None])[0],
        },
        "owner": (_texts(block.select_one(".owner-info")) or [None, None])[-1],
        "greyhound": details,
        "career": career.get("career", {}),
        "track_distance": {
            **(career.get("track_distance") or {}),
            **track_distance,
        },
        "win_percent": career.get("win_percent"),
        "place_percent": career.get("place_percent"),
        "prize_money": career.get("prize_money"),
        "best_win_times_other_tracks": best_wins,
        "winning_distance_counts": _parse_distance_wins(histories[0]) if histories else {},
        "box_history": _parse_box_history(histories[1]) if len(histories) >= 2 else {},
    }


def _parse_capture_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(
            str(value or datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")
        )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_jump_datetime(race_info: Mapping[str, Any]) -> datetime | None:
    race_date = str(race_info.get("date") or race_info.get("race_date") or "").strip()
    race_time = str(race_info.get("race_time") or race_info.get("jump_time") or "").strip()
    if not race_date or not race_time:
        return None
    location = venue_weather_location(
        race_info.get("venue") or race_info.get("venue_name")
    )
    timezone_name = location.timezone if location is not None else "Australia/Melbourne"
    cleaned = re.sub(r"\s+", " ", race_time.upper().replace(".", ":"))
    parsed_time = None
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H%M"):
        try:
            parsed_time = datetime.strptime(cleaned, fmt).time()
            break
        except ValueError:
            continue
    if parsed_time is None:
        return None
    try:
        parsed_date = datetime.strptime(race_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    return datetime(
        parsed_date.year,
        parsed_date.month,
        parsed_date.day,
        parsed_time.hour,
        parsed_time.minute,
        tzinfo=ZoneInfo(timezone_name),
    )


def _source_url_safe(source_url: Any) -> bool:
    parsed = urlparse(str(source_url or ""))
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc.endswith("thedogs.com.au")
        and "/racing/" in parsed.path
        and parsed.path.rstrip("/").endswith("/expert-form")
        and not any(part in parsed.path.lower() for part in ("result", "dividend", "payout"))
    )


def build_expert_form_metadata_payload(
    html: str,
    *,
    race_info: Mapping[str, Any],
    source_url: str,
    captured_at: Any = None,
) -> dict[str, Any]:
    """Parse TheDogs expert-form page metadata and apply source/timing gates."""

    captured_dt = _parse_capture_timestamp(captured_at)
    rejected: list[str] = []
    if BeautifulSoup is None:
        rejected.append("beautifulsoup_missing")
        runners: list[dict[str, Any]] = []
    else:
        soup = BeautifulSoup(str(html or ""), "html.parser")
        runners = [
            runner
            for block in soup.select(".layout--sidebar--expert")
            if (runner := _parse_runner_block(block)) is not None
        ]
    if not runners:
        rejected.append("expert_form_runner_metadata_missing")
    if not _source_url_safe(source_url):
        rejected.append("expert_form_source_url_not_allowed")
    jump_dt = _parse_jump_datetime(race_info)
    if jump_dt is None:
        rejected.append("expert_form_jump_time_unverified")
    elif captured_dt >= jump_dt.astimezone(timezone.utc):
        rejected.append("expert_form_metadata_captured_at_not_before_jump")

    leakage_safe = not rejected
    return {
        "schema_version": "thedogs_expert_form_metadata_v1",
        "source": "thedogs_expert_form_page",
        "source_url": source_url,
        "captured_at": captured_dt.isoformat().replace("+00:00", "Z"),
        "metadata_is_leakage_safe": leakage_safe,
        "runner_count": len(runners),
        "runners": runners if leakage_safe else [],
        "rejected_reasons": rejected,
    }


def safe_expert_form_metadata_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return sidecar Expert Form metadata only when source and timing stay safe."""

    default = {
        "schema_version": "thedogs_expert_form_metadata_v1",
        "source": None,
        "source_url": None,
        "captured_at": None,
        "metadata_is_leakage_safe": False,
        "runner_count": 0,
        "runners": [],
        "rejected_reasons": ["expert_form_metadata_missing"],
    }
    if not isinstance(payload, Mapping):
        return {**default, "rejected_reasons": ["sidecar_not_object"]}
    metadata = payload.get("expert_form_metadata")
    if not isinstance(metadata, Mapping):
        return default

    rejected = list(metadata.get("rejected_reasons") or [])
    source = metadata.get("source")
    source_url = metadata.get("source_url")
    captured_at = metadata.get("captured_at")
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    if source != "thedogs_expert_form_page":
        rejected.append(f"expert_form_source_not_allowed:{source or 'missing'}")
    if not _source_url_safe(source_url):
        rejected.append("expert_form_source_url_not_allowed")

    captured_dt = None
    try:
        captured_dt = _parse_capture_timestamp(captured_at)
    except Exception:
        rejected.append("expert_form_captured_at_unparseable")
    jump_dt = _parse_jump_datetime(race_info)
    if jump_dt is None:
        rejected.append("expert_form_jump_time_unverified")
    elif captured_dt is not None and captured_dt >= jump_dt.astimezone(timezone.utc):
        rejected.append("expert_form_metadata_captured_at_not_before_jump")

    runners = metadata.get("runners")
    if not isinstance(runners, list) or not runners:
        rejected.append("expert_form_runner_metadata_missing")
        runners = []
    safe = metadata.get("metadata_is_leakage_safe") is True and not rejected
    return {
        "schema_version": str(metadata.get("schema_version") or "thedogs_expert_form_metadata_v1"),
        "source": source,
        "source_url": source_url,
        "captured_at": captured_at,
        "metadata_is_leakage_safe": safe,
        "runner_count": len(runners) if safe else 0,
        "runners": list(runners) if safe else [],
        "rejected_reasons": rejected,
    }
