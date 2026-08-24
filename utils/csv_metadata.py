"""
CSV metadata extraction utilities for greyhound race files.
Provides lightweight metadata extraction from race CSV files with fallback to filename parsing.
"""

import csv
import hashlib
import io
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union
from urllib.parse import urlparse

from config.venue_mapping import normalize_venue
from utils.runner_completeness import (
    align_csv_text_to_canonical_final_runner_set,
    analyze_csv_text_runner_completeness,
    normalise_runner_name,
)

# Optional heavy dependency - make pandas optional in constrained test envs
try:
    import pandas as pd  # noqa: F401
except Exception:  # pragma: no cover
    pd = None


SAFE_TARGET_DISTANCE_COLUMNS = (
    "Race Distance",
    "race_distance",
    "target_distance",
    "current_race_distance",
)
SAFE_TARGET_GRADE_COLUMNS = (
    "Race Grade",
    "race_grade",
    "target_grade",
    "current_race_grade",
)
SAFE_SIDECAR_TARGET_SOURCES = {
    "canonical_pre_race_page",
    "sidecar_target_metadata",
    "explicit_csv_sidecar",
}
CANONICAL_SIDECAR_TARGET_SOURCES = {
    "canonical_pre_race_page",
    "sidecar_target_metadata",
}
THEDOGS_MEETING_CARD_GRADE_SOURCE = "thedogs_meeting_card_exact_race"
THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE = "thedogs_exact_race_page"
THEDOGS_EXACT_GRADE_SOURCES = frozenset(
    {
        THEDOGS_MEETING_CARD_GRADE_SOURCE,
        THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE,
    }
)
THEDOGS_VENUE_CODE_OVERRIDES = {
    "ALBION": "ALBION",
    "ALBIONPARK": "ALBION",
    "ANGLEPARK": "APK",
    "AP": "ALBION",
    "APK": "APK",
    "GOSF": "GOSF",
    "GOSFORD": "GOSF",
    "MOUNT": "MOUNT",
    "MOUNTGAMBIER": "MOUNT",
    "MTG": "MOUNT",
    "TARE": "TAREE",
    "TAREE": "TAREE",
}
UNSAFE_TARGET_SOURCE_MARKERS = (
    "embedded_form_history",
    "post_result",
    "result_page",
    "sportsbet_result",
)
POST_RESULT_URL_MARKERS = ("result", "results", "dividend", "dividends", "payout", "payouts")
THEDOGS_CANONICAL_HOST = "www.thedogs.com.au"
THEDOGS_SAFE_RACE_QUERIES = {"", "trial=false", "trial=true"}
WEATHER_TRACK_PLACEHOLDERS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "tba",
    "tbd",
    "to be advised",
    "to be confirmed",
    "0",
    "0.0",
    "20.0",
    "50.0",
}
TRACK_CONDITION_ALLOWED_PATTERNS = (
    r"good(?:\s*\d+)?",
    r"fast",
    r"slow",
    r"heavy(?:\s*\d+)?",
    r"soft(?:\s*\d+)?",
    r"dead(?:\s*\d+)?",
    r"firm(?:\s*\d+)?",
    r"wet",
    r"muddy",
    r"sloppy",
    r"fair",
    r"excellent",
    r"normal",
    r"rain\s+affected",
    r"weather\s+affected",
)
SAFE_WEATHER_TRACK_SOURCES = {
    "canonical_pre_race_page",
    "sidecar_weather_track_metadata",
    "explicit_csv_sidecar",
    "open_meteo_forecast_api",
    "sportsbet_pre_race_page",
}
UNSAFE_WEATHER_TRACK_SOURCE_MARKERS = (
    "embedded_form_history",
    "post_result",
    "result_page",
    "sportsbet_result",
)
TRACK_CONDITION_FIELDS = (
    "track_condition",
    "track condition",
    "trackCondition",
    "trackConditionText",
)
WEATHER_FIELDS = (
    "weather",
    "weather_condition",
    "weather condition",
    "weatherCondition",
    "weatherConditionText",
)
FORM_GUIDE_SPEC_VERSION = "form_guide_pipe_v1"
THEDOGS_EXPERT_FORM_COLUMNS = (
    "Dog Name",
    "Sex",
    "PLC",
    "BOX",
    "WGT",
    "DIST",
    "DATE",
    "TRACK",
    "G",
    "TIME",
    "WIN",
    "BON",
    "1 SEC",
    "MGN",
    "W/2G",
    "PIR",
    "SP",
)
SUPPORTED_FORM_GUIDE_DELIMITERS = {",", "|"}
NORMALIZATION_SOURCE = "canonical_thedogs_export"
PIPE_DELIMITER = "|"


def normalize_target_distance(value: Any) -> Optional[str]:
    """Normalize explicit pre-race distance metadata, preserving fail-closed behavior."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\b(\d{3,4})\s*m?\b", text, re.I)
    if not match:
        return None
    return f"{match.group(1)}m"


def normalize_target_grade(value: Any) -> Optional[str]:
    """Normalize explicit pre-race grade/class metadata without inferring from venue tokens."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text)
    patterns = (
        r"\b\d+(?:st|nd|rd|th)(?:/\d+(?:st|nd|rd|th))?\s+Grade\b",
        r"\bGrade\s*\d+\b",
        r"\bNo\s+Grade\b",
        r"\bNon\s+Graded\b",
        r"^\s*Other(?:\s+\d{3,4}\s*m?)?\s*$",
        r"\bNG\d+(?:-\d+)?\b",
        r"\bM\d+(?:/M\d+)*\b",
        r"^\s*PM(?:\s+\d{3,4}\s*m?)?\s*$",
        r"\bR/?W\b",
        r"\bN\s*[/.-]\s*P\b",
        r"\bG\d+\b",
        r"\bP\d+\b",
        r"\b\d+\s*-\s*\d+\s+Win\b",
        r"\bBest\s*8\b",
        r"\bMaiden\b",
        r"\bNovice\b",
        r"\bOpen\b",
        r"\bMixed\b",
        r"\bRestricted\b",
        r"\bFree For All\b",
        r"\bFFA\b",
        r"\bPathways\b",
        r"\bInvitation(?:al)?\b",
        r"\bSpecial\s+Event\b",
        r"\bGroup\s*\d+\b",
        r"\bFinal\b",
        r"\bHeat\b",
        r"\bMasters\b",
    )
    for pattern in patterns:
        match = re.search(pattern, compact, re.I)
        if match:
            value = match.group(0).strip()
            value = re.sub(r"\s*-\s*", "-", value)
            if re.fullmatch(r"(?:G|P)\d+", value, re.I):
                return value.upper()
            if re.fullmatch(r"NG\d+(?:-\d+)?", value, re.I):
                return value.upper()
            if re.fullmatch(r"M\d+(?:/M\d+)*", value, re.I):
                return value.upper()
            if re.fullmatch(r"PM(?:\s+\d{3,4}\s*m?)?", value, re.I):
                return "PM"
            if re.fullmatch(r"Other(?:\s+\d{3,4}\s*m?)?", value, re.I):
                return "Other"
            if re.fullmatch(r"R/?W", value, re.I):
                return "R/W"
            if re.fullmatch(r"N\s*[/.-]\s*P", value, re.I):
                return "N/P"
            if re.fullmatch(r"\d+-\d+\s+Win", value, re.I):
                return re.sub(r"\bWIN\b", "Win", value.upper())
            if re.search(r"\d+(?:st|nd|rd|th)", value, re.I):
                value = re.sub(
                    r"\b(\d+)(ST|ND|RD|TH)\b",
                    lambda m: f"{m.group(1)}{m.group(2).lower()}",
                    value.upper(),
                )
                return re.sub(r"\bGRADE\b", "Grade", value)
            return value.upper() if value.upper() == "FFA" else value.title()
    return None


def _exact_target_grade_core(value: Any) -> Optional[str]:
    """Return one finite whole-value grade token, never a substring match."""

    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return None
    labeled = re.fullmatch(r"(?:Race\s+)?(?:Grade|Class)\s*:\s*(.+)", text, re.I)
    if labeled:
        text = labeled.group(1).strip()
    text = re.sub(r"\s+\d{3,4}\s*m\s*$", "", text, flags=re.I).strip()
    exact_patterns = (
        r"\d+(?:st|nd|rd|th)(?:/\d+(?:st|nd|rd|th))*\s+Grade",
        r"Grade\s*\d+",
        r"No\s+Grade",
        r"Non\s+Graded",
        r"Other",
        r"NG\d+(?:-\d+)?",
        r"M\d+(?:/M\d+)*",
        r"PM",
        r"R/?W",
        r"N\s*[/.-]\s*P",
        r"G\d+",
        r"P\d+",
        r"\d+\s*-\s*\d+\s+Win",
        r"Best\s*8",
        r"Maiden",
        r"Novice",
        r"Open",
        r"Mixed(?:\s+\d+(?:/\d+)+)?",
        r"Restricted(?:\s+Win(?:\s+(?:Heat|Final))?)?",
        r"Free\s+For\s+All",
        r"FFA",
        r"Pathways",
        r"Invitation(?:al)?",
        r"Special\s+Event",
        r"Group\s*\d+",
        r"Final",
        r"Heat",
        r"Masters",
        r"Tier\s+3\s*-\s*(?:Maiden|Grade\s*\d+|Restricted\s+Win)",
    )
    if any(re.fullmatch(pattern, text, re.I) for pattern in exact_patterns):
        return text
    return None


def normalize_exact_target_grade(value: Any) -> Optional[str]:
    """Normalize one complete grade value without any substring extraction."""

    core = _exact_target_grade_core(value)
    if core is None:
        return None
    upper = re.sub(r"\s+", " ", core.strip().upper())
    tier = re.fullmatch(
        r"TIER\s+3\s*-\s*(MAIDEN|GRADE\s*\d+|RESTRICTED\s+WIN)",
        upper,
    )
    if tier:
        upper = tier.group(1)
    ordinal = re.fullmatch(
        r"\d+(?:ST|ND|RD|TH)(?:/\d+(?:ST|ND|RD|TH))*\s+GRADE",
        upper,
    )
    if ordinal:
        numbers = [str(int(item)) for item in re.findall(r"\d+", upper)]
        if len(numbers) == 1:
            return f"Grade {numbers[0]}"
        mixed = "/".join(numbers)
        if mixed in {
            "2/3",
            "2/3/4",
            "3/4",
            "3/4/5",
            "4/5",
            "5/6",
            "6/7",
        }:
            return f"Mixed {mixed}"
        return None
    grade_number = re.fullmatch(r"(?:GRADE\s*|G)(\d+)", upper)
    if grade_number:
        return f"Grade {int(grade_number.group(1))}"
    mixed = re.fullmatch(r"MIXED\s+(\d+(?:/\d+)+)", upper)
    if mixed:
        value = mixed.group(1)
        return (
            f"Mixed {value}"
            if value
            in {"2/3", "2/3/4", "3/4", "3/4/5", "4/5", "5/6", "6/7"}
            else None
        )
    if re.fullmatch(r"NG\d+(?:-\d+)?", upper):
        return upper
    if re.fullmatch(r"M\d+(?:/M\d+)*", upper):
        return upper
    if re.fullmatch(r"P\d+", upper):
        return upper
    range_win = re.fullmatch(r"(\d+)\s*-\s*(\d+)\s+WIN", upper)
    if range_win:
        return f"{int(range_win.group(1))}-{int(range_win.group(2))} Win"
    aliases = {
        "NO GRADE": "No Grade",
        "NON GRADED": "Non Graded",
        "OTHER": "Other",
        "PM": "PM",
        "R/W": "R/W",
        "RW": "R/W",
        "N/P": "N/P",
        "N-P": "N/P",
        "N.P": "N/P",
        "BEST 8": "Best 8",
        "MAIDEN": "Maiden",
        "NOVICE": "Novice",
        "OPEN": "Open",
        "MIXED": "Mixed",
        "RESTRICTED": "Restricted",
        "RESTRICTED WIN": "Restricted Win",
        "RESTRICTED WIN HEAT": "Restricted Win",
        "RESTRICTED WIN FINAL": "Restricted Win",
        "FREE FOR ALL": "Free For All",
        "FFA": "Free For All",
        "PATHWAYS": "Pathways",
        "INVITATION": "Invitation",
        "INVITATIONAL": "Invitation",
        "SPECIAL EVENT": "Special Event",
        "FINAL": "Final",
        "HEAT": "Heat",
        "MASTERS": "Masters",
    }
    if upper in aliases:
        return aliases[upper]
    group = re.fullmatch(r"GROUP\s*(\d+)", upper)
    return f"Group {int(group.group(1))}" if group else None


def target_grade_equivalence_key(value: Any) -> Optional[str]:
    """Return a comparison key for exact grade aliases without changing storage."""

    normalized = normalize_exact_target_grade(value)
    if normalized is None:
        return None
    grade_number = re.fullmatch(r"Grade\s+(\d+)", normalized, re.I)
    if grade_number:
        return f"GRADE:{int(grade_number.group(1))}"
    mixed = re.fullmatch(r"Mixed\s+(\d+(?:/\d+)+)", normalized, re.I)
    if mixed:
        return f"MIXED:{mixed.group(1)}"
    return re.sub(r"[^A-Z0-9]+", "", str(normalized or "").upper()) or None


def canonical_thedogs_race_identity(value: Any) -> Optional[Dict[str, Any]]:
    """Parse one strict, pre-result canonical TheDogs race URL."""

    try:
        parsed = urlparse(str(value or "").strip())
        unsafe_authority = bool(
            parsed.username or parsed.password or parsed.port is not None
        )
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != THEDOGS_CANONICAL_HOST
        or unsafe_authority
        or parsed.fragment
        or parsed.query.lower() not in THEDOGS_SAFE_RACE_QUERIES
    ):
        return None
    stripped_path = parsed.path.rstrip("/")
    parts = [part for part in stripped_path.split("/") if part]
    if (
        stripped_path != "/" + "/".join(parts)
        or len(parts) not in {4, 5}
        or parts[0].lower() != "racing"
        or not re.fullmatch(r"[a-z0-9-]+", parts[1], re.I)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", parts[2])
        or not parts[3].isdigit()
        or int(parts[3]) < 1
    ):
        return None
    try:
        parsed_date = datetime.strptime(parts[2], "%Y-%m-%d")
    except ValueError:
        return None
    if parsed_date.strftime("%Y-%m-%d") != parts[2]:
        return None
    unsafe_tokens = set(POST_RESULT_URL_MARKERS)
    if len(parts) == 5:
        if (
            not re.fullmatch(r"[a-z0-9-]+", parts[4], re.I)
            or parts[4].lower() in unsafe_tokens
        ):
            return None
    canonical_parts = ["racing", parts[1].lower(), parts[2], str(int(parts[3]))]
    if len(parts) == 5:
        canonical_parts.append(parts[4].lower())
    return {
        "canonical_url": f"https://{THEDOGS_CANONICAL_HOST}/{'/'.join(canonical_parts)}",
        "race_date": parts[2],
        "race_number": int(parts[3]),
        "venue_slug": parts[1].lower(),
    }


def canonical_thedogs_meeting_card_url(value: Any, *, race_date: str) -> Optional[str]:
    """Return the exact date-card URL that can carry live race-card provenance."""

    expected = f"https://{THEDOGS_CANONICAL_HOST}/racing/{race_date}"
    try:
        parsed = urlparse(str(value or "").strip())
        unsafe_authority = bool(
            parsed.username or parsed.password or parsed.port is not None
        )
    except (TypeError, ValueError):
        return None
    candidate = f"https://{THEDOGS_CANONICAL_HOST}{parsed.path.rstrip('/')}"
    if (
        parsed.scheme.lower() != "https"
        or (parsed.hostname or "").lower() != THEDOGS_CANONICAL_HOST
        or unsafe_authority
        or parsed.query
        or parsed.fragment
        or candidate.lower() != expected.lower()
    ):
        return None
    return expected


def canonical_thedogs_venue_identity(value: Any) -> Optional[str]:
    """Return a dependency-light venue identity shared by proof and diagnostics."""

    raw = str(value or "").strip()
    if not raw:
        return None
    token = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if token in THEDOGS_VENUE_CODE_OVERRIDES:
        return THEDOGS_VENUE_CODE_OVERRIDES[token]
    candidates = (
        raw,
        raw.replace("-", " "),
        raw.replace("-", "_"),
        raw.replace("_", " "),
        raw.replace("_", "-"),
    )
    for candidate in candidates:
        normalized = normalize_venue(candidate)
        if normalized != candidate.upper():
            normalized_token = re.sub(r"[^A-Z0-9]", "", normalized.upper())
            return THEDOGS_VENUE_CODE_OVERRIDES.get(
                normalized_token, normalized_token
            )
    return THEDOGS_VENUE_CODE_OVERRIDES.get(token, token)


def normalize_weather_track_text(value: Any) -> Optional[str]:
    """Normalize explicit pre-race weather/track text and reject placeholders."""

    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value).strip())
    if text.lower() in WEATHER_TRACK_PLACEHOLDERS:
        return None
    return text or None


def normalize_track_condition_text(value: Any) -> Optional[str]:
    """Normalize explicit track condition text and reject race-title/promo text."""

    text = normalize_weather_track_text(value)
    if not text:
        return None
    if any(
        re.fullmatch(pattern, text, re.I)
        for pattern in TRACK_CONDITION_ALLOWED_PATTERNS
    ):
        return text
    return None


def _first_named_value(mapping: Mapping[str, Any], fields: tuple[str, ...]) -> Any:
    lower_map = {str(key).lower(): key for key in mapping.keys()}
    for field in fields:
        actual = lower_map.get(field.lower())
        if actual is None:
            continue
        value = mapping.get(actual)
        if value not in (None, ""):
            return value
    return None


def is_safe_sidecar_target_source(source: Any) -> bool:
    text = str(source or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in UNSAFE_TARGET_SOURCE_MARKERS):
        return False
    return text in SAFE_SIDECAR_TARGET_SOURCES or text.startswith(
        ("target_column:", "filename:")
    )


def is_canonical_sidecar_target_source(source: Any) -> bool:
    """Return True only for canonical pre-race sidecar target metadata sources."""

    text = str(source or "").strip()
    if not is_safe_sidecar_target_source(text):
        return False
    return text in CANONICAL_SIDECAR_TARGET_SOURCES


def is_safe_weather_track_source(source: Any) -> bool:
    text = str(source or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in UNSAFE_WEATHER_TRACK_SOURCE_MARKERS):
        return False
    parts = [
        part.strip()
        for part in re.split(r"\s*\+\s*|\s*,\s*", text)
        if part.strip()
    ]
    if not parts:
        return False
    return all(part in SAFE_WEATHER_TRACK_SOURCES for part in parts)


def _is_thedogs_source_url(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    return host == "thedogs.com.au" or host.endswith(".thedogs.com.au")


def _looks_post_result_source_url(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    try:
        parsed = urlparse(text)
        searchable = " ".join(
            part for part in (parsed.path, parsed.query, parsed.fragment) if part
        )
    except Exception:
        searchable = text
    tokens = {token for token in re.split(r"[^a-z0-9]+", searchable) if token}
    return bool(tokens.intersection(POST_RESULT_URL_MARKERS))


def _safe_source_url(value: Any, rejected: list[str]) -> bool:
    if not value:
        rejected.append("source_url_missing")
        return False
    if not _is_thedogs_source_url(value):
        rejected.append("source_url_not_thedogs")
        return False
    if _looks_post_result_source_url(value):
        rejected.append("source_url_looks_post_result")
        return False
    return True


def _is_open_meteo_source_url(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    return parsed.scheme == "https" and host == "api.open-meteo.com"


def _is_sportsbet_source_url(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except Exception:
        return False
    host = parsed.netloc.lower().split("@")[-1].split(":")[0]
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    return host == "sportsbet.com.au" or host.endswith(".sportsbet.com.au")


def _safe_weather_track_source_url(
    value: Any,
    source: Any,
    rejected: list[str],
) -> bool:
    source_text = str(source or "").strip()
    source_parts = [
        part.strip()
        for part in re.split(r"\s*\+\s*|\s*,\s*", source_text)
        if part.strip()
    ] or [source_text]

    def _url_allowed_for_part(part: str, url_value: Any) -> bool:
        if part in {
            "canonical_pre_race_page",
            "sidecar_weather_track_metadata",
            "explicit_csv_sidecar",
        }:
            return _is_thedogs_source_url(url_value)
        if part == "open_meteo_forecast_api":
            return _is_open_meteo_source_url(url_value)
        if part == "sportsbet_pre_race_page":
            return _is_sportsbet_source_url(url_value)
        return False

    if isinstance(value, Mapping):
        ok = True
        for part in source_parts:
            url_value = value.get(part)
            if not url_value:
                rejected.append(f"source_url_missing:{part}")
                ok = False
                continue
            if _looks_post_result_source_url(url_value):
                rejected.append(f"source_url_looks_post_result:{part}")
                ok = False
                continue
            if not _url_allowed_for_part(part, url_value):
                rejected.append(
                    f"source_url_not_allowed_for_weather_track_source:{part}"
                )
                ok = False
        return ok

    if not value:
        rejected.append("source_url_missing")
        return False
    if len(source_parts) > 1:
        rejected.append("composite_source_url_requires_mapping")
        return False
    if _looks_post_result_source_url(value):
        rejected.append("source_url_looks_post_result")
        return False
    allowed = False
    for part in source_parts:
        allowed = allowed or _url_allowed_for_part(part, value)
    if not allowed:
        rejected.append("source_url_not_allowed_for_weather_track_source")
    return allowed


def _weather_track_source_url(
    payload: Mapping[str, Any],
    race_info: Mapping[str, Any],
    shadow_metadata: Mapping[str, Any],
) -> Any:
    return (
        payload.get("weather_track_metadata_source_url")
        or race_info.get("weather_track_metadata_source_url")
        or shadow_metadata.get("weather_track_metadata_source_url")
        or payload.get("metadata_source_url")
        or payload.get("race_url")
        or shadow_metadata.get("source_url")
        or race_info.get("url")
    )


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(str(value).strip())
    except Exception:
        return None


def _exact_grade_provenance_is_valid(
    payload: Mapping[str, Any],
    *,
    grade_source: str,
    canonical_race_url: Any,
    grade_value: Any,
) -> bool:
    """Validate every field required by an exact hash-bound live grade source."""

    race_info = (
        payload.get("race_info")
        if isinstance(payload.get("race_info"), Mapping)
        else {}
    )

    def one_value(key: str) -> Any:
        values = [
            value
            for value in (payload.get(key), race_info.get(key))
            if value not in (None, "")
        ]
        if not values or any(value != values[0] for value in values[1:]):
            return None
        return values[0]

    schema = one_value("target_grade_context_schema")
    exact_value = one_value("target_grade_exact_value")
    declared_key = one_value("target_grade_equivalence_key")
    target_race_url = one_value("target_grade_race_url")
    target_race_date = one_value("target_grade_race_date")
    target_race_number = _safe_int(one_value("target_grade_race_number"))
    target_venue = one_value("target_grade_venue")
    grade_source_url = one_value("target_grade_source_url")
    grade_source_sha256 = str(
        one_value("target_grade_source_sha256") or ""
    ).strip().lower()
    requested_identity = canonical_thedogs_race_identity(canonical_race_url)
    target_identity = canonical_thedogs_race_identity(target_race_url)
    grade_source_identity = canonical_thedogs_race_identity(grade_source_url)
    normalized_exact = normalize_exact_target_grade(exact_value)
    normalized_grade = normalize_exact_target_grade(grade_value)
    expected_schema = {
        THEDOGS_MEETING_CARD_GRADE_SOURCE: "thedogs_meeting_card_exact_race_v1",
        THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE: "thedogs_exact_race_page_v1",
    }.get(grade_source)
    source_url_is_valid = bool(
        (
            grade_source == THEDOGS_MEETING_CARD_GRADE_SOURCE
            and requested_identity is not None
            and canonical_thedogs_meeting_card_url(
                grade_source_url,
                race_date=str(requested_identity["race_date"]),
            )
            is not None
        )
        or (
            grade_source == THEDOGS_EXACT_RACE_PAGE_GRADE_SOURCE
            and requested_identity is not None
            and grade_source_identity is not None
            and grade_source_identity["canonical_url"]
            == requested_identity["canonical_url"]
        )
    )
    if (
        expected_schema is None
        or schema != expected_schema
        or requested_identity is None
        or target_identity is None
        or target_identity["canonical_url"] != requested_identity["canonical_url"]
        or target_race_date != requested_identity["race_date"]
        or target_race_number != requested_identity["race_number"]
        or canonical_thedogs_venue_identity(target_venue)
        != canonical_thedogs_venue_identity(requested_identity["venue_slug"])
        or normalized_exact is None
        or normalized_grade != normalized_exact
        or declared_key != target_grade_equivalence_key(exact_value)
        or re.fullmatch(r"[0-9a-f]{64}", grade_source_sha256) is None
        or not source_url_is_valid
    ):
        return False
    race_dates = [
        str(value).strip()
        for value in (
            payload.get("race_date"),
            payload.get("date"),
            race_info.get("race_date"),
            race_info.get("date"),
        )
        if value not in (None, "")
    ]
    race_numbers = [
        _safe_int(value)
        for value in (
            payload.get("race_number"),
            race_info.get("race_number"),
        )
        if value not in (None, "")
    ]
    race_venues = [
        value
        for value in (
            payload.get("venue"),
            payload.get("venue_name"),
            race_info.get("venue"),
            race_info.get("venue_name"),
        )
        if value not in (None, "")
    ]
    return bool(
        all(value == requested_identity["race_date"] for value in race_dates)
        and all(value == requested_identity["race_number"] for value in race_numbers)
        and all(
            canonical_thedogs_venue_identity(value)
            == canonical_thedogs_venue_identity(requested_identity["venue_slug"])
            for value in race_venues
        )
    )


def _grade_source_is_safe(
    source: Any,
    payload: Mapping[str, Any],
    *,
    canonical_race_url: Any,
    grade_value: Any,
    canonical: bool = False,
) -> bool:
    text = str(source or "").strip()
    if text in THEDOGS_EXACT_GRADE_SOURCES:
        return _exact_grade_provenance_is_valid(
            payload,
            grade_source=text,
            canonical_race_url=canonical_race_url,
            grade_value=grade_value,
        )
    return (
        is_canonical_sidecar_target_source(text)
        if canonical
        else is_safe_sidecar_target_source(text)
    )


def _filename_race_number(csv_path: Union[str, os.PathLike]) -> Optional[int]:
    match = re.search(r"Race\s+(\d+)", os.path.basename(os.fspath(csv_path)), re.I)
    return _safe_int(match.group(1)) if match else None


def _canonical_url_race_number(url: Any) -> Optional[int]:
    """Extract the race-number path segment from a canonical TheDogs race URL."""

    text = str(url or "").split("?", 1)[0].split("#", 1)[0].strip("/")
    if not text:
        return None
    parts = [part for part in text.split("/") if part]
    for idx, part in enumerate(parts):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part) and idx + 1 < len(parts):
            race_number = _safe_int(parts[idx + 1])
            if race_number is not None:
                return race_number
    for part in reversed(parts):
        race_number = _safe_int(part)
        if race_number is not None:
            return race_number
    return None


def _sidecar_path(csv_path: Union[str, os.PathLike]) -> str:
    return f"{os.fspath(csv_path)}.metadata.json"


def _sidecar_field(payload: Dict[str, Any], race_info: Dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value not in (None, ""):
        return value
    return race_info.get(key)


def verify_canonical_sidecar_payload(
    payload: Mapping[str, Any],
    *,
    csv_path: Union[str, os.PathLike],
    race_number: Optional[int] = None,
    canonical_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify an in-memory sidecar payload using the canonical live-capture gate."""

    path = _sidecar_path(csv_path)
    capture_race_number = _safe_int(race_number) or _filename_race_number(csv_path)
    default: Dict[str, Any] = {
        "target_metadata_status": "missing",
        "target_metadata_failure_reason": "sidecar_metadata_missing",
        "target_distance": None,
        "target_grade": None,
        "target_distance_source": None,
        "target_grade_source": None,
        "metadata_is_leakage_safe": False,
        "metadata_source_detail": None,
        "canonical_race_url": canonical_url,
        "race_time_mapping_status": None,
        "race_time_source": None,
        "canonical_url_race_number": None,
        "capture_race_number": capture_race_number,
        "sidecar_path": path,
        "failure_reasons": ["sidecar_metadata_missing"],
    }
    if not isinstance(payload, Mapping):
        result = dict(default)
        result["target_metadata_failure_reason"] = "sidecar_metadata_not_object"
        result["failure_reasons"] = ["sidecar_metadata_not_object"]
        return result

    payload_dict = dict(payload)
    race_info = (
        payload_dict.get("race_info")
        if isinstance(payload_dict.get("race_info"), dict)
        else {}
    )
    canonical_race_url = (
        canonical_url
        or payload_dict.get("race_url")
        or race_info.get("url")
        or payload_dict.get("metadata_source_url")
    )
    distance_source = payload_dict.get("target_distance_source")
    grade_source = payload_dict.get("target_grade_source")
    distance = normalize_target_distance(payload_dict.get("target_distance"))
    grade_value = payload_dict.get("target_grade")
    grade = (
        normalize_exact_target_grade(grade_value)
        if grade_source in THEDOGS_EXACT_GRADE_SOURCES
        else normalize_target_grade(grade_value)
    )
    leakage_safe = payload_dict.get("metadata_is_leakage_safe") is True
    race_time_mapping_status = _sidecar_field(
        payload_dict, race_info, "race_time_mapping_status"
    )
    race_time_source = _sidecar_field(payload_dict, race_info, "race_time_source")
    canonical_url_race_number = _canonical_url_race_number(canonical_race_url)

    missing: list[str] = []
    unsafe: list[str] = []
    mismatch: list[str] = []
    if not distance:
        missing.append("missing_target_distance")
    if not grade:
        missing.append("missing_target_grade")
    if not canonical_race_url:
        missing.append("missing_canonical_race_url")
    if not leakage_safe:
        unsafe.append("metadata_is_leakage_safe_not_true")
    if not is_canonical_sidecar_target_source(distance_source):
        unsafe.append(f"noncanonical_target_distance_source:{distance_source or 'missing'}")
    if not _grade_source_is_safe(
        grade_source,
        payload_dict,
        canonical_race_url=canonical_race_url,
        grade_value=grade_value,
        canonical=True,
    ):
        unsafe.append(f"noncanonical_target_grade_source:{grade_source or 'missing'}")
    if str(race_time_mapping_status or "") != "exact_url_match":
        mismatch.append(
            f"race_time_mapping_status_not_exact_url_match:{race_time_mapping_status or 'missing'}"
        )
    if str(race_time_source or "") != "canonical_race_url":
        mismatch.append(f"race_time_source_not_canonical_race_url:{race_time_source or 'missing'}")
    if capture_race_number is None:
        mismatch.append("capture_race_number_missing")
    if canonical_url_race_number is None:
        mismatch.append("canonical_url_race_number_missing")
    elif capture_race_number is not None and canonical_url_race_number != capture_race_number:
        mismatch.append(
            f"canonical_url_race_number_mismatch:{canonical_url_race_number}!={capture_race_number}"
        )

    status = "verified"
    reasons: list[str] = []
    if missing:
        status = "missing"
        reasons = missing
    elif unsafe:
        status = "unsafe"
        reasons = unsafe
    elif mismatch:
        status = "mismatch"
        reasons = mismatch

    return {
        "target_metadata_status": status,
        "target_metadata_failure_reason": None if status == "verified" else ";".join(reasons),
        "target_distance": distance if status == "verified" else None,
        "target_grade": grade if status == "verified" else None,
        "target_distance_source": str(distance_source) if distance_source not in (None, "") else None,
        "target_grade_source": str(grade_source) if grade_source not in (None, "") else None,
        "metadata_is_leakage_safe": status == "verified",
        "metadata_source_detail": {
            "distance": f"sidecar_metadata:{distance_source}",
            "grade": f"sidecar_metadata:{grade_source}",
        }
        if distance_source or grade_source
        else None,
        "canonical_race_url": str(canonical_race_url) if canonical_race_url else None,
        "race_time_mapping_status": race_time_mapping_status,
        "race_time_source": race_time_source,
        "canonical_url_race_number": canonical_url_race_number,
        "capture_race_number": capture_race_number,
        "sidecar_path": path,
        "failure_reasons": reasons,
    }


def verify_canonical_sidecar_target_metadata(
    csv_path: Union[str, os.PathLike],
    *,
    race_number: Optional[int] = None,
    canonical_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify target distance/grade can be persisted from canonical sidecar metadata.

    This verifier is intentionally stricter than ``load_safe_sidecar_target_metadata``:
    live snapshot persistence requires both target fields, a canonical pre-race
    source, exact URL-backed race-time mapping, and a canonical URL whose race
    number matches the race being captured.
    """

    path = _sidecar_path(csv_path)
    capture_race_number = _safe_int(race_number) or _filename_race_number(csv_path)
    default: Dict[str, Any] = {
        "target_metadata_status": "missing",
        "target_metadata_failure_reason": "sidecar_metadata_missing",
        "target_distance": None,
        "target_grade": None,
        "target_distance_source": None,
        "target_grade_source": None,
        "metadata_is_leakage_safe": False,
        "metadata_source_detail": None,
        "canonical_race_url": canonical_url,
        "race_time_mapping_status": None,
        "race_time_source": None,
        "canonical_url_race_number": None,
        "capture_race_number": capture_race_number,
        "sidecar_path": path,
        "failure_reasons": ["sidecar_metadata_missing"],
    }
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        result = dict(default)
        result["target_metadata_failure_reason"] = f"sidecar_metadata_unreadable:{type(exc).__name__}"
        result["failure_reasons"] = [result["target_metadata_failure_reason"]]
        return result
    if not isinstance(payload, dict):
        result = dict(default)
        result["target_metadata_failure_reason"] = "sidecar_metadata_not_object"
        result["failure_reasons"] = ["sidecar_metadata_not_object"]
        return result
    return verify_canonical_sidecar_payload(
        payload,
        csv_path=csv_path,
        race_number=race_number,
        canonical_url=canonical_url,
    )


def build_safe_target_metadata_payload(
    race_info: Optional[Dict[str, Any]] = None,
    *,
    source_url: Optional[str] = None,
    source: str = "canonical_pre_race_page",
    allow_generic_fields: bool = True,
) -> Dict[str, Any]:
    """Build sidecar target metadata from explicit, pre-race race-card fields only."""

    race_info = dict(race_info or {})
    distance_source = race_info.get("target_distance_source") or source
    grade_source = race_info.get("target_grade_source") or source
    distance_value = race_info.get("target_distance")
    if distance_value in (None, "") and allow_generic_fields:
        distance_value = race_info.get("distance")
    grade_value = race_info.get("target_grade")
    if grade_value in (None, "") and allow_generic_fields:
        grade_value = race_info.get("grade")
    distance = normalize_target_distance(distance_value)
    grade = (
        normalize_exact_target_grade(grade_value)
        if grade_source in THEDOGS_EXACT_GRADE_SOURCES
        else normalize_target_grade(grade_value)
    )
    payload: Dict[str, Any] = {
        "target_distance": None,
        "target_distance_source": "default_missing_target",
        "target_grade": None,
        "target_grade_source": "default_missing_target",
        "metadata_is_leakage_safe": False,
        "metadata_source_url": source_url,
    }
    source_url_safe = (
        _is_thedogs_source_url(source_url)
        and not _looks_post_result_source_url(source_url)
    )
    if distance and is_safe_sidecar_target_source(distance_source):
        payload["target_distance"] = distance
        payload["target_distance_source"] = distance_source
    if grade and _grade_source_is_safe(
        grade_source,
        race_info,
        canonical_race_url=source_url,
        grade_value=grade_value,
    ):
        payload["target_grade"] = grade
        payload["target_grade_source"] = grade_source
    payload["metadata_is_leakage_safe"] = bool(
        source_url_safe
        and payload["target_distance"] is not None
        and payload["target_grade"] is not None
    )
    return payload


def build_safe_weather_track_metadata_payload(
    race_info: Optional[Mapping[str, Any]] = None,
    *,
    source_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Build sidecar weather/track metadata from explicit pre-race fields only."""

    race_info_dict = dict(race_info or {})
    rejected: list[str] = list(
        race_info_dict.get("rejected_weather_track_metadata_sources") or []
    )
    payload: Dict[str, Any] = {
        "track_condition": None,
        "weather": None,
        "weather_condition": None,
        "weather_track_metadata_source": None,
        "weather_track_metadata_source_url": None,
        "weather_track_metadata_is_leakage_safe": False,
        "rejected_weather_track_metadata_sources": rejected,
    }
    source = (
        race_info_dict.get("weather_track_metadata_source")
        or "canonical_pre_race_page"
    )
    metadata_source_url = (
        race_info_dict.get("weather_track_metadata_source_url")
        or source_url
    )
    if not is_safe_weather_track_source(source):
        rejected.append(f"unsafe_weather_track_source:{source}")
        return payload
    if not _safe_weather_track_source_url(metadata_source_url, source, rejected):
        return payload

    track_condition = normalize_track_condition_text(
        _first_named_value(race_info_dict, TRACK_CONDITION_FIELDS)
    )
    weather = normalize_weather_track_text(_first_named_value(race_info_dict, WEATHER_FIELDS))
    if not track_condition:
        rejected.append("track_condition_missing_or_placeholder")
    if not weather:
        rejected.append("weather_missing_or_placeholder")
    if track_condition:
        payload["track_condition"] = track_condition
    if weather:
        payload["weather"] = weather
        payload["weather_condition"] = weather
    if track_condition or weather:
        payload["weather_track_metadata_source"] = source
        payload["weather_track_metadata_source_url"] = metadata_source_url
        payload["weather_track_metadata_is_leakage_safe"] = True
    if isinstance(race_info_dict.get("weather_track_metadata_detail"), Mapping):
        payload["weather_track_metadata_detail"] = dict(
            race_info_dict["weather_track_metadata_detail"]
        )
    return payload


def safe_weather_track_metadata_from_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract weather/track metadata only from verified pre-race sidecar context."""

    default: Dict[str, Any] = {
        "track_condition": None,
        "weather": None,
        "weather_condition": None,
        "weather_track_metadata_source": None,
        "weather_track_metadata_source_url": None,
        "weather_track_metadata_detail": None,
        "weather_track_metadata_is_leakage_safe": False,
        "metadata_source_url": None,
        "metadata_captured_at": None,
        "race_date": None,
        "race_time": None,
        "rejected_weather_track_metadata_sources": [],
    }
    if not isinstance(payload, Mapping):
        return {
            **default,
            "rejected_weather_track_metadata_sources": ["sidecar_not_object"],
        }

    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    shadow_metadata = (
        payload.get("prejump_shadow_metadata")
        if isinstance(payload.get("prejump_shadow_metadata"), Mapping)
        else {}
    )
    rejected = list(payload.get("rejected_weather_track_metadata_sources") or [])
    source_url = _weather_track_source_url(payload, race_info, shadow_metadata)
    metadata = {
        "metadata_source_url": source_url,
        "weather_track_metadata_source_url": source_url,
        "metadata_captured_at": payload.get("metadata_captured_at")
        or shadow_metadata.get("metadata_captured_at"),
        "race_date": race_info.get("date") or shadow_metadata.get("race_date"),
        "race_time": race_info.get("race_time") or shadow_metadata.get("jump_time"),
    }
    source = (
        payload.get("weather_track_metadata_source")
        or shadow_metadata.get("weather_track_metadata_source")
        or "sidecar_weather_track_metadata"
    )
    source_safe = is_safe_weather_track_source(source)
    if not source_safe:
        rejected.append(f"unsafe_weather_track_source:{source}")
    source_url_safe = _safe_weather_track_source_url(source_url, source, rejected)
    explicit_weather_track_safe = (
        payload.get("weather_track_metadata_is_leakage_safe") is True
        or shadow_metadata.get("weather_track_metadata_is_leakage_safe") is True
    )
    if not explicit_weather_track_safe:
        rejected.append("weather_track_metadata_is_leakage_safe_not_true")

    timing_safe = False
    capture_dt = _parse_prejump_contract_timestamp(metadata["metadata_captured_at"])
    if capture_dt is None:
        rejected.append("metadata_captured_at_unparseable")
    else:
        jump_dt, jump_error = _parse_prejump_contract_jump_datetime(
            race_date=metadata["race_date"],
            jump_time=metadata["race_time"],
            capture_dt=capture_dt,
        )
        if jump_dt is None:
            rejected.append(
                f"metadata_capture_timing_unverified:{jump_error or 'unknown'}"
            )
        elif (jump_dt - capture_dt).total_seconds() <= 0:
            rejected.append("metadata_captured_at_not_before_jump")
        else:
            timing_safe = True
    leakage_safe = (
        (
            payload.get("metadata_is_leakage_safe") is True
            or (
                shadow_metadata.get("status") == "PASS"
                and shadow_metadata.get("metadata_is_leakage_safe") is True
            )
        )
        and source_url_safe
        and source_safe
        and explicit_weather_track_safe
        and timing_safe
    )
    if not leakage_safe:
        return {
            **default,
            **metadata,
            "rejected_weather_track_metadata_sources": rejected
            or ["sidecar_not_verified_pre_race_context"],
        }

    track_condition = normalize_track_condition_text(
        _first_named_value(payload, TRACK_CONDITION_FIELDS)
        or _first_named_value(race_info, TRACK_CONDITION_FIELDS)
        or _first_named_value(shadow_metadata, TRACK_CONDITION_FIELDS)
    )
    weather = normalize_weather_track_text(
        _first_named_value(payload, WEATHER_FIELDS)
        or _first_named_value(race_info, WEATHER_FIELDS)
        or _first_named_value(shadow_metadata, WEATHER_FIELDS)
    )
    if not track_condition:
        rejected.append("track_condition_missing_or_placeholder")
    if not weather:
        rejected.append("weather_missing_or_placeholder")

    return {
        **metadata,
        "track_condition": track_condition,
        "weather": weather,
        "weather_condition": weather,
        "weather_track_metadata_source": str(source) if source not in (None, "") else None,
        "weather_track_metadata_source_url": source_url,
        "weather_track_metadata_detail": payload.get("weather_track_metadata_detail")
        if isinstance(payload.get("weather_track_metadata_detail"), Mapping)
        else None,
        "weather_track_metadata_is_leakage_safe": bool(track_condition or weather),
        "rejected_weather_track_metadata_sources": rejected,
    }


def load_safe_weather_track_metadata(csv_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read leakage-safe weather/track metadata from a CSV sidecar, if present."""

    sidecar_path = _sidecar_path(csv_path)
    default: Dict[str, Any] = {
        "track_condition": None,
        "weather": None,
        "weather_condition": None,
        "weather_track_metadata_source": None,
        "weather_track_metadata_source_url": None,
        "weather_track_metadata_detail": None,
        "weather_track_metadata_is_leakage_safe": False,
        "metadata_source_url": None,
        "metadata_captured_at": None,
        "race_date": None,
        "race_time": None,
        "rejected_weather_track_metadata_sources": ["sidecar_metadata_missing"],
    }
    if not os.path.exists(sidecar_path):
        return default
    try:
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {
            **default,
            "rejected_weather_track_metadata_sources": ["sidecar_metadata_unreadable"],
        }
    return safe_weather_track_metadata_from_payload(payload)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _display_delimiter(delimiter: Optional[str]) -> Optional[str]:
    if delimiter == "\t":
        return "\\t"
    return delimiter


def detect_form_guide_delimiter(content: str) -> Optional[str]:
    """Detect the dominant delimiter for a downloaded form-guide export."""

    text = str(content or "")
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",|;\t")
        return dialect.delimiter
    except Exception:
        first_line = next((line for line in text.splitlines() if line.strip()), "")
        counts = {delimiter: first_line.count(delimiter) for delimiter in ",|;\t"}
        delimiter, count = max(counts.items(), key=lambda item: item[1])
        return delimiter if count > 0 else None


def _read_form_guide_rows(content: str, delimiter: str) -> tuple[list[list[str]], Optional[str]]:
    try:
        rows = list(csv.reader(io.StringIO(content), delimiter=delimiter))
    except Exception as exc:
        return [], f"csv_parse_error:{type(exc).__name__}"
    if not rows:
        return [], "empty_csv"
    expected_len = len(rows[0])
    for idx, row in enumerate(rows, start=1):
        if len(row) != expected_len:
            return [], f"row_{idx}_column_count_mismatch:{len(row)}!={expected_len}"
    return rows, None


def _filename_target_date(path: Union[str, os.PathLike]) -> Optional[datetime]:
    match = re.search(r"Race\s+\d+\s+-\s+.+?\s+-\s+(\d{4}-\d{2}-\d{2})\.csv$", Path(path).name, re.I)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d")
    except Exception:
        return None


def _normalize_cell(value: Any) -> str:
    return str(value or "").lstrip("\ufeff").strip()


def _normalize_output_cell(value: Any) -> str:
    return str(value or "").lstrip("\ufeff")


def _validate_thedogs_export_rows(
    rows: list[list[str]],
    *,
    accepted_csv_path: Union[str, os.PathLike],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    header = tuple(_normalize_cell(cell) for cell in rows[0])
    if header != THEDOGS_EXPERT_FORM_COLUMNS:
        reasons.append("schema_header_mismatch")
    if len(header) < len(THEDOGS_EXPERT_FORM_COLUMNS):
        reasons.append("schema_missing_expected_columns")
        return False, reasons

    target_date = _filename_target_date(accepted_csv_path)
    if target_date is None:
        reasons.append("target_date_missing_from_filename")

    dog_name_index = 0
    date_index = THEDOGS_EXPERT_FORM_COLUMNS.index("DATE")
    current_dog = None
    primary_runner_rows = 0
    historical_rows = 0
    for row_number, row in enumerate(rows[1:], start=2):
        if not any(_normalize_cell(cell) for cell in row):
            continue
        dog_cell = _normalize_cell(row[dog_name_index]).strip('"')
        if dog_cell:
            if re.match(r"^\d{1,2}\s*[\.\):-]\s*.+", dog_cell):
                current_dog = dog_cell
                primary_runner_rows += 1
            else:
                reasons.append(f"row_{row_number}_dog_name_missing_box_prefix")
        elif current_dog is None:
            reasons.append(f"row_{row_number}_blank_dog_name_before_primary_runner")

        row_date_text = _normalize_cell(row[date_index])
        if row_date_text:
            try:
                row_date = datetime.strptime(row_date_text, "%Y-%m-%d")
            except Exception:
                reasons.append(f"row_{row_number}_invalid_history_date:{row_date_text}")
                continue
            historical_rows += 1
            if target_date is not None and row_date >= target_date:
                reasons.append(
                    f"row_{row_number}_non_historical_date:{row_date_text}>={target_date.strftime('%Y-%m-%d')}"
                )

    if primary_runner_rows == 0:
        reasons.append("no_box_prefixed_target_runner_rows")
    if historical_rows == 0:
        reasons.append("no_historical_rows")
    return not reasons, reasons


def _rows_to_pipe_text(rows: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, delimiter=PIPE_DELIMITER, lineterminator="\n")
    writer.writerows([[_normalize_output_cell(cell) for cell in row] for row in rows])
    return output.getvalue()


def build_csv_download_provenance_payload(
    *,
    filepath: Union[str, os.PathLike],
    race_url: Optional[str],
    csv_info: Any,
    content: str,
    completeness: Any,
    race_info: Optional[Mapping[str, Any]] = None,
    source: Optional[str] = None,
    normalization: Optional[Mapping[str, Any]] = None,
    filename: Optional[str] = None,
    allow_generic_fields: bool = True,
) -> Dict[str, Any]:
    resolved_csv_url = None
    csv_method = None
    if isinstance(csv_info, str):
        resolved_csv_url = csv_info
        csv_method = "GET"
    elif isinstance(csv_info, Mapping):
        resolved_csv_url = csv_info.get("url")
        csv_method = csv_info.get("type") or "unknown"

    race_info_dict = dict(race_info or {})
    captured_at = _utc_timestamp()
    payload: Dict[str, Any] = {
        "schema_version": "form_guide_download_provenance_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata_captured_at": captured_at,
        "race_url": race_url,
        "race_info": {
            key: value
            for key, value in race_info_dict.items()
            if key
            in {
                "date",
                "distance",
                "grade",
                "race_name",
                "race_number",
                "race_time",
                "race_time_mapping_status",
                "race_time_source",
                "title",
                "track_condition",
                "target_grade_context_schema",
                "target_grade_equivalence_key",
                "target_grade_exact_value",
                "target_grade_race_date",
                "target_grade_race_number",
                "target_grade_race_url",
                "target_grade_source_url",
                "target_grade_source_sha256",
                "target_grade_venue",
                "url",
                "venue",
                "venue_name",
                "weather",
                "weather_condition",
            }
            and value not in (None, "")
        },
        "resolved_csv_url": resolved_csv_url,
        "csv_method": csv_method,
        "content_length": len(str(content).encode("utf-8")),
        "content_sha256": hashlib.sha256(str(content).encode("utf-8")).hexdigest(),
        "runner_completeness": (
            completeness.as_dict() if hasattr(completeness, "as_dict") else dict(completeness or {})
        ),
    }
    if source:
        payload["source"] = source
    if filename:
        payload["filename"] = filename
    for key in (
        "target_grade_context_schema",
        "target_grade_equivalence_key",
        "target_grade_exact_value",
        "target_grade_race_date",
        "target_grade_race_number",
        "target_grade_race_url",
        "target_grade_source_url",
        "target_grade_source_sha256",
        "target_grade_venue",
    ):
        if race_info_dict.get(key) not in (None, ""):
            payload[key] = race_info_dict[key]
    payload.update(
        build_safe_target_metadata_payload(
            race_info_dict,
            source_url=race_url,
            source="canonical_pre_race_page",
            allow_generic_fields=allow_generic_fields,
        )
    )
    payload.update(
        build_safe_weather_track_metadata_payload(
            race_info_dict,
            source_url=race_url,
        )
    )
    expert_form_metadata = race_info_dict.get("expert_form_metadata")
    if isinstance(expert_form_metadata, Mapping):
        payload["expert_form_metadata"] = dict(expert_form_metadata)
    if normalization:
        payload.update(dict(normalization))
    payload["prejump_shadow_metadata"] = build_prejump_shadow_metadata_payload(payload)
    return payload


def _participant_box_name_list(payload: Mapping[str, Any]) -> list[Dict[str, Any]]:
    for key in ("runner_completeness_after_canonical_alignment", "runner_completeness"):
        section = payload.get(key)
        if not isinstance(section, Mapping):
            continue
        participants = section.get("participants")
        if not isinstance(participants, list):
            continue
        rows: list[Dict[str, Any]] = []
        for participant in participants:
            if not isinstance(participant, Mapping):
                continue
            box_number = _safe_int(participant.get("box_number") or participant.get("box"))
            dog_name = str(participant.get("dog_name") or participant.get("name") or "").strip()
            if box_number is None or not dog_name:
                continue
            row = {"box_number": box_number, "dog_name": dog_name}
            native_id = participant.get("source_native_runner_id")
            if (
                isinstance(native_id, str)
                and native_id.isascii()
                and native_id.isdecimal()
            ):
                row["source_native_runner_id"] = native_id
            if participant.get("scratch_state") == "ACTIVE":
                row["scratch_state"] = "ACTIVE"
            rows.append(row)
        if rows:
            return rows
    return []


def build_prejump_shadow_metadata_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an explicit pre-race metadata contract for shadow/live CSV sidecars."""

    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    source_url = (
        payload.get("metadata_source_url")
        or payload.get("race_url")
        or race_info.get("url")
    )
    distance_source = payload.get("target_distance_source")
    grade_source = payload.get("target_grade_source")
    target_distance = normalize_target_distance(payload.get("target_distance"))
    target_grade_value = payload.get("target_grade")
    target_grade = (
        normalize_exact_target_grade(target_grade_value)
        if grade_source in THEDOGS_EXACT_GRADE_SOURCES
        else normalize_target_grade(target_grade_value)
    )
    participants = _participant_box_name_list(payload)
    alignment = (
        payload.get("canonical_runner_alignment")
        if isinstance(payload.get("canonical_runner_alignment"), Mapping)
        else {}
    )
    canonical_runner_source_url = (
        alignment.get("canonical_source_url")
        or alignment.get("canonical_runner_source_url")
        or alignment.get("canonical_runner_set_source_url")
        or alignment.get("source_url")
    )
    fail_reasons: list[str] = []
    if payload.get("metadata_is_leakage_safe") is not True:
        fail_reasons.append("metadata_is_leakage_safe_not_true")
    if not target_distance or not is_safe_sidecar_target_source(distance_source):
        fail_reasons.append("target_distance_missing_or_unsafe")
    if not target_grade or not _grade_source_is_safe(
        grade_source,
        payload,
        canonical_race_url=source_url,
        grade_value=target_grade_value,
    ):
        fail_reasons.append("target_grade_missing_or_unsafe")
    if not (race_info.get("date") or payload.get("race_date")):
        fail_reasons.append("race_date_missing")
    if not (race_info.get("venue") or payload.get("venue")):
        fail_reasons.append("venue_missing")
    if _safe_int(race_info.get("race_number") or payload.get("race_number")) is None:
        fail_reasons.append("race_number_missing")
    if not (
        race_info.get("race_time")
        or race_info.get("jump_time")
        or payload.get("jump_time")
        or payload.get("jump_datetime")
    ):
        fail_reasons.append("jump_time_missing")
    if not source_url:
        fail_reasons.append("source_url_missing")
    elif not _is_thedogs_source_url(source_url):
        fail_reasons.append("source_url_not_thedogs")
    elif _looks_post_result_source_url(source_url):
        fail_reasons.append("source_url_looks_post_result")
    if not participants:
        fail_reasons.append("runner_box_name_list_missing")
    if alignment:
        if alignment.get("status") != "aligned":
            fail_reasons.append("canonical_runner_alignment_not_aligned")
        if alignment.get("canonical_runner_set_status") != "available":
            fail_reasons.append("canonical_runner_set_not_available")
        if not canonical_runner_source_url:
            fail_reasons.append("canonical_runner_source_url_missing")
        elif not _is_thedogs_source_url(canonical_runner_source_url):
            fail_reasons.append("canonical_runner_source_url_not_thedogs")
        elif _looks_post_result_source_url(canonical_runner_source_url):
            fail_reasons.append("canonical_runner_source_url_looks_post_result")
    else:
        fail_reasons.append("canonical_runner_alignment_missing")

    return {
        "schema_version": "prejump_shadow_metadata_v1",
        "status": "PASS" if not fail_reasons else "FAIL",
        "fail_reasons": fail_reasons,
        "metadata_is_leakage_safe": payload.get("metadata_is_leakage_safe") is True,
        "race_date": race_info.get("date") or payload.get("race_date"),
        "venue": race_info.get("venue") or payload.get("venue"),
        "race_number": _safe_int(race_info.get("race_number") or payload.get("race_number")),
        "jump_time": (
            race_info.get("race_time")
            or race_info.get("jump_time")
            or payload.get("jump_time")
            or payload.get("jump_datetime")
        ),
        "metadata_captured_at": (
            payload.get("metadata_captured_at")
            or payload.get("created_at")
            or payload.get("generated_at")
        ),
        "distance": target_distance,
        "grade": target_grade,
        "target_distance_safe": target_distance,
        "target_distance_source": str(distance_source) if distance_source not in (None, "") else None,
        "target_grade_safe": target_grade,
        "target_grade_source": str(grade_source) if grade_source not in (None, "") else None,
        "source_url": str(source_url) if source_url not in (None, "") else None,
        "source_native_race_id": payload.get("source_native_race_id"),
        "source_native_identity_origin": payload.get(
            "source_native_identity_origin"
        ),
        "runner_box_name_list": participants,
        "canonical_final_runner_alignment": {
            "status": alignment.get("status"),
            "canonical_runner_set_status": alignment.get("canonical_runner_set_status"),
            "canonical_runner_count": alignment.get("canonical_runner_count"),
            "prediction_runner_count": alignment.get("prediction_runner_count"),
            "source_url": canonical_runner_source_url,
            "reason": alignment.get("reason"),
            "native_identity_status": alignment.get("native_identity_status"),
        },
    }


def normalize_verified_thedogs_export_content(
    content: str,
    *,
    accepted_csv_path: Union[str, os.PathLike],
    raw_export_path: Union[str, os.PathLike],
    sidecar_payload: Mapping[str, Any],
    runner_completeness: Optional[Mapping[str, Any]] = None,
    canonical_runner_set: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize only verified canonical TheDogs export data to pipe format."""

    original_delimiter = detect_form_guide_delimiter(content)
    base: Dict[str, Any] = {
        "form_guide_spec_version": FORM_GUIDE_SPEC_VERSION,
        "normalization_source": NORMALIZATION_SOURCE,
        "normalization_timestamp": _utc_timestamp(),
        "original_delimiter": _display_delimiter(original_delimiter),
        "normalized_delimiter": PIPE_DELIMITER,
        "raw_export_path": str(raw_export_path),
        "accepted_csv_path": str(accepted_csv_path),
        "raw_content_length": len(str(content).encode("utf-8")),
        "raw_content_sha256": hashlib.sha256(str(content).encode("utf-8")).hexdigest(),
    }
    if original_delimiter not in SUPPORTED_FORM_GUIDE_DELIMITERS:
        return {
            **base,
            "delimiter_status": "rejected",
            "normalization_status": "rejected",
            "normalization_failure_reason": f"unsupported_delimiter:{_display_delimiter(original_delimiter)}",
            "normalized_content": None,
        }

    rows, parse_error = _read_form_guide_rows(content, original_delimiter)
    if parse_error:
        return {
            **base,
            "delimiter_status": "rejected",
            "normalization_status": "rejected",
            "normalization_failure_reason": parse_error,
            "normalized_content": None,
        }

    content_for_normalization = content
    effective_delimiter = original_delimiter
    effective_runner_completeness: Optional[Mapping[str, Any]] = runner_completeness
    canonical_alignment: Optional[Dict[str, Any]] = None
    if canonical_runner_set is not None:
        aligned_content, canonical_alignment = align_csv_text_to_canonical_final_runner_set(
            content,
            canonical_runner_set,
            source=str(accepted_csv_path),
        )
        base["canonical_runner_alignment"] = canonical_alignment
        if canonical_runner_set.get("native_identity_status") == "available":
            base["source_native_race_id"] = canonical_runner_set.get(
                "source_native_race_id"
            )
            native_identity_evidence = canonical_runner_set.get(
                "native_identity_evidence"
            )
            if isinstance(native_identity_evidence, Mapping):
                base["native_identity_evidence"] = dict(native_identity_evidence)
                base["source_native_identity_origin"] = (
                    "thedogs_odds_api_exact_runner_set"
                )
        if canonical_alignment.get("status") == "aligned":
            content_for_normalization = aligned_content
            effective_delimiter = (
                detect_form_guide_delimiter(content_for_normalization) or original_delimiter
            )
            rows, parse_error = _read_form_guide_rows(
                content_for_normalization,
                effective_delimiter,
            )
            if parse_error:
                return {
                    **base,
                    "delimiter_status": "rejected",
                    "normalization_status": "rejected",
                    "normalization_failure_reason": parse_error,
                    "normalized_content": None,
                }
            effective_runner_completeness = analyze_csv_text_runner_completeness(
                content_for_normalization,
                source=str(accepted_csv_path),
            ).as_dict()
            canonical_active = {
                (
                    _safe_int(participant.get("box_number")),
                    normalise_runner_name(participant.get("dog_name") or ""),
                ): participant
                for participant in canonical_runner_set.get(
                    "final_runner_participants", []
                )
                if isinstance(participant, Mapping)
            }
            for participant in effective_runner_completeness.get("participants", []):
                if not isinstance(participant, dict):
                    continue
                identity = (
                    _safe_int(participant.get("box_number") or participant.get("box")),
                    normalise_runner_name(
                        participant.get("dog_name") or participant.get("name") or ""
                    ),
                )
                if identity in canonical_active:
                    # This is not inferred from CSV presence.  The canonical
                    # pre-race producer explicitly excluded scratched and
                    # unpromoted reserve rows before reporting this active set.
                    participant["scratch_state"] = "ACTIVE"
                    native_id = canonical_active[identity].get(
                        "source_native_runner_id"
                    )
                    if (
                        isinstance(native_id, str)
                        and native_id.isascii()
                        and native_id.isdecimal()
                    ):
                        participant["source_native_runner_id"] = native_id
            base["runner_completeness_after_canonical_alignment"] = (
                dict(effective_runner_completeness)
            )

    schema_ok, schema_reasons = _validate_thedogs_export_rows(
        rows,
        accepted_csv_path=accepted_csv_path,
    )
    runner_status = dict(effective_runner_completeness or {}).get("status")
    metadata_verification = verify_canonical_sidecar_payload(
        sidecar_payload,
        csv_path=accepted_csv_path,
    )
    verification = {
        "schema_status": "verified" if schema_ok else "rejected",
        "schema_failure_reasons": schema_reasons,
        "runner_set_status": runner_status,
        "target_metadata_status": metadata_verification.get("target_metadata_status"),
        "target_metadata_failure_reason": metadata_verification.get(
            "target_metadata_failure_reason"
        ),
        "race_time_mapping_status": metadata_verification.get("race_time_mapping_status"),
        "race_time_source": metadata_verification.get("race_time_source"),
        "canonical_url_race_number": metadata_verification.get("canonical_url_race_number"),
        "capture_race_number": metadata_verification.get("capture_race_number"),
    }
    if canonical_alignment is not None:
        verification.update(
            {
                "canonical_runner_set_status": canonical_alignment.get(
                    "canonical_runner_set_status"
                ),
                "canonical_runner_alignment_status": canonical_alignment.get("status"),
                "canonical_runner_alignment_reason": canonical_alignment.get("reason"),
                "canonical_runner_count": canonical_alignment.get(
                    "canonical_runner_count"
                ),
                "canonical_prediction_runner_count": canonical_alignment.get(
                    "prediction_runner_count"
                ),
            }
        )
    failure_reasons: list[str] = []
    if not schema_ok:
        failure_reasons.extend(schema_reasons)
    if runner_status != "COMPLETE":
        failure_reasons.append(f"runner_set_not_complete:{runner_status or 'missing'}")
    if (
        canonical_alignment is not None
        and canonical_alignment.get("canonical_runner_set_status") == "available"
        and canonical_alignment.get("status") != "aligned"
    ):
        failure_reasons.append(
            "final_runner_set_not_aligned:"
            + str(canonical_alignment.get("reason") or "unknown")
        )
    if metadata_verification.get("target_metadata_status") != "verified":
        failure_reasons.append(
            "target_metadata_not_verified:"
            + str(metadata_verification.get("target_metadata_failure_reason") or metadata_verification.get("target_metadata_status"))
        )

    if failure_reasons:
        return {
            **base,
            "delimiter_status": "verified",
            "normalization_status": "rejected",
            "normalization_failure_reason": ";".join(failure_reasons),
            "normalization_verification": verification,
            "normalized_content": None,
        }

    normalized_content = _rows_to_pipe_text(rows)
    normalization_action = (
        "already_pipe" if original_delimiter == PIPE_DELIMITER else "converted_to_pipe"
    )
    if canonical_alignment is not None and canonical_alignment.get("status") == "aligned":
        normalization_action = (
            "canonical_aligned_already_pipe"
            if original_delimiter == PIPE_DELIMITER
            else "canonical_aligned_and_converted_to_pipe"
        )
    return {
        **base,
        "delimiter_status": "verified",
        "normalization_status": "verified",
        "normalization_failure_reason": None,
        "normalization_action": normalization_action,
        "normalization_verification": verification,
        "normalized_content": normalized_content,
    }


def load_safe_sidecar_target_metadata(csv_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """Read leakage-safe target metadata from a CSV sidecar, if present.

    Existing sidecars may include race_info distance/grade without provenance. Those are
    intentionally ignored until a sidecar carries explicit target fields, safe source
    labels, and a pre-race TheDogs source URL.
    """

    sidecar_path = f"{csv_path}.metadata.json"
    default = {
        "target_distance": None,
        "target_distance_source": "default_missing_target",
        "target_grade": None,
        "target_grade_source": "default_missing_target",
        "metadata_is_leakage_safe": False,
        "metadata_source_url": None,
        "rejected_metadata_sources": [],
    }
    if not os.path.exists(sidecar_path):
        return default
    try:
        with open(sidecar_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return default
    if not isinstance(payload, Mapping):
        return default

    shadow_metadata = (
        payload.get("prejump_shadow_metadata")
        if isinstance(payload.get("prejump_shadow_metadata"), Mapping)
        else {}
    )
    rejected = list(payload.get("rejected_metadata_sources") or [])
    result = dict(default)
    source_url = (
        payload.get("metadata_source_url")
        or payload.get("race_url")
        or shadow_metadata.get("source_url")
    )
    result["metadata_source_url"] = source_url

    source_url_safe = False
    if not source_url:
        rejected.append("source_url_missing")
    elif not _is_thedogs_source_url(source_url):
        rejected.append("source_url_not_thedogs")
    elif _looks_post_result_source_url(source_url):
        rejected.append("source_url_looks_post_result")
    else:
        source_url_safe = True

    shadow_contract_safe = True
    if shadow_metadata:
        shadow_status = shadow_metadata.get("status")
        if shadow_status != "PASS":
            fail_reasons = shadow_metadata.get("fail_reasons")
            if isinstance(fail_reasons, list) and fail_reasons:
                reason_text = ",".join(str(reason) for reason in fail_reasons)
            else:
                reason_text = str(shadow_status or "unknown")
            rejected.append(f"prejump_shadow_metadata_failed:{reason_text}")
            shadow_contract_safe = False

    leakage_safe = (
        (
            bool(payload.get("metadata_is_leakage_safe"))
            or (
                shadow_metadata.get("status") == "PASS"
                and shadow_metadata.get("metadata_is_leakage_safe") is True
            )
        )
        and source_url_safe
        and shadow_contract_safe
    )

    distance_source = (
        payload.get("target_distance_source")
        or shadow_metadata.get("target_distance_source")
        or "sidecar_target_metadata"
    )
    distance = normalize_target_distance(
        payload.get("target_distance")
        or shadow_metadata.get("target_distance_safe")
        or shadow_metadata.get("distance")
    )
    if distance and leakage_safe and is_safe_sidecar_target_source(distance_source):
        result["target_distance"] = distance
        result["target_distance_source"] = str(distance_source)
        result["metadata_is_leakage_safe"] = True
    elif payload.get("target_distance") not in (None, ""):
        rejected.append(f"unsafe_sidecar_target_distance:{distance_source}")

    grade_source = (
        payload.get("target_grade_source")
        or shadow_metadata.get("target_grade_source")
        or "sidecar_target_metadata"
    )
    grade_value = (
        payload.get("target_grade")
        or shadow_metadata.get("target_grade_safe")
        or shadow_metadata.get("grade")
    )
    grade = (
        normalize_exact_target_grade(grade_value)
        if grade_source in THEDOGS_EXACT_GRADE_SOURCES
        else normalize_target_grade(grade_value)
    )
    if grade and leakage_safe and _grade_source_is_safe(
        grade_source,
        payload,
        canonical_race_url=source_url,
        grade_value=grade_value,
    ):
        result["target_grade"] = grade
        result["target_grade_source"] = str(grade_source)
        result["metadata_is_leakage_safe"] = True
    elif payload.get("target_grade") not in (None, ""):
        rejected.append(f"unsafe_sidecar_target_grade:{grade_source}")

    if rejected:
        result["rejected_metadata_sources"] = rejected
    return result


def existing_prejump_sidecar_contract_status(
    csv_path: Union[str, os.PathLike],
) -> Dict[str, Any]:
    """Return whether an accepted pre-jump CSV can be safely reused."""

    path = Path(csv_path)
    sidecar_path = Path(f"{path}.metadata.json")
    report: Dict[str, Any] = {
        "schema_version": "existing_prejump_sidecar_contract_status_v1",
        "csv_path": str(path),
        "sidecar_path": str(sidecar_path),
        "status": "FAIL",
        "reasons": [],
        "runner_completeness": None,
    }
    if not path.exists():
        report["reasons"].append("csv_missing")
        return report
    try:
        completeness = analyze_csv_text_runner_completeness(
            path.read_text(encoding="utf-8-sig", errors="replace"),
            source=str(path),
        )
    except Exception as exc:
        report["reasons"].append(f"csv_unreadable:{type(exc).__name__}")
        completeness = None
    if completeness is not None:
        report["runner_completeness"] = completeness.as_dict()
        if not completeness.is_complete:
            report["reasons"].append(
                "existing_csv_runner_set_incomplete:"
                + ",".join(completeness.reasons)
            )

    if not sidecar_path.exists():
        report["reasons"].append("sidecar_metadata_missing")
        return report
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["reasons"].append(f"sidecar_metadata_unreadable:{type(exc).__name__}")
        return report
    if not isinstance(payload, Mapping):
        report["reasons"].append("sidecar_metadata_not_object")
        return report

    shadow_metadata = payload.get("prejump_shadow_metadata")
    if not isinstance(shadow_metadata, Mapping):
        report["reasons"].append("prejump_shadow_metadata_missing")
        shadow_metadata = {}
    elif shadow_metadata.get("status") != "PASS":
        report["reasons"].append("prejump_shadow_metadata_not_pass")

    required_fields = {
        "race_date": shadow_metadata.get("race_date"),
        "venue": shadow_metadata.get("venue"),
        "race_number": shadow_metadata.get("race_number"),
        "jump_time": shadow_metadata.get("jump_time"),
        "metadata_captured_at": shadow_metadata.get("metadata_captured_at")
        or payload.get("metadata_captured_at"),
        "target_distance_safe": shadow_metadata.get("target_distance_safe")
        or payload.get("target_distance"),
        "target_grade_safe": shadow_metadata.get("target_grade_safe")
        or payload.get("target_grade"),
        "source_url": shadow_metadata.get("source_url") or payload.get("race_url"),
    }
    missing = [
        field_name
        for field_name, value in required_fields.items()
        if value in (None, "")
    ]
    if missing:
        report["reasons"].append("missing_required_fields:" + ",".join(missing))
    if not any(
        field in missing for field in ("race_date", "jump_time", "metadata_captured_at")
    ):
        capture_dt = _parse_prejump_contract_timestamp(
            required_fields.get("metadata_captured_at")
        )
        if capture_dt is None:
            report["reasons"].append("metadata_captured_at_unparseable")
        else:
            jump_dt, jump_error = _parse_prejump_contract_jump_datetime(
                race_date=required_fields.get("race_date"),
                jump_time=required_fields.get("jump_time"),
                capture_dt=capture_dt,
            )
            if jump_dt is None:
                report["reasons"].append(
                    f"metadata_capture_timing_unverified:{jump_error or 'unknown'}"
                )
            elif (jump_dt - capture_dt).total_seconds() <= 0:
                report["reasons"].append("metadata_captured_at_not_before_jump")
    report["reasons"].extend(
        _prejump_contract_url_reasons(
            required_fields.get("source_url"),
            field_name="source_url",
        )
    )

    runners = shadow_metadata.get("runner_box_name_list")
    if not isinstance(runners, list) or not runners:
        report["reasons"].append("runner_box_name_list_missing")

    alignment = shadow_metadata.get("canonical_final_runner_alignment")
    if not isinstance(alignment, Mapping):
        report["reasons"].append("canonical_final_runner_alignment_missing")
    else:
        if alignment.get("status") != "aligned":
            report["reasons"].append("canonical_final_runner_alignment_not_aligned")
        if alignment.get("canonical_runner_set_status") != "available":
            report["reasons"].append("canonical_runner_set_not_available")
        if not alignment.get("source_url"):
            report["reasons"].append("canonical_runner_source_url_missing")
        else:
            report["reasons"].extend(
                _prejump_contract_url_reasons(
                    alignment.get("source_url"),
                    field_name="canonical_runner_source_url",
                )
            )

    if not report["reasons"]:
        report["status"] = "PASS"
    return report


def _parse_prejump_contract_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return parsed


def _parse_prejump_contract_jump_datetime(
    *,
    race_date: Any,
    jump_time: Any,
    capture_dt: datetime,
) -> tuple[datetime | None, str | None]:
    try:
        parsed_date = datetime.strptime(str(race_date).strip()[:10], "%Y-%m-%d").date()
    except Exception:
        return None, "race_date_unparseable"
    text = str(jump_time or "").strip()
    if not text:
        return None, "jump_time_missing"
    normalized = text
    if len(normalized) >= 5 and normalized[-5] in {"+", "-"} and normalized[-4:].isdigit():
        normalized = f"{normalized[:-5]}{normalized[-5:-2]}:{normalized[-2:]}"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        parsed = None
    if parsed is not None:
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=capture_dt.tzinfo)
        return parsed, None
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M", "%H:%M:%S"):
        try:
            parsed_time = datetime.strptime(text.upper(), fmt).time()
        except ValueError:
            continue
        return (
            datetime.combine(parsed_date, parsed_time).replace(tzinfo=capture_dt.tzinfo),
            None,
        )
    return None, "jump_time_unparseable"


def _prejump_contract_url_reasons(value: Any, *, field_name: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if not _is_thedogs_source_url(text):
        return [f"{field_name}_not_thedogs"]
    if _looks_post_result_source_url(text):
        return [f"{field_name}_looks_post_result"]
    return []


def parse_race_csv_meta(file_path: str) -> Dict[str, Any]:
    """
    Extract lightweight metadata from race CSV files with intelligent fallback.

    Handles common column aliases and fallback to regex parsing of filename
    patterns like "Race 11 - TAREE - 2025-08-02.csv". Gracefully skips
    malformed files and includes "status":"error" entry in response if needed.

    Args:
        file_path: Path to the CSV file to analyze

    Returns:
        Dict containing extracted metadata with the following structure:
        {
            "race_number": int,           # Race number (from filename or data)
            "venue": str,                 # Racing venue/track
            "race_date": str,            # Race date (YYYY-MM-DD format)
            "distance": str,             # Race distance (from CSV data)
            "grade": str,                # Race grade/class (from CSV data)
            "field_size": int,           # Number of runners
            "source": str,               # "csv_data", "filename", or "mixed"
            "status": str,               # "success" or "error"
            "error_message": str,        # Error details (if status="error")
            "filename": str,             # Original filename
            "file_exists": bool,         # Whether file exists
            "file_size": int            # File size in bytes (if exists)
        }

    Examples:
        >>> parse_race_csv_meta("Race 11 - TAREE - 2025-08-02.csv")
        {
            "race_number": 11,
            "venue": "TAREE",
            "race_date": "2025-08-02",
            "distance": "300",
            "grade": "5",
            "field_size": 8,
            "source": "mixed",
            "status": "success",
            "error_message": "",
            "filename": "Race 11 - TAREE - 2025-08-02.csv",
            "file_exists": True,
            "file_size": 4521
        }
    """

    # Initialize response with defaults
    response = {
        "race_number": 0,
        "venue": "Unknown",
        "race_date": "Unknown",
        "distance": "Unknown",
        "grade": "Unknown",
        "field_size": 0,
        "source": "unknown",
        "status": "success",
        "error_message": "",
        "filename": os.path.basename(file_path),
        "file_exists": False,
        "file_size": 0,
    }

    try:
        # Check if file exists and get basic file info
        if os.path.exists(file_path):
            response["file_exists"] = True
            response["file_size"] = os.path.getsize(file_path)
        else:
            response["status"] = "error"
            response["error_message"] = f"File not found: {file_path}"
            return response

        # Track what data sources we use
        data_sources = []

        # Step 1: Extract what we can from filename using regex
        filename_meta = _extract_from_filename(os.path.basename(file_path))
        if filename_meta:
            response.update(filename_meta)
            data_sources.append("filename")

        # Step 2: Try to extract from CSV data with error handling
        try:
            csv_meta = _extract_from_csv_data(file_path)
            if csv_meta:
                # Merge CSV data, preferring CSV over filename for data fields
                # but keeping filename data for race identification
                for key, value in csv_meta.items():
                    if (
                        value is not None and value != "" and value != "Unknown"
                    ):  # Only override with valid CSV data
                        response[key] = value
                data_sources.append("csv_data")
        except Exception as csv_error:
            # CSV parsing failed, but we might still have filename data
            response.update(
                {
                    "status": "error" if not data_sources else "success",
                    "error_message": f"CSV parsing failed: {str(csv_error)}",
                }
            )

        # Step 3: Set source indicator
        if len(data_sources) > 1:
            response["source"] = "mixed"
        elif data_sources:
            response["source"] = data_sources[0]
        else:
            response["source"] = "none"
            response["status"] = "error"
            response["error_message"] = "No metadata could be extracted"

        return response

    except Exception as e:
        # Catch-all for any unexpected errors
        response.update(
            {
                "status": "error",
                "error_message": f"Unexpected error: {str(e)}",
                "source": "error",
            }
        )
        return response


def _extract_from_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from filename using regex patterns.

    Supports patterns like:
    - "Race 11 - TAREE - 2025-08-02.csv"
    - "Race 5 - GEE - 08 July 2025.csv"
    - "20250730162231_Race 3 - TAR - 28 June 2025.csv"

    Args:
        filename: The filename to parse

    Returns:
        Dict with extracted metadata or None if no pattern matches
    """

    # Remove any timestamp prefix and .csv extension
    clean_name = re.sub(r"^\d+_", "", filename)  # Remove timestamp prefix
    clean_name = re.sub(
        r"\.csv$", "", clean_name, flags=re.IGNORECASE
    )  # Remove extension

    # Pattern 1: "Race 11 - TAREE - 2025-08-02" (ISO date format)
    venue_pattern = r"([A-Z0-9_]+(?:-[A-Z0-9_]+)*)"

    pattern1 = rf"Race\s+(\d+)\s*-\s*{venue_pattern}\s*-\s*(\d{{4}}-\d{{2}}-\d{{2}})"
    match1 = re.search(pattern1, clean_name, re.IGNORECASE)

    if match1:
        race_num, venue, date_str = match1.groups()
        return {
            "race_number": int(race_num),
            "venue": venue.upper(),
            "race_date": date_str,  # Already in YYYY-MM-DD format
        }

    # Pattern 2: "Race 5 - GEE - 08 July 2025" (human readable date)
    pattern2 = rf"Race\s+(\d+)\s*-\s*{venue_pattern}\s*-\s*(\d{{1,2}})\s+(\w+)\s+(\d{{4}})"
    match2 = re.search(pattern2, clean_name, re.IGNORECASE)

    if match2:
        race_num, venue, day, month_name, year = match2.groups()

        # Convert month name to number
        date_str = _parse_human_date(day, month_name, year)
        if date_str:
            return {
                "race_number": int(race_num),
                "venue": venue.upper(),
                "race_date": date_str,
            }

    # Pattern 3: Try to extract just race number and venue if date parsing fails
    pattern3 = rf"Race\s+(\d+)\s*-\s*{venue_pattern}"
    match3 = re.search(pattern3, clean_name, re.IGNORECASE)

    if match3:
        race_num, venue = match3.groups()
        return {
            "race_number": int(race_num),
            "venue": venue.upper(),
            "race_date": "Unknown",
        }

    return None


def _parse_human_date(day: str, month_name: str, year: str) -> Optional[str]:
    """
    Convert human readable date to YYYY-MM-DD format.

    Args:
        day: Day of month (e.g., "08", "28")
        month_name: Month name (e.g., "July", "June")
        year: Year (e.g., "2025")

    Returns:
        Date string in YYYY-MM-DD format or None if parsing fails
    """

    month_mapping = {
        "january": "01",
        "jan": "01",
        "february": "02",
        "feb": "02",
        "march": "03",
        "mar": "03",
        "april": "04",
        "apr": "04",
        "may": "05",
        "june": "06",
        "jun": "06",
        "july": "07",
        "jul": "07",
        "august": "08",
        "aug": "08",
        "september": "09",
        "sep": "09",
        "sept": "09",
        "october": "10",
        "oct": "10",
        "november": "11",
        "nov": "11",
        "december": "12",
        "dec": "12",
    }

    month_num = month_mapping.get(month_name.lower())
    if month_num:
        day_padded = day.zfill(2)  # Ensure 2-digit day
        return f"{year}-{month_num}-{day_padded}"

    return None


def _looks_like_embedded_form_history(df: "pd.DataFrame") -> bool:
    """Detect form-guide CSVs where row values are historical starts, not target race fields."""
    try:
        columns = {str(c).strip().upper() for c in df.columns}
        historical_columns = {
            "PLC",
            "TIME",
            "WIN",
            "BON",
            "MGN",
            "W/2G",
            "PIR",
            "SP",
            "DATE",
            "TRACK",
        }
        if "DOG NAME" not in columns or len(columns.intersection(historical_columns)) < 4:
            return False
        names = df["Dog Name"].dropna().astype(str)
        return bool(names.str.match(r"^\s*\d{1,2}\s*[\.\):-]").any())
    except Exception:
        return False


def _extract_from_csv_data(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract metadata from CSV file contents.

    Looks for common columns like TRACK, DIST, G (grade), and analyzes
    the data to determine race characteristics.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dict with extracted metadata or None if extraction fails
    """

    try:
        # Try pandas first for robust CSV handling
        if pd is None:
            raise ImportError("pandas not available")
        df = pd.read_csv(
            file_path,
            nrows=50,
            sep=None,
            engine="python",
            encoding="utf-8-sig",
        )

        # Clean up the dataframe - remove rows where all values are empty quotes
        df = df.replace('""', "")  # Replace empty quotes with empty strings
        df = df.replace("", pd.NA)  # Convert empty strings to NaN

        result = {}
        embedded_form_history = _looks_like_embedded_form_history(df)
        if embedded_form_history:
            result["csv_row_context"] = "embedded_form_history"
            result["target_metadata_from_csv"] = False
            result["metadata_is_leakage_safe"] = False
        rejected_metadata_sources = []

        def _first_non_empty(columns):
            for column in columns:
                if column not in df.columns:
                    continue
                try:
                    values = df[column].dropna()
                    for value in values:
                        if str(value).strip() != "":
                            return value, column
                except Exception:
                    continue
            return None, None

        # Extract venue from TRACK column (most reliable source)
        if "TRACK" in df.columns and not embedded_form_history:
            venues = df["TRACK"].dropna().unique()
            # Get the most common venue (in case of mixed data)
            if len(venues) > 0:
                venue_counts = df["TRACK"].value_counts()
                raw_venue = str(venue_counts.index[0]).upper()
                result["venue"] = standardize_venue_name(raw_venue)

        safe_distance, safe_distance_col = _first_non_empty(SAFE_TARGET_DISTANCE_COLUMNS)
        safe_grade, safe_grade_col = _first_non_empty(SAFE_TARGET_GRADE_COLUMNS)

        if safe_distance is not None:
            result["distance"] = str(safe_distance)
            result["distance_source"] = f"target_column:{safe_distance_col}"
            result["target_metadata_from_csv"] = True
            result["metadata_is_leakage_safe"] = True
        elif "DIST" in df.columns and embedded_form_history:
            rejected_metadata_sources.append("embedded_form_history:DIST")
        # Extract distance from DIST column only when rows are not embedded form history.
        elif "DIST" in df.columns:
            distances = df["DIST"].dropna().unique()
            if len(distances) > 0:
                # Get most common distance
                distance_counts = df["DIST"].value_counts()
                result["distance"] = str(distance_counts.index[0])
                result["distance_source"] = "csv_target_row:DIST"
                result["metadata_is_leakage_safe"] = True

        if safe_grade is not None:
            result["grade"] = str(safe_grade)
            result["grade_source"] = f"target_column:{safe_grade_col}"
            result["target_metadata_from_csv"] = True
            result["metadata_is_leakage_safe"] = True
        elif "G" in df.columns and embedded_form_history:
            rejected_metadata_sources.append("embedded_form_history:G")
        # Extract grade from G column only when rows are not embedded form history.
        elif "G" in df.columns:
            grades = df["G"].dropna().unique()
            if len(grades) > 0:
                # Get most common grade
                grade_counts = df["G"].value_counts()
                result["grade"] = str(grade_counts.index[0])
                result["grade_source"] = "csv_target_row:G"
                result["metadata_is_leakage_safe"] = True

        if embedded_form_history:
            for rejected_col in (
                "PLC",
                "TIME",
                "BON",
                "MGN",
                "WIN",
                "PIR",
                "finish_position",
                "winner",
                "winner_name",
                "payout",
            ):
                if rejected_col in df.columns:
                    rejected_metadata_sources.append(f"post_result_field:{rejected_col}")
        if rejected_metadata_sources:
            result["rejected_metadata_sources"] = rejected_metadata_sources

        # Calculate field size (number of unique dogs/boxes)
        if "Dog Name" in df.columns and embedded_form_history:
            # Count unique dogs (excluding empty quotes and NaN)
            dogs = df["Dog Name"].dropna()
            dogs = dogs[dogs != '""']  # Remove empty quotes
            # Only count dogs that don't start with a number followed by period (these are the primary entries)
            primary_dogs = dogs[dogs.astype(str).str.match(r"^\s*\d{1,2}\s*[\.\):-]")]
            result["field_size"] = len(primary_dogs)
        elif "BOX" in df.columns and not embedded_form_history:
            boxes = df["BOX"].dropna().unique()
            result["field_size"] = len(boxes)
        elif "Dog Name" in df.columns:
            dogs = df["Dog Name"].dropna()
            dogs = dogs[dogs != '""']
            result["field_size"] = len(dogs)

        # Try to extract race date from DATE column
        if "DATE" in df.columns and not embedded_form_history:
            dates = df["DATE"].dropna().unique()
            if len(dates) > 0:
                # Get most common date and try to parse it
                date_counts = df["DATE"].value_counts()
                most_common_date = str(date_counts.index[0])

                # Try to parse the date into standard format
                parsed_date = _standardize_date(most_common_date)
                if parsed_date:
                    result["race_date"] = parsed_date

        return result if result else None

    except Exception as e:
        # If pandas fails, try basic CSV reader as fallback
        try:
            with open(file_path, "r", newline="", encoding="utf-8-sig") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=",|;\t")
                except Exception:
                    dialect = csv.excel
                reader = csv.DictReader(f, dialect=dialect)

                # Read first few rows to analyze
                rows = []
                for i, row in enumerate(reader):
                    if i >= 20:  # Only read first 20 rows
                        break
                    rows.append(row)

                if not rows:
                    return None

                result = {}

                # Simple extraction from first few rows
                for row in rows:
                    if "TRACK" in row and row["TRACK"] and row["TRACK"] != '""':
                        result["venue"] = row["TRACK"].upper()
                        break

                return result if result else None

        except Exception as fallback_error:
            # Both pandas and basic CSV failed
            return None


def _standardize_date(date_str: str) -> Optional[str]:
    """
    Convert various date formats to YYYY-MM-DD standard format.

    Args:
        date_str: Date string in various formats

    Returns:
        Standardized date string or None if parsing fails
    """

    # List of common date formats to try
    date_formats = [
        "%Y-%m-%d",  # 2025-08-02
        "%d/%m/%Y",  # 02/08/2025
        "%m/%d/%Y",  # 08/02/2025
        "%d-%m-%Y",  # 02-08-2025
        "%Y%m%d",  # 20250802
        "%d %B %Y",  # 02 August 2025
        "%d %b %Y",  # 02 Aug 2025
    ]

    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


# Venue name standardization mapping
VENUE_ALIASES = {
    "TARE": "TAREE",
    "TAR": "TAREE",
    "BEN": "BENDIGO",
    "GEE": "GEELONG",
    "BAL": "BALLARAT",
    "WAR": "WARRNAMBOOL",
    "SHE": "SHEPPARTON",
    "MEA": "THE_MEADOWS",
    "SAN": "SANDOWN",
    "HOR": "HORSHAM",
    "RICH": "RICHMOND",
    "GRDN": "THE_GARDENS",
    "GOSF": "GOSFORD",
    "MAIT": "MAITLAND",
    "RICS": "RICHMOND",
    "MUSW": "MUSWELLBROOK",
    "NOWR": "NOWRA",
    "PPK": "PENRITH",
    "GUNN": "GUNNEDAH",
    "AP_K": "ANGLE_PARK",
    "CANN": "CANNINGTON",
    "MAND": "MANDURAH",
    "MURR": "MURRAY_BRIDGE",
    "SAL": "SALE",
    "MOUNT": "MOUNT_GAMBIER",
    "HEA": "HEALESVILLE",
    "W_PK": "WENTWORTH_PARK",
    "DAPT": "DAPTO",
}


def standardize_venue_name(venue: str) -> str:
    """
    Standardize venue names using common aliases.

    Args:
        venue: Raw venue name from filename or CSV

    Returns:
        Standardized venue name
    """

    venue_upper = venue.upper().strip()
    return VENUE_ALIASES.get(venue_upper, venue_upper)
