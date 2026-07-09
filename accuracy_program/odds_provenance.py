"""Strict pre-jump odds provenance and EV eligibility decisions."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping
from urllib.parse import urlparse

try:
    from config.venue_mapping import normalize_venue
except Exception:

    def normalize_venue(value: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "", str(value or "").upper())


TRUSTED_ODDS_SOURCES = {"sportsbet"}
POST_RACE_OR_RESULT_ODDS_MARKETS = {
    "dividend",
    "dividends",
    "payout",
    "result",
    "results",
    "sp",
    "starting_price",
    "startingprice",
}
POST_RACE_SOURCE_URL_MARKERS = ("dividend", "payout", "result")
POST_RACE_SOURCE_TABLES = {"dog_race_data", "race_results", "results"}
STRICT_ODDS_MATCH_METHODS = {
    "dog_name_box",
    "name_box",
    "race_id_box_name",
    "race_id_box_name_exact",
    "strict_identity",
}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _normalize_identity(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def normalize_odds_source(value: Any) -> str:
    return re.sub(r"[^a-z0-9_]", "_", str(value or "").strip().lower()).strip("_")


def _canonical_race_identity(value: Any) -> tuple[str, str, int] | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    filename_match = re.match(
        r"^\s*Race\s+(\d+)\s*-\s*(.+?)\s*-\s*(\d{4}-\d{2}-\d{2})\s*$",
        raw,
        re.IGNORECASE,
    )
    if filename_match:
        race_number = filename_match.group(1)
        venue = filename_match.group(2)
        race_date = filename_match.group(3)
    else:
        canonical_match = re.match(
            r"^\s*(.+?)_(\d{4}-\d{2}-\d{2})_(\d+)\s*$",
            raw,
            re.IGNORECASE,
        )
        if not canonical_match:
            return None
        venue = canonical_match.group(1)
        race_date = canonical_match.group(2)
        race_number = canonical_match.group(3)

    try:
        parsed_race_number = int(race_number)
    except Exception:
        return None

    venue_code = normalize_venue(str(venue).replace("/", "_"))
    if not venue_code or venue_code == "UNKNOWN":
        return None
    return (str(venue_code).upper(), str(race_date), parsed_race_number)


def race_ids_match(snapshot_race_id: Any, odds_race_id: Any) -> tuple[bool, str | None]:
    if str(snapshot_race_id) == str(odds_race_id):
        return True, None

    snapshot_identity = _canonical_race_identity(snapshot_race_id)
    odds_identity = _canonical_race_identity(odds_race_id)
    if snapshot_identity is not None and snapshot_identity == odds_identity:
        return True, "canonical_race_id_box_dog"
    return False, None


def parse_odds_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    for candidate in (raw, raw.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt)
        except ValueError:
            continue
    return None


def seconds_between(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    compare_later = later
    compare_earlier = earlier
    if compare_earlier.tzinfo is not None and compare_later.tzinfo is None:
        compare_later = compare_later.replace(tzinfo=compare_earlier.tzinfo)
    elif compare_earlier.tzinfo is None and compare_later.tzinfo is not None:
        compare_later = compare_later.replace(tzinfo=None)
    return (compare_later - compare_earlier).total_seconds()


def _first_value(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_market_type(value: Any, odds_snapshot: Mapping[str, Any]) -> str:
    raw = str(value or "").strip().lower()
    if raw:
        return re.sub(r"[^a-z0-9_]", "_", raw)
    return "win" if odds_snapshot.get("market_odds_win") is not None else ""


def _source_url_is_post_race(raw_url: Any) -> bool:
    url = str(raw_url or "").strip().lower()
    if not url:
        return False
    if any(marker in url for marker in POST_RACE_SOURCE_URL_MARKERS):
        return True
    try:
        parsed = urlparse(url)
        text = " ".join(part for part in (parsed.path, parsed.query, parsed.fragment) if part)
    except Exception:
        text = url
    tokens = {token for token in re.split(r"[^a-z0-9]+", text.lower()) if token}
    return (
        "sp" in tokens
        or "startingprice" in tokens
        or ("starting" in tokens and "price" in tokens)
    )


def _normalize_odds_level(value: Any, row: Mapping[str, Any]) -> str:
    raw = str(value or "").strip().lower()
    if raw:
        return re.sub(r"[^a-z0-9_]", "_", raw)
    if row.get("box_number") is not None and (
        row.get("dog_clean_name") or row.get("dog_name") or row.get("name")
    ):
        return "dog"
    return "unknown"


def build_odds_snapshot(
    row: Mapping[str, Any],
    *,
    prediction_timestamp: str,
    feature_freeze_timestamp: str | None = None,
    jump_datetime: str | None = None,
    stale_odds_after_minutes: float = 30.0,
) -> dict[str, Any]:
    """Build the canonical odds snapshot shape from prediction/live odds fields."""

    odds_snapshot = {
        "market_odds_win": _safe_float(
            row.get("market_odds_win", row.get("odds_win", row.get("live_odds")))
        ),
        "odds_implied_prob": _safe_float(row.get("odds_implied_prob")),
        "odds_implied_prob_norm": _safe_float(row.get("odds_implied_prob_norm")),
    }
    odds_snapshot = {key: value for key, value in odds_snapshot.items() if value is not None}
    odds_timestamp = _first_value(
        row,
        (
            "odds_timestamp",
            "market_odds_timestamp",
            "live_odds_timestamp",
            "odds_updated_at",
            "odds_last_updated",
        ),
    )
    if odds_timestamp:
        odds_snapshot["odds_timestamp"] = str(odds_timestamp)

    market_type = _normalize_market_type(
        _first_value(row, ("odds_market_type", "market_type", "market_odds_type")),
        odds_snapshot,
    )
    if market_type:
        odds_snapshot["market_type"] = market_type
    odds_level = _normalize_odds_level(
        _first_value(row, ("odds_level", "market_odds_level", "live_odds_level")),
        row,
    )
    if odds_level:
        odds_snapshot["odds_level"] = odds_level

    odds_dt = parse_odds_timestamp(odds_timestamp)
    prediction_dt = parse_odds_timestamp(prediction_timestamp)
    feature_freeze_dt = parse_odds_timestamp(feature_freeze_timestamp)
    if odds_dt is not None and prediction_dt is not None:
        age_seconds = seconds_between(prediction_dt, odds_dt)
        odds_snapshot["odds_age_seconds_at_prediction"] = age_seconds
        odds_snapshot["odds_age_minutes_at_prediction"] = (
            age_seconds / 60.0 if age_seconds is not None else None
        )
        odds_snapshot["odds_captured_before_prediction"] = (
            age_seconds is not None and age_seconds >= 0
        )
        odds_snapshot["odds_stale_at_prediction"] = (
            age_seconds is not None and age_seconds > stale_odds_after_minutes * 60.0
        )
        odds_snapshot["stale_odds_after_minutes"] = stale_odds_after_minutes

    if odds_dt is not None and feature_freeze_dt is not None:
        freeze_age_seconds = seconds_between(feature_freeze_dt, odds_dt)
        odds_snapshot["odds_age_seconds_at_feature_freeze"] = freeze_age_seconds
        odds_snapshot["odds_age_minutes_at_feature_freeze"] = (
            freeze_age_seconds / 60.0 if freeze_age_seconds is not None else None
        )
        odds_snapshot["odds_captured_before_feature_freeze"] = (
            freeze_age_seconds is not None and freeze_age_seconds >= 0
        )

    jump_dt = parse_odds_timestamp(jump_datetime)
    if odds_dt is not None and jump_dt is not None:
        jump_age_seconds = seconds_between(jump_dt, odds_dt)
        odds_snapshot["odds_age_seconds_at_jump"] = jump_age_seconds
        odds_snapshot["odds_age_minutes_at_jump"] = (
            jump_age_seconds / 60.0 if jump_age_seconds is not None else None
        )
        odds_snapshot["odds_captured_before_jump"] = (
            jump_age_seconds is not None and jump_age_seconds >= 0
        )

    provenance = {
        "source": _first_value(
            row,
            (
                "odds_source",
                "market_odds_source",
                "live_odds_source",
                "odds_provider",
                "market_source",
            ),
        ),
        "source_url": _first_value(
            row,
            (
                "odds_source_url",
                "market_odds_source_url",
                "live_odds_source_url",
                "source_url",
                "sportsbet_url",
            ),
        ),
        "source_table": _first_value(
            row,
            (
                "odds_source_table",
                "market_odds_source_table",
                "live_odds_source_table",
            ),
        ),
        "odds_id": _first_value(row, ("odds_id", "live_odds_id", "market_odds_id")),
        "odds_race_id": _first_value(row, ("odds_race_id", "market_odds_race_id")),
        "odds_dog_name": _first_value(
            row,
            (
                "odds_dog_name",
                "odds_dog_clean_name",
                "market_odds_dog_name",
                "market_odds_dog_clean_name",
            ),
        ),
        "odds_box_number": _safe_int(
            _first_value(row, ("odds_box_number", "market_odds_box_number"))
        ),
        "match_type": _first_value(row, ("odds_match_type", "market_odds_match_type")),
        "match_method": _first_value(row, ("odds_match_method", "market_odds_match_method")),
        "match_key": _first_value(row, ("odds_match_key", "market_odds_match_key")),
        "match_confidence": _first_value(
            row, ("odds_match_confidence", "market_odds_match_confidence")
        ),
        "candidate_count": _safe_int(
            _first_value(
                row,
                (
                    "odds_candidate_count",
                    "market_odds_candidate_count",
                    "odds_match_candidate_count",
                ),
            )
        ),
        "duplicate_count": _safe_int(
            _first_value(
                row,
                (
                    "odds_duplicate_count",
                    "market_odds_duplicate_count",
                    "duplicate_odds_count",
                ),
            )
        ),
        "sportsbet_box_source": _first_value(
            row,
            (
                "odds_sportsbet_box_source",
                "sportsbet_box_source",
                "market_odds_sportsbet_box_source",
            ),
        ),
        "sportsbet_list_position": _safe_int(
            _first_value(
                row,
                (
                    "odds_sportsbet_list_position",
                    "sportsbet_list_position",
                    "market_odds_sportsbet_list_position",
                ),
            )
        ),
        "sportsbet_raw_runner_text": _first_value(
            row,
            (
                "odds_sportsbet_raw_runner_text",
                "sportsbet_raw_runner_text",
                "market_odds_sportsbet_raw_runner_text",
            ),
        ),
        "capture_mode": _first_value(
            row,
            (
                "odds_capture_mode",
                "capture_mode",
                "market_odds_capture_mode",
            ),
        ),
        "fetch_timestamp": _first_value(
            row,
            (
                "odds_fetch_timestamp",
                "fetch_timestamp",
                "fetched_at",
                "odds_fetched_at",
                "market_odds_fetched_at",
            ),
        ),
    }
    provenance = {key: value for key, value in provenance.items() if value not in (None, "")}
    if provenance:
        odds_snapshot["odds_provenance"] = provenance
    return {key: value for key, value in odds_snapshot.items() if value is not None}


def build_odds_snapshot_from_row(
    row: Mapping[str, Any],
    *,
    prediction_time: datetime,
    feature_freeze_time: datetime | None = None,
    jump_time: datetime | None = None,
    duplicate_count: int,
    stale_odds_after_minutes: float = 30.0,
) -> dict[str, Any]:
    """Adapt a live-odds DB row into the canonical odds snapshot shape."""

    adapted = {
        "market_odds_win": row.get("odds_decimal"),
        "market_type": row.get("market_type") or "win",
        "odds_level": row.get("odds_level") or "dog",
        "odds_timestamp": row.get("timestamp") or row.get("capture_timestamp"),
        "odds_source": row.get("source"),
        "odds_source_url": row.get("source_url"),
        "odds_source_table": "live_odds",
        "odds_id": row.get("id"),
        "odds_race_id": row.get("race_id"),
        "odds_dog_name": row.get("dog_clean_name") or row.get("dog_name"),
        "odds_box_number": row.get("box_number"),
        "odds_match_type": row.get("_match_basis") or "race_id_box_name",
        "odds_match_method": "race_id_box_name_exact",
        "odds_match_confidence": 1.0,
        "odds_candidate_count": duplicate_count,
        "odds_duplicate_count": duplicate_count,
        "odds_sportsbet_box_source": row.get("sportsbet_box_source") or "unknown",
        "odds_sportsbet_list_position": row.get("sportsbet_list_position"),
        "odds_sportsbet_raw_runner_text": row.get("sportsbet_raw_runner_text"),
        "odds_capture_mode": row.get("capture_mode"),
    }
    snapshot = build_odds_snapshot(
        adapted,
        prediction_timestamp=prediction_time.isoformat(),
        feature_freeze_timestamp=feature_freeze_time.isoformat()
        if feature_freeze_time is not None
        else None,
        jump_datetime=jump_time.isoformat() if jump_time is not None else None,
        stale_odds_after_minutes=stale_odds_after_minutes,
    )
    provenance = snapshot.setdefault("odds_provenance", {})
    provenance.setdefault("sportsbet_raw_runner_text", row.get("sportsbet_raw_runner_text"))
    provenance.setdefault("capture_mode", row.get("capture_mode"))
    return snapshot


def _odds_match_method(provenance: Mapping[str, Any]) -> str | None:
    method = provenance.get("match_method") or provenance.get("match_type")
    return str(method).strip() if method not in (None, "") else None


def classify_prejump_odds(
    runner: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any] | None = None,
    *,
    snapshot_race_id: Any = None,
) -> dict[str, Any]:
    """Classify leakage-safe dog-level odds eligibility for EV calculation."""

    snapshot = odds_snapshot if isinstance(odds_snapshot, Mapping) else {}
    odds = _safe_float(
        snapshot.get("market_odds_win")
        or runner.get("odds")
        or runner.get("odds_win")
        or runner.get("market_odds_win")
    )
    provenance = (
        snapshot.get("odds_provenance")
        if isinstance(snapshot.get("odds_provenance"), Mapping)
        else {}
    )
    method = _odds_match_method(provenance)
    effective_method = method
    canonical_match_method: str | None = None

    def result(status: str) -> dict[str, Any]:
        valid = status == "valid_pre_jump_dog_odds"
        return {
            "odds_match_status": status,
            "odds_match_method": effective_method,
            "odds_exclusion_reason": None if valid else status,
            "odds_provenance_status": "complete" if valid else "excluded",
            "is_ev_eligible": valid,
            "normalized_win_odds": odds if valid else None,
            "market_implied_raw": (1.0 / odds) if valid and odds and odds > 0 else None,
            "reasons": [] if valid else [status],
        }

    if odds is None or odds <= 1.0:
        return result("no_odds_row")

    market_type = _normalize_market_type(snapshot.get("market_type"), snapshot)
    if market_type in POST_RACE_OR_RESULT_ODDS_MARKETS:
        return result("post_race_or_sp_only")
    if market_type and market_type != "win":
        return result("race_level_only_odds")

    if str(snapshot.get("odds_level") or "").lower() in {"race", "race_level", "market"}:
        return result("race_level_only_odds")

    if not snapshot.get("odds_timestamp"):
        return result("missing_timestamp")
    if snapshot.get("odds_captured_before_prediction") is not True:
        return result("timestamp_after_prediction")
    if (
        "odds_captured_before_feature_freeze" in snapshot
        and snapshot.get("odds_captured_before_feature_freeze") is not True
    ):
        return result("timestamp_after_feature_freeze")
    if (
        "odds_captured_before_jump" in snapshot
        and snapshot.get("odds_captured_before_jump") is not True
    ):
        return result("timestamp_after_jump")
    if snapshot.get("odds_stale_at_prediction") is True:
        return result("stale_beyond_ttl")

    source = normalize_odds_source(provenance.get("source") or runner.get("odds_source"))
    if source not in TRUSTED_ODDS_SOURCES:
        return result("untrusted_source")
    source_url = str(provenance.get("source_url") or "").strip()
    if not source_url:
        return result("missing_source_url")
    if _source_url_is_post_race(source_url):
        return result("post_race_or_sp_only")
    source_table = str(provenance.get("source_table") or "").strip().lower()
    if source_table in POST_RACE_SOURCE_TABLES:
        return result("post_race_or_sp_only")

    if snapshot_race_id and provenance.get("odds_race_id"):
        race_match, canonical_method = race_ids_match(
            snapshot_race_id,
            provenance.get("odds_race_id"),
        )
        if not race_match:
            return result("race_id_mismatch")
        if canonical_method:
            canonical_match_method = canonical_method

    sportsbet_box_source = str(provenance.get("sportsbet_box_source") or "").strip().lower()
    if sportsbet_box_source in {"list_position_fallback", "ambiguous_box_source"}:
        return result("ambiguous_box_source")

    runner_box = _safe_int(runner.get("box_number"))
    odds_box = _safe_int(provenance.get("odds_box_number"))
    if runner_box is not None and odds_box is not None and runner_box != odds_box:
        return result("box_mismatch")

    runner_name = runner.get("dog_name") or runner.get("dog_clean_name") or runner.get("name")
    odds_name = provenance.get("odds_dog_name")
    if odds_name and _normalize_identity(odds_name) != _normalize_identity(runner_name):
        return result("dog_name_mismatch")

    if _safe_int(provenance.get("duplicate_count")) and int(provenance["duplicate_count"]) > 1:
        return result("duplicate_odds_rows")
    if _safe_int(provenance.get("candidate_count")) and int(provenance["candidate_count"]) > 1:
        return result("duplicate_odds_rows")

    confidence = _safe_float(provenance.get("match_confidence"))
    explicit_identity_match = (
        runner_box is not None
        and odds_box is not None
        and runner_box == odds_box
        and bool(odds_name)
        and _normalize_identity(odds_name) == _normalize_identity(runner_name)
    )
    if canonical_match_method and explicit_identity_match:
        effective_method = canonical_match_method
    method_key = str(effective_method or "").strip().lower()
    strict_method_match = method_key in STRICT_ODDS_MATCH_METHODS
    if not explicit_identity_match and not strict_method_match:
        return result("ambiguous_runner_identity")
    if confidence is not None and confidence < 0.99:
        return result("ambiguous_runner_identity")

    return result("valid_pre_jump_dog_odds")


def classify_odds_snapshot_for_ev(
    runner: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any] | None = None,
    *,
    snapshot_race_id: Any = None,
) -> dict[str, Any]:
    return classify_prejump_odds(
        runner,
        odds_snapshot,
        snapshot_race_id=snapshot_race_id,
    )


def is_ev_eligible(
    runner: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any],
    *,
    snapshot_race_id: Any = None,
) -> bool:
    return classify_prejump_odds(
        runner,
        odds_snapshot,
        snapshot_race_id=snapshot_race_id,
    )["is_ev_eligible"] is True


def ev_win_if_eligible(
    win_prob_norm: Any,
    runner: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any],
    *,
    snapshot_race_id: Any = None,
) -> float | None:
    if not is_ev_eligible(runner, odds_snapshot, snapshot_race_id=snapshot_race_id):
        return None
    probability = _safe_float(win_prob_norm)
    odds = _safe_float(odds_snapshot.get("market_odds_win"))
    if probability is None or odds is None:
        return None
    return probability * odds - 1.0
