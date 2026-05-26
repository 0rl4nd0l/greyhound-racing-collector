"""Prediction-before-result snapshot construction and safe persistence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


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

RESULT_FIELD_NAMES = {
    "actual_results",
    "actual_finish_position",
    "actual_margin",
    "actual_place",
    "actual_position",
    "actual_win",
    "actual_winner",
    "beaten_margin",
    "finish_position",
    "label",
    "label_quality",
    "label_source",
    "labels",
    "placing",
    "official_result",
    "official_results",
    "race_result",
    "race_results",
    "result",
    "result_status",
    "results_status",
    "scraped_finish_position",
    "scraped_raw_result",
    "winner",
    "winner_margin",
    "winner_name",
    "winner_odds",
    "winning_time",
}


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            return {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


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


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if str(value).strip().lower() in {"0", "false", "no", "off"}:
        return False
    return None


def _normalize_identity(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def _quality_flags(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("quality_flags")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = row.get("data_quality_flags")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = []
    if isinstance(raw, str):
        return [value.strip() for value in raw.split(",") if value.strip()]
    if isinstance(raw, list):
        return [str(value) for value in raw if value not in (None, "")]
    return []


def _add_quality_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _parse_timestamp(value: Any) -> datetime | None:
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


def _seconds_between(later: datetime, earlier: datetime) -> float:
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


def _build_odds_snapshot(
    row: Mapping[str, Any],
    *,
    prediction_timestamp: str,
    feature_freeze_timestamp: str,
    jump_datetime: str | None,
    stale_odds_after_minutes: float,
) -> dict[str, Any]:
    odds_snapshot = {
        "market_odds_win": _safe_float(
            row.get("market_odds_win", row.get("odds_win", row.get("live_odds")))
        ),
        "odds_implied_prob": _safe_float(row.get("odds_implied_prob")),
        "odds_implied_prob_norm": _safe_float(row.get("odds_implied_prob_norm")),
    }
    odds_snapshot = {k: v for k, v in odds_snapshot.items() if v is not None}
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

    odds_dt = _parse_timestamp(odds_timestamp)
    prediction_dt = _parse_timestamp(prediction_timestamp)
    feature_freeze_dt = _parse_timestamp(feature_freeze_timestamp)
    if odds_dt is not None and prediction_dt is not None:
        age_seconds = _seconds_between(prediction_dt, odds_dt)
        odds_snapshot["odds_age_seconds_at_prediction"] = age_seconds
        odds_snapshot["odds_age_minutes_at_prediction"] = age_seconds / 60.0
        odds_snapshot["odds_captured_before_prediction"] = age_seconds >= 0
        odds_snapshot["odds_stale_at_prediction"] = (
            age_seconds > stale_odds_after_minutes * 60.0
        )
        odds_snapshot["stale_odds_after_minutes"] = stale_odds_after_minutes

    if odds_dt is not None and feature_freeze_dt is not None:
        freeze_age_seconds = _seconds_between(feature_freeze_dt, odds_dt)
        odds_snapshot["odds_age_seconds_at_feature_freeze"] = freeze_age_seconds
        odds_snapshot["odds_captured_before_feature_freeze"] = freeze_age_seconds >= 0

    jump_dt = _parse_timestamp(jump_datetime)
    if odds_dt is not None and jump_dt is not None:
        odds_snapshot["odds_captured_before_jump"] = (
            _seconds_between(jump_dt, odds_dt) >= 0
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
        "match_method": _first_value(
            row, ("odds_match_method", "market_odds_match_method")
        ),
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
    provenance = {k: v for k, v in provenance.items() if v not in (None, "")}
    if provenance:
        odds_snapshot["odds_provenance"] = provenance
    return odds_snapshot


def _odds_match_method(provenance: Mapping[str, Any]) -> str | None:
    method = provenance.get("match_method") or provenance.get("match_type")
    return str(method).strip() if method not in (None, "") else None


def classify_odds_snapshot_for_ev(
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

    def result(status: str) -> dict[str, Any]:
        valid = status == "valid_pre_jump_dog_odds"
        return {
            "odds_match_status": status,
            "odds_match_method": method,
            "odds_exclusion_reason": None if valid else status,
            "odds_provenance_status": "complete" if valid else "excluded",
            "is_ev_eligible": valid,
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
        return result("timestamp_after_prediction")
    if (
        "odds_captured_before_jump" in snapshot
        and snapshot.get("odds_captured_before_jump") is not True
    ):
        return result("timestamp_after_jump")
    if snapshot.get("odds_stale_at_prediction") is True:
        return result("stale_beyond_ttl")

    source = str(provenance.get("source") or runner.get("odds_source") or "").strip().lower()
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
        if str(provenance.get("odds_race_id")) != str(snapshot_race_id):
            return result("race_id_mismatch")

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
    method_key = str(method or "").strip().lower()
    explicit_identity_match = (
        runner_box is not None
        and odds_box is not None
        and runner_box == odds_box
        and bool(odds_name)
        and _normalize_identity(odds_name) == _normalize_identity(runner_name)
    )
    strict_method_match = method_key in STRICT_ODDS_MATCH_METHODS
    if not explicit_identity_match and not strict_method_match:
        return result("ambiguous_runner_identity")
    if confidence is not None and confidence < 0.99:
        return result("ambiguous_runner_identity")

    return result("valid_pre_jump_dog_odds")


def _odds_valid_for_ev(
    runner: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any],
    *,
    snapshot_race_id: Any = None,
) -> bool:
    return classify_odds_snapshot_for_ev(
        runner,
        odds_snapshot,
        snapshot_race_id=snapshot_race_id,
    )["is_ev_eligible"] is True


def _ev_win(
    win_prob_norm: Any,
    row: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any],
    *,
    snapshot_race_id: Any = None,
) -> float | None:
    if not _odds_valid_for_ev(row, odds_snapshot, snapshot_race_id=snapshot_race_id):
        return None
    probability = _safe_float(win_prob_norm)
    odds = _safe_float(odds_snapshot.get("market_odds_win"))
    if probability is None or odds is None:
        return None
    return probability * odds - 1.0


def _runner_odds_quality_flags(
    row: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any],
    *,
    odds_eligibility: Mapping[str, Any],
) -> list[str]:
    flags = _quality_flags(row)
    for flag in _quality_flags({"quality_flags": row.get("provenance_quality_flags")}):
        _add_quality_flag(flags, flag)
    odds = _safe_float(odds_snapshot.get("market_odds_win"))
    if odds is None:
        _add_quality_flag(flags, "missing_live_odds")
        exclusion_reason = odds_eligibility.get("odds_exclusion_reason")
        if exclusion_reason:
            _add_quality_flag(flags, f"odds_excluded:{exclusion_reason}")
        return flags
    if not odds_snapshot.get("odds_timestamp"):
        _add_quality_flag(flags, "missing_odds_timestamp")
    if odds_snapshot.get("odds_captured_before_prediction") is not True:
        _add_quality_flag(flags, "odds_not_captured_before_prediction")
    if (
        "odds_captured_before_jump" in odds_snapshot
        and odds_snapshot.get("odds_captured_before_jump") is not True
    ):
        _add_quality_flag(flags, "odds_not_captured_before_jump")
    if odds_snapshot.get("odds_stale_at_prediction") is True:
        _add_quality_flag(flags, "stale_live_odds")
    exclusion_reason = odds_eligibility.get("odds_exclusion_reason")
    if exclusion_reason:
        _add_quality_flag(flags, f"odds_excluded:{exclusion_reason}")
    if odds_eligibility.get("is_ev_eligible") is not True:
        _add_quality_flag(flags, "invalid_pre_jump_odds")
    return flags


def _runner_inclusion_reason(row: Mapping[str, Any]) -> str:
    value = row.get("runner_inclusion_reason")
    if value is not None and str(value).strip() != "":
        return str(value)
    flags = set(_quality_flags(row))
    if "optimizer_retained_low_quality_for_runner_alignment" in flags:
        return "model_scored_low_confidence_retained"
    if row.get("quality_filter_status") == "retained_for_runner_alignment":
        return "model_scored_low_confidence_retained"
    return "model_scored"


def _metadata_source_detail(row: Mapping[str, Any]) -> Any:
    value = row.get("metadata_source_detail")
    if isinstance(value, Mapping):
        return dict(value)
    if value is not None and str(value).strip() != "":
        return str(value)
    detail = {}
    if row.get("distance_source") not in (None, ""):
        detail["distance"] = row.get("distance_source")
    if row.get("grade_source") not in (None, ""):
        detail["grade"] = row.get("grade_source")
    return detail or None


def _snapshot_readiness(
    predictions: list[dict[str, Any]],
    *,
    lifecycle_status: Any,
    prediction_timestamp: str,
    feature_freeze_timestamp: str,
    source_runner_completeness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from utils.runner_completeness import analyze_prediction_runner_match

    priced_rows = [
        row
        for row in predictions
        if isinstance(row.get("odds_snapshot"), Mapping)
        and row["odds_snapshot"].get("market_odds_win") is not None
    ]
    missing_live_odds_count = len(predictions) - len(priced_rows)
    stale_count = sum(
        1
        for row in priced_rows
        if row["odds_snapshot"].get("odds_stale_at_prediction") is True
    )
    missing_timestamp_count = sum(
        1
        for row in priced_rows
        if not row["odds_snapshot"].get("odds_timestamp")
    )
    not_before_prediction_count = sum(
        1
        for row in priced_rows
        if row["odds_snapshot"].get("odds_captured_before_prediction") is not True
    )
    not_before_jump_count = sum(
        1
        for row in priced_rows
        if row["odds_snapshot"].get("odds_captured_before_jump") is not True
    )
    missing_odds_explicit = all(
        "missing_live_odds" in (row.get("data_quality_flags") or [])
        for row in predictions
        if not (
            isinstance(row.get("odds_snapshot"), Mapping)
            and row["odds_snapshot"].get("market_odds_win") is not None
        )
    )
    source_report = dict(source_runner_completeness or {})
    source_status = str(source_report.get("status") or "UNVERIFIED")
    runner_match = analyze_prediction_runner_match(predictions, source_report)
    source_verified = bool(source_report)
    requirements = {
        "result_free": True,
        "pre_jump_lifecycle": lifecycle_status == "upcoming_not_jumped",
        "prediction_timestamp_present": bool(prediction_timestamp),
        "feature_freeze_timestamp_present": bool(feature_freeze_timestamp),
        "runner_rows_present": bool(predictions),
        "runner_rows_have_identity": all(
            row.get("dog_name") and row.get("box_number") is not None
            for row in predictions
        )
        if predictions
        else False,
        "runner_rows_have_probabilities": all(
            row.get("win_prob_norm") is not None for row in predictions
        )
        if predictions
        else False,
        "source_runner_set_complete": (
            source_status == "COMPLETE" if source_verified else True
        ),
        "predictions_match_source_runner_set": (
            runner_match.get("status") == "MATCHED" if source_verified else True
        ),
        "priced_runners_have_odds_timestamps": missing_timestamp_count == 0,
        "priced_runners_captured_before_prediction": not_before_prediction_count == 0,
        "priced_runners_captured_before_jump": not_before_jump_count == 0,
        "missing_live_odds_explicit": missing_odds_explicit,
    }
    return {
        "schema_version": "snapshot_readiness_v1",
        "status": "READY" if all(requirements.values()) else "NOT_READY",
        "requirements": requirements,
        "counts": {
            "runner_count": len(predictions),
            "priced_runner_count": len(priced_rows),
            "missing_live_odds_count": missing_live_odds_count,
            "stale_live_odds_count": stale_count,
            "missing_odds_timestamp_count": missing_timestamp_count,
            "odds_not_captured_before_prediction_count": not_before_prediction_count,
            "odds_not_captured_before_jump_count": not_before_jump_count,
        },
        "source_runner_completeness": source_report or None,
        "prediction_runner_match": runner_match if source_verified else None,
    }


def _prediction_rows(
    prediction_result: Mapping[str, Any],
    *,
    prediction_timestamp: str,
    feature_freeze_timestamp: str,
    jump_datetime: str | None,
    stale_odds_after_minutes: float,
) -> list[dict[str, Any]]:
    rows = prediction_result.get("predictions")
    if not isinstance(rows, list):
        rows = prediction_result.get("enhanced_predictions")
    if not isinstance(rows, list):
        return []

    snapshot_rows: list[dict[str, Any]] = []
    race_context = _as_dict(prediction_result.get("race_context"))
    snapshot_race_id = (
        prediction_result.get("race_id")
        or prediction_result.get("raceId")
        or race_context.get("race_id")
    )
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        leaked = sorted(str(key) for key in row.keys() if str(key) in RESULT_FIELD_NAMES)
        if leaked:
            raise ValueError(
                "result field leaked into runner prediction: " + ", ".join(leaked)
            )
        odds_snapshot = _build_odds_snapshot(
            row,
            prediction_timestamp=prediction_timestamp,
            feature_freeze_timestamp=feature_freeze_timestamp,
            jump_datetime=jump_datetime,
            stale_odds_after_minutes=stale_odds_after_minutes,
        )
        win_prob_norm = _safe_float(
            row.get("win_prob_norm", row.get("win_probability"))
        )
        odds_eligibility = classify_odds_snapshot_for_ev(
            row,
            odds_snapshot,
            snapshot_race_id=snapshot_race_id,
        )
        flags = _runner_odds_quality_flags(
            row,
            odds_snapshot,
            odds_eligibility=odds_eligibility,
        )
        odds_provenance = odds_snapshot.get("odds_provenance")
        odds_source = (
            odds_provenance.get("source")
            if isinstance(odds_provenance, Mapping)
            else None
        )
        odds_match_confidence = (
            odds_provenance.get("match_confidence")
            if isinstance(odds_provenance, Mapping)
            else None
        )
        snapshot_rows.append(
            {
                "dog_name": row.get("dog_clean_name") or row.get("dog_name") or row.get("name"),
                "box_number": _safe_int(row.get("box_number")),
                "win_prob_raw": _safe_float(
                    row.get("win_prob_raw", row.get("win_probability_raw"))
                ),
                "win_prob_norm": win_prob_norm,
                "predicted_rank": _safe_int(row.get("predicted_rank", row.get("rank"))),
                "confidence_score": _safe_float(
                    row.get("confidence_score", row.get("confidence"))
                ),
                "odds": _safe_float(odds_snapshot.get("market_odds_win")),
                "odds_timestamp": odds_snapshot.get("odds_timestamp"),
                "odds_source": odds_source,
                "odds_match_confidence": _safe_float(odds_match_confidence),
                "odds_match_status": odds_eligibility.get("odds_match_status"),
                "odds_match_method": odds_eligibility.get("odds_match_method"),
                "odds_exclusion_reason": odds_eligibility.get("odds_exclusion_reason"),
                "odds_provenance_status": odds_eligibility.get("odds_provenance_status"),
                "odds_snapshot": odds_snapshot,
                "ev_win": _ev_win(
                    win_prob_norm,
                    row,
                    odds_snapshot,
                    snapshot_race_id=snapshot_race_id,
                ),
                "history_source": row.get("history_source"),
                "history_match_status": row.get("history_match_status"),
                "db_history_match_status": row.get("db_history_match_status"),
                "db_result_history_count": _safe_int(row.get("db_result_history_count")),
                "runner_inclusion_reason": _runner_inclusion_reason(row),
                "distance_source": row.get("distance_source"),
                "grade_source": row.get("grade_source"),
                "metadata_source_detail": _metadata_source_detail(row),
                "metadata_is_leakage_safe": _safe_bool(row.get("metadata_is_leakage_safe")),
                "rejected_metadata_sources": _quality_flags(
                    {"quality_flags": row.get("rejected_metadata_sources")}
                ),
                "data_quality_flags": flags,
                "data_shape": {
                    key: row.get(key)
                    for key in (
                        "parser_context",
                        "target_field_warning",
                        "distance_source",
                        "grade_source",
                        "metadata_source_detail",
                        "metadata_is_leakage_safe",
                        "rejected_metadata_sources",
                        "history_source",
                        "history_match_status",
                        "db_history_match_status",
                        "field_size_source",
                        "csv_historical_races",
                        "csv_prefixed_history_rows",
                        "csv_blank_history_rows",
                        "csv_historical_sources",
                        "csv_history_rows_dropped_post_target",
                    )
                    if row.get(key) not in (None, "")
                },
            }
        )
    return snapshot_rows


def _race_identity(prediction_result: Mapping[str, Any], lifecycle_data: Mapping[str, Any]) -> dict[str, Any]:
    race_context = _as_dict(prediction_result.get("race_context"))
    race_date = (
        lifecycle_data.get("race_date")
        or race_context.get("race_date")
        or prediction_result.get("race_date")
    )
    venue = (
        lifecycle_data.get("venue")
        or race_context.get("venue")
        or prediction_result.get("venue")
    )
    race_number = (
        lifecycle_data.get("race_number")
        or race_context.get("race_number")
        or prediction_result.get("race_number")
    )
    stable_parts = [
        str(part)
        for part in (race_date, venue, race_number)
        if part not in (None, "")
    ]
    return {
        "race_date": race_date,
        "venue": venue,
        "race_number": _safe_int(race_number),
        "jump_time": lifecycle_data.get("jump_time")
        or race_context.get("jump_time")
        or prediction_result.get("race_time"),
        "stable_race_key": "|".join(stable_parts) if stable_parts else None,
    }


def build_prediction_snapshot(
    prediction_result: Mapping[str, Any],
    *,
    source_file_path: str | None = None,
    lifecycle: Any = None,
    prediction_timestamp: str | None = None,
    feature_freeze_timestamp: str | None = None,
    stale_odds_after_minutes: float = 30.0,
    source_runner_completeness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a result-free snapshot record from an already computed prediction.

    The snapshot intentionally contains no target-result labels. It can be
    written by callers only after they choose a safe storage target.
    """

    lifecycle_data = _as_dict(lifecycle)
    lifecycle_status = (
        lifecycle_data.get("status")
        or lifecycle_data.get("lifecycle_status")
        or prediction_result.get("lifecycle_status")
    )
    timestamp = prediction_timestamp or _iso_now()
    feature_freeze = feature_freeze_timestamp or timestamp
    jump_datetime = (
        lifecycle_data.get("jump_datetime")
        or lifecycle_data.get("start_datetime")
        or prediction_result.get("jump_datetime")
        or prediction_result.get("start_datetime")
        or prediction_result.get("race_start_time")
    )
    race_id = (
        prediction_result.get("race_id")
        or prediction_result.get("raceId")
        or (Path(source_file_path).stem if source_file_path else None)
    )
    model_version = (
        prediction_result.get("model_version")
        or prediction_result.get("primary_model_id")
        or prediction_result.get("model_info")
        or "unknown"
    )
    identity = _race_identity(prediction_result, lifecycle_data)
    if source_runner_completeness is None and source_file_path:
        try:
            from utils.runner_completeness import analyze_csv_runner_completeness

            source_path = Path(source_file_path)
            if source_path.exists():
                source_runner_completeness = analyze_csv_runner_completeness(
                    source_path
                ).as_dict()
        except Exception:
            source_runner_completeness = None
    source_runner_completeness = dict(source_runner_completeness or {})
    frozen_participants = list(source_runner_completeness.get("participants") or [])

    snapshot = {
        "schema_version": "prediction_snapshot_v1",
        "race_id": race_id,
        "stable_race_key": identity["stable_race_key"],
        "race_date": identity["race_date"],
        "venue": identity["venue"],
        "race_number": identity["race_number"],
        "jump_time": identity["jump_time"],
        "jump_datetime": jump_datetime,
        "source_file_path": source_file_path,
        "lifecycle_status": lifecycle_status,
        "lifecycle_status_reason": lifecycle_data.get("status_reason")
        or lifecycle_data.get("lifecycle_status_reason"),
        "model_version": model_version,
        "prediction_timestamp": timestamp,
        "feature_freeze_timestamp": feature_freeze,
        "is_pre_jump_snapshot": lifecycle_status == "upcoming_not_jumped",
        "snapshot_state": (
            "pre_jump_feature_freeze"
            if lifecycle_status == "upcoming_not_jumped"
            else "not_bet_qualified_lifecycle"
        ),
        "predictions": _prediction_rows(
            prediction_result,
            prediction_timestamp=timestamp,
            feature_freeze_timestamp=feature_freeze,
            jump_datetime=str(jump_datetime) if jump_datetime else None,
            stale_odds_after_minutes=stale_odds_after_minutes,
        ),
        "source_runner_completeness": source_runner_completeness or None,
        "expected_runner_count": source_runner_completeness.get("runner_count"),
        "frozen_participants": frozen_participants,
        "runner_set_status": source_runner_completeness.get("status"),
        "runner_set_complete": source_runner_completeness.get("status") == "COMPLETE"
        if source_runner_completeness
        else None,
        "data_quality_flags": list(prediction_result.get("quality_flags") or []),
        "snapshot_provenance": {
            "builder": "accuracy_program.snapshots.build_prediction_snapshot",
            "source_file_path": source_file_path,
        },
    }
    snapshot["snapshot_readiness"] = _snapshot_readiness(
        snapshot["predictions"],
        lifecycle_status=lifecycle_status,
        prediction_timestamp=timestamp,
        feature_freeze_timestamp=feature_freeze,
        source_runner_completeness=source_runner_completeness,
    )
    assert_no_result_fields(snapshot)
    return snapshot


def assert_no_result_fields(value: Any) -> None:
    """Raise ValueError if a snapshot contains result/label fields."""

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_text = str(key)
                if key_text.lower() in RESULT_FIELD_NAMES:
                    raise ValueError(f"result field leaked into snapshot: {path}.{key_text}")
                visit(child, f"{path}.{key_text}")
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                visit(child, f"{path}[{idx}]")

    visit(value, "snapshot")


def _slug(value: Any, *, fallback: str = "unknown") -> str:
    raw = str(value or fallback).strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")
    return slug[:120] or fallback


def snapshot_output_path(snapshot: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Return the non-overwriting JSON path for a frozen snapshot."""

    race_date = _slug(snapshot.get("race_date"))
    venue = _slug(snapshot.get("venue"))
    race_number = _slug(snapshot.get("race_number"), fallback="race")
    race_id = _slug(snapshot.get("race_id"))
    timestamp = _slug(str(snapshot.get("prediction_timestamp") or _iso_now()).replace(":", ""))
    digest = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    directory = Path(output_dir) / race_date / venue
    return directory / f"race-{race_number}_{race_id}_{timestamp}_{digest}.json"


def persist_prediction_snapshot(
    snapshot: Mapping[str, Any],
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Persist a result-free snapshot JSON and append an audit manifest line.

    The function writes only under the caller-provided output directory and
    never overwrites an existing snapshot file.
    """

    assert_no_result_fields(snapshot)
    path = snapshot_output_path(snapshot, output_dir)
    manifest_path = Path(output_dir) / "manifest.jsonl"
    report = {
        "status": "dry_run" if dry_run else "persisted",
        "path": str(path),
        "manifest_path": str(manifest_path),
        "race_id": snapshot.get("race_id"),
        "stable_race_key": snapshot.get("stable_race_key"),
        "prediction_timestamp": snapshot.get("prediction_timestamp"),
        "lifecycle_status": snapshot.get("lifecycle_status"),
    }
    if dry_run:
        return report

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "prediction_snapshot_manifest_v1",
        "created_at": _iso_now(),
        "snapshot_path": str(path),
        "race_id": snapshot.get("race_id"),
        "stable_race_key": snapshot.get("stable_race_key"),
        "prediction_timestamp": snapshot.get("prediction_timestamp"),
        "feature_freeze_timestamp": snapshot.get("feature_freeze_timestamp"),
        "lifecycle_status": snapshot.get("lifecycle_status"),
        "runner_count": len(snapshot.get("predictions") or []),
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, default=str) + "\n")

    report["bytes"] = len(text.encode("utf-8"))
    return report
