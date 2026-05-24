"""Prediction-before-result snapshot construction and safe persistence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


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
    "result",
    "result_status",
    "results_status",
    "scraped_finish_position",
    "scraped_raw_result",
    "winner_margin",
    "winner_name",
    "winner_odds",
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


def _quality_flags(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("quality_flags") or row.get("data_quality_flags") or []
    if isinstance(raw, str):
        return [raw]
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


def _build_odds_snapshot(
    row: Mapping[str, Any],
    *,
    prediction_timestamp: str,
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

    odds_dt = _parse_timestamp(odds_timestamp)
    prediction_dt = _parse_timestamp(prediction_timestamp)
    if odds_dt is not None and prediction_dt is not None:
        age_seconds = _seconds_between(prediction_dt, odds_dt)
        odds_snapshot["odds_age_seconds_at_prediction"] = age_seconds
        odds_snapshot["odds_age_minutes_at_prediction"] = age_seconds / 60.0
        odds_snapshot["odds_captured_before_prediction"] = age_seconds >= 0
        odds_snapshot["odds_stale_at_prediction"] = (
            age_seconds > stale_odds_after_minutes * 60.0
        )
        odds_snapshot["stale_odds_after_minutes"] = stale_odds_after_minutes

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
        "match_type": _first_value(row, ("odds_match_type", "market_odds_match_type")),
        "match_key": _first_value(row, ("odds_match_key", "market_odds_match_key")),
        "match_confidence": _first_value(
            row, ("odds_match_confidence", "market_odds_match_confidence")
        ),
    }
    provenance = {k: v for k, v in provenance.items() if v not in (None, "")}
    if provenance:
        odds_snapshot["odds_provenance"] = provenance
    return odds_snapshot


def _odds_valid_for_ev(odds_snapshot: Mapping[str, Any]) -> bool:
    odds = _safe_float(odds_snapshot.get("market_odds_win"))
    if odds is None or odds <= 1.0:
        return False
    if not odds_snapshot.get("odds_timestamp"):
        return False
    if odds_snapshot.get("odds_captured_before_prediction") is not True:
        return False
    if (
        "odds_captured_before_jump" in odds_snapshot
        and odds_snapshot.get("odds_captured_before_jump") is not True
    ):
        return False
    return True


def _ev_win(win_prob_norm: Any, odds_snapshot: Mapping[str, Any]) -> float | None:
    if not _odds_valid_for_ev(odds_snapshot):
        return None
    probability = _safe_float(win_prob_norm)
    odds = _safe_float(odds_snapshot.get("market_odds_win"))
    if probability is None or odds is None:
        return None
    return probability * odds - 1.0


def _runner_odds_quality_flags(
    row: Mapping[str, Any], odds_snapshot: Mapping[str, Any]
) -> list[str]:
    flags = _quality_flags(row)
    odds = _safe_float(odds_snapshot.get("market_odds_win"))
    if odds is None:
        _add_quality_flag(flags, "missing_live_odds")
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
    if not _odds_valid_for_ev(odds_snapshot):
        _add_quality_flag(flags, "invalid_pre_jump_odds")
    return flags


def _snapshot_readiness(
    predictions: list[dict[str, Any]],
    *,
    lifecycle_status: Any,
    prediction_timestamp: str,
    feature_freeze_timestamp: str,
) -> dict[str, Any]:
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
    requirements = {
        "result_free": True,
        "pre_jump_lifecycle": lifecycle_status == "upcoming_not_jumped",
        "prediction_timestamp_present": bool(prediction_timestamp),
        "feature_freeze_timestamp_present": bool(feature_freeze_timestamp),
        "runner_rows_present": bool(predictions),
        "runner_rows_have_probabilities": all(
            row.get("win_prob_norm") is not None for row in predictions
        )
        if predictions
        else False,
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
    }


def _prediction_rows(
    prediction_result: Mapping[str, Any],
    *,
    prediction_timestamp: str,
    jump_datetime: str | None,
    stale_odds_after_minutes: float,
) -> list[dict[str, Any]]:
    rows = prediction_result.get("predictions")
    if not isinstance(rows, list):
        rows = prediction_result.get("enhanced_predictions")
    if not isinstance(rows, list):
        return []

    snapshot_rows: list[dict[str, Any]] = []
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
            jump_datetime=jump_datetime,
            stale_odds_after_minutes=stale_odds_after_minutes,
        )
        win_prob_norm = _safe_float(
            row.get("win_prob_norm", row.get("win_probability"))
        )
        flags = _runner_odds_quality_flags(row, odds_snapshot)
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
                "odds_snapshot": odds_snapshot,
                "ev_win": _ev_win(win_prob_norm, odds_snapshot),
                "data_quality_flags": flags,
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
            jump_datetime=str(jump_datetime) if jump_datetime else None,
            stale_odds_after_minutes=stale_odds_after_minutes,
        ),
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
