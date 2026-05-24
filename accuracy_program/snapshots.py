"""Prediction-before-result snapshot construction."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


RESULT_FIELD_NAMES = {
    "actual_results",
    "actual_winner",
    "beaten_margin",
    "finish_position",
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


def _prediction_rows(prediction_result: Mapping[str, Any]) -> list[dict[str, Any]]:
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
        odds_snapshot = {
            "market_odds_win": _safe_float(
                row.get("market_odds_win", row.get("odds_win", row.get("live_odds")))
            ),
            "odds_implied_prob": _safe_float(row.get("odds_implied_prob")),
            "odds_implied_prob_norm": _safe_float(row.get("odds_implied_prob_norm")),
        }
        odds_snapshot = {k: v for k, v in odds_snapshot.items() if v is not None}
        odds_timestamp = (
            row.get("odds_timestamp")
            or row.get("market_odds_timestamp")
            or row.get("live_odds_timestamp")
            or row.get("odds_updated_at")
            or row.get("odds_last_updated")
        )
        if odds_timestamp:
            odds_snapshot["odds_timestamp"] = str(odds_timestamp)
        snapshot_rows.append(
            {
                "dog_name": row.get("dog_clean_name") or row.get("dog_name") or row.get("name"),
                "box_number": _safe_int(row.get("box_number")),
                "win_prob_raw": _safe_float(
                    row.get("win_prob_raw", row.get("win_probability_raw"))
                ),
                "win_prob_norm": _safe_float(
                    row.get("win_prob_norm", row.get("win_probability"))
                ),
                "rank": _safe_int(row.get("predicted_rank", row.get("rank"))),
                "confidence": _safe_float(
                    row.get("confidence_score", row.get("confidence"))
                ),
                "odds_snapshot": odds_snapshot,
                "ev_win": _safe_float(row.get("ev_win")),
                "data_quality_flags": list(row.get("quality_flags") or []),
            }
        )
    return snapshot_rows


def build_prediction_snapshot(
    prediction_result: Mapping[str, Any],
    *,
    source_file_path: str | None = None,
    lifecycle: Any = None,
    prediction_timestamp: str | None = None,
    feature_freeze_timestamp: str | None = None,
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

    snapshot = {
        "race_id": race_id,
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
        "predictions": _prediction_rows(prediction_result),
        "data_quality_flags": list(prediction_result.get("quality_flags") or []),
    }
    assert_no_result_fields(snapshot)
    return snapshot


def assert_no_result_fields(value: Any) -> None:
    """Raise ValueError if a snapshot contains result/label fields."""

    def visit(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                key_text = str(key)
                if key_text in RESULT_FIELD_NAMES:
                    raise ValueError(f"result field leaked into snapshot: {path}.{key_text}")
                visit(child, f"{path}.{key_text}")
        elif isinstance(node, list):
            for idx, child in enumerate(node):
                visit(child, f"{path}[{idx}]")

    visit(value, "snapshot")
