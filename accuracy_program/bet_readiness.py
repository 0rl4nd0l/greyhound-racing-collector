"""Prediction quality gates that separate prediction from betting advice."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping


ABSTAIN_FLAGS = {
    "stale_form_guide",
    "jumped_pending_results",
    "missing_live_odds",
    "stale_live_odds",
    "thin_history",
    "probabilities_too_uniform",
    "single_model_only",
    "market_model_disagreement",
    "low_calibration_confidence",
}


def _predictions(prediction_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = prediction_result.get("predictions")
    if not isinstance(rows, list):
        rows = prediction_result.get("enhanced_predictions")
    return [row for row in rows or [] if isinstance(row, dict)]


def _append_flag(target: dict[str, Any], flag: str) -> None:
    flags = target.get("quality_flags")
    if not isinstance(flags, list):
        flags = []
    if flag not in flags:
        flags.append(flag)
    target["quality_flags"] = flags


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


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


def _odds_timestamp(row: Mapping[str, Any]) -> datetime | None:
    odds_snapshot = row.get("odds_snapshot")
    candidates = []
    if isinstance(odds_snapshot, Mapping):
        candidates.extend(
            odds_snapshot.get(key)
            for key in (
                "odds_timestamp",
                "market_odds_timestamp",
                "live_odds_timestamp",
                "odds_updated_at",
                "odds_last_updated",
            )
        )
    candidates.extend(
        row.get(key)
        for key in (
            "odds_timestamp",
            "market_odds_timestamp",
            "live_odds_timestamp",
            "odds_updated_at",
            "odds_last_updated",
        )
    )
    for candidate in candidates:
        parsed = _parse_timestamp(candidate)
        if parsed is not None:
            return parsed
    return None


def _history_count(row: Mapping[str, Any]) -> int | None:
    for key in (
        "csv_historical_races",
        "csv_prefixed_history_rows",
        "historical_races",
        "history_races",
        "starts",
    ):
        value = row.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except Exception:
            continue
    return None


def _probability_uniform_flag(rows: list[Mapping[str, Any]], threshold: float) -> bool:
    probs = [
        _safe_float(row.get("win_prob_norm", row.get("win_probability")))
        for row in rows
    ]
    probs = [p for p in probs if p is not None]
    if len(probs) < 2:
        return False
    return max(probs) - min(probs) <= threshold


def _model_count(prediction_result: Mapping[str, Any]) -> int | None:
    for key in ("ensemble_models_used", "ensemble_models", "model_count"):
        value = prediction_result.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    model_ids = prediction_result.get("model_ids_used")
    if isinstance(model_ids, list):
        return len(model_ids)
    return None


def apply_bet_readiness_gates(
    prediction_result: dict[str, Any],
    *,
    lifecycle: Mapping[str, Any] | None = None,
    min_history_races: int = 3,
    uniform_threshold: float = 0.015,
    stale_odds_after_minutes: float = 30.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Add abstain gates without changing rank, probability, or EV values."""

    rows = _predictions(prediction_result)
    lifecycle = lifecycle or prediction_result.get("lifecycle") or {}
    lifecycle_status = (
        lifecycle.get("status")
        or lifecycle.get("lifecycle_status")
        or prediction_result.get("lifecycle_status")
    )
    flags: list[str] = []
    reasons: dict[str, Any] = {}

    def add(flag: str, reason: Any = True) -> None:
        if flag not in flags:
            flags.append(flag)
        reasons[flag] = reason

    if lifecycle_status == "stale_form_guide":
        add("stale_form_guide", "race is a stale form guide, not a live pre-jump target")
    if lifecycle_status == "jumped_pending_results":
        add("jumped_pending_results", "race has jumped and official labels are still pending")
    if lifecycle_status == "resulted":
        add("jumped_pending_results", "race has result evidence and is not bet-eligible")

    missing_odds = []
    stale_odds = []
    now_dt = now or datetime.now()
    for row in rows:
        odds = _safe_float(row.get("market_odds_win", row.get("odds_win", row.get("live_odds"))))
        ev = _safe_float(row.get("ev_win"))
        if odds is None or odds <= 1:
            _append_flag(row, "missing_live_odds")
            missing_odds.append(row.get("dog_clean_name") or row.get("dog_name"))
        elif ev is None:
            _append_flag(row, "missing_live_odds")
            missing_odds.append(row.get("dog_clean_name") or row.get("dog_name"))
        odds_ts = _odds_timestamp(row)
        if odds_ts is not None:
            compare_now = now_dt
            if odds_ts.tzinfo is not None and compare_now.tzinfo is None:
                compare_now = compare_now.replace(tzinfo=odds_ts.tzinfo)
            elif odds_ts.tzinfo is None and compare_now.tzinfo is not None:
                compare_now = compare_now.replace(tzinfo=None)
            age_minutes = (compare_now - odds_ts).total_seconds() / 60.0
            if age_minutes > stale_odds_after_minutes:
                _append_flag(row, "stale_live_odds")
                stale_odds.append(row.get("dog_clean_name") or row.get("dog_name"))
    if rows and missing_odds:
        add("missing_live_odds", {"runner_count": len(missing_odds)})
    if rows and stale_odds:
        add(
            "stale_live_odds",
            {
                "runner_count": len(stale_odds),
                "stale_after_minutes": stale_odds_after_minutes,
            },
        )

    thin = []
    for row in rows:
        count = _history_count(row)
        if count is not None and count < min_history_races:
            _append_flag(row, "thin_history")
            thin.append(row.get("dog_clean_name") or row.get("dog_name"))
    if thin:
        add("thin_history", {"runner_count": len(thin), "min_history_races": min_history_races})

    if _probability_uniform_flag(rows, uniform_threshold):
        add("probabilities_too_uniform", {"threshold": uniform_threshold})
        for row in rows:
            _append_flag(row, "probabilities_too_uniform")

    count = _model_count(prediction_result)
    if count is None:
        if any("single_model_no_ensemble_agreement" in (row.get("quality_flags") or []) for row in rows):
            add("single_model_only", "single model quality flag already present")
    elif count <= 1:
        add("single_model_only", {"model_count": count})
        for row in rows:
            _append_flag(row, "single_model_only")

    market_context = prediction_result.get("market_context") or {}
    disagreement_count = int(market_context.get("large_disagreement_count") or 0)
    if disagreement_count > 0 or any(
        "large_model_market_disagreement" in (row.get("quality_flags") or [])
        for row in rows
    ):
        add("market_model_disagreement", {"large_disagreement_count": disagreement_count})
        for row in rows:
            if "large_model_market_disagreement" in (row.get("quality_flags") or []):
                _append_flag(row, "market_model_disagreement")

    model_version = str(prediction_result.get("model_version") or "").strip().lower()
    metrics = prediction_result.get("metrics")
    if (
        not model_version
        or model_version == "unknown"
        or prediction_result.get("degraded")
        or prediction_result.get("synthetic")
        or metrics in (None, {})
    ):
        add("low_calibration_confidence", "no current temporal calibration metrics attached")

    flag_map = {flag: flag in flags for flag in sorted(ABSTAIN_FLAGS)}
    probs = [
        _safe_float(row.get("win_prob_norm", row.get("win_probability")))
        for row in rows
    ]
    probs = [p for p in probs if p is not None]
    history_counts = [_history_count(row) for row in rows]
    history_counts = [count for count in history_counts if count is not None]
    market_context = prediction_result.get("market_context") or {}
    status = "bet_qualified" if not flags else "prediction_available_not_bet_qualified"
    result = {
        "schema_version": "bet_readiness_v1",
        "status": status,
        "ready": not flags,
        "abstain": bool(flags),
        "abstain_reasons": flags,
        "bet_qualified": not flags,
        "abstain_flags": flags,
        "flags": flag_map,
        "reasons": reasons,
        "inputs": {
            "lifecycle_status": lifecycle_status,
            "market_odds_count": market_context.get("market_odds_count"),
            "stale_odds_count": len(stale_odds),
            "stale_odds_after_minutes": stale_odds_after_minutes,
            "min_history_races": min(history_counts) if history_counts else None,
            "probability_spread": (
                max(probs) - min(probs) if len(probs) >= 2 else None
            ),
            "calibration_metrics_present": metrics not in (None, {}),
        },
    }
    prediction_result["bet_readiness"] = result
    return result
