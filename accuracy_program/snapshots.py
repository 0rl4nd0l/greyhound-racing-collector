"""Prediction-before-result snapshot construction and safe persistence."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from accuracy_program.calibration import power_normalize_prediction_group
from accuracy_program.odds_provenance import (
    build_odds_snapshot as _build_canonical_odds_snapshot,
    classify_odds_snapshot_for_ev as _classify_canonical_odds_snapshot_for_ev,
    ev_win_if_eligible as _canonical_ev_win_if_eligible,
)

try:
    from config.venue_mapping import normalize_venue
except Exception:

    def normalize_venue(value: str) -> str:
        return re.sub(r"[^A-Z0-9_]", "", str(value or "").upper())


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


def _build_odds_snapshot(
    row: Mapping[str, Any],
    *,
    prediction_timestamp: str,
    feature_freeze_timestamp: str,
    jump_datetime: str | None,
    stale_odds_after_minutes: float,
) -> dict[str, Any]:
    return _build_canonical_odds_snapshot(
        row,
        prediction_timestamp=prediction_timestamp,
        feature_freeze_timestamp=feature_freeze_timestamp,
        jump_datetime=jump_datetime,
        stale_odds_after_minutes=stale_odds_after_minutes,
    )


def classify_odds_snapshot_for_ev(
    runner: Mapping[str, Any],
    odds_snapshot: Mapping[str, Any] | None = None,
    *,
    snapshot_race_id: Any = None,
) -> dict[str, Any]:
    """Classify leakage-safe dog-level odds eligibility for EV calculation."""
    return _classify_canonical_odds_snapshot_for_ev(
        runner,
        odds_snapshot,
        snapshot_race_id=snapshot_race_id,
    )


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
    return _canonical_ev_win_if_eligible(
        win_prob_norm,
        row,
        odds_snapshot,
        snapshot_race_id=snapshot_race_id,
    )


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
        "odds_captured_before_feature_freeze" in odds_snapshot
        and odds_snapshot.get("odds_captured_before_feature_freeze") is not True
    ):
        _add_quality_flag(flags, "odds_not_captured_before_feature_freeze")
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


def _ev_readiness(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether captured odds are trustworthy enough to expose EV."""

    priced_count = 0
    eligible_count = 0
    ev_present_count = 0
    ev_leak_count = 0
    exclusion_counts: Counter[str] = Counter()
    for row in predictions:
        odds = row.get("odds")
        has_odds = odds is not None
        if has_odds:
            priced_count += 1
        match_status = str(row.get("odds_match_status") or "")
        provenance_status = str(row.get("odds_provenance_status") or "")
        eligible = (
            has_odds
            and match_status == "valid_pre_jump_dog_odds"
            and provenance_status == "complete"
        )
        if eligible:
            eligible_count += 1
        ev_present = row.get("ev_win") is not None
        if ev_present:
            ev_present_count += 1
        if ev_present and not eligible:
            ev_leak_count += 1
        if has_odds and not eligible:
            reason = (
                row.get("odds_exclusion_reason")
                or row.get("odds_match_status")
                or "unknown_odds_exclusion"
            )
            exclusion_counts[str(reason)] += 1
        if not has_odds:
            exclusion_counts["missing_live_odds"] += 1

    runner_count = len(predictions)
    requirements = {
        "all_runners_priced": runner_count > 0 and priced_count == runner_count,
        "all_priced_odds_ev_eligible": priced_count > 0 and eligible_count == priced_count,
        "ev_present_only_for_eligible_odds": ev_leak_count == 0,
        "ev_null_for_unpriced_or_ineligible": (
            ev_present_count == eligible_count
        ),
    }
    return {
        "schema_version": "ev_readiness_v1",
        "status": "EV_READY" if all(requirements.values()) else "EV_NOT_READY",
        "runner_count": runner_count,
        "priced_runner_count": priced_count,
        "ev_eligible_runner_count": eligible_count,
        "ev_present_runner_count": ev_present_count,
        "ev_null_runner_count": runner_count - ev_present_count,
        "ev_leak_count": ev_leak_count,
        "odds_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "requirements": requirements,
    }


def _snapshot_readiness(
    predictions: list[dict[str, Any]],
    *,
    lifecycle_status: Any,
    prediction_timestamp: str,
    feature_freeze_timestamp: str,
    source_runner_completeness: Mapping[str, Any] | None = None,
    final_runner_set_verification: Mapping[str, Any] | None = None,
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
    not_before_feature_freeze_count = sum(
        1
        for row in priced_rows
        if (
            "odds_captured_before_feature_freeze" in row["odds_snapshot"]
            and row["odds_snapshot"].get("odds_captured_before_feature_freeze") is not True
        )
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
    final_runner_report = dict(final_runner_set_verification or {})
    final_runner_status = str(final_runner_report.get("final_runner_set_status") or "")
    ev_readiness = _ev_readiness(predictions)
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
        "priced_runners_captured_before_feature_freeze": not_before_feature_freeze_count == 0,
        "priced_runners_captured_before_jump": not_before_jump_count == 0,
        "missing_live_odds_explicit": missing_odds_explicit,
    }
    if final_runner_report:
        requirements["final_runner_set_verified"] = final_runner_status == "verified"
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
            "odds_not_captured_before_feature_freeze_count": not_before_feature_freeze_count,
            "odds_not_captured_before_jump_count": not_before_jump_count,
        },
        "source_runner_completeness": source_report or None,
        "prediction_runner_match": runner_match if source_verified else None,
        "final_runner_set_verification": final_runner_report or None,
        "ev_readiness": ev_readiness,
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


def _apply_report_only_calibration(
    predictions: list[dict[str, Any]],
    calibration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    algorithm = str(calibration.get("algorithm") or "")
    if algorithm != "power_normalize_per_race":
        raise ValueError("unsupported_report_only_calibration_algorithm")
    input_key = str(calibration.get("input_probability_key") or "win_prob_norm")
    output_key = str(
        calibration.get("output_probability_key")
        or "calibrated_win_prob_report_only"
    )
    output_rank_key = str(
        calibration.get("output_rank_key")
        or "calibrated_predicted_rank_report_only"
    )
    alpha = calibration.get("alpha")
    original_probabilities = [row.get(input_key) for row in predictions]
    original_ranks = [row.get("predicted_rank") for row in predictions]
    calibrated = power_normalize_prediction_group(
        predictions,
        alpha=alpha,
        input_key=input_key,
        output_key=output_key,
        output_rank_key=output_rank_key,
    )
    return calibrated, {
        "algorithm": algorithm,
        "alpha": float(alpha),
        "input_probability_key": input_key,
        "output_probability_key": output_key,
        "output_rank_key": output_rank_key,
        "status": "APPLIED_REPORT_ONLY",
        "canonical_probabilities_unchanged": (
            [row.get(input_key) for row in calibrated] == original_probabilities
        ),
        "canonical_ranks_unchanged": (
            [row.get("predicted_rank") for row in calibrated] == original_ranks
        ),
        "uses_labels_at_runtime": False,
        "uses_odds_at_runtime": False,
        "model_artifact_written": False,
        "registry_mutation_allowed": False,
        "promotion_allowed": False,
        "betting_allowed": False,
    }


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
    final_runner_set_verification: Mapping[str, Any] | None = None,
    report_only_calibration: Mapping[str, Any] | None = None,
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
    final_runner_set_verification = dict(final_runner_set_verification or {})
    prediction_rows = _prediction_rows(
        prediction_result,
        prediction_timestamp=timestamp,
        feature_freeze_timestamp=feature_freeze,
        jump_datetime=str(jump_datetime) if jump_datetime else None,
        stale_odds_after_minutes=stale_odds_after_minutes,
    )
    report_only_calibration_state: dict[str, Any] | None = None
    if report_only_calibration is not None:
        prediction_rows, report_only_calibration_state = _apply_report_only_calibration(
            prediction_rows,
            report_only_calibration,
        )

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
        "predictions": prediction_rows,
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
    if report_only_calibration_state is not None:
        snapshot["report_only_calibration"] = report_only_calibration_state
    if final_runner_set_verification:
        snapshot.update(
            {
                "final_runner_set_status": final_runner_set_verification.get(
                    "final_runner_set_status"
                ),
                "final_runner_set_source": final_runner_set_verification.get(
                    "final_runner_set_source"
                )
                or final_runner_set_verification.get("final_runner_source"),
                "final_runner_set_source_url": final_runner_set_verification.get(
                    "final_runner_set_source_url"
                )
                or final_runner_set_verification.get("final_runner_source_url"),
                "final_runner_set_mismatch_reason": final_runner_set_verification.get(
                    "mismatch_reason"
                ),
                "canonical_active_boxes": final_runner_set_verification.get(
                    "canonical_active_boxes"
                )
                or final_runner_set_verification.get("final_runner_boxes"),
                "source_active_boxes": final_runner_set_verification.get(
                    "source_active_boxes"
                ),
                "canonical_scratch_boxes": final_runner_set_verification.get(
                    "canonical_scratch_boxes"
                )
                or final_runner_set_verification.get("scratched_boxes"),
                "source_reserve_boxes": final_runner_set_verification.get(
                    "source_reserve_boxes"
                ),
                "final_runner_set_verification": final_runner_set_verification,
            }
        )
    snapshot["snapshot_readiness"] = _snapshot_readiness(
        snapshot["predictions"],
        lifecycle_status=lifecycle_status,
        prediction_timestamp=timestamp,
        feature_freeze_timestamp=feature_freeze,
        source_runner_completeness=source_runner_completeness,
        final_runner_set_verification=final_runner_set_verification,
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
    require_final_runner_verification: bool = False,
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
        "final_runner_set_status": snapshot.get("final_runner_set_status"),
    }
    if dry_run:
        return report
    if (
        require_final_runner_verification
        and snapshot.get("lifecycle_status") == "upcoming_not_jumped"
        and snapshot.get("final_runner_set_status") != "verified"
    ):
        report["status"] = "skipped_pre_jump_runner_set_unverified"
        report["reason"] = "pre_jump_runner_set_unverified"
        report["final_runner_set_mismatch_reason"] = snapshot.get(
            "final_runner_set_mismatch_reason"
        )
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
        "final_runner_set_status": snapshot.get("final_runner_set_status"),
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, default=str) + "\n")

    report["bytes"] = len(text.encode("utf-8"))
    return report
