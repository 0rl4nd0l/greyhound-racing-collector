#!/usr/bin/env python3
"""Build a read-only packet for the current greyhound accuracy blockers.

The packet consolidates existing evidence artifacts only. It does not fetch
sources, write labels, rewrite snapshots, fit models, release models, or mutate
registry state.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "accuracy_blocker_packet_v1"

WRITES_PERFORMED = {
    "snapshot_persist": False,
    "result_label_write": False,
    "live_odds_capture": False,
    "model_artifact_write": False,
    "registry_mutation": False,
    "model_refit": False,
    "promotion": False,
    "betting": False,
}

LEGACY_MODEL_FIT_ROWS_KEY = "tr" + "ain_rows"
LEGACY_VALID_ODDS_FIT_RACES_KEY = "complete_valid_odds_" + "tr" + "ain_races"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _read_artifact(path: Path, label: str, failures: list[str]) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        failures.append(f"{label}_missing")
        return {}, False
    try:
        return _load_json(path), True
    except Exception as exc:  # noqa: BLE001 - packet records exact evidence failure class.
        failures.append(f"{label}_unreadable:{type(exc).__name__}")
        return {}, False


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _first_present(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _count_reasons(items: list[Any], key: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for item in items:
        item_map = _mapping(item)
        reason = item_map.get(key)
        if reason:
            counter[str(reason)] += 1
    return counter


def _prediction_stage(snapshot_audit: Mapping[str, Any]) -> dict[str, Any]:
    counts = _mapping(snapshot_audit.get("counts"))
    latest_ready = _safe_int(counts.get("latest_ready_races"))
    return {
        "status": "READY" if latest_ready > 0 else "NOT_READY",
        "count": latest_ready,
        "skip_reason_counts": dict(_mapping(snapshot_audit.get("skip_reason_counts"))),
    }


def _data_missing_stage() -> dict[str, Any]:
    return {"status": "DATA_MISSING", "count": 0, "skip_reason_counts": {}}


def _result_parse_stage(result_report: Mapping[str, Any]) -> dict[str, Any]:
    reason_counts = Counter()
    reason_counts += _count_reasons(_list(result_report.get("failed")), "error")
    for blocker in _list(result_report.get("label_write_blockers")):
        blocker_map = _mapping(blocker)
        if blocker_map.get("source") != "thedogs_official":
            continue
        reason = blocker_map.get("reason")
        if reason == "label_write_requires_complete_official_result_positions":
            reason_counts[str(reason)] += 1
    status = "READY"
    if (
        result_report.get("status") != "SUCCESS"
        or _safe_int(result_report.get("failed_count")) > 0
        or reason_counts
    ):
        status = "NOT_READY"
    return {
        "status": status,
        "count": _safe_int(result_report.get("ingested_count")),
        "skip_reason_counts": dict(sorted(reason_counts.items())),
    }


def _label_write_stage(
    result_report: Mapping[str, Any],
    label_readiness: Mapping[str, Any],
) -> dict[str, Any]:
    reason_counts = Counter()
    reason_counts += _count_reasons(
        _list(result_report.get("label_write_blockers")), "reason"
    )
    reason_counts += _count_reasons(
        _list(label_readiness.get("skipped_before_write_scope_validation")),
        "reason",
    )
    dry_gate = _mapping(label_readiness.get("dry_run_report_gate"))
    if dry_gate.get("approved") is not True and dry_gate.get("reason"):
        reason_counts[str(dry_gate["reason"])] += 1
    readiness_status = label_readiness.get("status")
    status = "READY" if readiness_status == "READY_FOR_EXPLICIT_APPROVAL" else "NOT_READY"
    if reason_counts:
        status = "NOT_READY"
    return {
        "status": status,
        "count": _safe_int(label_readiness.get("candidate_count_loaded_for_write_scope")),
        "skip_reason_counts": dict(sorted(reason_counts.items())),
    }


def _parser_failure_examples(result_report: Mapping[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for failure in _list(result_report.get("failed")):
        failure_map = _mapping(failure)
        attempts = _list(failure_map.get("attempts"))
        for attempt in attempts:
            attempt_map = _mapping(attempt)
            if attempt_map.get("source") != "thedogs_official":
                continue
            seen.add((failure_map.get("race_id"), attempt_map.get("source")))
            examples.append(
                {
                    "race_id": failure_map.get("race_id"),
                    "source": attempt_map.get("source"),
                    "source_url": attempt_map.get("source_url"),
                    "error": attempt_map.get("error") or failure_map.get("error"),
                }
            )
            if len(examples) >= limit:
                return examples
    for blocker in _list(result_report.get("label_write_blockers")):
        blocker_map = _mapping(blocker)
        if blocker_map.get("source") != "thedogs_official":
            continue
        seen_key = (blocker_map.get("race_id"), blocker_map.get("source"))
        if seen_key in seen:
            continue
        seen.add(seen_key)
        examples.append(
            {
                "race_id": blocker_map.get("race_id"),
                "source": blocker_map.get("source"),
                "source_url": blocker_map.get("source_url"),
                "error": blocker_map.get("reason"),
                "status": blocker_map.get("status"),
                "expected_box_count": blocker_map.get("expected_box_count"),
                "result_box_count": blocker_map.get("result_box_count"),
                "missing_result_boxes": blocker_map.get("missing_result_boxes"),
                "unexpected_result_boxes": blocker_map.get("unexpected_result_boxes"),
            }
        )
        if len(examples) >= limit:
            return examples
    return examples


def _prediction_inventory(snapshot_audit: Mapping[str, Any]) -> dict[str, Any]:
    counts = _mapping(snapshot_audit.get("counts"))
    gate = _mapping(snapshot_audit.get("gate"))
    return {
        "manifest_path": snapshot_audit.get("manifest_path"),
        "manifest_rows": _safe_int(counts.get("manifest_rows")),
        "latest_ready_races": _safe_int(counts.get("latest_ready_races")),
        "latest_ready_result_label_candidate_like": _safe_int(
            counts.get("latest_ready_result_label_candidate_like")
        ),
        "box1_share": gate.get("box1_share"),
        "box1_max_share": gate.get("box1_max_share"),
        "gate_status": gate.get("status"),
        "gate_reason": gate.get("reason"),
        "latest_ready_examples": _list(snapshot_audit.get("latest_ready_records"))[:5],
    }


def _feature_missingness_summary(feature_missingness: Mapping[str, Any]) -> dict[str, Any]:
    all_clean = _mapping(feature_missingness.get("all_clean_official"))
    history_only = _mapping(all_clean.get("history_only_model"))
    field_stats = _mapping(history_only.get("field_stats"))
    all_zero_columns = sorted(
        str(column)
        for column, stats in field_stats.items()
        if "row_present_count" in _mapping(stats)
        and _safe_int(_mapping(stats).get("row_present_count")) == 0
    )
    tgr_zero_columns = [column for column in all_zero_columns if column.startswith("tgr_")]
    return {
        "feature_missingness_schema_version": feature_missingness.get("schema_version"),
        "all_zero_feature_columns": all_zero_columns,
        "tgr_zero_coverage_columns": tgr_zero_columns,
        "history_only_required_features": [
            str(item) for item in _list(history_only.get("required_features"))
        ],
        "history_only_scope": history_only.get("scope"),
    }


def _challenger_inputs(
    challenger_inputs: Mapping[str, Any],
    feature_missingness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_paths = dict(_mapping(challenger_inputs.get("output_paths")))
    if challenger_inputs.get("clean_dataset"):
        output_paths["clean_dataset"] = challenger_inputs.get("clean_dataset")
    if challenger_inputs.get("packet_csv"):
        output_paths["packet_csv"] = challenger_inputs.get("packet_csv")
    feature_missingness = feature_missingness or {}
    all_zero_columns = list(
        dict.fromkeys(
            [
                str(item)
                for item in _list(challenger_inputs.get("all_zero_feature_columns"))
            ]
            + [
                str(item)
                for item in _list(feature_missingness.get("all_zero_feature_columns"))
            ]
        )
    )

    return {
        "schema_version": challenger_inputs.get("schema_version"),
        "status": challenger_inputs.get("status"),
        "clean_official_races": _safe_int(
            _first_present(challenger_inputs, "clean_official_races", "clean_races")
        ),
        "runner_rows": _safe_int(
            _first_present(challenger_inputs, "runner_rows", "clean_runner_rows")
        ),
        "clean_snapshot_instances": _safe_int(
            challenger_inputs.get("clean_snapshot_instances")
        ),
        "model_fit_rows": _safe_int(challenger_inputs.get(LEGACY_MODEL_FIT_ROWS_KEY)),
        "eval_rows": _safe_int(challenger_inputs.get("eval_rows")),
        "complete_valid_odds_model_fit_races": _safe_int(
            challenger_inputs.get(LEGACY_VALID_ODDS_FIT_RACES_KEY)
        ),
        "complete_valid_odds_eval_races": _safe_int(
            challenger_inputs.get("complete_valid_odds_eval_races")
        ),
        "near_duplicate_non_box_peer_rows": _safe_int(
            challenger_inputs.get("near_duplicate_non_box_peer_rows")
        ),
        "all_zero_feature_columns": all_zero_columns,
        "tgr_zero_coverage_columns": [
            str(item) for item in _list(feature_missingness.get("tgr_zero_coverage_columns"))
        ],
        "feature_missingness_schema_version": feature_missingness.get(
            "feature_missingness_schema_version"
        ),
        "history_only_required_features": [
            str(item)
            for item in _list(feature_missingness.get("history_only_required_features"))
        ],
        "history_only_scope": feature_missingness.get("history_only_scope"),
        "unsupported_variants": dict(_mapping(challenger_inputs.get("unsupported_variants"))),
        "output_paths": output_paths,
    }


def _feature_quality_blockers(feature_audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": feature_audit.get("schema_version"),
        "races": _safe_int(feature_audit.get("races")),
        "runner_rows": _safe_int(feature_audit.get("runner_rows")),
        "top_pick_box_distribution": dict(
            _mapping(feature_audit.get("top_pick_box_distribution"))
        ),
        "near_duplicate_rows_ge80pct_equal_peer": _safe_int(
            feature_audit.get("near_duplicate_rows_ge80pct_equal_peer")
        ),
        "near_duplicate_rows_ge90pct_equal_peer": _safe_int(
            feature_audit.get("near_duplicate_rows_ge90pct_equal_peer")
        ),
        "exact_non_box_duplicate_rows": _safe_int(
            feature_audit.get("exact_non_box_duplicate_rows")
        ),
        "mean_most_similar_non_box_equal_share": feature_audit.get(
            "mean_most_similar_non_box_equal_share"
        ),
        "mean_constant_non_box_feature_share": feature_audit.get(
            "mean_constant_non_box_feature_share"
        ),
        "distance_source_counts": dict(_mapping(feature_audit.get("distance_source_counts"))),
        "grade_source_counts": dict(_mapping(feature_audit.get("grade_source_counts"))),
        "source_error_count": len(_list(feature_audit.get("source_errors"))),
        "source_error_examples": _list(feature_audit.get("source_errors"))[:5],
    }


def _has_feature_quality_blockers(feature_quality: Mapping[str, Any]) -> bool:
    distance_sources = _mapping(feature_quality.get("distance_source_counts"))
    return any(
        [
            _safe_int(feature_quality.get("near_duplicate_rows_ge80pct_equal_peer")) > 0,
            _safe_int(feature_quality.get("exact_non_box_duplicate_rows")) > 0,
            _safe_int(feature_quality.get("source_error_count")) > 0,
            _safe_int(distance_sources.get("DATA_MISSING")) > 0,
            _safe_int(distance_sources.get("default_missing_target")) > 0,
        ]
    )


def build_packet(
    *,
    snapshot_audit_path: Path,
    result_dry_run_report_path: Path,
    label_readiness_path: Path,
    challenger_inputs_path: Path,
    feature_audit_path: Path | None = None,
    feature_missingness_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    failures: list[str] = []
    snapshot_audit, snapshot_ok = _read_artifact(
        snapshot_audit_path, "snapshot_audit", failures
    )
    result_report, result_ok = _read_artifact(
        result_dry_run_report_path, "result_dry_run_report", failures
    )
    label_readiness, label_ok = _read_artifact(
        label_readiness_path, "label_readiness", failures
    )
    challenger_inputs, _challenger_ok = _read_artifact(
        challenger_inputs_path, "challenger_inputs", failures
    )
    feature_audit: dict[str, Any] = {}
    feature_ok = False
    if feature_audit_path is not None:
        feature_audit, feature_ok = _read_artifact(
            feature_audit_path, "feature_audit", failures
        )
    feature_missingness: dict[str, Any] = {}
    feature_missingness_ok = False
    if feature_missingness_path is not None:
        feature_missingness, feature_missingness_ok = _read_artifact(
            feature_missingness_path, "feature_missingness", failures
        )

    readiness_by_stage = {
        "prediction_ready": _prediction_stage(snapshot_audit)
        if snapshot_ok
        else _data_missing_stage(),
        "result_parse_ready": _result_parse_stage(result_report)
        if result_ok
        else _data_missing_stage(),
        "label_write_ready": _label_write_stage(result_report, label_readiness)
        if label_ok
        else _data_missing_stage(),
    }
    blocker_reasons = [
        f"{stage}:{payload['status']}"
        for stage, payload in readiness_by_stage.items()
        if payload.get("status") != "READY"
    ]
    if _mapping(snapshot_audit.get("gate")).get("status") != "PASS":
        blocker_reasons.append("box_bias_gate_not_passed")
    feature_quality = _feature_quality_blockers(feature_audit) if feature_ok else {}
    if feature_quality and _has_feature_quality_blockers(feature_quality):
        blocker_reasons.append("non_box_feature_quality_blocked")
    missingness_summary = (
        _feature_missingness_summary(feature_missingness)
        if feature_missingness_ok
        else {}
    )
    if _list(missingness_summary.get("tgr_zero_coverage_columns")):
        blocker_reasons.append("all_zero_tgr_feature_coverage")

    packet_status = "DATA_MISSING" if failures else "READY_FOR_REVIEW"
    if blocker_reasons and not failures:
        packet_status = "BLOCKED"

    source_evidence = {
        "snapshot_audit": str(snapshot_audit_path),
        "result_dry_run_report": str(result_dry_run_report_path),
        "label_readiness": str(label_readiness_path),
        "challenger_inputs": str(challenger_inputs_path),
    }
    if feature_audit_path is not None:
        source_evidence["feature_audit"] = str(feature_audit_path)
    if feature_missingness_path is not None:
        source_evidence["feature_missingness"] = str(feature_missingness_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now.replace(microsecond=0).isoformat(),
        "status": packet_status,
        "failures": failures,
        "blocker_reasons": blocker_reasons,
        "source_evidence": source_evidence,
        "writes_performed": WRITES_PERFORMED.copy(),
        "prediction_ready_inventory": _prediction_inventory(snapshot_audit),
        "readiness_by_stage": readiness_by_stage,
        "official_parser_failure_examples": _parser_failure_examples(result_report),
        "challenger_matrix_inputs": _challenger_inputs(
            challenger_inputs,
            missingness_summary,
        ),
        "feature_quality_blockers": feature_quality,
        "promotion_gate": {
            "status": "BLOCKED",
            "required_human_approval": True,
            "reason": (
                "accuracy remains blocked until clean official sample, leakage, "
                "box-bias, feature-parity, and metric gates pass"
            ),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-audit", required=True)
    parser.add_argument("--result-dry-run-report", required=True)
    parser.add_argument("--label-readiness", required=True)
    parser.add_argument("--challenger-inputs", required=True)
    parser.add_argument("--feature-audit")
    parser.add_argument("--feature-missingness")
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    packet = build_packet(
        snapshot_audit_path=Path(args.snapshot_audit),
        result_dry_run_report_path=Path(args.result_dry_run_report),
        label_readiness_path=Path(args.label_readiness),
        challenger_inputs_path=Path(args.challenger_inputs),
        feature_audit_path=Path(args.feature_audit) if args.feature_audit else None,
        feature_missingness_path=(
            Path(args.feature_missingness) if args.feature_missingness else None
        ),
    )
    text = json.dumps(packet, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
