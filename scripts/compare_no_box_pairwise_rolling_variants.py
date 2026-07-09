#!/usr/bin/env python3
"""Compare report-only no-box pairwise rolling variants.

This helper reads rolling-window reports produced by
evaluate_no_box_pairwise_rolling_windows.py and ranks only report-local,
non-persisted variants. It is a decision packet, not a training or promotion
step: no DB rows, labels, snapshots, manifests, datasets, model artifacts,
registries, TGR settings, betting decisions, or EV artifacts are changed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_no_box_pairwise_ranking_smoke import (  # noqa: E402
    _assert_output_dir_safe,
    _load_json,
    _safe_float,
)
from scripts.evaluate_no_box_pairwise_rolling_windows import (  # noqa: E402
    SCHEMA_VERSION as ROLLING_SCHEMA_VERSION,
)


SCHEMA_VERSION = "no_box_pairwise_rolling_variant_comparison_v1"
STATUS_OK = "REPORT_ONLY_NO_BOX_PAIRWISE_ROLLING_VARIANT_COMPARISON"
STATUS_FAILURES = "REPORT_ONLY_NO_BOX_PAIRWISE_ROLLING_VARIANT_COMPARISON_WITH_FAILURES"

WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "dataset_regeneration": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}

CSV_FIELDS = [
    "diagnostic_rank",
    "variant_key",
    "status",
    "source_status",
    "history_db_fill_policy",
    "label_proxy_audit_status",
    "race_count",
    "evaluated_race_count",
    "row_count",
    "usable_feature_count",
    "complete_field_races",
    "sample_size_status",
    "window_count",
    "top1_accuracy",
    "top3_hit_rate",
    "mean_winner_rank",
    "top1_delta_vs_baseline",
    "top3_delta_vs_baseline",
    "diagnostic_rank_eligible",
    "promotion_ready",
    "blocking_reasons",
    "report_path",
]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _pipe(values: Sequence[Any]) -> str:
    return "|".join(str(value) for value in values if value not in (None, ""))


def _writes_true(report: Mapping[str, Any]) -> list[str]:
    writes = _mapping(report.get("writes_performed"))
    return sorted(str(key) for key, value in writes.items() if value is not False)


def _is_rejected(status: str) -> bool:
    return "REJECTED" in status or "FAILED" in status


def _variant_row(
    *,
    variant_key: str,
    report_path: str,
    report: Mapping[str, Any],
    failures: list[str],
) -> dict[str, Any]:
    status = str(report.get("status") or "")
    source = _mapping(report.get("source_packet"))
    validation = _mapping(report.get("validation"))
    metrics = _mapping(report.get("aggregate_metrics"))
    window_summary = _mapping(report.get("window_metric_summary"))
    complete_gate = _mapping(report.get("race_grouped_complete_field_gate"))
    rolling_policy = _mapping(report.get("rolling_window_policy"))
    write_flags_true = _writes_true(report)
    blocking_reasons = []

    if report.get("schema_version") != ROLLING_SCHEMA_VERSION:
        blocking_reasons.append("schema_mismatch")
        failures.append(f"{variant_key}:schema_mismatch")
    if report.get("report_only") is not True:
        blocking_reasons.append("report_only_not_true")
        failures.append(f"{variant_key}:report_only_not_true")
    if write_flags_true:
        blocking_reasons.append("write_flags_true:" + ",".join(write_flags_true))
        failures.append(f"{variant_key}:write_flags_true:{','.join(write_flags_true)}")
    if validation.get("status") not in (None, "PASS"):
        blocking_reasons.append(f"validation_status:{validation.get('status')}")
    if rolling_policy.get("reserved_races_predicted") is not False:
        blocking_reasons.append("reserved_races_predicted_not_false")
        failures.append(f"{variant_key}:reserved_races_predicted_not_false")
    if _is_rejected(status):
        blocking_reasons.append(f"status:{status}")
    if report.get("sample_size_status") == "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES":
        blocking_reasons.append("underpowered_below_50_actual_win_races")
    if complete_gate.get("status") == "SKIPPED_INSUFFICIENT_COMPLETE_FIELD_RACES":
        blocking_reasons.append("insufficient_complete_field_races_for_full_ranking_gate")

    evaluated = status == "REPORT_ONLY_PAIRWISE_ROLLING_WINDOWS_EVALUATED"
    diagnostic_rank_eligible = (
        evaluated
        and validation.get("status") == "PASS"
        and not write_flags_true
        and rolling_policy.get("reserved_races_predicted") is False
        and metrics.get("race_count") is not None
    )
    return {
        "variant_key": variant_key,
        "report_path": report_path,
        "status": status,
        "source_status": source.get("status"),
        "history_db_fill_policy": source.get("history_db_fill_policy"),
        "label_proxy_audit_status": source.get("label_proxy_audit_status"),
        "race_count": validation.get("race_count"),
        "evaluated_race_count": metrics.get("race_count"),
        "row_count": validation.get("row_count"),
        "usable_feature_count": validation.get("usable_feature_count"),
        "complete_field_races": validation.get("complete_field_races"),
        "sample_size_status": report.get("sample_size_status"),
        "window_count": window_summary.get("window_count"),
        "top1_accuracy": _safe_float(metrics.get("top1_accuracy")),
        "top3_hit_rate": _safe_float(metrics.get("top3_hit_rate")),
        "mean_winner_rank": _safe_float(metrics.get("mean_winner_rank")),
        "expected_random_top1": _safe_float(metrics.get("expected_random_top1")),
        "expected_random_top3": _safe_float(metrics.get("expected_random_top3")),
        "top1_range": _safe_float(window_summary.get("top1_range")),
        "top3_range": _safe_float(window_summary.get("top3_range")),
        "reserved_final_races": rolling_policy.get("reserved_final_races"),
        "reserved_races_predicted": rolling_policy.get("reserved_races_predicted"),
        "diagnostic_rank_eligible": diagnostic_rank_eligible,
        "promotion_ready": False,
        "blocking_reasons": blocking_reasons,
        "diagnostic_rank": None,
        "top1_delta_vs_baseline": None,
        "top3_delta_vs_baseline": None,
    }


def _rank_rows(rows: list[dict[str, Any]]) -> None:
    eligible = [row for row in rows if row.get("diagnostic_rank_eligible") is True]
    eligible.sort(
        key=lambda row: (
            -(_safe_float(row.get("top1_accuracy")) or -1.0),
            -(_safe_float(row.get("top3_hit_rate")) or -1.0),
            _safe_float(row.get("mean_winner_rank")) or 999999.0,
            str(row.get("variant_key") or ""),
        )
    )
    for index, row in enumerate(eligible, start=1):
        row["diagnostic_rank"] = index


def _baseline_row(rows: Sequence[Mapping[str, Any]], baseline_key: str | None) -> Mapping[str, Any]:
    if baseline_key:
        for row in rows:
            if row.get("variant_key") == baseline_key:
                return row
    for row in rows:
        key = str(row.get("variant_key") or "").lower()
        if "plain" in key and row.get("diagnostic_rank_eligible") is True:
            return row
    for row in rows:
        if row.get("diagnostic_rank_eligible") is True:
            return row
    return {}


def _add_baseline_deltas(rows: list[dict[str, Any]], baseline: Mapping[str, Any]) -> None:
    base_top1 = _safe_float(baseline.get("top1_accuracy"))
    base_top3 = _safe_float(baseline.get("top3_hit_rate"))
    for row in rows:
        top1 = _safe_float(row.get("top1_accuracy"))
        top3 = _safe_float(row.get("top3_hit_rate"))
        row["top1_delta_vs_baseline"] = (
            top1 - base_top1 if top1 is not None and base_top1 is not None else None
        )
        row["top3_delta_vs_baseline"] = (
            top3 - base_top3 if top3 is not None and base_top3 is not None else None
        )


def _is_history_feature_variant(row: Mapping[str, Any]) -> bool:
    key = str(row.get("variant_key") or "").lower()
    policy = str(row.get("history_db_fill_policy") or "").lower()
    return "history" in key or bool(policy)


def build_variant_comparison_packet(
    *,
    variant_reports: Mapping[str, tuple[str, Mapping[str, Any]]],
    baseline_key: str | None = None,
) -> dict[str, Any]:
    failures: list[str] = []
    rows = [
        _variant_row(
            variant_key=variant_key,
            report_path=report_path,
            report=report,
            failures=failures,
        )
        for variant_key, (report_path, report) in sorted(variant_reports.items())
    ]
    _rank_rows(rows)
    baseline = _baseline_row(rows, baseline_key)
    if baseline:
        _add_baseline_deltas(rows, baseline)

    best = next((row for row in rows if row.get("diagnostic_rank") == 1), {})
    evaluated_rows = [row for row in rows if row.get("diagnostic_rank_eligible") is True]
    rejected_rows = [row for row in rows if _is_rejected(str(row.get("status") or ""))]
    top1_delta = _safe_float(best.get("top1_delta_vs_baseline"))
    top3_delta = _safe_float(best.get("top3_delta_vs_baseline"))
    history_gain_status = "DATA_MISSING"
    if best and baseline:
        if not _is_history_feature_variant(best):
            history_gain_status = "BEST_DIAGNOSTIC_VARIANT_IS_NOT_HISTORY_FEATURE_SET"
        elif (top1_delta or 0.0) > 0 or (top3_delta or 0.0) > 0:
            history_gain_status = "PROMISING_UNDERPOWERED_DIAGNOSTIC"
        else:
            history_gain_status = "NO_DIAGNOSTIC_GAIN_OVER_BASELINE"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS_FAILURES if failures else STATUS_OK,
        "failures": failures,
        "report_only": True,
        "safe_to_write_now": False,
        "label_write_approved": False,
        "model_promotion_allowed": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "summary": {
            "variant_count": len(rows),
            "diagnostic_evaluated_variant_count": len(evaluated_rows),
            "rejected_variant_count": len(rejected_rows),
            "baseline_variant_key": baseline.get("variant_key"),
            "best_diagnostic_variant_key": best.get("variant_key"),
            "best_top1_accuracy": best.get("top1_accuracy"),
            "best_top3_hit_rate": best.get("top3_hit_rate"),
            "best_mean_winner_rank": best.get("mean_winner_rank"),
            "best_top1_delta_vs_baseline": top1_delta,
            "best_top3_delta_vs_baseline": top3_delta,
            "history_feature_gain_status": history_gain_status,
            "all_variants_underpowered_below_50_actual_win_races": all(
                row.get("sample_size_status") == "UNDERPOWERED_BELOW_50_ACTUAL_WIN_RACES"
                for row in evaluated_rows
            )
            if evaluated_rows
            else None,
            "max_complete_field_races": max(
                (int(row.get("complete_field_races") or 0) for row in rows),
                default=0,
            ),
            "reserved_races_predicted_count": sum(
                1 for row in rows if row.get("reserved_races_predicted") is not False
            ),
            "recommended_next_action": (
                "treat_best_history_feature_variant_as_promising_but_underpowered;"
                "expand_official_safe_labels_and_repeat_rolling_windows_before_any_promotion"
                if history_gain_status == "PROMISING_UNDERPOWERED_DIAGNOSTIC"
                else "expand_official_safe_labels_and_repeat_rolling_windows"
            ),
        },
        "objective_progress": {
            "pre_race_history_features": history_gain_status,
            "race_grouped_ranking_models": (
                "REPORT_ONLY_PAIRWISE_ROLLING_COMPARISON_AVAILABLE"
                if evaluated_rows
                else "DATA_MISSING"
            ),
            "ablation_tests": "COVERED_BY_SOURCE_REPORTS_NOT_REEVALUATED_HERE",
            "stratified_error_analysis": "COVERED_BY_SOURCE_REPORTS_NOT_REEVALUATED_HERE",
            "official_safe_label_expansion": "STILL_REQUIRES_EXPLICIT_LABEL_OR_DB_WRITE_APPROVAL",
            "rolling_temporal_validation": (
                "REPORT_ONLY_WINDOWS_AVAILABLE_SECOND_HOLDOUT_RESERVED"
                if evaluated_rows
                else "DATA_MISSING"
            ),
        },
        "variant_rows": rows,
        "forbidden_without_explicit_approval": [
            "db_write",
            "label_write",
            "metadata_write",
            "official_fetch",
            "snapshot_or_manifest_mutation",
            "dataset_regeneration",
            "model_training_or_promotion",
            "registry_update",
            "enable_tgr",
            "betting_or_ev_action",
        ],
    }


def _csv_rows(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in _list(packet.get("variant_rows")):
        row_map = dict(_mapping(row))
        row_map["blocking_reasons"] = _pipe(_list(row_map.get("blocking_reasons")))
        rows.append(row_map)
    return rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "no_box_pairwise_rolling_variant_comparison.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "no_box_pairwise_rolling_variant_comparison.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_csv_rows(packet))
    _write_summary(output_dir / "SUMMARY.md", packet)


def _write_summary(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# No-Box Pairwise Rolling Variant Comparison",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB rows, labels, snapshots, manifests, datasets, model artifacts, "
        "registries, TGR settings, betting decisions, EV actions, or official "
        "fetches were changed or performed.",
        "",
        "## Summary",
        "",
        f"- Variants: `{summary.get('variant_count')}`",
        f"- Evaluated diagnostic variants: `{summary.get('diagnostic_evaluated_variant_count')}`",
        f"- Rejected variants: `{summary.get('rejected_variant_count')}`",
        f"- Baseline: `{summary.get('baseline_variant_key')}`",
        f"- Best diagnostic variant: `{summary.get('best_diagnostic_variant_key')}`",
        f"- Best Top1: `{summary.get('best_top1_accuracy')}`",
        f"- Best Top3: `{summary.get('best_top3_hit_rate')}`",
        f"- Top1 delta vs baseline: `{summary.get('best_top1_delta_vs_baseline')}`",
        f"- Top3 delta vs baseline: `{summary.get('best_top3_delta_vs_baseline')}`",
        f"- History feature gain status: `{summary.get('history_feature_gain_status')}`",
        f"- Max complete-field races: `{summary.get('max_complete_field_races')}`",
        f"- Safe to write now: `{packet.get('safe_to_write_now')}`",
        "",
        "## Next",
        "",
        str(summary.get("recommended_next_action")),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_variant_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("variant_must_be_key_equals_path")
    key, raw_path = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("variant_key_missing")
    return key, Path(raw_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant spec in the form key=/path/to/no_box_pairwise_rolling_windows_report.json",
    )
    parser.add_argument("--baseline-key", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    reports: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for item in args.variant:
        key, path = _parse_variant_arg(item)
        resolved = path.expanduser().resolve()
        reports[key] = (str(resolved), _load_json(resolved))
    packet = build_variant_comparison_packet(
        variant_reports=reports,
        baseline_key=args.baseline_key,
    )
    write_outputs(Path(args.output_dir), packet)
    print(
        json.dumps(
            {"status": packet["status"], "summary": packet["summary"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if packet["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
