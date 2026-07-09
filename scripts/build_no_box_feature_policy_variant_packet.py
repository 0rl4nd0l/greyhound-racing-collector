#!/usr/bin/env python3
"""Build a report-only no-box feature-policy variant packet.

The helper derives a new feature-row packet from an existing no-box dog-form
feature join by removing explicitly approved diagnostic feature columns. It
does not read or write the DB, fetch official sources, write labels, train or
persist models, mutate snapshots or manifests, update registries, promote
models, enable TGR, or make EV/betting decisions.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
DEFAULT_PACKET_ROOT = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "expanded_historical_shadow_evaluation_20260609T_accuracy_improvement_packet_v23"
)
DEFAULT_SOURCE_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_actual_win_dog_form_feature_join_combined_37_history_db_masked_no_outcome_proxy_correct_db"
)
DEFAULT_TRIAGE_REPORT = (
    DEFAULT_PACKET_ROOT
    / "no_box_same_distance_miss_triage_combined_37_history_db_masked_no_outcome_proxy_correct_db"
    / "same_distance_miss_triage_report.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PACKET_ROOT
    / "no_box_actual_win_dog_form_feature_join_combined_37_history_db_masked_no_outcome_proxy_quarantine_same_distance_rates_correct_db"
)
SCHEMA_VERSION = "no_box_feature_policy_variant_packet_v1"
ROWS_SCHEMA_VERSION = "no_box_actual_win_dog_form_feature_rows_v1"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "metadata_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "model_persistence": False,
    "registry_mutation": False,
    "promotion": False,
    "tgr_enablement": False,
    "betting_decision": False,
    "ev_action": False,
}
FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL = [
    "write_official_safe_labels",
    "mutate_db",
    "regenerate_canonical_dataset",
    "train_or_promote_model",
    "update_registry",
    "enable_tgr",
    "betting_or_ev_action",
]
CSV_BASE_FIELDS = [
    "race_id",
    "legacy_race_id",
    "identity_key",
    "race_date",
    "venue",
    "race_number",
    "dog_name_key",
    "dog_name",
    "actual_win",
    "candidate_kind",
    "field_scope",
    "field_complete_for_ranking",
    "feature_join_status",
]
FORBIDDEN_ROW_FIELDS = {
    "box_number",
    "official_box_number",
    "db_box_number",
    "finish_position",
    "official_finish_position",
    "db_finish_position",
    "db_result_position",
    "result_position",
    "placing",
    "scraped_finish_position",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{line_number}")
            rows.append(row)
    return rows


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root = root or ROOT
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{logical}") from exc
    return resolved, relative


def _repo_relative_text(path: Path, root: Path | None = None) -> str:
    return _repo_output_path(path, root)[1]


def _assert_output_dir_safe(output_dir: Path) -> Path:
    resolved, relative = _repo_output_path(output_dir)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _feature_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("feature_") and key != "feature_join_status"
        }
    )


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _drop_features_from_triage(triage_report: Mapping[str, Any] | None) -> list[str]:
    if not triage_report:
        return []
    summary = triage_report.get("summary") or {}
    return [str(item) for item in summary.get("quarantine_candidate_features") or []]


def _validate_source(
    packet: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    drop_features: Sequence[str],
    expected_races: int | None,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    if packet.get("report_only") is not True:
        failures.append("source_packet_report_only_not_true")
    if packet.get("status") != "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY":
        failures.append(f"source_packet_status_not_ready:{packet.get('status')}")
    writes = packet.get("writes_performed") or {}
    for key in ("db_write", "label_write", "model_training", "registry_mutation", "promotion"):
        if writes.get(key) is not False:
            failures.append(f"source_packet_{key}_not_false")
    for feature in drop_features:
        if not feature.startswith("feature_"):
            failures.append(f"drop_feature_must_start_with_feature_:{feature}")
    source_features = set(_feature_columns(rows))
    missing_drop_features = [feature for feature in drop_features if feature not in source_features]
    if missing_drop_features:
        warnings.append("drop_features_not_present:" + ",".join(sorted(missing_drop_features)))
    race_ids = {str(row.get("race_id") or "") for row in rows}
    if expected_races is not None and len(race_ids) != expected_races:
        failures.append(f"expected_races_mismatch:{expected_races}:{len(race_ids)}")
    for index, row in enumerate(rows, start=1):
        forbidden = sorted(FORBIDDEN_ROW_FIELDS & set(row))
        if forbidden:
            failures.append(f"row_{index}_forbidden_fields_present:{','.join(forbidden)}")
        if row.get("box_features_allowed") is not False:
            failures.append(f"row_{index}_box_features_allowed_not_false")
        if row.get("finish_order_labels_allowed") is not False:
            failures.append(f"row_{index}_finish_order_labels_allowed_not_false")
        if row.get("top3_labels_allowed") is not False:
            failures.append(f"row_{index}_top3_labels_allowed_not_false")
        if row.get("label_write_approved") is not False:
            failures.append(f"row_{index}_label_write_approved_not_false")
        if _safe_int(row.get("actual_win")) not in (0, 1):
            failures.append(f"row_{index}_actual_win_not_binary")
    positive_counts: Counter[str] = Counter()
    for row in rows:
        positive_counts[str(row.get("race_id") or "")] += _safe_int(row.get("actual_win")) or 0
    for race_id, count in sorted(positive_counts.items()):
        if count != 1:
            failures.append(f"race_{race_id}_actual_win_positive_count:{count}")
    return {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "warnings": warnings,
        "source_row_count": len(rows),
        "source_race_count": len(race_ids),
        "source_feature_count": len(source_features),
    }


def build_feature_policy_variant(
    *,
    source_packet: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    source_packet_path: str | None = None,
    source_rows_path: str | None = None,
    triage_report: Mapping[str, Any] | None = None,
    triage_report_path: str | None = None,
    drop_features: Sequence[str] | None = None,
    variant_key: str = "quarantine_same_distance_rates",
    expected_races: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    resolved_drop_features = list(dict.fromkeys([*(drop_features or []), *_drop_features_from_triage(triage_report)]))
    validation = _validate_source(
        source_packet,
        source_rows,
        resolved_drop_features,
        expected_races,
    )
    kept_rows = []
    for row in source_rows:
        kept = dict(row)
        for feature in resolved_drop_features:
            kept.pop(feature, None)
        kept["schema_version"] = ROWS_SCHEMA_VERSION
        kept_rows.append(kept)

    source_features = _feature_columns(source_rows)
    kept_features = _feature_columns(kept_rows)
    dropped_present = sorted(set(source_features) - set(kept_features))
    packet = copy.deepcopy(dict(source_packet))
    packet["schema_version"] = source_packet.get("schema_version")
    packet["status"] = (
        "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY"
        if validation["status"] == "PASS"
        else "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_FAILED"
    )
    packet["report_only"] = True
    packet["writes_performed"] = dict(WRITES_PERFORMED)
    packet["source_packet"] = source_packet_path
    packet["source_rows_jsonl"] = source_rows_path
    packet["source_triage_report"] = triage_report_path
    feature_policy = dict(packet.get("feature_policy") or {})
    feature_policy["policy_variant_key"] = variant_key
    feature_policy["derived_schema_version"] = SCHEMA_VERSION
    feature_policy["dropped_feature_columns"] = sorted(resolved_drop_features)
    feature_policy["dropped_feature_reason"] = (
        "report_only_quarantine_same_distance_rate_features_until_real_winner_coverage"
    )
    packet["feature_policy"] = feature_policy
    summary = dict(packet.get("summary") or {})
    summary.update(
        {
            "policy_variant_key": variant_key,
            "policy_variant_schema_version": SCHEMA_VERSION,
            "source_feature_column_count": len(source_features),
            "feature_column_count": len(kept_features),
            "dropped_feature_columns_requested": sorted(resolved_drop_features),
            "dropped_feature_columns_present": dropped_present,
            "dropped_feature_column_count": len(dropped_present),
            "features_with_non_null_values": sum(
                1 for feature in kept_features if any(row.get(feature) not in (None, "") for row in kept_rows)
            ),
            "variant_validation": validation,
            "joined_rows": len(kept_rows),
            "no_box_features_selected": True,
            "no_race_number_feature_selected": True,
            "no_calendar_features_selected": True,
        }
    )
    packet["summary"] = summary
    packet["forbidden_without_explicit_approval"] = list(FORBIDDEN_WITHOUT_EXPLICIT_APPROVAL)
    packet["recommended_next_action"] = (
        "run_no_box_pairwise_rolling_windows_on_policy_variant"
        if validation["status"] == "PASS"
        else "resolve_feature_policy_variant_contract_failures_before_eval"
    )
    return packet, kept_rows


def write_outputs(output_dir: Path, packet: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir = _assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "no_box_actual_win_feature_join_packet.json", packet)
    _write_jsonl(output_dir / "no_box_actual_win_feature_rows.jsonl", rows)
    feature_fields = _feature_columns(rows)
    with (output_dir / "no_box_actual_win_feature_rows.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=[*CSV_BASE_FIELDS, *feature_fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in [*CSV_BASE_FIELDS, *feature_fields]})
    summary = packet.get("summary") or {}
    lines = [
        "# No-Box Feature Policy Variant",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, metadata writes, official fetches, snapshot or manifest mutations, model training or persistence, registry updates, promotions, TGR enablement, betting decisions, or EV actions occurred.",
        "",
        f"- Variant: `{summary.get('policy_variant_key')}`",
        f"- Rows: `{summary.get('joined_rows')}`",
        f"- Source feature columns: `{summary.get('source_feature_column_count')}`",
        f"- Kept feature columns: `{summary.get('feature_column_count')}`",
        f"- Dropped feature columns: `{summary.get('dropped_feature_columns_present')}`",
        "",
        "## Next",
        "",
        str(packet.get("recommended_next_action")),
        "",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-packet",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "no_box_actual_win_feature_join_packet.json",
    )
    parser.add_argument(
        "--source-rows",
        type=Path,
        default=DEFAULT_SOURCE_DIR / "no_box_actual_win_feature_rows.jsonl",
    )
    parser.add_argument("--triage-report", type=Path, default=DEFAULT_TRIAGE_REPORT)
    parser.add_argument("--drop-feature", action="append", default=[])
    parser.add_argument("--variant-key", default="quarantine_same_distance_rates")
    parser.add_argument("--expected-races", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    triage = _load_json(args.triage_report) if args.triage_report and args.triage_report.exists() else None
    packet, rows = build_feature_policy_variant(
        source_packet=_load_json(args.source_packet),
        source_rows=_load_jsonl(args.source_rows),
        source_packet_path=str(args.source_packet),
        source_rows_path=str(args.source_rows),
        triage_report=triage,
        triage_report_path=str(args.triage_report) if triage else None,
        drop_features=args.drop_feature,
        variant_key=args.variant_key,
        expected_races=args.expected_races,
    )
    write_outputs(args.output_dir, packet, rows)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "output_dir": str(args.output_dir),
                "summary": {
                    key: packet["summary"].get(key)
                    for key in (
                        "source_feature_column_count",
                        "feature_column_count",
                        "dropped_feature_columns_present",
                        "variant_validation",
                    )
                },
            },
            sort_keys=True,
        )
    )
    return 0 if packet["status"] == "REPORT_ONLY_DOG_FORM_FEATURE_JOIN_READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
