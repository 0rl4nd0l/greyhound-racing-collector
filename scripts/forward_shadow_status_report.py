#!/usr/bin/env python3
"""Build a rolling report-only status for forward shadow reliability.

The report combines result-join metrics, sidecar metadata gates, live/training
coverage, and activation-gate decisions into one evidence packet. It does not
write predictions, labels, DB rows, registry entries, EV, betting output, or
production model pointers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = ROOT / "artifacts/full_evidence_orchestration_20260525"
OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/forward_shadow_status_"
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
EXPECTED_OFFICIAL_RACES = 214
EXPECTED_OFFICIAL_DOG_ROWS = 1493
DEFAULT_MIN_JOINED_RACES = 20

STATUS_DB_BLOCKED = "BLOCKED_DB_STATE"
STATUS_COLLECT_MORE = "CONTINUE_FORWARD_SHADOW_COLLECTION"
STATUS_READY_REVIEW = "READY_FOR_FORWARD_SHADOW_REVIEW_REPORT_ONLY"
STATUS_REVIEW_KEEP_QUARANTINED = "FORWARD_REVIEW_READY_KEEP_QUARANTINED"


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_hashes(paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS) -> dict[str, str | None]:
    return {relpath(path) or str(path): sha256_file(path) for path in paths}


def db_state(db_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "db_path": relpath(db_path),
        "status": "FAIL",
        "fail_reasons": [],
    }
    if not db_path.exists():
        report["fail_reasons"].append("db_missing")
        return report
    try:
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            official_races = connection.execute(
                "SELECT count(DISTINCT race_id) FROM race_metadata "
                "WHERE winner_source='thedogs_official'"
            ).fetchone()[0]
            official_dog_rows = connection.execute(
                "SELECT count(*) FROM dog_race_data WHERE data_source='thedogs_official'"
            ).fetchone()[0]
        finally:
            connection.close()
    except Exception as exc:  # pragma: no cover - defensive artifact reporting
        report["fail_reasons"].append(f"db_read_failed:{type(exc).__name__}")
        return report
    report.update(
        {
            "quick_check": quick_check,
            "official_races": int(official_races),
            "official_dog_rows": int(official_dog_rows),
        }
    )
    if quick_check != "ok":
        report["fail_reasons"].append("quick_check_not_ok")
    if official_races != EXPECTED_OFFICIAL_RACES:
        report["fail_reasons"].append("official_race_count_mismatch")
    if official_dog_rows != EXPECTED_OFFICIAL_DOG_ROWS:
        report["fail_reasons"].append("official_dog_row_count_mismatch")
    if not report["fail_reasons"]:
        report["status"] = "PASS"
    return report


def latest_artifact(root: Path, prefix: str, required_file: str) -> Path | None:
    candidates = [
        item
        for item in root.glob(f"{prefix}*")
        if item.is_dir() and (item / required_file).exists()
    ]
    return sorted(candidates)[-1] if candidates else None


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(metrics: Mapping[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "safe_joined_race_count": int(metrics.get("safe_joined_race_count") or 0),
        "pending_race_count": int(metrics.get("pending_race_count") or 0),
        "unsafe_match_count": int(metrics.get("unsafe_match_count") or 0),
        "top1": metrics.get("top1"),
        "top3": metrics.get("top3"),
        "mean_winner_rank": metrics.get("mean_winner_rank"),
        "brier": metrics.get("brier"),
        "logloss": metrics.get("logloss"),
        "probability_sum_max_error_joined_races": metrics.get(
            "probability_sum_max_error_joined_races"
        ),
        "winner_ranks": metrics.get("winner_ranks") or [],
    }


def activation_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    return {
        "final_status": report.get("final_status"),
        "activation_allowed_features": report.get("activation_allowed_features") or [],
        "kept_quarantined_features": report.get("kept_quarantined_features") or [],
    }


def sidecar_summary(report: Mapping[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    classification = report.get("classification") or {}
    return {
        "final_status": report.get("final_status"),
        "eligible_count": classification.get("eligible_count"),
        "malformed_count": classification.get("malformed_count"),
        "stale_count": classification.get("stale_count"),
        "prejump_sidecar_metadata_required": classification.get(
            "prejump_sidecar_metadata_required"
        ),
    }


def artifact_final_status(directory: Path | None) -> str | None:
    if directory is None:
        return None
    path = directory / "final_status.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def coverage_summary(
    report: Mapping[str, Any] | None,
    selected_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report = report or {}
    latest_metrics = (
        dict(selected_metrics)
        if selected_metrics is not None
        else report.get("latest_forward_metrics_summary") or {}
    )
    return {
        "final_status": report.get("final_status"),
        "blocked_reasons": report.get("blocked_reasons") or [],
        "latest_forward_metrics_summary": latest_metrics,
        "training_feature_coverage": report.get("training_feature_coverage") or {},
        "live_sidecar_feature_coverage": report.get("live_sidecar_feature_coverage") or {},
    }


def decide_status(
    *,
    db_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
    activation: Mapping[str, Any],
    min_joined_races: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if db_report.get("status") != "PASS":
        return STATUS_DB_BLOCKED, ["db_state_not_pass"]
    if int(metrics.get("safe_joined_race_count") or 0) < min_joined_races:
        reasons.append("safe_joined_race_count_below_review_min")
    if int(metrics.get("pending_race_count") or 0) > 0:
        reasons.append("pending_official_results_remain")
    if int(metrics.get("unsafe_match_count") or 0) > 0:
        reasons.append("unsafe_identity_matches_present")
    probability_error = metrics.get("probability_sum_max_error_joined_races")
    if probability_error is None or float(probability_error) > 1e-6:
        reasons.append("probability_sum_error_not_pass")
    kept_quarantined = activation.get("kept_quarantined_features") or []
    if kept_quarantined:
        reasons.append("features_remain_quarantined")
    if reasons:
        return STATUS_COLLECT_MORE, reasons
    if kept_quarantined:
        return STATUS_REVIEW_KEEP_QUARANTINED, reasons
    return STATUS_READY_REVIEW, []


def build_status_report(
    *,
    evidence_root: Path,
    db_path: Path,
    aggregate_result_dir: Path | None = None,
    result_join_dir: Path | None = None,
    sidecar_gate_dir: Path | None = None,
    live_feature_audit_dir: Path | None = None,
    activation_gate_dir: Path | None = None,
    coverage_gap_dir: Path | None = None,
    min_joined_races: int = DEFAULT_MIN_JOINED_RACES,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    aggregate_result_dir = aggregate_result_dir or latest_artifact(
        evidence_root, "forward_shadow_result_aggregate_", "aggregate_forward_metrics.json"
    )
    result_join_dir = result_join_dir or latest_artifact(
        evidence_root, "forward_shadow_result_join_", "shadow_forward_metrics.json"
    )
    sidecar_gate_dir = sidecar_gate_dir or latest_artifact(
        evidence_root, "prejump_sidecar_gate_audit_", "prejump_sidecar_gate_audit.json"
    )
    live_feature_audit_dir = live_feature_audit_dir or latest_artifact(
        evidence_root,
        "sidecar_target_metadata_live_feature_audit_",
        "sidecar_target_metadata_live_feature_audit.json",
    )
    activation_gate_dir = activation_gate_dir or latest_artifact(
        evidence_root, "shadow_feature_activation_gate_", "feature_activation_gate_report.json"
    )
    coverage_gap_dir = coverage_gap_dir or latest_artifact(
        evidence_root,
        "train_live_feature_coverage_gap_audit_",
        "train_live_feature_coverage_gap_audit.json",
    )

    db_report = db_state(db_path)
    if aggregate_result_dir is not None:
        metrics = metric_summary(load_json(aggregate_result_dir / "aggregate_forward_metrics.json"))
        result_metric_source = "aggregate_forward_metrics"
    else:
        metrics = metric_summary(
            load_json(result_join_dir / "shadow_forward_metrics.json") if result_join_dir else None
        )
        result_metric_source = "latest_single_result_join"
    activation = activation_summary(
        load_json(activation_gate_dir / "feature_activation_gate_report.json")
        if activation_gate_dir
        else None
    )
    sidecar_gate = sidecar_summary(
        load_json(sidecar_gate_dir / "prejump_sidecar_gate_audit.json")
        if sidecar_gate_dir
        else None
    )
    if not sidecar_gate.get("final_status"):
        sidecar_gate["final_status"] = artifact_final_status(sidecar_gate_dir)
    final_status, reasons = decide_status(
        db_report=db_report,
        metrics=metrics,
        activation=activation,
        min_joined_races=min_joined_races,
    )
    return {
        "schema_version": "forward_shadow_status_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "status_reasons": reasons,
        "min_joined_races_for_review": min_joined_races,
        "db_state": db_report,
        "forward_metrics": metrics,
        "activation_gate": activation,
        "prejump_sidecar_gate": sidecar_gate,
        "live_feature_audit": load_json(
            live_feature_audit_dir / "sidecar_target_metadata_live_feature_audit.json"
        )
        if live_feature_audit_dir
        else None,
        "coverage_gap": coverage_summary(
            load_json(coverage_gap_dir / "train_live_feature_coverage_gap_audit.json")
            if coverage_gap_dir
            else None,
            selected_metrics=metrics,
        ),
        "source_dirs": {
            "aggregate_result_dir": relpath(aggregate_result_dir),
            "result_join_dir": relpath(result_join_dir),
            "sidecar_gate_dir": relpath(sidecar_gate_dir),
            "live_feature_audit_dir": relpath(live_feature_audit_dir),
            "activation_gate_dir": relpath(activation_gate_dir),
            "coverage_gap_dir": relpath(coverage_gap_dir),
        },
        "result_metric_source": result_metric_source,
        "no_write_guarantees": {
            "production_promotion": False,
            "registry_mutation": False,
            "production_pointer_update": False,
            "production_prediction_write": False,
            "db_write": False,
            "label_write": False,
            "tgr_enabled": False,
            "betting_or_ev_output": False,
        },
    }


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_forward_shadow_status_artifact:{relative}")
    return logical.absolute()


def build_summary(report: Mapping[str, Any]) -> str:
    metrics = report.get("forward_metrics") or {}
    activation = report.get("activation_gate") or {}
    lines = [
        "# Forward Shadow Status",
        "",
        f"- Final status: `{report.get('final_status')}`",
        f"- Reasons: `{report.get('status_reasons')}`",
        f"- Safe joined races: `{metrics.get('safe_joined_race_count')}`",
        f"- Pending races: `{metrics.get('pending_race_count')}`",
        f"- Unsafe matches: `{metrics.get('unsafe_match_count')}`",
        f"- Top1: `{metrics.get('top1')}`",
        f"- Top3: `{metrics.get('top3')}`",
        f"- Mean winner rank: `{metrics.get('mean_winner_rank')}`",
        f"- Brier: `{metrics.get('brier')}`",
        f"- LogLoss: `{metrics.get('logloss')}`",
        f"- Quarantined features: `{activation.get('kept_quarantined_features')}`",
        "",
        "## Decision",
        "- Continue collecting forward shadow results.",
        "- Keep `quarantine_feature` for same-distance/same-grade timing fields.",
        "- Do not promote, enable TGR, mutate registry, write DB labels, or write betting/EV outputs.",
    ]
    return "\n".join(lines) + "\n"


def run_status_report(
    *,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    output_dir: Path | None = None,
    db_path: Path = ROOT / "greyhound_racing_data.db",
    min_joined_races: int = DEFAULT_MIN_JOINED_RACES,
) -> dict[str, Any]:
    generated_at = datetime.now().astimezone()
    output_dir = output_dir or evidence_root / f"forward_shadow_status_{generated_at.strftime('%Y%m%dT%H%M%S%z')}"
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    protected_before = protected_hashes()
    report = build_status_report(
        evidence_root=evidence_root,
        db_path=db_path,
        min_joined_races=min_joined_races,
        generated_at=generated_at,
    )
    protected_after = protected_hashes()
    report["protected_hashes_before"] = protected_before
    report["protected_hashes_after"] = protected_after
    report["protected_paths_unchanged"] = protected_before == protected_after
    if not report["protected_paths_unchanged"]:
        report["final_status"] = "BLOCKED_PROTECTED_PATH_MUTATION"
        report.setdefault("status_reasons", []).append("protected_paths_changed")
    write_json(output_dir / "forward_shadow_status_report.json", report)
    write_text(output_dir / "SUMMARY.md", build_summary(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    return {
        "output_dir": relpath(output_dir),
        "final_status": report["final_status"],
        "status_reasons": report["status_reasons"],
        "safe_joined_race_count": report["forward_metrics"]["safe_joined_race_count"],
        "pending_race_count": report["forward_metrics"]["pending_race_count"],
        "protected_paths_unchanged": report["protected_paths_unchanged"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--db", type=Path, default=ROOT / "greyhound_racing_data.db")
    parser.add_argument("--min-joined-races", type=int, default=DEFAULT_MIN_JOINED_RACES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_status_report(
        evidence_root=args.evidence_root,
        output_dir=args.output_dir,
        db_path=args.db,
        min_joined_races=args.min_joined_races,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
