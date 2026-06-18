#!/usr/bin/env python3
"""Build a report-only promotion-distance packet for greyhound accuracy work.

The packet summarizes how far the current evidence is from a promotion-ready
state. It reads existing rolling comparison, gated challenger, and high-accuracy
gate reports. It writes artifacts only and never trains, promotes, mutates a
registry, writes DB rows, emits EV/betting output, rewrites snapshots/manifests,
or enables TGR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import build_promotion_gate_contract_audit_packet as gate_contract_audit  # noqa: E402

OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "promotion_distance_report_"
)
REPORT_FILE = "promotion_distance_report.json"
SUMMARY_FILE = "SUMMARY.md"
MIN_ROLLING_RACES_FOR_REVIEW = 100
MIN_RESIDUAL_TRIGGERED_RACES_FOR_DIRECTIONAL_READ = 10
TARGET_TOP1_MARGIN_VS_MARKET = 0.02
PROMOTION_GATE_CONTRACT_POLICY = "dual_baseline_market_rank_primary_safety"
EXPECTED_PROMOTION_SELECTED_STAGE = "odds_augmented_model_research"
EXPECTED_PULL_REQUEST_BOUNDARY = {
    "promotion_pr_allowed": True,
    "direct_local_switch_allowed": False,
    "local_registry_mutation_allowed": False,
    "production_pointer_update_allowed": False,
    "requires_human_pr_review": True,
}

NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "ev_action": False,
    "betting_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "tgr_enabled": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_promotion_distance_report:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return payload


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "promotion_distance_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def finite_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def string_list(value: Any) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return []


def int_count_mapping(value: Any) -> dict[str, int]:
    return {
        str(reason): finite_int(count)
        for reason, count in sorted(mapping(value).items())
    }


def metric_gap_to_target(delta: float | None, target: float) -> float | None:
    if delta is None:
        return None
    return max(0.0, target - delta)


def high_accuracy_gate_contract_blockers(high_gate: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if high_gate.get("status") != "READY_FOR_PR_DRAFT":
        blockers.append(f"high_accuracy_gate_not_ready:{high_gate.get('status')}")
    blockers.extend(string_list(high_gate.get("blockers")))
    selected_stage = high_gate.get("selected_stage")
    if selected_stage != EXPECTED_PROMOTION_SELECTED_STAGE:
        blockers.append(f"high_accuracy_selected_stage_not_supported:{selected_stage}")
    boundary = mapping(high_gate.get("pull_request_boundary"))
    for key, expected in EXPECTED_PULL_REQUEST_BOUNDARY.items():
        if boundary.get(key) is not expected:
            blockers.append(
                f"high_accuracy_pull_request_boundary_invalid:{key}="
                f"{boundary.get(key)}"
            )
    return list(dict.fromkeys(blockers))


def gate_contract_candidate_summary(
    *,
    rolling: Mapping[str, Any],
    high_gate: Mapping[str, Any],
    target_top1_margin_vs_market: float,
) -> dict[str, Any]:
    audit = gate_contract_audit.build_report(
        rolling_report=rolling,
        thresholds=gate_contract_audit.GateAuditThresholds(
            min_market_top1_delta=0.0,
        ),
    )
    policy = next(
        (
            row
            for row in audit.get("policy_summaries") or []
            if mapping(row).get("policy_key") == PROMOTION_GATE_CONTRACT_POLICY
        ),
        {},
    )
    policy = mapping(policy)
    selected_candidate = high_gate.get("selected_candidate")
    audit_candidate = policy.get("selected_candidate")
    row = next(
        (
            item
            for item in audit.get("candidate_gate_matrix") or []
            if mapping(item).get("candidate_key") == selected_candidate
        ),
        {},
    )
    row = mapping(row)
    blocker_text = str(row.get(f"{PROMOTION_GATE_CONTRACT_POLICY}_blockers") or "")
    candidate_blockers = [item for item in blocker_text.split(";") if item]
    blockers: list[str] = []
    if audit.get("final_status") != "REPORT_ONLY_GATE_CHANGE_CANDIDATE":
        blockers.append(
            f"gate_contract_audit_not_ready:{audit.get('final_status')}"
        )
    if policy.get("status") != "PASS":
        blockers.append(f"gate_contract_policy_not_pass:{PROMOTION_GATE_CONTRACT_POLICY}")
    if not selected_candidate:
        blockers.append("high_accuracy_selected_candidate_missing")
    if selected_candidate and not row:
        blockers.append(f"selected_candidate_gate_matrix_row_missing:{selected_candidate}")
    if selected_candidate and selected_candidate != audit_candidate:
        blockers.append(
            "high_accuracy_selected_candidate_not_audit_selected:"
            f"{selected_candidate}!={audit_candidate}"
        )
    blockers.extend(candidate_blockers)
    top1_delta = finite_float(row.get("market_top1_delta"))
    return {
        "schema_version": "promotion_distance_gate_contract_candidate_v1",
        "status": "PASS" if not blockers else "BLOCKED",
        "audit_final_status": audit.get("final_status"),
        "policy_key": PROMOTION_GATE_CONTRACT_POLICY,
        "policy_status": policy.get("status"),
        "selected_candidate": selected_candidate,
        "audit_selected_candidate": audit_candidate,
        "candidate_minus_market": {
            "top1": top1_delta,
            "top3": finite_float(row.get("market_top3_delta")),
            "mean_winner_rank": finite_float(
                row.get("market_mean_winner_rank_delta")
            ),
            "brier": finite_float(row.get("market_brier_delta")),
            "logloss": finite_float(row.get("market_logloss_delta")),
            "box1_top_pick_share": finite_float(
                row.get("market_box1_top_pick_share_delta")
            ),
            "calibration_distance": finite_float(
                row.get("market_calibration_distance_delta")
            ),
        },
        "target_top1_margin_vs_market": target_top1_margin_vs_market,
        "top1_margin_gap_to_target": metric_gap_to_target(
            top1_delta,
            target_top1_margin_vs_market,
        ),
        "top1_target_is_advisory_when_gate_contract_passes": True,
        "blockers": list(dict.fromkeys(blockers)),
    }


def official_result_coverage_summary(rolling: Mapping[str, Any]) -> dict[str, Any]:
    direct = mapping(rolling.get("official_result_coverage"))
    missing_race_ids = string_list(
        rolling.get("source_official_result_evidence_db_missing_race_ids")
    )
    races_with_rows = string_list(
        rolling.get("source_official_result_evidence_db_races_with_rows")
    )
    runner_paths = string_list(rolling.get("source_official_result_runner_paths"))
    requested_race_count = finite_int(
        rolling.get("source_official_result_evidence_db_requested_race_count")
    )
    requested_race_ids = string_list(
        rolling.get("source_official_result_evidence_db_requested_race_ids")
    )
    if direct:
        requested_race_ids = string_list(direct.get("requested_race_ids"))
        requested_race_count = finite_int(direct.get("requested_race_count"))
    return {
        "source": "rolling_model_comparison",
        "requested_race_count": requested_race_count,
        "requested_race_count_source": direct.get("requested_race_count_source")
        if direct
        else "rolling_model_comparison_source_count",
        "requested_race_ids": requested_race_ids,
        "legacy_requested_race_count_without_ids": finite_int(
            direct.get("legacy_requested_race_count_without_ids")
            if direct
            else rolling.get(
                "source_official_result_evidence_db_legacy_requested_race_count_without_ids"
            )
        ),
        "races_with_rows_count": len(races_with_rows),
        "missing_race_count": len(missing_race_ids),
        "missing_race_ids": missing_race_ids,
        "races_with_rows": races_with_rows,
        "runner_path_count": len(runner_paths),
        "runner_paths_source_field": (
            "rolling_sample.source_official_result_runner_paths"
        ),
        "missing_exclusion_count": int_count_mapping(
            rolling.get("source_exclusion_reason_counts")
        ).get("official_result_missing", 0),
    }


def build_report(
    *,
    rolling_report_path: Path,
    pre_race_gated_report_path: Path,
    high_accuracy_gate_path: Path,
    output_dir: Path,
    generated_at: datetime | None = None,
    min_rolling_races: int = MIN_ROLLING_RACES_FOR_REVIEW,
    min_residual_triggered_races: int = MIN_RESIDUAL_TRIGGERED_RACES_FOR_DIRECTIONAL_READ,
    target_top1_margin_vs_market: float = TARGET_TOP1_MARGIN_VS_MARKET,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)

    rolling = load_json(rolling_report_path)
    gated = load_json(pre_race_gated_report_path)
    high_gate = load_json(high_accuracy_gate_path)

    sample_races = finite_int(rolling.get("sample_race_count"))
    sample_rows = finite_int(rolling.get("sample_runner_rows"))
    best_non_market_delta = rolling.get("best_non_market_minus_market")
    if not isinstance(best_non_market_delta, Mapping):
        best_non_market_delta = {}
    residual = gated.get("predeclared_residual_candidate")
    if not isinstance(residual, Mapping):
        residual = {}
    residual_delta = residual.get("candidate_minus_market")
    if not isinstance(residual_delta, Mapping):
        residual_delta = {}

    non_market_top1_delta = finite_float(best_non_market_delta.get("top1"))
    residual_top1_delta = finite_float(residual_delta.get("top1"))
    residual_triggered = finite_int(residual.get("triggered_race_count"))
    residual_floor = finite_int(
        residual.get("minimum_triggered_races_for_directional_read")
    ) or min_residual_triggered_races
    gate_contract_candidate = gate_contract_candidate_summary(
        rolling=rolling,
        high_gate=high_gate,
        target_top1_margin_vs_market=target_top1_margin_vs_market,
    )
    gate_contract_passed = gate_contract_candidate.get("status") == "PASS"

    blockers: list[str] = []
    blockers.extend(high_accuracy_gate_contract_blockers(high_gate))
    if sample_races < min_rolling_races:
        blockers.append("rolling_sample_below_review_floor")
    if not gate_contract_passed:
        blockers.extend(
            f"gate_contract_candidate_blocker:{blocker}"
            for blocker in string_list(gate_contract_candidate.get("blockers"))
        )
    if (
        not gate_contract_passed
        and (non_market_top1_delta or 0.0) < target_top1_margin_vs_market
    ):
        blockers.append("best_non_market_top1_margin_below_target")
    if not gate_contract_passed and residual_triggered < residual_floor:
        blockers.append("predeclared_residual_trigger_count_below_directional_floor")
    if not gate_contract_passed and (residual_top1_delta or 0.0) <= 0.0:
        blockers.append("predeclared_residual_top1_not_above_market")
    blockers = list(dict.fromkeys(blockers))

    official_result_coverage = official_result_coverage_summary(rolling)
    report = {
        "schema_version": "promotion_distance_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": "PROMOTION_DISTANCE_BLOCKED" if blockers else "PROMOTION_DISTANCE_REVIEW_READY",
        "output_dir": relpath(output_dir),
        "source_reports": {
            "rolling_model_comparison": relpath(rolling_report_path),
            "pre_race_gated_challenger": relpath(pre_race_gated_report_path),
            "high_accuracy_promotion_gate": relpath(high_accuracy_gate_path),
        },
        "rolling_sample": {
            "sample_race_count": sample_races,
            "sample_runner_rows": sample_rows,
            "minimum_races_for_review": min_rolling_races,
            "races_needed_for_review_floor": max(0, min_rolling_races - sample_races),
            "source_rejected_live_odds_candidate_count": finite_int(
                rolling.get("source_rejected_live_odds_candidate_count")
            ),
            "source_rows_with_rejected_live_odds_candidates": finite_int(
                rolling.get("source_rows_with_rejected_live_odds_candidates")
            ),
            "source_rejected_live_odds_candidate_reason_counts": {
                str(reason): finite_int(count)
                for reason, count in sorted(
                    mapping(
                        rolling.get(
                            "source_rejected_live_odds_candidate_reason_counts"
                        )
                    ).items()
                )
            },
            "source_exclusion_reason_counts": int_count_mapping(
                rolling.get("source_exclusion_reason_counts")
            ),
            "source_odds_exclusion_reason_counts": int_count_mapping(
                rolling.get("source_odds_exclusion_reason_counts")
            ),
            "source_official_result_evidence_db_missing_race_ids": string_list(
                rolling.get("source_official_result_evidence_db_missing_race_ids")
            ),
            "source_official_result_evidence_db_requested_race_ids": string_list(
                rolling.get("source_official_result_evidence_db_requested_race_ids")
            ),
            "source_official_result_evidence_db_requested_race_count": finite_int(
                rolling.get("source_official_result_evidence_db_requested_race_count")
            ),
            "source_official_result_evidence_db_legacy_requested_race_count_without_ids": finite_int(
                rolling.get(
                    "source_official_result_evidence_db_legacy_requested_race_count_without_ids"
                )
            ),
            "source_official_result_evidence_db_races_with_rows": string_list(
                rolling.get("source_official_result_evidence_db_races_with_rows")
            ),
            "source_official_result_runner_paths": string_list(
                rolling.get("source_official_result_runner_paths")
            ),
        },
        "official_result_coverage": official_result_coverage,
        "market_benchmark": {
            "best_candidate_key": rolling.get("best_candidate_key"),
            "best_non_market_candidate_key": rolling.get("best_non_market_candidate_key"),
            "best_non_market_minus_market": dict(best_non_market_delta),
            "target_top1_margin_vs_market": target_top1_margin_vs_market,
            "best_non_market_top1_margin_gap": metric_gap_to_target(
                non_market_top1_delta,
                target_top1_margin_vs_market,
            ),
            "selected_gate_contract_candidate_key": gate_contract_candidate.get(
                "selected_candidate"
            ),
            "selected_gate_contract_candidate_minus_market": dict(
                mapping(gate_contract_candidate.get("candidate_minus_market"))
            ),
            "selected_gate_contract_top1_margin_gap": (
                gate_contract_candidate.get("top1_margin_gap_to_target")
            ),
        },
        "gate_contract_candidate": gate_contract_candidate,
        "predeclared_residual_candidate": {
            "candidate_key": residual.get("candidate_key"),
            "status": residual.get("status"),
            "triggered_race_count": residual_triggered,
            "minimum_triggered_races_for_directional_read": residual_floor,
            "triggered_races_needed_for_directional_read": max(
                0,
                residual_floor - residual_triggered,
            ),
            "directional_read_ready": bool(residual.get("directional_read_ready", False)),
            "candidate_minus_market": dict(residual_delta),
            "top1_margin_gap_to_above_market": metric_gap_to_target(
                residual_top1_delta,
                0.000000001,
            ),
            "blockers": list(residual.get("blockers") or []),
            "promotion_blocking": not gate_contract_passed,
        },
        "promotion_gate": {
            "status": high_gate.get("status"),
            "blockers": list(high_gate.get("blockers") or []),
        },
        "blockers": blockers,
        "promotion_ready": not blockers,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_FILE, summary_markdown(report))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def summary_markdown(report: Mapping[str, Any]) -> str:
    rolling = report.get("rolling_sample") or {}
    official_result = report.get("official_result_coverage") or {}
    market = report.get("market_benchmark") or {}
    gate_contract = report.get("gate_contract_candidate") or {}
    residual = report.get("predeclared_residual_candidate") or {}
    gate = report.get("promotion_gate") or {}
    return "\n".join(
        [
            "# Promotion Distance Report",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            f"- Rolling sample races: `{rolling.get('sample_race_count')}` / `{rolling.get('minimum_races_for_review')}`",
            f"- Rolling sample rows: `{rolling.get('sample_runner_rows')}`",
            f"- Rolling source rejected live odds candidates: `{rolling.get('source_rejected_live_odds_candidate_count')}`",
            f"- Rolling source rows with rejected live odds candidates: `{rolling.get('source_rows_with_rejected_live_odds_candidates')}`",
            f"- Rolling source rejected live odds candidate reasons: `{rolling.get('source_rejected_live_odds_candidate_reason_counts')}`",
            f"- Rolling source exclusion reasons: `{rolling.get('source_exclusion_reason_counts')}`",
            f"- Rolling source odds exclusion reasons: `{rolling.get('source_odds_exclusion_reason_counts')}`",
            f"- Rolling source official-result missing race IDs: `{rolling.get('source_official_result_evidence_db_missing_race_ids')}`",
            f"- Official-result coverage requested races: `{official_result.get('requested_race_count')}`",
            f"- Official-result coverage requested race count source: `{official_result.get('requested_race_count_source')}`",
            f"- Official-result legacy requested race count without IDs: `{official_result.get('legacy_requested_race_count_without_ids')}`",
            f"- Official-result coverage races with rows: `{official_result.get('races_with_rows_count')}`",
            f"- Official-result coverage missing races: `{official_result.get('missing_race_count')}`",
            f"- Official-result missing exclusion count: `{official_result.get('missing_exclusion_count')}`",
            f"- Official-result runner path count: `{official_result.get('runner_path_count')}`",
            f"- Official-result runner paths source field: `{official_result.get('runner_paths_source_field')}`",
            f"- Best candidate: `{market.get('best_candidate_key')}`",
            f"- Best non-market candidate: `{market.get('best_non_market_candidate_key')}`",
            f"- Best non-market top1 gap to target margin: `{market.get('best_non_market_top1_margin_gap')}`",
            f"- Gate-contract candidate: `{gate_contract.get('selected_candidate')}`",
            f"- Gate-contract candidate status: `{gate_contract.get('status')}`",
            f"- Gate-contract policy: `{gate_contract.get('policy_key')}`",
            f"- Gate-contract candidate minus market: `{gate_contract.get('candidate_minus_market')}`",
            f"- Gate-contract top1 gap to target margin: `{gate_contract.get('top1_margin_gap_to_target')}`",
            f"- Gate-contract blockers: `{gate_contract.get('blockers')}`",
            f"- Residual candidate: `{residual.get('candidate_key')}`",
            f"- Residual triggered races: `{residual.get('triggered_race_count')}` / `{residual.get('minimum_triggered_races_for_directional_read')}`",
            f"- Residual races needed: `{residual.get('triggered_races_needed_for_directional_read')}`",
            f"- Residual directional read ready: `{residual.get('directional_read_ready')}`",
            f"- Residual promotion blocking: `{residual.get('promotion_blocking')}`",
            f"- Promotion gate: `{gate.get('status')}`",
            f"- Blockers: `{report.get('blockers')}`",
            "",
            "No training, promotion, registry mutation, DB write, label write, odds write, EV/betting action, snapshot rewrite, manifest rewrite, or TGR enablement was performed.",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rolling-report", type=Path, required=True)
    parser.add_argument("--pre-race-gated-report", type=Path, required=True)
    parser.add_argument("--high-accuracy-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"promotion_distance_report_{now_id()}"
    )
    report = build_report(
        rolling_report_path=args.rolling_report,
        pre_race_gated_report_path=args.pre_race_gated_report,
        high_accuracy_gate_path=args.high_accuracy_gate,
        output_dir=output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
