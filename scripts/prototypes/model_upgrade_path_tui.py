#!/usr/bin/env python3
"""PROTOTYPE ONLY.

Question: once the greyhound input data is trustworthy, what future upgrade
path actually makes sense? This throwaway TUI starts from the repo's live
baseline and lets an operator push the upgrade state by hand to see when
tabular/ranking, sequence NN, and LLM text-sidecar lanes become justified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from model_upgrade_path_logic import evaluate_state, initial_state, reduce_state

ROOT = Path(__file__).resolve().parents[2]

BEST_METADATA = ROOT / "model_registry" / "best_metadata.json"
LIVE_EVIDENCE = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "clean_live_evidence_cycle_20260603"
    / "evidence_decision.json"
)
LIVE_REPORT = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "clean_live_evidence_cycle_20260603"
    / "report.md"
)
DEBIAS_REPORT = (
    ROOT
    / "artifacts"
    / "full_evidence_orchestration_20260525"
    / "bounded_calibrated_debiasing_study_20260603"
    / "report.md"
)

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_box_bias_pct(text: str) -> float | None:
    match = re.search(
        r"Box 1 favorites share too high:\s*([0-9.]+)%\s*>\s*50.00?%",
        text,
    )
    if not match:
        match = re.search(r"Box 1 favorites share too high:\s*([0-9.]+)%", text)
    if not match:
        return None
    return float(match.group(1)) / 100.0


def _extract_soft_box_share(text: str) -> float | None:
    match = re.search(
        r"blend_champion_no_box_history.*?box1_share `([0-9.]+)`",
        text,
        re.DOTALL,
    )
    if not match:
        match = re.search(r"box1_share `([0-9.]+)`", text)
    if not match:
        return None
    return float(match.group(1))


def load_repo_snapshot() -> dict:
    best = _read_json(BEST_METADATA)
    live_evidence = _read_json(LIVE_EVIDENCE)

    live_report_text = LIVE_REPORT.read_text(encoding="utf-8", errors="ignore") if LIVE_REPORT.exists() else ""
    debias_report_text = DEBIAS_REPORT.read_text(encoding="utf-8", errors="ignore") if DEBIAS_REPORT.exists() else ""

    counts = live_evidence.get("counts") or {}
    validation = live_evidence.get("validation") or {}
    return {
        "current_model_id": best.get("model_id", "unknown"),
        "model_type": best.get("model_type", "unknown"),
        "training_rows": int(best.get("training_samples") or 0),
        "test_rows": int(best.get("test_samples") or 0),
        "feature_count": int(best.get("features_count") or 0),
        "top1_rate": float(best.get("top1_rate") or 0.0),
        "current_box1_share": _extract_box_bias_pct(live_report_text) or 0.90,
        "best_soft_box1_share": _extract_soft_box_share(debias_report_text) or 0.1852,
        "fresh_clean_labeled_races": int(counts.get("official_labels_written") or 0),
        "result_dry_run_candidates": int(counts.get("result_ingest_dry_run_candidates") or 0),
        "live_evidence_result": live_evidence.get("result", "unknown"),
        "box_bias_gate": validation.get("box_bias_gate", "red"),
    }


def render_frame(snapshot: dict, evaluation: dict) -> str:
    state = evaluation["state"]
    lines = [
        f"{BOLD}Prototype: Future Model Upgrade Path{RESET}",
        "Assumption: logic prototype for the greyhound training roadmap, not a UI mock.",
        "",
        f"{BOLD}Live Repo Snapshot{RESET}",
        json.dumps(snapshot, indent=2, sort_keys=True),
        "",
        f"{BOLD}Scenario State{RESET}",
        json.dumps(state, indent=2, sort_keys=True),
        "",
        f"{BOLD}Recommendation{RESET}",
        f"{evaluation['recommendation']}",
        f"{DIM}Core lane: {evaluation['next_core']}{RESET}",
        "",
        f"{BOLD}Candidates{RESET}",
    ]

    for name, details in evaluation["candidate_status"].items():
        status = "READY" if details["ready"] else "BLOCKED"
        lines.append(f"{BOLD}{name}{RESET}: {status}")
        lines.append(f"{DIM}{details['why']}{RESET}")
        if details["blockers"]:
            for blocker in details["blockers"]:
                lines.append(f"  - {blocker}")
        else:
            lines.append("  - no blocker")

    lines.extend(
        [
            "",
            f"{BOLD}Shortcuts{RESET}",
            f"{BOLD}1{RESET} current  {BOLD}2{RESET} tabular-ready  {BOLD}3{RESET} sequence-ready  {BOLD}4{RESET} llm-sidecar-ready",
            f"{BOLD}l/k{RESET} +/-100 clean labeled races  {BOLD}u/j{RESET} +/-1000 training rows",
            f"{BOLD}s{RESET} signal quality  {BOLD}b{RESET} box-bias gate  {BOLD}o{RESET} odds provenance",
            f"{BOLD}h{RESET} history depth  {BOLD}t{RESET} text corpus  {BOLD}d{RESET} soft debias signal  {BOLD}q{RESET} quit",
        ]
    )
    return "\n".join(lines)


def clear_screen() -> None:
    print("\033[2J\033[H", end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="store_true", help="print one frame and exit")
    args = parser.parse_args()

    snapshot = load_repo_snapshot()
    seed_rows = int(snapshot.get("training_rows") or 1441)
    state = initial_state(
        training_rows=seed_rows,
        clean_labeled_races=int(snapshot.get("fresh_clean_labeled_races") or 0),
    )

    while True:
        evaluation = evaluate_state(state)
        if not args.snapshot:
            clear_screen()
        print(render_frame(snapshot, evaluation))
        if args.snapshot:
            return 0

        command = input("\ncommand> ").strip().lower()
        if command == "q":
            return 0
        state = reduce_state(state, command, seed_rows)


if __name__ == "__main__":
    raise SystemExit(main())
