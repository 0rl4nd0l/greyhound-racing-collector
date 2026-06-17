#!/usr/bin/env python3
"""Build a no-write preflight for official result reserve substitutions.

This report consumes backlog unified evidence status rows that were already
quarantined by official-result ingestion. It never accepts, joins, inserts, or
promotes those rows; it only turns the reserve/scratch diagnostic into an
explicit blocker packet for future policy review.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)


SCHEMA_VERSION = "official_result_reserve_substitution_preflight_v1"
MANUAL_REVIEW_SCHEMA_VERSION = (
    "official_result_reserve_substitution_manual_review_packet_v1"
)
IMPACT_PREVIEW_SCHEMA_VERSION = (
    "official_result_reserve_substitution_policy_impact_preview_v1"
)
FINAL_EMPTY = "RESERVE_SUBSTITUTION_PREFLIGHT_EMPTY"
FINAL_BLOCKED = "RESERVE_SUBSTITUTION_PREFLIGHT_BLOCKED"
FINAL_READY = "RESERVE_SUBSTITUTION_PREFLIGHT_READY_FOR_POLICY_REVIEW"
MANUAL_REVIEW_EMPTY = "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_EMPTY"
MANUAL_REVIEW_BLOCKED = "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_BLOCKED"
MANUAL_REVIEW_READY = "RESERVE_SUBSTITUTION_MANUAL_REVIEW_PACKET_READY"
IMPACT_PREVIEW_EMPTY = "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_EMPTY"
IMPACT_PREVIEW_READY = "RESERVE_SUBSTITUTION_POLICY_IMPACT_PREVIEW_READY"
DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
PROTECTED_OUTPUT_DIRS = (
    ROOT / "artifacts/prediction_snapshots",
    ROOT / "model_registry",
    ROOT / "docs/model_registry",
    ROOT / "ml_models_v4",
    ROOT / "advanced_models",
)
NO_WRITE_GUARANTEES = {
    "db_write": False,
    "label_write": False,
    "canonical_result_label_write": False,
    "official_result_acceptance": False,
    "quarantine_bypass": False,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "production_promotion": False,
    "betting_action": False,
    "ev_action": False,
    "tgr_enabled": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, sort_keys=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def int_list(value: Any) -> list[int]:
    parsed: list[int] = []
    for item in sequence(value):
        try:
            parsed.append(int(item))
        except (TypeError, ValueError):
            continue
    return parsed


def unique_int_list(value: Any) -> list[int]:
    return sorted(set(int_list(value)))


def text_list(value: Any) -> list[str]:
    return [str(item) for item in sequence(value) if str(item or "").strip()]


def int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def int_count_mapping(value: Any) -> dict[str, int]:
    return {
        str(key): int_value(count)
        for key, count in sorted(mapping(value).items())
    }


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> None:
    resolved = output_dir.expanduser().resolve()
    if resolved == ROOT.resolve():
        raise ValueError("protected_output_dir:.")
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError:
        return
    for protected in PROTECTED_OUTPUT_DIRS:
        try:
            protected_relative = protected.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue
        if relative == protected_relative or protected_relative in relative.parents:
            raise ValueError(f"protected_output_dir:{protected_relative.as_posix()}")


def candidate_rows(status: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows_by_race: dict[str, Mapping[str, Any]] = {}
    for source in (
        status.get("top_gap_races"),
        status.get("top_official_result_missing_races"),
        mapping(status.get("gap_action_plan")).get("top_gap_races"),
        mapping(status.get("race_coverage")).get("top_official_result_missing_races"),
    ):
        for row in sequence(source):
            if not isinstance(row, Mapping):
                continue
            race_id = str(row.get("race_id") or "").strip()
            if race_id and row.get("official_result_quarantine_reserve_substitution_diagnostic"):
                rows_by_race.setdefault(race_id, row)
    return list(rows_by_race.values())


def evaluate_policy_source(
    *,
    source: Mapping[str, Any],
    participant_boxes: Sequence[int],
    result_boxes_outside: Sequence[int],
    candidate_reserve_boxes: Sequence[int],
    scratched_participant_boxes: Sequence[int],
) -> dict[str, Any]:
    result_boxes = int_list(source.get("result_boxes"))
    terminal_status_boxes = unique_int_list(source.get("terminal_status_boxes"))
    participant_set = set(participant_boxes)
    reserve_set = set(candidate_reserve_boxes)
    scratched_set = set(scratched_participant_boxes)
    names_by_box = mapping(source.get("dog_names_by_box"))
    blockers: list[str] = []

    if source.get("source") != "thedogs_official":
        blockers.append("official_source_identity_not_thedogs")
    if source.get("status") != "resulted":
        blockers.append("official_source_status_not_resulted")
    if not source.get("source_url"):
        blockers.append("official_source_url_missing")
    if not result_boxes:
        blockers.append("official_result_order_missing")
    if len(result_boxes) != len(set(result_boxes)):
        blockers.append("official_result_order_duplicate_boxes")
    if any(str(box) not in names_by_box for box in result_boxes):
        blockers.append("official_result_order_dog_names_missing")
    if set(result_boxes) - participant_set != set(result_boxes_outside):
        blockers.append("official_result_outside_boxes_do_not_match_source_order")
    if set(result_boxes_outside) != reserve_set:
        blockers.append("outside_result_boxes_do_not_match_candidate_reserves")
    if not reserve_set:
        blockers.append("candidate_reserve_boxes_missing")
    if not scratched_set:
        blockers.append("scratched_participant_boxes_missing")
    if len(reserve_set) != len(scratched_set):
        blockers.append("reserve_scratch_count_mismatch")
    if not scratched_set.issubset(participant_set):
        blockers.append("scratched_boxes_not_subset_of_participants")
    if not scratched_set.issubset(set(terminal_status_boxes)):
        blockers.append("scratched_participant_boxes_missing_terminal_status")

    expected_result_participants = participant_set - scratched_set
    result_participants = set(result_boxes) & participant_set
    if result_participants != expected_result_participants:
        blockers.append("result_participants_do_not_match_unscratched_participants")

    return {
        "source": source.get("source"),
        "source_url": source.get("source_url"),
        "source_status": source.get("status"),
        "policy_status": "PASS" if not blockers else "BLOCKED",
        "policy_blockers": sorted(set(blockers)),
        "result_boxes": result_boxes,
        "terminal_status_boxes": terminal_status_boxes,
        "result_boxes_outside_participants": sorted(set(result_boxes) - participant_set),
        "expected_result_participant_boxes": sorted(expected_result_participants),
        "actual_result_participant_boxes": sorted(result_participants),
        "candidate_reserve_boxes": sorted(reserve_set),
        "scratched_participant_boxes": sorted(scratched_set),
        "dog_names_by_result_box": {
            str(box): str(names_by_box.get(str(box)) or names_by_box.get(box) or "")
            for box in result_boxes
        },
    }


def preflight_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic = mapping(row.get("official_result_quarantine_reserve_substitution_diagnostic"))
    attempted_sources = [
        dict(source)
        for source in sequence(row.get("official_result_quarantine_attempted_source_box_sets"))
        if isinstance(source, Mapping)
    ]
    participant_boxes = int_list(row.get("official_result_quarantine_participant_boxes"))
    result_boxes_outside = int_list(diagnostic.get("result_boxes_outside_participants"))
    terminal_status_boxes = int_list(diagnostic.get("terminal_status_boxes"))
    candidate_reserve_boxes = int_list(diagnostic.get("candidate_reserve_boxes"))
    blockers: list[str] = []

    if diagnostic.get("classification") != "possible_reserve_substitution_manual_review_required":
        blockers.append("reserve_substitution_not_classified_possible")
    if diagnostic.get("acceptance_status") != "not_accepted_report_only":
        blockers.append("unexpected_acceptance_status")
    if result_boxes_outside:
        blockers.append("official_result_boxes_outside_frozen_participants_expected_for_reserve_policy")
    if terminal_status_boxes:
        blockers.append("official_terminal_statuses_present_expected_for_reserve_policy")
    if candidate_reserve_boxes and not set(candidate_reserve_boxes).issubset(
        set(result_boxes_outside)
    ):
        blockers.append("candidate_reserve_boxes_not_subset_of_outside_boxes")
    if not participant_boxes:
        blockers.append("participant_boxes_missing")
    if not attempted_sources:
        blockers.append("official_attempted_source_box_sets_missing")
    if not any(source.get("source_url") for source in attempted_sources):
        blockers.append("official_source_url_missing")
    missing_order_names = False
    for source in attempted_sources:
        names_by_box = mapping(source.get("dog_names_by_box"))
        source_result_boxes = int_list(source.get("result_boxes"))
        if any(str(box) not in names_by_box for box in source_result_boxes):
            missing_order_names = True
    if missing_order_names:
        blockers.append("official_result_order_dog_names_missing")

    source_policy_evaluations = [
        evaluate_policy_source(
            source=source,
            participant_boxes=participant_boxes,
            result_boxes_outside=result_boxes_outside,
            candidate_reserve_boxes=candidate_reserve_boxes,
            scratched_participant_boxes=int_list(
                diagnostic.get("scratched_participant_boxes")
            ),
        )
        for source in attempted_sources
    ]
    passing_sources = [
        source
        for source in source_policy_evaluations
        if source.get("policy_status") == "PASS"
    ]
    if not passing_sources:
        blockers.append("no_official_source_passed_reserve_substitution_policy")

    readiness_blockers = [
        blocker
        for blocker in blockers
        if blocker
        not in {
            "official_result_boxes_outside_frozen_participants_expected_for_reserve_policy",
            "official_terminal_statuses_present_expected_for_reserve_policy",
        }
    ]
    policy_review_status = (
        "READY_FOR_MANUAL_POLICY_REVIEW"
        if not readiness_blockers and passing_sources
        else "BLOCKED"
    )

    return {
        "race_id": row.get("race_id"),
        "race_date": row.get("race_date"),
        "venue": row.get("venue"),
        "recommended_action": "manual_review_reserve_substitution_policy",
        "preflight_status": policy_review_status,
        "policy_review_status": policy_review_status,
        "acceptance_status": "not_accepted_report_only",
        "acceptance_effect": "none_report_only",
        "blockers": sorted(set(blockers)),
        "readiness_blockers": sorted(set(readiness_blockers)),
        "dataset_join_blockers": [
            "official_result_remains_quarantined",
            "manual_policy_review_required_before_join",
        ],
        "source_url": (
            attempted_sources[0].get("source_url") if attempted_sources else None
        ),
        "participant_source": row.get("official_result_quarantine_participant_source"),
        "participant_boxes": participant_boxes,
        "participant_count": row.get("official_result_quarantine_participant_count"),
        "result_boxes_outside_participants": result_boxes_outside,
        "result_boxes_inside_participants": int_list(
            diagnostic.get("result_boxes_inside_participants")
        ),
        "candidate_reserve_boxes": candidate_reserve_boxes,
        "scratched_participant_boxes": int_list(
            diagnostic.get("scratched_participant_boxes")
        ),
        "terminal_status_boxes": terminal_status_boxes,
        "terminal_status_boxes_outside_participants": int_list(
            diagnostic.get("terminal_status_boxes_outside_participants")
        ),
        "attempted_source_box_sets": attempted_sources,
        "quarantine_errors": text_list(row.get("official_result_quarantine_errors")),
        "quarantine_reason": row.get("official_result_quarantine_reason"),
        "diagnostic": dict(diagnostic),
        "source_policy_evaluations": source_policy_evaluations,
    }


def build_preflight_packet(
    *,
    backlog_unified_evidence_status: Mapping[str, Any],
    backlog_unified_evidence_status_path: Path | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = generated_at or datetime.now().astimezone()
    candidates = [preflight_candidate(row) for row in candidate_rows(backlog_unified_evidence_status)]
    blocked = [row for row in candidates if row.get("preflight_status") == "BLOCKED"]
    ready = [
        row
        for row in candidates
        if row.get("preflight_status") == "READY_FOR_MANUAL_POLICY_REVIEW"
    ]
    blocker_counts: dict[str, int] = {}
    for row in blocked:
        for blocker in text_list(row.get("blockers")):
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
    if not candidates:
        final_status = FINAL_EMPTY
    elif blocked:
        final_status = FINAL_BLOCKED
    else:
        final_status = FINAL_READY
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "final_status": final_status,
        "source_artifacts": {
            "backlog_unified_evidence_status": relpath(
                backlog_unified_evidence_status_path
            ),
        },
        "candidate_count": len(candidates),
        "blocked_candidate_count": len(blocked),
        "ready_for_policy_review_count": len(ready),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "candidates": candidates,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def selected_policy_source(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
    evaluations = [
        source
        for source in sequence(candidate.get("source_policy_evaluations"))
        if isinstance(source, Mapping)
    ]
    for source in evaluations:
        if source.get("policy_status") == "PASS":
            return source
    return evaluations[0] if evaluations else {}


def substitution_mapping_hypothesis(
    candidate: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    reserve_boxes = unique_int_list(candidate.get("candidate_reserve_boxes"))
    scratched_boxes = unique_int_list(candidate.get("scratched_participant_boxes"))
    dog_names = mapping(source.get("dog_names_by_result_box"))
    blockers: list[str] = []

    if not reserve_boxes:
        blockers.append("candidate_reserve_boxes_missing")
    if not scratched_boxes:
        blockers.append("scratched_participant_boxes_missing")
    if len(reserve_boxes) != len(scratched_boxes):
        blockers.append("reserve_scratch_count_mismatch")

    pairs: list[dict[str, Any]] = []
    if not blockers:
        for scratched_box, reserve_box in zip(scratched_boxes, reserve_boxes):
            pairs.append(
                {
                    "scratched_participant_box": scratched_box,
                    "reserve_box": reserve_box,
                    "reserve_dog_name": str(
                        dog_names.get(str(reserve_box))
                        or dog_names.get(reserve_box)
                        or ""
                    ),
                    "mapping_basis": (
                        "sorted_count_aligned_reserve_and_scratched_boxes_"
                        "from_quarantine_diagnostic"
                    ),
                    "mapping_acceptance_status": "not_accepted",
                }
            )

    return {
        "mapping_status": (
            "report_only_policy_hypothesis" if not blockers else "blocked_report_only"
        ),
        "mapping_acceptance_status": "not_accepted",
        "mapping_blockers": sorted(set(blockers)),
        "pairs": pairs,
    }


def manual_review_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    source = selected_policy_source(candidate)
    mapping_hypothesis = substitution_mapping_hypothesis(candidate, source)
    packet_blockers = list(text_list(candidate.get("readiness_blockers")))
    packet_blockers.extend(text_list(mapping_hypothesis.get("mapping_blockers")))
    review_ready = (
        candidate.get("preflight_status") == "READY_FOR_MANUAL_POLICY_REVIEW"
        and not packet_blockers
    )

    return {
        "race_id": candidate.get("race_id"),
        "race_date": candidate.get("race_date"),
        "venue": candidate.get("venue"),
        "review_status": (
            "READY_FOR_MANUAL_REVIEW"
            if review_ready
            else "BLOCKED_FOR_MANUAL_REVIEW_PACKET"
        ),
        "approval_required": True,
        "automatic_acceptance_allowed": False,
        "dataset_join_allowed": False,
        "official_result_acceptance_allowed": False,
        "db_write": False,
        "acceptance_status": "not_accepted_report_only",
        "acceptance_effect": "none_report_only",
        "packet_blockers": sorted(set(packet_blockers)),
        "dataset_join_blockers": text_list(candidate.get("dataset_join_blockers")),
        "manual_review_questions": [
            "Does the official source prove each reserve dog replaced the scratched participant box?",
            "Does the official source result order align with the freeze-time participant set after removing scratched boxes?",
            "Should this reserve-substitution policy be approved for a future explicit join rule?",
        ],
        "approval_checklist": [
            "official_source_identity_verified",
            "official_source_url_verified",
            "result_order_and_dog_names_verified",
            "scratch_and_reserve_mapping_verified",
            "runner_set_after_scratches_verified",
            "explicit_policy_approval_recorded_before_any_join",
        ],
        "source": {
            "source": source.get("source"),
            "source_url": source.get("source_url") or candidate.get("source_url"),
            "source_status": source.get("source_status"),
            "policy_status": source.get("policy_status"),
            "policy_blockers": text_list(source.get("policy_blockers")),
            "sportsbook_source_identity": "not_applicable_official_result_source",
        },
        "race_facts": {
            "participant_source": candidate.get("participant_source"),
            "participant_boxes": int_list(candidate.get("participant_boxes")),
            "participant_count": candidate.get("participant_count"),
            "result_boxes": int_list(source.get("result_boxes")),
            "result_boxes_inside_participants": int_list(
                candidate.get("result_boxes_inside_participants")
            ),
            "result_boxes_outside_participants": int_list(
                candidate.get("result_boxes_outside_participants")
            ),
            "terminal_status_boxes": int_list(candidate.get("terminal_status_boxes")),
            "terminal_status_boxes_outside_participants": int_list(
                candidate.get("terminal_status_boxes_outside_participants")
            ),
            "candidate_reserve_boxes": int_list(
                candidate.get("candidate_reserve_boxes")
            ),
            "scratched_participant_boxes": int_list(
                candidate.get("scratched_participant_boxes")
            ),
            "dog_names_by_result_box": dict(
                mapping(source.get("dog_names_by_result_box"))
            ),
        },
        "mapping_hypothesis": mapping_hypothesis,
        "quarantine": {
            "quarantine_reason": candidate.get("quarantine_reason"),
            "quarantine_errors": text_list(candidate.get("quarantine_errors")),
            "official_result_remains_quarantined": True,
        },
    }


def build_manual_review_packet(
    *,
    preflight_packet: Mapping[str, Any],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = generated_at or datetime.now().astimezone()
    candidates = [
        manual_review_candidate(candidate)
        for candidate in sequence(preflight_packet.get("candidates"))
        if isinstance(candidate, Mapping)
    ]
    ready = [
        candidate
        for candidate in candidates
        if candidate.get("review_status") == "READY_FOR_MANUAL_REVIEW"
    ]
    blocked = [
        candidate
        for candidate in candidates
        if candidate.get("review_status") != "READY_FOR_MANUAL_REVIEW"
    ]
    if not candidates:
        final_status = MANUAL_REVIEW_EMPTY
    elif blocked:
        final_status = MANUAL_REVIEW_BLOCKED
    else:
        final_status = MANUAL_REVIEW_READY
    return {
        "schema_version": MANUAL_REVIEW_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "final_status": final_status,
        "preflight_final_status": preflight_packet.get("final_status"),
        "approval_required": True,
        "automatic_acceptance_allowed": False,
        "dataset_join_allowed": False,
        "official_result_acceptance_allowed": False,
        "db_write": False,
        "candidate_count": len(candidates),
        "ready_candidate_count": len(ready),
        "blocked_candidate_count": len(blocked),
        "ready_race_ids": [str(candidate.get("race_id")) for candidate in ready],
        "blocked_race_ids": [str(candidate.get("race_id")) for candidate in blocked],
        "source_artifacts": dict(mapping(preflight_packet.get("source_artifacts"))),
        "candidates": candidates,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def backlog_gap_action_plan(status: Mapping[str, Any]) -> Mapping[str, Any]:
    race_coverage = mapping(status.get("race_coverage"))
    return mapping(status.get("gap_action_plan")) or mapping(
        race_coverage.get("gap_action_plan")
    )


def policy_impact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    race_facts = mapping(candidate.get("race_facts"))
    mapping_hypothesis = mapping(candidate.get("mapping_hypothesis"))
    pairs = [
        dict(pair)
        for pair in sequence(mapping_hypothesis.get("pairs"))
        if isinstance(pair, Mapping)
    ]
    result_boxes = int_list(race_facts.get("result_boxes"))
    return {
        "race_id": candidate.get("race_id"),
        "race_date": candidate.get("race_date"),
        "venue": candidate.get("venue"),
        "review_status": candidate.get("review_status"),
        "approval_required": True,
        "automatic_acceptance_allowed": False,
        "dataset_join_allowed": False,
        "official_result_acceptance_allowed": False,
        "db_write": False,
        "blocked_reason": "manual_policy_review_required_before_join",
        "official_result_remains_quarantined": True,
        "result_box_count": len(result_boxes),
        "mapping_pair_count": len(pairs),
        "mapping_pairs": pairs,
        "source_url": mapping(candidate.get("source")).get("source_url"),
        "dataset_join_blockers": text_list(candidate.get("dataset_join_blockers")),
    }


def build_policy_impact_preview(
    *,
    backlog_unified_evidence_status: Mapping[str, Any],
    manual_review_packet: Mapping[str, Any],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated = generated_at or datetime.now().astimezone()
    candidates = [
        policy_impact_candidate(candidate)
        for candidate in sequence(manual_review_packet.get("candidates"))
        if isinstance(candidate, Mapping)
    ]
    ready_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("review_status") == "READY_FOR_MANUAL_REVIEW"
    ]
    gap_plan = backlog_gap_action_plan(backlog_unified_evidence_status)
    top_gap_ids = {
        str(row.get("race_id"))
        for row in sequence(
            backlog_unified_evidence_status.get("top_gap_races")
            or gap_plan.get("top_gap_races")
        )
        if isinstance(row, Mapping) and row.get("race_id")
    }
    ready_race_ids = {
        str(candidate.get("race_id"))
        for candidate in ready_candidates
        if candidate.get("race_id")
    }
    matched_gap_ids = sorted(ready_race_ids & top_gap_ids)
    potential_result_rows = sum(
        int_value(candidate.get("result_box_count")) for candidate in ready_candidates
    )
    mapping_pair_count = sum(
        int_value(candidate.get("mapping_pair_count")) for candidate in ready_candidates
    )
    return {
        "schema_version": IMPACT_PREVIEW_SCHEMA_VERSION,
        "generated_at": generated.isoformat(),
        "final_status": (
            IMPACT_PREVIEW_READY if ready_candidates else IMPACT_PREVIEW_EMPTY
        ),
        "preview_scope": "report_only_manual_policy_blockage_quantification",
        "approval_required": True,
        "automatic_acceptance_allowed": False,
        "dataset_join_allowed": False,
        "official_result_acceptance_allowed": False,
        "db_write": False,
        "candidate_count": len(candidates),
        "ready_candidate_count": len(ready_candidates),
        "ready_race_ids": sorted(ready_race_ids),
        "mapping_pair_count": mapping_pair_count,
        "potential_official_result_runner_rows_blocked_by_policy": (
            potential_result_rows
        ),
        "matched_backlog_top_gap_race_count": len(matched_gap_ids),
        "matched_backlog_top_gap_race_ids": matched_gap_ids,
        "backlog_sample_blocking_gap_count": int_value(
            backlog_unified_evidence_status.get("sample_blocking_gap_count")
            if "sample_blocking_gap_count" in backlog_unified_evidence_status
            else gap_plan.get("sample_blocking_gap_count")
        ),
        "backlog_unified_evidence_eligible_rows": int_value(
            backlog_unified_evidence_status.get("unified_evidence_eligible_rows")
        ),
        "backlog_gap_action_counts": int_count_mapping(
            backlog_unified_evidence_status.get("gap_action_counts")
            or gap_plan.get("action_counts")
        ),
        "backlog_evidence_missing_reason_counts": int_count_mapping(
            backlog_unified_evidence_status.get("evidence_missing_reason_counts")
            or gap_plan.get("evidence_missing_reason_counts")
        ),
        "preview_effect_if_policy_approved_later": (
            "listed_races_could_be_reconsidered_for_explicit_reserve_substitution_"
            "join_rule_after_policy_approval_only"
        ),
        "current_effect": "none_report_only_all_results_remain_quarantined",
        "candidates": ready_candidates,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }


def policy_impact_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Reserve Substitution Policy Impact Preview",
        "",
        f"- Final status: `{packet.get('final_status')}`",
        f"- Ready candidate count: `{packet.get('ready_candidate_count')}`",
        f"- Mapping pair count: `{packet.get('mapping_pair_count')}`",
        (
            "- Potential official-result runner rows blocked by policy: "
            f"`{packet.get('potential_official_result_runner_rows_blocked_by_policy')}`"
        ),
        (
            "- Matched backlog top gap races: "
            f"`{packet.get('matched_backlog_top_gap_race_count')}`"
        ),
        f"- Dataset join allowed: `{packet.get('dataset_join_allowed')}`",
        (
            "- Official result acceptance allowed: "
            f"`{packet.get('official_result_acceptance_allowed')}`"
        ),
        f"- DB write: `{packet.get('db_write')}`",
        "",
        "This preview is report-only and does not accept, join, write, train, promote, or trigger betting actions.",
        "",
        "## Candidates",
        "",
    ]
    for candidate in sequence(packet.get("candidates")):
        if not isinstance(candidate, Mapping):
            continue
        lines.extend(
            [
                f"- Race: `{candidate.get('race_id')}`",
                f"  - Result box count: `{candidate.get('result_box_count')}`",
                f"  - Mapping pair count: `{candidate.get('mapping_pair_count')}`",
                f"  - Dataset join blockers: `{candidate.get('dataset_join_blockers')}`",
                f"  - Source URL: `{candidate.get('source_url')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def manual_review_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Reserve Substitution Manual Review Packet",
        "",
        f"- Final status: `{packet.get('final_status')}`",
        f"- Candidate count: `{packet.get('candidate_count')}`",
        f"- Ready candidate count: `{packet.get('ready_candidate_count')}`",
        f"- Blocked candidate count: `{packet.get('blocked_candidate_count')}`",
        "- Automatic acceptance allowed: `False`",
        "- Dataset join allowed: `False`",
        "- Official result acceptance allowed: `False`",
        "",
        (
            "This packet is report-only. It records a mapping hypothesis for manual "
            "policy review and does not accept, join, write, train, promote, or "
            "trigger betting actions."
        ),
        "",
        "## Candidates",
        "",
    ]
    for candidate in sequence(packet.get("candidates")):
        if not isinstance(candidate, Mapping):
            continue
        facts = mapping(candidate.get("race_facts"))
        hypothesis = mapping(candidate.get("mapping_hypothesis"))
        lines.extend(
            [
                f"- Race: `{candidate.get('race_id')}`",
                f"  - Review status: `{candidate.get('review_status')}`",
                f"  - Source URL: `{mapping(candidate.get('source')).get('source_url')}`",
                f"  - Result boxes: `{facts.get('result_boxes')}`",
                f"  - Reserve boxes: `{facts.get('candidate_reserve_boxes')}`",
                f"  - Scratched participant boxes: `{facts.get('scratched_participant_boxes')}`",
                f"  - Mapping status: `{hypothesis.get('mapping_status')}`",
                f"  - Mapping pairs: `{hypothesis.get('pairs')}`",
                f"  - Dataset join blockers: `{candidate.get('dataset_join_blockers')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def summary_markdown(packet: Mapping[str, Any]) -> str:
    lines = [
        "# Official Result Reserve Substitution Preflight",
        "",
        f"- Final status: `{packet.get('final_status')}`",
        f"- Candidate count: `{packet.get('candidate_count')}`",
        f"- Blocked candidate count: `{packet.get('blocked_candidate_count')}`",
        f"- Ready for policy review count: `{packet.get('ready_for_policy_review_count')}`",
        f"- Blocker counts: `{packet.get('blocker_counts')}`",
        "",
        "No DB writes, label writes, quarantine bypasses, official-result acceptance, snapshot mutations, model training, registry changes, promotions, betting actions, EV actions, or TGR changes were performed.",
        "",
        "## Candidates",
        "",
    ]
    for candidate in sequence(packet.get("candidates")):
        if not isinstance(candidate, Mapping):
            continue
        lines.extend(
            [
                f"- Race: `{candidate.get('race_id')}`",
                f"  - Status: `{candidate.get('preflight_status')}`",
                f"  - Readiness blockers: `{candidate.get('readiness_blockers')}`",
                f"  - Dataset join blockers: `{candidate.get('dataset_join_blockers')}`",
                f"  - Expected reserve-policy facts: `{candidate.get('blockers')}`",
                f"  - Reserve boxes: `{candidate.get('candidate_reserve_boxes')}`",
                f"  - Scratched participant boxes: `{candidate.get('scratched_participant_boxes')}`",
                f"  - Source URL: `{candidate.get('source_url')}`",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def run_preflight(
    *,
    backlog_unified_evidence_status_path: Path,
    output_dir: Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    assert_output_dir_safe(output_dir)
    backlog_status = load_json(backlog_unified_evidence_status_path)
    packet = build_preflight_packet(
        backlog_unified_evidence_status=backlog_status,
        backlog_unified_evidence_status_path=backlog_unified_evidence_status_path,
        generated_at=generated_at,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manual_review_packet = build_manual_review_packet(
        preflight_packet=packet,
        generated_at=generated_at,
    )
    policy_impact_preview = build_policy_impact_preview(
        backlog_unified_evidence_status=backlog_status,
        manual_review_packet=manual_review_packet,
        generated_at=generated_at,
    )
    write_json(output_dir / "official_result_reserve_substitution_preflight.json", packet)
    write_json(
        output_dir / "reserve_substitution_manual_review_packet.json",
        manual_review_packet,
    )
    write_json(
        output_dir / "reserve_substitution_policy_impact_preview.json",
        policy_impact_preview,
    )
    write_jsonl(
        output_dir / "reserve_substitution_manual_review_candidates.jsonl",
        [
            candidate
            for candidate in sequence(manual_review_packet.get("candidates"))
            if isinstance(candidate, Mapping)
        ],
    )
    write_text(output_dir / "SUMMARY.md", summary_markdown(packet))
    write_text(
        output_dir / "MANUAL_REVIEW.md",
        manual_review_markdown(manual_review_packet),
    )
    write_text(
        output_dir / "POLICY_IMPACT_PREVIEW.md",
        policy_impact_markdown(policy_impact_preview),
    )
    write_text(output_dir / "final_status.txt", str(packet["final_status"]) + "\n")
    return packet


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog-unified-evidence-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_PARENT
        / f"official_result_reserve_substitution_preflight_{now_id()}"
    )
    packet = run_preflight(
        backlog_unified_evidence_status_path=args.backlog_unified_evidence_status,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "final_status": packet.get("final_status"),
                "output_dir": str(output_dir),
                "candidate_count": packet.get("candidate_count"),
                "blocked_candidate_count": packet.get("blocked_candidate_count"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
