import json
from datetime import datetime, timezone

import pytest

from scripts import build_official_result_reserve_substitution_preflight as preflight


def _status_payload():
    return {
        "schema_version": "shadow_autopilot_backlog_unified_evidence_status_v1",
        "status": "BACKLOG_UNIFIED_EVIDENCE_DATASETS_BUILT",
        "top_gap_races": [
            {
                "race_id": "Race 7 - TAREE - 2026-06-13",
                "race_date": "2026-06-13",
                "venue": "TAREE",
                "official_result_quarantine_reason": "ingest_failed_or_unsafe_match",
                "official_result_quarantine_errors": [
                    "result_boxes_not_in_participants:9"
                ],
                "official_result_quarantine_participant_source": (
                    "shadow_run_predictions"
                ),
                "official_result_quarantine_participant_count": 8,
                "official_result_quarantine_participant_boxes": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                ],
                "official_result_quarantine_attempted_source_box_sets": [
                    {
                        "source": "thedogs_official",
                        "source_url": (
                            "https://www.thedogs.com.au/racing/taree/"
                            "2026-06-13/7/example?trial=false"
                        ),
                        "status": "resulted",
                        "result_boxes": [2, 8, 4, 7, 3, 9, 6, 5],
                        "dog_names_by_box": {
                            "2": "Riverside Levi",
                            "8": "Beautiful Banter",
                            "4": "Bentley Elsa",
                            "7": "Crossed Toes",
                            "3": "Wonder Kid",
                            "9": "Reserve Runner",
                            "6": "Miss Clarence",
                            "5": "Rockin Panama",
                        },
                        "terminal_status_boxes": [1, 10],
                    }
                ],
                "official_result_quarantine_reserve_substitution_diagnostic": {
                    "classification": (
                        "possible_reserve_substitution_manual_review_required"
                    ),
                    "acceptance_status": "not_accepted_report_only",
                    "result_boxes_outside_participants": [9],
                    "result_boxes_inside_participants": [2, 3, 4, 5, 6, 7, 8],
                    "candidate_reserve_boxes": [9],
                    "scratched_participant_boxes": [1],
                    "terminal_status_boxes": [1, 10],
                    "terminal_status_boxes_outside_participants": [10],
                },
            }
        ],
    }


def test_reserve_substitution_preflight_surfaces_ready_manual_policy_review():
    packet = preflight.build_preflight_packet(
        backlog_unified_evidence_status=_status_payload(),
        generated_at=datetime(2026, 6, 14, 5, 29, tzinfo=timezone.utc),
    )

    assert packet["final_status"] == preflight.FINAL_READY
    assert packet["candidate_count"] == 1
    assert packet["blocked_candidate_count"] == 0
    assert packet["ready_for_policy_review_count"] == 1
    assert packet["no_write_guarantees"]["official_result_acceptance"] is False
    candidate = packet["candidates"][0]
    assert candidate["race_id"] == "Race 7 - TAREE - 2026-06-13"
    assert candidate["preflight_status"] == "READY_FOR_MANUAL_POLICY_REVIEW"
    assert candidate["acceptance_status"] == "not_accepted_report_only"
    assert candidate["acceptance_effect"] == "none_report_only"
    assert candidate["recommended_action"] == (
        "manual_review_reserve_substitution_policy"
    )
    assert candidate["candidate_reserve_boxes"] == [9]
    assert candidate["scratched_participant_boxes"] == [1]
    assert candidate["terminal_status_boxes"] == [1, 10]
    assert candidate["source_url"].endswith("/7/example?trial=false")
    assert set(candidate["blockers"]) == {
        "official_result_boxes_outside_frozen_participants_expected_for_reserve_policy",
        "official_terminal_statuses_present_expected_for_reserve_policy",
    }
    assert candidate["readiness_blockers"] == []
    assert candidate["dataset_join_blockers"] == [
        "official_result_remains_quarantined",
        "manual_policy_review_required_before_join",
    ]
    assert candidate["source_policy_evaluations"][0]["policy_status"] == "PASS"


def test_reserve_substitution_preflight_blocks_when_official_names_missing():
    payload = _status_payload()
    source = payload["top_gap_races"][0][
        "official_result_quarantine_attempted_source_box_sets"
    ][0]
    source["dog_names_by_box"].pop("9")

    packet = preflight.build_preflight_packet(
        backlog_unified_evidence_status=payload,
        generated_at=datetime(2026, 6, 14, 5, 29, tzinfo=timezone.utc),
    )

    assert packet["final_status"] == preflight.FINAL_BLOCKED
    assert packet["blocked_candidate_count"] == 1
    candidate = packet["candidates"][0]
    assert candidate["preflight_status"] == "BLOCKED"
    assert "official_result_order_dog_names_missing" in candidate["readiness_blockers"]
    assert (
        "no_official_source_passed_reserve_substitution_policy"
        in candidate["readiness_blockers"]
    )
    assert candidate["source_policy_evaluations"][0]["policy_status"] == "BLOCKED"


def test_manual_review_packet_is_report_only_mapping_hypothesis():
    preflight_packet = preflight.build_preflight_packet(
        backlog_unified_evidence_status=_status_payload(),
        generated_at=datetime(2026, 6, 14, 5, 29, tzinfo=timezone.utc),
    )

    packet = preflight.build_manual_review_packet(
        preflight_packet=preflight_packet,
        generated_at=datetime(2026, 6, 14, 5, 30, tzinfo=timezone.utc),
    )

    assert packet["final_status"] == preflight.MANUAL_REVIEW_READY
    assert packet["approval_required"] is True
    assert packet["automatic_acceptance_allowed"] is False
    assert packet["dataset_join_allowed"] is False
    assert packet["official_result_acceptance_allowed"] is False
    assert packet["db_write"] is False
    assert packet["ready_candidate_count"] == 1
    candidate = packet["candidates"][0]
    assert candidate["review_status"] == "READY_FOR_MANUAL_REVIEW"
    assert candidate["automatic_acceptance_allowed"] is False
    assert candidate["dataset_join_allowed"] is False
    assert candidate["official_result_acceptance_allowed"] is False
    assert candidate["db_write"] is False
    assert candidate["source"]["source"] == "thedogs_official"
    assert candidate["source"]["sportsbook_source_identity"] == (
        "not_applicable_official_result_source"
    )
    assert candidate["race_facts"]["result_boxes"] == [2, 8, 4, 7, 3, 9, 6, 5]
    assert candidate["race_facts"]["candidate_reserve_boxes"] == [9]
    assert candidate["race_facts"]["scratched_participant_boxes"] == [1]
    assert candidate["mapping_hypothesis"] == {
        "mapping_status": "report_only_policy_hypothesis",
        "mapping_acceptance_status": "not_accepted",
        "mapping_blockers": [],
        "pairs": [
            {
                "scratched_participant_box": 1,
                "reserve_box": 9,
                "reserve_dog_name": "Reserve Runner",
                "mapping_basis": (
                    "sorted_count_aligned_reserve_and_scratched_boxes_"
                    "from_quarantine_diagnostic"
                ),
                "mapping_acceptance_status": "not_accepted",
            }
        ],
    }
    assert candidate["quarantine"]["official_result_remains_quarantined"] is True
    assert "official_result_remains_quarantined" in candidate["dataset_join_blockers"]


def test_policy_impact_preview_quantifies_blocked_rows_without_joining():
    preflight_packet = preflight.build_preflight_packet(
        backlog_unified_evidence_status=_status_payload(),
        generated_at=datetime(2026, 6, 14, 5, 29, tzinfo=timezone.utc),
    )
    manual_packet = preflight.build_manual_review_packet(
        preflight_packet=preflight_packet,
        generated_at=datetime(2026, 6, 14, 5, 30, tzinfo=timezone.utc),
    )
    backlog_status = _status_payload()
    backlog_status.update(
        {
            "sample_blocking_gap_count": 1,
            "unified_evidence_eligible_rows": 42,
            "gap_action_counts": {
                "inspect_quarantined_official_result_runner_set": 1,
            },
            "evidence_missing_reason_counts": {
                "official_result_quarantined_unsafe_match": 1,
            },
        }
    )

    packet = preflight.build_policy_impact_preview(
        backlog_unified_evidence_status=backlog_status,
        manual_review_packet=manual_packet,
        generated_at=datetime(2026, 6, 14, 5, 31, tzinfo=timezone.utc),
    )

    assert packet["final_status"] == preflight.IMPACT_PREVIEW_READY
    assert packet["approval_required"] is True
    assert packet["automatic_acceptance_allowed"] is False
    assert packet["dataset_join_allowed"] is False
    assert packet["official_result_acceptance_allowed"] is False
    assert packet["db_write"] is False
    assert packet["candidate_count"] == 1
    assert packet["ready_candidate_count"] == 1
    assert packet["mapping_pair_count"] == 1
    assert packet["potential_official_result_runner_rows_blocked_by_policy"] == 8
    assert packet["matched_backlog_top_gap_race_count"] == 1
    assert packet["backlog_sample_blocking_gap_count"] == 1
    assert packet["backlog_unified_evidence_eligible_rows"] == 42
    assert packet["backlog_gap_action_counts"] == {
        "inspect_quarantined_official_result_runner_set": 1,
    }
    assert packet["backlog_evidence_missing_reason_counts"] == {
        "official_result_quarantined_unsafe_match": 1,
    }
    candidate = packet["candidates"][0]
    assert candidate["race_id"] == "Race 7 - TAREE - 2026-06-13"
    assert candidate["result_box_count"] == 8
    assert candidate["mapping_pair_count"] == 1
    assert candidate["dataset_join_allowed"] is False
    assert candidate["official_result_acceptance_allowed"] is False


def test_reserve_substitution_preflight_writes_report_only_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    status_path = tmp_path / "backlog_unified_evidence_datasets_status.json"
    status_path.write_text(json.dumps(_status_payload()), encoding="utf-8")
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/preflight"

    packet = preflight.run_preflight(
        backlog_unified_evidence_status_path=status_path,
        output_dir=output_dir,
        generated_at=datetime(2026, 6, 14, 5, 29, tzinfo=timezone.utc),
    )

    assert packet["final_status"] == preflight.FINAL_READY
    assert (output_dir / "official_result_reserve_substitution_preflight.json").exists()
    assert (output_dir / "reserve_substitution_manual_review_packet.json").exists()
    assert (output_dir / "reserve_substitution_manual_review_candidates.jsonl").exists()
    assert (output_dir / "reserve_substitution_policy_impact_preview.json").exists()
    assert (output_dir / "SUMMARY.md").exists()
    assert (output_dir / "MANUAL_REVIEW.md").exists()
    assert (output_dir / "POLICY_IMPACT_PREVIEW.md").exists()
    assert (output_dir / "final_status.txt").read_text(encoding="utf-8").strip() == (
        preflight.FINAL_READY
    )
    written = json.loads(
        (output_dir / "official_result_reserve_substitution_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    assert written["source_artifacts"]["backlog_unified_evidence_status"] == (
        "backlog_unified_evidence_datasets_status.json"
    )
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")
    assert "No DB writes" in summary
    assert "READY_FOR_MANUAL_POLICY_REVIEW" in summary
    manual_packet = json.loads(
        (output_dir / "reserve_substitution_manual_review_packet.json").read_text(
            encoding="utf-8"
        )
    )
    assert manual_packet["final_status"] == preflight.MANUAL_REVIEW_READY
    assert manual_packet["dataset_join_allowed"] is False
    assert manual_packet["candidates"][0]["mapping_hypothesis"]["pairs"] == [
        {
            "scratched_participant_box": 1,
            "reserve_box": 9,
            "reserve_dog_name": "Reserve Runner",
            "mapping_basis": (
                "sorted_count_aligned_reserve_and_scratched_boxes_"
                "from_quarantine_diagnostic"
            ),
            "mapping_acceptance_status": "not_accepted",
        }
    ]
    manual_review = (output_dir / "MANUAL_REVIEW.md").read_text(encoding="utf-8")
    assert "report-only" in manual_review
    assert "Dataset join allowed: `False`" in manual_review
    impact_preview = json.loads(
        (output_dir / "reserve_substitution_policy_impact_preview.json").read_text(
            encoding="utf-8"
        )
    )
    assert impact_preview["final_status"] == preflight.IMPACT_PREVIEW_READY
    assert impact_preview["dataset_join_allowed"] is False
    assert impact_preview["official_result_acceptance_allowed"] is False
    assert impact_preview[
        "potential_official_result_runner_rows_blocked_by_policy"
    ] == 8
    impact_markdown = (output_dir / "POLICY_IMPACT_PREVIEW.md").read_text(
        encoding="utf-8"
    )
    assert "Reserve Substitution Policy Impact Preview" in impact_markdown
    assert "DB write: `False`" in impact_markdown


def test_reserve_substitution_preflight_rejects_protected_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    monkeypatch.setattr(
        preflight,
        "PROTECTED_OUTPUT_DIRS",
        (tmp_path / "artifacts/prediction_snapshots",),
    )
    status_path = tmp_path / "status.json"
    status_path.write_text(json.dumps(_status_payload()), encoding="utf-8")

    with pytest.raises(ValueError, match="protected_output_dir"):
        preflight.run_preflight(
            backlog_unified_evidence_status_path=status_path,
            output_dir=tmp_path / "artifacts/prediction_snapshots/preflight",
        )
