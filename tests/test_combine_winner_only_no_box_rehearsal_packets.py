import json
from pathlib import Path

import pytest

from scripts.combine_winner_only_no_box_rehearsal_packets import (
    combine_rehearsal_packets,
    main,
)
from scripts.build_winner_only_no_box_rehearsal_packet import write_rehearsal_outputs


def _row(race_id: str, dog_name_key: str, actual_win: int, *, complete: bool = False) -> dict:
    return {
        "race_id": race_id,
        "legacy_race_id": race_id,
        "identity_key": f"2025-01-01|TEST|{race_id}",
        "race_date": "2025-01-01",
        "venue": "TEST",
        "race_number": 1,
        "dog_name_key": dog_name_key,
        "dog_name": dog_name_key.title(),
        "actual_win": actual_win,
        "candidate_kind": "complete_field" if complete else "partial_field",
        "field_scope": "complete_name_set_box_drift" if complete else "partial_db_name_subset_of_official_finishers",
        "field_complete_for_ranking": complete,
        "race_grouped_actual_win_ranking_allowed": complete,
        "target_source": "official_winner_name_metadata_confirmed",
        "label_scope": "actual_win_only",
        "box_features_allowed": False,
        "finish_order_labels_allowed": False,
        "top3_labels_allowed": False,
        "official_safe_label_candidate": False,
        "label_write_approved": False,
    }


def _packet(rows: list[dict]) -> dict:
    races = {row["race_id"] for row in rows}
    complete_races = {
        row["race_id"] for row in rows if row["field_complete_for_ranking"] is True
    }
    return {
        "schema_version": "winner_only_no_box_actual_win_rehearsal_v1",
        "generated_at": "2026-06-09T00:00:00+00:00",
        "status": "REPORT_ONLY_READY_FOR_NO_BOX_ACTUAL_WIN_EVALUATION",
        "failures": [],
        "report_only": True,
        "write_ready": False,
        "label_write_approved": False,
        "model_training_performed": False,
        "model_promotion_allowed": False,
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "metadata_write": False,
            "official_fetch": False,
            "snapshot_mutation": False,
            "manifest_mutation": False,
            "model_training": False,
            "registry_mutation": False,
            "promotion": False,
            "tgr_enablement": False,
            "betting_decision": False,
            "ev_action": False,
        },
        "source_packets": {"fixture": True},
        "scope": {
            "row_schema_version": "winner_only_no_box_actual_win_rows_v1",
            "forbidden_row_fields": [],
        },
        "summary": {
            "confirmed_winner_only_candidate_count": len(races),
            "complete_field_confirmed_winner_only_candidate_count": len(complete_races),
            "partial_field_confirmed_winner_only_candidate_count": len(races) - len(complete_races),
            "race_grouped_ranking_ready_candidate_count": len(complete_races),
            "blocked_or_non_candidate_records_seen": 2,
            "p1_recovery_candidates": 1,
        },
        "forbidden_without_explicit_approval": [],
        "race_summaries": [
            {"race_id": race_id, "status": "MATERIALIZED_ACTUAL_WIN_ONLY_ROWS"}
            for race_id in sorted(races)
        ],
        "blocked_or_non_candidate_records": [],
        "recovery_queue": {
            "summary": {"recovery_queue_count": 1},
            "records": [
                {
                    "priority": "P1",
                    "recovery_lane": "name_set_and_parser_repair_candidate",
                    "race_id": sorted(races)[0],
                }
            ],
        },
    }


def _write_source(directory: Path, rows: list[dict]) -> Path:
    directory.mkdir(parents=True)
    (directory / "winner_only_no_box_actual_win_rehearsal_packet.json").write_text(
        json.dumps(_packet(rows), sort_keys=True),
        encoding="utf-8",
    )
    with (directory / "winner_only_no_box_actual_win_rows.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return directory


def test_combines_report_only_packets_and_preserves_gates(tmp_path: Path):
    first = _write_source(
        tmp_path / "first",
        [_row("R1", "winner dog", 1, complete=True), _row("R1", "other dog", 0, complete=True)],
    )
    second = _write_source(
        tmp_path / "second",
        [_row("R2", "alpha", 1), _row("R2", "beta", 0), _row("R2", "gamma", 0)],
    )

    packet, rows = combine_rehearsal_packets([first, second], expected_races=2)

    assert packet["failures"] == []
    assert packet["report_only"] is True
    assert packet["write_ready"] is False
    assert packet["summary"]["confirmed_winner_only_candidate_count"] == 2
    assert packet["summary"]["complete_field_confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["partial_field_confirmed_winner_only_candidate_count"] == 1
    assert packet["summary"]["actual_win_positive_rows"] == 2
    assert packet["summary"]["no_box_row_policy_pass"] is True
    assert len(rows) == 5

    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/combined"
    write_rehearsal_outputs(output_dir, packet, rows, root=tmp_path)
    written = json.loads(
        (output_dir / "winner_only_no_box_actual_win_rehearsal_packet.json").read_text(
            encoding="utf-8",
        )
    )
    assert written["source_packets"]["combined_packet_count"] == 2


def test_combiner_fails_closed_on_duplicate_race(tmp_path: Path):
    first = _write_source(
        tmp_path / "first",
        [_row("R1", "winner dog", 1), _row("R1", "other dog", 0)],
    )
    second = _write_source(
        tmp_path / "second",
        [_row("R1", "winner dog", 1), _row("R1", "other dog", 0)],
    )

    packet, _ = combine_rehearsal_packets([first, second])

    assert packet["status"] == "REPORT_ONLY_COMBINED_WITH_FAILURES"
    assert any("duplicate_race_id_across_sources:R1" == failure for failure in packet["failures"])


def test_combiner_fails_closed_on_forbidden_row_field(tmp_path: Path):
    row = _row("R1", "winner dog", 1)
    row["box_number"] = 1
    source = _write_source(tmp_path / "source", [row])

    packet, _ = combine_rehearsal_packets([source])

    assert packet["status"] == "REPORT_ONLY_COMBINED_WITH_FAILURES"
    assert any("row_forbidden_fields:" in failure for failure in packet["failures"])


def test_combiner_cli_rejects_output_outside_repo(tmp_path: Path):
    source = _write_source(
        tmp_path / "source",
        [_row("R1", "winner dog", 1), _row("R1", "other dog", 0)],
    )

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        main(
            [
                "--packet",
                str(source),
                "--output-dir",
                str(tmp_path / "outside-combined"),
            ]
        )
