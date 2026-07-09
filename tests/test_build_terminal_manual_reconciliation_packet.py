import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_terminal_scope_reconciliation_packet as terminal_scope_guard
from scripts.build_terminal_manual_reconciliation_packet import (
    build_manual_reconciliation_packet,
    main,
)


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


def _db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table dog_race_data (
            race_id text,
            dog_name text,
            dog_clean_name text,
            box_number integer,
            finish_position integer,
            placing integer,
            scraped_finish_position text,
            data_source text
        )
        """
    )
    conn.executemany(
        "insert into dog_race_data values (?,?,?,?,?,?,?,?)",
        [
            ("R1", "1. Alpha", "Alpha", 1, 1, None, None, None),
            ("R1", "2. Beta", "Beta", 2, 2, None, None, None),
            ("R1", "3. Scratch Row", "Scratch Row", 3, None, None, None, None),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _lookup(path: Path) -> Path:
    payload = {
        "schema_version": "official_reverify_lookup_dry_run_v1",
        "status": "REPORT_ONLY",
        "writes_performed": {
            "db_write": False,
            "label_write": False,
            "official_fetch": True,
        },
        "results": [
            {
                "legacy_race_id": "R1",
                "positions": [
                    {"dog_name": "Alpha", "box_number": 1, "finish_position": 1},
                    {"dog_name": "Gamma", "box_number": 2, "finish_position": 2},
                    {"dog_name": "Delta", "box_number": 4, "finish_position": 3},
                ],
                "terminal_statuses": [{"box_number": 3, "status": "SCR"}],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _terminal_scope_packet(path: Path, db: Path, lookup: Path) -> Path:
    payload = {
        "schema_version": "terminal_scope_reconciliation_packet_v1",
        "status": "REPORT_ONLY_TERMINAL_SCOPE_RECONCILIATION_PACKET",
        "report_only": True,
        "safe_to_write_now": False,
        "writes_performed": dict(WRITES_PERFORMED),
        "source_evidence": {
            "db": str(db),
            "lookup_packets": [str(lookup)],
        },
        "race_diagnostics": [
            {
                "race_id": "R1",
                "review_lane": "P2_TOP1_MISS_PARSED_OFFICIAL_REVIEW",
                "forecast_gate": "POST_REPAIR_FORECAST_STILL_INCOMPLETE",
                "reconciliation_lane": "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED",
                "naive_insert_candidates": [
                    {
                        "race_id": "R1",
                        "candidate_id": "R1::gamma",
                        "official_dog_name": "Gamma",
                        "name_key": "gamma",
                        "box_number": 2,
                        "finish_position": 2,
                    },
                    {
                        "race_id": "R1",
                        "candidate_id": "R1::delta",
                        "official_dog_name": "Delta",
                        "name_key": "delta",
                        "box_number": 4,
                        "finish_position": 3,
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_terminal_manual_reconciliation_builds_row_policy_items(tmp_path: Path):
    db = _db(tmp_path / "greyhound.sqlite")
    lookup = _lookup(tmp_path / "lookup.json")
    terminal = _terminal_scope_packet(tmp_path / "terminal.json", db, lookup)

    packet = build_manual_reconciliation_packet(terminal_scope_packet_path=terminal)

    assert packet["schema_version"] == "terminal_manual_reconciliation_packet_v1"
    assert packet["status"] == "REPORT_ONLY_TERMINAL_MANUAL_RECONCILIATION_PACKET"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["races_considered"] == 1
    assert packet["summary"]["races_excluded_from_insert_only_apply_count"] == 1
    assert packet["summary"]["missing_official_finisher_manual_count"] == 2
    assert packet["summary"]["extra_db_row_manual_count"] == 2
    item_types = packet["summary"]["item_type_counts"]
    assert item_types["missing_official_finisher_box_conflict_policy_required"] == 1
    assert item_types["missing_official_finisher_insert_deferred_until_terminal_policy"] == 1
    assert item_types["terminal_status_db_row_policy_required"] == 1
    assert item_types["extra_db_row_on_terminal_status_box_policy_required"] == 1


def test_terminal_manual_reconciliation_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(terminal_scope_guard, "ROOT", tmp_path)
    db = _db(tmp_path / "greyhound.sqlite")
    lookup = _lookup(tmp_path / "lookup.json")
    terminal = _terminal_scope_packet(tmp_path / "terminal.json", db, lookup)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/manual"

    exit_code = main(
        [
            "--terminal-scope-packet",
            str(terminal),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "terminal_manual_reconciliation_packet.json").read_text())
    assert payload["summary"]["manual_item_count"] == 5
    assert (output_dir / "terminal_manual_reconciliation_items.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_terminal_manual_reconciliation_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/manual"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        terminal_scope_guard._assert_output_dir_safe(outside, root=tmp_path)


def test_terminal_manual_reconciliation_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        terminal_scope_guard._assert_output_dir_safe(tmp_path / "reports/manual", root=tmp_path)
