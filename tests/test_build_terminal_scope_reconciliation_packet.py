import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_terminal_scope_reconciliation_packet as terminal_scope
from scripts.build_terminal_scope_reconciliation_packet import (
    build_terminal_scope_packet,
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
        ],
    )
    conn.commit()
    conn.close()
    return path


def _forecast(path: Path) -> Path:
    payload = {
        "schema_version": "post_repair_label_gate_forecast_packet_v1",
        "status": "REPORT_ONLY_POST_REPAIR_LABEL_GATE_FORECAST_PACKET",
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "forecast_rows": [
            {
                "race_id": "R1",
                "review_lane": "P2_TOP1_MISS_PARSED_OFFICIAL_REVIEW",
                "forecast_gate": "POST_REPAIR_FORECAST_STILL_INCOMPLETE",
                "terminal_status_count": 1,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _policy(path: Path) -> Path:
    payload = {
        "schema_version": "missing_runner_insert_policy_packet_v1",
        "status": "REPORT_ONLY_MISSING_RUNNER_INSERT_POLICY_PACKET",
        "report_only": True,
        "writes_performed": dict(WRITES_PERFORMED),
        "candidate_rows": [
            {
                "race_id": "R1",
                "candidate_id": "R1::gamma",
                "official_dog_name": "Gamma",
                "name_key": "gamma",
                "box_number": 2,
                "finish_position": 2,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
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
                ],
                "terminal_statuses": [{"box_number": 3, "status": "SCR"}],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_terminal_scope_reconciliation_blocks_insert_only_conflicts(tmp_path: Path):
    packet = build_terminal_scope_packet(
        forecast_packet_path=_forecast(tmp_path / "forecast.json"),
        insert_policy_packet_path=_policy(tmp_path / "policy.json"),
        lookup_packet_paths=[_lookup(tmp_path / "lookup.json")],
        db_path=_db(tmp_path / "greyhound.sqlite"),
    )

    assert packet["schema_version"] == "terminal_scope_reconciliation_packet_v1"
    assert packet["status"] == "REPORT_ONLY_TERMINAL_SCOPE_RECONCILIATION_PACKET"
    assert packet["safe_to_write_now"] is False
    assert packet["writes_performed"]["db_write"] is False
    assert packet["summary"]["safe_for_insert_only_count"] == 0
    assert packet["summary"]["insert_candidate_box_conflict_count"] == 1
    row = packet["race_diagnostics"][0]
    assert row["reconciliation_lane"] == "TERMINAL_SCOPE_MANUAL_RECONCILIATION_REQUIRED"
    assert row["missing_official_finisher_name_keys"] == ["gamma"]
    assert row["extra_db_name_keys"] == ["beta"]
    assert row["insert_candidates_with_box_conflict"][0]["official_dog_name"] == "Gamma"


def test_terminal_scope_reconciliation_cli_writes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(terminal_scope, "ROOT", tmp_path)
    output_dir = tmp_path / "artifacts/full_evidence_orchestration_20260525/terminal_scope"
    exit_code = main(
        [
            "--forecast-packet",
            str(_forecast(tmp_path / "forecast.json")),
            "--insert-policy-packet",
            str(_policy(tmp_path / "policy.json")),
            "--lookup-packet",
            str(_lookup(tmp_path / "lookup.json")),
            "--db",
            str(_db(tmp_path / "greyhound.sqlite")),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    payload = json.loads((output_dir / "terminal_scope_reconciliation_packet.json").read_text())
    assert payload["summary"]["safe_for_insert_only_count"] == 0
    assert (output_dir / "terminal_scope_reconciliation.csv").exists()
    assert "No DB rows" in (output_dir / "SUMMARY.md").read_text(encoding="utf-8")


def test_terminal_scope_reconciliation_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/terminal_scope"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        terminal_scope._assert_output_dir_safe(outside, root=tmp_path)


def test_terminal_scope_reconciliation_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        terminal_scope._assert_output_dir_safe(tmp_path / "reports/terminal_scope", root=tmp_path)
