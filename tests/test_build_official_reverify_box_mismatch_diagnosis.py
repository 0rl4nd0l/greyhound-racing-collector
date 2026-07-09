import json
import sqlite3
from pathlib import Path

import pytest

import scripts.build_official_reverify_box_mismatch_diagnosis as diagnosis
from scripts.build_official_reverify_box_mismatch_diagnosis import (
    build_box_mismatch_diagnosis,
    main,
)


def _db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        create table dog_race_data (
            race_id text,
            dog_name text,
            box_number integer,
            finish_position integer,
            placing integer,
            scraped_finish_position text,
            data_source text
        )
        """
    )
    rows = [
        ("R1", "6. Weblec Bubbles", 1, 4, None, None, None),
        ("R1", "4. Carry On Natasha", 4, 2, None, None, None),
        ("R1", "2. That Smile", 5, 5, None, None, None),
        ("R1", "1. Creeper", 7, 3, None, None, None),
        ("R1", "8. Fencer Max", 8, 1, None, None, None),
    ]
    conn.executemany("insert into dog_race_data values (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _preflight(path: Path, *, missing_name: bool = False) -> Path:
    positions = [
        {"box_number": 8, "dog_name": "Fencer Max", "finish_position": 1},
        {"box_number": 2, "dog_name": "That Smile", "finish_position": 2},
        {"box_number": 4, "dog_name": "Carry On Natasha", "finish_position": 3},
        {"box_number": 1, "dog_name": "Creeper", "finish_position": 4},
        {
            "box_number": 6,
            "dog_name": "Different Dog" if missing_name else "Weblec Bubbles",
            "finish_position": 5,
        },
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": "official_reverify_label_preflight_v1",
                "status": "PREFLIGHT_READY_WITH_BLOCKERS",
                "writes_performed": {
                    "db_write": False,
                    "label_write": False,
                    "official_fetch": False,
                    "snapshot_mutation": False,
                    "manifest_mutation": False,
                    "model_training": False,
                    "registry_mutation": False,
                    "promotion": False,
                    "betting_decision": False,
                },
                "candidate_preflight": [
                    {
                        "legacy_race_id": "R1",
                        "lookup_key": {"venue": "TEST", "race_date": "2025-01-01", "race_number": 1},
                        "source_url": "https://example.invalid/race",
                        "resolved_db_race_id": "R1",
                        "positions": positions,
                        "preflight_status": "BLOCKED",
                        "blockers": ["db_has_existing_result_rows"],
                        "row_alignment": {
                            "box_set_matches_official": False,
                            "official_box_numbers": [1, 2, 4, 6, 8],
                            "existing_box_numbers": [1, 4, 5, 7, 8],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_box_mismatch_diagnosis_builds_name_matched_no_write_review(tmp_path: Path):
    packet = build_box_mismatch_diagnosis(
        preflight_packet_path=_preflight(tmp_path / "preflight.json"),
        db_path=_db(tmp_path / "greyhound.db"),
    )

    assert packet["schema_version"] == "official_reverify_box_mismatch_diagnosis_v1"
    assert packet["status"] == "REPORT_ONLY_NAME_MATCHED_BOX_MISMATCH_REVIEW_READY"
    assert packet["writes_performed"]["db_write"] is False
    assert packet["writes_performed"]["label_write"] is False
    assert packet["summary"]["candidate_count"] == 1
    assert packet["summary"]["safe_to_write_now_count"] == 0
    candidate = packet["candidates"][0]
    assert candidate["exact_name_set_match"] is True
    assert candidate["official_box_numbers"] == [1, 2, 4, 6, 8]
    assert candidate["db_box_numbers"] == [1, 4, 5, 7, 8]
    assert len(candidate["proposed_no_write_updates"]) == 5
    that_smile = next(
        row for row in candidate["proposed_no_write_updates"] if row["name_key"] == "that smile"
    )
    assert that_smile["before"]["box_number"] == 5
    assert that_smile["after"]["box_number"] == 2
    assert that_smile["after"]["data_source"] == "thedogs_official"


def test_box_mismatch_diagnosis_blocks_when_name_set_differs(tmp_path: Path):
    packet = build_box_mismatch_diagnosis(
        preflight_packet_path=_preflight(tmp_path / "preflight.json", missing_name=True),
        db_path=_db(tmp_path / "greyhound.db"),
    )

    assert packet["status"] == "REPORT_ONLY_BOX_MISMATCH_MANUAL_REVIEW_REQUIRED"
    candidate = packet["candidates"][0]
    assert candidate["exact_name_set_match"] is False
    assert "runner_name_set_not_exact" in candidate["blockers"]
    assert "different dog" in candidate["missing_db_name_keys"]
    assert "weblec bubbles" in candidate["extra_db_name_keys"]


def test_box_mismatch_diagnosis_cli_writes_json_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(diagnosis, "ROOT", tmp_path)
    output = tmp_path / "artifacts/full_evidence_orchestration_20260525/box_mismatch/diagnosis.json"
    report = tmp_path / "artifacts/full_evidence_orchestration_20260525/box_mismatch/SUMMARY.md"
    status = main(
        [
            "--preflight-packet",
            str(_preflight(tmp_path / "preflight.json")),
            "--db",
            str(_db(tmp_path / "greyhound.db")),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert status == 0
    assert json.loads(output.read_text())["status"] == (
        "REPORT_ONLY_NAME_MATCHED_BOX_MISMATCH_REVIEW_READY"
    )
    assert "No DB writes" in report.read_text()


def test_box_mismatch_diagnosis_rejects_absolute_output_outside_repo(tmp_path: Path):
    outside = tmp_path.parent / "artifacts/full_evidence_orchestration_20260525/box_mismatch/diagnosis.json"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        diagnosis._assert_output_path_safe(outside, root=tmp_path)


def test_box_mismatch_diagnosis_rejects_in_repo_non_artifact_output(tmp_path: Path):
    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        diagnosis._assert_output_path_safe(tmp_path / "reports/diagnosis.json", root=tmp_path)
