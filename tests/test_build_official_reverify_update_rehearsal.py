import json
import sqlite3
from pathlib import Path

from scripts.build_official_reverify_update_rehearsal import build_update_rehearsal, main


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "greyhound.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                results_status TEXT,
                winner_name TEXT,
                winner_source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE dog_race_data (
                race_id TEXT,
                dog_name TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                placing INTEGER,
                scraped_finish_position TEXT,
                data_source TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, results_status, winner_name, winner_source)
            VALUES ('AP_K_2025-07-08_3', 'pending', '', NULL)
            """
        )
        for box, dog_name, old_position in [
            (1, "4. Districts Dub", 5),
            (2, "7. Our Boy Turbo", 6),
            (4, "8. Stella Swift", 6),
            (5, "1. Hayride Warner", 2),
            (7, "2. Shiloh Tanaeya", 5),
            (8, "5. Cantara Pearl", 6),
        ]:
            conn.execute(
                """
                INSERT INTO dog_race_data
                    (race_id, dog_name, box_number, finish_position, data_source)
                VALUES ('AP_K_2025-07-08_3', ?, ?, ?, '')
                """,
                (dog_name, box, old_position),
            )
    return db_path


def _preflight(path: Path) -> Path:
    payload = {
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
                "legacy_race_id": "ap_k_2025-07-08_3",
                "resolved_db_race_id": "AP_K_2025-07-08_3",
                "lookup_key": {"venue": "AP_K", "race_date": "2025-07-08", "race_number": 3},
                "blockers": ["db_has_existing_result_rows"],
                "positions": [
                    {"box_number": 1, "finish_position": 1},
                    {"box_number": 8, "finish_position": 2},
                    {"box_number": 7, "finish_position": 3},
                    {"box_number": 5, "finish_position": 4},
                    {"box_number": 4, "finish_position": 5},
                    {"box_number": 2, "finish_position": 6},
                ],
                "row_alignment": {
                    "box_set_matches_official": True,
                    "existing_box_numbers": [1, 2, 4, 5, 7, 8],
                    "official_box_numbers": [1, 2, 4, 5, 7, 8],
                },
            },
            {
                "legacy_race_id": "mismatch",
                "resolved_db_race_id": "GEE_2025-07-22_5",
                "lookup_key": {"venue": "GEE", "race_date": "2025-07-22", "race_number": 5},
                "blockers": ["db_has_existing_result_rows"],
                "positions": [{"box_number": 1, "finish_position": 1}],
                "row_alignment": {
                    "box_set_matches_official": False,
                    "existing_box_numbers": [1, 4],
                    "official_box_numbers": [1],
                },
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_update_rehearsal_builds_exact_no_write_update_plan(tmp_path):
    packet = build_update_rehearsal(
        preflight_packet_path=_preflight(tmp_path / "preflight.json"),
        db_path=_make_db(tmp_path),
    )

    assert packet["schema_version"] == "official_reverify_update_rehearsal_v1"
    assert packet["status"] == "READY_FOR_EXPLICIT_APPROVED_UPDATE_WRITE"
    assert packet["summary"] == {
        "candidate_count": 1,
        "ready_count": 1,
        "blocked_count": 0,
        "skipped_non_exact_lane_count": 1,
    }
    assert packet["writes_performed"]["db_write"] is False
    candidate = packet["candidates"][0]
    assert candidate["race_id"] == "AP_K_2025-07-08_3"
    assert candidate["winner"] == {
        "box_number": 1,
        "dog_name": "Districts Dub",
    }
    assert candidate["dog_updates"][0] == {
        "box_number": 1,
        "dog_name": "4. Districts Dub",
        "before": {"finish_position": 5, "placing": None, "scraped_finish_position": None, "data_source": ""},
        "after": {"finish_position": 1, "placing": 1, "scraped_finish_position": "1", "data_source": "thedogs_official"},
    }
    assert candidate["metadata_update"]["after"] == {
        "results_status": "resulted",
        "winner_name": "Districts Dub",
        "winner_source": "thedogs_official",
    }


def test_update_rehearsal_cli_writes_json_and_report(tmp_path):
    output = tmp_path / "rehearsal.json"
    report = tmp_path / "rehearsal.md"

    exit_code = main(
        [
            "--preflight-packet",
            str(_preflight(tmp_path / "preflight.json")),
            "--db",
            str(_make_db(tmp_path)),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["ready_count"] == 1
    assert report.exists()
