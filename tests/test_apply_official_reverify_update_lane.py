import json
import sqlite3
from pathlib import Path

import scripts.apply_official_reverify_update_lane as apply_lane
from scripts.apply_official_reverify_update_lane import apply_update_lane


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
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, results_status, winner_name, winner_source)
            VALUES ('UNTOUCHED', 'pending', '', NULL)
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
                    (race_id, dog_name, box_number, finish_position, placing, scraped_finish_position, data_source)
                VALUES ('AP_K_2025-07-08_3', ?, ?, ?, NULL, NULL, NULL)
                """,
                (dog_name, box, old_position),
            )
        conn.execute(
            """
            INSERT INTO dog_race_data
                (race_id, dog_name, box_number, finish_position, placing, scraped_finish_position, data_source)
            VALUES ('UNTOUCHED', '1. Other Dog', 1, NULL, NULL, NULL, NULL)
            """
        )
    return db_path


def _rehearsal(path: Path) -> Path:
    payload = {
        "schema_version": "official_reverify_update_rehearsal_v1",
        "status": "READY_FOR_EXPLICIT_APPROVED_UPDATE_WRITE",
        "failures": [],
        "summary": {
            "candidate_count": 1,
            "ready_count": 1,
            "blocked_count": 0,
            "skipped_non_exact_lane_count": 470,
        },
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
            "betting_decision": False,
        },
        "candidates": [
            {
                "race_id": "AP_K_2025-07-08_3",
                "status": "READY",
                "blockers": [],
                "winner": {"box_number": 1, "dog_name": "Districts Dub"},
                "dog_updates": [
                    {
                        "box_number": 1,
                        "dog_name": "4. Districts Dub",
                        "before": {
                            "finish_position": 5,
                            "placing": None,
                            "scraped_finish_position": None,
                            "data_source": None,
                        },
                        "after": {
                            "finish_position": 1,
                            "placing": 1,
                            "scraped_finish_position": "1",
                            "data_source": "thedogs_official",
                        },
                    },
                    {
                        "box_number": 2,
                        "dog_name": "7. Our Boy Turbo",
                        "before": {
                            "finish_position": 6,
                            "placing": None,
                            "scraped_finish_position": None,
                            "data_source": None,
                        },
                        "after": {
                            "finish_position": 6,
                            "placing": 6,
                            "scraped_finish_position": "6",
                            "data_source": "thedogs_official",
                        },
                    },
                    {
                        "box_number": 4,
                        "dog_name": "8. Stella Swift",
                        "before": {
                            "finish_position": 6,
                            "placing": None,
                            "scraped_finish_position": None,
                            "data_source": None,
                        },
                        "after": {
                            "finish_position": 5,
                            "placing": 5,
                            "scraped_finish_position": "5",
                            "data_source": "thedogs_official",
                        },
                    },
                    {
                        "box_number": 5,
                        "dog_name": "1. Hayride Warner",
                        "before": {
                            "finish_position": 2,
                            "placing": None,
                            "scraped_finish_position": None,
                            "data_source": None,
                        },
                        "after": {
                            "finish_position": 4,
                            "placing": 4,
                            "scraped_finish_position": "4",
                            "data_source": "thedogs_official",
                        },
                    },
                    {
                        "box_number": 7,
                        "dog_name": "2. Shiloh Tanaeya",
                        "before": {
                            "finish_position": 5,
                            "placing": None,
                            "scraped_finish_position": None,
                            "data_source": None,
                        },
                        "after": {
                            "finish_position": 3,
                            "placing": 3,
                            "scraped_finish_position": "3",
                            "data_source": "thedogs_official",
                        },
                    },
                    {
                        "box_number": 8,
                        "dog_name": "5. Cantara Pearl",
                        "before": {
                            "finish_position": 6,
                            "placing": None,
                            "scraped_finish_position": None,
                            "data_source": None,
                        },
                        "after": {
                            "finish_position": 2,
                            "placing": 2,
                            "scraped_finish_position": "2",
                            "data_source": "thedogs_official",
                        },
                    },
                ],
                "metadata_update": {
                    "before": {
                        "results_status": "pending",
                        "winner_name": "",
                        "winner_source": None,
                    },
                    "after": {
                        "results_status": "resulted",
                        "winner_name": "Districts Dub",
                        "winner_source": "thedogs_official",
                    },
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_approved_exact_lane_applies_labels_and_creates_backup(tmp_path):
    db_path = _make_db(tmp_path)
    output_dir = tmp_path / "apply"

    packet = apply_update_lane(
        rehearsal_packet_path=_rehearsal(tmp_path / "rehearsal.json"),
        db_path=db_path,
        output_dir=output_dir,
        write_labels_approved=True,
    )

    assert packet["status"] == "APPLIED"
    assert packet["summary"] == {
        "candidate_count": 1,
        "races_updated": 1,
        "dog_rows_updated": 6,
        "metadata_rows_updated": 1,
        "skipped_non_exact_lane_count": 470,
    }
    assert Path(packet["backup"]["path"]).exists()
    assert packet["writes_performed"]["db_write"] is True
    assert packet["writes_performed"]["label_write"] is True
    assert packet["writes_performed"]["metadata_write"] is True
    assert packet["writes_performed"]["model_training"] is False

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT box_number, finish_position, placing, scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = 'AP_K_2025-07-08_3'
            ORDER BY box_number
            """
        ).fetchall()
        metadata = conn.execute(
            """
            SELECT results_status, winner_name, winner_source
            FROM race_metadata
            WHERE race_id = 'AP_K_2025-07-08_3'
            """
        ).fetchone()
        untouched = conn.execute(
            """
            SELECT finish_position, data_source
            FROM dog_race_data
            WHERE race_id = 'UNTOUCHED'
            """
        ).fetchone()

    assert rows == [
        (1, 1, 1, "1", "thedogs_official"),
        (2, 6, 6, "6", "thedogs_official"),
        (4, 5, 5, "5", "thedogs_official"),
        (5, 4, 4, "4", "thedogs_official"),
        (7, 3, 3, "3", "thedogs_official"),
        (8, 2, 2, "2", "thedogs_official"),
    ]
    assert metadata == ("resulted", "Districts Dub", "thedogs_official")
    assert untouched == (None, None)
    assert (output_dir / "official_reverify_update_apply.json").exists()
    assert (output_dir / "report.md").exists()


def test_update_lane_refuses_without_approval_flag(tmp_path):
    db_path = _make_db(tmp_path)
    output_dir = tmp_path / "apply"

    packet = apply_update_lane(
        rehearsal_packet_path=_rehearsal(tmp_path / "rehearsal.json"),
        db_path=db_path,
        output_dir=output_dir,
        write_labels_approved=False,
    )

    assert packet["status"] == "NOT_APPROVED"
    assert packet["failures"] == ["write_labels_approved_flag_missing"]
    assert packet["writes_performed"]["db_write"] is False
    assert not (output_dir / "db_backups").exists()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT finish_position, placing, scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = 'AP_K_2025-07-08_3'
              AND box_number = 1
            """
        ).fetchone()
        metadata = conn.execute(
            """
            SELECT results_status, winner_name, winner_source
            FROM race_metadata
            WHERE race_id = 'AP_K_2025-07-08_3'
            """
        ).fetchone()

    assert row == (5, None, None, None)
    assert metadata == ("pending", "", None)


def test_update_lane_refuses_when_live_preimage_no_longer_matches(tmp_path):
    db_path = _make_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE dog_race_data
               SET finish_position = 1
             WHERE race_id = 'AP_K_2025-07-08_3'
               AND box_number = 1
            """
        )
    output_dir = tmp_path / "apply"

    packet = apply_update_lane(
        rehearsal_packet_path=_rehearsal(tmp_path / "rehearsal.json"),
        db_path=db_path,
        output_dir=output_dir,
        write_labels_approved=True,
    )

    assert packet["status"] == "PREIMAGE_MISMATCH"
    assert packet["failures"] == ["AP_K_2025-07-08_3:box:1:finish_position_changed"]
    assert Path(packet["backup"]["path"]).exists()
    assert packet["writes_performed"]["db_write"] is False
    with sqlite3.connect(db_path) as conn:
        source_rows = conn.execute(
            """
            SELECT box_number, finish_position, placing, scraped_finish_position, data_source
            FROM dog_race_data
            WHERE race_id = 'AP_K_2025-07-08_3'
            ORDER BY box_number
            """
        ).fetchall()
        metadata = conn.execute(
            """
            SELECT results_status, winner_name, winner_source
            FROM race_metadata
            WHERE race_id = 'AP_K_2025-07-08_3'
            """
        ).fetchone()

    assert source_rows == [
        (1, 1, None, None, None),
        (2, 6, None, None, None),
        (4, 6, None, None, None),
        (5, 2, None, None, None),
        (7, 5, None, None, None),
        (8, 6, None, None, None),
    ]
    assert metadata == ("pending", "", None)


def test_update_lane_reports_post_apply_quick_check_failure(tmp_path, monkeypatch):
    db_path = _make_db(tmp_path)
    real_quick_check = apply_lane._quick_check
    call_count = 0

    def fake_quick_check(conn):
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            return "quick_check_failed_after_apply"
        return real_quick_check(conn)

    monkeypatch.setattr(apply_lane, "_quick_check", fake_quick_check)

    packet = apply_update_lane(
        rehearsal_packet_path=_rehearsal(tmp_path / "rehearsal.json"),
        db_path=db_path,
        output_dir=tmp_path / "apply",
        write_labels_approved=True,
    )

    assert packet["status"] == "APPLIED_WITH_POST_QUICK_CHECK_FAILURE"
    assert packet["failures"] == ["source_quick_check_failed_after_apply"]
    assert packet["writes_performed"]["db_write"] is True
    assert packet["source_quick_check_after"] == "quick_check_failed_after_apply"
