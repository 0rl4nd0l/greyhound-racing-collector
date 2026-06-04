import json
import sqlite3
from pathlib import Path

from scripts.build_legacy_label_verification_packet import build_packet, main


def _create_db(path: Path) -> Path:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                race_date TEXT,
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
                dog_clean_name TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                placing INTEGER,
                scraped_finish_position TEXT,
                data_source TEXT
            )
            """
        )
    return path


def _insert_race(
    db_path: Path,
    race_id: str,
    *,
    rows: list[tuple[int, str, int | None]],
    data_source: str | None,
    results_status: str = "complete",
    winner_source: str | None = None,
) -> None:
    winner_name = next((name for _, name, pos in rows if pos == 1), None)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO race_metadata
                (race_id, race_date, results_status, winner_name, winner_source)
            VALUES (?, '2026-01-01', ?, ?, ?)
            """,
            (race_id, results_status, winner_name, winner_source),
        )
        for box, name, finish_position in rows:
            conn.execute(
                """
                INSERT INTO dog_race_data (
                    race_id, dog_name, dog_clean_name, box_number,
                    finish_position, placing, scraped_finish_position, data_source
                )
                VALUES (?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (race_id, name, name.upper(), box, finish_position, data_source),
            )


def test_legacy_label_verification_classifies_sources_and_exact_official_matches(tmp_path):
    legacy_db = _create_db(tmp_path / "legacy.sqlite")
    official_db = _create_db(tmp_path / "official.sqlite")

    full_rows = [
        (1, "Alpha", 1),
        (2, "Bravo", 2),
        (3, "Charlie", 3),
        (4, "Delta", 4),
    ]
    _insert_race(
        legacy_db,
        "Race 1 - TEST - 2026-01-01",
        rows=full_rows,
        data_source="enhanced_processor_with_results",
    )
    _insert_race(
        official_db,
        "Race 1 - TEST - 2026-01-01",
        rows=full_rows,
        data_source="thedogs_official",
        results_status="resulted",
        winner_source="thedogs_official",
    )

    _insert_race(
        legacy_db,
        "Race 2 - TEST - 2026-01-01",
        rows=[
            (1, "Echo", 2),
            (2, "Foxtrot", 1),
            (3, "Golf", 3),
            (4, "Hotel", 4),
        ],
        data_source="navigator_results",
    )
    _insert_race(
        official_db,
        "Race 2 - TEST - 2026-01-01",
        rows=[
            (1, "Echo", 1),
            (2, "Foxtrot", 2),
            (3, "Golf", 3),
            (4, "Hotel", 4),
        ],
        data_source="thedogs_official",
        results_status="resulted",
        winner_source="thedogs_official",
    )

    _insert_race(
        legacy_db,
        "Race 3 - TEST - 2026-01-01",
        rows=[(1, "India", 1), (2, "Juliet", 2)],
        data_source="embedded_form_guide",
    )
    _insert_race(
        legacy_db,
        "Race 4 - TEST - 2026-01-01",
        rows=[(1, "Kilo", 1), (2, "Lima", 2)],
        data_source=None,
    )
    _insert_race(
        legacy_db,
        "Race 5 - TEST - 2026-01-01",
        rows=[(1, "Mike", 1), (2, "November", 2), (3, "Oscar", 3), (4, "Papa", 4)],
        data_source="sportsbet_results_top4",
        results_status="partial_sportsbet_results",
        winner_source="sportsbet_results_top4",
    )

    packet = build_packet(
        legacy_db_paths=[legacy_db],
        official_db_path=official_db,
    )

    assert packet["schema_version"] == "legacy_label_verification_packet_v1"
    assert packet["status"] == "REPORT_ONLY"
    assert packet["writes_performed"] == {
        "db_write": False,
        "label_promotion": False,
        "snapshot_mutation": False,
        "model_training": False,
        "registry_mutation": False,
        "scrape_or_fetch": False,
    }
    assert packet["summary"]["races_scanned"] == 5
    assert packet["summary"]["classification_counts"] == {
        "embedded_history_only": 1,
        "legacy_unknown_provenance": 1,
        "official_mismatch": 1,
        "partial_or_winner_only": 1,
        "verified_official_candidate": 1,
    }
    assert packet["summary"]["source_counts"] == {
        "NULL": 1,
        "embedded_form_guide": 1,
        "enhanced_processor_with_results": 1,
        "navigator_results": 1,
        "sportsbet_results_top4": 1,
    }
    assert packet["summary"]["verified_official_candidates"] == {
        "races": 1,
        "runner_rows": 4,
    }

    by_race = {item["race_id"]: item for item in packet["race_classifications"]}
    assert by_race["Race 1 - TEST - 2026-01-01"]["verification"] == {
        "status": "MATCH",
        "official_reference_rows": 4,
        "legacy_rows": 4,
        "mismatches": [],
    }
    assert by_race["Race 2 - TEST - 2026-01-01"]["verification"]["status"] == "MISMATCH"
    assert by_race["Race 3 - TEST - 2026-01-01"]["reason"] == "embedded_form_guide_not_result_label"
    assert by_race["Race 4 - TEST - 2026-01-01"]["reason"] == "legacy_null_source_requires_reverification"
    assert by_race["Race 5 - TEST - 2026-01-01"]["reason"] == "partial_or_winner_only_source"


def test_legacy_label_verification_fails_closed_without_official_reference(tmp_path):
    legacy_db = _create_db(tmp_path / "legacy.sqlite")
    _insert_race(
        legacy_db,
        "Race 1 - TEST - 2026-01-01",
        rows=[(1, "Alpha", 1), (2, "Bravo", 2)],
        data_source="enhanced_processor_with_results",
    )

    packet = build_packet(legacy_db_paths=[legacy_db], official_db_path=None)

    assert packet["status"] == "REPORT_ONLY"
    assert packet["summary"]["classification_counts"] == {
        "result_like_reverify_candidate": 1
    }
    assert packet["race_classifications"][0]["verification"] == {
        "status": "NO_OFFICIAL_REFERENCE_DB",
        "official_reference_rows": 0,
        "legacy_rows": 2,
        "mismatches": [],
    }
    assert packet["recommended_next_actions"][0].startswith("Do not promote")


def test_legacy_label_verification_requires_explicit_official_source_column(tmp_path):
    legacy_db = _create_db(tmp_path / "legacy.sqlite")
    official_db = tmp_path / "official_without_source.sqlite"
    _insert_race(
        legacy_db,
        "Race 1 - TEST - 2026-01-01",
        rows=[(1, "Alpha", 1), (2, "Bravo", 2)],
        data_source="enhanced_processor_with_results",
    )
    with sqlite3.connect(official_db) as conn:
        conn.execute(
            """
            CREATE TABLE dog_race_data (
                race_id TEXT,
                dog_name TEXT,
                box_number INTEGER,
                finish_position INTEGER
            )
            """
        )
        conn.executemany(
            """
            INSERT INTO dog_race_data
                (race_id, dog_name, box_number, finish_position)
            VALUES ('Race 1 - TEST - 2026-01-01', ?, ?, ?)
            """,
            [("Alpha", 1, 1), ("Bravo", 2, 2)],
        )

    packet = build_packet(
        legacy_db_paths=[legacy_db],
        official_db_path=official_db,
    )

    assert packet["summary"]["classification_counts"] == {
        "result_like_reverify_candidate": 1
    }
    assert packet["race_classifications"][0]["verification"]["status"] == (
        "OFFICIAL_REFERENCE_MISSING"
    )


def test_legacy_label_verification_cli_writes_report_only_packet(tmp_path):
    legacy_db = _create_db(tmp_path / "legacy.sqlite")
    _insert_race(
        legacy_db,
        "Race 1 - TEST - 2026-01-01",
        rows=[(1, "Alpha", 1), (2, "Bravo", 2)],
        data_source=None,
    )
    output = tmp_path / "packet.json"

    exit_code = main(
        [
            "--legacy-db",
            str(legacy_db),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "REPORT_ONLY"
    assert payload["summary"]["classification_counts"] == {
        "legacy_unknown_provenance": 1
    }
    assert payload["read_only_safety"]["sqlite_mode"] == "mode=ro + PRAGMA query_only=ON"
