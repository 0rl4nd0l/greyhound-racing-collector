import sqlite3

from ingestion.staging_writer import RaceMeta
from scripts.ingest_csv_history import ensure_staging_tables, upsert_race_metadata


def test_upsert_race_metadata_persists_safe_weather_track_fields(tmp_path):
    db_path = tmp_path / "staging.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE race_metadata (
                race_id TEXT PRIMARY KEY,
                venue TEXT,
                race_number INTEGER,
                race_date TEXT,
                race_name TEXT,
                grade TEXT,
                distance TEXT,
                track_condition TEXT,
                weather TEXT,
                field_size INTEGER,
                extraction_timestamp TEXT,
                data_source TEXT
            )
            """
        )
        ensure_staging_tables(conn)

        meta = RaceMeta(
            race_date="2026-06-01",
            venue="TEST",
            race_number=1,
            grade="Grade 5",
            distance="350m",
            track_condition="Slow",
            weather="Showers",
        )
        upsert_race_metadata(conn, meta, field_size=8)

        staging_row = conn.execute(
            """
            SELECT track_condition, weather
            FROM csv_race_metadata_staging
            WHERE race_id = ?
            """,
            (meta.race_id,),
        ).fetchone()
        canonical_row = conn.execute(
            """
            SELECT track_condition, weather
            FROM race_metadata
            WHERE race_id = ?
            """,
            (meta.race_id,),
        ).fetchone()
    finally:
        conn.close()

    assert staging_row == ("Slow", "Showers")
    assert canonical_row == ("Slow", "Showers")
