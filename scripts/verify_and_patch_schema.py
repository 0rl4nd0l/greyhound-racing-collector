#!/usr/bin/env python3
"""
Verify and patch the SQLite schema so all expected tables/columns exist for the app.
- Archive-first: creates a pre-patch backup under archive/db_schema_patches/<TS>/
- Non-destructive: only adds missing tables/columns and ensures indexes exist
- Minimal types chosen to match SQLite and the code expectations

Usage:
  python scripts/verify_and_patch_schema.py [--db /path/to.db]

DB resolution order:
  --db arg > $GREYHOUND_DB_PATH > $DATABASE_PATH > ./greyhound_racing_data.db
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.db_utils import open_sqlite_readonly, open_sqlite_writable

# -------------------------
# Helpers
# -------------------------


def resolve_db_path(cli_db: str | None) -> Path:
    if cli_db:
        return Path(cli_db).resolve()
    # Schema patching is a write operation, prefer staging DB
    env = (
        os.getenv("STAGING_DB_PATH")
        or os.getenv("GREYHOUND_DB_PATH")
        or os.getenv("DATABASE_PATH")
    )
    if env:
        return Path(env).expanduser().resolve()
    return Path("./greyhound_racing_data_stage.db").resolve()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def backup_db(db_path: Path, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    backup_path = out_dir / "pre_patch.sqlite"
    # Try sqlite online backup; fallback to cp
    try:
        import subprocess

        subprocess.run(
            ["sqlite3", str(db_path), ".backup", str(backup_path)], check=False
        )
        if backup_path.exists() and backup_path.stat().st_size > 0:
            return backup_path
    except Exception:
        pass
    # Fallback file copy
    import shutil

    shutil.copyfile(db_path, backup_path)
    return backup_path


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def get_columns(conn: sqlite3.Connection, table: str) -> Dict[str, str]:
    cols: Dict[str, str] = {}
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        for row in cur.fetchall():
            # row: cid, name, type, notnull, dflt_value, pk
            name = str(row[1])
            ctype = str(row[2] or "")
            cols[name] = ctype.upper()
    except Exception:
        pass
    return cols


def index_exists(conn: sqlite3.Connection, table: str, index_name: str) -> bool:
    try:
        cur = conn.execute(f"PRAGMA index_list({table})")
        for row in cur.fetchall():
            # seq, name, unique, origin, partial
            if str(row[1]) == index_name:
                return True
    except Exception:
        pass
    return False


def has_duplicates(conn: sqlite3.Connection, table: str, col: str) -> bool:
    try:
        cur = conn.execute(
            f"SELECT {col}, COUNT(*) FROM {table} GROUP BY {col} HAVING COUNT(*)>1 LIMIT 1"
        )
        return cur.fetchone() is not None
    except Exception:
        return True


def has_composite_duplicates(
    conn: sqlite3.Connection, table: str, cols: List[str]
) -> bool:
    """Return True if duplicates exist for the composite key defined by cols."""
    try:
        col_list = ", ".join(cols)
        cur = conn.execute(
            f"SELECT {col_list}, COUNT(*) FROM {table} GROUP BY {col_list} HAVING COUNT(*)>1 LIMIT 1"
        )
        return cur.fetchone() is not None
    except Exception:
        return True


# -------------------------
# Expected schema definition
# -------------------------
# SQLite-friendly types
T_INT = "INTEGER"
T_REAL = "REAL"
T_TEXT = "TEXT"
T_DT = "DATETIME"
T_NUM = "NUMERIC"

EXPECTED: Dict[str, Dict[str, str]] = {
    # Core tables
    "race_metadata": {
        "id": T_INT,
        "race_id": T_TEXT,
        "venue": T_TEXT,
        "race_number": T_INT,
        "race_date": T_TEXT,
        "race_name": T_TEXT,
        "grade": T_TEXT,
        "distance": T_TEXT,
        "track_condition": T_TEXT,
        "weather": T_TEXT,
        "temperature": T_REAL,
        "humidity": T_REAL,
        "wind_speed": T_REAL,
        "wind_direction": T_TEXT,
        "track_record": T_TEXT,
        "prize_money_total": T_REAL,
        "prize_money_breakdown": T_TEXT,
        "race_time": T_TEXT,
        "field_size": T_INT,
        "url": T_TEXT,
        "extraction_timestamp": T_DT,
        "data_source": T_TEXT,
        "winner_name": T_TEXT,
        "winner_odds": T_REAL,
        "winner_margin": T_REAL,
        "race_status": T_TEXT,
        "data_quality_note": T_TEXT,
        "actual_field_size": T_INT,
        "scratched_count": T_INT,
        "scratch_rate": T_REAL,
        "box_analysis": T_TEXT,
        "weather_condition": T_TEXT,
        "precipitation": T_REAL,
        "pressure": T_REAL,
        "visibility": T_REAL,
        "weather_location": T_TEXT,
        "weather_timestamp": T_DT,
        "weather_adjustment_factor": T_REAL,
        "sportsbet_url": T_TEXT,
        "venue_slug": T_TEXT,
        "start_datetime": T_DT,
    },
    "dog_race_data": {
        "id": T_INT,
        "race_id": T_TEXT,
        "dog_name": T_TEXT,
        "dog_clean_name": T_TEXT,
        "dog_id": T_INT,
        "box_number": T_INT,
        "trainer_name": T_TEXT,
        "trainer_id": T_INT,
        "weight": T_REAL,
        "running_style": T_TEXT,
        "odds_decimal": T_REAL,
        "odds_fractional": T_TEXT,
        "starting_price": T_REAL,
        "individual_time": T_TEXT,
        "sectional_1st": T_TEXT,
        "sectional_2nd": T_TEXT,
        "sectional_3rd": T_TEXT,
        "margin": T_TEXT,
        "beaten_margin": T_REAL,
        "was_scratched": T_INT,
        "blackbook_link": T_TEXT,
        "extraction_timestamp": T_DT,
        "data_source": T_TEXT,
        "form_guide_json": T_TEXT,
        "historical_records": T_TEXT,
        "performance_rating": T_REAL,
        "speed_rating": T_REAL,
        "class_rating": T_REAL,
        "recent_form": T_TEXT,
        "win_probability": T_REAL,
        "place_probability": T_REAL,
        "scraped_trainer_name": T_TEXT,
        "scraped_reaction_time": T_TEXT,
        "scraped_nbtt": T_TEXT,
        "scraped_race_classification": T_TEXT,
        "scraped_raw_result": T_TEXT,
        "scraped_finish_position": T_TEXT,
        "best_time": T_REAL,
        "data_quality_note": T_TEXT,
        "finish_position": T_INT,
        "odds": T_TEXT,
        "trainer": T_TEXT,
        "winning_time": T_TEXT,
        "placing": T_INT,
        "form": T_TEXT,
    },
    "dogs": {
        "dog_id": T_INT,
        "dog_name": T_TEXT,
        "total_races": T_INT,
        "total_wins": T_INT,
        "total_places": T_INT,
        "best_time": T_REAL,
        "average_position": T_REAL,
        "last_race_date": T_TEXT,
        "created_at": T_DT,
        "weight": T_NUM,
        "age": T_INT,
        "id": T_INT,
        "color": T_TEXT,
        "owner": T_TEXT,
        "trainer": T_TEXT,
        "sex": T_TEXT,
    },
    # Frequently used auxiliaries
    "enhanced_expert_data": {
        "id": T_INT,
        "race_id": T_TEXT,
        "dog_clean_name": T_TEXT,
        "pir_rating": T_REAL,
        "first_sectional": T_TEXT,
        "win_time": T_TEXT,
        "bonus_time": T_TEXT,
    },
    "prediction_history": {
        "id": T_INT,
        "race_id": T_TEXT,
        "model_name": T_TEXT,
        "model_version": T_TEXT,
        "prediction_data": T_TEXT,
        "confidence_score": T_REAL,
        "actual_results": T_TEXT,
        "accuracy_score": T_REAL,
        "created_at": T_DT,
        "updated_at": T_DT,
    },
    "ml_model_registry": {
        "id": T_INT,
        "model_name": T_TEXT,
        "model_version": T_TEXT,
        "model_type": T_TEXT,
        "file_path": T_TEXT,
        "metrics": T_TEXT,
        "parameters": T_TEXT,
        "training_data_hash": T_TEXT,
        "is_active": T_INT,
        "created_at": T_DT,
        "updated_at": T_DT,
    },
    "processed_race_files": {
        "id": T_INT,
        "file_hash": T_TEXT,
        "race_date": T_TEXT,
        "venue": T_TEXT,
        "race_no": T_INT,
        "file_path": T_TEXT,
        "file_size": T_INT,
        "processed_at": T_DT,
        "status": T_TEXT,
        "error_message": T_TEXT,
    },
    "db_meta": {
        "id": T_INT,
        "meta_key": T_TEXT,
        "meta_value": T_TEXT,
        "last_updated": T_DT,
    },
    # Odds system
    "live_odds": {
        "id": T_INT,
        "race_id": T_TEXT,
        "venue": T_TEXT,
        "race_number": T_INT,
        "race_date": T_TEXT,
        "race_time": T_TEXT,
        "dog_name": T_TEXT,
        "dog_clean_name": T_TEXT,
        "box_number": T_INT,
        "odds_decimal": T_REAL,
        "odds_fractional": T_TEXT,
        "market_type": T_TEXT,
        "source": T_TEXT,
        "timestamp": T_DT,
        "is_current": T_INT,
    },
    "odds_history": {
        "id": T_INT,
        "race_id": T_TEXT,
        "dog_clean_name": T_TEXT,
        "odds_decimal": T_REAL,
        "odds_change": T_REAL,
        "timestamp": T_DT,
        "source": T_TEXT,
    },
    "value_bets": {
        "id": T_INT,
        "race_id": T_TEXT,
        "dog_clean_name": T_TEXT,
        "predicted_probability": T_REAL,
        "market_odds": T_REAL,
        "implied_probability": T_REAL,
        "value_percentage": T_REAL,
        "confidence_level": T_TEXT,
        "bet_recommendation": T_TEXT,
        "timestamp": T_DT,
    },
    "predictions": {
        "id": T_INT,
        "race_id": T_TEXT,
        "dog_clean_name": T_TEXT,
        "predicted_probability": T_REAL,
        "confidence_level": T_TEXT,
        "prediction_source": T_TEXT,
        "timestamp": T_DT,
    },
    "race_analytics": {
        "id": T_INT,
        "race_id": T_TEXT,
    },
    "gpt_analysis": {
        "id": T_INT,
        "race_id": T_TEXT,
        "analysis_type": T_TEXT,
        "analysis_data": T_TEXT,
    },
}

# Index definitions: (table, index_name, columns, unique?)
INDEXES: List[Tuple[str, str, List[str], bool]] = [
    ("race_metadata", "idx_race_metadata_venue_date", ["venue", "race_date"], False),
    (
        "race_metadata",
        "idx_race_metadata_race_id",
        ["race_id"],
        False,
    ),  # upgraded to unique if possible
    ("dog_race_data", "idx_dog_race_data_race_id", ["race_id"], False),
    ("dog_race_data", "idx_dog_race_data_dog_name", ["dog_clean_name"], False),
    ("dog_race_data", "idx_dog_race_data_finish_position", ["finish_position"], False),
    ("dogs", "idx_dogs_clean_name", ["dog_name"], False),
    ("dogs", "idx_dogs_trainer", ["trainer"], False),
    ("prediction_history", "idx_prediction_history_race_id", ["race_id"], False),
    (
        "prediction_history",
        "idx_prediction_history_model",
        ["model_name", "model_version"],
        False,
    ),
    ("race_analytics", "idx_race_analytics_race_id", ["race_id"], False),
    ("db_meta", "idx_db_meta_key", ["meta_key"], False),
    ("processed_race_files", "idx_processed_files_hash", ["file_hash"], False),
    (
        "processed_race_files",
        "idx_processed_files_race_key",
        ["race_date", "venue", "race_no"],
        False,
    ),
    (
        "processed_race_files",
        "idx_processed_files_processed_at",
        ["processed_at"],
        False,
    ),
    # Added per schema tests: missing FK indexes
    (
        "dog_performance_ft_extra",
        "idx_dog_performance_ft_extra_performance_id",
        ["performance_id"],
        False,
    ),
    ("dog_race_data_backup", "idx_dog_race_data_backup_race_id", ["race_id"], False),
    ("dogs_ft_extra", "idx_dogs_ft_extra_dog_id", ["dog_id"], False),
    ("expert_form_analysis", "idx_expert_form_analysis_race_id", ["race_id"], False),
    ("races_ft_extra", "idx_races_ft_extra_race_id", ["race_id"], False),
]


# -------------------------
# Creation DDLs for missing tables (minimal)
# -------------------------
CREATE_TABLE_SQL: Dict[str, str] = {
    "race_metadata": (
        """
        CREATE TABLE IF NOT EXISTS race_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT UNIQUE,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            race_name TEXT,
            grade TEXT,
            distance TEXT,
            track_condition TEXT,
            weather TEXT,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            wind_direction TEXT,
            track_record TEXT,
            prize_money_total REAL,
            prize_money_breakdown TEXT,
            race_time TEXT,
            field_size INTEGER,
            url TEXT,
            extraction_timestamp DATETIME,
            data_source TEXT,
            winner_name TEXT,
            winner_odds REAL,
            winner_margin REAL,
            race_status TEXT,
            data_quality_note TEXT,
            actual_field_size INTEGER,
            scratched_count INTEGER,
            scratch_rate REAL,
            box_analysis TEXT,
            weather_condition TEXT,
            precipitation REAL,
            pressure REAL,
            visibility REAL,
            weather_location TEXT,
            weather_timestamp DATETIME,
            weather_adjustment_factor REAL,
            sportsbet_url TEXT,
            venue_slug TEXT,
            start_datetime DATETIME
        )
        """
    ),
    "dog_race_data": (
        """
        CREATE TABLE IF NOT EXISTS dog_race_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            dog_id INTEGER,
            box_number INTEGER,
            trainer_name TEXT,
            trainer_id INTEGER,
            weight REAL,
            running_style TEXT,
            odds_decimal REAL,
            odds_fractional TEXT,
            starting_price REAL,
            individual_time TEXT,
            sectional_1st TEXT,
            sectional_2nd TEXT,
            sectional_3rd TEXT,
            margin TEXT,
            beaten_margin REAL,
            was_scratched INTEGER,
            blackbook_link TEXT,
            extraction_timestamp DATETIME,
            data_source TEXT,
            form_guide_json TEXT,
            historical_records TEXT,
            performance_rating REAL,
            speed_rating REAL,
            class_rating REAL,
            recent_form TEXT,
            win_probability REAL,
            place_probability REAL,
            scraped_trainer_name TEXT,
            scraped_reaction_time TEXT,
            scraped_nbtt TEXT,
            scraped_race_classification TEXT,
            scraped_raw_result TEXT,
            scraped_finish_position TEXT,
            best_time REAL,
            data_quality_note TEXT,
            finish_position INTEGER,
            odds TEXT,
            trainer TEXT,
            winning_time TEXT,
            placing INTEGER,
            form TEXT
        )
        """
    ),
    "dogs": (
        """
        CREATE TABLE IF NOT EXISTS dogs (
            dog_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dog_name TEXT UNIQUE NOT NULL,
            total_races INTEGER,
            total_wins INTEGER,
            total_places INTEGER,
            best_time REAL,
            average_position REAL,
            last_race_date TEXT,
            created_at DATETIME,
            weight NUMERIC,
            age INTEGER,
            id INTEGER,
            color TEXT,
            owner TEXT,
            trainer TEXT,
            sex TEXT
        )
        """
    ),
    "enhanced_expert_data": (
        """
        CREATE TABLE IF NOT EXISTS enhanced_expert_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_clean_name TEXT,
            pir_rating REAL,
            first_sectional TEXT,
            win_time TEXT,
            bonus_time TEXT
        )
        """
    ),
    "prediction_history": (
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            model_name TEXT,
            model_version TEXT,
            prediction_data TEXT,
            confidence_score REAL,
            actual_results TEXT,
            accuracy_score REAL,
            created_at DATETIME,
            updated_at DATETIME
        )
        """
    ),
    "ml_model_registry": (
        """
        CREATE TABLE IF NOT EXISTS ml_model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT,
            model_version TEXT,
            model_type TEXT,
            file_path TEXT,
            metrics TEXT,
            parameters TEXT,
            training_data_hash TEXT,
            is_active INTEGER,
            created_at DATETIME,
            updated_at DATETIME
        )
        """
    ),
    "processed_race_files": (
        """
        CREATE TABLE IF NOT EXISTS processed_race_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_hash TEXT,
            race_date TEXT,
            venue TEXT,
            race_no INTEGER,
            file_path TEXT,
            file_size INTEGER,
            processed_at DATETIME,
            status TEXT,
            error_message TEXT
        )
        """
    ),
    "db_meta": (
        """
        CREATE TABLE IF NOT EXISTS db_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meta_key TEXT UNIQUE,
            meta_value TEXT,
            last_updated DATETIME
        )
        """
    ),
    "live_odds": (
        """
        CREATE TABLE IF NOT EXISTS live_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            race_time TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            odds_decimal REAL,
            odds_fractional TEXT,
            market_type TEXT,
            source TEXT,
            timestamp DATETIME,
            is_current INTEGER
        )
        """
    ),
    "odds_history": (
        """
        CREATE TABLE IF NOT EXISTS odds_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_clean_name TEXT,
            odds_decimal REAL,
            odds_change REAL,
            timestamp DATETIME,
            source TEXT
        )
        """
    ),
    "value_bets": (
        """
        CREATE TABLE IF NOT EXISTS value_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_clean_name TEXT,
            predicted_probability REAL,
            market_odds REAL,
            implied_probability REAL,
            value_percentage REAL,
            confidence_level TEXT,
            bet_recommendation TEXT,
            timestamp DATETIME
        )
        """
    ),
    "predictions": (
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_clean_name TEXT,
            predicted_probability REAL,
            confidence_level TEXT,
            prediction_source TEXT,
            timestamp DATETIME
        )
        """
    ),
    "race_analytics": (
        """
        CREATE TABLE IF NOT EXISTS race_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT
        )
        """
    ),
    "gpt_analysis": (
        """
        CREATE TABLE IF NOT EXISTS gpt_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            analysis_type TEXT,
            analysis_data TEXT
        )
        """
    ),
}


# -------------------------
# Main logic
# -------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify and patch SQLite schema")
    ap.add_argument("--db", help="Path to SQLite DB", default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Apply changes even if DB appears populated",
    )
    args = ap.parse_args()

    db_path = resolve_db_path(args.db)
    if not db_path.exists():
        print(f"[schema] DB not found at {db_path}")
        return 2

    # Lockfile protection: if a .db.lock exists next to the DB, require --force
    lockfile = db_path.with_suffix(db_path.suffix + ".lock")
    if (
        lockfile.exists()
        and not args.force
        and os.getenv("FORCE", "0") not in ("1", "true", "TRUE")
    ):
        print(
            f"[schema] Lockfile present: {lockfile}. Refusing to patch without --force."
        )
        return 3

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    archive_dir = Path("archive") / "db_schema_patches" / ts
    ensure_dir(archive_dir)

    # Backup
    backup = backup_db(db_path, archive_dir)
    print(f"[schema] Pre-patch backup created at {backup}")

    conn = open_sqlite_writable(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass

    report_lines: List[str] = []

    # Safety: if DB appears heavily populated, require --force (prevents accidental patch on prod-like DB)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        tbls = int(cur.fetchone()[0] or 0)
    except Exception:
        tbls = 0
    try:
        cur = conn.execute("SELECT COUNT(*) FROM dog_race_data")
        dog_rows = int(cur.fetchone()[0] or 0)
    except Exception:
        dog_rows = 0
    if (
        dog_rows >= 1000
        and not args.force
        and os.getenv("FORCE", "0") not in ("1", "true", "TRUE")
    ):
        print(
            f"[schema] DB has {dog_rows} dog_race_data rows; refusing to patch without --force."
        )
        print(
            f"[schema] Use: python scripts/verify_and_patch_schema.py --db {db_path} --force"
        )
        return 4

    # Ensure tables exist
    for table, ddl in CREATE_TABLE_SQL.items():
        if not table_exists(conn, table):
            try:
                conn.executescript(ddl)
                report_lines.append(f"[create] table {table}")
            except Exception as e:
                report_lines.append(f"[warn] failed to create table {table}: {e}")

    # Ensure columns exist (ALTER TABLE add missing)
    for table, expected_cols in EXPECTED.items():
        existing = get_columns(conn, table)
        if not existing:
            # If still missing (create failed), skip
            continue
        for col, coltype in expected_cols.items():
            if col not in existing:
                try:
                    # Default values only where safe and expected; otherwise NULL
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                    report_lines.append(f"[alter] {table} ADD COLUMN {col} {coltype}")
                    # Special-case defaults
                    if table == "dog_race_data" and col == "was_scratched":
                        conn.execute(
                            "UPDATE dog_race_data SET was_scratched=0 WHERE was_scratched IS NULL"
                        )
                except Exception as e:
                    report_lines.append(
                        f"[warn] failed to add column {table}.{col}: {e}"
                    )

    # Ensure indexes
    for table, idx_name, cols, unique in INDEXES:
        if not table_exists(conn, table):
            continue
        if index_exists(conn, table, idx_name):
            continue
        try:
            col_list = ", ".join(cols)
            stmt = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({col_list})"
            # Attempt unique on race_metadata.race_id if no duplicates
            if table == "race_metadata" and cols == ["race_id"]:
                if not has_duplicates(conn, "race_metadata", "race_id"):
                    stmt = f"CREATE UNIQUE INDEX IF NOT EXISTS uq_race_metadata_race_id ON race_metadata (race_id)"
                    idx_name = "uq_race_metadata_race_id"
            conn.execute(stmt)
            report_lines.append(f"[index] {idx_name} on {table}({', '.join(cols)})")
        except Exception as e:
            report_lines.append(f"[warn] failed to create index {idx_name}: {e}")

    # Staged-ingestion uniqueness: dog_race_data (race_id, dog_clean_name, box_number)
    try:
        if table_exists(conn, "dog_race_data") and not index_exists(
            conn, "dog_race_data", "uq_dog_race_data_rdb"
        ):
            cols = ["race_id", "dog_clean_name", "box_number"]
            if not has_composite_duplicates(conn, "dog_race_data", cols):
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_dog_race_data_rdb ON dog_race_data (race_id, dog_clean_name, box_number)"
                )
                report_lines.append(
                    "[index] uq_dog_race_data_rdb on dog_race_data(race_id, dog_clean_name, box_number)"
                )
    except Exception as e:
        report_lines.append(
            f"[warn] failed to ensure composite unique index on dog_race_data: {e}"
        )

    conn.commit()

    # Write report
    report_path = archive_dir / "patch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) or "[info] No changes required\n")

    print(f"[schema] Patch completed. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
