#!/usr/bin/env python3
"""
Safe DB repair: ensure dogs(dog_name) index exists and drop stray clean_name index if invalid.

Usage:
  python migrations/add_dogs_index_safety.py [--db /path/to/db.sqlite]

- Detects DB path from GREYHOUND_DB_PATH or argument, else greyhound_racing_data.db in repo root
- Creates idx_dogs_dog_name on dogs(dog_name) if not present
- If 'clean_name' column is absent, drops idx_dogs_clean_name index if it exists
- Prints a summary of actions taken

This script is intentionally independent of Alembic to avoid altering migration history.
Run under your project venv.
"""

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall() or []}
        return column in cols
    except Exception:
        return False


def index_exists(conn: sqlite3.Connection, index_name: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        )
        return cur.fetchone() is not None
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", dest="db", help="Path to SQLite DB")
    args = parser.parse_args()

    db_path = (
        args.db
        or os.getenv("GREYHOUND_DB_PATH")
        or str(Path.cwd() / "greyhound_racing_data.db")
    )
    db_path = str(Path(db_path).resolve())

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        sys.exit(1)

    print(f"🔧 Using database: {db_path}")
    conn = sqlite3.connect(db_path)

    try:
        # Ensure dogs table exists
        if not column_exists(conn, "dogs", "dog_name"):
            print("❌ Table 'dogs' or column 'dog_name' not found — nothing to do")
            sys.exit(2)

        # Create index on dog_name if missing
        created = False
        if not index_exists(conn, "idx_dogs_dog_name"):
            print("➕ Creating index idx_dogs_dog_name ON dogs(dog_name)…")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dogs_dog_name ON dogs(dog_name)")
            created = True
        else:
            print("✅ Index idx_dogs_dog_name already exists")

        # Drop stray clean_name index if clean_name column does not exist
        if not column_exists(conn, "dogs", "clean_name") and index_exists(
            conn, "idx_dogs_clean_name"
        ):
            print("➖ Dropping idx_dogs_clean_name (no clean_name column present)…")
            conn.execute("DROP INDEX IF EXISTS idx_dogs_clean_name")
        else:
            print("ℹ️ No stray idx_dogs_clean_name found or clean_name column exists")

        conn.commit()
        print("\n📋 Summary:
 - idx_dogs_dog_name: {}\n - idx_dogs_clean_name: handled if stray\n".format("created" if created else "already present"))
        print("🎉 Repair complete")
    except Exception as e:
        print(f"💥 Repair failed: {e}")
        sys.exit(3)
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

