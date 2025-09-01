#!/usr/bin/env python3
import os
import sqlite3
from pathlib import Path

# Helper utilities for opening SQLite DBs consistently across the project.
# - Read-only analytics connections use URI mode=ro and PRAGMA query_only=ON
# - Write connections use regular mode and are expected to target STAGING_DB_PATH


def get_analytics_db_path(default: str = "greyhound_racing_data.db") -> str:
    return os.getenv("ANALYTICS_DB_PATH") or os.getenv("GREYHOUND_DB_PATH") or default


def get_staging_db_path(default: str = "greyhound_racing_data_stage.db") -> str:
    return os.getenv("STAGING_DB_PATH") or default


def open_sqlite_readonly(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_analytics_db_path()
    uri = f"file:{str(Path(path).resolve())}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    return conn


def open_sqlite_writable(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or get_staging_db_path()
    conn = sqlite3.connect(str(Path(path).resolve()))
    try:
        conn.execute("PRAGMA foreign_keys=ON")
    except Exception:
        pass
    return conn
