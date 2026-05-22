#!/usr/bin/env python3
import logging
import os
import sqlite3
from pathlib import Path

# Helper utilities for opening SQLite DBs consistently across the project.
# - Read-only analytics connections use URI mode=ro and PRAGMA query_only=ON
# - Write connections use regular mode and are expected to target STAGING_DB_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def _resolve_db_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def get_analytics_db_path(default: str = "greyhound_racing_data.db") -> str:
    """Return the first configured analytics DB that exists.

    App startup loads local .env files, and stale absolute paths from another
    host otherwise break prediction reads even when the repo-local DB exists.
    Analytics connections are read-only, so a missing configured path is not a
    usable target.
    """
    for env_name in ("ANALYTICS_DB_PATH", "GREYHOUND_DB_PATH", "DATABASE_PATH"):
        value = os.getenv(env_name)
        if not value:
            continue

        path = _resolve_db_path(value)
        if path.is_file():
            return str(path)

        logger.warning(
            "Ignoring %s=%s because analytics database was not found at %s",
            env_name,
            value,
            path,
        )

    return str(_resolve_db_path(default))


def get_staging_db_path(default: str = "greyhound_racing_data_stage.db") -> str:
    return str(_resolve_db_path(os.getenv("STAGING_DB_PATH") or default))


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
