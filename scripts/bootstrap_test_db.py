#!/usr/bin/env python3
"""
Archive-first bootstrap for a clean development database.

- Archives existing greyhound_racing_data.db to archive/databases/ with a timestamp
- Recreates schema exactly from models.py (SQLAlchemy Base.metadata.create_all)
- Attempts to stamp Alembic to head (non-fatal if Alembic is unavailable)

This script is safe to run repeatedly and respects the project's archive-first policy.
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repo root is importable so `models.py` can be found when running this script directly
import sys  # noqa: E402
sys.path.insert(0, str(REPO_ROOT))  # noqa: E402

# Import models Base (canonical schema)
from models import Base  # noqa: E402

DB_DEFAULT_FILE = REPO_ROOT / "greyhound_racing_data.db"
ARCHIVE_DIR = REPO_ROOT / "archive" / "databases"
DB_URL = os.getenv("DATABASE_URL", "sqlite:///greyhound_racing_data.db")


def _db_file_from_url(url: str) -> Path:
    """Resolve the SQLite DB file path from a SQLAlchemy-style URL.

    Supports patterns:
    - sqlite:///relative.db        -> repo-local file
    - sqlite:////absolute/path.db  -> absolute file
    - sqlite:///:memory:           -> non-file; fall back to default repo-local file
    """
    try:
        # In-memory DB -> treat as non-archivable; use default repo-local file path
        if url.startswith("sqlite:///:memory:"):
            return DB_DEFAULT_FILE

        # Absolute path (four slashes)
        if url.startswith("sqlite:////"):
            path_part = url[len("sqlite:////"):]
            # Ensure leading slash for absolute path
            return Path(f"/{path_part}")

        # Relative path (three slashes)
        if url.startswith("sqlite:///"):
            path_part = url[len("sqlite:///"):]
            return (REPO_ROOT / path_part).resolve()
    except Exception:
        pass
    return DB_DEFAULT_FILE


def archive_existing_db(db_file: Path) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if db_file.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = ARCHIVE_DIR / f"{db_file.stem}.{ts}{db_file.suffix}"
        shutil.move(str(db_file), str(dest))
        print(f"Archived existing DB to {dest}")
    else:
        print("No existing DB found to archive (skipping)")


def create_schema(url: str) -> None:
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    print("Created schema from models.py")

    # Optional: Alembic stamp head for state alignment
    try:
        from alembic.config import Config
        from alembic import command

        cfg = Config(str(REPO_ROOT / "alembic.ini"))
        command.stamp(cfg, "head")
        print("Alembic stamped to head")
    except Exception as e:
        print(f"(Non-fatal) Alembic stamp skipped: {e}")


def main() -> int:
    db_file = _db_file_from_url(DB_URL)
    print(f"Using DATABASE_URL={DB_URL}")
    print(f"Resolved DB file: {db_file}")
    archive_existing_db(db_file)
    create_schema(DB_URL)
    print("Bootstrap complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

