import sqlite3
from pathlib import Path

import pytest

from scripts import db_utils


DB_ENV_VARS = (
    "ANALYTICS_DB_PATH",
    "GREYHOUND_DB_PATH",
    "DATABASE_PATH",
    "STAGING_DB_PATH",
)


@pytest.fixture(autouse=True)
def clear_db_env(monkeypatch):
    for name in DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _touch_sqlite(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS marker (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    return path


def test_analytics_path_falls_back_from_stale_env_to_existing_database_path(
    tmp_path, monkeypatch
):
    local_db = _touch_sqlite(tmp_path / "greyhound_racing_data.db")
    stale_db = tmp_path / "missing-host" / "greyhound_racing_data_writable.db"

    monkeypatch.setenv("ANALYTICS_DB_PATH", str(stale_db))
    monkeypatch.setenv("GREYHOUND_DB_PATH", str(stale_db))
    monkeypatch.setenv("DATABASE_PATH", str(local_db))

    assert Path(db_utils.get_analytics_db_path()).resolve() == local_db.resolve()


def test_analytics_path_prefers_existing_analytics_env(tmp_path, monkeypatch):
    analytics_db = _touch_sqlite(tmp_path / "analytics.db")
    legacy_db = _touch_sqlite(tmp_path / "legacy.db")

    monkeypatch.setenv("ANALYTICS_DB_PATH", str(analytics_db))
    monkeypatch.setenv("GREYHOUND_DB_PATH", str(legacy_db))

    assert Path(db_utils.get_analytics_db_path()).resolve() == analytics_db.resolve()


def test_open_sqlite_readonly_prevents_writes(tmp_path):
    db_path = _touch_sqlite(tmp_path / "readonly.db")

    conn = db_utils.open_sqlite_readonly(str(db_path))
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE blocked (id INTEGER)")
    finally:
        conn.close()


def test_staging_path_resolves_to_project_root(monkeypatch):
    monkeypatch.setenv("STAGING_DB_PATH", "relative_stage.db")

    assert (
        Path(db_utils.get_staging_db_path())
        == db_utils.PROJECT_ROOT / "relative_stage.db"
    )
