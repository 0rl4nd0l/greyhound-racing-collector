import os
import sqlite3
import tempfile
import shutil

# Ensure app starts in testing mode to avoid watchers and heavy init
os.environ["TESTING"] = "1"

import pytest

# Import the Flask app and module-level DATABASE_PATH
import app as app_module
from app import app


@pytest.fixture(scope="module")
def temp_db_path():
    d = tempfile.mkdtemp(prefix="gh_db_test_")
    path = os.path.join(d, "test.sqlite")
    # Initialize empty sqlite database
    conn = sqlite3.connect(path)
    # Set a pragma to write something so the file exists
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.commit()
    conn.close()
    try:
        yield path
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(autouse=True)
def configure_app_db(temp_db_path, monkeypatch):
    # Ensure Flask app is in testing mode
    app.config["TESTING"] = True
    # Override module-level and config DB paths so all endpoints use our temp DB
    monkeypatch.setattr(app_module, "DATABASE_PATH", temp_db_path, raising=False)
    app.config["DATABASE_PATH"] = temp_db_path
    yield


def test_ws_db_endpoint():
    with app.test_client() as client:
        resp = client.get("/ws/db")
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload is not None
        assert payload.get("type") == "db_metrics"
        assert payload.get("status") in ("ok", "error")
        assert "stats" in payload
        for k in ("total_tables", "total_races", "total_dogs", "database_size_bytes"):
            assert k in payload["stats"]


def test_sse_events_endpoint():
    with app.test_client() as client:
        resp = client.get("/api/db/events", headers={"Accept": "text/event-stream"})
        assert resp.status_code == 200
        ctype = resp.headers.get("Content-Type", "")
        assert "text/event-stream" in ctype
        data = resp.data.decode("utf-8", errors="ignore")
        assert "heartbeat" in data


def test_database_stats_endpoint():
    with app.test_client() as client:
        resp = client.get("/api/database/stats")
        assert resp.status_code == 200
        j = resp.get_json()
        assert j.get("success") is True
        for key in ("total_tables", "total_races", "total_dogs", "database_size"):
            assert key in j


def test_slow_queries_empty_then_populated(temp_db_path):
    with app.test_client() as client:
        # Initially table likely does not exist
        r1 = client.get("/api/database/queries/slow")
        assert r1.status_code == 200
        j1 = r1.get_json()
        assert j1.get("success") is True
        # enabled may be False if table doesn't exist yet
        assert j1.get("count", 0) >= 0

        # Enable explain/analyze to ensure table exists
        r2 = client.get("/api/enable-explain-analyze")
        assert r2.status_code == 200
        j2 = r2.get_json()
        assert j2.get("success") is True

        # Insert some slow queries directly
        conn = sqlite3.connect(temp_db_path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO query_monitoring (query, execution_time, query_plan) VALUES (?, ?, ?)",
            ("SELECT 1", 123.45, "PLAN"),
        )
        cur.execute(
            "INSERT INTO query_monitoring (query, execution_time, query_plan) VALUES (?, ?, ?)",
            ("SELECT 2", 12.34, "PLAN"),
        )
        conn.commit()
        conn.close()

        # Query with min_ms filter so only the first row returns
        r3 = client.get("/api/database/queries/slow?min_ms=50&limit=10")
        assert r3.status_code == 200
        j3 = r3.get_json()
        assert j3.get("success") is True
        assert j3.get("enabled") is True
        assert j3.get("count", 0) >= 1
        queries = j3.get("queries", [])
        assert any(q.get("query") == "SELECT 1" for q in queries)
        for q in queries:
            if q.get("execution_time") is not None:
                assert q["execution_time"] >= 50


def test_database_tables_counts(temp_db_path):
    # Create two user tables and populate them
    conn = sqlite3.connect(temp_db_path)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS race_metadata (race_id TEXT PRIMARY KEY, venue TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS dog_race_data (id INTEGER PRIMARY KEY AUTOINCREMENT, dog_name TEXT)")
    cur.execute("INSERT INTO race_metadata (race_id, venue) VALUES (?, ?)", ("RID1", "TESTV"))
    cur.execute("INSERT INTO race_metadata (race_id, venue) VALUES (?, ?)", ("RID2", "TESTV"))
    cur.execute("INSERT INTO dog_race_data (dog_name) VALUES ('A')")
    conn.commit()
    conn.close()

    with app.test_client() as client:
        resp = client.get("/api/database/tables")
        assert resp.status_code == 200
        j = resp.get_json()
        assert j.get("success") is True
        names = {t["name"]: t for t in j.get("tables", [])}
        assert "race_metadata" in names
        assert "dog_race_data" in names
        assert names["race_metadata"]["row_count"] == 2
        assert names["dog_race_data"]["row_count"] == 1


def test_overview_endpoint():
    with app.test_client() as client:
        resp = client.get("/api/database/overview")
        assert resp.status_code == 200
        j = resp.get_json()
        assert j.get("success") is True
        assert isinstance(j.get("health_score"), int)
        for k in ("queries_per_second", "avg_latency", "active_connections"):
            assert k in j


def test_sse_events_includes_metric():
    with app.test_client() as client:
        resp = client.get("/api/db/events", headers={"Accept": "text/event-stream"})
        assert resp.status_code == 200
        data = resp.data.decode("utf-8", errors="ignore")
        # Both heartbeat and metric events should be present in our minimal SSE
        assert "heartbeat" in data
        assert "metric" in data

