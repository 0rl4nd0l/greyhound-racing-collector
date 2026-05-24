import os
import sqlite3
import tempfile
from datetime import date

import pytest


@pytest.fixture(scope="module")
def test_app_client():
    # Use a temporary DB for this test module
    fd, tmp_db = tempfile.mkstemp(prefix="odds_dashboard_", suffix=".db")
    os.close(fd)
    os.environ["DATABASE_PATH"] = tmp_db

    # Ensure schema exists with integrator's schema before importing app
    from sportsbet_odds_integrator import SportsbetOddsIntegrator

    _pre_integrator = SportsbetOddsIntegrator(db_path=tmp_db)

    # Import app after env var is set so it picks up DATABASE_PATH
    from app import app as flask_app
    from app import sportsbet_integrator

    # Ensure the integrator points to our temp DB
    try:
        if getattr(sportsbet_integrator, "db_path", None) != tmp_db:
            sportsbet_integrator.db_path = tmp_db
    except Exception:
        pass

    client = flask_app.test_client()

    # Seed DB with one race and two dogs directly (avoid schema mismatches)
    race_id = "sandown_r1_20250830"
    conn = sqlite3.connect(tmp_db)
    cur = conn.cursor()

    # Create live_odds table if not exists (aligned with integrator's usage)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS live_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            venue TEXT,
            race_number INTEGER,
            race_date DATE,
            race_time TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            odds_decimal REAL,
            odds_fractional TEXT,
            market_type TEXT DEFAULT 'win',
            source TEXT DEFAULT 'sportsbet',
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_current BOOLEAN DEFAULT TRUE
        )
        """
    )

    # Insert two dogs for a single race
    cur.executemany(
        """
        INSERT INTO live_odds (race_id, venue, race_number, race_date, race_time, dog_name, dog_clean_name, box_number, odds_decimal, odds_fractional)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                race_id,
                "Sandown",
                1,
                date.today().isoformat(),
                "12:05",
                "Fleet Runner",
                "FLEET RUNNER",
                1,
                2.8,
                "2.80",
            ),
            (
                race_id,
                "Sandown",
                1,
                date.today().isoformat(),
                "12:05",
                "Night Star",
                "NIGHT STAR",
                2,
                4.2,
                "4.20",
            ),
        ],
    )

    # Create race_metadata table if not exists, then ensure both url and sportsbet_url columns exist
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS race_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT UNIQUE,
            venue TEXT,
            race_number INTEGER,
            race_date DATE,
            race_time TEXT
        )
        """
    )
    # Ensure columns for url variants exist, regardless of which schema created the table first
    try:
        cur.execute("ALTER TABLE race_metadata ADD COLUMN url TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE race_metadata ADD COLUMN sportsbet_url TEXT")
    except sqlite3.OperationalError:
        pass
    cur.execute(
        """
        INSERT OR REPLACE INTO race_metadata (race_id, venue, race_number, race_date, race_time, sportsbet_url, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            race_id,
            "Sandown",
            1,
            date.today().isoformat(),
            "12:05",
            "https://www.sportsbet.com.au/betting/racing/greyhound/sandown",
            "https://www.sportsbet.com.au/betting/racing/greyhound/sandown",
        ),
    )

    # Create value_bets table if not exists and insert one value bet
    cur.execute(
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
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cur.execute(
        """
        INSERT INTO value_bets (race_id, dog_clean_name, predicted_probability, market_odds, implied_probability, value_percentage, confidence_level, bet_recommendation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            race_id,
            "FLEET RUNNER",
            0.45,
            2.80,
            1 / 2.8,
            ((0.45 - (1 / 2.8)) / (1 / 2.8)) * 100,
            "HIGH",
            "STRONG BET - auto-test",
        ),
    )

    conn.commit()
    conn.close()

    # Sanity check: rows visible via integrator connection
    _conn = sqlite3.connect(sportsbet_integrator.db_path)
    _cur = _conn.cursor()
    _cur.execute("SELECT COUNT(*) FROM live_odds")
    assert _cur.fetchone()[0] >= 2
    _cur.execute("SELECT COUNT(*) FROM value_bets")
    assert _cur.fetchone()[0] >= 1
    _conn.close()

    yield client

    # Cleanup
    try:
        os.remove(tmp_db)
    except Exception:
        pass


def test_live_odds_summary_ok(test_app_client):
    resp = test_app_client.get("/api/sportsbet/live_odds")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data and data.get("success") is True
    assert isinstance(data.get("odds_summary"), list)
    # Should contain at least one race (seeded)
    assert len(data["odds_summary"]) >= 1


def test_value_bets_ok(test_app_client):
    resp = test_app_client.get("/api/sportsbet/value_bets")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data and data.get("success") is True
    assert isinstance(data.get("value_bets"), list)
    # Should contain at least one seeded record
    assert data.get("count", 0) >= 1


def test_odds_dashboard_route_renders(test_app_client):
    # Verify the odds dashboard HTML route renders the minimal template
    resp = test_app_client.get("/odds_dashboard")
    assert resp.status_code == 200
    text = resp.data.decode("utf-8").lower()
    assert "sportsbet odds dashboard" in text
