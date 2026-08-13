import json
import sqlite3
from pathlib import Path

import pytest

from scripts import append_august_official_results as subject


RACE_ID = "Race 1 - WPK - 2026-08-03"
URL = "https://www.thedogs.com.au/racing/wentworth-park/2026-08-03/1/results?trial=false"


def _db(path: Path, *, duplicate_identity: bool = False) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            f"""
            CREATE TABLE live_odds (
              id INTEGER PRIMARY KEY, race_id TEXT, race_date TEXT, venue TEXT,
              race_number INTEGER, source_url TEXT, box_number INTEGER,
              dog_name TEXT, dog_clean_name TEXT, odds_decimal REAL,
              capture_timestamp TEXT, market_type TEXT, source TEXT,
              odds_level TEXT, sportsbet_box_source TEXT, capture_mode TEXT,
              race_time TEXT
            );
            CREATE TABLE {subject.RACE_TABLE} (
              id INTEGER PRIMARY KEY, race_id TEXT NOT NULL, race_date TEXT NOT NULL,
              venue TEXT, race_number INTEGER, race_time TEXT, start_datetime TEXT,
              source TEXT NOT NULL, source_url TEXT NOT NULL, status TEXT NOT NULL,
              winner_name TEXT, winner_box INTEGER, position_count INTEGER NOT NULL,
              participant_count INTEGER, box_order_json TEXT NOT NULL,
              participant_source TEXT, captured_at TEXT NOT NULL, inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
              source_artifact_dir TEXT NOT NULL, row_json TEXT NOT NULL,
              UNIQUE(race_id, source_url, box_order_json)
            );
            CREATE TABLE {subject.RUNNER_TABLE} (
              id INTEGER PRIMARY KEY, race_id TEXT NOT NULL, race_date TEXT NOT NULL,
              venue TEXT, race_number INTEGER, source TEXT NOT NULL, source_url TEXT NOT NULL,
              box_number INTEGER NOT NULL, dog_name TEXT NOT NULL, finish_position INTEGER NOT NULL,
              is_winner INTEGER NOT NULL, captured_at TEXT NOT NULL, inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
              source_artifact_dir TEXT NOT NULL, row_json TEXT NOT NULL,
              UNIQUE(race_id, source_url, box_number, dog_name, finish_position)
            );
            """
        )
        rows = [(1, "Alpha", 2.4), (2, "Bravo", 3.2)]
        for box, name, odds in rows:
            conn.execute(
                "INSERT INTO live_odds VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    RACE_ID,
                    "2026-08-03",
                    "WPK",
                    1,
                    "https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-1-1",
                    box,
                    name,
                    name.upper(),
                    odds,
                    "2026-08-03T10:00:00+10:00",
                    "win",
                    "sportsbet",
                    "dog",
                    "explicit_dom",
                    "autonomous_prejump_t60m",
                    "11:00",
                ),
            )
        if duplicate_identity:
            conn.execute(
                "INSERT INTO live_odds VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    RACE_ID,
                    "2026-08-03",
                    "WPK",
                    2,
                    "https://www.sportsbet.com.au/greyhound-racing/australia-nz/wentworth-park/race-1-1",
                    3,
                    "Charlie",
                    "CHARLIE",
                    4.0,
                    "2026-08-03T10:00:00+10:00",
                    "win",
                    "sportsbet",
                    "dog",
                    "explicit_dom",
                    "autonomous_prejump_t60m",
                    "11:00",
                ),
            )


def _html(rows=((1, "Alpha", "1st"), (2, "Bravo", "2nd"))) -> bytes:
    body = "".join(
        f'<tr class="race-runner"><td class="race-runners__finish-position">{position}</td>'
        f'<td class="race-runners__box"><span name="rug_{box}">{box}</span></td>'
        f'<td class="race-runners__name">{name}</td></tr>'
        for box, name, position in rows
    )
    return f'<table class="race-runners--result">{body}</table>'.encode()


def test_exact_match_appends_and_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    raw = tmp_path / "raw.html"
    raw.write_bytes(_html())
    kwargs = {"raw_path": raw, "source_url": URL, "raw_fetched_at": "2026-08-03T12:00:00+10:00"}
    first = subject.run(
        db_path=db, race_id=RACE_ID, output_dir=tmp_path / "evidence", execute=True, **kwargs
    )
    second = subject.run(
        db_path=db, race_id=RACE_ID, output_dir=tmp_path / "evidence", execute=True, **kwargs
    )
    assert (first["status"], first["inserted_race_rows"], first["inserted_runner_rows"]) == (
        "APPENDED",
        1,
        2,
    )
    assert second["status"] == "NOOP_ALREADY_PRESENT"
    with sqlite3.connect(db) as conn:
        race_json = json.loads(
            conn.execute(f"SELECT row_json FROM {subject.RACE_TABLE}").fetchone()[0]
        )
        assert race_json["raw_sha256"] == subject.sha256_bytes(_html())
        assert conn.execute(f"SELECT count(*) FROM {subject.RUNNER_TABLE}").fetchone()[0] == 2


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        (((1, "Alpha", "1st"), (2, "Charlie", "2nd")), "runner_set_mismatch"),
        (((1, "Alpha", "1st"),), "runner_set_mismatch"),
        (((1, "Alpha", "1st"), (2, "Bravo", "3rd")), "incomplete_finish_positions"),
    ],
)
def test_rejects_runner_mismatch_and_partial_results(tmp_path, rows, reason):
    db = tmp_path / "test.db"
    _db(db)
    sealed = subject.seal_race_from_db(db, RACE_ID)
    with pytest.raises(subject.MatchRejected, match=reason):
        subject.validate_official_html(sealed, URL, _html(rows))


def test_rejects_ambiguous_race_identity(tmp_path):
    db = tmp_path / "test.db"
    _db(db, duplicate_identity=True)
    with pytest.raises(subject.MatchRejected, match="ambiguous_race_number"):
        subject.seal_race_from_db(db, RACE_ID)


def test_rejects_provenance_url_mismatch(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    sealed = subject.seal_race_from_db(db, RACE_ID)
    with pytest.raises(subject.MatchRejected, match="official_url_identity_mismatch"):
        subject.validate_official_html(
            sealed,
            "https://www.thedogs.com.au/racing/wentworth-park/2026-08-03/2/results?trial=false",
            _html(),
        )


def test_rejects_conflicting_existing_evidence_without_partial_insert(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    raw = tmp_path / "raw.html"
    raw.write_bytes(_html())
    kwargs = {"raw_path": raw, "source_url": URL, "raw_fetched_at": "2026-08-03T12:00:00+10:00"}
    subject.run(
        db_path=db, race_id=RACE_ID, output_dir=tmp_path / "evidence", execute=True, **kwargs
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            f"UPDATE {subject.RACE_TABLE} SET row_json = ?",
            (json.dumps({"raw_sha256": "conflict"}),),
        )
    with pytest.raises(subject.MatchRejected, match="conflicting_or_unverifiable"):
        subject.run(
            db_path=db, race_id=RACE_ID, output_dir=tmp_path / "evidence", execute=True, **kwargs
        )
    with sqlite3.connect(db) as conn:
        assert conn.execute(f"SELECT count(*) FROM {subject.RUNNER_TABLE}").fetchone()[0] == 2


def test_rejects_existing_artifact_with_failed_hash_verification(tmp_path, monkeypatch):
    payload = _html()
    monkeypatch.setattr(subject, "sha256_bytes", lambda _: "0" * 64)
    path = tmp_path / f"thedogs_official_{'0' * 64}.html"
    path.write_bytes(b"different")
    with pytest.raises(subject.MatchRejected, match="raw_artifact_hash_collision"):
        subject.write_raw_artifact(tmp_path, payload)


def test_rejects_post_jump_or_unclassified_odds(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE live_odds SET capture_timestamp = '2026-08-03T11:01:00+10:00'")
    with pytest.raises(subject.MatchRejected, match="odds_capture_not_before"):
        subject.seal_race_from_db(db, RACE_ID)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE live_odds SET capture_timestamp = '2026-08-03T10:00:00+10:00', capture_mode = NULL"
        )
    with pytest.raises(subject.MatchRejected, match="non_prejump_odds_capture"):
        subject.seal_race_from_db(db, RACE_ID)


def test_rejects_sportsbet_venue_or_race_url_mismatch(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE live_odds SET source_url = 'https://www.sportsbet.com.au/greyhound-racing/australia-nz/hobart/race-2-1'"
        )
    with pytest.raises(subject.MatchRejected, match="sportsbet_url_identity_mismatch"):
        subject.seal_race_from_db(db, RACE_ID)


@pytest.mark.parametrize("missing", ["race", "runner"])
def test_rejects_partial_existing_bundle_without_writes(tmp_path, missing):
    db = tmp_path / "test.db"
    _db(db)
    raw = tmp_path / "raw.html"
    raw.write_bytes(_html())
    kwargs = {"raw_path": raw, "source_url": URL, "raw_fetched_at": "2026-08-03T12:00:00+10:00"}
    subject.run(
        db_path=db, race_id=RACE_ID, output_dir=tmp_path / "evidence", execute=True, **kwargs
    )
    with sqlite3.connect(db) as conn:
        if missing == "race":
            conn.execute(f"DELETE FROM {subject.RACE_TABLE}")
        else:
            conn.execute(f"DELETE FROM {subject.RUNNER_TABLE} WHERE box_number = 2")
    with pytest.raises(subject.MatchRejected, match="conflicting_or_incomplete"):
        subject.run(
            db_path=db, race_id=RACE_ID, output_dir=tmp_path / "evidence", execute=True, **kwargs
        )
    with sqlite3.connect(db) as conn:
        counts = (
            conn.execute(f"SELECT count(*) FROM {subject.RACE_TABLE}").fetchone()[0],
            conn.execute(f"SELECT count(*) FROM {subject.RUNNER_TABLE}").fetchone()[0],
        )
    assert counts == ((0, 2) if missing == "race" else (1, 1))


def test_raw_html_requires_explicit_fetch_provenance(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    raw = tmp_path / "raw.html"
    raw.write_bytes(_html())
    with pytest.raises(subject.MatchRejected, match="requires_source_url_and_fetched_at"):
        subject.run(
            db_path=db,
            race_id=RACE_ID,
            output_dir=tmp_path / "evidence",
            execute=False,
            raw_path=raw,
            source_url=URL,
        )


def test_rejects_official_fetch_time_before_scheduled_start(tmp_path):
    db = tmp_path / "test.db"
    _db(db)
    raw = tmp_path / "raw.html"
    raw.write_bytes(_html())
    with pytest.raises(subject.MatchRejected, match="fetched_before_scheduled_start"):
        subject.run(
            db_path=db,
            race_id=RACE_ID,
            output_dir=tmp_path / "evidence",
            execute=False,
            raw_path=raw,
            source_url=URL,
            raw_fetched_at="2026-08-03T10:30:00+10:00",
        )
