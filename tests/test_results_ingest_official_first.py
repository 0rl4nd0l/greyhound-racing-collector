import importlib.util
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _load_ingest_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ingest_results_for_date.py"
    spec = importlib.util.spec_from_file_location(
        "_ingest_results_for_date_for_tests", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_sportsbet_result_text_extracts_top_four_boxes():
    module = _load_ingest_module()
    text = """
    Warragul
    18:15
    R4 Barn Function Area
    6,8,3,4
    """

    parsed = module.parse_sportsbet_result_text(text)

    assert parsed[4]["time"] == "18:15"
    assert parsed[4]["race_name"] == "Barn Function Area"
    assert parsed[4]["boxes"] == [6, 8, 3, 4]


def test_parse_participants_from_csv_uses_only_prefixed_runner_rows(tmp_path):
    module = _load_ingest_module()
    csv_path = tmp_path / "Race 4 - WRGL - 2026-05-21.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Dog Name,PLC,TIME",
                "1. Rumour Not Fact,3,22.51",
                ",1,22.31",
                "2. Mr. Fahrenheit,5,22.81",
                "2. Mr. Fahrenheit,6,22.92",
            ]
        ),
        encoding="utf-8",
    )

    participants = module.parse_participants_from_csv(csv_path)

    assert participants == [
        {"box_number": 1, "dog_name": "Rumour Not Fact"},
        {"box_number": 2, "dog_name": "Mr. Fahrenheit"},
    ]


def test_parse_thedogs_result_text_matches_ordinals_to_local_runners():
    module = _load_ingest_module()
    participants = [
        {"box_number": 1, "dog_name": "Rumour Not Fact"},
        {"box_number": 2, "dog_name": "Mr. Fahrenheit"},
        {"box_number": 3, "dog_name": "Tootsie"},
        {"box_number": 4, "dog_name": "Offensive Lady"},
        {"box_number": 5, "dog_name": "Socks And Slides"},
        {"box_number": 6, "dog_name": "Rhine Star"},
        {"box_number": 7, "dog_name": "Chorus Line"},
        {"box_number": 8, "dog_name": "Valve Bounce"},
    ]
    text = """
    Results
    1st
    6. Rhine Star
    2nd
    8. Valve Bounce
    3rd
    3. Tootsie
    """

    positions = module.parse_thedogs_result_text(text, participants)

    assert positions == {6: 1, 8: 2, 3: 3}


def _make_ingest_db(tmp_path):
    db_path = tmp_path / "results.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            race_time TEXT,
            start_datetime TEXT,
            sportsbet_url TEXT,
            results_status TEXT DEFAULT 'pending',
            winner_name TEXT,
            winner_odds REAL,
            winner_source TEXT,
            scraping_attempts INTEGER DEFAULT 0,
            last_scraped_at TEXT,
            extraction_timestamp TEXT,
            actual_field_size INTEGER,
            field_size INTEGER,
            url TEXT,
            parse_confidence REAL,
            data_quality_note TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE dog_race_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_name TEXT,
            dog_clean_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            placing INTEGER,
            scraped_finish_position TEXT,
            extraction_timestamp TEXT,
            data_source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE live_odds (
            race_id TEXT,
            market_type TEXT,
            box_number INTEGER,
            odds_decimal REAL,
            is_current INTEGER,
            timestamp TEXT
        )
        """
    )
    conn.commit()
    return db_path, conn


def _candidate(module, tmp_path):
    csv_path = tmp_path / "Race 4 - WRGL - 2026-05-21.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Dog Name",
                "1. Alpha Runner",
                "2. Bravo Runner",
                "3. Charlie Runner",
                "4. Delta Runner",
                "5. Echo Runner",
                "6. Foxtrot Runner",
                "7. Golf Runner",
                "8. Hotel Runner",
            ]
        ),
        encoding="utf-8",
    )
    participants = module.parse_participants_from_csv(csv_path)
    return module.RaceCandidate(
        race_id="Race 4 - WRGL - 2026-05-21",
        venue="WRGL",
        race_number=4,
        race_date="2026-05-21",
        race_time="16:30",
        start_datetime=None,
        sportsbet_url=(
            "https://www.sportsbet.com.au/betting/greyhound-racing/"
            "australia-nz/warragul/race-4"
        ),
        csv_path=csv_path,
        participants=participants,
        lifecycle_status="jumped_pending_results",
    )


def test_thedogs_fetch_success_uses_resulted_status(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Body:
        text = "Results\n1st\n6. Foxtrot Runner\n2nd\n8. Hotel Runner\n"

    class Driver:
        title = "Race results"

        def get(self, url):
            self.current_url = url

        def find_element(self, *_args):
            return Body()

    result = module.TheDogsResultFetcher(Driver(), wait_seconds=0).fetch(candidate)

    assert result.source == "thedogs_official"
    assert result.status == "resulted"
    assert result.positions_by_box == {6: 1, 8: 2}


def test_sportsbet_fetcher_marks_top_four_as_partial(tmp_path):
    module = _load_ingest_module()
    candidate = _candidate(module, tmp_path)

    class Body:
        text = "Warragul\n18:15\nR4 Barn Function Area\n6,8,3,4\n"

    class Driver:
        current_url = "https://www.sportsbet.com.au/results/2026-05-21/racing/greyhound-racing-4/warragul-123"

        def get(self, url):
            self.current_url = url

        def find_element(self, *_args):
            return Body()

    fetcher = module.SportsbetResultFetcher(Driver(), "2026-05-21", wait_seconds=0)
    fetcher.category_links = {"warragul": Driver.current_url}

    result = fetcher.fetch(candidate)

    assert result.source == "sportsbet_results_top4"
    assert result.status == "partial_sportsbet_results"
    assert result.positions_by_box == {6: 1, 8: 2, 3: 3, 4: 4}


def test_write_sportsbet_fallback_records_partial_status_and_thedogs_error(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            candidate.race_id,
            candidate.venue,
            candidate.race_number,
            candidate.race_date,
            candidate.race_time,
            candidate.sportsbet_url,
        ),
    )
    conn.execute(
        """
        INSERT INTO live_odds
            (race_id, market_type, box_number, odds_decimal, is_current, timestamp)
        VALUES (?, 'win', 6, 3.2, 1, '2026-05-21T16:29:00')
        """,
        (candidate.race_id,),
    )
    conn.commit()

    official_error = module.SourceResult(
        source="thedogs_official",
        status="error",
        source_url="https://www.thedogs.com.au/racing/warragul/2026-05-21/4",
        positions_by_box={},
        raw_order=[],
        error="thedogs_403_forbidden",
    )
    fallback = module.SourceResult(
        source="sportsbet_results_top4",
        status="partial_sportsbet_results",
        source_url="https://www.sportsbet.com.au/results/2026-05-21/racing/greyhound-racing-4/warragul-123",
        positions_by_box={6: 1, 8: 2, 3: 3, 4: 4},
        raw_order=[6, 8, 3, 4],
    )

    module.write_result(conn, candidate, fallback, [official_error, fallback], dry_run=False)
    conn.commit()

    row = conn.execute(
        """
        SELECT winner_name, winner_odds, winner_source, results_status, data_quality_note
        FROM race_metadata
        WHERE race_id = ?
        """,
        (candidate.race_id,),
    ).fetchone()
    dog_rows = conn.execute(
        "SELECT box_number, finish_position, data_source FROM dog_race_data ORDER BY box_number"
    ).fetchall()
    conn.close()

    assert db_path.exists()
    assert row[0] == "Foxtrot Runner"
    assert row[1] == 3.2
    assert row[2] == "sportsbet_results_top4"
    assert row[3] == "partial_sportsbet_results"
    assert "thedogs_official:thedogs_403_forbidden" in row[4]
    assert dog_rows[5] == (6, 1, "sportsbet_results_top4")
    assert dog_rows[0] == (1, None, "sportsbet_results_top4")


def test_write_official_result_uses_resulted_lifecycle_status(tmp_path):
    module = _load_ingest_module()
    _db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, ?, ?, ?, ?, ?, 'jumped_pending_results')
        """,
        (
            candidate.race_id,
            candidate.venue,
            candidate.race_number,
            candidate.race_date,
            candidate.race_time,
            candidate.sportsbet_url,
        ),
    )
    official = module.SourceResult(
        source="thedogs_official",
        status="resulted",
        source_url="https://www.thedogs.com.au/racing/warragul/2026-05-21/4",
        positions_by_box={6: 1, 8: 2, 3: 3, 4: 4, 1: 5, 2: 6, 5: 7, 7: 8},
        raw_order=[6, 8, 3, 4, 1, 2, 5, 7],
    )

    module.write_result(conn, candidate, official, [official], dry_run=False)
    conn.commit()

    row = conn.execute(
        "SELECT winner_source, results_status, parse_confidence FROM race_metadata WHERE race_id = ?",
        (candidate.race_id,),
    ).fetchone()
    conn.close()

    assert row == ("thedogs_official", "resulted", 1.0)


def test_dry_run_write_result_does_not_mutate_database(tmp_path):
    module = _load_ingest_module()
    _db_path, conn = _make_ingest_db(tmp_path)
    candidate = _candidate(module, tmp_path)
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, ?, ?, ?, ?, ?, 'pending')
        """,
        (
            candidate.race_id,
            candidate.venue,
            candidate.race_number,
            candidate.race_date,
            candidate.race_time,
            candidate.sportsbet_url,
        ),
    )
    conn.commit()
    result = module.SourceResult(
        source="sportsbet_results_top4",
        status="partial_sportsbet_results",
        source_url="https://example.test/results",
        positions_by_box={6: 1},
        raw_order=[6],
    )

    summary = module.write_result(conn, candidate, result, [result], dry_run=True)
    conn.commit()

    status = conn.execute(
        "SELECT results_status FROM race_metadata WHERE race_id = ?",
        (candidate.race_id,),
    ).fetchone()[0]
    dog_count = conn.execute("SELECT COUNT(*) FROM dog_race_data").fetchone()[0]
    conn.close()

    assert summary["dry_run"] is True
    assert status == "pending"
    assert dog_count == 0


def test_backup_db_creates_readable_pre_write_copy(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    conn.close()

    backup_path = module.backup_db(db_path)

    assert backup_path.exists()
    assert sqlite3.connect(backup_path).execute("PRAGMA quick_check").fetchone()[0] == "ok"


def test_load_candidates_skips_today_race_before_jump(tmp_path):
    module = _load_ingest_module()
    db_path, conn = _make_ingest_db(tmp_path)
    upcoming_dir = tmp_path / "upcoming"
    upcoming_dir.mkdir()
    candidate_csv = upcoming_dir / "Race 4 - WRGL - 2026-05-21.csv"
    candidate_csv.write_text("Dog Name\n1. Alpha Runner\n", encoding="utf-8")
    conn.execute(
        """
        INSERT INTO race_metadata
            (race_id, venue, race_number, race_date, race_time, sportsbet_url, results_status)
        VALUES (?, 'WRGL', 4, '2026-05-21', '18:30', ?, 'pending')
        """,
        (
            "Race 4 - WRGL - 2026-05-21",
            "https://www.sportsbet.com.au/betting/greyhound-racing/australia-nz/warragul/race-4",
        ),
    )
    conn.commit()
    conn.close()

    candidates, skipped = module.load_candidates(
        db_path,
        "2026-05-21",
        upcoming_dir,
        [],
        now=datetime(2026, 5, 21, 17, 37, tzinfo=ZoneInfo("Australia/Melbourne")),
    )

    assert candidates == []
    assert skipped == [
        {
            "race_id": "Race 4 - WRGL - 2026-05-21",
            "reason": "race_not_jumped:upcoming_not_jumped",
        }
    ]
