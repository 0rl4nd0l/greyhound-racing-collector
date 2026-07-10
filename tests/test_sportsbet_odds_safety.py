import sqlite3
from datetime import date
from pathlib import Path

import pytest

import odds_auto_integrator
from odds_auto_integrator import _copy_current_odds_to_alias
from sportsbet_odds_integrator import (
    SPORTSBET_LIST_POSITION_BOX_SOURCE,
    SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
    SportsbetOddsIntegrator,
    sportsbet_paired_fixed_prices,
    sportsbet_paired_market_rows,
    sportsbet_runner_header_count,
    sportsbet_runner_box_metadata,
)
from utils import feature_flags


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def test_save_odds_metadata_upsert_preserves_existing_race_metadata_row(tmp_path):
    db_path = tmp_path / "odds.db"
    integrator = SportsbetOddsIntegrator(db_path=str(db_path))
    race_id = integrator._canonical_race_id("Sandown", "2026-05-21", 4)

    with sqlite3.connect(db_path) as conn:
        _ensure_column(conn, "race_metadata", "winner_name", "TEXT")
        _ensure_column(conn, "race_metadata", "grade", "TEXT")
        conn.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, race_time,
                sportsbet_url, winner_name, grade
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                race_id,
                "Sandown",
                4,
                "2026-05-21",
                "12:01",
                "https://old.example/race",
                "Preserved Winner",
                "Grade 5",
            ),
        )
        existing_id = conn.execute(
            "SELECT id FROM race_metadata WHERE race_id = ?", (race_id,)
        ).fetchone()[0]
        conn.commit()

    integrator.save_odds_to_database(
        {
            "race_id": "ignored_noncanonical_id",
            "venue": "Sandown",
            "venue_slug": "sandown",
            "race_number": 4,
            "race_date": "2026-05-21",
            "race_time": "12:08",
            "venue_url": "https://new.example/race",
            "odds_data": [],
        }
    )

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, winner_name, grade, race_time, sportsbet_url
            FROM race_metadata
            WHERE race_id = ?
            """,
            (race_id,),
        ).fetchall()

    assert rows == [
        (
            existing_id,
            "Preserved Winner",
            "Grade 5",
            "12:08",
            "https://new.example/race",
        )
    ]


def test_alias_metadata_update_preserves_unrelated_columns(tmp_path):
    db_path = tmp_path / "alias.db"
    SportsbetOddsIntegrator(db_path=str(db_path))
    source_race_id = "SAND_2026-05-21_4"
    alias_race_id = "Race 4 - Sandown - 2026-05-21"

    with sqlite3.connect(db_path) as conn:
        _ensure_column(conn, "race_metadata", "winner_name", "TEXT")
        conn.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, race_time,
                sportsbet_url, url, venue_slug, start_datetime
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_race_id,
                "Sandown",
                4,
                "2026-05-21",
                "13:00",
                "https://source.example/race",
                "https://source.example/race",
                "sandown",
                "2026-05-21T13:00:00",
            ),
        )
        conn.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, race_time,
                sportsbet_url, winner_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alias_race_id,
                "Old Venue",
                4,
                "2026-05-21",
                "12:00",
                "https://old.example/race",
                "Alias Winner",
            ),
        )
        alias_id = conn.execute(
            "SELECT id FROM race_metadata WHERE race_id = ?", (alias_race_id,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, race_time, dog_name,
                dog_clean_name, box_number, odds_decimal, odds_fractional,
                market_type, source, is_current, topN
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_race_id,
                "Sandown",
                4,
                "2026-05-21",
                "13:00",
                "Fast Dog",
                "FAST DOG",
                1,
                2.5,
                "2.50",
                "win",
                "sportsbet",
                1,
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, race_time, dog_name,
                dog_clean_name, box_number, odds_decimal, odds_fractional,
                market_type, source, is_current, topN
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alias_race_id,
                "Old Venue",
                4,
                "2026-05-21",
                "12:00",
                "Old Dog",
                "OLD DOG",
                2,
                7.5,
                "7.50",
                "win",
                "sportsbet",
                1,
                None,
            ),
        )
        conn.commit()

    inserted = _copy_current_odds_to_alias(
        str(db_path),
        source_race_id,
        alias_race_id,
        "Sandown",
        4,
        "2026-05-21",
    )

    with sqlite3.connect(db_path) as conn:
        metadata = conn.execute(
            """
            SELECT id, winner_name, race_time, sportsbet_url, venue, race_number, race_date
            FROM race_metadata
            WHERE race_id = ?
            """,
            (alias_race_id,),
        ).fetchone()
        current_alias_odds = conn.execute(
            """
            SELECT dog_clean_name, odds_decimal
            FROM live_odds
            WHERE race_id = ? AND is_current = 1
            """,
            (alias_race_id,),
        ).fetchall()
        old_alias_current = conn.execute(
            """
            SELECT COUNT(*)
            FROM live_odds
            WHERE race_id = ? AND dog_clean_name = 'OLD DOG' AND is_current = 1
            """,
            (alias_race_id,),
        ).fetchone()[0]

    assert inserted == 1
    assert metadata == (
        alias_id,
        "Alias Winner",
        "13:00",
        "https://source.example/race",
        "Sandown",
        4,
        "2026-05-21",
    )
    assert current_alias_odds == [("FAST DOG", 2.5)]
    assert old_alias_current == 0


def test_append_pre_jump_capture_defaults_to_canonical_race_id_and_preserves_source_url(
    tmp_path,
):
    db_path = tmp_path / "canonical_capture.db"
    integrator = SportsbetOddsIntegrator(db_path=str(db_path))
    source_url = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
        "horsham/race-7-10514874"
    )

    report = integrator.append_pre_jump_odds_snapshot(
        {
            "race_id": "sportsbet-race-10514874",
            "venue": "Horsham",
            "race_number": 7,
            "race_date": "2026-05-26",
            "race_time": "16:36",
            "venue_url": source_url,
        },
        [
            {
                "dog_name": "Pablo Vendetta",
                "dog_clean_name": "PABLO VENDETTA",
                "box_number": 1,
                "odds_decimal": 3.5,
                "sportsbet_box_source": SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
                "sportsbet_list_position": 1,
            }
        ],
        capture_mode="opt_in_live_pre_jump_snapshot",
        capture_timestamp="2026-05-26T15:42:29",
    )

    assert report["status"] == "SUCCESS"
    assert report["race_id"] == "HOR_2026-05-26_7"
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT race_id, source_url, capture_mode, dog_clean_name, box_number,
                   sportsbet_box_source, sportsbet_list_position
            FROM live_odds
            """
        ).fetchone()

        assert row == (
            "HOR_2026-05-26_7",
            source_url,
            "opt_in_live_pre_jump_snapshot",
            "PABLO VENDETTA",
            1,
            SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
            1,
        )


def test_append_pre_jump_capture_can_skip_race_metadata_write(tmp_path):
    db_path = tmp_path / "capture_no_metadata.db"
    integrator = SportsbetOddsIntegrator(db_path=str(db_path))
    source_url = (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
        "wentworth-park/race-1"
    )

    report = integrator.append_pre_jump_odds_snapshot(
        {
            "race_id": "Race 1 - WPK - 2026-06-10",
            "venue": "WPK",
            "race_number": 1,
            "race_date": "2026-06-10",
            "race_time": "15:00",
            "venue_url": source_url,
            "preserve_race_id": True,
        },
        [
            {
                "dog_name": "Alpha",
                "dog_clean_name": "ALPHA",
                "box_number": 1,
                "odds_decimal": 2.4,
                "sportsbet_box_source": SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
            }
        ],
        capture_mode="autonomous_prejump_t30m",
        capture_timestamp="2026-06-10T14:40:00+10:00",
        write_race_metadata=False,
    )

    assert report["status"] == "SUCCESS"
    assert report["inserted_rows"] == 1
    with sqlite3.connect(db_path) as conn:
        metadata_count = conn.execute(
            "SELECT COUNT(*) FROM race_metadata WHERE race_id = ?",
            ("Race 1 - WPK - 2026-06-10",),
        ).fetchone()[0]
        live_count = conn.execute(
            "SELECT COUNT(*) FROM live_odds WHERE race_id = ? AND capture_mode = ?",
            ("Race 1 - WPK - 2026-06-10", "autonomous_prejump_t30m"),
        ).fetchone()[0]

    assert metadata_count == 0
    assert live_count == 1


def test_sportsbet_runner_box_metadata_uses_explicit_runner_text_over_list_position():
    metadata = sportsbet_runner_box_metadata(
        list_position=5,
        runner_text="6. Memories\n(6)\nF: 244311\n9.00\n3.25",
    )

    assert metadata["box_number"] == 6
    assert metadata["sportsbet_box_source"] == SPORTSBET_RUNNER_TEXT_BOX_SOURCE
    assert metadata["sportsbet_list_position"] == 5


def test_sportsbet_runner_box_metadata_prefers_parenthesized_final_box_for_reserve():
    metadata = sportsbet_runner_box_metadata(
        list_position=8,
        runner_text="9. High Rollin' (6)\nF: 388638\n8.00",
    )

    assert metadata["box_number"] == 6
    assert metadata["sportsbet_box_source"] == SPORTSBET_RUNNER_TEXT_BOX_SOURCE
    assert metadata["sportsbet_list_position"] == 8


def test_sportsbet_runner_box_metadata_accepts_abbreviated_runner_name_periods():
    metadata = sportsbet_runner_box_metadata(
        list_position=3,
        runner_text="3. Dr. Will (3)\nF: 241112\n8.50",
    )

    assert metadata["box_number"] == 3
    assert metadata["sportsbet_box_source"] == SPORTSBET_RUNNER_TEXT_BOX_SOURCE
    assert metadata["sportsbet_list_position"] == 3


def test_sportsbet_runner_header_count_detects_single_runner_text():
    assert sportsbet_runner_header_count("3. Dr. Will (3)\nF: 241112\n8.50") == 1


def test_sportsbet_runner_header_count_detects_concatenated_race_card_text():
    raw_text = "\n".join(
        [
            "Expand Form",
            "Flucs",
            "Fast Form",
            "Runner",
            "Early Speed",
            "Open",
            "Fluc 1",
            "Fluc 2",
            "Win",
            "Fixed",
            "Place",
            "Fixed",
            "Each Way",
            "Fixed",
            "1. Big Cutie (1)",
            "F: 145581",
            "T: Meredith Verhagen",
            "Early Speed:",
            "3.00",
            "2.15",
            "2.20",
            "2.15",
            "Fav",
            "1.17",
            "EW",
            "2. Our Hedley (2)",
            "F: 525153",
            "T: Jonathan Childs",
            "Early Speed:",
            "7.00",
            "6.50",
            "6.00",
            "6.50",
            "1.67",
            "EW",
            "3. Super Skunk (3)",
            "F: 222118",
            "T: Meredith Verhagen",
            "Early Speed:",
            "6.00",
            "5.00",
            "5.50",
            "5.00",
            "1.67",
            "EW",
            "4. Golden Effects (4)",
        ]
    )

    assert sportsbet_runner_header_count(raw_text) == 4


def test_sportsbet_runner_box_metadata_marks_list_position_only_as_ambiguous():
    metadata = sportsbet_runner_box_metadata(
        list_position=5,
        runner_text="Memories\nF: 244311\n9.00\n3.25",
    )

    assert metadata["box_number"] == 5
    assert metadata["sportsbet_box_source"] == SPORTSBET_LIST_POSITION_BOX_SOURCE
    assert metadata["sportsbet_list_position"] == 5


class _FakeSportsbetBy:
    CSS_SELECTOR = "css selector"
    XPATH = "xpath"


class _FakeSportsbetEC:
    @staticmethod
    def presence_of_element_located(_locator):
        return lambda _driver: True


class _FakeSportsbetWait:
    def __init__(self, _driver, _timeout):
        pass

    def until(self, condition):
        return condition(None)


class _FakeSportsbetElement:
    def __init__(self, text):
        self.text = text


class _FakeSportsbetCard:
    def __init__(self, runner_number, dog_name, odds_text):
        self.runner_number = runner_number
        self.dog_name = dog_name
        self.odds_text = odds_text
        self.text = f"{runner_number}. {dog_name} ({runner_number})\n{odds_text}"

    def find_element(self, _by, selector):
        if selector == "div[data-automation-id='racecard-outcome-name'] span":
            return _FakeSportsbetElement(f"{self.runner_number}. {self.dog_name}")
        if selector == "[data-automation-id*='price-text']":
            return _FakeSportsbetElement(self.odds_text)
        raise LookupError(selector)

    def find_elements(self, _by, _selector):
        return []


class _FakeSportsbetRunnerCardDriver:
    current_url = "https://www.sportsbet.com.au/test-race"
    page_source = "<html></html>"

    def __init__(self, cards):
        self.cards = cards

    def find_elements(self, _by, selector):
        if "racecard-outcome-name" in selector or selector.startswith(
            "div[data-automation-id^='racecard-outcome-']"
        ):
            return self.cards
        return []

    def execute_script(self, *_args):
        return None

    def save_screenshot(self, _path):
        return True


def test_runner_card_extractor_scans_all_candidates_before_deduping(monkeypatch, tmp_path):
    monkeypatch.setattr("sportsbet_odds_integrator.time.sleep", lambda _seconds: None)
    runners = [
        (1, "Alpha One", "2.10"),
        (2, "Bravo Two", "3.20"),
        (3, "Charlie Three", "4.30"),
        (4, "Delta Four", "5.40"),
        (5, "Echo Five", "6.50"),
        (6, "Foxtrot Six", "7.60"),
        (7, "Golf Seven", "8.70"),
        (8, "Hotel Eight", "9.80"),
    ]
    cards = []
    for runner_number, dog_name, win_odds in runners:
        cards.append(_FakeSportsbetCard(runner_number, dog_name, win_odds))
        cards.append(_FakeSportsbetCard(runner_number, dog_name, "1.50"))

    integrator = SportsbetOddsIntegrator(
        db_path=str(tmp_path / "odds.db"),
        setup_database=False,
    )
    integrator.driver = _FakeSportsbetRunnerCardDriver(cards)
    monkeypatch.setattr(
        integrator,
        "_selenium_primitives",
        lambda: (
            _FakeSportsbetBy,
            _FakeSportsbetWait,
            _FakeSportsbetEC,
            TimeoutError,
        ),
    )

    extracted = integrator.extract_odds_strategy_runner_cards()

    assert [runner["box_number"] for runner in extracted] == list(range(1, 9))
    assert [runner["dog_clean_name"] for runner in extracted] == [
        "ALPHA ONE",
        "BRAVO TWO",
        "CHARLIE THREE",
        "DELTA FOUR",
        "ECHO FIVE",
        "FOXTROT SIX",
        "GOLF SEVEN",
        "HOTEL EIGHT",
    ]
    assert [runner["odds_decimal"] for runner in extracted] == [
        2.10,
        3.20,
        4.30,
        5.40,
        6.50,
        7.60,
        8.70,
        9.80,
    ]
    assert len(extracted) == 8


def test_race_page_uses_explicit_paired_prices_without_generic_place_click(
    monkeypatch, tmp_path
):
    class FakeRacePageDriver:
        current_url = ""
        title = "Race 4 Healesville Betting Odds"

        def get(self, url):
            self.current_url = url

        def execute_script(self, *_args):
            return "complete"

        def find_elements(self, *_args):
            return []

    class FakeReadyWait:
        def __init__(self, _driver, _timeout):
            pass

        def until(self, _condition):
            return True

    source_rows = [
        {
            "dog_name": "Proud Mary",
            "dog_clean_name": "PROUD MARY",
            "box_number": 1,
            "odds_decimal": 5.0,
            "odds_fractional": "5.00",
            "sportsbet_box_source": SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
            "sportsbet_list_position": 1,
            "sportsbet_raw_runner_text": (
                "1. Proud Mary (1)\nF: 853614\nT: KATRINA EVANS\n"
                "Early Speed:\n4.40\n4.80\n5.00\n1.80\nEW\nNo Awards"
            ),
        },
        {
            "dog_name": "Devil In Me",
            "dog_clean_name": "DEVIL IN ME",
            "box_number": 2,
            "odds_decimal": 5.0,
            "odds_fractional": "5.00",
            "sportsbet_box_source": SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
            "sportsbet_list_position": 2,
            "sportsbet_raw_runner_text": (
                "2. Devil In Me (2)\nF: X75332\nT: KAY KING\n"
                "Early Speed:\n4.50\n5.00\n5.50\n5.00\n1.67\nEW\nPlace Rate"
            ),
        },
    ]
    extractor_calls = []
    integrator = SportsbetOddsIntegrator(
        db_path=str(tmp_path / "odds.db"),
        setup_database=False,
    )
    integrator.driver = FakeRacePageDriver()
    monkeypatch.setattr("sportsbet_odds_integrator.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        integrator,
        "_selenium_primitives",
        lambda: (_FakeSportsbetBy, FakeReadyWait, _FakeSportsbetEC, TimeoutError),
    )
    monkeypatch.setattr(
        integrator,
        "extract_odds_strategy_runner_cards",
        lambda: extractor_calls.append("runner_cards") or list(source_rows),
    )
    monkeypatch.setattr(
        integrator,
        "_select_place_market",
        lambda: (_ for _ in ()).throw(
            AssertionError("generic Place click must not run for paired source rows")
        ),
    )
    monkeypatch.setattr(integrator, "extract_race_number_from_page", lambda _venue: 4)

    race = integrator.get_race_odds_from_page(
        {
            "race_id": "HEA_2026-07-10_4",
            "venue": "Healesville",
            "venue_slug": "healesville",
            "race_number": 4,
            "race_date": date(2026, 7, 10),
            "venue_url": (
                "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
                "healesville/race-4-10680641"
            ),
        }
    )

    assert extractor_calls == ["runner_cards"]
    assert [row["odds_decimal"] for row in race["odds_data"]] == [5.0, 5.0]
    assert [row["odds_decimal"] for row in race["odds_data_place"]] == [1.8, 1.67]
    assert [row["box_number"] for row in race["odds_data_place"]] == [1, 2]
    assert race["place_topN"] == 3


def test_paired_price_parser_requires_explicit_ew_pair_and_sane_market_order():
    assert sportsbet_paired_fixed_prices(
        "7. Dinosaur Deano (7)\nEarly Speed:\n2.60\n2.90\n3.00\n"
        "2.90\nFav\n1.30\nEW\nPlace Rate"
    ) == (2.9, 1.3)
    assert sportsbet_paired_fixed_prices(
        "7. Dinosaur Deano (7)\nEarly Speed:\n2.90\n1.30"
    ) is None
    assert sportsbet_paired_fixed_prices(
        "7. Dinosaur Deano (7)\nEarly Speed:\n1.30\n2.90\nEW"
    ) is None


def test_paired_market_rows_uses_box_aligned_raw_pair_when_button_price_is_place():
    favourite_row = {
        "dog_name": "Dinosaur Deano",
        "box_number": 7,
        "odds_decimal": 1.30,
        "sportsbet_raw_runner_text": (
            "7. Dinosaur Deano (7)\nEarly Speed:\n2.60\n2.90\n3.00\n"
            "2.90\nFav\n1.30\nEW"
        ),
    }
    wrong_box_row = {
        "dog_name": "Wrong Box Runner",
        "box_number": 8,
        "odds_decimal": 4.80,
        "sportsbet_raw_runner_text": (
            "6. Wrong Box Runner (6)\nEarly Speed:\n4.40\n4.80\n1.80\nEW"
        ),
    }

    win_rows, place_rows = sportsbet_paired_market_rows(
        [favourite_row, wrong_box_row]
    )

    assert [row["dog_name"] for row in win_rows] == ["Dinosaur Deano"]
    assert [row["odds_decimal"] for row in win_rows] == [2.90]
    assert [row["dog_name"] for row in place_rows] == ["Dinosaur Deano"]
    assert [row["odds_decimal"] for row in place_rows] == [1.30]


def test_race_page_recovers_win_and_place_from_second_explicit_paired_render(
    monkeypatch, tmp_path
):
    class FakeRacePageDriver:
        current_url = ""
        title = "Race 6 Healesville Betting Odds"

        def get(self, url):
            self.current_url = url

        def execute_script(self, *_args):
            return "complete"

        def find_elements(self, *_args):
            return []

    class FakeReadyWait:
        def __init__(self, _driver, _timeout):
            pass

        def until(self, _condition):
            return True

    runners = [
        (1, "Fiddles", 4.20, 1.44),
        (2, "Little Master", 10.00, 2.40),
        (3, "Archie Mcivor", 4.80, 1.57),
        (4, "Fast Love", 17.00, 3.40),
        (5, "Sunny Stride", 67.00, 10.00),
        (6, "Minter Scorch", 6.00, 1.91),
        (7, "Winking Willie", 2.45, 1.22),
        (8, "Leon's Entity", 6.50, 1.88),
    ]

    def row(box, name, win, place, *, paired):
        raw_text = f"{box}. {name} ({box})\nF: 123456\nT: TEST TRAINER\nEarly Speed:"
        if paired:
            raw_text += f"\n3.60\n3.70\n{win:.2f}\n{place:.2f}\nEW"
        return {
            "dog_name": name,
            "dog_clean_name": name.upper(),
            "box_number": box,
            "odds_decimal": win,
            "odds_fractional": f"{win:.2f}",
            "sportsbet_box_source": SPORTSBET_RUNNER_TEXT_BOX_SOURCE,
            "sportsbet_list_position": box,
            "sportsbet_raw_runner_text": raw_text,
        }

    partial_first_render = [row(*runner, paired=False) for runner in runners[:4]]
    complete_second_render = [row(*runner, paired=True) for runner in runners]
    extractor_results = iter([partial_first_render, complete_second_render])
    integrator = SportsbetOddsIntegrator(
        db_path=str(tmp_path / "odds.db"),
        setup_database=False,
    )
    integrator.driver = FakeRacePageDriver()
    monkeypatch.setattr("sportsbet_odds_integrator.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        integrator,
        "_selenium_primitives",
        lambda: (_FakeSportsbetBy, FakeReadyWait, _FakeSportsbetEC, TimeoutError),
    )
    monkeypatch.setattr(
        integrator,
        "extract_odds_strategy_runner_cards",
        lambda: next(extractor_results),
    )
    monkeypatch.setattr(integrator, "_select_place_market", lambda: 3)
    monkeypatch.setattr(integrator, "extract_race_number_from_page", lambda _venue: 6)

    race = integrator.get_race_odds_from_page(
        {
            "race_id": "HEA_2026-07-10_6",
            "venue": "Healesville",
            "venue_slug": "healesville",
            "race_number": 6,
            "race_date": date(2026, 7, 10),
            "venue_url": (
                "https://www.sportsbet.com.au/greyhound-racing/australia-nz/"
                "healesville/race-6-10680654"
            ),
        }
    )

    assert [row["box_number"] for row in race["odds_data"]] == list(range(1, 9))
    assert [row["odds_decimal"] for row in race["odds_data"]] == [
        runner[2] for runner in runners
    ]
    assert [row["odds_decimal"] for row in race["odds_data_place"]] == [
        runner[3] for runner in runners
    ]


def test_alias_odds_copy_rolls_back_when_metadata_upsert_fails(tmp_path, monkeypatch):
    import sportsbet_odds_integrator as odds_module

    db_path = tmp_path / "alias_rollback.db"
    SportsbetOddsIntegrator(db_path=str(db_path))
    source_race_id = "SAND_2026-05-21_4"
    alias_race_id = "Race 4 - Sandown - 2026-05-21"

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, race_time
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_race_id, "Sandown", 4, "2026-05-21", "13:00"),
        )
        conn.execute(
            """
            INSERT INTO race_metadata (
                race_id, venue, race_number, race_date, race_time
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (alias_race_id, "Sandown", 4, "2026-05-21", "12:00"),
        )
        conn.execute(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, race_time, dog_name,
                dog_clean_name, box_number, odds_decimal, odds_fractional,
                market_type, source, is_current, topN
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_race_id,
                "Sandown",
                4,
                "2026-05-21",
                "13:00",
                "Fast Dog",
                "FAST DOG",
                1,
                2.5,
                "2.50",
                "win",
                "sportsbet",
                1,
                None,
            ),
        )
        conn.execute(
            """
            INSERT INTO live_odds (
                race_id, venue, race_number, race_date, race_time, dog_name,
                dog_clean_name, box_number, odds_decimal, odds_fractional,
                market_type, source, is_current, topN
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alias_race_id,
                "Sandown",
                4,
                "2026-05-21",
                "12:00",
                "Old Dog",
                "OLD DOG",
                2,
                7.5,
                "7.50",
                "win",
                "sportsbet",
                1,
                None,
            ),
        )
        conn.commit()

    def fail_metadata_upsert(*_args, **_kwargs):
        raise RuntimeError("metadata failure")

    monkeypatch.setattr(
        odds_module,
        "safe_upsert_race_metadata",
        fail_metadata_upsert,
    )

    with pytest.raises(RuntimeError, match="metadata failure"):
        _copy_current_odds_to_alias(
            str(db_path),
            source_race_id,
            alias_race_id,
            "Sandown",
            4,
            "2026-05-21",
        )

    with sqlite3.connect(db_path) as conn:
        current_alias_odds = conn.execute(
            """
            SELECT dog_clean_name, is_current
            FROM live_odds
            WHERE race_id = ?
            ORDER BY id
            """,
            (alias_race_id,),
        ).fetchall()
        race_time = conn.execute(
            "SELECT race_time FROM race_metadata WHERE race_id = ?",
            (alias_race_id,),
        ).fetchone()[0]

    assert current_alias_odds == [("OLD DOG", 1)]
    assert race_time == "12:00"


def test_auto_scrape_odds_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ENABLE_AUTO_SCRAPE_ODDS", raising=False)
    monkeypatch.delenv("AUTO_SCRAPE_ODDS_DOM_FALLBACK_MAX_CARDS", raising=False)
    flags_path = tmp_path / "feature_flags.yaml"
    monkeypatch.setattr(feature_flags, "_FEATURE_FLAGS_PATH", tmp_path / "missing.yaml")

    flags, sources = feature_flags.load_flags()

    assert flags["ENABLE_AUTO_SCRAPE_ODDS"] is False
    assert sources["ENABLE_AUTO_SCRAPE_ODDS"] == "default"
    assert feature_flags.auto_scrape_odds_enabled() is False
    assert feature_flags.auto_scrape_dom_fallback_limit() == 3

    flags_path.write_text('ENABLE_AUTO_SCRAPE_ODDS: "false"\n')
    monkeypatch.setattr(feature_flags, "_FEATURE_FLAGS_PATH", flags_path)
    yaml_flags, yaml_sources = feature_flags.load_flags()
    assert yaml_flags["ENABLE_AUTO_SCRAPE_ODDS"] is False
    assert yaml_sources["ENABLE_AUTO_SCRAPE_ODDS"] == "yaml"
    assert feature_flags.auto_scrape_odds_enabled() is False


def test_auto_odds_ensure_requires_internal_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv("ENABLE_AUTO_SCRAPE_ODDS", raising=False)
    monkeypatch.setattr(feature_flags, "_FEATURE_FLAGS_PATH", tmp_path / "missing.yaml")

    db_path = tmp_path / "no_auto_scrape.db"
    summary = odds_auto_integrator.ensure_odds_for_target_race(
        str(db_path),
        "Sandown",
        4,
        "2026-05-21",
    )

    assert summary["success"] is False
    assert summary["win_count"] == 0
    assert summary["place_count"] == 0
    assert summary["opt_in_source"] == "ENABLE_AUTO_SCRAPE_ODDS from default"
    assert summary["warnings"] == [
        "auto odds scraping disabled; ENABLE_AUTO_SCRAPE_ODDS from default"
    ]
    assert not db_path.exists()


class _FakeAnchor:
    def __init__(self, text="", href="", aria_label=""):
        self.text = text
        self._href = href
        self._aria_label = aria_label

    def get_attribute(self, name):
        if name == "href":
            return self._href
        if name == "aria-label":
            return self._aria_label
        return ""


class _FakeDriver:
    def __init__(self, *, landing_anchors=None, region_anchors=None):
        self.landing_anchors = landing_anchors or []
        self.region_anchors = region_anchors or []
        self.urls = []
        self.current_url = ""

    def get(self, url):
        self.current_url = url
        self.urls.append(url)

    def execute_script(self, *_args):
        return "complete"

    def find_elements(self, *_args):
        if self.current_url.endswith("/betting/greyhound-racing"):
            return self.landing_anchors
        if self.current_url.endswith("/greyhound-racing/australia-nz"):
            return self.region_anchors
        return []


def test_sportsbet_meeting_url_resolves_alias_from_region_page(monkeypatch):
    monkeypatch.setattr(odds_auto_integrator.time, "sleep", lambda _seconds: None)
    driver = _FakeDriver(
        region_anchors=[
            _FakeAnchor(
                text="",
                href="https://www.sportsbet.com.au/greyhound-racing/australia-nz/q1-lakeside",
                aria_label="Q1 Lakeside",
            )
        ]
    )

    resolved = odds_auto_integrator._find_meeting_url_for_target(
        driver,
        "https://www.sportsbet.com.au",
        "QOT",
    )

    assert resolved == (
        "https://www.sportsbet.com.au/greyhound-racing/australia-nz/q1-lakeside"
    )


def test_auto_odds_falls_back_to_sportsbet_meeting_when_landing_misses_target(
    monkeypatch, tmp_path
):
    import sportsbet_odds_integrator as odds_module

    monkeypatch.setattr(odds_auto_integrator.time, "sleep", lambda _seconds: None)
    driver = _FakeDriver(
        landing_anchors=[
            _FakeAnchor(
                text="R3 Horsham\n16m",
                href="https://www.sportsbet.com.au/greyhound-racing/australia-nz/horsham/race-3-10514870",
            )
        ],
        region_anchors=[
            _FakeAnchor(
                text="Horsham",
                href="https://www.sportsbet.com.au/greyhound-racing/australia-nz/horsham",
            )
        ],
    )

    class FakeIntegrator:
        instances = []

        def __init__(self, db_path, allow_auto_scrape_odds=None):
            self.db_path = db_path
            self.allow_auto_scrape_odds = allow_auto_scrape_odds
            self.base_url = "https://www.sportsbet.com.au"
            self.greyhound_url = f"{self.base_url}/betting/greyhound-racing"
            self.driver = driver
            self.selected_race_info = None
            self.meeting_calls = []
            FakeIntegrator.instances.append(self)

        def setup_driver(self):
            return True

        def find_specific_race_from_meeting(
            self, meeting_url, target_race_number, expected_venue=None
        ):
            self.meeting_calls.append(
                (meeting_url, target_race_number, expected_venue)
            )
            return (
                "https://www.sportsbet.com.au/greyhound-racing/"
                "australia-nz/horsham/race-1-10514868"
            )

        def get_race_odds_from_page(self, race_info):
            self.selected_race_info = dict(race_info)
            return {
                **race_info,
                "venue": "Horsham",
                "race_date": "2026-05-26",
                "race_number": 1,
                "odds_data": [
                    {
                        "dog_name": "Fast Dog",
                        "dog_clean_name": "FAST DOG",
                        "box_number": 1,
                        "odds_decimal": 2.5,
                    }
                ],
            }

        def _canonical_race_id(self, venue, race_date, race_number):
            return f"HOR_{race_date}_{race_number}"

        def append_pre_jump_odds_snapshot(self, race_info, odds_data, capture_mode):
            return {
                "inserted_rows": len(odds_data),
                "warnings": [],
                "capture_mode": capture_mode,
                "race_id": race_info["race_id"],
            }

        def close_driver(self):
            pass

    monkeypatch.setattr(odds_module, "SportsbetOddsIntegrator", FakeIntegrator)

    summary = odds_auto_integrator.ensure_odds_for_target_race(
        str(tmp_path / "odds.db"),
        "HOR",
        1,
        "2026-05-26",
        allow_auto_scrape_odds=True,
        append_only=True,
    )

    fake = FakeIntegrator.instances[0]
    assert summary["success"] is True
    assert summary["discovery_method"] == "sportsbet_meeting_exact_race"
    assert summary["win_count"] == 1
    assert summary["captured_rows"] == 2
    assert summary["warnings"] == []
    assert fake.meeting_calls == [
        (
            "https://www.sportsbet.com.au/greyhound-racing/australia-nz/horsham",
            1,
            "HOR",
        )
    ]
    assert fake.selected_race_info["venue_url"] == (
        "https://www.sportsbet.com.au/greyhound-racing/"
        "australia-nz/horsham/race-1-10514868"
    )


def test_fetch_odds_for_target_race_uses_read_only_integrator(monkeypatch, tmp_path):
    import sportsbet_odds_integrator as odds_module

    monkeypatch.setattr(odds_auto_integrator.time, "sleep", lambda _seconds: None)
    driver = _FakeDriver(
        landing_anchors=[
            _FakeAnchor(
                text="R1 Horsham\n16m",
                href=(
                    "https://www.sportsbet.com.au/greyhound-racing/"
                    "australia-nz/horsham/race-1-10514868"
                ),
            )
        ],
    )

    class FakeIntegrator:
        instances = []

        def __init__(
            self,
            db_path,
            allow_auto_scrape_odds=None,
            setup_database=True,
        ):
            self.db_path = db_path
            self.allow_auto_scrape_odds = allow_auto_scrape_odds
            self.setup_database = setup_database
            self.base_url = "https://www.sportsbet.com.au"
            self.greyhound_url = f"{self.base_url}/betting/greyhound-racing"
            self.driver = driver
            FakeIntegrator.instances.append(self)

        def setup_driver(self):
            return True

        def get_race_odds_from_page(self, race_info):
            return {
                **race_info,
                "venue": "Horsham",
                "race_date": "2026-05-26",
                "race_number": 1,
                "odds_data": [
                    {
                        "dog_name": "Fast Dog",
                        "dog_clean_name": "FAST DOG",
                        "box_number": 1,
                        "odds_decimal": 2.5,
                    }
                ],
                "odds_data_place": [],
            }

        def _canonical_race_id(self, venue, race_date, race_number):
            return f"HOR_{race_date}_{race_number}"

        def close_driver(self):
            pass

    monkeypatch.setattr(odds_module, "SportsbetOddsIntegrator", FakeIntegrator)
    db_path = tmp_path / "fetch_only.db"

    summary = odds_auto_integrator.fetch_odds_for_target_race(
        str(db_path),
        "HOR",
        1,
        "2026-05-26",
        allow_auto_scrape_odds=True,
    )

    fake = FakeIntegrator.instances[0]
    assert fake.setup_database is False
    assert summary["success"] is True
    assert summary["write_performed"] is False
    assert summary["win_count"] == 1
    assert summary["race_id"] == "HOR_2026-05-26_1"
    assert summary["odds_data"][0]["dog_name"] == "Fast Dog"
    assert not db_path.exists()


def test_dom_fallback_page_scraping_requires_opt_in_and_is_limited(tmp_path, monkeypatch):
    db_path = tmp_path / "dom.db"
    integrator = SportsbetOddsIntegrator(
        db_path=str(db_path),
        allow_auto_scrape_odds=False,
        dom_fallback_card_limit=2,
    )
    fallback_races = [
        {
            "race_id": f"race_{idx}",
            "venue": "Sandown",
            "race_number": idx,
            "venue_url": "url",
        }
        for idx in range(1, 4)
    ]
    calls = []

    def fake_get_race_odds_from_page(race_info):
        calls.append(race_info["race_id"])
        return {
            **race_info,
            "odds_data": [{"dog_clean_name": "FAST DOG", "odds_decimal": 2.5}],
        }

    monkeypatch.setattr(
        integrator,
        "get_race_odds_from_page",
        fake_get_race_odds_from_page,
    )

    disabled_result = integrator._enhance_dom_fallback_races(
        fallback_races,
        reason="unit-test disabled path",
    )
    assert calls == []
    assert [race["race_id"] for race in disabled_result] == ["race_1", "race_2"]
    assert all(not race.get("odds_data") for race in disabled_result)

    integrator.allow_auto_scrape_odds = True
    enabled_result = integrator._enhance_dom_fallback_races(
        fallback_races,
        reason="unit-test enabled path",
    )
    assert calls == ["race_1", "race_2"]
    assert [race["race_id"] for race in enabled_result] == ["race_1", "race_2"]
    assert all(race.get("odds_data") for race in enabled_result)


def test_integrator_can_skip_database_setup_for_read_only_fetch(tmp_path):
    db_path = tmp_path / "read_only_fetch.db"

    SportsbetOddsIntegrator(db_path=str(db_path), setup_database=False)

    assert not db_path.exists()


def test_append_pre_jump_snapshot_can_skip_race_metadata_write(tmp_path):
    db_path = tmp_path / "append_without_metadata.db"
    integrator = SportsbetOddsIntegrator(db_path=str(db_path))

    race_info = {
        "race_id": "READ_ONLY_R1",
        "preserve_race_id": True,
        "venue": "Wentworth Park",
        "race_number": 1,
        "race_date": "2026-05-24",
        "race_time": "10:30",
        "venue_url": "https://www.sportsbet.com.au/greyhound-racing/wpk-r1",
    }
    report = integrator.append_pre_jump_odds_snapshot(
        race_info,
        [
            {
                "dog_name": "Alpha Runner",
                "dog_clean_name": "Alpha Runner",
                "box_number": 1,
                "odds_decimal": 3.0,
            }
        ],
        capture_timestamp="2026-05-24T09:55:00",
        write_race_metadata=False,
    )

    assert report["status"] == "SUCCESS"
    assert report["inserted_rows"] == 1
    with sqlite3.connect(db_path) as conn:
        metadata_rows = conn.execute(
            "SELECT COUNT(*) FROM race_metadata WHERE race_id = ?",
            ("READ_ONLY_R1",),
        ).fetchone()[0]
        odds_rows = conn.execute(
            "SELECT COUNT(*) FROM live_odds WHERE race_id = ?",
            ("READ_ONLY_R1",),
        ).fetchone()[0]

    assert metadata_rows == 0
    assert odds_rows == 1


def test_ev_win_contract_uses_win_probability_times_odds_minus_one_when_odds_exist():
    import importlib.util
    import numpy as np
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ml_system_v4_real_for_odds_safety",
        repo_root / "ml_system_v4.py",
    )
    assert spec and spec.loader
    real_ml_system_v4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_ml_system_v4)
    _MLSystemV4 = real_ml_system_v4._MLSystemV4

    class FixedProbabilityModel:
        def predict_proba(self, features):
            return np.array([[0.60, 0.40], [0.80, 0.20]])

    system = _MLSystemV4.__new__(_MLSystemV4)
    system.calibrated_pipeline = FixedProbabilityModel()
    system.feature_columns = None
    system._use_iso_calibration = False
    system._iso_calibrator = None

    race_data = pd.DataFrame(
        {
            "dog_clean_name": ["DOG A", "DOG B"],
            "box_number": [1, 2],
            "weight": [30.0, 31.0],
            "distance": [515, 515],
            "historical_avg_position": [2.0, 4.0],
            "historical_win_rate": [0.25, 0.10],
            "venue_specific_avg_position": [2.5, 4.5],
            "days_since_last_race": [7, 8],
            "venue": ["Sandown", "Sandown"],
            "race_date": [date.today().isoformat(), date.today().isoformat()],
        }
    )

    result_with_odds = _MLSystemV4.predict_race(
        system,
        race_data,
        race_id="SAND_2026-05-21_4",
        market_odds={"DOG A": 4.0},
    )
    by_name = {
        prediction["dog_clean_name"]: prediction
        for prediction in result_with_odds["predictions"]
    }
    assert by_name["DOG A"]["ev_win"] == pytest.approx(
        by_name["DOG A"]["win_prob_norm"] * 4.0 - 1.0
    )
    assert "ev_win" not in by_name["DOG B"]

    result_without_odds = _MLSystemV4.predict_race(
        system,
        race_data,
        race_id="SAND_2026-05-21_4",
        market_odds=None,
    )
    assert all(
        "ev_win" not in prediction
        for prediction in result_without_odds["predictions"]
    )
