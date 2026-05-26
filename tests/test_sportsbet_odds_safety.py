import sqlite3
from datetime import date
from pathlib import Path

import pytest

import odds_auto_integrator
from odds_auto_integrator import _copy_current_odds_to_alias
from sportsbet_odds_integrator import SportsbetOddsIntegrator
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
            SELECT race_id, source_url, capture_mode, dog_clean_name, box_number
            FROM live_odds
            """
        ).fetchone()

    assert row == (
        "HOR_2026-05-26_7",
        source_url,
        "opt_in_live_pre_jump_snapshot",
        "PABLO VENDETTA",
        1,
    )


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
