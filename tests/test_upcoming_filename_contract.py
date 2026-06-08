from datetime import date

from scripts import (
    alias_upcoming_api_names_safe,
    archive_past_upcoming,
    normalize_upcoming_csvs,
    normalize_upcoming_to_api_pattern,
    validate_upcoming_races,
)
from upcoming_race_browser import UpcomingRaceBrowser
from utils.csv_metadata import _extract_from_filename
from utils.race_lifecycle import extract_target_metadata_from_filename


HYPHENATED_VENUE_NAME = "Race 1 - LADBROKES-Q1-LAKESIDE - 2026-05-29.csv"


class FixedDate(date):
    @classmethod
    def today(cls):
        return date(2026, 5, 29)


def test_hyphenated_alphanumeric_venue_filename_is_contract_compliant(
    tmp_path, monkeypatch
):
    path = tmp_path / HYPHENATED_VENUE_NAME
    path.write_text(
        "Dog Name|BOX\n"
        "Alpha Runner|1\n"
        "Bravo Runner|2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(validate_upcoming_races, "date", FixedDate)

    race_no, venue, race_date, problems = validate_upcoming_races.validate_filename(path)

    assert problems == []
    assert race_no == 1
    assert venue == "LADBROKES-Q1-LAKESIDE"
    assert race_date == date(2026, 5, 29)
    assert validate_upcoming_races.validate_file(path, strict_future=False) == []

    parsed_no, parsed_venue, parsed_date = archive_past_upcoming.parse_filename(path)
    assert parsed_no == 1
    assert parsed_venue == "LADBROKES-Q1-LAKESIDE"
    assert parsed_date == date(2026, 5, 29)

    assert normalize_upcoming_csvs.FILENAME_PATTERN.match(path.name).group(2) == (
        "LADBROKES-Q1-LAKESIDE"
    )
    assert normalize_upcoming_to_api_pattern.extract_meta(path.name) == {
        "race": 1,
        "venue": "LADBROKES-Q1-LAKESIDE",
        "date": "2026-05-29",
    }
    assert alias_upcoming_api_names_safe.extract_from_name(path.name) == (
        1,
        "LADBROKES-Q1-LAKESIDE",
        "2026-05-29",
    )
    assert extract_target_metadata_from_filename(path)["venue"] == (
        "LADBROKES-Q1-LAKESIDE"
    )


def test_legacy_filename_metadata_parsers_preserve_hyphenated_alphanumeric_venue(
    tmp_path,
):
    browser = object.__new__(UpcomingRaceBrowser)
    browser.base_url = "https://www.thedogs.com.au"
    browser.upcoming_dir = str(tmp_path)
    browser.venue_map = {}

    info = browser.extract_race_info_from_csv_filename(
        HYPHENATED_VENUE_NAME, "2026-05-29"
    )

    assert info is not None
    assert info["race_number"] == 1
    assert info["venue"] == "LADBROKES-Q1-LAKESIDE"
    assert info["venue_name"] == "Ladbrokes Q1 Lakeside"
    assert info["url"] == (
        "https://www.thedogs.com.au/racing/ladbrokes-q1-lakeside/2026-05-29/1"
    )
    assert _extract_from_filename(HYPHENATED_VENUE_NAME) == {
        "race_number": 1,
        "venue": "LADBROKES-Q1-LAKESIDE",
        "race_date": "2026-05-29",
    }
