"""Prospective regression from CANN5's retained pre-jump header only.

Original page SHA256: 033059795a324a9971b3b96845900ba7c4d8441f7e43dcc5a9e546eea5f05cd4.
This reduced fixture contains no runners, odds, histories or results. It is not
a replacement source receipt and cannot retroactively admit the original race.
"""

import hashlib

import pytest
from bs4 import BeautifulSoup

from upcoming_race_browser import UpcomingRaceBrowser
from utils.csv_metadata import (
    normalize_exact_target_grade,
    normalize_target_grade,
    target_grade_equivalence_key,
    verify_canonical_sidecar_payload,
)


RACE_URL = (
    "https://www.thedogs.com.au/racing/cannington/2026-10-03/5/"
    "get-your-quaddie-on?trial=false"
)
HEADER = (
    '<div class="race-header"><span class="race-box__number">R5</span>'
    '<div class="race-header__info__grade">NOV 601m</div>'
    '<div class="race-header__info__name">Get Your Quaddie On!</div></div>'
)


def parsed_header(html=HEADER, source_hash=None):
    browser = object.__new__(UpcomingRaceBrowser)
    browser.venue_map = {}
    return browser._extract_safe_target_metadata_from_page(
        BeautifulSoup(html, "html.parser"),
        RACE_URL,
        source_sha256=source_hash or hashlib.sha256(html.encode()).hexdigest(),
    )


def canonical_input(tmp_path, parsed):
    return verify_canonical_sidecar_payload(
        {
            **parsed,
            "race_url": RACE_URL,
            "race_time_mapping_status": "exact_url_match",
            "race_time_source": "canonical_race_url",
        },
        csv_path=tmp_path / "Race 5 - CANN - 2026-10-03.csv",
    )


def test_retained_nov_header_reaches_verified_target_input(tmp_path):
    parsed = parsed_header()
    verified = canonical_input(tmp_path, parsed)
    assert verified["target_metadata_status"] == "verified", verified[
        "target_metadata_failure_reason"
    ]
    assert verified["target_grade"] == "Novice"
    assert verified["target_distance"] == "601m"
    assert parsed["target_grade_source"] == "thedogs_exact_race_page"
    assert parsed["target_grade_equivalence_key"] == "NOVICE"
    assert parsed["target_grade_race_number"] == 5
    assert parsed["target_grade_race_date"] == "2026-10-03"
    assert parsed["target_grade_source_sha256"] == hashlib.sha256(
        HEADER.encode()
    ).hexdigest()


@pytest.mark.parametrize("value", ["NOV", "nov", "NOV 601m", " Nov 601m "])
def test_explicit_nov_alias_matches_existing_novice_semantics(value):
    assert normalize_target_grade(value) == "Novice"
    assert normalize_exact_target_grade(value) == "Novice"
    assert target_grade_equivalence_key(value) == "NOVICE"


@pytest.mark.parametrize("value", ["NOVEL", "UNKNOWN CLASS", "NOV/5", "NOV UNKNOWN"])
def test_unknown_or_compound_nov_header_stays_rejected(tmp_path, value):
    parsed = parsed_header(HEADER.replace("NOV 601m", f"{value} 601m"))
    assert canonical_input(tmp_path, parsed)["target_metadata_status"] == "missing"
    assert normalize_target_grade(value) is None
    assert normalize_exact_target_grade(value) is None


@pytest.mark.parametrize("value", ["NOV/OPEN", "NOV MAIDEN", "NOV-Grade 5"])
def test_combined_grade_does_not_become_an_exact_nov_alias(value):
    assert normalize_exact_target_grade(value) is None
    assert target_grade_equivalence_key(value) is None


@pytest.mark.parametrize("html", [HEADER.replace("R5", "R6"), HEADER + HEADER])
def test_alias_does_not_weaken_header_race_identity(tmp_path, html):
    verified = canonical_input(tmp_path, parsed_header(html))
    assert verified["target_metadata_status"] != "verified"


@pytest.mark.parametrize(
    "field,value",
    [
        ("target_grade_race_number", 6),
        ("target_grade_venue", "MAND"),
        ("target_grade_race_date", "2026-10-04"),
        ("target_grade_source_sha256", "not-a-sha256"),
        (
            "target_grade_source_url",
            "https://www.thedogs.com.au/racing/cannington/2026-10-03/6",
        ),
    ],
)
def test_alias_keeps_exact_source_proof_required(tmp_path, field, value):
    parsed = parsed_header()
    assert canonical_input(tmp_path, parsed)["target_metadata_status"] == "verified"
    parsed[field] = value
    assert canonical_input(tmp_path, parsed)["target_metadata_status"] != "verified"
