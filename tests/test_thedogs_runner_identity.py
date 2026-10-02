from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from utils.runner_completeness import extract_canonical_runner_set_from_html
from utils.thedogs_runner_identity import (
    TheDogsRunnerIdentityError,
    extract_thedogs_profile_identity,
)


def row(markup):
    return BeautifulSoup(markup, "html.parser").select_one("tr")


def test_distinct_entry_and_profile_namespaces():
    markup = '<tr data-dog-id="123"><td data-runner-id="789"><a href="/dogs/runner/789/odds">entry</a><a href="/dogs/123/example">profile</a></td></tr>'
    assert extract_thedogs_profile_identity(row(markup), require_profile_link=True) == "123"


@pytest.mark.parametrize("prefix", ["", "https://www.thedogs.com.au", "https://thedogs.com.au"])
def test_corroborating_profile_link(prefix):
    markup = f'<tr><td data-dog-id="123"><a href="{prefix}/dogs/123/example">dog</a></td></tr>'
    assert extract_thedogs_profile_identity(row(markup), require_profile_link=True) == "123"


@pytest.mark.parametrize("identity", ["", "0", "0123", "-1", "abc", "１２３"])
def test_missing_or_invalid_profile_id(identity):
    with pytest.raises(TheDogsRunnerIdentityError, match="^dog_profile_id_missing_invalid_or_ambiguous$"):
        extract_thedogs_profile_identity(row(f'<tr data-dog-id="{identity}"></tr>'), require_profile_link=False)


def test_conflicting_same_row_profile_attributes():
    with pytest.raises(TheDogsRunnerIdentityError, match="^dog_profile_id_missing_invalid_or_ambiguous$"):
        extract_thedogs_profile_identity(row('<tr data-dog-id="123"><td data-dog-id="456"></td></tr>'), require_profile_link=False)


@pytest.mark.parametrize("link", [
    "/dogs/456/example", "/dogs/abc/example", "/dogs/0123/example",
    "https://evil.invalid/dogs/123/example", "http://www.thedogs.com.au/dogs/123/example",
    "https://www.thedogs.com.au.evil.invalid/dogs/123/example",
    "https://user@www.thedogs.com.au/dogs/123/example", "/dogs/123/example?alternate=1",
    "/dogs/123/example#alternate", "https:///dogs/123/example",
])
def test_invalid_or_conflicting_profile_link_fails_even_when_optional(link):
    with pytest.raises(TheDogsRunnerIdentityError):
        extract_thedogs_profile_identity(row(f'<tr data-dog-id="123"><td><a href="{link}">dog</a></td></tr>'), require_profile_link=False)


def test_official_requires_profile_corroboration():
    entry_only = row('<tr data-dog-id="123"><td><a href="/dogs/runner/789">entry</a></td></tr>')
    assert extract_thedogs_profile_identity(entry_only, require_profile_link=False) == "123"
    with pytest.raises(TheDogsRunnerIdentityError, match="^dog_profile_link_missing$"):
        extract_thedogs_profile_identity(entry_only, require_profile_link=True)


def test_retained_live_prerace_page_has_same_row_bridge_without_profile_links():
    fixture = Path(__file__).parent / "fixtures/thedogs_live_20260825/8151dedd4cf52bfe409791fa3286f1ac5b6ba56b2f41237f57faa87e35f7adbf.race-page.html"
    html = fixture.read_text()
    canonical = extract_canonical_runner_set_from_html(html)
    expected_entries = {r["source_native_runner_id"] for r in canonical["final_runner_participants"]}
    bridge = {}
    for source_row in BeautifulSoup(html, "html.parser").select("tr.race-runner"):
        dog_id = extract_thedogs_profile_identity(source_row, require_profile_link=False)
        ids = {str(e.get("data-runner-id")) for e in source_row.select("[data-runner-id]")}
        if len(ids) == 1:
            bridge[next(iter(ids))] = dog_id
        with pytest.raises(TheDogsRunnerIdentityError, match="^dog_profile_link_missing$"):
            extract_thedogs_profile_identity(source_row, require_profile_link=True)
    assert expected_entries <= bridge.keys()
    assert len({bridge[entry] for entry in expected_entries}) == len(expected_entries)
    assert expected_entries.isdisjoint(bridge.values())
