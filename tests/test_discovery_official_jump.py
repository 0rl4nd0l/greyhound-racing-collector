"""Official calendar date and absolute jump time have distinct meanings."""
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import json

from bs4 import BeautifulSoup
import pytest

import upcoming_race_browser as browser_module
from upcoming_race_browser import UpcomingRaceBrowser
from scripts.refresh_prejump_upcoming import _parse_race_jump_datetime

ZONE = ZoneInfo('Australia/Melbourne')
URL = 'https://www.thedogs.com.au/racing/cannington/2026-10-03/11/tabtouch-download-today'
# Exact timing metadata retained from official canonical response on October3.
JUMP = datetime(2026, 10, 4, 0, 13, tzinfo=ZONE)
ELEMENT = f'<formatted-time data-format="datetime_short" data-timestamp="{int(JUMP.timestamp())}">00:13 04 Oct</formatted-time>'


class Response:
    status_code = 200
    def __init__(self, body):
        self.content = body.encode()
        self.text = body
    def close(self):
        pass


def discover(monkeypatch, tmp_path, *, meeting=False, element=ELEMENT):
    browser = UpcomingRaceBrowser()
    browser.upcoming_dir = str(tmp_path)
    browser.bounded_meeting_discovery = meeting
    clock = element.replace('datetime_short', 'time_24') if meeting else ''
    card = f'<a href="{URL}">R11{clock}</a>'
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return Response(card if url.endswith('/racing/2026-10-03') else element)
    monkeypatch.setattr(browser.session, 'get', get)
    rows = browser._scrape_live_races_for_date('2026-10-03')
    return browser, rows, calls


@pytest.mark.parametrize('meeting', [False, True])
def test_official_midnight_jump_survives_discovery_and_filter(monkeypatch, tmp_path, meeting):
    browser, rows, calls = discover(monkeypatch, tmp_path, meeting=meeting)
    assert len(rows) == 1
    assert rows[0]['date'] == '2026-10-03'
    assert rows[0]['scheduled_jump_datetime'] == JUMP.isoformat()
    assert _parse_race_jump_datetime(rows[0], now=datetime(2026,10,3,13,5,tzinfo=ZONE)) == JUMP
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026,10,3,13,5,tzinfo=tz)
    monkeypatch.setattr(browser_module, 'datetime', FixedClock)
    monkeypatch.setattr(browser, 'get_races_for_date', lambda date: rows)
    assert browser.get_upcoming_races(0) == rows
    assert len(calls) == (1 if meeting else 2)


def test_explicit_pm_suffix_preserved():
    browser = UpcomingRaceBrowser.__new__(UpcomingRaceBrowser)
    soup = BeautifulSoup('<formatted-time data-format="datetime_short">1:41 PM</formatted-time>', 'html.parser')
    assert browser._extract_formatted_race_time(soup) == '1:41 PM'


@pytest.mark.parametrize('value', ['2026-10-04T00:13:00', 'bad', '2026-10-06T00:13:00+10:00'])
def test_invalid_absolute_jump_never_falls_back_to_clock(value):
    assert _parse_race_jump_datetime({'date':'2026-10-03','race_time':'12:13 AM','scheduled_jump_datetime':value}, now=datetime(2026,10,3,13,5,tzinfo=ZONE)) is None


@pytest.mark.parametrize('meeting', [False, True])
def test_conflicting_official_epochs_fail_closed(monkeypatch, tmp_path, meeting):
    bad = ELEMENT + ELEMENT.replace(str(int(JUMP.timestamp())), str(int(JUMP.timestamp())+60))
    _, rows, _ = discover(monkeypatch, tmp_path, meeting=meeting, element=bad)
    assert not rows or rows[0]['race_time'] is None


def test_sidecar_preserves_jump_instant_and_source_racing_day(tmp_path):
    from utils.csv_metadata import build_csv_download_provenance_payload
    from utils.race_lifecycle import classify_race_file, UPCOMING_NOT_JUMPED
    import json
    info = {'date':'2026-10-03','venue':'CAN','race_number':11,'race_time':'12:13 AM','scheduled_jump_datetime':JUMP.isoformat()}
    payload = build_csv_download_provenance_payload(filepath=tmp_path/'sample.csv', completeness={'status':'COMPLETE'}, race_url=URL, csv_info=URL+'/export-expert-form', content='Dog Name\n', race_info=info)
    assert payload['race_info']['scheduled_jump_datetime'] == JUMP.isoformat()
    csv = tmp_path/'Race 11 - CAN - 2026-10-03.csv'
    csv.write_text('Dog Name\n')
    Path(str(csv)+'.metadata.json').write_text(json.dumps(payload))
    lifecycle = classify_race_file(csv, now=datetime(2026,10,3,23,50,tzinfo=ZONE))
    assert lifecycle.status == UPCOMING_NOT_JUMPED
    assert lifecycle.race_date == '2026-10-03'
    assert lifecycle.jump_datetime == JUMP.isoformat()


def test_live_epoch_replaces_cached_timing_without_changing_identity(monkeypatch, tmp_path):
    browser, live, _ = discover(monkeypatch, tmp_path)
    cached = {**live[0], 'race_time':'12:13 AM'}
    cached.pop('scheduled_jump_datetime')
    browser.enhance_limit = 0
    monkeypatch.setattr(browser, '_get_cached_races_for_date', lambda date:[cached])
    monkeypatch.setattr(browser, '_scrape_live_races_for_date', lambda date:live)
    rows = browser.get_races_for_date(datetime(2026,10,3))
    assert len(rows) == 1
    assert rows[0]['date'] == '2026-10-03'
    assert rows[0]['scheduled_jump_datetime'] == JUMP.isoformat()
    assert _parse_race_jump_datetime(rows[0]) == JUMP


def test_invalid_live_timing_does_not_reuse_cached_clock(monkeypatch):
    browser = UpcomingRaceBrowser()
    browser.enhance_limit = 0
    cached = {'date':'2026-10-03', 'venue':'CAN', 'race_number':'11', 'race_time':'11:00 PM'}
    live = {**cached, 'race_time':None, 'race_time_mapping_status':'invalid_official_timing'}
    monkeypatch.setattr(browser, '_get_cached_races_for_date', lambda date:[cached])
    monkeypatch.setattr(browser, '_scrape_live_races_for_date', lambda date:[live])
    rows = browser.get_races_for_date(datetime(2026,10,3))
    assert rows[0]['race_time'] is None


def test_empty_live_calendar_cannot_claim_fresh_cached_inventory(monkeypatch):
    browser = UpcomingRaceBrowser()
    monkeypatch.setattr(browser.session, 'get', lambda *args,**kwargs:Response('<html></html>'))
    assert browser._scrape_live_races_for_date('2026-10-03') == []
    assert browser.discovery_failures[0]['error_type'] == 'EmptyDiscovery'


def test_unparsed_calendar_link_is_a_discovery_failure(monkeypatch, tmp_path):
    browser, _, _ = discover(monkeypatch, tmp_path)
    monkeypatch.setattr(browser, 'extract_race_info_from_link', lambda *args,**kwargs:None)
    assert browser._scrape_live_races_for_date('2026-10-03') == []
    assert browser.discovery_failures[0]['error_type'] == 'RaceIdentityUnavailable'


def test_index_accepts_only_explicit_consistent_next_day_jump():
    from scripts.refresh_prejump_upcoming import race_window_record
    from race_collection.synchronous_manual_capture import _normalize_current_index_rows, CaptureOneRejected
    row = race_window_record({'date':'2026-10-03','venue':'CAN','race_number':11,'race_time':'12:13 AM','scheduled_jump_datetime':JUMP.isoformat(),'url':URL}, now=datetime(2026,10,3,23,50,tzinfo=ZONE))
    row['source_native_race_id'] = '123456'
    normalized = _normalize_current_index_rows({'selected_count':1,'selected_races':[row]}, max_races=32)
    assert normalized[0]['date'] == '2026-10-03'
    assert normalized[0]['scheduled_jump_datetime'] == JUMP.isoformat()
    row.pop('scheduled_jump_datetime')
    with pytest.raises(CaptureOneRejected):
        _normalize_current_index_rows({'selected_count':1,'selected_races':[row]}, max_races=32)
