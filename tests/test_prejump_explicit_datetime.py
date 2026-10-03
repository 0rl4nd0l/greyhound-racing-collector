"""Prospective metadata-only regression for retained CANN11 timing, no live data."""
from datetime import datetime
from urllib.parse import parse_qs, urlsplit

import pytest

from utils.expert_form_metadata import build_expert_form_metadata_payload
from utils.prejump_weather import collect_open_meteo_weather_metadata
from utils.prejump_sportsbet import collect_sportsbet_track_metadata

RACE = {
    'date': '2026-10-03', 'venue': 'CANN', 'race_number': '11',
    'race_time': '12:13 AM', 'scheduled_jump_datetime': '2026-10-04T00:13:00+10:00',
}
CAPTURE = '2026-10-03T13:12:45.231525Z'
# Minimal invented runner fixture; no retained form rows or outcomes.
HTML = '<div class="layout--sidebar--expert"><div class="expert-form-runner__details__dog__name">Fixture Dog</div></div>'
URL = 'https://www.thedogs.com.au/racing/cannington/2026-10-03/11/tabtouch-download-today/expert-form'

class Response:
    def raise_for_status(self): pass
    def close(self): pass
    def json(self): return {'hourly': {'time': ['2026-10-03T22:00'], 'weather_code': [0]}}

class Session:
    def __init__(self): self.urls = []
    def get(self, url, **kwargs): self.urls.append(url); return Response()

def expert(race=RACE, captured_at=CAPTURE):
    return build_expert_form_metadata_payload(HTML, race_info=race, source_url=URL, captured_at=captured_at)

def track(race=RACE, **changes):
    event = {'id': 1, 'type': 'greyhound', 'classId': '4', 'competitionName': 'Cannington',
             'raceNumber': 11, 'startTime': datetime.fromisoformat(RACE['scheduled_jump_datetime']).timestamp(),
             'trackStatus': 'Good', **changes}
    return collect_sportsbet_track_metadata(race, snapshot={'events': [event]})

def test_retained_cann11_time_accepts_prejump_expert_fixture():
    result = expert()
    assert result['metadata_is_leakage_safe'] is True, result['rejected_reasons']

def test_retained_cann11_weather_uses_actual_perth_day():
    session = Session()
    result = collect_open_meteo_weather_metadata(RACE, session=session)
    query = parse_qs(urlsplit(session.urls[0]).query)
    assert query['start_date'] == ['2026-10-03']
    assert result['weather_track_metadata_detail']['race_time_venue_local'] == '2026-10-03T22:13:00+08:00'

def test_retained_cann11_track_matches_correct_instant():
    assert track().get('track_condition') == 'Good'

@pytest.mark.parametrize('change', [
    {'scheduled_jump_datetime': '2026-10-04T00:13:00'},
    {'scheduled_jump_datetime': 'invalid'},
    {'scheduled_jump_datetime': None},
    {'scheduled_jump_datetime': '2026-10-05T00:13:00+10:00'},
    {'scheduled_jump_datetime': '2026-10-02T00:13:00+10:00'},
    {'race_time': '12:14 AM'}, {'race_time': ''},
    {'jump_time': '12:14 AM'},
    {'scheduled_jump_datetime': '2026-10-04T00:13:01+10:00'},
    {'display_timezone': 'Australia/Perth'}, {'date': 'invalid'},
])
def test_invalid_explicit_evidence_never_falls_back(change):
    race = {**RACE, **change}
    assert 'expert_form_jump_time_unverified' in expert(race)['rejected_reasons']
    session = Session()
    weather = collect_open_meteo_weather_metadata(race, session=session)
    assert weather['rejected_weather_track_metadata_sources'] == ['weather_race_time_unparseable']
    assert session.urls == []
    assert track(race)['rejected_weather_track_metadata_sources'] == ['sportsbet_race_time_unparseable']

@pytest.mark.parametrize('captured', ['2026-10-03T14:13:00Z', '2026-10-03T14:13:01Z'])
def test_at_or_after_real_jump_remains_rejected(captured):
    assert 'expert_form_metadata_captured_at_not_before_jump' in expert(captured_at=captured)['rejected_reasons']

@pytest.mark.parametrize('changes', [{'raceNumber': 10}, {'competitionName': 'Sale'}, {'startTime': datetime.fromisoformat(RACE['scheduled_jump_datetime']).timestamp() - 86400}])
def test_track_identity_and_date_stay_strict(changes):
    assert track(**changes).get('track_condition') is None

def test_equivalent_utc_timestamp_is_same_authoritative_instant():
    assert expert({**RACE, 'scheduled_jump_datetime': '2026-10-03T14:13:00Z'})['metadata_is_leakage_safe'] is True

def test_same_day_explicit_time_and_legacy_expert_path():
    race = {**RACE, 'race_time': '11:13 PM', 'scheduled_jump_datetime': '2026-10-03T23:13:00+10:00'}
    assert expert(race)['metadata_is_leakage_safe'] is True
    legacy = {'date': '2026-10-03', 'venue': 'CANN', 'race_time': '10:13 PM'}
    assert expert(legacy)['metadata_is_leakage_safe'] is True
