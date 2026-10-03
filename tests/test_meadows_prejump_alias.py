"""Exact-page venue identity must use the same existing mapping as discovery."""
from datetime import datetime
from urllib.parse import urlparse,parse_qs
from zoneinfo import ZoneInfo

import pytest
from bs4 import BeautifulSoup

from upcoming_race_browser import UpcomingRaceBrowser
from utils.prejump_weather import venue_weather_location,collect_open_meteo_weather_metadata
from utils.prejump_sportsbet import collect_sportsbet_track_metadata
from tests.test_prejump_weather_track_metadata import FakeWeatherSession,_weather_payload
from tests.test_prejump_sportsbet import FakeSportsbetSession,_sale_r9_event


def parsed_race(slug='meadows',date='2026-10-03'):
    browser=object.__new__(UpcomingRaceBrowser)
    browser.venue_map={'the-meadows':'MEA'}
    url=f'https://www.thedogs.com.au/racing/{slug}/{date}/6/fixture'
    soup=BeautifulSoup('<html><title>Meadows Race 6</title></html>','html.parser')
    race=browser.extract_detailed_race_info(soup,url)
    return {**race,'url':url,'race_time':'8:17 PM'}


@pytest.mark.parametrize('slug',['meadows','the-meadows'])
def test_exact_page_preserves_existing_canonical_meadows_location(slug):
    race=parsed_race(slug)
    assert race['venue']=='MEA'
    location=venue_weather_location(race['venue'])
    assert location.timezone=='Australia/Melbourne'
    assert (location.latitude,location.longitude)==(-37.6822,144.9528)


def test_unrecognized_venue_does_not_fuzzily_become_meadows():
    race=parsed_race('meadows-unrecognized-track')
    assert race['venue']!='MEA'
    assert venue_weather_location(race['venue']) is None


def test_exact_page_meadows_identity_reaches_existing_weather_collector():
    session=FakeWeatherSession(_weather_payload())
    race={**parsed_race(date='2026-06-18'),'race_time':'7:12 PM'}
    result=collect_open_meteo_weather_metadata(race,session=session)
    assert result['weather_track_metadata_is_leakage_safe'] is True
    assert len(session.calls)==1
    params=parse_qs(urlparse(session.calls[0][0]).query)
    assert float(params['latitude'][0])==-37.6822 and float(params['longitude'][0])==144.9528
    assert params['timezone']==['Australia/Melbourne']


def test_exact_page_meadows_identity_matches_only_exact_sportsbet_event():
    event=_sale_r9_event(competitionName='The Meadows',raceNumber=6,
        startTime=int(datetime(2026,10,3,20,17,tzinfo=ZoneInfo('Australia/Melbourne')).timestamp()))
    race=parsed_race()
    result=collect_sportsbet_track_metadata(race,session=FakeSportsbetSession([event]))
    assert result['weather_track_metadata_is_leakage_safe'] is True
    assert result['track_condition']=='Good'
    rejected=collect_sportsbet_track_metadata(race,
        session=FakeSportsbetSession([{**event,'competitionName':'Unrelated Track'}]))
    assert 'track_condition' not in rejected
    assert rejected.get('weather_track_metadata_is_leakage_safe') is not True
