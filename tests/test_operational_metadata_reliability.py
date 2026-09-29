"""Regression cases from the fixed September 28 exclusion set; invented inputs."""
import json

import pytest
from bs4 import BeautifulSoup

from upcoming_race_browser import UpcomingRaceBrowser
from utils.csv_metadata import canonical_thedogs_venue_identity
from tests.test_refresh_shared_sportsbet_snapshot import fixture, access, refresh


@pytest.mark.parametrize('number,header,grade', [(1, 'Maiden 515m', 'Maiden'), (2, '6th Grade 515m', 'Grade 6')])
def test_launceston_explicit_grade_survives_exact_provenance(number, header, grade):
    # Only the current pre-jump header is projected from the observed structure.
    browser = object.__new__(UpcomingRaceBrowser)
    browser.venue_map = {}
    url = f'https://www.thedogs.com.au/racing/launceston/2026-09-28/{number}/invented?trial=false'
    soup = BeautifulSoup(f'<div class="race-header"><span class="race-box__number">R{number}</span><div class="race-header__info__grade">{header}</div></div>', 'html.parser')
    parsed = browser._extract_safe_target_metadata_from_page(soup, url, source_sha256='a'*64)
    assert parsed.get('target_grade') == grade
    assert parsed['metadata_is_leakage_safe'] is True
    assert len({canonical_thedogs_venue_identity(alias) for alias in ('LCTN', 'LAU', 'LAUNCESTON')}) == 1
    assert canonical_thedogs_venue_identity('HOBT') != canonical_thedogs_venue_identity('LCTN')


@pytest.mark.parametrize('slug,name', [('maitland','Maitland'), ('grafton','Grafton'), ('launceston','Launceston')])
def test_missing_venue_metadata_through_spawned_refresh(tmp_path, slug, name):
    http = fixture(tmp_path, 1)
    # Existing acquisition/parser/publication path, with a genuinely supplied
    # forecast and track status. No default weather or track value is injected.
    http.write_text(http.read_text().replace('/sale/', f'/{slug}/').replace('Sale', name))
    report = refresh(tmp_path, http, access(tmp_path), 1, lane='full')
    assert report['current_index_race_count'] == 1, report['current_index_metadata_selection']
    coverage = report['sidecar_metadata_coverage']['races'][0]
    assert coverage['safe_all_weather_track_expert_form_present'] is True
    assert coverage['source_native_race_id'] == '99000'


def test_q_straight_is_not_a_lakeside_or_parklands_alias():
    from scripts.refresh_prejump_upcoming import VENUE_EXCLUSION_ALIAS_GROUPS
    straight = next(group for group in VENUE_EXCLUSION_ALIAS_GROUPS if 'QOT' in group)
    assert 'LADBROKES-Q-STRAIGHT' in straight
    assert not straight.intersection({'Q1', 'Q2', 'LADBROKES-Q1-LAKESIDE', 'LADBROKES-Q2-PARKLANDS'})
    assert canonical_thedogs_venue_identity('LADBROKES-Q-STRAIGHT') == canonical_thedogs_venue_identity('QOT')
    assert len({canonical_thedogs_venue_identity(value) for value in (
        'LADBROKES-Q-STRAIGHT', 'LADBROKES-Q1-LAKESIDE', 'LADBROKES-Q2-PARKLANDS')}) == 3


@pytest.mark.parametrize('missing', ['weather', 'track', 'identity'])
def test_mapping_does_not_invent_required_evidence(tmp_path, missing):
    http = fixture(tmp_path, 1)
    data = json.loads(http.read_text().replace('/sale/', '/maitland/').replace('Sale', 'Maitland'))
    for key, value in data['responses'].items():
        if missing == 'weather':
            value['body'] = value['body'].replace('<dd>Fine</dd>', '<dd>Unknown</dd>')
            if 'open-meteo' in key:
                value['body'] = '{}'
        elif missing == 'track' and 'NextEvents' in key:
            events = json.loads(value['body'])
            for event in events:
                event.pop('trackStatus', None)
            value['body'] = json.dumps(events)
        elif missing == 'identity':
            value['body'] = value['body'].replace('data-race-id="99000"', '')
    http.write_text(json.dumps(data))
    report = refresh(tmp_path, http, access(tmp_path), 1, lane='full')
    assert report['current_index_race_count'] == 0
    assert report['current_index_metadata_selection']['excluded_race_count'] == 1


@pytest.mark.parametrize('slug', ['ladbrokes-q-straight', 'ladbrokes-q1-lakeside', 'ladbrokes-q2-parklands'])
def test_exact_grade_proof_preserves_distinct_q_venue(slug):
    from utils.csv_metadata import build_safe_target_metadata_payload
    browser = object.__new__(UpcomingRaceBrowser)
    browser.venue_map = {}
    url = f'https://www.thedogs.com.au/racing/{slug}/2026-09-29/6/invented?trial=false'
    soup = BeautifulSoup('<div class="race-header"><span class="race-box__number">R6</span><div class="race-header__info__grade">6th Grade 390m</div></div>', 'html.parser')
    parsed = browser._extract_safe_target_metadata_from_page(soup, url, source_sha256='a'*64)
    assert parsed.get('target_grade') == 'Grade 6'
    assert parsed['metadata_is_leakage_safe'] is True
    assert parsed['target_grade_venue'] == canonical_thedogs_venue_identity(slug)
    assert browser._canonical_thedogs_race_identity(url)[3] == parsed['target_grade_venue']
    wrong = 'ladbrokes-q2-parklands' if slug != 'ladbrokes-q2-parklands' else 'ladbrokes-q1-lakeside'
    tampered = {**parsed, 'target_grade_venue': canonical_thedogs_venue_identity(wrong)}
    assert build_safe_target_metadata_payload(tampered, source_url=url)['target_grade'] is None
