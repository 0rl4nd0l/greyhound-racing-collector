"""Distinct Queensland layouts remain exact from discovery to sealed forecast."""
from datetime import datetime
import pytest
from scripts.predict_race_now import _request_race
from src.predictor.on_demand import PredictionBlocked, validate_prediction_result_v2
from src.predictor.receipt_preflight import _sportsbet_source_matches
from tests.test_prediction_bundle_sealed import ready_result
from utils.race_identity_equivalence import configured_venue_identity, race_identity_equivalent
from config.venue_mapping import normalize_venue

@pytest.mark.parametrize('layout,slug,sportsbet', [('Q1','ladbrokes-q1-lakeside','q1-lakeside'),('Q2','ladbrokes-q2-parklands','q2-parklands')])
@pytest.mark.parametrize('change',['none','other_layout','straight','wrong_number','wrong_date','unknown'])
def test_layout_request_and_sealed_result_require_exact_source(layout,slug,sportsbet,change):
    venue=slug.upper();race_id=f'Race 7 - {venue} - 2026-07-19'
    url=f'https://www.thedogs.com.au/racing/{slug}/2026-07-19/7/example-stake'
    source_slug=slug
    if change=='other_layout':source_slug='ladbrokes-q2-parklands' if layout=='Q1' else 'ladbrokes-q1-lakeside'
    if change=='straight':source_slug='ladbrokes-q-straight'
    if change=='unknown':source_slug=slug+'-unknown'
    url=url.replace(slug,source_slug)
    if change=='wrong_number':url=url.replace('/7/','/8/')
    if change=='wrong_date':url=url.replace('2026-07-19','2026-07-20')
    target={'race_number':7,'date':'2026-07-19','venue':venue,'race_url':url}
    result=ready_result();result['race'].update(race_id=race_id,url=url,venue=venue,venue_slug=source_slug,race_number=7)
    assert normalize_venue(venue)=='QOT'
    if change=='none':
        assert configured_venue_identity(venue)==layout
        assert _request_race(target,race_id=race_id,jump=datetime.fromisoformat('2026-07-19T13:00:00+10:00'))['venue']==venue
        assert validate_prediction_result_v2(result)['race']==result['race']
        assert race_identity_equivalent(race_id,race_id,source_url=url)
        assert _sportsbet_source_matches(f'https://www.sportsbet.com.au/greyhound-racing/australia-nz/{sportsbet}/race-7-12345',race_id)
    else:
        with pytest.raises(PredictionBlocked,match='EXACT_RACE_IDENTITY_UNAVAILABLE'):
            _request_race(target,race_id=race_id,jump=datetime.fromisoformat('2026-07-19T13:00:00+10:00'))
        with pytest.raises(PredictionBlocked,match='PREDICTION_BUNDLE_INVALID'):
            validate_prediction_result_v2(result)
        assert not race_identity_equivalent(race_id,race_id,source_url=url)

@pytest.mark.parametrize('caller,source',[('LADBROKES-Q1-LAKESIDE','q2-parklands'),('LADBROKES-Q1-LAKESIDE','q-straight'),('LADBROKES-Q2-PARKLANDS','q1-lakeside'),('LADBROKES-Q2-PARKLANDS','q-straight'),('QOT','q1-lakeside'),('QOT','q2-parklands')])
def test_sportsbet_rejects_cross_layout_receipts(caller,source):
    assert not _sportsbet_source_matches(f'https://www.sportsbet.com.au/greyhound-racing/australia-nz/{source}/race-7-12345',f'Race 7 - {caller} - 2026-07-19')
