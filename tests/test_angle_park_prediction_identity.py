"""Preserve the source-bound AP_K identity through all prediction validators."""
from datetime import datetime

import pytest

from scripts.predict_race_now import _request_race
from src.predictor.on_demand import PredictionBlocked, validate_prediction_result_v2
from src.predictor.receipt_preflight import _sportsbet_source_matches
from tests.test_prediction_bundle_sealed import ready_result
from utils.race_identity_equivalence import configured_venue_identity

RACE_ID='Race 7 - AP_K - 2026-07-19'
URL='https://www.thedogs.com.au/racing/angle-park/2026-07-19/7/example-stake'
SPORTSBET='https://www.sportsbet.com.au/greyhound-racing/australia-nz/angle-park/race-7-12345'
JUMP=datetime.fromisoformat('2026-07-19T13:00:00+10:00')


def target():
    return {'race_number':7,'date':'2026-07-19','venue':'AP_K','race_url':URL}


def test_request_preserves_exact_angle_park_race_identity():
    result=_request_race(target(),race_id=RACE_ID,jump=JUMP)
    assert result['race_id']==RACE_ID
    assert result['venue']=='AP_K'
    assert result['venue_slug']=='angle-park'


def test_source_match_uses_existing_explicit_canonical_override():
    assert configured_venue_identity('ANGLE-PARK')=='APWE'  # Legacy identity remains unchanged.
    assert configured_venue_identity('AP_K')=='APK'
    assert _sportsbet_source_matches(SPORTSBET,RACE_ID)
    assert _sportsbet_source_matches(SPORTSBET,RACE_ID.replace('AP_K','APWE'))
    assert _sportsbet_source_matches(SPORTSBET.replace('angle-park','launceston'),RACE_ID.replace('AP_K','LAU'))
    assert configured_venue_identity('ALBION-PARK') != configured_venue_identity('AP_K')


def test_sealed_result_preserves_same_source_bound_identity():
    result=ready_result()
    result['race'].update(race_id=RACE_ID,url=URL,venue='AP_K',venue_slug='angle-park',race_number=7)
    validated=validate_prediction_result_v2(result)
    assert validated['race']==result['race']


@pytest.mark.parametrize('change',['albion','unknown_slug','wrong_number','wrong_date','wrong_id','unconfigured_venue'])
def test_admission_rejects_inconsistent_or_unconfigured_identity(change):
    race=target();race_id=RACE_ID
    if change=='albion':race['race_url']=URL.replace('angle-park','albion-park')
    elif change=='unknown_slug':race['race_url']=URL.replace('angle-park','angle-park-unknown')
    elif change=='wrong_number':race['race_url']=URL.replace('/7/','/8/')
    elif change=='wrong_date':race['race_url']=URL.replace('2026-07-19','2026-07-20')
    elif change=='wrong_id':race_id='Race 7 - AP - 2026-07-19'
    else:race['venue']='AP@K'
    with pytest.raises(PredictionBlocked,match='EXACT_RACE_IDENTITY_UNAVAILABLE'):
        _request_race(race,race_id=race_id,jump=JUMP)


@pytest.mark.parametrize('url',[SPORTSBET.replace('angle-park','albion-park'),SPORTSBET.replace('angle-park','unknown-park'),SPORTSBET.replace('race-7','race-8'),SPORTSBET+'?results=true'])
def test_sportsbet_receipt_requires_exact_venue_race_and_safe_url(url):
    assert not _sportsbet_source_matches(url,RACE_ID)


@pytest.mark.parametrize('change',['albion','unknown_slug','wrong_number','wrong_date','wrong_id','unconfigured_venue'])
def test_sealed_result_rejects_inconsistent_identity(change):
    result=ready_result()
    race=result['race']
    race.update(race_id=RACE_ID,url=URL,venue='AP_K',venue_slug='angle-park',race_number=7)
    if change=='albion':race.update(url=URL.replace('angle-park','albion-park'),venue_slug='albion-park')
    elif change=='unknown_slug':race.update(url=URL.replace('angle-park','unknown-park'),venue_slug='unknown-park')
    elif change=='wrong_number':race['race_number']=8
    elif change=='wrong_date':race['race_date']='2026-07-20'
    elif change=='wrong_id':race['race_id']='Race 7 - AP - 2026-07-19'
    else:race['venue']='AP@K'
    with pytest.raises(PredictionBlocked,match='PREDICTION_BUNDLE_INVALID'):
        validate_prediction_result_v2(result)


def test_configured_display_name_is_not_a_source_code_substitution():
    race={'race_number':5,'date':'2026-07-19','venue':'GUNNEDAH',
          'race_url':'https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5'}
    with pytest.raises(PredictionBlocked,match='EXACT_RACE_IDENTITY_UNAVAILABLE'):
        _request_race(race,race_id='Race 5 - GUNN - 2026-07-19',jump=JUMP)
    result=ready_result();result['race']['venue']='GUNNEDAH'
    with pytest.raises(PredictionBlocked,match='PREDICTION_BUNDLE_INVALID'):
        validate_prediction_result_v2(result)
