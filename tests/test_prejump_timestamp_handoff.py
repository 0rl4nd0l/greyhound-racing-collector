"""The real sidecar producer must meet the frozen scorer's timing contract."""
from datetime import date, datetime

import pytest

from scripts.predict_market_form_residual import ManualPredictionError, _jump_timestamp
from utils.csv_metadata import build_prejump_shadow_metadata_payload


def produce(*, racing_date='2026-10-03', clock='5:41 PM', scheduled=None):
    race_info = {'date': racing_date, 'race_time': clock}
    if scheduled is not None:
        race_info['scheduled_jump_datetime'] = scheduled
    payload = {'race_info': race_info}
    payload['prejump_shadow_metadata'] = build_prejump_shadow_metadata_payload(payload)
    return payload


@pytest.mark.parametrize(('racing_date', 'scheduled'), [
    ('2026-10-03', '2026-10-03T17:41:00+10:00'),
    ('2026-10-04', '2026-10-04T17:41:00+11:00'),
    ('2026-10-03', '2026-10-03T07:41:00Z'),
])
def test_scheduled_instant_reaches_real_scorer_without_becoming_clock_text(racing_date, scheduled):
    payload = produce(racing_date=racing_date, scheduled=scheduled)
    assert _jump_timestamp(payload, date.fromisoformat(racing_date)) == datetime.fromisoformat(scheduled)
    assert payload['prejump_shadow_metadata']['jump_time'] == '5:41 PM'
    assert payload['prejump_shadow_metadata']['jump_datetime'] == scheduled


def test_scheduled_instant_does_not_hide_conflicting_display_clock():
    payload = produce(scheduled='2026-10-03T17:41:00+10:00', clock='5:42 PM')
    with pytest.raises(ManualPredictionError, match='^jump_timestamp_mismatch$'):
        _jump_timestamp(payload, date(2026, 10, 3))


def test_naive_scheduled_instant_does_not_fall_back_to_valid_display_clock():
    payload = produce(scheduled='2026-10-03T17:41:00')
    with pytest.raises(ManualPredictionError, match='^jump_timestamp_timezone_missing$'):
        _jump_timestamp(payload, date(2026, 10, 3))


@pytest.mark.parametrize('scheduled', [None, False, ''])
def test_explicit_invalid_scheduled_instant_remains_rejected(scheduled):
    payload = produce()
    payload['race_info']['scheduled_jump_datetime'] = scheduled
    payload['prejump_shadow_metadata'] = build_prejump_shadow_metadata_payload(payload)
    with pytest.raises(ManualPredictionError, match='^jump_timestamp_invalid$'):
        _jump_timestamp(payload, date(2026, 10, 3))


def test_clock_only_legacy_handoff_remains_valid():
    payload = produce()
    assert 'jump_datetime' not in payload['prejump_shadow_metadata']
    assert _jump_timestamp(payload, date(2026, 10, 3)) == datetime.fromisoformat('2026-10-03T17:41:00+10:00')


def authoritative_overnight(*, racing_date='2026-10-03', scheduled='2026-10-04T00:13:00+10:00', clock='12:13 AM'):
    payload = produce(racing_date=racing_date, clock=clock, scheduled=scheduled)
    payload['race_info'].update(race_time_source='canonical_race_url', race_time_mapping_status='exact_url_match')
    return payload


@pytest.mark.parametrize(('racing_date', 'scheduled', 'clock'), [
    ('2026-10-03', '2026-10-04T00:13:00+10:00', '12:13 AM'),
    ('2026-10-03', '2026-10-03T14:13:00Z', '12:13 AM'),
    ('2026-10-03', '2026-10-04T03:13:00+11:00', '3:13 AM'),
    ('2026-10-04', '2026-10-05T00:13:00+11:00', '12:13 AM'),
])
def test_explicit_official_next_day_instant_preserves_source_day(racing_date, scheduled, clock):
    payload = authoritative_overnight(racing_date=racing_date, scheduled=scheduled, clock=clock)
    assert _jump_timestamp(payload, date.fromisoformat(racing_date)) == datetime.fromisoformat(scheduled)
    assert payload['prejump_shadow_metadata']['race_date'] == racing_date
    assert payload['race_info']['date'] == racing_date


@pytest.mark.parametrize('missing', ['scheduled_jump_datetime', 'race_time_source', 'race_time_mapping_status'])
def test_next_day_requires_explicit_canonical_timing_proof(missing):
    payload = authoritative_overnight()
    del payload['race_info'][missing]
    with pytest.raises(ManualPredictionError):
        _jump_timestamp(payload, date(2026, 10, 3))


@pytest.mark.parametrize('scheduled', ['2026-10-02T00:13:00+10:00', '2026-10-05T00:13:00+11:00'])
def test_explicit_instant_cannot_select_arbitrary_date(scheduled):
    payload = authoritative_overnight(scheduled=scheduled)
    with pytest.raises(ManualPredictionError, match='jump_date_target_date_mismatch'):
        _jump_timestamp(payload, date(2026, 10, 3))


def test_next_day_still_rejects_conflicting_clock():
    payload = authoritative_overnight(clock='12:14 AM')
    with pytest.raises(ManualPredictionError, match='jump_timestamp_mismatch'):
        _jump_timestamp(payload, date(2026, 10, 3))


def test_scheduled_alias_conflict_cannot_be_hidden_by_shadow():
    payload = authoritative_overnight()
    payload['race_info']['scheduled_jump_datetime'] = '2026-10-04T00:14:00+10:00'
    with pytest.raises(ManualPredictionError, match='jump_timestamp_mismatch'):
        _jump_timestamp(payload, date(2026, 10, 3))


def test_dst_skipped_local_clock_cannot_alias_valid_instant():
    payload = authoritative_overnight(scheduled='2026-10-04T03:13:00+11:00', clock='2:13 AM')
    with pytest.raises(ManualPredictionError, match='jump_timestamp_mismatch'):
        _jump_timestamp(payload, date(2026, 10, 3))
