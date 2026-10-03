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


def test_after_midnight_source_day_remains_explicit_and_not_silently_rebased():
    payload = produce(clock='12:13 AM', scheduled='2026-10-04T00:13:00+10:00')
    assert payload['prejump_shadow_metadata']['race_date'] == '2026-10-03'
    assert payload['prejump_shadow_metadata']['jump_datetime'] == '2026-10-04T00:13:00+10:00'
    # Separate downstream source-day support is required; this producer repair
    # must not disguise a date disagreement to make the old scorer accept it.
    with pytest.raises(ManualPredictionError, match='^jump_timestamp_mismatch$'):
        _jump_timestamp(payload, date(2026, 10, 3))
