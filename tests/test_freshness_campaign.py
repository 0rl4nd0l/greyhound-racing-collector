"""Cumulative limits never depend on package or process identity."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import create_once


def make_campaign(root):
    create_once(root / 'authorization.json', dict(schema_version='collector_engineering_campaign_v1',
        campaign_id='synthetic', max_capture_attempts=12, max_logical_requests=48000, max_live_seconds=10800))
    create_once(root / 'ledger.json', dict(campaign_id='synthetic', attempts=[], logical_requests=0, launches={}))
    return Campaign(root)


def test_attempts_shared_across_instances_and_aliases(tmp_path):
    campaign = make_campaign(tmp_path)
    item = dict(race_id='canonical', race_id_aliases=['canonical', 'alias'], capture_window_minutes=10)
    campaign.consume(tmp_path / 'first', item)
    restarted = Campaign(tmp_path)
    with pytest.raises(ValueError, match='window_consumed'):
        restarted.consume(tmp_path / 'other', {**item, 'race_id': 'alias'})
    for number in range(11):
        restarted.consume(tmp_path / str(number), {**item, 'race_id': str(number), 'race_id_aliases': [str(number)]})
    assert not campaign.available()
    with pytest.raises(ValueError, match='allowance_consumed'):
        campaign.consume(tmp_path / 'last', {**item, 'race_id': 'last'})


def test_request_limit_survives_new_launch(tmp_path):
    campaign = make_campaign(tmp_path)
    with campaign.ledger() as value:
        value['logical_requests'] = 47999
    campaign.request()
    with pytest.raises(ValueError, match='cap_exhausted'):
        Campaign(tmp_path).request()
    with campaign.ledger() as value:
        assert value['logical_requests'] == 48000


def test_live_lease_charges_crashes_and_only_restoration_releases_time(tmp_path):
    campaign = make_campaign(tmp_path)
    now = datetime.now(timezone.utc)
    campaign.begin('first', now=now, deadline=now + timedelta(seconds=6600))
    with pytest.raises(ValueError, match='already_exists'):
        Campaign(tmp_path).begin('second', now=now, deadline=now + timedelta(seconds=6600))
    with pytest.raises(ValueError, match='lease_closed'):
        campaign.admit('first', now + timedelta(seconds=6600))
    campaign.close('first', now=now + timedelta(seconds=120))
    campaign.begin('second', now=now + timedelta(seconds=120), deadline=now + timedelta(seconds=6720))
    campaign.close('second', now=now + timedelta(seconds=6720))
    with pytest.raises(ValueError, match='time_exhausted'):
        campaign.begin('third', now=now + timedelta(seconds=6720), deadline=now + timedelta(seconds=13320))
