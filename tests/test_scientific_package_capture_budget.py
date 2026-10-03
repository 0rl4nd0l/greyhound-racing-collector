"""Package admission uses the same scientific capture domain as native Campaign."""
import json
from pathlib import Path

import pytest

from race_collection.freshness_campaign import Campaign
from race_collection.live_freshness_contract import create_once, digest
from scripts import prepare_freshness_rehearsal as prep
from tests.test_short_operational_observation import prepared
from tests.fixtures.incident_engineering_case import make_incident


def scientific_campaign(prepared, tmp_path, monkeypatch, *, science=2, incident=20, engineering=6):
    root = prepared['campaign_root']
    authority, reference = make_incident(tmp_path/'incident')
    base = {'schema_version':'collector_engineering_campaign_v1','campaign_id':'SYNTHETIC',
            'max_capture_attempts':12,'max_logical_requests':48000,'max_live_seconds':10800}
    create_once(root/'authorization.json', base)
    create_once(root/'persistent-programme-authority.json', {
        'schema_version':'collector_persistent_programme_v1','status':'AUTHORIZED_PERSISTENT_PROGRAMME',
        'campaign_id':'SYNTHETIC','programme_id':'study','authority_reference':'synthetic',
        'prior_effective_authorization_sha256':digest(base),'starts_at':'2026-10-01T12:00:00+10:00',
        'expires_at':'2027-01-21T12:00:00+11:00','max_capture_attempts':1012,
        'max_logical_requests':1352000,'max_live_seconds':591600,
        'initial_counters':{'capture_attempts':0,'logical_requests':0,'live_seconds':0}})
    tags = {'incident_authority':reference,'incident_authority_sha256':reference['sha256'],
            'incident_id':authority['incident_id'],'incident_slot':'001'}
    attempts = ([{} for _ in range(science)] + [{'engineering_authority':'synthetic-preprogramme'} for _ in range(engineering)]
                + [dict(tags) for _ in range(incident)])
    ledger = {'campaign_id':'SYNTHETIC','attempts':attempts,'logical_requests':0,'launches':{},'source_holds':[]}
    (root/'ledger.json').write_text(json.dumps(ledger))
    monkeypatch.setattr('race_collection.freshness_campaign.Campaign', Campaign)
    return Campaign(root), ledger


@pytest.mark.parametrize('science,incident,engineering,expected', [
    (2,0,6,998), (2,20,6,998), (997,20,6,3), (2,0,12,998), (2,0,20,990)])
def test_packaged_scientific_jobs_use_scientific_remaining_capacity(prepared,tmp_path,monkeypatch,
                                                                  science,incident,engineering,expected):
    campaign, ledger = scientific_campaign(prepared,tmp_path,monkeypatch,
        science=science,incident=incident,engineering=engineering)
    before = (campaign.root/'ledger.json').read_bytes()
    assert campaign.programme_usage(ledger)['capture_attempts'] == science
    prep.prepare(**prepared)
    plan = json.loads((prepared['output']/'plan.json').read_bytes())
    assert plan['operational_predictions']['max_jobs'] == expected
    assert (campaign.root/'ledger.json').read_bytes() == before
    assert campaign.available() is True


def test_exhausted_scientific_population_does_not_publish_an_admissible_package(prepared,tmp_path,monkeypatch):
    campaign, _ = scientific_campaign(prepared,tmp_path,monkeypatch,science=1000,incident=0,engineering=0)
    assert campaign.available() is False
    before = (campaign.root/'ledger.json').read_bytes()
    with pytest.raises(ValueError, match='capture_allowance_consumed'):
        prep.prepare(**prepared)
    assert not (prepared['output']/'plan.json').exists()
    assert (campaign.root/'ledger.json').read_bytes() == before


def test_unverifiable_engineering_tag_cannot_free_scientific_capacity(prepared,tmp_path,monkeypatch):
    campaign, ledger = scientific_campaign(prepared,tmp_path,monkeypatch)
    ledger['attempts'][-1]['incident_authority_sha256'] = 'f'*64
    (campaign.root/'ledger.json').write_text(json.dumps(ledger))
    before = (campaign.root/'ledger.json').read_bytes()
    with pytest.raises(ValueError): prep.prepare(**prepared)
    assert not (prepared['output']/'plan.json').exists()
    assert (campaign.root/'ledger.json').read_bytes() == before
