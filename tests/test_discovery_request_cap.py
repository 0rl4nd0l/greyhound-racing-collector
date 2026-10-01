"""Invented canonical discovery through the real request guard and refresh seam."""
from datetime import datetime
import json

import pytest
import requests


@pytest.mark.parametrize('cap', ['scope', 'campaign'])
@pytest.mark.parametrize('canonical_successes', [0, 1])
def test_charged_cap_during_canonical_discovery_is_failed_not_success(
        tmp_path, monkeypatch, cap, canonical_successes):
    from race_collection import live_freshness_contract as contracts
    from scripts import refresh_prejump_upcoming as refresh
    from tests.test_live_freshness_candidate import contract_value
    from upcoming_race_browser import UpcomingRaceBrowser

    value = {**contract_value(tmp_path), 'max_logical_requests': 10 + canonical_successes}
    if cap == 'campaign':
        root = tmp_path / 'campaign'
        root.mkdir()
        authorization = dict(schema_version='collector_engineering_campaign_v1', campaign_id='SYNTHETIC',
                             max_capture_attempts=12, max_logical_requests=48000, max_live_seconds=10800)
        (root / 'authorization.json').write_text(json.dumps(authorization))
        (root / 'ledger.json').write_text(json.dumps(dict(campaign_id='SYNTHETIC', attempts=[],
            launches={}, logical_requests=47999-canonical_successes, source_holds=[])))
        value.update(campaign_root=str(root), campaign_authorization_sha256=contracts.digest(authorization),
                     max_capture_attempts=12, max_logical_requests=24000, cleanup_seconds=1860)
    stamp = datetime.fromisoformat(value['starts_at'])
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return stamp.astimezone(tz) if tz else stamp.replace(tzinfo=None)
    monkeypatch.setattr(contracts, 'datetime', Clock)
    monkeypatch.setenv('GREYHOUND_LIVE_EXECUTION', '1')
    scope = contracts.FreshnessContract(value)
    if scope.campaign:
        scope.campaign.begin(value['rehearsal_id'], now=stamp, deadline=scope.end)
    scope.session.mkdir(parents=True)
    (scope.session / 'request-count.json').write_text(json.dumps({'started': 9}))
    calls = []
    day = datetime.now().date().isoformat()
    def transport(session, method, url, **kwargs):
        calls.append(url)
        response = requests.Response()
        response.status_code = 200
        response._content = (''.join(
            f'<a href="/racing/invented/{day}/{number}/">Race {number}</a>'
            for number in (1, 2)) if len(calls) == 1 else
            '<formatted-time data-format="time_24">23:59</formatted-time>').encode()
        return response
    monkeypatch.setattr(requests.Session, 'request', transport)
    browser = UpcomingRaceBrowser.__new__(UpcomingRaceBrowser)
    browser.base_url = 'https://www.thedogs.com.au'
    browser.venue_map = {}
    browser.session = requests.Session()
    browser.get_races_for_date = lambda date: browser._scrape_live_races_for_date(date.isoformat())
    browser._get_cached_races_for_date = lambda date: [dict(race_number=3, date=day,
        race_time='23:59', venue='INVENTED', url=f'{browser.base_url}/racing/invented/{day}/3/')]
    browser.download_race_csv = lambda *args, **kwargs: pytest.fail('failed discovery must not download')
    monkeypatch.setattr(refresh, '_refresh_browser', lambda *args, **kwargs: browser)
    args = refresh.build_parser().parse_args([
        '--upcoming-dir', str(tmp_path / 'upcoming'), '--days-ahead', '0', '--require-safe-metadata'])
    restore = contracts.install_request_guard(scope)
    try:
        report = refresh.refresh_prejump_upcoming(args)
    finally:
        restore()
    assert calls == [f'https://www.thedogs.com.au/racing/{day}'] + (
        [f'https://www.thedogs.com.au/racing/invented/{day}/1/'] if canonical_successes else [])
    assert json.loads((scope.session / 'request-count.json').read_bytes()) == {'started': 10 + canonical_successes}
    stop_reason = 'REQUEST_CAP_EXHAUSTED' if cap == 'scope' else 'CAMPAIGN_REQUEST_CAP_EXHAUSTED'
    stop_bytes = (scope.session / 'STOP.json').read_bytes()
    assert json.loads(stop_bytes) == {'reason': stop_reason}
    if scope.campaign:
        ledger = json.loads((root / 'ledger.json').read_bytes())
        assert ledger['logical_requests'] == 48000 and ledger['attempts'] == []
        assert ledger['launches'][value['rehearsal_id']]['charged_seconds'] == 5400
    with pytest.raises(ValueError, match='operating_scope_stopped'):
        scope.admit(stamp, seconds=0)
    assert (scope.session / 'STOP.json').read_bytes() == stop_bytes
    assert report['status'] == 'DISCOVERY_FAILED'
    assert report['selected_count'] == report['current_index_race_count'] == report['accepted_csv_count'] == 0
    assert report['downloads'] == []
    assert report['next_preferred_window']['recommended_rerun_after_local'] is None
    assert report['total_races_found'] == 0 and report['considered_races'] == []
    assert report['discovery_failures'][0]['stop_reason'] == stop_reason
    assert report['discovery_failures'][0]['error_type'] == 'RequestGuardStopped'
    # The existing bounded transport-outage recovery must reject this real
    # stopped refresh, including when a preceding capture is already consumed.
    from tests.test_scheduled_refresh_outage import discovery_outage
    from scripts import run_freshness_rehearsal as supervisor
    output, plan, rid, checkpoint, retained_report = discovery_outage(tmp_path / 'cycle', after_capture=True)
    phase_bytes = {p: p.read_bytes() for p in checkpoint.glob('phase-*-result.json')}
    retained_report.write_text(json.dumps(report))
    assert contracts.classify_refresh_outage(plan['evidence_root'], rid) is None
    assert not supervisor.record_refresh_outage(output, plan, rid, set())
    assert all(p.read_bytes() == raw for p, raw in phase_bytes.items())
    assert not (output / 'refresh-deferrals').exists()
    from pathlib import Path
    from race_collection import synchronous_manual_capture as publisher
    evidence = Path(plan['evidence_root'])
    state = evidence / 'runtime/state.json'
    publication = publisher.publish_current_race_index(state_path=state, evidence_root=evidence,
        source_refresh_report_path=retained_report, run_id=rid, enforce_monotonic=True)
    assert publication['status'] == 'REJECTED'
    assert publication['failure_detail']['reason'] == 'refresh_not_accepted_success'
    assert not publisher.current_race_index_path(state).exists()
