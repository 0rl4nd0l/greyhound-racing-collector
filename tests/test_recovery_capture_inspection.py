"""Capture entrypoint wiring; fabricated CDP and no network."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('resource_type', ['Document', 'XHR', 'Image'])
def test_capture_retains_inspection_even_when_navigation_is_denied(tmp_path, monkeypatch, resource_type):
    from tests.fixtures.freshness_transport.fake_cdp import BrowserTransport
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    import drivers
    import odds_auto_integrator

    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(tmp_path/'access.json'))
    monkeypatch.setenv('GREYHOUND_SPORTSBET_RESPONSE_INSPECTION', '1')
    SportsbetAccess().initialize(access_basis={'status':'permitted','reference':'synthetic'})
    transport = BrowserTransport()
    unbounded_dom_calls = []
    class Driver:
        service = SimpleNamespace(process=transport.process)
        def start_devtools(self): return None, transport
        def get(self, url):
            transport.response('https://www.sportsbet.com.au/apigw/unknown?token=secret', 429,
                               {'Retry-After':'3600'}, resource_type=resource_type)
            assert transport.stopped.wait(3)
        def get_log(self, name): return []
        def execute_script(self, script):
            unbounded_dom_calls.append(script)
            raise RuntimeError('unresponsive renderer')
        def quit(self): transport.quit()
    monkeypatch.setattr(drivers, 'get_chrome_driver', lambda **kwargs: Driver())
    metrics = tmp_path/'requests.json'
    with pytest.raises(SportsbetAccessBlocked):
        odds_auto_integrator.fetch_odds_for_target_race(str(tmp_path/'unused.sqlite'), 'BEN', 3,
                '2026-09-24', True, request_metrics_path=metrics)
    report = json.loads(metrics.with_suffix('.responses.json').read_text())
    assert report['operation_id'].startswith('browser:')
    assert report['network_responses'][-1]['resource_type'] == resource_type
    assert report['network_responses'][-1]['status'] == 429
    assert report['network_responses'][-1]['retry_headers']['retry-after'] == '3600'
    assert any(m['name'] == 'browser_ready' for m in report['marks'])
    assert 'secret' not in json.dumps(report)
    assert transport.process.poll() is not None
    assert len(SportsbetAccess().read()['denials']) == 1
    assert unbounded_dom_calls == []
