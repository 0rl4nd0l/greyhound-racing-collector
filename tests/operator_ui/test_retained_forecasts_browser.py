"""Offline browser contract tests; all HTTP responses are local fixtures."""
import json
import shutil
from pathlib import Path

import pytest
from flask import Flask, render_template

playwright = pytest.importorskip('playwright.sync_api')
ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize('engineering', [False, True])
def test_retained_source_labels_and_distinct_verified_timestamps(engineering):
    executable = shutil.which('google-chrome') or shutil.which('chromium')
    if executable is None:
        pytest.skip('Local Chromium is required for offline browser acceptance')
    app = Flask(__name__, template_folder=str(ROOT / 'templates'))
    with app.test_request_context():
        html = render_template('operator_ui_forecasts.jinja')
    forecast = {
        'prediction_id': 'fixture-prediction', 'verification': 'VERIFIED',
        'temporal_status': 'HISTORICAL',
        'race': {'venue': 'CASO', 'race_number': 2, 'race_date': '2026-10-01',
                 'jump_timestamp': '2026-10-01T06:10:00Z'},
        'quote_at': '2026-10-01T06:00:01Z',
        'prediction_at': '2026-10-01T06:00:02Z',
        'independent_verification_at': '2026-10-01T06:00:03Z',
        'verification_at': '2026-10-01T06:00:04Z',
        'independent_chain_audit_at': None,
        'model': {'identity': 'market_form_residual_v1', 'sha256': 'a' * 64},
        'manifest_sha256': 'b' * 64,
        'runners': [{'box': 1, 'name': 'Fixture runner', 'win_odds': 2.5,
                     'market_probability': 0.4, 'model_probability': 0.5, 'model_rank': 1}],
    }
    sources = [{'source': kind, 'state': 'EMPTY', 'forecasts': [], 'errors': []}
               for kind in ['operational', 'programme']]
    if engineering:
        sources.append({'source': 'engineering', 'state': 'VERIFIED',
                        'forecasts': [forecast], 'errors': []})
    else:
        sources[0].update(state='VERIFIED', forecasts=[forecast])
    payload = {'schema': 'operator_ui_retained_forecasts_v1',
               'observed_at': '2026-10-01T06:30:00Z', 'sources': sources,
               'programme': {'state': 'CANARY_NOT_VERIFIED', 'collecting': False,
                             'next_session': None, 'next_session_end': None}}
    requests = []
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(executable_path=executable, headless=True,
                                        args=['--no-sandbox', '--disable-background-networking'])
        page = browser.new_page()
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def respond(route):
            request = route.request
            path = request.url.removeprefix('http://offline.invalid').split('?', 1)[0]
            requests.append((request.method, path))
            assert request.method == 'GET'
            if path == '/operator-ui/forecasts':
                return route.fulfill(body=html, content_type='text/html')
            if path.startswith('/static/'):
                return route.fulfill(path=str(ROOT / path.lstrip('/')))
            if path == '/operator-ui/api/v1/predictions/retained':
                return route.fulfill(body=json.dumps(payload), content_type='application/json')
            if path == '/operator-ui/api/v1/collector':
                return route.fulfill(body='{"classification":"STALE"}', content_type='application/json')
            raise AssertionError(f'Unexpected browser request: {path}')

        page.route('**/*', respond)
        page.goto('http://offline.invalid/operator-ui/forecasts')
        headings = ['Operational forecasts', 'October programme forecasts']
        if engineering:
            headings.append('Engineering rehearsals')
        playwright.expect(page.locator('#forecast-sources h2')).to_have_text(headings)
        card = page.locator('article[data-prediction-id="fixture-prediction"]')
        playwright.expect(card).to_contain_text(
            'Quote: 01/10/2026, 16:00:01 AEST · Prediction: 01/10/2026, 16:00:02 AEST · '
            'Verified: 01/10/2026, 16:00:04 AEST')
        playwright.expect(card).to_contain_text('Independent verification: 01/10/2026, 16:00:03 AEST')
        playwright.expect(card.locator('tbody td')).to_have_text(
            ['1', 'Fixture runner', '2.50', '40.0000%', '50.0000%', '1'])
        playwright.expect(page.locator('#programme-status')).to_contain_text('CANARY NOT VERIFIED')
        assert not errors
        assert all(method == 'GET' for method, _ in requests)
        browser.close()
