"""Synthetic, network-denied supervised CSV acquisition boundaries."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from upcoming_race_browser import UpcomingRaceBrowser

RACE = "https://www.thedogs.com.au/racing/synthetic/2099-06-09/1/fixture"
EXPERT = RACE + "/expert-form"
EXPORT = RACE + "/export-expert-form"


@pytest.fixture
def browser(tmp_path, monkeypatch):
    monkeypatch.setenv("GREYHOUND_LIVE_EXECUTION", "fixture")
    obj = UpcomingRaceBrowser.__new__(UpcomingRaceBrowser)
    obj.base_url = "https://www.thedogs.com.au"
    obj.upcoming_dir = str(tmp_path)
    monkeypatch.setattr(obj, "extract_detailed_race_info", lambda *args: {
        "race_number": 1, "venue": "SYN", "date": "2099-06-09", "race_time": "13:00",
    })
    monkeypatch.setattr(obj, "_collect_safe_track_metadata_from_sportsbet", lambda *args: {})
    monkeypatch.setattr(obj, "_collect_safe_weather_metadata_from_forecast", lambda *args: {})
    return obj


def response(url, status=200, text="<html></html>", headers=None):
    value = requests.Response()
    value.status_code = status
    value.url = url
    value._content = text.encode()
    value.headers.update(headers or {"Content-Type": "text/html"})
    return value


@pytest.mark.parametrize("stage,status", [("main", 502), ("expert", 503), ("export", 504), ("expert", 401), ("expert", 403), ("expert", 429)])
def test_supervised_http_failure_is_typed_and_never_falls_back(browser, stage, status):
    failed_url = {"main": RACE, "expert": EXPERT, "export": EXPORT}[stage]
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        assert calls == [RACE, EXPERT, EXPORT][:len(calls)]
        if url == failed_url:
            return response(url, status, headers={"Retry-After": "120", "X-RateLimit-Reset": "12345", "Set-Cookie": "private-secret"})
        return response(url, text=f'<a href="{EXPORT}">Export CSV</a>')
    browser.session = SimpleNamespace(get=get)
    result = browser.download_race_csv(RACE)
    assert result["success"] is False
    assert result["source_http_status"] == status
    assert result["source_retry_after"] == "120"
    assert result["source_rate_limit_reset"] == "12345"
    assert "private-secret" not in json.dumps(result)
    assert calls == [RACE, EXPERT, EXPORT][: ["main", "expert", "export"].index(stage) + 1]
    assert not list(Path(browser.upcoming_dir).glob("*.csv"))


def test_supervised_missing_export_makes_only_main_and_expert_requests(browser, monkeypatch):
    import sys
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        assert url in {RACE, EXPERT}
        assert len(calls) <= 2
        return response(url)
    def forbidden(*args, **kwargs):
        pytest.fail("repeated expert-form scraper fallback was invoked")
    monkeypatch.setitem(sys.modules, "expert_form_csv_scraper", SimpleNamespace(ExpertFormCsvScraper=forbidden))
    browser.session = SimpleNamespace(get=get)
    assert browser.download_race_csv(RACE) == {"success": False, "error": "No CSV download link found", "source_failure_category": "observed_export_absent"}
    assert calls == [RACE, EXPERT]


@pytest.mark.parametrize("method", ["get", "post"])
def test_supervised_observed_form_submits_once_with_only_observed_fields(browser, method):
    calls = []
    form = f'<form action="{EXPORT}" method="{method}"><input name="sort_by" value=""><button type="submit" name="observed_export" value="csv">CSV</button></form>'
    def get(url, **kwargs):
        calls.append(("get", url, kwargs))
        if url == RACE:
            return response(url)
        if url == EXPERT:
            return response(url, text=form)
        assert method == "get" and url == EXPORT
        assert kwargs["params"] == {"sort_by": "", "observed_export": "csv"}
        return response(url, 502)
    def post(url, **kwargs):
        calls.append(("post", url, kwargs))
        assert method == "post" and url == EXPORT
        assert kwargs["data"] == {"sort_by": "", "observed_export": "csv"}
        return response(url, 502)
    browser.session = SimpleNamespace(get=get, post=post)
    result = browser.download_race_csv(RACE)
    assert result["source_http_status"] == 502
    assert [(method, url) for method, url, kwargs in calls] == [("get", RACE), ("get", EXPERT), (method, EXPORT)]


def test_live_zero_retry_exposes_first_5xx_response_policy(monkeypatch):
    import utils.http_client as module
    monkeypatch.setenv("GREYHOUND_LIVE_EXECUTION", "fixture")
    monkeypatch.setattr(module, "_shared_session", None)
    session = module.get_shared_session()
    try:
        retry = session.get_adapter(RACE).max_retries
        assert retry.total == 0
        assert retry.raise_on_status is False
        assert not retry.is_retry("GET", 429, has_retry_after=True)
    finally:
        session.close()


@pytest.mark.parametrize("status", [502, 429])
def test_supervised_discovery_time_failure_has_one_request_and_no_sleep(browser, monkeypatch, status):
    import upcoming_race_browser as module
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        return response(url, status, headers={"Retry-After": "60"})
    monkeypatch.setattr(module.time, "sleep", lambda delay: pytest.fail("supervised discovery retried/slept"))
    browser.session = SimpleNamespace(get=get)
    assert browser._scrape_race_time_from_page(RACE) is None
    assert calls == [RACE]


def test_live_adapter_returns_first_502_without_retrying(browser, monkeypatch):
    import io
    import urllib3.connectionpool
    import utils.http_client as module
    from urllib3.response import HTTPResponse
    calls = []
    def make_request(pool, connection, method, url, *args, **kwargs):
        calls.append((method, url))
        return HTTPResponse(body=io.BytesIO(b"synthetic failure"), status=502,
                            headers={"Retry-After": "120"}, preload_content=False)
    monkeypatch.setattr(urllib3.connectionpool.HTTPSConnectionPool, "_make_request", make_request)
    monkeypatch.setattr(module, "_shared_session", None)
    session = module.get_shared_session()
    session.trust_env = False
    try:
        value = session.get(RACE)
        assert value.status_code == 502
        assert value.headers["Retry-After"] == "120"
        value.close()
        assert len(calls) == 1
    finally:
        session.close()


def test_observed_csv_form_ignores_other_race_navigation_and_pdf_submit(browser):
    calls = []
    other = RACE.replace('/1/fixture', '/2/download-the-app') + '/expert-form'
    form = f'''<a href="{other}">Download the app</a>
      <form action="{EXPERT}" method="get">
      <input name="expert_form[sort_by]" value="">
      <button name="button" type="submit">Apply</button>
      <button name="export_pdf" value="true" type="submit">Export PDF</button>
      <button name="export_csv" value="true" type="submit">Export CSV</button></form>'''
    def get(url, **kwargs):
        calls.append(url)
        assert url in {RACE, EXPERT}, "requested another race's navigation link"
        if url == RACE:
            return response(url)
        if 'params' in kwargs:
            assert kwargs['params'] == {'expert_form[sort_by]': '', 'export_csv': 'true'}
            return response(url, 502)
        return response(url, text=form)
    browser.session = SimpleNamespace(get=get)
    assert browser.download_race_csv(RACE)['source_http_status'] == 502
    assert calls == [RACE, EXPERT, EXPERT]


@pytest.mark.parametrize('kind', ['link', 'form', 'delivered'])
def test_supervised_export_rejects_cross_race_source_targets(browser, kind):
    calls = []
    other = RACE.replace('/1/fixture', '/2/other') + '/export-expert-form'
    def get(url, **kwargs):
        calls.append(url)
        assert url != other, 'cross-race export requested'
        if url == RACE:
            return response(url)
        if url == EXPERT:
            if kind == 'link':
                return response(url, text=f'<a href="{other}">CSV</a>')
            if kind == 'form':
                return response(url, text=f'<form action="{other}"><button name="export_csv" value="true">CSV</button></form>')
            return response(url, text=f'<a href="{EXPORT}">CSV</a>')
        assert url == EXPORT
        return response(url, text=other)
    browser.session = SimpleNamespace(get=get)
    result = browser.download_race_csv(RACE)
    assert result['success'] is False
    assert result['error'] == ('Observed export URL invalid' if kind == 'delivered' else 'No CSV download link found')
    assert calls == [RACE, EXPERT] + ([EXPORT] if kind == 'delivered' else [])


def test_supervised_delivered_export_url_never_repeats_same_request(browser):
    calls = []
    def get(url, **kwargs):
        calls.append(url)
        if url == RACE:
            return response(url)
        if url == EXPERT:
            return response(url, text=f'<a href="{EXPORT}">CSV</a>')
        assert url == EXPORT
        return response(url, text=EXPORT)
    browser.session = SimpleNamespace(get=get)
    result = browser.download_race_csv(RACE)
    assert result['success'] is False
    assert result['error'] == 'Observed export URL invalid'
    assert calls == [RACE, EXPERT, EXPORT]
