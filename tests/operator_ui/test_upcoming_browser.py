"""Browser fixtures for the real connected UI; every request stays offline."""
import json
import shutil
from pathlib import Path

import pytest
from flask import Flask, render_template

playwright = pytest.importorskip("playwright.sync_api")
sync_playwright, expect = playwright.sync_playwright, playwright.expect

ROOT = Path(__file__).parents[2]


@pytest.fixture
def console():
    executable = shutil.which("google-chrome") or shutil.which("chromium")
    if executable is None:
        pytest.skip("Local Chromium is required for offline browser acceptance")
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    with app.test_request_context():
        html = render_template("operator_ui_connected.jinja", connected=True)
        landing = render_template("operator_ui_forecasts.jinja")
    race = dict(race_id="exact-race-1", route_id="race-route-1", venue="Bulli",
                racing_date="2099-04-01", race_number=2, jump_utc="2099-04-01T06:36:00Z",
                runner_set_sha256="c" * 64, runners=[dict(box=1, name="Fixture runner")])
    state = dict(classification="AVAILABLE/FRESH", races=[race], posts=[])
    resources = {"races/upcoming": "upcoming_races", "models": "models", "predictions/recent": "recent_predictions"}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=executable, args=["--no-sandbox", "--disable-background-networking"])
        page = browser.new_page()
        def respond(route):
            path = route.request.url.split("127.0.0.1:5055", 1)[-1].split("?", 1)[0]
            body, status = {}, 200
            if path.startswith("/static/"):
                file = ROOT / path.lstrip("/")
                return route.fulfill(path=str(file), content_type="text/javascript" if file.suffix == ".js" else "text/css")
            if path in ("/operator-ui", "/operator-ui/forecasts"):
                return route.fulfill(body=html if path == "/operator-ui" else landing, content_type="text/html")
            if path == "/operator-ui/login":
                body = dict(classification="CSRF_TOKEN", csrf_token="fixture-token")
            elif path.endswith("r3-capability"):
                body = dict(schema="operator_ui_r3_capability_v1", authorized=True, runtime_configured=True, level=2)
            elif path.endswith("prediction-jobs"):
                state["posts"].append(json.loads(route.request.post_data))
                assert route.request.headers["x-csrf-token"] == "fixture-token"
                body, status = state.get("submission", (dict(schema="operator_ui_prediction_error_v1", classification="PENDING_RECEIPT"), 409))
            elif '/prediction-jobs/' in path and 'reconnect' in state:
                body = state['reconnect']
            else:
                endpoint = path.removeprefix("/operator-ui/api/v1/")
                if endpoint == "overview" and "overview_reply" in state:
                    body, status = state["overview_reply"]
                    return route.fulfill(status=status, content_type="application/json", body=json.dumps(body))
                classification = state["classification"] if endpoint == "races/upcoming" else "AVAILABLE/FRESH"
                data = {"races": state["races"]} if endpoint == "races/upcoming" and classification == "AVAILABLE/FRESH" else {}
                if endpoint == "models":
                    data = {"models": [dict(role="LATEST_RESEARCH", model_id="market_form_residual_v1", config_id="market-form-residual-v1")]}
                body = dict(schema="operator_ui_level_1_api_v1", api_version="v1", resource=resources.get(endpoint, endpoint), classification=classification, stale=classification == "STALE", server_observed_at="2099-04-01T06:30:00Z", evidence={"age_seconds": 0}, data=data)
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))
        page.route("**/*", respond)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        yield page, state
        assert not errors
        browser.close()


def test_landing_links_to_upcoming_and_run_prediction(console):
    page, _ = console
    page.goto("http://127.0.0.1:5055/operator-ui/forecasts")
    expect(page.get_by_role("link", name="Upcoming races", exact=True)).to_have_attribute("href", "/operator-ui#upcoming-races")
    expect(page.get_by_role("link", name="Run prediction", exact=True)).to_have_attribute("href", "/operator-ui#manual-prediction")


def test_upcoming_selection_uses_existing_csrf_submission_and_reports_blocker(console):
    page, state = console
    page.goto("http://127.0.0.1:5055/operator-ui")
    page.get_by_role("button", name="Select Bulli R2").click()
    expect(page.locator("#runner-confirmation")).to_contain_text("1 Fixture runner")
    page.get_by_role("button", name="Run prediction", exact=True).click()
    expect(page.locator("#job-status")).to_contain_text("PENDING_RECEIPT")
    assert len(state["posts"]) == 1
    assert state["posts"][0]["race_id"] == "exact-race-1"
    assert state["posts"][0]["odds_source_id"] == "receipt"
    assert state["posts"][0]["model_id"] == "latest-research"


@pytest.mark.parametrize('race_id', ['exact-race-1', 'Race 7 - TWN - 2026-09-29'])
def test_verified_submission_response_renders_probabilities(console, race_id):
    page, state = console
    state['races'][0]['race_id'] = race_id
    state["submission"] = (dict(
        schema="operator_ui_prediction_job_response_v1", job_id="job_" + "a" * 32,
        phase="PREDICTION_READY", terminal=True, race_id=race_id,
        jump_timestamp="2099-04-01T06:36:00Z", runner_set_sha256="c" * 64,
        model_id="latest-research", resolved_model_identity="market_form_residual_v1",
        config_id="manual-default", odds_source_id="receipt", timeline=[],
        result=dict(schema="operator_ui_verified_prediction_result_v1", verification_status="VERIFIED",
                    probabilities=[dict(rank=1, runner_id="fixture-runner", box=1, name="Fixture runner", probability=0.25)], evidence={}),
    ), 202)
    page.goto("http://127.0.0.1:5055/operator-ui")
    page.get_by_role("button", name="Select Bulli R2").click()
    page.get_by_role("button", name="Run prediction", exact=True).click()
    expect(page.locator("#job-result")).to_contain_text("Strict sealed-v2 verification: VERIFIED")
    expect(page.locator("#job-result")).to_contain_text("0.25")
    assert len(state["posts"]) == 1


def test_race_that_has_jumped_cannot_be_submitted(console):
    page, state = console
    state["races"][0]["jump_utc"] = "2000-04-01T06:36:00Z"
    page.goto("http://127.0.0.1:5055/operator-ui")
    page.get_by_role("button", name="Select Bulli R2").click()
    expect(page.locator("#prediction-readiness")).to_contain_text("scheduled jump has passed")
    expect(page.locator("#prediction-submit")).to_be_disabled()


def test_saved_job_link_reopens_verified_forecast_without_submission(console):
    page,state=console
    state['reconnect']=dict(schema='operator_ui_prediction_job_response_v1',job_id='job_'+'a'*32,
        phase='PREDICTION_READY',terminal=True,race_id='Race 7 - TWN - 2026-09-29',
        jump_timestamp='2026-09-29T20:35:00+10:00',runner_set_sha256='c'*64,
        model_id='latest-research',resolved_model_identity='market_form_residual_v1',config_id='manual-default',odds_source_id='receipt',timeline=[],
        result=dict(schema='operator_ui_verified_prediction_result_v1',verification_status='VERIFIED',
            probabilities=[dict(rank=1,runner_id='HUNTERSBOY',box=8,name='Hunters Boy',probability=0.527)],evidence={}))
    page.goto('http://127.0.0.1:5055/operator-ui?job=job_'+'a'*32+'#manual-prediction')
    expect(page.locator('#job-result')).to_contain_text('Hunters Boy')
    expect(page.locator('#job-result')).to_contain_text('VERIFIED')
    assert state['posts']==[]
    assert not state["posts"]


@pytest.mark.parametrize("kind", ["audit", "provider", "empty", "unregistered", "malformed"])
def test_overview_distinguishes_api_errors_from_offline_and_preserves_evidence(console, kind):
    from src.operator_ui.security import NON_OPERATIONAL_ERROR, PROVIDER_ERROR

    page, state = console
    payload = dict(schema="operator_ui_level_1_api_v1", api_version="v1", resource="overview",
                   classification="UNAVAILABLE/DATA_MISSING", stale=False,
                   server_observed_at="2099-04-01T06:30:00Z", evidence={"age_seconds": None})
    if kind == "audit":
        state["overview_reply"] = (NON_OPERATIONAL_ERROR, 503)
        classification, message = "NON_OPERATIONAL/AUDIT_UNAVAILABLE", "could not confirm the access audit"
    elif kind == "provider":
        state["overview_reply"] = (PROVIDER_ERROR, 503)
        classification, message = "NON_OPERATIONAL/PROVIDER_ERROR", "could not supply valid evidence"
    elif kind == "malformed":
        state["overview_reply"] = ({**NON_OPERATIONAL_ERROR, "data": {"unverified": "do not display"}}, 503)
        classification, message = "INVALID/INTEGRITY_FAILED", "could not be validated"
    else:
        if kind == "unregistered":
            payload["reason"] = "ADAPTER_NOT_REGISTERED"
        state["overview_reply"] = (payload, 200)
        classification = "UNAVAILABLE/DATA_MISSING"
        message = "overview is not configured" if kind == "unregistered" else "No operational values disclosed"
    page.goto("http://127.0.0.1:5055/operator-ui")
    panel = page.locator('[data-resource="overview"]')
    expect(panel.locator(".resource-state")).to_have_text(classification)
    expect(panel).to_contain_text(message)
    expect(panel).not_to_contain_text("NON_OPERATIONAL/OFFLINE")
    expect(panel).not_to_contain_text("do not display")
    if kind in ("empty", "unregistered"):
        expect(panel.locator("summary")).to_contain_text("2099-04-01T06:30:00Z")
    else:
        expect(panel.locator("summary")).to_have_text("Source and freshness evidence unavailable")
        expect(page.locator("#prediction-submit")).to_be_disabled()


@pytest.mark.parametrize("classification,races,reason", [
    ("STALE", None, "too old"),
    ("INVALID/INTEGRITY_FAILED", None, "failed verification"),
    ("UNAVAILABLE/DATA_MISSING", None, "unavailable"),
    ("AVAILABLE/FRESH", [], "No upcoming races"),
])
def test_refresh_removes_previous_races_and_explains_blocked_prediction(console, classification, races, reason):
    page, state = console
    page.goto("http://127.0.0.1:5055/operator-ui")
    page.get_by_role("button", name="Select Bulli R2").click()
    expect(page.locator("#prediction-submit")).to_be_enabled()
    state["classification"] = classification
    if races is not None:
        state["races"] = races
    page.get_by_role("button", name="Refresh upcoming races", exact=True).click()
    expect(page.locator("#prediction-readiness")).to_contain_text(reason)
    expect(page.locator("#manual-prediction")).to_be_visible()
    expect(page.locator("#prediction-submit")).to_be_disabled()
    expect(page.locator("#prediction-race option[value='exact-race-1']")).to_have_count(0)
    expect(page.get_by_role("button", name="Select Bulli R2")).to_have_count(0)
    assert not state["posts"]
