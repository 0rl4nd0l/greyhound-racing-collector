from unittest.mock import patch

import app as dashboard_app


def _unexpected_call(*_args, **_kwargs):
    raise AssertionError("prototype route called an operational collaborator")


def test_operator_ui_prototype_is_fixture_only_and_semantic():
    client = dashboard_app.app.test_client()

    with (
        patch.object(dashboard_app.sqlite3, "connect", side_effect=_unexpected_call),
        patch.object(dashboard_app.subprocess, "run", side_effect=_unexpected_call),
        patch.object(dashboard_app.subprocess, "Popen", side_effect=_unexpected_call),
        patch.object(
            dashboard_app.db_manager,
            "get_database_stats",
            side_effect=_unexpected_call,
        ),
        patch.object(
            dashboard_app.db_manager,
            "get_recent_races",
            side_effect=_unexpected_call,
        ),
    ):
        response = client.get("/operator-ui/prototype")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert html.count("PROTOTYPE DATA") >= 5
    assert "RESEARCH ONLY — NOT FOR BETTING" in html
    assert "<aside" in html
    assert "<nav" in html
    assert "<main" in html
    assert "<header" in html
    assert "<footer" in html
    assert 'class="status status--' in html
    assert "Freshness" in html
    assert "Evidence reference" in html
    assert "Illustrative only" in html
    assert "not live" in html
    assert "/static/css/operator-ui.css" in html
    assert "/static/css/style.css" not in html
    assert "/static/js/a11y.js" not in html
    assert 'id="mode-banner"' not in html
    assert "window.E2E_DISABLE_REALTIME" not in html
    assert "onclick=" not in html


def test_existing_root_route_is_still_distinct():
    assert dashboard_app.app.view_functions["index"] is not (
        dashboard_app.app.view_functions["operator_ui_prototype"]
    )
