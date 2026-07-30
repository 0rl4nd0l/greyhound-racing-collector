import ast
import re
from pathlib import Path
from unittest.mock import patch

from flask import render_template_string

import app as dashboard_app

EXPECTED_STATES = {
    "healthy": "AVAILABLE/FRESH",
    "stale": "STALE",
    "unavailable": "UNAVAILABLE/DATA_MISSING",
    "waiting": "WAITING",
    "running": "RUNNING",
    "blocked": "BLOCKED",
}
EXPECTED_AREAS = {
    "next-race": "Next race",
    "collector-summary": "Collector summary",
    "corpus-funnel": "Corpus funnel",
    "model-identity": "Model identity",
    "recent-predictions": "Recent predictions",
    "system-health": "System health",
    "activity-feed": "Activity feed",
}


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
    assert html.count("PROTOTYPE DATA") >= 35
    assert "RESEARCH ONLY — NOT FOR BETTING" in html
    assert "<aside" in html
    assert "<nav" in html
    assert "<main" in html
    assert "<header" in html
    assert "<footer" in html
    assert 'class="status status--' in html
    assert html.count("Updated at") == 7
    assert html.count("Evidence source") == 7
    assert "illustrative" in html.lower()
    assert "nothing here is live" in html.lower()
    assert "/static/css/operator-ui.css" in html
    assert "/static/css/style.css" not in html
    assert "/static/js/a11y.js" not in html
    assert 'id="mode-banner"' not in html
    assert "window.E2E_DISABLE_REALTIME" not in html
    assert "onclick=" not in html
    assert "<form" not in html
    assert 'method="post"' not in html.lower()
    assert "Launch prediction" in html
    assert re.search(r"<button[^>]*disabled", html)
    assert (
        "2099-04-01 · Sandown Park · Race 6 · Jump 09:30 UTC · "
        "Fixture race ID FIXTURE-RACE-20990401-SANDOWN-R06"
    ) in html
    for area_id in EXPECTED_AREAS:
        assert f'href="#{area_id}"' in html


def test_dashboard_renders_all_areas_states_and_adjacent_prototype_values():
    response = dashboard_app.app.test_client().get("/operator-ui/prototype")
    html = response.get_data(as_text=True)

    for area_id, title in EXPECTED_AREAS.items():
        match = re.search(
            rf'<article[^>]+id="{area_id}".*?</article>',
            html,
            flags=re.DOTALL,
        )
        assert match, f"missing dashboard area {area_id}"
        card = match.group(0)
        assert title in card
        assert "Updated at" in card
        assert "Evidence source" in card
        assert card.count("PROTOTYPE DATA") >= 4
        for fixture_value in re.findall(
            r'<span class="fixture-value[^"]*">(.*?)</span>\s*</span>',
            card,
            flags=re.DOTALL,
        ):
            assert "PROTOTYPE DATA" in fixture_value

    for tone, label in EXPECTED_STATES.items():
        assert f"status--{tone}" in html
        assert label in html


def test_component_macros_render_all_six_typed_variants():
    template = """
    {% from "operator_ui_components.html" import status %}
    {% for tone, label in states.items() %}{{ status(label, tone) }}{% endfor %}
    """
    with dashboard_app.app.test_request_context():
        html = render_template_string(
            template,
            states=EXPECTED_STATES,
        )

    assert html.count('class="status status--') == 6
    assert html.count("PROTOTYPE DATA") == 6
    for tone, label in EXPECTED_STATES.items():
        assert f"status--{tone}" in html
        assert label in html


def test_prototype_route_has_no_operational_collaborators_or_post_controls():
    source = Path(dashboard_app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    route_function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "operator_ui_prototype"
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for statement in route_function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls == {"render_template"}

    html = dashboard_app.app.test_client().get(
        "/operator-ui/prototype"
    ).get_data(as_text=True)
    forbidden = (
        "<form",
        'method="post"',
        "service call",
        "database",
        "subprocess",
        "collector action",
        "prediction submission",
    )
    for value in forbidden:
        assert value not in html.lower()


def test_existing_root_route_is_still_distinct():
    assert dashboard_app.app.view_functions["index"] is not (
        dashboard_app.app.view_functions["operator_ui_prototype"]
    )
