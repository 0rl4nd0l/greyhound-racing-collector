import ast
import re
from pathlib import Path

from flask import Flask, render_template

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "static/js/operator-ui-connected.js"


def test_connected_client_uses_only_fixed_get_api_and_bounded_returned_ids():
    source = SCRIPT.read_text(encoding="utf-8")
    assert set(re.findall(r"['\"](/operator-ui/api/v1/[^'\"]+)['\"]", source)) == {
        "/operator-ui/api/v1/overview", "/operator-ui/api/v1/races/upcoming",
        "/operator-ui/api/v1/races/", "/operator-ui/api/v1/predictions/recent",
        "/operator-ui/api/v1/predictions/", "/operator-ui/api/v1/collector",
        "/operator-ui/api/v1/corpus", "/operator-ui/api/v1/models",
        "/operator-ui/api/v1/system", "/operator-ui/api/v1/audit",
        "/operator-ui/api/v1/prediction-jobs",
        "/operator-ui/api/v1/r3-capability",
    }
    assert "method:'GET'" in source
    assert "safeId.test(id)" in source and "encodeURIComponent(id)" in source
    assert "method:'POST'" in source and "X-CSRF-Token" in source
    state_source = (ROOT / "static/js/operator-ui-state.js").read_text(encoding="utf-8")
    assert "setTimeout" in state_source and "EventSource" not in source + state_source
    for prohibited in ("URLSearchParams", "FormData", "XMLHttpRequest", "WebSocket"):
        assert prohibited not in source


def test_connected_client_has_explicit_isolated_truthful_accessible_states():
    source = SCRIPT.read_text(encoding="utf-8")
    for state in ("AVAILABLE/FRESH", "STALE", "DIVERGENT", "INVALID/INTEGRITY_FAILED", "NON_OPERATIONAL/PROVIDER_ERROR", "NON_OPERATIONAL/OFFLINE", "NON_OPERATIONAL/AUTHENTICATION_REQUIRED", "Loading"):
        assert state in source
    assert "Promise.allSettled" in source
    assert "Resources were fetched independently" in source
    assert "'/operator-ui/login'" in source
    assert "beforeprint" in source and "afterprint" in source
    css = (ROOT / "static/css/operator-ui.css").read_text(encoding="utf-8")
    for state in ("fresh", "stale", "divergent", "invalid", "unavailable"):
        assert f"resource-panel--{state}" in css
    assert 'content: "Status: "' in css


def test_prediction_intent_and_reconnect_are_durable_and_bounded():
    source = SCRIPT.read_text(encoding="utf-8")
    state = (ROOT / "static/js/operator-ui-state.js").read_text(encoding="utf-8")
    assert "createOperatorState" in source and "r3-capability" in source
    assert "operatorUiPredictionIntentV1" in state and "maximum = 6" in state
    assert "modelCatalog" in source
    assert "prediction-retransmit" in source and "retransmitButton.type='button'" in source
    assert "retransmitButton.addEventListener('click'" in source


def test_connected_template_default_off_and_prototype_are_distinct():
    app = Flask(__name__, template_folder=str(ROOT / "templates"), static_folder=str(ROOT / "static"))
    app.add_url_rule("/operator-ui/prototype", endpoint="operator_ui_prototype", view_func=lambda: "")
    with app.test_request_context("/operator-ui"):
        connected = render_template("operator_ui_connected.jinja", connected=True)
        disabled = render_template("operator_ui_connected.jinja", connected=False)
    assert "operator-ui-connected.js" in connected
    assert "operator-ui-connected.js" not in disabled
    assert "CONNECTED MODE OFF" in disabled
    assert "UNAVAILABLE/DATA_MISSING" in disabled
    assert "PROTOTYPE DATA" not in connected
    assert "RESEARCH ONLY — NOT FOR BETTING" in connected


def test_connected_route_source_has_no_direct_operational_access():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "operator_ui_connected")
    source = ast.unparse(function).lower()
    for prohibited in ("sqlite", "subprocess", "open(", "systemctl", "database", "collector", "browser", "lock"):
        assert prohibited not in source
