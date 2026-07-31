import ast
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

import app as dashboard_app
from utils import module_monitor


class _PrototypeMarkerParser(HTMLParser):
    wrapper_classes = ("fixture-value", "status")
    void_elements = {
        "area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr",
    }

    def __init__(self):
        super().__init__()
        self.wrappers = {class_name: [] for class_name in self.wrapper_classes}
        self._stack = []

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "").split()
        if "prototype-marker" in classes and self._stack:
            for class_name, index in self._stack[-1][1]:
                self.wrappers[class_name][index] = True

        frame = []
        for class_name in self.wrapper_classes:
            if class_name in classes:
                frame.append((class_name, len(self.wrappers[class_name])))
                self.wrappers[class_name].append(False)
        if tag not in self.void_elements:
            self._stack.append((tag, frame))

    def handle_endtag(self, tag):
        assert self._stack[-1][0] == tag
        self._stack.pop()


def _unexpected_call(*_args, **_kwargs):
    raise AssertionError("prototype route called an operational collaborator")


def _html(path="/operator-ui"):
    response = dashboard_app.app.test_client().get(path)
    assert response.status_code == 200
    return response.get_data(as_text=True)


def test_operator_ui_get_is_fixture_only_and_has_persistent_boundaries():
    with (
        patch.object(dashboard_app.sqlite3, "connect", side_effect=_unexpected_call),
        patch.object(dashboard_app.subprocess, "run", side_effect=_unexpected_call),
        patch.object(dashboard_app.subprocess, "Popen", side_effect=_unexpected_call),
        patch.object(dashboard_app.db_manager, "get_database_stats", side_effect=_unexpected_call),
        patch.object(dashboard_app.db_manager, "get_recent_races", side_effect=_unexpected_call),
        patch.object(
            module_monitor, "log_request_modules", side_effect=_unexpected_call
        ) as monitor_log,
    ):
        html = _html("/operator-ui")
        _html("/operator-ui/prototype")
        monitor_log.assert_not_called()

    document = _PrototypeMarkerParser()
    document.feed(html)
    assert document.wrappers["fixture-value"]
    assert document.wrappers["status"]
    assert all(document.wrappers["fixture-value"])
    assert all(document.wrappers["status"])
    assert html.count("RESEARCH ONLY — NOT FOR BETTING") >= 3
    assert "<form" not in html
    assert "onclick=" not in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "/static/js/operator-ui-prototype.js" in html


def test_exact_race_binding_and_all_fail_closed_fixture_reasons_render():
    html = _html()
    for value in (
        "race_01JQ7SANDOWN20990401R06",
        "CAN-AU-VIC-20990401-SAN-R06 / TD-987654",
        "https://fixture.invalid/racing/sandown-park/2099-04-01/6",
        "sandown-park",
        "2099-04-01T10:30:00Z",
        "Australia/Melbourne",
        "90766b65ba7f184d53b57c520fd9af1962797c9370984769d93eecc631716cea",
        "dog_ember_01 / TD-DOG-101",
        "ACTIVE",
    ):
        assert value in html
    script = Path("static/js/operator-ui-prototype.js").read_text()
    for reason in (
        "ambiguous race identity",
        "scheduled jump has passed",
        "runner identity is missing",
        "300-second",
        "scheduled jump identity is missing",
        "window or model configuration is unsupported",
        "evidence conflict",
        "evidence is unavailable",
    ):
        assert reason in script
    assert "fetch" not in script
    assert "XMLHttpRequest" not in script


def test_lifecycle_outcomes_and_governance_surfaces_are_complete():
    html = _html()
    for phase in (
        "SUBMITTED", "VALIDATED", "WAITING_FOR_CLAIM", "CLAIMED",
        "ATTEMPT_STARTED", "RESPONSE_RECORDED", "RECEIPT_VERIFIED",
        "CONSUMED", "SCORING", "PREDICTION_READY",
    ):
        assert phase in html
    for outcome in (
        "RECEIPT_READY", "REQUEST_EXPIRED", "RACE_NOT_FOUND",
        "CAPTURE_WINDOW_CLOSED", "IDENTITY_MISMATCH", "CAPTURE_FAILED",
        "BUSY", "CANCELLED", "INSUFFICIENT_PREJUMP_MARGIN",
    ):
        assert outcome in html
    for section in ("collector", "corpus", "models", "system", "audit"):
        assert f'id="{section}"' in html
        assert f'href="#{section}"' in html


def test_result_copy_exposes_identity_without_prohibited_forecasting_claims():
    html = _html()
    result = html.split('id="prediction-result"', 1)[1].split("</section>", 1)[0].lower()
    for identity in ("model", "configuration", "source", "bundle"):
        assert identity in result
    for prohibited in ("best bet", "edge", "staking", "profit", "race outcome:"):
        assert prohibited not in result


def test_route_is_pure_render_template_and_root_remains_distinct():
    source = Path(dashboard_app.__file__).read_text(encoding="utf-8")
    route_function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "operator_ui_prototype"
    )
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for statement in route_function.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls == {"render_template"}
    assert dashboard_app.app.view_functions["index"] is not dashboard_app.app.view_functions["operator_ui_prototype"]
    assert _html("/operator-ui/prototype") == _html("/operator-ui")
