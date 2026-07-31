import pytest
from flask import Flask
from werkzeug.security import generate_password_hash

from src.operator_ui.api import install_level_1_api, register_level_1_provider
from src.operator_ui.bootstrap import CONFIG_KEY, bind_configured_live_evidence
from src.operator_ui.live_adapters import LiveEvidenceAdapters
from src.operator_ui.security import install_connected_mode


def installed_app(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True, OPERATOR_UI_CONNECTED_MODE=True,
        OPERATOR_UI_SECRET_KEY="stable-connected-secret-" + "x" * 32,
        OPERATOR_UI_USERNAME="viewer",
        OPERATOR_UI_PASSWORD_HASH=generate_password_hash("correct horse"),
        OPERATOR_UI_LEVEL=1,
        OPERATOR_UI_AUDIT_DB_PATH=str(tmp_path / "audit.sqlite3"),
        OPERATOR_UI_JOB_DB_PATH=str(tmp_path / "jobs.sqlite3"),
        DATABASE_PATH=str(tmp_path / "canonical.sqlite3"),
        OPERATOR_UI_DEPLOYED_COMMIT="c" * 40,
        OPERATOR_UI_DEPLOYED_TREE="d" * 40,
        OPERATOR_UI_DEPLOYED_VERSION="test-v1",
    )
    install_connected_mode(app)
    assert install_level_1_api(app)
    return app


def exact_adapter():
    return object.__new__(LiveEvidenceAdapters)


def test_missing_config_is_fail_closed_and_side_effect_free(tmp_path):
    app = installed_app(tmp_path)
    assert bind_configured_live_evidence(app) is False
    assert app.extensions["operator_ui_level_1_api_providers"] == {}


def test_exact_adapter_is_bound_all_at_once_without_overview_or_audit(tmp_path):
    app, adapter = installed_app(tmp_path), exact_adapter()
    app.config[CONFIG_KEY] = adapter
    assert bind_configured_live_evidence(app) is True
    registry = app.extensions["operator_ui_level_1_api_providers"]
    assert set(registry) == {"upcoming_races", "race_detail", "recent_predictions", "prediction_detail", "collector", "corpus", "models", "system"}
    assert "overview" not in registry and "audit" not in registry
    assert registry["upcoming_races"] == adapter.upcoming


def test_unknown_partial_duplicate_and_replacement_bindings_are_denied(tmp_path):
    app = installed_app(tmp_path)
    app.config[CONFIG_KEY] = object()
    with pytest.raises(TypeError): bind_configured_live_evidence(app)
    adapter = exact_adapter()
    app.config[CONFIG_KEY] = adapter
    register_level_1_provider(app, "system", adapter.system)
    with pytest.raises(ValueError): bind_configured_live_evidence(app)
    assert set(app.extensions["operator_ui_level_1_api_providers"]) == {"system"}
    app = installed_app(tmp_path / "second"); app.config[CONFIG_KEY] = adapter
    bind_configured_live_evidence(app)
    with pytest.raises(RuntimeError): bind_configured_live_evidence(app)


def test_binding_occurs_only_when_called_never_during_requests(tmp_path):
    app, adapter = installed_app(tmp_path), exact_adapter()
    app.config[CONFIG_KEY] = adapter
    client = app.test_client()
    client.get("/operator-ui/api/v1/overview")
    assert app.extensions["operator_ui_level_1_api_providers"] == {}
