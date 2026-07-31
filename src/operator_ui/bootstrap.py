"""Finite startup binding for an already-constructed live evidence adapter."""

from __future__ import annotations

from flask import Flask

from .api import register_level_1_provider
from .live_adapters import LiveEvidenceAdapters
from .r3_api import R3Services, install_r3_api


CONFIG_KEY = "OPERATOR_UI_LIVE_EVIDENCE_ADAPTERS"
R3_CONFIG_KEY = "OPERATOR_UI_R3_SERVICES"
_BOUND_KEY = "operator_ui_live_evidence_bound"
_METHODS = (
    ("upcoming_races", "upcoming"),
    ("race_detail", "race_detail"),
    ("recent_predictions", "recent_predictions"),
    ("prediction_detail", "prediction_detail"),
    ("collector", "collector"),
    ("corpus", "corpus"),
    ("models", "models"),
    ("system", "system"),
)


def bind_configured_live_evidence(app: Flask) -> bool:
    """Bind the exact configured adapter once, at application bootstrap."""
    if not isinstance(app, Flask):
        raise TypeError("a Flask application is required")
    adapter = app.config.get(CONFIG_KEY)
    if adapter is None:
        return False
    if type(adapter) is not LiveEvidenceAdapters:
        raise TypeError("configured live evidence must be an exact LiveEvidenceAdapters")
    if app.extensions.get(_BOUND_KEY):
        raise RuntimeError("live evidence adapter is already bound")

    registry = app.extensions.get("operator_ui_level_1_api_providers")
    if not isinstance(registry, dict):
        raise RuntimeError("Operator UI Level-1 API is not installed")
    resources = tuple(resource for resource, _method in _METHODS)
    if any(resource in registry for resource in resources):
        raise ValueError("partial or replacement live evidence binding is forbidden")

    providers = tuple((resource, getattr(adapter, method)) for resource, method in _METHODS)
    if any(not callable(provider) for _resource, provider in providers):
        raise TypeError("live evidence adapter method is not callable")
    for resource, provider in providers:
        register_level_1_provider(app, resource, provider)
    app.extensions[_BOUND_KEY] = adapter
    return True


def bind_configured_r3(app: Flask) -> bool:
    """Bind the explicit server-owned R3 composition; default is safely off."""
    services = app.config.get(R3_CONFIG_KEY)
    if services is None:
        return False
    if type(services) is not R3Services:
        raise TypeError("configured R3 services must be exact R3Services")
    return install_r3_api(app, services)


__all__ = ["CONFIG_KEY", "R3_CONFIG_KEY", "bind_configured_live_evidence", "bind_configured_r3"]
