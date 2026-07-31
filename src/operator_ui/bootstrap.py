"""Finite startup binding for an already-constructed live evidence adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from flask import Flask

from .api import register_level_1_provider
from .live_adapters import LiveEvidenceAdapters
from .r3_api import R3Services, install_r3_api
from .job_store import Job, JobStore


CONFIG_KEY = "OPERATOR_UI_LIVE_EVIDENCE_ADAPTERS"
R3_CONFIG_KEY = "OPERATOR_UI_R3_SERVICES"
R3_RUNTIME_CONFIG_KEY = "OPERATOR_UI_R3_RUNTIME"
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
    runtime = app.config.get(R3_RUNTIME_CONFIG_KEY)
    if services is not None and runtime is not None:
        raise TypeError("configure one R3 composition source")
    if runtime is not None:
        services = build_r3_services(runtime)
    if services is None:
        return False
    if type(services) is not R3Services:
        raise TypeError("configured R3 services must be exact R3Services")
    return install_r3_api(app, services)


@dataclass(frozen=True, slots=True)
class R3RuntimeConfig:
    """Trusted startup-only allowlist; disabled is the authoritative default."""
    enabled: bool = False
    job_store_path: Path | None = None
    separate_from: tuple[Path, ...] = ()
    resolve_submission: Callable[[Mapping[str, str], Any], Any] | None = None
    launch_once: Callable[[str], None] | None = None
    read_verified_result: Callable[[Job], Mapping[str, Any] | None] | None = None
    clock: Callable[[], Any] | None = None


def build_r3_services(config: R3RuntimeConfig) -> R3Services | None:
    if type(config) is not R3RuntimeConfig:
        raise TypeError("R3 runtime configuration must be typed")
    if config.enabled is not True:
        return None
    if config.job_store_path is None or not config.separate_from:
        raise ValueError("enabled R3 requires isolated fixed stores")
    callbacks=(config.resolve_submission,config.launch_once,config.read_verified_result,config.clock)
    if any(not callable(value) for value in callbacks):
        raise ValueError("enabled R3 requires fixed repository callbacks")
    store=JobStore(config.job_store_path,separate_from=config.separate_from)
    return R3Services(store,config.resolve_submission,config.launch_once,config.read_verified_result,clock=config.clock)


__all__ = ["CONFIG_KEY", "R3_CONFIG_KEY", "R3_RUNTIME_CONFIG_KEY", "R3RuntimeConfig", "build_r3_services", "bind_configured_live_evidence", "bind_configured_r3"]
