import importlib
import json
import os

import pytest


class FakeRegistry:
    def __init__(self, calls):
        self.calls = calls

    def set_best_selection_policy(self, metric):
        self.calls.append(("set", metric))

    def auto_promote_best_by_metric(self, metric, prediction_type="win"):
        self.calls.append(("promote", metric, prediction_type))
        return "fake_model_id"

    def get_best_model(self):
        # Return None to avoid dataclasses.asdict() on non-dataclass
        return None


@pytest.fixture(autouse=True)
def _ensure_testing_env(monkeypatch):
    # Put app in test-friendly mode before import
    monkeypatch.setenv("TESTING", "1")
    # Ensure watchers are off by default
    monkeypatch.setenv("WATCH_UPCOMING", "0")
    monkeypatch.setenv("WATCH_DOWNLOADS", "0")
    yield


def _bootstrap_app(monkeypatch):
    # Import (or reload) the app fresh for each test to ensure clean state
    if "app" in list(globals().keys()):
        import app as _app

        importlib.reload(_app)
    import app as _app

    return _app


def test_promote_uses_top1_rate_by_env(monkeypatch, tmp_path):
    # Configure metric via env
    monkeypatch.setenv("REGISTRY_PROMOTE_METRIC", "top1_rate")
    app = _bootstrap_app(monkeypatch)

    calls = []
    # Patch registry accessor
    monkeypatch.setattr(app, "get_model_registry", lambda: FakeRegistry(calls))

    client = app.app.test_client()
    resp = client.post("/api/model_registry/promote_correct_winners")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data and data.get("success") is True

    # Verify calls
    assert any(c[0] == "set" and c[1] == "top1_rate" for c in calls)
    assert any(
        c[0] == "promote" and c[1] == "top1_rate" and c[2] == "win" for c in calls
    )

    # Verify refresh signal payload
    signal_path = os.path.join("model_registry", "refresh_signal.json")
    assert os.path.exists(signal_path), "refresh_signal.json not created"
    with open(signal_path, "r", encoding="utf-8") as f:
        sig = json.load(f)
    assert sig.get("selection_policy") == "top1_rate"
    assert sig.get("prediction_type") == "win"


def test_promote_uses_correct_winners_when_configured(monkeypatch):
    # Configure metric via env
    monkeypatch.setenv("REGISTRY_PROMOTE_METRIC", "correct_winners")
    app = _bootstrap_app(monkeypatch)

    calls = []
    monkeypatch.setattr(app, "get_model_registry", lambda: FakeRegistry(calls))

    client = app.app.test_client()
    resp = client.post("/api/model_registry/promote_correct_winners")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data and data.get("success") is True

    # Verify calls
    assert any(c[0] == "set" and c[1] == "correct_winners" for c in calls)
    assert any(
        c[0] == "promote" and c[1] == "correct_winners" and c[2] == "win" for c in calls
    )

    # Verify refresh signal payload
    signal_path = os.path.join("model_registry", "refresh_signal.json")
    assert os.path.exists(signal_path), "refresh_signal.json not created"
    with open(signal_path, "r", encoding="utf-8") as f:
        sig = json.load(f)
    assert sig.get("selection_policy") == "correct_winners"
    assert sig.get("prediction_type") == "win"
