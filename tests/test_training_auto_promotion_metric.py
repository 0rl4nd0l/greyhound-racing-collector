import importlib
import os

import pytest


class DummyPopen:
    def __init__(
        self,
        cmd,
        stdout=None,
        stderr=None,
        text=None,
        bufsize=None,
        universal_newlines=None,
        env=None,
    ):
        # Simulate trainer output: one non-JSON line, then metrics as JSON
        self._lines = [
            "Starting training...\n",
            '{"metrics": {"test_accuracy": 0.51, "test_auc": 0.63}}\n',
        ]
        self._idx = 0
        self.returncode = 0
        self.pid = 12345
        # stdout needs to provide readline
        self.stdout = self
        self.stderr = None

    # Interface used by app.run_training_background
    def readline(self):
        if self._idx < len(self._lines):
            line = self._lines[self._idx]
            self._idx += 1
            return line
        return ""

    def poll(self):
        # Non-None indicates process finished; return 0 to match success
        return 0

    def wait(self):
        return 0


class FakeRegistry:
    def __init__(self, calls):
        self.calls = calls

    def set_best_selection_policy(self, metric):
        self.calls.append(("set", metric))

    def auto_promote_best_by_metric(self, metric, prediction_type="win"):
        self.calls.append(("promote", metric, prediction_type))
        return "fake_model_id"

    def get_best_model(self):
        return None


@pytest.fixture(autouse=True)
def _ensure_testing_env(monkeypatch):
    # Encourage app to take test-friendly branches
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("WATCH_UPCOMING", "0")
    monkeypatch.setenv("WATCH_DOWNLOADS", "0")
    yield


def _bootstrap_app(monkeypatch):
    # Import/reload app fresh
    if "app" in list(globals().keys()):
        import app as _app

        importlib.reload(_app)
    import app as _app

    return _app


def test_automated_training_auto_promotes_with_configured_metric(monkeypatch):
    # Configure promotion metric
    monkeypatch.setenv("REGISTRY_PROMOTE_METRIC", "top1_rate")

    # Bootstrap app
    app = _bootstrap_app(monkeypatch)

    # Patch script existence to avoid depending on files
    real_exists = os.path.exists

    def fake_exists(path):
        p = str(path)
        if p.endswith("scripts/train_register_v4_gb.py") or p.endswith(
            "scripts/train_register_v4_hgb.py"
        ):
            return True
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", fake_exists)

    # Patch Popen to avoid running real subprocess
    monkeypatch.setattr(app.subprocess, "Popen", DummyPopen)

    # Patch get_model_registry (imported inside function) to capture calls
    calls = []
    import model_registry as _mr

    monkeypatch.setattr(_mr, "get_model_registry", lambda: FakeRegistry(calls))

    # Execute background training synchronously
    app.run_training_background("automated_training")

    # Verify registry calls show metric 'top1_rate'
    assert any(c[0] == "set" and c[1] == "top1_rate" for c in calls), calls
    assert any(
        c[0] == "promote" and c[1] == "top1_rate" and c[2] == "win" for c in calls
    ), calls
