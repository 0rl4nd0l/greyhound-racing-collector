import json
import os
import subprocess
import sys

import pytest


def test_denial_survives_restart_and_allows_only_one_bounded_recovery(tmp_path):
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked

    path = tmp_path / "access.json"
    gate = SportsbetAccess(path, clock=lambda: 1000)
    gate.initialize(access_basis={"status": "permitted", "reference": "fabricated fixture"})
    with gate.operation("python") as operation:
        operation.response(429, {"Retry-After": "60"})
    restarted = SportsbetAccess(path, clock=lambda: 1059)
    with pytest.raises(SportsbetAccessBlocked):
        with restarted.operation("browser"):
            pytest.fail("cooldown admitted traffic")
    # Provider's shorter interval does not shorten our conservative 30-minute floor.
    recovered = SportsbetAccess(path, clock=lambda: 2800)
    with recovered.operation("python") as operation:
        assert operation.recovery
        operation.response(200, {})
    with recovered.operation("browser") as operation:
        operation.response(429, {})
    with pytest.raises(SportsbetAccessBlocked):
        with SportsbetAccess(path, clock=lambda: 999999).operation("python"):
            pytest.fail("renewed denial got a second recovery")
    state = json.loads(path.read_text())
    assert state["recovery_attempts"] == 1
    assert len(state["denials"]) == 2


def test_concurrent_and_crashed_process_cannot_reuse_source_admission(tmp_path):
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked

    path = tmp_path / "access.json"
    gate = SportsbetAccess(path)
    gate.initialize(access_basis={"status": "permitted", "reference": "fabricated fixture"})
    script = """
import sys,os
from utils.sportsbet_access import SportsbetAccess,SportsbetAccessBlocked
try:
    with SportsbetAccess(sys.argv[1]).operation('python'):
        os._exit(9)
except SportsbetAccessBlocked:
    sys.exit(4)
"""
    with gate.operation("browser"):
        result = subprocess.run([sys.executable, "-c", script, str(path)])
        assert result.returncode == 4
    assert subprocess.run([sys.executable, "-c", script, str(path)]).returncode == 9
    with pytest.raises(SportsbetAccessBlocked):
        with SportsbetAccess(path).operation("browser"):
            pytest.fail("process death cleared an uncertain operation")


@pytest.mark.parametrize("header,expected", [(None, 2800), ("nonsense", 2800), ("99999", 100999)])
def test_missing_invalid_and_stricter_guidance(tmp_path, header, expected):
    from utils.sportsbet_access import SportsbetAccess

    gate = SportsbetAccess(tmp_path / "access.json", clock=lambda: 1000)
    gate.initialize(access_basis={"status": "permitted", "reference": "fabricated fixture"})
    gate.retain_denial(429, {"Retry-After": header} if header else {})
    assert gate.read()["not_before"] == expected


@pytest.mark.parametrize("basis", ["unresolved", "prohibited"])
def test_retry_timing_never_grants_access(tmp_path, basis):
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked

    gate = SportsbetAccess(tmp_path / "access.json", clock=lambda: 999999)
    gate.initialize(access_basis={"status": basis, "reference": "fabricated fixture"})
    with pytest.raises(SportsbetAccessBlocked):
        with gate.operation("python"):
            pytest.fail("missing or prohibited access admitted")


def test_python_denial_blocks_browser_before_driver_creation(tmp_path, monkeypatch):
    import requests
    from utils.http_client import SourceCoordinatedSession
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    from utils.sportsbet_browser import create_sportsbet_driver

    path = tmp_path / "access.json"
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(path))
    SportsbetAccess().initialize(access_basis={"status": "permitted", "reference": "fabricated fixture"})
    def transport(adapter, request, **kwargs):
        result = requests.Response()
        result.status_code = 429
        result.headers = {"Retry-After": "60"}
        result.request = request
        return result
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", transport)
    client = SourceCoordinatedSession()
    assert client.get("https://www.sportsbet.com.au/fixture").status_code == 429
    with pytest.raises(SportsbetAccessBlocked):
        create_sportsbet_driver(lambda: pytest.fail("held browser was constructed"))
    with pytest.raises(SportsbetAccessBlocked):
        SourceCoordinatedSession().get("https://www.sportsbet.com.au/fixture")


def test_browser_observes_async_denial_without_another_navigation(tmp_path, monkeypatch):
    import threading
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    from utils.sportsbet_browser import create_sportsbet_driver

    path = tmp_path / "access.json"
    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(path))
    SportsbetAccess().initialize(access_basis={"status": "permitted", "reference": "fabricated fixture"})
    stopped = threading.Event()

    class Driver:
        pending = []
        navigations = 0
        closed = False
        def get(self, url):
            self.navigations += 1
        def get_log(self, name):
            rows, self.pending = self.pending, []
            return rows
        def execute_cdp_cmd(self, command, params):
            stopped.set()
        def quit(self):
            self.closed = True

    driver = create_sportsbet_driver(Driver)
    try:
        driver.get("https://www.sportsbet.com.au/fixture")
        driver.pending = [{"message": json.dumps({"message": {
            "method": "Network.responseReceived", "params": {"response": {
                "url": "https://www.sportsbet.com.au/fixture", "status": 429,
                "headers": {"Retry-After": "120"},
            }}}})}]
        assert stopped.wait(2), "asynchronous source traffic not stopped"
        with pytest.raises(SportsbetAccessBlocked):
            driver.get("https://www.sportsbet.com.au/second")
        assert driver.navigations == 1
    finally:
        driver.quit()
    assert driver.closed
    assert SportsbetAccess().read()["phase"] == "COOLDOWN"


def test_real_systemd_adapter_accepts_disabled_exit_status(monkeypatch):
    from scripts.run_freshness_rehearsal import SystemdControl

    def disabled(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="disabled\n")
    monkeypatch.setattr(subprocess, "check_output", disabled)
    assert SystemdControl().command("is-enabled", "synthetic.timer") == "disabled\n"
    with pytest.raises(subprocess.CalledProcessError):
        SystemdControl().command("start", "synthetic.timer")
