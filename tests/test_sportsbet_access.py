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


@pytest.mark.parametrize("failure", ["denial", "xhr_retry", "disconnect", "logs"])
def test_browser_observes_failure_during_pending_navigation(tmp_path, monkeypatch, failure):
    import threading
    from types import SimpleNamespace
    from tests.fixtures.freshness_transport.fake_cdp import BrowserTransport
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    from utils.sportsbet_browser import create_sportsbet_driver

    monkeypatch.setenv("GREYHOUND_SPORTSBET_ACCESS_STATE", str(tmp_path / "access.json"))
    SportsbetAccess().initialize(access_basis={"status": "permitted", "reference": "fabricated fixture"})
    entered = threading.Event()
    transport = BrowserTransport()

    class Driver:
        service = SimpleNamespace(process=transport.process)
        navigations = 0
        def start_devtools(self):
            return None, transport
        def get(self, url):
            self.navigations += 1
            entered.set()
            if failure in {"denial", "xhr_retry"}:
                assert transport.stopped.wait(3), "CDP could not stop pending WebDriver navigation"
            else:
                transport.process.wait(timeout=3)
        def get_log(self, name):
            raise RuntimeError("fabricated observation loss")
        def quit(self):
            transport.quit()

    driver = create_sportsbet_driver(Driver)
    errors = []
    def navigate():
        try:
            driver.get("https://www.sportsbet.com.au/fixture")
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=navigate)
    worker.start()
    try:
        assert entered.wait(2)
        if failure == "denial":
            transport.response("https://www.sportsbet.com.au/fixture", 429, {"Retry-After": "120"})
        elif failure == "xhr_retry":
            transport.response("https://www.sportsbet.com.au/fixture", 503, {"Retry-After": "3600"}, resource_type="XHR")
        elif failure == "disconnect":
            transport._ws.on_close(None)
        else:
            with pytest.raises(RuntimeError):
                driver.get_log("performance")
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], SportsbetAccessBlocked)
        with pytest.raises(SportsbetAccessBlocked):
            driver.get("https://www.sportsbet.com.au/second")
        assert driver.navigations == 1
    finally:
        driver.quit()
    assert transport.process.poll() is not None
    assert SportsbetAccess().read()["phase"] == ("COOLDOWN" if failure == "denial" else "STOP")
    with pytest.raises(SportsbetAccessBlocked):
        with SportsbetAccess().operation("python"):
            pytest.fail("restart bypassed hold")


def test_real_systemd_adapter_accepts_disabled_exit_status(monkeypatch):
    from scripts.run_freshness_rehearsal import SystemdControl

    def disabled(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], output="disabled\n")
    monkeypatch.setattr(subprocess, "check_output", disabled)
    assert SystemdControl().command("is-enabled", "synthetic.timer") == "disabled\n"
    with pytest.raises(subprocess.CalledProcessError):
        SystemdControl().command("start", "synthetic.timer")


def test_browser_cleanup_retains_descendant_after_service_parent_exits(tmp_path):
    import time
    from utils.sportsbet_browser import owned_processes, process_identity, stop_owned_processes

    marker = tmp_path / "child.pid"
    parent = subprocess.Popen([sys.executable, "-c", """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
open(sys.argv[1], 'w').write(str(child.pid))
time.sleep(300)
""", str(marker)])
    retained = {}
    birth = process_identity(parent.pid)[1]
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        child = int(marker.read_text())
        retained = owned_processes(parent.pid, birth)
        assert child in retained
        parent.terminate()
        parent.wait(timeout=3)
        stop_owned_processes(parent.pid, birth, retained)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                state = open(f'/proc/{child}/stat').read().rsplit(')', 1)[1].split()[0]
            except FileNotFoundError:
                break
            if state == 'Z':
                break
            time.sleep(.01)
        else:
            pytest.fail('reparented owned browser survived cleanup')
    finally:
        stop_owned_processes(parent.pid, birth, retained)
        if parent.poll() is None:
            parent.terminate()
        parent.wait(timeout=3)
