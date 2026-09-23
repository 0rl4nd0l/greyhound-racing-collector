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
        operation.accept_data()
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


def test_queued_denial_is_drained_before_navigation(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace
    from tests.fixtures.freshness_transport.fake_cdp import BrowserTransport
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked, SourceOperation
    from utils.sportsbet_browser import create_sportsbet_driver

    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(tmp_path / 'access.json'))
    SportsbetAccess().initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    processing, release, navigated = threading.Event(), threading.Event(), threading.Event()
    original = SourceOperation.response
    def paused_response(self, *args):
        processing.set()
        assert release.wait(3)
        return original(self, *args)
    monkeypatch.setattr(SourceOperation, 'response', paused_response)
    transport = BrowserTransport()
    driver = SimpleNamespace(service=SimpleNamespace(process=transport.process),
        start_devtools=lambda: (None, transport), get=lambda url: navigated.set(),
        get_log=lambda name: [], quit=transport.quit)
    driver = create_sportsbet_driver(lambda: driver)
    errors = []
    def navigate():
        try:
            driver.get('https://www.sportsbet.com.au/next')
        except SportsbetAccessBlocked as error:
            errors.append(error)
    worker = threading.Thread(target=navigate)
    try:
        transport.response('https://www.sportsbet.com.au/previous', 429)
        assert processing.wait(2)
        worker.start()
        assert not navigated.wait(.1)
        release.set()
        worker.join(timeout=3)
        assert not worker.is_alive() and len(errors) == 1 and not navigated.is_set()
    finally:
        release.set()
        driver.quit()


def test_http_date_guidance_preserves_server_interval_and_unknown_reset_holds(tmp_path):
    from utils.sportsbet_access import SportsbetAccess
    gate = SportsbetAccess(tmp_path / 'access.json', clock=lambda: 1000)
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    gate.retain_denial(429, {'Date': 'Thu, 01 Jan 1970 00:00:00 GMT',
                            'Retry-After': 'Thu, 01 Jan 1970 03:00:00 GMT'})
    assert gate.read()['not_before'] == 11800
    gate.retain_denial(429, {'X-RateLimit-Reset': 'unclear-units'})
    assert gate.read()['phase'] == 'STOP'

@pytest.mark.parametrize('payload', [b'<html>challenge</html>', b'{}', b'[]'])
def test_python_recovery_requires_usable_next_events(tmp_path, monkeypatch, payload):
    import time
    import requests
    from utils.http_client import SourceCoordinatedSession
    from utils.prejump_sportsbet import fetch_sportsbet_next_events_snapshot
    from utils.sportsbet_access import SportsbetAccess

    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(tmp_path / 'access.json'))
    gate = SportsbetAccess()
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    SportsbetAccess(clock=lambda: time.time() - 1801).retain_denial(429)
    def transport(adapter, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = payload
        response.request = request
        return response
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    fetch_sportsbet_next_events_snapshot(session=SourceCoordinatedSession())
    assert gate.read()['phase'] == 'STOP'
    assert gate.read()['recovery_attempts'] == 1
    assert len(gate.read()['denials']) == 1


def test_browser_http_success_alone_cannot_complete_recovery(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace
    from tests.fixtures.freshness_transport.fake_cdp import BrowserTransport
    from utils.sportsbet_access import SportsbetAccess
    from utils.sportsbet_browser import create_sportsbet_driver

    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(tmp_path / 'access.json'))
    gate = SportsbetAccess()
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    SportsbetAccess(clock=lambda: time.time() - 1801).retain_denial(429)
    transport = BrowserTransport()
    class Driver:
        service = SimpleNamespace(process=transport.process)
        def start_devtools(self): return None, transport
        def get(self, url): transport.response(url, 200)
        def get_log(self, name): return []
        def quit(self): transport.quit()
    driver = create_sportsbet_driver(Driver)
    try:
        driver.get('https://www.sportsbet.com.au/fixture')
    finally:
        driver.quit()
    assert gate.read()['phase'] == 'STOP'
    assert gate.read()['recovery_attempts'] == 1


def test_python_recovery_accepts_current_identifiable_metadata(tmp_path, monkeypatch):
    import time
    import requests
    from utils.http_client import SourceCoordinatedSession
    from utils.prejump_sportsbet import fetch_sportsbet_next_events_snapshot
    from utils.sportsbet_access import SportsbetAccess

    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(tmp_path / 'access.json'))
    gate = SportsbetAccess()
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    SportsbetAccess(clock=lambda: time.time() - 1801).retain_denial(429)
    def transport(adapter, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps([dict(id=123, classId='4', raceNumber=1,
            competitionName='Sale', startTime=time.time() + 600, trackStatus='Good')]).encode()
        response.request = request
        return response
    monkeypatch.setattr(requests.adapters.HTTPAdapter, 'send', transport)
    snapshot = fetch_sportsbet_next_events_snapshot(session=SourceCoordinatedSession())
    assert len(snapshot['events']) == 1
    assert gate.read()['phase'] == 'OPEN'
    assert gate.read()['recovery_attempts'] == 1


def test_operating_policy_stops_combined_lane_bursts_and_survives_restart(tmp_path):
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked
    gate = SportsbetAccess(tmp_path / 'access.json', clock=lambda: 1000)
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    with gate.locked():
        value = gate.read()
        value['operating_policy'] = {'reference': 'fabricated bounded trial',
            'python_per_60_seconds': 2, 'browser_per_60_seconds': 1,
            'browser_navigation_cap': 2}
        gate.write(value)
    for kind in ('python', 'browser', 'python'):
        with gate.operation(kind):
            pass
    with pytest.raises(SportsbetAccessBlocked, match='operating_policy'):
        with gate.operation('python'):
            pytest.fail('combined source burst admitted')
    assert gate.read()['phase'] == 'STOP'
    assert len(gate.read()['operations']) == 3
    with pytest.raises(SportsbetAccessBlocked):
        with SportsbetAccess(gate.path, clock=lambda: 2000).operation('browser'):
            pytest.fail('policy stop expired on restart')
