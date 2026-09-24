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
    if failure in {"denial", "xhr_retry"}:
        observed=SportsbetAccess().read()["denials"][-1]["response_observation"]
        assert observed["source_url_without_query"] == "https://www.sportsbet.com.au/fixture"
        assert observed["resource_type"] == ("XHR" if failure == "xhr_retry" else "Document")
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


@pytest.mark.parametrize("operation", ["navigation", "readiness"])
def test_queued_denial_is_drained_before_navigation(tmp_path, monkeypatch, operation):
    import threading
    from types import SimpleNamespace
    from tests.fixtures.freshness_transport.fake_cdp import BrowserTransport
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked, SourceOperation
    from utils.sportsbet_browser import create_sportsbet_driver

    monkeypatch.setenv('GREYHOUND_SPORTSBET_ACCESS_STATE', str(tmp_path / 'access.json'))
    SportsbetAccess().initialize(access_basis={'status': 'permitted', 'reference': 'fabricated fixture'})
    processing, release, navigated = threading.Event(), threading.Event(), threading.Event()
    original = SourceOperation.response
    def paused_response(self, *args, **kwargs):
        processing.set()
        assert release.wait(3)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SourceOperation, 'response', paused_response)
    transport = BrowserTransport()
    driver = SimpleNamespace(service=SimpleNamespace(process=transport.process),
        start_devtools=lambda: (None, transport), get=lambda url: navigated.set(),
        get_log=lambda name: [], quit=transport.quit)
    driver = create_sportsbet_driver(lambda: driver)
    errors = []
    def navigate():
        try:
            if operation == 'navigation':
                driver.get('https://www.sportsbet.com.au/next')
            else:
                driver.sportsbet_check_access()
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


def test_denial_observation_omits_credentials_queries_and_fragments(tmp_path):
    from utils.sportsbet_access import SportsbetAccess
    gate=SportsbetAccess(tmp_path/'access.json')
    gate.initialize(access_basis={'status':'permitted','reference':'invented offline test'})
    with gate.operation('python') as operation:
        operation.response(429,{},source_url='https://user:secret@www.sportsbet.com.au/api/events?token=private#secret',resource_type='python')
    observed=gate.read()['denials'][0]['response_observation']
    assert observed['source_url_without_query']=='https://www.sportsbet.com.au/api/events'
    assert observed['resource_type']=='python' and observed['monotonic_seconds']>0
    assert gate.read()['phase']=='COOLDOWN'


def test_denial_links_durable_operation_and_process_without_rewriting_history(tmp_path):
    import os
    from utils.sportsbet_access import SportsbetAccess
    gate = SportsbetAccess(tmp_path / 'access.json', clock=lambda: 10000)
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'synthetic fixture'})
    previous = {'at': 1, 'kind': 'python'}
    with gate.locked():
        value = gate.read()
        value['operating_policy'] = {'reference': 'synthetic fixture',
                                    'python_per_60_seconds': 10,
                                    'browser_per_60_seconds': 1,
                                    'browser_navigation_cap': 2}
        value['operations'] = [previous.copy()]
        value['recovery_attempts'] = 1
        gate.write(value)
    with gate.operation('python') as operation:
        active = gate.read()['active']
        row = gate.read()['operations'][-1]
        assert row['operation_id'] == active
        assert row['owner_pid'] == os.getpid()
        assert row['owner_parent_pid'] == os.getppid()
        assert row['owner_process_start_ticks']
        assert row['owner_boot_id']
        assert 'closed_at' not in row
        operation.response(429, {}, source_url='https://www.sportsbet.com.au/fixture')
        assert gate.read()['denials'][-1]['operation_id'] == active
    value = gate.read()
    assert value['operations'][0] == previous
    assert value['operations'][-1]['closed_at'] == 10000
    assert value['operations'][-1]['final_phase'] == 'STOP'
    assert value['recovery_attempts'] == 1


@pytest.mark.parametrize('kind', ['python', 'browser'])
def test_consumed_recovery_stop_blocks_independent_process_before_transport(tmp_path, kind):
    import os
    import subprocess
    import sys
    from utils.sportsbet_access import SportsbetAccess
    path = tmp_path / 'access.json'
    gate = SportsbetAccess(path)
    gate.initialize(access_basis={'status': 'permitted', 'reference': 'synthetic fixture'})
    with gate.locked():
        state = gate.read()
        state['recovery_attempts'] = 1
        gate.write(state)
    with gate.operation('python') as operation:
        operation.response(429, {})
    before = path.read_bytes()
    code = '''
import sys
import requests
from utils.sportsbet_access import SportsbetAccessBlocked
from utils.http_client import SourceCoordinatedSession
from utils.sportsbet_browser import create_sportsbet_driver

def forbidden(*args, **kwargs):
    raise AssertionError('transport or browser factory invoked despite shared STOP')
requests.adapters.HTTPAdapter.send = forbidden
try:
    if sys.argv[1] == 'python':
        SourceCoordinatedSession().get('https://www.sportsbet.com.au/fabricated')
    else:
        create_sportsbet_driver(forbidden)
except SportsbetAccessBlocked:
    pass
else:
    raise AssertionError('independent process ignored shared STOP')
'''
    subprocess.run([sys.executable, '-c', code, kind], check=True, timeout=10,
                   env={**os.environ, 'GREYHOUND_SPORTSBET_ACCESS_STATE': str(path)})
    assert path.read_bytes() == before


def test_supervised_transport_has_no_implicit_retry_for_any_provider(monkeypatch):
    import utils.http_client as client
    monkeypatch.setenv("GREYHOUND_LIVE_EXECUTION", "1")
    monkeypatch.setattr(client, "_shared_session", None)
    session = client.get_shared_session()
    try:
        for url in ("https://www.sportsbet.com.au/fixture", "https://www.thedogs.com.au/fixture", "https://api.open-meteo.com/fixture"):
            assert session.get_adapter(url).max_retries.total == 0
    finally:
        session.close()


@pytest.mark.parametrize("condition", ["OPEN", "STOP", "RECOVERY", "COOLDOWN", "cooldown_elapsed", "denial", "expired", "operation_cap", "unresolved"])
def test_service_condition_can_queue_behind_owned_operation_without_transport_admission(
    tmp_path, monkeypatch, condition,
):
    from scripts import check_sportsbet_access as service_condition
    from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked

    gate = SportsbetAccess(tmp_path / "access.json", clock=lambda: 1000)
    gate.initialize(access_basis={"status": "permitted", "reference": "synthetic admission fixture"})
    monkeypatch.setattr(service_condition, "SportsbetAccess", lambda: gate)
    with gate.operation("browser") as operation:
        if condition in {"STOP", "RECOVERY", "COOLDOWN"}:
            operation.value["phase"] = condition
            operation.value["not_before"] = 1100
        elif condition in {"expired", "operation_cap"}:
            operation.value["diagnostic_authority"] = {
                "expires_at": 999 if condition == "expired" else 1100,
                "operation_start": 0,
                "max_operations": 2 if condition == "expired" else 1,
            }
            operation.value["operations"] = [{"kind": "browser", "at": 1000}]
        elif condition == "unresolved":
            operation.value["access_basis"]["status"] = "unresolved"
        elif condition == "denial":
            operation.response(403, {})
        elif condition == "cooldown_elapsed":
            operation.value["phase"] = "COOLDOWN"
            operation.value["not_before"] = 900
        gate.write(operation.value)
        before = gate.path.read_bytes()

        assert service_condition.main() == (0 if condition == "OPEN" else 1)
        with pytest.raises(SportsbetAccessBlocked):
            gate.check_admission()
        with pytest.raises(SportsbetAccessBlocked):
            with gate.operation("python"):
                pytest.fail("service admission granted concurrent provider ownership")
        assert gate.path.read_bytes() == before
        assert gate.read()["active"] == operation.value["active"]
    if condition == "OPEN":
        gate.check_admission()
        with gate.operation("python"):
            pass
