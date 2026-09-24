"""Independent CDP denial observation while WebDriver navigation is pending."""
import os
import json
import queue
from pathlib import Path
import signal
import threading
import time

from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked, is_sportsbet


def cdp(method, params=None):
    result = yield {"method": method, "params": params or {}}
    return result


def process_identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[1]), fields[19]


def owned_processes(pid, birth):
    """Snapshot descendants while the service parent still proves ownership."""
    try:
        if process_identity(pid)[1] != birth:
            return {}
    except FileNotFoundError:
        return {}
    identities = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            try:
                identities[int(entry.name)] = process_identity(int(entry.name))
            except (OSError, ValueError):
                pass
    owned = [pid]
    for parent in owned:
        owned.extend(child for child, (ppid, _) in identities.items()
                     if ppid == parent and child not in owned)
    return {child: identities[child][1] for child in owned if child in identities}


def stop_owned_processes(pid, birth, retained=None):
    """Retained birth identities survive reparenting without a host-wide kill."""
    identities = dict(retained or {})
    identities.update(owned_processes(pid, birth))
    def alive(child, expected):
        try:
            fields = (Path('/proc') / str(child) / 'stat').read_text().rsplit(')', 1)[1].split()
            return fields[19] == expected and fields[0] != 'Z'
        except FileNotFoundError:
            return False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for child, expected in reversed(list(identities.items())):
            try:
                if alive(child, expected):
                    os.kill(child, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + .5
        while time.monotonic() < deadline:
            if not any(alive(child, expected) for child, expected in identities.items()):
                return
            time.sleep(.01)
    if any(alive(child, expected) for child, expected in identities.items()):
        raise RuntimeError('owned_browser_cleanup_incomplete')


def create_sportsbet_driver(factory, *, response_inspection=None, **kwargs):
    admission = SportsbetAccess().operation('browser')
    operation = admission.__enter__()
    if response_inspection is not None:
        response_inspection.operation_id = operation.value["active"]
    driver = connection = None
    expected_disconnect = threading.Event()
    observation_failed = threading.Event()
    state_lock = threading.RLock()
    channel_lock = threading.Lock()
    events = queue.Queue()
    navigations = 0
    closed = False
    descendants = {}

    def channel(method, params=None):
        with channel_lock:
            return connection.execute(cdp(method, params))

    def lost_observation(*args):
        if expected_disconnect.is_set() or observation_failed.is_set():
            return
        observation_failed.set()
        try:
            with state_lock:
                operation.failed = True
                operation.value['phase'] = 'STOP'
                operation.gate.write(operation.value)
        finally:
            stop_owned_processes(owner_pid, owner_birth, descendants)

    def monitor():
        while True:
            params = events.get()
            try:
                if params is None:
                    return
                source = params['response']
                if not is_sportsbet(source.get('url', '')):
                    continue
                status = source['status']
                with state_lock:
                    if status >= 400 or params.get('type') == 'Document':
                        operation.response(status, source.get('headers', {}),
                                           source_url=source.get('url'), resource_type=params.get('type'))
                    held = operation.value['phase'] in {'COOLDOWN', 'STOP'}
                if held and not expected_disconnect.is_set():
                    channel('Network.setBlockedURLs', {'urls': ['*://*.sportsbet.com.au/*', '*://sportsbet.com.au/*']})
                    channel('Page.stopLoading')
            except Exception:
                lost_observation()
            finally:
                events.task_done()

    try:
        driver = factory(**kwargs)
        if response_inspection is not None:
            capabilities = getattr(driver, 'capabilities', {})
            response_inspection.browser_identity = {
                'browser_version': capabilities.get('browserVersion'),
                'driver_version': capabilities.get('chrome', {}).get('chromedriverVersion', '').split(' ')[0],
            }
        owner_pid = driver.service.process.pid
        if owner_pid == os.getpid():
            raise RuntimeError('invalid_browser_process_owner')
        owner_birth = process_identity(owner_pid)[1]
        descendants.update(owned_processes(owner_pid, owner_birth))
        _, connection = driver.start_devtools()
        # Selenium owns this independent WebSocket. Loss cannot fall back to an
        # unobserved browser; callbacks do not share WebDriver's command queue.
        connection._ws.on_close = lost_observation
        connection._ws.on_error = lost_observation
        original_message = connection._ws.on_message
        def received(ws, message):
            original_message(ws, message)
            try:
                event = json.loads(message)
                if event.get('method') == 'Network.responseReceived':
                    events.put(event['params'])
                if response_inspection is not None:
                    try:
                        response_inspection.observe(event)
                    except Exception:
                        # Optional research instrumentation must not replace or
                        # delay the independent source-denial observer.
                        response_inspection.fail()
            except Exception:
                lost_observation()
        connection._ws.on_message = received
        # The pinned Selenium transport exposes no public disconnect callback or
        # response timeout setter. Keep these adapter details isolated here.
        connection._response_wait_timeout = 2
        watcher = threading.Thread(target=monitor, name='sportsbet-source-observer', daemon=True)
        watcher.start()
        channel('Network.enable')
        if response_inspection is not None:
            response_inspection.mark('browser_ready')
    except BaseException:
        import sys
        failure = sys.exc_info()
        expected_disconnect.set()
        try:
            if driver is not None:
                driver.quit()
        finally:
            if connection is not None:
                connection._ws.close()
                connection.close()
            if 'watcher' in locals():
                events.join()
                events.put(None)
                watcher.join()
            admission.__exit__(*failure)
        raise

    navigate, get_log, quit_driver = driver.get, driver.get_log, driver.quit

    def get(url):
        nonlocal navigations
        events.join()
        descendants.update(owned_processes(owner_pid, owner_birth))
        with state_lock:
            operation.check()
            if not is_sportsbet(url):
                operation.failed = True
                raise SportsbetAccessBlocked('sportsbet_browser_route_changed')
            navigation_cap = operation.value.get('operating_policy', {}).get(
                'browser_navigation_cap', 2 if operation.recovery else None)
            if navigation_cap is not None and navigations >= navigation_cap:
                operation.failed = True
                raise SportsbetAccessBlocked('sportsbet_browser_navigation_cap')
            navigations += 1
        if response_inspection is not None:
            response_inspection.navigate()
        try:
            result = navigate(url)
        except BaseException:
            with state_lock:
                # A known denial remains a cooldown, not an ambiguous crash.
                if operation.value['phase'] not in {'COOLDOWN', 'STOP'}:
                    operation.failed = True
            raise
        events.join()
        descendants.update(owned_processes(owner_pid, owner_birth))
        with state_lock:
            operation.check()
        if response_inspection is not None:
            response_inspection.mark('navigation_complete')
        return result

    def logs(name):
        try:
            return get_log(name)
        except Exception:
            lost_observation()
            raise

    def quit():
        nonlocal closed
        if closed:
            return
        closed = True
        expected_disconnect.set()
        try:
            quit_driver()
        except BaseException:
            operation.failed = True
            stop_owned_processes(owner_pid, owner_birth, descendants)
            raise
        finally:
            try:
                connection._ws.close()
                connection.close()
                events.join()
                events.put(None)
                watcher.join()
            except BaseException:
                operation.failed = True
                stop_owned_processes(owner_pid, owner_birth, descendants)
                raise
            finally:
                with state_lock:
                    admission.__exit__(None, None, None)

    def navigation_remaining():
        """Planning information only; get() remains the enforcing boundary."""
        with state_lock:
            operation.check()
            cap = operation.value.get('operating_policy', {}).get(
                'browser_navigation_cap', 2 if operation.recovery else None)
            return None if cap is None else max(0, cap - navigations)

    driver.sportsbet_navigation_remaining = navigation_remaining

    def accept_data():
        events.join()
        with state_lock:
            operation.accept_data()

    def inspect_response_shapes():
        if closed:
            raise RuntimeError('sportsbet_browser_closed')
        if response_inspection is None:
            raise RuntimeError('sportsbet_response_inspection_disabled')
        response_inspection.mark('rendered_extraction_complete')
        def body(request_id):
            events.join()
            with state_lock:
                operation.check()
            return channel('Network.getResponseBody', {'requestId': request_id})
        events.join()
        with state_lock:
            operation.check()
        response_inspection.inspect_bodies(body)
        return response_inspection.report()

    driver.sportsbet_accept_validated_data = accept_data
    driver.sportsbet_inspect_response_shapes = inspect_response_shapes
    def snapshot_dom_counts():
        # Reading the already delivered DOM does not admit source traffic. Use
        # the independent two-second CDP transport and a one-second renderer cap.
        value = channel('Runtime.evaluate', {
            'expression': """({runner_elements: document.querySelectorAll(
                '[data-automation-id*=racecard-outcome-name]').length,
                price_elements: document.querySelectorAll(
                '[data-automation-id*=price-text]').length})""",
            'returnByValue': True, 'timeout': 1000,
        })
        return value.get('result', {}).get('value')
    driver.sportsbet_snapshot_dom_counts = snapshot_dom_counts
    driver.get, driver.get_log, driver.quit = get, logs, quit
    return driver
