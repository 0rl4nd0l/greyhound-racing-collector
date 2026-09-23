"""Keep source ownership until browser cleanup, including its asynchronous work."""

import json

from utils.sportsbet_access import SportsbetAccess, SportsbetAccessBlocked, is_sportsbet


def create_sportsbet_driver(factory, **kwargs):
    admission = SportsbetAccess().operation("browser")
    operation = admission.__enter__()
    try:
        driver = factory(**kwargs)
    except BaseException:
        import sys
        admission.__exit__(*sys.exc_info())
        raise
    navigate, get_log, quit_driver = driver.get, driver.get_log, driver.quit
    buffered = []
    navigations = 0
    closed = False

    def drain():
        try:
            rows = get_log("performance")
            buffered.extend(rows)
            for row in rows:
                event = json.loads(row["message"])["message"]
                if event.get("method") == "Network.responseReceived":
                    response = event["params"]["response"]
                    if is_sportsbet(response.get("url", "")):
                        if response["status"] in {401, 403, 429} or event["params"].get("type") == "Document":
                            operation.response(response["status"], response.get("headers", {}))
            if operation.value["phase"] in {"COOLDOWN", "STOP"}:
                driver.execute_cdp_cmd("Network.setBlockedURLs", {"urls": ["*://*.sportsbet.com.au/*", "*://sportsbet.com.au/*"]})
                driver.execute_cdp_cmd("Page.stopLoading", {})
        except Exception:
            operation.failed = True
            raise

    def get(url):
        nonlocal navigations
        operation.check()
        if not is_sportsbet(url):
            operation.failed = True
            raise SportsbetAccessBlocked("sportsbet_browser_route_changed")
        if operation.recovery and navigations >= 2:
            operation.failed = True
            raise SportsbetAccessBlocked("sportsbet_recovery_navigation_cap")
        navigations += 1
        try:
            return navigate(url)
        except BaseException:
            operation.failed = True
            raise
        finally:
            drain()
            operation.check()

    def logs(name):
        if name != "performance":
            return get_log(name)
        drain()
        result = list(buffered)
        buffered.clear()
        return result

    def quit():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            drain()
        finally:
            try:
                quit_driver()
            except BaseException:
                operation.failed = True
                raise
            finally:
                admission.__exit__(None, None, None)

    driver.get, driver.get_log, driver.quit = get, logs, quit
    return driver
