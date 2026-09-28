"""No-install execution contract for the bounded live profile only."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys

_installer_guard_installed = False

REQUIRED_MODULES = (
    "requests",
    "bs4",
    "pandas",
    "selenium.webdriver",
    "playwright.sync_api",
    "flask",
)


def installed_browser_binaries():
    chrome = Path(os.environ.get("GREYHOUND_CHROME_BINARY", "/opt/google/chrome/chrome")).resolve(
        strict=True
    )
    driver = os.environ.get("GREYHOUND_CHROMEDRIVER") or shutil.which("chromedriver")
    if not driver:
        candidates = sorted(
            (Path.home() / ".cache/selenium/chromedriver/linux64").glob("*/chromedriver")
        )
        if len(candidates) != 1:
            raise ValueError("installed_chromedriver_must_be_explicit")
        driver = candidates[0]
    paths = {"chrome": chrome, "chromedriver": Path(driver).resolve(strict=True)}
    for path in paths.values():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("installed_browser_binary_unavailable")
    return {
        key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key, path in paths.items()
    }


def require_profile_dependencies():
    for name in REQUIRED_MODULES:
        importlib.import_module(name)


def configure_profile_execution(contract_path):
    """Propagates the same checked interpreter/browser binding to every child."""
    manifest = Path(contract_path).parent / "runtime-identity.json"
    value = json.loads(manifest.read_bytes())
    if Path(contract_path).exists():
        from race_collection.live_freshness_contract import digest

        contract = json.loads(Path(contract_path).read_bytes())
        if digest(value) != contract.get("runtime_sha256"):
            raise ValueError("execution_runtime_manifest_changed")
    if value["executable"] != sys.executable or value["prefix"] != sys.prefix:
        raise ValueError("capture_interpreter_binding_changed")
    require_profile_dependencies()
    for key, binary in value["browser_binaries"].items():
        path = Path(binary["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != binary["sha256"]:
            raise ValueError("installed_browser_binary_changed")
        os.environ["GREYHOUND_CHROME_BINARY" if key == "chrome" else "GREYHOUND_CHROMEDRIVER"] = (
            str(path)
        )
    os.environ["GREYHOUND_LIVE_EXECUTION"] = "bounded80-v1"
    os.environ["GREYHOUND_LIVE_CONTRACT"] = str(contract_path)
    os.environ["GREYHOUND_RUNTIME_MANIFEST"] = str(manifest)
    if Path(contract_path).exists() and contract.get("operational_predictions"):
        os.environ["GREYHOUND_SPORTSBET_RESPONSE_INSPECTION"] = "1"
    global _installer_guard_installed
    if not _installer_guard_installed:

        def installer_guard(event, args):
            if event != "subprocess.Popen":
                return
            executable, command = str(args[0]), args[1]
            tokens = command if isinstance(command, (list, tuple)) else [command]
            if Path(executable).name in {"uv", "pip", "pip3"} or (
                "-m" in tokens and any(x in {"pip", "ensurepip"} for x in tokens)
            ):
                from race_collection.live_freshness_contract import FreshnessContract, create_once

                if Path(contract_path).exists():
                    scope = FreshnessContract.load(contract_path)
                    scope.stop("DEPENDENCY_INSTALLATION_BLOCKED")
                    marker = scope.session / ("blocked-installer-" + str(os.getpid()) + ".json")
                    if not marker.exists():
                        create_once(marker, {"executable": Path(executable).name, "blocked": True})
                raise RuntimeError("profile_dependency_installation_forbidden")

        sys.addaudithook(installer_guard)
        _installer_guard_installed = True
    os.environ["UV_OFFLINE"] = "1"
    os.environ["PIP_NO_INDEX"] = "1"
    return value


class BrowserNetworkAccounting:
    """Separate browser event observations from guarded Python logical requests.

    CDP events are observations, not a claim to count every wire request. Chrome
    startup/update/background traffic and buffered events remain explicit gaps.
    """

    def __init__(self, driver, output):
        from race_collection.live_phase_checkpoint import atomic_json

        self.driver, self.output, self.write = driver, output, atomic_json
        self.value = {
            "browser_navigation_attempts": 0,
            "observed_provider_requests": 0,
            "observed_other_requests": 0,
            "observed_by_host": {},
            "coverage": "CDP_REQUEST_EVENTS_ONLY; startup/background/wire retries not guaranteed",
            "performance_log_errors": [],
        }
        self.seen = set()
        self.navigate = driver.get
        driver.get = self.get
        self.write(output, self.value)

    def get(self, url):
        from urllib.parse import urlsplit

        host = (urlsplit(url).hostname or "").lower()
        if host != "sportsbet.com.au" and not host.endswith(".sportsbet.com.au"):
            self.value["blocked_navigation"] = host
            self.write(self.output, self.value)
            raise ValueError("unexpected_browser_navigation")
        contract = os.environ.get("GREYHOUND_LIVE_CONTRACT")
        if contract:
            from datetime import datetime, timezone
            from race_collection.live_freshness_contract import FreshnessContract
            scope = FreshnessContract.load(contract)
            scope.admit(datetime.now(timezone.utc), seconds=0)
            if scope.campaign:
                try:
                    scope.campaign.request()
                except ValueError:
                    scope.stop("CAMPAIGN_REQUEST_CAP_EXHAUSTED")
                    raise
        self.value["browser_navigation_attempts"] += 1
        self.write(self.output, self.value)
        try:
            return self.navigate(url)
        finally:
            self.drain()

    def drain(self):
        from urllib.parse import urlsplit

        try:
            for row in self.driver.get_log("performance"):
                event = json.loads(row["message"])["message"]
                if event.get("method") == "Network.responseReceived":
                    response = event["params"]["response"]
                    host = (urlsplit(response.get("url", "")).hostname or "").lower()
                    if response.get("status") in {401, 403, 429} and any(
                        host == domain or host.endswith("." + domain)
                        for domain in ("sportsbet.com.au", "thedogs.com.au")
                    ):
                        from datetime import datetime, timezone
                        from utils.http_client import source_retry_headers

                        denial = {
                            "host": host,
                            "status": response["status"],
                            "observed_at": datetime.now(timezone.utc).isoformat(),
                            "retry_headers": source_retry_headers(response.get("headers")),
                        }
                        self.value.setdefault("source_access_denied", denial)
                        denials = self.value.setdefault("source_access_denials", [])
                        if len(denials) < 128:
                            denials.append(denial)
                        else:
                            self.value["source_access_denials_truncated"] = True
                if event.get("method") != "Network.requestWillBeSent":
                    continue
                params = event["params"]
                url = params["request"]["url"]
                key = (params["requestId"], url, params.get("timestamp"))
                if key in self.seen:
                    continue
                self.seen.add(key)
                host = (urlsplit(url).hostname or "").lower()
                if not host:
                    continue
                provider = any(
                    host == d or host.endswith("." + d)
                    for d in ("sportsbet.com.au", "thedogs.com.au")
                )
                category = "observed_provider_requests" if provider else "observed_other_requests"
                self.value[category] += 1
                self.value["observed_by_host"][host] = (
                    self.value["observed_by_host"].get(host, 0) + 1
                )
        except Exception as error:
            self.value["performance_log_errors"].append(type(error).__name__)
        self.write(self.output, self.value)
        if self.value.get("source_access_denied"):
            contract = os.environ.get("GREYHOUND_LIVE_CONTRACT")
            if contract:
                from race_collection.live_freshness_contract import FreshnessContract
                FreshnessContract.load(contract).stop("SOURCE_ACCESS_DENIED")
            raise ValueError("source_access_denied")
