"""Source adapters installed only by the offline subprocess fixture."""

import json
import os
from pathlib import Path
import time
from datetime import datetime, timedelta


def install():
    fixture = json.loads(Path(os.environ["FRESHNESS_FABRICATED_SOURCE"]).read_bytes())
    import sys

    def expire_capture_clock():
        # Clock is a controlled external input. Planner, reservation, subprocess,
        # fetch/append time gates and storage remain the actual packaged code.
        module = sys.modules.get("__main__")
        if not str(getattr(module, "__file__", "")).endswith("autonomous_live_odds_capture.py"):
            return
        boundary = datetime.fromisoformat(fixture["sidecar"]["prejump_shadow_metadata"]["jump_time"]) - timedelta(minutes=10)
        class ExpiredDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return boundary.astimezone(tz) if tz else boundary
        module.datetime = ExpiredDateTime

    import upcoming_race_browser
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException
    from bs4 import BeautifulSoup

    def upcoming(self, days_ahead=0):
        assert days_ahead == 0
        return [fixture["race"]]

    def download(self, url, **kwargs):
        assert url == fixture["race"]["url"]
        directory = Path(os.environ["UPCOMING_RACES_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        csv = directory / fixture["filename"]
        csv.write_text(fixture["csv"])
        csv.with_name(csv.name + ".metadata.json").write_text(json.dumps(fixture["sidecar"]))
        return {"success": True, "csv_path": str(csv)}

    upcoming_race_browser.UpcomingRaceBrowser.get_upcoming_races = upcoming
    upcoming_race_browser.UpcomingRaceBrowser.download_race_csv = download

    class Element:
        def __init__(self, node):
            self.node = node

        @property
        def text(self):
            return self.node.get_text("\n", strip=True)

        def get_attribute(self, name):
            return self.node.get(name, "")

        def find_elements(self, by, selector):
            if by == "xpath":
                return []
            return [Element(x) for x in self.node.select(selector)]

        def find_element(self, by, selector):
            rows = self.find_elements(by, selector)
            if not rows:
                raise NoSuchElementException(selector)
            return rows[0]

        def is_displayed(self):
            return True

        def is_enabled(self):
            return True

        def click(self):
            pass

    class Driver(Element):
        current_url = ""
        title = "Murray Bridge Straight Race 9"

        def __init__(self, *args, **kwargs):
            # The actual production driver factory must supply explicit binaries.
            assert kwargs["service"].path == os.environ["GREYHOUND_CHROMEDRIVER"]
            assert kwargs["options"].binary_location == os.environ["GREYHOUND_CHROME_BINARY"]
            self.logs = []
            super().__init__(BeautifulSoup("", "html.parser"))

        def get(self, url):
            assert url.startswith("https://www.sportsbet.com.au/")
            self.current_url = url
            html = fixture["race_html"] if "/race-" in url else fixture["landing_html"]
            self.node = BeautifulSoup(html, "html.parser")
            self.page_source = html
            self.logs.append(
                {
                    "message": json.dumps(
                        {
                            "message": {
                                "method": "Network.requestWillBeSent",
                                "params": {
                                    "requestId": str(len(self.logs)),
                                    "request": {"url": url},
                                },
                            }
                        }
                    )
                }
            )
            status = 429 if fixture["scenario"].startswith("source_denial") else 200
            self.logs.append({"message": json.dumps({"message": {
                "method": "Network.responseReceived",
                "params": {"type": "Document", "response": {
                    "url": url, "status": status,
                    "headers": {"Retry-After": "60"} if status == 429 else {},
                }},
            }})})
            if fixture["scenario"] == "expired_append" and "/race-" in url:
                expire_capture_clock()
            if fixture["scenario"] == "interrupted":
                Path(fixture["transport_marker"]).write_text(str(os.getpid()))
                time.sleep(3)
                raise RuntimeError("fabricated interrupted source response")

        def execute_script(self, script, *args):
            return "complete" if "readyState" in script else None

        def execute_cdp_cmd(self, command, params):
            assert command in {"Network.setBlockedURLs", "Page.stopLoading"}
            return {}

        def get_log(self, name):
            logs, self.logs = self.logs, []
            return logs

        def quit(self):
            Path(fixture["cleanup_marker"]).write_text(str(os.getpid()))

        def save_screenshot(self, path):
            return False

    # Service construction otherwise reserves a TCP port before transport creation.
    # This fabricated WebDriver never opens that port.
    import selenium.webdriver.chrome.service as service_module

    original_service = service_module.Service
    service_module.Service = lambda executable_path, **kwargs: original_service(
        executable_path, port=9999, **kwargs
    )
    webdriver.Chrome = Driver
    if fixture["scenario"] == "delayed_start":
        from race_collection import live_execution
        original_dependencies = live_execution.require_profile_dependencies
        def dependencies_with_delayed_clock():
            original_dependencies()
            expire_capture_clock()
        live_execution.require_profile_dependencies = dependencies_with_delayed_clock
