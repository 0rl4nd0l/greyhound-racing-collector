#!/usr/bin/env python3
"""One bounded public page observation; no bookmaker access or raw payload output."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit

from utils.sportsbet_response_inspection import response_shape

SAFE_FIELDS = frozenset(
    "props pageProps race races meeting meetings event events runners runner bookmakers bookmaker odds fixedOdds fixedWin fixedPlace winOdds placeOdds prices markets market selections selection data attributes included relationships startTime startDate startDateTime advertisedStartTime jumpTime raceTime raceNumber raceStatus status isResulted hasResults scratched isScratched isSuspended suspended closed isClosed number box boxNumber runnerNumber runnerId eventId raceId meetingId bookmakerId name slug id url canonicalUrl timestamp updatedAt lastUpdated sequence version win place price decimal places placeTerms results".split()
)


def now():
    return datetime.now(timezone.utc).isoformat()


def public_host(url):
    host = (urlsplit(url).hostname or "").lower()
    return host == "odds.com.au" or host.endswith(".odds.com.au")


def route_metadata(url):
    parsed = urlsplit(url)
    # A clean public browser has no account identity. Still redact opaque path
    # tokens and ALL query values; never retain request/response headers.
    path = "/".join(
        part if re.fullmatch(r"[a-zA-Z0-9_.-]{0,100}", part) else "{redacted}"
        for part in parsed.path.split("/")
    )
    return {"host": parsed.hostname, "path": path, "query_present": bool(parsed.query)}


class InlineJSON(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.buffer = []
        self.items = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "script" and (
            attrs.get("id") == "__NEXT_DATA__"
            or attrs.get("type") in {"application/json", "application/ld+json"}
        ):
            self.active = True
            self.buffer = []

    def handle_data(self, data):
        if self.active:
            self.buffer.append(data)

    def handle_endtag(self, tag):
        if tag == "script" and self.active:
            self.items.append("".join(self.buffer))
            self.active = False


def metadata(raw):
    """Known schedule/state keys only; never runners, prices, form or results."""
    try:
        tree = json.loads(raw)
    except (ValueError, RecursionError):
        return []
    out = []
    budget = 10000
    time_keys = {
        "startTime",
        "startDate",
        "startDateTime",
        "advertisedStartTime",
        "jumpTime",
        "raceTime",
    }

    def visit(value, path, depth):
        nonlocal budget
        budget -= 1
        if budget < 0 or depth > 15 or len(out) >= 30:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"results", "result", "form", "history", "pastperformances"}:
                    continue
                safe = key if key in SAFE_FIELDS else "{field}"
                if (
                    key in time_keys
                    and isinstance(item, str)
                    and re.fullmatch(r"[0-9T:Z+ .\-/]{5,40}", item)
                ):
                    out.append({"path": path + [safe], "time": item})
                elif key in time_keys and isinstance(item, (int, float)):
                    out.append({"path": path + [safe], "time_numeric": item})
                elif key in {"isResulted", "hasResults"} and isinstance(item, bool):
                    out.append({"path": path + [safe], "flag": item})
                elif (
                    key in {"raceStatus", "status"}
                    and isinstance(item, str)
                    and item.lower()
                    in {
                        "open",
                        "closed",
                        "suspended",
                        "resulted",
                        "final",
                        "interim",
                        "upcoming",
                        "live",
                        "finished",
                    }
                ):
                    out.append({"path": path + [safe], "state": item.lower()})
                elif isinstance(item, (dict, list)):
                    visit(item, path + [safe], depth + 1)
        elif isinstance(value, list):
            for item in value[:30]:
                visit(item, path + ["[]"], depth + 1)

    visit(tree, [], 0)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not public_host(args.url) or urlsplit(args.url).scheme != "https":
        raise ValueError("odds_public_url_required")
    output = Path(args.output)
    if output.exists():
        raise ValueError("observation_output_exists_no_retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "odds-public-inspection-v1",
        "started_at": now(),
        "url": route_metadata(args.url),
        "request_cap": 60,
        "duration_cap_seconds": 30,
        "navigation_cap": 1,
        "requests_allowed": [],
        "requests_blocked": [],
        "responses": [],
        "stop": None,
        "provider_operations": {
            "odds_page_navigation": 0,
            "sportsbet": 0,
            "bookmaker_redirects": 0,
        },
    }

    def persist():
        output.write_text(json.dumps(record, indent=2) + "\n")

    persist()  # Durable operation intent before browser transport.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--disable-background-networking"],
        )
        context = browser.new_context(service_workers="block")

        # HTTP routing does not cover websocket handshakes. Until a reviewed
        # public stream contract exists, block all streams before any page loads.
        def block_socket(websocket):
            record["requests_blocked"].append(
                {
                    "resource_type": "websocket",
                    "observed_at": now(),
                    **route_metadata(websocket.url),
                }
            )
            persist()
            websocket.close()

        context.route_web_socket("**/*", block_socket)
        page = context.new_page()
        started = time.monotonic()
        ids = {}

        def route(request_route):
            request = request_route.request
            item = {
                "index": len(record["requests_allowed"]) + 1,
                "method": request.method,
                "resource_type": request.resource_type,
                "observed_at": now(),
                **route_metadata(request.url),
            }
            allowed = (
                public_host(request.url)
                and not record["stop"]
                and time.monotonic() - started < 30
                and len(record["requests_allowed"]) < 60
            )
            # No second document, redirects, click-throughs or outcome routes.
            if request.redirected_from or any(
                x in urlsplit(request.url).path.lower().split("/") for x in ("results", "result")
            ):
                allowed = False
            if request.is_navigation_request() and request.url != args.url:
                allowed = False
            if allowed:
                ids[id(request)] = item["index"]
                record["requests_allowed"].append(item)
                persist()
                request_route.continue_()
            else:
                record["requests_blocked"].append(item)
                if public_host(request.url) and len(record["requests_allowed"]) >= 60:
                    record["stop"] = "request_cap"
                persist()
                request_route.abort()

        context.route("**/*", route)

        def response(resp):
            if not public_host(resp.url):
                return
            row = {
                "request_index": ids.get(id(resp.request)),
                "status": resp.status,
                "observed_at": now(),
                "elapsed_seconds": time.monotonic() - started,
                **route_metadata(resp.url),
            }
            record["responses"].append(row)
            if resp.status in {401, 403, 429}:
                record["stop"] = "denial"
                row["retry_after_present"] = "retry-after" in resp.headers
                persist()
                return
            if record["stop"]:
                persist()
                return
            content_type = resp.headers.get("content-type", "")
            if resp.status == 200 and (
                "json" in content_type or resp.request.resource_type == "document"
            ):
                try:
                    raw = resp.text()
                    row["body_bytes"] = len(raw.encode())
                    if len(raw) <= 2_000_000:
                        if resp.request.resource_type == "document":
                            lower = raw.lower()
                            if any(
                                marker in lower
                                for marker in (
                                    "cf-chl-",
                                    "verify you are human",
                                    "access denied",
                                    "captcha",
                                )
                            ):
                                record["stop"] = "challenge_indicator"
                            parsed = InlineJSON()
                            parsed.feed(raw)
                            row["inline_json"] = [
                                {"shape": response_shape(item), "schedule_state": metadata(item)}
                                for item in parsed.items[:4]
                            ]
                            # Only future-selection links from normal navigation; no text/results.
                            row["race_links"] = list(
                                dict.fromkeys(
                                    re.findall(
                                        r'href=["\'](/greyhounds/[a-z0-9-]+/[^"\'?#]*race-\d+/)["\']',
                                        raw,
                                    )
                                )
                            )[:20]
                        else:
                            row["shape"] = response_shape(raw)
                            row["schedule_state"] = metadata(raw)
                except Exception:
                    row["body_state"] = "unavailable"
            persist()

        page.on("response", response)
        try:
            record["provider_operations"]["odds_page_navigation"] = 1
            persist()
            page.goto(args.url, wait_until="domcontentloaded", timeout=20000)
            while time.monotonic() - started < 15 and not record["stop"]:
                page.wait_for_timeout(100)
        except Exception as error:
            record["navigation_error_type"] = type(error).__name__
        finally:
            context.close()
            browser.close()
    record["finished_at"] = now()
    persist()
    print(
        json.dumps(
            {
                "output": str(output),
                "allowed": len(record["requests_allowed"]),
                "blocked": len(record["requests_blocked"]),
                "statuses": [r["status"] for r in record["responses"]],
                "stop": record["stop"],
            }
        )
    )


if __name__ == "__main__":
    main()
