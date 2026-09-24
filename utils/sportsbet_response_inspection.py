"""Bounded, value-free inspection of an already admitted browser operation.

This is schema discovery, never an odds source. No HTTP, replay, raw-body storage,
provider timestamp inference, or access-state ownership lives here.
"""

from datetime import datetime, timezone
import json
import math
import threading
import time
from urllib.parse import parse_qsl, urlsplit

# Unknown route tokens and query values are deliberately not retained. Expand
# only after review; do not mistake a redacted route for an executable endpoint.
_ROUTE_TOKENS = frozenset(
    "apigw sportsbook-racing Sportsbook Racing NextEvents Events Event Markets Market Prices Price Races Race RacingEvents racing events event markets market prices price races race v1 v2".split()
)
_QUERY_IDS = frozenset(("eventId", "raceId", "marketId", "competitionId"))
_FIELDS = frozenset(
    "id eventId raceId marketId selectionId runnerId name eventName marketName runnerName competitionId competitionName raceNumber startTime status state active suspended closed scratched isScratched isSuspended isActive markets selections runners events prices price odds decimal win place fixed fixedWin fixedPlace numerator denominator places numberOfPlaces placeTerms timestamp sequence version data payload snapshot updates type".split()
)


def safe_route(url):
    """Only gateway metadata; never persist credentials or arbitrary text."""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "wss"} or parsed.hostname != "www.sportsbet.com.au":
            return None
        if not parsed.path.startswith("/apigw/"):
            return None
        segments = parsed.path.split("/")
        redacted = False
        safe = []
        for part in segments:
            if part == "" or part in _ROUTE_TOKENS:
                safe.append(part)
            else:
                safe.append("{redacted}")
                redacted = True
        query = {}
        omitted = 0
        for key, value in parse_qsl(parsed.query, max_num_fields=64):
            if key in _QUERY_IDS and value.isascii() and value.isdigit() and len(value) <= 18:
                query[key] = value
            elif key == "groupByFilters" and value in {"true", "false"}:
                query[key] = value
            elif key == "racingFilters" and all(
                v
                in {
                    "GH_DOMESTIC",
                    "GH_INTERNATIONAL",
                    "HR_DOMESTIC",
                    "HR_INTERNATIONAL",
                    "HA_DOMESTIC",
                    "HA_INTERNATIONAL",
                }
                for v in value.split(",")
            ):
                query[key] = value
            else:
                omitted += 1
        return {
            "path": "/".join(safe),
            "path_redacted": redacted,
            "query": query,
            "omitted_query_fields": omitted,
        }
    except (ValueError, TypeError):
        return None


def response_shape(raw, *, max_bytes=1_000_000):
    """Expose known field names and types only, including under unknown parents.

    Even numeric scalars are suppressed: they may be finishing placements.
    Unknown property names can themselves contain protected values. Traversal is
    finite, and truncation is explicit; this cannot establish runner coverage.
    """
    if not isinstance(raw, str) or len(raw) > max_bytes or len(raw.encode("utf-8")) > max_bytes:
        return {"state": "body_limit"}
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError):
        return {"state": "not_json"}
    remaining = 2048
    truncated = False

    def visit(item, depth):
        nonlocal remaining, truncated
        remaining -= 1
        if remaining < 0 or depth > 12:
            truncated = True
            return {"type": "truncated"}
        if isinstance(item, dict):
            fields = {}
            unknown = []
            for key, child in item.items():
                if remaining <= 0:
                    truncated = True
                    break
                shape = visit(child, depth + 1)
                if key in _FIELDS:
                    fields[key] = shape
                else:
                    unknown.append(shape)
            return {"type": "object", "fields": fields, "unnamed_fields": unknown}
        if isinstance(item, list):
            sample = [visit(child, depth + 1) for child in item[: min(16, max(0, remaining))]]
            if len(sample) < len(item):
                truncated = True
            return {"type": "array", "length": len(item), "sample": sample}
        return {
            "type": (
                "null"
                if item is None
                else (
                    "boolean"
                    if isinstance(item, bool)
                    else "number" if isinstance(item, (int, float)) else "string"
                )
            )
        }

    return {"state": "shape_only", "shape": visit(value, 0), "truncated": truncated}


class ResponseInspection:
    """One bounded observer, explicitly supplied to create_sportsbet_driver.

    Caller owns canonical race/window binding and writes report() to its existing
    evidence directory. This object grants no reservation or execution authority.
    """

    def __init__(self, *, expires_at, clock=None, monotonic=time.monotonic):
        if expires_at.tzinfo is None:
            raise ValueError("inspection_requires_aware_expiry")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.monotonic = monotonic
        self.expires_at = expires_at
        self.started = monotonic()
        self.lock = threading.RLock()
        self.rows = {}
        self.marks = []
        self.navigation = 0
        self.stopped = None
        self.dropped = 0
        self.websocket_frames = 0
        self.body_reads = 0
        self.operation_id = None
        self.network_responses = []
        self.mark("browser_start")

    def _stamp(self):
        return {
            "observed_at": self.clock().isoformat(),
            "elapsed_seconds": max(0, self.monotonic() - self.started),
        }

    def _open(self):
        if self.stopped:
            return False
        if self.clock() >= self.expires_at or self.monotonic() - self.started >= 50:
            self.stopped = "expired"
            return False
        return True

    def mark(self, name):
        if name not in {
            "browser_start",
            "browser_ready",
            "navigation_start",
            "navigation_complete",
            "rendered_extraction_complete",
            "inspection_complete",
        }:
            raise ValueError("unknown_inspection_mark")
        with self.lock:
            if len(self.marks) < 16:
                self.marks.append({"name": name, "navigation": self.navigation, **self._stamp()})

    def navigate(self):
        with self.lock:
            self.navigation += 1
            self.mark("navigation_start")

    def fail(self):
        with self.lock:
            self.stopped = "inspection_error"

    def observe(self, event):
        with self.lock:
            if not self._open():
                return
            method, params = event.get("method"), event.get("params", {})
            if method == "Network.responseReceived" and len(self.network_responses) < 256:
                from utils.sportsbet_access import is_sportsbet
                from utils.http_client import source_retry_headers
                response = params.get("response", {})
                url = response.get("url", "")
                if is_sportsbet(url):
                    import hashlib
                    parsed = urlsplit(url)
                    resource = params.get("type", "unknown")
                    route = safe_route(url)
                    category = ("document" if resource == "Document" else
                                "structured_data" if resource in {"XHR", "Fetch"} else
                                "static_asset" if resource in {"Image", "Font", "Stylesheet"} else
                                "script" if resource == "Script" else "unknown")
                    self.network_responses.append({
                        "host": parsed.hostname, "route": route,
                        "path_sha256": hashlib.sha256(parsed.path.encode()).hexdigest(),
                        "resource_type": resource, "category": category,
                        "status": response.get("status"),
                        "retry_headers": source_retry_headers(response.get("headers", {})),
                        **self._stamp(),
                    })
            if method == "Network.webSocketFrameReceived":
                self.websocket_frames += 1
                return  # Never inspect stream payloads or infer ordering.
            request_id = params.get("requestId")
            if not isinstance(request_id, str):
                return
            if method == "Network.requestWillBeSent":
                request = params.get("request", {})
                if request_id in self.rows:
                    self.rows[request_id]["state"] = "redirect_or_reused_id"
                    return
                route = safe_route(request.get("url", ""))
                if route is None:
                    return
                if len(self.rows) >= 64:
                    self.dropped += 1
                    return
                verb = request.get("method")
                self.rows[request_id] = {
                    "route": route,
                    "method": verb if verb in {"GET", "POST", "OPTIONS"} else "other",
                    "navigation": self.navigation,
                    "request": self._stamp(),
                    "state": "requested",
                }
            elif request_id in self.rows:
                row = self.rows[request_id]
                if row["navigation"] != self.navigation:
                    row["state"] = "navigation_changed"
                    return
                if row["state"] in {
                    "redirect_or_reused_id",
                    "navigation_changed",
                    "route_changed",
                    "loading_failed",
                    "invalid_sequence",
                }:
                    return
                if method == "Network.responseReceived":
                    if row["state"] != "requested":
                        row["state"] = "invalid_sequence"
                        return
                    response = params.get("response", {})
                    route = safe_route(response.get("url", ""))
                    if route != row["route"]:
                        row["state"] = "route_changed"
                        return
                    status = response.get("status")
                    row.update(
                        response=self._stamp(),
                        status=status if isinstance(status, int) else None,
                        json_mime=response.get("mimeType") == "application/json",
                        from_cache=bool(
                            response.get("fromDiskCache") or response.get("fromPrefetchCache")
                        ),
                        from_service_worker=bool(response.get("fromServiceWorker")),
                        state="response",
                    )
                elif method == "Network.loadingFinished" and row["state"] == "response":
                    size = params.get("encodedDataLength")
                    row.update(
                        completed=self._stamp(),
                        state="complete",
                        encoded_bytes=(
                            size
                            if isinstance(size, (int, float)) and math.isfinite(size) and size >= 0
                            else None
                        ),
                    )
                elif method == "Network.loadingFailed":
                    row["state"] = "loading_failed"

    def inspect_bodies(self, read_body):
        """Read at most four already delivered CDP bodies, not provider requests.

        Call only through driver's guarded inspection method after ordinary DOM
        extraction. Unknown JSON is projected to a value-free schema in memory.
        """
        with self.lock:
            request_ids = list(self.rows)
        for request_id in request_ids:
            with self.lock:
                if not self._open() or self.body_reads >= 4:
                    break
                row = self.rows[request_id]
                if (
                    row["navigation"] != self.navigation
                    or row["state"] != "complete"
                    or "body" in row
                ):
                    continue
                if (
                    row.get("status") != 200
                    or not row.get("json_mime")
                    or row.get("from_cache")
                    or row.get("from_service_worker")
                ):
                    row["body"] = {"state": "ineligible_response"}
                    continue
                if row.get("encoded_bytes") is None or row["encoded_bytes"] > 1_000_000:
                    row["body"] = {"state": "body_limit"}
                    continue
                self.body_reads += 1
            try:
                body = read_body(request_id)
                projected = (
                    {"state": "base64_not_inspected"}
                    if body.get("base64Encoded")
                    else response_shape(body.get("body"))
                )
            except Exception:
                projected = {"state": "body_unavailable"}  # Never persist exception text.
            with self.lock:
                if row["navigation"] != self.navigation or row["state"] != "complete":
                    projected = {"state": "changed_during_inspection"}
                row["body"] = projected if self._open() else {"state": "expired_during_inspection"}
                row["body_inspected"] = self._stamp()
        self.mark("inspection_complete")

    def report(self):
        with self.lock:
            self._open()
            return json.loads(
                json.dumps(
                    {
                        "schema": "sportsbet-response-inspection-v1",
                        "purpose": "schema_only_not_capture",
                        "provider_timestamp": None,
                        "stop_reason": self.stopped,
                        "dropped_requests": self.dropped,
                        "websocket_frame_count": self.websocket_frames,
                        "body_reads": self.body_reads,
                        "operation_id": self.operation_id,
                        "network_responses": self.network_responses,
                        "marks": self.marks,
                        "responses": list(self.rows.values()),
                    }
                )
            )
