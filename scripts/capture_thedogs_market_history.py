#!/usr/bin/env python3
"""Capture one immutable prospective TheDogs market snapshot.

The collector accepts one exact race plan, warms the current TheDogs meeting
and race pages, fetches the exact ``/odds`` page once, then fetches the native
runner-odds JSON used by that page once.  It writes only an immutable raw HTML
file and its receipt.  It never retries, discovers a substitute race, writes a
database, or performs training/scoring.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import requests
from requests.adapters import HTTPAdapter

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCHEMA_VERSION = "thedogs_market_snapshot_receipt_v1"
SCHEMA_VERSION = "thedogs_market_snapshot_receipt_v2"
RECEIPT_SCHEMA_VERSIONS = frozenset({LEGACY_SCHEMA_VERSION, SCHEMA_VERSION})
PLAN_SCHEMA_VERSION = "thedogs_market_snapshot_plan_v1"
SOURCE_CLASS = "thedogs_prospective_point_in_time_market_history"
NOMINAL_WINDOWS_MINUTES = (120, 60, 30, 10, 2)
NOMINAL_WINDOWS = tuple(f"T-{minutes}" for minutes in NOMINAL_WINDOWS_MINUTES)
DEFAULT_EARLY_TOLERANCE_SECONDS = 30
DEFAULT_LATE_TOLERANCE_SECONDS = 90
SERVER_DATE_TOLERANCE_SECONDS = 5
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)
PAGE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Encoding": "identity",
    "Accept-Language": "en-US,en;q=0.9",
}
RECEIPT_HTTP_HEADERS = (
    "age",
    "cache-control",
    "content-encoding",
    "content-length",
    "content-type",
    "date",
    "etag",
    "last-modified",
    "server",
    "via",
    "x-request-id",
    "x-runtime",
    "x-varnish-cache",
)
EXACT_ODDS_PATH = re.compile(
    r"^/racing/([a-z0-9-]+)/([0-9]{4}-[0-9]{2}-[0-9]{2})/"
    r"([1-9][0-9]*)/([a-z0-9-]+)/odds$"
)


class CaptureError(ValueError):
    """Raised when capture evidence cannot satisfy the prospective contract."""


@dataclass(frozen=True)
class SourceRunner:
    native_runner_id: str
    runner_name: str
    box: int
    page_effective_box: int | None
    active: bool


@dataclass(frozen=True)
class TimedResponse:
    requested_url: str
    request_start_utc: datetime
    request_end_utc: datetime
    final_url: str
    status_code: int
    headers: dict[str, str]
    body: bytes


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: Any, *, field: str) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise CaptureError(f"{field}_required")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise CaptureError(f"{field}_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CaptureError(f"{field}_timezone_required")
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def serialized_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def exact_odds_identity(value: Any) -> dict[str, str | int]:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "www.thedogs.com.au"
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise CaptureError("exact_thedogs_odds_url_required")
    match = EXACT_ODDS_PATH.fullmatch(parsed.path)
    if match is None:
        raise CaptureError("exact_thedogs_odds_url_required")
    venue_slug, race_date, race_number, race_slug = match.groups()
    return {
        "odds_url": url,
        "race_url": url[: -len("/odds")],
        "meeting_url": f"https://www.thedogs.com.au/racing/{race_date}",
        "venue_slug": venue_slug,
        "race_date": race_date,
        "race_number": int(race_number),
        "race_slug": race_slug,
    }


def nominal_window_minutes(value: Any) -> int:
    text = str(value or "").strip().upper()
    if text not in NOMINAL_WINDOWS:
        raise CaptureError("unsupported_nominal_window")
    return int(text.removeprefix("T-"))


def planned_time(jump_utc: datetime, nominal_window: str) -> datetime:
    return jump_utc - timedelta(minutes=nominal_window_minutes(nominal_window))


def validate_due_window(
    *,
    now: datetime,
    jump_utc: datetime,
    nominal_window: str,
    early_tolerance_seconds: int,
    late_tolerance_seconds: int,
) -> datetime:
    nominal = planned_time(jump_utc, nominal_window)
    if now < nominal - timedelta(seconds=early_tolerance_seconds):
        raise CaptureError("nominal_window_not_due")
    if now > nominal + timedelta(seconds=late_tolerance_seconds):
        raise CaptureError("nominal_window_missed")
    return nominal


class _RunnerParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.runners: list[SourceRunner] = []
        self._tbody_runner_id: str | None = None
        self._row: dict[str, Any] | None = None
        self._collect_name = False
        self._collect_effective_box = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        lowered = tag.lower()
        classes = set(str(attributes.get("class") or "").split())
        if lowered == "tbody":
            match = re.fullmatch(
                r"/dogs/runner/([0-9]+)/odds",
                str(attributes.get("data-content-url") or ""),
            )
            self._tbody_runner_id = match.group(1) if match else None
        elif lowered == "tr" and "race-runner" in classes:
            self._row = {
                "native_runner_id": self._tbody_runner_id,
                "runner_name": "",
                "box": None,
                "effective_box_texts": [],
                "active": "race-runner--scratched" not in classes,
                "element_runner_ids": set(),
            }
        elif self._row is not None and lowered == "sprite-svg":
            match = re.fullmatch(r"rug_([1-9][0-9]*)", str(attributes.get("name") or ""))
            if match:
                self._row["box"] = int(match.group(1))
        elif self._row is not None and lowered in {
            "runner-odd",
            "runner-odd-fluctuation-low",
            "runner-odd-fluctuation-high",
            "runner-odd-fluctuation-sparkline",
        }:
            runner_id = str(attributes.get("data-runner-id") or "").strip()
            if runner_id:
                self._row["element_runner_ids"].add(runner_id)
        elif self._row is not None and lowered == "div" and (
            "race-runners__name__dog" in classes
        ):
            self._collect_name = True
        elif (
            self._row is not None
            and lowered == "span"
            and "race-runners__name__box" in classes
        ):
            self._row["effective_box_texts"].append("")
            self._collect_effective_box = True
        elif self._collect_name and lowered == "span":
            self._collect_name = False

    def handle_data(self, data: str) -> None:
        if self._row is not None and self._collect_effective_box:
            self._row["effective_box_texts"][-1] += data
        elif self._row is not None and self._collect_name and data.strip():
            self._row["runner_name"] += data.strip()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "span" and self._collect_effective_box:
            self._collect_effective_box = False
        if lowered == "div" and self._collect_name:
            self._collect_name = False
        if lowered == "tr" and self._row is not None:
            row = self._row
            self._row = None
            runner_id = str(row["native_runner_id"] or "").strip()
            if not runner_id:
                if (
                    row["active"]
                    or row["element_runner_ids"]
                    or row["runner_name"].strip().lower() != "vacant box"
                ):
                    raise CaptureError("native_runner_identity_missing_from_odds_page")
                return
            element_ids = row["element_runner_ids"]
            if element_ids and element_ids != {runner_id}:
                raise CaptureError("native_runner_identity_mismatch_in_odds_page")
            if not row["runner_name"] or row["box"] is None:
                raise CaptureError("runner_structure_incomplete_in_odds_page")
            page_effective_box = None
            effective_box_texts = [
                value.strip()
                for value in row["effective_box_texts"]
                if value.strip()
            ]
            if effective_box_texts:
                if len(effective_box_texts) != 1:
                    raise CaptureError("effective_box_source_ambiguous_in_odds_page")
                match = re.fullmatch(
                    r"\(\s*from\s+box\s+([1-8])\s*\)",
                    effective_box_texts[0],
                    flags=re.IGNORECASE,
                )
                if match is None:
                    raise CaptureError("effective_box_source_invalid_in_odds_page")
                page_effective_box = int(match.group(1))
            self.runners.append(
                SourceRunner(
                    native_runner_id=runner_id,
                    runner_name=row["runner_name"],
                    box=row["box"],
                    page_effective_box=page_effective_box,
                    active=bool(row["active"]),
                )
            )
        elif lowered == "tbody":
            self._tbody_runner_id = None


def parse_source_runners(source_html: bytes) -> tuple[SourceRunner, ...]:
    try:
        text = source_html.decode("utf-8")
    except UnicodeError as exc:
        raise CaptureError("odds_page_utf8_invalid") from exc
    parser = _RunnerParser()
    parser.feed(text)
    if not parser.runners:
        raise CaptureError("odds_page_runner_set_missing")
    ids = [runner.native_runner_id for runner in parser.runners]
    boxes = [runner.box for runner in parser.runners]
    if len(ids) != len(set(ids)):
        raise CaptureError("duplicate_native_runner_id")
    if len(boxes) != len(set(boxes)):
        raise CaptureError("duplicate_runner_box")
    if not any(runner.active for runner in parser.runners):
        raise CaptureError("active_runner_set_empty")
    return tuple(parser.runners)


def parse_jump_from_source(source_html: bytes) -> datetime:
    try:
        text = source_html.decode("utf-8")
    except UnicodeError as exc:
        raise CaptureError("jump_source_utf8_invalid") from exc
    matches = re.findall(
        r"<formatted-time\b(?=[^>]*\bdata-format=[\"']datetime_short[\"'])"
        r"(?=[^>]*\bdata-timestamp=[\"']([0-9]+)[\"'])[^>]*>",
        text,
        flags=re.IGNORECASE,
    )
    if len(set(matches)) != 1:
        raise CaptureError("jump_source_timestamp_not_unique")
    return datetime.fromtimestamp(int(matches[0]), tz=timezone.utc)


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(PAGE_HEADERS)
    adapter = HTTPAdapter(max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _receipt_headers(headers: Mapping[str, Any]) -> dict[str, str]:
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return {key: lowered[key] for key in RECEIPT_HTTP_HEADERS if key in lowered}


def timed_get(
    session: Any,
    url: str,
    *,
    referer: str | None,
    accept: str,
    clock: Callable[[], datetime],
    xhr: bool = False,
) -> TimedResponse:
    headers = {"Accept": accept}
    if referer:
        headers.update(
            {
                "Referer": referer,
                "Sec-Fetch-Site": "same-origin",
            }
        )
    if xhr:
        headers["X-Requested-With"] = "XMLHttpRequest"
    start = clock().astimezone(timezone.utc)
    response = session.get(url, timeout=30, headers=headers, allow_redirects=False)
    end = clock().astimezone(timezone.utc)
    body = bytes(response.content)
    return TimedResponse(
        requested_url=url,
        request_start_utc=start,
        request_end_utc=end,
        final_url=str(response.url),
        status_code=int(response.status_code),
        headers=_receipt_headers(response.headers),
        body=body,
    )


def validate_response(
    response: TimedResponse,
    *,
    exact_url: str,
    content_type_prefix: str,
) -> None:
    if response.status_code != 200:
        raise CaptureError(f"source_http_status_{response.status_code}")
    if response.final_url != exact_url or response.requested_url != exact_url:
        raise CaptureError("source_redirect_or_url_mismatch")
    content_type = response.headers.get("content-type", "").lower()
    if not content_type.startswith(content_type_prefix):
        raise CaptureError("source_content_type_invalid")
    content_encoding = response.headers.get("content-encoding", "").lower()
    if content_encoding not in {"", "identity"}:
        raise CaptureError("source_content_encoding_not_identity")
    if not response.body:
        raise CaptureError("source_body_empty")
    if response.request_end_utc < response.request_start_utc:
        raise CaptureError("request_receipt_time_order_invalid")
    server_date_text = response.headers.get("date")
    if not server_date_text:
        raise CaptureError("source_server_date_missing")
    try:
        server_date = parsedate_to_datetime(server_date_text).astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise CaptureError("source_server_date_invalid") from exc
    age_text = response.headers.get("age", "0").strip()
    try:
        age_seconds = int(age_text)
    except ValueError as exc:
        raise CaptureError("source_server_age_invalid") from exc
    if age_seconds < 0 or age_seconds > 300:
        raise CaptureError("source_server_age_invalid")
    latest_server_date = server_date + timedelta(seconds=age_seconds)
    tolerance = timedelta(seconds=SERVER_DATE_TOLERANCE_SECONDS)
    if not (
        server_date <= response.request_end_utc + tolerance
        and latest_server_date >= response.request_start_utc - tolerance
    ):
        raise CaptureError("source_server_date_outside_request_interval")


def _api_url(runners: tuple[SourceRunner, ...]) -> str:
    query = urlencode(
        [("runner_ids[]", runner.native_runner_id) for runner in runners]
    )
    return f"https://www.thedogs.com.au/api/runners/odds?{query}"


def _safe_price(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 1.0 else None


def _quote_for_runner(
    payload: Mapping[str, Any], runner: SourceRunner
) -> Mapping[str, Any] | None:
    runner_odds = payload.get("runner_odds")
    if not isinstance(runner_odds, Mapping):
        raise CaptureError("odds_api_runner_odds_missing")
    quotes = runner_odds.get(runner.native_runner_id)
    if not isinstance(quotes, list) or not quotes:
        if runner.active:
            raise CaptureError("active_runner_current_price_missing")
        return None
    fixed_quotes = [
        quote
        for quote in quotes
        if isinstance(quote, Mapping)
        and isinstance(quote.get("market"), Mapping)
        and quote["market"].get("code") == "fixed_win"
    ]
    if len(fixed_quotes) != 1:
        raise CaptureError("runner_fixed_win_quote_not_unique")
    quote = fixed_quotes[0]
    if str(quote.get("runner_id") or "").strip() != runner.native_runner_id:
        raise CaptureError("native_runner_quote_id_mismatch")
    return quote


def normalize_api_snapshot(
    payload: Mapping[str, Any], source_runners: tuple[SourceRunner, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    rows: list[dict[str, Any]] = []
    active_effective_boxes: list[int] = []
    provider_values: set[tuple[str, str, str]] = set()
    active_provider_unknown = 0
    native_race_ids: set[str] = set()
    for runner in source_runners:
        quote = _quote_for_runner(payload, runner)
        price = _safe_price(quote.get("price")) if quote is not None else None
        if runner.active and price is None:
            raise CaptureError("active_runner_current_price_invalid")
        if not runner.active and price is not None:
            raise CaptureError("scratched_runner_has_active_price")
        bookmaker = quote.get("bookmaker") if quote is not None else None
        provider = None
        if isinstance(bookmaker, Mapping):
            provider = {
                "id": str(bookmaker.get("id") or ""),
                "code": str(bookmaker.get("code") or "").strip(),
                "name": str(bookmaker.get("name") or "").strip(),
            }
            if any(provider.values()):
                if runner.active:
                    provider_values.add(
                        (provider["id"], provider["code"], provider["name"])
                    )
            else:
                provider = None
        if runner.active and provider is None:
            active_provider_unknown += 1
        market = quote.get("market") if quote is not None else None
        native_race_id = ""
        if isinstance(market, Mapping):
            native_race_id = str(market.get("race_id") or "").strip()
            if native_race_id:
                native_race_ids.add(native_race_id)
        api_run_box = quote.get("run_box") if quote is not None else None
        parsed_api_run_box = None
        if api_run_box is not None:
            if isinstance(api_run_box, bool) or re.fullmatch(
                r"[1-8]", str(api_run_box).strip()
            ) is None:
                raise CaptureError("effective_box_source_invalid")
            parsed_api_run_box = int(str(api_run_box).strip())
        if (
            runner.page_effective_box is not None
            and runner.page_effective_box == runner.box
        ):
            raise CaptureError("effective_box_source_conflict")
        expected_api_box = (
            runner.page_effective_box
            if runner.page_effective_box is not None
            else runner.box
        )
        if runner.active:
            if parsed_api_run_box is None:
                raise CaptureError("effective_box_source_invalid")
            if parsed_api_run_box != expected_api_box:
                raise CaptureError("effective_box_source_conflict")
            effective_box = parsed_api_run_box
            active_effective_boxes.append(effective_box)
            if runner.page_effective_box is None:
                page_source = "thedogs_odds_page_sprite_svg_rug"
                resolution = "normal_page_and_api_box_match"
            else:
                page_source = (
                    "thedogs_odds_page_race_runners_name_box_from_box"
                )
                resolution = "explicit_replacement_box_match"
        else:
            if (
                parsed_api_run_box is not None
                and parsed_api_run_box != expected_api_box
            ):
                raise CaptureError("effective_box_source_conflict")
            effective_box = None
            page_source = (
                "thedogs_odds_page_race_runners_name_box_from_box"
                if runner.page_effective_box is not None
                else "thedogs_odds_page_sprite_svg_rug"
            )
            resolution = "inactive_runner_no_effective_box"
        rows.append(
            {
                "native_runner_id": runner.native_runner_id,
                "runner_name": runner.runner_name,
                "box": runner.box,
                "page_box": runner.box,
                "page_effective_box": runner.page_effective_box,
                "api_run_box": api_run_box,
                "effective_box": effective_box,
                "effective_box_provenance": {
                    "page_source": page_source,
                    "api_source": (
                        "thedogs_runner_odds_api_fixed_win_run_box"
                        if quote is not None
                        else "thedogs_runner_odds_api_quote_absent_for_inactive_runner"
                    ),
                    "resolution": resolution,
                },
                "active": runner.active,
                "current_price": price,
                "provider": provider,
            }
        )
    if len(active_effective_boxes) != len(set(active_effective_boxes)):
        raise CaptureError("active_effective_box_not_unique")
    if len(native_race_ids) != 1:
        raise CaptureError("native_race_identity_not_unique")
    if len(provider_values) > 1:
        raise CaptureError("provider_identity_not_unique")
    if provider_values and active_provider_unknown:
        raise CaptureError("provider_identity_incomplete")
    if provider_values:
        provider_id, code, name = next(iter(provider_values))
        provider = {
            "classification": "provider_explicit",
            "id": provider_id,
            "code": code,
            "name": name,
            "source": "thedogs_runner_odds_api_bookmaker",
        }
    else:
        provider = {
            "classification": "provider_unknown",
            "id": None,
            "code": None,
            "name": None,
            "source": "provider_not_explicit_in_source_payload",
        }
    return rows, provider, next(iter(native_race_ids))


def _response_receipt(response: TimedResponse, *, include_body: bool) -> dict[str, Any]:
    payload = {
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "request_start_utc": iso_utc(response.request_start_utc),
        "request_end_utc": iso_utc(response.request_end_utc),
        "status_code": response.status_code,
        "headers": response.headers,
        "body_sha256": sha256_bytes(response.body),
        "body_bytes": len(response.body),
    }
    if include_body:
        payload["body_base64"] = base64.b64encode(response.body).decode("ascii")
    return payload


def _write_new_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.chmod(stat.S_IWUSR | stat.S_IRUSR)
            path.unlink()
        except OSError:
            pass
        raise


def _stored_response(
    payload: Any,
    *,
    field: str,
    exact_url: str,
    content_type_prefix: str,
    body: bytes | None = None,
    require_body: bool = False,
) -> TimedResponse:
    if not isinstance(payload, Mapping):
        raise CaptureError(f"{field}_receipt_missing")
    observed_body = body
    if require_body:
        encoded = payload.get("body_base64")
        if not isinstance(encoded, str) or not encoded:
            raise CaptureError(f"{field}_raw_bytes_missing")
        try:
            observed_body = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CaptureError(f"{field}_raw_bytes_invalid") from exc
    if observed_body is not None:
        if sha256_bytes(observed_body) != payload.get("body_sha256"):
            raise CaptureError(f"{field}_raw_hash_mismatch")
        if len(observed_body) != payload.get("body_bytes"):
            raise CaptureError(f"{field}_raw_size_mismatch")
    elif re.fullmatch(r"[0-9a-f]{64}", str(payload.get("body_sha256") or "")) is None:
        raise CaptureError(f"{field}_body_hash_invalid")
    response = TimedResponse(
        requested_url=str(payload.get("requested_url") or ""),
        request_start_utc=parse_timestamp(
            payload.get("request_start_utc"), field=f"{field}_request_start_utc"
        ),
        request_end_utc=parse_timestamp(
            payload.get("request_end_utc"), field=f"{field}_request_end_utc"
        ),
        final_url=str(payload.get("final_url") or ""),
        status_code=payload.get("status_code"),
        headers=dict(payload.get("headers") or {}),
        body=observed_body if observed_body is not None else b"receipt-body-not-embedded",
    )
    validate_response(
        response,
        exact_url=exact_url,
        content_type_prefix=content_type_prefix,
    )
    return response


def assert_output_dir_safe(output_dir: Path, *, repo_root: Path = ROOT) -> Path:
    logical = output_dir if output_dir.is_absolute() else repo_root / output_dir
    try:
        resolved_root = repo_root.resolve(strict=False)
        resolved = logical.resolve(strict=False)
        relative = resolved.relative_to(resolved_root)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise CaptureError("output_dir_must_be_inside_repo") from exc
    if "capture" not in relative.parts and "canary" not in relative.parts:
        raise CaptureError("output_dir_must_be_capture_or_canary_path")
    current = logical
    while current != repo_root and current != current.parent:
        if current.exists() and current.is_symlink():
            raise CaptureError("output_dir_symlink_not_allowed")
        current = current.parent
    return resolved


def _validate_existing(
    *, plan: Mapping[str, Any], raw_path: Path, receipt_path: Path
) -> dict[str, Any] | None:
    if not raw_path.exists() and not receipt_path.exists():
        return None
    if not raw_path.is_file() or not receipt_path.is_file():
        raise CaptureError("conflicting_snapshot_partial")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError("conflicting_snapshot_receipt_invalid") from exc
    if not isinstance(receipt, dict):
        raise CaptureError("conflicting_snapshot_receipt_invalid")
    raw = raw_path.read_bytes()
    observed_raw_hash = sha256_bytes(raw)
    if receipt.get("schema_version") not in RECEIPT_SCHEMA_VERSIONS:
        raise CaptureError("conflicting_snapshot_receipt_schema")
    expected = {
        "race_id": str(plan.get("race_id") or "").strip(),
        "odds_url": str(plan.get("odds_url") or "").strip(),
        "nominal_window": str(plan.get("nominal_window") or "").strip().upper(),
        "raw_html_sha256": observed_raw_hash,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise CaptureError("conflicting_snapshot")
    core_hash = str(receipt.get("receipt_core_sha256") or "")
    core = {key: value for key, value in receipt.items() if key != "receipt_core_sha256"}
    if core_hash != sha256_bytes(canonical_json_bytes(core)):
        raise CaptureError("conflicting_snapshot_receipt_hash")
    try:
        identity = exact_odds_identity(plan.get("odds_url"))
        plan_jump = parse_timestamp(plan.get("jump_timestamp"), field="jump_timestamp")
        receipt_jump = parse_timestamp(
            receipt.get("jump_timestamp"), field="jump_timestamp"
        )
        capture_end = parse_timestamp(
            receipt.get("capture_end_utc"), field="capture_end_utc"
        )
    except CaptureError as exc:
        raise CaptureError("conflicting_snapshot_receipt_invalid") from exc
    if (
        plan_jump != receipt_jump
        or capture_end >= receipt_jump
        or receipt.get("race_url") != plan.get("race_url")
        or receipt.get("race_identity") != identity
        or receipt.get("source_class") != SOURCE_CLASS
        or receipt.get("field_complete") is not True
        or receipt.get("raw_html_bytes") != raw_path.stat().st_size
    ):
        raise CaptureError("conflicting_snapshot")
    active_ids = {str(value) for value in receipt.get("active_native_runner_ids") or []}
    expected_active_ids = {
        str(value).strip() for value in plan.get("expected_active_runner_ids", [])
    }
    if not active_ids or (expected_active_ids and expected_active_ids != active_ids):
        raise CaptureError("conflicting_snapshot")
    try:
        source_runners = parse_source_runners(raw)
        source_all_ids = {runner.native_runner_id for runner in source_runners}
        source_active_ids = {
            runner.native_runner_id for runner in source_runners if runner.active
        }
        if source_active_ids != expected_active_ids or active_ids != source_active_ids:
            raise CaptureError("runner_set_mismatch")
        if set(receipt.get("all_native_runner_ids") or []) != source_all_ids:
            raise CaptureError("all_runner_set_mismatch")
        meeting = _stored_response(
            receipt.get("warm_meeting_http"),
            field="warm_meeting",
            exact_url=str(identity["meeting_url"]),
            content_type_prefix="text/html",
        )
        jump_source = _stored_response(
            receipt.get("jump_source"),
            field="jump_source",
            exact_url=str(identity["race_url"]),
            content_type_prefix="text/html",
            require_body=True,
        )
        odds = _stored_response(
            receipt.get("odds_page_http"),
            field="odds_page",
            exact_url=str(identity["odds_url"]),
            content_type_prefix="text/html",
            body=raw,
        )
        api = _stored_response(
            receipt.get("odds_api_http"),
            field="odds_api",
            exact_url=_api_url(source_runners),
            content_type_prefix="application/json",
            require_body=True,
        )
        if not (
            meeting.request_end_utc <= jump_source.request_start_utc
            <= jump_source.request_end_utc
            <= odds.request_start_utc
            <= odds.request_end_utc
            <= api.request_start_utc
            <= api.request_end_utc
        ):
            raise CaptureError("request_chain_time_order_invalid")
        if (
            receipt.get("capture_start_utc") != iso_utc(meeting.request_start_utc)
            or receipt.get("request_start_utc") != iso_utc(odds.request_start_utc)
            or receipt.get("request_end_utc") != iso_utc(api.request_end_utc)
            or receipt.get("capture_end_utc") != iso_utc(api.request_end_utc)
        ):
            raise CaptureError("request_receipt_time_mismatch")
        jump_body = base64.b64decode(
            str(receipt["jump_source"]["body_base64"]), validate=True
        )
        if parse_jump_from_source(jump_body) != plan_jump:
            raise CaptureError("jump_source_timestamp_mismatch")
        api_body = base64.b64decode(
            str(receipt["odds_api_http"]["body_base64"]), validate=True
        )
        api_payload = json.loads(api_body.decode("utf-8"))
        if not isinstance(api_payload, Mapping):
            raise CaptureError("odds_api_json_invalid")
        normalized_rows, normalized_provider, native_race_id = normalize_api_snapshot(
            api_payload, source_runners
        )
        if receipt.get("schema_version") == LEGACY_SCHEMA_VERSION:
            legacy_keys = {
                "native_runner_id",
                "runner_name",
                "box",
                "active",
                "current_price",
                "provider",
            }
            projected_rows = [
                {key: value for key, value in row.items() if key in legacy_keys}
                for row in normalized_rows
            ]
        else:
            projected_rows = normalized_rows
        if (
            receipt.get("runners") != projected_rows
            or receipt.get("provider") != normalized_provider
            or receipt.get("source_native_race_id") != native_race_id
        ):
            raise CaptureError("receipt_projection_mismatch")
        nominal_window = str(receipt.get("nominal_window") or "")
        if receipt.get("nominal_capture_utc") != iso_utc(
            planned_time(plan_jump, nominal_window)
        ):
            raise CaptureError("nominal_capture_timestamp_mismatch")
        validate_due_window(
            now=capture_end,
            jump_utc=plan_jump,
            nominal_window=nominal_window,
            early_tolerance_seconds=int(receipt.get("early_tolerance_seconds")),
            late_tolerance_seconds=int(receipt.get("late_tolerance_seconds")),
        )
    except (CaptureError, KeyError, TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError("conflicting_snapshot") from exc
    if receipt.get("open_low_high_are_temporal_observations") is not False:
        raise CaptureError("conflicting_snapshot")
    return {
        "status": "SKIPPED_IDENTICAL_SNAPSHOT",
        "accepted": True,
        "raw_html_path": str(raw_path),
        "receipt_path": str(receipt_path),
        "raw_html_sha256": observed_raw_hash,
        "receipt_sha256": sha256_bytes(receipt_path.read_bytes()),
    }


def capture_snapshot(
    plan: Mapping[str, Any],
    output_dir: Path,
    *,
    session: Any | None = None,
    current_time: datetime | None = None,
    clock: Callable[[], datetime] = utc_now,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise CaptureError("plan_schema_version_invalid")
    race_id = str(plan.get("race_id") or "").strip()
    if not race_id:
        raise CaptureError("race_id_required")
    identity = exact_odds_identity(plan.get("odds_url"))
    if plan.get("race_url") != identity["race_url"]:
        raise CaptureError("race_url_odds_url_mismatch")
    jump_utc = parse_timestamp(plan.get("jump_timestamp"), field="jump_timestamp")
    nominal_window = str(plan.get("nominal_window") or "").strip().upper()
    nominal_window_minutes(nominal_window)
    expected_values = plan.get("expected_active_runner_ids")
    if not isinstance(expected_values, list) or not expected_values:
        raise CaptureError("expected_active_runner_ids_required")
    expected_ids = {str(value).strip() for value in expected_values}
    if "" in expected_ids or len(expected_ids) != len(expected_values):
        raise CaptureError("expected_active_runner_ids_invalid")
    safe_output_dir = assert_output_dir_safe(output_dir, repo_root=repo_root)
    raw_path = safe_output_dir / "raw.html"
    receipt_path = safe_output_dir / "receipt.json"
    existing = _validate_existing(
        plan=plan,
        raw_path=raw_path,
        receipt_path=receipt_path,
    )
    if existing is not None:
        return existing
    now = (current_time or clock()).astimezone(timezone.utc)
    early_tolerance_seconds = int(
        plan.get("early_tolerance_seconds", DEFAULT_EARLY_TOLERANCE_SECONDS)
    )
    late_tolerance_seconds = int(
        plan.get("late_tolerance_seconds", DEFAULT_LATE_TOLERANCE_SECONDS)
    )
    nominal_at = validate_due_window(
        now=now,
        jump_utc=jump_utc,
        nominal_window=nominal_window,
        early_tolerance_seconds=early_tolerance_seconds,
        late_tolerance_seconds=late_tolerance_seconds,
    )
    active_session = session or make_session()
    meeting = timed_get(
        active_session,
        str(identity["meeting_url"]),
        referer=None,
        accept=PAGE_HEADERS["Accept"],
        clock=clock,
    )
    validate_response(
        meeting,
        exact_url=str(identity["meeting_url"]),
        content_type_prefix="text/html",
    )
    race = timed_get(
        active_session,
        str(identity["race_url"]),
        referer=str(identity["meeting_url"]),
        accept=PAGE_HEADERS["Accept"],
        clock=clock,
    )
    validate_response(
        race,
        exact_url=str(identity["race_url"]),
        content_type_prefix="text/html",
    )
    source_jump = parse_jump_from_source(race.body)
    if source_jump != jump_utc:
        raise CaptureError("jump_source_timestamp_mismatch")
    odds = timed_get(
        active_session,
        str(identity["odds_url"]),
        referer=str(identity["race_url"]),
        accept=PAGE_HEADERS["Accept"],
        clock=clock,
    )
    validate_response(
        odds,
        exact_url=str(identity["odds_url"]),
        content_type_prefix="text/html",
    )
    source_runners = parse_source_runners(odds.body)
    observed_active_ids = {
        runner.native_runner_id for runner in source_runners if runner.active
    }
    if expected_ids != observed_active_ids:
        raise CaptureError("expected_native_runner_set_mismatch")
    api_url = _api_url(source_runners)
    api = timed_get(
        active_session,
        api_url,
        referer=str(identity["odds_url"]),
        accept="application/json",
        clock=clock,
        xhr=True,
    )
    validate_response(
        api,
        exact_url=api_url,
        content_type_prefix="application/json",
    )
    try:
        api_payload = json.loads(api.body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError("odds_api_json_invalid") from exc
    if not isinstance(api_payload, Mapping):
        raise CaptureError("odds_api_json_invalid")
    rows, provider, native_race_id = normalize_api_snapshot(
        api_payload, source_runners
    )
    expected_native_race_id = str(plan.get("expected_native_race_id") or "").strip()
    if expected_native_race_id and expected_native_race_id != native_race_id:
        raise CaptureError("expected_native_race_id_mismatch")
    capture_end = api.request_end_utc
    if capture_end >= jump_utc:
        raise CaptureError("capture_not_strictly_prejump")
    if capture_end < nominal_at - timedelta(seconds=early_tolerance_seconds):
        raise CaptureError("capture_completed_before_nominal_window")
    if capture_end > nominal_at + timedelta(seconds=late_tolerance_seconds):
        raise CaptureError("capture_completed_after_nominal_window")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_class": SOURCE_CLASS,
        "race_id": race_id,
        "source_native_race_id": native_race_id,
        "race_identity": identity,
        "race_url": identity["race_url"],
        "odds_url": identity["odds_url"],
        "nominal_window": nominal_window,
        "nominal_capture_utc": iso_utc(nominal_at),
        "early_tolerance_seconds": early_tolerance_seconds,
        "late_tolerance_seconds": late_tolerance_seconds,
        "jump_timestamp": iso_utc(jump_utc),
        "capture_start_utc": iso_utc(meeting.request_start_utc),
        "capture_end_utc": iso_utc(capture_end),
        "request_start_utc": iso_utc(odds.request_start_utc),
        "request_end_utc": iso_utc(capture_end),
        "raw_html_sha256": sha256_bytes(odds.body),
        "raw_html_bytes": len(odds.body),
        "jump_source": _response_receipt(race, include_body=True),
        "odds_page_http": _response_receipt(odds, include_body=False),
        "odds_api_http": _response_receipt(api, include_body=True),
        "warm_meeting_http": _response_receipt(meeting, include_body=False),
        "provider": provider,
        "field_complete": True,
        "active_runner_count": len(observed_active_ids),
        "all_runner_count": len(source_runners),
        "active_native_runner_ids": sorted(observed_active_ids),
        "all_native_runner_ids": sorted(
            runner.native_runner_id for runner in source_runners
        ),
        "runners": rows,
        "open_low_high_are_temporal_observations": False,
        "request_attempts": {
            "meeting_page": 1,
            "jump_source_race_page": 1,
            "exact_odds_page": 1,
            "native_runner_odds_api": 1,
        },
        "no_write_guarantees": {
            "canonical_db_write": False,
            "runtime_write": False,
            "service_or_timer_change": False,
            "training": False,
            "scoring": False,
            "promotion": False,
            "betting": False,
            "august_cohort_use": False,
        },
    }
    receipt["receipt_core_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    raw_bytes = odds.body
    receipt_bytes = serialized_json_bytes(receipt)
    _write_new_immutable(raw_path, raw_bytes)
    _write_new_immutable(receipt_path, receipt_bytes)
    return {
        "status": "CAPTURE_ACCEPTED",
        "accepted": True,
        "race_id": race_id,
        "nominal_window": nominal_window,
        "capture_end_utc": iso_utc(capture_end),
        "raw_html_path": str(raw_path),
        "receipt_path": str(receipt_path),
        "raw_html_sha256": sha256_bytes(raw_bytes),
        "receipt_sha256": sha256_bytes(receipt_bytes),
        "active_runner_count": len(observed_active_ids),
        "provider": provider,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--current-time", help="Test/canary clock override with timezone")
    args = parser.parse_args()
    try:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        if not isinstance(plan, Mapping):
            raise CaptureError("plan_must_be_json_object")
        current_time = (
            parse_timestamp(args.current_time, field="current_time")
            if args.current_time
            else None
        )
        result = capture_snapshot(
            plan,
            args.output_dir,
            current_time=current_time,
        )
    except (CaptureError, OSError, json.JSONDecodeError, requests.RequestException) as exc:
        result = {"status": "CAPTURE_REJECTED", "accepted": False, "blocker": str(exc)}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
