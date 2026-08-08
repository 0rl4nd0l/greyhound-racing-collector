"""One-attempt live source child for the bounded GHU-051 executor.

This module is intentionally not a capture contract of its own. It accepts
only the exact values exported by the parent executor, visits that one URL in
the dedicated manual profile, and emits the existing child envelope with a
versioned live schema. It reads runner rows and explicit WIN odds only; no
page body, form history, result fields, or alternate URL is consumed.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.predictor.manual_independent_capture import canonical_bytes
from src.predictor.manual_independent_capture_executor import (
    LIVE_CHILD_SCHEMA_VERSION,
)
from utils.csv_metadata import canonical_thedogs_race_identity

CONTENT_TYPE = "application/vnd.greyhound.manual-live+json"
_MAX_RUNNER_COUNT = 10
_PAGE_TIMEOUT_MS = 20_000
_ODDS_RE = re.compile(r"^\d+(?:\.\d+)?$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CHALLENGE_SELECTORS = (
    "input[type='password']",
    "[data-captcha]",
    "iframe[src*='challenge']",
    ".cf-challenge",
    "#challenge-running",
)
_OUTCOME_SELECTORS = (
    "[data-result]",
    "[data-outcome]",
    "[data-finishing-position]",
    ".race-result",
    ".race-results",
    ".race-result__winner",
)
_OUTCOME_URL_MARKERS = ("/result", "/results", "dividend", "payout", "outcome")


class LiveCaptureRejected(RuntimeError):
    """Fail-closed source rejection with no retry instruction."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or "\x00" in value:
        raise LiveCaptureRejected("EXACT_INPUT_MISSING")
    return value


def _safe_directory(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise LiveCaptureRejected("PROFILE_PATH_INVALID")
    try:
        info = path.lstat()
    except OSError as exc:
        raise LiveCaptureRejected("PROFILE_PATH_INVALID") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise LiveCaptureRejected("PROFILE_PATH_INVALID")
    return path


def _decimal_odds(value: Any) -> float:
    if not isinstance(value, str) or not _ODDS_RE.fullmatch(value.strip()):
        raise LiveCaptureRejected("ODDS_INVALID")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise LiveCaptureRejected("ODDS_INVALID") from exc
    if not parsed.is_finite() or parsed <= 1:
        raise LiveCaptureRejected("ODDS_INVALID")
    result = float(parsed)
    if not math.isfinite(result) or result <= 1:
        raise LiveCaptureRejected("ODDS_INVALID")
    return result


def _identity(race_id: str, box_number: int, display_name: str) -> str:
    normalized_name = re.sub(r"[^A-Z0-9]", "", display_name.upper())
    if not normalized_name:
        raise LiveCaptureRejected("RUNNER_SET_MISMATCH")
    return f"{race_id}|BOX:{box_number}|DOG:{normalized_name}".upper()


def _rows_to_runners(rows: Sequence[Mapping[str, Any]], race_id: str) -> list[dict[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise LiveCaptureRejected("SOURCE_MALFORMED")
    if not 1 <= len(rows) <= _MAX_RUNNER_COUNT:
        raise LiveCaptureRejected("SOURCE_MALFORMED")
    runners: list[dict[str, Any]] = []
    seen_boxes: set[int] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "box_number",
            "display_name",
            "source_native_runner_id",
            "win_odds",
        }:
            raise LiveCaptureRejected("SOURCE_MALFORMED")
        box = row["box_number"]
        name = row["display_name"]
        native_id = row["source_native_runner_id"]
        if (
            isinstance(box, bool)
            or not isinstance(box, int)
            or not 1 <= box <= 10
            or box in seen_boxes
            or not isinstance(name, str)
            or not name
            or name != name.strip()
            or (
                native_id is not None
                and (not isinstance(native_id, str) or not native_id.strip())
            )
        ):
            raise LiveCaptureRejected("RUNNER_SET_MISMATCH")
        seen_boxes.add(box)
        runners.append(
            {
                "box_number": box,
                "display_name": name,
                "identity": _identity(race_id, box, name),
                "source_native_runner_id": native_id,
                "decimal_odds": _decimal_odds(row["win_odds"]),
            }
        )
    if [row["box_number"] for row in runners] != sorted(seen_boxes):
        raise LiveCaptureRejected("RUNNER_SET_MISMATCH")
    return runners


def _page_probe(page: Any, exact_url: str) -> tuple[int, str]:
    try:
        response = page.goto(
            exact_url,
            wait_until="domcontentloaded",
            timeout=_PAGE_TIMEOUT_MS,
        )
        final_url = str(page.url)
        status_code = int(response.status) if response is not None else 0
        title = str(page.title()).strip().lower()
        challenge_count = sum(
            int(page.locator(selector).count()) for selector in _CHALLENGE_SELECTORS
        )
        outcome_count = sum(
            int(page.locator(selector).count()) for selector in _OUTCOME_SELECTORS
        )
    except Exception as exc:  # Playwright errors are terminal, never retried.
        raise LiveCaptureRejected("SOURCE_ATTEMPT_FAILED") from exc
    if challenge_count or any(token in title for token in ("login", "challenge", "captcha")):
        raise LiveCaptureRejected("SOURCE_CHALLENGE")
    if outcome_count:
        raise LiveCaptureRejected("OUTCOME_MATERIAL_FORBIDDEN")
    return status_code, final_url


def _install_navigation_guard(page: Any, exact_url: str) -> None:
    def guard(route: Any) -> None:
        request = route.request
        request_url = str(request.url)
        if request_url != exact_url and (
            request.is_navigation_request()
            or any(marker in request_url.lower() for marker in _OUTCOME_URL_MARKERS)
        ):
            route.abort()
            return
        route.continue_()

    page.route("**/*", guard)


def capture_from_page(
    page: Any,
    *,
    exact_url: str,
    race_id: str,
    race_identity_sha256: str | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Capture one mocked or real page without reading unrestricted page text."""

    identity = canonical_thedogs_race_identity(exact_url)
    if identity is None or identity["canonical_url"] != exact_url:
        raise LiveCaptureRejected("EXACT_INPUT_INVALID")
    if not race_id or not isinstance(race_id, str):
        raise LiveCaptureRejected("EXACT_INPUT_INVALID")
    status_code, final_url = _page_probe(page, exact_url)
    if final_url != exact_url:
        raise LiveCaptureRejected("IDENTITY_MISMATCH")
    if status_code != 200:
        raise LiveCaptureRejected("SOURCE_ATTEMPT_FAILED")
    try:
        rows = page.locator("tr.race-runner").evaluate_all(
            """
            rows => rows.map(row => ({
              box_number: Number(
                row.querySelector('.race-runners__box')?.getAttribute('data-box')
                || row.querySelector('sprite-svg[name^="rug_"]')?.getAttribute('name')?.replace('rug_', '')
                || ''
              ),
              display_name: (
                row.querySelector('.race-runners__name__dog')
                || row.querySelector('.race-runners__name')
              )?.textContent?.trim() || '',
              source_native_runner_id: row.getAttribute('data-runner-id')
                || row.getAttribute('data-native-runner-id') || null,
              win_odds: row.querySelector('[data-win-odds]')?.getAttribute('data-win-odds')
                || row.querySelector('.race-runners__odds')?.textContent?.trim() || ''
            }))
            """
        )
    except Exception as exc:
        raise LiveCaptureRejected("SOURCE_MALFORMED") from exc
    runners = _rows_to_runners(rows, race_id)
    timestamp = now()
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise LiveCaptureRejected("SOURCE_MALFORMED")
    timestamp = timestamp.replace(microsecond=0)
    body = canonical_bytes(
        {
            "runners": [
                {
                    "box_number": row["box_number"],
                    "display_name": row["display_name"],
                    "decimal_odds": row["decimal_odds"],
                }
                for row in runners
            ]
        }
    )
    identity_sha = race_identity_sha256 or _required_env(
        "MANUAL_CAPTURE_RACE_IDENTITY_SHA256"
    )
    if _SHA256_RE.fullmatch(identity_sha) is None:
        raise LiveCaptureRejected("EXACT_INPUT_INVALID")
    return {
        "schema_version": LIVE_CHILD_SCHEMA_VERSION,
        "requested_race_url": exact_url,
        "race_identity_sha256": identity_sha,
        "runners": runners,
        "source": {
            "content_class": "prejump_sidecar",
            "source_timestamp": timestamp.isoformat(),
            "final_url": final_url,
            "status_code": status_code,
            "content_type": CONTENT_TYPE,
            "bytes_base64": base64.b64encode(body).decode("ascii"),
        },
    }


def _main() -> int:
    if os.environ.get("MANUAL_CAPTURE_EXECUTOR_PROTOCOL") != "ghu051-bounded-v1":
        raise LiveCaptureRejected("EXECUTOR_PROTOCOL_REQUIRED")
    exact_url = _required_env("MANUAL_CAPTURE_EXACT_URL")
    _safe_directory(_required_env("MANUAL_CAPTURE_PROFILE"))
    _safe_directory(_required_env("MANUAL_CAPTURE_RUN_DIR"))
    race_id = _required_env("MANUAL_CAPTURE_RACE_ID")
    race_sha = _required_env("MANUAL_CAPTURE_RACE_IDENTITY_SHA256")
    if _SHA256_RE.fullmatch(race_sha) is None:
        raise LiveCaptureRejected("EXACT_INPUT_INVALID")
    identity = canonical_thedogs_race_identity(exact_url)
    if identity is None or identity["canonical_url"] != exact_url:
        raise LiveCaptureRejected("EXACT_INPUT_INVALID")
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise LiveCaptureRejected("LIVE_RUNTIME_UNAVAILABLE") from exc
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=_required_env("MANUAL_CAPTURE_PROFILE"),
            headless=True,
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            _install_navigation_guard(page, exact_url)
            record = capture_from_page(
                page,
                exact_url=exact_url,
                race_id=race_id,
            )
            print(json.dumps(record, sort_keys=True, separators=(",", ":")))
            return 0
        finally:
            context.close()


def main() -> int:
    try:
        return _main()
    except LiveCaptureRejected as exc:
        print(exc.code, file=sys.stderr)
        return 78
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - fail-closed boundary
        print(f"SOURCE_ATTEMPT_FAILED:{type(exc).__name__}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CONTENT_TYPE", "LiveCaptureRejected", "capture_from_page", "main"]
