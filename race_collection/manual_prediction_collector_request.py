"""Append-only exact-race handoff between the manual predictor and collector."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REQUEST_SCHEMA = "manual-prediction-collector-request-v1"
CLAIM_SCHEMA = "manual-prediction-collector-claim-v1"
ATTEMPT_SCHEMA = "manual-prediction-collector-attempt-v1"
RESPONSE_SCHEMA = "manual-prediction-collector-response-v1"
RECEIPT_SCHEMA = "manual-prediction-collector-receipt-v1"
CONSUME_SCHEMA = "manual-prediction-collector-consume-v1"
EXACT_RECEIPT_SCHEMA = "manual-prediction-exact-receipt-index-v1"
COLLECTOR_EXACT_RECEIPT_SCHEMA = "collector-exact-capture-receipt-v1"
PROTOCOL_DIRECTORY = "manual_prediction_collector_requests_v1"

RECEIPT_READY = "RECEIPT_READY"
TERMINAL_STATUSES = frozenset(
    {
        RECEIPT_READY,
        "REQUEST_EXPIRED",
        "RACE_NOT_FOUND",
        "CAPTURE_WINDOW_CLOSED",
        "IDENTITY_MISMATCH",
        "CAPTURE_FAILED",
    }
)
_RACE_KEYS = {
    "race_id",
    "url",
    "venue",
    "race_number",
    "race_date",
    "jump_timestamp",
}
_REQUEST_KEYS = {
    "schema_version",
    "request_id",
    "race",
    "created_at",
    "expires_at",
    "research_only",
    "attempt_authority",
    "requested_output",
    "expected_runners",
    "expected_runner_set_sha256",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class ProtocolRejected(ValueError):
    """A protocol record or transition is unsafe or ambiguous."""

    def __init__(self, code: str, **details: Any):
        super().__init__(code)
        self.code = code
        self.details = details


@dataclass(frozen=True, slots=True)
class CollectorRequest:
    request: Mapping[str, Any]
    claim: Mapping[str, Any]
    request_sha256: str
    claim_sha256: str
    recovered: bool = False


def canonical_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolRejected("JSON_INVALID") from exc
    return (text + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _known_text(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ProtocolRejected("IDENTITY_INVALID", field=field)
    return value


def _timestamp(value: Any, field: str) -> tuple[datetime, str]:
    if isinstance(value, datetime):
        parsed = value
    elif type(value) is str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ProtocolRejected("TIMESTAMP_INVALID", field=field) from exc
    else:
        raise ProtocolRejected("TIMESTAMP_INVALID", field=field)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProtocolRejected("TIMESTAMP_INVALID", field=field)
    return parsed, parsed.isoformat()


def _hash(value: Any, field: str) -> str:
    if type(value) is not str or _HASH_RE.fullmatch(value) is None:
        raise ProtocolRejected("HASH_INVALID", field=field)
    return value


def _source_url(value: Any) -> str:
    url = _known_text(value, "race.url")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.thedogs.com.au", "thedogs.com.au"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ProtocolRejected("IDENTITY_INVALID", field="race.url")
    return url


def _race(value: Any) -> dict[str, Any]:
    if type(value) is not dict or set(value) != _RACE_KEYS:
        raise ProtocolRejected("IDENTITY_INVALID", field="race")
    race_id = _known_text(value["race_id"], "race.race_id")
    venue = _known_text(value["venue"], "race.venue")
    if venue != venue.upper():
        raise ProtocolRejected("IDENTITY_INVALID", field="race.venue")
    if type(value["race_number"]) is not int or not 1 <= value["race_number"] <= 20:
        raise ProtocolRejected("IDENTITY_INVALID", field="race.race_number")
    try:
        race_date = date.fromisoformat(str(value["race_date"])).isoformat()
    except ValueError as exc:
        raise ProtocolRejected("IDENTITY_INVALID", field="race.race_date") from exc
    jump, jump_text = _timestamp(value["jump_timestamp"], "race.jump_timestamp")
    if jump.date().isoformat() != race_date:
        raise ProtocolRejected("IDENTITY_INVALID", field="race.jump_timestamp")
    return {
        "race_id": race_id,
        "url": _source_url(value["url"]),
        "venue": venue,
        "race_number": value["race_number"],
        "race_date": race_date,
        "jump_timestamp": jump_text,
    }


def _runner_rows(values: Any) -> list[dict[str, Any]]:
    if type(values) not in {list, tuple}:
        raise ProtocolRejected("IDENTITY_INVALID", field="expected_runners")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[int, str]] = set()
    for value in values:
        if type(value) is not dict or set(value) != {
            "box_number",
            "dog_name",
            "identity",
        }:
            raise ProtocolRejected("IDENTITY_INVALID", field="expected_runners")
        box = value["box_number"]
        if type(box) is not int or not 1 <= box <= 10:
            raise ProtocolRejected(
                "IDENTITY_INVALID", field="expected_runners.box_number"
            )
        name = _known_text(value["dog_name"], "expected_runners.dog_name")
        identity = _known_text(value["identity"], "expected_runners.identity")
        if identity != identity.upper():
            raise ProtocolRejected(
                "IDENTITY_INVALID", field="expected_runners.identity"
            )
        key = (box, identity)
        if key in keys:
            raise ProtocolRejected("IDENTITY_INVALID", field="expected_runners")
        keys.add(key)
        rows.append({"box_number": box, "dog_name": name, "identity": identity})
    return sorted(rows, key=lambda row: (row["box_number"], row["identity"]))


def runner_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str | None:
    normalized = _runner_rows(list(rows))
    if not normalized:
        return None
    identities = sorted(f"{row['box_number']}:{row['identity']}" for row in normalized)
    return sha256_bytes(canonical_bytes(identities))


def _validate_collector_source_report(
    raw: bytes,
    *,
    race_id: str,
    collector_run_id: str,
    capture_attempt_sha256: str,
    append_report_sha256: str,
) -> None:
    try:
        report = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolRejected("RECORD_MALFORMED") from exc
    if (
        type(report) is not dict
        or canonical_bytes(report) != raw
        or report.get("schema_version") != "collector_exact_capture_source_v1"
        or report.get("race_id") != race_id
        or report.get("collector_run_id") != collector_run_id
    ):
        raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
    attempts = report.get("attempts")
    matches = [
        attempt
        for attempt in attempts or []
        if isinstance(attempt, Mapping)
        and attempt.get("race_id") == race_id
        and attempt.get("status") == "APPENDED"
    ]
    if len(matches) != 1:
        raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
    attempt = matches[0]
    append_report = attempt.get("append_report")
    if (
        sha256_bytes(canonical_bytes(attempt)) != capture_attempt_sha256
        or not isinstance(append_report, Mapping)
        or sha256_bytes(canonical_bytes(append_report)) != append_report_sha256
    ):
        raise ProtocolRejected("HASH_DRIFT", field="capture_attempt")


class ManualPredictionCollectorProtocol:
    """Atomic request lifecycle with one claim, attempt, response, and consume."""

    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()

    def request_path(self, request_id: str) -> Path:
        return self.root / "requests" / f"{request_id}.json"

    def claim_path(self, request_id: str) -> Path:
        return self.root / "claims" / f"{request_id}.json"

    def attempt_path(self, request_id: str) -> Path:
        return self.root / "attempts" / f"{request_id}.json"

    def response_path(self, request_id: str) -> Path:
        return self.root / "responses" / f"{request_id}.json"

    def receipt_path(self, request_id: str) -> Path:
        return self.root / "receipts" / f"{request_id}.json"

    def consumed_path(self, request_id: str) -> Path:
        return self.root / "consumed" / f"{request_id}.json"

    def exact_receipt_directory(self, race_id: str) -> Path:
        name = hashlib.sha256(race_id.encode()).hexdigest()
        return self.root / "exact-receipts" / name

    def exact_receipt_path(self, race_id: str, request_id: str) -> Path:
        return self.exact_receipt_directory(race_id) / f"{request_id}.json"

    def collector_exact_receipt_directory(self, race_id: str) -> Path:
        name = hashlib.sha256(race_id.encode()).hexdigest()
        return self.root / "collector-exact-receipts" / name

    def collector_exact_receipt_path(
        self, race_id: str, capture_attempt_sha256: str
    ) -> Path:
        return (
            self.collector_exact_receipt_directory(race_id)
            / f"{capture_attempt_sha256}.json"
        )

    def _safe_directory(self, path: Path) -> None:
        try:
            if self.root.is_symlink():
                raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not path.is_relative_to(self.root):
                raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
            current = self.root
            for part in path.parent.relative_to(self.root).parts:
                current /= part
                if current.is_symlink():
                    raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
                current.mkdir(exist_ok=True, mode=0o700)
        except OSError as exc:
            raise ProtocolRejected("PROTOCOL_PATH_UNSAFE") from exc

    def _publish_once(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        duplicate_code: str,
        exact_recovery: bool = False,
    ) -> tuple[bytes, bool]:
        content = canonical_bytes(dict(payload))
        self._safe_directory(path)
        if path.exists() or path.is_symlink():
            if (
                exact_recovery
                and not path.is_symlink()
                and path.is_file()
                and path.read_bytes() == content
            ):
                return content, False
            raise ProtocolRejected(duplicate_code)
        descriptor, name = tempfile.mkstemp(prefix=".incoming-", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ProtocolRejected(duplicate_code) from exc
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return content, True

    @contextmanager
    def _admission_lock(self):
        path = self.root / "coordination" / "request-admission.lock"
        self._safe_directory(path)
        if path.is_symlink():
            raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            current = path.stat(follow_symlinks=False)
            if (
                path.is_symlink()
                or not path.is_file()
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _load_object(
        self, path: Path, missing_code: str
    ) -> tuple[dict[str, Any], bytes]:
        self._safe_directory(path)
        if path.is_symlink() or not path.is_file():
            raise ProtocolRejected(missing_code)
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProtocolRejected("RECORD_MALFORMED") from exc
        if type(value) is not dict or canonical_bytes(value) != raw:
            raise ProtocolRejected("RECORD_MALFORMED")
        return value, raw

    @staticmethod
    def _request_id(value: Any) -> str:
        text = _known_text(value, "request_id")
        try:
            parsed = uuid.UUID(text)
        except ValueError as exc:
            raise ProtocolRejected("REQUEST_ID_INVALID") from exc
        if str(parsed) != text or parsed.version != 4:
            raise ProtocolRejected("REQUEST_ID_INVALID")
        return text

    def _validate_request(self, value: Any) -> dict[str, Any]:
        if type(value) is not dict or set(value) != _REQUEST_KEYS:
            raise ProtocolRejected("REQUEST_MALFORMED")
        if value["schema_version"] != REQUEST_SCHEMA:
            raise ProtocolRejected("REQUEST_MALFORMED")
        request_id = self._request_id(value["request_id"])
        race = _race(value["race"])
        created, created_text = _timestamp(value["created_at"], "created_at")
        expires, expires_text = _timestamp(value["expires_at"], "expires_at")
        jump, _ = _timestamp(race["jump_timestamp"], "race.jump_timestamp")
        if not created < expires or not created < jump:
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        if (
            value["research_only"] is not True
            or value["attempt_authority"] != "one_attempt"
        ):
            raise ProtocolRejected("AUTHORITY_INVALID")
        requested = value["requested_output"]
        expected_requested = {
            "response_schema": RESPONSE_SCHEMA,
            "receipt_schema": RECEIPT_SCHEMA,
            "terminal_statuses": sorted(TERMINAL_STATUSES),
        }
        if requested != expected_requested:
            raise ProtocolRejected("OUTPUT_CONTRACT_INVALID")
        runners = _runner_rows(value["expected_runners"])
        expected_hash = runner_set_sha256(runners)
        if value["expected_runner_set_sha256"] != expected_hash:
            raise ProtocolRejected("IDENTITY_MISMATCH")
        return {
            "schema_version": REQUEST_SCHEMA,
            "request_id": request_id,
            "race": race,
            "created_at": created_text,
            "expires_at": expires_text,
            "research_only": True,
            "attempt_authority": "one_attempt",
            "requested_output": expected_requested,
            "expected_runners": runners,
            "expected_runner_set_sha256": expected_hash,
        }

    def _read_request(self, request_id: str) -> tuple[dict[str, Any], bytes]:
        value, raw = self._load_object(
            self.request_path(self._request_id(request_id)),
            "REQUEST_NOT_FOUND",
        )
        request = self._validate_request(value)
        if request["request_id"] != request_id:
            raise ProtocolRejected("IDENTITY_MISMATCH")
        return request, raw

    def publish_request(
        self,
        *,
        race: Mapping[str, Any],
        expected_runners: Sequence[Mapping[str, Any]] = (),
        created_at: datetime,
        expires_at: datetime,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        created, created_text = _timestamp(created_at, "created_at")
        _, expires_text = _timestamp(expires_at, "expires_at")
        resolved_request_id = request_id or str(uuid.uuid4())
        self._request_id(resolved_request_id)
        with self._admission_lock():
            if self.request_path(resolved_request_id).exists():
                raise ProtocolRejected("REPLAYED_REQUEST")
            for path in sorted((self.root / "requests").glob("*.json")):
                existing, _ = self._load_object(path, "REQUEST_NOT_FOUND")
                validated = self._validate_request(existing)
                response = self.response_path(validated["request_id"])
                expires, _ = _timestamp(validated["expires_at"], "expires_at")
                if not response.exists() and created < expires:
                    raise ProtocolRejected(
                        "ACTIVE_REQUEST_EXISTS",
                        request_id=validated["request_id"],
                    )
            runners = _runner_rows(list(expected_runners))
            payload = self._validate_request(
                {
                    "schema_version": REQUEST_SCHEMA,
                    "request_id": resolved_request_id,
                    "race": dict(race),
                    "created_at": created_text,
                    "expires_at": expires_text,
                    "research_only": True,
                    "attempt_authority": "one_attempt",
                    "requested_output": {
                        "response_schema": RESPONSE_SCHEMA,
                        "receipt_schema": RECEIPT_SCHEMA,
                        "terminal_statuses": sorted(TERMINAL_STATUSES),
                    },
                    "expected_runners": runners,
                    "expected_runner_set_sha256": runner_set_sha256(runners),
                }
            )
            raw, _ = self._publish_once(
                self.request_path(payload["request_id"]),
                payload,
                duplicate_code="REPLAYED_REQUEST",
            )
            return {**payload, "request_sha256": sha256_bytes(raw)}

    def _validate_claim(
        self, value: Any, *, request: Mapping[str, Any], request_raw: bytes
    ) -> dict[str, Any]:
        expected_keys = {
            "schema_version",
            "request_id",
            "request_sha256",
            "collector_run_id",
            "claimed_at",
            "safe_boundary",
        }
        if type(value) is not dict or set(value) != expected_keys:
            raise ProtocolRejected("CLAIM_MALFORMED")
        if (
            value["schema_version"] != CLAIM_SCHEMA
            or value["request_id"] != request["request_id"]
            or value["request_sha256"] != sha256_bytes(request_raw)
            or value["safe_boundary"] is not True
        ):
            raise ProtocolRejected("HASH_DRIFT")
        _known_text(value["collector_run_id"], "collector_run_id")
        _timestamp(value["claimed_at"], "claimed_at")
        return dict(value)

    def _validate_attempt(
        self,
        value: Any,
        *,
        context: CollectorRequest,
    ) -> dict[str, Any]:
        expected_keys = {
            "schema_version",
            "request_id",
            "request_sha256",
            "claim_sha256",
            "collector_run_id",
            "started_at",
            "attempt_number",
        }
        if type(value) is not dict or set(value) != expected_keys:
            raise ProtocolRejected("ATTEMPT_MALFORMED")
        if (
            value["schema_version"] != ATTEMPT_SCHEMA
            or value["request_id"] != context.request["request_id"]
            or value["request_sha256"] != context.request_sha256
            or value["claim_sha256"] != context.claim_sha256
            or value["attempt_number"] != 1
        ):
            raise ProtocolRejected("HASH_DRIFT")
        _known_text(value["collector_run_id"], "collector_run_id")
        started, _ = _timestamp(value["started_at"], "started_at")
        claimed, _ = _timestamp(context.claim["claimed_at"], "claimed_at")
        expires, _ = _timestamp(context.request["expires_at"], "expires_at")
        jump, _ = _timestamp(
            context.request["race"]["jump_timestamp"],
            "race.jump_timestamp",
        )
        if started < claimed or started >= expires or started >= jump:
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        return dict(value)

    def _context(self, request_id: str, *, recovered: bool) -> CollectorRequest:
        request, request_raw = self._read_request(request_id)
        claim_value, claim_raw = self._load_object(
            self.claim_path(request_id),
            "CLAIM_NOT_FOUND",
        )
        claim = self._validate_claim(
            claim_value,
            request=request,
            request_raw=request_raw,
        )
        return CollectorRequest(
            request=request,
            claim=claim,
            request_sha256=sha256_bytes(request_raw),
            claim_sha256=sha256_bytes(claim_raw),
            recovered=recovered,
        )

    def claimed_request(self, request_id: str) -> CollectorRequest:
        """Load one existing claim without creating or recovering a transition."""

        return self._context(self._request_id(request_id), recovered=True)

    def claim_request(
        self,
        request_id: str,
        *,
        now: datetime,
        collector_run_id: str,
    ) -> CollectorRequest:
        request, request_raw = self._read_request(request_id)
        _, claimed_at = _timestamp(now, "claimed_at")
        claim = {
            "schema_version": CLAIM_SCHEMA,
            "request_id": request_id,
            "request_sha256": sha256_bytes(request_raw),
            "collector_run_id": _known_text(collector_run_id, "collector_run_id"),
            "claimed_at": claimed_at,
            "safe_boundary": True,
        }
        self._publish_once(
            self.claim_path(request_id),
            claim,
            duplicate_code="DUPLICATE_CLAIM",
        )
        return self._context(request["request_id"], recovered=False)

    def _outstanding_claims(self) -> list[str]:
        output: list[str] = []
        for path in sorted((self.root / "claims").glob("*.json")):
            request_id = path.stem
            if not self.response_path(request_id).exists():
                output.append(request_id)
        return output

    def outstanding_request_ids(self) -> list[str]:
        output: list[str] = []
        for path in sorted((self.root / "requests").glob("*.json")):
            request_id = path.stem
            self._read_request(request_id)
            if not self.response_path(request_id).exists():
                output.append(request_id)
        return output

    def prepare_collector_request(
        self,
        *,
        now: datetime,
        collector_run_id: str,
        active_capture: bool,
    ) -> CollectorRequest | None:
        if active_capture:
            return None
        now_value, _ = _timestamp(now, "collector_boundary_at")
        outstanding = self._outstanding_claims()
        if len(outstanding) > 1:
            raise ProtocolRejected("MULTIPLE_OUTSTANDING_CLAIMS")
        if outstanding:
            context = self._context(outstanding[0], recovered=True)
            if self.attempt_path(context.request["request_id"]).exists():
                attempt, _ = self._load_object(
                    self.attempt_path(context.request["request_id"]),
                    "ATTEMPT_NOT_FOUND",
                )
                self._validate_attempt(attempt, context=context)
                self.publish_terminal(
                    context,
                    status="CAPTURE_FAILED",
                    now=now_value,
                    reason="collector_recovered_after_started_attempt",
                )
                return None
            return self._terminal_or_context(context, now_value)

        requests: list[tuple[datetime, str]] = []
        for path in sorted((self.root / "requests").glob("*.json")):
            request, _ = self._read_request(path.stem)
            if self.response_path(request["request_id"]).exists():
                continue
            created, _ = _timestamp(request["created_at"], "created_at")
            requests.append((created, request["request_id"]))
        for _, request_id in sorted(requests):
            context = self.claim_request(
                request_id,
                now=now_value,
                collector_run_id=collector_run_id,
            )
            terminal = self._terminal_or_context(context, now_value)
            if terminal is not None:
                return terminal
        return None

    def _terminal_or_context(
        self, context: CollectorRequest, now: datetime
    ) -> CollectorRequest | None:
        created, _ = _timestamp(context.request["created_at"], "created_at")
        expires, _ = _timestamp(context.request["expires_at"], "expires_at")
        jump, _ = _timestamp(
            context.request["race"]["jump_timestamp"],
            "race.jump_timestamp",
        )
        if now < created:
            self.publish_terminal(
                context,
                status="CAPTURE_FAILED",
                now=now,
                reason="request_created_at_is_in_future",
            )
            return None
        if now >= expires:
            self.publish_terminal(
                context,
                status="REQUEST_EXPIRED",
                now=now,
                reason="request_expired_before_collector_attempt",
            )
            return None
        if now >= jump:
            self.publish_terminal(
                context,
                status="CAPTURE_WINDOW_CLOSED",
                now=now,
                reason="race_jump_reached_before_collector_attempt",
            )
            return None
        return context

    def begin_attempt(
        self,
        context: CollectorRequest,
        *,
        now: datetime,
        collector_run_id: str,
    ) -> dict[str, Any]:
        if self.response_path(context.request["request_id"]).exists():
            raise ProtocolRejected("RESPONSE_ALREADY_TERMINAL")
        now_value, started_at = _timestamp(now, "attempt_started_at")
        if self._terminal_or_context(context, now_value) is None:
            raise ProtocolRejected("CAPTURE_WINDOW_CLOSED")
        payload = {
            "schema_version": ATTEMPT_SCHEMA,
            "request_id": context.request["request_id"],
            "request_sha256": context.request_sha256,
            "claim_sha256": context.claim_sha256,
            "collector_run_id": _known_text(collector_run_id, "collector_run_id"),
            "started_at": started_at,
            "attempt_number": 1,
        }
        self._validate_attempt(payload, context=context)
        self._publish_once(
            self.attempt_path(context.request["request_id"]),
            payload,
            duplicate_code="DUPLICATE_ATTEMPT",
        )
        return payload

    def _response_payload(
        self,
        context: CollectorRequest,
        *,
        status: str,
        now: datetime,
        reason: str | None,
        receipt: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if status not in TERMINAL_STATUSES:
            raise ProtocolRejected("STATUS_UNKNOWN")
        if (status == RECEIPT_READY) != (receipt is not None):
            raise ProtocolRejected("RESPONSE_MALFORMED")
        attempt_sha = None
        attempt_path = self.attempt_path(context.request["request_id"])
        if attempt_path.exists():
            attempt, attempt_raw = self._load_object(attempt_path, "ATTEMPT_NOT_FOUND")
            self._validate_attempt(attempt, context=context)
            attempt_sha = sha256_bytes(attempt_raw)
        _, responded_at = _timestamp(now, "responded_at")
        return {
            "schema_version": RESPONSE_SCHEMA,
            "request_id": context.request["request_id"],
            "request_sha256": context.request_sha256,
            "claim_sha256": context.claim_sha256,
            "attempt_sha256": attempt_sha,
            "status": status,
            "responded_at": responded_at,
            "race": dict(context.request["race"]),
            "reason": reason,
            "receipt": dict(receipt) if receipt is not None else None,
        }

    def publish_terminal(
        self,
        context: CollectorRequest,
        *,
        status: str,
        now: datetime,
        reason: str,
    ) -> dict[str, Any]:
        if status == RECEIPT_READY:
            raise ProtocolRejected("RESPONSE_MALFORMED")
        now_value, _ = _timestamp(now, "responded_at")
        expires, _ = _timestamp(context.request["expires_at"], "expires_at")
        jump, _ = _timestamp(
            context.request["race"]["jump_timestamp"],
            "race.jump_timestamp",
        )
        if now_value >= expires:
            status = "REQUEST_EXPIRED"
            reason = "request_expired_before_terminal_response"
        elif now_value >= jump:
            status = "CAPTURE_WINDOW_CLOSED"
            reason = "race_jump_reached_before_terminal_response"
        payload = self._response_payload(
            context,
            status=status,
            now=now_value,
            reason=_known_text(reason, "reason"),
            receipt=None,
        )
        self._publish_once(
            self.response_path(context.request["request_id"]),
            payload,
            duplicate_code="DUPLICATE_RESPONSE",
        )
        return payload

    def publish_receipt_ready(
        self,
        context: CollectorRequest,
        *,
        now: datetime,
        handoff: Mapping[str, Any],
        normalized_receipt: Mapping[str, Any],
    ) -> dict[str, Any]:
        now_value, _ = _timestamp(now, "receipt_emitted_at")
        if self._terminal_or_context(context, now_value) is None:
            raise ProtocolRejected("CAPTURE_WINDOW_CLOSED")
        public_handoff = {
            str(key): value
            for key, value in handoff.items()
            if not str(key).startswith("_")
        }
        handoff_schema = public_handoff.get("schema_version")
        if (
            handoff_schema
            not in {
                "on_demand_verified_master_packet_v1",
                "on_demand_verified_collector_capture_v2",
            }
            or public_handoff.get("race_id") != context.request["race"]["race_id"]
            or normalized_receipt.get("schema_version") != "on_demand_odds_receipt_v1"
            or normalized_receipt.get("race_id") != context.request["race"]["race_id"]
        ):
            raise ProtocolRejected("IDENTITY_MISMATCH")
        source_hashes = {
            key: _hash(public_handoff.get(key), key)
            for key in (
                "source_report_sha256",
                "source_form_sha256",
                "source_sidecar_sha256",
            )
        }
        for label, hash_key in (
            ("report", "source_report_sha256"),
            ("form", "source_form_sha256"),
            ("sidecar", "source_sidecar_sha256"),
        ):
            raw = handoff.get(f"_{label}_bytes")
            if (
                not isinstance(raw, bytes)
                or sha256_bytes(raw) != source_hashes[hash_key]
            ):
                raise ProtocolRejected("HASH_DRIFT", field=hash_key)
        if handoff_schema == "on_demand_verified_master_packet_v1":
            packet_hashes = {
                key: _hash(public_handoff.get(key), key)
                for key in (
                    "packet_record_checksum_sha256",
                    "packet_effective_state_sha256",
                )
            }
            source_evidence = {
                "source_url": normalized_receipt.get("source_url"),
                **source_hashes,
                "packet_record_schema_version": public_handoff.get(
                    "packet_record_schema_version"
                ),
                "packet_record_checksum_sha256": packet_hashes[
                    "packet_record_checksum_sha256"
                ],
                "packet_effective_state_schema_version": public_handoff.get(
                    "packet_effective_state_schema_version"
                ),
                "packet_effective_state_sha256": packet_hashes[
                    "packet_effective_state_sha256"
                ],
            }
        else:
            source_evidence = {
                "source_url": normalized_receipt.get("source_url"),
                **source_hashes,
                "capture_attempt_sha256": _hash(
                    public_handoff.get("capture_attempt_sha256"),
                    "capture_attempt_sha256",
                ),
                "append_report_sha256": _hash(
                    public_handoff.get("append_report_sha256"),
                    "append_report_sha256",
                ),
            }
        actual_runner_hash = _hash(
            normalized_receipt.get("runner_set_sha256"),
            "runner_set_sha256",
        )
        win_market = (normalized_receipt.get("markets") or {}).get("win")
        if not isinstance(win_market, list):
            raise ProtocolRejected("IDENTITY_MISMATCH")
        actual_runners = _runner_rows(
            [
                {
                    "box_number": row.get("box_number"),
                    "dog_name": row.get("dog_name"),
                    "identity": row.get("identity"),
                }
                for row in win_market
                if isinstance(row, Mapping)
            ]
        )
        if runner_set_sha256(actual_runners) != actual_runner_hash:
            raise ProtocolRejected("IDENTITY_MISMATCH")
        expected_runner_hash = context.request["expected_runner_set_sha256"]
        if (
            expected_runner_hash is not None
            and expected_runner_hash != actual_runner_hash
        ):
            raise ProtocolRejected("IDENTITY_MISMATCH")
        _, emitted_at = _timestamp(now_value, "receipt_emitted_at")
        _, captured_at = _timestamp(
            normalized_receipt.get("captured_at"),
            "receipt.captured_at",
        )
        receipt_payload = {
            "schema_version": RECEIPT_SCHEMA,
            "request_id": context.request["request_id"],
            "request_sha256": context.request_sha256,
            "race": dict(context.request["race"]),
            "runners": actual_runners,
            "runner_set_sha256": actual_runner_hash,
            "captured_at": captured_at,
            "emitted_at": emitted_at,
            "source_evidence": source_evidence,
            "sealed_handoff": public_handoff,
        }
        receipt_raw, _ = self._publish_once(
            self.receipt_path(context.request["request_id"]),
            receipt_payload,
            duplicate_code="DUPLICATE_RECEIPT",
            exact_recovery=True,
        )
        receipt_reference = {
            "schema_version": RECEIPT_SCHEMA,
            "path": self.receipt_path(context.request["request_id"])
            .relative_to(self.root)
            .as_posix(),
            "sha256": sha256_bytes(receipt_raw),
        }
        if handoff_schema == "on_demand_verified_collector_capture_v2":
            artifacts: dict[str, Any] = {}
            evidence_root = self.root.parent.resolve()
            for label, hash_key in (
                ("report", "source_report_sha256"),
                ("form", "source_form_sha256"),
                ("sidecar", "source_sidecar_sha256"),
            ):
                path_value = handoff.get(f"_{label}_path")
                path = Path(path_value) if isinstance(path_value, (str, Path)) else None
                if (
                    path is None
                    or path.is_symlink()
                    or not path.is_file()
                    or not path.is_absolute()
                ):
                    raise ProtocolRejected("SOURCE_FILE_UNSAFE", field=label)
                resolved = path.resolve()
                try:
                    relative = resolved.relative_to(evidence_root)
                except ValueError as exc:
                    raise ProtocolRejected(
                        "SOURCE_FILE_UNSAFE", field=label
                    ) from exc
                raw = path.read_bytes()
                if sha256_bytes(raw) != source_hashes[hash_key]:
                    raise ProtocolRejected("HASH_DRIFT", field=hash_key)
                artifacts[label] = {
                    "path": relative.as_posix(),
                    "sha256": source_hashes[hash_key],
                }
            exact_payload = {
                "schema_version": EXACT_RECEIPT_SCHEMA,
                "race_id": context.request["race"]["race_id"],
                "request_id": context.request["request_id"],
                "receipt": receipt_reference,
                "artifacts": artifacts,
                "form_name": _known_text(handoff.get("_form_name"), "form_name"),
            }
            self._publish_once(
                self.exact_receipt_path(
                    context.request["race"]["race_id"],
                    context.request["request_id"],
                ),
                exact_payload,
                duplicate_code="DUPLICATE_EXACT_RECEIPT",
                exact_recovery=True,
            )
        response = self._response_payload(
            context,
            status=RECEIPT_READY,
            now=now,
            reason=None,
            receipt=receipt_reference,
        )
        self._publish_once(
            self.response_path(context.request["request_id"]),
            response,
            duplicate_code="DUPLICATE_RESPONSE",
        )
        return response

    def publish_collector_exact_receipt(
        self,
        *,
        collector_run_id: str,
        emitted_at: datetime,
        handoff: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Publish one scheduled collector capture for bounded exact reuse."""

        public_handoff = {
            str(key): value
            for key, value in handoff.items()
            if not str(key).startswith("_")
        }
        required_handoff = {
            "schema_version",
            "race_id",
            "race",
            "append_timestamp",
            "runner_set_sha256",
            "source_report_sha256",
            "source_form_sha256",
            "source_sidecar_sha256",
            "capture_attempt_sha256",
            "append_report_sha256",
        }
        if (
            set(public_handoff) != required_handoff
            or public_handoff.get("schema_version")
            != "on_demand_verified_collector_capture_v2"
        ):
            raise ProtocolRejected("RECEIPT_MALFORMED")
        race = _race(public_handoff["race"])
        race_id = _known_text(public_handoff["race_id"], "race_id")
        if race["race_id"] != race_id:
            raise ProtocolRejected("IDENTITY_MISMATCH")
        captured, captured_text = _timestamp(
            public_handoff["append_timestamp"], "append_timestamp"
        )
        jump, _ = _timestamp(race["jump_timestamp"], "race.jump_timestamp")
        emitted, emitted_text = _timestamp(emitted_at, "emitted_at")
        if captured >= jump or emitted < captured:
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        capture_attempt_sha = _hash(
            public_handoff["capture_attempt_sha256"],
            "capture_attempt_sha256",
        )
        collector_run = _known_text(collector_run_id, "collector_run_id")
        for key in (
            "runner_set_sha256",
            "source_report_sha256",
            "source_form_sha256",
            "source_sidecar_sha256",
            "append_report_sha256",
        ):
            _hash(public_handoff[key], key)

        evidence_root = self.root.parent.resolve()
        artifacts: dict[str, Any] = {}
        for label, hash_key in (
            ("report", "source_report_sha256"),
            ("form", "source_form_sha256"),
            ("sidecar", "source_sidecar_sha256"),
        ):
            path_value = handoff.get(f"_{label}_path")
            path = Path(path_value) if isinstance(path_value, (str, Path)) else None
            if (
                path is None
                or path.is_symlink()
                or not path.is_file()
                or not path.is_absolute()
            ):
                raise ProtocolRejected("SOURCE_FILE_UNSAFE", field=label)
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(evidence_root)
            except ValueError as exc:
                raise ProtocolRejected("SOURCE_FILE_UNSAFE", field=label) from exc
            raw = path.read_bytes()
            expected_hash = _hash(public_handoff[hash_key], hash_key)
            if sha256_bytes(raw) != expected_hash:
                raise ProtocolRejected("HASH_DRIFT", field=hash_key)
            artifacts[label] = {
                "path": relative.as_posix(),
                "sha256": expected_hash,
            }
        _validate_collector_source_report(
            Path(handoff["_report_path"]).read_bytes(),
            race_id=race_id,
            collector_run_id=collector_run,
            capture_attempt_sha256=capture_attempt_sha,
            append_report_sha256=public_handoff["append_report_sha256"],
        )

        payload = {
            "schema_version": COLLECTOR_EXACT_RECEIPT_SCHEMA,
            "race_id": race_id,
            "collector_run_id": collector_run,
            "captured_at": captured_text,
            "emitted_at": emitted_text,
            "sealed_handoff": public_handoff,
            "artifacts": artifacts,
            "form_name": _known_text(handoff.get("_form_name"), "form_name"),
        }
        self._publish_once(
            self.collector_exact_receipt_path(race_id, capture_attempt_sha),
            payload,
            duplicate_code="DUPLICATE_COLLECTOR_EXACT_RECEIPT",
            exact_recovery=True,
        )
        return payload

    def discover_collector_exact_handoff(
        self,
        *,
        race_id: str,
        current_time: datetime,
        max_age_seconds: int,
    ) -> dict[str, Any] | None:
        """Load one exact scheduled capture without scanning other races."""

        if max_age_seconds <= 0:
            raise ProtocolRejected("RECEIPT_MAX_AGE_INVALID")
        now, _ = _timestamp(current_time, "current_time")
        race_id = _known_text(race_id, "race_id")
        directory = self.collector_exact_receipt_directory(race_id)
        if not directory.exists():
            return None
        if directory.is_symlink() or not directory.is_dir():
            raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
        paths = sorted(directory.glob("*.json"), reverse=True)
        if len(paths) > 32:
            raise ProtocolRejected("EXACT_RECEIPT_INDEX_UNBOUNDED")

        evidence_root = self.root.parent.resolve()
        candidates: list[tuple[datetime, dict[str, Any]]] = []
        for path in paths:
            value, _ = self._load_object(
                path, "COLLECTOR_EXACT_RECEIPT_NOT_FOUND"
            )
            expected_keys = {
                "schema_version",
                "race_id",
                "collector_run_id",
                "captured_at",
                "emitted_at",
                "sealed_handoff",
                "artifacts",
                "form_name",
            }
            if (
                set(value) != expected_keys
                or value.get("schema_version")
                != COLLECTOR_EXACT_RECEIPT_SCHEMA
                or value.get("race_id") != race_id
            ):
                raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
            _known_text(value["collector_run_id"], "collector_run_id")
            captured, _ = _timestamp(value["captured_at"], "captured_at")
            emitted, _ = _timestamp(value["emitted_at"], "emitted_at")
            age = (now - captured).total_seconds()
            if emitted < captured or emitted > now or age < 0:
                raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
            if age > max_age_seconds:
                continue
            handoff = value.get("sealed_handoff")
            required_handoff = {
                "schema_version",
                "race_id",
                "race",
                "append_timestamp",
                "runner_set_sha256",
                "source_report_sha256",
                "source_form_sha256",
                "source_sidecar_sha256",
                "capture_attempt_sha256",
                "append_report_sha256",
            }
            if (
                type(handoff) is not dict
                or set(handoff) != required_handoff
                or handoff.get("schema_version")
                != "on_demand_verified_collector_capture_v2"
                or handoff.get("race_id") != race_id
                or handoff.get("append_timestamp") != value["captured_at"]
            ):
                raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
            race = _race(handoff["race"])
            jump, _ = _timestamp(
                race["jump_timestamp"], "race.jump_timestamp"
            )
            if race["race_id"] != race_id or captured >= jump:
                raise ProtocolRejected("IDENTITY_MISMATCH")
            capture_attempt_sha = _hash(
                handoff.get("capture_attempt_sha256"),
                "capture_attempt_sha256",
            )
            for key in (
                "runner_set_sha256",
                "source_report_sha256",
                "source_form_sha256",
                "source_sidecar_sha256",
                "append_report_sha256",
            ):
                _hash(handoff[key], key)
            if path != self.collector_exact_receipt_path(
                race_id, capture_attempt_sha
            ):
                raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")

            artifacts = value.get("artifacts")
            if type(artifacts) is not dict or set(artifacts) != {
                "report",
                "form",
                "sidecar",
            }:
                raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
            raw_artifacts: dict[str, bytes] = {}
            artifact_paths: dict[str, Path] = {}
            for label, source_key in (
                ("report", "source_report_sha256"),
                ("form", "source_form_sha256"),
                ("sidecar", "source_sidecar_sha256"),
            ):
                artifact = artifacts[label]
                if type(artifact) is not dict or set(artifact) != {
                    "path",
                    "sha256",
                }:
                    raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
                relative = Path(
                    _known_text(artifact["path"], f"artifacts.{label}.path")
                )
                if relative.is_absolute():
                    raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
                artifact_path = evidence_root / relative
                if artifact_path.is_symlink() or not artifact_path.is_file():
                    raise ProtocolRejected("SOURCE_FILE_UNSAFE", field=label)
                resolved = artifact_path.resolve()
                try:
                    resolved.relative_to(evidence_root)
                except ValueError as exc:
                    raise ProtocolRejected("PROTOCOL_PATH_UNSAFE") from exc
                raw = artifact_path.read_bytes()
                expected_hash = _hash(
                    artifact["sha256"], f"artifacts.{label}.sha256"
                )
                if (
                    sha256_bytes(raw) != expected_hash
                    or handoff.get(source_key) != expected_hash
                ):
                    raise ProtocolRejected("HASH_DRIFT", field=label)
                raw_artifacts[label] = raw
                artifact_paths[label] = resolved
            _validate_collector_source_report(
                raw_artifacts["report"],
                race_id=race_id,
                collector_run_id=value["collector_run_id"],
                capture_attempt_sha256=capture_attempt_sha,
                append_report_sha256=handoff["append_report_sha256"],
            )
            candidates.append(
                (
                    captured,
                    {
                        **handoff,
                        "_report_bytes": raw_artifacts["report"],
                        "_form_bytes": raw_artifacts["form"],
                        "_sidecar_bytes": raw_artifacts["sidecar"],
                        "_report_path": artifact_paths["report"],
                        "_form_path": artifact_paths["form"],
                        "_sidecar_path": artifact_paths["sidecar"],
                        "_form_name": _known_text(
                            value["form_name"], "form_name"
                        ),
                    },
                )
            )
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _validate_response(
        self,
        value: Any,
        *,
        context: CollectorRequest,
        request_raw: bytes,
        claim_raw: bytes,
    ) -> dict[str, Any]:
        request = context.request
        expected_keys = {
            "schema_version",
            "request_id",
            "request_sha256",
            "claim_sha256",
            "attempt_sha256",
            "status",
            "responded_at",
            "race",
            "reason",
            "receipt",
        }
        if type(value) is not dict or set(value) != expected_keys:
            raise ProtocolRejected("RESPONSE_MALFORMED")
        if value["status"] not in TERMINAL_STATUSES:
            raise ProtocolRejected("STATUS_UNKNOWN")
        if (
            value["schema_version"] != RESPONSE_SCHEMA
            or value["request_id"] != request["request_id"]
            or value["request_sha256"] != sha256_bytes(request_raw)
            or value["claim_sha256"] != sha256_bytes(claim_raw)
            or value["race"] != request["race"]
        ):
            raise ProtocolRejected("HASH_DRIFT")
        responded, _ = _timestamp(value["responded_at"], "responded_at")
        claimed, _ = _timestamp(context.claim["claimed_at"], "claimed_at")
        expires, _ = _timestamp(request["expires_at"], "expires_at")
        jump, _ = _timestamp(
            request["race"]["jump_timestamp"],
            "race.jump_timestamp",
        )
        if responded < claimed:
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        if responded >= expires and value["status"] != "REQUEST_EXPIRED":
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        if (
            responded < expires
            and responded >= jump
            and value["status"] != "CAPTURE_WINDOW_CLOSED"
        ):
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        if (value["status"] == RECEIPT_READY) != (type(value["receipt"]) is dict):
            raise ProtocolRejected("RESPONSE_MALFORMED")
        if value["status"] == RECEIPT_READY and value["reason"] is not None:
            raise ProtocolRejected("RESPONSE_MALFORMED")
        if value["status"] != RECEIPT_READY:
            _known_text(value["reason"], "reason")
        attempt_path = self.attempt_path(request["request_id"])
        if value["attempt_sha256"] is not None:
            attempt, attempt_raw = self._load_object(attempt_path, "ATTEMPT_NOT_FOUND")
            self._validate_attempt(attempt, context=context)
            if value["attempt_sha256"] != sha256_bytes(attempt_raw):
                raise ProtocolRejected("HASH_DRIFT")
            started, _ = _timestamp(attempt["started_at"], "started_at")
            if responded < started:
                raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        elif attempt_path.exists():
            raise ProtocolRejected("HASH_DRIFT")
        elif value["status"] == RECEIPT_READY:
            raise ProtocolRejected("RESPONSE_MALFORMED")
        return dict(value)

    def read_response(self, request_id: str) -> dict[str, Any] | None:
        request, request_raw = self._read_request(request_id)
        response_path = self.response_path(request_id)
        if not response_path.exists():
            return None
        claim_value, claim_raw = self._load_object(
            self.claim_path(request_id), "CLAIM_NOT_FOUND"
        )
        claim = self._validate_claim(
            claim_value, request=request, request_raw=request_raw
        )
        context = CollectorRequest(
            request=request,
            claim=claim,
            request_sha256=sha256_bytes(request_raw),
            claim_sha256=sha256_bytes(claim_raw),
            recovered=True,
        )
        response, _ = self._load_object(response_path, "RESPONSE_NOT_FOUND")
        return self._validate_response(
            response,
            context=context,
            request_raw=request_raw,
            claim_raw=claim_raw,
        )

    def wait_for_response(
        self,
        request_id: str,
        *,
        timeout_seconds: float,
        poll_seconds: float,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any] | None:
        if timeout_seconds < 0 or poll_seconds <= 0:
            raise ProtocolRejected("WAIT_INVALID")
        deadline = monotonic() + timeout_seconds
        while True:
            response = self.read_response(request_id)
            if response is not None:
                return response
            remaining = deadline - monotonic()
            if remaining <= 0:
                return None
            sleep(min(poll_seconds, remaining))

    def _receipt_from_response(
        self, request: Mapping[str, Any], response: Mapping[str, Any]
    ) -> dict[str, Any]:
        reference = response.get("receipt")
        if type(reference) is not dict or set(reference) != {
            "schema_version",
            "path",
            "sha256",
        }:
            raise ProtocolRejected("RESPONSE_MALFORMED")
        if reference["schema_version"] != RECEIPT_SCHEMA:
            raise ProtocolRejected("RESPONSE_MALFORMED")
        relative = Path(_known_text(reference["path"], "receipt.path"))
        expected = self.receipt_path(request["request_id"])
        if relative.is_absolute() or self.root / relative != expected:
            raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
        receipt, receipt_raw = self._load_object(expected, "RECEIPT_NOT_FOUND")
        if _hash(reference["sha256"], "receipt.sha256") != sha256_bytes(receipt_raw):
            raise ProtocolRejected("HASH_DRIFT")
        receipt_keys = {
            "schema_version",
            "request_id",
            "request_sha256",
            "race",
            "runners",
            "runner_set_sha256",
            "captured_at",
            "emitted_at",
            "source_evidence",
            "sealed_handoff",
        }
        if (
            set(receipt) != receipt_keys
            or receipt.get("schema_version") != RECEIPT_SCHEMA
            or receipt.get("request_id") != request["request_id"]
            or receipt.get("request_sha256") != response["request_sha256"]
            or receipt.get("race") != request["race"]
        ):
            raise ProtocolRejected("IDENTITY_MISMATCH")
        runners = _runner_rows(receipt["runners"])
        runner_hash = _hash(receipt["runner_set_sha256"], "runner_set_sha256")
        if runner_set_sha256(runners) != runner_hash or (
            request["expected_runner_set_sha256"] is not None
            and request["expected_runner_set_sha256"] != runner_hash
        ):
            raise ProtocolRejected("IDENTITY_MISMATCH")
        captured, _ = _timestamp(receipt["captured_at"], "receipt.captured_at")
        emitted, emitted_text = _timestamp(receipt["emitted_at"], "receipt.emitted_at")
        responded, responded_text = _timestamp(response["responded_at"], "responded_at")
        jump, _ = _timestamp(
            request["race"]["jump_timestamp"],
            "race.jump_timestamp",
        )
        if (
            captured > emitted
            or captured >= jump
            or emitted_text != responded_text
            or emitted != responded
        ):
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        source = receipt["source_evidence"]
        sealed = receipt["sealed_handoff"]
        if type(sealed) is not dict:
            raise ProtocolRejected("RECEIPT_MALFORMED")
        if sealed.get("schema_version") == "on_demand_verified_master_packet_v1":
            source_keys = {
                "source_url",
                "source_report_sha256",
                "source_form_sha256",
                "source_sidecar_sha256",
                "packet_record_schema_version",
                "packet_record_checksum_sha256",
                "packet_effective_state_schema_version",
                "packet_effective_state_sha256",
            }
            text_keys = {
                "source_url",
                "packet_record_schema_version",
                "packet_effective_state_schema_version",
            }
        elif (
            sealed.get("schema_version")
            == "on_demand_verified_collector_capture_v2"
        ):
            source_keys = {
                "source_url",
                "source_report_sha256",
                "source_form_sha256",
                "source_sidecar_sha256",
                "capture_attempt_sha256",
                "append_report_sha256",
            }
            text_keys = {"source_url"}
        else:
            raise ProtocolRejected("RECEIPT_MALFORMED")
        if type(source) is not dict or set(source) != source_keys:
            raise ProtocolRejected("RECEIPT_MALFORMED")
        _known_text(source["source_url"], "source_evidence.source_url")
        for key in text_keys - {"source_url"}:
            _known_text(source[key], f"source_evidence.{key}")
        for key in source_keys - text_keys:
            _hash(source[key], f"source_evidence.{key}")
        if (
            sealed.get("race_id") != request["race"]["race_id"]
            or any(
                sealed.get(key) != source.get(key)
                for key in source_keys - {"source_url"}
            )
        ):
            raise ProtocolRejected("HASH_DRIFT")
        if (
            sealed.get("schema_version")
            == "on_demand_verified_collector_capture_v2"
            and (
                sealed.get("race") != request["race"]
                or sealed.get("runner_set_sha256") != runner_hash
                or sealed.get("append_timestamp") != receipt["captured_at"]
            )
        ):
            raise ProtocolRejected("IDENTITY_MISMATCH")
        return receipt

    def discover_exact_handoff(
        self,
        *,
        race_id: str,
        current_time: datetime,
        max_age_seconds: int,
    ) -> dict[str, Any] | None:
        """Load the newest exact valid sealed receipt without scanning other races."""

        if max_age_seconds <= 0:
            raise ProtocolRejected("RECEIPT_MAX_AGE_INVALID")
        now, _ = _timestamp(current_time, "current_time")
        directory = self.exact_receipt_directory(
            _known_text(race_id, "race_id")
        )
        if not directory.exists():
            return None
        if directory.is_symlink() or not directory.is_dir():
            raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
        paths = sorted(directory.glob("*.json"), reverse=True)
        if len(paths) > 32:
            raise ProtocolRejected("EXACT_RECEIPT_INDEX_UNBOUNDED")
        candidates: list[tuple[datetime, dict[str, Any]]] = []
        evidence_root = self.root.parent.resolve()
        for path in paths:
            index, _ = self._load_object(path, "EXACT_RECEIPT_NOT_FOUND")
            if (
                set(index)
                != {
                    "schema_version",
                    "race_id",
                    "request_id",
                    "receipt",
                    "artifacts",
                    "form_name",
                }
                or index.get("schema_version") != EXACT_RECEIPT_SCHEMA
                or index.get("race_id") != race_id
            ):
                raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
            request_id = self._request_id(index.get("request_id"))
            if path != self.exact_receipt_path(race_id, request_id):
                raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
            request, request_raw = self._read_request(request_id)
            if request["race"]["race_id"] != race_id:
                raise ProtocolRejected("IDENTITY_MISMATCH")
            reference = index.get("receipt")
            if type(reference) is not dict:
                raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
            relative = Path(
                _known_text(reference.get("path"), "receipt.path")
            )
            receipt_path = self.receipt_path(request_id)
            if relative.is_absolute() or self.root / relative != receipt_path:
                raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
            receipt_value, _ = self._load_object(
                receipt_path, "RECEIPT_NOT_FOUND"
            )
            synthetic_response = {
                "receipt": reference,
                "request_sha256": sha256_bytes(request_raw),
                "responded_at": receipt_value.get("emitted_at"),
            }
            receipt = self._receipt_from_response(request, synthetic_response)
            captured, _ = _timestamp(
                receipt["captured_at"], "receipt.captured_at"
            )
            age = (now - captured).total_seconds()
            if age < 0:
                raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
            if age > max_age_seconds:
                continue
            artifacts = index.get("artifacts")
            if type(artifacts) is not dict or set(artifacts) != {
                "report",
                "form",
                "sidecar",
            }:
                raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
            raw_artifacts: dict[str, bytes] = {}
            artifact_paths: dict[str, Path] = {}
            for label, source_key in (
                ("report", "source_report_sha256"),
                ("form", "source_form_sha256"),
                ("sidecar", "source_sidecar_sha256"),
            ):
                artifact = artifacts[label]
                if type(artifact) is not dict or set(artifact) != {
                    "path",
                    "sha256",
                }:
                    raise ProtocolRejected("EXACT_RECEIPT_MALFORMED")
                relative_artifact = Path(
                    _known_text(artifact["path"], f"artifacts.{label}.path")
                )
                if relative_artifact.is_absolute():
                    raise ProtocolRejected("PROTOCOL_PATH_UNSAFE")
                artifact_path = evidence_root / relative_artifact
                if artifact_path.is_symlink() or not artifact_path.is_file():
                    raise ProtocolRejected("SOURCE_FILE_UNSAFE", field=label)
                resolved = artifact_path.resolve()
                try:
                    resolved.relative_to(evidence_root)
                except ValueError as exc:
                    raise ProtocolRejected("PROTOCOL_PATH_UNSAFE") from exc
                raw = artifact_path.read_bytes()
                expected_hash = _hash(
                    artifact["sha256"], f"artifacts.{label}.sha256"
                )
                if (
                    sha256_bytes(raw) != expected_hash
                    or expected_hash
                    != receipt["source_evidence"][source_key]
                ):
                    raise ProtocolRejected("HASH_DRIFT", field=label)
                raw_artifacts[label] = raw
                artifact_paths[label] = resolved
            candidates.append(
                (
                    captured,
                    {
                        **receipt["sealed_handoff"],
                        "_report_bytes": raw_artifacts["report"],
                        "_form_bytes": raw_artifacts["form"],
                        "_sidecar_bytes": raw_artifacts["sidecar"],
                        "_report_path": artifact_paths["report"],
                        "_form_path": artifact_paths["form"],
                        "_sidecar_path": artifact_paths["sidecar"],
                        "_form_name": _known_text(
                            index["form_name"], "form_name"
                        ),
                    },
                )
            )
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def consume_response(self, request_id: str, *, now: datetime) -> dict[str, Any]:
        if self.consumed_path(request_id).exists():
            raise ProtocolRejected("RESPONSE_ALREADY_CONSUMED")
        response = self.read_response(request_id)
        if response is None:
            raise ProtocolRejected("RESPONSE_NOT_FOUND")
        request, _ = self._read_request(request_id)
        receipt = (
            self._receipt_from_response(request, response)
            if response["status"] == RECEIPT_READY
            else None
        )
        _, response_raw = self._load_object(
            self.response_path(request_id),
            "RESPONSE_NOT_FOUND",
        )
        consumed, consumed_at = _timestamp(now, "consumed_at")
        responded, _ = _timestamp(response["responded_at"], "responded_at")
        if consumed < responded:
            raise ProtocolRejected("TIMESTAMP_ORDER_INVALID")
        consume = {
            "schema_version": CONSUME_SCHEMA,
            "request_id": request_id,
            "response_sha256": sha256_bytes(response_raw),
            "status": response["status"],
            "consumed_at": consumed_at,
            "consume_once": True,
        }
        self._publish_once(
            self.consumed_path(request_id),
            consume,
            duplicate_code="RESPONSE_ALREADY_CONSUMED",
        )
        return {"response": response, "receipt": receipt, "consume": consume}

    @staticmethod
    def verify_ready_handoff(
        receipt: Mapping[str, Any],
        *,
        handoff: Mapping[str, Any],
        normalized_receipt: Mapping[str, Any],
    ) -> None:
        public_handoff = {
            str(key): value
            for key, value in handoff.items()
            if not str(key).startswith("_")
        }
        if (
            receipt.get("sealed_handoff") != public_handoff
            or receipt.get("runner_set_sha256")
            != normalized_receipt.get("runner_set_sha256")
            or receipt.get("race", {}).get("race_id") != handoff.get("race_id")
        ):
            raise ProtocolRejected("HASH_DRIFT")

    def prioritize_capture_plan(
        self,
        context: CollectorRequest,
        plan: Mapping[str, Any],
        *,
        now: datetime,
    ) -> dict[str, Any]:
        now_value, _ = _timestamp(now, "plan_prioritized_at")
        if self._terminal_or_context(context, now_value) is None:
            raise ProtocolRejected("CAPTURE_WINDOW_CLOSED")
        rows = [
            dict(row) for row in plan.get("races") or [] if isinstance(row, Mapping)
        ]
        expected = context.request["race"]

        def exact(row: Mapping[str, Any]) -> bool:
            try:
                jump, jump_text = _timestamp(
                    row.get("jump_datetime"),
                    "plan.jump_datetime",
                )
            except ProtocolRejected:
                return False
            del jump
            return (
                row.get("race_id") == expected["race_id"]
                and row.get("thedogs_source_url") == expected["url"]
                and row.get("venue") == expected["venue"]
                and row.get("race_number") == expected["race_number"]
                and row.get("race_date") == expected["race_date"]
                and jump_text == expected["jump_timestamp"]
                and (
                    not context.request["expected_runners"]
                    or runner_set_sha256(row.get("expected_runners") or [])
                    == context.request["expected_runner_set_sha256"]
                )
            )

        matches = [row for row in rows if exact(row)]
        related = [
            row
            for row in rows
            if row.get("race_id") == expected["race_id"]
            or row.get("thedogs_source_url") == expected["url"]
        ]
        if not matches:
            raise ProtocolRejected("IDENTITY_MISMATCH" if related else "RACE_NOT_FOUND")
        if len(matches) != 1:
            raise ProtocolRejected("IDENTITY_MISMATCH")
        target = matches[0]
        if target.get("status") != "READY_TO_CAPTURE":
            raise ProtocolRejected("CAPTURE_WINDOW_CLOSED")
        ordered = [target, *(row for row in rows if row is not target)]
        counts: dict[str, int] = {}
        for row in ordered:
            status = str(row.get("status") or "UNKNOWN")
            counts[status] = counts.get(status, 0) + 1
        return {
            **dict(plan),
            "races": ordered,
            "status_counts": counts,
            "ready_count": counts.get("READY_TO_CAPTURE", 0),
            "manual_request_id": context.request["request_id"],
            "manual_request_prioritized": True,
        }
