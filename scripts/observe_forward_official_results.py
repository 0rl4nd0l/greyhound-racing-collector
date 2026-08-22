#!/usr/bin/env python3
"""Run one bounded pass over prospective official-result observations."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_collection.forward_sealed_corpus import (  # noqa: E402
    FORWARD_CORPUS_ORIGIN,
    STATUS_SCHEMA,
    ForwardCorpusRejected,
    ForwardSealedCorpus,
    _normalize_official_result,
    canonical_json,
)
from scripts.ingest_results_for_date import (  # noqa: E402
    THEDOGS_PUBLIC_HEADERS,
    parse_thedogs_result_html_runner_rows,
    thedogs_result_urls_from_race_url,
)

COLLECTOR_ID = "forward-official-result-observer-v1"
TERMINAL_STATES = {"EXAMPLE_CLOSED", "RESULT_CHANGED_BEFORE_CLOSURE"}
KNOWN_STATES = {
    "RESULT_PENDING",
    "RESULT_FIRST_OBSERVED",
    "RESULT_STABILITY_CONFIRMED",
    *TERMINAL_STATES,
}
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
SOURCE_REJECTION_BACKOFF = timedelta(hours=1)
SEMANTIC_FINGERPRINT_ALGORITHM = (
    "thedogs-official-result-semantic-projection-sha256-v1"
)
REJECTION_DEFERRAL_SCHEMA = "forward-official-result-rejection-deferral-v2"
OFFICIAL_RESULT_REQUEST_HEADERS = {
    **THEDOGS_PUBLIC_HEADERS,
    "Accept-Encoding": "identity",
}


class SourceEnvelopeRejected(ValueError):
    """Fetched bytes cannot enter the official-result normalization stage."""

    def __init__(self, reason: str, *, response_hash: str | None = None) -> None:
        super().__init__(reason)
        self.response_hash = response_hash


def _aware_now(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ForwardCorpusRejected("observer clock must return an aware datetime")
    return value


def _identity(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]
    return f"{prefix}-{digest}"


def canonical_result_url(race_url: str) -> str:
    """Derive the sealed non-trial race URL used by the installed resolver."""
    parsed = urlsplit(race_url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"www.thedogs.com.au", "thedogs.com.au"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or "/racing/" not in parsed.path
        or parsed.path.rstrip("/").endswith("/results")
    ):
        raise ForwardCorpusRejected("sealed race-card URL has unsafe or ambiguous identity")
    expected = race_url.rstrip("/") + "?trial=false"
    matches = [
        candidate
        for candidate in thedogs_result_urls_from_race_url(race_url)
        if candidate == expected
    ]
    if len(matches) != 1:
        raise ForwardCorpusRejected("canonical TheDogs result URL is missing or ambiguous")
    return matches[0]


def _source_capture(corpus: ForwardSealedCorpus, race_id: str) -> dict[str, Any]:
    pre = corpus._load_receipt(race_id, "prejump")
    if pre is None:
        raise ForwardCorpusRejected("validated race lost its pre-jump receipt")
    value = json.loads(
        corpus._read_artifact(pre.get("source_capture_checksum"), "source capture")
    )
    if (
        type(value) is not dict
        or value.get("race_id") != race_id
        or value.get("reconstructed") is not False
        or value.get("identity_authority") != "source-native"
    ):
        raise ForwardCorpusRejected("source capture is not prospective source-native material")
    return value


def _race_state(status: Mapping[str, Any], race_id: str) -> str:
    matches = [row for row in status["races"] if row.get("race_id") == race_id]
    if len(matches) != 1 or matches[0].get("state") not in KNOWN_STATES:
        raise ForwardCorpusRejected("validated corpus status has ambiguous race state")
    return str(matches[0]["state"])


def _counts(status: Mapping[str, Any], excluded: int) -> dict[str, int]:
    states = Counter(row["state"] for row in status["races"])
    return {
        "pending": states["RESULT_PENDING"],
        "observed": states["RESULT_FIRST_OBSERVED"],
        "stable": states["RESULT_STABILITY_CONFIRMED"],
        "closed": states["EXAMPLE_CLOSED"],
        "conflict": states["RESULT_CHANGED_BEFORE_CLOSURE"],
        "excluded": excluded,
    }


class _LockOwnership:
    def __init__(self, descriptor: int, directory_descriptor: int, payload: bytes) -> None:
        self.descriptor = descriptor
        self.directory_descriptor = directory_descriptor
        self.payload = payload
        stat = os.fstat(descriptor)
        self.device = stat.st_dev
        self.inode = stat.st_ino


def _acquire_lock(path: Path, cycle_id: str) -> _LockOwnership:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "forward-official-result-observer-lock-v1",
        "cycle_id": cycle_id,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }
    encoded = canonical_json(payload)
    directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError as error:
            raise FileExistsError("forward sealed corpus lock is busy") from error
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
            return _LockOwnership(descriptor, directory_descriptor, encoded)
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            raise
        finally:
            fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
    except BaseException:
        os.close(directory_descriptor)
        raise


def _release_lock(path: Path, owned: _LockOwnership) -> bool:
    try:
        fcntl.flock(owned.directory_descriptor, fcntl.LOCK_EX)
        try:
            current_stat = os.stat(path, follow_symlinks=False)
            if (current_stat.st_dev, current_stat.st_ino) != (owned.device, owned.inode):
                return False
            os.lseek(owned.descriptor, 0, os.SEEK_SET)
            if os.read(owned.descriptor, len(owned.payload) + 1) != owned.payload:
                return False
            path.unlink()
            return True
        except (FileNotFoundError, OSError):
            return False
        finally:
            fcntl.flock(owned.directory_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(owned.descriptor)
        os.close(owned.directory_descriptor)


def _raw_response_body(response: Any) -> bytes:
    response.raw.decode_content = False
    body = response.raw.read(MAX_RESPONSE_BYTES + 1, decode_content=False)
    if len(body) > MAX_RESPONSE_BYTES:
        raise RuntimeError("official response exceeds maximum byte size")
    return body


def _fetch_source_response(session: Any, url: str, timeout_seconds: float) -> dict[str, Any]:
    response = session.get(
        url,
        headers=dict(OFFICIAL_RESULT_REQUEST_HEADERS),
        timeout=timeout_seconds,
        allow_redirects=False,
        stream=True,
    )
    try:
        body = _raw_response_body(response)
        response_hash = hashlib.sha256(body).hexdigest()
        if response.url != url:
            raise SourceEnvelopeRejected(
                "official response final URL changed source identity",
                response_hash=response_hash,
            )
        if not 200 <= response.status_code < 300:
            reason = (
                "official response redirected"
                if 300 <= response.status_code < 400
                else "official response HTTP status is unsupported"
            )
            raise SourceEnvelopeRejected(reason, response_hash=response_hash)
        encoding = response.headers.get("Content-Encoding")
        if encoding is not None and encoding.strip().lower() not in {"", "identity"}:
            raise SourceEnvelopeRejected(
                f"unsupported Content-Encoding: {encoding}",
                response_hash=response_hash,
            )
        content_type = response.headers.get("Content-Type")
        if type(content_type) is not str or not content_type.strip():
            raise SourceEnvelopeRejected(
                "official response content type is missing",
                response_hash=response_hash,
            )
        media_type, separator, parameters = content_type.partition(";")
        if media_type.strip().casefold() != "text/html" or (
            separator and parameters.strip().casefold().replace(" ", "") != "charset=utf-8"
        ):
            raise SourceEnvelopeRejected(
                "official response content type is unsupported",
                response_hash=response_hash,
            )
        return {
            "body": body,
            "status_code": response.status_code,
            "content_type": content_type,
            "final_url": response.url,
            "source_document_last_modified": response.headers.get("Last-Modified"),
        }
    finally:
        response.close()


def _verify_retained_rejected_response(
    corpus: ForwardSealedCorpus,
    *,
    race_id: str,
    collector_id: str,
    session_id: str,
    run_id: str,
    request_id: str,
    request_url: str,
    response_hash: str,
) -> None:
    """Prove normalization, not persistence or identity validation, rejected the stage."""
    pre = corpus._load_receipt(race_id, "prejump")
    if pre is None:
        raise ForwardCorpusRejected("rejected response lost its pre-jump receipt")
    request_directory = corpus._request_directory(corpus.root, race_id, request_id)
    stage = corpus._load_request_receipt(
        request_directory / "response-stage.json", "response-stage"
    )
    if stage is None:
        raise ForwardCorpusRejected("rejected response-stage receipt was not persisted")
    body = corpus._verify_response_stage(pre, stage)
    expected = {
        "race_id": race_id,
        "collector_id": collector_id,
        "session_id": session_id,
        "run_id": run_id,
        "request_id": request_id,
        "request_url": request_url,
    }
    if any(stage.get(key) != value for key, value in expected.items()):
        raise ForwardCorpusRejected("rejected response-stage identity drift")
    if hashlib.sha256(body).hexdigest() != response_hash:
        raise ForwardCorpusRejected("rejected response-stage raw-byte hash drift")


def _semantic_deferral_fingerprint(
    body: bytes,
    *,
    race_id: str,
    source_native_race_id: str,
    request_url: str,
    frozen_runners: Sequence[Mapping[str, Any]],
) -> str | None:
    """Fingerprint a complete parsed result projection, never page scaffolding."""
    try:
        from bs4 import BeautifulSoup

        markup = body.decode("utf-8")
        parsed = parse_thedogs_result_html_runner_rows(markup)
    except (ImportError, UnicodeDecodeError, TypeError, ValueError):
        return None
    if not parsed or len(parsed) != len(frozen_runners):
        return None
    soup = BeautifulSoup(markup, "html.parser")
    result_rows = soup.select("table.race-runners--result tr.race-runner")
    if len(result_rows) != len(parsed):
        return None
    displayed_result_texts = []
    response_native_runner_ids = []
    for result_row in result_rows:
        position_cell = result_row.select_one("td.race-runners__finish-position")
        name_cell = result_row.select_one("td.race-runners__name")
        if position_cell is None or name_cell is None:
            return None
        displayed_result_texts.append(position_cell.get_text(" ", strip=True))
        native_runner_ids = {
            str(element.get("data-dog-id"))
            for element in name_cell.select("blackbook-dog[data-dog-id]")
            if str(element.get("data-dog-id") or "").strip()
        }
        for link in name_cell.select("a[href]"):
            parsed_link = urlsplit(str(link.get("href") or ""))
            parts = parsed_link.path.split("/")
            if (
                not parsed_link.scheme
                and not parsed_link.netloc
                and not parsed_link.query
                and not parsed_link.fragment
                and len(parts) == 4
                and parts[0] == ""
                and parts[1] == "dogs"
                and parts[2]
                and parts[3]
            ):
                native_runner_ids.add(parts[2])
        if len(native_runner_ids) != 1:
            return None
        response_native_runner_ids.append(native_runner_ids.pop())
    if len(set(response_native_runner_ids)) != len(response_native_runner_ids):
        return None
    frozen_by_box = {runner.get("box_number"): runner for runner in frozen_runners}
    if len(frozen_by_box) != len(frozen_runners):
        return None
    projected_rows = []
    parsed_boxes = []
    for row, displayed_result_text, response_native_runner_id in zip(
        parsed, displayed_result_texts, response_native_runner_ids, strict=True
    ):
        if type(row) is not dict or set(row) != {
            "box_number",
            "finish_position",
            "dog_name",
            "status",
        }:
            return None
        box_number = row["box_number"]
        finish_position = row["finish_position"]
        dog_name = row["dog_name"]
        status = row["status"]
        if (
            type(box_number) is not int
            or (finish_position is not None and type(finish_position) is not int)
            or (dog_name is not None and type(dog_name) is not str)
            or (status is not None and type(status) is not str)
        ):
            return None
        parsed_boxes.append(box_number)
        frozen = frozen_by_box.get(box_number)
        projected_rows.append(
            {
                "box_number": box_number,
                "displayed_finish_position": finish_position,
                "displayed_finish_or_status_text": displayed_result_text,
                "displayed_runner_name": dog_name,
                "frozen_runner_name": frozen.get("name") if frozen else None,
                "source_declared_terminal_status": status,
                "response_source_native_runner_id": response_native_runner_id,
                "frozen_source_native_runner_id": (
                    frozen.get("source_native_runner_id") if frozen else None
                ),
            }
        )
    if len(set(parsed_boxes)) != len(parsed_boxes) or set(parsed_boxes) != set(frozen_by_box):
        return None
    projected_rows.sort(key=lambda row: canonical_json(row))
    projection = {
        "fingerprint_algorithm_version": SEMANTIC_FINGERPRINT_ALGORITHM,
        "race_id": race_id,
        "request_url": request_url,
        "source_native_race_id": source_native_race_id,
        "runners": projected_rows,
    }
    return hashlib.sha256(canonical_json(projection)).hexdigest()


def _verify_retained_deferral_response(
    corpus: ForwardSealedCorpus, deferral: Mapping[str, Any]
) -> None:
    pre = corpus._load_receipt(deferral["race_id"], "prejump")
    if pre is None:
        raise ForwardCorpusRejected("retained rejection lost its pre-jump receipt")
    request_directory = corpus._request_directory(
        corpus.root, deferral["race_id"], deferral["retained_request_id"]
    )
    stage = corpus._load_request_receipt(
        request_directory / "response-stage.json", "response-stage"
    )
    if stage is None:
        raise ForwardCorpusRejected("retained rejection response-stage receipt is missing")
    body = corpus._verify_response_stage(pre, stage)
    if any(
        stage.get(field) != deferral[expected]
        for field, expected in (
            ("race_id", "race_id"),
            ("request_id", "retained_request_id"),
            ("request_url", "request_url"),
        )
    ):
        raise ForwardCorpusRejected("retained rejection response-stage identity drift")
    if hashlib.sha256(body).hexdigest() != deferral["retained_raw_response_hash"]:
        raise ForwardCorpusRejected("retained rejection raw-byte hash drift")


def _validated_rejection_deferrals(
    value: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if type(value) not in {list, tuple}:
        raise ForwardCorpusRejected("source rejection deferral metadata is invalid")
    result: dict[str, dict[str, Any]] = {}
    for item in value:
        if type(item) is not dict:
            raise ForwardCorpusRejected("source rejection deferral metadata is invalid")
        legacy = set(item) == {
            "race_id",
            "response_hash",
            "reason",
            "rejected_at",
            "next_eligible_at",
        }
        versioned = set(item) == {
            "schema_version",
            "race_id",
            "source_native_race_id",
            "request_url",
            "retained_request_id",
            "retained_raw_response_hash",
            "semantic_fingerprint",
            "fingerprint_algorithm_version",
            "reason",
            "rejected_at",
            "next_eligible_at",
            "deferral_decision",
            "pending_state",
        }
        if not legacy and not versioned:
            raise ForwardCorpusRejected("source rejection deferral metadata is invalid")
        race_id = item["race_id"]
        response_hash = (
            item["response_hash"] if legacy else item["retained_raw_response_hash"]
        )
        reason = item["reason"]
        if (
            type(race_id) is not str
            or not race_id
            or len(race_id.encode()) > 128
            or race_id in result
            or type(response_hash) is not str
            or len(response_hash) != 64
            or any(character not in "0123456789abcdef" for character in response_hash)
            or type(reason) is not str
            or not reason
        ):
            raise ForwardCorpusRejected("source rejection deferral metadata is invalid")
        if versioned:
            semantic_fingerprint = item["semantic_fingerprint"]
            algorithm = item["fingerprint_algorithm_version"]
            if (
                item["schema_version"] != REJECTION_DEFERRAL_SCHEMA
                or algorithm != SEMANTIC_FINGERPRINT_ALGORITHM
                or type(semantic_fingerprint) is not str
                or len(semantic_fingerprint) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in semantic_fingerprint
                )
                or any(
                    type(item[field]) is not str
                    or not item[field]
                    or len(item[field].encode()) > 512
                    for field in (
                        "source_native_race_id",
                        "request_url",
                        "retained_request_id",
                    )
                )
                or item["deferral_decision"]
                not in {"SOURCE_REJECTED", "SOURCE_REJECTION_DEFERRED"}
                or item["pending_state"] != "RESULT_PENDING"
            ):
                raise ForwardCorpusRejected(
                    "source rejection deferral metadata is invalid"
                )
        try:
            rejected_at = datetime.fromisoformat(item["rejected_at"])
            next_eligible_at = datetime.fromisoformat(item["next_eligible_at"])
        except (TypeError, ValueError) as error:
            raise ForwardCorpusRejected(
                "source rejection deferral metadata is invalid"
            ) from error
        if (
            rejected_at.tzinfo is None
            or rejected_at.utcoffset() is None
            or next_eligible_at.tzinfo is None
            or next_eligible_at.utcoffset() is None
            or next_eligible_at - rejected_at != SOURCE_REJECTION_BACKOFF
            or next_eligible_at
            > datetime.now().astimezone() + SOURCE_REJECTION_BACKOFF
        ):
            raise ForwardCorpusRejected("source rejection deferral metadata is invalid")
        result[race_id] = dict(item)
    return result


def _rejection_deferral(
    *,
    race_id: str,
    source_native_race_id: str,
    request_url: str,
    retained_request_id: str,
    response_hash: str,
    semantic_fingerprint: str,
    reason: str,
    rejected_at: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": REJECTION_DEFERRAL_SCHEMA,
        "race_id": race_id,
        "source_native_race_id": source_native_race_id,
        "request_url": request_url,
        "retained_request_id": retained_request_id,
        "retained_raw_response_hash": response_hash,
        "semantic_fingerprint": semantic_fingerprint,
        "fingerprint_algorithm_version": SEMANTIC_FINGERPRINT_ALGORITHM,
        "reason": reason,
        "rejected_at": rejected_at.isoformat(timespec="microseconds"),
        "next_eligible_at": (rejected_at + SOURCE_REJECTION_BACKOFF).isoformat(
            timespec="microseconds"
        ),
        "deferral_decision": "SOURCE_REJECTED",
        "pending_state": "RESULT_PENDING",
    }


def _record_source_rejection(
    race_report: dict[str, Any],
    error: Exception,
    corpus: ForwardSealedCorpus,
    race_id: str,
) -> None:
    race_report["decision"] = "SOURCE_REJECTED"
    race_report["source_rejection"] = f"{type(error).__name__}: {error}"
    try:
        race_report["after_state"] = _race_state(corpus.status(), race_id)
    except Exception as status_error:
        race_report["decision"] = "ERROR"
        race_report["error"] = (
            "source rejection status check failed: "
            f"{type(status_error).__name__}: {status_error}"
        )


def _deferral_applies(
    deferral: Mapping[str, Any] | None,
    *,
    semantic_fingerprint: str | None,
    observed_at: datetime,
) -> bool:
    if deferral is None or observed_at >= datetime.fromisoformat(
        deferral["next_eligible_at"]
    ):
        return False
    if "schema_version" not in deferral:
        return False
    return bool(
        semantic_fingerprint
        and deferral["semantic_fingerprint"] == semantic_fingerprint
    )


def _record_deferred_rejection(
    race_report: dict[str, Any],
    deferral: Mapping[str, Any],
) -> None:
    recorded = dict(deferral)
    if "schema_version" in recorded:
        recorded["deferral_decision"] = "SOURCE_REJECTION_DEFERRED"
    race_report["decision"] = "SOURCE_REJECTION_DEFERRED"
    race_report["source_rejection"] = "DeferredSourceRejection: " + deferral["reason"]
    race_report["deferral_reason"] = "identical_result_semantics_before_next_eligibility"
    race_report["rejection_deferral"] = recorded


def _attach_new_deferral(
    race_report: dict[str, Any],
    active_deferrals: dict[str, dict[str, Any]],
    *,
    race_id: str,
    source_native_race_id: str,
    request_url: str,
    request_id: str,
    response_hash: str | None,
    semantic_fingerprint: str | None,
    reason: str,
    rejected_at: datetime,
) -> None:
    if (
        race_report["decision"] != "SOURCE_REJECTED"
        or response_hash is None
        or semantic_fingerprint is None
    ):
        return
    deferral = _rejection_deferral(
        race_id=race_id,
        source_native_race_id=source_native_race_id,
        request_url=request_url,
        retained_request_id=request_id,
        response_hash=response_hash,
        semantic_fingerprint=semantic_fingerprint,
        reason=reason,
        rejected_at=rejected_at,
    )
    race_report["rejection_deferral"] = deferral
    active_deferrals[race_id] = deferral


def observe_once(
    *,
    corpus_root: Path,
    cycle_id: str,
    timeout_seconds: float = 30.0,
    clock: Callable[[], datetime] | None = None,
    session_factory: Callable[[], Any] = requests.Session,
    corpus_factory: Callable[..., ForwardSealedCorpus] = ForwardSealedCorpus,
    previous_rejection_deferrals: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Observe each eligible nonterminal validated race at most once."""
    if not cycle_id or len(cycle_id.encode()) > 96:
        raise ValueError("cycle_id must be a non-empty bounded caller identity")
    if not 0 < timeout_seconds <= 120:
        raise ValueError("timeout_seconds must be in (0, 120]")
    observer_clock = clock or (lambda: datetime.now().astimezone())
    root = Path(corpus_root).resolve()
    lock_path = root / "forward-sealed-corpus.lock"
    report: dict[str, Any] = {
        "schema_version": "forward-official-result-observer-run-v1",
        "corpus_origin": FORWARD_CORPUS_ORIGIN,
        "cycle_id": cycle_id,
        "collector_id": COLLECTOR_ID,
        "session_id": _identity("session", cycle_id),
        "run_id": _identity("run", cycle_id),
        "status": "RUNNING",
        "attempted_race_ids": [],
        "races": [],
        "package_hashes": [],
        "source_rejection_count": 0,
        "source_rejected_race_ids": [],
        "deferred_rejection_count": 0,
        "source_rejection_deferrals": [],
    }
    owned: _LockOwnership | None = None
    try:
        owned = _acquire_lock(lock_path, cycle_id)
    except FileExistsError as error:
        report.update(
            status="LOCK_BUSY",
            error=str(error),
            counts={
                key: 0
                for key in ("pending", "observed", "stable", "closed", "conflict", "excluded")
            },
            lock_released=False,
        )
        return report
    try:
        corpus = corpus_factory(root, clock=observer_clock)
        before = corpus.status()
        if before.get("schema_version") != STATUS_SCHEMA:
            raise ForwardCorpusRejected("corpus status is not the accepted prospective schema")
        active_deferrals = _validated_rejection_deferrals(previous_rejection_deferrals)
        race_ids = {row.get("race_id") for row in before["races"]}
        if not set(active_deferrals) <= race_ids:
            raise ForwardCorpusRejected("source rejection deferral race identity is unknown")
        state_by_race = {row["race_id"]: row["state"] for row in before["races"]}
        for deferred_race_id, deferral in active_deferrals.items():
            if "schema_version" not in deferral:
                continue
            source = _source_capture(corpus, deferred_race_id)
            if (
                state_by_race[deferred_race_id] != deferral["pending_state"]
                or source["source_native_race_id"]
                != deferral["source_native_race_id"]
                or canonical_result_url(source["canonical_source_url"])
                != deferral["request_url"]
            ):
                raise ForwardCorpusRejected(
                    "source rejection deferral race identity is inconsistent"
                )
            _verify_retained_deferral_response(corpus, deferral)
        session = session_factory()
        try:
            for index, row in enumerate(before["races"]):
                race_id = row["race_id"]
                race_report: dict[str, Any] = {
                    "race_id": race_id,
                    "before_state": row["state"],
                    "after_state": row["state"],
                    "request_id": None,
                    "decision": "SKIPPED",
                    "receipt_hash": None,
                    "raw_response_hash": None,
                    "semantic_fingerprint": None,
                    "fingerprint_algorithm_version": None,
                    "normalization_attempted": False,
                    "source_rejection": None,
                    "deferral_reason": None,
                    "rejection_deferral": None,
                    "error": None,
                }
                report["races"].append(race_report)
                if row["state"] in TERMINAL_STATES:
                    active_deferrals.pop(race_id, None)
                    race_report["decision"] = "SKIPPED_TERMINAL"
                    continue
                try:
                    source = _source_capture(corpus, race_id)
                    jump = datetime.fromisoformat(source["scheduled_jump_at"])
                    if jump.tzinfo is None or jump.utcoffset() is None:
                        raise ForwardCorpusRejected("scheduled jump is not timezone-aware")
                    eligibility_time = _aware_now(observer_clock)
                    if row["state"] == "RESULT_PENDING" and eligibility_time < (
                        jump + timedelta(minutes=5)
                    ):
                        race_report["decision"] = "SKIPPED_PRE_BOUNDARY"
                        continue
                    request_url = canonical_result_url(source["canonical_source_url"])
                    request_id = _identity("request", cycle_id, str(index), race_id)
                    race_report["request_id"] = request_id
                    report["attempted_race_ids"].append(race_id)
                    prior_deferral = active_deferrals.get(race_id)
                    source_response = _fetch_source_response(
                        session, request_url, timeout_seconds
                    )
                    response_hash = hashlib.sha256(source_response["body"]).hexdigest()
                    race_report["raw_response_hash"] = response_hash
                    semantic_fingerprint = _semantic_deferral_fingerprint(
                        source_response["body"],
                        race_id=race_id,
                        source_native_race_id=source["source_native_race_id"],
                        request_url=request_url,
                        frozen_runners=source["runners"],
                    )
                    race_report["semantic_fingerprint"] = semantic_fingerprint
                    if semantic_fingerprint is not None:
                        race_report["fingerprint_algorithm_version"] = (
                            SEMANTIC_FINGERPRINT_ALGORITHM
                        )
                    if _deferral_applies(
                        prior_deferral,
                        semantic_fingerprint=semantic_fingerprint,
                        observed_at=eligibility_time,
                    ):
                        _record_deferred_rejection(race_report, prior_deferral)
                        active_deferrals[race_id] = race_report["rejection_deferral"]
                        continue
                    active_deferrals.pop(race_id, None)
                    normalization_rejection: ForwardCorpusRejected | None = None
                    race_report["normalization_attempted"] = True
                    try:
                        _normalize_official_result(
                            source_response["body"],
                            race_id=race_id,
                            frozen_runners=source["runners"],
                        )
                    except ForwardCorpusRejected as error:
                        normalization_rejection = error
                    if normalization_rejection is not None and not source_response["body"]:
                        _record_source_rejection(
                            race_report, normalization_rejection, corpus, race_id
                        )
                        _attach_new_deferral(
                            race_report,
                            active_deferrals,
                            race_id=race_id,
                            source_native_race_id=source["source_native_race_id"],
                            request_url=request_url,
                            request_id=request_id,
                            response_hash=response_hash,
                            semantic_fingerprint=semantic_fingerprint,
                            reason=str(normalization_rejection),
                            rejected_at=eligibility_time,
                        )
                        continue

                    try:
                        receipt = corpus.capture_result(
                            race_id=race_id,
                            collector_id=COLLECTOR_ID,
                            session_id=report["session_id"],
                            run_id=report["run_id"],
                            request_id=request_id,
                            request_url=request_url,
                            transport=lambda _url: source_response,
                        )
                    except ForwardCorpusRejected as error:
                        if normalization_rejection is None:
                            raise
                        try:
                            _verify_retained_rejected_response(
                                corpus,
                                race_id=race_id,
                                collector_id=COLLECTOR_ID,
                                session_id=report["session_id"],
                                run_id=report["run_id"],
                                request_id=request_id,
                                request_url=request_url,
                                response_hash=response_hash,
                            )
                        except ForwardCorpusRejected as verification_error:
                            raise verification_error from error
                        _record_source_rejection(
                            race_report, normalization_rejection, corpus, race_id
                        )
                        _attach_new_deferral(
                            race_report,
                            active_deferrals,
                            race_id=race_id,
                            source_native_race_id=source["source_native_race_id"],
                            request_url=request_url,
                            request_id=request_id,
                            response_hash=response_hash,
                            semantic_fingerprint=semantic_fingerprint,
                            reason=str(normalization_rejection),
                            rejected_at=eligibility_time,
                        )
                        continue
                    if normalization_rejection is not None:
                        raise ForwardCorpusRejected(
                            "corpus accepted source bytes rejected by observer normalization"
                        )
                    race_report["receipt_hash"] = hashlib.sha256(
                        canonical_json(receipt)
                    ).hexdigest()
                    active_deferrals.pop(race_id, None)
                    race_report["after_state"] = _race_state(corpus.status(), race_id)
                    race_report["decision"] = race_report["after_state"]
                    if race_report["after_state"] == "RESULT_STABILITY_CONFIRMED":
                        corpus.close(race_id=race_id)
                        package = corpus.build_package()
                        race_report["after_state"] = _race_state(corpus.status(), race_id)
                        race_report["decision"] = race_report["after_state"]
                        report["package_hashes"].append(
                            {
                                "race_id": race_id,
                                "package_checksum": str(package.package_checksum),
                                "manifest_checksum": str(package.manifest_checksum),
                            }
                        )
                except SourceEnvelopeRejected as error:
                    race_report["raw_response_hash"] = error.response_hash
                    if _deferral_applies(
                        prior_deferral,
                        semantic_fingerprint=None,
                        observed_at=eligibility_time,
                    ):
                        _record_deferred_rejection(race_report, prior_deferral)
                        active_deferrals[race_id] = race_report["rejection_deferral"]
                    else:
                        active_deferrals.pop(race_id, None)
                        _record_source_rejection(race_report, error, corpus, race_id)
                    _attach_new_deferral(
                        race_report,
                        active_deferrals,
                        race_id=race_id,
                        source_native_race_id=source["source_native_race_id"],
                        request_url=request_url,
                        request_id=request_id,
                        response_hash=error.response_hash,
                        semantic_fingerprint=None,
                        reason=str(error),
                        rejected_at=eligibility_time,
                    )
                except ForwardCorpusRejected as error:
                    race_report["decision"] = "ERROR"
                    race_report["error"] = f"{type(error).__name__}: {error}"
                    try:
                        race_report["after_state"] = _race_state(corpus.status(), race_id)
                    except Exception as status_error:
                        race_report["error"] = (
                            f"{race_report['error']}; status error: "
                            f"{type(status_error).__name__}: {status_error}"
                        )
                except Exception as error:  # isolate operational race failures
                    race_report["decision"] = "ERROR"
                    race_report["error"] = f"{type(error).__name__}: {error}"
                    try:
                        race_report["after_state"] = _race_state(corpus.status(), race_id)
                    except Exception as status_error:
                        race_report["error"] += (
                            f"; status error: {type(status_error).__name__}: {status_error}"
                        )
        finally:
            close = getattr(session, "close", None)
            if callable(close):
                close()
        final_status = corpus.status()
        report["counts"] = _counts(final_status, 0)
        source_rejected = [
            row
            for row in report["races"]
            if row["decision"] in {"SOURCE_REJECTED", "SOURCE_REJECTION_DEFERRED"}
        ]
        report["deferred_rejection_count"] = sum(
            row["decision"] == "SOURCE_REJECTION_DEFERRED" for row in report["races"]
        )
        report["source_rejection_count"] = len(source_rejected)
        report["source_rejected_race_ids"] = [row["race_id"] for row in source_rejected]
        report["source_rejection_deferrals"] = [
            active_deferrals[race_id] for race_id in sorted(active_deferrals)
        ]
        if any(row["error"] for row in report["races"]):
            report["status"] = "COMPLETED_WITH_ERRORS"
        elif source_rejected:
            report["status"] = "COMPLETED_WITH_REJECTIONS"
        else:
            report["status"] = "COMPLETED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
        report.setdefault(
            "counts",
            {
                key: 0
                for key in ("pending", "observed", "stable", "closed", "conflict", "excluded")
            },
        )
    finally:
        report["lock_released"] = _release_lock(lock_path, owned) if owned else False
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = observe_once(
        corpus_root=args.corpus_root,
        cycle_id=args.cycle_id,
        timeout_seconds=args.timeout_seconds,
    )
    print(canonical_json(report).decode())
    return 0 if report["status"] in {"COMPLETED", "COMPLETED_WITH_REJECTIONS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
