"""Finite operational scope and append-only, shared capture consumption.

No history or network reads. Reconciliation is supplied by the quiescent admission
step. Absence, ambiguity, interrupted writes and spent reservations fail closed.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from race_collection.live_phase_checkpoint import atomic_json


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def create_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL consumption precedes all network work. A truncated file is consumed.
    with path.open("xb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FreshnessContract:
    def __init__(self, value):
        self.value = value
        if (
            value.get("schema_version") != "freshness_rehearsal_contract_v1"
            or value.get("profile") != "bounded80-v1"
        ):
            raise ValueError("invalid_freshness_contract")
        self.start = datetime.fromisoformat(value["starts_at"])
        self.end = datetime.fromisoformat(value["ends_at"])
        if (
            self.start.utcoffset() is None
            or self.end.utcoffset() is None
            or (self.end - self.start).total_seconds() != 5400
        ):
            raise ValueError("invalid_scope_duration")
        zone = ZoneInfo("Australia/Melbourne")
        local_start, local_end = self.start.astimezone(zone), self.end.astimezone(zone)
        cutoff = local_start.replace(hour=21, minute=20, second=0, microsecond=0)
        if (
            local_start.date().isoformat() != value["source_date"]
            or local_end.date() != local_start.date()
            or value["cleanup_seconds"] != 1200
            or local_end + timedelta(seconds=1200) > cutoff
        ):
            raise ValueError("one_date_scope_required")
        if value["max_capture_attempts"] != 1 or not 0 < value["max_logical_requests"] <= 24000:
            raise ValueError("invalid_scope_allowance")
        for key in ("lock_path", "evidence_root", "db_path"):
            if not Path(value[key]).is_absolute():
                raise ValueError("absolute_scope_paths_required")
        if not value.get("rehearsal_id") or not value.get("reconciliation_sha256"):
            raise ValueError("scope_identity_required")
        self.root = Path(value["lock_path"]).parent / "live-freshness-attempts-v1"
        self.session = self.root / (
            "rehearsal-" + hashlib.sha256(value["rehearsal_id"].encode()).hexdigest()
        )

    @classmethod
    def load(cls, path):
        if path is None:
            raise ValueError("live_profile_requires_contract")
        return cls(json.loads(Path(path).read_bytes()))

    def admit(self, now, *, seconds):
        if (
            now.utcoffset() is None
            or seconds < 0
            or now < self.start
            or now + timedelta(seconds=seconds) > self.end
        ):
            raise ValueError("operating_scope_closed")
        if (
            now.astimezone(ZoneInfo("Australia/Melbourne")).date().isoformat()
            != self.value["source_date"]
        ):
            raise ValueError("one_date_scope_required")
        if (self.session / "STOP.json").exists():
            raise ValueError("operating_scope_stopped")

    def check_paths(self, *, lock_path, evidence_root, db_path):
        for key, path in (
            ("lock_path", lock_path),
            ("evidence_root", evidence_root),
            ("db_path", db_path),
        ):
            if Path(self.value[key]).resolve() != Path(path).resolve():
                raise ValueError("operating_scope_path_mismatch")

    def stop(self, reason):
        try:
            create_once(self.session / "STOP.json", {"reason": reason})
        except FileExistsError:
            pass


class AttemptAllowance:
    def __init__(self, scope):
        self.scope = scope
        self.claim = scope.session / "capture-reservation.json"
        self.reconciliation = scope.session / "reconciliation.json"

    def initialize(self, reconciliation):
        if (
            reconciliation.get("schema_version") != "freshness_attempt_reconciliation_v1"
            or reconciliation.get("complete") is not True
            or not reconciliation.get("sources")
            or digest(reconciliation) != self.scope.value["reconciliation_sha256"]
        ):
            raise ValueError("attempt_reconciliation_incomplete")
        create_once(self.reconciliation, reconciliation)
        create_once(self.scope.session / "scope.json", self.scope.value)

    def _accounting(self):
        value = json.loads(self.reconciliation.read_bytes())
        if (
            digest(value) != self.scope.value["reconciliation_sha256"]
            or value.get("complete") is not True
        ):
            raise ValueError("attempt_reconciliation_changed")
        return value

    def available(self):
        self._accounting()
        return not self.claim.exists()

    @staticmethod
    def key(item):
        race = str(item.get("race_id") or "")
        window = item.get("capture_window_minutes")
        if not race or window not in (60, 30, 10, 2):
            raise ValueError("capture_window_identity_required")
        return race, window

    def consumed(self, item):
        key = self.key(item)
        keys = {(alias, key[1]) for alias in item.get("race_id_aliases", [key[0]])} | {key}
        return any(self.key(row) in keys for row in self._accounting()["consumed"]) or any(
            (self.scope.root / "windows" / (digest(list(candidate)) + ".json")).exists()
            for candidate in keys
        )

    def reserve(self, item, *, now):
        self.scope.admit(now, seconds=155)
        if not self.available() or self.consumed(item):
            raise ValueError("capture_allowance_consumed")
        claim = {
            "schema_version": "freshness_capture_reservation_v1",
            "contract_sha256": digest(self.scope.value),
            "item": item,
            "reserved_at": now.isoformat(),
            "status": "CONSUMED",
        }
        create_once(self.claim, claim)
        # If this second write fails, the session is still spent; never substitute.
        key = self.key(item)
        keys = {(alias, key[1]) for alias in item.get("race_id_aliases", [key[0]])} | {key}
        for candidate in sorted(keys):
            create_once(self.scope.root / "windows" / (digest(list(candidate)) + ".json"), claim)
        return self.claim

    def start_fetch(self, claim_path, item, *, now):
        if Path(claim_path).resolve() != self.claim.resolve():
            raise ValueError("capture_reservation_path_mismatch")
        claim = json.loads(self.claim.read_bytes())
        if claim["contract_sha256"] != digest(self.scope.value) or self.key(
            claim["item"]
        ) != self.key(item):
            raise ValueError("capture_reservation_identity_changed")
        for path, expected_hash in claim["item"].get("input_files", {}).items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("capture_reserved_input_changed")
        if (
            claim["item"].get("input_files")
            and item.get("csv_path") not in claim["item"]["input_files"]
        ):
            raise ValueError("capture_reserved_input_substituted")
        expected = claim["item"]["race_identity"]
        actual_jump = item.get("jump_datetime", item.get("race_identity", {}).get("jump_datetime"))
        if expected["jump_datetime"] != actual_jump:
            raise ValueError("capture_reservation_jump_changed")
        self.scope.admit(now, seconds=50)
        create_once(
            self.claim.with_suffix(".fetch.json"),
            {"started_at": now.isoformat(), "reservation_sha256": digest(claim)},
        )

    def finish(self, claim_path, result):
        if Path(claim_path).resolve() != self.claim.resolve():
            raise ValueError("capture_reservation_path_mismatch")
        create_once(self.claim.with_suffix(".terminal.json"), {"result": result})


def install_request_guard(scope):
    """Account all requests.Session calls, including auxiliary clients and retries.

    Counts logical calls; transport-level retries remain separately unbounded by
    this counter. Used only inside an already admitted acquisition subprocess.
    """
    import fcntl
    import requests

    original = requests.Session.request

    def request(session, method, url, **kwargs):
        scope.admit(datetime.now(timezone.utc), seconds=0)
        scope.session.mkdir(parents=True, exist_ok=True)
        with (scope.session / "request-count.lock").open("a") as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            path = scope.session / "request-count.json"
            count = json.loads(path.read_bytes())["started"] if path.exists() else 0
            if count >= scope.value["max_logical_requests"]:
                scope.stop("REQUEST_CAP_EXHAUSTED")
                raise ValueError("request_cap_exhausted")
            atomic_json(path, {"started": count + 1})
        return original(session, method, url, **kwargs)

    requests.Session.request = request
    return lambda: setattr(requests.Session, "request", original)


def reject_unreserved_capture(db_path, item, reservation=None):
    """Shared scheduled/manual fetch seam; inert before accounting is installed."""
    from race_collection.synchronous_manual_capture import CANONICAL_LOCK_RELATIVE_PATH

    root = (
        Path(db_path).resolve().parent / CANONICAL_LOCK_RELATIVE_PATH
    ).parent / "live-freshness-attempts-v1"
    if not root.exists():
        return
    key = AttemptAllowance.key(item)
    path = root / "windows" / (digest(list(key)) + ".json")
    if path.exists() and reservation is None:
        raise ValueError("capture_window_previously_consumed")
    if reservation is None:
        now = datetime.now(timezone.utc)
        for scope_path in root.glob("rehearsal-*/scope.json"):
            scope = FreshnessContract.load(scope_path)
            if scope.start <= now <= scope.end + timedelta(seconds=1200):
                raise ValueError("capture_reserved_for_operational_rehearsal")


def reconcile_projections(projections):
    """Merge bounded operational projections, never infer absence from DB rows alone.

    Admission supplies all seven domain projections under the shared lock. Each
    projection authenticates its source inventory and classifies incomplete input.
    An ambiguous identity blocks instead of silently skipping a potentially spent
    window. Original source files are never rewritten.
    """
    required = {
        "scheduled_progress",
        "scheduled_reports",
        "manual_claims",
        "manual_attempts",
        "phase_checkpoints",
        "prior_rehearsals",
        "live_odds",
    }
    if set(projections) != required:
        raise ValueError("reconciliation_domains_incomplete")
    consumed, sources = {}, []
    for domain, projection in projections.items():
        if projection.get("complete") is not True or not projection.get("inventory_sha256"):
            raise ValueError("reconciliation_inventory_incomplete")
        sources.append({"domain": domain, "sha256": projection["inventory_sha256"]})
        for row in projection["consumed"]:
            key = AttemptAllowance.key(row)
            consumed[key] = {"race_id": key[0], "capture_window_minutes": key[1]}
    return {
        "schema_version": "freshness_attempt_reconciliation_v1",
        "complete": True,
        "consumed": [consumed[key] for key in sorted(consumed)],
        "sources": sources,
    }


def verify_source_package(root, expected_digest):
    root = Path(root).resolve()
    identity = json.loads((root / "SOURCE_IDENTITY.json").read_bytes())
    if digest(identity) != expected_digest:
        raise ValueError("source_package_identity_changed")
    for name, expected in identity["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("source_package_file_changed")
    return identity
