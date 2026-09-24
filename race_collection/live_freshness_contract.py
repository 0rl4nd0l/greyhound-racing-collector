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
            or value["cleanup_seconds"] != (1860 if value.get("campaign_root") else 1200)
            or local_end + timedelta(seconds=value["cleanup_seconds"]) > cutoff
        ):
            raise ValueError("one_date_scope_required")
        self.campaign = None
        if value.get("campaign_root"):
            from race_collection.freshness_campaign import Campaign
            self.campaign = Campaign(value["campaign_root"])
            if digest(self.campaign.value) != value["campaign_authorization_sha256"]:
                raise ValueError("campaign_authorization_changed")
        if (value["max_capture_attempts"] != (self.campaign.value['max_capture_attempts'] if self.campaign else 1)
                or not 0 < value["max_logical_requests"] <= (48000 if self.campaign else 24000)):
            raise ValueError("invalid_scope_allowance")
        for key in ("lock_path", "evidence_root", "db_path"):
            if not Path(value[key]).is_absolute():
                raise ValueError("absolute_scope_paths_required")
        operational = value.get("operational_predictions")
        if operational:
            if self.campaign is None:
                raise ValueError("operational_campaign_required")
            expected = self.campaign.root / "operational-predictions/capture.sqlite3"
            capture = Path(value["db_path"]).resolve()
            if (capture != expected.resolve() or capture != Path(operational["capture_db_path"]).resolve()
                    or capture == Path(operational["history_db_path"]).resolve()):
                raise ValueError("operational_database_separation_required")
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
        if self.campaign:
            self.campaign.admit(self.value["rehearsal_id"], now + timedelta(seconds=seconds))
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
        return self.scope.campaign.available() if self.scope.campaign else not self.claim.exists()

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
        if self.scope.campaign:
            with self.scope.campaign.ledger() as ledger:
                if any((self.scope.value.get("operational_predictions") or row["window"] == key[1]) and any((alias, key[1]) in keys for alias in row["aliases"])
                       for row in ledger["attempts"]):
                    return True
        return any(self.key(row) in keys for row in self._accounting()["consumed"]) or any(
            (self.scope.root / "windows" / (digest(list(candidate)) + ".json")).exists()
            for candidate in keys
        )

    @staticmethod
    def check_window(item, *, now, required_seconds=0):
        """Planner opens at target; the original next-window boundary is exclusive."""
        _, window = AttemptAllowance.key(item)
        jump = datetime.fromisoformat(item["race_identity"]["jump_datetime"])
        from scripts.autonomous_live_odds_capture import capture_window_bounds
        opens, closes = capture_window_bounds(jump_datetime=jump,
                                             capture_window_minutes=window,
                                             tolerance_seconds=0)
        if now.utcoffset() is None or jump.utcoffset() is None:
            raise ValueError("capture_window_timezone_required")
        if now < opens:
            raise ValueError("capture_reservation_not_open")
        if now >= closes:
            raise ValueError("capture_reservation_expired")
        if now + timedelta(seconds=required_seconds) >= closes:
            raise ValueError("capture_window_insufficient_time")
        return closes

    def reserve(self, item, *, now):
        self.scope.admit(now, seconds=155)
        if not self.available() or self.consumed(item):
            raise ValueError("capture_allowance_consumed")
        closes = self.check_window(item, now=now)
        claim = {
            "expires_at": closes.isoformat(),
            "schema_version": "freshness_capture_reservation_v1",
            "contract_sha256": digest(self.scope.value),
            "item": item,
            "reserved_at": now.isoformat(),
            "status": "CONSUMED",
        }
        if self.scope.campaign:
            import uuid
            self.claim = self.scope.session / "captures" / uuid.uuid4().hex / "capture-reservation.json"
            self.scope.campaign.consume(self.claim, item)
        create_once(self.claim, claim)
        # If this second write fails, the session is still spent; never substitute.
        key = self.key(item)
        keys = {(alias, key[1]) for alias in item.get("race_id_aliases", [key[0]])} | {key}
        for candidate in sorted(keys):
            create_once(self.scope.root / "windows" / (digest(list(candidate)) + ".json"), claim)
        return self.claim

    def claims(self):
        if self.scope.campaign:
            return sorted(self.scope.session.glob("captures/*/capture-reservation.json"))
        return [self.claim] if self.claim.exists() else []

    def select_claim(self, path):
        candidate = Path(path).resolve()
        if self.scope.campaign:
            if (candidate.parent.parent != (self.scope.session / "captures").resolve()
                    or candidate.name != "capture-reservation.json"):
                raise ValueError("capture_reservation_path_mismatch")
            with self.scope.campaign.ledger() as ledger:
                rows = [row for row in ledger["attempts"] if row["claim"] == str(candidate)]
                if len(rows) != 1:
                    raise ValueError("capture_campaign_reservation_missing")
                if rows[0]["item"] != json.loads(candidate.read_bytes())["item"]:
                    raise ValueError("capture_campaign_reservation_changed")
            self.claim = candidate
        elif candidate != self.claim.resolve():
            raise ValueError("capture_reservation_path_mismatch")

    def bind_capture_plan(self, claim_path, plan):
        """Authenticate a native planner alias, then carry the reserved canonical ID.

        Alias membership alone is insufficient: exact input bytes, source URL,
        native source identity, jump, window and runner set must agree. Rebinding
        neither starts a fetch nor creates/replenishes an allowance.
        """
        from race_collection.synchronous_manual_capture import runner_set_sha256
        from utils.runner_completeness import normalise_runner_name

        self.select_claim(claim_path)
        claim = json.loads(self.claim.read_bytes())
        if claim["contract_sha256"] != digest(self.scope.value):
            raise ValueError("capture_reservation_identity_changed")
        reserved = claim["item"]
        identity = reserved["race_identity"]
        rows = plan.get("races", [])
        if len(rows) != 1:
            raise ValueError("capture_reservation_plan_ambiguous")
        item = dict(rows[0])
        aliases = reserved.get("race_id_aliases", [reserved["race_id"]])
        if (
            not isinstance(aliases, list)
            or not 0 < len(aliases) <= 16
            or len(set(aliases)) != len(aliases)
            or reserved["race_id"] not in aliases
            or item.get("race_id") not in aliases
            or identity.get("race_id") != reserved["race_id"]
        ):
            raise ValueError("capture_reservation_identity_changed")
        for alias in aliases:
            parts = alias.split(" - ")
            if (
                len(parts) != 3
                or parts[0] != f"Race {item['race_number']}"
                or parts[2] != item["race_date"]
            ):
                raise ValueError("capture_reservation_alias_invalid")
        files = reserved.get("input_files", {})
        if not files or item.get("csv_path") not in files or item.get("sidecar_path") not in files:
            raise ValueError("capture_reserved_input_substituted")
        for path, expected in files.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("capture_reserved_input_changed")
        metadata = json.loads(Path(item["sidecar_path"]).read_bytes())["prejump_shadow_metadata"]
        if (
            item.get("thedogs_source_url") != identity.get("race_url")
            or not identity.get("source_native_race_id")
            or str(metadata.get("source_native_race_id")) != str(identity["source_native_race_id"])
            or any(
                normalise_runner_name(row["dog_name"]) != row["identity"]
                for row in item.get("expected_runners", [])
            )
            or runner_set_sha256(item.get("expected_runners", []))
            != reserved.get("capture_runner_set_sha256")
        ):
            raise ValueError("capture_reservation_source_identity_changed")
        if item.get("jump_datetime") != identity["jump_datetime"]:
            raise ValueError("capture_reservation_jump_changed")
        if item.get("capture_window_minutes") != reserved["capture_window_minutes"]:
            raise ValueError("capture_reservation_identity_changed")
        item.update(
            race_id=reserved["race_id"],
            canonical_race_id=reserved["race_id"],
            planner_race_id=item.get("planner_race_id", item["race_id"]),
            race_id_aliases=list(aliases),
            race_identity=dict(identity),
        )
        return {**plan, "races": [item]}

    def start_fetch(self, claim_path, item, *, now):
        self.select_claim(claim_path)
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
        self.check_window(claim["item"], now=now)
        create_once(
            self.claim.with_suffix(".fetch.json"),
            {"started_at": now.isoformat(), "reservation_sha256": digest(claim)},
        )

    def finish(self, claim_path, result):
        self.select_claim(claim_path)
        create_once(self.claim.with_suffix(".terminal.json"), {"result": result})


def install_request_guard(scope):
    """Account all requests.Session calls, including auxiliary clients and retries.

    Counts logical calls; transport-level retries remain separately unbounded by
    this counter. Used only inside an already admitted acquisition subprocess.
    """
    import fcntl
    import requests
    from urllib.parse import urlsplit

    original = requests.Session.request

    def request(session, method, url, **kwargs):
        scope.admit(datetime.now(timezone.utc), seconds=0)
        scope.session.mkdir(parents=True, exist_ok=True)
        with (scope.session / "request-count.lock").open("a") as mutex:
            fcntl.flock(mutex, fcntl.LOCK_EX)
            host = (urlsplit(url).hostname or "").lower()
            provider = any(
                host == domain or host.endswith("." + domain)
                for domain in ("thedogs.com.au", "sportsbet.com.au")
            )
            auxiliary = host == "api.open-meteo.com"
            network_path = scope.session / "network-count.json"
            network = (
                json.loads(network_path.read_bytes())
                if network_path.exists()
                else {
                    "provider_started": 0,
                    "auxiliary_started": 0,
                    "unexpected_blocked": 0,
                    "by_host": {},
                    "wire_retries": "UNMEASURED",
                }
            )
            category = (
                "provider_started"
                if provider
                else ("auxiliary_started" if auxiliary else "unexpected_blocked")
            )
            if category == "unexpected_blocked":
                network[category] += 1
                network["by_host"][host] = network["by_host"].get(host, 0) + 1
                atomic_json(network_path, network)
                scope.stop("UNEXPECTED_NETWORK_BLOCKED")
                raise ValueError("unexpected_network_host")
            path = scope.session / "request-count.json"
            count = json.loads(path.read_bytes())["started"] if path.exists() else 0
            if count >= scope.value["max_logical_requests"]:
                scope.stop("REQUEST_CAP_EXHAUSTED")
                raise ValueError("request_cap_exhausted")
            if scope.campaign:
                try:
                    scope.campaign.request()
                except ValueError:
                    scope.stop("CAMPAIGN_REQUEST_CAP_EXHAUSTED")
                    raise
            network[category] += 1
            network["by_host"][host] = network["by_host"].get(host, 0) + 1
            atomic_json(network_path, network)
            atomic_json(path, {"started": count + 1})
        response = original(session, method, url, **kwargs)
        if getattr(response, "status_code", None) in {401, 403, 429}:
            scope.stop("SOURCE_ACCESS_DENIED")
            from utils.http_client import source_retry_headers

            atomic_json(scope.session / ("source-access-denied-" + str(os.getpid()) + ".json"), {
                "host": host,
                "status": response.status_code,
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "retry_headers": source_retry_headers(getattr(response, "headers", {})),
            })
            raise ValueError("source_access_denied")
        return response

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
