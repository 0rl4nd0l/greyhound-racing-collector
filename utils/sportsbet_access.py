"""Durable source admission, independent of lane, launch, and campaign directories.

No request is made here. An unresolved access basis never becomes permission by
waiting. The fallback is our engineering policy, not a provider recovery promise.
"""

from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import fcntl
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid


FALLBACK_SECONDS = 1800
FALLBACK_CAP_SECONDS = 7200


class SportsbetAccessBlocked(RuntimeError):
    pass


def state_path():
    return Path(os.environ.get(
        "GREYHOUND_SPORTSBET_ACCESS_STATE",
        str(Path.home() / ".local/state/greyhound/sportsbet-access.json"),
    )).absolute()


def is_sportsbet(url):
    host = (urlsplit(url).hostname or "").lower()
    return host == "sportsbet.com.au" or host.endswith(".sportsbet.com.au")


class SportsbetAccess:
    def __init__(self, path=None, *, clock=None):
        self.path = Path(path) if path is not None else state_path()
        self.clock = clock or time.time

    @contextmanager
    def locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            # Fail closed rather than letting service deadlines queue source work.
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise SportsbetAccessBlocked("sportsbet_source_busy") from error
            yield

    def read(self):
        try:
            value = json.loads(self.path.read_bytes())
            valid = (
                value["schema"] == "sportsbet_access_v1"
                and value["access_basis"]["status"] in {"permitted", "unresolved", "prohibited"}
                and bool(value["access_basis"]["reference"])
                and value["phase"] in {"OPEN", "COOLDOWN", "RECOVERY", "STOP"}
                and type(value["recovery_attempts"]) is int and 0 <= value["recovery_attempts"] <= 1
                and isinstance(value["denials"], list)
                and math.isfinite(value["not_before"])
                and (value["active"] is None or isinstance(value["active"], str))
            )
            if not valid:
                raise ValueError("invalid_source_state")
            policy = value.get("operating_policy")
            if policy is not None:
                if not policy.get("reference") or any(
                    type(policy.get(key)) is not int or not 0 < policy[key] <= ceiling
                    for key, ceiling in (("python_per_60_seconds", 10),
                                         ("browser_per_60_seconds", 1),
                                         ("browser_navigation_cap", 2))
                ):
                    raise ValueError("invalid_operating_policy")
                operations = value.get("operations", [])
                if not isinstance(operations, list) or any(
                    row["kind"] not in {"python", "browser"} or not math.isfinite(row["at"])
                    for row in operations
                ):
                    raise ValueError("invalid_operation_accounting")
            return value
        except (OSError, ValueError, TypeError, KeyError, AssertionError) as error:
            raise SportsbetAccessBlocked("sportsbet_access_state_missing_or_invalid") from error

    def write(self, value):
        temporary = self.path.with_name(self.path.name + "." + uuid.uuid4().hex)
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)
        directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def initialize(self, *, access_basis):
        if access_basis.get("status") not in {"permitted", "unresolved", "prohibited"} or not access_basis.get("reference"):
            raise ValueError("explicit_access_basis_required")
        with self.locked():
            if self.path.exists():
                raise FileExistsError("sportsbet_access_state_already_exists")
            self.write({
                "schema": "sportsbet_access_v1", "access_basis": access_basis,
                "phase": "OPEN", "not_before": 0, "recovery_attempts": 0,
                "active": None, "denials": [],
                "fallback_policy": "engineering_30min_floor_2h_cap_one_recovery",
            })

    def blocks_restoration(self):
        try:
            value = self.read()
        except SportsbetAccessBlocked:
            return True
        return (value["access_basis"]["status"] != "permitted"
                or value["phase"] != "OPEN" or value["active"] is not None)

    def check_admission(self):
        """Read-only timer preflight; actual transport still claims under lock."""
        value = self.read()
        if (value["access_basis"]["status"] != "permitted" or value["active"] is not None
                or value["phase"] in {"STOP", "RECOVERY"}
                or (value["phase"] == "COOLDOWN" and
                    (self.clock() < value["not_before"] or value["recovery_attempts"]))):
            raise SportsbetAccessBlocked("sportsbet_source_hold")

    def retain_denial(self, status, headers=None, *, reason="retained_evidence", observed_at=None):
        """Import retained operational evidence without making a source request."""
        with self.locked():
            value = self.read()
            self._denial(value, status, headers, reason, observed_at=observed_at)

    def _denial(self, value, status, headers, reason, *, observed_at):
        from utils.http_client import source_retry_headers

        now = self.clock()
        guidance = source_retry_headers(headers)
        raw = guidance.get("retry-after")
        deadline = None
        if raw:
            try:
                if raw.isdigit():
                    deadline = now + int(raw)
                else:
                    stamp = parsedate_to_datetime(raw)
                    if stamp.tzinfo is not None:
                        deadline = stamp.timestamp()
                        # A server Date ahead/behind our clock must not shorten its interval.
                        if guidance.get("date"):
                            server = parsedate_to_datetime(guidance["date"])
                            if server.tzinfo is not None:
                                deadline = max(deadline, now + stamp.timestamp() - server.timestamp())
                if deadline is not None and not math.isfinite(deadline):
                    deadline = None
            except (ValueError, TypeError, OverflowError):
                deadline = None
        fallback = min(FALLBACK_CAP_SECONDS, FALLBACK_SECONDS * 2 ** min(len(value["denials"]), 2))
        value["not_before"] = max(value["not_before"], now + fallback, deadline or 0)
        value["denials"].append({
            "observed_at_epoch": observed_at, "recorded_at_epoch": now,
            "status": status, "retry_headers": guidance,
            "provider_not_before_epoch": deadline, "fallback_seconds": fallback,
            "reason": reason,
        })
        unclear_reset = any(key in guidance for key in ("ratelimit-reset", "x-ratelimit-reset"))
        value["phase"] = (
            "STOP" if status != 429 or value["recovery_attempts"] or value["phase"] == "STOP" or unclear_reset
            else "COOLDOWN"
        )
        self.write(value)

    @contextmanager
    def operation(self, kind):
        with self.locked():
            value = self.read()
            if value["access_basis"]["status"] != "permitted":
                raise SportsbetAccessBlocked("sportsbet_access_basis_" + value["access_basis"]["status"])
            if value["active"] is not None or value["phase"] in {"RECOVERY", "STOP"}:
                raise SportsbetAccessBlocked("sportsbet_hold_or_interrupted_operation")
            recovery = value["phase"] == "COOLDOWN"
            if recovery and (self.clock() < value["not_before"] or value["recovery_attempts"]):
                raise SportsbetAccessBlocked("sportsbet_cooldown")
            policy = value.get("operating_policy")
            if policy is not None:
                now = self.clock()
                operations = value.setdefault("operations", [])
                recent = [row for row in operations if row["at"] > now - 60]
                limit = policy.get(kind + "_per_60_seconds", 0)
                if (kind not in {"python", "browser"} or len(operations) >= 512
                        or sum(row["kind"] == kind for row in recent) >= limit
                        or any(row["at"] > now for row in operations)):
                    value["phase"] = "STOP"
                    value["policy_stop"] = {"at": now, "kind": kind,
                                            "reason": "operating_policy_limit_or_clock"}
                    self.write(value)
                    raise SportsbetAccessBlocked("sportsbet_operating_policy_stop")
                operations.append({"at": now, "kind": kind})
            if recovery:
                value["recovery_attempts"] += 1
                value["phase"] = "RECOVERY"
            value["active"] = kind + ":" + uuid.uuid4().hex
            self.write(value)  # Durable consumption before any transport call.
            operation = SourceOperation(self, value, recovery)
            try:
                yield operation
            except BaseException:
                operation.failed = True
                raise
            finally:
                if operation.failed or (recovery and not operation.success):
                    value["phase"] = "STOP"
                elif recovery and value["phase"] == "RECOVERY":
                    value["phase"] = "OPEN"
                value["active"] = None
                self.write(value)


class SourceOperation:
    def __init__(self, gate, value, recovery):
        self.gate, self.value, self.recovery = gate, value, recovery
        self.failed = self.success = self.transport_success = False

    def accept_data(self):
        """Called only after the route's data checks pass, while still owning it."""
        self.check()
        if not self.transport_success:
            raise SportsbetAccessBlocked("sportsbet_validated_data_without_response")
        self.success = True

    def check(self):
        if self.value["phase"] in {"COOLDOWN", "STOP"}:
            raise SportsbetAccessBlocked("sportsbet_source_hold")

    def response(self, status, headers):
        from utils.http_client import source_retry_headers
        guidance = source_retry_headers(headers)
        instructed = any(key in guidance for key in ("retry-after", "ratelimit-reset", "x-ratelimit-reset"))
        if status in {401, 403, 429} or (status >= 400 and instructed):
            self.gate._denial(self.value, status, headers, "source_response", observed_at=self.gate.clock())
        elif 200 <= status < 300:
            self.transport_success = True
        elif self.recovery:
            self.failed = True
