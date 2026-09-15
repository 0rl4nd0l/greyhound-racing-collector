"""Default-off, in-process coordination of the existing R3 prediction lane."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .job_store import Phase, RaceAlreadyRecorded, TERMINAL_PHASES, canonical, utc_text
from .r3_api import (
    R3Rejected,
    confirm_prediction_operation,
    dispatch_prediction,
    submit_prediction,
    verified_prediction_result,
)


@dataclass(frozen=True)
class JournalActivation:
    """Operator-reviewed continuation identity, never an experimental protocol."""

    activation_id: str
    not_before: datetime
    admit_until: datetime
    maximum_jobs: int
    source_commit: str
    protocol_sha256: str
    model_sha256: str
    config_sha256: str
    excluded_race_ids: tuple[str, ...]

    @classmethod
    def from_fields(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != set(cls.__dataclass_fields__) | {"schema"}
            or value["schema"] != "operator_ui_journal_activation_v1"
        ):
            raise ValueError("invalid activation schema")
        fields = {key: item for key, item in value.items() if key != "schema"}
        for key in ("not_before", "admit_until"):
            fields[key] = datetime.fromisoformat(fields[key].replace("Z", "+00:00"))
        if not isinstance(fields["excluded_race_ids"], list):
            raise ValueError("invalid activation exclusions")
        fields["excluded_race_ids"] = tuple(fields["excluded_race_ids"])
        activation = cls(**fields)
        if activation.fields() != value:
            raise ValueError("activation is not canonical")
        return activation

    def fields(self):
        if str(uuid.UUID(self.activation_id)) != self.activation_id:
            raise ValueError("invalid activation identity")
        for value, size in (
            (self.source_commit, 40),
            (self.protocol_sha256, 64),
            (self.model_sha256, 64),
            (self.config_sha256, 64),
        ):
            if (
                not isinstance(value, str)
                or len(value) != size
                or set(value) - set("0123456789abcdef")
            ):
                raise ValueError("invalid activation binding")
        start, end = utc_text(self.not_before), utc_text(self.admit_until)
        if self.admit_until <= self.not_before:
            raise ValueError("invalid admission window")
        if type(self.maximum_jobs) is not int or not 1 <= self.maximum_jobs <= 64:
            raise ValueError("invalid job allowance")
        if (
            not isinstance(self.excluded_race_ids, tuple)
            or len(set(self.excluded_race_ids)) != len(self.excluded_race_ids)
            or any(not isinstance(r, str) or not r for r in self.excluded_race_ids)
        ):
            raise ValueError("invalid retained race exclusions")
        return {
            **asdict(self),
            "not_before": start,
            "admit_until": end,
            "excluded_race_ids": list(self.excluded_race_ids),
            "schema": "operator_ui_journal_activation_v1",
        }


def _publish(path: Path, value: dict) -> None:
    """Publish complete immutable bytes; a competing writer cannot overwrite."""
    raw = canonical(value)
    descriptor, temporary = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("unsafe journal record")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or canonical(value) != raw:
        raise ValueError("invalid journal record")
    return value


def start_journal_coordinator(coordinator, logger):
    """Bind recurrence to R3's process; no separate scheduler or prediction lane."""
    if coordinator.activation is None:
        return None
    # Persist the future-only identity before starting any recurring observation.
    coordinator.tick()
    stop = threading.Event()

    def observe():
        while not stop.wait(60):
            try:
                report = coordinator.tick()
                logger.info("R3 journal observation: %s", json.dumps(report, sort_keys=True))
            except R3Rejected as exc:
                logger.info("R3 journal source pending: %s", exc.classification)
            except Exception as exc:
                logger.error("R3 journal stopped for inspection: %s", type(exc).__name__)
                return

    threading.Thread(target=observe, name="operator-ui-r3-journal", daemon=True).start()
    return stop


class JournalCoordinator:
    """One bounded observation cycle; construction never starts background work."""

    def __init__(
        self,
        *,
        activation=None,
        root=None,
        services=None,
        audit=None,
        races=None,
        results=None,
        clock=lambda: datetime.now(timezone.utc),
    ):
        self.activation, self.root, self.clock = activation, root, clock
        self.services, self.audit, self.races = services, audit, races
        self.results = results

    def tick(self):
        if self.activation is None:
            return {"state": "DISABLED"}
        activation = self.activation.fields()
        root = Path(self.root).absolute()
        if root.resolve() != root:
            raise ValueError("unsafe journal root")
        path = root / "activation.json"
        now = self.clock()
        utc_text(now)
        if not path.exists():
            if now >= self.activation.not_before:
                raise ValueError("first activation requires a future cutoff")
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                _publish(path, activation)
            except FileExistsError:
                pass
        if _read(path) != activation:
            raise ValueError("persisted activation differs")
        if now < self.activation.not_before:
            return {"state": "WAITING_START"}
        descriptor = os.open(
            root / "coordinator.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {"state": "OWNER_BUSY"}
            return self._cycle(now)
        finally:
            os.close(descriptor)

    def _confirm(self, intent):
        return confirm_prediction_operation(
            self.services,
            self.audit,
            intent,
            session_identifier=self.activation.activation_id,
            client_identity="server-owned-r3-journal",
        )

    def _selection(self, race_id):
        return {
            "race_id": race_id,
            "model_id": "latest-research",
            "config_id": "manual-default",
            "odds_source_id": "receipt",
            "idempotency_key": str(uuid.uuid5(uuid.UUID(self.activation.activation_id), race_id)),
        }

    def _guard(self, job_input):
        jump = datetime.fromisoformat(job_input.jump_timestamp.replace("Z", "+00:00"))
        now = self.clock()
        if (
            job_input.model_sha256 != self.activation.model_sha256
            or job_input.config_sha256 != self.activation.config_sha256
            or job_input.resolved_model_identity != "market_form_residual_v1"
            or job_input.model_selector != "latest-research"
            or job_input.config_id != "manual-default"
            or job_input.odds_source != "receipt"
        ):
            raise R3Rejected("ACTIVATION_MODEL_CONFIG_MISMATCH")
        if not self.activation.not_before <= now < self.activation.admit_until or jump <= max(
            now, self.activation.not_before
        ):
            raise R3Rejected("OUTSIDE_FUTURE_ADMISSION_WINDOW")

    def _cycle(self, now):
        store = self.services.job_store
        jobs = store.recorded_jobs()
        actor = "r3-journal:" + self.activation.activation_id
        owned = [job for job in jobs if job.actor_identity == actor]
        report = {
            "state": "OBSERVED",
            "jobs": [job.job_id for job in owned],
            "admissions": {},
            "closures": {},
            "recovery": {},
        }
        # Reconcile the original queue, never claim or re-launch a consumed attempt.
        for job in owned:
            if (
                job.phase in {Phase.SUBMITTED, Phase.VALIDATED, Phase.WAITING_FOR_CLAIM}
                and not job.attempt_claimed
            ):
                try:
                    self._guard(job.input)
                except R3Rejected as exc:
                    report["recovery"][job.job_id] = exc.classification
                    continue
                submit_prediction(
                    self.services,
                    self._selection(job.input.race_id),
                    identity=actor,
                    confirm_audit=self._confirm,
                    input_guard=self._guard,
                )
            elif job.phase is Phase.PRODUCER_COMPLETED:
                dispatch_prediction(self.services, job, self._confirm)
        owned = [store.get(job.job_id) for job in owned]
        for job in owned:
            if job.phase is Phase.PREDICTION_READY:
                report["closures"][job.job_id] = self._close(job, now)
        if report["recovery"]:
            report["state"] = "STOPPED_UNCLAIMED_ADMISSION"
            return report
        if any(job.phase in TERMINAL_PHASES - {Phase.PREDICTION_READY} for job in owned):
            report["state"] = "STOPPED_AFTER_FAILURE"
            return report
        if any(job.phase not in TERMINAL_PHASES for job in owned):
            report["state"] = "PREDICTION_PENDING"
            return report
        if now >= self.activation.admit_until or len(owned) >= self.activation.maximum_jobs:
            report["state"] = "ADMISSION_CLOSED"
            return report
        recorded = {job.input.race_id for job in jobs} | set(self.activation.excluded_race_ids)
        try:
            races = tuple(self.races())
        except R3Rejected as exc:
            report.update(state="SOURCE_PENDING", reason=exc.classification)
            return report
        if len(races) > 64:
            raise ValueError("unbounded journal index")
        identifiers = [race.get("race_id") for race in races]
        candidates = []
        for race in races:
            race_id = race.get("race_id")
            if not isinstance(race_id, str) or not race_id or identifiers.count(race_id) != 1:
                raise ValueError("ambiguous journal race identity")
            if race_id in recorded:
                report["admissions"][race_id] = "ALREADY_RECORDED"
                continue
            try:
                jump = datetime.fromisoformat(race["jump_datetime"].replace("Z", "+00:00"))
                utc_text(jump)
            except (KeyError, TypeError, ValueError):
                report["admissions"][race_id] = "EXACT_RACE_IDENTITY_UNAVAILABLE"
                continue
            if jump <= max(now, self.activation.not_before):
                report["admissions"][race_id] = "OUTSIDE_FUTURE_ADMISSION_WINDOW"
                continue
            candidates.append((jump, race_id))
        for _, race_id in sorted(candidates):
            try:
                job, _ = submit_prediction(
                    self.services,
                    self._selection(race_id),
                    identity=actor,
                    confirm_audit=self._confirm,
                    input_guard=self._guard,
                )
            except R3Rejected as exc:
                report["admissions"][race_id] = exc.classification
                continue
            except RaceAlreadyRecorded:
                report["admissions"][race_id] = "ALREADY_RECORDED"
                continue
            report["admissions"][race_id] = "ADMITTED"
            report["jobs"].append(job.job_id)
            # One lane, one new admission per cycle. A failure stops the next cycle.
            break
        return report

    def _close(self, job, now):
        path = Path(self.root) / f"{job.job_id}.closure.json"
        bundle = verified_prediction_result(self.services, job)
        if bundle is None:
            return "PREDICTION_VERIFICATION_FAILED"
        identity = {
            "job_id": job.job_id,
            "race_id": job.input.race_id,
            "activation_sha256": hashlib.sha256(canonical(self.activation.fields())).hexdigest(),
            "input_identity_sha256": job.input.identity_sha256,
            "logical_bundle_sha256": bundle.index_entry["logical_bundle_sha256"],
            "prediction_id": bundle.result["prediction_id"],
        }
        if path.exists():
            retained = _read(path)
            if (
                any(retained.get(key) != value for key, value in identity.items())
                or retained.get("official_result_sha256")
                != hashlib.sha256(canonical(retained.get("official_result"))).hexdigest()
            ):
                raise ValueError("closure identity differs")
            if self.results is None:
                return "CLOSURE_RECHECK_PENDING"
            observed = self.results.read(job, bundle, now=now)
            if observed["state"] != "RESULT_AVAILABLE":
                return "CLOSURE_RECHECK_PENDING:" + observed["reason"]
            if retained["official_result"] != observed["evidence"]:
                raise ValueError("closure official evidence differs")
            return "CLOSED"
        jump = datetime.fromisoformat(job.input.jump_timestamp.replace("Z", "+00:00"))
        if now <= jump or self.results is None:
            return "RESULT_PENDING"
        result = self.results.read(job, bundle, now=now)
        if result["state"] != "RESULT_AVAILABLE":
            return (
                result["state"] + ":" + result["reason"]
                if result["state"] == "RESULT_REJECTED"
                else "RESULT_PENDING"
            )
        _publish(
            path,
            {
                **identity,
                "schema": "operator_ui_research_closure_v1",
                "closed_at": utc_text(now),
                "research_only": True,
                "metrics_computed": False,
                "evaluation_eligible": True,
                "official_result": result["evidence"],
                "official_result_sha256": result["evidence_sha256"],
            },
        )
        return "CLOSED"
