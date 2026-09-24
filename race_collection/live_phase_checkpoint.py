import hashlib
import json
import os
import fcntl
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


@contextmanager
def native_publication_lock(evidence_root: Path, *, exclusive: bool, timeout_seconds: float = 5.0):
    """Serialize short native publication groups with their local observer."""
    path = Path(evidence_root) / 'shadow_autopilot_daemon_runtime' / 'native-publication.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('native_publication_lock_not_regular')
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('native_publication_lock_timeout')
                time.sleep(.005)
        yield
    finally:
        os.close(descriptor)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PhaseCheckpoint:
    def __init__(self, path: Path, *, identity: str, cycle_id: str, output_dir: Path):
        self.path = path
        previous = json.loads(path.read_text()) if path.exists() else None
        if previous and any(phase["status"] == "STARTED" for phase in previous["phases"]):
            raise ValueError("live_phase_interrupted_requires_reconciliation")
        if previous and previous.get("status") == "RUNNING":
            if previous.get("identity") != identity:
                raise ValueError("live_phase_identity_changed")
            for phase in previous["phases"]:
                result_path = Path(phase["result_path"])
                result_path.resolve().relative_to(Path(previous["output_dir"]).resolve())
                if hashlib.sha256(result_path.read_bytes()).hexdigest() != phase["result_sha256"]:
                    raise ValueError("live_phase_retained_result_changed")
                if (
                    phase["budget_exceeded"]
                    or json.loads(result_path.read_text()).get("status") != "PASS"
                ):
                    raise ValueError("live_phase_failed_boundary_requires_reconciliation")
            self.value = previous
        else:
            self.value = {
                "schema_version": "collector_live_phase_checkpoint_v1",
                "identity": identity,
                "cycle_id": cycle_id,
                "output_dir": str(output_dir),
                "status": "RUNNING",
                "phases": [],
                "pending": [],
                "maintenance": "DEFERRED_LIVE_FRESHNESS_PRIORITY",
            }

    def begin(self, kind: str, inputs: dict[str, Any], observed_at: str) -> dict[str, Any]:
        if self.value["status"] != "RUNNING":
            raise ValueError("live_phase_terminal_checkpoint")
        if any(phase["status"] == "STARTED" for phase in self.value["phases"]):
            raise ValueError("live_phase_already_started")
        phase = {
            "number": len(self.value["phases"]),
            "kind": kind,
            "status": "STARTED",
            "inputs": inputs,
            "started_at": observed_at,
            "inputs_sha256": hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest(),
        }
        self.value["phases"].append(phase)
        atomic_json(self.path, self.value)
        return phase

    def complete(self, result: dict[str, Any], *, elapsed: float, overrun: bool) -> None:
        phase = self.value["phases"][-1]
        if phase["status"] != "STARTED":
            raise ValueError("live_phase_not_started")
        result_path = Path(self.value["output_dir"]) / f"phase-{phase['number']}-result.json"
        if result_path.exists():
            raise ValueError("live_phase_result_already_exists")
        atomic_json(result_path, result)
        phase.update(
            status="COMPLETE",
            result_path=str(result_path),
            result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
            elapsed_seconds=elapsed,
            budget_exceeded=overrun,
        )
        atomic_json(self.path, self.value)

    def finish(self, status: str) -> None:
        self.value["status"] = status
        atomic_json(Path(self.value["output_dir"]) / "phase-checkpoint.json", self.value)
        atomic_json(self.path, self.value)
