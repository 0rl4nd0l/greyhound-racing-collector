"""Feature-only replay of retained bytes. No scorer or live-input fallback."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

def generate_retained_features(root: Path, files: dict) -> bytes:
    """The worker imports generator code from the retained source ZIP only."""
    request = {role: entry["path"] for role, entry in files.items()}
    worker = (root / request["feature_replay_worker"]).resolve()
    if root.resolve() not in worker.parents:
        raise ValueError("REPLAY_WORKER_PATH_INVALID")
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(worker), str(root.resolve())],
        input=json.dumps(request).encode(), stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, timeout=30, check=False,
    )
    if result.returncode != 0:
        phase = json.loads(result.stdout).get("failure_phase") if result.stdout else None
        if phase not in {"INPUT_SETUP", "ENVIRONMENT_CHECK", "GENERATOR_IMPORT", "FEATURE_GENERATION", "FEATURE_PROJECTION"}:
            phase = "UNKNOWN"
        raise ValueError("RETAINED_FEATURE_GENERATION_FAILED:" + phase)
    # Validate canonical machine output, without displaying any feature values.
    rows = json.loads(result.stdout)
    if not isinstance(rows, list) or not rows:
        raise ValueError("RETAINED_FEATURES_EMPTY")
    return result.stdout


def replay_retained_inputs(root: Path) -> dict:
    """Verify all bindings, replay in a fresh process and return only a digest."""
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    completion = json.loads((root / "completion.json").read_bytes())
    if hashlib.sha256(raw).hexdigest() != completion["manifest_sha256"]:
        raise ValueError("RETENTION_MANIFEST_CHANGED")
    from datetime import datetime

    if manifest.get("authorization_config_sha256") is not None:
        terminal = json.loads((root.parent / "terminal.json").read_bytes())
        if (terminal.get("status") != "RETAINED"
                or terminal.get("manifest_sha256") != completion["manifest_sha256"]
                or terminal.get("config_sha256") != manifest["authorization_config_sha256"]
                or not datetime.fromisoformat(terminal["accepted_at"]) < datetime.fromisoformat(manifest["prediction_cutoff"])):
            raise ValueError("SCHEDULED_RETENTION_NOT_ACCEPTED")
    if not datetime.fromisoformat(completion["inputs_sealed_at"]) < datetime.fromisoformat(manifest["prediction_cutoff"]):
        raise ValueError("RETENTION_COMPLETED_LATE")
    for entry in list(manifest["files"].values()) + [manifest["history"]]:
        path = (root / entry["path"]).resolve()
        if root.resolve() not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("RETAINED_INPUT_CHANGED")
    generated = generate_retained_features(root, manifest["files"])
    digest = hashlib.sha256(generated).hexdigest()
    if digest != manifest["feature_values_sha256"] or (root / "feature_values.json").read_bytes() != generated:
        raise ValueError("RETAINED_FEATURE_REPLAY_MISMATCH")
    return {"status": "IDENTICAL_FEATURE_REPLAY", "feature_values_sha256": digest}
