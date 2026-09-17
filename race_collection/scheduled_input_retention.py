"""Opt-in bounded retention inside the existing successful WIN-capture lane."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from race_collection.prospective_input_retention import RetentionRejected, failure_code, retain_inputs

ROOT = Path(__file__).resolve().parents[1]
SCOPE = "complete_checkpointed_database_and_captured_sources_machine_only"


def _now():
    return datetime.now(timezone.utc)


class ScheduledInputRetention:
    def __init__(self, *, config_path, evidence_root, protocol_root, collector_run_id, history_source):
        self.raw = b""
        self.config = {}
        self.attempted = 0
        self.config_error = False
        try:
            self.raw = Path(config_path).read_bytes()
            self.config = json.loads(self.raw)
            if not isinstance(self.config, dict):
                raise ValueError("configuration object required")
        except (OSError, ValueError):
            self.config_error = True
        self.context = dict(evidence_root=str(evidence_root), protocol_root=str(protocol_root),
                            collector_run_id=collector_run_id, history_source=str(history_source))

    def __call__(self, *, plan_item, attempt, receipt_publish):
        start = time.monotonic()
        config = self.config
        if self.config_error:
            return {"status": "REJECTED", "reason": "RETENTION_CONFIG_INVALID"}
        authority = config.get("authority", {})
        # This gate precedes any form/history payload access or DB stat/copy.
        if (config.get("schema_version") != "scheduled_input_retention_v1"
                or authority.get("approved") is not True
                or authority.get("scope") != SCOPE
                or not authority.get("approval_reference")
                or plan_item["race_id"] not in authority.get("race_ids", [])
                or authority.get("history_source") != self.context["history_source"]):
            return {"status": "REJECTED", "reason": "HISTORY_ACCESS_NOT_AUTHORIZED"}
        now = _now()
        if not datetime.fromisoformat(authority["not_before"]) <= now < datetime.fromisoformat(authority["expires_at"]):
            return {"status": "REJECTED", "reason": "HISTORY_ACCESS_NOT_AUTHORIZED"}
        jump = datetime.fromisoformat(plan_item["jump_datetime"])
        if (not isinstance(config.get("max_seconds"), (float, int))
                or not math.isfinite(config["max_seconds"]) or config["max_seconds"] <= 0
                or any(type(config.get(key)) is not int or config[key] <= 0 for key in (
                    "cutoff_seconds_before_jump", "max_history_source_bytes", "max_bundle_bytes"))):
            return {"status": "REJECTED", "reason": "RETENTION_CONFIG_INVALID"}
        if self.attempted:
            return {"status": "REJECTED", "reason": "RUN_RETENTION_BUDGET_EXHAUSTED"}
        cutoff = jump - timedelta(seconds=config["cutoff_seconds_before_jump"])
        budget = min(float(config["max_seconds"]), (cutoff - now).total_seconds())
        if budget <= 0 or config["cutoff_seconds_before_jump"] <= 0:
            return {"status": "REJECTED", "reason": "RETENTION_WINDOW_CLOSED"}
        source_bytes = Path(self.context["history_source"]).stat().st_size
        if source_bytes > config["max_history_source_bytes"]:
            return {"status": "REJECTED", "reason": "HISTORY_STORAGE_BUDGET_EXCEEDED"}
        # One consumed attempt per source race and approved configuration.
        identity = hashlib.sha256(json.dumps([plan_item["race_id"], hashlib.sha256(self.raw).hexdigest()]).encode()).hexdigest()
        parent = Path(config["output_root"])
        parent.mkdir(parents=True, exist_ok=True)
        claim = parent / identity
        try:
            claim.mkdir(mode=0o700)
        except FileExistsError:
            return {"status": "REJECTED", "reason": "RETENTION_ALREADY_ATTEMPTED"}
        self.attempted += 1
        request = dict(config=config, config_sha256=hashlib.sha256(self.raw).hexdigest(),
                       context=self.context, plan_item=plan_item, attempt=attempt,
                       receipt_publish=receipt_publish, cutoff=cutoff.isoformat())
        request_path = claim / "request.json"
        request_path.write_text(json.dumps(request))
        worker = subprocess.Popen(
            [sys.executable, "-B", "-m", "race_collection.scheduled_input_retention", str(request_path)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            worker.wait(timeout=budget)
            result_path = claim / "worker-result.json"
            result = json.loads(result_path.read_bytes()) if result_path.is_file() else {"status": "REJECTED", "reason": "RETENTION_WORKER_FAILED"}
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait()
            result = {"status": "REJECTED", "reason": "RETENTION_TIME_BUDGET_EXCEEDED"}
        # No bundle is usable without the parent's terminal acceptance receipt.
        accepted_at = _now()
        if accepted_at >= cutoff:
            result = {"status": "REJECTED", "reason": "RETENTION_WINDOW_CLOSED"}
        result.update(elapsed_seconds=round(time.monotonic()-start, 3), history_source_bytes=source_bytes,
                      config_sha256=request["config_sha256"], accepted_at=accepted_at.isoformat())
        if result["status"] != "RETAINED":
            import shutil
            shutil.rmtree(claim / "bundle", ignore_errors=True)
            # The child may have died before removing its private scratch copy.
            for path in claim.glob(".input-stage-*"):
                shutil.rmtree(path, ignore_errors=True)
        (claim / "terminal.json").write_text(json.dumps(result, sort_keys=True) + "\n")
        return result


def _retain(request_path: Path):
    from race_collection.manual_prediction_collector_request import ManualPredictionCollectorProtocol
    from race_collection.scheduled_forward_corpus import _scheduled_handoff
    from scripts.capture_thedogs_market_history import verify_primary_race_page_evidence

    request = json.loads(request_path.read_bytes())
    config, context = request["config"], request["context"]
    protocol = ManualPredictionCollectorProtocol(context["protocol_root"])
    handoff, plan = _scheduled_handoff(
        protocol=protocol, plan_item=request["plan_item"], attempt=request["attempt"],
        receipt_publish=request["receipt_publish"], collector_run_id=context["collector_run_id"], emitted_at=_now(),
    )
    metadata = json.loads(handoff["_sidecar_bytes"])
    raw_form = Path(metadata["raw_export_path"])
    # Scheduled odds refresh can mirror a sidecar while its authenticated raw
    # export remains in the original download lane. Retain those original bytes.
    if (raw_form.is_symlink() or not raw_form.resolve().is_relative_to(Path(context["evidence_root"]).resolve())
            or raw_form.parent.name != "raw_exports"
            or raw_form.name != Path(handoff["_form_path"]).name):
        raise RetentionRejected("RAW_SOURCE_BINDING_INVALID")
    raw = raw_form.read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata["raw_content_sha256"] or len(raw) != metadata["raw_content_length"]:
        raise RetentionRejected("RAW_SOURCE_BINDING_INVALID")
    page = metadata["primary_race_page_evidence"]
    source_root = raw_form.parent.parent
    verify_primary_race_page_evidence(artifact_root=source_root, reference=page)
    files = {role: (Path(spec["path"]), spec["sha256"]) for role, spec in config["static_files"].items()}
    files.update({
        "normalized_form": (Path(handoff["_form_path"]), handoff["source_form_sha256"]),
        "form_metadata": (Path(handoff["_sidecar_path"]), handoff["source_sidecar_sha256"]),
        "odds_report": (Path(handoff["_report_path"]), handoff["source_report_sha256"]),
        "raw_form": (raw_form, metadata["raw_content_sha256"]),
        "primary_page": (source_root / page["raw_path"], page["body_sha256"]),
        "primary_page_receipt": (source_root / page["receipt_path"], page["receipt_sha256"]),
    })
    exact = protocol.collector_exact_receipt_path(plan["race_id"], handoff["capture_attempt_sha256"])
    files["exact_odds_receipt"] = (exact, hashlib.sha256(exact.read_bytes()).hexdigest())
    if sum(path.stat().st_size for path, _ in files.values()) > config["max_bundle_bytes"]:
        raise RetentionRejected("BUNDLE_STORAGE_BUDGET_EXCEEDED")
    observed = max(datetime.fromisoformat(metadata["metadata_captured_at"].replace("Z", "+00:00")), datetime.fromisoformat(handoff["append_timestamp"]))
    manifest = retain_inputs(
        destination=request_path.parent / "bundle", race_id=plan["race_id"],
        runner_names=[str(r.get("dog_name") or r.get("display_name") or "") for r in plan["expected_runners"]],
        observed_at=observed, prediction_cutoff=datetime.fromisoformat(request["cutoff"]),
        jump_at=datetime.fromisoformat(plan["jump_datetime"]),
        history_source=Path(context["history_source"]), files=files,
        generate_features=True, max_bundle_bytes=config["max_bundle_bytes"],
        clock=_now,
        authorization_config_sha256=request["config_sha256"],
    )
    completion = json.loads((request_path.parent / "bundle/completion.json").read_bytes())
    return {"status": "RETAINED", "bundle_bytes": manifest["bundle_bytes"],
            "feature_values_sha256": manifest["feature_values_sha256"],
            "manifest_sha256": completion["manifest_sha256"],
            "eligibility_assessed": False, "predictions_generated": False}


if __name__ == "__main__":
    path = Path(sys.argv[1])
    try:
        result = _retain(path)
    except Exception as error:
        result = {"status": "REJECTED", "reason": failure_code(error)}
    (path.parent / "worker-result.json").write_text(json.dumps(result) + "\n")
