"""Finite, outcome-blind use of the existing retained-input R3 worker.

The collector lock is never taken here. The caller supplies one already consumed
capture claim; a campaign-wide exclusive race record forbids retry/substitution.
No journal, corpus admission, result acquisition or evaluation is configured.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from race_collection.live_freshness_contract import FreshnessContract, create_once
from race_collection.live_phase_checkpoint import atomic_json

OPERATION = "operational_prediction"


def prepare_retention(output, source, python):
    """Pin the same generator, model and complete environment used by the worker."""
    import subprocess
    import zipfile
    archive = output / "operational-generator.zip"
    paths = [source / name for name in ("scripts/__init__.py", "scripts/utils.py",
        "scripts/run_shadow_non_tgr_rf_evaluation.py", "scripts/run_feature_recovery_execution_v1.py")]
    paths += list((source / "utils").glob("*.py")) + list((source / "config").glob("*.py"))
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(paths):
            handle.write(path, path.relative_to(source))
    lock = output / "operational-environment.json"
    raw = subprocess.check_output([str(python), "-c", "import json,platform,importlib.metadata as m; print(json.dumps({'python':platform.python_version(),'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"])
    lock.write_bytes(raw)
    sources = {"model": source / "artifacts/frozen_models/market_form_residual_v1/model.json",
        "model_manifest": source / "artifacts/frozen_models/market_form_residual_v1/manifest.json",
        "configuration": source / "configs/prediction/manual-default.json",
        "feature_schema": source / "accuracy_program/repaired_non_tgr_schema.json",
        "feature_replay_worker": source / "scripts/retained_feature_worker.py",
        "generator_source_archive": archive, "environment_lock": lock}
    config = {"schema_version": "scheduled_input_retention_v1", "max_seconds": 90,
        "cutoff_seconds_before_jump": 60, "max_history_source_bytes": 4_000_000_000,
        "max_bundle_bytes": 200_000_000,
        "static_files": {key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for key, path in sources.items()}}
    create_once(output / "operational-retention.json", config)
    return hashlib.sha256((output / "operational-retention.json").read_bytes()).hexdigest()


class Supervisor:
    """At most one concurrent bounded prediction, with durable race consumption."""
    def __init__(self, output, plan, scope):
        self.output, self.plan, self.scope = output, plan, scope
        self.child = None
        self.log = None

    def tick(self):
        import subprocess
        if self.child is not None:
            if self.child.poll() is None:
                return
            self.log.close()
            self.child = None
        if not self.plan.get("operational_predictions") or (self.scope.end-now()).total_seconds() < 200:
            return
        from race_collection.live_freshness_contract import AttemptAllowance
        for claim in AttemptAllowance(self.scope).claims():
            verification = self.output / "capture-verifications" / (claim.parent.name + ".json")
            if not verification.exists():
                continue
            item = json.loads(claim.read_bytes())["item"]
            root = self.scope.campaign.root / "operational-predictions"
            identity = hashlib.sha256(item["race_id"].encode()).hexdigest()
            if (root / "races" / identity).exists() or (root / "dispatches" / (identity + ".json")).exists():
                continue
            self.scope.admit(now(), seconds=200)
            create_once(root / "dispatches" / (identity + ".json"), {
                "race_id": item["race_id"], "capture_claim": str(claim), "dispatched_at": now().isoformat()})
            self.log = (root / "dispatches" / (identity + ".log")).open("xb")
            self.child = subprocess.Popen([self.plan["python"], "-B", "-m", "race_collection.operational_prediction",
                str(self.output / "plan.json"), str(claim)], cwd=self.plan["source_root"],
                stdout=self.log, stderr=self.log)
            return

    def drain(self):
        if self.child is not None:
            self.child.wait(timeout=240)
            self.log.close()
            self.child = None


def now():
    return datetime.now(timezone.utc)


def run(plan_path: Path, claim_path: Path):
    from race_collection.manual_prediction_collector_request import ManualPredictionCollectorProtocol
    from race_collection.scheduled_input_retention import ScheduledInputRetention, SCOPE
    from race_collection.synchronous_manual_capture import bounded_current_race_index
    from src.operator_ui.job_store import JobInput, JobStore, OperationalIndexProvenance, Phase, resolve_audit_confirmation
    from src.operator_ui.prediction_worker import WorkerConfig, ServerChoice, run_once
    from src.operator_ui.r3_api import finalize_producer_bundle
    from src.predictor.on_demand import resolve_model, sha256_file, canonical_bytes
    from scripts.run_freshness_rehearsal import verify_claim_receipt

    started = time.monotonic()
    plan = json.loads(plan_path.read_bytes())
    scope = FreshnessContract.load(plan_path.parent / "contract.json")
    scope.admit(now(), seconds=180)
    if not plan.get("operational_predictions") or scope.campaign is None:
        raise ValueError("operational_prediction_not_authorized")
    claim_path = claim_path.resolve(strict=True)
    claim_path.relative_to(scope.session / "captures")
    reserved = json.loads(claim_path.read_bytes())
    item = reserved["item"]
    race_id = item["race_id"]
    root = scope.campaign.root / "operational-predictions"
    record = root / "races" / hashlib.sha256(race_id.encode()).hexdigest()
    record.mkdir(parents=True, exist_ok=False, mode=0o700)
    create_once(record / "identity.json", {"race_id": race_id, "capture_claim": str(claim_path),
                "plan_sha256": sha256_file(plan_path), "operation": OPERATION, "started_at": now().isoformat()})
    timing = {"started_at": now().isoformat(), "operation": OPERATION, "race_id": race_id}
    stage = "receipt_validation"
    try:
        evidence = Path(plan["evidence_root"])
        protocol_root = evidence / "manual_prediction_collector_requests_v1"
        protocol = ManualPredictionCollectorProtocol(protocol_root)
        handoff = protocol.discover_collector_exact_handoff(race_id=race_id, current_time=now(), max_age_seconds=300)
        if handoff is None:
            raise ValueError("exact_receipt_unavailable")
        verify_claim_receipt(claim_path, handoff, evidence, Path(plan["source_root"]))
        source = json.loads(handoff["_report_bytes"])
        jump = datetime.fromisoformat(item["race_identity"]["jump_datetime"])
        timing["price_observed_at"] = handoff["append_timestamp"]
        timing["receipt_validation_seconds"] = time.monotonic() - started
        stage = "retention"
        phase_start = time.monotonic()
        config = json.loads((plan_path.parent / "operational-retention.json").read_bytes())
        config.update(output_root=str(record / "retention"), authority={
            "approved": True, "scope": SCOPE, "approval_reference": plan["operational_predictions"]["authorization"],
            "race_ids": [race_id], "history_source": plan["db_path"],
            "not_before": plan["starts_at"], "expires_at": plan["ends_at"],
        })
        config_path = record / "retention-config.json"
        create_once(config_path, config)
        retainer = ScheduledInputRetention(config_path=config_path, evidence_root=evidence,
            protocol_root=protocol_root, collector_run_id=source["collector_run_id"], history_source=plan["db_path"])
        # Replay the authenticated scheduled handoff, not a mutable capture plan.
        retained = retainer(plan_item=source["source_plan_item"], attempt=source["source_attempt"], receipt_publish=None)
        timing["retention_seconds"] = time.monotonic() - phase_start
        if retained["status"] != "RETAINED":
            raise ValueError(retained["reason"])
        bundles = list((record / "retention").glob("*/bundle"))
        if len(bundles) != 1:
            raise ValueError("retained_bundle_ambiguous")
        stage = "job_admission"
        scope.admit(now(), seconds=100)
        index_path = evidence / "shadow_autopilot_daemon_runtime/manual_prediction_current_race_index.json"
        view = bounded_current_race_index(current_time=now(), timeout_seconds=5, index_path=index_path,
            evidence_root=evidence, max_age_seconds=300, return_verified_view=True)
        observed = datetime.fromisoformat(view.source_generated_at)
        if not 0 <= (now() - observed).total_seconds() <= 300:
            raise ValueError("CURRENT_INDEX_STALE")
        races = [r for r in view.races if r["race_id"] == race_id]
        if len(races) != 1 or races[0]["jump_datetime"] != item["race_identity"]["jump_datetime"]:
            raise ValueError("captured_race_changed")
        race = races[0]
        if race["runner_set_sha256"] != item["race_identity"]["runner_set_sha256"]:
            raise ValueError("captured_runners_changed")
        source_root = Path(plan["source_root"])
        model = resolve_model("latest-research")
        config_path = source_root / "configs/prediction/manual-default.json"
        choice = ServerChoice(config_path, "manual-default", sha256_file(config_path), model.resolved,
            model.model_sha256, model.manifest_sha256, model.schema_sha256, model.model_path, model.manifest_path, model.schema_path)
        inp = JobInput(race_id, race["jump_datetime"], race["runner_set_sha256"], "latest-research", model.resolved,
            model.model_sha256, model.manifest_sha256, model.schema_sha256, "manual-default", choice.config_sha256,
            "receipt", tuple({"box": r["box"], "name": r["display_name"], "identity": r["identity"],
                              "source_native_runner_id": r["source_native_runner_id"]} for r in race["runners"]),
            OperationalIndexProvenance.from_verified_current_race_index(view), retained["manifest_sha256"])
        authority = object()
        store = JobStore(root / "jobs.sqlite3", separate_from=(Path(plan["db_path"]),), verifier_authority=authority)
        def confirm(intent):
            raw = canonical_bytes(intent)
            digest = hashlib.sha256(raw).hexdigest()
            path = root / "audit" / (digest + ".json")
            if not path.exists():
                create_once(path, intent)
            return resolve_audit_confirmation(intent, digest)
        worker = WorkerConfig(Path(plan["python"]), source_root, {"latest-research": choice}, Path(plan["db_path"]),
            root / "bundles", (evidence,), protocol_root, index_path, evidence, 5, 45, 90, 2,
            retained_input_bindings={race_id: {"path": str(bundles[0]), "manifest_sha256": retained["manifest_sha256"]}})
        job = store.create(actor_identity="operational-campaign", actor_level=2, operation=OPERATION,
            idempotency_key=hashlib.sha256(race_id.encode()).hexdigest(), job_input=inp, now=now(), confirm_audit=confirm)
        create_once(record / "job.json", {"job_id": job.job_id, "input_sha256": inp.identity_sha256})
        for phase, status in ((Phase.VALIDATED, "VALID"), (Phase.WAITING_FOR_CLAIM, "WAITING")):
            job = store.transition(job.job_id, phase, now=now(), status=status, reason="operational_admission", confirm_audit=confirm)
        timing.update(index_observed_at=view.source_generated_at, index_age_at_dispatch_seconds=(now()-observed).total_seconds(),
                      seconds_to_jump_at_dispatch=(jump-now()).total_seconds())
        stage = "prediction_subprocess"
        phase_start = time.monotonic()
        job = run_once(store, job.job_id, worker, now=now, confirm_audit=confirm)
        timing["prediction_subprocess_seconds"] = time.monotonic() - phase_start
        stage = "bundle_verification"
        phase_start = time.monotonic()
        job = finalize_producer_bundle(root / "bundles", store, job, capability=authority, now=now(), confirm_audit=confirm)
        timing["bundle_verification_seconds"] = time.monotonic() - phase_start
        timing.update(status=job.phase.value, job_id=job.job_id, seconds_to_jump_at_verification=(jump-now()).total_seconds(),
                      price_to_verification_seconds=(now()-datetime.fromisoformat(handoff["append_timestamp"])).total_seconds(),
                      index_to_verification_seconds=(now()-observed).total_seconds())
        if job.phase is not Phase.PREDICTION_READY:
            timing["reason"] = job.reason
    except Exception as exc:
        timing.update(status="FAILED", stage=stage, reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
    finally:
        timing.update(total_seconds=time.monotonic()-started, completed_at=now().isoformat())
        atomic_json(record / "terminal.json", timing)
    return timing


if __name__ == "__main__":
    # Predictions must never acquire remote data, including through libraries.
    from scripts.check_freshness_service import deny_network
    deny_network()
    os.umask(0o077)
    result = run(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({k: result[k] for k in ("status", "total_seconds")}))
    raise SystemExit(0 if result["status"] == "PREDICTION_READY" else 2)
