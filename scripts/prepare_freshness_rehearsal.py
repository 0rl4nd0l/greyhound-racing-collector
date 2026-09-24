#!/usr/bin/env python3
"""Prepare a finite local operational package. Does not execute or install it."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from race_collection.live_freshness_contract import create_once, digest
from scripts import shadow_autopilot_daemon as daemon
from scripts.check_freshness_runtime import probe_runtime

UNITS = (
    "shadow-autopilot.service",
    "shadow-autopilot.timer",
    "shadow-autopilot-odds-capture.service",
    "shadow-autopilot-odds-capture.timer",
)


def prepare(*, output, start, python, db, lock, reconciliation_roots, installed_dir, campaign_root=None, operational_predictions=False):
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    history_db = db.resolve(strict=True)
    if operational_predictions:
        if campaign_root is None:
            raise ValueError("operational_predictions_require_existing_campaign")
        db = campaign_root.resolve() / "operational-predictions/capture.sqlite3"
        db.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if db.resolve() == history_db:
            raise ValueError("operational_history_write_collision")
        from sportsbet_odds_integrator import SportsbetOddsIntegrator
        SportsbetOddsIntegrator(str(db), allow_auto_scrape_odds=False)
        db.chmod(0o600)
    source = output / "source"
    source.mkdir()
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit], cwd=ROOT, text=True
    ).splitlines()
    identities = {}
    for name in files:
        path = Path(name)
        frozen = operational_predictions and (name in {"artifacts/frozen_models/market_form_residual_v1/model.json", "artifacts/frozen_models/market_form_residual_v1/manifest.json", "accuracy_program/repaired_non_tgr_schema.json", "tests/test_run_shadow_non_tgr_rf_evaluation.py"})
        if path.parts[0] in {"tests", "artifacts", ".git", "docs"} and not frozen:
            continue
        if not frozen and path.suffix != ".py" and not (
            path.parts[0] in {"configs", "config", "ops"}
            and path.suffix in {".json", ".toml", ".yaml", ".yml", ".service", ".timer"}
        ):
            continue
        raw = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        target = source / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        identities[name] = hashlib.sha256(raw).hexdigest()
    identity = {"commit": commit, "tree": tree, "files": identities}
    create_once(source / "SOURCE_IDENTITY.json", identity)
    runtime_identity = probe_runtime(python=python, source_root=source)
    create_once(output / "runtime-identity.json", runtime_identity)
    # Publication retains its root parent's mutation witnesses. Supervisor
    # progress/receipt files change independently, so give the collector a
    # dedicated parent that observers never write into.
    evidence = output / "collector" / "evidence"
    evidence.mkdir(parents=True)
    runtime = evidence / "shadow_autopilot_daemon_runtime"
    runtime.mkdir()
    contract_path = output / "contract.json"
    common = dict(
        service_dir=output / "units",
        repo_path=source,
        python_path=python,
        evidence_root=evidence,
        db_path=db,
        lock_path=lock,
        live_freshness=True,
        live_freshness_profile="bounded80-v1",
        live_freshness_contract=contract_path,
    )
    daemon.write_service_files(
        **common,
        timeout_seconds=600,
        state_path=runtime / "state.json",
        odds_capture_state_path=runtime / "odds_capture_state.json",
        pause_path=None,
    )
    daemon.write_odds_capture_service_files(
        **common,
        timeout_seconds=600,
        state_path=runtime / "odds_capture_state.json",
        refresh_limit=16,
    )
    service_checks = {}
    for name in (UNITS[0], UNITS[2]):
        completed = subprocess.run(
            [
                str(python),
                "-B",
                str(source / "scripts/check_freshness_service.py"),
                "--unit",
                str(output / "units" / name),
            ],
            text=True,
            capture_output=True,
            timeout=45,
        )
        if completed.returncode:
            raise ValueError("generated_service_preflight_failed: " + completed.stderr[-3000:])
        service_checks[name] = json.loads(completed.stdout.splitlines()[-1])
    create_once(output / "service-preflight.json", service_checks)
    # An immutable archive of code/config only; no retained data/model/test fixtures.
    import tarfile

    with tarfile.open(output / "source.tar", "w") as archive:
        for target in sorted(source.rglob("*")):
            if target.is_file():
                archive.add(target, arcname=target.relative_to(output), recursive=False)
    unit_hashes = {
        name: hashlib.sha256((output / "units" / name).read_bytes()).hexdigest() for name in UNITS
    }
    baseline = {
        name: hashlib.sha256((installed_dir / name).read_bytes()).hexdigest()
        for name in (*UNITS, "greyhound-operator-ui-r3.service")
    }
    campaign = None
    if campaign_root is not None:
        from race_collection.freshness_campaign import Campaign
        campaign = Campaign(campaign_root)
    from utils.sportsbet_access import state_path

    plan = {
        "sportsbet_access_state": str(state_path()),
        "baseline_source_coordination_verified": False,
        **({"campaign_root": str(campaign.root),
            "campaign_authorization_sha256": digest(campaign.value)} if campaign else {}),
        "schema_version": "freshness_scheduled_rehearsal_plan_v1",
        "status": "PREPARED_NOT_AUTHORIZED",
        "rehearsal_id": output.name,
        "commit": commit,
        "tree": tree,
        "source_root": str(source),
        "source_identity_sha256": digest(identity),
        "source_archive_sha256": hashlib.sha256((output / "source.tar").read_bytes()).hexdigest(),
        "python": str(python),
        "python_sha256": hashlib.sha256(python.resolve().read_bytes()).hexdigest(),
        "runtime_sha256": digest(runtime_identity),
        "starts_at": start.isoformat(),
        "ends_at": (start + timedelta(minutes=90)).isoformat(),
        "admission_starts_at": (start - timedelta(minutes=30)).isoformat(),
        "cleanup_seconds": 1860 if campaign else 1200,
        "sample_period_seconds": 2,
        "max_sample_gap_seconds": 5,
        "readiness_warmup_seconds": 1200,
        "first_index_deadline_seconds": 180,
        "profile": "bounded80-v1",
        "max_capture_attempts": 12 if campaign else 1,
        "max_logical_requests": 48000 if campaign else 24000,
        "capture_allowance": "PENDING_QUIESCENT_RECONCILIATION",
        "evidence_root": str(evidence),
        "lock_path": str(lock),
        "db_path": str(db),
        "installed_dir": str(installed_dir),
        "unit_sha256": unit_hashes,
        "baseline_unit_sha256": baseline,
        "reconciliation_roots": reconciliation_roots,
        "r3_measurement": "candidate-native-readers; installed R3 binding is not rewritten",
        "stop_on": [
            "scope_failure",
            "phase_or_overhead_failure",
            "request_cap",
            "native_integrity_failure",
            "age_bound_over_270",
            "sample_gap_over_5",
            "clock_discontinuity",
            "source_or_unit_change",
            "consumed_or_changed_capture",
            "native_readiness_failure_after_warmup",
            "unapproved_lock_owner",
            "unattributed_or_overbudget_timer_dispatch",
        ],
        "restoration": "exact four collector files; natural drain and reserved-window closure; keep only collector timers inactive and disabled while source is held or baseline source coordination is unverified; never change R3; record previous timer states",
    }
    if operational_predictions:
        if campaign is None:
            raise ValueError("operational_predictions_require_existing_campaign")
        from race_collection.operational_prediction import prepare_retention
        plan["operational_predictions"] = {
            "authorization": "user:collection-to-prediction-20260924",
            "retention_config_sha256": prepare_retention(output, source, python),
            "operation": "operational_prediction",
            "history_db_path": str(history_db),
            "capture_db_path": str(db),
            "max_jobs": 12 - len(json.loads((campaign.root / "ledger.json").read_bytes())["attempts"]),
            "result_access": False, "research_activation": False,
        }
    create_once(output / "plan.json", plan)
    return {
        "plan": str(output / "plan.json"),
        "plan_sha256": digest(plan),
        "commit": commit,
        "tree": tree,
        "source_archive_sha256": plan["source_archive_sha256"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--operational-predictions", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--reconciliation-roots", type=Path, required=True)
    parser.add_argument("--installed-dir", type=Path, default=Path.home() / ".config/systemd/user")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                campaign_root=args.campaign_root,
                operational_predictions=args.operational_predictions,
                output=args.output,
                start=datetime.fromisoformat(args.start),
                python=args.python,
                db=args.db,
                lock=args.lock,
                reconciliation_roots=json.loads(args.reconciliation_roots.read_bytes()),
                installed_dir=args.installed_dir,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
