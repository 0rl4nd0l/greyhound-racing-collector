#!/usr/bin/env python3
"""Prepare a finite local operational package. Does not execute or install it."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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


def utc_now():
    return datetime.now(timezone.utc)


def prepare(*, output, start, python, db, lock, reconciliation_roots, installed_dir, campaign_root=None, operational_predictions=False, observation_minutes=90, start_after_minutes=None, comparison_plan=None, prediction_root=None, engineering_authority=None, development_authority=None, reduced_request_cap=None, incident_authority=None, incident_slot=None):
    if engineering_authority is not None and (
            not operational_predictions or campaign_root is None
            or comparison_plan is not None or prediction_root is not None):
        raise ValueError('engineering_requires_separate_operational_predictions')
    if development_authority is not None and (engineering_authority is not None
            or not operational_predictions or campaign_root is None or comparison_plan is not None):
        raise ValueError('development_requires_separate_operational_predictions')
    if incident_authority is not None and (engineering_authority is not None or development_authority is not None
            or not operational_predictions or campaign_root is None or comparison_plan is None or prediction_root is None):
        raise ValueError("incident_requires_native_comparison_path")
    incident_args = ({"incident_authority": incident_authority, "incident_slot": incident_slot}
                     if incident_authority is not None else {})
    comparison_binding = None
    if comparison_plan is not None:
        if not operational_predictions:
            raise ValueError("comparison_requires_existing_prediction_path")
        from src.predictor.future_comparison import load_plan
        comparison_plan = comparison_plan.resolve(strict=True)
        comparison_sha = hashlib.sha256(comparison_plan.read_bytes()).hexdigest()
        comparison_value, _ = load_plan(comparison_plan, comparison_sha)
        if incident_authority is not None and (
                comparison_value['status'] != 'AUTHORIZED_ENGINEERING'
                or any(comparison_value.get(key) != item for key, item in incident_args.items())):
            raise ValueError('incident_comparison_scope_mismatch')
        comparison_binding = {"path": str(comparison_plan), "sha256": comparison_sha}
    operational = bool(campaign_root and operational_predictions)
    if (type(observation_minutes) is not int
            or not (observation_minutes == 110 if development_authority else 5 <= observation_minutes <= 90 if operational else observation_minutes == 90)):
        raise ValueError("invalid_operational_observation_duration")
    short_observation = observation_minutes < 60
    if start_after_minutes is not None and (start is not None
            or type(start_after_minutes) is not int or not 5 <= start_after_minutes <= 30):
        raise ValueError('invalid_relative_execution_window')
    if start is None and start_after_minutes is None:
        raise ValueError('execution_window_required')
    output = output.absolute()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    history_db = db.resolve(strict=True)
    if operational_predictions:
        if campaign_root is None:
            raise ValueError("operational_predictions_require_existing_campaign")
        if prediction_root is not None:
            from race_collection.freshness_campaign import Campaign
            campaign = Campaign(campaign_root, development_authority=development_authority, **incident_args)
            approved = getattr(campaign, "incident", None) or campaign.development or campaign.programme
            if (not approved or str(prediction_root) != approved.get('prediction_root')
                    or not prediction_root.is_absolute() or prediction_root.resolve() != prediction_root):
                raise ValueError('prediction_root_not_in_approved_programme')
        db = (prediction_root or campaign_root.resolve() / "operational-predictions") / "capture.sqlite3"
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
        frozen = operational_predictions and (name in {"artifacts/frozen_models/market_form_residual_v1/model.json", "artifacts/frozen_models/market_form_residual_v1/manifest.json", "accuracy_program/repaired_non_tgr_schema.json", "tests/test_run_shadow_non_tgr_rf_evaluation.py"} or name.startswith("artifacts/research_comparison/frozen_20260924/"))
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
    if short_observation:
        timer_path = output / "units" / UNITS[1]
        timer = timer_path.read_text()
        expected = f"OnActiveSec={daemon.DEFAULT_TIMER_FREQUENCY}"
        if timer.count(expected) != 1:
            raise ValueError("unexpected_full_timer_start")
        timer_path.write_text(timer.replace(expected, "OnActiveSec=1s"))
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
        campaign = (Campaign(campaign_root, engineering_authority=engineering_authority)
                    if engineering_authority is not None else Campaign(campaign_root, development_authority=development_authority, **incident_args))
    from utils.sportsbet_access import state_path

    plan = {
        **incident_args,
        **({'development_authority': development_authority} if development_authority is not None else {}),
        **({'engineering_authority': engineering_authority} if engineering_authority is not None else {}),
        **({"prediction_root": str(prediction_root)} if prediction_root is not None else {}),
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
        "cleanup_seconds": 600 if development_authority else 1860 if campaign else 1200,
        "sample_period_seconds": 2,
        "max_sample_gap_seconds": 5,
        "readiness_warmup_seconds": 180 if short_observation else 1200,
        **({"minimum_completed_full_cycles": 1, "minimum_distinct_captures": 1}
           if short_observation else {}),
        **({"minimum_completed_odds_cycles": 3} if observation_minutes < 10 else {}),
        "first_index_deadline_seconds": 180,
        "profile": "bounded80-v1",
        "max_capture_attempts": campaign.value['max_capture_attempts'] if campaign else 1,
        "max_logical_requests": campaign.incident.get("max_python_requests_per_window", campaign.value["max_logical_requests"]) if incident_authority else (16000 if getattr(campaign,"programme",None) else campaign.value['max_logical_requests']) if campaign else 24000,
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
            "authorization": (campaign.development["authority_reference"] if development_authority else "user:collection-to-prediction-20260924"),
            "retention_config_sha256": prepare_retention(output, source, python),
            "operation": "operational_prediction",
            "history_db_path": str(history_db),
            "capture_db_path": str(db),
            "max_jobs": (campaign.incident["max_capture_attempts_per_window"] if incident_authority else 6 if development_authority else campaign.value['max_capture_attempts'] - len(json.loads((campaign.root / "ledger.json").read_bytes())["attempts"])),
            "result_access": False, "research_activation": False,
        }
        if comparison_binding is not None:
            plan["frozen_comparison"] = comparison_binding
    if reduced_request_cap is not None:
        if type(reduced_request_cap) is not int or not 0 < reduced_request_cap <= plan['max_logical_requests']:
            raise ValueError('invalid_reduced_request_cap')
        plan['max_logical_requests'] = reduced_request_cap
    # Select relative windows only after expensive export/runtime/retention work.
    # Seal once using the same canonical encoding the launch preflight verifies.
    if start_after_minutes is not None:
        start = utc_now() + timedelta(minutes=start_after_minutes)
    if start.utcoffset() is None:
        raise ValueError('ambiguous_execution_window')
    end = start + timedelta(minutes=observation_minutes)
    if engineering_authority is not None:
        campaign.check_programme_time()
        if (campaign.study_programme and end + timedelta(seconds=plan['cleanup_seconds'])
                >= datetime.fromisoformat(campaign.study_programme['starts_at'])):
            raise ValueError('engineering_window_overlaps_programme')
    if incident_authority is not None:
        slot = next(row for row in campaign.incident["slots"] if row["id"] == incident_slot)
        if (start != datetime.fromisoformat(slot["starts_at"]) or end != datetime.fromisoformat(slot["ends_at"])
                or end + timedelta(seconds=plan["cleanup_seconds"]) != datetime.fromisoformat(slot["cleanup_by"])):
            raise ValueError("incident_scope_window_changed")
    from zoneinfo import ZoneInfo
    zone = ZoneInfo('Australia/Melbourne')
    if start.astimezone(zone).date() != end.astimezone(zone).date():
        raise ValueError('execution_window_crosses_source_date')
    cleanup_at = end + timedelta(seconds=plan['cleanup_seconds'])
    if start.astimezone(zone).date() != cleanup_at.astimezone(zone).date():
        from race_collection.incident_engineering import late_cleanup_deadline
        cutoff = late_cleanup_deadline(incident_authority) if incident_authority else None
        if cutoff is None or cleanup_at > cutoff:
            raise ValueError('execution_window_crosses_source_date')
    if development_authority and (start.astimezone(zone).date().isoformat() not in campaign.development['dates']
            or start.astimezone(zone).strftime('%H:%M:%S.%f') != '12:40:00.000000'):
        raise ValueError('development_scope_window_changed')
    plan.update(starts_at=start.isoformat(), ends_at=end.isoformat(),
                admission_starts_at=(start-timedelta(minutes=30)).isoformat())
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
    parser.add_argument("--engineering-authority", help="Explicit separate pre-programme operational authority; uses existing engineering limits")
    parser.add_argument("--comparison-plan", type=Path, help="Explicit approved comparison binding; omitted by default")
    parser.add_argument("--observation-minutes", type=int, default=90)
    parser.add_argument("--output", type=Path, required=True)
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument("--start")
    window.add_argument("--start-after-minutes", type=int)
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
                engineering_authority=args.engineering_authority,
                comparison_plan=args.comparison_plan,
                observation_minutes=args.observation_minutes,
                output=args.output,
                start=datetime.fromisoformat(args.start) if args.start else None,
                start_after_minutes=args.start_after_minutes,
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
