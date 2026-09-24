"""Default-off candidate adapter for the existing official-result collector.

Only persisted, verified R3 predictions can nominate a race. No shadow model,
mutable input directory, scheduler, or result transport is introduced here.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from scripts import ingest_results_for_date as ingest
from src.operator_ui.job_store import JobStore, JobStoreError, Phase, utc_text
from src.operator_ui.journal_results import OfficialResultSource
from src.operator_ui.r3_api import _verified_result, build_verified_bundle_reader
from utils.csv_metadata import canonical_thedogs_race_identity, canonical_thedogs_venue_identity
from utils.runner_completeness import RunnerRow, analyze_runner_rows


def r3_prediction_candidates(*, job_store_path: Path, prediction_bundles: Path,
                             result_database: Path, target_date: str,
                             current_time: datetime, race_ids, output_dir: Path,
                             limit: int = 128):
    """Return bounded candidates authenticated against job and sealed bundle truth."""
    utc_text(current_time)
    if not 1 <= limit <= 128:
        raise ValueError("R3 result discovery bounds invalid")
    report = {"schema_version": "r3_official_result_candidates_v1",
              "candidate_source": "verified_r3_predictions", "candidate_count": 0,
              "candidate_race_ids": [], "limit": limit,
              "result_horizon": "all_persisted_ready_jobs", "requested_target_date": target_date}
    try:
        store = JobStore(job_store_path, separate_from=(result_database,), readonly=True)
        jobs = store.recorded_jobs()
    except (OSError, ValueError, JobStoreError) as exc:
        return [], [{"reason": "R3_JOB_STORE_UNAVAILABLE", "detail": str(exc)}], report
    read_bundle = build_verified_bundle_reader(prediction_bundles, store)
    candidates, skipped = [], []
    selected_ids = set(race_ids)
    for job in jobs:
        if job.phase is not Phase.PREDICTION_READY or selected_ids and job.input.race_id not in selected_ids:
            continue
        reason = None
        try:
            jump = datetime.fromisoformat(job.input.jump_timestamp.replace("Z", "+00:00"))
            if jump >= current_time:
                reason = "R3_RACE_NOT_OFF"
            elif datetime.fromisoformat(job.phase_at.replace("Z", "+00:00")) > current_time:
                reason = "R3_READY_TIMESTAMP_IN_FUTURE"
            else:
                bundle = read_bundle(job)
                if _verified_result(job, bundle, list(store.events(job.job_id))) is None:
                    raise ValueError("R3_PREDICTION_VERIFICATION_FAILED")
                race = bundle.result["race"]
                identity = canonical_thedogs_race_identity(race["url"])
                if (identity is None or identity["canonical_url"] != race["url"]
                    or identity["race_date"] != race["race_date"]
                    or identity["race_number"] != race["race_number"]
                    or canonical_thedogs_venue_identity(identity["venue_slug"]) != canonical_thedogs_venue_identity(race["venue"])):
                    raise ValueError("R3_CANONICAL_RACE_IDENTITY_MISMATCH")
                generated = datetime.fromisoformat(bundle.result["generated_at"].replace("Z", "+00:00"))
                utc_text(generated)
                if not generated < jump < current_time:
                    raise ValueError("R3_PREDICTION_TIMESTAMP_INVALID")
                existing = OfficialResultSource(result_database).read(job, bundle, now=current_time)
                if existing["state"] == "RESULT_AVAILABLE":
                    reason = "R3_RESULT_ALREADY_AVAILABLE"
                elif existing["state"] == "RESULT_REJECTED" or existing.get("reason") in {"RESULT_SOURCE_BUSY", "RESULT_SOURCE_CHANGED", "RESULT_SOURCE_UNSAFE"}:
                    reason = existing["reason"]
                elif len(candidates) >= limit:
                    reason = "R3_RESULT_CANDIDATE_LIMIT"
                else:
                    rows = [RunnerRow(box_number=r["box"], dog_name=r["name"])
                            for r in job.input.ordered_runners]
                    # R3 completeness was sealed at admission; validate the
                    # entire admitted field, without a shadow-model minimum.
                    completeness = analyze_runner_rows(rows, source="verified_r3_prediction",
                                                        min_complete_runners=len(rows)).as_dict()
                    if completeness.get("status") != "COMPLETE":
                        raise ValueError("R3_RUNNERS_INCOMPLETE")
                    participants = [{"box_number": r["box"], "dog_name": r["name"]}
                                    for r in job.input.ordered_runners]
                    candidates.append(ingest.RaceCandidate(
                        race_id=job.input.race_id, venue=race["venue"],
                        race_number=race["race_number"], race_date=race["race_date"],
                        race_time=None, start_datetime=job.input.jump_timestamp,
                        # The ingester uses participants directly. This locator
                        # is diagnostic only; no CSV is read or manufactured.
                        sportsbet_url=None, csv_path=output_dir / "r3-sealed-inputs-not-csv",
                        participants=participants, lifecycle_status="JUMPED_AWAITING_RESULT",
                        participant_source="verified_r3_prediction", csv_participants=participants,
                        runner_completeness=completeness, canonical_thedogs_url=race["url"]))
        except (KeyError, TypeError, ValueError, JobStoreError) as exc:
            reason = str(exc)
        if reason:
            skipped.append({"job_id": job.job_id, "race_id": job.input.race_id, "reason": reason})
    report.update(candidate_count=len(candidates), skipped_count=len(skipped),
                  candidate_race_ids=[c.race_id for c in candidates])
    return candidates, skipped, report
