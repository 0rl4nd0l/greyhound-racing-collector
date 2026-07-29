#!/usr/bin/env python3
"""Collect official result labels for accepted Expert Form feature rows.

This packet is report-only. It reads already accepted pre-jump Expert Form
feature rows, revalidates their source sidecars, fetches official TheDogs
result pages, validates the result boxes against the frozen CSV runner set, and
writes only artifact-local JSONL/report files. It does not write DB labels,
mutate canonical schemas, rewrite snapshots, promote models, emit EV, or bet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts import ingest_results_for_date as ingest  # noqa: E402
from scripts.autonomous_official_result_capture import (  # noqa: E402
    validate_official_result_evidence_rows,
)
from scripts.run_shadow_non_tgr_rf_evaluation import (  # noqa: E402
    protected_path_snapshot,
    protected_path_verification,
    shadow_relpath,
)
from utils.expert_form_metadata import safe_expert_form_metadata_from_payload  # noqa: E402
from utils.race_lifecycle import RESULTED  # noqa: E402
from utils.runner_completeness import normalise_runner_name  # noqa: E402


OUTPUT_PREFIX = (
    "artifacts/full_evidence_orchestration_20260525/"
    "expert_form_official_result_labels_"
)
DEFAULT_FEATURE_ROWS_GLOB = (
    "artifacts/full_evidence_orchestration_20260525/"
    "expert_form_shadow_feature_row_backfill_*_report_only/shadow_feature_rows.json"
)
FINAL_READY = "OFFICIAL_RESULT_LABELS_READY_REPORT_ONLY"
FINAL_PARTIAL = "OFFICIAL_RESULT_LABELS_PARTIAL_REPORT_ONLY"
FINAL_DATA_MISSING = "DATA_MISSING_OFFICIAL_RESULT_LABELS"

NO_WRITE_GUARANTEES = {
    "report_only": True,
    "canonical_schema_mutation": False,
    "training_artifact_write": False,
    "registry_mutation": False,
    "production_prediction_write": False,
    "db_write": False,
    "canonical_label_write": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
    "ev_output": False,
    "betting_output": False,
}


def now_id(value: datetime | None = None) -> str:
    return (value or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def relpath(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def assert_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(
            f"output_dir_must_be_expert_form_official_result_labels_artifact:{relative}"
        )
    return logical


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def read_json_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def latest_feature_rows_path() -> Path | None:
    matches = sorted(ROOT.glob(DEFAULT_FEATURE_ROWS_GLOB))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.as_posix()))


def race_identity_from_id(race_id: str) -> dict[str, Any]:
    match = re.match(
        r"^\s*Race\s+(\d+)\s+-\s*(.+?)\s+-\s*(\d{4}-\d{2}-\d{2})\s*$",
        str(race_id or ""),
        re.IGNORECASE,
    )
    if not match:
        return {"race_number": None, "venue": None, "race_date": None}
    return {
        "race_number": int(match.group(1)),
        "venue": match.group(2).strip(),
        "race_date": match.group(3),
    }


def canonical_url_from_sidecar(payload: Mapping[str, Any]) -> str | None:
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    race_url = str(race_info.get("url") or "").strip()
    if race_url:
        return race_url
    metadata = (
        payload.get("expert_form_metadata")
        if isinstance(payload.get("expert_form_metadata"), Mapping)
        else {}
    )
    source_url = str(metadata.get("source_url") or "").strip()
    return re.sub(r"/expert-form/?$", "", source_url) if source_url else None


def race_time_from_sidecar(payload: Mapping[str, Any]) -> str | None:
    race_info = payload.get("race_info") if isinstance(payload.get("race_info"), Mapping) else {}
    value = str(race_info.get("race_time") or "").strip()
    return value or None


def selected_feature_races(
    rows: Sequence[Mapping[str, Any]],
    *,
    race_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    by_race: dict[str, dict[str, Any]] = {}
    for row in rows:
        race_id = str(row.get("race_id") or "").strip()
        if not race_id:
            continue
        if race_ids and race_id not in race_ids:
            continue
        current = by_race.get(race_id)
        candidate = {
            "race_id": race_id,
            "source_csv_path": row.get("source_csv_path") or row.get("source_csv"),
            "source_sidecar_path": row.get("source_sidecar_path"),
        }
        if current is None or str(candidate.get("source_sidecar_path") or "") > str(
            current.get("source_sidecar_path") or ""
        ):
            by_race[race_id] = candidate
    selected = [by_race[key] for key in sorted(by_race)]
    if limit is not None and limit >= 0:
        selected = selected[:limit]
    return selected


def build_candidate(row: Mapping[str, Any]) -> tuple[ingest.RaceCandidate | None, dict[str, Any]]:
    race_id = str(row.get("race_id") or "").strip()
    base = {
        "race_id": race_id,
        "source_csv_path": row.get("source_csv_path"),
        "source_sidecar_path": row.get("source_sidecar_path"),
    }
    identity = race_identity_from_id(race_id)
    if not race_id or identity["race_number"] is None:
        return None, {**base, "reason": "race_id_unparseable"}

    csv_path = ROOT / str(row.get("source_csv_path") or "")
    sidecar_path = ROOT / str(row.get("source_sidecar_path") or "")
    if not row.get("source_csv_path") or not csv_path.exists():
        return None, {**base, "reason": "source_csv_missing"}
    if not row.get("source_sidecar_path") or not sidecar_path.exists():
        return None, {**base, "reason": "source_sidecar_missing"}

    try:
        sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, {**base, "reason": f"source_sidecar_unreadable:{type(exc).__name__}"}
    if not isinstance(sidecar_payload, Mapping):
        return None, {**base, "reason": "source_sidecar_not_object"}

    safe_metadata = safe_expert_form_metadata_from_payload(sidecar_payload)
    if safe_metadata.get("metadata_is_leakage_safe") is not True:
        return None, {
            **base,
            "reason": "expert_form_sidecar_not_leakage_safe",
            "rejected_reasons": list(safe_metadata.get("rejected_reasons") or []),
        }

    runner_completeness = ingest.analyze_csv_runner_completeness(csv_path).as_dict()
    if runner_completeness.get("status") != "COMPLETE":
        return None, {
            **base,
            "reason": "source_csv_runner_set_incomplete",
            "runner_completeness": runner_completeness,
        }
    participants = ingest.parse_participants_from_csv(csv_path)
    if not participants:
        return None, {**base, "reason": "source_csv_participants_missing"}

    candidate = ingest.RaceCandidate(
        race_id=race_id,
        venue=str(identity["venue"] or ""),
        race_number=int(identity["race_number"]),
        race_date=str(identity["race_date"] or ""),
        race_time=race_time_from_sidecar(sidecar_payload),
        start_datetime=None,
        sportsbet_url=None,
        csv_path=csv_path,
        participants=participants,
        lifecycle_status="result_label_refresh_report_only",
        participant_source="expert_form_backfill_csv",
        csv_participants=participants,
        runner_completeness=runner_completeness,
        canonical_thedogs_url=canonical_url_from_sidecar(sidecar_payload),
    )
    return candidate, {**base, "reason": None}


def source_result_diagnostic(result: ingest.SourceResult) -> dict[str, Any]:
    return ingest._source_result_diagnostic(result)  # noqa: SLF001


def comparable_result_dog_name(value: Any) -> str:
    cleaned = re.sub(r"\bNBT\b\.?$", "", str(value or "").strip(), flags=re.IGNORECASE)
    return normalise_runner_name(cleaned)


def result_dog_name_validation_error(
    candidate: ingest.RaceCandidate,
    result: ingest.SourceResult,
) -> str | None:
    if not result.dog_names_by_box:
        return None
    participant_by_box = {
        int(participant["box_number"]): str(participant["dog_name"])
        for participant in candidate.participants or []
        if participant.get("box_number") is not None
    }
    for box in sorted(result.positions_by_box):
        official_name = str(result.dog_names_by_box.get(int(box)) or "").strip()
        participant_name = participant_by_box.get(int(box), "")
        if not official_name or not participant_name:
            continue
        if comparable_result_dog_name(official_name) != comparable_result_dog_name(participant_name):
            return f"result_dog_name_mismatch_for_box:{int(box)}"
    return None


def fetch_official_result(
    candidate: ingest.RaceCandidate,
    *,
    use_browser_fallback: bool,
) -> tuple[ingest.SourceResult, str | None]:
    public_http = ingest._PersistentPublicHttpClient()  # noqa: SLF001
    try:
        fetcher = ingest.TheDogsResultFetcher(
            None,
            http_session=public_http,
        )
        if not use_browser_fallback:
            urls = fetcher._result_urls(candidate)  # noqa: SLF001
            http_result = fetcher._fetch_via_http(candidate, urls)  # noqa: SLF001
            if http_result:
                return http_result, None
            return (
                ingest.SourceResult(
                    source="thedogs_official",
                    status="error",
                    source_url=urls[0] if urls else None,
                    positions_by_box={},
                    raw_order=[],
                    error="no_thedogs_result_response",
                    attempted_urls=[],
                ),
                None,
            )

        driver, By, browser_error = ingest.optional_browser_driver(headless=True)
        try:
            fetcher = ingest.TheDogsResultFetcher(
                driver,
                by=By,
                http_session=public_http,
            )
            return fetcher.fetch(candidate), browser_error
        finally:
            if driver is not None:
                try:
                    driver.quit()
                except Exception:
                    pass
    finally:
        public_http.close()


def artifact_rows_for_result(
    candidate: ingest.RaceCandidate,
    result: ingest.SourceResult,
    *,
    captured_at: str,
    artifact_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    box_to_name = {
        int(participant["box_number"]): str(participant["dog_name"])
        for participant in candidate.participants
    }
    winner_box = result.winner_box
    race_row = {
        "schema_version": "autonomous_official_result_race_v1",
        "captured_at": captured_at,
        "race_id": candidate.race_id,
        "venue": candidate.venue,
        "race_number": candidate.race_number,
        "race_date": candidate.race_date,
        "race_time": candidate.race_time,
        "start_datetime": candidate.start_datetime,
        "source": result.source,
        "source_url": result.source_url,
        "status": result.status,
        "winner_name": box_to_name.get(int(winner_box)) if winner_box is not None else None,
        "winner_box": winner_box,
        "box_order": list(result.raw_order or []),
        "participant_source": candidate.participant_source,
        "position_count": len(result.positions_by_box),
        "participant_count": len(candidate.participants or []),
        "scope": {
            "candidate_source": "expert_form_shadow_feature_row_backfill",
            "source_artifact_dir": relpath(artifact_dir),
        },
    }
    runner_rows = []
    for box, position in sorted(result.positions_by_box.items(), key=lambda item: (item[1], item[0])):
        box_number = int(box)
        runner_rows.append(
            {
                "schema_version": "autonomous_official_result_runner_v1",
                "captured_at": captured_at,
                "race_id": candidate.race_id,
                "venue": candidate.venue,
                "race_number": candidate.race_number,
                "race_date": candidate.race_date,
                "source": result.source,
                "source_url": result.source_url,
                "box_number": box_number,
                "dog_name": box_to_name.get(box_number),
                "finish_position": int(position),
                "is_winner": int(position) == 1,
            }
        )
    return {"race_rows": [race_row], "runner_rows": runner_rows}


def winner_label_rows_for_result(
    candidate: ingest.RaceCandidate,
    result: ingest.SourceResult,
    *,
    captured_at: str,
) -> list[dict[str, Any]]:
    winner_box = result.winner_box
    if winner_box is None:
        return []
    rows: list[dict[str, Any]] = []
    for participant in candidate.participants or []:
        box_number = int(participant["box_number"])
        finish_position = result.positions_by_box.get(box_number)
        rows.append(
            {
                "schema_version": "expert_form_official_result_winner_label_runner_v1",
                "captured_at": captured_at,
                "race_id": candidate.race_id,
                "venue": candidate.venue,
                "race_number": candidate.race_number,
                "race_date": candidate.race_date,
                "source": result.source,
                "source_url": result.source_url,
                "box_number": box_number,
                "dog_name": str(participant["dog_name"]),
                "finish_position": int(finish_position) if finish_position is not None else None,
                "finish_position_available": finish_position is not None,
                "is_winner": box_number == int(winner_box),
                "label_scope": "winner_only_full_frozen_field",
            }
        )
    return rows


def collect_labels(
    *,
    expert_feature_rows_path: Path | None,
    race_ids: set[str] | None = None,
    limit: int | None = None,
    use_browser_fallback: bool = False,
    output_dir: Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or datetime.now().astimezone()
    captured_at = generated_at.isoformat()
    feature_rows = read_json_rows(expert_feature_rows_path)
    selected = selected_feature_races(feature_rows, race_ids=race_ids, limit=limit)

    candidate_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    race_rows: list[dict[str, Any]] = []
    runner_rows: list[dict[str, Any]] = []
    winner_label_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    browser_errors: Counter[str] = Counter()

    for row in selected:
        candidate, skipped = build_candidate(row)
        if candidate is None:
            skipped_rows.append(skipped)
            quarantine_rows.append(
                {
                    "schema_version": "expert_form_official_result_quarantine_v1",
                    "captured_at": captured_at,
                    "race_id": skipped.get("race_id"),
                    "reason": skipped.get("reason") or "candidate_rejected",
                    "item": skipped,
                }
            )
            continue

        candidate_rows.append(
            {
                "race_id": candidate.race_id,
                "venue": candidate.venue,
                "race_number": candidate.race_number,
                "race_date": candidate.race_date,
                "race_time": candidate.race_time,
                "canonical_thedogs_url": candidate.canonical_thedogs_url,
                "source_csv_path": relpath(candidate.csv_path),
                "participant_count": len(candidate.participants or []),
                "participant_source": candidate.participant_source,
            }
        )
        result, browser_error = fetch_official_result(
            candidate,
            use_browser_fallback=use_browser_fallback,
        )
        if browser_error:
            browser_errors[str(browser_error)] += 1
        validation_error = ingest.result_validation_error(candidate, result)
        name_validation_error = result_dog_name_validation_error(candidate, result)
        if validation_error is None and name_validation_error is not None:
            validation_error = name_validation_error
        if validation_error:
            attempted_source = source_result_diagnostic(result)
            item = {
                "race_id": candidate.race_id,
                "errors": [validation_error],
                "attempted_sources": [attempted_source],
                "participant_source": candidate.participant_source,
                "participant_count": len(candidate.participants or []),
                "participant_boxes": sorted(
                    int(participant["box_number"])
                    for participant in candidate.participants
                    if participant.get("box_number") is not None
                ),
                "participants": [
                    {
                        "box_number": int(participant["box_number"]),
                        "dog_name": str(participant["dog_name"]),
                    }
                    for participant in candidate.participants
                    if participant.get("box_number") is not None
                ],
            }
            quarantine_rows.append(
                {
                    "schema_version": "expert_form_official_result_quarantine_v1",
                    "captured_at": captured_at,
                    "race_id": candidate.race_id,
                    "reason": "official_result_validation_failed",
                    "errors": [validation_error],
                    "item": item,
                    "candidate": {
                        "venue": candidate.venue,
                        "race_number": candidate.race_number,
                        "race_date": candidate.race_date,
                        "canonical_thedogs_url": candidate.canonical_thedogs_url,
                        "participant_count": len(candidate.participants or []),
                        "participant_boxes": sorted(
                            int(participant["box_number"])
                            for participant in candidate.participants
                            if participant.get("box_number") is not None
                        ),
                    },
                    "attempted_source": attempted_source,
                }
            )
            continue
        if result.source != "thedogs_official" or result.status != RESULTED:
            quarantine_rows.append(
                {
                    "schema_version": "expert_form_official_result_quarantine_v1",
                    "captured_at": captured_at,
                    "race_id": candidate.race_id,
                    "reason": "non_official_or_non_resulted_source",
                    "attempted_source": source_result_diagnostic(result),
                }
            )
            continue
        built = artifact_rows_for_result(
            candidate,
            result,
            captured_at=captured_at,
            artifact_dir=output_dir,
        )
        race_rows.extend(built["race_rows"])
        runner_rows.extend(built["runner_rows"])
        winner_label_rows.extend(
            winner_label_rows_for_result(
                candidate,
                result,
                captured_at=captured_at,
            )
        )

    result_race_ids = {str(row.get("race_id")) for row in race_rows}
    selected_race_ids = {str(row.get("race_id")) for row in selected}
    status_counts = Counter(
        "RESULT_READY" if race_id in result_race_ids else "NO_SAFE_OFFICIAL_RESULT"
        for race_id in selected_race_ids
    )
    if not selected:
        final_status = FINAL_DATA_MISSING
    elif len(result_race_ids) == len(selected_race_ids):
        final_status = FINAL_READY
    elif result_race_ids:
        final_status = FINAL_PARTIAL
    else:
        final_status = FINAL_DATA_MISSING
    validation = validate_official_result_evidence_rows(
        {"race_rows": race_rows, "runner_rows": runner_rows}
    )

    return {
        "schema_version": "expert_form_official_result_labels_report_v1",
        "generated_at": captured_at,
        "final_status": final_status,
        "expert_feature_rows_path": relpath(expert_feature_rows_path) if expert_feature_rows_path else None,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
        "browser_fallback_enabled": use_browser_fallback,
        "browser_errors": dict(sorted(browser_errors.items())),
        "official_result_evidence_validation": {
            key: value
            for key, value in validation.items()
            if key not in {"race_rows", "runner_rows"}
        },
        "coverage_summary": {
            "expert_feature_rows": len(feature_rows),
            "selected_races": len(selected_race_ids),
            "candidate_races": len(candidate_rows),
            "safe_official_result_races": len(result_race_ids),
            "safe_official_result_runner_rows": len(runner_rows),
            "winner_label_runner_rows": len(winner_label_rows),
            "quarantine_rows": len(quarantine_rows),
            "skipped_candidate_rows": len(skipped_rows),
            "status_counts": dict(sorted(status_counts.items())),
        },
        "selected_race_ids": sorted(selected_race_ids),
        "result_race_ids": sorted(result_race_ids),
        "candidate_rows": candidate_rows,
        "skipped_candidates": skipped_rows,
        "race_rows": race_rows,
        "runner_rows": runner_rows,
        "winner_label_rows": winner_label_rows,
        "quarantine_rows": quarantine_rows,
    }


def summary_md(report: Mapping[str, Any], output_dir: Path) -> str:
    coverage = report.get("coverage_summary") or {}
    return "\n".join(
        [
            "# Expert Form Official Result Labels Packet",
            "",
            f"Final status: `{report.get('final_status')}`",
            "",
            "## Coverage",
            "",
            f"- Selected races: `{coverage.get('selected_races')}`",
            f"- Candidate races: `{coverage.get('candidate_races')}`",
            f"- Safe official-result races: `{coverage.get('safe_official_result_races')}`",
            f"- Safe official-result runner rows: `{coverage.get('safe_official_result_runner_rows')}`",
            f"- Winner-label runner rows: `{coverage.get('winner_label_runner_rows')}`",
            f"- Quarantine rows: `{coverage.get('quarantine_rows')}`",
            "",
            "## Artifacts",
            "",
            f"- `{relpath(output_dir / 'official_result_races.jsonl')}`",
            f"- `{relpath(output_dir / 'official_result_runners.jsonl')}`",
            f"- `{relpath(output_dir / 'official_result_winner_label_runners.jsonl')}`",
            f"- `{relpath(output_dir / 'official_result_quarantine.jsonl')}`",
            f"- `{relpath(output_dir / 'expert_form_official_result_label_report.json')}`",
            "",
            "No DB label write, canonical schema mutation, registry mutation, snapshot rewrite, EV output, or betting output was performed.",
            "",
        ]
    )


def write_packet(report: Mapping[str, Any], output_dir: Path, protected: Mapping[str, Any]) -> None:
    output_dir = assert_output_dir_safe(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "official_result_races.jsonl", report["race_rows"])
    write_jsonl(output_dir / "official_result_runners.jsonl", report["runner_rows"])
    write_jsonl(
        output_dir / "official_result_winner_label_runners.jsonl",
        report["winner_label_rows"],
    )
    write_jsonl(output_dir / "official_result_quarantine.jsonl", report["quarantine_rows"])
    write_jsonl(output_dir / "candidate_races.jsonl", report["candidate_rows"])
    report_for_disk = dict(report)
    report_for_disk.pop("race_rows", None)
    report_for_disk.pop("runner_rows", None)
    report_for_disk.pop("winner_label_rows", None)
    report_for_disk.pop("quarantine_rows", None)
    report_for_disk.pop("candidate_rows", None)
    report_for_disk["artifacts"] = {
        "official_result_races": relpath(output_dir / "official_result_races.jsonl"),
        "official_result_runners": relpath(output_dir / "official_result_runners.jsonl"),
        "official_result_winner_label_runners": relpath(
            output_dir / "official_result_winner_label_runners.jsonl"
        ),
        "official_result_quarantine": relpath(output_dir / "official_result_quarantine.jsonl"),
        "candidate_races": relpath(output_dir / "candidate_races.jsonl"),
    }
    write_json(output_dir / "expert_form_official_result_label_report.json", report_for_disk)
    write_json(output_dir / "protected_path_verification.json", protected)
    manifest = {
        "schema_version": "expert_form_official_result_labels_manifest_v1",
        "generated_at": report["generated_at"],
        "files": {
            "report": relpath(output_dir / "expert_form_official_result_label_report.json"),
            "summary": relpath(output_dir / "SUMMARY.md"),
            "final_status": relpath(output_dir / "final_status.txt"),
            "official_result_races": relpath(output_dir / "official_result_races.jsonl"),
            "official_result_runners": relpath(output_dir / "official_result_runners.jsonl"),
            "official_result_winner_label_runners": relpath(
                output_dir / "official_result_winner_label_runners.jsonl"
            ),
            "official_result_quarantine": relpath(output_dir / "official_result_quarantine.jsonl"),
            "candidate_races": relpath(output_dir / "candidate_races.jsonl"),
            "protected_path_verification": relpath(output_dir / "protected_path_verification.json"),
        },
        "no_write_guarantees": report["no_write_guarantees"],
    }
    write_json(output_dir / "output_manifest.json", manifest)
    write_text(output_dir / "SUMMARY.md", summary_md(report, output_dir))
    write_text(output_dir / "final_status.txt", str(report["final_status"]) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-feature-rows", type=Path, default=None)
    parser.add_argument("--race-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--browser-fallback", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    feature_rows_path = args.expert_feature_rows or latest_feature_rows_path()
    output_dir = args.output_dir or (
        ROOT
        / "artifacts/full_evidence_orchestration_20260525"
        / f"expert_form_official_result_labels_{now_id()}_report_only"
    )
    output_dir = assert_output_dir_safe(output_dir)
    protected_before = protected_path_snapshot()
    report = collect_labels(
        expert_feature_rows_path=feature_rows_path,
        race_ids=set(args.race_id or []) or None,
        limit=args.limit,
        use_browser_fallback=bool(args.browser_fallback),
        output_dir=output_dir,
    )
    protected = protected_path_verification(protected_before)
    write_packet(report, output_dir, protected)
    print(
        json.dumps(
            {
                "final_status": report["final_status"],
                "output_dir": shadow_relpath(output_dir),
                "coverage_summary": report["coverage_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
