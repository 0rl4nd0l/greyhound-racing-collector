#!/usr/bin/env python3
"""Resolve missing official race numbers for manual reverify candidates.

This is a report-only helper for rows whose manual verification packet has a
date, venue, distance, and winner but no official race number. It may fetch
public TheDogs pages, but it never writes labels, DB rows, snapshots, manifests,
model state, registry state, promotions, or betting output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingest_results_for_date import (
    THEDOGS_PUBLIC_HEADERS,
    _clean_official_runner_name,
    norm_name,
    parse_thedogs_result_html_runner_rows,
    rendered_text_from_html,
    response_is_forbidden,
    thedogs_result_rows_present,
    title_from_html,
)
from scripts.run_official_reverify_lookup_dry_run import (
    StatelessHttpClient,
    _candidate_urls,
    _venue_slug,
)


ROOT = REPO_ROOT
ALLOWED_OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/"
SCHEMA_VERSION = "official_race_number_lookup_dry_run_v1"
WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "official_fetch": True,
    "snapshot_mutation": False,
    "manifest_mutation": False,
    "model_training": False,
    "registry_mutation": False,
    "promotion": False,
    "betting_decision": False,
}

RESOLVED_QUEUE_FIELDS = [
    "identity_key",
    "legacy_race_id",
    "venue",
    "race_date",
    "race_number",
    "target_distance",
    "selected_metadata_grade",
    "winner_name",
    "resolution_status",
    "source_url",
]


def clean_thedogs_result_runner_name(raw: str) -> str:
    text = _clean_official_runner_name(raw) or ""
    return re.sub(r"\s+NBT\s*$", "", text, flags=re.IGNORECASE).strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"queue_row_not_object:{line_number}")
            rows.append(row)
    return rows


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _repo_output_path(path: Path, root: Path | None = None) -> tuple[Path, str]:
    root_path = (root or ROOT).expanduser().resolve(strict=False)
    logical = path.expanduser()
    if not logical.is_absolute():
        logical = root_path / logical
    resolved = logical.resolve(strict=False)
    try:
        relative = resolved.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{resolved}") from exc
    return resolved, relative


def _assert_report_output_dir_safe(output_dir: Path, root: Path | None = None) -> Path:
    resolved, relative = _repo_output_path(output_dir, root)
    if not relative.startswith(ALLOWED_OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _normalized_grade(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _target_winner_key(row: Mapping[str, Any]) -> str:
    return norm_name(clean_thedogs_result_runner_name(str(row.get("winner_name") or "")))


def _distance_text_match(text: str, target_distance: Any) -> bool:
    distance = _safe_float(target_distance)
    if distance is None:
        return False
    distance_int = int(distance)
    collapsed = " ".join(str(text or "").split()).lower()
    return f"{distance_int}m" in collapsed


def _fetch_candidate_race(
    *,
    row: Mapping[str, Any],
    slug: str,
    race_number: int,
    http_client: Any,
    timeout_seconds: float,
) -> dict[str, Any]:
    race_date = str(row.get("race_date") or "").strip()
    urls = _candidate_urls(slug=slug, race_date=race_date, race_number=race_number)
    target_winner_key = _target_winner_key(row)
    target_grade_key = _normalized_grade(row.get("selected_metadata_grade"))
    target_distance = row.get("target_distance")
    last_status = "official_result_not_found"
    last_source_url = urls[0] if urls else None

    for url in urls:
        try:
            response = http_client.get(
                url,
                headers=THEDOGS_PUBLIC_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - report-only packet should capture fetch failures.
            last_status = f"official_fetch_error:{type(exc).__name__}"
            continue

        markup = getattr(response, "text", "") or ""
        source_url = getattr(response, "url", url)
        last_source_url = source_url
        text = rendered_text_from_html(markup)
        status_code = getattr(response, "status_code", None)
        if response_is_forbidden(status_code, title_from_html(markup), text):
            return {
                "race_number": race_number,
                "status": "OFFICIAL_FETCH_FORBIDDEN",
                "source_url": source_url,
                "winner_match": False,
                "distance_match": False,
                "grade_text_match": False,
                "official_winner_name": None,
                "official_runner_rows": [],
                "attempted_urls": urls,
            }
        if status_code and status_code >= 400:
            last_status = f"official_http_{status_code}"
            continue

        official_rows = parse_thedogs_result_html_runner_rows(markup)
        winner_rows = [
            item
            for item in official_rows
            if _safe_int(_mapping(item).get("finish_position")) == 1
        ]
        winner_name = str(_mapping(winner_rows[0]).get("dog_name") or "") if winner_rows else ""
        winner_key = norm_name(clean_thedogs_result_runner_name(winner_name))
        distance_match = _distance_text_match(text, target_distance)
        text_grade_key = _normalized_grade(text)
        grade_text_match = bool(target_grade_key and target_grade_key in text_grade_key)
        winner_match = bool(target_winner_key and winner_key == target_winner_key)
        parsed_status = (
            "OFFICIAL_RESULT_PARSED"
            if official_rows
            else "OFFICIAL_RESULT_TABLE_WITHOUT_STRICT_ROWS"
            if thedogs_result_rows_present(markup)
            else "OFFICIAL_RESULT_NOT_PARSED"
        )
        return {
            "race_number": race_number,
            "status": parsed_status,
            "source_url": source_url,
            "winner_match": winner_match,
            "distance_match": distance_match,
            "grade_text_match": grade_text_match,
            "official_winner_name": winner_name or None,
            "official_runner_rows": official_rows,
            "attempted_urls": urls,
        }

    return {
        "race_number": race_number,
        "status": "OFFICIAL_RESULT_NOT_PARSED",
        "skip_reasons": [last_status],
        "source_url": last_source_url,
        "winner_match": False,
        "distance_match": False,
        "grade_text_match": False,
        "official_winner_name": None,
        "official_runner_rows": [],
        "attempted_urls": urls,
    }


def _resolved_queue_row(row: Mapping[str, Any], match: Mapping[str, Any]) -> dict[str, Any]:
    race_number = _safe_int(match.get("race_number"))
    resolved = dict(row)
    original_blockers = list(resolved.get("blockers") or [])
    resolved["original_lookup_status"] = resolved.get("lookup_status")
    resolved["original_blockers"] = original_blockers
    resolved["lookup_status"] = "PARSE_READY"
    resolved["lookup_key"] = {
        "venue": resolved.get("venue"),
        "race_date": resolved.get("race_date"),
        "race_number": race_number,
    }
    resolved["race_number"] = race_number
    resolved["blockers"] = []
    resolved["next_action"] = "official_result_dry_run_lookup_before_any_label_write"
    resolved["race_number_resolution"] = {
        "status": "RESOLVED_OFFICIAL_WINNER_AND_DISTANCE_MATCH",
        "source_url": match.get("source_url"),
        "official_winner_name": match.get("official_winner_name"),
        "winner_match": match.get("winner_match"),
        "distance_match": match.get("distance_match"),
        "grade_text_match": match.get("grade_text_match"),
        "label_write_approved": False,
        "approval_required_before_label_write": True,
    }
    return resolved


def _result_for_queue_row(
    *,
    row: Mapping[str, Any],
    http_client: Any,
    max_race_number: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    base = {
        "schema_version": "official_race_number_lookup_result_v1",
        "identity_key": row.get("identity_key"),
        "legacy_race_id": row.get("legacy_race_id"),
        "venue": row.get("venue"),
        "race_date": row.get("race_date"),
        "target_distance": row.get("target_distance"),
        "selected_metadata_grade": row.get("selected_metadata_grade"),
        "winner_name": row.get("winner_name"),
        "original_lookup_status": row.get("lookup_status"),
        "original_blockers": list(row.get("blockers") or []),
    }
    if row.get("lookup_status") != "PARSE_BLOCKED":
        return {
            **base,
            "resolution_status": "SKIPPED_NOT_PARSE_BLOCKED",
            "resolved_queue_row": None,
            "candidate_races": [],
        }

    race_date = str(row.get("race_date") or "").strip()
    venue = str(row.get("venue") or "").strip().upper().replace(" ", "_")
    slug = _venue_slug(venue)
    if not race_date or not venue:
        return {
            **base,
            "resolution_status": "INSUFFICIENT_LOOKUP_INPUT",
            "resolved_queue_row": None,
            "candidate_races": [],
        }
    if not slug:
        return {
            **base,
            "resolution_status": "VENUE_SLUG_MISSING",
            "resolved_queue_row": None,
            "candidate_races": [],
        }

    candidate_races = [
        _fetch_candidate_race(
            row=row,
            slug=slug,
            race_number=race_number,
            http_client=http_client,
            timeout_seconds=timeout_seconds,
        )
        for race_number in range(1, max(1, int(max_race_number)) + 1)
    ]
    matches = [
        candidate
        for candidate in candidate_races
        if candidate.get("winner_match") is True and candidate.get("distance_match") is True
    ]
    if len(matches) == 1:
        return {
            **base,
            "resolution_status": "RESOLVED_OFFICIAL_WINNER_AND_DISTANCE_MATCH",
            "resolved_race_number": matches[0].get("race_number"),
            "resolved_source_url": matches[0].get("source_url"),
            "resolved_queue_row": _resolved_queue_row(row, matches[0]),
            "candidate_races": candidate_races,
        }
    if len(matches) > 1:
        status = "MULTIPLE_OFFICIAL_WINNER_DISTANCE_MATCHES_REVIEW_REQUIRED"
    else:
        status = "NO_OFFICIAL_WINNER_DISTANCE_MATCH"
    return {
        **base,
        "resolution_status": status,
        "resolved_queue_row": None,
        "candidate_races": candidate_races,
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESOLVED_QUEUE_FIELDS)
        writer.writeheader()
        for row in rows:
            resolution = _mapping(row.get("race_number_resolution"))
            writer.writerow(
                {
                    "identity_key": row.get("identity_key"),
                    "legacy_race_id": row.get("legacy_race_id"),
                    "venue": row.get("venue"),
                    "race_date": row.get("race_date"),
                    "race_number": row.get("race_number"),
                    "target_distance": row.get("target_distance"),
                    "selected_metadata_grade": row.get("selected_metadata_grade"),
                    "winner_name": row.get("winner_name"),
                    "resolution_status": resolution.get("status"),
                    "source_url": resolution.get("source_url"),
                }
            )


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = _mapping(packet.get("summary"))
    lines = [
        "# Official Race Number Lookup Dry Run",
        "",
        f"Status: `{packet.get('status')}`.",
        "",
        "No DB writes, label writes, snapshot mutations, manifest mutations, model training, model-registry state changes, promotions, betting decisions, or expected-value assertions were performed.",
        "",
        "## Summary",
        "",
        f"- Queue rows seen: `{summary.get('queue_rows_seen')}`",
        f"- Parse-blocked rows evaluated: `{summary.get('parse_blocked_rows_seen')}`",
        f"- Resolved rows: `{summary.get('resolved_count')}`",
        f"- Unresolved rows: `{summary.get('unresolved_count')}`",
        f"- Resolution status counts: `{summary.get('resolution_status_counts')}`",
        "",
        "## Next Step",
        "",
        "Run the resolved queue through `scripts/run_official_reverify_lookup_dry_run.py` as another dry run, then reconcile identity. Do not write labels without explicit approval.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_race_number_lookup_packet(
    *,
    queue_path: Path,
    output_dir: Path,
    http_client: Any | None = None,
    max_candidates: int | None = None,
    max_race_number: int = 14,
    timeout_seconds: float = 20.0,
    progress_every: int = 0,
) -> dict[str, Any]:
    output_dir = _assert_report_output_dir_safe(output_dir)
    rows = _load_jsonl(queue_path)
    if max_candidates is not None:
        rows = rows[: max(0, int(max_candidates))]
    client = http_client or StatelessHttpClient(timeout_seconds=timeout_seconds)
    results = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        results.append(
            _result_for_queue_row(
                row=row,
                http_client=client,
                max_race_number=max_race_number,
                timeout_seconds=timeout_seconds,
            )
        )
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(
                json.dumps(
                    {
                        "progress": index,
                        "total": total,
                        "generated_at": datetime.now(timezone.utc)
                        .replace(microsecond=0)
                        .isoformat(),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    resolved_rows = [
        _mapping(result.get("resolved_queue_row"))
        for result in results
        if result.get("resolved_queue_row")
    ]
    unresolved_results = [
        result
        for result in results
        if result.get("original_lookup_status") == "PARSE_BLOCKED"
        and not result.get("resolved_queue_row")
    ]
    resolution_counts = Counter(str(result.get("resolution_status") or "DATA_MISSING") for result in results)
    output_dir.mkdir(parents=True, exist_ok=True)
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "REPORT_ONLY",
        "source_evidence": {
            "queue": str(queue_path.expanduser().resolve()),
            "output_dir": str(output_dir.expanduser().resolve()),
        },
        "summary": {
            "queue_rows_seen": len(rows),
            "parse_blocked_rows_seen": sum(
                1 for row in rows if row.get("lookup_status") == "PARSE_BLOCKED"
            ),
            "official_fetch_attempted_count": sum(
                1
                for result in results
                for candidate in result.get("candidate_races") or []
                if candidate.get("attempted_urls")
            ),
            "resolved_count": len(resolved_rows),
            "unresolved_count": len(unresolved_results),
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "resolved_queue_jsonl": "resolved_official_reverify_queue.jsonl",
        "resolved_queue_csv": "resolved_official_reverify_queue.csv",
        "unresolved_results_jsonl": "unresolved_official_race_number_lookup_results.jsonl",
        "resolved_queue_rows": resolved_rows,
        "results": results,
    }
    (output_dir / "official_race_number_lookup_packet.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "resolved_official_reverify_queue.jsonl", resolved_rows)
    _write_jsonl(output_dir / "unresolved_official_race_number_lookup_results.jsonl", unresolved_results)
    _write_csv(output_dir / "resolved_official_reverify_queue.csv", resolved_rows)
    _write_report(output_dir / "report.md", packet)
    return packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-race-number", type=int, default=14)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = build_race_number_lookup_packet(
        queue_path=args.queue,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        max_race_number=args.max_race_number,
        timeout_seconds=args.timeout_seconds,
        progress_every=args.progress_every,
    )
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
