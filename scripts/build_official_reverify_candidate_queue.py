#!/usr/bin/env python3
"""Build a report-only queue for official re-verification candidates.

This script consumes a legacy-label verification packet and emits a JSONL queue
of result-like races that need official TheDogs re-verification. It does not
fetch official sources, write labels, mutate snapshots, train, or promote.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "official_reverify_candidate_queue_v1"

WRITES_PERFORMED = {
    "db_write": False,
    "label_write": False,
    "official_fetch": False,
    "snapshot_mutation": False,
    "model_training": False,
    "registry_mutation": False,
}

MONTHS = {
    "JANUARY": "01",
    "FEBRUARY": "02",
    "MARCH": "03",
    "APRIL": "04",
    "MAY": "05",
    "JUNE": "06",
    "JULY": "07",
    "AUGUST": "08",
    "SEPTEMBER": "09",
    "OCTOBER": "10",
    "NOVEMBER": "11",
    "DECEMBER": "12",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("json_root_not_object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parse_legacy_race_id(race_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = str(race_id or "").strip()
    if not text:
        return None, ["legacy_race_id_missing"]

    legacy_match = re.fullmatch(
        r"(?P<venue>[A-Z0-9_]+)_(?P<race_number>\d+)_(?P<day>\d{1,2})_"
        r"(?P<month>[A-Za-z]+)_(?P<year>\d{4})",
        text,
    )
    if legacy_match:
        month = MONTHS.get(legacy_match.group("month").upper())
        if not month:
            return None, ["legacy_race_id_month_not_parseable"]
        return (
            {
                "venue": legacy_match.group("venue"),
                "race_number": int(legacy_match.group("race_number")),
                "race_date": (
                    f"{legacy_match.group('year')}-{month}-"
                    f"{int(legacy_match.group('day')):02d}"
                ),
            },
            [],
        )

    coded_match = re.fullmatch(
        r"R0*(?P<race_number>\d+)_(?P<race_date>\d{4}-\d{2}-\d{2})_"
        r"(?P<venue>[A-Z0-9_]+)",
        text,
    )
    if coded_match:
        return (
            {
                "venue": coded_match.group("venue"),
                "race_number": int(coded_match.group("race_number")),
                "race_date": coded_match.group("race_date"),
            },
            [],
        )

    venue_date_match = re.fullmatch(
        r"(?P<venue>[A-Za-z0-9_]+)_(?P<race_date>\d{4}-\d{2}-\d{2})_"
        r"(?P<race_number>\d+)",
        text,
    )
    if venue_date_match:
        return (
            {
                "venue": venue_date_match.group("venue").upper(),
                "race_number": int(venue_date_match.group("race_number")),
                "race_date": venue_date_match.group("race_date"),
            },
            [],
        )

    return None, ["legacy_race_id_not_parseable"]


def _candidate_from_classification(item: Mapping[str, Any]) -> dict[str, Any]:
    race_id = str(item.get("race_id") or "")
    lookup_key, blockers = _parse_legacy_race_id(race_id)
    metadata = _mapping(item.get("metadata"))
    verification = _mapping(item.get("verification"))
    lookup_status = "PARSE_READY" if lookup_key and not blockers else "PARSE_BLOCKED"
    return {
        "schema_version": "official_reverify_candidate_v1",
        "legacy_race_id": race_id,
        "legacy_db_path": item.get("legacy_db_path"),
        "legacy_source": item.get("source"),
        "legacy_runner_rows": item.get("legacy_runner_rows"),
        "legacy_winner_name": metadata.get("winner_name"),
        "legacy_race_date_raw": metadata.get("race_date"),
        "verification_status": verification.get("status"),
        "lookup_status": lookup_status,
        "lookup_key": lookup_key,
        "blockers": blockers,
        "next_action": (
            "official_result_dry_run_lookup"
            if lookup_status == "PARSE_READY"
            else "manual_identifier_mapping_required"
        ),
        "writes_performed": dict(WRITES_PERFORMED),
    }


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def build_queue(
    *,
    verification_packet_path: Path,
    queue_output_path: Path,
) -> dict[str, Any]:
    packet = _load_json(verification_packet_path)
    race_classifications = packet.get("race_classifications")
    if not isinstance(race_classifications, list):
        race_classifications = []

    candidates = [
        _candidate_from_classification(item)
        for item in race_classifications
        if isinstance(item, Mapping)
        and item.get("classification") == "result_like_reverify_candidate"
    ]
    _write_jsonl(queue_output_path, candidates)

    source_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    venue_counts: Counter[str] = Counter()
    dates: list[str] = []
    for candidate in candidates:
        source_counts[str(candidate.get("legacy_source") or "DATA_MISSING")] += 1
        status_counts[str(candidate.get("lookup_status") or "DATA_MISSING")] += 1
        for blocker in candidate.get("blockers") or []:
            blocker_counts[str(blocker)] += 1
        lookup_key = _mapping(candidate.get("lookup_key"))
        if lookup_key:
            venue_counts[str(lookup_key.get("venue") or "DATA_MISSING")] += 1
            if lookup_key.get("race_date"):
                dates.append(str(lookup_key["race_date"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "REPORT_ONLY",
        "source_evidence": {
            "verification_packet": str(verification_packet_path.expanduser().resolve()),
            "queue_output": str(queue_output_path.expanduser().resolve()),
        },
        "summary": {
            "candidate_count": len(candidates),
            "parse_ready_count": status_counts.get("PARSE_READY", 0),
            "parse_blocked_count": status_counts.get("PARSE_BLOCKED", 0),
            "source_counts": dict(sorted(source_counts.items())),
            "lookup_status_counts": dict(sorted(status_counts.items())),
            "blocker_counts": dict(sorted(blocker_counts.items())),
            "parse_ready_date_range": {
                "min": min(dates) if dates else None,
                "max": max(dates) if dates else None,
            },
            "top_parse_ready_venues": dict(venue_counts.most_common(20)),
        },
        "queue_preview": candidates[:25],
        "writes_performed": dict(WRITES_PERFORMED),
        "recommended_next_actions": [
            "Run official-result lookup dry-run only for PARSE_READY candidates.",
            "Do not write labels from this queue without complete official positions and explicit approval.",
            "Manually map PARSE_BLOCKED legacy identifiers before any official lookup attempt.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-packet", required=True)
    parser.add_argument("--queue-output", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_queue(
        verification_packet_path=Path(args.verification_packet),
        queue_output_path=Path(args.queue_output),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("status") == "REPORT_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
