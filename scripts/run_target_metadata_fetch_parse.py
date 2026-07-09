#!/usr/bin/env python3
"""Fetch and parse target distance/grade metadata from an approved manifest.

This is a report-only helper for the greyhound accuracy work. Real TheDogs
HTTP fetches require the exact approval phrase below. Fixture mode is available
for tests and parser development without any network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingest_results_for_date import (  # noqa: E402
    THEDOGS_PUBLIC_HEADERS,
    rendered_text_from_html,
    response_is_forbidden,
    title_from_html,
)


SCHEMA_VERSION = "target_metadata_fetch_parse_report_v1"
RESULT_SCHEMA_VERSION = "target_metadata_fetch_parse_result_v1"
APPROVAL_TEXT = (
    "Approve fetch/parse-only target distance and grade collection for the current "
    "214 TheDogs URL races, with no DB writes and no label writes."
)

REQUIRED_MANIFEST_FIELDS = (
    "race_id",
    "race_date",
    "venue",
    "race_number",
    "thedogs_url",
)
OUTPUT_FIELDS = (
    "race_id",
    "source_url",
    "http_status",
    "parse_status",
    "parsed_distance",
    "parsed_grade",
    "parsed_race_time",
    "parsed_start_datetime",
    "source_snippet_hash",
    "parse_notes",
    "safe_for_metadata_review",
)
ALLOWED_OUTPUT_PREFIX = Path("artifacts/full_evidence_orchestration_20260525")


class StatelessHttpClient:
    def __init__(self, *, timeout_seconds: float):
        import requests

        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.cookies.clear()

    def get(self, url: str, **kwargs):
        kwargs.setdefault("headers", THEDOGS_PUBLIC_HEADERS)
        kwargs.setdefault("timeout", self.timeout_seconds)
        kwargs.setdefault("allow_redirects", True)
        kwargs["cookies"] = {}
        return self.session.get(url, **kwargs)


class FixtureHttpClient:
    def __init__(self, fixture_dir: Path):
        self.fixture_dir = fixture_dir

    def get(self, url: str, **_kwargs):
        fixture_path = self.fixture_dir / f"{_fixture_key_from_url(url)}.html"
        if fixture_path.exists():
            return _FixtureResponse(
                status_code=200,
                text=fixture_path.read_text(encoding="utf-8"),
                url=url,
            )
        return _FixtureResponse(status_code=404, text="", url=url)


class _FixtureResponse:
    def __init__(self, *, status_code: int, text: str, url: str):
        self.status_code = status_code
        self.text = text
        self.url = url


def _fixture_key_from_url(url: str) -> str:
    parts = [part for part in str(url).split("?")[0].split("/") if part]
    try:
        racing_index = parts.index("racing")
    except ValueError:
        return "unknown"
    slug = parts[racing_index + 1] if len(parts) > racing_index + 1 else "unknown"
    race_date = parts[racing_index + 2] if len(parts) > racing_index + 2 else "unknown"
    race_number = parts[racing_index + 3] if len(parts) > racing_index + 3 else "unknown"
    return f"{slug}_{race_date}_{race_number}"


def _assert_report_output_dir_safe(output_dir: Path) -> Path:
    root = REPO_ROOT.expanduser().resolve()
    candidate = output_dir.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"output_dir_must_be_inside_repo:{resolved}") from exc
    if not (relative == ALLOWED_OUTPUT_PREFIX or ALLOWED_OUTPUT_PREFIX in relative.parents):
        raise ValueError(f"output_dir_must_be_under_artifacts:{relative}")
    return resolved


def _writes_performed(*, official_fetch: bool) -> dict[str, bool]:
    return {
        "db_write": False,
        "label_write": False,
        "metadata_write": False,
        "official_fetch": bool(official_fetch),
        "snapshot_mutation": False,
        "manifest_mutation": False,
        "model_training": False,
        "model_persistence": False,
        "registry_mutation": False,
        "github_write": False,
        "promotion": False,
        "ev_action": False,
        "betting_decision": False,
    }


def _normalise_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _clean_grade(value: str | None) -> str | None:
    cleaned = _normalise_spaces(value or "")
    cleaned = re.sub(
        r"\b(?:Distance|Dist|Start\s*Time|Race\s*Time|Time|Prize|Track)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.strip(" :-\t\r\n")
    return cleaned or None


def _line_value(lines: list[str], labels: tuple[str, ...]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    for index, line in enumerate(lines):
        stripped = line.strip()
        same_line = re.match(
            rf"^(?:Race\s+)?(?:{label_pattern})\s*[:\-]?\s*(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if same_line:
            value = _normalise_spaces(same_line.group(1))
            if value:
                return value
        if re.match(
            rf"^(?:Race\s+)?(?:{label_pattern})\s*[:\-]?$",
            stripped,
            flags=re.IGNORECASE,
        ):
            for next_line in lines[index + 1 : index + 4]:
                value = _normalise_spaces(next_line)
                if value:
                    return value
    return None


def _parse_distance(text: str, lines: list[str], notes: list[str]) -> str | None:
    labelled = _line_value(lines, ("Distance", "Dist"))
    if labelled:
        match = re.search(r"\b([2-9]\d{2,3})\s*m\b", labelled, flags=re.IGNORECASE)
        if match:
            return f"{int(match.group(1))}m"

    compact = _normalise_spaces(text)
    match = re.search(
        r"\b(?:Distance|Dist)\s*[:\-]?\s*([2-9]\d{2,3})\s*m\b",
        compact,
        flags=re.IGNORECASE,
    )
    if match:
        return f"{int(match.group(1))}m"

    fallback = re.search(r"\b([2-9]\d{2,3})\s*m\b", compact, flags=re.IGNORECASE)
    if fallback:
        notes.append("distance_fallback_unlabelled_metre_token")
        return f"{int(fallback.group(1))}m"
    notes.append("distance_missing")
    return None


def _parse_grade(lines: list[str], notes: list[str]) -> str | None:
    labelled = _line_value(lines, ("Grade", "Class"))
    cleaned = _clean_grade(labelled)
    if cleaned:
        return cleaned

    compact = _normalise_spaces(" ".join(lines))
    compact_match = re.search(
        r"\b(?:Grade|Class)\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9 /\-]{0,40}?)"
        r"(?=\s+(?:Distance|Dist|Start\s*Time|Race\s*Time|Time|Prize|Track)\b|$)",
        compact,
        flags=re.IGNORECASE,
    )
    cleaned = _clean_grade(compact_match.group(1) if compact_match else None)
    if cleaned:
        return cleaned

    for line in lines:
        match = re.search(
            r"\b(?:Grade|Class)\s*[:\-]\s*([A-Za-z0-9][A-Za-z0-9 /\-]{0,40})",
            line,
            flags=re.IGNORECASE,
        )
        cleaned = _clean_grade(match.group(1) if match else None)
        if cleaned:
            return cleaned
    notes.append("grade_missing")
    return None


def _parse_race_time(text: str, lines: list[str], notes: list[str]) -> str | None:
    labelled = _line_value(lines, ("Start Time", "Race Time", "Time"))
    candidates = [labelled or "", _normalise_spaces(text)]
    for candidate in candidates:
        match = re.search(
            r"\b([01]?\d|2[0-3]):([0-5]\d)\s*([AP]M)?\b",
            candidate,
            flags=re.IGNORECASE,
        )
        if match:
            hour = match.group(1)
            minute = match.group(2)
            suffix = (match.group(3) or "").upper()
            return f"{int(hour)}:{minute}{(' ' + suffix) if suffix else ''}"
    notes.append("race_time_missing")
    return None


def _start_datetime(race_date: str, race_time: str | None, notes: list[str]) -> str | None:
    if not race_time:
        notes.append("start_datetime_missing_no_time")
        return None
    raw_date = str(race_date or "").strip()
    try:
        parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError:
        notes.append("start_datetime_missing_bad_date")
        return None

    normalised_time = re.sub(
        r"\s*([AP]M)$",
        r" \1",
        race_time.strip().upper(),
        flags=re.IGNORECASE,
    )
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            parsed_time = datetime.strptime(normalised_time, fmt).time()
            return datetime.combine(parsed_date, parsed_time).isoformat()
        except ValueError:
            continue
    notes.append("start_datetime_missing_bad_time")
    return None


def parse_target_metadata_from_text(
    text: str,
    *,
    race_date: str,
) -> dict[str, Any]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    notes: list[str] = []
    parsed_distance = _parse_distance(text, lines, notes)
    parsed_grade = _parse_grade(lines, notes)
    parsed_race_time = _parse_race_time(text, lines, notes)
    parsed_start_datetime = _start_datetime(race_date, parsed_race_time, notes)

    if parsed_distance and parsed_grade:
        parse_status = "METADATA_PARSED"
    elif parsed_distance or parsed_grade or parsed_race_time:
        parse_status = "METADATA_PARTIAL"
    else:
        parse_status = "METADATA_NOT_PARSED"

    return {
        "parse_status": parse_status,
        "parsed_distance": parsed_distance,
        "parsed_grade": parsed_grade,
        "parsed_race_time": parsed_race_time,
        "parsed_start_datetime": parsed_start_datetime,
        "parse_notes": sorted(set(notes)),
    }


def _load_manifest(path: Path, *, expected_races: int | None) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [
            field
            for field in REQUIRED_MANIFEST_FIELDS
            if field not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"manifest_missing_fields:{','.join(missing)}")
        rows = [dict(row) for row in reader]

    if expected_races is not None and len(rows) != int(expected_races):
        raise ValueError(f"expected_races_mismatch:{len(rows)}!={int(expected_races)}")

    race_ids = [str(row.get("race_id") or "").strip() for row in rows]
    urls = [str(row.get("thedogs_url") or "").strip() for row in rows]
    if any(not race_id for race_id in race_ids):
        raise ValueError("manifest_race_id_missing")
    if any(not url for url in urls):
        raise ValueError("manifest_thedogs_url_missing")
    duplicate_race_ids = [item for item, count in Counter(race_ids).items() if count > 1]
    duplicate_urls = [item for item, count in Counter(urls).items() if count > 1]
    if duplicate_race_ids:
        raise ValueError(f"manifest_duplicate_race_id:{duplicate_race_ids[0]}")
    if duplicate_urls:
        raise ValueError(f"manifest_duplicate_thedogs_url:{duplicate_urls[0]}")
    return rows


def _source_hash(text: str) -> str:
    snippet = _normalise_spaces(text)[:4000]
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def _empty_result(row: Mapping[str, Any], *, parse_status: str, note: str) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "race_id": row.get("race_id"),
        "race_date": row.get("race_date"),
        "venue": row.get("venue"),
        "race_number": row.get("race_number"),
        "manifest_url": row.get("thedogs_url"),
        "source_url": row.get("thedogs_url"),
        "http_status": None,
        "parse_status": parse_status,
        "parsed_distance": None,
        "parsed_grade": None,
        "parsed_race_time": None,
        "parsed_start_datetime": None,
        "source_snippet_hash": None,
        "parse_notes": [note],
        "safe_for_metadata_review": False,
    }


def _fetch_and_parse_row(
    *,
    row: Mapping[str, Any],
    http_client,
    timeout_seconds: float,
) -> dict[str, Any]:
    manifest_url = str(row.get("thedogs_url") or "").strip()
    try:
        response = http_client.get(
            manifest_url,
            headers=THEDOGS_PUBLIC_HEADERS,
            timeout=timeout_seconds,
            allow_redirects=True,
        )
    except Exception as exc:  # noqa: BLE001 - report-only packet records fetch failures.
        return _empty_result(row, parse_status="OFFICIAL_FETCH_ERROR", note=type(exc).__name__)

    markup = getattr(response, "text", "") or ""
    source_url = getattr(response, "url", manifest_url) or manifest_url
    http_status = getattr(response, "status_code", None)
    text = rendered_text_from_html(markup)
    if response_is_forbidden(http_status, title_from_html(markup), text):
        result = _empty_result(
            row,
            parse_status="OFFICIAL_FETCH_FORBIDDEN",
            note="official_fetch_forbidden",
        )
        result["http_status"] = http_status
        result["source_url"] = source_url
        return result
    if http_status and int(http_status) >= 400:
        result = _empty_result(
            row,
            parse_status="OFFICIAL_HTTP_ERROR",
            note=f"official_http_{http_status}",
        )
        result["http_status"] = http_status
        result["source_url"] = source_url
        return result

    parsed = parse_target_metadata_from_text(text, race_date=str(row.get("race_date") or ""))
    source_matches_manifest = str(source_url) == manifest_url
    safe_for_metadata_review = (
        parsed["parse_status"] == "METADATA_PARSED" and source_matches_manifest
    )
    parse_notes = list(parsed["parse_notes"])
    if not source_matches_manifest:
        parse_notes.append("source_url_mismatch")

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "race_id": row.get("race_id"),
        "race_date": row.get("race_date"),
        "venue": row.get("venue"),
        "race_number": row.get("race_number"),
        "manifest_url": manifest_url,
        "source_url": source_url,
        "http_status": http_status,
        "parse_status": parsed["parse_status"],
        "parsed_distance": parsed["parsed_distance"],
        "parsed_grade": parsed["parsed_grade"],
        "parsed_race_time": parsed["parsed_race_time"],
        "parsed_start_datetime": parsed["parsed_start_datetime"],
        "source_snippet_hash": _source_hash(text),
        "parse_notes": sorted(set(parse_notes)),
        "safe_for_metadata_review": safe_for_metadata_review,
    }


def _write_csv(path: Path, results: list[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS))
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in OUTPUT_FIELDS}
            row["parse_notes"] = ";".join(result.get("parse_notes") or [])
            row["safe_for_metadata_review"] = str(bool(result.get("safe_for_metadata_review")))
            writer.writerow(row)


def _write_summary(path: Path, packet: Mapping[str, Any]) -> None:
    summary = packet["summary"]
    lines = [
        "# Target Metadata Fetch/Parse Packet",
        "",
        f"Status: `{packet['status']}`.",
        "",
        (
            "No DB writes, label writes, metadata writes, snapshot or manifest "
            "mutations, model training, registry updates, GitHub writes, EV "
            "actions, or betting decisions were performed."
        ),
        "",
        "## Summary",
        "",
        f"- Manifest rows seen: `{summary['manifest_rows_seen']}`",
        f"- Candidate rows processed: `{summary['candidate_rows_processed']}`",
        f"- Official fetch attempted: `{summary['official_fetch_attempted_count']}`",
        f"- Metadata parsed: `{summary['metadata_parsed_count']}`",
        f"- Safe for metadata review: `{summary['safe_for_metadata_review_count']}`",
        f"- Parse status counts: `{summary['parse_status_counts']}`",
        "",
        "## Approval Boundary",
        "",
        f"- Approval text matched: `{packet['approval']['approval_text_matched']}`",
        f"- Fixture mode: `{packet['approval']['fixture_mode']}`",
        "",
        "## Next",
        "",
        (
            "Review safe candidate rows before any canonical metadata write. "
            "Label writes remain unapproved."
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_target_metadata_packet(
    *,
    manifest_path: Path,
    output_dir: Path,
    approve_fetch_parse: str | None = None,
    fixture_dir: Path | None = None,
    http_client=None,
    expected_races: int | None = None,
    max_candidates: int | None = None,
    timeout_seconds: float = 20.0,
    progress_every: int = 0,
) -> dict[str, Any]:
    if str(os.environ.get("APPROVE_RESULT_LABEL_WRITE") or "").strip():
        raise ValueError("refusing_metadata_lookup_while_APPROVE_RESULT_LABEL_WRITE_is_set")
    if str(os.environ.get("APPROVE_TARGET_METADATA_WRITE") or "").strip():
        raise ValueError("refusing_report_only_lookup_while_APPROVE_TARGET_METADATA_WRITE_is_set")

    output_dir = _assert_report_output_dir_safe(output_dir)
    fixture_mode = fixture_dir is not None
    approval_text_matched = str(approve_fetch_parse or "").strip() == APPROVAL_TEXT
    if not fixture_mode and not approval_text_matched:
        raise ValueError("official_fetch_requires_exact_approval")

    rows = _load_manifest(manifest_path, expected_races=expected_races)
    manifest_rows_seen = len(rows)
    if max_candidates is not None:
        rows = rows[: max(0, int(max_candidates))]

    client = http_client
    if client is None:
        client = (
            FixtureHttpClient(fixture_dir)
            if fixture_mode
            else StatelessHttpClient(timeout_seconds=timeout_seconds)
        )

    results: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        results.append(
            _fetch_and_parse_row(
                row=row,
                http_client=client,
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

    parse_status_counts = Counter(
        str(result.get("parse_status") or "DATA_MISSING") for result in results
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "REPORT_ONLY",
        "approval": {
            "required_for_real_fetch": APPROVAL_TEXT,
            "approval_text_matched": approval_text_matched,
            "fixture_mode": fixture_mode,
        },
        "source_evidence": {
            "manifest": str(manifest_path.expanduser().resolve()),
            "output_dir": str(output_dir.expanduser().resolve()),
        },
        "summary": {
            "manifest_rows_seen": manifest_rows_seen,
            "candidate_rows_processed": len(results),
            "official_fetch_attempted_count": 0 if fixture_mode else len(results),
            "metadata_parsed_count": sum(
                1 for result in results if result.get("parse_status") == "METADATA_PARSED"
            ),
            "safe_for_metadata_review_count": sum(
                1 for result in results if result.get("safe_for_metadata_review") is True
            ),
            "parse_status_counts": dict(sorted(parse_status_counts.items())),
        },
        "writes_performed": _writes_performed(official_fetch=not fixture_mode),
        "results": results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / "target_metadata_fetch_parse_packet.json"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "target_metadata_candidates.csv", results)
    _write_summary(output_dir / "SUMMARY.md", packet)
    return packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-races", type=int, default=None)
    parser.add_argument("--approve-fetch-parse", default=None)
    parser.add_argument("--fixture-dir")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    build_target_metadata_packet(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        approve_fetch_parse=args.approve_fetch_parse,
        fixture_dir=Path(args.fixture_dir) if args.fixture_dir else None,
        expected_races=args.expected_races,
        max_candidates=args.max_candidates,
        timeout_seconds=args.timeout_seconds,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
