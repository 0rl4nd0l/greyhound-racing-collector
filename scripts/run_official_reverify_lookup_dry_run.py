#!/usr/bin/env python3
"""Run report-only official result lookup for legacy reverify candidates.

The input is the JSONL queue produced by build_official_reverify_candidate_queue.
This script may fetch official TheDogs pages, but it never writes labels,
snapshots, manifests, model files, registry state, promotions, or betting output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ingest_results_for_date import (
    THEDOGS_BASE,
    THEDOGS_PUBLIC_HEADERS,
    VENUE_TO_THEDOGS_SLUG,
    parse_thedogs_result_html,
    parse_thedogs_result_html_runner_rows,
    parse_thedogs_result_html_terminal_statuses,
    rendered_text_from_html,
    response_is_forbidden,
    thedogs_result_rows_present,
    title_from_html,
)


SCHEMA_VERSION = "official_reverify_lookup_dry_run_v1"
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

EXTRA_THEDOGS_SLUGS = {
    "DUB": "dubbo",
    "GEE": "geelong",
    "GOSF": "gosford",
    "GOUL": "goulburn",
    "HEA": "healesville",
    "HOR": "horsham",
    "MURR": "murray-bridge",
    "NOR": "northam",
    "TAR": "taree",
    "TARE": "taree",
    "TRA": "traralgon",
    "TWN": "townsville",
    "WAG": "wagga",
    "WAGGA": "wagga",
}

PROTECTED_OUTPUT_DIRS = (
    Path("artifacts/prediction_snapshots"),
    Path("model_registry"),
    Path("docs/model_registry"),
    Path("ml_models_v4"),
    Path("advanced_models"),
)


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
        key = _fixture_key_from_url(url)
        fixture_path = self.fixture_dir / f"{key}.html"
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


def _assert_report_output_dir_safe(output_dir: Path) -> None:
    lexical = output_dir.expanduser()
    if not lexical.is_absolute():
        lexical = REPO_ROOT / lexical
    try:
        lexical_relative = lexical.absolute().relative_to(REPO_ROOT)
    except ValueError:
        lexical_relative = None
    if lexical_relative == Path("."):
        raise ValueError("protected_output_dir:.")
    if lexical_relative is not None:
        for protected in PROTECTED_OUTPUT_DIRS:
            if lexical_relative == protected or protected in lexical_relative.parents:
                raise ValueError(f"protected_output_dir:{protected.as_posix()}")

    resolved = output_dir.expanduser().resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return

    if relative == Path("."):
        raise ValueError("protected_output_dir:.")
    for protected in PROTECTED_OUTPUT_DIRS:
        if relative == protected or protected in relative.parents:
            raise ValueError(f"protected_output_dir:{protected.as_posix()}")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _venue_slug(venue: Any) -> str | None:
    key = str(venue or "").strip().upper().replace(" ", "_")
    if not key:
        return None
    return VENUE_TO_THEDOGS_SLUG.get(key) or EXTRA_THEDOGS_SLUGS.get(key)


def _candidate_urls(*, slug: str, race_date: str, race_number: int) -> list[str]:
    base = f"{THEDOGS_BASE}/racing/{slug}/{race_date}/{int(race_number)}"
    return [
        f"{base}/results?trial=false",
        f"{base}/results",
        f"{base}?trial=false",
        base,
    ]


def _position_payload(
    positions_by_box: Mapping[int, int],
    *,
    dog_names_by_box: Mapping[int, str | None] | None = None,
) -> list[dict[str, Any]]:
    dog_names = dog_names_by_box or {}
    payload = []
    for box, position in sorted(
        positions_by_box.items(), key=lambda item: (int(item[1]), int(item[0]))
    ):
        row = {"box_number": int(box), "finish_position": int(position)}
        if int(box) in dog_names:
            row["dog_name"] = dog_names.get(int(box))
        payload.append(row)
    return payload


def _terminal_payload(statuses_by_box: Mapping[int, str]) -> list[dict[str, Any]]:
    return [
        {"box_number": int(box), "status": str(status)}
        for box, status in sorted(statuses_by_box.items(), key=lambda item: int(item[0]))
    ]


def _label_write_reasons(
    *,
    positions_by_box: Mapping[int, int],
    terminal_statuses_by_box: Mapping[int, str],
    legacy_runner_rows: Any,
) -> list[str]:
    reasons: list[str] = []
    positions = [int(value) for value in positions_by_box.values()]
    if not positions:
        reasons.append("official_positions_missing")
        return reasons
    if 1 not in positions:
        reasons.append("official_first_place_missing")
    if len(positions) != len(set(positions)):
        reasons.append("official_duplicate_finish_positions")

    try:
        expected_rows = int(legacy_runner_rows)
    except (TypeError, ValueError):
        expected_rows = 0
    if expected_rows <= 0:
        reasons.append("legacy_runner_count_missing")
    if expected_rows > 0 and len(positions_by_box) != expected_rows:
        reasons.append("official_positions_incomplete_for_legacy_runner_count")
    if terminal_statuses_by_box:
        reasons.append("official_terminal_statuses_present")
    if expected_rows > 0 and sorted(positions) != list(range(1, len(positions) + 1)):
        reasons.append("official_finish_positions_not_contiguous")
    return sorted(set(reasons))


def _fetch_official_result(
    *,
    candidate: Mapping[str, Any],
    http_client,
    timeout_seconds: float,
) -> dict[str, Any]:
    lookup_key = _mapping(candidate.get("lookup_key"))
    venue = lookup_key.get("venue")
    slug = _venue_slug(venue)
    if not slug:
        return {
            "lookup_status": "VENUE_SLUG_MISSING",
            "result_parse_ready": False,
            "label_write_ready": False,
            "skip_reasons": ["venue_slug_missing"],
            "attempted_urls": [],
            "source_url": None,
            "positions": [],
            "terminal_statuses": [],
        }

    race_date = str(lookup_key.get("race_date") or "").strip()
    race_number = int(lookup_key.get("race_number") or 0)
    urls = _candidate_urls(slug=slug, race_date=race_date, race_number=race_number)
    last_error = "official_result_not_found"
    for url in urls:
        try:
            response = http_client.get(
                url,
                headers=THEDOGS_PUBLIC_HEADERS,
                timeout=timeout_seconds,
                allow_redirects=True,
            )
        except Exception as exc:  # noqa: BLE001 - dry-run packet must capture fetch failures.
            last_error = f"official_fetch_error:{type(exc).__name__}"
            continue

        markup = getattr(response, "text", "") or ""
        source_url = getattr(response, "url", url)
        text = rendered_text_from_html(markup)
        status_code = getattr(response, "status_code", None)
        if response_is_forbidden(status_code, title_from_html(markup), text):
            return {
                "lookup_status": "OFFICIAL_FETCH_FORBIDDEN",
                "result_parse_ready": False,
                "label_write_ready": False,
                "skip_reasons": ["official_fetch_forbidden"],
                "attempted_urls": urls,
                "source_url": source_url,
                "positions": [],
                "terminal_statuses": [],
            }
        if status_code and status_code >= 400:
            last_error = f"official_http_{status_code}"
            continue

        official_runner_rows = parse_thedogs_result_html_runner_rows(markup)
        official_dog_names_by_box = {
            int(row["box_number"]): row.get("dog_name")
            for row in official_runner_rows
            if row.get("finish_position") is not None and row.get("box_number") is not None
        }
        positions_by_box = parse_thedogs_result_html(markup)
        terminal_statuses_by_box = parse_thedogs_result_html_terminal_statuses(markup)
        if positions_by_box:
            skip_reasons = _label_write_reasons(
                positions_by_box=positions_by_box,
                terminal_statuses_by_box=terminal_statuses_by_box,
                legacy_runner_rows=candidate.get("legacy_runner_rows"),
            )
            return {
                "lookup_status": "OFFICIAL_RESULT_PARSED",
                "result_parse_ready": True,
                "label_write_ready": not skip_reasons,
                "skip_reasons": skip_reasons,
                "attempted_urls": urls,
                "source_url": source_url,
                "positions": _position_payload(
                    positions_by_box,
                    dog_names_by_box=official_dog_names_by_box,
                ),
                "terminal_statuses": _terminal_payload(terminal_statuses_by_box),
                "official_runner_rows": official_runner_rows,
            }
        if thedogs_result_rows_present(markup):
            last_error = "official_result_table_without_strict_positions"
        else:
            last_error = "official_positions_not_found"

    return {
        "lookup_status": "OFFICIAL_RESULT_NOT_PARSED",
        "result_parse_ready": False,
        "label_write_ready": False,
        "skip_reasons": [last_error],
        "attempted_urls": urls,
        "source_url": urls[0] if urls else None,
        "positions": [],
        "terminal_statuses": [],
    }


def _result_for_queue_row(
    *,
    candidate: Mapping[str, Any],
    http_client,
    timeout_seconds: float,
) -> dict[str, Any]:
    base = {
        "schema_version": "official_reverify_lookup_result_v1",
        "legacy_race_id": candidate.get("legacy_race_id"),
        "legacy_runner_rows": candidate.get("legacy_runner_rows"),
        "lookup_key": candidate.get("lookup_key"),
    }
    if candidate.get("lookup_status") != "PARSE_READY":
        return {
            **base,
            "lookup_status": "QUEUE_PARSE_BLOCKED",
            "result_parse_ready": False,
            "label_write_ready": False,
            "skip_reasons": ["legacy_lookup_parse_blocked"],
            "attempted_urls": [],
            "source_url": None,
            "positions": [],
            "terminal_statuses": [],
        }
    return {
        **base,
        **_fetch_official_result(
            candidate=candidate,
            http_client=http_client,
            timeout_seconds=timeout_seconds,
        ),
    }


def _write_report(path: Path, packet: Mapping[str, Any]) -> None:
    summary = packet["summary"]
    lines = [
        "# Official Reverify Lookup Dry Run",
        "",
        "Status: `REPORT_ONLY`.",
        "",
        "No DB writes, label writes, snapshot mutations, manifest mutations, model training, model-registry state changes, promotions, betting decisions, or expected-value assertions were performed.",
        "",
        "## Summary",
        "",
        f"- Queue rows seen: `{summary['queue_rows_seen']}`",
        f"- Official fetch attempted: `{summary['official_fetch_attempted_count']}`",
        f"- Result parse ready: `{summary['result_parse_ready_count']}`",
        f"- Label write ready: `{summary['label_write_ready_count']}`",
        f"- Lookup status counts: `{summary['lookup_status_counts']}`",
        f"- Label-write skip reasons: `{summary['label_write_skip_reason_counts']}`",
        "",
        "## Next Step",
        "",
        "Review label-write-ready rows and parser failures. Do not write labels until complete official positions are verified and explicit approval is provided.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_lookup_packet(
    *,
    queue_path: Path,
    output_dir: Path,
    http_client=None,
    max_candidates: int | None = None,
    timeout_seconds: float = 20.0,
    progress_every: int = 0,
) -> dict[str, Any]:
    _assert_report_output_dir_safe(output_dir)
    candidates = _load_jsonl(queue_path)
    if max_candidates is not None:
        candidates = candidates[: max(0, int(max_candidates))]

    client = http_client or StatelessHttpClient(timeout_seconds=timeout_seconds)
    results = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        results.append(
            _result_for_queue_row(
                candidate=candidate,
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

    lookup_status_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    for result in results:
        lookup_status_counts[str(result.get("lookup_status") or "DATA_MISSING")] += 1
        for reason in result.get("skip_reasons") or []:
            skip_reason_counts[str(reason)] += 1

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
            "queue_rows_seen": len(candidates),
            "official_fetch_attempted_count": sum(
                1 for result in results if result.get("attempted_urls")
            ),
            "result_parse_ready_count": sum(
                1 for result in results if result.get("result_parse_ready") is True
            ),
            "label_write_ready_count": sum(
                1 for result in results if result.get("label_write_ready") is True
            ),
            "lookup_status_counts": dict(sorted(lookup_status_counts.items())),
            "label_write_skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        },
        "writes_performed": dict(WRITES_PERFORMED),
        "results": results,
    }
    packet_path = output_dir / "official_reverify_lookup_packet.json"
    packet_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report(output_dir / "report.md", packet)
    return packet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--fixture-dir")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    if str(os.environ.get("APPROVE_RESULT_LABEL_WRITE") or "").strip():
        raise SystemExit("refusing dry-run lookup while APPROVE_RESULT_LABEL_WRITE is set")

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    http_client = (
        FixtureHttpClient(Path(args.fixture_dir))
        if args.fixture_dir
        else StatelessHttpClient(timeout_seconds=args.timeout_seconds)
    )
    packet = build_lookup_packet(
        queue_path=Path(args.queue),
        output_dir=Path(args.output_dir),
        http_client=http_client,
        max_candidates=args.max_candidates,
        timeout_seconds=args.timeout_seconds,
        progress_every=args.progress_every,
    )
    print(json.dumps({"status": packet["status"], "summary": packet["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
