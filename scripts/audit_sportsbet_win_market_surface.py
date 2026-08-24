#!/usr/bin/env python3
"""Build a provenance-safe, report-only Sportsbet WIN development surface.

The auditor never writes the source database or any supplied experiment bundle.
It binds development rows to exact ``live_odds`` rows by race ID, box, and
capture timestamp, then reconstructs WIN only from a source-explicit paired
WIN/PLACE rendering retained in ``sportsbet_raw_runner_text``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = "sportsbet_win_market_surface_audit_v1"
SIDECAR_SCHEMA_VERSION = "sportsbet_win_canonical_sidecar_v1"
MATRIX_SCHEMA_VERSION = "canonical_training_matrix_sportsbet_win_recovered_v1"
ANALYSIS_DATE = "2026-08-15"

VERIFIED_WIN = "VERIFIED_WIN"
RECOVERABLE_WIN = "RECOVERABLE_WIN"
PLACE_MISLABEL = "PLACE_MISLABEL"
UNPARSABLE = "UNPARSABLE"
CONFLICTING = "CONFLICTING"
MISSING = "MISSING"
CANONICAL_STATUSES = frozenset({VERIFIED_WIN, RECOVERABLE_WIN, PLACE_MISLABEL})

DECIMAL_PRICE_RE = re.compile(r"^\d+(?:\.\d{1,2})$")
RUNNER_HEADER_RE = re.compile(
    r"^\s*(\d{1,2})\s*[.)]\s+[A-Za-z][A-Za-z'. -]+(?:\s*\((\d{1,2})\))?\s*$"
)


@dataclass(frozen=True)
class WinEvidence:
    classification: str
    canonical_win_odds: float | None
    paired_win_odds: float | None
    paired_place_odds: float | None
    reason: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_root_cause_lineage(
    repo_root: Path, source_revision: str, repair_revision: str
) -> dict[str, Any]:
    current_head = git_output(repo_root, "rev-parse", "HEAD")
    source_code = git_output(
        repo_root, "show", f"{source_revision}:sportsbet_odds_integrator.py"
    )
    required_source_signals = {
        "generic_first_price_selector": (
            "find_element(By.CSS_SELECTOR, \"[data-automation-id*='price-text']\")"
            in source_code
        ),
        "separate_place_market_interaction": (
            "topN = self._select_place_market()" in source_code
        ),
        "paired_column_parser_absent": (
            "def sportsbet_paired_fixed_prices" not in source_code
        ),
    }
    if not all(required_source_signals.values()):
        raise ValueError(
            f"source_revision_root_cause_signal_missing:{required_source_signals}"
        )
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            repair_revision,
            source_revision,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode not in (0, 1):
        raise ValueError("git_ancestry_check_failed")
    return {
        "audit_repo_root": str(repo_root),
        "audit_repo_head": current_head,
        "source_code_signals": required_source_signals,
        "repair_is_ancestor_of_defective_revision": ancestor.returncode == 0,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timezone_missing:{text}")
    return parsed


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _runner_header_boxes(lines: Sequence[str]) -> list[int]:
    boxes: list[int] = []
    for line in lines:
        match = RUNNER_HEADER_RE.fullmatch(line)
        if not match:
            continue
        box = int(match.group(2) or match.group(1))
        if 1 <= box <= 10:
            boxes.append(box)
    return boxes


def classify_win_evidence(
    *, raw_text: Any, expected_box: int, stored_odds: float
) -> WinEvidence:
    """Classify one retained Sportsbet runner row without heuristic fallback."""

    text = str(raw_text or "").replace("\xa0", " ").strip()
    if not text:
        return WinEvidence(MISSING, None, None, None, "raw_runner_text_missing")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_boxes = _runner_header_boxes(lines)
    if not header_boxes:
        return WinEvidence(UNPARSABLE, None, None, None, "runner_header_unparseable")
    if len(header_boxes) != 1:
        return WinEvidence(CONFLICTING, None, None, None, "multiple_runner_headers")
    if header_boxes[0] != int(expected_box):
        return WinEvidence(CONFLICTING, None, None, None, "raw_runner_box_conflict")

    ew_indexes = [index for index, line in enumerate(lines) if line.upper() == "EW"]
    if not ew_indexes:
        return WinEvidence(UNPARSABLE, None, None, None, "ew_control_missing")
    if len(ew_indexes) != 1:
        return WinEvidence(CONFLICTING, None, None, None, "multiple_ew_controls")

    prices = [
        float(line)
        for line in lines[: ew_indexes[0]]
        if DECIMAL_PRICE_RE.fullmatch(line) and float(line) > 1.0
    ]
    if len(prices) < 2:
        return WinEvidence(UNPARSABLE, None, None, None, "paired_prices_incomplete")

    win_price, place_price = prices[-2:]
    if place_price > win_price:
        return WinEvidence(
            CONFLICTING,
            None,
            win_price,
            place_price,
            "paired_market_order_conflict",
        )

    if math.isclose(stored_odds, win_price, rel_tol=0.0, abs_tol=1e-12):
        classification = VERIFIED_WIN
        reason = "stored_price_matches_source_paired_win"
    elif math.isclose(stored_odds, place_price, rel_tol=0.0, abs_tol=1e-12):
        classification = PLACE_MISLABEL
        reason = "stored_price_matches_source_paired_place"
    else:
        classification = RECOVERABLE_WIN
        reason = "stored_price_differs_from_both_source_paired_prices"
    return WinEvidence(classification, win_price, win_price, place_price, reason)


def qualify_races(rows: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        grouped[str(row["race_id"])].append(str(row["classification"]))
    return {
        race_id: bool(statuses) and all(status in CANONICAL_STATUSES for status in statuses)
        for race_id, statuses in grouped.items()
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
    return rows


def read_matrix_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            row = json.loads(stripped)
            row_digest = sha256_bytes(stripped + b"\n")
            row["_matrix_row_index"] = len(rows)
            row["_matrix_row_sha256"] = row_digest
            rows.append(row)
    return rows


def connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def validate_inputs(
    matrix_root: Path, model_report_path: Path, source_db: Path
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
    paths = {
        "matrix_jsonl": matrix_root / "training_matrix.jsonl",
        "matrix_csv": matrix_root / "training_matrix.csv",
        "matrix_exclusions": matrix_root / "exclusions.jsonl",
        "matrix_report": matrix_root / "report.json",
        "model_report": model_report_path,
        "source_db": source_db,
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    matrix_report = read_json(paths["matrix_report"])
    model_report = read_json(model_report_path)

    filename_keys = {
        "training_matrix.jsonl": "matrix_jsonl",
        "training_matrix.csv": "matrix_csv",
        "exclusions.jsonl": "matrix_exclusions",
    }
    for filename, key in filename_keys.items():
        expected_matrix = str(matrix_report["files"][filename]["sha256"])
        expected_model = str(model_report["matrix_files"][filename])
        if hashes[key] != expected_matrix or hashes[key] != expected_model:
            raise ValueError(f"sealed_matrix_hash_mismatch:{filename}")
    if hashes["source_db"] != str(model_report["source_snapshot_sha256"]):
        raise ValueError("source_db_hash_mismatch")
    if matrix_report.get("leakage_status") != "PASS":
        raise ValueError("matrix_leakage_status_not_pass")
    if matrix_report.get("odds_policy") != (
        "latest valid Sportsbet win odds per box strictly before jump"
    ):
        raise ValueError("unexpected_matrix_odds_policy")
    if model_report.get("august_boundary", {}).get("outcomes_opened"):
        raise ValueError("august_outcomes_opened")

    rows = read_matrix_rows(paths["matrix_jsonl"])
    if len(rows) != int(matrix_report["included_runner_rows"]):
        raise ValueError("matrix_runner_count_mismatch")
    if len({str(row["race_id"]) for row in rows}) != int(
        matrix_report["included_races"]
    ):
        raise ValueError("matrix_race_count_mismatch")
    return rows, hashes, matrix_report, model_report


def load_selected_odds(
    connection: sqlite3.Connection,
    race_ids: set[str],
    jump_by_race: Mapping[str, datetime],
) -> dict[tuple[str, int], dict[str, Any]]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    for sqlite_row in connection.execute("SELECT * FROM live_odds ORDER BY id"):
        row = dict(sqlite_row)
        race_id = str(row.get("race_id") or "")
        if race_id not in race_ids:
            continue
        try:
            box = int(row.get("box_number"))
            odds = float(row.get("odds_decimal"))
            captured = parse_datetime(row.get("capture_timestamp"))
        except (TypeError, ValueError):
            continue
        if str(row.get("market_type") or "").casefold() != "win":
            continue
        if str(row.get("source") or "").casefold() != "sportsbet":
            continue
        if not math.isfinite(odds) or odds <= 1.0 or captured >= jump_by_race[race_id]:
            continue
        if urlparse(str(row.get("source_url") or "")).netloc.casefold() != (
            "www.sportsbet.com.au"
        ):
            continue
        key = (race_id, box)
        prior = selected.get(key)
        if prior is None or parse_datetime(prior["capture_timestamp"]) < captured:
            selected[key] = row
    return selected


def build_development_surface(
    matrix_rows: list[dict[str, Any]], source_db: Path, source_db_sha256: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], str]:
    jumps: dict[str, datetime] = {}
    race_boxes: dict[str, set[int]] = defaultdict(set)
    for row in matrix_rows:
        race_id = str(row["race_id"])
        jump = parse_datetime(row["jump_at"])
        if race_id in jumps and jumps[race_id] != jump:
            raise ValueError(f"matrix_jump_conflict:{race_id}")
        jumps[race_id] = jump
        box = int(row["box_number"])
        if box in race_boxes[race_id]:
            raise ValueError(f"matrix_duplicate_box:{race_id}:box_{box}")
        race_boxes[race_id].add(box)

    connection = connect_read_only(source_db)
    try:
        selected = load_selected_odds(connection, set(jumps), jumps)
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()
    if integrity != "ok":
        raise ValueError(f"sqlite_integrity_check_failed:{integrity}")

    sidecar: list[dict[str, Any]] = []
    for row in matrix_rows:
        race_id = str(row["race_id"])
        box = int(row["box_number"])
        odds_row = selected.get((race_id, box))
        if odds_row is None:
            evidence = WinEvidence(MISSING, None, None, None, "selected_live_odds_row_missing")
            odds_row = {}
        else:
            if str(row.get("odds_capture_timestamp")) != str(
                odds_row.get("capture_timestamp")
            ):
                raise ValueError(f"matrix_capture_timestamp_mismatch:{race_id}:box_{box}")
            if normalize_name(row.get("dog_name")) != normalize_name(
                odds_row.get("dog_name")
            ):
                raise ValueError(f"matrix_live_odds_name_conflict:{race_id}:box_{box}")
            evidence = classify_win_evidence(
                raw_text=odds_row.get("sportsbet_raw_runner_text"),
                expected_box=box,
                stored_odds=float(odds_row["odds_decimal"]),
            )

        raw_text = str(odds_row.get("sportsbet_raw_runner_text") or "")
        item = {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "race_id": race_id,
            "race_date": str(row["race_date"]),
            "box_number": box,
            "dog_name": str(row["dog_name"]),
            "matrix_row_index": int(row["_matrix_row_index"]),
            "matrix_row_sha256": str(row["_matrix_row_sha256"]),
            "source_db_sha256": source_db_sha256,
            "source_table": "live_odds",
            "source_row_id": odds_row.get("id"),
            "source_row_identity": (
                f"{source_db_sha256}:live_odds:{odds_row['id']}"
                if odds_row.get("id") is not None
                else None
            ),
            "source": odds_row.get("source"),
            "source_url": odds_row.get("source_url"),
            "capture_timestamp": odds_row.get("capture_timestamp"),
            "capture_mode": odds_row.get("capture_mode"),
            "stored_market_type": odds_row.get("market_type"),
            "stored_odds_decimal": odds_row.get("odds_decimal"),
            "raw_runner_text_sha256": sha256_bytes(raw_text.encode("utf-8"))
            if raw_text
            else None,
            **asdict(evidence),
        }
        sidecar.append(item)

    race_qualified = qualify_races(sidecar)
    for item in sidecar:
        item["race_qualified"] = race_qualified[item["race_id"]]
        item["sidecar_row_sha256"] = sha256_bytes(canonical_json_bytes(item))

    canonical_matrix: list[dict[str, Any]] = []
    matrix_by_identity = {
        int(row["_matrix_row_index"]): row for row in matrix_rows
    }
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sidecar:
        if item["race_qualified"]:
            by_race[str(item["race_id"])].append(item)

    absolute_deltas: list[float] = []
    race_l1: list[float] = []
    favourite_changes = 0
    price_changes = 0
    for race_id in sorted(by_race):
        items = by_race[race_id]
        raw = {int(item["box_number"]): 1.0 / float(item["canonical_win_odds"]) for item in items}
        total = sum(raw.values())
        corrected = {box: implied / total for box, implied in raw.items()}
        old = {
            int(item["box_number"]): float(
                matrix_by_identity[int(item["matrix_row_index"])][
                    "market_implied_probability"
                ]
            )
            for item in items
        }
        old_favourite = min(old, key=lambda box: (-old[box], box))
        new_favourite = min(corrected, key=lambda box: (-corrected[box], box))
        favourite_changes += int(old_favourite != new_favourite)
        l1 = 0.0
        for item in sorted(items, key=lambda value: int(value["box_number"])):
            index = int(item["matrix_row_index"])
            original = matrix_by_identity[index]
            box = int(item["box_number"])
            delta = corrected[box] - old[box]
            absolute_deltas.append(abs(delta))
            l1 += abs(delta)
            price_changes += int(
                not math.isclose(
                    float(item["stored_odds_decimal"]),
                    float(item["canonical_win_odds"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            output_row = {
                key: value
                for key, value in original.items()
                if not key.startswith("_matrix_")
            }
            output_row["schema_version"] = MATRIX_SCHEMA_VERSION
            output_row["market_implied_probability_original"] = old[box]
            output_row["market_implied_probability"] = corrected[box]
            output_row["market_probability_delta"] = delta
            output_row["canonical_sportsbet_win_odds"] = item["canonical_win_odds"]
            output_row["sportsbet_win_evidence_classification"] = item[
                "classification"
            ]
            output_row["sportsbet_win_source_row_id"] = item["source_row_id"]
            output_row["sportsbet_win_raw_sha256"] = item[
                "raw_runner_text_sha256"
            ]
            output_row["sportsbet_win_sidecar_row_sha256"] = item[
                "sidecar_row_sha256"
            ]
            canonical_matrix.append(output_row)
        race_l1.append(l1)

    probability_report = {
        "qualified_races": len(by_race),
        "qualified_runner_rows": len(canonical_matrix),
        "price_changed_runner_rows": price_changes,
        "probability_changed_runner_rows": sum(delta > 1e-15 for delta in absolute_deltas),
        "mean_absolute_probability_delta": mean(absolute_deltas),
        "median_absolute_probability_delta": median(absolute_deltas),
        "maximum_absolute_probability_delta": max(absolute_deltas),
        "mean_race_l1_probability_distance": mean(race_l1),
        "maximum_race_l1_probability_distance": max(race_l1),
        "favourite_identity_changed_races": favourite_changes,
    }
    return sidecar, canonical_matrix, probability_report, integrity


def _classification_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row["classification"]) for row in rows)
    return {
        status: counts.get(status, 0)
        for status in (
            VERIFIED_WIN,
            RECOVERABLE_WIN,
            PLACE_MISLABEL,
            UNPARSABLE,
            CONFLICTING,
            MISSING,
        )
    }


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return (
        str(row["race_id"]),
        int(row["box_number"]),
        str(row.get("odds_capture_timestamp", row.get("capture_timestamp"))),
    )


def _artifact_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths if path.is_file()}


def require_hash_references(paths: Sequence[Path], expected: Mapping[str, str]) -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    missing = [name for name, digest in expected.items() if digest not in text]
    if missing:
        raise ValueError(f"dependency_input_hash_reference_missing:{missing}")


def _dependency_summary(
    *,
    experiment: str,
    classification: str,
    rows: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Path],
    binding: str,
    reason: str,
) -> dict[str, Any]:
    counts = _classification_counts(rows)
    affected = counts[PLACE_MISLABEL] + counts[RECOVERABLE_WIN]
    unrecoverable = counts[UNPARSABLE] + counts[CONFLICTING] + counts[MISSING]
    return {
        "experiment": experiment,
        "classification": classification,
        "exact_input_runner_rows": len(rows),
        "exact_input_races": len({str(row["race_id"]) for row in rows}),
        "classification_counts": counts,
        "affected_runner_rows": affected,
        "unrecoverable_runner_rows": unrecoverable,
        "exact_row_binding": binding,
        "artifact_sha256": _artifact_hashes(artifacts),
        "reason": reason,
    }


def bind_legacy_training_matrix(
    matrix_path: Path, sidecar: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Bind an older matrix to this surface by exact row identity and probability."""

    rows = read_jsonl(matrix_path)
    by_key = {_row_key(row): dict(row) for row in sidecar}
    if len(by_key) != len(sidecar):
        raise ValueError("canonical_sidecar_duplicate_row_key")

    stored_probability: dict[tuple[str, int, str], float] = {}
    by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in sidecar:
        by_race[str(row["race_id"])].append(row)
    for race_rows in by_race.values():
        denominator = sum(1.0 / float(row["stored_odds_decimal"]) for row in race_rows)
        for row in race_rows:
            stored_probability[_row_key(row)] = (
                1.0 / float(row["stored_odds_decimal"])
            ) / denominator

    bound: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for row in rows:
        key = _row_key(row)
        if key in seen:
            raise ValueError(f"legacy_training_matrix_duplicate_row:{key}")
        seen.add(key)
        source = by_key.get(key)
        if source is None:
            raise ValueError(f"legacy_training_matrix_row_not_in_surface:{key}")
        if not math.isclose(
            float(row["market_implied_probability"]),
            stored_probability[key],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"legacy_training_matrix_probability_mismatch:{key}")
        bound.append(source)
    if seen != set(by_key):
        raise ValueError("legacy_training_matrix_does_not_bind_full_surface")
    return bound


def audit_historical_tier_a(
    *,
    runners_path: Path,
    provenance_path: Path,
    manifest_path: Path,
    evaluation_path: Path,
    source_db: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve exact Tier-A runner rows to retained immutable Sportsbet evidence."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    exact_sets = manifest["exact_eligibility_sets"]
    if exact_sets["all_exact_runner_ids_file"]["sha256"] != sha256_file(
        runners_path
    ):
        raise ValueError("tier_a_runner_manifest_hash_mismatch")
    if exact_sets["tier_a_race_and_runner_provenance_file"][
        "sha256"
    ] != sha256_file(provenance_path):
        raise ValueError("tier_a_provenance_manifest_hash_mismatch")

    with runners_path.open(encoding="utf-8", newline="") as handle:
        tier_a = [
            row for row in csv.DictReader(handle) if row["strongest_tier"] == "A"
        ]
    race_ids = sorted({str(row["race_id"]) for row in tier_a})
    sample = evaluation["sample"]
    if int(sample["runner_count"]) != len(tier_a) or int(
        sample["race_count"]
    ) != len(race_ids):
        raise ValueError("tier_a_evaluation_population_mismatch")
    if int(provenance["race_count"]) != len(race_ids) or set(
        provenance["races"]
    ) != set(race_ids):
        raise ValueError("tier_a_provenance_population_mismatch")
    market = evaluation["market"]
    if market.get("provider") != "Sportsbet" or market.get("market") != "WIN":
        raise ValueError("tier_a_evaluation_not_sportsbet_win")

    placeholders = ",".join("?" for _ in race_ids)
    connection = connect_read_only(source_db)
    try:
        source_rows = [
            dict(row)
            for row in connection.execute(
                f"SELECT * FROM live_odds WHERE source = 'sportsbet' "
                f"AND market_type = 'win' AND race_id IN ({placeholders})",
                race_ids,
            )
        ]
    finally:
        connection.close()
    indexed: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        indexed[
            (
                str(row["race_id"]),
                int(row["box_number"]),
                str(row["capture_timestamp"]),
            )
        ].append(row)

    audited: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for runner in tier_a:
        key = (
            str(runner["race_id"]),
            int(runner["box_number"]),
            str(runner["odds_capture_timestamp"]),
        )
        if key in seen:
            raise ValueError(f"tier_a_duplicate_runner_key:{key}")
        seen.add(key)
        candidates = indexed.get(key, [])
        if len(candidates) != 1:
            raise ValueError(f"tier_a_source_row_cardinality:{key}:{len(candidates)}")
        source = candidates[0]
        if normalize_name(source["dog_name"]) != normalize_name(runner["dog_name"]):
            raise ValueError(f"tier_a_runner_name_conflict:{key}")
        if not math.isclose(
            float(source["odds_decimal"]),
            float(runner["strict_win_odds"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"tier_a_stored_price_conflict:{key}")
        evidence = classify_win_evidence(
            raw_text=source.get("sportsbet_raw_runner_text"),
            expected_box=int(source["box_number"]),
            stored_odds=float(source["odds_decimal"]),
        )
        audited.append(
            {
                "race_id": key[0],
                "box_number": key[1],
                "odds_capture_timestamp": key[2],
                "source_row_id": int(source["id"]),
                **asdict(evidence),
            }
        )
    return audited, {
        "evaluation_disposition": evaluation["disposition"],
        "production_decision": evaluation["production_decision"],
        "model_sha256": evaluation["model"]["sha256"],
        "race_ids_sha256": sample["race_ids_sha256"],
    }


def build_dependency_audit(
    *,
    sidecar: list[dict[str, Any]],
    source_db: Path,
    input_hashes: Mapping[str, str],
    experiments_root: Path,
    favourite_report: Path,
    legacy_matrix_root: Path,
    legacy_model_report: Path,
    historical_tier_a_runners: Path,
    historical_tier_a_provenance: Path,
    historical_tier_a_manifest: Path,
    historical_tier_a_evaluation: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_matrix_index = {int(row["matrix_row_index"]): row for row in sidecar}
    by_key = {
        (
            str(row["race_id"]),
            int(row["box_number"]),
            str(row["capture_timestamp"]),
        ): row
        for row in sidecar
    }
    dependencies: list[dict[str, Any]] = []

    legacy_matrix_path = legacy_matrix_root / "training_matrix.jsonl"
    require_hash_references(
        [legacy_model_report],
        {"legacy_matrix_jsonl": sha256_file(legacy_matrix_path)},
    )
    legacy_rows = bind_legacy_training_matrix(legacy_matrix_path, sidecar)
    dependencies.append(
        _dependency_summary(
            experiment="canonical training model experiment 20260812",
            classification="RERUN_REQUIRED",
            rows=legacy_rows,
            artifacts=[legacy_matrix_path, legacy_model_report],
            binding=(
                "report binds the exact matrix hash; every race_id, box, capture timestamp, "
                "and normalized stored-price probability matches this contaminated surface"
            ),
            reason=(
                "market_implied_probability was a model feature and benchmark, so its model "
                "comparisons and calibration metrics require an unchanged-protocol rerun"
            ),
        )
    )

    tier_a_rows, tier_a_detail = audit_historical_tier_a(
        runners_path=historical_tier_a_runners,
        provenance_path=historical_tier_a_provenance,
        manifest_path=historical_tier_a_manifest,
        evaluation_path=historical_tier_a_evaluation,
        source_db=source_db,
    )
    tier_a_dependency = _dependency_summary(
        experiment="historical Tier-A frozen-RF vs WIN-market evaluation 20260716",
        classification="RERUN_REQUIRED",
        rows=tier_a_rows,
        artifacts=[
            historical_tier_a_runners,
            historical_tier_a_provenance,
            historical_tier_a_manifest,
            historical_tier_a_evaluation,
        ],
        binding=(
            "Tier-A exact race_id, box, capture timestamp, runner, and stored price all bind "
            "uniquely to immutable live_odds rows"
        ),
        reason=(
            "the model-only predictions remain frozen evidence, but the normalized WIN baseline, "
            "model-vs-market deltas, and KEEP_BASELINE decision consume contaminated prices"
        ),
    )
    tier_a_dependency["claim_impact"] = {
        "model_only_metrics": "VALID_UNAFFECTED",
        "market_and_model_vs_market_metrics": "RERUN_REQUIRED",
        "production_decision": "RERUN_REQUIRED",
        **tier_a_detail,
    }
    dependencies.append(tier_a_dependency)

    model_root = (
        experiments_root
        / "fresh_model_protocol_v1_model_frozen_20260814_v2_report_only"
    )
    require_hash_references(
        [model_root / "report.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    dependencies.append(
        _dependency_summary(
            experiment="fresh model protocol v1 frozen selection / market baseline",
            classification="RERUN_REQUIRED",
            rows=sidecar,
            artifacts=[model_root / "report.json", model_root / "selected_model.pkl"],
            binding="model report binds exact contaminated matrix SHA-256 and source DB SHA-256",
            reason="selected market baseline and comparison metrics consume corrected market probabilities",
        )
    )

    protocol_v2_root = experiments_root / "fresh_model_protocol_v2_report_only"
    require_hash_references(
        [protocol_v2_root / "protocol_v2.json", protocol_v2_root / "report.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    dependencies.append(
        _dependency_summary(
            experiment="fresh model protocol v2 frozen selection",
            classification="RERUN_REQUIRED",
            rows=sidecar,
            artifacts=[
                protocol_v2_root / "protocol_v2.json",
                protocol_v2_root / "report.json",
            ],
            binding="v2 report reproduces the same contaminated market-baseline selection metrics",
            reason="the frozen candidate comparison inherits the same base probability surface",
        )
    )

    mapping_path = (
        experiments_root
        / "thedogs_development_matrix_identity_backfill_20260814_report_only"
        / "accepted_identity_mapping.jsonl"
    )
    mapped_indices: list[int] = []
    for race in read_jsonl(mapping_path):
        for runner in race["runners"]:
            index = int(runner["matrix_row_index"])
            if by_matrix_index[index]["matrix_row_sha256"] != str(
                runner["matrix_row_sha256"]
            ):
                raise ValueError(f"target_mapping_matrix_row_hash_mismatch:{index}")
            mapped_indices.append(index)
    mapped_rows = [by_matrix_index[index] for index in mapped_indices]

    history_root = (
        experiments_root / "sportsbet_history_experiment_20260815_v2_report_only"
    )
    history_initial_root = (
        experiments_root / "sportsbet_history_experiment_20260815_report_only"
    )
    require_hash_references(
        [history_initial_root / "protocol.json", history_initial_root / "report.json"],
        {"matrix_jsonl": input_hashes["matrix_jsonl"]},
    )
    dependencies.append(
        _dependency_summary(
            experiment="basic Sportsbet history initial run",
            classification="INVALIDATED",
            rows=mapped_rows,
            artifacts=[
                history_initial_root / "protocol.json",
                history_initial_root / "report.json",
                mapping_path,
            ],
            binding="initial protocol binds the exact matrix hash and accepted matrix-row mapping",
            reason=(
                "the run is already superseded by its source-snapshot correction and also "
                "consumes contaminated market probabilities"
            ),
        )
    )
    require_hash_references(
        [history_root / "protocol.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    dependencies.append(
        _dependency_summary(
            experiment="basic Sportsbet history v2",
            classification="RERUN_REQUIRED",
            rows=mapped_rows,
            artifacts=[
                history_root / "protocol.json",
                history_root / "report.json",
                mapping_path,
            ],
            binding="accepted mapping supplies exact matrix_row_index values; protocol binds matrix hash",
            reason="NO_INCREMENTAL_SIGNAL was measured against contaminated market probabilities",
        )
    )

    speed_root = (
        experiments_root
        / "sportsbet_speed_context_experiment_20260815_clean_rerun_report_only"
    )
    speed_initial_root = (
        experiments_root / "sportsbet_speed_context_experiment_20260815_report_only"
    )
    require_hash_references(
        [speed_initial_root / "protocol.json", speed_initial_root / "report.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    dependencies.append(
        _dependency_summary(
            experiment="speed/context initial run",
            classification="INVALIDATED",
            rows=mapped_rows,
            artifacts=[
                speed_initial_root / "protocol.json",
                speed_initial_root / "report.json",
            ],
            binding="initial protocol binds the exact matrix/source hashes and accepted rows",
            reason=(
                "the clean rerun already invalidated this bundle for feature leakage, and its "
                "market baseline is independently contaminated"
            ),
        )
    )
    require_hash_references(
        [speed_root / "protocol.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    dependencies.append(
        _dependency_summary(
            experiment="clean speed/context rerun",
            classification="RERUN_REQUIRED",
            rows=mapped_rows,
            artifacts=[speed_root / "protocol.json", speed_root / "report.json"],
            binding="protocol binds the same matrix hash and exact target-mapped matrix rows",
            reason="clean feature provenance does not repair the contaminated market baseline",
        )
    )

    calibration_root = (
        experiments_root / "sportsbet_calibration_experiment_20260815_report_only"
    )
    require_hash_references(
        [calibration_root / "protocol.json"],
        {"matrix_jsonl": input_hashes["matrix_jsonl"]},
    )
    dependencies.append(
        _dependency_summary(
            experiment="Sportsbet calibration",
            classification="INVALIDATED",
            rows=mapped_rows,
            artifacts=[
                calibration_root / "protocol.json",
                calibration_root / "report.json",
            ],
            binding="protocol directly declares sealed matrix market_implied_probability as its input",
            reason="calibration transforms and metrics directly interpret contaminated normalized probabilities",
        )
    )

    raw_root = experiments_root / "raw_race_shape_experiment_20260815_report_only"
    require_hash_references(
        [raw_root / "protocol.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    raw_matrix_path = raw_root / "feature_matrix.jsonl"
    raw_rows = [by_key[_row_key(row)] for row in read_jsonl(raw_matrix_path)]
    dependencies.append(
        _dependency_summary(
            experiment="raw race-shape Sportsbet baseline",
            classification="RERUN_REQUIRED",
            rows=raw_rows,
            artifacts=[
                raw_root / "protocol.json",
                raw_root / "report.json",
                raw_matrix_path,
            ],
            binding="feature matrix retains exact race_id, box_number, and odds_capture_timestamp",
            reason="candidate comparison uses contaminated matrix market probabilities as baseline/input",
        )
    )

    favourite_rows = sidecar
    require_hash_references(
        [favourite_report],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    dependencies.append(
        _dependency_summary(
            experiment="favourite benchmark",
            classification="VALID_UNAFFECTED",
            rows=favourite_rows,
            artifacts=[favourite_report],
            binding="benchmark report binds exact matrix/source hashes and all selected development rows",
            reason="its only verdict is an evidence block; it emitted no economic or calibration result",
        )
    )

    movement_root = (
        experiments_root / "prejump_market_structure_experiment_20260815_report_only"
    )
    require_hash_references(
        [movement_root / "protocol.json", movement_root / "report.json"],
        {
            "matrix_jsonl": input_hashes["matrix_jsonl"],
            "source_db": input_hashes["source_db"],
        },
    )
    movement_source = movement_root / "sportsbet_source_extract.jsonl"
    movement_extract = read_jsonl(movement_source)
    source_ids = [int(row["id"]) for row in movement_extract]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("movement_source_extract_duplicate_live_odds_id")
    placeholders = ",".join("?" for _ in source_ids)
    connection = connect_read_only(source_db)
    try:
        db_rows = {
            int(row["id"]): dict(row)
            for row in connection.execute(
                f"SELECT * FROM live_odds WHERE id IN ({placeholders})", source_ids
            )
        }
    finally:
        connection.close()
    if set(db_rows) != set(source_ids):
        raise ValueError("movement_source_row_id_missing_from_bound_db")
    movement_rows: list[dict[str, Any]] = []
    development_source_ids = {int(row["source_row_id"]) for row in sidecar}
    exact_development_overlap = 0
    for extract_row in movement_extract:
        row = db_rows[int(extract_row["id"])]
        evidence = classify_win_evidence(
            raw_text=row.get("sportsbet_raw_runner_text"),
            expected_box=int(row["box_number"]),
            stored_odds=float(row["odds_decimal"]),
        )
        movement_rows.append({**extract_row, **asdict(evidence)})
        exact_development_overlap += int(int(row["id"]) in development_source_ids)
    movement_dependency = _dependency_summary(
        experiment="fixed-window T-30/T-10 market structure",
        classification="RERUN_REQUIRED",
        rows=movement_rows,
        artifacts=[
            movement_root / "protocol.json",
            movement_root / "report.json",
            movement_source,
        ],
        binding="source extract retains exact live_odds IDs resolved against the immutable source DB",
        reason="independent row-ID audit proves its fixed-window rows also include mislabeled/unverified prices",
    )
    movement_dependency["exact_source_row_id_overlap_with_development_surface"] = (
        exact_development_overlap
    )
    dependencies.append(movement_dependency)

    fixed_window_detail = {
        "source_extract_sha256": sha256_file(movement_source),
        "source_rows": len(movement_rows),
        "source_races": len({str(row["race_id"]) for row in movement_rows}),
        "classification_counts": _classification_counts(movement_rows),
        "exact_source_row_id_overlap_with_development_surface": exact_development_overlap,
        "conclusion": (
            "independently affected; this is not inferred from the development matrix"
        ),
    }
    return dependencies, fixed_window_detail


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_dependency_csv(path: Path, dependencies: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "experiment",
        "classification",
        "exact_input_races",
        "exact_input_runner_rows",
        "affected_runner_rows",
        "unrecoverable_runner_rows",
        "exact_row_binding",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dependency in dependencies:
            writer.writerow({field: dependency.get(field) for field in fields})


def write_report_markdown(path: Path, report: Mapping[str, Any]) -> None:
    coverage = report["corrected_coverage"]
    probability = report["old_vs_corrected_probabilities"]
    lines = [
        "# Sportsbet WIN market provenance audit",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        "## Root cause",
        "",
        report["root_cause"]["summary"],
        "",
        "## Corrected coverage",
        "",
        f"- Development: {coverage['development_races']:,} races / {coverage['development_runner_rows']:,} runners.",
        f"- Fully verified without repair: {coverage['fully_verified_original_races']:,} races.",
        f"- Canonical complete-field WIN: {coverage['qualified_races']:,} races / {coverage['qualified_runner_rows']:,} runners.",
        f"- Complete only after deterministic repair: {coverage['repaired_qualified_races']:,} races.",
        f"- Excluded: {coverage['excluded_races']:,} races / {coverage['excluded_runner_rows']:,} runners.",
        f"- Repaired prices: {coverage['repaired_runner_rows']:,} runner rows.",
        f"- Sidecar SHA-256: `{report['canonical_artifacts']['canonical_win_sidecar_sha256']}`.",
        f"- Matrix SHA-256: `{report['canonical_artifacts']['canonical_win_matrix_sha256']}`.",
        "",
        "## Old versus corrected normalized probabilities",
        "",
        f"- Changed runner probabilities: {probability['probability_changed_runner_rows']:,}.",
        f"- Mean absolute delta: {probability['mean_absolute_probability_delta']:.12f}.",
        f"- Maximum absolute delta: {probability['maximum_absolute_probability_delta']:.12f}.",
        f"- Favourite identity changed: {probability['favourite_identity_changed_races']:,} races.",
        "",
        "## Experiment dependency audit",
        "",
        "| Experiment | Status | Exact races | Exact rows | Affected | Unrecoverable |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["experiment_dependency_audit"]:
        lines.append(
            f"| {item['experiment']} | {item['classification']} | "
            f"{item['exact_input_races']:,} | {item['exact_input_runner_rows']:,} | "
            f"{item['affected_runner_rows']:,} | {item['unrecoverable_runner_rows']:,} |"
        )
    for heading in ("BLOCKING", "IMPORTANT", "OPTIONAL"):
        lines.extend(["", f"## {heading}", ""])
        lines.extend(f"- {finding}" for finding in report["findings"][heading])
    lines.extend(["", "## Exact minimal rerun list", ""])
    lines.extend(
        f"{index}. {item}" for index, item in enumerate(report["minimal_rerun_list"], 1)
    )
    lines.extend(["", "## Claims", "", "Supported:"])
    lines.extend(f"- {item}" for item in report["claims"]["supported"])
    lines.extend(["", "Unsupported:"])
    lines.extend(f"- {item}" for item in report["claims"]["unsupported"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(
    *,
    repo_root: Path,
    matrix_root: Path,
    model_report: Path,
    source_db: Path,
    experiments_root: Path,
    favourite_report: Path,
    legacy_matrix_root: Path,
    legacy_model_report: Path,
    historical_tier_a_runners: Path,
    historical_tier_a_provenance: Path,
    historical_tier_a_manifest: Path,
    historical_tier_a_evaluation: Path,
    output_dir: Path,
) -> dict[str, Any]:
    matrix_rows, input_hashes, matrix_report, frozen_model_report = validate_inputs(
        matrix_root, model_report, source_db
    )
    source_revision = str(frozen_model_report.get("repo_sha") or "")
    repair_revision = "314091604fc245185638cbf30be07ed7241301d9"
    lineage = verify_root_cause_lineage(repo_root, source_revision, repair_revision)
    if lineage["repair_is_ancestor_of_defective_revision"]:
        raise ValueError("paired_market_repair_present_in_declared_source_revision")
    hashes_before = dict(input_hashes)
    auxiliary_inputs = [
        legacy_matrix_root / "training_matrix.jsonl",
        legacy_model_report,
        historical_tier_a_runners,
        historical_tier_a_provenance,
        historical_tier_a_manifest,
        historical_tier_a_evaluation,
    ]
    auxiliary_hashes_before = _artifact_hashes(auxiliary_inputs)
    sidecar, canonical_matrix, probability_report, integrity = build_development_surface(
        matrix_rows, source_db, input_hashes["source_db"]
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    sidecar_path = output_dir / "canonical_win_sidecar.jsonl"
    matrix_path = output_dir / "canonical_win_matrix.jsonl"
    dependency_path = output_dir / "experiment_dependency_audit.csv"
    report_path = output_dir / "report.json"
    markdown_path = output_dir / "REPORT.md"
    write_jsonl(sidecar_path, sidecar)
    write_jsonl(matrix_path, canonical_matrix)

    dependencies, fixed_window_detail = build_dependency_audit(
        sidecar=sidecar,
        source_db=source_db,
        input_hashes=input_hashes,
        experiments_root=experiments_root,
        favourite_report=favourite_report,
        legacy_matrix_root=legacy_matrix_root,
        legacy_model_report=legacy_model_report,
        historical_tier_a_runners=historical_tier_a_runners,
        historical_tier_a_provenance=historical_tier_a_provenance,
        historical_tier_a_manifest=historical_tier_a_manifest,
        historical_tier_a_evaluation=historical_tier_a_evaluation,
    )
    write_dependency_csv(dependency_path, dependencies)

    hashes_after = {
        "matrix_jsonl": sha256_file(matrix_root / "training_matrix.jsonl"),
        "matrix_csv": sha256_file(matrix_root / "training_matrix.csv"),
        "matrix_exclusions": sha256_file(matrix_root / "exclusions.jsonl"),
        "matrix_report": sha256_file(matrix_root / "report.json"),
        "model_report": sha256_file(model_report),
        "source_db": sha256_file(source_db),
    }
    if hashes_before != hashes_after:
        raise ValueError("immutable_input_hash_changed_during_audit")
    auxiliary_hashes_after = _artifact_hashes(auxiliary_inputs)
    if auxiliary_hashes_before != auxiliary_hashes_after:
        raise ValueError("immutable_auxiliary_input_hash_changed_during_audit")

    counts = _classification_counts(sidecar)
    qualified_races = {str(row["race_id"]) for row in sidecar if row["race_qualified"]}
    rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sidecar:
        rows_by_race[str(row["race_id"])].append(row)
    fully_verified_original_races = sum(
        all(row["classification"] == VERIFIED_WIN for row in race_rows)
        for race_rows in rows_by_race.values()
    )
    qualified_rows = sum(bool(row["race_qualified"]) for row in sidecar)
    development_races = len({str(row["race_id"]) for row in sidecar})
    verdict = (
        "WIN_MARKET_SURFACE_RECOVERED"
        if qualified_races and len(qualified_races) == development_races
        else "WIN_MARKET_PARTIAL_RECOVERY"
        if qualified_races
        else "WIN_MARKET_EVIDENCE_UNRECOVERABLE"
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "analysis_date": ANALYSIS_DATE,
        "verdict": verdict,
        "scope": {
            "report_only": True,
            "models_trained": False,
            "models_rescored": False,
            "august_or_forward_outcomes_opened": False,
            "source_db_open_mode": "read-only immutable query-only",
            "matrix_policy": matrix_report["odds_policy"],
            "upstream_model_status": frozen_model_report["status"],
        },
        "root_cause": {
            "source_snapshot_repo_sha": frozen_model_report.get("repo_sha"),
            "defective_source_revision": source_revision,
            "paired_market_repair_revision": repair_revision,
            **lineage,
            "summary": (
                "The source revision scraped a generic first matching price element into the WIN list, "
                "then separately interacted with PLACE, without binding the stored WIN row to the "
                "source-explicit WIN/PLACE columns retained in the same runner text. The paired-market "
                "repair existed on a non-ancestor branch, so the development snapshot was built from "
                "the unprotected lineage."
            ),
            "defect_boundary": (
                "generic DOM price selection plus market_type assignment before source-explicit paired-column validation"
            ),
        },
        "input_sha256_before": hashes_before,
        "input_sha256_after": hashes_after,
        "auxiliary_input_sha256_before": auxiliary_hashes_before,
        "auxiliary_input_sha256_after": auxiliary_hashes_after,
        "immutable_inputs_unchanged": True,
        "sqlite_integrity_check": integrity,
        "classification_policy": {
            VERIFIED_WIN: "stored value equals source-explicit paired WIN",
            PLACE_MISLABEL: "stored WIN-labelled value equals source-explicit paired PLACE; WIN reconstructed",
            RECOVERABLE_WIN: "stored value equals neither paired price; source-explicit WIN reconstructed",
            UNPARSABLE: "retained raw text cannot prove a paired WIN value",
            CONFLICTING: "retained raw identity or paired-column evidence conflicts",
            MISSING: "required retained raw/source row evidence is absent",
        },
        "corrected_coverage": {
            "development_races": development_races,
            "development_runner_rows": len(sidecar),
            "classification_counts": counts,
            "qualified_races": len(qualified_races),
            "qualified_runner_rows": qualified_rows,
            "fully_verified_original_races": fully_verified_original_races,
            "repaired_qualified_races": len(qualified_races)
            - fully_verified_original_races,
            "excluded_races": development_races - len(qualified_races),
            "excluded_runner_rows": len(sidecar) - qualified_rows,
            "repaired_runner_rows": counts[PLACE_MISLABEL] + counts[RECOVERABLE_WIN],
            "qualification_rule": (
                "every matrix runner row in the race must have VERIFIED_WIN, RECOVERABLE_WIN, or PLACE_MISLABEL evidence"
            ),
        },
        "old_vs_corrected_probabilities": probability_report,
        "fixed_window_independent_audit": fixed_window_detail,
        "experiment_dependency_audit": dependencies,
        "canonical_artifacts": {
            "canonical_win_sidecar": sidecar_path.name,
            "canonical_win_sidecar_sha256": sha256_file(sidecar_path),
            "canonical_win_matrix": matrix_path.name,
            "canonical_win_matrix_sha256": sha256_file(matrix_path),
            "dependency_audit": dependency_path.name,
            "dependency_audit_sha256": sha256_file(dependency_path),
        },
        "findings": {
            "BLOCKING": [
                "The original 1,182-race development surface is not canonical Sportsbet WIN evidence.",
                "Races containing UNPARSABLE, CONFLICTING, or MISSING rows remain excluded; no odds-magnitude or neighbouring-row fallback is permitted.",
                "All empirical model-selection, calibration, and fixed-window conclusions identified as RERUN_REQUIRED or INVALIDATED lack current authority until separately authorized reruns.",
            ],
            "IMPORTANT": [
                "The original matrix, source snapshot, experiment bundles, labels, frozen models, and August cohort were unchanged.",
                "PLACE_MISLABEL rows are recoverable because the same immutable raw runner text explicitly proves both columns; PLACE was never treated as WIN evidence.",
                "The fixed-window surface was audited independently by exact live_odds IDs and is affected on its own evidence, not by assumption from the development matrix.",
                "The older Tier-A frozen-RF model-only outputs remain intact, but its WIN-market comparison and KEEP_BASELINE decision require rerun.",
            ],
            "OPTIONAL": [
                "A later authorized rerun can consume only the canonical complete-field matrix and must declare its reduced race population and new hashes.",
            ],
        },
        "claims": {
            "supported": [
                "The retained raw Sportsbet evidence deterministically reconstructs a partial canonical WIN development surface.",
                "The dependency table proves exact row/hash intersections for the named prior experiments.",
                "The favourite benchmark's blocking verdict remains valid because it emitted no economic result.",
                "The 20260812 training experiment is exactly row-and-probability bound to this contaminated surface.",
                "The historical Tier-A frozen-RF evaluation is exactly source-row bound; only its model-only outputs remain unaffected.",
            ],
            "unsupported": [
                "Any prior calibration, ROI, P&L, EV, drawdown, losing-sequence, market-vs-model, or promotion claim derived from the contaminated values.",
                "Any claim that excluded races have reconstructed WIN prices.",
                "Any August, forward, deployment, promotion, or betting conclusion.",
            ],
        },
        "minimal_rerun_list": [
            "Use the new canonical complete-field WIN matrix as the sole shared market input; do not reuse the contaminated probability column.",
            "Rerun the 20260812 canonical training comparison and the frozen fresh-model comparison on their exact canonical intersections; do not tune protocols.",
            "Rerun the historical Tier-A frozen-RF market baseline and model-vs-market comparison from its exact rows; preserve the frozen model predictions and do not retrain.",
            "Rerun basic history, clean speed/context, and raw race-shape comparisons on their exact canonical intersections; do not tune protocols.",
            "Rerun calibration from raw canonical WIN probabilities and invalidate the existing fitted calibration artifacts.",
            "Rebuild the T-30/T-10 movement matrix from its independently audited exact source-row IDs, excluding incomplete paired-WIN races before any refit.",
            "Run the favourite economic benchmark only if a revised, explicit scope permits the canonical reduced population; the original full-population benchmark remains evidence-blocked.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report_markdown(markdown_path, report)

    checksum_paths = [sidecar_path, matrix_path, dependency_path, report_path, markdown_path]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--model-report", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--experiments-root", type=Path, required=True)
    parser.add_argument("--favourite-report", type=Path, required=True)
    parser.add_argument("--legacy-matrix-root", type=Path, required=True)
    parser.add_argument("--legacy-model-report", type=Path, required=True)
    parser.add_argument("--historical-tier-a-runners", type=Path, required=True)
    parser.add_argument("--historical-tier-a-provenance", type=Path, required=True)
    parser.add_argument("--historical-tier-a-manifest", type=Path, required=True)
    parser.add_argument("--historical-tier-a-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        repo_root=args.repo_root.resolve(),
        matrix_root=args.matrix_root.resolve(),
        model_report=args.model_report.resolve(),
        source_db=args.source_db.resolve(),
        experiments_root=args.experiments_root.resolve(),
        favourite_report=args.favourite_report.resolve(),
        legacy_matrix_root=args.legacy_matrix_root.resolve(),
        legacy_model_report=args.legacy_model_report.resolve(),
        historical_tier_a_runners=args.historical_tier_a_runners.resolve(),
        historical_tier_a_provenance=args.historical_tier_a_provenance.resolve(),
        historical_tier_a_manifest=args.historical_tier_a_manifest.resolve(),
        historical_tier_a_evaluation=args.historical_tier_a_evaluation.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"verdict": report["verdict"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
