#!/usr/bin/env python3
"""Build a deterministic, report-only Sportsbet favourite benchmark.

The legacy path audits its immutable source database read-only.  The corrected
path consumes only the sealed canonical WIN surface for prices and uses the
sealed upstream matrix solely to verify identity, timestamps, and labels.
Neither path opens a held-out August/forward cohort.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


SCHEMA_VERSION = "sportsbet_favourite_benchmark_v1"
CORRECTED_SCHEMA_VERSION = "sportsbet_favourite_benchmark_canonical_win_v1"
ANALYSIS_DATE = "2026-08-15"
DEVELOPMENT_END = "2026-08-02"
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_REPLICATES = 20_000
EXPECTED_CANONICAL_MATRIX_SHA256 = (
    "eb1783d4cc07e6980463a097c97fdac9f5370b08f493ca15addf768aa0b014b6"
)
EXPECTED_CANONICAL_SIDECAR_SHA256 = (
    "880ae93680e56991fa2c9eb316732cbc71bc7ff713525efcf83750ceace4493d"
)
EXPECTED_CANONICAL_RACES = 1_153
EXPECTED_CANONICAL_RUNNERS = 8_234
EXPECTED_DEVELOPMENT_RACES = 1_182
EXPECTED_DEVELOPMENT_RUNNERS = 8_350
CANONICAL_WIN_CLASSES = frozenset(
    {"VERIFIED_WIN", "RECOVERABLE_WIN", "PLACE_MISLABEL"}
)
ODDS_BANDS = (
    ("lt_2_00", None, 2.0, "<2.00"),
    ("2_00_to_2_99", 2.0, 3.0, "2.00-2.99"),
    ("3_00_to_4_99", 3.0, 5.0, "3.00-4.99"),
    ("gte_5_00", 5.0, None, ">=5.00"),
)
SPORTSBET_DECIMAL_PRICE_RE = re.compile(r"^\d+(?:\.\d{1,2})$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def read_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid_sha256_manifest_line:{line_number}")
        name = parts[1].lstrip("*")
        if name in entries:
            raise ValueError(f"duplicate_sha256_manifest_entry:{name}")
        entries[name] = parts[0]
    return entries


def read_original_matrix(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"original_matrix_json_invalid_line:{line_number}") from exc
            row["_matrix_row_sha256"] = hashlib.sha256(stripped + b"\n").hexdigest()
            rows.append(row)
    return rows


def parse_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timezone_missing:{text}")
    return parsed


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def paired_fixed_prices(raw_text: Any) -> tuple[float, float] | None:
    """Recover the source-explicit WIN and PLACE pair before the EW control."""

    lines = [line.strip() for line in str(raw_text or "").splitlines()]
    try:
        ew_index = next(index for index, line in enumerate(lines) if line.upper() == "EW")
    except StopIteration:
        return None
    prices = [
        float(line)
        for line in lines[:ew_index]
        if SPORTSBET_DECIMAL_PRICE_RE.fullmatch(line) and float(line) > 1.0
    ]
    if len(prices) < 2:
        return None
    win_price, place_price = prices[-2:]
    if place_price > win_price:
        return None
    return win_price, place_price


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_intervals(
    selections: Sequence[Mapping[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)
    count = len(selections)
    net = [float(row["net_pnl_units"]) for row in selections]
    won = [int(row["won"]) for row in selections]
    implied = [float(row["normalized_market_probability"]) for row in selections]
    roi_samples: list[float] = []
    win_rate_samples: list[float] = []
    calibration_gap_samples: list[float] = []
    for _ in range(replicates):
        net_sum = 0.0
        win_sum = 0
        implied_sum = 0.0
        for _ in range(count):
            index = rng.randrange(count)
            net_sum += net[index]
            win_sum += won[index]
            implied_sum += implied[index]
        roi_samples.append(net_sum / count)
        win_rate = win_sum / count
        win_rate_samples.append(win_rate)
        calibration_gap_samples.append(win_rate - implied_sum / count)
    return {
        "method": "race-level nonparametric percentile bootstrap",
        "confidence_level": 0.95,
        "replicates": replicates,
        "seed": seed,
        "roi": {
            "lower": percentile(roi_samples, 0.025),
            "upper": percentile(roi_samples, 0.975),
        },
        "win_rate": {
            "lower": percentile(win_rate_samples, 0.025),
            "upper": percentile(win_rate_samples, 0.975),
        },
        "observed_minus_normalized_implied_win_rate": {
            "lower": percentile(calibration_gap_samples, 0.025),
            "upper": percentile(calibration_gap_samples, 0.975),
        },
    }


def odds_band(odds: float) -> tuple[str, str]:
    for key, lower, upper, label in ODDS_BANDS:
        if (lower is None or odds >= lower) and (upper is None or odds < upper):
            return key, label
    raise AssertionError(f"odds_outside_bands:{odds}")


def summarise_bets(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    bets = len(rows)
    staked = float(bets)
    returned = sum(float(row["returned_units"]) for row in rows)
    wins = sum(int(row["won"]) for row in rows)
    odds = [float(row["decimal_odds"]) for row in rows]
    return {
        "races": bets,
        "bets": bets,
        "wins": wins,
        "win_rate": wins / bets if bets else None,
        "mean_odds": mean(odds) if odds else None,
        "average_odds": mean(odds) if odds else None,
        "median_odds": median(odds) if odds else None,
        "total_staked_units": staked,
        "total_returned_units": returned,
        "net_pnl_units": returned - staked,
        "flat_stake_roi": (returned - staked) / staked if bets else None,
        "average_return_per_bet_units": returned / bets if bets else None,
        "average_net_per_bet_units": (returned - staked) / bets if bets else None,
    }


def drawdown_and_losing_sequence(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    equity = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    current_losses = 0
    longest_losses = 0
    for row in rows:
        equity += float(row["net_pnl_units"])
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
        if int(row["won"]):
            current_losses = 0
        else:
            current_losses += 1
            longest_losses = max(longest_losses, current_losses)
    return {
        "ordering": "jump_at ascending, then race_id ascending",
        "starting_equity_units": 0.0,
        "maximum_drawdown_units": maximum_drawdown,
        "longest_losing_sequence_bets": longest_losses,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_matrix(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"matrix_json_invalid_line:{line_number}") from exc
    return rows


def connect_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def load_selected_odds(
    conn: sqlite3.Connection, race_ids: set[str], jumps: Mapping[str, datetime]
) -> dict[str, dict[int, dict[str, Any]]]:
    selected: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for sqlite_row in conn.execute("SELECT * FROM live_odds ORDER BY id"):
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
        if not math.isfinite(odds) or odds <= 1.0 or captured >= jumps[race_id]:
            continue
        if "sportsbet.com.au" not in str(row.get("source_url") or "").casefold():
            continue
        existing = selected[race_id].get(box)
        if existing is None or parse_datetime(existing["capture_timestamp"]) < captured:
            selected[race_id][box] = row
    return selected


def load_official_runners(
    conn: sqlite3.Connection, race_ids: set[str]
) -> tuple[dict[str, dict[int, dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sqlite_row in conn.execute(
        """
        SELECT * FROM autonomous_official_result_evidence_runners
        ORDER BY race_id, box_number, captured_at, inserted_at, id
        """
    ):
        row = dict(sqlite_row)
        race_id = str(row.get("race_id") or "")
        if race_id in race_ids:
            grouped[race_id].append(row)
    selected: dict[str, dict[int, dict[str, Any]]] = {}
    conflicts: list[str] = []
    for race_id, rows in grouped.items():
        by_box: dict[int, dict[str, Any]] = {}
        for row in rows:
            box = int(row["box_number"])
            prior = by_box.get(box)
            signature = (
                normalize_name(row.get("dog_name")),
                int(row["finish_position"]),
                int(row["is_winner"]),
            )
            if prior is not None:
                prior_signature = (
                    normalize_name(prior.get("dog_name")),
                    int(prior["finish_position"]),
                    int(prior["is_winner"]),
                )
                if signature != prior_signature:
                    conflicts.append(f"{race_id}:box_{box}:official_runner_conflict")
                    continue
            by_box[box] = row
        selected[race_id] = by_box
    return selected, conflicts


def validate_inputs(
    matrix_root: Path, model_report_path: Path, source_db: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, str]]:
    matrix_report_path = matrix_root / "report.json"
    matrix_path = matrix_root / "training_matrix.jsonl"
    csv_path = matrix_root / "training_matrix.csv"
    exclusions_path = matrix_root / "exclusions.jsonl"
    matrix_report = read_json(matrix_report_path)
    model_report = read_json(model_report_path)
    input_paths = {
        "matrix_jsonl": matrix_path,
        "matrix_csv": csv_path,
        "matrix_exclusions": exclusions_path,
        "matrix_report": matrix_report_path,
        "model_report": model_report_path,
        "source_db": source_db,
    }
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    for filename in ("training_matrix.jsonl", "training_matrix.csv", "exclusions.jsonl"):
        expected = str(matrix_report["files"][filename]["sha256"])
        actual = hashes[
            {
                "training_matrix.jsonl": "matrix_jsonl",
                "training_matrix.csv": "matrix_csv",
                "exclusions.jsonl": "matrix_exclusions",
            }[filename]
        ]
        if actual != expected or model_report["matrix_files"][filename] != actual:
            raise ValueError(f"sealed_matrix_hash_mismatch:{filename}")
    if hashes["source_db"] != model_report["source_snapshot_sha256"]:
        raise ValueError("source_db_hash_mismatch")
    if matrix_report["leakage_status"] != "PASS":
        raise ValueError("matrix_leakage_status_not_pass")
    if matrix_report["label_policy"] != "append-only official result evidence only":
        raise ValueError("unexpected_label_policy")
    if matrix_report["odds_policy"] != "latest valid Sportsbet win odds per box strictly before jump":
        raise ValueError("unexpected_odds_policy")
    if model_report["august_boundary"]["outcomes_opened"] or model_report["august_boundary"]["scored"]:
        raise ValueError("august_boundary_not_untouched_in_model_report")
    rows = read_matrix(matrix_path)
    race_ids = {str(row["race_id"]) for row in rows}
    if len(rows) != int(matrix_report["included_runner_rows"]):
        raise ValueError("matrix_runner_count_mismatch")
    if len(race_ids) != int(matrix_report["included_races"]):
        raise ValueError("matrix_race_count_mismatch")
    if max(str(row["race_date"]) for row in rows) != DEVELOPMENT_END:
        raise ValueError("unexpected_development_end")
    return rows, matrix_report, model_report, hashes


def validate_corrected_inputs(
    canonical_root: Path, original_matrix_path: Path
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]], dict[str, Any], dict[str, str], dict[str, Any]]:
    matrix_path = canonical_root / "canonical_win_matrix.jsonl"
    sidecar_path = canonical_root / "canonical_win_sidecar.jsonl"
    audit_report_path = canonical_root / "report.json"
    manifest_path = canonical_root / "SHA256SUMS"
    original_matrix_report_path = original_matrix_path.with_name("report.json")
    input_paths = {
        "canonical_win_matrix": matrix_path,
        "canonical_win_sidecar": sidecar_path,
        "canonical_audit_report": audit_report_path,
        "canonical_sha256_manifest": manifest_path,
        "upstream_label_matrix": original_matrix_path,
        "upstream_label_report": original_matrix_report_path,
    }
    hashes = {name: sha256_file(path) for name, path in input_paths.items()}
    if hashes["canonical_win_matrix"] != EXPECTED_CANONICAL_MATRIX_SHA256:
        raise ValueError("canonical_win_matrix_hash_mismatch")
    if hashes["canonical_win_sidecar"] != EXPECTED_CANONICAL_SIDECAR_SHA256:
        raise ValueError("canonical_win_sidecar_hash_mismatch")

    manifest = read_sha256_manifest(manifest_path)
    manifest_targets = {
        "canonical_win_matrix.jsonl": hashes["canonical_win_matrix"],
        "canonical_win_sidecar.jsonl": hashes["canonical_win_sidecar"],
        "report.json": hashes["canonical_audit_report"],
    }
    for name, actual in manifest_targets.items():
        if manifest.get(name) != actual:
            raise ValueError(f"canonical_manifest_hash_mismatch:{name}")

    audit = read_json(audit_report_path)
    coverage = audit.get("corrected_coverage", {})
    artifacts = audit.get("canonical_artifacts", {})
    if audit.get("verdict") != "WIN_MARKET_PARTIAL_RECOVERY":
        raise ValueError("unexpected_canonical_audit_verdict")
    if audit.get("immutable_inputs_unchanged") is not True:
        raise ValueError("canonical_audit_inputs_not_immutable")
    if audit.get("scope", {}).get("august_or_forward_outcomes_opened") is not False:
        raise ValueError("canonical_audit_opened_august_or_forward")
    if artifacts.get("canonical_win_matrix_sha256") != hashes["canonical_win_matrix"]:
        raise ValueError("canonical_report_matrix_hash_mismatch")
    if artifacts.get("canonical_win_sidecar_sha256") != hashes["canonical_win_sidecar"]:
        raise ValueError("canonical_report_sidecar_hash_mismatch")
    expected_coverage = {
        "development_races": EXPECTED_DEVELOPMENT_RACES,
        "development_runner_rows": EXPECTED_DEVELOPMENT_RUNNERS,
        "qualified_races": EXPECTED_CANONICAL_RACES,
        "qualified_runner_rows": EXPECTED_CANONICAL_RUNNERS,
        "excluded_races": 29,
        "excluded_runner_rows": 116,
        "repaired_qualified_races": 862,
        "repaired_runner_rows": 1_313,
    }
    for name, expected in expected_coverage.items():
        if int(coverage.get(name, -1)) != expected:
            raise ValueError(f"canonical_report_coverage_mismatch:{name}")
    declared_original_hash = audit.get("input_sha256_before", {}).get("matrix_jsonl")
    if hashes["upstream_label_matrix"] != declared_original_hash:
        raise ValueError("upstream_label_matrix_hash_mismatch")
    declared_original_report_hash = audit.get("input_sha256_before", {}).get(
        "matrix_report"
    )
    if hashes["upstream_label_report"] != declared_original_report_hash:
        raise ValueError("upstream_label_report_hash_mismatch")
    original_matrix_report = read_json(original_matrix_report_path)
    if original_matrix_report.get("label_policy") != "append-only official result evidence only":
        raise ValueError("unexpected_upstream_label_policy")
    if original_matrix_report.get("leakage_status") != "PASS":
        raise ValueError("upstream_label_matrix_leakage_status_not_pass")
    if int(original_matrix_report.get("included_races", -1)) != EXPECTED_DEVELOPMENT_RACES:
        raise ValueError("upstream_label_report_race_population_mismatch")
    if int(original_matrix_report.get("included_runner_rows", -1)) != EXPECTED_DEVELOPMENT_RUNNERS:
        raise ValueError("upstream_label_report_runner_population_mismatch")
    if audit.get("input_sha256_before") != audit.get("input_sha256_after"):
        raise ValueError("canonical_primary_inputs_changed")

    rows = read_matrix(matrix_path)
    sidecar_rows = read_matrix(sidecar_path)
    original_rows = read_original_matrix(original_matrix_path)
    if len(rows) != EXPECTED_CANONICAL_RUNNERS:
        raise ValueError("canonical_runner_population_mismatch")
    if len(sidecar_rows) != EXPECTED_DEVELOPMENT_RUNNERS:
        raise ValueError("canonical_sidecar_population_mismatch")
    if len(original_rows) != EXPECTED_DEVELOPMENT_RUNNERS:
        raise ValueError("upstream_label_population_mismatch")

    sidecar_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    classification_counts: Counter[str] = Counter()
    repaired_races: set[str] = set()
    qualified_sidecar_rows = 0
    for sidecar in sidecar_rows:
        index = int(sidecar["matrix_row_index"])
        if index < 0 or index >= len(original_rows):
            raise ValueError(f"sidecar_matrix_index_out_of_range:{index}")
        stored_sidecar_hash = str(sidecar["sidecar_row_sha256"])
        unhashed_sidecar = dict(sidecar)
        del unhashed_sidecar["sidecar_row_sha256"]
        if hashlib.sha256(canonical_json_bytes(unhashed_sidecar)).hexdigest() != stored_sidecar_hash:
            raise ValueError(f"sidecar_row_hash_mismatch:{index}")
        original = original_rows[index]
        if original["_matrix_row_sha256"] != sidecar["matrix_row_sha256"]:
            raise ValueError(f"upstream_label_row_hash_mismatch:{index}")
        identity_fields = ("race_id", "race_date", "box_number", "dog_name")
        if any(str(original[field]) != str(sidecar[field]) for field in identity_fields):
            raise ValueError(f"sidecar_upstream_identity_mismatch:{index}")
        key = (
            str(sidecar["race_id"]),
            int(sidecar["box_number"]),
            str(sidecar["capture_timestamp"]),
        )
        if key in sidecar_by_key:
            raise ValueError(f"duplicate_sidecar_runner_key:{key}")
        sidecar_by_key[key] = sidecar
        classification = str(sidecar["classification"])
        classification_counts[classification] += 1
        if bool(sidecar["race_qualified"]):
            qualified_sidecar_rows += 1
            if classification not in CANONICAL_WIN_CLASSES:
                raise ValueError(f"noncanonical_qualified_sidecar_row:{key}")
            if classification != "VERIFIED_WIN":
                repaired_races.add(str(sidecar["race_id"]))

    declared_classes = {
        str(name): int(value)
        for name, value in coverage.get("classification_counts", {}).items()
    }
    if set(classification_counts) - set(declared_classes) or any(
        classification_counts[name] != expected
        for name, expected in declared_classes.items()
    ):
        raise ValueError("sidecar_classification_counts_mismatch")
    if qualified_sidecar_rows != EXPECTED_CANONICAL_RUNNERS:
        raise ValueError("qualified_sidecar_population_mismatch")
    if len(repaired_races) != int(coverage["repaired_qualified_races"]):
        raise ValueError("repaired_race_count_mismatch")

    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_matrix_keys: set[tuple[str, int, str]] = set()
    lead_seconds: list[float] = []
    label_fields = (
        "race_id",
        "race_date",
        "jump_at",
        "venue",
        "race_number",
        "field_size",
        "box_number",
        "dog_name",
        "label_finish_position",
        "label_is_winner",
        "odds_capture_timestamp",
    )
    for row in rows:
        key = (
            str(row["race_id"]),
            int(row["box_number"]),
            str(row["odds_capture_timestamp"]),
        )
        if key in seen_matrix_keys:
            raise ValueError(f"duplicate_canonical_runner_key:{key}")
        seen_matrix_keys.add(key)
        sidecar = sidecar_by_key.get(key)
        if sidecar is None or not bool(sidecar["race_qualified"]):
            raise ValueError(f"canonical_row_missing_qualified_sidecar:{key}")
        index = int(sidecar["matrix_row_index"])
        original = original_rows[index]
        if any(str(row[field]) != str(original[field]) for field in label_fields):
            raise ValueError(f"canonical_label_or_identity_mismatch:{key}")
        if str(row.get("schema_version")) != "canonical_training_matrix_sportsbet_win_recovered_v1":
            raise ValueError(f"unexpected_canonical_matrix_schema:{key}")
        if row["sportsbet_win_sidecar_row_sha256"] != sidecar["sidecar_row_sha256"]:
            raise ValueError(f"canonical_sidecar_binding_mismatch:{key}")
        if int(row["sportsbet_win_source_row_id"]) != int(sidecar["source_row_id"]):
            raise ValueError(f"canonical_source_row_mismatch:{key}")
        if row["sportsbet_win_raw_sha256"] != sidecar["raw_runner_text_sha256"]:
            raise ValueError(f"canonical_raw_hash_mismatch:{key}")
        if row["sportsbet_win_evidence_classification"] != sidecar["classification"]:
            raise ValueError(f"canonical_classification_mismatch:{key}")
        odds = float(row["canonical_sportsbet_win_odds"])
        if not math.isfinite(odds) or odds <= 1.0 or not math.isclose(
            odds, float(sidecar["canonical_win_odds"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"canonical_odds_mismatch:{key}")
        captured = parse_datetime(row["odds_capture_timestamp"])
        jump = parse_datetime(row["jump_at"])
        if captured >= jump:
            raise ValueError(f"canonical_odds_not_prejump:{key}")
        lead_seconds.append((jump - captured).total_seconds())
        by_race[str(row["race_id"])].append(row)

    if len(by_race) != EXPECTED_CANONICAL_RACES:
        raise ValueError("canonical_race_population_mismatch")
    qualified_keys = {
        key for key, sidecar in sidecar_by_key.items() if bool(sidecar["race_qualified"])
    }
    if seen_matrix_keys != qualified_keys:
        raise ValueError("canonical_complete_field_key_set_mismatch")

    ties = 0
    for race_id, race_rows in by_race.items():
        boxes = [int(row["box_number"]) for row in race_rows]
        field_sizes = {int(row["field_size"]) for row in race_rows}
        if len(field_sizes) != 1 or len(race_rows) != next(iter(field_sizes)):
            raise ValueError(f"canonical_incomplete_field:{race_id}")
        if len(set(boxes)) != len(boxes):
            raise ValueError(f"canonical_duplicate_box:{race_id}")
        if len({str(row["jump_at"]) for row in race_rows}) != 1:
            raise ValueError(f"canonical_jump_conflict:{race_id}")
        if sum(int(row["label_is_winner"]) for row in race_rows) != 1:
            raise ValueError(f"canonical_winner_count_not_one:{race_id}")
        for row in race_rows:
            if int(row["label_is_winner"]) != int(int(row["label_finish_position"]) == 1):
                raise ValueError(f"canonical_winner_label_conflict:{race_id}")
        raw_implied = [1.0 / float(row["canonical_sportsbet_win_odds"]) for row in race_rows]
        total_implied = sum(raw_implied)
        for row, implied in zip(race_rows, raw_implied):
            probability = float(row["market_implied_probability"])
            if not math.isclose(probability, implied / total_implied, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"canonical_probability_mismatch:{race_id}")
        shortest = min(float(row["canonical_sportsbet_win_odds"]) for row in race_rows)
        tied = sum(
            math.isclose(float(row["canonical_sportsbet_win_odds"]), shortest, rel_tol=0.0, abs_tol=1e-12)
            for row in race_rows
        )
        ties += int(tied > 1)

    dates = [str(row["race_date"]) for row in rows]
    if max(dates) > DEVELOPMENT_END:
        raise ValueError("canonical_population_crosses_development_boundary")
    validation = {
        "audit_verdict": audit["verdict"],
        "immutable_inputs_unchanged": True,
        "manifest_entries_verified": sorted(manifest_targets),
        "canonical_complete_field_races": len(by_race),
        "canonical_runner_rows": len(rows),
        "excluded_incomplete_or_unparseable_races": int(coverage["excluded_races"]),
        "excluded_runner_rows": int(coverage["excluded_runner_rows"]),
        "repaired_runner_rows": int(coverage["repaired_runner_rows"]),
        "repaired_races": len(repaired_races),
        "classification_counts_all_development_rows": dict(sorted(declared_classes.items())),
        "labels_verified_against_upstream_row_hashes": len(rows),
        "label_policy": original_matrix_report["label_policy"],
        "upstream_label_matrix_use": "identity, timestamp, and official-result label verification only; no stored odds or probabilities used in benchmark calculation",
        "canonical_win_matrix_sha256": hashes["canonical_win_matrix"],
        "canonical_win_sidecar_sha256": hashes["canonical_win_sidecar"],
        "races_with_exactly_one_winner": len(by_race),
        "all_capture_timestamps_timezone_aware_and_strictly_prejump": True,
        "minimum_odds_lead_seconds": min(lead_seconds),
        "median_odds_lead_seconds": median(lead_seconds),
        "maximum_odds_lead_seconds": max(lead_seconds),
        "date_min": min(dates),
        "date_max": max(dates),
        "races_with_tied_shortest_price": ties,
        "benchmark_additional_race_skips": 0,
        "august_or_forward_outcomes_opened": False,
    }
    return rows, sidecar_by_key, audit, hashes, validation


def build_corrected_benchmark(
    rows: Sequence[Mapping[str, Any]],
    sidecar_by_key: Mapping[tuple[str, int, str], Mapping[str, Any]],
    audit: Mapping[str, Any],
    validation: Mapping[str, Any],
    replicates: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_race[str(row["race_id"])].append(row)

    selections: list[dict[str, Any]] = []
    for race_id, race_rows in by_race.items():
        favourite = min(
            race_rows,
            key=lambda row: (
                float(row["canonical_sportsbet_win_odds"]),
                int(row["box_number"]),
                str(row.get("dog_name") or "").casefold(),
            ),
        )
        box = int(favourite["box_number"])
        capture_timestamp = str(favourite["odds_capture_timestamp"])
        key = (race_id, box, capture_timestamp)
        sidecar = sidecar_by_key[key]
        odds = float(favourite["canonical_sportsbet_win_odds"])
        won = int(favourite["label_is_winner"])
        returned = odds if won else 0.0
        shortest = min(
            float(row["canonical_sportsbet_win_odds"]) for row in race_rows
        )
        tied = [
            row
            for row in race_rows
            if math.isclose(
                float(row["canonical_sportsbet_win_odds"]),
                shortest,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ]
        highest_probability = max(
            float(row["market_implied_probability"]) for row in race_rows
        )
        if not math.isclose(
            float(favourite["market_implied_probability"]),
            highest_probability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"shortest_price_not_highest_probability:{race_id}")
        band_key, band_label = odds_band(odds)
        selections.append(
            {
                "race_id": race_id,
                "race_date": str(favourite["race_date"]),
                "jump_at": str(favourite["jump_at"]),
                "venue": str(favourite["venue"]),
                "race_number": int(favourite["race_number"]),
                "box_number": box,
                "dog_name": str(favourite["dog_name"]),
                "decimal_odds": odds,
                "normalized_market_probability": float(
                    favourite["market_implied_probability"]
                ),
                "raw_inverse_odds_probability": 1.0 / odds,
                "odds_capture_timestamp": capture_timestamp,
                "source_url": str(sidecar["source_url"]),
                "sportsbet_win_evidence_classification": str(
                    sidecar["classification"]
                ),
                "source_row_id": int(sidecar["source_row_id"]),
                "odds_band": band_key,
                "odds_band_label": band_label,
                "tied_shortest_odds_count": len(tied),
                "won": won,
                "finish_position": int(favourite["label_finish_position"]),
                "stake_units": 1.0,
                "returned_units": returned,
                "net_pnl_units": returned - 1.0,
            }
        )
    selections.sort(key=lambda row: (parse_datetime(row["jump_at"]), row["race_id"]))
    if len(selections) != EXPECTED_CANONICAL_RACES:
        raise ValueError("benchmark_selection_population_mismatch")

    aggregate = summarise_bets(selections)
    path_metrics = drawdown_and_losing_sequence(selections)
    uncertainty = bootstrap_intervals(selections, replicates, seed)
    roi_interval = uncertainty["roi"]
    if roi_interval["lower"] > 0.0:
        verdict = "FAVOURITE_BENCHMARK_PROFITABLE_IN_SAMPLE"
    elif roi_interval["upper"] < 0.0:
        verdict = "FAVOURITE_BENCHMARK_UNPROFITABLE_IN_SAMPLE"
    else:
        verdict = "FAVOURITE_BENCHMARK_INCONCLUSIVE"

    bands: dict[str, Any] = {}
    for band_key, _, _, label in ODDS_BANDS:
        summary = summarise_bets(
            [row for row in selections if row["odds_band"] == band_key]
        )
        summary["definition"] = label
        summary["interpretation"] = "descriptive_only_not_a_strategy_variant"
        bands[band_key] = summary
    if sum(int(item["bets"]) for item in bands.values()) != len(selections):
        raise ValueError("odds_band_population_mismatch")

    observed_win_rate = float(aggregate["win_rate"])
    normalized_implied = mean(
        float(row["normalized_market_probability"]) for row in selections
    )
    raw_implied = mean(float(row["raw_inverse_odds_probability"]) for row in selections)
    calibration = {
        "observed_favourite_wins": int(aggregate["wins"]),
        "observed_favourite_win_rate": observed_win_rate,
        "aggregate_expected_wins_from_normalized_market": sum(
            float(row["normalized_market_probability"]) for row in selections
        ),
        "mean_normalized_market_probability": normalized_implied,
        "observed_minus_normalized_implied_win_rate": observed_win_rate
        - normalized_implied,
        "mean_raw_inverse_odds_probability": raw_implied,
        "observed_minus_raw_inverse_odds_win_rate": observed_win_rate
        - raw_implied,
        "bootstrap_interval_for_observed_minus_normalized_implied_win_rate": uncertainty[
            "observed_minus_normalized_implied_win_rate"
        ],
        "note": "Normalized comparison is descriptive favourite calibration; raw inverse odds retain bookmaker overround and connect directly to decimal-price returns.",
    }

    coverage = audit["corrected_coverage"]
    report = {
        "schema_version": CORRECTED_SCHEMA_VERSION,
        "analysis_date": ANALYSIS_DATE,
        "verdict": verdict,
        "scope": {
            "population": "every race in the sealed canonical complete-field Sportsbet WIN development surface",
            "date_min": validation["date_min"],
            "date_max": validation["date_max"],
            "development_races_before_market_evidence_exclusions": int(
                coverage["development_races"]
            ),
            "development_runner_rows_before_market_evidence_exclusions": int(
                coverage["development_runner_rows"]
            ),
            "eligible_races": len(selections),
            "eligible_runner_rows": len(rows),
            "excluded_incomplete_or_unparseable_races": int(coverage["excluded_races"]),
            "excluded_runner_rows": int(coverage["excluded_runner_rows"]),
            "benchmark_additional_race_skips": 0,
            "august_or_forward_cohort_opened": False,
            "models_modified_selected_or_rescored": False,
            "database_opened": False,
            "report_only": True,
        },
        "predeclared_rule": {
            "selections_per_eligible_race": 1,
            "selection": "shortest verified canonical Sportsbet WIN decimal price, equivalent to highest race-normalized inverse-odds probability",
            "stake_units": 1.0,
            "race_skipping": "none after canonical complete-field eligibility",
            "tie_break": "lowest numeric box, then case-folded dog name",
            "return": "decimal odds units when the selected runner wins, otherwise zero",
            "odds_bands": [label for _, _, _, label in ODDS_BANDS],
            "odds_band_use": "descriptive only; no post-hoc band selection or tuning",
        },
        "verdict_rule": {
            "blocked": "any canonical market, label, timestamp, complete-field, population, or input-hash validation failure",
            "profitable": "95% race-bootstrap ROI interval wholly above zero",
            "unprofitable": "95% race-bootstrap ROI interval wholly below zero",
            "inconclusive": "95% race-bootstrap ROI interval includes zero",
        },
        "evidence_validation": dict(validation),
        "aggregate_metrics": aggregate,
        "path_metrics": path_metrics,
        "uncertainty": uncertainty,
        "market_calibration_comparison": calibration,
        "odds_band_diagnostics": bands,
        "findings": {
            "BLOCKING": [],
            "IMPORTANT": [
                "This is an in-sample diagnostic on a consumed development population, not untouched prospective evidence.",
                "The 29 incomplete or unparseable races were excluded by the sealed canonical surface before this benchmark; no further race was skipped.",
                "The economic calculation uses only corrected canonical Sportsbet WIN prices and fixed one-unit stakes.",
                "The result does not establish a verified betting edge or authorize betting, model selection, promotion, filters, or staking changes.",
            ],
            "OPTIONAL": [
                "Retain this frozen rule and result only as a development economic diagnostic; do not tune it or its descriptive bands on these outcomes."
            ],
        },
        "claims": {
            "strongest_supported": "The frozen one-unit favourite rule had the reported in-sample economic performance on exactly the sealed canonical complete-field Sportsbet WIN development population.",
            "unsupported": [
                "The rule has a verified betting edge or is a betting recommendation.",
                "Any reported odds band is a selected profitable strategy.",
                "The result generalizes to consumed August evidence, untouched forward evidence, live execution, or future races.",
                "The benchmark changes model selection, promotion readiness, staking, or production policy.",
            ],
        },
        "main_modelling_programme_impact": "NONE. This development-only economic diagnostic does not select, train, tune, rescore, promote, deploy, or authorize betting.",
    }
    return selections, report


def build_benchmark(
    rows: list[dict[str, Any]], source_db: Path, replicates: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    jumps: dict[str, datetime] = {}
    for row in rows:
        race_id = str(row["race_id"])
        by_race[race_id].append(row)
        jump = parse_datetime(row["jump_at"])
        if race_id in jumps and jumps[race_id] != jump:
            raise ValueError(f"jump_timestamp_conflict:{race_id}")
        jumps[race_id] = jump
    race_ids = set(by_race)
    conn = connect_read_only(source_db)
    try:
        selected_odds = load_selected_odds(conn, race_ids, jumps)
        official, official_conflicts = load_official_runners(conn, race_ids)
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    if integrity != "ok":
        raise ValueError(f"sqlite_integrity_check_failed:{integrity}")
    if official_conflicts:
        raise ValueError(f"official_result_conflicts:{official_conflicts[:3]}")

    selections: list[dict[str, Any]] = []
    capture_modes: Counter[str] = Counter()
    odds_levels: Counter[str] = Counter()
    box_sources: Counter[str] = Counter()
    source_domains: Counter[str] = Counter()
    official_sources: Counter[str] = Counter()
    official_domains: Counter[str] = Counter()
    capture_spreads: list[float] = []
    lead_seconds: list[float] = []
    paired_price_status: Counter[str] = Counter()
    paired_price_status_by_race: dict[str, Counter[str]] = defaultdict(Counter)
    market_evidence_failures: list[dict[str, Any]] = []
    for race_id, race_rows in by_race.items():
        boxes = {int(row["box_number"]) for row in race_rows}
        if len(boxes) != len(race_rows):
            raise ValueError(f"duplicate_matrix_box:{race_id}")
        if set(selected_odds.get(race_id, {})) != boxes:
            raise ValueError(f"strict_odds_runner_set_changed:{race_id}")
        if set(official.get(race_id, {})) != boxes:
            raise ValueError(f"official_runner_set_changed:{race_id}")
        if sum(int(row["label_is_winner"]) for row in race_rows) != 1:
            raise ValueError(f"matrix_winner_count_not_one:{race_id}")
        raw_implied: dict[int, float] = {}
        captures: list[datetime] = []
        for row in race_rows:
            box = int(row["box_number"])
            odds_row = selected_odds[race_id][box]
            odds = float(odds_row["odds_decimal"])
            captured = parse_datetime(odds_row["capture_timestamp"])
            paired = paired_fixed_prices(odds_row.get("sportsbet_raw_runner_text"))
            if paired is None:
                paired_status = "raw_pair_unparseable"
                parsed_win = None
                parsed_place = None
            else:
                parsed_win, parsed_place = paired
                if math.isclose(odds, parsed_win, rel_tol=0.0, abs_tol=1e-12):
                    paired_status = "stored_matches_paired_win"
                elif math.isclose(odds, parsed_place, rel_tol=0.0, abs_tol=1e-12):
                    paired_status = "stored_matches_paired_place"
                else:
                    paired_status = "stored_matches_neither_paired_price"
            paired_price_status[paired_status] += 1
            paired_price_status_by_race[race_id][paired_status] += 1
            if paired_status != "stored_matches_paired_win":
                market_evidence_failures.append(
                    {
                        "race_id": race_id,
                        "race_date": str(row["race_date"]),
                        "box_number": box,
                        "dog_name": str(row["dog_name"]),
                        "capture_timestamp": str(odds_row["capture_timestamp"]),
                        "stored_market_type": str(odds_row["market_type"]),
                        "stored_odds_decimal": odds,
                        "parsed_paired_win_odds": parsed_win,
                        "parsed_paired_place_odds": parsed_place,
                        "validation_status": paired_status,
                        "source_url": str(odds_row["source_url"]),
                    }
                )
            if str(row["odds_capture_timestamp"]) != str(odds_row["capture_timestamp"]):
                raise ValueError(f"matrix_odds_timestamp_mismatch:{race_id}:box_{box}")
            official_row = official[race_id][box]
            if int(row["label_finish_position"]) != int(official_row["finish_position"]):
                raise ValueError(f"official_finish_mismatch:{race_id}:box_{box}")
            if int(row["label_is_winner"]) != int(official_row["is_winner"]):
                raise ValueError(f"official_winner_mismatch:{race_id}:box_{box}")
            if normalize_name(row["dog_name"]) != normalize_name(official_row["dog_name"]):
                raise ValueError(f"official_name_mismatch:{race_id}:box_{box}")
            raw_implied[box] = 1.0 / odds
            captures.append(captured)
            capture_modes[str(odds_row.get("capture_mode") or "missing")] += 1
            odds_levels[str(odds_row.get("odds_level") or "missing")] += 1
            box_sources[str(odds_row.get("sportsbet_box_source") or "missing")] += 1
            source_domains[urlparse(str(odds_row["source_url"])).netloc] += 1
            official_sources[str(official_row.get("source") or "missing")] += 1
            official_domains[urlparse(str(official_row["source_url"])).netloc] += 1
            lead_seconds.append((jumps[race_id] - captured).total_seconds())
        total_implied = sum(raw_implied.values())
        for row in race_rows:
            box = int(row["box_number"])
            reconstructed = raw_implied[box] / total_implied
            if not math.isclose(
                reconstructed,
                float(row["market_implied_probability"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"normalized_probability_mismatch:{race_id}:box_{box}")
        capture_spreads.append((max(captures) - min(captures)).total_seconds())
        shortest = min(float(selected_odds[race_id][box]["odds_decimal"]) for box in boxes)
        tied = [row for row in race_rows if float(selected_odds[race_id][int(row["box_number"])]["odds_decimal"]) == shortest]
        favourite = min(
            tied,
            key=lambda row: (
                int(row["box_number"]),
                str(row.get("dog_name") or "").casefold(),
            ),
        )
        box = int(favourite["box_number"])
        odds_row = selected_odds[race_id][box]
        odds = float(odds_row["odds_decimal"])
        won = int(favourite["label_is_winner"])
        returned = odds if won else 0.0
        band_key, band_label = odds_band(odds)
        selections.append(
            {
                "race_id": race_id,
                "race_date": str(favourite["race_date"]),
                "jump_at": str(favourite["jump_at"]),
                "venue": str(favourite["venue"]),
                "race_number": int(favourite["race_number"]),
                "box_number": box,
                "dog_name": str(favourite["dog_name"]),
                "decimal_odds": odds,
                "normalized_market_probability": float(favourite["market_implied_probability"]),
                "raw_inverse_odds_probability": 1.0 / odds,
                "odds_capture_timestamp": str(odds_row["capture_timestamp"]),
                "source_url": str(odds_row["source_url"]),
                "odds_band": band_key,
                "odds_band_label": band_label,
                "tied_shortest_odds_count": len(tied),
                "won": won,
                "finish_position": int(favourite["label_finish_position"]),
                "stake_units": 1.0,
                "returned_units": returned,
                "net_pnl_units": returned - 1.0,
            }
        )
    selections.sort(key=lambda row: (parse_datetime(row["jump_at"]), row["race_id"]))
    if market_evidence_failures:
        aggregate: dict[str, Any] = {}
        path_metrics: dict[str, Any] = {}
        uncertainty: dict[str, Any] = {}
        calibration_comparison: dict[str, Any] = {}
        bands: dict[str, Any] = {}
        verdict = "BLOCKED_MARKET_OR_LABEL_EVIDENCE"
    else:
        aggregate = summarise_bets(selections)
        path_metrics = drawdown_and_losing_sequence(selections)
        uncertainty = bootstrap_intervals(selections, replicates, seed)
        roi_interval = uncertainty["roi"]
        if roi_interval["lower"] > 0.0:
            verdict = "FAVOURITE_BENCHMARK_PROFITABLE_IN_SAMPLE"
        elif roi_interval["upper"] < 0.0:
            verdict = "FAVOURITE_BENCHMARK_UNPROFITABLE_IN_SAMPLE"
        else:
            verdict = "FAVOURITE_BENCHMARK_INCONCLUSIVE"
        bands = {}
        for key, _, _, label in ODDS_BANDS:
            summary = summarise_bets([row for row in selections if row["odds_band"] == key])
            summary["definition"] = label
            summary["interpretation"] = "descriptive_only_not_a_strategy_variant"
            bands[key] = summary
        observed_win_rate = float(aggregate["win_rate"])
        normalized_implied = mean(float(row["normalized_market_probability"]) for row in selections)
        raw_implied = mean(float(row["raw_inverse_odds_probability"]) for row in selections)
        calibration_comparison = {
            "observed_favourite_win_rate": observed_win_rate,
            "aggregate_expected_wins_from_normalized_market": sum(float(row["normalized_market_probability"]) for row in selections),
            "mean_normalized_market_probability": normalized_implied,
            "observed_minus_normalized_implied_win_rate": observed_win_rate - normalized_implied,
            "mean_raw_inverse_odds_probability": raw_implied,
            "observed_minus_raw_inverse_odds_win_rate": observed_win_rate - raw_implied,
            "note": "Normalized comparison assesses favourite calibration; raw inverse odds retain bookmaker overround and connect directly to decimal-price returns.",
        }
    evidence = {
        "sqlite_integrity_check": integrity,
        "odds_source": "rows stored as Sportsbet decimal WIN odds; source-paired price validation reported separately",
        "odds_selection_policy": "latest individually valid Sportsbet WIN odds per box strictly before jump, matching the sealed matrix builder",
        "capture_mode_runner_rows": dict(sorted(capture_modes.items())),
        "odds_level_runner_rows": dict(sorted(odds_levels.items())),
        "sportsbet_box_source_runner_rows": dict(sorted(box_sources.items())),
        "sportsbet_source_url_domains": dict(sorted(source_domains.items())),
        "official_label_sources": dict(sorted(official_sources.items())),
        "official_label_url_domains": dict(sorted(official_domains.items())),
        "minimum_odds_lead_seconds": min(lead_seconds),
        "median_odds_lead_seconds": median(lead_seconds),
        "maximum_odds_lead_seconds": max(lead_seconds),
        "races_with_single_capture_timestamp": sum(value == 0 for value in capture_spreads),
        "races_with_mixed_capture_timestamps": sum(value > 0 for value in capture_spreads),
        "maximum_within_race_capture_spread_seconds": max(capture_spreads),
        "paired_price_validation_runner_rows": dict(sorted(paired_price_status.items())),
        "paired_price_validation_races": {
            "races_total": len(paired_price_status_by_race),
            "races_all_stored_prices_match_paired_win": sum(
                counts["stored_matches_paired_win"] == sum(counts.values())
                for counts in paired_price_status_by_race.values()
            ),
            "races_with_stored_place_routed_as_win": sum(
                counts["stored_matches_paired_place"] > 0
                for counts in paired_price_status_by_race.values()
            ),
            "races_with_unparseable_raw_pair": sum(
                counts["raw_pair_unparseable"] > 0
                for counts in paired_price_status_by_race.values()
            ),
            "races_with_neither_price_match": sum(
                counts["stored_matches_neither_paired_price"] > 0
                for counts in paired_price_status_by_race.values()
            ),
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_date": ANALYSIS_DATE,
        "verdict": verdict,
        "scope": {
            "population": "all accepted races in the sealed development matrix",
            "date_min": min(row["race_date"] for row in selections),
            "date_max": max(row["race_date"] for row in selections),
            "runner_rows": len(rows),
            "races": len(selections),
            "excluded_races_before_benchmark": 184,
            "benchmark_additional_race_skips": 0,
            "august_or_forward_cohort_opened": False,
            "models_modified_or_selected": False,
            "database_open_mode": "read-only immutable query-only",
        },
        "predeclared_rule": {
            "selections_per_race": 1,
            "selection": "shortest valid decimal Sportsbet WIN odds, equivalent to highest race-normalized inverse-odds probability",
            "stake_units": 1.0,
            "race_skipping": "none beyond pre-existing matrix evidence exclusions",
            "tie_break": "lowest numeric box, then case-folded dog name",
            "return": "decimal_odds units when winner, otherwise zero",
            "odds_bands": [label for _, _, _, label in ODDS_BANDS],
            "odds_band_use": "descriptive only; no post-hoc band selection",
        },
        "verdict_rule": {
            "blocked": "any market, timing, label, population, or input-hash validation failure",
            "profitable": "95% race-bootstrap ROI interval wholly above zero",
            "unprofitable": "95% race-bootstrap ROI interval wholly below zero",
            "inconclusive": "95% race-bootstrap ROI interval includes zero",
        },
        "evidence_validation": evidence,
        "aggregate_metrics": aggregate,
        "path_metrics": path_metrics,
        "uncertainty": uncertainty,
        "market_calibration_comparison": calibration_comparison,
        "odds_band_diagnostics": bands,
        "findings": {
            "BLOCKING": [],
            "IMPORTANT": [
                "This is a consumed development-population diagnostic, not untouched or prospective evidence.",
                "The benchmark cannot establish a verified market edge or authorize betting, promotion, model selection, or staking changes.",
                "Odds are the sealed builder's latest valid per-box pre-jump observations; races with mixed capture timestamps are quantified in evidence_validation.",
            ],
            "OPTIONAL": [
                "Retain this fixed rule and report as an economic benchmark for future separately authorized evaluations; do not tune it on these results."
            ],
        },
        "claims": {
            "strongest_supported": "The frozen one-unit favourite rule had the reported in-sample economic performance on the exact sealed development population.",
            "unsupported": [
                "The rule has a verified betting edge.",
                "Any odds band is a profitable strategy.",
                "The result generalizes to August, a forward cohort, live execution, or future races.",
                "The benchmark changes model selection or promotion readiness.",
            ],
        },
        "main_modelling_programme_impact": "NONE; benchmark only, with no model selection, tuning, promotion, or future-cohort consumption.",
    }
    if market_evidence_failures:
        pair_rows = evidence["paired_price_validation_runner_rows"]
        pair_races = evidence["paired_price_validation_races"]
        report["verdict"] = "BLOCKED_MARKET_OR_LABEL_EVIDENCE"
        blocked_metric = {
            "status": "NOT_VALIDLY_COMPUTABLE",
            "reason": "Stored market_type=win prices do not consistently match source-explicit paired WIN prices.",
        }
        report["aggregate_metrics"] = dict(blocked_metric)
        report["path_metrics"] = dict(blocked_metric)
        report["uncertainty"] = dict(blocked_metric)
        report["market_calibration_comparison"] = dict(blocked_metric)
        report["odds_band_diagnostics"] = {
            key: {
                "definition": label,
                "status": "NOT_VALIDLY_COMPUTABLE",
                "interpretation": "descriptive_only_not_a_strategy_variant",
                "reason": blocked_metric["reason"],
            }
            for key, _, _, label in ODDS_BANDS
        }
        report["findings"] = {
            "BLOCKING": [
                "The sealed matrix's stored market_type=win values are not reliable WIN-price evidence: "
                f"{pair_rows.get('stored_matches_paired_place', 0):,} runner rows match the source-explicit paired PLACE price, "
                f"{pair_rows.get('stored_matches_neither_paired_price', 0):,} match neither paired price, and "
                f"{pair_rows.get('raw_pair_unparseable', 0):,} cannot be pair-parsed from retained raw text.",
                f"Only {pair_races['races_all_stored_prices_match_paired_win']:,} of {pair_races['races_total']:,} races have every stored price matching the paired WIN price. "
                "The no-additional-skipping rule prevents a partial-population substitute.",
            ],
            "IMPORTANT": [
                "Because the normalized market probabilities were derived from these stored prices, prior development-market calibration and scoring claims require a separate provenance re-audit before being treated as fully verified; this diagnostic does not recompute them.",
                "This task did not repair data, relabel odds, rebuild the matrix, or open August/forward evidence.",
            ],
            "OPTIONAL": [
                "A separately authorized report-only recovery could rebuild this same consumed development population from source-explicit paired WIN prices; "
                f"it must fail closed on the {pair_rows.get('raw_pair_unparseable', 0):,} unparseable and "
                f"{pair_rows.get('stored_matches_neither_paired_price', 0):,} neither-match rows rather than skip races."
            ],
        }
        report["claims"] = {
            "strongest_supported": "The requested favourite benchmark is blocked because the sealed development odds cannot consistently establish the actual Sportsbet WIN return price.",
            "unsupported": [
                "Any favourite ROI, P&L, drawdown, losing sequence, odds-band return, or market-calibration statistic computed from the stored market_type=win values.",
                "The rule has a verified betting edge or verified in-sample loss.",
                "The result generalizes to August, a forward cohort, live execution, or future races.",
                "This evidence changes model selection or promotion readiness.",
            ],
        }
        report["main_modelling_programme_impact"] = (
            "NO MODEL-SELECTION CHANGE. The benchmark is evidence-blocked and cannot influence tuning, promotion, or future-cohort use; however, the stored WIN-price routing defect is an input-provenance issue for any existing development-market calibration or economic claim and should be reviewed separately."
        )
        return [], market_evidence_failures, report
    return selections, [], report


def write_outputs(
    output_dir: Path,
    selections: Sequence[Mapping[str, Any]],
    market_evidence_failures: Sequence[Mapping[str, Any]],
    report: dict[str, Any],
    input_hashes: Mapping[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    output_paths: dict[str, Path] = {}
    if selections:
        selections_path = output_dir / "favourite_selections.csv"
        with selections_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selections[0]))
            writer.writeheader()
            writer.writerows(selections)
        output_paths[selections_path.name] = selections_path
    if market_evidence_failures:
        failures_path = output_dir / "market_evidence_failures.csv"
        with failures_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(market_evidence_failures[0]))
            writer.writeheader()
            writer.writerows(market_evidence_failures)
        output_paths[failures_path.name] = failures_path
    report["input_sha256"] = dict(sorted(input_hashes.items()))
    report["output_sha256"] = {
        **{name: sha256_file(path) for name, path in sorted(output_paths.items())},
        "builder_script": sha256_file(Path(__file__).resolve()),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_entries = {
        **{name: sha256_file(path) for name, path in sorted(output_paths.items())},
        "report.json": sha256_file(report_path),
    }
    manifest = output_dir / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(manifest_entries.items())),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--model-report", type=Path)
    parser.add_argument("--source-db", type=Path)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--original-matrix", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates_must_be_positive")
    if args.canonical_root is not None:
        if args.original_matrix is None:
            raise ValueError("canonical_mode_requires_original_matrix")
        if any(value is not None for value in (args.matrix_root, args.model_report, args.source_db)):
            raise ValueError("canonical_mode_rejects_legacy_inputs")
        rows, sidecar_by_key, audit, input_hashes, validation = validate_corrected_inputs(
            args.canonical_root.resolve(), args.original_matrix.resolve()
        )
        selections, report = build_corrected_benchmark(
            rows,
            sidecar_by_key,
            audit,
            validation,
            args.bootstrap_replicates,
            args.bootstrap_seed,
        )
        market_evidence_failures: list[dict[str, Any]] = []
    else:
        if any(value is None for value in (args.matrix_root, args.model_report, args.source_db)):
            raise ValueError("legacy_mode_requires_matrix_model_and_source_db")
        rows, matrix_report, model_report, input_hashes = validate_inputs(
            args.matrix_root.resolve(),
            args.model_report.resolve(),
            args.source_db.resolve(),
        )
        selections, market_evidence_failures, report = build_benchmark(
            rows, args.source_db.resolve(), args.bootstrap_replicates, args.bootstrap_seed
        )
        report["scope"].update(
            {
                "pre_existing_exclusion_counts": matrix_report["exclusion_counts"],
                "label_policy": matrix_report["label_policy"],
                "odds_policy": matrix_report["odds_policy"],
                "upstream_model_report_status": model_report["status"],
            }
        )
    write_outputs(
        args.output_dir.resolve(),
        selections,
        market_evidence_failures,
        report,
        input_hashes,
    )
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "population_races": report["scope"].get(
                    "races", report["scope"].get("eligible_races")
                ),
                "market_evidence_failure_rows": len(market_evidence_failures),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
