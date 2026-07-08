#!/usr/bin/env python3
"""Build a report-only runner-level matrix for certified scorecard races."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_scorecard_residual_feature_packet import (  # noqa: E402
    FEATURE_FAMILIES,
    is_nondefault,
    is_present,
)

OUTPUT_PREFIX = "artifacts/full_evidence_orchestration_20260525/" "scorecard_runner_matrix_"
DEFAULT_DB = ROOT / "greyhound_racing_data.db"
REPORT_FILE = "scorecard_runner_matrix_report.json"
MATRIX_JSONL = "scorecard_runner_matrix.jsonl"
MATRIX_CSV = "scorecard_runner_matrix.csv"
SUMMARY_MD = "SUMMARY.md"
FINAL_READY = "SCORECARD_RUNNER_MATRIX_READY"
FINAL_DATA_MISSING = "SCORECARD_RUNNER_MATRIX_DATA_MISSING"
FINAL_REPRODUCTION_FAILED = "SCORECARD_RUNNER_MATRIX_REPRODUCTION_FAILED"

NO_WRITE_GUARANTEES = {
    "training": False,
    "production_promotion": False,
    "registry_mutation": False,
    "production_pointer_update": False,
    "active_model_replacement": False,
    "db_write": False,
    "label_write": False,
    "odds_write": False,
    "official_result_write": False,
    "daemon_control": False,
    "betting_or_ev_action": False,
    "snapshot_rewrite": False,
    "manifest_rewrite": False,
}


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


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
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(OUTPUT_PREFIX):
        raise ValueError(f"output_dir_must_be_scorecard_runner_matrix:{relative}")
    return logical.absolute()


def unique_dir(base: Path) -> Path:
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = Path(f"{base}_{index:03d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"output_dir_collision_exhausted:{base}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_manifest(output_dir: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        files[relpath(path) or str(path)] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "schema_version": "scorecard_runner_matrix_output_manifest_v1",
        "output_dir": relpath(output_dir),
        "files": files,
    }


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, Mapping):
                rows.append(dict(payload))
    return rows


def load_feature_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        race_id = str(row.get("race_id") or "")
        box = parse_int(row.get("box_number"))
        if race_id and box is not None:
            keyed[(race_id, box)] = dict(row)
    return keyed


def parse_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def sqlite_table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def sqlite_select_expr(columns: set[str], column: str, default: str = "NULL") -> str:
    return column if column in columns else f"{default} AS {column}"


def evidence_row_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("captured_at") or ""),
        str(row.get("inserted_at") or ""),
        parse_int(row.get("id")) or -1,
    )


def race_result_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str, int]:
    return (
        parse_int(row.get("position_count")) or -1,
        *evidence_row_sort_key(row),
    )


def official_results_by_box(
    conn: sqlite3.Connection,
    race_id: str,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    audit = {
        "status": "OK",
        "race_rows": 0,
        "runner_rows": 0,
        "race_identity_conflict_count": 0,
        "race_winner_conflict_count": 0,
        "runner_conflict_count": 0,
        "runner_duplicate_row_count": 0,
        "selected_position_count": None,
    }
    for table in (
        "autonomous_official_result_evidence_races",
        "autonomous_official_result_evidence_runners",
    ):
        if not sqlite_table_exists(conn, table):
            audit["status"] = "DATA_MISSING_OFFICIAL_RESULT_TABLE"
            audit["missing_table"] = table
            return {}, audit

    race_columns = sqlite_table_columns(conn, "autonomous_official_result_evidence_races")
    race_select_fields = [
        sqlite_select_expr(race_columns, "id", "0"),
        sqlite_select_expr(race_columns, "race_date"),
        sqlite_select_expr(race_columns, "venue"),
        sqlite_select_expr(race_columns, "race_number"),
        sqlite_select_expr(race_columns, "captured_at", "''"),
        sqlite_select_expr(race_columns, "inserted_at", "''"),
        sqlite_select_expr(race_columns, "status"),
        sqlite_select_expr(race_columns, "winner_name"),
        sqlite_select_expr(race_columns, "winner_box"),
        sqlite_select_expr(race_columns, "position_count"),
        sqlite_select_expr(race_columns, "participant_count"),
        sqlite_select_expr(race_columns, "box_order_json"),
    ]
    race_rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT {", ".join(race_select_fields)}
            FROM autonomous_official_result_evidence_races
            WHERE race_id = ?
            ORDER BY captured_at, inserted_at, id
            """,
            (race_id,),
        ).fetchall()
    ]
    audit["race_rows"] = len(race_rows)
    identity_signatures = {
        (row.get("race_date"), row.get("venue"), parse_int(row.get("race_number")))
        for row in race_rows
    }
    winner_signatures = {
        (
            str(row.get("status") or "").strip().lower(),
            str(row.get("winner_name") or "").strip().casefold(),
            parse_int(row.get("winner_box")),
        )
        for row in race_rows
    }
    audit["race_identity_conflict_count"] = max(0, len(identity_signatures) - 1)
    audit["race_winner_conflict_count"] = max(0, len(winner_signatures) - 1)
    if race_rows:
        selected_race = max(race_rows, key=race_result_sort_key)
        audit["selected_position_count"] = parse_int(selected_race.get("position_count"))

    runner_columns = sqlite_table_columns(conn, "autonomous_official_result_evidence_runners")
    runner_select_fields = [
        sqlite_select_expr(runner_columns, "id", "0"),
        sqlite_select_expr(runner_columns, "source"),
        sqlite_select_expr(runner_columns, "source_url"),
        sqlite_select_expr(runner_columns, "captured_at", "''"),
        sqlite_select_expr(runner_columns, "inserted_at", "''"),
        sqlite_select_expr(runner_columns, "box_number"),
        sqlite_select_expr(runner_columns, "dog_name"),
        sqlite_select_expr(runner_columns, "finish_position"),
        sqlite_select_expr(runner_columns, "is_winner"),
    ]
    runner_rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT {", ".join(runner_select_fields)}
            FROM autonomous_official_result_evidence_runners
            WHERE race_id = ?
            ORDER BY box_number, captured_at, inserted_at, id
            """,
            (race_id,),
        ).fetchall()
    ]
    audit["runner_rows"] = len(runner_rows)
    results: dict[int, dict[str, Any]] = {}
    seen_by_box: Counter[int] = Counter()
    for row in runner_rows:
        box = parse_int(row.get("box_number"))
        if box is None:
            continue
        seen_by_box[box] += 1
        if seen_by_box[box] > 1:
            audit["runner_duplicate_row_count"] += 1
        result = {
            "box_number": box,
            "dog_name": row.get("dog_name"),
            "finish_position": parse_int(row.get("finish_position")),
            "is_winner": parse_int(row.get("is_winner")),
            "source": row.get("source"),
            "source_url": row.get("source_url"),
            "captured_at": row.get("captured_at"),
            "dedupe_sort_key": evidence_row_sort_key(row),
        }
        existing = results.get(box)
        if existing is None:
            results[box] = result
            continue
        if (
            existing.get("finish_position") != result.get("finish_position")
            or existing.get("is_winner") != result.get("is_winner")
            or str(existing.get("dog_name") or "").strip().casefold()
            != str(result.get("dog_name") or "").strip().casefold()
        ):
            audit["runner_conflict_count"] += 1
            continue
        if result["dedupe_sort_key"] >= existing.get("dedupe_sort_key", ("", "", -1)):
            results[box] = result

    if any(
        int(audit.get(key) or 0) > 0
        for key in (
            "race_identity_conflict_count",
            "race_winner_conflict_count",
            "runner_conflict_count",
        )
    ):
        audit["status"] = "CONFLICTING_OFFICIAL_RESULT_EVIDENCE"
    elif not results:
        audit["status"] = "DATA_MISSING_OFFICIAL_RESULT_RUNNERS"
    return results, audit


def prediction_probability(row: Mapping[str, Any]) -> float | None:
    for key in (
        "shadow_rf_calibrated_probability",
        "stage2_probability",
        "predicted_probability",
        "win_probability",
        "probability",
    ):
        parsed = parse_float(row.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    return None


def normalized_probability_by_box(raw: Mapping[int, float | None]) -> dict[int, float]:
    positives = {
        int(box): float(value)
        for box, value in raw.items()
        if value is not None and float(value) > 0
    }
    total = sum(positives.values())
    if total <= 0:
        return {}
    return {box: value / total for box, value in positives.items()}


def safe_logloss(probability: float | None) -> float | None:
    if probability is None:
        return None
    clipped = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    return -math.log(clipped)


def mean(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def validate_strict_odds_row(row: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(row.get("market_type") or "").strip().lower() != "win":
        reasons.append("odds_market_not_win")
    if str(row.get("source") or "").strip().lower() != "sportsbet":
        reasons.append("odds_source_not_sportsbet")
    decimal = parse_float(row.get("odds_decimal"))
    if decimal is None or decimal <= 1.0:
        reasons.append("odds_decimal_invalid")
    source_url = str(row.get("source_url") or "").lower()
    if "sportsbet.com.au" not in source_url:
        reasons.append("odds_source_url_not_sportsbet")
    if not str(row.get("capture_timestamp") or "").strip():
        reasons.append("odds_capture_timestamp_missing")
    if str(row.get("odds_level") or "").strip().lower() != "dog":
        reasons.append("odds_level_not_dog")
    if str(row.get("sportsbet_box_source") or "").strip() != "runner_text":
        reasons.append("unsupported_sportsbet_box_source")
    if parse_int(row.get("box_number")) is None:
        reasons.append("odds_box_number_missing")
    if not str(row.get("dog_name") or row.get("dog_clean_name") or "").strip():
        reasons.append("odds_dog_name_missing")
    return reasons


def latest_strict_odds_by_box(conn: sqlite3.Connection, race_id: str) -> dict[int, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM live_odds
        WHERE race_id = ?
        ORDER BY box_number, capture_timestamp, timestamp, id
        """,
        (race_id,),
    ).fetchall()
    by_box: dict[int, dict[str, Any]] = {}
    for sqlite_row in rows:
        row = dict(sqlite_row)
        if validate_strict_odds_row(row):
            continue
        box = parse_int(row.get("box_number"))
        if box is None:
            continue
        candidate_timestamp = str(row.get("capture_timestamp") or row.get("timestamp") or "")
        existing = by_box.get(box)
        existing_timestamp = (
            str(existing.get("capture_timestamp") or existing.get("timestamp") or "")
            if existing
            else ""
        )
        if existing is None or candidate_timestamp >= existing_timestamp:
            by_box[box] = row
    return by_box


def ranked_model_boxes(predictions_by_box: Mapping[int, Mapping[str, Any]]) -> list[int]:
    rankable: list[tuple[float, int]] = []
    for box, row in predictions_by_box.items():
        rank = parse_float(row.get("predicted_rank"))
        probability = prediction_probability(row)
        if rank is not None:
            key = rank
        elif probability is not None:
            key = -probability
        else:
            key = float(box)
        rankable.append((key, int(box)))
    return [box for _, box in sorted(rankable)]


def ranked_market_boxes(
    odds_by_box: Mapping[int, Mapping[str, Any]], boxes: Iterable[int]
) -> list[int]:
    candidates: list[tuple[float, int]] = []
    for box in boxes:
        odds = odds_by_box.get(int(box)) or {}
        decimal = parse_float(odds.get("odds_decimal"))
        if decimal is not None:
            candidates.append((decimal, int(box)))
    return [box for _, box in sorted(candidates)]


def feature_counts(feature_row: Mapping[str, Any] | None, family: str) -> tuple[int, int]:
    if not feature_row:
        return 0, 0
    fields = FEATURE_FAMILIES[family]
    present = sum(1 for field in fields if is_present(feature_row.get(field)))
    nondefault = sum(1 for field in fields if is_nondefault(feature_row.get(field)))
    return present, nondefault


def metrics_from_matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("race_id"))].append(row)
    model_top1 = 0
    model_top3 = 0
    market_top1 = 0
    market_top3 = 0
    model_ranks: list[float] = []
    market_ranks: list[float] = []
    model_logloss: list[float] = []
    market_logloss: list[float] = []
    for race_rows in groups.values():
        winner_rows = [row for row in race_rows if int(row.get("actual_win") or 0) == 1]
        if len(winner_rows) != 1:
            continue
        winner = winner_rows[0]
        model_rank = parse_int(winner.get("model_rank"))
        market_rank = parse_int(winner.get("market_rank"))
        model_probability = parse_float(winner.get("win_prob_norm"))
        market_probability = parse_float(winner.get("market_implied_probability_normalized"))
        if model_rank is not None:
            model_ranks.append(float(model_rank))
            model_top1 += int(model_rank == 1)
            model_top3 += int(model_rank <= 3)
        if market_rank is not None:
            market_ranks.append(float(market_rank))
            market_top1 += int(market_rank == 1)
            market_top3 += int(market_rank <= 3)
        loss = safe_logloss(model_probability)
        if loss is not None:
            model_logloss.append(loss)
        loss = safe_logloss(market_probability)
        if loss is not None:
            market_logloss.append(loss)
    race_count = len(groups)
    return {
        "race_count": race_count,
        "model_top1_accuracy": model_top1 / race_count if race_count else None,
        "model_top3_accuracy": model_top3 / race_count if race_count else None,
        "model_mean_winner_rank": mean(model_ranks),
        "model_logloss": mean(model_logloss),
        "market_top1_accuracy": market_top1 / race_count if race_count else None,
        "market_top3_accuracy": market_top3 / race_count if race_count else None,
        "market_mean_winner_rank": mean(market_ranks),
        "market_logloss": mean(market_logloss),
    }


def scorecard_metrics(scorecard_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(scorecard_rows)
    return {
        "race_count": count,
        "model_top1_accuracy": (
            sum(1 for row in scorecard_rows if str(row.get("model_top1_correct")) == "True") / count
            if count
            else None
        ),
        "model_top3_accuracy": (
            sum(1 for row in scorecard_rows if str(row.get("model_top3_correct")) == "True") / count
            if count
            else None
        ),
        "model_mean_winner_rank": mean(
            parse_float(row.get("model_winner_rank")) for row in scorecard_rows
        ),
        "model_logloss": mean(parse_float(row.get("model_logloss")) for row in scorecard_rows),
        "market_top1_accuracy": (
            sum(1 for row in scorecard_rows if str(row.get("market_top1_correct")) == "True")
            / count
            if count
            else None
        ),
        "market_top3_accuracy": (
            sum(1 for row in scorecard_rows if str(row.get("market_top3_correct")) == "True")
            / count
            if count
            else None
        ),
        "market_mean_winner_rank": mean(
            parse_float(row.get("market_winner_rank")) for row in scorecard_rows
        ),
        "market_logloss": mean(parse_float(row.get("market_logloss")) for row in scorecard_rows),
    }


def metrics_match(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    tolerance: float = 1e-12,
) -> tuple[bool, dict[str, Any]]:
    diffs: dict[str, Any] = {}
    ok = True
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if expected_value is None and actual_value is None:
            diffs[key] = 0.0
            continue
        if expected_value is None or actual_value is None:
            ok = False
            diffs[key] = "missing"
            continue
        delta = abs(float(expected_value) - float(actual_value))
        diffs[key] = delta
        if delta > tolerance:
            ok = False
    return ok, diffs


def prediction_path(scorecard_row: Mapping[str, Any]) -> Path:
    return Path(str(scorecard_row.get("winner_prediction_source_path") or ""))


def build_packet(
    *,
    scorecard_csv: Path,
    db_path: Path,
    output_dir: Path,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    output_dir = unique_dir(assert_output_dir_safe(output_dir))
    output_dir.mkdir(parents=True, exist_ok=False)
    generated_at = generated_at or datetime.now().astimezone()

    scorecard_rows = load_csv(scorecard_csv)
    predictions_by_path: dict[Path, list[dict[str, Any]]] = {}
    features_by_path: dict[Path, dict[tuple[str, int], dict[str, Any]]] = {}
    matrix_rows: list[dict[str, Any]] = []
    skipped = Counter()
    official_status_counts = Counter()
    official_join_counts = Counter()
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    try:
        for score_row in scorecard_rows:
            race_id = str(score_row.get("race_id") or "")
            winner_box = parse_int(score_row.get("winner_box"))
            source_path = prediction_path(score_row)
            if source_path not in predictions_by_path:
                if not source_path.exists():
                    predictions_by_path[source_path] = []
                else:
                    predictions_by_path[source_path] = load_jsonl(source_path)
            feature_path = source_path.parent / "shadow_feature_rows.json"
            if feature_path not in features_by_path:
                features_by_path[feature_path] = load_feature_rows(feature_path)
            prediction_rows = [
                row
                for row in predictions_by_path[source_path]
                if str(row.get("race_id")) == race_id
            ]
            predictions_by_box: dict[int, dict[str, Any]] = {}
            for row in prediction_rows:
                box = parse_int(
                    row.get("box") if row.get("box") not in (None, "") else row.get("box_number")
                )
                if box is not None:
                    predictions_by_box[box] = row
            shadow_boxes = set(predictions_by_box)
            if not race_id or winner_box is None or not shadow_boxes:
                skipped["prediction_rows_missing"] += 1
                continue
            odds_by_box = latest_strict_odds_by_box(conn, race_id)
            if not shadow_boxes.issubset(set(odds_by_box)):
                skipped["strict_odds_incomplete_for_shadow_boxes"] += 1
                continue
            model_order = ranked_model_boxes(predictions_by_box)
            market_order = ranked_market_boxes(odds_by_box, shadow_boxes)
            if winner_box not in model_order or winner_box not in market_order:
                skipped["winner_missing_from_model_or_market_order"] += 1
                continue
            official_by_box, official_audit = official_results_by_box(conn, race_id)
            official_status = str(official_audit.get("status") or "DATA_MISSING")
            official_status_counts[official_status] += 1
            if official_status != "OK":
                skipped[f"official_result_{official_status.lower()}"] += 1
                continue
            if not shadow_boxes.issubset(set(official_by_box)):
                skipped["official_result_incomplete_for_shadow_boxes"] += 1
                continue
            winner_results = [
                result
                for box, result in official_by_box.items()
                if box in shadow_boxes
                and (
                    parse_int(result.get("finish_position")) == 1
                    or parse_int(result.get("is_winner")) == 1
                )
            ]
            if (
                len(winner_results) != 1
                or parse_int(winner_results[0].get("box_number")) != winner_box
            ):
                skipped["official_result_winner_mismatch"] += 1
                continue
            raw_model = {
                box: prediction_probability(row) for box, row in predictions_by_box.items()
            }
            raw_market = {
                box: 1.0 / float(odds_by_box[box]["odds_decimal"]) for box in shadow_boxes
            }
            model_probs = normalized_probability_by_box(raw_model)
            market_probs = normalized_probability_by_box(raw_market)
            feature_rows = features_by_path[feature_path]
            for box in sorted(shadow_boxes):
                prediction = predictions_by_box[box]
                odds = odds_by_box[box]
                official = official_by_box.get(box) or {}
                feature = feature_rows.get((race_id, box))
                finish_position = parse_int(official.get("finish_position"))
                official_join_status = (
                    "JOINED_OFFICIAL_RESULT"
                    if finish_position is not None
                    else "DATA_MISSING_OFFICIAL_RESULT"
                )
                official_join_counts[official_join_status] += 1
                row = {
                    "race_id": race_id,
                    "label_race_id": race_id,
                    "snapshot_instance_id": race_id,
                    "race_date": score_row.get("race_date"),
                    "venue": score_row.get("venue"),
                    "race_number": score_row.get("race_number"),
                    "runner_count": len(shadow_boxes),
                    "box_number": box,
                    "box": box,
                    "dog_name": prediction.get("dog_name")
                    or odds.get("dog_name")
                    or official.get("dog_name"),
                    "actual_win": 1 if box == winner_box else 0,
                    "is_winner": 1 if box == winner_box else 0,
                    "finish_position": finish_position,
                    "label_quality": "official_or_complete_result",
                    "result_detail_quality": "finish_position",
                    "official_result_join_status": official_join_status,
                    "official_result_source": official.get("source"),
                    "official_result_source_url": official.get("source_url"),
                    "official_result_captured_at": official.get("captured_at"),
                    "model_rank": model_order.index(box) + 1,
                    "market_rank": market_order.index(box) + 1,
                    "predicted_rank": prediction.get("predicted_rank"),
                    "shadow_rf_calibrated_probability": prediction_probability(prediction),
                    "win_prob_norm": model_probs.get(box),
                    "market_implied_probability_raw": raw_market.get(box),
                    "market_implied_probability_normalized": market_probs.get(box),
                    "odds_win": odds.get("odds_decimal"),
                    "market_odds_win": odds.get("odds_decimal"),
                    "odds_capture_timestamp": odds.get("capture_timestamp"),
                    "odds_capture_mode": odds.get("capture_mode"),
                    "odds_level": odds.get("odds_level"),
                    "sportsbet_box_source": odds.get("sportsbet_box_source"),
                    "prediction_source_path": str(source_path),
                    "feature_rows_path": str(feature_path),
                    "feature_row_join_status": (
                        "JOINED_FEATURE_ROW" if feature else "DATA_MISSING_FEATURE_ROW"
                    ),
                    "schema_version": "scorecard_runner_matrix_row_v1",
                }
                for family in FEATURE_FAMILIES:
                    present, nondefault = feature_counts(feature, family)
                    row[f"{family}_present_field_count"] = present
                    row[f"{family}_nondefault_field_count"] = nondefault
                matrix_rows.append(row)
    finally:
        conn.close()

    expected_metrics = scorecard_metrics(scorecard_rows)
    actual_metrics = metrics_from_matrix(matrix_rows)
    reproduction_ok, reproduction_diffs = metrics_match(expected_metrics, actual_metrics)
    final_status = (
        FINAL_READY
        if matrix_rows and reproduction_ok and not skipped
        else FINAL_REPRODUCTION_FAILED if matrix_rows else FINAL_DATA_MISSING
    )
    report = {
        "schema_version": "scorecard_runner_matrix_report_v1",
        "generated_at": generated_at.isoformat(),
        "final_status": final_status,
        "recommended_decision": (
            "RUN_EXISTING_BOX_BIAS_STUDY_WITH_EVALUATION_DATASET"
            if final_status == FINAL_READY
            else "DATA_MISSING_REPAIR_MATRIX_BEFORE_CHALLENGER"
        ),
        "scorecard_csv": relpath(scorecard_csv),
        "db_path": str(db_path),
        "output_dir": relpath(output_dir),
        "scorecard_race_count": len(scorecard_rows),
        "matrix_race_count": len({row["race_id"] for row in matrix_rows}),
        "matrix_runner_row_count": len(matrix_rows),
        "skipped_race_reason_counts": dict(sorted(skipped.items())),
        "expected_scorecard_metrics": expected_metrics,
        "matrix_reproduced_metrics": actual_metrics,
        "metric_reproduction_ok": reproduction_ok,
        "metric_reproduction_abs_diffs": reproduction_diffs,
        "prediction_source_path_count": len(predictions_by_path),
        "feature_rows_path_count": len(features_by_path),
        "feature_join_counts": dict(
            Counter(row["feature_row_join_status"] for row in matrix_rows).most_common()
        ),
        "official_result_status_counts": dict(sorted(official_status_counts.items())),
        "official_result_join_counts": dict(official_join_counts.most_common()),
        "matrix_jsonl": relpath(output_dir / MATRIX_JSONL),
        "matrix_csv": relpath(output_dir / MATRIX_CSV),
        "evaluation_dataset_compatible": True,
        "no_write_guarantees": dict(NO_WRITE_GUARANTEES),
    }
    write_jsonl(output_dir / MATRIX_JSONL, matrix_rows)
    write_csv(output_dir / MATRIX_CSV, matrix_rows)
    write_json(output_dir / REPORT_FILE, report)
    write_text(output_dir / SUMMARY_MD, summary_markdown(report))
    write_text(output_dir / "final_status.txt", final_status + "\n")
    write_json(output_dir / "output_manifest.json", output_manifest(output_dir))
    return report


def summary_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Scorecard Runner Matrix",
            "",
            f"Final status: `{report.get('final_status')}`",
            f"Recommended decision: `{report.get('recommended_decision')}`",
            f"Scorecard races: `{report.get('scorecard_race_count')}`",
            f"Matrix races: `{report.get('matrix_race_count')}`",
            f"Runner rows: `{report.get('matrix_runner_row_count')}`",
            f"Metric reproduction: `{report.get('metric_reproduction_ok')}`",
            "",
            "```json",
            json.dumps(report.get("matrix_reproduced_metrics") or {}, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scorecard-csv", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or (ROOT / f"{OUTPUT_PREFIX}{now_id()}_report_only")
    report = build_packet(
        scorecard_csv=args.scorecard_csv,
        db_path=args.db,
        output_dir=output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
