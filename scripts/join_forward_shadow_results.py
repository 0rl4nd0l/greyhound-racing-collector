#!/usr/bin/env python3
"""Join forward shadow predictions to official TheDogs results.

This command is report-only. It reads a completed shadow prediction artifact,
looks up official TheDogs result pages, applies exact identity gates, computes
metrics for safe joins, and writes a fresh shadow-only result-join artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse

import requests


ROOT = Path(__file__).resolve().parents[1]
ROOT_STR = str(ROOT)
sys.path = [path for path in sys.path if path != ROOT_STR]
sys.path.insert(0, ROOT_STR)

from scripts.ingest_results_for_date import (  # noqa: E402
    THEDOGS_PUBLIC_HEADERS,
    parse_thedogs_result_html_runner_rows,
    remap_promoted_reserve_runner_rows,
    thedogs_result_rows_present,
    thedogs_result_urls_from_race_url,
)
from utils.race_lifecycle import extract_target_metadata_from_filename  # noqa: E402


DEFAULT_OUTPUT_PARENT = ROOT / "artifacts/full_evidence_orchestration_20260525"
DEFAULT_DB_PATH = ROOT / "greyhound_racing_data.db"
DEFAULT_PROTECTED_PATHS = (
    ROOT / "greyhound_racing_data.db",
    ROOT / "greyhound_racing_data_writable.db",
    ROOT / "model_registry/best_metadata.json",
    ROOT / "docs/model_contracts/v4_feature_contract.json",
    ROOT / "artifacts/prediction_snapshots/manifest.jsonl",
)
EXPECTED_OFFICIAL_RACES = 214
EXPECTED_OFFICIAL_DOG_ROWS = 1493
RESULT_JOIN_PREFIX = "artifacts/full_evidence_orchestration_20260525/forward_shadow_result_join_"
PROBABILITY_COLUMN = "shadow_rf_calibrated_probability"
NON_NAME_RESULT_BADGES = frozenset({"NBT"})
SCRATCH_STATUSES = frozenset({"SCR", "L/SCR", "LSCR"})

FINAL_STATUS_JOINED = "FORWARD_SHADOW_RESULTS_JOINED"
FINAL_STATUS_WAITING = "WAITING_FOR_OFFICIAL_RESULTS"
FINAL_STATUS_PARTIAL = "PARTIAL_JOIN_PENDING_MORE_RESULTS"
FINAL_STATUS_IDENTITY_BLOCKED = "BLOCKED_IDENTITY_MATCH_FAILURE"
FINAL_STATUS_DB_BLOCKED = "BLOCKED_DB_STATE"


FetchHtml = Callable[[str], Mapping[str, Any]]


def now_id(now: datetime | None = None) -> str:
    return (now or datetime.now().astimezone()).strftime("%Y%m%dT%H%M%S%z")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), ROOT.resolve())
    except ValueError:
        return str(path)


def assert_result_join_output_dir_safe(output_dir: Path) -> Path:
    logical = output_dir if output_dir.is_absolute() else ROOT / output_dir
    try:
        relative = logical.absolute().relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ValueError("output_dir_must_be_inside_repo") from exc
    if ".." in relative.parts:
        raise ValueError("output_dir_must_not_contain_parent_traversal")
    if not relative.as_posix().startswith(RESULT_JOIN_PREFIX):
        raise ValueError(f"output_dir_must_be_forward_shadow_result_join_artifact:{relative}")
    return logical.absolute()


def unique_default_output_dir(output_parent: Path, generated_at: datetime) -> Path:
    base = output_parent / f"forward_shadow_result_join_{now_id(generated_at)}"
    output_dir = assert_result_join_output_dir_safe(base)
    if not output_dir.exists():
        return output_dir
    for index in range(1, 1000):
        candidate = assert_result_join_output_dir_safe(Path(f"{base}_{index:03d}"))
        if not candidate.exists():
            return candidate
    raise RuntimeError("forward_shadow_result_join_output_dir_collision_exhausted")


def verify_db_state(db_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "forward_shadow_result_join_db_state_v1",
        "db_path": relpath(db_path),
        "expected_official_races": EXPECTED_OFFICIAL_RACES,
        "expected_official_dog_rows": EXPECTED_OFFICIAL_DOG_ROWS,
        "status": "FAIL",
        "fail_reasons": [],
    }
    if not db_path.exists():
        report["fail_reasons"].append("db_missing")
        return report

    try:
        report["sha256"] = sha256_file(db_path)
        connection = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        try:
            quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
            official_races = connection.execute(
                "SELECT count(DISTINCT race_id) FROM race_metadata "
                "WHERE winner_source='thedogs_official'"
            ).fetchone()[0]
            official_dog_rows = connection.execute(
                "SELECT count(*) FROM dog_race_data "
                "WHERE data_source='thedogs_official'"
            ).fetchone()[0]
        finally:
            connection.close()
    except Exception as exc:  # pragma: no cover - defensive artifact reporting
        report["fail_reasons"].append(f"db_read_failed:{exc!r}")
        return report

    report.update(
        {
            "quick_check": quick_check,
            "official_races": official_races,
            "official_dog_rows": official_dog_rows,
        }
    )
    if quick_check != "ok":
        report["fail_reasons"].append("quick_check_not_ok")
    if official_races != EXPECTED_OFFICIAL_RACES:
        report["fail_reasons"].append("official_race_count_mismatch")
    if official_dog_rows != EXPECTED_OFFICIAL_DOG_ROWS:
        report["fail_reasons"].append("official_dog_row_count_mismatch")
    if not report["fail_reasons"]:
        report["status"] = "PASS"
    return report


def protected_path_hashes(paths: Sequence[Path] | None = None) -> dict[str, str | None]:
    paths = DEFAULT_PROTECTED_PATHS if paths is None else paths
    return {relpath(path): sha256_file(path) for path in paths}


def normalize_result_identity_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.)\-:]\s*", "", text)
    changed = True
    while changed:
        changed = False
        for badge in NON_NAME_RESULT_BADGES:
            updated = re.sub(rf"\s+{re.escape(badge)}\s*$", "", text, flags=re.IGNORECASE)
            if updated != text:
                text = updated.strip()
                changed = True
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def display_name_without_result_badges(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^\s*\d{1,2}\s*[\.)\-:]\s*", "", text)
    for badge in NON_NAME_RESULT_BADGES:
        text = re.sub(rf"\s+{re.escape(badge)}\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def clip_probability(value: float) -> float:
    return min(max(float(value), 1e-15), 1.0 - 1e-15)


def parse_current_time(value: str | None) -> datetime:
    if not value:
        return datetime.now().astimezone()
    text = value.strip()
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-4:].isdigit():
        text = f"{text[:-5]}{text[-5:-2]}:{text[-2:]}"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def date_from_thedogs_url(url: str) -> str | None:
    match = re.search(r"/racing/[^/]+/(\d{4}-\d{2}-\d{2})/", str(url or ""))
    return match.group(1) if match else None


def base_url_without_query(url: str) -> str:
    parsed = urlparse(str(url or ""))
    return parsed._replace(query="", fragment="").geturl().rstrip("/")


def official_result_candidate_urls(race_url: str | None) -> list[str]:
    if not race_url:
        return []
    candidates = [
        race_url,
        base_url_without_query(race_url),
        *thedogs_result_urls_from_race_url(race_url),
    ]
    output: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def default_fetch_html(url: str) -> Mapping[str, Any]:
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            url,
            headers=THEDOGS_PUBLIC_HEADERS,
            timeout=20,
            allow_redirects=True,
        )
        return {
            "url": url,
            "final_url": response.url,
            "status_code": response.status_code,
            "text": response.text or "",
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - network-dependent
        return {
            "url": url,
            "final_url": None,
            "status_code": None,
            "text": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def load_shadow_predictions(shadow_run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predictions_path = shadow_run_dir / "shadow_predictions.csv"
    predictions: list[dict[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []
    with predictions_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            parsed = dict(row)
            try:
                parsed["box"] = int(parsed["box"])
                parsed["predicted_rank"] = int(parsed["predicted_rank"])
                parsed[PROBABILITY_COLUMN] = float(parsed[PROBABILITY_COLUMN])
                parsed["shadow_rf_uncalibrated_probability"] = float(
                    parsed["shadow_rf_uncalibrated_probability"]
                )
            except (TypeError, ValueError, KeyError) as exc:
                malformed_rows.append(
                    {
                        "line_number": line_number,
                        "race_id": parsed.get("race_id"),
                        "dog_name": parsed.get("dog_name"),
                        "reason": "invalid_prediction_row_numeric_field",
                        "error": f"{type(exc).__name__}: {exc}",
                        "row": parsed,
                    }
                )
                continue
            predictions.append(parsed)
    return predictions, malformed_rows


def race_key_from_metadata(meta: Mapping[str, Any], fallback_filename: str | None = None) -> str | None:
    filename = meta.get("filename") or fallback_filename
    if filename:
        stem = Path(str(filename)).stem
        if re.match(r"^Race\s+\d+\s+-\s+.+\s+-\s+\d{4}-\d{2}-\d{2}$", stem, re.I):
            return stem

    info = meta.get("race_info") or {}
    venue = info.get("venue") or meta.get("venue")
    race_number = info.get("race_number") or meta.get("race_number")
    race_date = info.get("date") or meta.get("race_date")
    if venue and race_number and race_date:
        return f"Race {int(race_number)} - {str(venue).upper()} - {race_date}"
    return None


def load_shadow_race_metadata(
    shadow_run_dir: Path,
    refresh_metadata_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    race_meta: dict[str, dict[str, Any]] = {}
    for meta_path in sorted((shadow_run_dir / "eligible_inputs").glob("source_*/*.metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        race_id = race_key_from_metadata(meta)
        if not race_id:
            continue
        info = meta.get("race_info") or {}
        filename = str(meta.get("filename") or meta_path.name.replace(".metadata.json", ""))
        filename_meta = extract_target_metadata_from_filename(filename)
        race_meta[race_id] = {
            "race_id": race_id,
            "source_metadata_path": relpath(meta_path),
            "source_csv_path": relpath(meta_path.with_name(meta_path.name.replace(".metadata.json", ""))),
            "race_url": meta.get("race_url") or meta.get("metadata_source_url") or info.get("url"),
            "venue": info.get("venue") or filename_meta.get("venue"),
            "race_number": int(info.get("race_number") or filename_meta.get("race_number")),
            "race_date": info.get("date") or filename_meta.get("race_date"),
            "race_time": info.get("race_time"),
            "metadata_is_leakage_safe": meta.get("metadata_is_leakage_safe"),
            "canonical_runner_alignment": (
                dict(meta.get("canonical_runner_alignment"))
                if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                else None
            ),
            "runner_completeness_after_canonical_alignment": (
                dict(meta.get("runner_completeness_after_canonical_alignment"))
                if isinstance(meta.get("runner_completeness_after_canonical_alignment"), Mapping)
                else None
            ),
        }

    if refresh_metadata_path and refresh_metadata_path.exists():
        refresh = json.loads(refresh_metadata_path.read_text(encoding="utf-8"))
        for item in refresh.get("selected_races") or []:
            race_url = str(item.get("race_url") or "")
            race_date = item.get("race_date") or date_from_thedogs_url(race_url)
            venue = item.get("venue")
            race_number = item.get("race_number")
            if not (race_date and venue and race_number):
                continue
            race_id = f"Race {int(race_number)} - {str(venue).upper()} - {race_date}"
            race_meta.setdefault(race_id, {"race_id": race_id})
            race_meta[race_id].update(
                {
                    "race_url": race_url or race_meta[race_id].get("race_url"),
                    "venue": str(venue).upper(),
                    "race_number": int(race_number),
                    "race_date": race_date,
                    "jump_datetime": item.get("jump_datetime"),
                    "refresh_source_path": relpath(refresh_metadata_path),
                }
            )
    return race_meta


def classify_result_identity_join(
    *,
    race_id: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    official_rows: Sequence[Mapping[str, Any]],
    race_url: str | None = None,
    result_url: str | None = None,
) -> dict[str, Any]:
    participant_rows = [
        {"box_number": row.get("box"), "dog_name": row.get("dog_name")}
        for row in prediction_rows
    ]
    reserve_remap = remap_promoted_reserve_runner_rows(
        [dict(row) for row in official_rows],
        participant_rows,
    )
    official_rows = reserve_remap["rows"]

    by_box: dict[int, Mapping[str, Any]] = {}
    duplicate_boxes: list[int] = []
    for official in official_rows:
        box = int(official.get("box_number") or 0)
        if not box:
            continue
        if box in by_box:
            duplicate_boxes.append(box)
        by_box[box] = official

    expected_boxes = {int(row.get("box") or 0) for row in prediction_rows}
    official_boxes = set(by_box)
    missing_predicted_boxes = sorted(expected_boxes - official_boxes)
    extra_official_boxes = sorted(official_boxes - expected_boxes)

    allowed_extra_scratched: list[dict[str, Any]] = []
    disallowed_extra: list[dict[str, Any]] = []
    for box in extra_official_boxes:
        official = by_box[box]
        status = str(official.get("status") or "").upper()
        item = {
            "box": box,
            "status": status,
            "dog_name": official.get("dog_name"),
        }
        if status in SCRATCH_STATUSES:
            allowed_extra_scratched.append(item)
        else:
            disallowed_extra.append(item)

    name_mismatches: list[dict[str, Any]] = []
    for prediction in prediction_rows:
        box = int(prediction.get("box") or 0)
        official = by_box.get(box)
        if official is None:
            continue
        prediction_identity = normalize_result_identity_name(prediction.get("dog_name"))
        official_identity = normalize_result_identity_name(official.get("dog_name"))
        if prediction_identity != official_identity:
            name_mismatches.append(
                {
                    "box": box,
                    "prediction_dog_name": prediction.get("dog_name"),
                    "official_dog_name": official.get("dog_name"),
                    "official_dog_name_after_badge_strip": display_name_without_result_badges(
                        official.get("dog_name")
                    ),
                    "prediction_identity": prediction_identity,
                    "official_identity": official_identity,
                }
            )

    winners = [
        official
        for official in official_rows
        if official.get("finish_position") == 1
        and int(official.get("box_number") or 0) in expected_boxes
    ]

    identity_errors: list[str] = []
    if duplicate_boxes:
        identity_errors.append("duplicate_official_boxes")
    if missing_predicted_boxes:
        identity_errors.append("missing_predicted_boxes_in_official_result")
    if disallowed_extra:
        identity_errors.append("extra_official_non_scratch_boxes_outside_prediction_set")
    if name_mismatches:
        identity_errors.append("dog_name_mismatch_after_exact_badge_stripping")
    if len(winners) != 1:
        identity_errors.append("winner_count_not_exactly_one")

    status = "UNSAFE_QUARANTINED" if identity_errors else "SAFE_EXACT_BOX_AND_NAME_MATCH"
    return {
        "race_id": race_id,
        "status": status,
        "identity_errors": identity_errors,
        "race_url": race_url,
        "result_url": result_url,
        "official_runner_rows": [dict(row) for row in official_rows],
        "reserve_box_remappings": reserve_remap["remappings"],
        "ignored_terminal_status_rows": reserve_remap["ignored_terminal_status_rows"],
        "rejected_reserve_box_remappings": reserve_remap["rejected_remappings"],
        "duplicate_official_boxes": duplicate_boxes,
        "missing_predicted_boxes": missing_predicted_boxes,
        "extra_official_boxes": extra_official_boxes,
        "allowed_extra_scratched_official_boxes": allowed_extra_scratched,
        "disallowed_extra_official_boxes": disallowed_extra,
        "name_mismatches": name_mismatches,
        "winner_count": len(winners),
        "winner_box": int(winners[0]["box_number"]) if len(winners) == 1 else None,
    }


def joined_rows_for_safe_race(
    *,
    race_meta: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    official_by_box = {
        int(row["box_number"]): row for row in identity.get("official_runner_rows", [])
    }
    winner_box = int(identity["winner_box"])
    winner_rank = None
    joined_rows: list[dict[str, Any]] = []
    for prediction in sorted(prediction_rows, key=lambda row: int(row.get("predicted_rank") or 999)):
        box = int(prediction["box"])
        official = official_by_box[box]
        is_winner = box == winner_box
        if is_winner:
            winner_rank = int(prediction["predicted_rank"])
        joined_rows.append(
            {
                "race_id": race_meta.get("race_id"),
                "race_date": race_meta.get("race_date"),
                "venue": race_meta.get("venue"),
                "race_number": race_meta.get("race_number"),
                "jump_datetime": race_meta.get("jump_datetime"),
                "race_url": race_meta.get("race_url"),
                "result_url": identity.get("result_url"),
                "box": box,
                "dog_name": prediction.get("dog_name"),
                "official_dog_name": official.get("dog_name"),
                "official_dog_name_after_badge_strip": display_name_without_result_badges(
                    official.get("dog_name")
                ),
                "finish_position": official.get("finish_position"),
                "is_winner": is_winner,
                "predicted_rank": int(prediction["predicted_rank"]),
                "shadow_rf_calibrated_probability": float(prediction[PROBABILITY_COLUMN]),
                "shadow_rf_uncalibrated_probability": float(
                    prediction["shadow_rf_uncalibrated_probability"]
                ),
                "calibration_method": prediction.get("calibration_method"),
                "tgr_enabled": str(prediction.get("tgr_enabled")).lower() == "true",
                "identity_match_status": "exact_box_and_normalized_name",
                "identity_name_badge_strip": sorted(NON_NAME_RESULT_BADGES),
                "allowed_extra_scratched_official_boxes": identity.get(
                    "allowed_extra_scratched_official_boxes", []
                ),
            }
        )

    top_pick = sorted(prediction_rows, key=lambda row: int(row.get("predicted_rank") or 999))[0]
    summary = {
        "race_id": race_meta.get("race_id"),
        "winner_box": winner_box,
        "winner_predicted_rank": winner_rank,
        "runner_count": len(prediction_rows),
        "top_pick_box": int(top_pick["box"]),
        "top_pick_dog_name": top_pick.get("dog_name"),
        "top_pick_won": winner_rank == 1,
        "winner_in_top3": bool(winner_rank is not None and winner_rank <= 3),
    }
    return joined_rows, summary


def probability_reliability_bins(labels: Sequence[int], probabilities: Sequence[float]) -> list[dict[str, Any]]:
    bins = []
    for index in range(10):
        start = index / 10
        end = (index + 1) / 10
        members = [
            (label, probability)
            for label, probability in zip(labels, probabilities)
            if probability >= start and (probability < end or (end == 1.0 and probability <= end))
        ]
        if members:
            labels_in_bin = [label for label, _ in members]
            probs_in_bin = [probability for _, probability in members]
            bins.append(
                {
                    "bin_start": start,
                    "bin_end": end,
                    "count": len(members),
                    "mean_predicted_probability": sum(probs_in_bin) / len(probs_in_bin),
                    "observed_rate": sum(labels_in_bin) / len(labels_in_bin),
                }
            )
        else:
            bins.append(
                {
                    "bin_start": start,
                    "bin_end": end,
                    "count": 0,
                    "mean_predicted_probability": None,
                    "observed_rate": None,
                }
            )
    return bins


def logistic_calibration_review(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    positive = sum(labels)
    negative = len(labels) - positive
    if len(labels) < 30 or positive < 5 or negative < 5:
        return {
            "status": "insufficient_sample",
            "sample_size": len(labels),
            "positive_labels": positive,
            "negative_labels": negative,
            "minimum_required": {
                "sample_size": 30,
                "positive_labels": 5,
                "negative_labels": 5,
            },
            "slope": None,
            "intercept": None,
        }

    logits = [math.log(clip_probability(p) / (1.0 - clip_probability(p))) for p in probabilities]
    if len(set(round(logit, 12) for logit in logits)) < 2:
        return {
            "status": "insufficient_probability_variation",
            "sample_size": len(labels),
            "positive_labels": positive,
            "negative_labels": negative,
            "slope": None,
            "intercept": None,
        }

    intercept = 0.0
    slope = 1.0
    for _ in range(50):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for logit, label in zip(logits, labels):
            eta = max(min(intercept + slope * logit, 35.0), -35.0)
            mean = 1.0 / (1.0 + math.exp(-eta))
            weight = max(mean * (1.0 - mean), 1e-9)
            diff = label - mean
            g0 += diff
            g1 += diff * logit
            h00 += weight
            h01 += weight * logit
            h11 += weight * logit * logit
        determinant = h00 * h11 - h01 * h01
        if abs(determinant) < 1e-12:
            return {
                "status": "singular_fit",
                "sample_size": len(labels),
                "positive_labels": positive,
                "negative_labels": negative,
                "slope": None,
                "intercept": None,
            }
        delta_intercept = (g0 * h11 - g1 * h01) / determinant
        delta_slope = (h00 * g1 - h01 * g0) / determinant
        intercept += delta_intercept
        slope += delta_slope
        if abs(delta_intercept) + abs(delta_slope) < 1e-8:
            break

    return {
        "status": "computed",
        "sample_size": len(labels),
        "positive_labels": positive,
        "negative_labels": negative,
        "intercept": intercept,
        "slope": slope,
        "method": "logistic_regression_on_logit_probability",
    }


def metric_reports(
    *,
    shadow_run_dir: Path,
    manifest: Mapping[str, Any],
    predictions_by_race: Mapping[str, Sequence[Mapping[str, Any]]],
    safe_races: Sequence[Mapping[str, Any]],
    joined_rows: Sequence[Mapping[str, Any]],
    pending_count: int,
    unsafe_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    labels = [1 if row.get("is_winner") else 0 for row in joined_rows]
    probabilities = [float(row[PROBABILITY_COLUMN]) for row in joined_rows]
    joined_by_race: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined_rows:
        joined_by_race[str(row.get("race_id"))].append(row)

    if safe_races:
        winner_ranks = [int(row["winner_predicted_rank"]) for row in safe_races]
        top1 = sum(1 for row in safe_races if row.get("top_pick_won")) / len(safe_races)
        top3 = sum(1 for row in safe_races if row.get("winner_in_top3")) / len(safe_races)
        brier = sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(
            labels
        )
        logloss = sum(
            -math.log(
                clip_probability(
                    float(next(row for row in rows if row.get("is_winner"))[PROBABILITY_COLUMN])
                )
            )
            for rows in joined_by_race.values()
        ) / len(joined_by_race)
        probability_sum_max_error = max(
            abs(sum(float(row[PROBABILITY_COLUMN]) for row in rows) - 1.0)
            for rows in joined_by_race.values()
        )
        metrics = {
            "schema_version": "forward_shadow_metrics_v1",
            "status": "COMPUTED_FOR_SAFE_JOINED_RACES",
            "source_shadow_run": relpath(shadow_run_dir),
            "safe_joined_race_count": len(safe_races),
            "safe_joined_runner_count": len(joined_rows),
            "pending_race_count": pending_count,
            "unsafe_match_count": unsafe_count,
            "top1": top1,
            "top3": top3,
            "winner_ranks": winner_ranks,
            "mean_winner_rank": sum(winner_ranks) / len(winner_ranks),
            "brier": brier,
            "logloss": logloss,
            "logloss_method": "mean_negative_log_calibrated_probability_assigned_to_winner_per_race",
            "probability_sum_max_error_joined_races": probability_sum_max_error,
            "joined_races": list(safe_races),
            "calibration_method": manifest.get("calibration_method"),
            "all_missing_train_policy": manifest.get("all_missing_train_policy"),
            "tgr_enabled": manifest.get("tgr_enabled"),
        }
    else:
        metrics = {
            "schema_version": "forward_shadow_metrics_v1",
            "status": "NO_SAFE_JOINED_RACES",
            "source_shadow_run": relpath(shadow_run_dir),
            "safe_joined_race_count": 0,
            "safe_joined_runner_count": 0,
            "pending_race_count": pending_count,
            "unsafe_match_count": unsafe_count,
            "top1": None,
            "top3": None,
            "winner_ranks": [],
            "mean_winner_rank": None,
            "brier": None,
            "logloss": None,
            "probability_sum_max_error_joined_races": None,
            "joined_races": [],
            "calibration_method": manifest.get("calibration_method"),
            "all_missing_train_policy": manifest.get("all_missing_train_policy"),
            "tgr_enabled": manifest.get("tgr_enabled"),
        }

    calibration = {
        "schema_version": "forward_shadow_calibration_review_v1",
        "source_shadow_run": relpath(shadow_run_dir),
        "safe_joined_race_count": len(safe_races),
        "safe_joined_runner_count": len(joined_rows),
        "calibration_method": manifest.get("calibration_method"),
        "brier": metrics.get("brier"),
        "logloss": metrics.get("logloss"),
        "reliability_bins": (
            probability_reliability_bins(labels, probabilities) if joined_rows else []
        ),
        "slope_intercept": (
            logistic_calibration_review(labels, probabilities)
            if joined_rows
            else {"status": "no_safe_joined_rows", "slope": None, "intercept": None}
        ),
    }

    all_top_picks = [
        sorted(rows, key=lambda row: int(row.get("predicted_rank") or 999))[0]
        for rows in predictions_by_race.values()
        if rows
    ]
    all_top_pick_counts = Counter(str(row.get("box")) for row in all_top_picks)
    joined_top_pick_counts = Counter(str(row.get("top_pick_box")) for row in safe_races)
    box_bias = {
        "schema_version": "forward_shadow_box_bias_review_v1",
        "source_shadow_run": relpath(shadow_run_dir),
        "all_shadow_race_count": len(predictions_by_race),
        "all_shadow_top_pick_box_distribution": dict(
            sorted(all_top_pick_counts.items(), key=lambda item: int(item[0]))
        ),
        "all_shadow_box_1_top_pick_share": (
            all_top_pick_counts.get("1", 0) / len(all_top_picks) if all_top_picks else None
        ),
        "safe_joined_race_count": len(safe_races),
        "safe_joined_top_pick_box_distribution": dict(
            sorted(joined_top_pick_counts.items(), key=lambda item: int(item[0]))
        ),
        "safe_joined_box_1_top_pick_share": (
            joined_top_pick_counts.get("1", 0) / len(safe_races) if safe_races else None
        ),
    }
    return metrics, calibration, box_bias


def final_status_for_counts(safe_count: int, pending_count: int, unsafe_count: int) -> str:
    if safe_count and not pending_count and not unsafe_count:
        return FINAL_STATUS_JOINED
    if safe_count and (pending_count or unsafe_count):
        return FINAL_STATUS_PARTIAL
    if unsafe_count and not pending_count:
        return FINAL_STATUS_IDENTITY_BLOCKED
    return FINAL_STATUS_WAITING


def build_summary(
    *,
    shadow_run_dir: Path,
    generated_at: datetime,
    final_status: str,
    db_before: Mapping[str, Any] | None,
    db_after: Mapping[str, Any] | None,
    protected_unchanged: bool,
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    lines = [
        "# Forward Shadow Result Join",
        "",
        f"- Source shadow run: `{relpath(shadow_run_dir)}`",
        f"- Generated at: `{generated_at.isoformat()}`",
        f"- Final verdict: `{final_status}`",
        f"- Safe joined races: `{metrics.get('safe_joined_race_count')}`",
        f"- Safe joined runner rows: `{metrics.get('safe_joined_runner_count')}`",
        f"- Pending races: `{metrics.get('pending_race_count')}`",
        f"- Unsafe/quarantined result matches: `{metrics.get('unsafe_match_count')}`",
        f"- DB state before/after: `{db_before}` / `{db_after}`",
        f"- Protected paths unchanged: `{protected_unchanged}`",
        f"- Calibration: `{manifest.get('calibration_method')}`",
        f"- All-missing-train policy: `{manifest.get('all_missing_train_policy')}`",
        f"- TGR enabled: `{manifest.get('tgr_enabled')}`",
        "",
        "## Metrics",
    ]
    if metrics.get("safe_joined_race_count"):
        lines.extend(
            [
                f"- Top1: `{metrics.get('top1')}`",
                f"- Top3: `{metrics.get('top3')}`",
                f"- Mean winner rank: `{metrics.get('mean_winner_rank')}`",
                f"- Brier: `{metrics.get('brier')}`",
                f"- LogLoss: `{metrics.get('logloss')}`",
                f"- Winner ranks: `{metrics.get('winner_ranks')}`",
            ]
        )
    else:
        lines.append("- Metrics not computed because no race passed the safe identity join gate.")
    lines.extend(
        [
            "",
            "## Identity Policy",
            "- Race identity came from the shadow source sidecars and TheDogs race URLs.",
            "- Runner identity required exact box and exact normalized dog name. Fuzzy-only matches were not accepted.",
            "- Known non-name result badges were stripped before exact comparison and recorded in the identity report.",
            "- Extra official scratched boxes outside the prediction field were allowed only with an official scratch status.",
            "",
            "## Stop State",
            "- Stopped after result reporting.",
            "- No production promotion, registry mutation, pointer update, TGR enablement, betting/EV output, DB writes, or label writes were performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def join_forward_shadow_results(
    *,
    shadow_run_dir: Path,
    output_parent: Path = DEFAULT_OUTPUT_PARENT,
    output_dir: Path | None = None,
    refresh_metadata_path: Path | None = None,
    db_path: Path = DEFAULT_DB_PATH,
    current_time: datetime | None = None,
    fetch_html: FetchHtml = default_fetch_html,
    verify_db: bool = True,
) -> dict[str, Any]:
    generated_at = current_time or datetime.now().astimezone()
    shadow_run_dir = shadow_run_dir.resolve()
    output_dir = (
        assert_result_join_output_dir_safe(output_dir)
        if output_dir is not None
        else unique_default_output_dir(output_parent, generated_at)
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    db_before = verify_db_state(db_path) if verify_db else None
    if verify_db and (not db_before or db_before.get("status") != "PASS"):
        write_text(output_dir / "final_status.txt", FINAL_STATUS_DB_BLOCKED + "\n")
        write_json(
            output_dir / "identity_match_report.json",
            {
                "schema_version": "forward_shadow_identity_match_report_v2",
                "generated_at": generated_at.isoformat(),
                "source_shadow_run": relpath(shadow_run_dir),
                "db_state_before": db_before,
                "summary": {"verdict": FINAL_STATUS_DB_BLOCKED},
            },
        )
        return {
            "output_dir": relpath(output_dir),
            "verdict": FINAL_STATUS_DB_BLOCKED,
            "db_state_before": db_before,
        }

    protected_before = protected_path_hashes()
    manifest = json.loads((shadow_run_dir / "shadow_manifest.json").read_text(encoding="utf-8"))
    predictions, malformed_prediction_rows = load_shadow_predictions(shadow_run_dir)
    predictions_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        predictions_by_race[str(row["race_id"])].append(row)
    for rows in predictions_by_race.values():
        rows.sort(key=lambda row: int(row.get("predicted_rank") or 999))
    malformed_prediction_rows_by_race: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in malformed_prediction_rows:
        race_id = str(row.get("race_id") or "").strip()
        if race_id:
            malformed_prediction_rows_by_race[race_id].append(dict(row))

    race_meta = load_shadow_race_metadata(shadow_run_dir, refresh_metadata_path)
    joined_rows: list[dict[str, Any]] = []
    safe_races: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []
    unsafe_matches: list[dict[str, Any]] = []
    race_attempts: list[dict[str, Any]] = []

    for race_id in sorted(set(predictions_by_race) | set(malformed_prediction_rows_by_race)):
        rows = predictions_by_race.get(race_id, [])
        meta = dict(race_meta.get(race_id, {"race_id": race_id}))
        meta["race_id"] = race_id
        race_url = meta.get("race_url")
        jump_datetime = parse_datetime(meta.get("jump_datetime"))
        attempt: dict[str, Any] = {
            "race_id": race_id,
            "race_url": race_url,
            "venue": meta.get("venue"),
            "race_number": meta.get("race_number"),
            "race_date": meta.get("race_date"),
            "jump_datetime": meta.get("jump_datetime"),
            "prediction_runner_count": len(rows),
            "prejump_runner_alignment": {
                "canonical_runner_alignment_status": (
                    (meta.get("canonical_runner_alignment") or {}).get("status")
                    if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                    else None
                ),
                "canonical_runner_set_status": (
                    (meta.get("canonical_runner_alignment") or {}).get(
                        "canonical_runner_set_status"
                    )
                    if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                    else None
                ),
                "canonical_runner_count": (
                    (meta.get("canonical_runner_alignment") or {}).get("canonical_runner_count")
                    if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                    else None
                ),
                "canonical_prediction_runner_count": (
                    (meta.get("canonical_runner_alignment") or {}).get(
                        "prediction_runner_count"
                    )
                    if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                    else None
                ),
                "remapped_participants": (
                    (meta.get("canonical_runner_alignment") or {}).get("remapped_participants")
                    if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                    else []
                ),
                "dropped_participants": (
                    (meta.get("canonical_runner_alignment") or {}).get("dropped_participants")
                    if isinstance(meta.get("canonical_runner_alignment"), Mapping)
                    else []
                ),
                "runner_completeness_after_canonical_alignment": meta.get(
                    "runner_completeness_after_canonical_alignment"
                ),
            },
            "lookup_urls": [],
            "result_status": None,
            "identity_status": None,
        }
        malformed_for_race = malformed_prediction_rows_by_race.get(race_id, [])
        if malformed_for_race:
            attempt["result_status"] = "PREDICTION_ROWS_MALFORMED"
            attempt["identity_status"] = "UNSAFE_QUARANTINED"
            attempt["identity_errors"] = ["malformed_prediction_rows_for_race"]
            attempt["malformed_prediction_rows"] = malformed_for_race
            unsafe_matches.append(
                {
                    "race_id": race_id,
                    "status": "UNSAFE_RESULT_MATCH_QUARANTINED",
                    "reason": ["malformed_prediction_rows_for_race"],
                    "race_url": race_url,
                    "malformed_prediction_rows": malformed_for_race,
                    "prejump_runner_alignment": attempt["prejump_runner_alignment"],
                }
            )
            race_attempts.append(attempt)
            continue

        if not race_url:
            attempt["result_status"] = "NO_RACE_URL"
            attempt["identity_status"] = "PENDING_OFFICIAL_RESULT"
            pending_results.append(
                {
                    "race_id": race_id,
                    "status": "PENDING_OFFICIAL_OUTCOME",
                    "reason": "no_race_url_available_for_lookup",
                    "metadata": meta,
                }
            )
            race_attempts.append(attempt)
            continue

        official_rows: list[dict[str, Any]] = []
        result_url = None
        lookup_error = None
        for candidate_url in official_result_candidate_urls(str(race_url)):
            response = fetch_html(candidate_url)
            text = str(response.get("text") or "")
            lookup = {
                "url": candidate_url,
                "final_url": response.get("final_url"),
                "status_code": response.get("status_code"),
                "content_length": len(text),
                "result_rows_present": False,
                "parsed_runner_rows": 0,
                "error": response.get("error"),
            }
            lookup_error = response.get("error") or lookup_error
            parsed = parse_thedogs_result_html_runner_rows(text)
            lookup["result_rows_present"] = bool(thedogs_result_rows_present(text))
            lookup["parsed_runner_rows"] = len(parsed)
            attempt["lookup_urls"].append(lookup)
            if lookup["result_rows_present"] and parsed:
                official_rows = [dict(row) for row in parsed]
                result_url = candidate_url
                break

        if not official_rows:
            reason = "official_result_rows_not_present"
            if jump_datetime and generated_at < jump_datetime:
                reason = "race_not_jumped_at_join_time"
            attempt["result_status"] = "OFFICIAL_RESULT_NOT_AVAILABLE"
            attempt["identity_status"] = "PENDING_OFFICIAL_RESULT"
            pending_results.append(
                {
                    "race_id": race_id,
                    "status": "PENDING_OFFICIAL_OUTCOME",
                    "reason": reason,
                    "race_url": race_url,
                    "jump_datetime": meta.get("jump_datetime"),
                    "lookup_error": lookup_error,
                }
            )
            race_attempts.append(attempt)
            continue

        attempt["result_status"] = "OFFICIAL_RESULT_ROWS_FOUND"
        attempt["winning_result_url"] = result_url
        identity = classify_result_identity_join(
            race_id=race_id,
            prediction_rows=rows,
            official_rows=official_rows,
            race_url=str(race_url),
            result_url=result_url,
        )
        attempt.update(
            {
                "identity_status": identity["status"],
                "identity_errors": identity["identity_errors"],
                "official_runner_rows": identity["official_runner_rows"],
                "reserve_box_remappings": identity["reserve_box_remappings"],
                "ignored_terminal_status_rows": identity["ignored_terminal_status_rows"],
                "rejected_reserve_box_remappings": identity["rejected_reserve_box_remappings"],
                "duplicate_official_boxes": identity["duplicate_official_boxes"],
                "missing_predicted_boxes": identity["missing_predicted_boxes"],
                "extra_official_boxes": identity["extra_official_boxes"],
                "allowed_extra_scratched_official_boxes": identity[
                    "allowed_extra_scratched_official_boxes"
                ],
                "disallowed_extra_official_boxes": identity["disallowed_extra_official_boxes"],
                "name_mismatches": identity["name_mismatches"],
                "winner_count": identity["winner_count"],
            }
        )
        if identity["status"] != "SAFE_EXACT_BOX_AND_NAME_MATCH":
            unsafe_matches.append(
                {
                    "race_id": race_id,
                    "status": "UNSAFE_RESULT_MATCH_QUARANTINED",
                    "reason": identity["identity_errors"],
                    "race_url": race_url,
                    "winning_result_url": result_url,
                    "missing_predicted_boxes": identity["missing_predicted_boxes"],
                    "allowed_extra_scratched_official_boxes": identity[
                        "allowed_extra_scratched_official_boxes"
                    ],
                    "disallowed_extra_official_boxes": identity[
                        "disallowed_extra_official_boxes"
                    ],
                    "reserve_box_remappings": identity["reserve_box_remappings"],
                    "ignored_terminal_status_rows": identity["ignored_terminal_status_rows"],
                    "rejected_reserve_box_remappings": identity[
                        "rejected_reserve_box_remappings"
                    ],
                    "name_mismatches": identity["name_mismatches"],
                    "official_runner_rows": identity["official_runner_rows"],
                    "prejump_runner_alignment": attempt["prejump_runner_alignment"],
                }
            )
            race_attempts.append(attempt)
            continue

        race_joined_rows, race_summary = joined_rows_for_safe_race(
            race_meta=meta,
            prediction_rows=rows,
            identity=identity,
        )
        joined_rows.extend(race_joined_rows)
        safe_races.append(race_summary)
        race_attempts.append(attempt)

    metrics, calibration, box_bias = metric_reports(
        shadow_run_dir=shadow_run_dir,
        manifest=manifest,
        predictions_by_race=predictions_by_race,
        safe_races=safe_races,
        joined_rows=joined_rows,
        pending_count=len(pending_results),
        unsafe_count=len(unsafe_matches),
    )
    final_status = final_status_for_counts(
        len(safe_races), len(pending_results), len(unsafe_matches)
    )
    db_after = verify_db_state(db_path) if verify_db else None
    protected_after = protected_path_hashes()
    protected_unchanged = protected_before == protected_after
    if verify_db and (
        not db_after or db_after.get("status") != "PASS" or db_after != db_before
    ):
        final_status = FINAL_STATUS_DB_BLOCKED

    identity_report = {
        "schema_version": "forward_shadow_identity_match_report_v2",
        "generated_at": generated_at.isoformat(),
        "source_shadow_run": relpath(shadow_run_dir),
        "identity_policy": {
            "race_match": "source sidecar race_url plus date venue race_number metadata",
            "runner_match": "exact box and exact normalized dog name for every predicted runner",
            "fuzzy_matching_allowed": False,
            "normalization": "casefold whitespace/punctuation normalization with deterministic stripping of known non-name result badges only",
            "stripped_non_name_badges": sorted(NON_NAME_RESULT_BADGES),
            "extra_official_boxes": "allowed only when official terminal status is SCR/L-SCR/LSCR and box was absent from the pre-jump prediction field",
            "promoted_reserve_boxes": "TheDogs result rugs 9/10 may be remapped to a frozen participant box only with a source '(from box N)' note and exact cleaned dog-name match",
            "nonwinner_missing_finish_position": "allowed after exact runner identity and exactly one official winner are established",
            "prejump_runner_set": "prediction sidecars may carry canonical pre-jump final-starter alignment evidence; it is reported with every race attempt and unsafe match",
        },
        "shadow_manifest_summary": {
            "prediction_rows": manifest.get("prediction_rows"),
            "race_count": manifest.get("race_count"),
            "calibration_method": manifest.get("calibration_method"),
            "all_missing_train_policy": manifest.get("all_missing_train_policy"),
            "tgr_enabled": manifest.get("tgr_enabled"),
            "output_mode": manifest.get("output_mode"),
        },
        "db_state_before": db_before,
        "db_state_after": db_after,
        "protected_hashes_before": protected_before,
        "protected_hashes_after": protected_after,
        "protected_paths_unchanged": protected_unchanged,
        "summary": {
            "safe_joined_race_count": len(safe_races),
            "safe_joined_runner_count": len(joined_rows),
            "pending_race_count": len(pending_results),
            "unsafe_match_count": len(unsafe_matches),
            "malformed_prediction_row_count": len(malformed_prediction_rows),
            "verdict": final_status,
        },
        "malformed_prediction_rows": malformed_prediction_rows,
        "race_attempts": race_attempts,
    }

    write_jsonl(output_dir / "joined_shadow_predictions.jsonl", joined_rows)
    write_json(output_dir / "shadow_forward_metrics.json", metrics)
    write_json(output_dir / "identity_match_report.json", identity_report)
    write_json(
        output_dir / "malformed_prediction_rows.json",
        {
            "schema_version": "forward_shadow_malformed_prediction_rows_v1",
            "malformed_prediction_row_count": len(malformed_prediction_rows),
            "malformed_prediction_rows": malformed_prediction_rows,
        },
    )
    write_json(
        output_dir / "pending_results.json",
        {
            "schema_version": "forward_shadow_pending_results_v1",
            "pending_race_count": len(pending_results),
            "pending_results": pending_results,
        },
    )
    write_json(
        output_dir / "unsafe_result_matches.json",
        {
            "schema_version": "forward_shadow_unsafe_result_matches_v1",
            "unsafe_match_count": len(unsafe_matches),
            "unsafe_result_matches": unsafe_matches,
        },
    )
    write_json(output_dir / "calibration_review.json", calibration)
    write_json(output_dir / "box_bias_review.json", box_bias)
    write_text(
        output_dir / "SUMMARY.md",
        build_summary(
            shadow_run_dir=shadow_run_dir,
            generated_at=generated_at,
            final_status=final_status,
            db_before=db_before,
            db_after=db_after,
            protected_unchanged=protected_unchanged,
            metrics=metrics,
            manifest=manifest,
        ),
    )
    write_text(output_dir / "final_status.txt", final_status + "\n")

    return {
        "output_dir": relpath(output_dir),
        "verdict": final_status,
        "safe_joined_race_count": len(safe_races),
        "safe_joined_runner_count": len(joined_rows),
        "pending_race_count": len(pending_results),
        "unsafe_match_count": len(unsafe_matches),
        "malformed_prediction_row_count": len(malformed_prediction_rows),
        "metrics": metrics,
        "db_state_before": db_before,
        "db_state_after": db_after,
        "protected_paths_unchanged": protected_unchanged,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-run-dir", required=True, type=Path)
    parser.add_argument("--output-parent", default=DEFAULT_OUTPUT_PARENT, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--refresh-metadata", type=Path)
    parser.add_argument("--db", default=DEFAULT_DB_PATH, type=Path)
    parser.add_argument("--current-time", help="ISO timestamp used for pending race-time decisions")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = join_forward_shadow_results(
        shadow_run_dir=args.shadow_run_dir,
        output_parent=args.output_parent,
        output_dir=args.output_dir,
        refresh_metadata_path=args.refresh_metadata,
        db_path=args.db,
        current_time=parse_current_time(args.current_time),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("verdict") != FINAL_STATUS_DB_BLOCKED else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
