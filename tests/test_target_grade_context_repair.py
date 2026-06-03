from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.rebuild_same_distance_feature_packet import (
    _clean_lookup,
    _compute_grade_context_bundle,
    _histories_by_dog,
    _join_key,
    _load_csv,
    _load_jsonl,
    _normalize_grade_text,
    _parse_date,
    _same_distance_band,
    _resolve_target_grade_metadata,
    _resolve_target_metadata,
    _is_prior_to_target,
    assert_output_dir_safe,
    repair_packet,
)


PACKET_CSV = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "clean_history_feature_packet_20260602/pre_race_history_feature_packet.csv"
)
CLEAN_DATASET = Path(
    "artifacts/full_evidence_orchestration_20260525/"
    "isolated_challenger_box_bias_study_20260602/clean_official_dataset.jsonl"
)
DB_PATH = Path("greyhound_racing_data_writable.db")


def _packet_rows() -> list[dict[str, object]]:
    if not PACKET_CSV.exists():
        pytest.skip("repair packet artifact is not present")
    return _load_csv(PACKET_CSV)


def _clean_rows() -> list[dict[str, object]]:
    if not CLEAN_DATASET.exists():
        pytest.skip("clean official holdout artifact is not present")
    return _load_jsonl(CLEAN_DATASET)


def _clean_map() -> dict[tuple[str, str, str], dict[str, object]]:
    return _clean_lookup(_clean_rows())


def test_target_grade_requires_safe_provenance():
    rows = _packet_rows()
    clean_lookup = _clean_map()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for row in rows:
            if row.get("packet") != "historical":
                continue
            meta = _resolve_target_grade_metadata(conn, row, clean_lookup.get(_join_key(row)))
            if meta["target_grade_provenance_status"] in {"MISSING", "AMBIGUOUS_OR_MISSING"}:
                assert meta["target_grade_safe"] is None
                assert meta["target_grade_normalized"] is None
                break
        else:
            pytest.skip("no unsafe historical target-grade row was found")


def test_embedded_grade_not_used_as_target_metadata():
    rows = _packet_rows()
    clean_lookup = _clean_map()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for row in rows:
            if row.get("packet") != "historical":
                continue
            meta = _resolve_target_grade_metadata(conn, row, clean_lookup.get(_join_key(row)))
            if meta["target_grade_provenance_status"] not in {"MISSING", "AMBIGUOUS_OR_MISSING"}:
                continue
            forged = dict(row)
            forged["last_start_grade"] = "Grade 1"
            forged["grade"] = "Grade 1"
            forged["DIST"] = "999m"
            forged["G"] = "Grade 1"
            forged_meta = _resolve_target_grade_metadata(
                conn,
                forged,
                clean_lookup.get(_join_key(row)),
            )
            assert forged_meta["target_grade_safe"] is None
            assert forged_meta["target_grade_provenance_status"] == meta[
                "target_grade_provenance_status"
            ]
            break
        else:
            pytest.skip("no unsafe historical target-grade row was found")


def test_grade_vocab_normalization_preserves_unmapped():
    normalized, status = _normalize_grade_text("MIXED 4/5")
    assert normalized == "Mixed 4/5"
    assert status == "CANONICAL"

    unmapped, unmapped_status = _normalize_grade_text("??")
    assert unmapped is None
    assert unmapped_status == "UNMAPPED"


def test_class_transition_uses_only_prior_rows():
    rows = _packet_rows()
    history_index = _histories_by_dog(DB_PATH)
    clean_lookup = _clean_map()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        for row in rows:
            if row.get("packet") != "historical":
                continue
            target_meta = _resolve_target_metadata(conn, row)
            target_grade_meta = _resolve_target_grade_metadata(
                conn,
                row,
                clean_lookup.get(_join_key(row)),
            )
            if target_grade_meta["target_grade_normalized"] in (None, ""):
                continue

            bundle = _compute_grade_context_bundle(
                row=row,
                target_meta=target_meta,
                target_grade_meta=target_grade_meta,
                history_index=history_index,
            )
            dog_key = "".join(ch for ch in str(row.get("dog_name") or "").upper() if ch.isalnum())
            history = history_index.get(dog_key, [])
            target_date = _parse_date(target_meta.get("target_race_date") or row.get("race_date"))
            target_dt = target_meta.get("target_datetime")
            same_grade_prior_rows = [
                history_row
                for history_row in history
                if _is_prior_to_target(history_row, target_date, target_dt)
                and _normalize_grade_text(history_row.get("grade"))[0]
                == target_grade_meta["target_grade_normalized"]
            ]
            all_same_grade_rows = [
                history_row
                for history_row in history
                if _normalize_grade_text(history_row.get("grade"))[0]
                == target_grade_meta["target_grade_normalized"]
            ]
            if len(all_same_grade_rows) <= len(same_grade_prior_rows):
                continue

            assert bundle["same_grade_start_count"] == len(same_grade_prior_rows)
            assert bundle["same_grade_start_count"] < len(all_same_grade_rows)
            target_distance = target_meta.get("target_distance")
            same_distance_same_grade_prior_rows = [
                history_row
                for history_row in same_grade_prior_rows
                if _same_distance_band(history_row.get("distance"), target_distance)
            ]
            if bundle["same_distance_same_grade_start_count"] is not None:
                assert bundle["same_distance_same_grade_start_count"] == len(
                    same_distance_same_grade_prior_rows
                )
            break
        else:
            pytest.skip("no suitable historical row with prior same-grade history was found")


def test_target_grade_train_eval_schema_parity(tmp_path):
    if not PACKET_CSV.exists() or not CLEAN_DATASET.exists() or not DB_PATH.exists():
        pytest.skip("required artifacts are not present")

    output_dir = Path(
        "artifacts/full_evidence_orchestration_20260525/"
        "bounded_target_grade_repair_20260603/test_tmp_schema_parity"
    )
    result = repair_packet(
        input_packet=PACKET_CSV,
        clean_dataset=CLEAN_DATASET,
        output_dir=output_dir,
        db_path=DB_PATH,
    )
    assert result["leakage_status"] == "PASS"
    assert result["parity_status"] == "PASS"

    parity = json.loads((output_dir / "train_eval_schema_parity.json").read_text())
    assert parity["status"] == "PASS"
    assert "target_grade_safe" in parity["historical_present_fields"]
    assert "grade_change_indicator" in parity["rolling_present_fields"]
    assert "same_distance_same_grade_start_count" in parity["historical_present_fields"]


def test_no_production_writes_from_target_grade_repair():
    with pytest.raises(ValueError):
        assert_output_dir_safe(Path("artifacts/prediction_snapshots"))
    with pytest.raises(ValueError):
        assert_output_dir_safe(Path("model_registry"))
