from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from race_collection.forward_sealed_corpus import (
    ForwardCorpusRejected,
    ForwardSealedCorpus,
)
from race_collection.manual_prediction_collector_request import (
    ManualPredictionCollectorProtocol,
    ProtocolRejected,
)
from race_collection.scheduled_forward_corpus import admit_scheduled_capture
from race_collection.synchronous_manual_capture import (
    CollectorBusy,
    publish_scheduled_capture_receipts,
)
from src.predictor.on_demand import canonical_bytes, sha256_bytes

NOW = datetime.fromisoformat("2026-08-03T12:00:00+10:00")
JUMP = NOW + timedelta(minutes=20)
RACE_ID = "Race 1 - WARRNAMBOOL - 2026-08-03"
RACE_URL = "https://www.thedogs.com.au/racing/warrnambool/2026-08-03/1?trial=false"
RUNNERS = [
    {"box_number": 1, "dog_name": "Bravo", "identity": "BRAVO"},
    {"box_number": 2, "dog_name": "Alpha", "identity": "ALPHA"},
]


@dataclass(frozen=True)
class ScheduledFixture:
    protocol: ManualPredictionCollectorProtocol
    evidence_root: Path
    corpus_root: Path
    plan_item: dict[str, Any]
    attempt: dict[str, Any]
    receipt_publish: dict[str, Any]


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _fixture(
    tmp_path: Path,
    *,
    mutate_plan: Callable[[dict[str, Any]], None] | None = None,
    mutate_sidecar: Callable[[dict[str, Any]], None] | None = None,
    mutate_raw: Callable[[bytes], bytes] | None = None,
) -> ScheduledFixture:
    evidence_root = tmp_path / "evidence"
    output_dir = evidence_root / "autonomous_live_odds_capture_fixture"
    output_dir.mkdir(parents=True)
    protocol = ManualPredictionCollectorProtocol(
        evidence_root / "manual_prediction_collector_requests_v1"
    )
    corpus_root = evidence_root / "forward-corpus"
    (corpus_root / "artifacts").mkdir(parents=True)
    (corpus_root / "races").mkdir()

    form = output_dir / "Race 1 - WARRNAMBOOL - 2026-08-03.csv"
    form.write_text("Dog Name,BOX\n1. Bravo,\n2. Alpha,\n", encoding="utf-8")
    sidecar_path = form.with_name(form.name + ".metadata.json")
    raw_dir = output_dir / "raw_exports"
    raw_dir.mkdir()
    raw_path = raw_dir / form.name
    raw_bytes = form.read_bytes()
    if mutate_raw is not None:
        raw_bytes = mutate_raw(raw_bytes)
    raw_path.write_bytes(raw_bytes)

    sidecar = {
        "normalization_status": "verified",
        "metadata_is_leakage_safe": True,
        "accepted_csv_path": str(form.resolve()),
        "content_sha256": sha256_bytes(form.read_bytes()),
        "raw_export_path": str(raw_path.resolve()),
        "raw_content_sha256": sha256_bytes(raw_bytes),
        "raw_content_length": len(raw_bytes),
        "metadata_captured_at": NOW.isoformat(),
        "race_url": RACE_URL,
        "race_info": {"distance": "525m"},
        "canonical_runner_alignment": {
            "status": "aligned",
            "canonical_source_url": RACE_URL,
            "missing_canonical_participants": [],
            "dropped_participants": [],
        },
        "runner_completeness_after_canonical_alignment": {
            "status": "COMPLETE",
            "participants": [
                {"box_number": 1, "dog_name": "Bravo"},
                {"box_number": 2, "dog_name": "Alpha"},
            ],
        },
        "prejump_shadow_metadata": {
            "status": "PASS",
            "metadata_is_leakage_safe": True,
            "fail_reasons": [],
            "race_date": "2026-08-03",
            "race_number": "1",
            "venue": "WARRNAMBOOL",
            "distance": "525m",
            "source_url": RACE_URL,
            "metadata_captured_at": NOW.isoformat(),
            "runner_box_name_list": [
                {"box_number": 1, "dog_name": "Bravo"},
                {"box_number": 2, "dog_name": "Alpha"},
            ],
            "canonical_final_runner_alignment": {
                "status": "aligned",
                "canonical_runner_set_status": "available",
            },
        },
    }
    if mutate_sidecar is not None:
        mutate_sidecar(sidecar)
    sidecar_path.write_bytes(canonical_bytes(sidecar))

    plan_item = {
        "schema_version": "autonomous_live_odds_capture_plan_item_v1",
        "status": "READY_TO_CAPTURE",
        "csv_path": str(form.resolve()),
        "sidecar_path": str(sidecar_path.resolve()),
        "race_id": RACE_ID,
        "race_id_aliases": [],
        "venue": "WARRNAMBOOL",
        "race_number": 1,
        "race_date": "2026-08-03",
        "race_time": "12:20",
        "jump_datetime": JUMP.isoformat(),
        "minutes_to_jump": 20.0,
        "capture_window_minutes": 30,
        "window_status": "due_now_or_passed_pre_jump",
        "thedogs_source_url": RACE_URL,
        "runner_set_validation": {"status": "PASS", "expected_runners": RUNNERS},
        "expected_runners": [dict(row) for row in RUNNERS],
        "blockers": [],
    }
    if mutate_plan is not None:
        mutate_plan(plan_item)
    validation_rows = [
        {
            "dog_name": row["dog_name"],
            "dog_clean_name": row["dog_name"],
            "box_number": row["box_number"],
            "identity": row["identity"],
            "odds_decimal": 2.0 + row["box_number"],
            "sportsbet_box_source": "explicit_dom",
        }
        for row in plan_item.get("expected_runners", [])
        if set(row) == {"box_number", "dog_name", "identity"}
    ]
    attempt = {
        "schema_version": "autonomous_live_odds_capture_attempt_v1",
        "race_id": str(plan_item.get("race_id")),
        "status": "APPENDED",
        "capture_window_minutes": 30,
        "fetch_time": NOW.isoformat(),
        "append_time": (NOW + timedelta(seconds=2)).isoformat(),
        "reasons": [],
        "validation": {
            "schema_version": "autonomous_live_odds_capture_validation_v1",
            "status": "PASS",
            "source_url": "https://www.sportsbet.com.au/greyhounds/warrnambool/race-1",
            "accepted_rows": validation_rows,
            "accepted_place_rows": validation_rows,
            "reasons": [],
        },
        "append_report": {
            "status": "SUCCESS",
            "race_id": str(plan_item.get("race_id")),
            "inserted_rows": 2 * len(validation_rows),
            "append_only": True,
            "capture_timestamp": (NOW + timedelta(seconds=2)).isoformat(),
        },
    }
    receipt_publish = publish_scheduled_capture_receipts(
        protocol=protocol,
        evidence_root=evidence_root,
        collector_run_id="scheduled-run-1",
        plan_item=plan_item,
        attempt=attempt,
        output_dir=output_dir,
        emitted_at=NOW + timedelta(seconds=3),
    )
    return ScheduledFixture(
        protocol=protocol,
        evidence_root=evidence_root,
        corpus_root=corpus_root,
        plan_item=plan_item,
        attempt=attempt,
        receipt_publish=receipt_publish,
    )


def _admit(
    fixture: ScheduledFixture, *, emitted_at: datetime = NOW + timedelta(seconds=4)
):
    return admit_scheduled_capture(
        protocol=fixture.protocol,
        evidence_root=fixture.evidence_root,
        corpus_root=fixture.corpus_root,
        collector_run_id="scheduled-run-1",
        plan_item=fixture.plan_item,
        attempt=fixture.attempt,
        receipt_publish=fixture.receipt_publish,
        emitted_at=emitted_at,
    )


def test_fixture_scheduled_capture_admits_once_and_exact_replay_is_byte_stable(
    tmp_path,
):
    fixture = _fixture(tmp_path)

    admitted = _admit(fixture)
    first_tree = _tree(fixture.corpus_root)
    refreshed_plan = deepcopy(fixture.plan_item)
    refreshed_plan.update(
        {
            "csv_path": "/new-cycle/unread.csv",
            "sidecar_path": "/new-cycle/unread.csv.metadata.json",
            "minutes_to_jump": -1.0,
            "capture_window_minutes": 5,
            "window_status": "due_now_or_passed_pre_jump",
            "race_id_aliases": ["refreshed-alias"],
        }
    )
    replayed = admit_scheduled_capture(
        protocol=fixture.protocol,
        evidence_root=fixture.evidence_root,
        corpus_root=fixture.corpus_root,
        collector_run_id="scheduled-run-2",
        plan_item=refreshed_plan,
        emitted_at=JUMP + timedelta(seconds=1),
    )

    assert admitted["status"] == "PREJUMP_CAPTURED"
    assert replayed["status"] == "EXACT_REPLAY"
    assert _tree(fixture.corpus_root) == first_tree
    receipts = list((fixture.corpus_root / "races").glob("*/prejump.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_bytes())
    assert receipt["runner_ids"] == ["Alpha", "Bravo"]
    assert receipt["feature_schema_checksum"] == (
        "sha256:9c9ae3c10a800d98ed07dad298eb9b31dff10cfd1d073e0cb52879af6b8010e8"
    )
    assert receipt["missingness_policy_checksum"] == (
        "sha256:ae39177c5d1ed77eb7b40c09acb2d0ac92b2a258aa3beae1aae584dd8c08687f"
    )
    source_capture = json.loads(
        ForwardSealedCorpus(fixture.corpus_root)._read_artifact(
            receipt["source_capture_checksum"], "source capture"
        )
    )
    assert source_capture["canonical_source_url"] == RACE_URL.removesuffix(
        "?trial=false"
    )


def test_receipt_only_recovery_uses_sealed_plan_after_schedule_refresh(tmp_path):
    fixture = _fixture(tmp_path)
    refreshed_plan = deepcopy(fixture.plan_item)
    refreshed_plan.update(
        {
            "csv_path": "/new-cycle/unread.csv",
            "sidecar_path": "/new-cycle/unread.csv.metadata.json",
            "minutes_to_jump": 10.0,
            "capture_window_minutes": 15,
            "window_status": "due_now_or_passed_pre_jump",
            "race_id_aliases": ["refreshed-alias"],
        }
    )

    recovered = admit_scheduled_capture(
        protocol=fixture.protocol,
        evidence_root=fixture.evidence_root,
        corpus_root=fixture.corpus_root,
        collector_run_id="scheduled-run-2",
        plan_item=refreshed_plan,
        emitted_at=NOW + timedelta(minutes=10),
    )

    assert recovered["status"] == "PREJUMP_CAPTURED"
    assert len(list((fixture.corpus_root / "races").glob("*/prejump.json"))) == 1


def test_first_postjump_admission_rejects_without_writing_corpus_bytes(tmp_path):
    fixture = _fixture(tmp_path)
    before = _tree(fixture.corpus_root)

    with pytest.raises(ForwardCorpusRejected, match="prospectively"):
        _admit(fixture, emitted_at=JUMP)

    assert _tree(fixture.corpus_root) == before


@pytest.mark.parametrize(
    ("case", "mutate_plan", "mutate_sidecar", "mutate_raw", "match"),
    [
        (
            "identity",
            None,
            lambda value: value["prejump_shadow_metadata"].update(
                {"race_date": "2026-08-04"}
            ),
            None,
            "race identity",
        ),
        (
            "runner",
            None,
            None,
            lambda raw: raw.replace(b"2. Alpha", b"2. Charlie"),
            "runner identities",
        ),
        (
            "hash",
            None,
            lambda value: value.update({"raw_content_sha256": "f" * 64}),
            None,
            "raw source bytes disagree",
        ),
        (
            "missing-source",
            None,
            lambda value: value.update({"raw_export_path": "/missing/source.csv"}),
            None,
            "raw source path",
        ),
    ],
)
def test_disagreements_and_missing_production_inputs_fail_closed_without_corpus_write(
    tmp_path, case, mutate_plan, mutate_sidecar, mutate_raw, match
):
    del case
    fixture = _fixture(
        tmp_path,
        mutate_plan=mutate_plan,
        mutate_sidecar=mutate_sidecar,
        mutate_raw=mutate_raw,
    )
    before = _tree(fixture.corpus_root)

    with pytest.raises(ForwardCorpusRejected, match=match):
        _admit(fixture)

    assert _tree(fixture.corpus_root) == before


def test_missing_production_feature_is_rejected_before_receipt_publication(tmp_path):
    with pytest.raises(ProtocolRejected, match="IDENTITY_INVALID"):
        _fixture(
            tmp_path,
            mutate_plan=lambda value: value["expected_runners"][0].pop("box_number"),
        )


def test_manual_prediction_plan_is_not_a_second_acquisition_path(tmp_path):
    fixture = _fixture(tmp_path)
    manual = {**fixture.plan_item, "schema_version": "manual_prediction_bundle_v1"}
    before = _tree(fixture.corpus_root)

    with pytest.raises(ForwardCorpusRejected, match="manual or unsupported"):
        admit_scheduled_capture(
            protocol=fixture.protocol,
            evidence_root=fixture.evidence_root,
            corpus_root=fixture.corpus_root,
            collector_run_id="scheduled-run-1",
            plan_item=manual,
            emitted_at=NOW + timedelta(seconds=4),
        )

    assert _tree(fixture.corpus_root) == before


def test_corpus_lock_contention_is_no_steal_and_no_write(tmp_path):
    fixture = _fixture(tmp_path)
    lock = fixture.corpus_root / "forward-sealed-corpus.lock"
    lock.write_text('{"run_id":"other-producer","pid":1}', encoding="utf-8")
    before = _tree(fixture.corpus_root)

    with pytest.raises(CollectorBusy):
        _admit(fixture)

    assert lock.exists()
    assert _tree(fixture.corpus_root) == before
