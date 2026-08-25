from __future__ import annotations

import copy
import json
import ast
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import src.predictor.on_demand as sealed
from src.predictor.on_demand import PredictionBlocked, canonical_bytes, sha256_bytes
from src.operator_ui.job_store import Job, JobInput, OperationalIndexProvenance, Phase
from src.operator_ui.r3_api import _verified_result


PREDICTION_ID = "12345678-1234-4123-8123-123456789abc"
GENERATED = datetime.fromisoformat("2026-07-19T12:00:00+10:00")
DIRECTORY = "prediction_20260719T120000000000+1000_0123456789ab"
ZERO_SHA = "0" * 64


def ready_result(*, job_id: str | None = None) -> dict[str, Any]:
    race = {
        "race_id": "Race 5 - GUNN - 2026-07-19",
        "url": "https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5",
        "race_date": "2026-07-19",
        "venue": "GUNN",
        "venue_slug": "gunnedah",
        "race_number": 5,
        "jump_timestamp": "2026-07-19T13:00:00+10:00",
    }
    runners = [
        {"box_number": 1, "display_name": "ONE", "identity": "ONE", "source_native_runner_id": "101"},
        {"box_number": 2, "display_name": "TWO", "identity": "TWO", "source_native_runner_id": None},
    ]
    predictions = [
        {"rank": 1, "box_number": 1, "dog_name": "ONE", "identity": "ONE", "source_native_runner_id": "101", "probability": 0.6},
        {"rank": 2, "box_number": 2, "dog_name": "TWO", "identity": "TWO", "source_native_runner_id": None, "probability": 0.4},
    ]
    return {
        "schema_version": sealed.PREDICTION_RESULT_SCHEMA_V2,
        "prediction_id": PREDICTION_ID,
        "job_id": job_id,
        "generated_at": GENERATED.isoformat(),
        "status": "PREDICTION_READY",
        "blocker_stage": None,
        "blocker": None,
        "research_only": True,
        "production_persisted": False,
        "betting_output": False,
        "race": race,
        "model": {
            "requested": "market-only",
            "resolved": "market_only_v1",
            "alias_resolved": True,
            "schema_sha256": ZERO_SHA,
            "artifact_identity": "UNAVAILABLE_NOT_APPLICABLE",
            "artifact_sha256": None,
            "artifact_manifest_identity": "UNAVAILABLE_NOT_APPLICABLE",
            "artifact_manifest_sha256": None,
        },
        "config": {"sha256": ZERO_SHA},
        "evidence": {
            "request": "request.json",
            "config": "config.json",
            "model_schema": "model/config.schema.json",
            "model_artifact": None,
            "model_manifest": None,
            "runner_set_sha256": sealed.sealed_runner_set_sha256(race, runners),
            "prediction_output_sha256": sha256_bytes(canonical_bytes(predictions)),
            "protocol_chain": {"request_id":"request-1","request_sha256":ZERO_SHA,"claim_sha256":ZERO_SHA,"attempt_sha256":ZERO_SHA,"response_sha256":ZERO_SHA,"receipt_sha256":ZERO_SHA,"consume_sha256":ZERO_SHA,"authenticated_receipt_sha256":ZERO_SHA},
            "authenticated_cutoff": {"history_seal_sha256":ZERO_SHA,"cutoff_timestamp":"2026-07-19T13:00:00+10:00","source_sha256":ZERO_SHA,"sealed_sha256":ZERO_SHA},
        },
        "prediction": {
            "predictions": predictions
        },
    }


def blocked_result(stage: str = "VALIDATION") -> dict[str, Any]:
    value = ready_result()
    value.update(
        status="PREDICTION_BLOCKED",
        blocker_stage=stage,
        blocker={"code": "POST_JUMP"},
        prediction=None,
    )
    value["evidence"]["prediction_output_sha256"] = None
    return value


def request_for(result: dict[str, Any]) -> dict[str, Any]:
    model = result["model"]
    runners = [
        {"box_number": row["box_number"], "display_name": row["dog_name"], "identity": row["identity"], "source_native_runner_id": row["source_native_runner_id"]}
        for row in (result.get("prediction") or {"predictions": ready_result()["prediction"]["predictions"]})["predictions"]
    ]
    runners.sort(key=lambda row: (row["box_number"], row["identity"]))
    return {
        "schema_version": "on_demand_prediction_request_v1",
        "prediction_id": result["prediction_id"],
        "job_id": result["job_id"],
        "race_query": result["race"]["race_id"],
        "race_id": result["race"]["race_id"],
        "jump_timestamp": result["race"]["jump_timestamp"],
        "request_timestamp": "2026-07-19T11:59:00+10:00",
        "odds_source": "receipt",
        "model": {
            "requested": model["requested"],
            "resolved": model["resolved"],
            "alias_resolved": model["alias_resolved"],
            "model_sha256": model["artifact_sha256"],
            "manifest_sha256": model["artifact_manifest_sha256"],
            "schema_sha256": model["schema_sha256"],
        },
        "config_sha256": result["config"]["sha256"],
        "research_only": True,
        "runners": runners,
        "runner_set_sha256": sealed.sealed_runner_set_sha256(result["race"], runners),
    }


def make_bundle(root: Path, result: dict[str, Any] | None = None, *, operational_provenance: dict[str,str] | None = None) -> tuple[Path, dict[str, Any]]:
    result = copy.deepcopy(result or ready_result())
    bundle = root / DIRECTORY
    (bundle / "model").mkdir(parents=True)
    prediction_request=request_for(result)
    if operational_provenance is not None:
        prediction_request["schema_version"]="on_demand_prediction_request_v2"
        prediction_request["operational_index_provenance"]=operational_provenance
    files = {
        "result.json": canonical_bytes(result),
        "request.json": canonical_bytes(prediction_request),
        "config.json": b"{}\n",
        "model/config.schema.json": b"{}\n",
    }
    protocol_race={key:result["race"][key] for key in ("race_id","url","venue","race_number","race_date","jump_timestamp")}
    protocol_runners=[{"box_number":row["box_number"],"dog_name":row["dog_name"],"identity":row["identity"]} for row in (result.get("prediction") or {"predictions":ready_result()["prediction"]["predictions"]})["predictions"]]
    protocol_request={"schema_version":"manual-prediction-collector-request-v1","request_id":"request-1","created_at":"2026-07-19T11:55:00+10:00","expires_at":"2026-07-19T12:30:00+10:00","race":protocol_race,"expected_runners":protocol_runners,"expected_runner_set_sha256":result["evidence"]["runner_set_sha256"],"requested_output":"normalized_odds_receipt","research_only":True,"attempt_authority":"one_attempt"}
    protocol={"request":protocol_request}
    protocol["claim"]={"schema_version":"manual-prediction-collector-claim-v1","request_id":"request-1","request_sha256":sha256_bytes(canonical_bytes(protocol["request"])),"collector_run_id":"collector-1","claimed_at":"2026-07-19T12:00:00+10:00","safe_boundary":True}
    protocol["attempt"]={"schema_version":"manual-prediction-collector-attempt-v1","request_id":"request-1","request_sha256":protocol["claim"]["request_sha256"],"claim_sha256":sha256_bytes(canonical_bytes(protocol["claim"])),"collector_run_id":"collector-1","attempt_number":1,"started_at":"2026-07-19T12:01:00+10:00"}
    sealed_handoff={"schema_version":"on_demand_verified_collector_capture_v2","race_id":result["race"]["race_id"],"race":protocol_race,"runner_set_sha256":result["evidence"]["runner_set_sha256"],"append_timestamp":"2026-07-19T12:02:00+10:00","source_report_sha256":"2"*64,"source_form_sha256":"3"*64,"source_sidecar_sha256":"4"*64,"capture_attempt_sha256":"5"*64,"append_report_sha256":"6"*64}
    source_evidence={"source_url":"https://example.invalid/source","source_report_sha256":"2"*64,"source_form_sha256":"3"*64,"source_sidecar_sha256":"4"*64,"capture_attempt_sha256":"5"*64,"append_report_sha256":"6"*64}
    receipt={"schema_version":"manual-prediction-collector-receipt-v1","request_id":"request-1","request_sha256":protocol["claim"]["request_sha256"],"race":protocol_race,"runners":protocol_runners,"runner_set_sha256":result["evidence"]["runner_set_sha256"],"captured_at":"2026-07-19T12:02:00+10:00","emitted_at":"2026-07-19T12:03:00+10:00","source_evidence":source_evidence,"sealed_handoff":sealed_handoff}
    receipt_sha=sha256_bytes(canonical_bytes(receipt))
    receipt_reference={"schema_version":"manual-prediction-collector-receipt-v1","path":"receipts/request-1.json","sha256":receipt_sha}
    protocol["response"]={"schema_version":"manual-prediction-collector-response-v1","request_id":"request-1","request_sha256":protocol["claim"]["request_sha256"],"claim_sha256":protocol["attempt"]["claim_sha256"],"attempt_sha256":sha256_bytes(canonical_bytes(protocol["attempt"])),"race":protocol_race,"status":"RECEIPT_READY","reason":None,"responded_at":"2026-07-19T12:03:00+10:00","receipt":receipt_reference}
    protocol["receipt"]=receipt
    protocol["consume"]={"schema_version":"manual-prediction-collector-consume-v1","request_id":"request-1","response_sha256":sha256_bytes(canonical_bytes(protocol["response"])),"status":"RECEIPT_READY","consumed_at":"2026-07-19T12:04:00+10:00","consume_once":True}
    artifacts={"report":{"path":"capture/report.json","sha256":"2"*64},"form":{"path":"capture/form.csv","sha256":"3"*64},"sidecar":{"path":"capture/sidecar.json","sha256":"4"*64}}
    protocol["authenticated_receipt"]={"schema_version":"manual-prediction-exact-receipt-index-v1","request_id":"request-1","race_id":result["race"]["race_id"],"receipt":receipt_reference,"artifacts":artifacts,"form_name":"form.csv"}
    for name,value in protocol.items():files[f"protocol/{name}.json"]=canonical_bytes(value)
    sealed_db=b"sealed fixture database"
    history={"schema_version":"sealed_prediction_history_v1","cutoff_timestamp":result["race"]["jump_timestamp"],"source_sha256":"1"*64,"sealed_sha256":sha256_bytes(sealed_db),"target_race_id":result["race"]["race_id"],"cutoff_basis":"race_date_strictly_before_target_jump_date","safe_race_count":0,"safe_dog_row_count":0,"excluded_target_metadata_rows":0,"excluded_at_or_after_cutoff_metadata_rows":0,"excluded_ambiguous_date_metadata_rows":0,"target_rows_materialized":0,"at_or_after_cutoff_rows_materialized":0}
    files["features/sealed_history.db"]=sealed_db;files["features/history_seal.json"]=canonical_bytes(history)
    result["evidence"]["protocol_chain"]={"request_id":"request-1",**{f"{name}_sha256":sha256_bytes(files[f"protocol/{name}.json"]) for name in ("request","claim","attempt","response","receipt","consume")},"authenticated_receipt_sha256":sha256_bytes(files["protocol/authenticated_receipt.json"])}
    result["evidence"]["authenticated_cutoff"]={"history_seal_sha256":sha256_bytes(files["features/history_seal.json"]),"cutoff_timestamp":result["race"]["jump_timestamp"],"source_sha256":history["source_sha256"],"sealed_sha256":history["sealed_sha256"]}
    result["config"]["sha256"] = sha256_bytes(files["config.json"])
    result["model"]["schema_sha256"] = sha256_bytes(files["model/config.schema.json"])
    if result["model"]["resolved"] != "market_only_v1":
        files["model/model.json"] = b'{"model":"fixture"}\n'
        files["model/manifest.json"] = b'{"manifest":"fixture"}\n'
        result["model"]["artifact_sha256"] = sha256_bytes(files["model/model.json"])
        result["model"]["artifact_manifest_sha256"] = sha256_bytes(files["model/manifest.json"])
    files["result.json"] = canonical_bytes(result)
    prediction_request=request_for(result)
    if operational_provenance is not None:
        prediction_request["schema_version"]="on_demand_prediction_request_v2"
        prediction_request["operational_index_provenance"]=operational_provenance
    files["request.json"] = canonical_bytes(prediction_request)
    for name, raw in files.items():
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest = sealed.build_prediction_bundle_manifest_v2(
        bundle, prediction_id=PREDICTION_ID, job_id=result["job_id"]
    )
    manifest_raw = canonical_bytes(manifest)
    (bundle / "bundle_manifest.json").write_bytes(manifest_raw)
    entry = sealed.prediction_bundle_index_entry(
        bundle=bundle, result=result, manifest_raw=manifest_raw
    )
    return bundle, entry


def operator_result(job_id: str) -> dict[str, Any]:
    value=ready_result(job_id=job_id); value["model"].update(requested="latest-research",resolved="market_form_residual_v1",alias_resolved=True,artifact_identity="AVAILABLE",artifact_manifest_identity="AVAILABLE",artifact_sha256=ZERO_SHA,artifact_manifest_sha256=ZERO_SHA)
    value["evidence"].update(model_artifact="model/model.json",model_manifest="model/manifest.json")
    return value


def operator_provenance(**changes) -> OperationalIndexProvenance:
    values={"schema":"operator_ui_operational_index_admission_v1","index_schema_version":"collector_current_race_index_v2","run_id":"collector-run","packet_sha256":ZERO_SHA,"source_refresh_sha256":ZERO_SHA,"publication_sha256":ZERO_SHA,"state_sha256":ZERO_SHA,"report_sha256":ZERO_SHA}
    values.update(changes); return OperationalIndexProvenance(**values)


def assert_blocked(callable_: Any, *args: Any, **kwargs: Any) -> PredictionBlocked:
    with pytest.raises(PredictionBlocked) as captured:
        callable_(*args, **kwargs)
    return captured.value


def _attacker_resealed_protocol(
    bundle: Path,
    *,
    protocol_path: str | None = None,
    value: Any = None,
    result_path: str | None = None,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Mutate one semantic relation, then authentically reseal every outer byte."""
    names = ("request", "claim", "attempt", "response", "receipt", "consume", "authenticated_receipt")
    values = {name: json.loads((bundle / f"protocol/{name}.json").read_bytes()) for name in names}
    result = json.loads((bundle / "result.json").read_bytes())

    def assign(root: dict[str, Any], dotted: str, replacement: Any) -> None:
        parts = dotted.split(".")
        target = root
        for part in parts[:-1]:
            target = target[int(part)] if isinstance(target, list) else target[part]
        if replacement is _DELETE:
            del target[parts[-1]]
        elif replacement is _ADD:
            target["attacker_extra"] = "resealed"
        else:
            if isinstance(target, list):
                target[int(parts[-1])] = replacement
            else:
                target[parts[-1]] = replacement

    if protocol_path is not None:
        owner, dotted = protocol_path.split(".", 1)
        assign(values[owner], dotted, value)
    if result_path is not None:
        assign(result, result_path, value)

    protected = protocol_path
    raws: dict[str, bytes] = {}
    raws["request"] = canonical_bytes(values["request"])
    request_sha = sha256_bytes(raws["request"])
    for owner in ("claim", "attempt", "response", "receipt"):
        if protected != f"{owner}.request_sha256":
            values[owner]["request_sha256"] = request_sha
    raws["claim"] = canonical_bytes(values["claim"])
    claim_sha = sha256_bytes(raws["claim"])
    for owner in ("attempt", "response"):
        if protected != f"{owner}.claim_sha256":
            values[owner]["claim_sha256"] = claim_sha
    raws["attempt"] = canonical_bytes(values["attempt"])
    attempt_sha = sha256_bytes(raws["attempt"])
    if protected != "response.attempt_sha256":
        values["response"]["attempt_sha256"] = attempt_sha
    raws["receipt"] = canonical_bytes(values["receipt"])
    receipt_sha = sha256_bytes(raws["receipt"])
    reference = {"schema_version": "manual-prediction-collector-receipt-v1", "path": "receipts/request-1.json", "sha256": receipt_sha}
    if protected is None or not protected.startswith("response.receipt"):
        values["response"]["receipt"] = copy.deepcopy(reference)
    if protected is None or not protected.startswith("authenticated_receipt.receipt"):
        values["authenticated_receipt"]["receipt"] = copy.deepcopy(reference)
    raws["response"] = canonical_bytes(values["response"])
    response_sha = sha256_bytes(raws["response"])
    if protected != "consume.response_sha256":
        values["consume"]["response_sha256"] = response_sha
    for name in ("consume", "authenticated_receipt"):
        raws[name] = canonical_bytes(values[name])
    contents = {path.relative_to(bundle).as_posix(): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    contents.update({f"protocol/{name}.json": raw for name, raw in raws.items()})
    result["evidence"]["protocol_chain"] = {
        "request_id": result["evidence"]["protocol_chain"]["request_id"],
        **{f"{name}_sha256": sha256_bytes(raws[name]) for name in ("request", "claim", "attempt", "response", "receipt", "consume")},
        "authenticated_receipt_sha256": sha256_bytes(raws["authenticated_receipt"]),
    }
    if result_path == "evidence.protocol_chain.request_id":
        result["evidence"]["protocol_chain"]["request_id"] = value
    return contents, result


_DELETE = object()
_ADD = object()


SCHEMA_AND_MEMBERSHIP_ATTACKS = [
    (f"{name}.schema_version", "attacker-v1")
    for name in ("request", "claim", "attempt", "response", "receipt", "consume", "authenticated_receipt")
] + [
    (f"{name}.schema_version", _ADD)
    for name in ("request", "claim", "attempt", "response", "receipt", "consume", "authenticated_receipt")
]


RELATION_ATTACKS = [
    ("request.race.race_id", "wrong-race"), ("request.expected_runners.0.identity", "WRONG"),
    ("request.expected_runner_set_sha256", "f" * 64), ("request.research_only", False),
    ("request.attempt_authority", "retry"), ("claim.request_id", "wrong"),
    ("attempt.request_id", "wrong"), ("response.request_id", "wrong"),
    ("receipt.request_id", "wrong"), ("consume.request_id", "wrong"),
    ("authenticated_receipt.request_id", "wrong"), ("claim.request_sha256", "f" * 64),
    ("attempt.request_sha256", "f" * 64), ("response.request_sha256", "f" * 64),
    ("receipt.request_sha256", "f" * 64), ("attempt.claim_sha256", "f" * 64),
    ("response.claim_sha256", "f" * 64), ("response.attempt_sha256", "f" * 64),
    ("consume.response_sha256", "f" * 64), ("claim.safe_boundary", False),
    ("attempt.collector_run_id", "wrong-run"), ("attempt.attempt_number", 2),
    ("request.created_at", "2026-07-19T12:00:01+10:00"),
    ("claim.claimed_at", "2026-07-19T12:01:01+10:00"),
    ("attempt.started_at", "2026-07-19T12:30:00+10:00"),
    ("receipt.captured_at", "2026-07-19T12:00:59+10:00"),
    ("receipt.emitted_at", "2026-07-19T12:03:01+10:00"),
    ("response.responded_at", "2026-07-19T12:02:59+10:00"),
    ("consume.consumed_at", "2026-07-19T12:30:00+10:00"),
    ("request.expires_at", "2026-07-19T12:04:00+10:00"),
    ("response.status", "FAILED"), ("response.reason", "attacker"),
    ("response.receipt.schema_version", "wrong"), ("response.receipt.path", "receipts/wrong.json"),
    ("response.receipt.sha256", "f" * 64), ("receipt.race.race_id", "wrong-race"),
    ("receipt.runners.0.identity", "WRONG"), ("receipt.runner_set_sha256", "f" * 64),
    ("receipt.sealed_handoff.race.race_id", "wrong-race"),
    ("receipt.sealed_handoff.race_id", "wrong-race"),
    ("receipt.sealed_handoff.runner_set_sha256", "f" * 64),
] + [
    (f"receipt.sealed_handoff.{field}", "f" * 64)
    for field in ("source_report_sha256", "source_form_sha256", "source_sidecar_sha256", "capture_attempt_sha256", "append_report_sha256")
] + [
    ("authenticated_receipt.race_id", "wrong-race"),
    ("authenticated_receipt.form_name", "wrong.csv"),
    ("authenticated_receipt.artifacts.report", _DELETE),
    ("authenticated_receipt.artifacts.report.attacker_extra", _ADD),
    ("authenticated_receipt.artifacts.report.path", "../report.json"),
    ("authenticated_receipt.artifacts.report.path", "evidence/./report.json"),
    ("authenticated_receipt.artifacts.report.path", ""),
    ("authenticated_receipt.artifacts.report.path", "evidence/report\n.json"),
    ("authenticated_receipt.artifacts.form.path", "capture/not-form.csv"),
    ("authenticated_receipt.artifacts.sidecar.path", "/tmp/sidecar.json"),
    ("authenticated_receipt.artifacts.report.sha256", "f" * 64),
    ("authenticated_receipt.artifacts.form.sha256", "f" * 64),
    ("authenticated_receipt.artifacts.sidecar.sha256", "f" * 64),
]


@pytest.mark.parametrize(("path", "value"), SCHEMA_AND_MEMBERSHIP_ATTACKS + RELATION_ATTACKS)
def test_each_sealed_protocol_relation_blocks_after_attacker_reseals_downstream_bytes(tmp_path: Path, path: str, value: Any):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle, protocol_path=path, value=value)
    assert_blocked(sealed._validate_sealed_protocol, contents, result)


def test_authenticated_receipt_accepts_real_evidence_root_relative_artifact_paths(tmp_path: Path):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle)
    exact = json.loads(contents["protocol/authenticated_receipt.json"])
    exact["artifacts"]["report"]["path"] = "shadow_runs/2026-07-19/request-1/report.json"
    exact["artifacts"]["form"]["path"] = "collector/forms/request-1/thedogs-form.csv"
    exact["artifacts"]["sidecar"]["path"] = "collector/sidecars/request-1/source.json"
    exact["form_name"] = "thedogs-form.csv"
    raw = canonical_bytes(exact)
    contents["protocol/authenticated_receipt.json"] = raw
    result["evidence"]["protocol_chain"]["authenticated_receipt_sha256"] = sha256_bytes(raw)
    sealed._validate_sealed_protocol(contents, result)


def test_authenticated_receipt_requires_distinct_artifact_paths(tmp_path: Path):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle)
    exact = json.loads(contents["protocol/authenticated_receipt.json"])
    exact["artifacts"]["sidecar"]["path"] = exact["artifacts"]["report"]["path"]
    raw = canonical_bytes(exact)
    contents["protocol/authenticated_receipt.json"] = raw
    result["evidence"]["protocol_chain"]["authenticated_receipt_sha256"] = sha256_bytes(raw)
    assert_blocked(sealed._validate_sealed_protocol, contents, result)


def test_authentic_history_seal_is_accepted_by_sealed_protocol_verifier(tmp_path: Path):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle)
    source = tmp_path / "authentic-source.db"
    connection = sqlite3.connect(source)
    connection.execute(
        "CREATE TABLE race_metadata (race_id TEXT, race_date TEXT)"
    )
    connection.execute(
        "CREATE TABLE dog_race_data (race_id TEXT, dog_name TEXT)"
    )
    connection.execute(
        "INSERT INTO race_metadata VALUES (?, ?)", ("past", "2026-07-18")
    )
    connection.execute(
        "INSERT INTO dog_race_data VALUES (?, ?)", ("past", "ONE")
    )
    connection.commit()
    connection.close()
    sealed_db = tmp_path / "authentic-sealed.db"
    history = sealed.seal_history_database(
        source=source,
        target=sealed_db,
        target_race_id=result["race"]["race_id"],
        cutoff=datetime.fromisoformat(result["race"]["jump_timestamp"]),
        runner_names=["ONE", "TWO"],
    )
    contents["features/sealed_history.db"] = sealed_db.read_bytes()
    contents["features/history_seal.json"] = canonical_bytes(history)
    result["evidence"]["authenticated_cutoff"] = {
        "history_seal_sha256": sha256_bytes(contents["features/history_seal.json"]),
        "cutoff_timestamp": history["cutoff_timestamp"],
        "source_sha256": history["source_sha256"],
        "sealed_sha256": history["sealed_sha256"],
    }

    sealed._validate_sealed_protocol(contents, result)


@pytest.mark.parametrize(("path", "value"), [
    ("schema_version", "wrong-v1"), ("schema_version", _ADD),
    ("cutoff_timestamp", "2026-07-19T12:59:59+10:00"), ("source_sha256", "f" * 64),
    ("sealed_sha256", "f" * 64), ("target_race_id", "wrong-race"),
    ("cutoff_basis", "target_jump_timestamp"),
    ("target_rows_materialized", 1), ("at_or_after_cutoff_rows_materialized", 1),
    *((field, value) for field in (
        "safe_race_count", "safe_dog_row_count", "excluded_target_metadata_rows",
        "excluded_at_or_after_cutoff_metadata_rows", "excluded_ambiguous_date_metadata_rows",
        "target_rows_materialized", "at_or_after_cutoff_rows_materialized",
    ) for value in (-1, True, 1.5, "0")),
    *((field, _DELETE) for field in (
        "schema_version", "cutoff_timestamp", "source_sha256", "sealed_sha256",
        "target_race_id", "cutoff_basis", "safe_race_count", "safe_dog_row_count",
        "excluded_target_metadata_rows", "excluded_at_or_after_cutoff_metadata_rows",
        "excluded_ambiguous_date_metadata_rows", "target_rows_materialized",
        "at_or_after_cutoff_rows_materialized",
    )),
])
def test_each_authenticated_cutoff_relation_blocks_after_attacker_reseals_history_digest(tmp_path: Path, path: str, value: Any):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle)
    history = json.loads(contents["features/history_seal.json"])
    if value is _ADD:
        history["attacker_extra"] = "resealed"
    elif value is _DELETE:
        del history[path]
    else:
        history[path] = value
    contents["features/history_seal.json"] = canonical_bytes(history)
    result["evidence"]["authenticated_cutoff"]["history_seal_sha256"] = sha256_bytes(contents["features/history_seal.json"])
    assert_blocked(sealed._validate_sealed_protocol, contents, result)


@pytest.mark.parametrize(("path", "value"), [
    ("evidence.authenticated_cutoff.cutoff_timestamp", "2026-07-19T12:59:59+10:00"),
    ("evidence.authenticated_cutoff.source_sha256", "f" * 64),
    ("evidence.authenticated_cutoff.sealed_sha256", "f" * 64),
    ("evidence.authenticated_cutoff.history_seal_sha256", "f" * 64),
    ("evidence.protocol_chain.request_id", "wrong-request"),
])
def test_result_sealed_authority_relations_are_not_only_syntax_checked(tmp_path: Path, path: str, value: Any):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle, result_path=path, value=value)
    assert_blocked(sealed._validate_sealed_protocol, contents, result)


def test_authenticated_cutoff_rejects_wrong_sealed_database_digest(tmp_path: Path):
    bundle, _ = make_bundle(tmp_path)
    contents, result = _attacker_resealed_protocol(bundle)
    contents["features/sealed_history.db"] = b"attacker replacement database"
    assert_blocked(sealed._validate_sealed_protocol, contents, result)


def test_publish_verify_index_and_detail_positive(tmp_path: Path):
    bundle, entry = make_bundle(tmp_path)
    index = sealed.publish_prediction_bundle_index_entry(tmp_path, entry)

    assert sealed.verify_prediction_bundle_index(tmp_path) == index
    verified = sealed.verify_indexed_prediction_bundle(tmp_path, entry)
    assert verified.directory == bundle.name
    assert verified.request == request_for(verified.result)
    assert verified.result["prediction"]["predictions"][0]["probability"] == 0.6
    assert not (tmp_path / sealed.PREDICTION_BUNDLE_LOCK_NAME).exists()
    view = sealed.verify_prediction_bundle_index(
        tmp_path, return_verified_view=True
    )
    assert isinstance(view, sealed.VerifiedPredictionBundleIndex)
    assert view.sha256 == sha256_bytes(view.canonical_bytes)
    assert datetime.fromisoformat(view.published_at).tzinfo is not None


def test_request_schema_downgrade_blocks_after_attacker_reseals_entire_chain(tmp_path: Path):
    bundle,_entry=make_bundle(tmp_path)
    result=json.loads((bundle/"result.json").read_bytes())
    values={name:json.loads((bundle/f"protocol/{name}.json").read_bytes()) for name in ("request","claim","attempt","response","receipt","consume","authenticated_receipt")}
    values["request"]["schema_version"]="synthetic-fallback-v1"
    raws={"request":canonical_bytes(values["request"])}
    request_sha=sha256_bytes(raws["request"])
    values["claim"]["request_sha256"]=request_sha;raws["claim"]=canonical_bytes(values["claim"]);claim_sha=sha256_bytes(raws["claim"])
    values["attempt"].update(request_sha256=request_sha,claim_sha256=claim_sha);raws["attempt"]=canonical_bytes(values["attempt"]);attempt_sha=sha256_bytes(raws["attempt"])
    values["receipt"]["request_sha256"]=request_sha;raws["receipt"]=canonical_bytes(values["receipt"]);receipt_sha=sha256_bytes(raws["receipt"])
    reference={"schema_version":"manual-prediction-collector-receipt-v1","path":"receipts/request-1.json","sha256":receipt_sha}
    values["response"].update(request_sha256=request_sha,claim_sha256=claim_sha,attempt_sha256=attempt_sha,receipt=reference);raws["response"]=canonical_bytes(values["response"]);response_sha=sha256_bytes(raws["response"])
    values["consume"]["response_sha256"]=response_sha;raws["consume"]=canonical_bytes(values["consume"])
    values["authenticated_receipt"]["receipt"]=reference;raws["authenticated_receipt"]=canonical_bytes(values["authenticated_receipt"])
    contents={path.relative_to(bundle).as_posix():path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    contents.update({f"protocol/{name}.json":raw for name,raw in raws.items()})
    result["evidence"]["protocol_chain"]={"request_id":"request-1",**{f"{name}_sha256":sha256_bytes(raws[name]) for name in ("request","claim","attempt","response","receipt","consume")},"authenticated_receipt_sha256":sha256_bytes(raws["authenticated_receipt"])}
    assert_blocked(sealed._validate_sealed_protocol,contents,result)


def test_operator_disclosure_binds_authenticated_request_odds_and_runner_projection(tmp_path: Path):
    job_id="job_"+"1"*32; result=operator_result(job_id); provenance=operator_provenance(); _bundle,entry=make_bundle(tmp_path,result,operational_provenance=provenance.fields())
    sealed.publish_prediction_bundle_index_entry(tmp_path,entry)
    verified=sealed.verify_indexed_prediction_bundle(tmp_path,entry)
    model=verified.result["model"]; request=verified.request; assert request is not None
    ordered=tuple({"box":row["box_number"],"name":row["display_name"],"identity":row["identity"],**({"source_native_runner_id":row["source_native_runner_id"]} if row["source_native_runner_id"] is not None else {})} for row in request["runners"])
    def job(odds):
        inp=JobInput(verified.result["race"]["race_id"],verified.result["race"]["jump_timestamp"],verified.result["evidence"]["runner_set_sha256"],"latest-research",model["resolved"],model["artifact_sha256"],model["artifact_manifest_sha256"],model["schema_sha256"],"manual-default",verified.result["config"]["sha256"],odds,ordered,provenance)
        return Job(job_id,"operator",2,"manual_prediction","0"*64,inp,"2026-07-19T01:00:00Z",Phase.PREDICTION_READY,"2026-07-19T02:00:00Z","READY","verified",None,None,True)
    chain=verified.result["evidence"]["protocol_chain"]; cutoff=verified.result["evidence"]["authenticated_cutoff"]
    events=[{"phase":"CLAIMED","facts":{"attempt_id":"attempt-1"}},{"phase":"RESPONSE_RECORDED","facts":{"attempt_id":"attempt-1","protocol_chain":chain,"authenticated_cutoff":cutoff}},{"phase":"PRODUCER_COMPLETED","facts":{"attempt_id":"attempt-1","protocol_chain":chain,"authenticated_cutoff":cutoff}}]
    assert _verified_result(job("auto"),verified,events) is None
    disclosed=_verified_result(job("receipt"),verified,events)
    assert disclosed is not None
    assert {(row["box"],row["runner_id"]) for row in disclosed["probabilities"]}=={(1,"ONE"),(2,"TWO")}


def test_operator_terminal_verifier_rejects_sealed_operational_index_provenance_divergence(tmp_path:Path):
    job_id="job_"+"1"*32; result=operator_result(job_id); admitted=operator_provenance()
    _bundle,entry=make_bundle(tmp_path,result,operational_provenance=operator_provenance(run_id="different-run").fields())
    sealed.publish_prediction_bundle_index_entry(tmp_path,entry); verified=sealed.verify_indexed_prediction_bundle(tmp_path,entry)
    model=verified.result["model"]; request=verified.request; assert request is not None
    ordered=tuple({"box":row["box_number"],"name":row["display_name"],"identity":row["identity"],**({"source_native_runner_id":row["source_native_runner_id"]} if row["source_native_runner_id"] is not None else {})} for row in request["runners"])
    inp=JobInput(verified.result["race"]["race_id"],verified.result["race"]["jump_timestamp"],verified.result["evidence"]["runner_set_sha256"],"latest-research",model["resolved"],model["artifact_sha256"],model["artifact_manifest_sha256"],model["schema_sha256"],"manual-default",verified.result["config"]["sha256"],"receipt",ordered,admitted)
    durable_job=Job(job_id,"operator",2,"manual_prediction","0"*64,inp,"2026-07-19T01:00:00Z",Phase.PREDICTION_READY,"2026-07-19T02:00:00Z","READY","verified",None,None,True)
    chain=verified.result["evidence"]["protocol_chain"]; cutoff=verified.result["evidence"]["authenticated_cutoff"]
    events=[{"phase":"CLAIMED","facts":{"attempt_id":"attempt-1"}},{"phase":"RESPONSE_RECORDED","facts":{"attempt_id":"attempt-1","protocol_chain":chain,"authenticated_cutoff":cutoff}},{"phase":"PRODUCER_COMPLETED","facts":{"attempt_id":"attempt-1","protocol_chain":chain,"authenticated_cutoff":cutoff}}]
    assert _verified_result(durable_job,verified,events) is None


@pytest.mark.parametrize("identity",["request_sha256","claim_sha256","attempt_sha256","response_sha256","receipt_sha256","consume_sha256","authenticated_receipt_sha256","cutoff_timestamp"])
def test_operator_disclosure_rejects_each_mutated_protocol_or_cutoff_identity(tmp_path:Path,identity:str):
    job_id="job_"+"1"*32; result=operator_result(job_id); _bundle,entry=make_bundle(tmp_path,result)
    sealed.publish_prediction_bundle_index_entry(tmp_path,entry); verified=sealed.verify_indexed_prediction_bundle(tmp_path,entry)
    request=verified.request; model=verified.result["model"]; assert request is not None
    ordered=tuple({"box":row["box_number"],"name":row["display_name"],"identity":row["identity"],"source_native_runner_id":row["source_native_runner_id"]} for row in request["runners"])
    inp=JobInput(verified.result["race"]["race_id"],verified.result["race"]["jump_timestamp"],verified.result["evidence"]["runner_set_sha256"],"latest-research",model["resolved"],model["artifact_sha256"],model["artifact_manifest_sha256"],model["schema_sha256"],"manual-default",verified.result["config"]["sha256"],"receipt",ordered)
    durable_job=Job(job_id,"operator",2,"manual_prediction","0"*64,inp,"2026-07-19T01:00:00Z",Phase.PREDICTION_READY,"2026-07-19T02:00:00Z","READY","verified",None,None,True)
    chain=copy.deepcopy(verified.result["evidence"]["protocol_chain"]); cutoff=copy.deepcopy(verified.result["evidence"]["authenticated_cutoff"])
    if identity=="cutoff_timestamp":cutoff[identity]="2026-07-19T12:31:00+10:00"
    else:chain[identity]="f"*64
    events=[{"phase":"CLAIMED","facts":{"attempt_id":"attempt-1"}},{"phase":"RESPONSE_RECORDED","facts":{"attempt_id":"attempt-1","protocol_chain":chain,"authenticated_cutoff":cutoff}},{"phase":"PRODUCER_COMPLETED","facts":{"attempt_id":"attempt-1","protocol_chain":chain,"authenticated_cutoff":cutoff}}]
    assert _verified_result(durable_job,verified,events) is None


def test_empty_index_publication_has_producer_owned_aware_time(tmp_path: Path):
    published_at = GENERATED - timedelta(minutes=5)
    index = sealed.publish_prediction_bundle_index_entry(
        tmp_path, None, _clock=lambda: published_at
    )
    assert index == {
        "schema_version": sealed.PREDICTION_BUNDLE_INDEX_SCHEMA,
        "published_at": published_at.astimezone(timezone.utc).isoformat(),
        "entries": [],
    }
    view = sealed.verify_prediction_bundle_index(
        tmp_path, return_verified_view=True
    )
    assert view.published_at == published_at.astimezone(timezone.utc).isoformat()


def test_duplicate_and_repeated_empty_publication_are_idempotent(tmp_path: Path):
    calls = iter((GENERATED, GENERATED + timedelta(minutes=1)))
    empty = sealed.publish_prediction_bundle_index_entry(tmp_path, None, _clock=lambda: next(calls))
    assert sealed.publish_prediction_bundle_index_entry(tmp_path, None, _clock=lambda: next(calls)) == empty
    _, entry = make_bundle(tmp_path)
    first = sealed.publish_prediction_bundle_index_entry(tmp_path, entry, _clock=lambda: GENERATED + timedelta(minutes=2))
    duplicate = sealed.publish_prediction_bundle_index_entry(tmp_path, entry, _clock=lambda: GENERATED + timedelta(days=1))
    assert duplicate == first


def test_publication_clock_is_utc_aware_monotonic_and_legacy_migrates(tmp_path: Path):
    legacy = {"schema_version": sealed.PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": []}
    (tmp_path / sealed.PREDICTION_BUNDLE_INDEX_NAME).write_bytes(canonical_bytes(legacy))
    migrated = sealed.publish_prediction_bundle_index_entry(
        tmp_path, None, _clock=lambda: GENERATED.astimezone(timezone(timedelta(hours=10)))
    )
    assert datetime.fromisoformat(migrated["published_at"]).utcoffset() == timedelta(0)
    with pytest.raises(PredictionBlocked) as regression:
        _, entry = make_bundle(tmp_path)
        sealed.publish_prediction_bundle_index_entry(
            tmp_path, entry, _clock=lambda: GENERATED - timedelta(seconds=1)
        )
    assert regression.value.code == "PREDICTION_BUNDLE_PUBLICATION_TIME_REGRESSION"
    with pytest.raises(PredictionBlocked):
        sealed.publish_prediction_bundle_index_entry(tmp_path, entry, _clock=lambda: datetime(2026, 1, 1))


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda value: value.update(extra=True), "unknown"),
        (lambda value: value.update(generated_at="2026-07-19T12:00:00"), "time"),
        (lambda value: value.update(status="READY"), "status"),
        (lambda value: value.update(blocker_stage="PROTOCOL"), "stage"),
        (lambda value: value["prediction"]["predictions"][0].update(probability=float("nan")), "finite"),
        (lambda value: value["prediction"]["predictions"][0].update(probability=0.3), "order"),
        (lambda value: value["prediction"]["predictions"][1].update(probability=0.3), "sum"),
        (lambda value: value["prediction"]["predictions"][1].update(box_number=1), "box"),
        (lambda value: value["prediction"]["predictions"][1].update(dog_name="ONE"), "runner"),
    ],
)
def test_result_schema_rejects_unknown_time_status_and_probability_attacks(mutation: Any, field: str):
    value = ready_result()
    mutation(value)
    assert_blocked(sealed.validate_prediction_result_v2, value)


@pytest.mark.parametrize(("code", "stage"), list(sealed.BLOCKER_STAGE_BY_CODE.items()))
def test_blocked_terminal_code_stage_pairs_are_exact(code: str, stage: str):
    value = blocked_result(stage)
    value["blocker"] = {"code": code}
    assert sealed.validate_prediction_result_v2(value)["blocker_stage"] == stage
    for wrong in {"PROTOCOL", "VALIDATION", "SCORING"} - {stage}:
        attack = copy.deepcopy(value)
        attack["blocker_stage"] = wrong
        assert_blocked(sealed.validate_prediction_result_v2, attack)


def test_market_only_is_the_only_nullable_artifact_identity():
    valid = ready_result()
    sealed.validate_prediction_result_v2(valid)
    invalid = copy.deepcopy(valid)
    invalid["model"]["resolved"] = "market_form_residual_v1"
    assert_blocked(sealed.validate_prediction_result_v2, invalid)


@pytest.mark.parametrize(
    "job_id",
    ["x", "job-" + "1" * 32, "job_" + "A" * 32, "job_" + "1" * 31,
     "job_" + "1" * 33, "xjob_" + "1" * 32, "job_" + "1" * 32 + "x"],
)
def test_job_id_is_null_or_exact_future_operations_store_format(job_id: str):
    value = ready_result(job_id=job_id)
    assert_blocked(sealed.validate_prediction_result_v2, value)
    assert_blocked(
        sealed.validate_prediction_bundle_manifest_v2,
        {"schema_version": sealed.PREDICTION_MANIFEST_SCHEMA_V2, "prediction_id": PREDICTION_ID, "job_id": job_id, "files": {"x": {"bytes": 0, "sha256": ZERO_SHA}}},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda race: race.update(url="http://www.thedogs.com.au/racing/gunnedah/2026-07-19/5"),
        lambda race: race.update(url="https://thedogs.com.au/racing/gunnedah/2026-07-19/5"),
        lambda race: race.update(url="https://www.thedogs.com.au/RACING/gunnedah/2026-07-19/5"),
        lambda race: race.update(url="https://www.thedogs.com.au/racing/Gunnedah/2026-07-19/5"),
        lambda race: race.update(url="https://www.thedogs.com.au/racing/gunnedah/2026-07-19/05"),
        lambda race: race.update(url="https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5/"),
        lambda race: race.update(url="https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5?q=1"),
        lambda race: race.update(url="https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5#x"),
        lambda race: race.update(url="https://www.thedogs.com.au/racing/gunnedah%2Fother/2026-07-19/5"),
        lambda race: race.update(race_date="2026-07-20"),
        lambda race: race.update(venue_slug="sandown"),
        lambda race: race.update(race_number=0),
        lambda race: race.update(race_number=-1),
        lambda race: race.update(race_number=5.0),
        lambda race: race.update(race_id="Race 5 - OTHER - 2026-07-19"),
    ],
)
def test_race_identity_rejects_every_alias_mismatch_and_substitution(mutation: Any):
    value = ready_result()
    mutation(value["race"])
    assert_blocked(sealed.validate_prediction_result_v2, value)


def test_runner_set_and_output_hash_reject_all_named_substitutions(tmp_path: Path):
    attacks = []
    for mutate in (
        lambda rows: rows.pop(),
        lambda rows: rows.append(copy.deepcopy(rows[0])),
        lambda rows: rows.reverse(),
        lambda rows: rows[0].update(display_name="SUBSTITUTE"),
        lambda rows: rows[0].update(identity="SUBSTITUTE"),
        lambda rows: rows[0].update(source_native_runner_id="invented"),
        lambda rows: rows[1].update(box_number=1),
    ):
        request = request_for(ready_result())
        mutate(request["runners"])
        attacks.append(request)
    for offset, request in enumerate(attacks):
        root = tmp_path / str(offset)
        bundle, entry = make_bundle(root)
        (bundle / "request.json").write_bytes(canonical_bytes(request))
        manifest = sealed.build_prediction_bundle_manifest_v2(bundle, prediction_id=PREDICTION_ID, job_id=None)
        raw = canonical_bytes(manifest)
        (bundle / "bundle_manifest.json").write_bytes(raw)
        entry = sealed.prediction_bundle_index_entry(bundle=bundle, result=ready_result(), manifest_raw=raw)
        assert_blocked(sealed.verify_indexed_prediction_bundle, root, entry)
    value = ready_result()
    value["prediction"]["predictions"].reverse()
    assert_blocked(sealed.validate_prediction_result_v2, value)
    value = ready_result()
    value["evidence"]["prediction_output_sha256"] = ZERO_SHA
    assert_blocked(sealed.validate_prediction_result_v2, value)


def test_shared_blocker_allowlist_covers_literal_and_dynamic_terminal_codes_once():
    import scripts.predict_race_now as producer
    assert producer.BLOCKER_STAGE_BY_CODE is sealed.BLOCKER_STAGE_BY_CODE
    source = Path(producer.__file__).read_text()
    tree = ast.parse(source)
    excluded_replay_only = {"REPLAY_INPUT_INVALID", "REPLAY_NONDETERMINISTIC", "REPLAY_SCORER_FAILED", "REPLAY_TAMPERED", "UNSEALED_BLOCKER_CODE"}
    reachable = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "PredictionBlocked"
        and node.args and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    } - excluded_replay_only
    assert reachable <= sealed.BLOCKER_STAGE_BY_CODE.keys()
    assert sealed.CURRENT_INDEX_BLOCKER_STAGE_BY_CODE == {
        code: "PROTOCOL" for code in {
            "CURRENT_INDEX_INVALID", "CURRENT_INDEX_PATH_UNSAFE",
            "CURRENT_INDEX_PUBLICATION_INVALID", "CURRENT_INDEX_PUBLICATION_MISSING",
            "CURRENT_INDEX_REPORT_INVALID", "CURRENT_INDEX_REPORT_MISSING",
            "CURRENT_INDEX_SIZE_INVALID", "CURRENT_INDEX_SOURCE_CHANGED",
            "CURRENT_INDEX_SOURCE_INVALID", "CURRENT_INDEX_SOURCE_MISSING",
            "CURRENT_INDEX_STALE", "CURRENT_INDEX_UNAVAILABLE", "CURRENT_INDEX_UNBOUNDED",
        }
    }
    assert sealed.CURRENT_INDEX_BLOCKER_STAGE_BY_CODE.items() <= sealed.BLOCKER_STAGE_BY_CODE.items()
    assert set(sealed.BLOCKER_STAGE_BY_CODE.values()) == {"PROTOCOL", "VALIDATION", "SCORING"}


def index_entry(offset: int, prediction_id: str = PREDICTION_ID) -> dict[str, Any]:
    return {
        "directory": f"prediction_20260719T12000000000{offset}+1000_{offset:012x}",
        "prediction_id": prediction_id,
        "job_id": None,
        "generated_at": (GENERATED + timedelta(seconds=offset)).isoformat(),
        "status": "PREDICTION_READY",
        "blocker_stage": None,
        "manifest_sha256": ZERO_SHA,
        "logical_bundle_sha256": ZERO_SHA,
    }


def test_index_schema_rejects_order_duplicates_count_and_unknown_fields():
    first = index_entry(0)
    second = index_entry(1, "22345678-1234-4123-8123-123456789abc")
    base = {"schema_version": sealed.PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": [second, first]}
    sealed.validate_prediction_bundle_index_v1(base)
    with pytest.raises(PredictionBlocked):
        sealed.validate_prediction_bundle_index_v1(
            base, require_publication_time=True
        )
    sealed.validate_prediction_bundle_index_v1(
        {**base, "published_at": GENERATED.isoformat()},
        require_publication_time=True,
    )
    for invalid in (
        {**base, "extra": True},
        {**base, "entries": [first, second]},
        {**base, "entries": [first, first]},
        {**base, "entries": [index_entry(0, f"{number:08x}-1234-4123-8123-123456789abc") for number in range(257)]},
    ):
        assert_blocked(sealed.validate_prediction_bundle_index_v1, invalid)


def test_index_rejects_noncanonical_duplicate_json_oversize_and_deadline(tmp_path: Path):
    path = tmp_path / sealed.PREDICTION_BUNDLE_INDEX_NAME
    path.write_bytes(b'{"entries":[],"schema_version":"on_demand_prediction_bundle_index_v1"}\n ')
    assert_blocked(sealed.verify_prediction_bundle_index, tmp_path)
    path.write_bytes(b'{"entries":[],"entries":[],"schema_version":"on_demand_prediction_bundle_index_v1"}\n')
    assert_blocked(sealed.verify_prediction_bundle_index, tmp_path)
    path.write_bytes(b"x" * (sealed.INDEX_MAX_BYTES + 1))
    assert_blocked(sealed.verify_prediction_bundle_index, tmp_path)
    path.write_bytes(canonical_bytes({"schema_version": sealed.PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": []}))
    times = iter((0.0, 2.0))
    assert_blocked(sealed.verify_prediction_bundle_index, tmp_path, monotonic=lambda: next(times))


@pytest.mark.parametrize("name", ["../escape", "a/../escape", "/absolute", "a\\b", "a//b", "./a"])
def test_manifest_rejects_traversal_and_platform_names(name: str):
    manifest = {
        "schema_version": sealed.PREDICTION_MANIFEST_SCHEMA_V2,
        "prediction_id": PREDICTION_ID,
        "job_id": None,
        "files": {name: {"bytes": 1, "sha256": ZERO_SHA}},
    }
    assert_blocked(sealed.validate_prediction_bundle_manifest_v2, manifest)


def test_manifest_rejects_duplicates_bounds_and_noncanonical_order():
    duplicate = b'{"files":{"a":{"bytes":1,"sha256":"' + ZERO_SHA.encode() + b'"},"a":{"bytes":1,"sha256":"' + ZERO_SHA.encode() + b'"}},"job_id":null,"prediction_id":"' + PREDICTION_ID.encode() + b'","schema_version":"on_demand_prediction_bundle_manifest_v2"}\n'
    assert_blocked(sealed._canonical_json, duplicate, max_bytes=sealed.BUNDLE_CONTROL_MAX_BYTES, label="manifest")
    files = {f"f{number:02d}": {"bytes": 1, "sha256": ZERO_SHA} for number in range(33)}
    value = {"schema_version": sealed.PREDICTION_MANIFEST_SCHEMA_V2, "prediction_id": PREDICTION_ID, "job_id": None, "files": files}
    assert_blocked(sealed.validate_prediction_bundle_manifest_v2, value)
    value["files"] = {"z": {"bytes": 1, "sha256": ZERO_SHA}, "a": {"bytes": 1, "sha256": ZERO_SHA}}
    assert_blocked(sealed.validate_prediction_bundle_manifest_v2, value)
    value["files"] = {"a": {"bytes": sealed.BUNDLE_AGGREGATE_MAX_BYTES + 1, "sha256": ZERO_SHA}}
    assert_blocked(sealed.validate_prediction_bundle_manifest_v2, value)


@pytest.mark.parametrize("kind", ["extra_file", "extra_directory", "missing", "changed", "symlink", "fifo"])
def test_detail_rejects_exact_membership_and_unsafe_types(tmp_path: Path, kind: str):
    bundle, entry = make_bundle(tmp_path)
    target = bundle / "config.json"
    if kind == "extra_file":
        (bundle / "extra").write_text("x")
    elif kind == "extra_directory":
        (bundle / "extra").mkdir()
    elif kind == "missing":
        target.unlink()
    elif kind == "changed":
        target.write_text("changed")
    elif kind == "symlink":
        target.unlink()
        target.symlink_to("request.json")
    else:
        target.unlink()
        os.mkfifo(target)
    started = time.monotonic()
    assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, entry)
    if kind == "fifo":
        assert time.monotonic() - started < 1.0


def test_manifest_producer_hashes_exact_descriptor_enumerated_membership(tmp_path: Path):
    bundle = tmp_path / DIRECTORY
    (bundle / "model").mkdir(parents=True)
    (bundle / "result.json").write_bytes(b"result\n")
    (bundle / "model" / "schema.json").write_bytes(b"schema\n")

    manifest = sealed.build_prediction_bundle_manifest_v2(
        bundle, prediction_id=PREDICTION_ID, job_id=None
    )

    assert manifest["files"] == {
        "model/schema.json": {"bytes": 7, "sha256": sha256_bytes(b"schema\n")},
        "result.json": {"bytes": 7, "sha256": sha256_bytes(b"result\n")},
    }


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_manifest_producer_rejects_unsafe_types(tmp_path: Path, kind: str):
    bundle = tmp_path / DIRECTORY
    bundle.mkdir()
    target = bundle / "unsafe"
    if kind == "symlink":
        target.symlink_to("missing")
    else:
        os.mkfifo(target)
    assert_blocked(
        sealed.build_prediction_bundle_manifest_v2,
        bundle,
        prediction_id=PREDICTION_ID,
        job_id=None,
    )


def test_manifest_producer_rejects_file_replacement_during_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = tmp_path / DIRECTORY
    bundle.mkdir()
    target = bundle / "result.json"
    target.write_bytes(b"same\n")
    inode = target.stat().st_ino
    original = sealed._read_fd

    def replace_after_read(descriptor: int, *args: Any, **kwargs: Any) -> bytes:
        raw = original(descriptor, *args, **kwargs)
        if os.fstat(descriptor).st_ino == inode and target.exists():
            target.rename(bundle / "old")
            target.write_bytes(raw)
            (bundle / "old").unlink()
        return raw

    monkeypatch.setattr(sealed, "_read_fd", replace_after_read)
    assert_blocked(
        sealed.build_prediction_bundle_manifest_v2,
        bundle,
        prediction_id=PREDICTION_ID,
        job_id=None,
    )


def test_manifest_producer_rejects_directory_replacement_and_extra_entry_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = tmp_path / DIRECTORY
    model = bundle / "model"
    model.mkdir(parents=True)
    (model / "schema.json").write_bytes(b"{}\n")
    inode = model.stat().st_ino
    original = sealed._directory_children

    def replace_after_enumeration(descriptor: int) -> tuple[str, ...]:
        children = original(descriptor)
        if os.fstat(descriptor).st_ino == inode and model.exists():
            model.rename(bundle / "retained-model")
            model.mkdir()
            (model / "extra").write_bytes(b"x")
        return children

    monkeypatch.setattr(sealed, "_directory_children", replace_after_enumeration)
    assert_blocked(
        sealed.build_prediction_bundle_manifest_v2,
        bundle,
        prediction_id=PREDICTION_ID,
        job_id=None,
    )


def test_detail_rejects_configured_root_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    _, entry = make_bundle(root)
    original = sealed._read_fd
    replaced = False

    def replace_after_read(*args: Any, **kwargs: Any) -> bytes:
        nonlocal replaced
        raw = original(*args, **kwargs)
        if not replaced:
            root.rename(tmp_path / "retained-root")
            root.mkdir()
            replaced = True
        return raw

    monkeypatch.setattr(sealed, "_read_fd", replace_after_read)
    assert_blocked(sealed.verify_indexed_prediction_bundle, root, entry)
    assert replaced


def test_detail_rejects_indexed_bundle_component_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    bundle, entry = make_bundle(root)
    original = sealed._read_fd
    replaced = False

    def replace_after_manifest_read(*args: Any, **kwargs: Any) -> bytes:
        nonlocal replaced
        raw = original(*args, **kwargs)
        if not replaced:
            bundle.rename(root / "retained-bundle")
            bundle.mkdir()
            replaced = True
        return raw

    monkeypatch.setattr(sealed, "_read_fd", replace_after_manifest_read)
    assert_blocked(sealed.verify_indexed_prediction_bundle, root, entry)
    assert replaced


def test_detail_rejects_derived_directory_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    bundle, entry = make_bundle(root)
    model = bundle / "model"
    model_inode = model.stat().st_ino
    original = sealed._directory_children
    replaced = False

    def replace_retained_directory(descriptor: int) -> tuple[str, ...]:
        nonlocal replaced
        if not replaced and os.fstat(descriptor).st_ino == model_inode:
            model.rename(bundle / "retained-model")
            model.mkdir()
            replaced = True
        return original(descriptor)

    monkeypatch.setattr(sealed, "_directory_children", replace_retained_directory)
    assert_blocked(sealed.verify_indexed_prediction_bundle, root, entry)
    assert replaced


def test_detail_rejects_regular_file_replacement_with_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "root"
    root.mkdir()
    bundle, entry = make_bundle(root)
    target = bundle / "config.json"
    target_inode = target.stat().st_ino
    original = sealed._read_fd
    replaced = False

    def replace_after_read(descriptor: int, *args: Any, **kwargs: Any) -> bytes:
        nonlocal replaced
        raw = original(descriptor, *args, **kwargs)
        if not replaced and os.fstat(descriptor).st_ino == target_inode:
            target.rename(bundle / "retained-config.json")
            target.write_bytes(raw)
            (bundle / "retained-config.json").unlink()
            replaced = True
        return raw

    monkeypatch.setattr(sealed, "_read_fd", replace_after_read)
    assert_blocked(sealed.verify_indexed_prediction_bundle, root, entry)
    assert replaced


def test_detail_rejects_manifest_index_request_and_job_binding(tmp_path: Path):
    bundle, entry = make_bundle(tmp_path, ready_result(job_id="job_" + "1" * 32))
    attacks = [
        {**entry, "job_id": "job_" + "2" * 32},
        {**entry, "manifest_sha256": ZERO_SHA},
        {**entry, "logical_bundle_sha256": ZERO_SHA},
        {**entry, "directory": DIRECTORY[:-1] + "c"},
    ]
    for attack in attacks:
        assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, attack)
    request = request_for(ready_result(job_id="job_" + "1" * 32))
    request["race_id"] = "other"
    (bundle / "request.json").write_bytes(canonical_bytes(request))
    assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, entry)


@pytest.mark.parametrize("field", ["prediction_id", "job_id"])
def test_detail_rejects_transplanted_byte_identical_request_identity(
    tmp_path: Path, field: str
):
    job_id = "job_" + "1" * 32
    result = ready_result(job_id=job_id)
    bundle, entry = make_bundle(tmp_path, result)
    transplanted = request_for(result)
    transplanted[field] = (
        "22345678-1234-4123-8123-123456789abc"
        if field == "prediction_id" else "job_" + "2" * 32
    )
    (bundle / "request.json").write_bytes(canonical_bytes(transplanted))
    manifest = sealed.build_prediction_bundle_manifest_v2(
        bundle, prediction_id=result["prediction_id"], job_id=job_id
    )
    manifest_raw = canonical_bytes(manifest)
    (bundle / "bundle_manifest.json").write_bytes(manifest_raw)
    entry = sealed.prediction_bundle_index_entry(
        bundle=bundle, result=result, manifest_raw=manifest_raw
    )
    assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, entry)


@pytest.mark.parametrize(
    ("url", "venue", "venue_slug"),
    [
        ("https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5", "GOSF", "gunnedah"),
        ("https://www.thedogs.com.au/racing/gunnedah/2026-07-19/5", "Gunnedah", "gunnedah"),
        ("https://www.thedogs.com.au/racing/gosford/2026-07-19/5", "GUNN", "gosford"),
    ],
)
def test_result_rejects_venue_substitution_alias_and_mapping_mismatch(
    url: str, venue: str, venue_slug: str
):
    value = ready_result()
    value["race"].update(url=url, venue=venue, venue_slug=venue_slug)
    value["race"]["race_id"] = sealed.stable_race_id({
        "race_number": 5, "venue": venue, "race_date": "2026-07-19", "url": url,
    })
    assert_blocked(sealed.validate_prediction_result_v2, value)


def test_lock_contention_stale_symlink_and_special_file_are_never_stolen(tmp_path: Path):
    _, entry = make_bundle(tmp_path)
    lock = tmp_path / sealed.PREDICTION_BUNDLE_LOCK_NAME
    for create in (
        lambda: lock.write_bytes(canonical_bytes({"pid": 999999, "token": "stale"})),
        lambda: lock.symlink_to("missing"),
        lambda: os.mkfifo(lock),
    ):
        create()
        assert_blocked(sealed.publish_prediction_bundle_index_entry, tmp_path, entry)
        assert lock.lexists() if hasattr(lock, "lexists") else os.path.lexists(lock)
        if lock.is_symlink() or lock.is_file():
            lock.unlink()
        else:
            os.unlink(lock)


def test_lock_release_refuses_replacement_and_non_owner(tmp_path: Path):
    root_fd = os.open(tmp_path, sealed._open_flags(directory=True))
    try:
        lock_fd, identity, payload = sealed._acquire_index_lock(
            root_fd, start=0.0, monotonic=lambda: 0.0
        )
        lock = tmp_path / sealed.PREDICTION_BUNDLE_LOCK_NAME
        lock.unlink()
        lock.write_bytes(payload)
        assert_blocked(sealed._release_index_lock, root_fd, lock_fd, identity, payload)
        assert lock.exists()
        os.close(lock_fd)
    finally:
        os.close(root_fd)


def test_concurrent_publish_never_loses_an_acknowledged_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, first = make_bundle(tmp_path)
    second = copy.deepcopy(first)
    second["prediction_id"] = "22345678-1234-4123-8123-123456789abc"
    second["directory"] = DIRECTORY[:-1] + "c"
    second["generated_at"] = (GENERATED + timedelta(seconds=1)).isoformat()
    started = threading.Barrier(2)

    def publish(entry: dict[str, Any]) -> tuple[bool, str]:
        started.wait()
        try:
            sealed.publish_prediction_bundle_index_entry(tmp_path, entry)
            return True, entry["prediction_id"]
        except PredictionBlocked:
            return False, entry["prediction_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(publish, (first, second)))
    acknowledged = {prediction_id for success, prediction_id in outcomes if success}
    indexed = {row["prediction_id"] for row in sealed.verify_prediction_bundle_index(tmp_path)["entries"]}
    assert acknowledged
    assert indexed == acknowledged


def test_legacy_v1_remains_verifiable_but_is_not_catalogued(tmp_path: Path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "result.json").write_bytes(canonical_bytes({"schema_version": "on_demand_race_prediction_v1"}))
    (legacy / "bundle_manifest.json").write_bytes(canonical_bytes(sealed.bundle_manifest(legacy)))
    assert sealed.verify_bundle(legacy)["schema_version"] == "on_demand_prediction_bundle_manifest_v1"
    index = {"schema_version": sealed.PREDICTION_BUNDLE_INDEX_SCHEMA, "entries": []}
    (tmp_path / sealed.PREDICTION_BUNDLE_INDEX_NAME).write_bytes(canonical_bytes(index))
    assert sealed.verify_prediction_bundle_index(tmp_path)["entries"] == []


def test_verifier_has_no_recursive_walk_or_collector_lock_authority():
    source = Path(sealed.__file__).read_text()
    verifier = source[source.index("def verify_indexed_prediction_bundle"):source.index("def build_prediction_bundle_manifest_v2")]
    assert ".rglob(" not in verifier
    assert ".walk(" not in verifier
    assert "collector_lock" not in source.lower()


@pytest.mark.parametrize("requested,resolved", sorted(sealed.MODEL_ALIASES.items()))
def test_c5_result_accepts_every_supported_model_selector_mapping(
    requested: str, resolved: str
):
    value = ready_result()
    value["model"].update(
        requested=requested,
        resolved=resolved,
        alias_resolved=requested != resolved,
    )
    if resolved != "market_only_v1":
        value["model"].update(
            artifact_identity="AVAILABLE",
            artifact_sha256=ZERO_SHA,
            artifact_manifest_identity="AVAILABLE",
            artifact_manifest_sha256=ZERO_SHA,
        )
        value["evidence"].update(
            model_artifact="model/model.json", model_manifest="model/manifest.json"
        )
    sealed.validate_prediction_result_v2(value)


@pytest.mark.parametrize(
    "requested,resolved,alias_resolved",
    [
        ("attacker-alias", "market_only_v1", True),
        ("attacker-alias", "market_only_v1", False),
        ("market-only", "market_form_residual_v1", True),
        ("market_only_v1", "market_only_v1", True),
        (" MARKET-ONLY ", "market_only_v1", False),
        (" MARKET_ONLY_V1 ", "market_only_v1", True),
        ("LATEST-RESEARCH", "market_only_v1", True),
    ],
)
def test_c5_result_rejects_self_asserted_model_selector_identity(
    requested: str, resolved: str, alias_resolved: bool
):
    value = ready_result()
    value["model"].update(
        requested=requested,
        resolved=resolved,
        alias_resolved=alias_resolved,
    )
    assert_blocked(sealed.validate_prediction_result_v2, value)


@pytest.mark.parametrize(
    "status,blocker_stage",
    [("PREDICTION_BLOCKED", None), ("PREDICTION_READY", "PROTOCOL")],
)
def test_c5_index_and_detail_reject_status_stage_disagreement(
    tmp_path: Path, status: str, blocker_stage: str | None
):
    _, entry = make_bundle(tmp_path)
    entry.update(status=status, blocker_stage=blocker_stage)
    index = {
        "schema_version": sealed.PREDICTION_BUNDLE_INDEX_SCHEMA,
        "entries": [entry],
    }
    assert_blocked(sealed.validate_prediction_bundle_index_v1, index)
    assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, entry)


@pytest.mark.parametrize("nested", [False, True])
def test_c5_manifest_producer_rejects_hardlinked_declared_files(
    tmp_path: Path, nested: bool
):
    bundle = tmp_path / DIRECTORY
    bundle.mkdir()
    first = bundle / "first"
    first.write_bytes(b"same inode\n")
    alias = bundle / "nested" / "alias" if nested else bundle / "alias"
    alias.parent.mkdir(parents=True, exist_ok=True)
    os.link(first, alias)
    assert_blocked(
        sealed.build_prediction_bundle_manifest_v2,
        bundle,
        prediction_id=PREDICTION_ID,
        job_id=None,
    )


def test_c5_manifest_producer_rejects_declared_file_hardlinked_to_manifest(
    tmp_path: Path
):
    bundle = tmp_path / DIRECTORY
    bundle.mkdir()
    manifest = bundle / "bundle_manifest.json"
    manifest.write_bytes(b"existing manifest\n")
    os.link(manifest, bundle / "alias")
    assert_blocked(
        sealed.build_prediction_bundle_manifest_v2,
        bundle,
        prediction_id=PREDICTION_ID,
        job_id=None,
    )


def _rewrite_manifest_for_hardlink_attack(
    bundle: Path, entry: dict[str, Any], relative: str, evidence: dict[str, Any]
) -> None:
    manifest_path = bundle / "bundle_manifest.json"
    value = sealed._canonical_json(
        manifest_path.read_bytes(),
        max_bytes=sealed.BUNDLE_CONTROL_MAX_BYTES,
        label="manifest",
    )
    value["files"][relative] = evidence
    value["files"] = dict(sorted(value["files"].items()))
    raw = canonical_bytes(value)
    manifest_path.write_bytes(raw)
    entry.update(
        manifest_sha256=sha256_bytes(raw),
        logical_bundle_sha256=sealed.logical_bundle_sha256(value),
    )


@pytest.mark.parametrize("nested", [False, True])
def test_c5_detail_rejects_hardlinked_declared_files(tmp_path: Path, nested: bool):
    bundle, entry = make_bundle(tmp_path)
    source = bundle / "config.json"
    alias = bundle / "nested" / "alias.json" if nested else bundle / "alias.json"
    alias.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, alias)
    relative = alias.relative_to(bundle).as_posix()
    _rewrite_manifest_for_hardlink_attack(
        bundle,
        entry,
        relative,
        {"bytes": source.stat().st_size, "sha256": sha256_bytes(source.read_bytes())},
    )
    assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, entry)


def test_c5_detail_rejects_declared_file_hardlinked_to_manifest(tmp_path: Path):
    bundle, entry = make_bundle(tmp_path)
    manifest_path = bundle / "bundle_manifest.json"
    os.link(manifest_path, bundle / "manifest-alias.json")
    _rewrite_manifest_for_hardlink_attack(
        bundle,
        entry,
        "manifest-alias.json",
        {"bytes": 0, "sha256": ZERO_SHA},
    )
    assert_blocked(sealed.verify_indexed_prediction_bundle, tmp_path, entry)
