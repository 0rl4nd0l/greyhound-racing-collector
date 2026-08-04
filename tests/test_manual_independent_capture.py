from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.predictor.manual_independent_capture import (
    ARTIFACT_PATH_BY_ROLE,
    AUTHORITY_MATRIX,
    AUTHORITY_PROFILE,
    CONFIG_SCHEMA_VERSION,
    CONTRACT_VERSION,
    DOWNSTREAM_ADMISSIBILITY,
    PHASE7_EXCLUSION_REASON,
    PROTECTED_PATH_KEYS,
    SAFETY_FIELDS,
    SOURCE_PATH_BY_CLASS,
    TERMINAL_ARTIFACT_SCHEMA_VERSION,
    TERMINAL_STATUS_BY_FAILURE_CODE,
    ManualIndependentCaptureRejected,
    authority_matrix,
    canonical_bytes,
    canonical_sha256,
    parse_canonical_json,
    validate_config,
    validate_terminal_artifact,
)
from src.predictor.on_demand import sealed_runner_set_sha256, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = ROOT / "configs/prediction/manual-independent-capture-v1"
COMMIT = "c20932008edaa02f602733253165f2cd7845a2a3"
TREE = "c4b5fc900e1a347c6fe0c889d3b300c7df8d2922"
RUN_ID = "10000000-0000-4000-8000-000000000001"
REQUEST_ID = "20000000-0000-4000-8000-000000000002"
MODEL_BYTES = b'{"model":"fixture-research-model"}\n'


def forbidden_paths(base: str = "/srv/greyhound-protected") -> dict[str, str]:
    return {name: f"{base}/{name.replace('_', '-')}" for name in PROTECTED_PATH_KEYS}


def config() -> dict:
    return json.loads((SCHEMA_ROOT / "example-config.json").read_bytes())


def race() -> dict:
    return {
        "url": "https://www.thedogs.com.au/racing/richmond/2026-08-04/1/race-name",
        "race_id": "Race 1 - RICH - 2026-08-04",
        "race_date": "2026-08-04",
        "venue": "RICH",
        "venue_slug": "richmond",
        "race_number": 1,
        "scheduled_start": "2026-08-04T00:10:05+00:00",
    }


def runners() -> list[dict]:
    return [
        {
            "box_number": 1,
            "display_name": "Alpha Dog",
            "identity": "ALPHA DOG",
            "source_native_runner_id": "dog-1",
            "decimal_odds": 2.5,
        },
        {
            "box_number": 2,
            "display_name": "Beta Dog",
            "identity": "BETA DOG",
            "source_native_runner_id": "dog-2",
            "decimal_odds": 3.75,
        },
    ]


def request() -> dict:
    selected = race()
    return {
        "request_id": REQUEST_ID,
        "requested_at": "2026-08-04T00:00:00+00:00",
        "requested_race_url": selected["url"],
        "selected_race": selected,
        "minimum_prejump_margin_seconds": 120,
        "attempt_authority": "one_attempt",
        "manual_concurrency": "one_manual_run",
        "safety": dict(SAFETY_FIELDS),
    }


def ready_artifact() -> tuple[dict, dict[str, bytes]]:
    cfg = config()
    req = request()
    selected = req["selected_race"]
    runner_rows = runners()
    canonical_runners = [
        {key: value for key, value in row.items() if key != "decimal_odds"}
        for row in runner_rows
    ]
    source = b"box,dog,source_timestamp\n1,Alpha Dog,2026-08-04T00:00:09+00:00\n"
    model = MODEL_BYTES
    capture = canonical_bytes({"runner_set": runner_rows})
    cfg_raw = canonical_bytes(cfg)
    member_bytes = {
        "capture/odds.json": capture,
        "config/config.json": cfg_raw,
        "model/model.json": model,
        "sources/form.csv": source,
    }
    artifact = {
        "schema_version": TERMINAL_ARTIFACT_SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "run_id": RUN_ID,
        "safety": dict(SAFETY_FIELDS),
        "authority_profile": AUTHORITY_PROFILE,
        "request": req,
        "timing": {
            "submitted_at": req["requested_at"],
            "readiness_checked_at": "2026-08-04T00:00:05+00:00",
            "deadline_at": "2026-08-04T00:01:35+00:00",
            "cleanup_deadline_at": None,
            "capture_timestamp": "2026-08-04T00:00:10+00:00",
            "readiness_prejump_margin_seconds": 600,
            "capture_prejump_margin_seconds": 595,
            "cancel_requested_at": None,
            "terminal_at": "2026-08-04T00:00:15+00:00",
        },
        "attempt": {"attempt_count": 1, "source_attempt_count": 1},
        "terminal": {"status": "CAPTURE_READY", "failure_code": None},
        "provenance": {
            "source_commit": COMMIT,
            "source_tree": TREE,
            "config_sha256": canonical_sha256(cfg),
            "model_sha256": sha256_bytes(model),
            "request_sha256": canonical_sha256(req),
            "race_identity_sha256": canonical_sha256(selected),
            "runner_set_sha256": sealed_runner_set_sha256(selected, canonical_runners),
            "odds_sha256": canonical_sha256(
                [
                    {
                        "box_number": row["box_number"],
                        "decimal_odds": row["decimal_odds"],
                    }
                    for row in runner_rows
                ]
            ),
            "source_files": [
                {
                    "path": "sources/form.csv",
                    "content_class": "prejump_form",
                    "outcome_scope": "target_same_future_outcomes_excluded",
                    "race_url": selected["url"],
                    "race_identity_sha256": canonical_sha256(selected),
                    "source_timestamp": "2026-08-04T00:00:09+00:00",
                    "bytes": len(source),
                    "sha256": sha256_bytes(source),
                }
            ],
            "artifact_hashes": [
                {
                    "role": "capture",
                    "path": "capture/odds.json",
                    "bytes": len(capture),
                    "sha256": sha256_bytes(capture),
                },
                {
                    "role": "config",
                    "path": "config/config.json",
                    "bytes": len(cfg_raw),
                    "sha256": sha256_bytes(cfg_raw),
                },
                {
                    "role": "model",
                    "path": "model/model.json",
                    "bytes": len(model),
                    "sha256": sha256_bytes(model),
                },
            ],
        },
        "capture": {"runner_set": runner_rows},
        "closure": {
            "bundle_closed": True,
            "closed_at": "2026-08-04T00:00:15+00:00",
            "phase7_accessed": False,
            "outcome_accessed": False,
            "canonical_write_claimed": False,
            "downstream_admissibility": DOWNSTREAM_ADMISSIBILITY,
        },
    }
    return artifact, member_bytes


def terminal_without_capture(
    code: str,
    status: str,
    *,
    source_attempt_count: int = 0,
    cancelled: bool = False,
) -> tuple[dict, dict[str, bytes]]:
    artifact, members = ready_artifact()
    artifact["terminal"] = {"status": status, "failure_code": code}
    artifact["attempt"]["source_attempt_count"] = source_attempt_count
    artifact["timing"]["capture_timestamp"] = None
    artifact["timing"]["capture_prejump_margin_seconds"] = None
    artifact["timing"]["cancel_requested_at"] = (
        "2026-08-04T00:00:12+00:00" if cancelled else None
    )
    artifact["timing"]["cleanup_deadline_at"] = (
        "2026-08-04T00:00:22+00:00" if cancelled else None
    )
    artifact["capture"]["runner_set"] = []
    artifact["provenance"]["source_files"] = []
    for name in ("runner_set_sha256", "odds_sha256"):
        artifact["provenance"][name] = None
    artifact["provenance"]["artifact_hashes"] = [
        row
        for row in artifact["provenance"]["artifact_hashes"]
        if row["role"] != "capture"
    ]
    members.pop("capture/odds.json")
    members.pop("sources/form.csv")
    return artifact, members


def valid_failure_artifact(code: str) -> tuple[dict, dict[str, bytes]]:
    status = TERMINAL_STATUS_BY_FAILURE_CODE[code]
    if code in {"FEATURE_BLOCKED", "SCORING_BLOCKED"}:
        artifact, members = ready_artifact()
        artifact["terminal"] = {"status": status, "failure_code": code}
        return artifact, members
    source_attempt_count = int(
        code
        in {
            "SOURCE_TIMEOUT",
            "SOURCE_MALFORMED",
            "IDENTITY_MISMATCH",
            "RUNNER_SET_MISMATCH",
            "ODDS_INVALID",
        }
    )
    artifact, members = terminal_without_capture(
        code,
        status,
        source_attempt_count=source_attempt_count,
        cancelled=code == "CANCELLED",
    )
    if code == "EXACT_RACE_INVALID":
        artifact["request"]["requested_race_url"] = "not-an-exact-race-url"
        artifact["request"]["selected_race"] = None
        artifact["timing"]["readiness_prejump_margin_seconds"] = None
        artifact["provenance"]["race_identity_sha256"] = None
        artifact["provenance"]["request_sha256"] = canonical_sha256(artifact["request"])
    elif code == "INSUFFICIENT_PREJUMP_MARGIN":
        artifact["request"]["selected_race"]["scheduled_start"] = (
            "2026-08-04T00:01:00+00:00"
        )
        artifact["timing"]["readiness_prejump_margin_seconds"] = 55
        artifact["provenance"]["request_sha256"] = canonical_sha256(artifact["request"])
        artifact["provenance"]["race_identity_sha256"] = canonical_sha256(
            artifact["request"]["selected_race"]
        )
    elif code in {"TIMED_OUT", "PROCESS_REAP_UNCONFIRMED"}:
        artifact["timing"]["cleanup_deadline_at"] = artifact["timing"]["deadline_at"]
        artifact["timing"]["terminal_at"] = artifact["timing"]["deadline_at"]
        artifact["closure"]["closed_at"] = artifact["timing"]["terminal_at"]
    return artifact, members


def validate(artifact: dict, members: dict[str, bytes]) -> dict:
    request_value = artifact["request"]
    return validate_terminal_artifact(
        artifact,
        config=config(),
        forbidden_paths=forbidden_paths(),
        member_bytes=members,
        expected_source_commit=COMMIT,
        expected_source_tree=TREE,
        expected_model_sha256=sha256_bytes(MODEL_BYTES),
        expected_source_files=deepcopy(artifact["provenance"]["source_files"]),
        expected_run_id=artifact["run_id"],
        expected_request_id=request_value["request_id"],
        expected_request_sha256=canonical_sha256(request_value),
        seen_run_ids=set(),
        seen_request_ids=set(),
        seen_request_sha256s=set(),
    )


def test_valid_config_and_terminal_artifact_canonical_round_trip():
    cfg = config()
    assert validate_config(cfg, forbidden_paths=forbidden_paths()) == cfg
    assert parse_canonical_json(canonical_bytes(cfg)) == cfg

    artifact, members = ready_artifact()
    validated = validate(artifact, members)
    assert validated == artifact
    assert parse_canonical_json(canonical_bytes(validated)) == artifact
    assert validated["safety"] == {
        "research_only": True,
        "canonical": False,
        "phase7_excluded": True,
        "phase7_eligible": False,
        "phase7_exclusion_reason": PHASE7_EXCLUSION_REASON,
    }


def test_json_schemas_and_example_publish_the_exact_fail_closed_vocabulary():
    config_schema = json.loads((SCHEMA_ROOT / "config.schema.json").read_bytes())
    terminal_schema = json.loads(
        (SCHEMA_ROOT / "terminal-artifact.schema.json").read_bytes()
    )
    assert config_schema["additionalProperties"] is False
    assert terminal_schema["additionalProperties"] is False
    assert (
        config_schema["properties"]["schema_version"]["const"] == CONFIG_SCHEMA_VERSION
    )
    assert terminal_schema["properties"]["schema_version"]["const"] == (
        TERMINAL_ARTIFACT_SCHEMA_VERSION
    )
    assert terminal_schema["$defs"]["safety"]["properties"]["canonical"] == {
        "const": False
    }
    published_codes = set(
        terminal_schema["properties"]["terminal"]["properties"]["failure_code"]["enum"]
    )
    assert published_codes == {None, *TERMINAL_STATUS_BY_FAILURE_CODE}
    validate_config(config(), forbidden_paths=forbidden_paths())


@pytest.mark.parametrize("location", ["config", "artifact", "nested"])
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_missing_and_unknown_fields_are_rejected(location: str, mutation: str):
    artifact, members = ready_artifact()
    cfg = config()
    target = {
        "config": cfg,
        "artifact": artifact,
        "nested": artifact["closure"],
    }[location]
    if mutation == "missing":
        target.pop(next(iter(target)))
    else:
        target["unexpected"] = True

    with pytest.raises(
        ManualIndependentCaptureRejected, match="CONTRACT_FIELDS_INVALID"
    ):
        validate_terminal_artifact(
            artifact,
            config=cfg,
            forbidden_paths=forbidden_paths(),
            member_bytes=members,
            expected_source_commit=COMMIT,
            expected_source_tree=TREE,
            expected_model_sha256=sha256_bytes(MODEL_BYTES),
            expected_source_files=deepcopy(
                artifact.get("provenance", {}).get("source_files", [])
            ),
            expected_run_id=artifact.get("run_id", RUN_ID),
            expected_request_id=artifact.get("request", {}).get(
                "request_id", REQUEST_ID
            ),
            expected_request_sha256=canonical_sha256(
                artifact.get("request", request())
            ),
            seen_run_ids=set(),
            seen_request_ids=set(),
            seen_request_sha256s=set(),
        )


@pytest.mark.parametrize(
    "path_field,path_value",
    [
        ("operations_root", "relative/manual"),
        ("operations_root", "/srv/../manual"),
        ("operations_root", "//srv/manual"),
        ("operations_root", "/srv/manual\nmisleading"),
        ("manual_lock", "/srv/greyhound-manual-operations/shared.lock"),
    ],
)
def test_unsafe_or_non_derived_manual_paths_are_rejected(
    path_field: str, path_value: str
):
    cfg = config()
    cfg["paths"][path_field] = path_value
    with pytest.raises(ManualIndependentCaptureRejected):
        validate_config(cfg, forbidden_paths=forbidden_paths())


@pytest.mark.parametrize(
    "manual_path",
    ["operations_root", "manual_root", "browser_profile", "runs_root", "manual_lock"],
)
@pytest.mark.parametrize("protected_key", sorted(PROTECTED_PATH_KEYS))
def test_manual_root_profile_lock_and_runs_cannot_overlap_protected_paths(
    manual_path: str, protected_key: str
):
    cfg = config()
    protected = forbidden_paths()
    protected[protected_key] = cfg["paths"][manual_path]
    with pytest.raises(
        ManualIndependentCaptureRejected, match="PATH_AUTHORITY_CONFLICT"
    ):
        validate_config(cfg, forbidden_paths=protected)


def test_symlinked_operations_root_cannot_alias_a_protected_root(tmp_path: Path):
    protected_root = tmp_path / "protected"
    protected_root.mkdir()
    alias = tmp_path / "manual-alias"
    alias.symlink_to(protected_root, target_is_directory=True)
    cfg = config()
    cfg["paths"] = {
        "operations_root": str(alias),
        "manual_root": str(alias / CONTRACT_VERSION),
        "browser_profile": str(alias / CONTRACT_VERSION / "browser-profile"),
        "runs_root": str(alias / CONTRACT_VERSION / "runs"),
        "manual_lock": str(alias / CONTRACT_VERSION / "manual-capture.lock"),
    }
    protected = forbidden_paths(str(tmp_path / "other-protected"))
    protected["canonical_database"] = str(protected_root)
    with pytest.raises(
        ManualIndependentCaptureRejected, match="PATH_AUTHORITY_CONFLICT"
    ):
        validate_config(cfg, forbidden_paths=protected)


def test_authority_matrix_forbids_all_shared_canonical_and_downstream_surfaces():
    matrix = authority_matrix()
    assert set(matrix["forbidden_reads"]) == PROTECTED_PATH_KEYS
    assert set(matrix["forbidden_writes"]) == PROTECTED_PATH_KEYS
    assert matrix["lock_authority"] == "manual_capture_lock_only"
    assert matrix["browser_authority"] == "manual_browser_profile_only"
    assert matrix["downstream_admissibility"] == DOWNSTREAM_ADMISSIBILITY
    assert "autonomous_browser_profile_root" in matrix["forbidden_reads"]
    assert "model_artifacts_root" in matrix["forbidden_writes"]
    matrix["allowed_reads"].append("canonical_database")
    assert "canonical_database" not in authority_matrix()["allowed_reads"]
    with pytest.raises(TypeError):
        AUTHORITY_MATRIX["lock_authority"] = "autonomous_shared_lock"  # type: ignore[index]
    with pytest.raises(TypeError):
        TERMINAL_STATUS_BY_FAILURE_CODE["MANUAL_BUSY"] = "CAPTURE_READY"  # type: ignore[index]
    with pytest.raises(TypeError):
        SOURCE_PATH_BY_CLASS["prejump_form"] = "winners.csv"  # type: ignore[index]
    with pytest.raises(TypeError):
        ARTIFACT_PATH_BY_ROLE["capture"] = "capture/results.json"  # type: ignore[index]


def test_hash_drift_in_source_model_config_and_request_is_rejected():
    artifact, members = ready_artifact()
    drifted = dict(members)
    drifted["sources/form.csv"] += b"drift"
    with pytest.raises(ManualIndependentCaptureRejected, match="HASH_DRIFT"):
        validate(artifact, drifted)

    artifact, members = ready_artifact()
    artifact["provenance"]["model_sha256"] = "0" * 64
    with pytest.raises(ManualIndependentCaptureRejected, match="MODEL_HASH_DRIFT"):
        validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["provenance"]["config_sha256"] = "0" * 64
    with pytest.raises(ManualIndependentCaptureRejected, match="CONFIG_HASH_DRIFT"):
        validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["provenance"]["request_sha256"] = "0" * 64
    with pytest.raises(ManualIndependentCaptureRejected, match="REQUEST_HASH_DRIFT"):
        validate(artifact, members)


@pytest.mark.parametrize(
    "field,value",
    [
        ("race_id", "Race 2 - RICHMOND - 2026-08-04"),
        ("race_number", 2),
        ("venue_slug", "wentworth-park"),
        ("url", "https://www.thedogs.com.au/racing/richmond/2026-08-04/2/race-name"),
    ],
)
def test_race_identity_disagreement_is_rejected(field: str, value: object):
    artifact, members = ready_artifact()
    artifact["request"]["selected_race"][field] = value
    with pytest.raises(
        ManualIndependentCaptureRejected, match="RACE_IDENTITY_DISAGREEMENT"
    ):
        validate(artifact, members)


def test_scheduled_start_must_share_the_canonical_race_date():
    artifact, members = ready_artifact()
    selected = artifact["request"]["selected_race"]
    selected["scheduled_start"] = "2027-08-04T00:10:05+00:00"
    artifact["provenance"]["request_sha256"] = canonical_sha256(artifact["request"])
    artifact["provenance"]["race_identity_sha256"] = canonical_sha256(selected)

    with pytest.raises(
        ManualIndependentCaptureRejected, match="RACE_IDENTITY_DISAGREEMENT"
    ):
        validate(artifact, members)


def test_runner_and_odds_hash_drift_are_rejected():
    artifact, members = ready_artifact()
    artifact["capture"]["runner_set"][0]["identity"] = "OTHER DOG"
    with pytest.raises(ManualIndependentCaptureRejected, match="IDENTITY_HASH_DRIFT"):
        validate(artifact, members)

    artifact, members = ready_artifact()
    conflicting_capture = canonical_bytes({"runner_set": runners()[::-1]})
    members["capture/odds.json"] = conflicting_capture
    capture_member = next(
        row
        for row in artifact["provenance"]["artifact_hashes"]
        if row["role"] == "capture"
    )
    capture_member["bytes"] = len(conflicting_capture)
    capture_member["sha256"] = sha256_bytes(conflicting_capture)
    with pytest.raises(ManualIndependentCaptureRejected, match="CAPTURE_HASH_DRIFT"):
        validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["capture"]["runner_set"][0]["decimal_odds"] = 9.0
    with pytest.raises(ManualIndependentCaptureRejected, match="IDENTITY_HASH_DRIFT"):
        validate(artifact, members)


@pytest.mark.parametrize("invalid_odds", [1, 1.0, 0.5, 0, -1])
def test_decimal_odds_must_be_finite_and_greater_than_one(invalid_odds: float):
    artifact, members = ready_artifact()
    artifact["capture"]["runner_set"][0]["decimal_odds"] = invalid_odds
    with pytest.raises(ManualIndependentCaptureRejected, match="ODDS_INVALID"):
        validate(artifact, members)


@pytest.mark.parametrize(
    "location,field,value",
    [
        ("safety", "canonical", True),
        ("safety", "research_only", False),
        ("safety", "phase7_eligible", True),
        ("safety", "phase7_excluded", False),
        ("closure", "canonical_write_claimed", True),
        ("closure", "phase7_accessed", True),
        ("closure", "outcome_accessed", True),
        ("closure", "downstream_admissibility", "phase7_eligible"),
    ],
)
def test_canonical_phase7_and_outcome_claims_are_rejected(
    location: str, field: str, value: object
):
    artifact, members = ready_artifact()
    artifact[location][field] = value
    with pytest.raises(ManualIndependentCaptureRejected):
        validate(artifact, members)


def test_outcome_fields_and_result_evidence_paths_are_not_expressible():
    artifact, members = ready_artifact()
    artifact["capture"]["runner_set"][0]["official_result"] = 1
    with pytest.raises(
        ManualIndependentCaptureRejected, match="CONTRACT_FIELDS_INVALID"
    ):
        validate(artifact, members)

    artifact, members = ready_artifact()
    source = artifact["provenance"]["source_files"][0]
    old_path = source["path"]
    source["path"] = "results/form.csv"
    members[source["path"]] = members.pop(old_path)
    with pytest.raises(
        ManualIndependentCaptureRejected, match="FORBIDDEN_ARTIFACT_LOCATOR"
    ):
        validate(artifact, members)

    for forbidden_variant in (
        "sources/official-results.json",
        "sources/officialresults.json",
        "sources/race_outcome.csv",
        "sources/raceoutcomes.csv",
        "capture/canonical-db.json",
        "capture/canonicaldb.json",
        "capture/phase-7.json",
        "capture/phase7data.json",
    ):
        artifact, members = ready_artifact()
        source = artifact["provenance"]["source_files"][0]
        old_path = source["path"]
        source["path"] = forbidden_variant
        members[source["path"]] = members.pop(old_path)
        with pytest.raises(
            ManualIndependentCaptureRejected, match="FORBIDDEN_ARTIFACT_LOCATOR"
        ):
            validate(artifact, members)

    for outcome_synonym in (
        "sources/winners.csv",
        "sources/placings.json",
        "sources/finishing-order.csv",
    ):
        artifact, members = ready_artifact()
        source = artifact["provenance"]["source_files"][0]
        old_path = source["path"]
        source["path"] = outcome_synonym
        members[source["path"]] = members.pop(old_path)
        with pytest.raises(
            ManualIndependentCaptureRejected, match="SOURCE_PATH_INVALID"
        ):
            validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["provenance"]["source_files"][0]["content_class"] = "prejump_race_source"
    with pytest.raises(ManualIndependentCaptureRejected, match="SOURCE_PATH_INVALID"):
        validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["provenance"]["source_files"][0]["outcome_scope"] = (
        "target_outcomes_included"
    )
    with pytest.raises(ManualIndependentCaptureRejected, match="OUTCOME_SCOPE_INVALID"):
        validate(artifact, members)


def test_replay_conflict_and_late_artifacts_fail_closed():
    artifact, members = ready_artifact()
    with pytest.raises(ManualIndependentCaptureRejected, match="REPLAYED_ARTIFACT"):
        validate_terminal_artifact(
            artifact,
            config=config(),
            forbidden_paths=forbidden_paths(),
            member_bytes=members,
            expected_source_commit=COMMIT,
            expected_source_tree=TREE,
            expected_model_sha256=sha256_bytes(MODEL_BYTES),
            expected_source_files=deepcopy(artifact["provenance"]["source_files"]),
            expected_run_id=RUN_ID,
            expected_request_id=REQUEST_ID,
            expected_request_sha256=canonical_sha256(artifact["request"]),
            seen_run_ids={RUN_ID},
            seen_request_ids=set(),
            seen_request_sha256s=set(),
        )
    with pytest.raises(ManualIndependentCaptureRejected, match="ARTIFACT_CONFLICT"):
        validate_terminal_artifact(
            artifact,
            config=config(),
            forbidden_paths=forbidden_paths(),
            member_bytes=members,
            expected_source_commit=COMMIT,
            expected_source_tree=TREE,
            expected_model_sha256=sha256_bytes(MODEL_BYTES),
            expected_source_files=deepcopy(artifact["provenance"]["source_files"]),
            expected_run_id=RUN_ID,
            expected_request_id=REQUEST_ID,
            expected_request_sha256="0" * 64,
            seen_run_ids=set(),
            seen_request_ids=set(),
            seen_request_sha256s=set(),
        )

    artifact["timing"]["terminal_at"] = "2026-08-04T00:01:36+00:00"
    artifact["closure"]["closed_at"] = artifact["timing"]["terminal_at"]
    with pytest.raises(ManualIndependentCaptureRejected, match="LATE_ARTIFACT"):
        validate(artifact, members)


def test_success_updates_required_replay_inventories_and_second_use_is_rejected():
    artifact, members = ready_artifact()
    seen_runs: set[str] = set()
    seen_requests: set[str] = set()
    seen_hashes: set[str] = set()
    expected_request_sha256 = canonical_sha256(artifact["request"])
    arguments = {
        "config": config(),
        "forbidden_paths": forbidden_paths(),
        "member_bytes": members,
        "expected_source_commit": COMMIT,
        "expected_source_tree": TREE,
        "expected_model_sha256": sha256_bytes(MODEL_BYTES),
        "expected_source_files": deepcopy(artifact["provenance"]["source_files"]),
        "expected_run_id": RUN_ID,
        "expected_request_id": REQUEST_ID,
        "expected_request_sha256": expected_request_sha256,
        "seen_run_ids": seen_runs,
        "seen_request_ids": seen_requests,
        "seen_request_sha256s": seen_hashes,
    }

    validate_terminal_artifact(artifact, **arguments)
    assert seen_runs == {RUN_ID}
    assert seen_requests == {REQUEST_ID}
    assert seen_hashes == {expected_request_sha256}
    with pytest.raises(ManualIndependentCaptureRejected, match="REPLAYED_ARTIFACT"):
        validate_terminal_artifact(artifact, **arguments)


def test_source_manifest_and_each_source_race_binding_are_trusted():
    artifact, members = ready_artifact()
    trusted_source_files = deepcopy(artifact["provenance"]["source_files"])
    arguments = {
        "config": config(),
        "forbidden_paths": forbidden_paths(),
        "member_bytes": members,
        "expected_source_commit": COMMIT,
        "expected_source_tree": TREE,
        "expected_model_sha256": sha256_bytes(MODEL_BYTES),
        "expected_source_files": trusted_source_files,
        "expected_run_id": RUN_ID,
        "expected_request_id": REQUEST_ID,
        "expected_request_sha256": canonical_sha256(artifact["request"]),
        "seen_run_ids": set(),
        "seen_request_ids": set(),
        "seen_request_sha256s": set(),
    }
    artifact["provenance"]["source_files"][0]["source_timestamp"] = (
        "2026-08-04T00:00:08+00:00"
    )
    with pytest.raises(
        ManualIndependentCaptureRejected, match="SOURCE_PROVENANCE_MISMATCH"
    ):
        validate_terminal_artifact(artifact, **arguments)

    artifact, members = ready_artifact()
    source = artifact["provenance"]["source_files"][0]
    source["race_url"] = (
        "https://www.thedogs.com.au/racing/richmond/2026-08-04/2/other-race"
    )
    source["race_identity_sha256"] = "0" * 64
    with pytest.raises(
        ManualIndependentCaptureRejected, match="RACE_IDENTITY_DISAGREEMENT"
    ):
        validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["provenance"]["source_files"][0]["source_timestamp"] = (
        "2026-08-04T00:00:04+00:00"
    )
    with pytest.raises(ManualIndependentCaptureRejected, match="SOURCE_TIMING_INVALID"):
        validate(artifact, members)


@pytest.mark.parametrize(
    "expected_override",
    [
        {"expected_run_id": "30000000-0000-4000-8000-000000000003"},
        {"expected_request_id": "30000000-0000-4000-8000-000000000003"},
        {"expected_request_sha256": "0" * 64},
    ],
)
def test_trusted_request_identity_is_mandatory(expected_override: dict[str, str]):
    artifact, members = ready_artifact()
    arguments = {
        "config": config(),
        "forbidden_paths": forbidden_paths(),
        "member_bytes": members,
        "expected_source_commit": COMMIT,
        "expected_source_tree": TREE,
        "expected_model_sha256": sha256_bytes(MODEL_BYTES),
        "expected_source_files": deepcopy(artifact["provenance"]["source_files"]),
        "expected_run_id": RUN_ID,
        "expected_request_id": REQUEST_ID,
        "expected_request_sha256": canonical_sha256(artifact["request"]),
        "seen_run_ids": set(),
        "seen_request_ids": set(),
        "seen_request_sha256s": set(),
    }
    arguments.update(expected_override)
    with pytest.raises(ManualIndependentCaptureRejected, match="ARTIFACT_CONFLICT"):
        validate_terminal_artifact(artifact, **arguments)


def test_exact_race_failure_cannot_hide_a_valid_canonical_url():
    artifact, members = valid_failure_artifact("EXACT_RACE_INVALID")
    artifact["request"]["requested_race_url"] = race()["url"]
    artifact["provenance"]["request_sha256"] = canonical_sha256(artifact["request"])
    with pytest.raises(
        ManualIndependentCaptureRejected, match="TERMINAL_FAILURE_CONFLICT"
    ):
        validate(artifact, members)


def test_cancellation_and_timeout_have_hard_terminal_semantics():
    cancelled, members = terminal_without_capture(
        "CANCELLED", "CANCELLED", cancelled=True
    )
    assert validate(cancelled, members)["terminal"]["failure_code"] == "CANCELLED"

    cancelled["timing"]["cancel_requested_at"] = None
    with pytest.raises(ManualIndependentCaptureRejected, match="CANCELLATION_INVALID"):
        validate(cancelled, members)

    timed_out, members = terminal_without_capture("TIMED_OUT", "TIMED_OUT")
    timed_out["timing"]["cleanup_deadline_at"] = timed_out["timing"]["deadline_at"]
    timed_out["timing"]["terminal_at"] = timed_out["timing"]["deadline_at"]
    timed_out["closure"]["closed_at"] = timed_out["timing"]["terminal_at"]
    assert validate(timed_out, members)["terminal"]["failure_code"] == "TIMED_OUT"

    timed_out["timing"]["terminal_at"] = "2026-08-04T00:01:34+00:00"
    timed_out["closure"]["closed_at"] = timed_out["timing"]["terminal_at"]
    with pytest.raises(ManualIndependentCaptureRejected, match="TIMING_INVALID"):
        validate(timed_out, members)

    timed_out, members = terminal_without_capture(
        "TIMED_OUT", "TIMED_OUT", cancelled=True
    )
    timed_out["timing"]["terminal_at"] = timed_out["timing"]["deadline_at"]
    timed_out["closure"]["closed_at"] = timed_out["timing"]["terminal_at"]
    with pytest.raises(ManualIndependentCaptureRejected, match="CANCELLATION_INVALID"):
        validate(timed_out, members)

    timed_out["terminal"] = {
        "status": "FAILED",
        "failure_code": "PROCESS_REAP_UNCONFIRMED",
    }
    timed_out["timing"]["terminal_at"] = timed_out["timing"]["cleanup_deadline_at"]
    timed_out["closure"]["closed_at"] = timed_out["timing"]["terminal_at"]
    assert validate(timed_out, members)["terminal"]["failure_code"] == (
        "PROCESS_REAP_UNCONFIRMED"
    )

    timed_out["timing"]["terminal_at"] = timed_out["timing"]["deadline_at"]
    timed_out["closure"]["closed_at"] = timed_out["timing"]["terminal_at"]
    with pytest.raises(ManualIndependentCaptureRejected, match="CANCELLATION_INVALID"):
        validate(timed_out, members)


def test_source_revision_is_bound_to_trusted_expected_commit_and_tree():
    artifact, members = ready_artifact()
    artifact["provenance"]["source_commit"] = "1" * 40
    with pytest.raises(
        ManualIndependentCaptureRejected, match="SOURCE_PROVENANCE_MISMATCH"
    ):
        validate(artifact, members)

    artifact, members = ready_artifact()
    artifact["provenance"]["source_tree"] = "2" * 40
    with pytest.raises(
        ManualIndependentCaptureRejected, match="SOURCE_PROVENANCE_MISMATCH"
    ):
        validate(artifact, members)


@pytest.mark.parametrize("failure_code", sorted(TERMINAL_STATUS_BY_FAILURE_CODE))
def test_every_published_failure_code_has_one_valid_terminal_shape(failure_code: str):
    artifact, members = valid_failure_artifact(failure_code)
    validated = validate(artifact, members)
    assert validated["terminal"] == {
        "status": TERMINAL_STATUS_BY_FAILURE_CODE[failure_code],
        "failure_code": failure_code,
    }


def test_failure_code_status_attempt_and_probability_conflicts_are_rejected():
    artifact, members = terminal_without_capture("MANUAL_BUSY", "BLOCKED")
    assert validate(artifact, members)["attempt"] == {
        "attempt_count": 1,
        "source_attempt_count": 0,
    }

    artifact["terminal"]["status"] = "FAILED"
    with pytest.raises(
        ManualIndependentCaptureRejected, match="TERMINAL_FAILURE_CONFLICT"
    ):
        validate(artifact, members)

    artifact, members = terminal_without_capture("MANUAL_BUSY", "BLOCKED")
    artifact["attempt"]["source_attempt_count"] = 1
    with pytest.raises(
        ManualIndependentCaptureRejected, match="ATTEMPT_AUTHORITY_INVALID"
    ):
        validate(artifact, members)

    artifact, members = terminal_without_capture("MANUAL_BUSY", "BLOCKED")
    artifact["probabilities"] = [{"box_number": 1, "probability": 1.0}]
    with pytest.raises(
        ManualIndependentCaptureRejected, match="CONTRACT_FIELDS_INVALID"
    ):
        validate(artifact, members)


def test_noncanonical_duplicate_and_nonfinite_json_are_rejected():
    with pytest.raises(
        ManualIndependentCaptureRejected, match="CANONICAL_JSON_INVALID"
    ):
        parse_canonical_json(b'{"b":1,"a":2}\n')
    with pytest.raises(
        ManualIndependentCaptureRejected, match="CANONICAL_JSON_INVALID"
    ):
        parse_canonical_json(b'{"a":1,"a":2}\n')
    with pytest.raises(
        ManualIndependentCaptureRejected, match="CANONICAL_JSON_INVALID"
    ):
        parse_canonical_json(b'{"a":NaN}\n')
