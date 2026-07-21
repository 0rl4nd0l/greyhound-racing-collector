import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from scripts import autonomous_live_odds_capture as autonomous_capture
from scripts import strict_win_odds_fixture_capture as fixture


RACE_ID = "Race 1 - TEST - 2026-07-09"
SOURCE_URL = "https://www.sportsbet.com.au/greyhound-racing/test/race-1"


def _plan(tmp_path: Path) -> dict:
    input_dir = tmp_path / "incoming"
    input_dir.mkdir()
    sidecar = input_dir / "Race 1 - TEST - 2026-07-09.csv.metadata.json"
    sidecar.write_text(
        json.dumps(
            {
                "source": "thedogs",
                "canonical_race_identity": RACE_ID,
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-09",
                "jump_datetime": "2026-07-09T12:30:00+10:00",
                "sportsbet_url": SOURCE_URL,
                "market_type": "win",
                "expected_runners": [
                    {"box_number": 1, "dog_name": "Alpha", "active": True},
                    {"box_number": 2, "dog_name": "Bravo", "active": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "candidate_race_count": 1,
        "ready_to_capture_race_count": 1,
        "items": [
            {
                "schema_version": "autonomous_live_odds_capture_plan_item_v1",
                "status": "READY_TO_CAPTURE",
                "skip_reasons": [],
                "sidecar_path": str(sidecar.relative_to(tmp_path)),
                "canonical_race_identity": RACE_ID,
                "venue": "TEST",
                "race_number": 1,
                "race_date": "2026-07-09",
                "jump_datetime": "2026-07-09T12:30:00+10:00",
                "sportsbet_url": SOURCE_URL,
                "market_type": "win",
                "capture_window_minutes": 10,
                "capture_mode": "autonomous_prejump_t10m",
                "expected_runners": [
                    {"box_number": 1, "dog_name": "Alpha", "active": True},
                    {"box_number": 2, "dog_name": "Bravo", "active": True},
                ],
            }
        ],
    }


def _fetch_result(*, odds_data=None, **overrides) -> dict:
    payload = {
        "success": True,
        "alias_race_id": RACE_ID,
        "race_id": "TEST_2026-07-09_1",
        "win_count": 2,
        "discovery_method": "sportsbet_landing",
        "capture_timestamp": "2026-07-09T12:20:00+10:00",
        "market_type": "win",
        "race_info": {
            "venue_url": SOURCE_URL,
            "venue": "TEST",
            "race_number": 1,
            "race_date": "2026-07-09",
            "jump_datetime": "2026-07-09T12:30:00+10:00",
        },
        "odds_data": odds_data
        if odds_data is not None
        else [
            {
                "box_number": 1,
                "dog_name": "Alpha",
                "odds_decimal": 2.4,
                "sportsbet_box_source": "runner_text",
                "sportsbet_list_position": 1,
                "sportsbet_raw_runner_text": "1. Alpha",
                "active": True,
            },
            {
                "box_number": 2,
                "dog_name": "Bravo",
                "odds_decimal": 3.1,
                "sportsbet_box_source": "runner_text",
                "sportsbet_list_position": 2,
                "sportsbet_raw_runner_text": "2. Bravo",
                "active": True,
            },
        ],
    }
    payload.update(overrides)
    return payload


def _output_dir(tmp_path: Path) -> Path:
    return (
        tmp_path / "artifacts/full_evidence_orchestration_20260525/"
        "strict_win_odds_fixture_capture_test_report_only"
    )


def _current_time() -> datetime:
    return datetime.fromisoformat("2026-07-09T12:20:00+10:00")


def _build_capture_plan(tmp_path: Path) -> dict:
    input_dir = tmp_path / "upcoming"
    input_dir.mkdir()
    csv_path = input_dir / "Race 1 - TEST - 2026-07-09.csv"
    csv_path.write_text("Dog Name,BOX\n1. Alpha,\n2. Bravo,\n", encoding="utf-8")
    autonomous_capture.write_json(
        autonomous_capture.sidecar_path_for(csv_path),
        {
            "metadata_is_leakage_safe": True,
            "prejump_shadow_metadata": {
                "status": "PASS",
                "metadata_is_leakage_safe": True,
                "race_date": "2026-07-09",
                "venue": "TEST",
                "race_number": "1",
                "jump_time": "2026-07-09T12:30:00+10:00",
                "source_url": "https://www.thedogs.com.au/racing/test/2026-07-09/1/example",
                "runner_box_name_list": [
                    {"box_number": 1, "dog_name": "Alpha"},
                    {"box_number": 2, "dog_name": "Bravo"},
                ],
                "canonical_final_runner_alignment": {
                    "status": "aligned",
                    "canonical_runner_set_status": "available",
                },
            },
        },
    )
    return autonomous_capture.build_capture_plan(
        [input_dir], current_time=_current_time()
    )


def _races_plan(tmp_path: Path) -> dict:
    legacy_plan = _plan(tmp_path)
    race = legacy_plan["items"][0]
    race["race_id"] = race.pop("canonical_race_identity")
    return {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "ready_count": 1,
        "races": [race],
    }


def _non_ready_plan(tmp_path: Path, *, status: str, schema: str) -> tuple[dict, dict]:
    plan = _build_capture_plan(tmp_path)
    race = plan["races"][0]
    race["status"] = status
    plan["ready_count"] = 0
    plan["status_counts"] = {status: 1}
    if schema == "canonical":
        return plan, race
    item = json.loads(json.dumps(race))
    item["canonical_race_identity"] = item.pop("race_id")
    return (
        {
            "schema_version": plan["schema_version"],
            "candidate_race_count": 1,
            "ready_to_capture_race_count": 0,
            "items": [item],
        },
        item,
    )


def _reseal_manifest(output_dir: Path, manifest: dict) -> None:
    manifest_base = dict(manifest)
    manifest_base.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = fixture.sha256_payload(manifest_base)
    (output_dir / "strict_win_fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _manifest_entry(output_dir: Path, role: str) -> tuple[dict, dict, Path]:
    manifest = json.loads(
        (output_dir / "strict_win_fixture_manifest.json").read_text(encoding="utf-8")
    )
    entry = next(row for row in manifest["files"] if row["role"] == role)
    return manifest, entry, output_dir / entry["path"]


def _rewrite_manifest_payload(output_dir: Path, role: str, payload: dict) -> None:
    manifest, entry, path = _manifest_entry(output_dir, role)
    payload_bytes = fixture.serialized_json_bytes(payload)
    path.write_bytes(payload_bytes)
    entry["sha256"] = fixture.sha256_bytes(payload_bytes)
    entry["bytes"] = len(payload_bytes)
    _reseal_manifest(output_dir, manifest)


def test_build_fixture_packet_seals_raw_fixture_projection_and_manifest(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload={"fetch_results": [_fetch_result()]},
        output_dir=output_dir,
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_SEALED_NO_DB_APPEND
    assert report["db_append_performed"] is False
    assert report["owner_approval_required_before_append"] is True
    assert report["validation_pass_count"] == 1
    manifest_path = output_dir / "strict_win_fixture_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == fixture.FINAL_SEALED_NO_DB_APPEND
    roles = {row["role"] for row in manifest["files"]}
    assert {"raw_fixture", "normalized_projection", "preseal_validation"} <= roles

    packet_validation = fixture.validate_packet(output_dir)
    assert packet_validation["status"] == "PASS"
    raw_entry = next(row for row in manifest["files"] if row["role"] == "raw_fixture")
    raw_fixture = json.loads(
        (output_dir / raw_entry["path"]).read_text(encoding="utf-8")
    )
    assert raw_fixture["schema_version"] == fixture.RAW_FIXTURE_SCHEMA
    assert raw_fixture["market_type"] == "win"
    assert raw_fixture["provenance"]["append_approved"] is False
    assert [row["match_status"] for row in raw_fixture["runner_rows"]] == [
        "box_name_exact",
        "box_name_exact",
    ]


def test_build_fixture_packet_consumes_build_capture_plan_output_directly(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _build_capture_plan(tmp_path)

    report = fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(
            alias_race_id="Race 9 - OTHER - 2026-07-09",
            race_id="OTHER_2026-07-09_9",
        ),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert plan["schema_version"] == "autonomous_live_odds_capture_plan_v1"
    assert plan["ready_count"] == 1
    assert plan["races"][0]["race_id"] == RACE_ID
    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert report["ready_plan_item_count"] == 1
    assert report["fixture_results"] == [
        {
            "race_id": RACE_ID,
            "status": "BLOCKED",
            "reasons": ["raw_fetch_result_missing_for_ready_race"],
            "fixture_path": None,
            "projection_path": None,
        }
    ]


@pytest.mark.parametrize("ready_count", [0, 2])
def test_ready_plan_items_rejects_producer_ready_count_mismatch(tmp_path, ready_count):
    plan = _build_capture_plan(tmp_path)
    plan["ready_count"] = ready_count

    with pytest.raises(ValueError, match="plan_ready_count_mismatch"):
        fixture.ready_plan_items(plan)


@pytest.mark.parametrize(
    ("race_id", "error"),
    [
        (None, "ready_plan_item_identity_missing"),
        ("   ", "ready_plan_item_identity_missing"),
        (123, "ready_plan_item_identity_malformed"),
    ],
)
def test_ready_plan_items_rejects_missing_or_malformed_producer_race_id(
    tmp_path, race_id, error
):
    plan = _build_capture_plan(tmp_path)
    plan["races"][0]["race_id"] = race_id

    with pytest.raises(ValueError, match=error):
        fixture.ready_plan_items(plan)


@pytest.mark.parametrize("status", ["BLOCKED", "NO_DUE_WINDOW"])
@pytest.mark.parametrize("schema", ["canonical", "legacy"])
@pytest.mark.parametrize(
    ("mutation", "value", "error"),
    [
        ("missing", None, "ready_plan_item_identity_missing"),
        ("null", None, "ready_plan_item_identity_missing"),
        ("blank", "   ", "ready_plan_item_identity_missing"),
        ("non_string", 123, "ready_plan_item_identity_malformed"),
    ],
)
def test_ready_plan_items_rejects_invalid_non_ready_identities(
    tmp_path, status, schema, mutation, value, error
):
    plan, item = _non_ready_plan(tmp_path, status=status, schema=schema)
    identity_key = "race_id" if schema == "canonical" else "canonical_race_identity"
    if mutation == "missing":
        item.pop(identity_key)
    else:
        item[identity_key] = value

    with pytest.raises(ValueError, match=error):
        fixture.ready_plan_items(plan)


@pytest.mark.parametrize("status", ["BLOCKED", "NO_DUE_WINDOW"])
@pytest.mark.parametrize("schema", ["canonical", "legacy"])
def test_ready_plan_items_rejects_conflicting_non_ready_alternate_identity(
    tmp_path, status, schema
):
    plan, item = _non_ready_plan(tmp_path, status=status, schema=schema)
    alternate_key = "canonical_race_identity" if schema == "canonical" else "race_id"
    item[alternate_key] = "Race 2 - TEST - 2026-07-09"

    with pytest.raises(ValueError, match="ready_plan_item_identity_conflict"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_rejects_conflicting_row_identities(tmp_path):
    plan = _build_capture_plan(tmp_path)
    plan["races"][0]["canonical_race_identity"] = "Race 2 - TEST - 2026-07-09"

    with pytest.raises(ValueError, match="ready_plan_item_identity_conflict"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_rejects_duplicate_producer_races(tmp_path):
    plan = _build_capture_plan(tmp_path)
    plan["races"].append(json.loads(json.dumps(plan["races"][0])))
    plan["ready_count"] = 2

    with pytest.raises(ValueError, match="duplicate_ready_plan_race_identities"):
        fixture.ready_plan_items(plan)


@pytest.mark.parametrize("status", ["BLOCKED", "NO_DUE_WINDOW"])
@pytest.mark.parametrize("schema", ["canonical", "legacy"])
def test_ready_plan_items_rejects_duplicate_non_ready_primary_identities(
    tmp_path, status, schema
):
    plan, item = _non_ready_plan(tmp_path, status=status, schema=schema)
    container_key = "races" if schema == "canonical" else "items"
    plan[container_key].append(json.loads(json.dumps(item)))
    plan["candidate_race_count"] = 2
    if schema == "canonical":
        plan["status_counts"] = {status: 2}

    with pytest.raises(ValueError, match="duplicate_ready_plan_race_identities"):
        fixture.ready_plan_items(plan)


@pytest.mark.parametrize("status", ["BLOCKED", "NO_DUE_WINDOW"])
@pytest.mark.parametrize("schema", ["canonical", "legacy"])
@pytest.mark.parametrize("collision", ["alias_alias", "alias_primary"])
def test_ready_plan_items_rejects_cross_record_alias_collisions(
    tmp_path, status, schema, collision
):
    plan, item = _non_ready_plan(tmp_path, status=status, schema=schema)
    container_key = "races" if schema == "canonical" else "items"
    identity_key = "race_id" if schema == "canonical" else "canonical_race_identity"
    item["race_id_aliases"] = ["SHARED-ALIAS"]
    second = json.loads(json.dumps(item))
    second[identity_key] = (
        "SHARED-ALIAS" if collision == "alias_primary" else "Race 2 - TEST - 2026-07-09"
    )
    if collision == "alias_primary":
        second["race_id_aliases"] = ["SECOND-ALIAS"]
    plan[container_key].append(second)
    plan["candidate_race_count"] = 2
    if schema == "canonical":
        plan["status_counts"] = {status: 2}

    with pytest.raises(ValueError, match="plan_race_identity_collision"):
        fixture.ready_plan_items(plan)


@pytest.mark.parametrize("status", ["BLOCKED", "NO_DUE_WINDOW"])
@pytest.mark.parametrize("schema", ["canonical", "legacy"])
@pytest.mark.parametrize("aliases", [None, "alias", [""], [123]])
def test_ready_plan_items_rejects_malformed_non_ready_aliases(
    tmp_path, status, schema, aliases
):
    plan, item = _non_ready_plan(tmp_path, status=status, schema=schema)
    item["race_id_aliases"] = aliases

    with pytest.raises(ValueError, match="ready_plan_item_aliases_malformed"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_retains_equivalent_legacy_items_support(tmp_path):
    plan = _build_capture_plan(tmp_path)
    legacy_item = json.loads(json.dumps(plan["races"][0]))
    legacy_item["canonical_race_identity"] = legacy_item.pop("race_id")
    plan["items"] = [legacy_item]
    plan["ready_to_capture_race_count"] = 1

    expected_item = dict(plan["races"][0])
    expected_item["canonical_race_identity"] = expected_item.pop("race_id")
    assert fixture.ready_plan_items(plan) == [expected_item]


def test_ready_plan_items_rejects_conflicting_mixed_schema_rows(tmp_path):
    plan = _build_capture_plan(tmp_path)
    legacy_item = json.loads(json.dumps(plan["races"][0]))
    legacy_item["canonical_race_identity"] = "Race 2 - TEST - 2026-07-09"
    legacy_item.pop("race_id")
    plan["items"] = [legacy_item]
    plan["ready_to_capture_race_count"] = 1

    with pytest.raises(ValueError, match="plan_mixed_schema_conflict"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_rejects_conflicting_non_ready_mixed_schema_rows(tmp_path):
    plan = _build_capture_plan(tmp_path)
    blocked = json.loads(json.dumps(plan["races"][0]))
    blocked["status"] = "BLOCKED"
    blocked["race_id"] = "Race 2 - TEST - 2026-07-09"
    plan["races"].append(blocked)
    plan["status_counts"] = {"READY_TO_CAPTURE": 1, "BLOCKED": 1}

    plan["items"] = json.loads(json.dumps(plan["races"]))
    for item in plan["items"]:
        item["canonical_race_identity"] = item.pop("race_id")
    plan["ready_to_capture_race_count"] = 1

    assert fixture.ready_plan_items(plan) == [plan["items"][0]]

    plan["items"][1]["status"] = "NO_DUE_WINDOW"
    with pytest.raises(ValueError, match="plan_mixed_schema_conflict"):
        fixture.ready_plan_items(plan)


def test_build_fixture_packet_accepts_zero_ready_producer_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _races_plan(tmp_path)
    plan["races"][0]["status"] = "BLOCKED"
    plan["races"][0]["race_id_aliases"] = [RACE_ID, "RACE-ONE-ALIAS"]
    second = json.loads(json.dumps(plan["races"][0]))
    second["status"] = "NO_DUE_WINDOW"
    second["race_id"] = "Race 2 - TEST - 2026-07-09"
    second["race_id_aliases"] = ["RACE-TWO-ALIAS"]
    plan["races"].append(second)
    plan["candidate_race_count"] = 2
    plan["ready_count"] = 0
    plan["status_counts"] = {"BLOCKED": 1, "NO_DUE_WINDOW": 1}

    report = fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(
            alias_race_id="Race 9 - OTHER - 2026-07-09",
            race_id="OTHER_2026-07-09_9",
        ),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_NO_READY_RACES
    assert report["ready_plan_item_count"] == 0
    assert report["fixture_results"] == []


def test_build_fixture_packet_rejects_invalid_non_ready_record_before_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _races_plan(tmp_path)
    plan["races"][0]["status"] = "BLOCKED"
    plan["races"][0]["race_id"] = None
    plan["candidate_race_count"] = 1
    plan["ready_count"] = 0
    plan["status_counts"] = {"BLOCKED": 1}
    output_dir = _output_dir(tmp_path)

    with pytest.raises(ValueError, match="ready_plan_item_identity_missing"):
        fixture.build_fixture_packet(
            plan=plan,
            fetch_payload=_fetch_result(),
            output_dir=output_dir,
            current_time=_current_time(),
        )

    assert not output_dir.exists()


def test_ready_plan_items_accepts_fully_validated_zero_ready_mixed_schema(tmp_path):
    plan, first = _non_ready_plan(tmp_path, status="BLOCKED", schema="canonical")
    first["race_id_aliases"] = [RACE_ID, "RACE-ONE-ALIAS"]
    second = json.loads(json.dumps(first))
    second["status"] = "NO_DUE_WINDOW"
    second["race_id"] = "Race 2 - TEST - 2026-07-09"
    second["race_id_aliases"] = ["RACE-TWO-ALIAS"]
    plan["races"].append(second)
    plan["candidate_race_count"] = 2
    plan["status_counts"] = {"BLOCKED": 1, "NO_DUE_WINDOW": 1}
    plan["items"] = json.loads(json.dumps(plan["races"]))
    for item in plan["items"]:
        item["canonical_race_identity"] = item.pop("race_id")
    plan["ready_to_capture_race_count"] = 0

    assert fixture.ready_plan_items(plan) == []


def test_producer_schema_fixture_creation_and_replay_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    report = fixture.build_fixture_packet(
        plan=_races_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_SEALED_NO_DB_APPEND
    assert report["validation_pass_count"] == 1
    assert fixture.validate_packet(output_dir)["status"] == "PASS"


def test_ready_plan_items_rejects_unknown_status_instead_of_discarding_row(tmp_path):
    plan = _build_capture_plan(tmp_path)
    plan["races"][0]["status"] = "UNKNOWN"
    plan["ready_count"] = 0
    plan["status_counts"] = {"UNKNOWN": 1}

    with pytest.raises(ValueError, match="plan_item_status_invalid"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_rejects_unaccounted_producer_record(tmp_path):
    plan = _build_capture_plan(tmp_path)
    blocked = json.loads(json.dumps(plan["races"][0]))
    blocked["status"] = "BLOCKED"
    blocked["race_id"] = "Race 2 - TEST - 2026-07-09"
    plan["races"].append(blocked)

    with pytest.raises(ValueError, match="plan_status_counts_mismatch"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_rejects_candidate_count_mismatch(tmp_path):
    plan = _build_capture_plan(tmp_path)
    plan["candidate_race_count"] = 2

    with pytest.raises(ValueError, match="plan_candidate_race_count_mismatch"):
        fixture.ready_plan_items(plan)


def test_ready_plan_items_rejects_ready_count_without_races_container():
    plan = {
        "schema_version": "autonomous_live_odds_capture_plan_v1",
        "ready_count": 1,
    }

    with pytest.raises(ValueError, match="plan_races_missing"):
        fixture.ready_plan_items(plan)


def test_build_fixture_packet_refuses_existing_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )

    with pytest.raises(FileExistsError):
        fixture.build_fixture_packet(
            plan=_plan(tmp_path),
            fetch_payload=_fetch_result(),
            output_dir=output_dir,
            current_time=_current_time(),
        )


def test_repeated_builds_emit_identical_sealed_contract_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    first_output = _output_dir(tmp_path)
    second_output = first_output.with_name(
        "strict_win_odds_fixture_capture_repeat_report_only"
    )

    fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(),
        output_dir=first_output,
        current_time=_current_time(),
    )
    fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(),
        output_dir=second_output,
        current_time=_current_time(),
    )

    for role in ("raw_fixture", "normalized_projection"):
        _, _, first_path = _manifest_entry(first_output, role)
        _, _, second_path = _manifest_entry(second_output, role)
        assert first_path.read_bytes() == second_path.read_bytes()
    assert (first_output / "strict_win_fixture_manifest.json").read_bytes() == (
        second_output / "strict_win_fixture_manifest.json"
    ).read_bytes()

    manifest, raw_entry, raw_path = _manifest_entry(first_output, "raw_fixture")
    _, projection_entry, _ = _manifest_entry(first_output, "normalized_projection")
    raw_fixture = json.loads(raw_path.read_text(encoding="utf-8"))
    assert (
        raw_fixture["fixture_id"]
        == "4457a73c0d90e9469ce172ef03804ce1d320c82d5ee061d4d4498d806b9c85d7"
    )
    assert (
        raw_entry["sha256"]
        == "8effcc9f8cc9b24ff9cbb2a87b8ffd667942d773740881e26faa860cebab82bc"
    )
    assert (
        projection_entry["sha256"]
        == "2eb2d494e959077397573c12ff9120633795e873df897bec4c462ce539eb5070"
    )
    assert (
        manifest["manifest_sha256"]
        == "2505ab0341470590fc8795bc4c75da89dd0c8e69bbc71603eb62c8c3e6ab363e"
    )
    assert (
        fixture.sha256_file(first_output / "strict_win_fixture_manifest.json")
        == "89073f67ebd88df2843644d4a8e3f9ed777dd274f2df925b5c5bddb41758dd20"
    )


def test_fixture_identity_is_independent_of_repository_checkout_path(
    tmp_path, monkeypatch
):
    fixtures = []
    for checkout_name in ("checkout_a", "checkout_b"):
        checkout_root = tmp_path / checkout_name
        checkout_root.mkdir()
        monkeypatch.setattr(fixture, "ROOT", checkout_root)
        plan = _plan(checkout_root)
        fixtures.append(
            fixture.build_raw_fixture(
                plan_item=plan["items"][0],
                fetch_result=_fetch_result(),
            )
        )

    assert fixtures[0]["provenance"]["collector"] == fixture.COLLECTOR_PATH
    assert fixture.canonical_bytes(fixtures[0]) == fixture.canonical_bytes(fixtures[1])


def test_build_fixture_packet_rejects_outcome_fields_before_writing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    with pytest.raises(ValueError, match="raw_fetch_result_contains_outcome_fields"):
        fixture.build_fixture_packet(
            plan=_plan(tmp_path),
            fetch_payload=_fetch_result(winner="Alpha"),
            output_dir=output_dir,
            current_time=_current_time(),
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("field", ["dividend", "margin", "payout"])
def test_build_fixture_packet_rejects_nested_outcome_fields_before_writing(
    tmp_path, monkeypatch, field
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fetch_result = _fetch_result()
    fetch_result["race_info"]["settlement"] = {field: 4.2}

    with pytest.raises(ValueError, match="raw_fetch_result_contains_outcome_fields"):
        fixture.build_fixture_packet(
            plan=_plan(tmp_path),
            fetch_payload=fetch_result,
            output_dir=output_dir,
            current_time=_current_time(),
        )

    assert not output_dir.exists()


def test_output_dir_guard_rejects_paths_outside_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    outside_repo = tmp_path.parent / f"{tmp_path.name}_outside"

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        fixture.assert_output_dir_safe(
            outside_repo / "artifacts/full_evidence_orchestration_20260525/"
            "strict_win_odds_fixture_capture_report_only"
        )


def test_output_dir_guard_rejects_in_repo_non_artifact_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="output_dir_must_be_under_artifacts"):
        fixture.assert_output_dir_safe(tmp_path / "reports/strict_win_fixture")


def test_output_dir_guard_rejects_artifact_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    artifact_root = tmp_path / "artifacts/full_evidence_orchestration_20260525"
    artifact_root.mkdir(parents=True)
    outside_repo = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_repo.mkdir()
    escape = artifact_root / "strict_win_odds_fixture_capture_escape"
    escape.symlink_to(outside_repo, target_is_directory=True)

    with pytest.raises(ValueError, match="output_dir_must_be_inside_repo"):
        fixture.assert_output_dir_safe(escape / "child")


def test_validator_rejects_count_only_capture_without_raw_runner_odds(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fetch_result = _fetch_result(odds_data=[])

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=fetch_result,
        output_dir=output_dir,
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    reasons = report["fixture_results"][0]["reasons"]
    assert "count_only_capture_no_raw_runner_odds" in reasons
    assert report["db_append_performed"] is False


@pytest.mark.parametrize(
    ("odds_data", "expected_reason"),
    [
        (
            [
                {
                    "box_number": 1,
                    "dog_name": "Alpha",
                    "odds_decimal": 2.4,
                    "sportsbet_raw_runner_text": "1. Alpha",
                },
                {
                    "box_number": 1,
                    "dog_name": "Alpha",
                    "odds_decimal": 2.5,
                    "sportsbet_raw_runner_text": "1. Alpha duplicate",
                },
            ],
            "duplicate_runner_boxes",
        ),
        (
            [
                {
                    "box_number": 1,
                    "dog_name": "Alpha",
                    "odds_decimal": 1.0,
                    "sportsbet_raw_runner_text": "1. Alpha",
                },
                {
                    "box_number": 2,
                    "dog_name": "Bravo",
                    "odds_decimal": 3.1,
                    "sportsbet_raw_runner_text": "2. Bravo",
                },
            ],
            "runner_0_invalid_odds_decimal",
        ),
        (
            [
                {
                    "box_number": 1,
                    "dog_name": "Alpha",
                    "odds_decimal": 2.4,
                    "sportsbet_raw_runner_text": "",
                },
                {
                    "box_number": 2,
                    "dog_name": "Bravo",
                    "odds_decimal": 3.1,
                    "sportsbet_raw_runner_text": "2. Bravo",
                },
            ],
            "runner_0_raw_runner_text_missing",
        ),
    ],
)
def test_validator_rejects_strict_runner_failures(
    tmp_path, monkeypatch, odds_data, expected_reason
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(odds_data=odds_data),
        output_dir=output_dir,
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert expected_reason in report["fixture_results"][0]["reasons"]
    assert report["db_append_performed"] is False


def test_validator_rejects_naive_capture_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(capture_timestamp="2026-07-09T12:20:00"),
        output_dir=output_dir,
        current_time=_current_time(),
    )

    reasons = report["fixture_results"][0]["reasons"]
    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert "capture_timestamp_not_timezone_aware" in reasons


def test_validator_rejects_missing_capture_timestamp_without_synthesizing_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fetch_result = _fetch_result()
    fetch_result.pop("capture_timestamp")

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=fetch_result,
        output_dir=output_dir,
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert (
        "capture_timestamp_missing_or_invalid"
        in report["fixture_results"][0]["reasons"]
    )
    _, _, raw_path = _manifest_entry(output_dir, "raw_fixture")
    raw_fixture = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw_fixture["capture_timestamp"] in (None, "")


@pytest.mark.parametrize("sidecar_kind", ["missing", "outside_symlink"])
def test_validator_requires_repository_contained_source_sidecar(
    tmp_path, monkeypatch, sidecar_kind
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    sidecar = tmp_path / plan["items"][0]["sidecar_path"]
    if sidecar_kind == "missing":
        sidecar.unlink()
    else:
        outside = tmp_path.parent / f"{tmp_path.name}_outside_sidecar.json"
        outside.write_text('{"source":"outside"}', encoding="utf-8")
        sidecar.unlink()
        sidecar.symlink_to(outside)

    report = fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert (
        "source_sidecar_missing_or_untrusted" in report["fixture_results"][0]["reasons"]
    )


@pytest.mark.parametrize(
    "source_url",
    [
        "https://sportsbet.com.au.evil.example/greyhound-racing/test/race-1",
        "https://notsportsbet.com.au/greyhound-racing/test/race-1",
        "http://www.sportsbet.com.au/greyhound-racing/test/race-1",
    ],
)
def test_validator_rejects_lookalike_or_non_https_sportsbet_urls(
    tmp_path, monkeypatch, source_url
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(race_info={"venue_url": source_url}),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert "sportsbet_source_url_untrusted" in report["fixture_results"][0]["reasons"]


def test_validator_accepts_exact_sportsbet_host_and_real_subdomain():
    assert fixture.source_url_is_trusted_sportsbet(
        "https://sportsbet.com.au/greyhound-racing/test/race-1"
    )
    assert fixture.source_url_is_trusted_sportsbet(
        "https://www.sportsbet.com.au/greyhound-racing/test/race-1"
    )


@pytest.mark.parametrize(
    ("expected_runners", "expected_reason"),
    [
        (
            [
                {"box_number": 1, "dog_name": "Alpha"},
                {"box_number": 1, "dog_name": "Bravo"},
            ],
            "duplicate_expected_runner_boxes",
        ),
        (
            [
                {"box_number": 1, "dog_name": "1. Alpha"},
                {"box_number": 2, "dog_name": "Alpha"},
            ],
            "duplicate_expected_runner_identities",
        ),
    ],
)
def test_validator_rejects_duplicate_expected_runners(
    tmp_path, monkeypatch, expected_runners, expected_reason
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    plan["items"][0]["expected_runners"] = expected_runners

    report = fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert expected_reason in report["fixture_results"][0]["reasons"]


def test_validator_rejects_post_jump_and_post_race_source(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(
            capture_timestamp="2026-07-09T12:31:00+10:00",
            race_info={
                "venue_url": "https://www.sportsbet.com.au/greyhound-racing/results/test/race-1"
            },
        ),
        output_dir=output_dir,
        current_time=_current_time(),
    )

    reasons = report["fixture_results"][0]["reasons"]
    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert "sportsbet_source_url_post_race" in reasons
    assert "capture_timestamp_not_before_jump" in reasons


def test_validate_packet_detects_fixture_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest = json.loads(
        (output_dir / "strict_win_fixture_manifest.json").read_text(encoding="utf-8")
    )
    raw_entry = next(row for row in manifest["files"] if row["role"] == "raw_fixture")
    raw_path = output_dir / raw_entry["path"]
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["runner_rows"][0]["odds_decimal"] = 99.0
    raw_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert any(
        reason.startswith("manifest_file_sha256_mismatch")
        for reason in packet_validation["reasons"]
    )


def test_validate_packet_recomputes_fixture_content_id(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, raw_entry, raw_path = _manifest_entry(output_dir, "raw_fixture")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    payload["runner_rows"][0]["odds_decimal"] = 2.5
    raw_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raw_entry["sha256"] = fixture.sha256_file(raw_path)
    raw_entry["bytes"] = raw_path.stat().st_size
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert (
        "fixture_id_content_mismatch"
        in packet_validation["fixture_validations"][0]["reasons"]
    )


def test_validate_packet_requires_exact_derived_projection(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, projection_entry, projection_path = _manifest_entry(
        output_dir, "normalized_projection"
    )
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    projection["rows"][0]["odds_decimal"] = 99.0
    projection_path.write_text(
        json.dumps(projection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    projection_entry["sha256"] = fixture.sha256_file(projection_path)
    projection_entry["bytes"] = projection_path.stat().st_size
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert (
        "projection_content_mismatch"
        in packet_validation["fixture_validations"][0]["reasons"]
    )


def test_fixture_validation_rechecks_source_sidecar_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    raw_fixture = fixture.build_raw_fixture(
        plan_item=plan["items"][0],
        fetch_result=_fetch_result(),
    )
    raw_fixture["provenance"]["source_sidecar_sha256"] = "0" * 64
    raw_fixture["fixture_id"] = fixture.fixture_content_id(raw_fixture)
    projection = fixture.normalized_projection_from_fixture(raw_fixture)

    validation = fixture.validate_fixture_payload(raw_fixture, projection)

    assert validation["status"] == "BLOCKED"
    assert "source_sidecar_sha256_mismatch" in validation["reasons"]


def test_v1_fixture_schema_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    raw_fixture = fixture.build_raw_fixture(
        plan_item=plan["items"][0],
        fetch_result=_fetch_result(),
    )
    raw_fixture["schema_version"] = "strict_win_odds_raw_replay_fixture_v1"
    raw_fixture["fixture_id"] = fixture.fixture_content_id(raw_fixture)
    projection = fixture.normalized_projection_from_fixture(raw_fixture)

    validation = fixture.validate_fixture_payload(raw_fixture, projection)

    assert validation["status"] == "BLOCKED"
    assert "fixture_schema_version_invalid" in validation["reasons"]


def test_validate_packet_rejects_manifest_path_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, raw_entry, _ = _manifest_entry(output_dir, "raw_fixture")
    outside = tmp_path / "outside_fixture.json"
    outside.write_text("{}\n", encoding="utf-8")
    raw_entry["path"] = os.path.relpath(outside, output_dir)
    raw_entry["sha256"] = fixture.sha256_file(outside)
    raw_entry["bytes"] = outside.stat().st_size
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert any(
        reason.startswith("manifest_file_path_outside_packet:")
        for reason in packet_validation["reasons"]
    )


def test_validate_packet_rejects_manifest_symlink_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, raw_entry, _ = _manifest_entry(output_dir, "raw_fixture")
    outside = tmp_path / "outside_fixture.json"
    outside.write_text("{}\n", encoding="utf-8")
    escape = output_dir / "fixture_escape.json"
    escape.symlink_to(outside)
    raw_entry["path"] = escape.relative_to(output_dir).as_posix()
    raw_entry["sha256"] = fixture.sha256_file(outside)
    raw_entry["bytes"] = outside.stat().st_size
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert any(
        reason.startswith("manifest_file_path_outside_packet:")
        for reason in packet_validation["reasons"]
    )


def test_validate_packet_rejects_manifest_itself_as_symlink_escape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest_path = output_dir / "strict_win_fixture_manifest.json"
    outside_manifest = tmp_path / "outside_manifest.json"
    manifest_path.replace(outside_manifest)
    manifest_path.symlink_to(outside_manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert packet_validation["reasons"] == ["manifest_path_outside_packet"]


def test_validate_packet_verifies_manifest_file_size(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, raw_entry, _ = _manifest_entry(output_dir, "raw_fixture")
    raw_entry["bytes"] += 1
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert any(
        reason.startswith("manifest_file_size_mismatch:")
        for reason in packet_validation["reasons"]
    )


def test_validate_packet_reports_malformed_projection_json(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, projection_entry, projection_path = _manifest_entry(
        output_dir, "normalized_projection"
    )
    projection_path.write_text("{not-json\n", encoding="utf-8")
    projection_entry["sha256"] = fixture.sha256_file(projection_path)
    projection_entry["bytes"] = projection_path.stat().st_size
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert any(
        reason.startswith("manifest_file_json_invalid:")
        for reason in packet_validation["reasons"]
    )


def test_validate_packet_accepts_packet_directory_symlink_inside_repo(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    packet_alias = tmp_path / "packet_alias"
    packet_alias.symlink_to(output_dir, target_is_directory=True)

    packet_validation = fixture.validate_packet(packet_alias)

    assert packet_validation["status"] == "PASS"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("raw_entry_fixture_id", "manifest_raw_fixture_id_mismatch"),
        ("projection_entry_fixture_id", "manifest_projection_fixture_id_mismatch"),
        ("fixture_count", "manifest_fixture_count_mismatch"),
        ("duplicate_projection", "manifest_projection_fixture_id_duplicate"),
    ],
)
def test_validate_packet_cross_checks_manifest_fixture_bookkeeping(
    tmp_path, monkeypatch, mutation, expected_reason
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest = json.loads(
        (output_dir / "strict_win_fixture_manifest.json").read_text(encoding="utf-8")
    )
    if mutation == "raw_entry_fixture_id":
        next(row for row in manifest["files"] if row["role"] == "raw_fixture")[
            "fixture_id"
        ] = "forged"
    elif mutation == "projection_entry_fixture_id":
        next(
            row for row in manifest["files"] if row["role"] == "normalized_projection"
        )["fixture_id"] = "forged"
    elif mutation == "fixture_count":
        manifest["fixture_count"] = 0
    else:
        projection_entry = next(
            row for row in manifest["files"] if row["role"] == "normalized_projection"
        )
        manifest["files"].append(dict(projection_entry))
    _reseal_manifest(output_dir, manifest)

    packet_validation = fixture.validate_packet(output_dir)

    assert packet_validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert expected_reason in packet_validation["reasons"]


def test_source_bytes_are_bound_once_per_capture_and_replay_boundary(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    fetch_path = tmp_path / "fetch.json"
    plan_bytes = fixture.serialized_json_bytes(plan)
    fetch_bytes = fixture.serialized_json_bytes(_fetch_result())
    plan_path.write_bytes(plan_bytes)
    fetch_path.write_bytes(fetch_bytes)
    changed_plan_bytes = plan_bytes.replace(
        b'"candidate_race_count": 1', b'"candidate_race_count": 2', 1
    )
    assert len(changed_plan_bytes) == len(plan_bytes)
    original_read = fixture.read_file_bytes
    reads: dict[Path, int] = {}

    def changing_read(path: Path) -> bytes:
        resolved = path.resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        if resolved == plan_path.resolve() and reads[resolved] == 2:
            return changed_plan_bytes
        return original_read(path)

    monkeypatch.setattr(fixture, "read_file_bytes", changing_read)
    report = fixture.build_fixture_packet(
        plan={},
        fetch_payload={},
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
        plan_path=plan_path,
        fetch_result_path=fetch_path,
    )

    assert reads[plan_path.resolve()] == 2
    assert "plan_source_sha256_mismatch" in report["packet_validation"]["reasons"]
    manifest = json.loads(
        (_output_dir(tmp_path) / fixture.MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert manifest["plan_sha256"] == fixture.sha256_bytes(plan_bytes)


def test_valid_source_files_are_hash_bound_and_replay_from_their_bytes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    fetch = _fetch_result()
    plan_path = tmp_path / "plan.json"
    fetch_path = tmp_path / "fetch.json"
    plan_bytes = fixture.serialized_json_bytes(plan)
    fetch_bytes = fixture.serialized_json_bytes(fetch)
    plan_path.write_bytes(plan_bytes)
    fetch_path.write_bytes(fetch_bytes)

    report = fixture.build_fixture_packet(
        plan={},
        fetch_payload={},
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
        plan_path=plan_path,
        fetch_result_path=fetch_path,
    )

    manifest = json.loads((_output_dir(tmp_path) / fixture.MANIFEST_PATH).read_bytes())
    assert report["status"] == fixture.FINAL_SEALED_NO_DB_APPEND
    assert report["packet_validation"]["status"] == "PASS"
    assert manifest["plan_sha256"] == fixture.sha256_bytes(plan_bytes)
    assert manifest["fetch_result_sha256"] == fixture.sha256_bytes(fetch_bytes)


def test_manifest_hashes_are_reproduced_from_exact_written_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest_bytes = (output_dir / fixture.MANIFEST_PATH).read_bytes()
    manifest = json.loads(manifest_bytes)

    for entry in manifest["files"]:
        payload_bytes = (output_dir / entry["path"]).read_bytes()
        assert entry["bytes"] == len(payload_bytes)
        assert entry["sha256"] == fixture.sha256_bytes(payload_bytes)


@pytest.mark.parametrize(
    "role",
    [
        "raw_fixture",
        "normalized_projection",
        "preseal_validation",
        "packet_report",
        "final_status",
        "manifest",
    ],
)
def test_validate_packet_rejects_forbidden_outcome_fields_in_every_json_role(
    tmp_path, monkeypatch, role
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    if role == "manifest":
        manifest_path = output_dir / fixture.MANIFEST_PATH
        manifest = json.loads(manifest_path.read_bytes())
        manifest["nested"] = {"official_result": {"winner": "synthetic"}}
        _reseal_manifest(output_dir, manifest)
    else:
        _, _, path = _manifest_entry(output_dir, role)
        payload = json.loads(path.read_bytes())
        payload["nested"] = {"outcome": "synthetic"}
        _rewrite_manifest_payload(output_dir, role, payload)

    validation = fixture.validate_packet(output_dir)

    assert validation["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert any(
        reason.startswith("forbidden_outcome_field:")
        for reason in validation["reasons"]
    )


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("place", "source_market_type_not_win"),
        ("wrong_url", "source_source_url_mismatch"),
        ("wrong_race", "source_race_number_mismatch"),
        ("wrong_roster", "source_runner_2_identity_mismatch"),
        ("missing_runner", "missing_expected_runner_boxes"),
        ("extra_runner", "extra_unexpected_runner_boxes"),
    ],
)
def test_capture_reconciles_exact_sportsbet_win_race(
    tmp_path, monkeypatch, mutation, expected_reason
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    fetch = _fetch_result()
    if mutation == "place":
        fetch["market_type"] = "place"
    elif mutation == "wrong_url":
        fetch["race_info"]["venue_url"] = (
            "https://www.sportsbet.com.au/greyhound-racing/test/race-2"
        )
    elif mutation == "wrong_race":
        fetch["race_info"]["race_number"] = 2
    elif mutation == "wrong_roster":
        fetch["odds_data"][1]["dog_name"] = "Charlie"
    elif mutation == "missing_runner":
        fetch["odds_data"].pop()
    else:
        fetch["odds_data"].append(
            {
                "box_number": 3,
                "dog_name": "Charlie",
                "odds_decimal": 5.0,
                "sportsbet_raw_runner_text": "3. Charlie",
                "active": True,
            }
        )

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=fetch,
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert expected_reason in report["fixture_results"][0]["reasons"]


def test_capture_rejects_priced_scratched_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    plan["items"][0]["expected_runners"][1]["active"] = False
    sidecar_path = tmp_path / plan["items"][0]["sidecar_path"]
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["expected_runners"][1]["active"] = False
    sidecar_path.write_bytes(fixture.serialized_json_bytes(sidecar))
    fetch = _fetch_result()
    fetch["odds_data"][1]["active"] = False

    report = fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=fetch,
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert (
        "runner_1_scratched_runner_has_price" in report["fixture_results"][0]["reasons"]
    )


def test_capture_rejects_explicitly_unsuccessful_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(success=False),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert "source_fetch_not_successful" in report["fixture_results"][0]["reasons"]


def test_capture_rejects_sidecar_roster_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    sidecar_path = tmp_path / plan["items"][0]["sidecar_path"]
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["expected_runners"][1]["dog_name"] = "Charlie"
    sidecar_path.write_bytes(fixture.serialized_json_bytes(sidecar))

    report = fixture.build_fixture_packet(
        plan=plan,
        fetch_payload=_fetch_result(),
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_BLOCKED_VALIDATION_FAILED
    assert (
        "sidecar_runner_2_identity_mismatch" in report["fixture_results"][0]["reasons"]
    )


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            lambda: {"fetch_results": [_fetch_result(), _fetch_result()]},
            "duplicate_fetch_record_identities",
        ),
        (
            lambda: {"fetch_results": [_fetch_result(), "malformed"]},
            "fetch_record_malformed:fetch_results:1",
        ),
        (
            lambda: {"fetch_results": [{"success": False}]},
            "fetch_record_identity_missing:0",
        ),
        (
            lambda: {
                "fetch_results": [
                    _fetch_result(odds_data=[_fetch_result()["odds_data"][0], None])
                ]
            },
            "fetch_runner_record_malformed:0:1",
        ),
    ],
)
def test_capture_rejects_duplicate_and_malformed_candidate_records(
    tmp_path, monkeypatch, payload, error
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)

    with pytest.raises(ValueError, match=error):
        fixture.build_fixture_packet(
            plan=_plan(tmp_path),
            fetch_payload=payload(),
            output_dir=output_dir,
            current_time=_current_time(),
        )

    assert not output_dir.exists()


def test_capture_rejects_competing_candidates_for_one_plan_item(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    plan["items"][0]["race_id_aliases"] = ["SECONDARY-ID"]
    primary = _fetch_result()
    secondary = _fetch_result(alias_race_id="OTHER", race_id="SECONDARY-ID")

    with pytest.raises(ValueError, match="competing_fetch_records_for_ready_race"):
        fixture.build_fixture_packet(
            plan=plan,
            fetch_payload={"fetch_results": [primary, secondary]},
            output_dir=_output_dir(tmp_path),
            current_time=_current_time(),
        )


def test_capture_rejects_one_candidate_matching_multiple_ready_races(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    plan = _plan(tmp_path)
    second_item = json.loads(json.dumps(plan["items"][0]))
    second_item["canonical_race_identity"] = "TEST_2026-07-09_1"
    plan["items"].append(second_item)
    plan["candidate_race_count"] = 2
    plan["ready_to_capture_race_count"] = 2
    output_dir = _output_dir(tmp_path)

    with pytest.raises(ValueError, match="fetch_record_matches_multiple_ready_races"):
        fixture.build_fixture_packet(
            plan=plan,
            fetch_payload=_fetch_result(),
            output_dir=output_dir,
            current_time=_current_time(),
        )

    assert not output_dir.exists()


def test_capture_ignores_well_formed_unrelated_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    unrelated = _fetch_result(
        alias_race_id="Race 9 - OTHER - 2026-07-09",
        race_id="OTHER_2026-07-09_9",
    )

    report = fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload={"fetch_results": [unrelated, _fetch_result()]},
        output_dir=_output_dir(tmp_path),
        current_time=_current_time(),
    )

    assert report["status"] == fixture.FINAL_SEALED_NO_DB_APPEND


def test_validate_packet_rejects_alternate_manifest_path(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest, raw_entry, raw_path = _manifest_entry(output_dir, "raw_fixture")
    alternate_path = raw_path.with_name("alternate_raw_fixture.json")
    raw_path.rename(alternate_path)
    raw_entry["path"] = alternate_path.relative_to(output_dir).as_posix()
    _reseal_manifest(output_dir, manifest)

    validation = fixture.validate_packet(output_dir)

    assert any(
        reason.startswith("manifest_role_path_invalid:raw_fixture:")
        for reason in validation["reasons"]
    )


def test_validate_packet_rejects_undeclared_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    (output_dir / "alternate.json").write_text("{}\n", encoding="utf-8")

    validation = fixture.validate_packet(output_dir)

    assert "packet_file_undeclared:alternate.json" in validation["reasons"]


def test_validate_packet_rejects_missing_declared_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    _, _, projection_path = _manifest_entry(output_dir, "normalized_projection")
    projection_path.unlink()

    validation = fixture.validate_packet(output_dir)

    assert any(
        reason.startswith("packet_file_missing:normalized/")
        for reason in validation["reasons"]
    )


def test_validate_packet_rejects_role_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    manifest = json.loads((output_dir / fixture.MANIFEST_PATH).read_bytes())
    preseal = next(
        row for row in manifest["files"] if row["role"] == "preseal_validation"
    )
    preseal["role"] = "packet_report"
    _reseal_manifest(output_dir, manifest)

    validation = fixture.validate_packet(output_dir)

    assert "manifest_role_conflict:packet_report:" in validation["reasons"]


def test_validate_packet_rejects_non_regular_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "ROOT", tmp_path)
    output_dir = _output_dir(tmp_path)
    fixture.build_fixture_packet(
        plan=_plan(tmp_path),
        fetch_payload=_fetch_result(),
        output_dir=output_dir,
        current_time=_current_time(),
    )
    (output_dir / "alias.json").symlink_to(output_dir / fixture.FINAL_STATUS_PATH)

    validation = fixture.validate_packet(output_dir)

    assert "packet_non_regular_entry:alias.json" in validation["reasons"]
