from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_form_only_v1_packet.py"
SPEC = importlib.util.spec_from_file_location("build_form_only_v1_packet", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def form_rows(participants: list[tuple[int, str]]) -> list[dict[str, object]]:
    rows = []
    for box, name in participants:
        rows.append({
            "Dog Name": f"{box}. {name}", "DATE": "2026-07-01", "TRACK": "BALLARAT",
            "DIST": "450", "G": "Grade 5", "PLC": str(box), "BOX": str(box), "MGN": "1.5",
        })
    return rows


FORM_FIELDS = ["Dog Name", "DATE", "TRACK", "DIST", "G", "PLC", "BOX", "MGN"]


def make_card(
    path: Path,
    race_id: str,
    participants: list[tuple[int, str]],
    *,
    capture: str,
    jump: str,
) -> tuple[Path, Path]:
    write_csv(path, FORM_FIELDS, form_rows(participants), delimiter="|")
    match = MODULE.re.fullmatch(r"Race (\d+) - (.+) - (\d{4}-\d{2}-\d{2})", race_id)
    assert match
    jump_dt = MODULE.parse_timestamp(jump)
    url = (
        f"https://www.thedogs.com.au/racing/{match.group(2).lower()}/"
        f"{match.group(3)}/{int(match.group(1))}/fixture"
    )
    metadata = {
        "metadata_is_leakage_safe": True,
        "metadata_captured_at": capture,
        "content_sha256": MODULE.sha256_path(path),
        "content_length": path.stat().st_size,
        "target_distance": 450,
        "target_grade": "Grade 5/6",
        "race_info": {
            "date": match.group(3),
            "venue": match.group(2),
            "race_number": int(match.group(1)),
            "race_time": jump_dt.strftime("%I:%M %p"),
            "race_time_mapping_status": "exact_url_match",
            "race_time_source": "canonical_race_url",
            "url": url,
        },
        "race_url": url,
        "metadata_source_url": url,
        "runner_completeness": {
            "status": "COMPLETE",
            "runner_count": len(participants),
            "participants": [
                {"box_number": box, "dog_name": name} for box, name in participants
            ],
        },
    }
    sidecar = Path(str(path) + ".metadata.json")
    write_json(sidecar, metadata)
    return path, sidecar


def record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": MODULE.sha256_path(path), "bytes": path.stat().st_size}


def make_fixture(
    tmp_path: Path,
    *,
    oot_date: str = "2026-07-11",
    oot_capture: str = "2026-07-11T11:00:00+10:00",
    oot_jump: str = "2026-07-11T12:00:00+10:00",
) -> dict[str, Path]:
    eligibility = tmp_path / "eligibility"
    training = tmp_path / "training"
    evidence = tmp_path / "evidence"
    freeze = tmp_path / "freeze"
    race_path = eligibility / "historical_win_eligibility_races_v1.csv"
    runner_path = eligibility / "historical_win_eligibility_runners_v1.csv"
    provenance_path = eligibility / "historical_win_tier_a_race_provenance_v1.json"
    training_path = training / "thedogs_training_rows_v1.csv"

    race_one = "Race 1 - BAL - 2026-07-09"
    race_two = "Race 2 - GEE - 2026-07-08"
    card_one, sidecar_one = make_card(
        evidence / "upcoming" / f"{race_one}.csv",
        race_one,
        [(1, "Same Dog"), (2, "Other Dog")],
        capture="2026-07-09T10:00:00+10:00",
        jump="2026-07-09T12:00:00+10:00",
    )
    card_two, sidecar_two = make_card(
        evidence / "upcoming" / f"{race_two}.csv",
        race_two,
        [(1, "Same Dog"), (2, "Scratched Dog")],
        capture="2026-07-08T10:00:00+10:00",
        jump="2026-07-08T12:00:00+10:00",
    )
    official_races = tmp_path / "labels" / "official_races.jsonl"
    official_runners = tmp_path / "labels" / "official_runners.jsonl"
    official_races.parent.mkdir(parents=True, exist_ok=True)
    official_races.write_text("official-race-label-proof\n", encoding="utf-8")
    official_runners.write_text("official-runner-label-proof\n", encoding="utf-8")
    shadow_path = tmp_path / "shadow" / "shadow_feature_rows.json"
    write_json(shadow_path, [
        {
            "race_id": race_one, "dog_name": "Same Dog", "box_number": 1,
            "prior_start_count": 1, "days_since_last_start": 8,
            "target_grade_normalized": "Grade 5/6", "target_grade_missing": 0,
        },
        {
            "race_id": race_one, "dog_name": "Other Dog", "box_number": 2,
            "prior_start_count": 1, "days_since_last_start": 8,
            "target_grade_normalized": "5/6", "target_grade_missing": 0,
        },
    ])

    write_csv(race_path, ["race_id", "used_for_training"], [
        {"race_id": race_one, "used_for_training": "1"},
        {"race_id": race_two, "used_for_training": "1"},
    ])
    write_csv(
        runner_path,
        ["race_id", "strongest_tier", "box_number", "dog_name", "runner_id"],
        [
            {"race_id": race_one, "strongest_tier": "A", "box_number": 1, "dog_name": "Same Dog", "runner_id": "sealed-a"},
            {"race_id": race_one, "strongest_tier": "A", "box_number": 2, "dog_name": "Other Dog", "runner_id": "sealed-b"},
        ],
    )
    training_fields = [
        "race_id", "csv_dog_name", "box_number", "source_csv_path", "source_csv_sha256",
        "race_date", "race_timestamp_utc", "target_distance", "target_grade",
        "listed_participants_count", "active_field_size", "scratched_runner_count",
        "reserve_runner_count", "has_scratch_or_reserve", "runner_status", "runner_id", "odds_url",
    ]
    training_rows = [
        {
            "race_id": race_one, "csv_dog_name": name, "box_number": box,
            "source_csv_path": str(card_one), "source_csv_sha256": MODULE.sha256_path(card_one),
            "race_date": "2026-07-09", "race_timestamp_utc": "2026-07-09T02:00:00+00:00",
            "target_distance": "450", "target_grade": "5/6", "listed_participants_count": "2",
            "active_field_size": "2", "scratched_runner_count": "0", "reserve_runner_count": "0",
            "has_scratch_or_reserve": "0", "runner_status": "active",
            "runner_id": f"sealed-{box}", "odds_url": "",
        }
        for box, name in [(1, "Same Dog"), (2, "Other Dog")]
    ]
    training_rows.append({
        "race_id": race_two, "csv_dog_name": "Same Dog", "box_number": 1,
        "source_csv_path": str(card_two), "source_csv_sha256": MODULE.sha256_path(card_two),
        "race_date": "2026-07-08", "race_timestamp_utc": "2026-07-08T02:00:00+00:00",
        "target_distance": "450", "target_grade": "Grade 5/6", "listed_participants_count": "2",
        "active_field_size": "1", "scratched_runner_count": "1", "reserve_runner_count": "0",
        "has_scratch_or_reserve": "1", "runner_status": "active", "runner_id": "sealed-c", "odds_url": "",
    })
    write_csv(training_path, training_fields, training_rows)
    provenance = {
        "races": {
            race_one: {
                "source_csv_path": str(card_one), "source_csv_sha256": MODULE.sha256_path(card_one),
                "sidecar_path": str(sidecar_one), "sidecar_sha256": MODULE.sha256_path(sidecar_one),
                "jump_timestamp": "2026-07-09T12:00:00+10:00",
                "official_race_artifact_path": str(official_races),
                "official_race_artifact_sha256": MODULE.sha256_path(official_races),
                "official_runner_artifact_path": str(official_runners),
                "official_runner_artifact_sha256": MODULE.sha256_path(official_runners),
                "official_urls": [], "feature_source_paths": [str(shadow_path)],
                "feature_source_sha256": [MODULE.sha256_path(shadow_path)],
            }
        }
    }
    write_json(provenance_path, provenance)

    oot_race = f"Race 3 - BAL - {oot_date}"
    oot_card, oot_sidecar = make_card(
        evidence / "upcoming" / f"{oot_race}.csv",
        oot_race,
        [(1, "Same Dog")],
        capture=oot_capture,
        jump=oot_jump,
    )
    inventory_fields = [
        "race_id", "role", "path", "sha256", "bytes", "capture_timestamp",
        "jump_timestamp", "status",
    ]
    write_csv(freeze / "out_of_time_source_inventory.csv", inventory_fields, [
        {
            "race_id": oot_race, "role": "raw_pre_race_card", "path": str(oot_card),
            "sha256": MODULE.sha256_path(oot_card), "bytes": oot_card.stat().st_size,
            "capture_timestamp": oot_capture, "jump_timestamp": oot_jump,
            "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
        },
        {
            "race_id": oot_race, "role": "raw_pre_race_sidecar", "path": str(oot_sidecar),
            "sha256": MODULE.sha256_path(oot_sidecar), "bytes": oot_sidecar.stat().st_size,
            "capture_timestamp": oot_capture, "jump_timestamp": oot_jump,
            "status": "OUTCOME_UNOPENED_OUT_OF_TIME",
        },
    ])
    write_csv(
        freeze / "out_of_time_exclusions.csv",
        ["race_id", "race_date", "source_path", "source_sha256", "source_bytes", "reason"],
        [],
    )
    write_json(freeze / "out_of_time_manifest.json", {
        "status": "OUTCOME_UNOPENED_OUT_OF_TIME", "outcomes_opened": False,
        "window_start": "2026-07-11", "window_end": "2026-08-09",
        "included_race_count": 1, "included_runner_count": 1,
    })

    top_files = {
        "eligibility_races": record(race_path),
        "eligibility_runners": record(runner_path),
        "tier_a_provenance": record(provenance_path),
        "training_rows": record(training_path),
    }
    dev_sources = [
        {"role": "development_card", **record(card_one)},
        {"role": "development_sidecar", **record(sidecar_one)},
        {"role": "development_label", **record(official_races)},
        {"role": "development_label", **record(official_runners)},
        {"role": "development_label", **record(training_path)},
        {"role": "development_card", **record(card_two)},
        {"role": "development_sidecar", **record(sidecar_two)},
        {"role": "shadow_reconciliation_source", **record(shadow_path)},
    ]
    freeze_records = [
        {"role": "out_of_time_freeze_source_inventory", **record(freeze / "out_of_time_source_inventory.csv")},
        {"role": "out_of_time_freeze_exclusions", **record(freeze / "out_of_time_exclusions.csv")},
        {"role": "out_of_time_freeze_manifest", **record(freeze / "out_of_time_manifest.json")},
    ]
    contract_path = tmp_path / "reproducibility.json"
    write_json(contract_path, {
        "schema_version": "form_only_v1_reproducibility_v4",
        "trusted_inputs": {
            "development": {
                "files": top_files, "authoritative_source_record_count": 7,
                "authoritative_source_set_sha256": MODULE.source_set_digest(dev_sources[:-1]),
            },
            "diagnostic": {
                "source_record_count": 1,
                "source_set_sha256": MODULE.source_set_digest(dev_sources[-1:]),
                "authority": "NON_AUTHORITATIVE_DIAGNOSTIC",
            },
            "out_of_time_freeze": {
                "path": str(freeze), "aggregate_sha256": MODULE.source_set_digest(freeze_records),
                "files": {
                    "source_inventory": {k: record(freeze / "out_of_time_source_inventory.csv")[k] for k in ("sha256", "bytes")},
                    "exclusions": {k: record(freeze / "out_of_time_exclusions.csv")[k] for k in ("sha256", "bytes")},
                    "manifest": {k: record(freeze / "out_of_time_manifest.json")[k] for k in ("sha256", "bytes")},
                },
            },
        },
        "expected_output": {},
    })
    return {
        "eligibility": eligibility, "training": training, "evidence": evidence,
        "freeze": freeze, "contract": contract_path, "oot_card": oot_card,
        "oot_sidecar": oot_sidecar, "card_one": card_one, "sidecar_one": sidecar_one,
        "card_two": card_two, "sidecar_two": sidecar_two,
        "official_races": official_races, "official_runners": official_runners,
        "shadow": shadow_path,
    }


def build_fixture(fixture: dict[str, Path], output: Path, *, enforce: bool) -> dict[str, object]:
    return MODULE.build_all(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], output,
        fixture["freeze"], fixture["contract"], enforce_expected_output=enforce,
    )


def bind_expected_output(fixture: dict[str, Path], summary: dict[str, object]) -> None:
    contract = json.loads(fixture["contract"].read_text(encoding="utf-8"))
    development = summary["development"]
    reconciliation = summary["reconciliation"]
    oot = summary["out_of_time"]
    manifests = summary["domain_manifests"]
    contract["expected_output"] = {
        "authoritative_counts": {
            "candidate_races": development["candidate_race_count"],
            "candidate_runners": development["candidate_runner_count"],
            "included_races": development["included_race_count"],
            "included_runners": development["included_runner_count"],
            "sidecar_only_exclusions": development["sidecar_only_runner_exclusion_count"],
            "out_of_time_races": oot["included_race_count"],
            "out_of_time_runners": oot["included_runner_count"],
        },
        "diagnostic_counts": {
            "overlap_races": reconciliation["overlap_race_count"],
            "overlap_runners": reconciliation["overlap_runner_count"],
            "history_differences": reconciliation["history_discrepancy_count"],
            "recency_differences": reconciliation["recency_discrepancy_count"],
            "grade_differences": reconciliation["grade_discrepancy_count"],
            "unexplained_differences": reconciliation["unexplained_mismatch_count"],
        },
        "domains": {
            domain: {
                "artifact_files": {row["path"]: row["sha256"] for row in manifest["files"]},
                "aggregate_sha256": manifest["aggregate_sha256"],
                "physical_aggregate_sha256": manifest["aggregate_sha256"],
                "declared_file_count": len(manifest["files"]),
                "artifacts": [
                    {
                        **row,
                        "type": "regular_file",
                        "role": MODULE.DOMAIN_ARTIFACT_ROLES[domain][row["path"]],
                    }
                    for row in manifest["files"]
                ],
            }
            for domain, manifest in manifests.items()
        },
    }
    write_json(fixture["contract"], contract)


def rebind_freeze(fixture: dict[str, Path]) -> None:
    contract = json.loads(fixture["contract"].read_text(encoding="utf-8"))
    freeze = fixture["freeze"]
    role_paths = {
        "source_inventory": freeze / "out_of_time_source_inventory.csv",
        "exclusions": freeze / "out_of_time_exclusions.csv",
        "manifest": freeze / "out_of_time_manifest.json",
    }
    records = []
    for role, path in role_paths.items():
        contract["trusted_inputs"]["out_of_time_freeze"]["files"][role] = {
            "sha256": MODULE.sha256_path(path), "bytes": path.stat().st_size,
        }
        records.append({"role": f"out_of_time_freeze_{role}", **record(path)})
    contract["trusted_inputs"]["out_of_time_freeze"]["aggregate_sha256"] = MODULE.source_set_digest(records)
    write_json(fixture["contract"], contract)


@pytest.mark.parametrize("raw", ["5/6", "Mixed 5/6", "Grade 5/6", "GRADE 5/6"])
def test_grade_aliases_are_unified(raw: str) -> None:
    assert MODULE.canonical_grade(raw) == "MIXED_5_6"


def test_history_normalizes_before_deduplication() -> None:
    rows = [
        {"DATE": "2026-07-01", "TRACK": "BAL", "DIST": "450", "G": "Grade 5", "PLC": "2", "BOX": "1", "MGN": "1.50"},
        {"DATE": "2026-07-01", "TRACK": "Ballarat", "DIST": "450.0", "G": "5th Grade", "PLC": "2.0", "BOX": "1.0", "MGN": "1.5"},
    ]
    accepted, rejected = MODULE.accepted_history(rows, date(2026, 7, 9))
    assert len(accepted) == 1
    assert [reason for reason, _row in rejected] == ["NORMALIZED_DUPLICATE_HISTORY"]


def test_unprovable_same_day_ordering_fails_closed() -> None:
    rows = [
        {"DATE": "2026-07-01", "TRACK": "BAL", "DIST": "450", "G": "5", "PLC": "1", "BOX": "1", "MGN": "0"},
        {"DATE": "2026-07-01", "TRACK": "GEE", "DIST": "460", "G": "6", "PLC": "5", "BOX": "2", "MGN": "4"},
    ]
    with pytest.raises(ValueError, match="unprovable same-day"):
        MODULE.accepted_history(rows, date(2026, 7, 9))


def test_verified_same_day_order_is_permutation_invariant() -> None:
    rows = [
        {"DATE": "2026-07-01", "RACE_NUMBER": "2", "TRACK": "GEE", "DIST": "460", "G": "6", "PLC": "5", "BOX": "2", "MGN": "4"},
        {"DATE": "2026-07-01", "RACE_NUMBER": "1", "TRACK": "BAL", "DIST": "450", "G": "5", "PLC": "1", "BOX": "1", "MGN": "0"},
    ]
    left, _ = MODULE.accepted_history(rows, date(2026, 7, 9))
    right, _ = MODULE.accepted_history(reversed(rows), date(2026, 7, 9))
    assert left == right
    assert [row["finish"] for row in left] == [5, 1]


def test_history_cap_and_recent_windows_are_newest_first() -> None:
    rows = [
        {"DATE": f"2026-06-{day:02d}", "TRACK": "BAL", "DIST": "450", "G": "5", "PLC": str((day % 8) + 1), "BOX": "1", "MGN": "1"}
        for day in range(1, 22)
    ]
    accepted, rejected = MODULE.accepted_history(rows, date(2026, 7, 9))
    assert len(accepted) == 20
    assert accepted[0]["date"] == date(2026, 6, 21)
    assert [reason for reason, _row in rejected].count("HISTORY_CAP_20") == 1
    feature = MODULE.feature_row("Race 1 - BAL - 2026-07-09", date(2026, 7, 9), "BAL", 450, "GRADE_5", 8, 1, "Dog", accepted)
    assert feature["prior_start_count"] == 20
    assert feature["recent_finish_mean_3"] != ""
    assert feature["recent_finish_mean_5"] != ""


def test_identity_is_derived_only_from_race_and_box() -> None:
    a = MODULE.row_id("Race 1 - BAL - 2026-07-09", 1, "Same Dog", scope="development")
    b = MODULE.row_id("Race 2 - BAL - 2026-07-09", 1, "Same Dog", scope="development")
    c = MODULE.row_id("Race 1 - BAL - 2026-07-09", 1, "Same Dog", scope="out_of_time")
    d = MODULE.row_id("Race 1 - BAL - 2026-07-09", 1, "Different Dog")
    assert len({a, b}) == 2
    assert a == c == d
    assert all("SAMEDOG" not in value for value in (a, b, c))


@pytest.mark.parametrize(
    ("capture", "accepted"),
    [("2026-07-11T11:00:00+10:00", True), ("2026-07-11T11:00:01+10:00", False)],
)
def test_t60_boundary_is_source_derived(tmp_path: Path, capture: str, accepted: bool) -> None:
    fixture = make_fixture(tmp_path, oot_capture=capture)
    if accepted:
        summary = build_fixture(fixture, tmp_path / "out", enforce=False)
        assert summary["out_of_time"]["included_race_count"] == 1
    else:
        with pytest.raises(ValueError, match="not available by T60"):
            build_fixture(fixture, tmp_path / "out", enforce=False)


def test_timezone_offsets_represent_same_instant_and_naive_capture_fails() -> None:
    assert MODULE.parse_timestamp("2026-07-11T11:00:00+10:00") == MODULE.parse_timestamp("2026-07-11T01:00:00Z")
    assert MODULE.parse_timestamp("2026-07-11T11:00:00") == MODULE.parse_timestamp("2026-07-11T11:00:00+10:00")
    with pytest.raises(ValueError, match="no source timezone"):
        MODULE.parse_timestamp("2026-07-11T11:00:00", require_timezone=True)


def test_offset_midnight_frozen_build_rederives_same_instant(tmp_path: Path) -> None:
    fixture = make_fixture(
        tmp_path,
        oot_capture="2026-07-10T15:00:00Z",
        oot_jump="2026-07-11T02:00:00+10:00",
    )
    summary = build_fixture(fixture, tmp_path / "out", enforce=False)
    assert summary["out_of_time"]["included_race_count"] == 1
    assert summary["out_of_time"]["source_display_timezone"] == "Australia/Melbourne"


def test_hash_valid_august_10_forgery_is_rejected(tmp_path: Path) -> None:
    fixture = make_fixture(
        tmp_path,
        oot_date="2026-08-10",
        oot_capture="2026-08-10T11:00:00+10:00",
        oot_jump="2026-08-10T12:00:00+10:00",
    )
    with pytest.raises(ValueError, match="outside Jul 11-Aug 9"):
        build_fixture(fixture, tmp_path / "out", enforce=False)


def test_card_sidecar_roster_rejects_swapped_box(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-11"
    card, sidecar = make_card(
        tmp_path / "upcoming" / f"{race_id}.csv", race_id, [(1, "A"), (2, "B")],
        capture="2026-07-11T10:00:00+10:00", jump="2026-07-11T12:00:00+10:00",
    )
    metadata = json.loads(sidecar.read_text())
    metadata["runner_completeness"]["participants"][0]["box_number"] = 2
    with pytest.raises(ValueError, match="duplicate or colliding"):
        MODULE.verify_card_sidecar_roster(card, metadata, race_id=race_id)


def test_sidecar_runner_count_must_equal_canonical_participants(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-11"
    card, sidecar = make_card(
        tmp_path / "upcoming" / f"{race_id}.csv", race_id, [(1, "A"), (2, "B")],
        capture="2026-07-11T10:00:00+10:00", jump="2026-07-11T12:00:00+10:00",
    )
    metadata = json.loads(sidecar.read_text())
    metadata["runner_completeness"]["runner_count"] = 1
    with pytest.raises(ValueError, match="runner_count disagrees"):
        MODULE.verify_card_sidecar_roster(card, metadata, race_id=race_id)


def test_sidecar_extra_requires_immutable_exclusion_evidence(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-09"
    card, sidecar = make_card(
        tmp_path / "upcoming" / f"{race_id}.csv", race_id, [(1, "A"), (2, "B")],
        capture="2026-07-09T10:00:00+10:00", jump="2026-07-09T12:00:00+10:00",
    )
    option = {
        "csv_path": card, "metadata": json.loads(sidecar.read_text()),
        "source_class": "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A", "active_roster_evidence": None,
    }
    with pytest.raises(ValueError, match="no immutable exclusion evidence"):
        MODULE.reconcile_development_roster(option, [{"box": 1, "dog_name": "A"}], race_id)


def test_active_label_runner_missing_from_complete_sidecar_fails_closed(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-09"
    card, sidecar = make_card(
        tmp_path / "upcoming" / f"{race_id}.csv", race_id, [(1, "A")],
        capture="2026-07-09T10:00:00+10:00", jump="2026-07-09T12:00:00+10:00",
    )
    option = {
        "csv_path": card, "metadata": json.loads(sidecar.read_text()),
        "source_class": "THEDOGS_PUBLISHED_HISTORY_NOT_TIER_A", "active_roster_evidence": None,
    }
    with pytest.raises(ValueError, match="active label runner absent"):
        MODULE.reconcile_development_roster(
            option,
            [{"box": 1, "dog_name": "A"}, {"box": 2, "dog_name": "B"}],
            race_id,
        )


@pytest.mark.parametrize(
    "participants",
    [
        [{"box": 1, "name": "A"}, {"box": 1, "name": "B"}],
        [{"box": 1, "name": "A"}, {"box": 2, "name": "A"}],
        [{"box": 1, "name": "A-B"}, {"box": 2, "name": "AB"}],
    ],
)
def test_duplicate_box_name_or_normalized_token_collisions_fail_closed(
    participants: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="duplicate or colliding"):
        MODULE.canonical_roster(participants, box_key="box", name_key="name", source="test")


@pytest.mark.parametrize("kind", [
    "eligibility_races", "eligibility_runners", "tier_a_provenance", "training_rows",
    "development_card", "development_sidecar", "official_race_label",
    "official_runner_label", "shadow", "freeze_inventory", "freeze_exclusions",
    "freeze_manifest", "oot_card", "oot_sidecar",
])
def test_one_bit_mutation_fails_closed(tmp_path: Path, kind: str) -> None:
    fixture = make_fixture(tmp_path)
    targets = {
        "eligibility_races": fixture["eligibility"] / "historical_win_eligibility_races_v1.csv",
        "eligibility_runners": fixture["eligibility"] / "historical_win_eligibility_runners_v1.csv",
        "tier_a_provenance": fixture["eligibility"] / "historical_win_tier_a_race_provenance_v1.json",
        "training_rows": fixture["training"] / "thedogs_training_rows_v1.csv",
        "development_card": fixture["card_one"],
        "development_sidecar": fixture["sidecar_one"],
        "official_race_label": fixture["official_races"],
        "official_runner_label": fixture["official_runners"],
        "shadow": fixture["shadow"],
        "freeze_inventory": fixture["freeze"] / "out_of_time_source_inventory.csv",
        "freeze_exclusions": fixture["freeze"] / "out_of_time_exclusions.csv",
        "freeze_manifest": fixture["freeze"] / "out_of_time_manifest.json",
        "oot_card": fixture["oot_card"],
        "oot_sidecar": fixture["oot_sidecar"],
    }
    target = targets[kind]
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(data)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        build_fixture(fixture, tmp_path / "out", enforce=False)


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_or_conflicting_shadow_keys_rejected(tmp_path: Path, conflicting: bool) -> None:
    path = tmp_path / "shadow.json"
    row = {"race_id": "Race 1 - BAL - 2026-07-09", "dog_name": "A", "box_number": 1}
    second = dict(row)
    if conflicting:
        second["box_number"] = 2
    write_json(path, [row, second])
    with pytest.raises(ValueError, match="shadow overlap key"):
        MODULE.load_shadow_feature_rows({row["race_id"]: record(path)})


def test_mismatch_predicates_report_unexplained() -> None:
    history = [{"date": date(2026, 7, 5)}]
    assert MODULE.mismatch_cause(
        "recency", differs=True, shadow={}, shadow_value=4, canonical_value=2,
        history=[{"date": date(2026, 7, 7)}, *history], rejected=[], target_date=date(2026, 7, 9),
    ) == "SHADOW_SELECTED_NONLATEST_COMPARED_RAW_ROW"
    assert MODULE.mismatch_cause(
        "history", differs=True, shadow={}, shadow_value=99, canonical_value=1,
        history=history, rejected=[], target_date=date(2026, 7, 9),
    ) == "UNEXPLAINED_HISTORY_DIFFERENCE"


def test_synthetic_full_build_is_deterministic_and_identity_safe(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    first = build_fixture(fixture, tmp_path / "build-a", enforce=False)
    bind_expected_output(fixture, first)
    second = build_fixture(fixture, tmp_path / "build-b", enforce=True)
    third = build_fixture(fixture, tmp_path / "build-c", enforce=True)
    assert second["artifact_manifest"] == third["artifact_manifest"] == first["artifact_manifest"]
    assert first["development"]["sidecar_only_runner_exclusion_count"] == 1
    assert first["out_of_time"]["included_runner_count"] == 1
    trainer = tmp_path / "build-a" / "trainer"
    dev_rows = list(csv.DictReader((trainer / "development_runners.csv").open()))
    oot_rows = list(csv.DictReader((trainer / "out_of_time_runners.csv").open()))
    assert set(dev_rows[0]).isdisjoint(MODULE.FORBIDDEN_ARTIFACT_FIELDS)
    assert set(oot_rows[0]).isdisjoint(MODULE.FORBIDDEN_ARTIFACT_FIELDS)
    assert {row["row_id"] for row in dev_rows}.isdisjoint({row["row_id"] for row in oot_rows})
    trainer = tmp_path / "build-a" / "trainer"
    exclusions = list(csv.DictReader((trainer / "development_exclusions.csv").open()))
    roster_exclusion = next(row for row in exclusions if row["reason"] == "HASH_BOUND_PUBLISHED_ACTIVE_ROSTER_EXCLUSION")
    assert set(roster_exclusion) == {
        "entity_type", "entity_id", "race_id", "reason", "history_date"
    }
    assert "SCRATCHEDDOG" not in (trainer / "development_exclusions.csv").read_text()


@pytest.mark.parametrize("mutation", ["count", "file_hash", "aggregate_hash"])
def test_expected_output_mutation_is_rejected(tmp_path: Path, mutation: str) -> None:
    fixture = make_fixture(tmp_path)
    summary = build_fixture(fixture, tmp_path / "build-a", enforce=False)
    bind_expected_output(fixture, summary)
    contract = json.loads(fixture["contract"].read_text())
    if mutation == "count":
        contract["expected_output"]["authoritative_counts"]["included_runners"] += 1
        match = "count mismatch"
    elif mutation == "file_hash":
        contract["expected_output"]["domains"]["trainer"]["artifact_files"]["development_runners.csv"] = "0" * 64
        match = "artifact hash mismatch"
    else:
        contract["expected_output"]["domains"]["trainer"]["aggregate_sha256"] = "0" * 64
        match = "aggregate hash mismatch"
    write_json(fixture["contract"], contract)
    with pytest.raises(ValueError, match=match):
        build_fixture(fixture, tmp_path / "build-b", enforce=True)


@pytest.mark.parametrize("mutation", ["sealed_key", "forbidden_json_key", "cross_split_id"])
def test_trainer_visible_identity_mutations_are_rejected(tmp_path: Path, mutation: str) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    build_fixture(fixture, output, enforce=False)
    trainer = output / "trainer"
    if mutation == "sealed_key":
        path = trainer / "development_runners.csv"
        path.write_text(path.read_text(encoding="utf-8") + "|dog:REUSABLE\n", encoding="utf-8")
        match = "sealed dog alignment key"
    elif mutation == "forbidden_json_key":
        path = trainer / "market_coverage.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dog_name"] = "REUSABLE"
        write_json(path, payload)
        match = "identity-bearing fields"
    else:
        dev_path = trainer / "development_runners.csv"
        oot_path = trainer / "out_of_time_runners.csv"
        with dev_path.open(encoding="utf-8", newline="") as handle:
            dev_id = next(csv.DictReader(handle))["row_id"]
        with oot_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["row_id"] = dev_id
        with oot_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        match = "links multiple races or splits"
    with pytest.raises(ValueError, match=match):
        MODULE.validate_trainer_visible_artifacts(trainer)


def test_scan_mode_rederives_source_and_rejects_banned_path(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    selected, exclusions = MODULE.scan_out_of_time_sources([fixture["evidence"]])
    assert len(selected) == 1
    banned = fixture["evidence"] / "reconstructed" / "upcoming" / "Race 4 - BAL - 2026-07-11.csv.metadata.json"
    banned.parent.mkdir(parents=True)
    banned.write_text("{}\n", encoding="utf-8")
    selected, exclusions = MODULE.scan_out_of_time_sources([fixture["evidence"]])
    assert len(selected) == 1
    assert any(row["reason"] == "RECONSTRUCTED_OR_NONCONTEMPORANEOUS_PATH_REJECTED" for row in exclusions)


def test_attacker_view_is_allowlisted_and_unlinkable(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    build_fixture(fixture, output, enforce=False)
    trainer = output / "trainer"
    control = output / "control_plane"
    manifest = json.loads((control / "trainer_input_manifest.json").read_text())
    allowed = {row["path"] for row in manifest["allowed_files"]}
    assert not any("source_inventory" in name or "alignment" in name for name in allowed)
    assert manifest["forbidden_roots"] == [
        "control_plane", "sealed_validation", "non_authoritative_diagnostic"
    ]
    attacker_text = "".join((trainer / name).read_text() for name in sorted(allowed))
    assert "Same Dog" not in attacker_text
    assert "SAMEDOG" not in attacker_text
    assert str(fixture["card_one"]) not in attacker_text
    dev = list(csv.DictReader((trainer / "development_runners.csv").open()))
    oot = list(csv.DictReader((trainer / "out_of_time_runners.csv").open()))
    assert not ({row["row_id"] for row in dev} & {row["row_id"] for row in oot})


def test_hash_rebound_sidecar_wrong_identity_still_fails(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-09"
    card, sidecar = make_card(
        tmp_path / "upcoming" / f"{race_id}.csv",
        race_id,
        [(1, "A")],
        capture="2026-07-09T10:00:00+10:00",
        jump="2026-07-09T12:00:00+10:00",
    )
    metadata = json.loads(sidecar.read_text())
    metadata["race_info"]["venue"] = "GEE"
    write_json(sidecar, metadata)
    rebound_hash = MODULE.sha256_path(sidecar)
    assert len(rebound_hash) == 64
    with pytest.raises(ValueError, match="venue mismatch"):
        MODULE.validate_sidecar_semantics(
            race_id,
            card,
            sidecar,
            metadata,
            expected_jump=MODULE.parse_timestamp("2026-07-09T12:00:00+10:00"),
            expected_roster=[(1, "A")],
        )


def test_equal_precedence_time_conflict_fails_without_path_tiebreak(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-09"
    card_a, sidecar_a = make_card(
        tmp_path / "a" / f"{race_id}.csv", race_id, [(1, "A")],
        capture="2026-07-09T10:00:00+10:00", jump="2026-07-09T12:00:00+10:00",
    )
    card_b, sidecar_b = make_card(
        tmp_path / "b" / f"{race_id}.csv", race_id, [(1, "B")],
        capture="2026-07-09T10:00:00+10:00", jump="2026-07-09T12:00:00+10:00",
    )
    def option(card: Path, sidecar: Path) -> dict[str, object]:
        metadata = json.loads(sidecar.read_text())
        return {
            "precedence": 0, "capture": MODULE.capture_timestamp(metadata),
            "jump": MODULE.sidecar_jump_timestamp(metadata, race_id),
            "csv_path": card, "sidecar_path": sidecar, "metadata": metadata,
            "csv_sha256": MODULE.sha256_path(card),
            "sidecar_sha256": MODULE.sha256_path(sidecar),
        }
    loaded = {"candidate_ids": [race_id], "source_options": {race_id: [
        option(card_a, sidecar_a), option(card_b, sidecar_b)
    ]}}
    with pytest.raises(ValueError, match="ambiguous equal-precedence"):
        MODULE.select_development_sources(loaded)


def test_numeric_zero_is_a_value() -> None:
    assert MODULE.safe_int(0) == 0
    assert MODULE.safe_float(0.0) == 0.0


def test_distinct_same_day_starts_with_verified_order_are_preserved() -> None:
    base = {
        "DATE": "2026-07-01", "TRACK": "BAL", "DIST": "450", "G": "5",
        "PLC": "2", "BOX": "1", "MGN": "1.5",
    }
    accepted, rejected = MODULE.accepted_history(
        [{**base, "RACE_NUMBER": "1"}, {**base, "RACE_NUMBER": "2"}],
        date(2026, 7, 9),
    )
    assert len(accepted) == 2
    assert not rejected


def test_duplicate_declaration_and_missing_bytes_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_text("x", encoding="utf-8")
    declaration = {"role": "card", **record(path)}
    with pytest.raises(ValueError, match="duplicate source declaration"):
        MODULE.source_set_digest([declaration, declaration])
    with pytest.raises(ValueError, match="byte declaration missing"):
        MODULE.verify_file_record(
            path,
            expected_sha256=MODULE.sha256_path(path),
            require_expected_bytes=True,
        )


def test_duplicate_live_discovery_and_same_time_conflict_fail(tmp_path: Path) -> None:
    race_id = "Race 1 - BAL - 2026-07-11"
    root_a = tmp_path / "a" / "upcoming"
    root_b = tmp_path / "b" / "upcoming"
    make_card(
        root_a / f"{race_id}.csv", race_id, [(1, "A")],
        capture="2026-07-11T10:00:00+10:00", jump="2026-07-11T12:00:00+10:00",
    )
    make_card(
        root_b / f"{race_id}.csv", race_id, [(1, "B")],
        capture="2026-07-11T10:00:00+10:00", jump="2026-07-11T12:00:00+10:00",
    )
    with pytest.raises(ValueError, match="duplicate live discovery path"):
        MODULE.scan_out_of_time_sources([root_a, root_a])
    with pytest.raises(ValueError, match="ambiguous same-time live sources"):
        MODULE.scan_out_of_time_sources([root_a, root_b])


def test_empty_or_missing_trainer_artifact_fails_closed(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    build_fixture(fixture, output, enforce=False)
    trainer = output / "trainer"
    (trainer / "development_features.csv").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty trainer artifact"):
        MODULE.validate_trainer_visible_artifacts(trainer)
    with pytest.raises(ValueError, match="missing or empty generated artifact"):
        MODULE.write_artifact_manifest(trainer, MODULE.TRAINER_ARTIFACT_NAMES)


def test_actual_trainer_readable_set_equals_declared_surface(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    build_fixture(fixture, output, enforce=False)
    trainer = output / "trainer"
    control = output / "control_plane"
    manifest = json.loads((control / "trainer_input_manifest.json").read_text())
    declared = {row["path"] for row in manifest["allowed_files"]}
    actual = {path.name for path in trainer.iterdir()}
    assert actual == declared == set(MODULE.TRAINER_ARTIFACT_ROLES)
    assert len(actual) == len(MODULE.TRAINER_ARTIFACT_ROLES)
    assert (control / "trainer_input_manifest.json").is_file()
    assert (control / "artifact-manifest.sha256").is_file()
    assert not any(path.is_file() for path in output.iterdir())
    assert MODULE.validate_trainer_read_surface(output) is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("unexpected_13th", "surface mismatch"),
        ("dotfile", "surface mismatch"),
        ("symlink", "not a regular file"),
        ("hardlink", "not a regular single-link file"),
        ("directory", "not a regular"),
        ("renamed", "surface mismatch"),
        ("missing", "surface mismatch"),
        ("length", "byte length mismatch"),
        ("hash", "sha256 mismatch"),
        ("duplicate", "duplicate trainer declaration"),
        ("escape", "unsafe trainer declaration"),
    ],
)
def test_trainer_loader_fails_closed_before_returning_data(
    tmp_path: Path, mutation: str, match: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    trainer = output / "trainer"
    control_manifest = output / "control_plane" / "trainer_input_manifest.json"
    manifest = json.loads(control_manifest.read_text())
    target = trainer / "development_features.csv"
    if mutation == "unexpected_13th":
        (trainer / "unexpected-13th.txt").write_text("x", encoding="utf-8")
    elif mutation == "dotfile":
        (trainer / ".hidden").write_text("x", encoding="utf-8")
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(output / "sealed_validation" / "development_source_inventory.csv")
    elif mutation == "hardlink":
        target.unlink()
        target.hardlink_to(output / "sealed_validation" / "development_source_inventory.csv")
    elif mutation == "directory":
        target.unlink()
        target.mkdir()
    elif mutation == "renamed":
        target.rename(trainer / "renamed.csv")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "length":
        target.write_bytes(target.read_bytes() + b"x")
    elif mutation == "hash":
        payload = target.read_bytes()
        target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    elif mutation == "duplicate":
        manifest["allowed_files"].append(dict(manifest["allowed_files"][0]))
        write_json(control_manifest, manifest)
    elif mutation == "escape":
        manifest["allowed_files"][0]["path"] = "../sealed_validation/development_source_inventory.csv"
        write_json(control_manifest, manifest)
    with pytest.raises(ValueError, match=match):
        MODULE.load_verified_trainer_inputs(output, fixture["contract"])


@pytest.mark.parametrize("mutation,match", [
    ("duplicate", "duplicate trainer declaration"),
    ("escape", "unsafe trainer declaration path"),
])
def test_internal_manifest_declaration_checks_fail_closed(
    tmp_path: Path, mutation: str, match: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    build_fixture(fixture, output, enforce=False)
    control_manifest = output / "control_plane" / "trainer_input_manifest.json"
    manifest = json.loads(control_manifest.read_text())
    if mutation == "duplicate":
        manifest["allowed_files"].append(dict(manifest["allowed_files"][0]))
    else:
        manifest["allowed_files"][0]["path"] = "../sealed_validation/development_source_inventory.csv"
    write_json(control_manifest, manifest)
    with pytest.raises(ValueError, match=match):
        MODULE.validate_trainer_read_surface(output)


def test_git_tracked_descriptor_pins_control_plane_bytes(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    control_manifest = output / "control_plane" / "trainer_input_manifest.json"
    control_manifest.write_bytes(control_manifest.read_bytes() + b"\n")
    assert MODULE.validate_trainer_read_surface(output) is None
    with pytest.raises(ValueError, match="control_plane artifact"):
        MODULE.load_verified_trainer_inputs(output, fixture["contract"])


def test_build_rejects_symlinked_packet_root_before_writing(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    target = tmp_path / "symlink-target"
    target.mkdir()
    output = tmp_path / "build"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="packet root.*symlink"):
        build_fixture(fixture, output, enforce=False)
    assert not list(target.iterdir())


@pytest.mark.parametrize("mutation", ["domain_symlink", "domain_entry", "root_entry"])
def test_build_rejects_preexisting_output_entries_before_writing(
    tmp_path: Path, mutation: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    output.mkdir()
    if mutation == "domain_symlink":
        target = tmp_path / "diagnostic-target"
        target.mkdir()
        (output / "non_authoritative_diagnostic").symlink_to(
            target, target_is_directory=True
        )
        match = "unexpected pre-existing entries"
    elif mutation == "domain_entry":
        (output / "trainer").mkdir()
        (output / "trainer" / "unexpected.txt").write_text("do not overwrite", encoding="utf-8")
        match = "packet domain has unexpected pre-existing entries"
    else:
        (output / "unexpected.txt").write_text("do not overwrite", encoding="utf-8")
        match = "packet root has unexpected pre-existing entries"
    before = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
    with pytest.raises(ValueError, match=match):
        build_fixture(fixture, output, enforce=False)
    after = sorted(str(path.relative_to(output)) for path in output.rglob("*"))
    assert after == before


@pytest.mark.parametrize(
    "mutation",
    [
        "packet_symlink", "trainer_symlink", "control_symlink",
        "sealed_symlink", "diagnostic_symlink",
    ],
)
def test_trainer_loader_rejects_symlinked_trust_paths(tmp_path: Path, mutation: str) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    packet = output
    if mutation == "packet_symlink":
        packet = tmp_path / "packet-alias"
        packet.symlink_to(output, target_is_directory=True)
    elif mutation == "trainer_symlink":
        trainer = output / "trainer"
        moved = tmp_path / "trainer-moved"
        trainer.rename(moved)
        trainer.symlink_to(moved, target_is_directory=True)
    elif mutation == "control_symlink":
        control = output / "control_plane"
        moved = tmp_path / "control-moved"
        control.rename(moved)
        control.symlink_to(moved, target_is_directory=True)
    else:
        domain_name = (
            "sealed_validation" if mutation == "sealed_symlink"
            else "non_authoritative_diagnostic"
        )
        domain = output / domain_name
        moved = tmp_path / f"{domain_name}-moved"
        domain.rename(moved)
        domain.symlink_to(moved, target_is_directory=True)
    if mutation == "diagnostic_symlink":
        assert len(MODULE.load_verified_trainer_inputs(packet, fixture["contract"])) == 10
        with pytest.raises(ValueError, match="symlink|directory"):
            MODULE.validate_complete_packet(packet, fixture["contract"])
    else:
        with pytest.raises(ValueError, match="symlink|directory"):
            MODULE.load_verified_trainer_inputs(packet, fixture["contract"])


@pytest.mark.parametrize(
    "mutation", ["unexpected_file", "dotfile", "symlink", "directory"]
)
def test_trainer_loader_rejects_unexpected_packet_root_entries(
    tmp_path: Path, mutation: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    if mutation == "unexpected_file":
        (output / "unexpected.txt").write_text("x", encoding="utf-8")
    elif mutation == "dotfile":
        (output / ".hidden").write_text("x", encoding="utf-8")
    elif mutation == "symlink":
        (output / "trainer-alias").symlink_to(output / "trainer", target_is_directory=True)
    else:
        (output / "unexpected-directory").mkdir()
    assert len(MODULE.load_verified_trainer_inputs(output, fixture["contract"])) == 10
    with pytest.raises(ValueError, match="packet root surface mismatch"):
        MODULE.validate_complete_packet(output, fixture["contract"])


def test_diagnostic_build_cannot_change_authoritative_files(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    reproducibility = MODULE.load_reproducibility_contract(fixture["contract"])
    loaded = MODULE.load_development_sources(
        fixture["eligibility"], fixture["training"], reproducibility
    )
    loaded = MODULE.load_diagnostic_sources(loaded, reproducibility)
    authoritative = tmp_path / "authoritative"
    MODULE.build_development_packet(loaded, authoritative, tmp_path / "sealed")
    before = {
        name: MODULE.sha256_path(authoritative / name)
        for name in (
            "development_features.csv", "development_runners.csv",
            "development_races.csv", "development_manifest.json",
        )
    }
    mutated_shadow = tmp_path / "changed-diagnostic-path" / "shadow.json"
    payload = json.loads(fixture["shadow"].read_text())
    payload[0]["prior_start_count"] = 99
    write_json(mutated_shadow, payload)
    for record_value in loaded["shadow_source_by_race"].values():
        record_value.update(record(mutated_shadow))
    MODULE.build_overlap_reconciliation(loaded, tmp_path / "diagnostic")
    after = {name: MODULE.sha256_path(authoritative / name) for name in before}
    assert before == after


def test_cli_argument_contract_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    with pytest.raises(SystemExit):
        MODULE.parse_args()


@pytest.mark.parametrize("mutation", ["absent", "moved", "corrupt", "valid_mutation"])
def test_authoritative_build_is_operationally_independent_of_diagnostics(
    tmp_path: Path, mutation: str
) -> None:
    fixture = make_fixture(tmp_path)
    baseline = tmp_path / "authoritative-baseline"
    MODULE.build_authoritative_packet(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], baseline,
        fixture["freeze"], fixture["contract"], enforce_expected_output=False,
    )
    before = {
        domain: {
            path.name: MODULE.sha256_path(path)
            for path in sorted((baseline / domain).iterdir())
        }
        for domain in MODULE.AUTHORITATIVE_DOMAIN_ROOT_NAMES
    }

    if mutation == "absent":
        fixture["shadow"].unlink()
    elif mutation == "moved":
        moved = tmp_path / "moved-diagnostic" / fixture["shadow"].name
        moved.parent.mkdir()
        fixture["shadow"].rename(moved)
    elif mutation == "corrupt":
        fixture["shadow"].write_text("{not-json\n", encoding="utf-8")
    else:
        payload = json.loads(fixture["shadow"].read_text())
        payload[0]["prior_start_count"] = 999
        write_json(fixture["shadow"], payload)

    rebuilt = tmp_path / f"authoritative-{mutation}"
    MODULE.build_authoritative_packet(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], rebuilt,
        fixture["freeze"], fixture["contract"], enforce_expected_output=False,
    )
    after = {
        domain: {
            path.name: MODULE.sha256_path(path)
            for path in sorted((rebuilt / domain).iterdir())
        }
        for domain in MODULE.AUTHORITATIVE_DOMAIN_ROOT_NAMES
    }
    assert after == before
    assert not (rebuilt / "non_authoritative_diagnostic").exists()


def test_optional_diagnostic_phase_consumes_completed_authoritative_packet_read_only(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    bootstrap = build_fixture(fixture, tmp_path / "bootstrap", enforce=False)
    bind_expected_output(fixture, bootstrap)
    output = tmp_path / "authoritative"
    authoritative = MODULE.build_authoritative_packet(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], output,
        fixture["freeze"], fixture["contract"], enforce_expected_output=True,
    )
    before = {
        domain: {
            path.name: MODULE.sha256_path(path)
            for path in sorted((output / domain).iterdir())
        }
        for domain in MODULE.AUTHORITATIVE_DOMAIN_ROOT_NAMES
    }
    assert len(MODULE.load_verified_trainer_inputs(output, fixture["contract"])) == 10
    diagnostic = MODULE.build_optional_diagnostics(
        output, fixture["contract"], enforce_expected_output=True
    )
    after = {
        domain: {
            path.name: MODULE.sha256_path(path)
            for path in sorted((output / domain).iterdir())
        }
        for domain in MODULE.AUTHORITATIVE_DOMAIN_ROOT_NAMES
    }
    assert before == after
    assert diagnostic["authoritative_bytes_identical"] is True
    assert diagnostic["reconciliation"] == bootstrap["reconciliation"]
    assert authoritative["artifact_manifest"] == bootstrap["artifact_manifest"]
    assert MODULE.validate_complete_packet(output, fixture["contract"]) is None


def test_diagnostic_failure_fails_only_the_optional_phase(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    bootstrap = build_fixture(fixture, tmp_path / "bootstrap", enforce=False)
    bind_expected_output(fixture, bootstrap)
    output = tmp_path / "authoritative"
    MODULE.build_authoritative_packet(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], output,
        fixture["freeze"], fixture["contract"], enforce_expected_output=True,
    )
    expected_trainer = MODULE.load_verified_trainer_inputs(output, fixture["contract"])
    fixture["shadow"].unlink()
    with pytest.raises(ValueError, match="shadow diagnostic source"):
        MODULE.build_optional_diagnostics(output, fixture["contract"])
    assert MODULE.load_verified_trainer_inputs(output, fixture["contract"]) == expected_trainer


def test_authoritative_loader_ignores_absent_moved_corrupt_or_mutated_diagnostic_output(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    bootstrap = build_fixture(fixture, tmp_path / "bootstrap", enforce=False)
    bind_expected_output(fixture, bootstrap)
    output = tmp_path / "authoritative"
    MODULE.build_authoritative_packet(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], output,
        fixture["freeze"], fixture["contract"], enforce_expected_output=True,
    )
    expected = MODULE.load_verified_trainer_inputs(output, fixture["contract"])
    assert not (output / "non_authoritative_diagnostic").exists()
    MODULE.build_optional_diagnostics(output, fixture["contract"])
    diagnostic = output / "non_authoritative_diagnostic"
    moved = tmp_path / "moved-diagnostic-output"
    diagnostic.rename(moved)
    assert MODULE.load_verified_trainer_inputs(output, fixture["contract"]) == expected
    moved.rename(diagnostic)
    summary = diagnostic / "reconciliation_summary.json"
    summary.write_text("{corrupt\n", encoding="utf-8")
    assert MODULE.load_verified_trainer_inputs(output, fixture["contract"]) == expected
    with pytest.raises(ValueError):
        MODULE.validate_complete_packet(output, fixture["contract"])
    summary.write_text("{}\n", encoding="utf-8")
    assert MODULE.load_verified_trainer_inputs(output, fixture["contract"]) == expected


def test_authoritative_phase_does_not_validate_diagnostic_contract_shape(
    tmp_path: Path,
) -> None:
    fixture = make_fixture(tmp_path)
    bootstrap = build_fixture(fixture, tmp_path / "bootstrap", enforce=False)
    bind_expected_output(fixture, bootstrap)
    contract = json.loads(fixture["contract"].read_text())
    contract["trusted_inputs"]["diagnostic"] = ["malformed", "optional"]
    contract["expected_output"]["domains"]["non_authoritative_diagnostic"] = [
        "malformed", "optional"
    ]
    write_json(fixture["contract"], contract)
    output = tmp_path / "authoritative"
    MODULE.build_authoritative_packet(
        fixture["eligibility"], fixture["training"], [fixture["evidence"]], output,
        fixture["freeze"], fixture["contract"], enforce_expected_output=True,
    )
    assert len(list((output / "trainer").iterdir())) == 10
    assert len(MODULE.load_verified_trainer_inputs(output, fixture["contract"])) == 10
    with pytest.raises(ValueError, match="diagnostic trust domain"):
        MODULE.build_optional_diagnostics(
            output, fixture["contract"], enforce_expected_output=False
        )


def test_loader_rejects_ancestor_symlink_before_returning_bytes(tmp_path: Path) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "real-parent" / "build"
    output.parent.mkdir()
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(output.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|ancestor"):
        MODULE.load_verified_trainer_inputs(alias_parent / output.name, fixture["contract"])


def test_loader_rejects_component_swap_before_returning_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    trainer = output / "trainer"
    replacement = tmp_path / "replacement-trainer"
    moved = tmp_path / "moved-trainer"
    shutil.copytree(trainer, replacement)
    real_open = MODULE.os.open
    swapped = False

    def swap_before_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal swapped
        if path == "trainer" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            trainer.rename(moved)
            replacement.rename(trainer)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(MODULE.os, "open", swap_before_open)
    with pytest.raises(ValueError, match="changed during open|path swap"):
        MODULE.load_verified_trainer_inputs(output, fixture["contract"])


@pytest.mark.parametrize("domain", [
    "trainer", "control_plane", "sealed_validation", "non_authoritative_diagnostic",
])
@pytest.mark.parametrize("mutation", [
    "unexpected", "dotfile", "missing", "symlink", "hardlink", "directory", "hash",
])
def test_complete_packet_enforces_exact_declared_physical_sets(
    tmp_path: Path, domain: str, mutation: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    domain_root = output / domain
    target = next(path for path in sorted(domain_root.iterdir()) if path.is_file())
    if mutation == "unexpected":
        (domain_root / "unexpected.txt").write_text("x", encoding="utf-8")
    elif mutation == "dotfile":
        (domain_root / ".hidden").write_text("x", encoding="utf-8")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "symlink":
        target.unlink()
        target.symlink_to(fixture["card_one"])
    elif mutation == "hardlink":
        target.unlink()
        target.hardlink_to(fixture["card_one"])
    elif mutation == "directory":
        target.unlink()
        target.mkdir()
    else:
        payload = target.read_bytes()
        target.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
    with pytest.raises(ValueError):
        MODULE.validate_complete_packet(output, fixture["contract"])


@pytest.mark.parametrize("domain", [
    "trainer", "control_plane", "sealed_validation", "non_authoritative_diagnostic",
])
@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "type", "role", "length", "hash", "hash_map", "aggregate"],
)
def test_complete_packet_rejects_declaration_changes_in_every_domain(
    tmp_path: Path, domain: str, mutation: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    contract = json.loads(fixture["contract"].read_text())
    domain_contract = contract["expected_output"]["domains"][domain]
    declarations = domain_contract["artifacts"]
    if mutation == "duplicate":
        declarations.append(dict(declarations[0]))
    elif mutation == "type":
        declarations[0]["type"] = "directory"
    elif mutation == "role":
        declarations[0]["role"] = "WRONG_ROLE"
    elif mutation == "length":
        declarations[0]["bytes"] += 1
    elif mutation == "hash":
        declarations[0]["sha256"] = "0" * 64
    elif mutation == "hash_map":
        domain_contract["artifact_files"][declarations[0]["path"]] = "0" * 64
    else:
        domain_contract["aggregate_sha256"] = "0" * 64
    write_json(fixture["contract"], contract)
    with pytest.raises(ValueError):
        MODULE.validate_complete_packet(output, fixture["contract"])


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_complete_packet_requires_exact_descriptor_domain_set(
    tmp_path: Path, mutation: str
) -> None:
    fixture = make_fixture(tmp_path)
    output = tmp_path / "build"
    summary = build_fixture(fixture, output, enforce=False)
    bind_expected_output(fixture, summary)
    contract = json.loads(fixture["contract"].read_text())
    domains = contract["expected_output"]["domains"]
    if mutation == "missing":
        domains.pop("non_authoritative_diagnostic")
    else:
        domains["unexpected"] = dict(domains["trainer"])
    write_json(fixture["contract"], contract)
    with pytest.raises(ValueError, match="domain set"):
        MODULE.validate_complete_packet(output, fixture["contract"])
