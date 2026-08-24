import sqlite3
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from race_collection.domain import OperationId, RaceId
from race_collection.operations import SQLiteOperationsStore
from race_collection.forecasting import ForecastingAuthority
from race_collection.operations import BarrierNotSatisfied
from race_collection.synchronous_manual_capture import VerifiedCurrentRaceIndex
from race_collection.service import (
    ForwardBaselineCaptureConfiguration,
    ForwardBaselineCaptureService,
    ServiceUnavailable,
    run_forward_baseline_capture,
    main,
)
from src.predictor.on_demand import sha256_bytes


NOW = datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[2]


def verified_index(races):
    return VerifiedCurrentRaceIndex(
        schema_version="collector_current_race_index_v2",
        run_id="collector-run-159",
        source_generated_at=NOW.isoformat(),
        packet_sha256=sha256_bytes(b"verified-index"),
        packet_bytes=b"verified-index",
        races=tuple(races),
        source_refresh_report_path="refresh/report.json",
        source_refresh_report_sha256="2" * 64,
        publication_sha256="3" * 64,
        state_sha256="4" * 64,
        report_sha256="5" * 64,
    )


def candidate(number, *, venue="Sandown", racing_date="2026-08-23"):
    jump = datetime.fromisoformat(f"{racing_date}T03:{number:02d}:00+00:00")
    return {
        "race_id": f"Race {number} - {venue} - {racing_date}",
        "date": racing_date,
        "venue": venue,
        "race_number": number,
        "jump_datetime": jump.isoformat(),
        "source_native_race_id": str(159000 + number),
        "runners": [
            {
                "box": 1,
                "display_name": "Alpha",
                "identity": "alpha",
                "scratch_state": "ACTIVE",
                "source_native_runner_id": str(15900000 + number * 10 + 1),
            },
            {
                "box": 2,
                "display_name": "Beta",
                "identity": "beta",
                "scratch_state": "ACTIVE",
                "source_native_runner_id": str(15900000 + number * 10 + 2),
            },
        ],
    }


def configuration(tmp_path: Path) -> ForwardBaselineCaptureConfiguration:
    return ForwardBaselineCaptureConfiguration(
        cohort_id="issue-159-forward-baseline",
        corpus_root=tmp_path / "corpus",
        current_index_max_age=timedelta(minutes=15),
        feature_cutoff=timedelta(minutes=5),
        timezone="Australia/Melbourne",
    )


def complete_candidates():
    venues = ("Sandown", "Richmond", "Albion Park")
    return [
        candidate(
            number,
            venue=venues[(number - 1) % len(venues)],
            racing_date="2026-08-23" if number <= 10 else "2026-08-24",
        )
        for number in range(1, 21)
    ]


def table_counts(database: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database) as connection:
        return tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("operations", "racing_days", "races")
        )


def production_config(tmp_path: Path, database: Path) -> tuple[Path, Path, Path]:
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    index_path = evidence_root / "runtime/current.json"
    document = {
        "schema_version": "forward-baseline-capture-service-config-v1",
        "cohort_id": "issue-159-forward-baseline",
        "operations_database": str(database),
        "corpus_root": str(tmp_path / "corpus"),
        "evidence_root": str(evidence_root),
        "current_race_index_path": str(index_path),
        "current_index_max_age_seconds": 900,
        "current_index_timeout_seconds": 2,
        "feature_cutoff_seconds": 300,
        "timezone": "Australia/Melbourne",
    }
    config_path = tmp_path / "forward-baseline.json"
    config_path.write_bytes(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    )
    return config_path, evidence_root, index_path


def test_forward_baseline_configuration_schema_is_checked_in_and_closed():
    schema = json.loads(
        (ROOT / "config/forward_baseline_capture_service.schema.json").read_bytes()
    )

    assert schema["$id"] == "forward-baseline-capture-service-config-v1"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_production_preflight_awaits_complete_candidate_population_without_writes(
    tmp_path: Path,
):
    database = tmp_path / "operations.sqlite3"
    store = SQLiteOperationsStore(database)
    store.migrate()
    before = table_counts(database)

    report = ForwardBaselineCaptureService(store, configuration(tmp_path)).run(
        verified_index([candidate(1)]),
        now=NOW,
    )

    assert report == {
        "schema_version": "forward-baseline-capture-service-report-v1",
        "status": "AWAITING_COHORT_CANDIDATES",
        "candidate_race_count": 1,
        "candidate_venue_count": 1,
        "candidate_race_date_count": 1,
        "required_race_count": 20,
        "required_venue_count": 3,
        "required_race_date_count": 2,
    }
    assert table_counts(database) == before
    assert not configuration(tmp_path).corpus_root.exists()


def test_forward_cohort_freeze_still_requires_twenty_races_three_venues_two_dates(
    tmp_path: Path,
):
    cases = (
        [
            candidate(
                number,
                venue=("Sandown", "Richmond")[number % 2],
                racing_date="2026-08-23" if number <= 10 else "2026-08-24",
            )
            for number in range(1, 21)
        ],
        [candidate(number, venue=("Sandown", "Richmond", "Albion Park")[number % 3]) for number in range(1, 21)],
    )
    for index, races in enumerate(cases):
        database = tmp_path / f"operations-{index}.sqlite3"
        store = SQLiteOperationsStore(database)
        store.migrate()
        before = table_counts(database)

        report = ForwardBaselineCaptureService(
            store, configuration(tmp_path / str(index))
        ).run(verified_index(races), now=NOW)

        assert report["status"] == "AWAITING_COHORT_CANDIDATES"
        assert report["required_race_count"] == 20
        assert report["required_venue_count"] == 3
        assert report["required_race_date_count"] == 2
        assert table_counts(database) == before


def test_checked_in_production_entrypoint_uses_verified_v2_index_and_awaits_without_writes(
    tmp_path: Path,
):
    database = tmp_path / "operations.sqlite3"
    store = SQLiteOperationsStore(database)
    store.migrate()
    config_path, evidence_root, index_path = production_config(tmp_path, database)
    view = verified_index([candidate(1)])
    before = table_counts(database)

    report = run_forward_baseline_capture(
        config_path,
        now=NOW,
        current_index_reader=lambda **kwargs: (
            view
            if kwargs["return_verified_view"] is True
            and kwargs["index_path"] == index_path
            and kwargs["evidence_root"] == evidence_root
            else None
        ),
    )

    assert report["status"] == "AWAITING_COHORT_CANDIDATES"
    assert table_counts(database) == before
    assert not (tmp_path / "corpus").exists()


def test_race_collection_service_cli_routes_forward_baseline_report(capsys, tmp_path: Path):
    config_path = tmp_path / "forward-baseline.json"
    config_path.write_text("{}")
    observed = []

    exit_code = main(
        ["--forward-baseline-config", str(config_path)],
        forward_baseline_runner=lambda path: (
            observed.append(path)
            or {
                "schema_version": "forward-baseline-capture-service-report-v1",
                "status": "AWAITING_COHORT_CANDIDATES",
            }
        ),
    )

    assert exit_code == 0
    assert observed == [config_path]
    assert json.loads(capsys.readouterr().out)["status"] == "AWAITING_COHORT_CANDIDATES"


def test_production_preflight_rejects_name_as_native_runner_id_before_writes(
    tmp_path: Path,
):
    database = tmp_path / "operations.sqlite3"
    store = SQLiteOperationsStore(database)
    store.migrate()
    races = complete_candidates()
    races[7]["runners"][1]["source_native_runner_id"] = "Beta"
    before = table_counts(database)

    service = ForwardBaselineCaptureService(store, configuration(tmp_path))
    report = service.run(verified_index(races), now=NOW)

    assert report == {
        "schema_version": "forward-baseline-capture-service-report-v1",
        "status": "INTEGRITY_FAILED",
        "reason": "NUMERIC_SOURCE_NATIVE_RUNNER_IDS_REQUIRED",
        "race_id": races[7]["race_id"],
    }
    assert table_counts(database) == before
    assert not configuration(tmp_path).corpus_root.exists()


def test_production_preflight_rejects_malformed_selected_race_before_any_write(
    tmp_path: Path,
):
    database = tmp_path / "operations.sqlite3"
    store = SQLiteOperationsStore(database)
    store.migrate()
    races = complete_candidates()
    races[9]["jump_datetime"] = "not-a-timestamp"
    before = table_counts(database)

    service = ForwardBaselineCaptureService(store, configuration(tmp_path))
    report = service.run(verified_index(races), now=NOW)

    assert report == {
        "schema_version": "forward-baseline-capture-service-report-v1",
        "status": "INTEGRITY_FAILED",
        "reason": "COHORT_CANDIDATE_INVALID",
        "race_id": races[9]["race_id"],
    }
    assert table_counts(database) == before
    assert not configuration(tmp_path).corpus_root.exists()


def test_production_preflight_awaits_fresh_verified_view_before_any_write(tmp_path: Path):
    database = tmp_path / "operations.sqlite3"
    store = SQLiteOperationsStore(database)
    store.migrate()
    config_path, evidence_root, index_path = production_config(tmp_path, database)
    stale = verified_index(complete_candidates())
    stale = VerifiedCurrentRaceIndex(
        **{
            field: getattr(stale, field)
            for field in stale.__dataclass_fields__
            if field != "source_generated_at"
        },
        source_generated_at=(NOW - timedelta(minutes=16)).isoformat(),
    )
    before = table_counts(database)

    report = run_forward_baseline_capture(
        config_path,
        now=NOW,
        current_index_reader=lambda **values: (
            stale
            if values["index_path"] == index_path
            and values["evidence_root"] == evidence_root
            and values["return_verified_view"] is True
            else None
        ),
    )

    assert report["status"] == "AWAITING_COHORT_CANDIDATES"
    assert report["reason"] == "CURRENT_RACE_INDEX_NOT_READY_NOW"
    assert table_counts(database) == before
    assert not configuration(tmp_path).corpus_root.exists()


def test_production_entrypoint_freezes_exact_cohort_and_schedules_existing_lifecycle(
    tmp_path: Path,
):
    database = tmp_path / "operations.sqlite3"
    store = SQLiteOperationsStore(database)
    store.migrate()
    races = complete_candidates()

    config_path, evidence_root, index_path = production_config(tmp_path, database)
    view = verified_index(races)
    report = run_forward_baseline_capture(
        config_path,
        now=NOW,
        current_index_reader=lambda **kwargs: (
            view
            if kwargs["return_verified_view"] is True
            and kwargs["index_path"] == index_path
            and kwargs["evidence_root"] == evidence_root
            else None
        ),
    )
    service = ForwardBaselineCaptureService(store, configuration(tmp_path))

    assert report["status"] == "COHORT_FROZEN_AWAITING_SCHEDULED_CAPTURE"
    assert report["race_count"] == 20
    assert report["terminal_count"] == 0
    cohort_paths = list((configuration(tmp_path).corpus_root / "cohorts").glob("*.json"))
    assert len(cohort_paths) == 1
    cohort_bytes = cohort_paths[0].read_bytes()
    cohort = json.loads(cohort_bytes)
    assert cohort["cohort_id"] == "issue-159-forward-baseline"
    assert cohort["race_count"] == 20
    binding = store.forward_baseline_cohort(cohort["cohort_id"])
    assert binding["artifact_checksum"] == report["cohort_checksum"]
    assert binding["members"] == cohort["members"]
    by_native_race = {
        member["source_native_race_id"]: member for member in cohort["members"]
    }
    assert by_native_race[races[4]["source_native_race_id"]][
        "source_native_runner_ids"
    ] == [
        runner["source_native_runner_id"] for runner in races[4]["runners"]
    ]
    with store._connect() as connection:
        assert connection.execute("SELECT count(*) FROM racing_days").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM expected_races").fetchone()[0] == 20
    with pytest.raises(BarrierNotSatisfied, match="not terminal"):
        ForecastingAuthority(store).baseline_cohort_terminal_records(cohort_bytes)
    with pytest.raises(BarrierNotSatisfied, match="not terminal"):
        service.open_results(
            OperationId("op_" + "9" * 32),
            RaceId(cohort["members"][0]["race_id"]),
            NOW,
            cohort_bytes=cohort_bytes,
        )

    before = table_counts(database)
    substituted = complete_candidates()
    substituted[19]["source_native_race_id"] = "999999"
    conflict = service.run(verified_index(substituted), now=NOW)

    assert conflict["status"] == "INTEGRITY_FAILED"
    assert conflict["reason"] == "FROZEN_COHORT_BINDING_MISMATCH"
    assert table_counts(database) == before
    assert cohort_paths[0].read_bytes() == cohort_bytes

    refreshed = replace(
        view,
        source_generated_at=(NOW + timedelta(seconds=1)).isoformat(),
    )
    replay = service.run(refreshed, now=NOW + timedelta(seconds=1))

    assert replay["status"] == "COHORT_FROZEN_AWAITING_SCHEDULED_CAPTURE"
    assert cohort_paths[0].read_bytes() == cohort_bytes

    calls = []
    admitted = service.capture_scheduled(
        protocol=object(),
        evidence_root=tmp_path / "evidence",
        collector_run_id="collector-run-159",
        plan_item={"race_id": races[0]["race_id"]},
        verified_index=view,
        emitted_at=NOW,
        scheduled_admitter=lambda **values: (
            calls.append(values)
            or {"status": "PREJUMP_CAPTURED", "race_id": values["plan_item"]["race_id"]}
        ),
    )
    assert admitted["status"] == "PREJUMP_CAPTURED"
    assert calls[0]["corpus_root"] == configuration(tmp_path).corpus_root
    assert calls[0]["verified_index"] is view
    assert calls[0]["cohort_id"] == cohort["cohort_id"]
    assert str(calls[0]["cohort_checksum"]) == binding["artifact_checksum"]

    with pytest.raises(ServiceUnavailable, match="not READY_NOW"):
        service.capture_scheduled(
            protocol=object(),
            evidence_root=tmp_path / "evidence",
            collector_run_id="collector-run-stale",
            plan_item={"race_id": races[0]["race_id"]},
            verified_index=replace(
                view,
                source_generated_at=(NOW - timedelta(minutes=16)).isoformat(),
            ),
            emitted_at=NOW,
            scheduled_admitter=lambda **values: values,
        )

    mismatched_service = ForwardBaselineCaptureService(
        store,
        replace(configuration(tmp_path), corpus_root=tmp_path / "other-corpus"),
    )
    with pytest.raises(ServiceUnavailable, match="cohort authority disagrees"):
        mismatched_service.capture_scheduled(
            protocol=object(),
            evidence_root=tmp_path / "evidence",
            collector_run_id="collector-run-mismatch",
            plan_item={"race_id": races[0]["race_id"]},
            verified_index=view,
            emitted_at=NOW,
            scheduled_admitter=lambda **values: values,
        )
