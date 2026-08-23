import copy
import inspect
import json
import math
import shutil
import sqlite3
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from test_phase5_ordered_finish_training import (
    build_durable_forward,
    durable_forward_setup,
    synthetic_prepared_example,
)

from race_collection.artifacts import ArtifactStoreError, ChecksumMismatch, LocalArtifactStore
from race_collection.domain import (
    ArtifactChecksum,
    EvidenceAuthority,
    EvidenceField,
    FieldEvidence,
    FreezeAuthority,
    OperationId,
    ProgrammeRaceCandidate,
    RaceState,
    RacingDay,
    RacingDayId,
)
from race_collection.evaluation import (
    EligibleRaceIdentity,
    EvaluationAuthority,
    EvaluationRace,
    EvaluationRejected,
    ForecastEvidence,
    PromotionAuthority,
    PromotionPolicy,
    PromotionRejected,
    _evaluate_paired,
    _validate_report,
    diagnose_drift,
    forecast_checksum,
    forecast_document_checksum,
    promotion_decision,
)
from race_collection.forecast_service import (
    CanonicalForecastApplication,
    CanonicalForecastService,
    ForecastUnavailable,
)
from race_collection.forecasting import ForecastingAuthority, LegacyBundle, ModelRelease
from race_collection.model_bundle import BundleUnavailable, ChampionLoader, ModelBundleAuthority
from race_collection.operational import OperationalAuthority
from race_collection.operations import ConflictingOperation, SQLiteOperationsStore
from race_collection.ordered_finish import (
    forecast_ordered_finish,
    ordered_finish_from_probabilities,
)
from race_collection.sealing import FieldObservation, normalize_fields
from race_collection.training import (
    LinearStrengthModel,
    TrainingCorpusAuthority,
    canonical_json,
    checksum,
    train_challenger_bundle,
)


def digest(value):
    return ArtifactChecksum("sha256:" + value * 64)


def operation(value):
    return OperationId(f"op_{value:032x}")


def authenticated_forecast(
    race_id, bundle_id, bundle_checksum, evidence_checksum, distribution, computed_at
):
    provisional = ForecastEvidence(
        race_id,
        bundle_id,
        bundle_checksum,
        evidence_checksum,
        digest("0"),
        distribution,
        computed_at,
    )
    return replace(provisional, forecast_checksum=forecast_document_checksum(provisional))


def population(count=500, *, reverse_short=False, venues=("A", "B", "C")):
    races, champion, challenger = [], [], []
    champion_distribution = forecast_ordered_finish(("a", "b"), (0, 0))
    for index in range(count):
        race = EvaluationRace(
            f"race-{index:04}",
            f"2026-{1 + index // 280:02}-{1 + (index % 28):02}",
            venues[index % len(venues)],
            (300, 400, 500)[index % 3],
            ("G5", "G4")[index % 2],
            ("a", "b"),
            digest("a"),
            digest("b"),
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:01:00+00:00",
            "2026-01-01T00:02:00+00:00",
        )
        candidate_distribution = forecast_ordered_finish(
            ("a", "b"), (-4, 4) if reverse_short and index >= count - 100 else (4, -4)
        )
        races.append(race)
        champion.append(
            authenticated_forecast(
                race.race_id,
                "champion",
                digest("c"),
                digest("a"),
                champion_distribution,
                race.forecast_computed_at,
            )
        )
        challenger.append(
            authenticated_forecast(
                race.race_id,
                "challenger",
                digest("e"),
                digest("a"),
                candidate_distribution,
                race.forecast_computed_at,
            )
        )
    return races, {"champion": champion, "challenger": challenger}


def report(count=500, **kwargs):
    races, forecasts = population(count, **kwargs)
    return _evaluate_paired(
        races,
        forecasts,
        champion_bundle_id="champion",
        challenger_bundle_ids=("challenger",),
        policy=PromotionPolicy(bootstrap_samples=200),
        eligible_population=eligible(races),
    )


def eligible(races):
    return [
        EligibleRaceIdentity(race.race_id, race.evidence_checksum, race.result_checksum)
        for race in races
    ]


def test_all_metrics_share_one_distribution_and_exact_paired_population():
    result = report()
    candidate = result["long_horizon"]["challenger"]
    assert candidate["race_count"] == candidate["coverage_denominator"] == 500
    assert candidate["coverage"] == 1 and candidate["abstention"] == 0
    assert candidate["exact_top2_order_accuracy"] == 1
    assert candidate["exact_top3_order_accuracy"] == 1
    assert candidate["top3_containment"] == 1
    assert candidate["winner_mean_reciprocal_rank"] == 1
    assert len(candidate["win_calibration"]["bins"]) == 10
    assert len(candidate["top3_calibration"]["bins"]) == 10
    assert set(candidate["slices"]) == {"venue", "distance", "grade", "field_size"}
    assert math.isclose(
        sum(item["race_count"] for item in candidate["slices"]["venue"].values()), 500
    )
    assert result["wagering_scorecard"]["status"] == "NOT_RUN"
    assert result == report()  # deterministic race-level bootstrap and replay


def test_nonpaired_checksum_mismatch_and_exclusion_fail_closed():
    races, forecasts = population(3)
    forecasts["challenger"] = forecasts["challenger"][:-1]
    with pytest.raises(EvaluationRejected, match="exact paired"):
        _evaluate_paired(
            races,
            forecasts,
            champion_bundle_id="champion",
            challenger_bundle_ids=("challenger",),
            eligible_population=eligible(races),
        )
    races, forecasts = population(3)
    forecasts["challenger"][0] = replace(forecasts["challenger"][0], evidence_checksum=digest("9"))
    with pytest.raises(EvaluationRejected, match="checksum mismatch"):
        _evaluate_paired(
            races,
            forecasts,
            champion_bundle_id="champion",
            challenger_bundle_ids=("challenger",),
            eligible_population=eligible(races),
        )
    races, forecasts = population(3)
    forecasts["challenger"][0] = replace(forecasts["challenger"][0], forecast_checksum=digest("9"))
    with pytest.raises(EvaluationRejected, match="forecast checksum mismatch"):
        _evaluate_paired(
            races,
            forecasts,
            champion_bundle_id="champion",
            challenger_bundle_ids=("challenger",),
            eligible_population=eligible(races),
        )
    with pytest.raises(EvaluationRejected, match="excluded"):
        replace(races[0], eligible=False, exclusion="dead_heat")
    with pytest.raises(EvaluationRejected, match="temporally invalid"):
        replace(races[0], result_observed_at=races[0].forecast_computed_at)


def test_coverage_denominator_and_sealed_odds_report_are_explicitly_separate():
    races, forecasts = population(3)
    denominator = eligible(races) + [
        EligibleRaceIdentity("abstained-1", digest("3"), digest("4")),
        EligibleRaceIdentity("abstained-2", digest("3"), digest("4")),
    ]
    result = _evaluate_paired(
        races,
        forecasts,
        champion_bundle_id="champion",
        challenger_bundle_ids=("challenger",),
        eligible_population=denominator,
    )
    assert result["long_horizon"]["champion"]["coverage"] == 0.6
    assert result["long_horizon"]["champion"]["abstention"] == 0.4
    assert result["short_horizon"]["champion"]["coverage_denominator"] == 3
    odds_races = [
        replace(race, sealed_odds={"a": (2.0, 1.5), "b": (3.0, 1.8)}, odds_checksum=digest("6"))
        for race in races
    ]
    wagering = _evaluate_paired(
        odds_races,
        forecasts,
        champion_bundle_id="champion",
        challenger_bundle_ids=("challenger",),
        eligible_population=eligible(odds_races),
    )["wagering_scorecard"]
    assert wagering["report_only"] is True and wagering["real_betting"] is False
    assert set(wagering["models"]["champion"]) >= {"win", "place"}


def test_499_vs_500_and_venue_coverage_boundaries():
    assert promotion_decision(report(499), "challenger")["reasons"] == ["minimum_races"]
    assert promotion_decision(report(500), "challenger")["promote"] is True
    two_venues = report(500, venues=("A", "B"))
    assert "venue_coverage" in promotion_decision(two_venues, "challenger")["reasons"]


@pytest.mark.parametrize("minimum_races", (0, 1, 499))
def test_policy_and_forged_promotion_mapping_cannot_lower_500_race_floor(minimum_races):
    with pytest.raises(EvaluationRejected, match="500"):
        PromotionPolicy(minimum_races=minimum_races)
    forged = report(500)
    forged["policy"]["minimum_races"] = minimum_races
    with pytest.raises(EvaluationRejected, match="500"):
        promotion_decision(forged, "challenger")


def test_public_authenticated_499_race_evaluation_cannot_be_sealed(tmp_path):
    store, artifacts = _build_authentic_promotion_template(tmp_path, count=499)
    with pytest.raises(EvaluationRejected, match="complete durable 500-race population"):
        registered_report(store, artifacts)
    with store._connect() as db:
        assert tuple(
            db.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM phase6_forward_evaluation_races),"
                "(SELECT COUNT(*) FROM phase6_forecast_service_artifacts),"
                "(SELECT COUNT(*) FROM phase6_forecast_artifacts),"
                "(SELECT COUNT(*) FROM phase6_evaluation_evidence)"
            ).fetchone()
        ) == (499, 998, 998, 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_policy_rejects_nonfinite_thresholds(value):
    with pytest.raises(EvaluationRejected, match="finite"):
        PromotionPolicy(practical_loss_reduction=value)


def test_short_horizon_can_block_but_never_supply_long_horizon_gate():
    reversal = report(500, reverse_short=True)
    assert "short_horizon_reversal" in promotion_decision(reversal, "challenger")["reasons"]
    short_only = report(100)
    assert promotion_decision(short_only, "challenger")["decision"] == "retain_incumbent"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (
            lambda value: value["bootstrap"]["challenger"].update(probability_superior=0.5),
            "bootstrap_inconclusive",
        ),
        (
            lambda value: value["long_horizon"]["challenger"]["win_calibration"].update(ece=0.2),
            "calibration",
        ),
        (lambda value: value["long_horizon"]["challenger"].update(coverage=0.5), "coverage"),
        (
            lambda value: value["long_horizon"]["challenger"]["slices"]["venue"]["A"].update(
                mean_ordered_finish_nll=value["long_horizon"]["champion"]["slices"]["venue"]["A"][
                    "mean_ordered_finish_nll"
                ]
                + 1
            ),
            "slice:venue:A",
        ),
    ],
)
def test_inconclusive_and_guardrail_failures_retain_incumbent(mutation, expected):
    result = report()
    mutation(result)
    decision = promotion_decision(result, "challenger")
    assert decision["decision"] == "retain_incumbent"
    assert expected in decision["reasons"]


def test_drift_diagnosis_distinguishes_broad_and_champion_only():
    result = report()
    for model in result["short_horizon"]:
        result["short_horizon"][model]["mean_ordered_finish_nll"] = (
            result["long_horizon"][model]["mean_ordered_finish_nll"] + 1
        )
        result["short_horizon"][model]["race_losses"] = [
            result["short_horizon"][model]["mean_ordered_finish_nll"]
        ] * result["short_horizon"][model]["race_count"]
    assert diagnose_drift(result)["diagnosis"] == "data_domain_drift"
    result["short_horizon"]["challenger"]["mean_ordered_finish_nll"] = result["long_horizon"][
        "challenger"
    ]["mean_ordered_finish_nll"]
    assert diagnose_drift(result)["diagnosis"] == "model_drift"


def test_broad_drift_blocks_relatively_superior_challenger():
    result = report()
    for model in result["short_horizon"]:
        result["short_horizon"][model]["mean_ordered_finish_nll"] = (
            result["long_horizon"][model]["mean_ordered_finish_nll"] + 1
        )
        result["short_horizon"][model]["race_losses"] = [
            result["short_horizon"][model]["mean_ordered_finish_nll"]
        ] * result["short_horizon"][model]["race_count"]
    decision = promotion_decision(result, "challenger")
    assert diagnose_drift(result)["diagnosis"] == "data_domain_drift"
    assert decision["decision"] == "retain_incumbent"
    assert "data_domain_drift" in decision["reasons"]


@pytest.mark.parametrize("candidate_loss", (0.0, 0.1))
def test_perfect_incumbent_primary_loss_retains_without_nonfinite_reduction(candidate_loss):
    result = report()
    result["long_horizon"]["champion"]["mean_ordered_finish_nll"] = 0.0
    result["long_horizon"]["challenger"]["mean_ordered_finish_nll"] = candidate_loss
    decision = promotion_decision(result, "challenger")
    assert decision["decision"] == "retain_incumbent"
    assert "practical_loss" in decision["reasons"]
    assert decision["practical_loss_reduction"] == 0.0
    assert math.isfinite(decision["practical_loss_reduction"])


def test_migration_empty_and_repeat_paths(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "empty.sqlite3")
    store.migrate()
    store.migrate()
    with store._connect() as db:
        assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 30
        assert db.execute("SELECT COUNT(*) FROM phase6_promotion_records").fetchone()[0] == 0


def test_populated_upgrade_backfills_real_bundle_registration_run(tmp_path):
    class Phase5Store(SQLiteOperationsStore):
        def _migration_scripts(self):
            return tuple(item for item in super()._migration_scripts() if item[0] <= 17)

    path = tmp_path / "populated.sqlite3"
    phase5_store = Phase5Store(path)
    phase5_store.migrate()
    registration_operation = str(operation(490))
    checksum = "sha256:" + "4" * 64
    created_at = "2025-11-01T00:00:00+00:00"
    with phase5_store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (registration_operation, "register_canonical_bundle", "4" * 64, created_at),
        )
        db.execute(
            "INSERT INTO canonical_model_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "pre-phase6-bundle",
                "model",
                "canonical",
                None,
                checksum,
                "sealed-race-features-v1",
                "runner-win-probability-v1",
                checksum,
                checksum,
                checksum,
                checksum,
                checksum,
                "2025-10-01",
                checksum,
                checksum,
                checksum,
                created_at,
                registration_operation,
            ),
        )
        assert db.execute("SELECT COUNT(*) FROM phase6_runs").fetchone()[0] == 0

    SQLiteOperationsStore(path).migrate()
    with SQLiteOperationsStore(path)._connect() as db:
        assert tuple(
            db.execute("SELECT run_id,run_kind,started_at,operation_id FROM phase6_runs").fetchone()
        ) == (registration_operation, "registration", created_at, registration_operation)


def test_field_resolution_matches_normalizer_and_ignores_forged_critical_flag(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, _, race, *_ = setup
    forged_operation = str(operation(495))
    agreeing_rank_operation = str(operation(496))
    with store._connect() as db:
        db.executemany(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                (forged_operation, "field_evidence", "5" * 64, "2026-07-23T00:00:00+00:00"),
                (
                    agreeing_rank_operation,
                    "field_evidence",
                    "6" * 64,
                    "2026-07-23T00:00:00+00:00",
                ),
            ),
        )
        db.execute(
            "INSERT INTO field_evidence(race_id,field_name,authority,value_json,artifact_checksum,observed_at,critical,operation_id,source) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(race),
                "distance",
                "official_programme",
                json.dumps(450),
                "sha256:" + "6" * 64,
                "2026-07-23T00:00:00+00:00",
                0,
                agreeing_rank_operation,
                "a-source",
            ),
        )
        db.execute(
            "INSERT INTO field_evidence(race_id,field_name,authority,value_json,artifact_checksum,observed_at,critical,operation_id,source) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(race),
                "distance",
                "official_jump",
                json.dumps(500),
                "sha256:" + "5" * 64,
                "2026-07-23T00:00:00+00:00",
                1,  # Direct-SQL forgery: distance is intrinsically non-critical.
                forged_operation,
                "z-source",
            ),
        )
        rows = db.execute(
            "SELECT value_json,authority,source,artifact_checksum FROM field_evidence "
            "WHERE race_id=? AND field_name='distance'",
            (str(race),),
        ).fetchall()
        expected = normalize_fields(
            FieldObservation(
                EvidenceField.DISTANCE,
                json.loads(row["value_json"]),
                EvidenceAuthority(row["authority"]),
                False,
                row["source"],
                ArtifactChecksum(row["artifact_checksum"]),
            )
            for row in rows
        )["distance"]
        resolved = db.execute(
            "SELECT value_json,distinct_top_values,critical "
            "FROM phase6_resolved_field_evidence WHERE race_id=? AND field_name='distance'",
            (str(race),),
        ).fetchone()
        assert json.loads(resolved["value_json"]) == expected
        assert resolved["distinct_top_values"] == 2
        assert resolved["critical"] == 0


def test_field_resolution_excludes_unknown_authorities_and_field_identities(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, _, race, *_ = setup
    required = {
        "venue": "Ballarat",
        "distance": 450,
        "grade": "G5",
        "field_size": 2,
    }
    with store._connect() as db:
        for index, (field_name, valid_value) in enumerate(required.items()):
            unknown_authority_operation = str(operation(510 + index * 2))
            forged_field_operation = str(operation(511 + index * 2))
            for operation_id, payload in (
                (unknown_authority_operation, f"{index + 1:064x}"),
                (forged_field_operation, f"{index + 11:064x}"),
            ):
                db.execute(
                    "INSERT INTO operations VALUES(?,?,?,?)",
                    (
                        operation_id,
                        "field_evidence",
                        payload,
                        "2026-07-23T00:00:00+00:00",
                    ),
                )
            db.execute(
                "INSERT INTO field_evidence(race_id,field_name,authority,value_json,artifact_checksum,observed_at,critical,operation_id,source) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(race),
                    field_name,
                    "sql_forged_authority",
                    json.dumps("forged" if isinstance(valid_value, str) else 999),
                    "sha256:" + f"{index + 1:064x}",
                    "2026-07-23T00:00:00+00:00",
                    1,
                    unknown_authority_operation,
                    "forged-source",
                ),
            )
            forged_field_name = f"phase6_{field_name}"
            db.execute(
                "INSERT INTO field_evidence(race_id,field_name,authority,value_json,artifact_checksum,observed_at,critical,operation_id,source) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    str(race),
                    forged_field_name,
                    "official_programme",
                    json.dumps(valid_value),
                    "sha256:" + f"{index + 11:064x}",
                    "2026-07-23T00:00:00+00:00",
                    1,
                    forged_field_operation,
                    "forged-source",
                ),
            )
            resolved = db.execute(
                "SELECT value_json FROM phase6_resolved_field_evidence "
                "WHERE race_id=? AND field_name=?",
                (str(race), field_name),
            ).fetchone()
            assert json.loads(resolved["value_json"]) == valid_value
            assert (
                db.execute(
                    "SELECT COUNT(*) FROM phase6_resolved_field_evidence "
                    "WHERE race_id=? AND field_name=?",
                    (str(race), forged_field_name),
                ).fetchone()[0]
                == 0
            )


def test_populated_phase5_forward_row_registers_with_intact_source_trigger(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, _, race, *_ = setup
    example = build_durable_forward(setup)
    operation_id = "op_" + "deadbeef" * 4
    with store._connect() as db:
        assert db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='phase6_forward_evaluation_exact_source'"
        ).fetchone()
        db.execute(
            "INSERT OR IGNORE INTO operations VALUES(?,?,?,?)",
            (operation_id, "register_forward_evaluation", "f" * 64, "2026-07-23T00:00:00+00:00"),
        )
        for index, (authority, value) in enumerate(
            (
                ("official_programme", "Ballarat"),
                ("official_jump", "Ballarat"),
                ("source_card", "Geelong"),
            )
        ):
            field_operation = str(operation(500 + index))
            db.execute(
                "INSERT INTO operations VALUES(?,?,?,?)",
                (
                    field_operation,
                    "field_evidence",
                    f"{index + 1:064x}",
                    "2026-07-23T00:00:00+00:00",
                ),
            )
            db.execute(
                "INSERT INTO field_evidence(race_id,field_name,authority,value_json,artifact_checksum,observed_at,critical,operation_id) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (
                    str(race),
                    "venue",
                    authority,
                    json.dumps(value),
                    "sha256:" + f"{index + 1:064x}",
                    "2026-07-23T00:00:00+00:00",
                    1,
                    field_operation,
                ),
            )
        db.execute(
            "INSERT INTO phase6_forward_evaluation_races VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(race),
                example.example_id,
                example.racing_date,
                str(example.evidence_checksum),
                str(example.result_checksum),
                str(example.artifact_checksum),
                "Ballarat",
                450,
                "G5",
                2,
                "2026-07-23T00:00:00+00:00",
                operation_id,
            ),
        )
        assert db.execute("SELECT COUNT(*) FROM phase6_forward_evaluation_races").fetchone()[0] == 1
        mismatch_operation = str(operation(503))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                mismatch_operation,
                "register_forward_evaluation",
                "c" * 64,
                "2026-07-23T00:00:00+00:00",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="resolved"):
            db.execute(
                "INSERT INTO phase6_forward_evaluation_races VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(race),
                    example.example_id,
                    example.racing_date,
                    str(example.evidence_checksum),
                    str(example.result_checksum),
                    str(example.artifact_checksum),
                    "Ballarat",
                    999,
                    "G5",
                    2,
                    "2026-07-23T00:00:00+00:00",
                    mismatch_operation,
                ),
            )
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (str(operation(504)), "field_evidence", "a" * 64, "2026-07-23T00:00:00+00:00"),
        )
        db.execute(
            "INSERT INTO field_evidence(race_id,field_name,authority,value_json,artifact_checksum,observed_at,critical,operation_id) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (
                str(race),
                "venue",
                "official_programme",
                json.dumps("Geelong"),
                "sha256:" + "a" * 64,
                "2026-07-23T00:00:00+00:00",
                0,  # Direct-SQL forgery: venue remains intrinsically critical.
                str(operation(504)),
            ),
        )
        assert (
            db.execute(
                "SELECT critical FROM phase6_resolved_field_evidence "
                "WHERE race_id=? AND field_name='venue'",
                (str(race),),
            ).fetchone()[0]
            == 1
        )
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                "op_" + "feedface" * 4,
                "register_forward_evaluation",
                "b" * 64,
                "2026-07-23T00:00:00+00:00",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="resolved"):
            db.execute(
                "INSERT INTO phase6_forward_evaluation_races VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(race),
                    example.example_id,
                    example.racing_date,
                    str(example.evidence_checksum),
                    str(example.result_checksum),
                    str(example.artifact_checksum),
                    "Ballarat",
                    450,
                    "G5",
                    2,
                    "2026-07-23T00:00:00+00:00",
                    "op_" + "feedface" * 4,
                ),
            )


def test_public_forward_registration_replay_authenticates_exact_durable_row(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, artifacts, race, *_ = setup
    example = build_durable_forward(setup)
    authority = EvaluationAuthority(store, artifacts)
    registered_at = datetime(2026, 7, 23, 2, tzinfo=timezone.utc)
    operation_id = operation(5500)

    assert authority.register_forward_race(
        operation_id,
        training_example_id=example.example_id,
        registered_at=registered_at,
    )
    assert not authority.register_forward_race(
        operation_id,
        training_example_id=example.example_id,
        registered_at=registered_at,
    )
    with store._connect() as db:
        row = db.execute(
            "SELECT race_id,training_example_id,evidence_checksum,result_checksum,"
            "training_artifact_checksum,venue,distance_m,grade,field_size,registered_at "
            "FROM phase6_forward_evaluation_races WHERE operation_id=?",
            (str(operation_id),),
        ).fetchone()
    assert tuple(row) == (
        str(race),
        example.example_id,
        str(example.evidence_checksum),
        str(example.result_checksum),
        str(example.artifact_checksum),
        "Ballarat",
        450,
        "G5",
        2,
        registered_at.isoformat(timespec="microseconds"),
    )


@pytest.mark.parametrize(
    "field",
    (
        EvidenceField.VENUE,
        EvidenceField.DISTANCE,
        EvidenceField.GRADE,
        EvidenceField.FIELD_SIZE,
    ),
)
@pytest.mark.parametrize("authority_gap", ("missing", "ambiguous"))
def test_public_forward_registration_rejects_missing_or_ambiguous_slice_authority(
    tmp_path, field, authority_gap
):
    arguments = {
        f"{authority_gap}_slice": field,
    }
    with pytest.raises(EvaluationRejected, match="complete resolved Phase 2-5 authority"):
        _build_authentic_promotion_template(tmp_path, count=1, **arguments)
    store = SQLiteOperationsStore(tmp_path / "operations.db")
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM phase6_forward_evaluation_races").fetchone()[0] == 0


class _FixturePredictor:
    def __init__(self, artifact_checksum):
        self.artifact_checksum = artifact_checksum

    def predict(self, request):
        return self.artifact_checksum


def _build_authentic_promotion_template(
    root, count=500, *, missing_slice=None, ambiguous_slice=None, artifact_backed=False
):
    """Build expensive Phase 2/3 evidence once; tests copy its immutable baseline."""
    store = SQLiteOperationsStore(root / "operations.db")
    store.migrate()
    artifacts = LocalArtifactStore(root / "artifacts")
    backed = {
        label: artifacts.put(
            f"fixture-source-{label}".encode(), media_type="application/json"
        ).checksum
        for label in ("a", "2", "6", "e")
    }

    def source_checksum(label):
        return backed[label] if artifact_backed and label in backed else digest(label)

    now = "2026-07-22T00:00:00+00:00"
    lifecycle_at = datetime(2026, 7, 23, tzinfo=timezone.utc)
    result_at = lifecycle_at + timedelta(minutes=1)
    joined_at = lifecycle_at + timedelta(minutes=2)
    day = RacingDay(
        RacingDayId("day_" + "1" * 32),
        date(2026, 7, 23),
        "Australia/Melbourne",
        lifecycle_at,
    )
    store.create_racing_day(operation(6000), day)
    races = []
    evidence_by_race = {}
    for index in range(count):
        base = 10_000 + index * 20
        venue = ("Ballarat", "Geelong", "Sandown")[index % 3]
        race = store.record_expected_race(
            operation(base),
            day,
            ProgrammeRaceCandidate(
                "official", f"race-source-{index}", venue, index + 1, lifecycle_at
            ),
            source_checksum("a"),
            lifecycle_at,
        )
        odds = artifacts.put(
            canonical_json(
                {
                    "schema_version": "sealed-win-place-odds-v1",
                    "race_id": str(race),
                    "captured_at": (lifecycle_at - timedelta(seconds=1)).isoformat(),
                    "win_place": {"dog-a": [2.0, 1.5], "dog-b": [3.0, 1.8]},
                }
            ),
            media_type="application/json",
        )
        evidence = artifacts.put(
            canonical_json(
                {
                    "schema_version": "race-evidence-v1",
                    "normalization_version": "normalizer-v1",
                    "race_id": str(race),
                    "fields": {
                        "runner_set": ["dog-a", "dog-b"],
                        "runner_identity": {
                            "dog-a": "authoritative",
                            "dog-b": "authoritative",
                        },
                        "box": {"dog-a": 1, "dog-b": 2},
                        "runner_features": {
                            "dog-a": {"speed": 8 + index / 1000},
                            "dog-b": {"speed": 2},
                        },
                    },
                    "field_provenance": [
                        {
                            "field": "box",
                            "authority": "official_card",
                            "critical": True,
                            "value": {"dog-a": 1, "dog-b": 2},
                            "source": "official",
                            "artifact_checksum": str(source_checksum("a")),
                        }
                    ],
                    "freeze": {
                        "at": lifecycle_at.isoformat(),
                        "authority": "actual_jump",
                        "odds_checksum": str(odds.checksum),
                    },
                }
            ),
            media_type="application/json",
        )
        store.advance_race(operation(base + 1), race, RaceState.CARD_COLLECTED, lifecycle_at)
        store.advance_race(operation(base + 2), race, RaceState.COLLECTING_ODDS, lifecycle_at)
        for offset, field, value in (
            (3, EvidenceField.VENUE, venue),
            (4, EvidenceField.DISTANCE, 450),
            (5, EvidenceField.GRADE, "G5"),
            (6, EvidenceField.FIELD_SIZE, 2),
        ):
            if field is missing_slice:
                continue
            store.record_field_evidence(
                FieldEvidence(
                    operation(base + offset),
                    race,
                    field,
                    EvidenceAuthority.OFFICIAL_CARD,
                    value,
                    "official",
                    source_checksum("a"),
                    lifecycle_at,
                )
            )
            if field is ambiguous_slice:
                conflicting_value = {
                    EvidenceField.VENUE: "Warragul",
                    EvidenceField.DISTANCE: 451,
                    EvidenceField.GRADE: "G4",
                    EvidenceField.FIELD_SIZE: 3,
                }[field]
                store.record_field_evidence(
                    FieldEvidence(
                        operation(base + 9),
                        race,
                        field,
                        EvidenceAuthority.OFFICIAL_CARD,
                        conflicting_value,
                        "official-conflict",
                        digest("b"),
                        lifecycle_at,
                    )
                )
        store.seal_evidence(
            operation(base + 7),
            race_id=race,
            raw_checksum=source_checksum("a"),
            normalized_checksum=evidence.checksum,
            schema_version="race-evidence-v1",
            normalization_version="normalizer-v1",
            frozen_at=lifecycle_at,
            freeze_authority=FreezeAuthority.ACTUAL_JUMP,
            odds_checksum=odds.checksum,
            sealed_at=lifecycle_at,
            request_intent_digest=digest("f"),
        )
        store.advance_race(operation(base + 8), race, RaceState.AWAITING_DAY_CLOSE, lifecycle_at)
        races.append(race)
        evidence_by_race[race] = evidence

    phase3 = ForecastingAuthority(store)
    legacy_bundle = LegacyBundle(
        "legacy-bundle",
        "legacy-model",
        source_checksum("6"),
        10,
        source_checksum("2"),
        None,
        "raw_registry_model",
        {"fixture": True},
    )
    release = ModelRelease(
        "legacy-release", legacy_bundle.bundle_id, "policy-v1", {"fixture": True}
    )
    phase3.register_bundle(operation(6001), legacy_bundle, lifecycle_at)
    phase3.register_release(operation(6002), release, lifecycle_at)
    phase3.pin_day(operation(6003), day.id, release, lifecycle_at)
    store.close_racing_day(operation(6004), day, lifecycle_at)

    schema = canonical_json(
        {
            "bundle_id": "challenger-1",
            "contract_version": "sealed-race-features-v1",
            "evidence_schema_version": "race-evidence-v1",
            "normalization_version": "normalizer-v1",
            "fields": [
                {
                    "name": "speed",
                    "semantics": "forecast-required",
                    "source_field": "runner_features",
                }
            ],
        }
    )
    missingness = canonical_json(
        {
            "bundle_id": "challenger-1",
            "feature_contract_version": "sealed-race-features-v1",
            "imputation": {},
        }
    )
    corpus = TrainingCorpusAuthority(store, artifacts)
    evaluation = EvaluationAuthority(store, artifacts)
    for index, race in enumerate(races):
        base = 30_000 + index * 10
        phase3.begin_prediction(operation(base), race, lifecycle_at)
        phase3.predict(
            operation(base + 1),
            race,
            f"prediction-{index}",
            _FixturePredictor(evidence_by_race[race].checksum),
            lifecycle_at,
        )

    for index, race in enumerate(races):
        base = 30_000 + index * 10
        phase3.open_results(operation(base + 2), race, lifecycle_at)
        outcome = {
            "order": [1, 2],
            "official": True,
            "published_at": result_at.isoformat(),
            "provenance": {
                "source": "official",
                "source_record_id": f"result-source-{index}",
                "observed_at": result_at.isoformat(),
            },
            "exclusions": [],
        }
        result = artifacts.put(canonical_json(outcome), media_type="application/json")
        phase3.record_result_attempt(
            operation(base + 3),
            race,
            f"result-attempt-{index}",
            at=result_at,
            max_attempts=1,
            deadline=result_at,
            artifact_checksum=result.checksum,
            outcome=outcome,
        )
        phase3.join_training_example(
            operation(base + 4),
            race,
            f"phase3-example-{index}",
            source_checksum("e"),
            eligible=True,
            reason=None,
            at=result_at,
        )
        example = corpus.build_forward_example(
            operation(base + 5),
            phase3_example_id=f"phase3-example-{index}",
            example_id=f"canonical-example-{index}",
            schema_bytes=schema,
            schema_checksum=checksum(schema),
            missingness_bytes=missingness,
            missingness_checksum=checksum(missingness),
            joined_at=joined_at,
        )
        assert evaluation.register_forward_race(
            operation(base + 6),
            training_example_id=example.example_id,
            registered_at=lifecycle_at,
        )

    bundle_created = datetime(2025, 12, 1, tzinfo=timezone.utc)
    bundles = []
    for index, bundle_id in enumerate(("champion", "challenger")):
        schema = canonical_json(
            {
                "bundle_id": bundle_id,
                "contract_version": "sealed-race-features-v1",
                "evidence_schema_version": "race-evidence-v1",
                "normalization_version": "normalizer-v1",
                "fields": [
                    {
                        "name": "speed",
                        "semantics": "forecast-required",
                        "source_field": "runner_features",
                    }
                ],
            }
        )
        missingness = canonical_json(
            {
                "bundle_id": bundle_id,
                "feature_contract_version": "sealed-race-features-v1",
                "imputation": {},
            }
        )
        values = (2, 1) if index == 0 else (8, 1)
        finish = (1, 0) if index == 0 else (0, 1)
        bundles.append(
            train_challenger_bundle(
                model_id=f"model-{index}",
                bundle_id=bundle_id,
                examples=(
                    synthetic_prepared_example(f"{bundle_id}-1", "2026-07-01", values, finish),
                    synthetic_prepared_example(f"{bundle_id}-2", "2026-07-02", values, finish),
                ),
                schema_bytes=schema,
                missingness_bytes=missingness,
                validation={"method": "expanding-window-temporal-v1", "folds": []},
                seed=7341,
                artifacts=artifacts,
            )
        )
    bundle_authority = ModelBundleAuthority(store)
    for index, bundle in enumerate(bundles):
        bundle_authority.register(operation(700 + index), bundle, bundle_created)
    with store._connect() as db:
        db.execute("BEGIN")
        for index in range(4):
            setup_operation = str(operation(600 + index))
            db.execute(
                "INSERT INTO operations VALUES(?,?,?,?)",
                (setup_operation, "setup", str(index) * 64, now),
            )
        db.execute(
            "INSERT INTO canonical_serving_assignments VALUES(?,?,?,?,?,?,?,?)",
            (
                "assignment-old",
                "champion",
                str(bundles[0].bundle_checksum),
                now,
                "2026-07-22",
                "bootstrap",
                now,
                str(operation(602)),
            ),
        )
        db.execute(
            "INSERT INTO champion_pointer VALUES(1,?,?,?,?,?)",
            (
                "assignment-old",
                "champion",
                str(bundles[0].bundle_checksum),
                now,
                str(operation(603)),
            ),
        )
        db.commit()
    return store, artifacts


_PROMOTION_TEMPLATE = None


@pytest.fixture(scope="session", autouse=True)
def authentic_promotion_template(tmp_path_factory):
    global _PROMOTION_TEMPLATE
    _PROMOTION_TEMPLATE = tmp_path_factory.mktemp("authentic-phase6-template")
    _build_authentic_promotion_template(_PROMOTION_TEMPLATE)
    registered_report(
        SQLiteOperationsStore(_PROMOTION_TEMPLATE / "operations.db"),
        LocalArtifactStore(_PROMOTION_TEMPLATE / "artifacts"),
    )


def promotion_store(tmp_path):
    assert _PROMOTION_TEMPLATE is not None
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_PROMOTION_TEMPLATE / "operations.db", tmp_path / "operations.db")
    shutil.copytree(_PROMOTION_TEMPLATE / "artifacts", tmp_path / "artifacts")
    return (
        SQLiteOperationsStore(tmp_path / "operations.db"),
        LocalArtifactStore(tmp_path / "artifacts"),
    )


def registered_report(
    store, artifacts, *, evidence_id="evidence", force_evaluate=False, evaluated_at=None
):
    with store._connect() as db:
        sealed = db.execute(
            "SELECT artifact_checksum FROM phase6_evaluation_evidence WHERE evidence_id=?",
            (evidence_id,),
        ).fetchone()
        if sealed is not None and not force_evaluate:
            for row in db.execute("SELECT artifact_checksum FROM canonical_bundle_components"):
                artifacts.verify(ArtifactChecksum(row["artifact_checksum"]))
            policy = db.execute("SELECT artifact_checksum FROM phase6_policy_registry").fetchone()
            if policy is None:
                raise EvaluationRejected("promotion policy is not registered")
            artifacts.verify(ArtifactChecksum(policy["artifact_checksum"]))
            return json.loads(artifacts.read(ArtifactChecksum(sealed["artifact_checksum"])))
        forwards = db.execute(
            "SELECT * FROM phase6_forward_evaluation_races ORDER BY race_id"
        ).fetchall()
        checksums = {
            row["bundle_id"]: ArtifactChecksum(row["bundle_checksum"])
            for row in db.execute("SELECT bundle_id,bundle_checksum FROM canonical_model_bundles")
        }
        deferred_by_race = {
            row["race_id"]: row
            for row in db.execute(
                "SELECT race_id,prediction_id,computed_at FROM deferred_predictions"
            )
        }
        odds_by_race = {
            row["race_id"]: ArtifactChecksum(row["odds_checksum"])
            for row in db.execute("SELECT race_id,odds_checksum FROM sealed_evidence")
        }
    races = []
    for forward in forwards:
        training = json.loads(
            artifacts.read(ArtifactChecksum(forward["training_artifact_checksum"]))
        )
        odds_checksum = odds_by_race[forward["race_id"]]
        odds = json.loads(artifacts.read(odds_checksum))
        races.append(
            EvaluationRace(
                forward["race_id"],
                forward["racing_day"],
                forward["venue"],
                forward["distance_m"],
                forward["grade"],
                tuple(training["official_order"]),
                ArtifactChecksum(forward["evidence_checksum"]),
                ArtifactChecksum(forward["result_checksum"]),
                training["evidence_frozen_at"],
                deferred_by_race[forward["race_id"]]["computed_at"],
                training["result_published_at"],
                sealed_odds={runner: tuple(prices) for runner, prices in odds["win_place"].items()},
                odds_checksum=odds_checksum,
            )
        )
    computed = datetime.fromisoformat(races[0].forecast_computed_at)
    authority = EvaluationAuthority(store, artifacts)
    training_run = operation(790)
    registration_runs = {"champion": operation(700), "challenger": operation(701)}
    service_runs = {"champion": operation(792), "challenger": operation(792)}
    authority.begin_run(
        training_run, run_kind="training", started_at=datetime(2025, 10, 1, tzinfo=timezone.utc)
    )
    for bundle_id, service_run in service_runs.items():
        authority.begin_run(service_run, run_kind="forecast_service", started_at=computed)
    for index, bundle_id in enumerate(("champion", "challenger")):
        authority.record_bundle_lineage(
            operation(796 + index),
            bundle_id=bundle_id,
            registration_run_id=registration_runs[bundle_id],
            source_run_id=training_run,
        )
    loader = ChampionLoader(store, artifacts, deserializer=LinearStrengthModel.from_bytes)
    service = CanonicalForecastService(loader, artifacts)
    service_operation = 1000
    forecasts = {"champion": [], "challenger": []}
    with store._connect() as db:
        service_artifacts = {
            (row["race_id"], row["bundle_id"]): row
            for row in db.execute("SELECT * FROM phase6_forecast_service_artifacts")
        }
    if service_artifacts and len(service_artifacts) != len(races) * 2:
        raise AssertionError("authenticated service forecast baseline is incomplete")
    for bundle_id in ("champion", "challenger"):
        for index, race in enumerate(races):
            durable_service = service_artifacts.get((race.race_id, bundle_id))
            if durable_service is None:
                checksum_value = service.persist_evaluation_forecast(
                    operation(service_operation),
                    service_run_id=service_runs[bundle_id],
                    race_id=race.race_id,
                    bundle_id=bundle_id,
                    bundle_checksum=checksums[bundle_id],
                    evidence_checksum=race.evidence_checksum,
                    computed_at=datetime.fromisoformat(race.forecast_computed_at),
                    computation_id=f"computation-{bundle_id}-{index}",
                )
                with store._connect() as db:
                    durable_service = db.execute(
                        "SELECT * FROM phase6_forecast_service_artifacts "
                        "WHERE forecast_checksum=? AND race_id=? AND bundle_id=?",
                        (str(checksum_value), race.race_id, bundle_id),
                    ).fetchone()
            else:
                checksum_value = ArtifactChecksum(durable_service["forecast_checksum"])
            artifact_checksum = ArtifactChecksum(durable_service["artifact_checksum"])
            document = json.loads(artifacts.read(artifact_checksum))
            distribution = ordered_finish_from_probabilities(
                tuple(document["distribution"]["runner_ids"]),
                {
                    tuple(order): probability
                    for order, probability in document["distribution"]["orders"]
                },
            )
            forecasts[bundle_id].append(
                ForecastEvidence(
                    race.race_id,
                    bundle_id,
                    checksums[bundle_id],
                    race.evidence_checksum,
                    checksum_value,
                    distribution,
                    race.forecast_computed_at,
                )
            )
            service_operation += 1
    with store._connect() as db:
        assert tuple(
            db.execute(
                "SELECT COUNT(*),COUNT(DISTINCT service_run_id) "
                "FROM phase6_forecast_service_artifacts"
            ).fetchone()
        ) == (len(races) * 2, 1)
    evaluation_run = operation(800)
    authority.begin_run(
        evaluation_run,
        run_kind="evaluation",
        started_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )
    policy = PromotionPolicy()
    authority.register_policy(operation(801), policy, datetime(2026, 7, 21, tzinfo=timezone.utc))
    forecast_operation = 2000
    for values in forecasts.values():
        for forecast in values:
            authority.record_forecast(
                operation(forecast_operation), forecast, evaluation_run_id=evaluation_run
            )
            forecast_operation += 1
    durable_report = authority.evaluate_and_seal(
        operation(900),
        evidence_id=evidence_id,
        evaluated_at=evaluated_at
        or datetime.fromisoformat(races[-1].result_observed_at) + timedelta(minutes=1),
        races=races,
        model_forecasts=forecasts,
        evaluation_run_id=evaluation_run,
        champion_bundle_id="champion",
        challenger_bundle_ids=("challenger",),
        policy=policy,
        eligible_population=eligible(races),
    )
    with store._connect() as db:
        assert tuple(
            db.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM phase6_forward_evaluation_races),"
                "(SELECT COUNT(*) FROM phase6_forecast_service_artifacts),"
                "(SELECT COUNT(*) FROM phase6_forecast_artifacts),"
                "(SELECT COUNT(*) FROM phase6_evaluation_evidence WHERE evidence_id=?)",
                (evidence_id,),
            ).fetchone()
        ) == (500, 1000, 1000, 1)
    return durable_report


PHASE7_PROMOTION_APPROVED_AT = datetime(2026, 8, 6, 23, 30, tzinfo=timezone.utc)
PHASE7_PROMOTION_EFFECTIVE_DATE = "2026-08-07"


def seed_external_probation(store, artifacts, probation_id, count):
    """Issue synthetic probation only through the public Phase 7 authority."""
    from test_phase7_operational import NOW as PHASE7_NOW
    from test_phase7_operational import Phase7OperationalTests

    fixture = Phase7OperationalTests(methodName="runTest")
    fixture.store = store
    fixture.artifacts = artifacts
    fixture.authority = OperationalAuthority(store, artifacts, clock=lambda: PHASE7_NOW)
    with store._connect() as db:
        pointer = db.execute(
            "SELECT 1 FROM phase7_release_pointer WHERE singleton=1 "
            "AND authority='race_collection_service'"
        ).fetchone()
    if pointer is None:
        boundary = fixture.establish_cutover(
            930000,
            930100,
            date(2026, 7, 23),
            boundary_timezone="UTC",
        )
        predecessor = boundary
        for offset in range(14):
            predecessor = fixture.seed_complete_day(
                930200 + offset,
                date(2026, 7, 24) + timedelta(days=offset),
                release="candidate",
                predecessor=predecessor,
            )
        with store._operation(operation(930500), "synthetic_effective_day", {}) as (db, _):
            db.execute(
                "INSERT INTO racing_days VALUES(?,?,?,?,NULL)",
                (
                    "day_" + f"{930500:032x}",
                    PHASE7_PROMOTION_EFFECTIVE_DATE,
                    "UTC",
                    "2026-08-06T23:45:00+00:00",
                ),
            )
            db.execute(
                "INSERT INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                (
                    "day_" + f"{930500:032x}",
                    predecessor,
                    "sha256:" + f"{930500:064x}",
                    "2026-08-06T23:45:00+00:00",
                    str(operation(930500)),
                ),
            )
    with store._connect() as db:
        days = db.execute(
            "SELECT e.racing_day_id,d.local_date,d.closed_at,e.recorded_at,"
            "r.reconciled_at FROM phase7_day_evidence e "
            "JOIN racing_days d USING(racing_day_id) "
            "JOIN phase7_reconciliation r USING(racing_day_id) "
            "WHERE e.release_id='candidate' "
            "AND d.local_date BETWEEN '2026-07-24' AND '2026-08-06' "
            "ORDER BY d.local_date"
        ).fetchall()
        accepted = db.execute("SELECT count(*) FROM phase7_probation_acceptances").fetchone()[0]
    authority_floor = max(
        datetime.fromisoformat(value)
        for day in days
        for value in (day["closed_at"], day["recorded_at"], day["reconciled_at"])
    )
    acceptance_start = max(
        authority_floor + timedelta(microseconds=1),
        datetime(2026, 8, 6, 23, tzinfo=timezone.utc),
    )
    for offset, day in enumerate(days[accepted:count], accepted):
        fixture.authority.record_probation_day(
            operation(930600 + offset),
            racing_day_id=day["racing_day_id"],
            at=acceptance_start + timedelta(microseconds=offset),
        )
    if count == 14:
        sealed_at = acceptance_start + timedelta(microseconds=14)
        if sealed_at >= PHASE7_PROMOTION_APPROVED_AT:
            raise AssertionError("authentic probation cannot be sealed before approval")
        fixture.authority.seal_probation(
            operation(930700),
            probation_id=probation_id,
            at=sealed_at,
        )
    return days[-1]["racing_day_id"]


def racing_day_id(day):
    return "day_" + ("1" * 32 if day == 23 else f"{day:032x}")


def programme_checksum(day):
    return "sha256:" + f"{day:064x}"


def schedule_days(store):
    with store._connect() as db:
        for day in range(1, 25):
            operation_id = str(operation(2000 + day))
            db.execute(
                "INSERT OR IGNORE INTO operations VALUES(?,?,?,?)",
                (operation_id, "schedule_fixture", f"{day:064x}", "2026-07-01T00:00:00+00:00"),
            )
            db.execute(
                "INSERT OR IGNORE INTO racing_days VALUES(?,?,?,?,?)",
                (
                    racing_day_id(day),
                    f"2026-07-{day:02}",
                    "Australia/Melbourne",
                    "2026-07-01T00:00:00+00:00",
                    None,
                ),
            )
            db.execute(
                "INSERT OR IGNORE INTO phase6_racing_day_schedule VALUES(?,?,?,?,?)",
                (
                    racing_day_id(day),
                    racing_day_id(day - 1) if day > 1 else None,
                    programme_checksum(day),
                    "2026-07-01T00:00:00+00:00",
                    operation_id,
                ),
            )


def test_atomic_next_day_promotion_probation_training_and_failure_guards(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    store.migrate()  # populated repeat path
    authority = PromotionAuthority(store, artifacts)
    result = registered_report(store, artifacts)
    assert result["wagering_scorecard"]["status"] == "RUN"
    schedule_days(store)
    EvaluationAuthority(store, artifacts).begin_run(
        operation(849), run_kind="training", started_at=datetime(2026, 7, 23, tzinfo=timezone.utc)
    )
    EvaluationAuthority(store, artifacts).begin_run(
        operation(850),
        run_kind="promotion",
        started_at=datetime(2026, 7, 23, 1, tzinfo=timezone.utc),
    )
    approved = datetime(2026, 7, 23, 1, tzinfo=timezone.utc)
    EvaluationAuthority(store, artifacts).begin_run(
        operation(851),
        run_kind="promotion",
        started_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
    )
    with pytest.raises(PromotionRejected, match="runs overlap"):
        authority.promote(
            operation(5),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=approved,
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="reversed",
            probation_id="missing",
            promotion_run_id=operation(851),
            approval_racing_day_id=racing_day_id(23),
        )
    with pytest.raises(PromotionRejected, match="runs overlap"):
        authority.promote(
            operation(6),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=datetime(2026, 7, 22, tzinfo=timezone.utc),
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="backdated",
            probation_id="missing",
            promotion_run_id=operation(850),
            approval_racing_day_id=racing_day_id(23),
        )
    EvaluationAuthority(store, artifacts).begin_run(
        operation(852),
        run_kind="promotion",
        started_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
    )
    with pytest.raises(PromotionRejected, match="post-effective"):
        authority.promote(
            operation(7),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=datetime(2026, 7, 24, 1, tzinfo=timezone.utc),
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="late",
            probation_id="missing",
            promotion_run_id=operation(852),
            approval_racing_day_id=racing_day_id(23),
        )
    forged = copy.deepcopy(result)
    forged["population_checksum"] = str(digest("9"))
    with pytest.raises(PromotionRejected, match="population checksum"):
        authority.promote(
            operation(9),
            evidence_id="evidence",
            report=forged,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=approved,
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="forged",
            probation_id="missing",
            promotion_run_id=operation(850),
            approval_racing_day_id=racing_day_id(23),
        )
    with pytest.raises(PromotionRejected, match="promotion"):
        authority.promote(
            operation(10),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=approved,
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="gates passed",
            probation_id="missing",
            promotion_run_id=operation(849),
            approval_racing_day_id=racing_day_id(23),
        )
    with pytest.raises(PromotionRejected, match="probation"):
        authority.promote(
            operation(11),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=approved,
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="gates passed",
            probation_id="missing",
            promotion_run_id=operation(850),
            approval_racing_day_id=racing_day_id(23),
        )
    with store._connect() as db:
        assert db.execute("SELECT bundle_id FROM champion_pointer").fetchone()[0] == "champion"
        assert db.execute("SELECT COUNT(*) FROM phase6_promotion_records").fetchone()[0] == 0
    phase7_approval_day = seed_external_probation(store, artifacts, "probation-13", 13)
    with pytest.raises(PromotionRejected, match="probation"):
        authority.promote(
            operation(13),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=PHASE7_PROMOTION_APPROVED_AT,
            effective_racing_day=PHASE7_PROMOTION_EFFECTIVE_DATE,
            approver="policy",
            reason="gates passed",
            probation_id="probation-13",
            promotion_run_id=operation(850),
            approval_racing_day_id=phase7_approval_day,
        )
    phase7_approval_day = seed_external_probation(store, artifacts, "probation-14", 14)
    assignment = authority.promote(
        operation(15),
        evidence_id="evidence",
        report=result,
        challenger_bundle_id="challenger",
        assignment_id="assignment-new",
        promotion_record_id="promotion-new",
        approved_at=PHASE7_PROMOTION_APPROVED_AT,
        effective_racing_day=PHASE7_PROMOTION_EFFECTIVE_DATE,
        approver="automatic-policy",
        reason="all phase6 gates passed",
        probation_id="probation-14",
        promotion_run_id=operation(850),
        approval_racing_day_id=phase7_approval_day,
    )
    assert assignment.bundle_id == "challenger"
    assert (
        authority.promote(
            operation(15),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=PHASE7_PROMOTION_APPROVED_AT,
            effective_racing_day=PHASE7_PROMOTION_EFFECTIVE_DATE,
            approver="automatic-policy",
            reason="all phase6 gates passed",
            probation_id="probation-14",
            promotion_run_id=operation(850),
            approval_racing_day_id=phase7_approval_day,
        )
        == assignment
    )
    with pytest.raises(ConflictingOperation):
        authority.promote(
            operation(15),
            evidence_id="different-evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="assignment-new",
            promotion_record_id="promotion-new",
            approved_at=PHASE7_PROMOTION_APPROVED_AT,
            effective_racing_day=PHASE7_PROMOTION_EFFECTIVE_DATE,
            approver="automatic-policy",
            reason="all phase6 gates passed",
            probation_id="probation-14",
            promotion_run_id=operation(850),
            approval_racing_day_id=phase7_approval_day,
        )
    with store._connect() as db:
        pointer = db.execute("SELECT assignment_id,bundle_id FROM champion_pointer").fetchone()
        next_pointer = db.execute(
            "SELECT assignment_id,bundle_id,effective_racing_day,rollback_assignment_id FROM next_champion_pointer"
        ).fetchone()
        record = db.execute(
            "SELECT prior_assignment_id,next_assignment_id,component_checksums_json FROM phase6_promotion_records"
        ).fetchone()
        assert tuple(pointer) == ("assignment-old", "champion")
        assert tuple(next_pointer) == (
            "assignment-new",
            "challenger",
            PHASE7_PROMOTION_EFFECTIVE_DATE,
            "assignment-old",
        )
        assert tuple(record)[:2] == ("assignment-old", "assignment-new")
        assert len(json.loads(record[2])) == 9
        assignment_operation = db.execute(
            "SELECT operation_id FROM canonical_serving_assignments WHERE assignment_id='assignment-new'"
        ).fetchone()[0]
        assert assignment_operation.startswith("op_") and len(assignment_operation) == 35
        successors = db.execute(
            "SELECT s.racing_day_id,d.local_date,d.timezone "
            "FROM phase6_racing_day_schedule s "
            "JOIN racing_days d USING(racing_day_id) "
            "WHERE s.predecessor_racing_day_id=?",
            (phase7_approval_day,),
        ).fetchall()
        assert len(successors) == 1
        assert tuple(successors[0])[1:] == (PHASE7_PROMOTION_EFFECTIVE_DATE, "UTC")
        effective_racing_day_id = successors[0]["racing_day_id"]
        assert effective_racing_day_id != racing_day_id(24)
        with pytest.raises(sqlite3.OperationalError, match="view"):
            db.execute("UPDATE next_champion_pointer SET bundle_id='champion'")
    assert authority.resolve_scheduled_assignment(effective_racing_day_id) == "assignment-new"
    assert (
        authority.rollback_staged(
            operation(860),
            rollback_id="rollback-1",
            staged_assignment_id="assignment-new",
            reason="guarded rollback",
            rolled_back_at=PHASE7_PROMOTION_APPROVED_AT,
        )
        == "assignment-old"
    )
    with store._connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE phase6_assignment_history SET assignment_id='assignment-new'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM phase6_assignment_history")
        forged_operation = str(operation(861))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                forged_operation,
                "rollback_phase6_staged_assignment",
                "f" * 64,
                PHASE7_PROMOTION_APPROVED_AT.isoformat(),
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="exact staged target"):
            db.execute(
                "INSERT INTO phase6_rollback_records VALUES(?,?,?,?,?,?)",
                (
                    "forged-rollback",
                    "assignment-new",
                    "assignment-new",
                    "forged",
                    PHASE7_PROMOTION_APPROVED_AT.isoformat(),
                    forged_operation,
                ),
            )
        history_operation = str(operation(862))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                history_operation,
                "rollback_phase6_staged_assignment",
                "e" * 64,
                PHASE7_PROMOTION_APPROVED_AT.isoformat(),
            ),
        )
        unrelated_melbourne_day = racing_day_id(23)
        assert unrelated_melbourne_day not in (
            phase7_approval_day,
            effective_racing_day_id,
        )
        with pytest.raises(sqlite3.IntegrityError, match="authoritative chain"):
            db.execute(
                "INSERT INTO phase6_assignment_history VALUES(?,?,?,?,?,?,?)",
                (
                    "wrong-day-history",
                    unrelated_melbourne_day,
                    "assignment-old",
                    "rollback_restored",
                    "promotion-new:history",
                    PHASE7_PROMOTION_APPROVED_AT.isoformat(),
                    history_operation,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError, match="authoritative chain"):
            db.execute(
                "INSERT INTO phase6_assignment_history VALUES(?,?,?,?,?,?,?)",
                (
                    "wrong-action-history",
                    effective_racing_day_id,
                    "assignment-old",
                    "promoted",
                    None,
                    PHASE7_PROMOTION_APPROVED_AT.isoformat(),
                    history_operation,
                ),
            )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO phase6_assignment_history VALUES(?,?,?,?,?,?,?)",
                (
                    "competing-history",
                    effective_racing_day_id,
                    "assignment-old",
                    "rollback_restored",
                    "promotion-new:history",
                    PHASE7_PROMOTION_APPROVED_AT.isoformat(),
                    history_operation,
                ),
            )
    assert authority.resolve_scheduled_assignment(effective_racing_day_id) == "assignment-old"
    with store._connect() as db:
        assert tuple(
            db.execute("SELECT assignment_id,bundle_id FROM next_champion_pointer").fetchone()
        ) == ("assignment-old", "champion")
        assert [
            tuple(row)
            for row in db.execute(
                "SELECT action,assignment_id FROM phase6_assignment_history ORDER BY rowid"
            )
        ] == [("promoted", "assignment-new"), ("rollback_restored", "assignment-old")]
    assert (
        authority.rollback_staged(
            operation(860),
            rollback_id="rollback-1",
            staged_assignment_id="assignment-new",
            reason="guarded rollback",
            rolled_back_at=PHASE7_PROMOTION_APPROVED_AT,
        )
        == "assignment-old"
    )
    with pytest.raises(ConflictingOperation):
        authority.rollback_staged(
            operation(860),
            rollback_id="rollback-2",
            staged_assignment_id="assignment-new",
            reason="different intent",
            rolled_back_at=PHASE7_PROMOTION_APPROVED_AT,
        )


@pytest.mark.parametrize(
    "changed_field",
    ("assignment_id", "approved_at", "approver", "reason", "probation_id", "rolled_back_at"),
)
def test_promotion_and_rollback_replay_payload_covers_all_durable_intent(tmp_path, changed_field):
    store, artifacts = promotion_store(tmp_path)
    authority = PromotionAuthority(store, artifacts)
    report = registered_report(store, artifacts)
    approved = PHASE7_PROMOTION_APPROVED_AT
    phase7_approval_day = seed_external_probation(store, artifacts, "probation-14", 14)
    EvaluationAuthority(store, artifacts).begin_run(
        operation(850), run_kind="promotion", started_at=approved
    )
    arguments = {
        "evidence_id": "evidence",
        "report": report,
        "challenger_bundle_id": "challenger",
        "assignment_id": "assignment-new",
        "promotion_record_id": "promotion-new",
        "approved_at": approved,
        "effective_racing_day": PHASE7_PROMOTION_EFFECTIVE_DATE,
        "approver": "automatic-policy",
        "reason": "all phase6 gates passed",
        "probation_id": "probation-14",
        "promotion_run_id": operation(850),
        "approval_racing_day_id": phase7_approval_day,
    }
    expected = authority.promote(operation(15), **arguments)
    assert authority.promote(operation(15), **arguments) == expected

    if changed_field == "rolled_back_at":
        rollback_arguments = {
            "rollback_id": "rollback-1",
            "staged_assignment_id": "assignment-new",
            "reason": "guarded rollback",
            "rolled_back_at": approved,
        }
        assert authority.rollback_staged(operation(860), **rollback_arguments) == "assignment-old"
        assert authority.rollback_staged(operation(860), **rollback_arguments) == "assignment-old"
        rollback_arguments["rolled_back_at"] = approved + timedelta(seconds=1)
        with pytest.raises(ConflictingOperation):
            authority.rollback_staged(operation(860), **rollback_arguments)
        return

    changed = dict(arguments)
    changed[changed_field] = {
        "assignment_id": "assignment-other",
        "approved_at": approved + timedelta(seconds=1),
        "approver": "different-approver",
        "reason": "different-reason",
        "probation_id": "different-probation",
    }[changed_field]
    with pytest.raises(ConflictingOperation):
        authority.promote(operation(15), **changed)


def test_policy_and_evaluation_seal_replay_cover_durable_timestamps(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "policy.sqlite3")
    store.migrate()
    authority = EvaluationAuthority(store, LocalArtifactStore(tmp_path / "policy-artifacts"))
    policy = PromotionPolicy()
    registered_at = datetime(2026, 7, 21, tzinfo=timezone.utc)
    checksum = authority.register_policy(operation(4700), policy, registered_at)
    assert authority.register_policy(operation(4700), policy, registered_at) == checksum
    with pytest.raises(ConflictingOperation):
        authority.register_policy(operation(4700), policy, registered_at + timedelta(seconds=1))

    durable_store, artifacts = promotion_store(tmp_path / "seal")
    sealed_report = registered_report(durable_store, artifacts)
    with durable_store._connect() as db:
        sealed_row = db.execute(
            "SELECT evaluated_at,artifact_checksum FROM phase6_evaluation_evidence "
            "WHERE evidence_id='evidence'"
        ).fetchone()
        evaluated_at = datetime.fromisoformat(sealed_row["evaluated_at"])
    assert (
        registered_report(
            durable_store,
            artifacts,
            force_evaluate=True,
            evaluated_at=evaluated_at,
        )
        == sealed_report
    )
    with pytest.raises(ConflictingOperation):
        registered_report(
            durable_store,
            artifacts,
            force_evaluate=True,
            evaluated_at=evaluated_at + timedelta(seconds=1),
        )


def test_sealed_report_rejects_forged_drift_diagnosis(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    sealed_report = registered_report(store, artifacts)
    sealed_report["drift_diagnosis"] = {
        "diagnosis": "data_domain_drift",
        "action": "forged action",
    }
    with pytest.raises(EvaluationRejected, match="drift diagnosis"):
        _validate_report(sealed_report)


def test_forecast_record_and_service_persistence_replay_cover_run_authority(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    registered_report(store, artifacts)
    with store._connect() as db:
        row = db.execute(
            "SELECT s.* FROM phase6_forecast_service_artifacts s "
            "JOIN phase6_forecast_artifacts f ON f.forecast_checksum=s.artifact_checksum "
            "WHERE f.operation_id=?",
            (str(operation(2000)),),
        ).fetchone()
    document = json.loads(artifacts.read(ArtifactChecksum(row["artifact_checksum"])))
    distribution = ordered_finish_from_probabilities(
        tuple(document["distribution"]["runner_ids"]),
        {tuple(order): probability for order, probability in document["distribution"]["orders"]},
    )
    evidence = ForecastEvidence(
        row["race_id"],
        row["bundle_id"],
        ArtifactChecksum(row["bundle_checksum"]),
        ArtifactChecksum(row["evidence_checksum"]),
        ArtifactChecksum(row["forecast_checksum"]),
        distribution,
        row["generated_at"],
    )
    authority = EvaluationAuthority(store, artifacts)
    assert not authority.record_forecast(
        operation(2000), evidence, evaluation_run_id=operation(800)
    )
    with pytest.raises(ConflictingOperation):
        authority.record_forecast(operation(2000), evidence, evaluation_run_id=operation(799))

    service = CanonicalForecastService(
        ChampionLoader(store, artifacts, deserializer=LinearStrengthModel.from_bytes), artifacts
    )
    persistence = {
        "service_run_id": operation(792),
        "race_id": row["race_id"],
        "bundle_id": row["bundle_id"],
        "bundle_checksum": ArtifactChecksum(row["bundle_checksum"]),
        "evidence_checksum": ArtifactChecksum(row["evidence_checksum"]),
        "computed_at": datetime.fromisoformat(row["generated_at"]),
        "computation_id": "computation-champion-0",
    }
    assert service.persist_evaluation_forecast(operation(1000), **persistence) == ArtifactChecksum(
        row["forecast_checksum"]
    )
    persistence["service_run_id"] = operation(793)
    with pytest.raises(ConflictingOperation):
        service.persist_evaluation_forecast(operation(1000), **persistence)


@pytest.mark.parametrize("changed_field", ("requested_at", "service_run_id"))
def test_training_request_replay_covers_time_and_service_run(tmp_path, changed_field):
    store = SQLiteOperationsStore(tmp_path / f"request-{changed_field}.sqlite3")
    store.migrate()
    artifacts = LocalArtifactStore(tmp_path / f"artifacts-{changed_field}")
    service_run = operation(4710)
    requested_at = datetime(2026, 7, 23, tzinfo=timezone.utc)
    EvaluationAuthority(store, artifacts).begin_run(
        service_run, run_kind="forecast_service", started_at=requested_at
    )
    service = CanonicalForecastService(SimpleNamespace(store=store), artifacts)
    arguments = {
        "request_id": "training-request",
        "reason": "model drift",
        "requested_at": requested_at,
        "evidence_id": None,
        "service_run_id": service_run,
    }
    assert service.emit_training_request(operation(4711), **arguments)
    assert not service.emit_training_request(operation(4711), **arguments)
    changed = dict(arguments)
    changed[changed_field] = (
        requested_at + timedelta(seconds=1) if changed_field == "requested_at" else operation(4712)
    )
    with pytest.raises(ConflictingOperation):
        service.emit_training_request(operation(4711), **changed)


def test_corrupt_or_checksum_mismatched_registered_challenger_fails_closed(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    authority = PromotionAuthority(store, artifacts)
    approved = PHASE7_PROMOTION_APPROVED_AT
    phase7_approval_day = seed_external_probation(store, artifacts, "probation-14", 14)
    valid = registered_report(store, artifacts)
    EvaluationAuthority(store, artifacts).begin_run(
        operation(850), run_kind="promotion", started_at=approved
    )
    mismatched = copy.deepcopy(valid)
    mismatched["bundle_checksums"]["challenger"] = str(digest("9"))
    with pytest.raises(PromotionRejected, match="exact sealed evaluation"):
        authority.promote(
            operation(21),
            evidence_id="evidence",
            report=mismatched,
            challenger_bundle_id="challenger",
            assignment_id="a",
            promotion_record_id="p",
            approved_at=approved,
            effective_racing_day=PHASE7_PROMOTION_EFFECTIVE_DATE,
            approver="policy",
            reason="gates",
            probation_id="probation-14",
            promotion_run_id=operation(850),
            approval_racing_day_id=phase7_approval_day,
        )
    with store._connect() as db:
        db.execute("DROP TRIGGER canonical_bundle_components_append_only_delete")
        db.execute(
            "DELETE FROM canonical_bundle_components WHERE bundle_id='challenger' AND component_kind='evaluation'"
        )
    with pytest.raises(PromotionRejected, match="complete registered challenger"):
        authority.promote(
            operation(22),
            evidence_id="evidence",
            report=valid,
            challenger_bundle_id="challenger",
            assignment_id="a",
            promotion_record_id="p",
            approved_at=approved,
            effective_racing_day=PHASE7_PROMOTION_EFFECTIVE_DATE,
            approver="policy",
            reason="gates",
            probation_id="probation-14",
            promotion_run_id=operation(850),
            approval_racing_day_id=phase7_approval_day,
        )
    with store._connect() as db:
        assert tuple(
            db.execute("SELECT assignment_id,bundle_id FROM champion_pointer").fetchone()
        ) == ("assignment-old", "champion")


def test_evaluation_rejects_corrupt_registered_artifact_bytes(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    result = registered_report(store, artifacts)
    with store._connect() as db:
        checksum = ArtifactChecksum(
            db.execute(
                "SELECT artifact_checksum FROM canonical_bundle_components WHERE bundle_id='challenger' LIMIT 1"
            ).fetchone()[0]
        )
    artifacts.path_for(checksum).write_bytes(b"corrupt")
    with pytest.raises(ChecksumMismatch):
        registered_report(store, artifacts, force_evaluate=True)


def test_evaluation_rejects_unregistered_models(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "unregistered.sqlite3")
    store.migrate()
    races, forecasts = population(3)
    with pytest.raises(EvaluationRejected, match="complete immutable registration"):
        EvaluationAuthority(store, LocalArtifactStore(tmp_path / "artifacts")).evaluate(
            races,
            forecasts,
            evaluation_run_id=operation(800),
            champion_bundle_id="champion",
            challenger_bundle_ids=("challenger",),
            eligible_population=eligible(races),
        )


def test_evaluation_rejects_unauthenticated_forward_population(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    races, forecasts = population(3)
    with store._connect() as db:
        checksums = {
            row["bundle_id"]: ArtifactChecksum(row["bundle_checksum"])
            for row in db.execute("SELECT bundle_id,bundle_checksum FROM canonical_model_bundles")
        }
    forecasts = {
        model: [replace(item, bundle_checksum=checksums[model]) for item in items]
        for model, items in forecasts.items()
    }
    with pytest.raises(EvaluationRejected, match="complete durable forward population"):
        EvaluationAuthority(store, artifacts).evaluate(
            races,
            forecasts,
            evaluation_run_id=operation(800),
            champion_bundle_id="champion",
            challenger_bundle_ids=("challenger",),
            eligible_population=eligible(races),
        )


def test_forecast_service_can_only_emit_durable_training_request(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "service.sqlite3")
    store.migrate()
    service = CanonicalForecastService(
        SimpleNamespace(store=store), LocalArtifactStore(tmp_path / "artifacts")
    )
    run = operation(989)
    EvaluationAuthority(store, service.artifacts).begin_run(
        run, run_kind="forecast_service", started_at=datetime(2026, 7, 23, tzinfo=timezone.utc)
    )
    payload = {
        "operation_id": str(operation(990)),
        "service_run_id": str(run),
        "request_id": "training-request-1",
        "reason": "sustained model drift",
        "requested_at": "2026-07-23T00:00:00+00:00",
    }
    application = CanonicalForecastApplication(service)
    assert application.handle("POST", "/v1/training-requests", payload) == (
        {"success": True, "created": True},
        202,
    )
    assert application.handle("POST", "/v1/training-requests", payload) == (
        {"success": True, "created": False},
        202,
    )
    with store._connect() as db:
        assert tuple(
            db.execute("SELECT training_request_id,reason FROM phase6_training_requests").fetchone()
        ) == ("training-request-1", "sustained model drift")
        assert (
            db.execute("SELECT endpoint FROM phase6_service_training_requests").fetchone()[0]
            == "canonical_forecast_service"
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE phase6_service_training_requests SET endpoint='forged'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM phase6_service_training_requests")
    assert not any(
        hasattr(service, name) for name in ("train", "tune", "register_bundle", "promote")
    )
    assert application.handle("POST", "/v1/promotions", payload)[1] == 404


def test_sealed_report_validation_rejects_internal_metric_and_schema_forgery():
    valid = report()
    for mutation in (
        lambda value: value["long_horizon"]["champion"].update(mean_ordered_finish_nll=123.0),
        lambda value: value["long_horizon"]["champion"]["win_calibration"]["bins"][0].update(
            upper=0.2
        ),
        lambda value: value["bootstrap"]["challenger"].pop("lower_95"),
        lambda value: value["racing_day_views"].clear(),
        lambda value: value["wagering_scorecard"].update(real_betting=True),
    ):
        forged = copy.deepcopy(valid)
        mutation(forged)
        with pytest.raises(EvaluationRejected):
            _validate_report(forged)


def test_policy_artifact_and_real_bundle_lineage_are_fail_closed(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    result = registered_report(store, artifacts)
    with store._connect() as db:
        lineage = db.execute(
            "SELECT l.bundle_registration_operation_id,b.operation_id "
            "FROM phase6_bundle_lineage_v2 l JOIN canonical_model_bundles b USING(bundle_id) "
            "WHERE l.bundle_id='challenger'"
        ).fetchone()
        assert tuple(lineage) == (str(operation(701)), str(operation(701)))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute(
                "UPDATE phase6_bundle_lineage_v2 SET bundle_registration_operation_id=?",
                (str(operation(700)),),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM phase6_bundle_lineage_v2")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE phase6_service_computations SET computation_id='forged'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM phase6_service_computations")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM phase6_forecast_computation_bindings")
        policy_checksum = ArtifactChecksum(
            db.execute("SELECT artifact_checksum FROM phase6_policy_registry").fetchone()[0]
        )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM phase6_policy_registry")
        forward = db.execute(
            "SELECT race_id,evidence_checksum FROM phase6_forward_evaluation_races"
        ).fetchone()
        deferred = db.execute(
            "SELECT prediction_id,computed_at FROM deferred_predictions"
        ).fetchone()
        challenger_checksum = ArtifactChecksum(
            db.execute(
                "SELECT bundle_checksum FROM canonical_model_bundles WHERE bundle_id='challenger'"
            ).fetchone()[0]
        )
        forged_operation = str(operation(4104))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (forged_operation, "persist_evaluation_forecast", "d" * 64, deferred["computed_at"]),
        )
        with pytest.raises(sqlite3.IntegrityError, match="authority"):
            db.execute(
                "INSERT INTO phase6_service_computations VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    "timestamp-only-computation",
                    forward["race_id"],
                    "challenger",
                    str(challenger_checksum),
                    str(digest("f")),
                    deferred["computed_at"],
                    str(operation(792)),
                    deferred["prediction_id"],
                    forged_operation,
                ),
            )
        malformed = "op_" + "b" * 31 + "!"
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (malformed, "persist_evaluation_forecast", "b" * 64, deferred["computed_at"]),
        )
        existing = db.execute("SELECT * FROM phase6_forecast_service_artifacts LIMIT 1").fetchone()
        with pytest.raises(sqlite3.IntegrityError, match="checksum or operation identity"):
            db.execute(
                "INSERT INTO phase6_forecast_service_artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    "sha256:" + "a" * 63 + "!",
                    existing["race_id"],
                    existing["bundle_id"],
                    existing["bundle_checksum"],
                    existing["evidence_checksum"],
                    "sha256:" + "b" * 64,
                    existing["generated_at"],
                    existing["service_run_id"],
                    existing["deferred_prediction_id"],
                    malformed,
                ),
            )
    service = CanonicalForecastService(
        ChampionLoader(store, artifacts, deserializer=LinearStrengthModel.from_bytes), artifacts
    )
    with pytest.raises(ForecastUnavailable, match="authority"):
        service.persist_evaluation_forecast(
            operation(4100),
            service_run_id=operation(792),
            race_id="wrong-race",
            bundle_id="challenger",
            bundle_checksum=challenger_checksum,
            evidence_checksum=ArtifactChecksum(forward["evidence_checksum"]),
            computed_at=datetime.fromisoformat(deferred["computed_at"]),
            computation_id="wrong-race-computation",
        )
    with pytest.raises(BundleUnavailable):
        service.persist_evaluation_forecast(
            operation(4101),
            service_run_id=operation(792),
            race_id=forward["race_id"],
            bundle_id="challenger",
            bundle_checksum=digest("f"),
            evidence_checksum=ArtifactChecksum(forward["evidence_checksum"]),
            computed_at=datetime.fromisoformat(deferred["computed_at"]),
            computation_id="wrong-bundle-computation",
        )
    with pytest.raises(sqlite3.IntegrityError):
        service.persist_evaluation_forecast(
            operation(4102),
            service_run_id=operation(792),
            race_id=forward["race_id"],
            bundle_id="challenger",
            bundle_checksum=challenger_checksum,
            evidence_checksum=ArtifactChecksum(forward["evidence_checksum"]),
            computed_at=datetime.fromisoformat(deferred["computed_at"]),
            computation_id="computation-champion",
        )
    with pytest.raises(TypeError, match="order_probabilities"):
        service.persist_evaluation_forecast(
            operation(4103),
            service_run_id=operation(792),
            race_id=forward["race_id"],
            bundle_id="challenger",
            bundle_checksum=challenger_checksum,
            evidence_checksum=ArtifactChecksum(forward["evidence_checksum"]),
            computed_at=datetime.fromisoformat(deferred["computed_at"]),
            computation_id="fabricated-computation",
            order_probabilities={("dog-a", "dog-b"): 1.0},
        )
    artifacts.path_for(policy_checksum).write_bytes(b"corrupt-policy")
    with pytest.raises(ChecksumMismatch):
        registered_report(store, artifacts, force_evaluate=True)
    artifacts.path_for(policy_checksum).unlink()
    EvaluationAuthority(store, artifacts).begin_run(
        operation(4300),
        run_kind="promotion",
        started_at=datetime(2026, 7, 23, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(ArtifactStoreError):
        PromotionAuthority(store, artifacts).promote(
            operation(4301),
            evidence_id="evidence",
            report=result,
            challenger_bundle_id="challenger",
            assignment_id="missing-policy-assignment",
            promotion_record_id="missing-policy-promotion",
            approved_at=datetime(2026, 7, 23, 1, tzinfo=timezone.utc),
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="missing policy",
            probation_id="missing",
            promotion_run_id=operation(4300),
            approval_racing_day_id=racing_day_id(23),
        )


def test_naive_approval_and_caller_authored_service_distribution_are_rejected(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "boundaries.sqlite3")
    store.migrate()
    authority = PromotionAuthority(store, LocalArtifactStore(tmp_path / "artifacts"))
    with pytest.raises(ValueError, match="approved_at"):
        authority.promote(
            operation(4000),
            evidence_id="evidence",
            report=report(),
            challenger_bundle_id="challenger",
            assignment_id="assignment",
            promotion_record_id="promotion",
            approved_at=datetime(2026, 7, 23),
            effective_racing_day="2026-07-24",
            approver="policy",
            reason="invalid time",
            probation_id="probation",
            promotion_run_id=operation(4001),
            approval_racing_day_id=racing_day_id(23),
        )
    assert (
        "order_probabilities"
        not in inspect.signature(CanonicalForecastService.persist_evaluation_forecast).parameters
    )
    with store._connect() as db:
        policy_operation = str(operation(4002))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (policy_operation, "register_phase6_policy", "a" * 64, "2026-07-23T00:00:00+00:00"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="policy checksum"):
            db.execute(
                "INSERT INTO phase6_policy_registry VALUES(?,?,?,?,?)",
                (
                    "forged-policy",
                    "sha256:" + "a" * 64,
                    "sha256:" + "b" * 64,
                    "2026-07-23T00:00:00+00:00",
                    policy_operation,
                ),
            )
        malformed = "op_" + "a" * 31 + "z"
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (malformed, "begin_phase6_run", "c" * 64, "2026-07-23T00:00:00+00:00"),
        )
        with pytest.raises(sqlite3.IntegrityError, match="operation identity"):
            db.execute(
                "INSERT INTO phase6_runs VALUES(?,?,?,?)",
                (malformed, "evaluation", "2026-07-23T00:00:00+00:00", malformed),
            )


def test_probation_day_direct_sql_rejects_arbitrary_checksum_prefix(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    seed_external_probation(store, artifacts, "probation-prefix", 14)
    with store._connect() as db:
        authentic = list(
            db.execute(
                "SELECT * FROM phase6_probation_days "
                "WHERE probation_id='probation-prefix' ORDER BY racing_day LIMIT 1"
            ).fetchone()
        )
        for checksum_index, _name in enumerate(
            ("reconciliation", "restart", "ordering", "determinism"),
            2,
        ):
            values = list(authentic)
            values[checksum_index] = "forged:" + "a" * 64
            with pytest.raises(sqlite3.IntegrityError):
                db.execute(
                    "INSERT INTO phase6_probation_days VALUES(?,?,?,?,?,?,?)",
                    values,
                )


def test_trusted_evaluation_direct_sql_rejects_arbitrary_policy_checksum_prefix(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "trusted-policy-prefix.sqlite3")
    store.migrate()
    with store._connect() as db:
        with pytest.raises(sqlite3.IntegrityError, match="trusted evaluation identity"):
            db.execute(
                "INSERT INTO phase6_trusted_evaluations VALUES(?,?,?,?,?,?,?,?)",
                (
                    "missing-evidence",
                    str(operation(4190)),
                    "missing-assignment",
                    "2026-07-01T00:00:00+00:00",
                    "forged:" + "a" * 64,
                    "sha256:" + "b" * 64,
                    "2026-07-23T00:00:00+00:00",
                    str(operation(4191)),
                ),
            )


def test_direct_sql_bundle_lineage_forgery_and_reversed_time_are_rejected(tmp_path):
    store, artifacts = promotion_store(tmp_path)
    authority = EvaluationAuthority(store, artifacts)
    source = operation(4200)
    registration = operation(4201)
    authority.begin_run(
        source, run_kind="training", started_at=datetime(2025, 10, 1, tzinfo=timezone.utc)
    )
    authority.begin_run(
        registration,
        run_kind="registration",
        started_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
    )
    with store._connect() as db:
        forged_operation = str(operation(4202))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                forged_operation,
                "record_phase6_bundle_lineage",
                "e" * 64,
                "2025-11-01T00:00:00+00:00",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="real registration"):
            db.execute(
                "INSERT INTO phase6_bundle_lineage_v2 VALUES(?,?,?,?,?)",
                (
                    "challenger",
                    str(operation(700)),
                    str(registration),
                    str(source),
                    forged_operation,
                ),
            )
    late_source = operation(4203)
    early_registration = operation(4204)
    authority.begin_run(
        late_source, run_kind="training", started_at=datetime(2025, 11, 2, tzinfo=timezone.utc)
    )
    authority.begin_run(
        early_registration,
        run_kind="registration",
        started_at=datetime(2025, 11, 1, tzinfo=timezone.utc),
    )
    with store._connect() as db:
        reversed_operation = str(operation(4205))
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            (
                reversed_operation,
                "record_phase6_bundle_lineage",
                "f" * 64,
                "2025-11-02T00:00:00+00:00",
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="real registration"):
            db.execute(
                "INSERT INTO phase6_bundle_lineage_v2 VALUES(?,?,?,?,?)",
                (
                    "challenger",
                    str(operation(701)),
                    str(early_registration),
                    str(late_source),
                    reversed_operation,
                ),
            )
