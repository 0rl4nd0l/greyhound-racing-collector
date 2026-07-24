import json
import itertools
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from race_collection.artifacts import ChecksumMismatch, LocalArtifactStore
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
from race_collection.forecasting import ForecastingAuthority, LegacyBundle, ModelRelease
from race_collection.forecast_service import CanonicalForecastService, ForecastRequest
from race_collection.model_bundle import (
    BundleUnavailable,
    ChampionLoader,
    ModelBundleAuthority,
    ServingAssignment,
    validate_training_corpus_manifest,
)
from race_collection.operations import ConflictingOperation, SQLiteOperationsStore
from race_collection.ordered_finish import (
    OrderedFinishError,
    forecast_ordered_finish,
    ordered_finish_nll,
)
from race_collection.training import (
    LegacyRow,
    LinearStrengthModel,
    PreparedExample,
    CorpusRejected,
    TrainingExample,
    TrainingCorpusAuthority,
    audit_legacy_corpus,
    canonical_json,
    checksum,
    expanding_window_validation,
    fit_linear_strength_model,
    train_challenger_bundle,
    validate_evaluation_outcome,
    wagering_strategy_report,
)
from race_collection.features import (
    DerivationReport,
    DerivationResult,
    FeatureContract,
    FeatureMatrix,
)


NOW = datetime(2026, 7, 23, tzinfo=timezone.utc)


def digest(character):
    return ArtifactChecksum("sha256:" + character * 64)


def test_distribution_normalization_marginal_and_exact_order_identities():
    forecast = forecast_ordered_finish(("dog-c", "dog-a", "dog-b"), (1000, 1000, -1000))
    assert math.isclose(sum(forecast.order_probabilities.values()), 1.0)
    assert math.isclose(sum(forecast.win.values()), 1.0)
    assert math.isclose(sum(forecast.top_2.values()), 2.0)
    assert math.isclose(sum(forecast.top_3.values()), 3.0)
    assert math.isclose(sum(forecast.exacta.values()), 1.0)
    assert math.isclose(sum(forecast.trifecta.values()), 1.0)
    for runner in forecast.runner_ids:
        assert math.isclose(
            forecast.win[runner],
            sum(
                value for order, value in forecast.order_probabilities.items() if order[0] == runner
            ),
        )
        assert math.isclose(
            forecast.top_2[runner],
            sum(
                value
                for order, value in forecast.order_probabilities.items()
                if runner in order[:2]
            ),
        )
    assert forecast.ranking[:2] == ("dog-a", "dog-c")  # stable identity tie-break
    assert ordered_finish_nll(forecast, ("dog-a", "dog-c", "dog-b")) >= 0


def test_distribution_rejects_nonfinite_duplicate_and_oversized_fields():
    with pytest.raises(OrderedFinishError):
        forecast_ordered_finish(("a", "a"), (0, 1))
    with pytest.raises(OrderedFinishError):
        forecast_ordered_finish(("a",), (float("nan"),))
    with pytest.raises(OrderedFinishError):
        forecast_ordered_finish(tuple(map(str, range(9))), (0,) * 9)
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(OrderedFinishError, match="order_limit"):
            forecast_ordered_finish(("a",), (0,), order_limit=invalid)


def test_extreme_strengths_are_finite_normalized_for_every_runner_permutation():
    strengths_by_runner = {"dominant": 1000.0, "tiny-a": -1000.0, "tiny-b": -999.0, "mid": 0.0}
    for runners in itertools.permutations(strengths_by_runner):
        forecast = forecast_ordered_finish(
            runners, tuple(strengths_by_runner[runner] for runner in runners)
        )
        values = tuple(forecast.order_probabilities.values())
        assert all(math.isfinite(value) and value >= 0 for value in values)
        assert math.isclose(math.fsum(values), 1.0, rel_tol=1e-12, abs_tol=1e-12)
        assert all(math.isfinite(value) for value in forecast.top_2.values())


def test_one_and_two_runner_top_n_and_exotic_semantics():
    singleton = forecast_ordered_finish(("only",), (0,))
    assert singleton.win == singleton.top_2 == singleton.top_3 == {"only": 1.0}
    assert singleton.exacta == {}
    assert singleton.trifecta == {}
    pair = forecast_ordered_finish(("a", "b"), (1, 0))
    assert pair.top_2 == pair.top_3 == {"a": 1.0, "b": 1.0}
    assert math.isclose(math.fsum(pair.exacta.values()), 1.0)
    assert pair.trifecta == {}


def test_supported_eight_runner_maximum_is_finite_and_normalized():
    forecast = forecast_ordered_finish(tuple(f"dog-{index}" for index in range(8)), range(8))
    assert len(forecast.order_probabilities) == math.factorial(8)
    assert math.isclose(math.fsum(forecast.order_probabilities.values()), 1.0)


def test_legacy_audit_deduplicates_and_authoritative_supersedes_provisional():
    rows = [
        LegacyRow(
            "form", "p1", "dog-1", "2026-07-01", NOW.isoformat(), False, digest("a"), {"starts": 2}
        ),
        LegacyRow(
            "official",
            "a1",
            "dog-1",
            "2026-07-01",
            NOW.isoformat(),
            True,
            digest("b"),
            {"starts": 3},
        ),
        LegacyRow(
            "official",
            "a1",
            "dog-1",
            "2026-07-01",
            NOW.isoformat(),
            True,
            digest("b"),
            {"starts": 3},
        ),
    ]
    first = audit_legacy_corpus(rows)
    second = audit_legacy_corpus(reversed(rows))
    assert first == second
    assert first["promotion_grade"] is False
    assert first["counts"] == {
        "input_rows": 3,
        "unique_observations": 2,
        "deduplicated_dog_runs": 1,
        "quarantined_conflicts": 0,
    }
    run = first["dog_runs"][0]
    assert run["authoritative"] is True
    assert run["selected_observation"]["source"] == "official"
    assert run["superseded_observations"][0]["source"] == "form"


def test_legacy_audit_quarantines_equal_authority_source_conflict():
    rows = [
        LegacyRow("one", "1", "dog", "2026-07-01", NOW.isoformat(), True, digest("a"), {"box": 1}),
        LegacyRow("two", "2", "dog", "2026-07-01", NOW.isoformat(), True, digest("b"), {"box": 2}),
    ]
    audit = audit_legacy_corpus(rows)
    assert audit["dog_runs"] == []
    assert audit["conflicts"][0]["reason"] == "source_conflict"


@pytest.mark.parametrize(
    "exclusion",
    [
        "post_seal_scratch",
        "dead_heat",
        "abandoned",
        "order_changing_disqualification",
        "incomplete_result_provenance",
    ],
)
def test_evaluation_eligibility_excludes_ambiguous_outcomes(exclusion):
    outcome = {
        "official": True,
        "order": [1, 2],
        "provenance": {
            "source": "official",
            "source_record_id": "result-1",
            "observed_at": (NOW + timedelta(minutes=1)).isoformat(),
        },
        "published_at": (NOW + timedelta(minutes=1)).isoformat(),
        "exclusions": [exclusion],
    }
    with pytest.raises(CorpusRejected, match="not evaluation eligible"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


def test_evaluation_eligibility_rejects_temporal_leakage_and_accepts_exact_order():
    outcome = {
        "official": True,
        "order": [1, 2],
        "provenance": {
            "source": "official",
            "source_record_id": "result-1",
            "observed_at": (NOW + timedelta(minutes=1)).isoformat(),
        },
        "published_at": (NOW + timedelta(minutes=1)).isoformat(),
        "exclusions": [],
    }
    assert validate_evaluation_outcome(
        outcome,
        evidence_frozen_at=NOW,
        joined_at=NOW + timedelta(minutes=2),
        runner_ids=("a", "b"),
        box_by_runner={"a": 1, "b": 2},
    ) == ("a", "b")
    outcome["published_at"] = NOW.isoformat()
    with pytest.raises(CorpusRejected, match="temporal"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )
    outcome["published_at"] = (NOW + timedelta(minutes=1)).isoformat()
    outcome["provenance"]["observed_at"] = (NOW + timedelta(minutes=3)).isoformat()
    with pytest.raises(CorpusRejected, match="temporal"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


def test_evaluation_eligibility_rejects_incomplete_result_provenance():
    outcome = {
        "official": True,
        "order": [1, 2],
        "provenance": {"source": "official"},
        "published_at": (NOW + timedelta(minutes=1)).isoformat(),
        "exclusions": [],
    }
    with pytest.raises(CorpusRejected, match="provenance is incomplete"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


@pytest.mark.parametrize("exclusions", ["dead_heat", {"dead_heat"}, [1], ["future-unknown"]])
def test_evaluation_eligibility_fails_closed_on_exclusion_envelopes(exclusions):
    outcome = {
        "official": True,
        "order": [1, 2],
        "provenance": {
            "source": "official",
            "source_record_id": "result-1",
            "observed_at": (NOW + timedelta(minutes=1)).isoformat(),
        },
        "published_at": (NOW + timedelta(minutes=1)).isoformat(),
        "exclusions": exclusions,
    }
    with pytest.raises(CorpusRejected):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


def test_evaluation_eligibility_rejects_missing_required_provenance_even_with_extra_key():
    outcome = {
        "official": True,
        "order": [1, 2],
        "provenance": {
            "source": "official",
            "source_record_id": "result-1",
            "unrelated": "present",
        },
        "published_at": (NOW + timedelta(minutes=1)).isoformat(),
        "exclusions": [],
    }
    with pytest.raises(CorpusRejected, match="provenance is incomplete"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


@pytest.mark.parametrize("outcome", [None, [], "result"])
def test_evaluation_eligibility_rejects_nonobject_outcome_envelope(outcome):
    with pytest.raises(CorpusRejected, match="envelope must be an object"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("published_at", 7), ("published_at", "bad"), ("observed_at", "2026-07-23T00:01:00")],
)
def test_evaluation_eligibility_translates_malformed_timestamps(field, value):
    outcome = {
        "official": True,
        "order": [1, 2],
        "provenance": {
            "source": "official",
            "source_record_id": "result-1",
            "observed_at": (NOW + timedelta(minutes=1)).isoformat(),
        },
        "published_at": (NOW + timedelta(minutes=1)).isoformat(),
        "exclusions": [],
    }
    if field == "observed_at":
        outcome["provenance"][field] = value
    else:
        outcome[field] = value
    with pytest.raises(CorpusRejected, match="timestamp"):
        validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=NOW,
            joined_at=NOW + timedelta(minutes=2),
            runner_ids=("a", "b"),
            box_by_runner={"a": 1, "b": 2},
        )


def test_wagering_is_separate_report_only():
    report = wagering_strategy_report({"a": 0.6, "b": 0.4}, {"a": 2.0, "b": 2.0}, threshold=0.1)
    assert report["report_only"] is True
    assert report["places_bets"] is False
    assert report["forecast_quality_metric"] is False
    assert [ticket["runner_id"] for ticket in report["tickets"]] == ["a"]


def synthetic_prepared_example(example_id, racing_date, feature_values, finish_indices):
    runner_ids = tuple(f"dog-{index}" for index in range(len(feature_values)))
    matrix = FeatureMatrix(
        runner_ids,
        ("speed",),
        tuple((float(value),) for value in feature_values),
        digest("c"),
    )
    derived = DerivationResult(
        matrix,
        FeatureContract("sealed-race-features-v1", digest("d"), digest("e"), ("speed",)),
        DerivationReport("sealed-race-features-v1", digest("f"), digest("c"), {}, {}),
    )
    identity = TrainingExample(
        example_id,
        f"race-{example_id}",
        racing_date,
        digest("a"),
        digest("b"),
        runner_ids,
        tuple(runner_ids[index] for index in finish_indices),
        digest("c"),
        digest("9"),
    )
    return PreparedExample(identity, derived)


def test_training_and_expanding_window_are_deterministic_and_temporal():
    examples = [
        synthetic_prepared_example("1", "2026-07-01", (3, 2, 1), (0, 1, 2)),
        synthetic_prepared_example("2", "2026-07-02", (2, 1, 0), (0, 1, 2)),
        synthetic_prepared_example("3", "2026-07-03", (4, 2, 1), (0, 1, 2)),
    ]
    one = fit_linear_strength_model(examples, seed=7341)
    two = fit_linear_strength_model(examples, seed=7341)
    assert one.to_bytes() == two.to_bytes()
    assert LinearStrengthModel.from_bytes(one.to_bytes()) == one
    validation = expanding_window_validation(examples, seed=7341)
    assert validation["folds"][0]["trained_through"] == "2026-07-02"
    assert validation["folds"][0]["evaluation_example_id"] == "3"


@pytest.mark.parametrize(
    "rows",
    [[(1.0,)], [(1.0, 2.0, 3.0)], [(1.0, float("nan"))], [(1.0, "bad")]],
)
def test_linear_strength_model_rejects_dimension_and_value_errors(rows):
    model = LinearStrengthModel((1.0, 2.0))
    with pytest.raises(CorpusRejected):
        model.latent_strengths(rows)


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        b"not-json",
        b"[]",
        canonical_json({"schema_version": "wrong", "coefficients": [1.0], "intercept": 0.0}),
        canonical_json(
            {"schema_version": "linear-runner-strength-v1", "coefficients": [], "intercept": 0.0}
        ),
        canonical_json(
            {
                "schema_version": "linear-runner-strength-v1",
                "coefficients": [1.0],
                "intercept": 0.0,
                "extra": 1,
            }
        ),
        canonical_json(
            {
                "schema_version": "linear-runner-strength-v1",
                "coefficients": [True],
                "intercept": 0.0,
            }
        ),
        b'{"schema_version":"linear-runner-strength-v1","coefficients":[1e999],"intercept":0}',
    ],
)
def test_linear_strength_model_requires_exact_finite_envelope(payload):
    with pytest.raises(CorpusRejected):
        LinearStrengthModel.from_bytes(payload)


def test_linear_strength_model_rejects_nonfinite_constructor_parameters():
    with pytest.raises(CorpusRejected):
        LinearStrengthModel((float("inf"),))
    with pytest.raises(CorpusRejected):
        LinearStrengthModel((1.0,), float("nan"))


def test_expanding_window_never_trains_on_same_racing_day():
    examples = [
        synthetic_prepared_example("1", "2026-07-01", (3, 1), (0, 1)),
        synthetic_prepared_example("2", "2026-07-01", (2, 1), (0, 1)),
        synthetic_prepared_example("3", "2026-07-02", (3, 2), (0, 1)),
        synthetic_prepared_example("4", "2026-07-02", (4, 1), (0, 1)),
    ]
    folds = expanding_window_validation(examples, seed=7341)["folds"]
    assert {fold["evaluation_example_id"] for fold in folds} == {"3", "4"}
    assert all(fold["trained_through"] < fold["evaluation_racing_date"] for fold in folds)


def test_bundle_reproducibility_registration_loading_and_corruption(tmp_path):
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    schema = canonical_json(
        {
            "bundle_id": "challenger-1",
            "contract_version": "sealed-race-features-v1",
            "evidence_schema_version": "sealed-race-evidence-v1",
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
    kwargs = dict(
        model_id="model-1",
        bundle_id="challenger-1",
        examples=(
            synthetic_prepared_example("example-1", "2026-07-02", (2, 1), (0, 1)),
            synthetic_prepared_example("example-2", "2026-07-03", (3, 1), (0, 1)),
        ),
        schema_bytes=schema,
        missingness_bytes=missingness,
        validation={"method": "expanding-window-temporal-v1", "folds": []},
        seed=7341,
        artifacts=artifacts,
    )
    first = train_challenger_bundle(**kwargs)
    second = train_challenger_bundle(**{**kwargs, "examples": tuple(reversed(kwargs["examples"]))})
    assert first.bundle_checksum == second.bundle_checksum
    assert first.component("model").checksum == second.component("model").checksum
    assert first.manifest() == json.loads(artifacts.read(first.bundle_checksum))
    with pytest.raises(BundleUnavailable):
        validate_training_corpus_manifest(
            {
                "training_example_ids": ["example-1"],
                "training_examples": ["not-an-object"],
                "corpus_id": "sha256:" + "0" * 64,
            },
            first,
        )

    store = SQLiteOperationsStore(tmp_path / "operations.db")
    store.migrate()
    authority = ModelBundleAuthority(store)
    assert authority.register(OperationId("op_" + "1" * 32), first, NOW)
    store.migrate()  # repeat migration remains safe with populated Phase-5 bundle state
    with store._connect() as db:
        assert db.execute("SELECT COUNT(*) FROM canonical_model_bundles").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM canonical_serving_assignments").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM champion_pointer").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM canonical_day_assignments").fetchone()[0] == 0
    assignment = ServingAssignment(
        "assignment-1",
        first.bundle_id,
        first.bundle_checksum,
        NOW.isoformat(),
        "2026-07-23",
        "phase5-loadability-proof",
    )
    authority.register_assignment(OperationId("op_" + "2" * 32), assignment, NOW)
    authority.bootstrap_champion(OperationId("op_" + "3" * 32), assignment, NOW)

    class OrderedOnlyModel:
        def __init__(self, content):
            self.model = LinearStrengthModel.from_bytes(content)

        def latent_strengths(self, rows):
            return self.model.latent_strengths(rows)

        def predict_proba(self, rows):
            raise AssertionError("ordered contract must not call predict_proba")

    loader = ChampionLoader(store, artifacts, deserializer=OrderedOnlyModel)
    loaded = loader.load().bundle
    assert (loaded.bundle_id, loaded.bundle_checksum) == (first.bundle_id, first.bundle_checksum)

    class PredictOnlyModel:
        def predict_proba(self, rows):
            return [[0.5, 0.5] for _ in rows]

    with pytest.raises(BundleUnavailable, match="latent_strengths"):
        ChampionLoader(store, artifacts, deserializer=lambda _: PredictOnlyModel()).load()
    evidence = artifacts.put(
        canonical_json(
            {
                "schema_version": "sealed-race-evidence-v1",
                "normalization_version": "normalizer-v1",
                "race_id": "load-proof-race",
                "fields": {
                    "runner_set": ["dog-a", "dog-b"],
                    "runner_identity": {"dog-a": "authoritative", "dog-b": "authoritative"},
                    "runner_features": {"dog-a": {"speed": 3}, "dog-b": {"speed": 1}},
                },
                "freeze": {
                    "at": NOW.isoformat(),
                    "authority": "actual_jump",
                    "odds_checksum": str(digest("c")),
                },
            }
        ),
        media_type="application/json",
    )
    result = CanonicalForecastService(loader, artifacts, clock=lambda: NOW).forecast(
        ForecastRequest(evidence.checksum)
    )
    assert "exacta_probabilities" in result
    artifacts.path_for(first.component("model").checksum).write_bytes(b"corrupt")
    with pytest.raises(ChecksumMismatch):
        ChampionLoader(store, artifacts, deserializer=OrderedOnlyModel).load()


def test_challenger_corpus_rejects_empty_and_duplicate_identities(tmp_path):
    common = dict(
        model_id="model-1",
        bundle_id="challenger-1",
        schema_bytes=b"{}",
        missingness_bytes=b"{}",
        validation={"method": "fixture"},
        seed=1,
        artifacts=LocalArtifactStore(tmp_path / "artifacts"),
    )
    with pytest.raises(CorpusRejected, match="requires immutable"):
        train_challenger_bundle(examples=(), **common)
    example = synthetic_prepared_example("duplicate", "2026-07-01", (2, 1), (0, 1))
    with pytest.raises(CorpusRejected, match="duplicated"):
        train_challenger_bundle(examples=(example, example), **common)


def test_migration_is_repeatable_and_storage_rejects_sql_forgery(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.db")
    store.migrate()
    store.migrate()
    with store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            ("op_" + "2" * 32, "forgery", "0" * 64, NOW.isoformat()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="relations disagree"):
            db.execute(
                "INSERT INTO canonical_training_examples VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    "fake",
                    "fake-phase3",
                    "fake-race",
                    str(digest("a")),
                    str(digest("b")),
                    str(digest("c")),
                    str(digest("d")),
                    "2026-07-01",
                    NOW.isoformat(),
                    "op_" + "2" * 32,
                ),
            )


class _DurablePredictor:
    def predict(self, request):
        return digest("d")


def durable_forward_setup(tmp_path):
    store = SQLiteOperationsStore(tmp_path / "operations.db")
    store.migrate()
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    day_id = RacingDayId("day_" + "1" * 32)
    day = RacingDay(day_id, date(2026, 7, 23), "Australia/Melbourne", NOW)
    store.create_racing_day(OperationId("op_" + "1" * 32), day)
    race = store.record_expected_race(
        OperationId("op_" + "2" * 32),
        day,
        ProgrammeRaceCandidate("official", "race-source-1", "Ballarat", 1, NOW),
        digest("a"),
        NOW,
    )
    sealed_odds = artifacts.put(
        canonical_json(
            {
                "schema_version": "sealed-win-place-odds-v1",
                "race_id": str(race),
                "captured_at": (NOW - timedelta(seconds=1)).isoformat(),
                "win_place": {"dog-a": [2.0, 1.5], "dog-b": [3.0, 1.8]},
            }
        ),
        media_type="application/json",
    )
    evidence_document = {
        "schema_version": "race-evidence-v1",
        "normalization_version": "normalizer-v1",
        "race_id": str(race),
        "fields": {
            "runner_set": ["dog-a", "dog-b"],
            "runner_identity": {"dog-a": "authoritative", "dog-b": "authoritative"},
            "box": {"dog-a": 1, "dog-b": 2},
            "runner_features": {"dog-a": {"speed": 8}, "dog-b": {"speed": 2}},
        },
        "field_provenance": [
            {
                "field": "box",
                "authority": "official_card",
                "critical": True,
                "value": {"dog-a": 1, "dog-b": 2},
                "source": "official",
                "artifact_checksum": str(digest("a")),
            }
        ],
        "freeze": {
            "at": NOW.isoformat(),
            "authority": "actual_jump",
            "odds_checksum": str(sealed_odds.checksum),
        },
    }
    evidence = artifacts.put(canonical_json(evidence_document), media_type="application/json")
    store.advance_race(OperationId("op_" + "3" * 32), race, RaceState.CARD_COLLECTED, NOW)
    store.advance_race(OperationId("op_" + "4" * 32), race, RaceState.COLLECTING_ODDS, NOW)
    store.record_field_evidence(
        FieldEvidence(
            OperationId("op_" + "4" * 31 + "5"),
            race,
            EvidenceField.VENUE,
            EvidenceAuthority.OFFICIAL_CARD,
            "Ballarat",
            "official",
            digest("a"),
            NOW,
        )
    )
    for suffix, field, value in (
        ("6", EvidenceField.DISTANCE, 450),
        ("7", EvidenceField.GRADE, "G5"),
        ("8", EvidenceField.FIELD_SIZE, 2),
    ):
        store.record_field_evidence(
            FieldEvidence(
                OperationId("op_" + "4" * 31 + suffix),
                race,
                field,
                EvidenceAuthority.OFFICIAL_CARD,
                value,
                "official",
                digest("a"),
                NOW,
            )
        )
    store.seal_evidence(
        OperationId("op_" + "5" * 32),
        race_id=race,
        raw_checksum=digest("a"),
        normalized_checksum=evidence.checksum,
        schema_version="race-evidence-v1",
        normalization_version="normalizer-v1",
        frozen_at=NOW,
        freeze_authority=FreezeAuthority.ACTUAL_JUMP,
        odds_checksum=sealed_odds.checksum,
        sealed_at=NOW,
        request_intent_digest=digest("f"),
    )
    store.advance_race(OperationId("op_" + "6" * 32), race, RaceState.AWAITING_DAY_CLOSE, NOW)
    phase3 = ForecastingAuthority(store)
    bundle = LegacyBundle(
        "legacy-bundle",
        "legacy-model",
        digest("6"),
        10,
        digest("2"),
        None,
        "raw_registry_model",
        {"fixture": True},
    )
    release = ModelRelease("legacy-release", bundle.bundle_id, "policy-v1", {"fixture": True})
    phase3.register_bundle(OperationId("op_" + "7" * 32), bundle, NOW)
    phase3.register_release(OperationId("op_" + "8" * 32), release, NOW)
    phase3.pin_day(OperationId("op_" + "9" * 32), day_id, release, NOW)
    store.close_racing_day(OperationId("op_" + "a" * 32), day, NOW)
    phase3.begin_prediction(OperationId("op_" + "b" * 32), race, NOW)
    phase3.predict(OperationId("op_" + "c" * 32), race, "prediction-1", _DurablePredictor(), NOW)
    phase3.open_results(OperationId("op_" + "d" * 32), race, NOW)
    result_time = NOW + timedelta(minutes=1)
    outcome = {
        "order": [1, 2],
        "official": True,
        "published_at": result_time.isoformat(),
        "provenance": {
            "source": "official",
            "source_record_id": "result-source-1",
            "observed_at": result_time.isoformat(),
        },
        "exclusions": [],
    }
    result = artifacts.put(canonical_json(outcome), media_type="application/json")
    phase3.record_result_attempt(
        OperationId("op_" + "e" * 32),
        race,
        "result-attempt-1",
        at=result_time,
        max_attempts=1,
        deadline=result_time,
        artifact_checksum=result.checksum,
        outcome=outcome,
    )
    phase3.join_training_example(
        OperationId("op_" + "f" * 32),
        race,
        "phase3-example-1",
        digest("e"),
        eligible=True,
        reason=None,
        at=result_time,
    )
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
    return store, artifacts, race, evidence, result, schema, missingness


def build_durable_forward(setup, operation_id=None, example_id="canonical-example-1"):
    store, artifacts, _, _, _, schema, missingness = setup
    return TrainingCorpusAuthority(store, artifacts).build_forward_example(
        operation_id or OperationId("op_" + "0" * 32),
        phase3_example_id="phase3-example-1",
        example_id=example_id,
        schema_bytes=schema,
        schema_checksum=checksum(schema),
        missingness_bytes=missingness,
        missingness_checksum=checksum(missingness),
        joined_at=NOW + timedelta(minutes=2),
    )


def test_real_forward_authority_commits_maps_boxes_and_replays(tmp_path):
    setup = durable_forward_setup(tmp_path)
    first = build_durable_forward(setup)
    replay = build_durable_forward(setup)
    assert replay == first
    assert first.official_order == ("dog-a", "dog-b")
    assert "result_provenance" not in json.loads(setup[1].read(first.artifact_checksum))
    with pytest.raises(ConflictingOperation):
        build_durable_forward(setup, example_id="conflicting-example")


@pytest.mark.parametrize(
    "boxes", [{"dog-a": 1}, {"dog-a": 1, "dog-b": 1}, {"dog-a": 1, "dog-b": 2, "dog-c": 3}]
)
def test_real_forward_authority_rejects_ambiguous_box_mapping(tmp_path, boxes):
    setup = durable_forward_setup(tmp_path)
    _, artifacts, _, evidence, *_ = setup
    document = json.loads(artifacts.read(evidence.checksum))
    document["fields"]["box"] = boxes
    bad = artifacts.put(canonical_json(document), media_type="application/json")
    with setup[0]._connect() as db:
        db.execute("DROP TRIGGER sealed_evidence_append_only_update")
        db.execute(
            "UPDATE sealed_evidence SET normalized_checksum=? WHERE normalized_checksum=?",
            (str(bad.checksum), str(evidence.checksum)),
        )
        db.execute("DROP TRIGGER deferred_predictions_append_only_update")
        db.execute("UPDATE deferred_predictions SET evidence_checksum=?", (str(bad.checksum),))
    with pytest.raises(CorpusRejected, match="box mapping"):
        build_durable_forward(setup)


def test_real_forward_authority_requires_authoritative_box_provenance(tmp_path):
    setup = durable_forward_setup(tmp_path)
    _, artifacts, _, evidence, *_ = setup
    document = json.loads(artifacts.read(evidence.checksum))
    document["field_provenance"][0]["authority"] = "source_card"
    bad = artifacts.put(canonical_json(document), media_type="application/json")
    with setup[0]._connect() as db:
        db.execute("DROP TRIGGER sealed_evidence_append_only_update")
        db.execute(
            "UPDATE sealed_evidence SET normalized_checksum=? WHERE normalized_checksum=?",
            (str(bad.checksum), str(evidence.checksum)),
        )
        db.execute("DROP TRIGGER deferred_predictions_append_only_update")
        db.execute("UPDATE deferred_predictions SET evidence_checksum=?", (str(bad.checksum),))
    with pytest.raises(CorpusRejected, match="authoritative provenance"):
        build_durable_forward(setup)


def test_real_forward_authority_binds_prediction_seal_and_result_artifact(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, artifacts, race, evidence, result, *_ = setup
    other = artifacts.put(b'{"wrong":"seal"}', media_type="application/json")
    with store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            ("extra-seal-operation", "extra_seal_fixture", "0" * 64, NOW.isoformat()),
        )
        db.execute(
            "INSERT INTO sealed_evidence(race_id,raw_manifest_checksum,normalized_checksum,schema_version,normalization_version,frozen_at,freeze_authority,odds_checksum,sealed_at,operation_id,request_intent_digest) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                str(race),
                str(digest("a")),
                str(other.checksum),
                "v1",
                "v1",
                NOW.isoformat(),
                "actual_jump",
                str(digest("c")),
                NOW.isoformat(),
                "extra-seal-operation",
                str(digest("f")),
            ),
        )
    assert build_durable_forward(setup).evidence_checksum == evidence.checksum
    mismatched = artifacts.put(b'{"valid":"but-not-the-outcome"}', media_type="application/json")
    with store._connect() as db:
        db.execute("DROP TRIGGER result_attempts_append_only_update")
        db.execute("UPDATE result_attempts SET artifact_checksum=?", (str(mismatched.checksum),))
    with pytest.raises(CorpusRejected, match="artifact disagrees"):
        build_durable_forward(setup, operation_id=OperationId("op_" + "8" * 32))


def test_real_forward_authority_sql_forgery_is_rejected(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store = setup[0]
    with store._connect() as db:
        db.execute(
            "INSERT INTO operations VALUES(?,?,?,?)",
            ("forgery-operation", "forgery", "0" * 64, NOW.isoformat()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="relations disagree"):
            db.execute(
                "INSERT INTO canonical_training_examples VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    "forged",
                    "phase3-example-1",
                    str(setup[2]),
                    str(digest("8")),
                    str(setup[4].checksum),
                    str(digest("7")),
                    str(digest("6")),
                    "2026-07-23",
                    NOW.isoformat(),
                    "forgery-operation",
                ),
            )


def test_real_forward_authority_rejects_temporal_violation(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, artifacts = setup[:2]
    outcome = json.loads(artifacts.read(setup[4].checksum))
    outcome["published_at"] = NOW.isoformat()
    outcome["provenance"]["observed_at"] = NOW.isoformat()
    bad = artifacts.put(canonical_json(outcome), media_type="application/json")
    with store._connect() as db:
        db.execute("DROP TRIGGER result_attempts_append_only_update")
        db.execute(
            "UPDATE result_attempts SET artifact_checksum=?,outcome_json=?",
            (str(bad.checksum), canonical_json(outcome).decode()),
        )
    with pytest.raises(CorpusRejected, match="temporal"):
        build_durable_forward(setup)


def test_real_forward_authority_rejects_nonobject_result_and_evidence_envelopes(tmp_path):
    setup = durable_forward_setup(tmp_path)
    store, artifacts, _, evidence = setup[:4]
    bad_result = artifacts.put(b"[]", media_type="application/json")
    with store._connect() as db:
        db.execute("DROP TRIGGER result_attempts_append_only_update")
        db.execute(
            "UPDATE result_attempts SET artifact_checksum=?,outcome_json='[]'",
            (str(bad_result.checksum),),
        )
    with pytest.raises(CorpusRejected, match="result envelope"):
        build_durable_forward(setup)

    second = durable_forward_setup(tmp_path / "second")
    bad_evidence = second[1].put(b"[]", media_type="application/json")
    with second[0]._connect() as db:
        db.execute("DROP TRIGGER sealed_evidence_append_only_update")
        db.execute(
            "UPDATE sealed_evidence SET normalized_checksum=? WHERE normalized_checksum=?",
            (str(bad_evidence.checksum), str(second[3].checksum)),
        )
        db.execute("DROP TRIGGER deferred_predictions_append_only_update")
        db.execute(
            "UPDATE deferred_predictions SET evidence_checksum=?", (str(bad_evidence.checksum),)
        )
    with pytest.raises(CorpusRejected, match="evidence envelope"):
        build_durable_forward(second)


@pytest.mark.parametrize("leakage", ["top-level", "mixed-case", "nested-list"])
def test_real_forward_authority_rejects_result_feature_leakage(tmp_path, leakage):
    setup = durable_forward_setup(tmp_path)
    store, artifacts, _, evidence = setup[:4]
    document = json.loads(artifacts.read(evidence.checksum))
    if leakage == "top-level":
        document["fields"]["result_order"] = [1, 2]
    elif leakage == "mixed-case":
        document["fields"]["runner_features"]["dog-a"]["ReSuLt_OrDeR"] = 1
    else:
        document["fields"]["runner_features"]["dog-a"]["history"] = [{"winner": True}]
    bad = artifacts.put(canonical_json(document), media_type="application/json")
    with store._connect() as db:
        db.execute("DROP TRIGGER sealed_evidence_append_only_update")
        db.execute(
            "UPDATE sealed_evidence SET normalized_checksum=? WHERE normalized_checksum=?",
            (str(bad.checksum), str(evidence.checksum)),
        )
        db.execute("DROP TRIGGER deferred_predictions_append_only_update")
        db.execute("UPDATE deferred_predictions SET evidence_checksum=?", (str(bad.checksum),))
    with pytest.raises(CorpusRejected, match="post-result feature"):
        build_durable_forward(setup)
