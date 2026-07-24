"""Paired Phase-6 evaluation, drift diagnosis, and guarded next-day promotion."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any, Mapping, Sequence

from .artifacts import ArtifactStore, ArtifactStoreError, ChecksumMismatch
from .domain import ArtifactChecksum, OperationId, require_aware
from .model_bundle import COMPONENT_KINDS, ServingAssignment
from .operations import OperationsStoreError, SQLiteOperationsStore, iso_timestamp
from .operational import OperationalAuthority, OperationalRejected, verify_release_authority
from .ordered_finish import OrderedFinishForecast, ordered_finish_nll


class EvaluationRejected(ValueError):
    """Evidence cannot support an authoritative paired comparison."""


class PromotionRejected(OperationsStoreError):
    """A promotion gate or immutable durable relation failed closed."""


@dataclass(frozen=True, slots=True)
class EvaluationRace:
    race_id: str
    racing_day: str
    venue: str
    distance_m: int
    grade: str
    official_order: tuple[str, ...]
    evidence_checksum: ArtifactChecksum
    result_checksum: ArtifactChecksum
    evidence_frozen_at: str
    forecast_computed_at: str
    result_observed_at: str
    eligible: bool = True
    exclusion: str | None = None
    sealed_odds: Mapping[str, tuple[float, float]] | None = None
    odds_checksum: ArtifactChecksum | None = None

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.race_id, self.racing_day, self.venue, self.grade)
        ):
            raise EvaluationRejected("race identity and slice provenance are required")
        date.fromisoformat(self.racing_day)
        if type(self.distance_m) is not int or self.distance_m <= 0:
            raise EvaluationRejected("positive distance provenance is required")
        if len(self.official_order) < 2 or len(set(self.official_order)) != len(
            self.official_order
        ):
            raise EvaluationRejected("one unambiguous official order is required")
        if not isinstance(self.evidence_checksum, ArtifactChecksum) or not isinstance(
            self.result_checksum, ArtifactChecksum
        ):
            raise EvaluationRejected("exact evidence and result checksums are required")
        frozen = datetime.fromisoformat(self.evidence_frozen_at)
        forecasted = datetime.fromisoformat(self.forecast_computed_at)
        observed = datetime.fromisoformat(self.result_observed_at)
        if any(
            value.tzinfo is None or value.utcoffset() is None
            for value in (frozen, forecasted, observed)
        ):
            raise EvaluationRejected("evaluation timestamps must be timezone-aware")
        if not frozen <= forecasted < observed:
            raise EvaluationRejected("temporally invalid forecast or result evidence")
        if not self.eligible or self.exclusion is not None:
            raise EvaluationRejected("excluded or ambiguous race cannot enter evaluation")
        if (self.sealed_odds is None) != (self.odds_checksum is None):
            raise EvaluationRejected("sealed odds and checksum must be present together")
        if self.sealed_odds is not None and (
            set(self.sealed_odds) != set(self.official_order)
            or any(
                type(prices) is not tuple
                or len(prices) != 2
                or any(
                    type(price) not in (int, float) or not math.isfinite(price) or price <= 1
                    for price in prices
                )
                for prices in self.sealed_odds.values()
            )
        ):
            raise EvaluationRejected("sealed odds are incomplete or invalid")


@dataclass(frozen=True, slots=True)
class ForecastEvidence:
    race_id: str
    bundle_id: str
    bundle_checksum: ArtifactChecksum
    evidence_checksum: ArtifactChecksum
    forecast_checksum: ArtifactChecksum
    forecast: OrderedFinishForecast
    computed_at: str | None = None


@dataclass(frozen=True, slots=True)
class EligibleRaceIdentity:
    race_id: str
    evidence_checksum: ArtifactChecksum
    result_checksum: ArtifactChecksum


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    policy_id: str = "phase6-promotion-v1"
    minimum_races: int = 500
    minimum_venues: int = 3
    minimum_races_per_venue: int = 25
    short_horizon_races: int = 100
    practical_loss_reduction: float = 0.01
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260722
    superiority_probability: float = 0.95
    maximum_calibration_ece: float = 0.10
    maximum_coverage_drop: float = 0.0
    maximum_slice_loss_increase: float = 0.05
    maximum_short_loss_increase: float = 0.0

    def __post_init__(self) -> None:
        values = asdict(self)
        if type(self.policy_id) is not str or not self.policy_id.strip():
            raise EvaluationRejected("policy identity is required")
        if type(self.minimum_races) is not int or self.minimum_races < 500:
            raise EvaluationRejected(
                "policy minimum_races cannot be below the Phase 6 floor of 500"
            )
        for name in (
            "minimum_venues",
            "minimum_races_per_venue",
            "short_horizon_races",
            "bootstrap_samples",
        ):
            if type(values[name]) is not int or values[name] <= 0:
                raise EvaluationRejected(f"policy {name} must be a positive integer")
        for name, value in values.items():
            if name not in {
                "policy_id",
                "minimum_races",
                "minimum_venues",
                "minimum_races_per_venue",
                "short_horizon_races",
                "bootstrap_samples",
                "bootstrap_seed",
            } and (
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise EvaluationRejected(f"policy {name} must be finite")
        if not 0 < self.superiority_probability <= 1:
            raise EvaluationRejected("policy superiority probability is invalid")


def _digest(value: Any) -> ArtifactChecksum:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return ArtifactChecksum("sha256:" + hashlib.sha256(encoded).hexdigest())


def forecast_checksum(forecast: OrderedFinishForecast) -> ArtifactChecksum:
    """Independent checksum of the coherent probability distribution."""
    return _digest(
        {
            "runner_ids": list(forecast.runner_ids),
            "orders": [
                [list(order), probability]
                for order, probability in sorted(forecast.order_probabilities.items())
            ],
        }
    )


def forecast_document(evidence: ForecastEvidence) -> Mapping[str, Any]:
    """Canonical authenticated forecast identity, including its complete provenance."""
    if evidence.computed_at is None:
        raise EvaluationRejected("durable forecast computed_at is required")
    return {
        "schema_version": "phase6-ordered-forecast-v1",
        "race_id": evidence.race_id,
        "bundle_id": evidence.bundle_id,
        "bundle_checksum": str(evidence.bundle_checksum),
        "evidence_checksum": str(evidence.evidence_checksum),
        "computed_at": evidence.computed_at,
        "distribution": {
            "runner_ids": list(evidence.forecast.runner_ids),
            "orders": [
                [list(order), probability]
                for order, probability in sorted(evidence.forecast.order_probabilities.items())
            ],
        },
    }


def forecast_document_checksum(evidence: ForecastEvidence) -> ArtifactChecksum:
    return _digest(forecast_document(evidence))


def _calibration(probabilities: Sequence[float], outcomes: Sequence[int]) -> Mapping[str, Any]:
    bins = []
    weighted_error = 0.0
    for lower_index in range(10):
        lower, upper = lower_index / 10, (lower_index + 1) / 10
        members = [
            i for i, p in enumerate(probabilities) if lower <= p < upper or (upper == 1 and p == 1)
        ]
        if members:
            confidence = math.fsum(probabilities[i] for i in members) / len(members)
            observed = math.fsum(outcomes[i] for i in members) / len(members)
            weighted_error += len(members) * abs(confidence - observed)
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": len(members),
                    "mean_probability": confidence,
                    "observed_rate": observed,
                }
            )
        else:
            bins.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_probability": None,
                    "observed_rate": None,
                }
            )
    return {"ece": weighted_error / len(probabilities), "bins": bins}


def _metrics(pairs: Sequence[tuple[EvaluationRace, ForecastEvidence]]) -> Mapping[str, Any]:
    losses, reciprocal_ranks, containments, top2, top3 = [], [], [], [], []
    win_p, win_y, place_p, place_y = [], [], [], []
    for race, evidence in pairs:
        forecast, order = evidence.forecast, race.official_order
        if set(forecast.runner_ids) != set(order):
            raise EvaluationRejected(f"runner population mismatch for {race.race_id}")
        losses.append(ordered_finish_nll(forecast, order))
        reciprocal_ranks.append(1 / (forecast.ranking.index(order[0]) + 1))
        containments.append(int(set(forecast.ranking[:3]) == set(order[:3])))
        top2.append(int(forecast.ranking[:2] == order[:2]))
        top3.append(int(forecast.ranking[:3] == order[:3]))
        for runner in order:
            win_p.append(forecast.win[runner])
            win_y.append(int(runner == order[0]))
            place_p.append(forecast.top_3[runner])
            place_y.append(int(runner in order[:3]))
    count = len(pairs)
    return {
        "race_count": count,
        "mean_ordered_finish_nll": math.fsum(losses) / count,
        "race_losses": losses,
        "winner_mean_reciprocal_rank": math.fsum(reciprocal_ranks) / count,
        "top3_containment": math.fsum(containments) / count,
        "exact_top2_order_accuracy": math.fsum(top2) / count,
        "exact_top3_order_accuracy": math.fsum(top3) / count,
        "win_calibration": _calibration(win_p, win_y),
        "top3_calibration": _calibration(place_p, place_y),
    }


def _slice_key(race: EvaluationRace, kind: str) -> str:
    if kind == "venue":
        return race.venue
    if kind == "distance":
        return str(race.distance_m)
    if kind == "grade":
        return race.grade
    size = len(race.official_order)
    return "small(2-5)" if size <= 5 else "standard(6-8)"


def _score_model(
    races: Sequence[EvaluationRace], forecasts: Mapping[str, ForecastEvidence], eligible_count: int
) -> Mapping[str, Any]:
    pairs = [(race, forecasts[race.race_id]) for race in races]
    result = dict(_metrics(pairs))
    result["coverage"] = len(pairs) / eligible_count
    result["abstention"] = (eligible_count - len(pairs)) / eligible_count
    result["coverage_denominator"] = eligible_count
    result["slices"] = {}
    for kind in ("venue", "distance", "grade", "field_size"):
        groups: dict[str, list[tuple[EvaluationRace, ForecastEvidence]]] = {}
        for pair in pairs:
            groups.setdefault(_slice_key(pair[0], kind), []).append(pair)
        result["slices"][kind] = {key: _metrics(value) for key, value in sorted(groups.items())}
    return result


def _bootstrap(
    champion: Sequence[float], challenger: Sequence[float], policy: PromotionPolicy
) -> Mapping[str, Any]:
    rng = random.Random(policy.bootstrap_seed)
    differences = []
    for _ in range(policy.bootstrap_samples):
        indexes = [rng.randrange(len(champion)) for _ in champion]
        differences.append(math.fsum(challenger[i] - champion[i] for i in indexes) / len(indexes))
    differences.sort()
    probability = math.fsum(value < 0 for value in differences) / len(differences)
    return {
        "unit": "race",
        "samples": policy.bootstrap_samples,
        "seed": policy.bootstrap_seed,
        "challenger_minus_champion_mean": math.fsum(c - i for c, i in zip(challenger, champion))
        / len(champion),
        "lower_95": differences[int(0.025 * len(differences))],
        "upper_95": differences[min(len(differences) - 1, int(0.975 * len(differences)))],
        "probability_superior": probability,
    }


def _wagering_scorecard(
    races: Sequence[EvaluationRace], forecasts: Mapping[str, ForecastEvidence]
) -> Mapping[str, Any]:
    win_return = place_return = 0.0
    win_hits = place_hits = 0
    for race in races:
        forecast = forecasts[race.race_id].forecast
        winner_pick = forecast.ranking[0]
        place_pick = forecast.ranking[0]
        win_hit = winner_pick == race.official_order[0]
        place_hit = place_pick in race.official_order[:3]
        win_hits += win_hit
        place_hits += place_hit
        win_return += race.sealed_odds[winner_pick][0] if win_hit else 0.0  # type: ignore[index]
        place_return += race.sealed_odds[place_pick][1] if place_hit else 0.0  # type: ignore[index]
    count = len(races)
    return {
        "report_only": True,
        "real_betting": False,
        "unit_stake_per_race": 1.0,
        "win": {"bets": count, "hits": win_hits, "net_return": win_return - count},
        "place": {"bets": count, "hits": place_hits, "net_return": place_return - count},
    }


def _evaluate_paired(
    races: Sequence[EvaluationRace],
    model_forecasts: Mapping[str, Sequence[ForecastEvidence]],
    *,
    champion_bundle_id: str,
    challenger_bundle_ids: Sequence[str],
    policy: PromotionPolicy = PromotionPolicy(),
    eligible_population: Sequence[EligibleRaceIdentity],
) -> Mapping[str, Any]:
    """Pure metric kernel; callers must authenticate bundles before invoking it."""
    if not races or len({r.race_id for r in races}) != len(races):
        raise EvaluationRejected("evaluation races must be non-empty and unique")
    identities = (champion_bundle_id, *challenger_bundle_ids)
    if len(set(identities)) != len(identities) or set(model_forecasts) != set(identities):
        raise EvaluationRejected("champion and exact challenger forecast identities must align")
    race_ids = {r.race_id for r in races}
    by_model = {}
    checksums = {}
    for bundle_id in identities:
        values = model_forecasts[bundle_id]
        if (
            len({f.race_id for f in values}) != len(values)
            or {f.race_id for f in values} != race_ids
        ):
            raise EvaluationRejected("every model must cover the exact paired race population")
        by_model[bundle_id] = {f.race_id: f for f in values}
        bundle_checksums = {f.bundle_checksum for f in values}
        if len(bundle_checksums) != 1 or any(f.bundle_id != bundle_id for f in values):
            raise EvaluationRejected("forecast bundle provenance is mutable or inconsistent")
        if any(f.forecast_checksum != forecast_document_checksum(f) for f in values):
            raise EvaluationRejected("forecast checksum mismatch for canonical document")
        checksums[bundle_id] = str(next(iter(bundle_checksums)))
    for race in races:
        for bundle_id in identities:
            evidence = by_model[bundle_id][race.race_id]
            if evidence.evidence_checksum != race.evidence_checksum:
                raise EvaluationRejected("forecast/evaluation evidence checksum mismatch")
    ordered_races = sorted(races, key=lambda r: (r.racing_day, r.race_id))
    eligible = {
        item.race_id: (str(item.evidence_checksum), str(item.result_checksum))
        for item in eligible_population
    }
    if len(eligible) != len(eligible_population):
        raise EvaluationRejected("eligible population identities must be unique")
    paired_relations = {
        r.race_id: (str(r.evidence_checksum), str(r.result_checksum)) for r in ordered_races
    }
    if not paired_relations.items() <= eligible.items():
        raise EvaluationRejected("paired population is absent from exact eligible relations")
    eligible_count = len(eligible)
    scores = {
        identity: _score_model(ordered_races, by_model[identity], eligible_count)
        for identity in identities
    }
    short_races = ordered_races[-policy.short_horizon_races :]
    short = {
        identity: _score_model(short_races, by_model[identity], len(short_races))
        for identity in identities
    }
    days = {}
    for day in sorted({r.racing_day for r in short_races}):
        day_races = [r for r in short_races if r.racing_day == day]
        days[day] = {
            identity: _score_model(day_races, by_model[identity], len(day_races))
            for identity in identities
        }
    population = [
        {
            "race_id": r.race_id,
            "evidence_checksum": str(r.evidence_checksum),
            "result_checksum": str(r.result_checksum),
        }
        for r in ordered_races
    ]
    bootstraps = {
        challenger: _bootstrap(
            scores[champion_bundle_id]["race_losses"], scores[challenger]["race_losses"], policy
        )
        for challenger in challenger_bundle_ids
    }
    wagering: Mapping[str, Any] = {
        "status": "NOT_RUN",
        "report_only": True,
        "real_betting": False,
        "reason": "immutable sealed win/place odds unavailable",
    }
    if all(r.sealed_odds is not None for r in ordered_races):
        wagering = {
            "status": "RUN",
            "report_only": True,
            "real_betting": False,
            "sealed_odds_races": len(ordered_races),
            "odds_checksums": [str(r.odds_checksum) for r in ordered_races],
            "models": {
                identity: _wagering_scorecard(ordered_races, by_model[identity])
                for identity in identities
            },
        }
    return {
        "schema_version": "phase6-evaluation-v1",
        "policy": asdict(policy),
        "policy_checksum": str(_digest(asdict(policy))),
        "champion_bundle_id": champion_bundle_id,
        "challenger_bundle_ids": list(challenger_bundle_ids),
        "bundle_checksums": checksums,
        "population": population,
        "population_checksum": str(_digest(population)),
        "eligible_population": [
            {
                "race_id": race_id,
                "evidence_checksum": checksums[0],
                "result_checksum": checksums[1],
            }
            for race_id, checksums in sorted(eligible.items())
        ],
        "eligible_population_checksum": str(_digest(sorted(eligible.items()))),
        "long_horizon": scores,
        "short_horizon": short,
        "racing_day_views": days,
        "bootstrap": bootstraps,
        "wagering_scorecard": wagering,
    }


class EvaluationAuthority:
    """Authenticates immutable registered bundles before producing a scorecard."""

    def __init__(self, store: SQLiteOperationsStore, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts

    def begin_run(self, operation_id: OperationId, *, run_kind: str, started_at: datetime) -> bool:
        payload = {
            "run_id": str(operation_id),
            "kind": run_kind,
            "started_at": iso_timestamp(started_at),
        }
        with self.store._operation(operation_id, "begin_phase6_run", payload) as (db, replay):
            if replay:
                return False
            db.execute(
                "INSERT INTO phase6_runs VALUES(?,?,?,?)",
                (str(operation_id), run_kind, iso_timestamp(started_at), str(operation_id)),
            )
        return True

    def register_forward_race(
        self,
        operation_id: OperationId,
        *,
        training_example_id: str,
        registered_at: datetime,
    ) -> bool:
        """Derive one evaluation denominator row from authenticated Phase 2-5 state."""
        require_aware(registered_at, "registered_at")
        with self.store._connect() as db:
            example = db.execute(
                "SELECT * FROM canonical_training_examples WHERE training_example_id=?",
                (training_example_id,),
            ).fetchone()
            field_rows = {
                row["field_name"]: row
                for row in db.execute(
                    "SELECT field_name,value_json,distinct_top_values "
                    "FROM phase6_resolved_field_evidence "
                    "WHERE race_id=(SELECT race_id FROM canonical_training_examples "
                    "WHERE training_example_id=?) AND field_name IN "
                    "('venue','distance','grade','field_size')",
                    (training_example_id,),
                )
            }
        required_fields = {"venue", "distance", "grade", "field_size"}
        if (
            example is None
            or set(field_rows) != required_fields
            or any(row["distinct_top_values"] != 1 for row in field_rows.values())
        ):
            raise EvaluationRejected("forward race lacks complete resolved Phase 2-5 authority")
        fields = {
            field_name: json.loads(row["value_json"]) for field_name, row in field_rows.items()
        }
        for checksum in (
            example["evidence_checksum"],
            example["result_checksum"],
            example["artifact_checksum"],
        ):
            self.artifacts.verify(ArtifactChecksum(checksum))
        values = (
            example["race_id"],
            example["training_example_id"],
            example["racing_date"],
            example["evidence_checksum"],
            example["result_checksum"],
            example["artifact_checksum"],
            fields["venue"],
            fields["distance"],
            fields["grade"],
            fields["field_size"],
            iso_timestamp(registered_at),
            str(operation_id),
        )
        payload = {
            "race_id": values[0],
            "training_example_id": training_example_id,
            "racing_day": values[2],
            "evidence_checksum": values[3],
            "result_checksum": values[4],
            "training_artifact_checksum": values[5],
            "venue": values[6],
            "distance_m": values[7],
            "grade": values[8],
            "field_size": values[9],
            "distinct_top_values": {
                field_name: row["distinct_top_values"]
                for field_name, row in sorted(field_rows.items())
            },
            "registered_at": values[10],
        }
        with self.store._operation(operation_id, "register_forward_evaluation", payload) as (
            db,
            replay,
        ):
            if replay:
                durable = db.execute(
                    "SELECT race_id,training_example_id,racing_day,evidence_checksum,"
                    "result_checksum,training_artifact_checksum,venue,distance_m,grade,"
                    "field_size,registered_at,operation_id "
                    "FROM phase6_forward_evaluation_races WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if durable is None or tuple(durable) != values:
                    raise EvaluationRejected(
                        "forward race replay lacks its exact durable Phase 2-5 relation"
                    )
                return False
            db.execute(
                "INSERT INTO phase6_forward_evaluation_races VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                values,
            )
        return True

    def register_policy(
        self, operation_id: OperationId, policy: PromotionPolicy, registered_at: datetime
    ) -> ArtifactChecksum:
        document = asdict(policy)
        content = json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        artifact = self.artifacts.put(content, media_type="application/json")
        policy_checksum = _digest(document)
        payload = {
            "policy_id": policy.policy_id,
            "checksum": str(policy_checksum),
            "registered_at": iso_timestamp(registered_at),
        }
        with self.store._operation(operation_id, "register_phase6_policy", payload) as (db, replay):
            if not replay:
                db.execute(
                    "INSERT INTO phase6_policy_registry VALUES(?,?,?,?,?)",
                    (
                        policy.policy_id,
                        str(policy_checksum),
                        str(artifact.checksum),
                        iso_timestamp(registered_at),
                        str(operation_id),
                    ),
                )
        return policy_checksum

    def record_bundle_lineage(
        self,
        operation_id: OperationId,
        *,
        bundle_id: str,
        registration_run_id: OperationId,
        source_run_id: OperationId,
    ) -> bool:
        payload = {
            "bundle_id": bundle_id,
            "registration_run_id": str(registration_run_id),
            "source_run_id": str(source_run_id),
        }
        with self.store._operation(operation_id, "record_phase6_bundle_lineage", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            bundle = db.execute(
                "SELECT operation_id FROM canonical_model_bundles WHERE bundle_id=?",
                (bundle_id,),
            ).fetchone()
            if bundle is None:
                raise EvaluationRejected("bundle registration is unavailable")
            db.execute(
                "INSERT INTO phase6_bundle_lineage_v2 VALUES(?,?,?,?,?)",
                (
                    bundle_id,
                    bundle["operation_id"],
                    str(registration_run_id),
                    str(source_run_id),
                    str(operation_id),
                ),
            )
        return True

    def record_forecast(
        self,
        operation_id: OperationId,
        evidence: ForecastEvidence,
        *,
        evaluation_run_id: OperationId,
    ) -> bool:
        if evidence.computed_at is None:
            raise EvaluationRejected("durable forecast computed_at is required")
        document = forecast_document(evidence)
        content = json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        artifact = self.artifacts.put(content, media_type="application/json")
        if forecast_document_checksum(evidence) != evidence.forecast_checksum:
            raise EvaluationRejected("forecast document checksum disagrees")
        payload = {
            "race_id": evidence.race_id,
            "bundle_id": evidence.bundle_id,
            "checksum": str(artifact.checksum),
            "evaluation_run_id": str(evaluation_run_id),
        }
        with self.store._operation(operation_id, "record_phase6_forecast", payload) as (db, replay):
            if replay:
                return False
            run = db.execute(
                "SELECT started_at FROM phase6_runs WHERE run_id=? AND run_kind='evaluation'",
                (str(evaluation_run_id),),
            ).fetchone()
            bundle = db.execute(
                "SELECT created_at FROM canonical_model_bundles WHERE bundle_id=? AND bundle_checksum=?",
                (evidence.bundle_id, str(evidence.bundle_checksum)),
            ).fetchone()
            if run is None or bundle is None or bundle["created_at"] >= run["started_at"]:
                raise EvaluationRejected(
                    "forecast candidate was not registered before evaluation run"
                )
            origin = db.execute(
                "SELECT origin.artifact_checksum,origin.generated_at FROM phase6_forecast_service_artifacts origin "
                "JOIN phase6_forecast_computation_bindings binding USING(forecast_checksum) "
                "JOIN phase6_service_computations computation USING(computation_id) "
                "WHERE origin.forecast_checksum=? AND origin.race_id=? AND origin.bundle_id=? "
                "AND origin.bundle_checksum=? AND origin.evidence_checksum=? "
                "AND computation.race_id=origin.race_id AND computation.bundle_id=origin.bundle_id "
                "AND computation.bundle_checksum=origin.bundle_checksum "
                "AND computation.evidence_checksum=origin.evidence_checksum "
                "AND computation.computed_at=origin.generated_at",
                (
                    str(evidence.forecast_checksum),
                    evidence.race_id,
                    evidence.bundle_id,
                    str(evidence.bundle_checksum),
                    str(evidence.evidence_checksum),
                ),
            ).fetchone()
            if origin is None or origin["generated_at"] != evidence.computed_at:
                raise EvaluationRejected(
                    "forecast lacks an authenticated pre-result service artifact"
                )
            if origin["artifact_checksum"] != str(artifact.checksum):
                raise EvaluationRejected("forecast service artifact checksum disagrees")
            db.execute(
                "INSERT INTO phase6_forecast_artifacts VALUES(?,?,?,?,?,?,?,?)",
                (
                    evidence.race_id,
                    evidence.bundle_id,
                    str(evidence.bundle_checksum),
                    str(evidence.evidence_checksum),
                    str(artifact.checksum),
                    evidence.computed_at,
                    str(evaluation_run_id),
                    str(operation_id),
                ),
            )
        return True

    def _registrations(self, bundle_ids: Sequence[str]) -> Mapping[str, str]:
        authenticated = {}
        with self.store._connect() as db:
            for bundle_id in bundle_ids:
                row = db.execute(
                    "SELECT bundle_checksum FROM canonical_model_bundles WHERE bundle_id=?",
                    (bundle_id,),
                ).fetchone()
                components = db.execute(
                    "SELECT artifact_checksum FROM canonical_bundle_components WHERE bundle_id=?",
                    (bundle_id,),
                ).fetchall()
                if row is None or len(components) != len(COMPONENT_KINDS):
                    raise EvaluationRejected("model is not a complete immutable registration")
                self.artifacts.verify(ArtifactChecksum(row["bundle_checksum"]))
                for component in components:
                    self.artifacts.verify(ArtifactChecksum(component["artifact_checksum"]))
                authenticated[bundle_id] = row["bundle_checksum"]
        return authenticated

    def _authenticate_population(
        self,
        races: Sequence[EvaluationRace],
        eligible_population: Sequence[EligibleRaceIdentity],
    ) -> None:
        races_by_id = {race.race_id: race for race in races}
        with self.store._connect() as db:
            durable_population = {
                row["race_id"]: (row["evidence_checksum"], row["result_checksum"])
                for row in db.execute(
                    "SELECT race_id,evidence_checksum,result_checksum "
                    "FROM phase6_forward_evaluation_races"
                )
            }
            supplied_population = {
                identity.race_id: (
                    str(identity.evidence_checksum),
                    str(identity.result_checksum),
                )
                for identity in eligible_population
            }
            if supplied_population != durable_population:
                raise EvaluationRejected(
                    "eligible denominator is not the complete durable forward population"
                )
            for identity in eligible_population:
                row = db.execute(
                    "SELECT * FROM phase6_forward_evaluation_races WHERE race_id=? "
                    "AND evidence_checksum=? AND result_checksum=?",
                    (
                        identity.race_id,
                        str(identity.evidence_checksum),
                        str(identity.result_checksum),
                    ),
                ).fetchone()
                if row is None:
                    raise EvaluationRejected(
                        "eligible race lacks an exact durable forward evaluation relation"
                    )
                self.artifacts.read(identity.evidence_checksum)
                self.artifacts.read(identity.result_checksum)
                if identity.race_id in races_by_id:
                    race = races_by_id[identity.race_id]
                    try:
                        training = json.loads(
                            self.artifacts.read(ArtifactChecksum(row["training_artifact_checksum"]))
                        )
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise EvaluationRejected(
                            "canonical training artifact is invalid"
                        ) from error
                    required_training = {
                        "schema_version": "canonical-training-example-v1",
                        "origin": "forward-sealed",
                        "promotion_evidence_eligible": True,
                        "race_id": race.race_id,
                        "racing_date": race.racing_day,
                        "evidence_checksum": str(race.evidence_checksum),
                        "result_checksum": str(race.result_checksum),
                        "official_order": list(race.official_order),
                        "evidence_frozen_at": race.evidence_frozen_at,
                        "result_published_at": race.result_observed_at,
                    }
                    if (
                        any(training.get(key) != value for key, value in required_training.items())
                        or set(training.get("runner_ids", [])) != set(race.official_order)
                        or (
                            row["venue"] != race.venue
                            or row["distance_m"] != race.distance_m
                            or row["grade"] != race.grade
                            or row["field_size"] != len(race.official_order)
                        )
                    ):
                        raise EvaluationRejected(
                            "race metadata or result disagrees with canonical forward evidence"
                        )

    def evaluate(
        self,
        races: Sequence[EvaluationRace],
        model_forecasts: Mapping[str, Sequence[ForecastEvidence]],
        *,
        evaluation_run_id: OperationId,
        champion_bundle_id: str,
        challenger_bundle_ids: Sequence[str],
        eligible_population: Sequence[EligibleRaceIdentity],
        policy: PromotionPolicy = PromotionPolicy(),
    ) -> Mapping[str, Any]:
        identities = (champion_bundle_id, *challenger_bundle_ids)
        registrations = self._registrations(identities)
        self._authenticate_population(races, eligible_population)
        with self.store._connect() as db:
            run = db.execute(
                "SELECT started_at FROM phase6_runs WHERE run_id=? AND run_kind='evaluation'",
                (str(evaluation_run_id),),
            ).fetchone()
            pointer = db.execute(
                "SELECT bundle_id,bundle_checksum,assignment_id FROM champion_pointer WHERE singleton=1"
            ).fetchone()
            policy_row = db.execute(
                "SELECT * FROM phase6_policy_registry WHERE policy_id=?",
                (policy.policy_id,),
            ).fetchone()
            if run is None or pointer is None or pointer["bundle_id"] != champion_bundle_id:
                raise EvaluationRejected("report champion is not the durable current champion")
            if pointer["bundle_checksum"] != registrations[champion_bundle_id]:
                raise EvaluationRejected("current champion checksum disagrees")
            policy_content = json.dumps(
                asdict(policy), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
            if (
                policy_row is None
                or policy_row["policy_checksum"] != str(_digest(asdict(policy)))
                or policy_row["artifact_checksum"] != policy_row["policy_checksum"]
                or self.artifacts.read(ArtifactChecksum(policy_row["artifact_checksum"]))
                != policy_content
            ):
                raise EvaluationRejected("promotion policy is not the exact registered policy")
            for bundle_id, forecasts in model_forecasts.items():
                for forecast in forecasts:
                    durable = db.execute(
                        "SELECT * FROM phase6_forecast_artifacts WHERE race_id=? AND bundle_id=? "
                        "AND bundle_checksum=? AND evidence_checksum=? AND computed_at=? "
                        "AND evaluation_run_id=?",
                        (
                            forecast.race_id,
                            bundle_id,
                            str(forecast.bundle_checksum),
                            str(forecast.evidence_checksum),
                            forecast.computed_at,
                            str(evaluation_run_id),
                        ),
                    ).fetchone()
                    race = next((item for item in races if item.race_id == forecast.race_id), None)
                    if durable is None or race is None or forecast.computed_at is None:
                        raise EvaluationRejected("forecast lacks exact durable replay evidence")
                    computed = datetime.fromisoformat(forecast.computed_at)
                    if (
                        not datetime.fromisoformat(race.evidence_frozen_at)
                        <= computed
                        < datetime.fromisoformat(race.result_observed_at)
                    ):
                        raise EvaluationRejected("durable forecast is not result-independent")
                    artifact = json.loads(
                        self.artifacts.read(ArtifactChecksum(durable["forecast_checksum"]))
                    )
                    if (
                        artifact.get("race_id") != race.race_id
                        or artifact.get("bundle_id") != bundle_id
                        or artifact.get("computed_at") != forecast.computed_at
                        or artifact.get("evidence_checksum") != str(race.evidence_checksum)
                        or artifact.get("distribution")
                        != {
                            "runner_ids": list(forecast.forecast.runner_ids),
                            "orders": [
                                [list(order), probability]
                                for order, probability in sorted(
                                    forecast.forecast.order_probabilities.items()
                                )
                            ],
                        }
                    ):
                        raise EvaluationRejected("durable forecast artifact provenance disagrees")
            for race in races:
                start = db.execute(
                    "SELECT scheduled_jump FROM expected_races WHERE race_id=?",
                    (race.race_id,),
                ).fetchone()
                if start is None:
                    raise EvaluationRejected("race lacks an authenticated scheduled start")
                sealed = db.execute(
                    "SELECT odds_checksum FROM sealed_evidence WHERE race_id=? AND normalized_checksum=?",
                    (race.race_id, str(race.evidence_checksum)),
                ).fetchone()
                if sealed is None:
                    raise EvaluationRejected("race lacks its durable sealed-evidence relation")
                odds_checksum = ArtifactChecksum(sealed["odds_checksum"])
                try:
                    odds_document = json.loads(self.artifacts.read(odds_checksum))
                except ChecksumMismatch:
                    raise
                except ArtifactStoreError:
                    if race.sealed_odds is not None:
                        raise EvaluationRejected("claimed sealed odds artifact is unavailable")
                    continue
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise EvaluationRejected("sealed odds artifact is malformed") from error
                expected_prices = {
                    runner: list(prices) for runner, prices in (race.sealed_odds or {}).items()
                }
                if (
                    race.odds_checksum != odds_checksum
                    or odds_document.get("schema_version") != "sealed-win-place-odds-v1"
                    or odds_document.get("race_id") != race.race_id
                    or odds_document.get("win_place") != expected_prices
                    or datetime.fromisoformat(odds_document["captured_at"])
                    >= datetime.fromisoformat(start["scheduled_jump"])
                ):
                    raise EvaluationRejected("sealed odds identity, time, or checksum disagrees")
        result = dict(
            _evaluate_paired(
                races,
                model_forecasts,
                champion_bundle_id=champion_bundle_id,
                challenger_bundle_ids=challenger_bundle_ids,
                eligible_population=eligible_population,
                policy=policy,
            )
        )
        if result["bundle_checksums"] != registrations:
            raise EvaluationRejected("forecast checksums disagree with immutable registrations")
        result["registry_authentication"] = str(_digest(registrations))
        result["champion_assignment_id"] = pointer["assignment_id"]
        result["evaluation_run_id"] = str(evaluation_run_id)
        result["drift_diagnosis"] = diagnose_drift(result)
        return result

    def evaluate_and_seal(
        self,
        operation_id: OperationId,
        *,
        evidence_id: str,
        evaluated_at: datetime,
        races: Sequence[EvaluationRace],
        model_forecasts: Mapping[str, Sequence[ForecastEvidence]],
        evaluation_run_id: OperationId,
        champion_bundle_id: str,
        challenger_bundle_ids: Sequence[str],
        eligible_population: Sequence[EligibleRaceIdentity],
        policy: PromotionPolicy = PromotionPolicy(),
    ) -> Mapping[str, Any]:
        report = self.evaluate(
            races,
            model_forecasts,
            evaluation_run_id=evaluation_run_id,
            champion_bundle_id=champion_bundle_id,
            challenger_bundle_ids=challenger_bundle_ids,
            eligible_population=eligible_population,
            policy=policy,
        )
        _validate_report(report)
        content = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        checksum = _digest(report)
        self.artifacts.put(content, media_type="application/json", expected_checksum=checksum)
        challengers = report["challenger_bundle_ids"]
        if len(challengers) != 1:
            raise EvaluationRejected("one sealed promotion comparison is required per challenger")
        with self.store._connect() as db:
            durable_count = db.execute(
                "SELECT COUNT(*) FROM phase6_forward_evaluation_races"
            ).fetchone()[0]
        if (
            durable_count < 500
            or report["long_horizon"][report["champion_bundle_id"]]["race_count"] != durable_count
        ):
            raise EvaluationRejected(
                "sealed report is not the complete durable 500-race population"
            )
        payload = {
            "evidence_id": evidence_id,
            "artifact_checksum": str(checksum),
            "evaluated_at": iso_timestamp(evaluated_at),
        }
        with self.store._operation(operation_id, "seal_phase6_evaluation", payload) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT artifact_checksum FROM phase6_evaluation_evidence WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or row[0] != str(checksum):
                    raise EvaluationRejected("evaluation replay lacks exact durable evidence")
            else:
                db.execute(
                    "INSERT INTO phase6_evaluation_evidence VALUES(?,?,?,?,?,?,?,?)",
                    (
                        evidence_id,
                        report["champion_bundle_id"],
                        challengers[0],
                        report["population_checksum"],
                        str(checksum),
                        report["policy"]["policy_id"],
                        iso_timestamp(evaluated_at),
                        str(operation_id),
                    ),
                )
                challenger = db.execute(
                    "SELECT created_at FROM canonical_model_bundles WHERE bundle_id=?",
                    (challengers[0],),
                ).fetchone()
                if challenger is None:
                    raise EvaluationRejected("sealed challenger registration is unavailable")
                db.execute(
                    "INSERT INTO phase6_trusted_evaluations VALUES(?,?,?,?,?,?,?,?)",
                    (
                        evidence_id,
                        report["evaluation_run_id"],
                        report["champion_assignment_id"],
                        challenger["created_at"],
                        report["policy_checksum"],
                        str(checksum),
                        iso_timestamp(evaluated_at),
                        str(operation_id),
                    ),
                )
        return report


def diagnose_drift(
    report: Mapping[str, Any], *, degradation_threshold: float = 0.0
) -> Mapping[str, str]:
    champion = report["champion_bundle_id"]
    long, short = report["long_horizon"], report["short_horizon"]
    degraded = {
        model: short[model]["mean_ordered_finish_nll"] - long[model]["mean_ordered_finish_nll"]
        > degradation_threshold
        for model in long
    }
    if degraded and all(degraded.values()):
        return {
            "diagnosis": "data_domain_drift",
            "action": "block promotion; investigate evidence/domain and emit training request only",
        }
    if degraded.get(champion) and all(
        not value for model, value in degraded.items() if model != champion
    ):
        return {
            "diagnosis": "model_drift",
            "action": "block incumbent promotion decisions; nominate registered challenger investigation",
        }
    return {"diagnosis": "no_classified_drift", "action": "retain monitoring"}


def promotion_decision(report: Mapping[str, Any], challenger: str) -> Mapping[str, Any]:
    if (
        type(report.get("policy")) is not dict
        or type(report["policy"].get("minimum_races")) is not int
        or report["policy"]["minimum_races"] < 500
    ):
        raise EvaluationRejected("promotion policy minimum_races is below the Phase 6 floor of 500")
    policy = PromotionPolicy(**report["policy"])
    champion = report["champion_bundle_id"]
    long, short = report["long_horizon"], report["short_horizon"]
    incumbent, candidate = long[champion], long[challenger]
    reasons = []
    if incumbent["race_count"] < policy.minimum_races:
        reasons.append("minimum_races")
    venue_counts = [s["race_count"] for s in candidate["slices"]["venue"].values()]
    if (
        len(venue_counts) < policy.minimum_venues
        or sum(c >= policy.minimum_races_per_venue for c in venue_counts) < policy.minimum_venues
    ):
        reasons.append("venue_coverage")
    incumbent_loss = incumbent["mean_ordered_finish_nll"]
    if incumbent_loss == 0.0:
        reduction = 0.0
        reasons.append("practical_loss")
    else:
        reduction = (incumbent_loss - candidate["mean_ordered_finish_nll"]) / incumbent_loss
        if reduction < policy.practical_loss_reduction:
            reasons.append("practical_loss")
    if report["bootstrap"][challenger]["probability_superior"] < policy.superiority_probability:
        reasons.append("bootstrap_inconclusive")
    if (
        max(candidate["win_calibration"]["ece"], candidate["top3_calibration"]["ece"])
        > policy.maximum_calibration_ece
    ):
        reasons.append("calibration")
    if candidate["coverage"] + policy.maximum_coverage_drop < incumbent["coverage"]:
        reasons.append("coverage")
    for kind in candidate["slices"]:
        for key in set(candidate["slices"][kind]) & set(incumbent["slices"][kind]):
            if (
                candidate["slices"][kind][key]["mean_ordered_finish_nll"]
                > incumbent["slices"][kind][key]["mean_ordered_finish_nll"]
                + policy.maximum_slice_loss_increase
            ):
                reasons.append(f"slice:{kind}:{key}")
    if (
        short[challenger]["mean_ordered_finish_nll"]
        > short[champion]["mean_ordered_finish_nll"] + policy.maximum_short_loss_increase
    ):
        reasons.append("short_horizon_reversal")
    if diagnose_drift(report)["diagnosis"] == "data_domain_drift":
        reasons.append("data_domain_drift")
    return {
        "promote": not reasons,
        "decision": "promote" if not reasons else "retain_incumbent",
        "reasons": sorted(set(reasons)),
        "practical_loss_reduction": reduction,
    }


def _validate_report(report: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "policy",
        "policy_checksum",
        "champion_bundle_id",
        "challenger_bundle_ids",
        "bundle_checksums",
        "population",
        "population_checksum",
        "eligible_population",
        "eligible_population_checksum",
        "long_horizon",
        "short_horizon",
        "racing_day_views",
        "bootstrap",
        "wagering_scorecard",
        "registry_authentication",
        "champion_assignment_id",
        "evaluation_run_id",
        "drift_diagnosis",
    }
    if (
        type(report) is not dict
        or set(report) != expected
        or report.get("schema_version") != "phase6-evaluation-v1"
    ):
        raise EvaluationRejected("sealed report schema is incomplete or unknown")
    try:
        json.dumps(report, allow_nan=False)
        policy = PromotionPolicy(**report["policy"])
    except (TypeError, ValueError, KeyError) as error:
        raise EvaluationRejected("sealed report contains malformed or non-finite values") from error
    if report["policy_checksum"] != str(_digest(asdict(policy))):
        raise EvaluationRejected("sealed report policy checksum disagrees")
    identities = [report["champion_bundle_id"], *report["challenger_bundle_ids"]]
    if (
        len(identities) < 2
        or len(set(identities)) != len(identities)
        or set(report["bundle_checksums"]) != set(identities)
    ):
        raise EvaluationRejected("sealed report model identities disagree")
    population = report["population"]
    eligible = report["eligible_population"]
    if report["population_checksum"] != str(_digest(population)) or report[
        "eligible_population_checksum"
    ] != str(
        _digest(
            sorted(
                (item["race_id"], (item["evidence_checksum"], item["result_checksum"]))
                for item in eligible
            )
        )
    ):
        raise EvaluationRejected("sealed report population checksum disagrees")
    for horizon in ("long_horizon", "short_horizon"):
        if set(report[horizon]) != set(identities):
            raise EvaluationRejected("sealed report horizon identities disagree")
        for score in report[horizon].values():
            _validate_score(score, include_population=True)
    long_count = len(population)
    short_count = min(long_count, policy.short_horizon_races)
    if any(score["race_count"] != long_count for score in report["long_horizon"].values()) or any(
        score["race_count"] != short_count for score in report["short_horizon"].values()
    ):
        raise EvaluationRejected("horizon population is incoherent")
    day_total = 0
    for day, scores in report["racing_day_views"].items():
        try:
            date.fromisoformat(day)
        except (TypeError, ValueError) as error:
            raise EvaluationRejected("Racing Day view identity is invalid") from error
        if set(scores) != set(identities):
            raise EvaluationRejected("Racing Day model identities disagree")
        counts = set()
        for score in scores.values():
            _validate_score(score, include_population=True)
            counts.add(score["race_count"])
        if len(counts) != 1:
            raise EvaluationRejected("Racing Day paired population disagrees")
        day_total += next(iter(counts))
    if day_total != short_count:
        raise EvaluationRejected("Racing Day views do not partition the short horizon")
    for challenger in report["challenger_bundle_ids"]:
        bootstrap = report["bootstrap"].get(challenger)
        if (
            bootstrap is None
            or set(bootstrap)
            != {
                "unit",
                "samples",
                "seed",
                "challenger_minus_champion_mean",
                "lower_95",
                "upper_95",
                "probability_superior",
            }
            or bootstrap.get("unit") != "race"
            or bootstrap.get("samples") != policy.bootstrap_samples
            or bootstrap.get("seed") != policy.bootstrap_seed
        ):
            raise EvaluationRejected("bootstrap evidence configuration disagrees")
        if not (
            bootstrap["lower_95"]
            <= bootstrap["challenger_minus_champion_mean"]
            <= bootstrap["upper_95"]
            and 0 <= bootstrap["probability_superior"] <= 1
        ):
            raise EvaluationRejected("bootstrap evidence is internally inconsistent")
    wagering = report["wagering_scorecard"]
    if wagering.get("status") == "NOT_RUN":
        if set(wagering) != {"status", "report_only", "real_betting", "reason"}:
            raise EvaluationRejected("wagering absence schema disagrees")
    elif wagering.get("status") == "RUN":
        if (
            set(wagering)
            != {
                "status",
                "report_only",
                "real_betting",
                "sealed_odds_races",
                "odds_checksums",
                "models",
            }
            or wagering["sealed_odds_races"] != long_count
            or set(wagering["models"]) != set(identities)
        ):
            raise EvaluationRejected("wagering scorecard population disagrees")
    else:
        raise EvaluationRejected("wagering scorecard status is invalid")
    if wagering["report_only"] is not True or wagering["real_betting"] is not False:
        raise EvaluationRejected("wagering scorecard crossed the report-only boundary")
    if report["drift_diagnosis"] != diagnose_drift(report):
        raise EvaluationRejected("drift diagnosis disagrees with report evidence")


def _validate_score(score: Mapping[str, Any], *, include_population: bool) -> None:
    base = {
        "race_count",
        "mean_ordered_finish_nll",
        "race_losses",
        "winner_mean_reciprocal_rank",
        "top3_containment",
        "exact_top2_order_accuracy",
        "exact_top3_order_accuracy",
        "win_calibration",
        "top3_calibration",
    }
    expected = base | (
        {"coverage", "abstention", "coverage_denominator", "slices"}
        if include_population
        else set()
    )
    if type(score) is not dict or set(score) != expected:
        raise EvaluationRejected("metric schema is incomplete or unknown")
    count = score["race_count"]
    losses = score["race_losses"]
    if type(count) is not int or count <= 0 or type(losses) is not list or len(losses) != count:
        raise EvaluationRejected("metric population is incoherent")
    if not math.isclose(score["mean_ordered_finish_nll"], math.fsum(losses) / count):
        raise EvaluationRejected("primary loss does not derive from race losses")
    for name in (
        "winner_mean_reciprocal_rank",
        "top3_containment",
        "exact_top2_order_accuracy",
        "exact_top3_order_accuracy",
    ):
        if not 0 <= score[name] <= 1:
            raise EvaluationRejected("metric probability is outside its domain")
    for calibration in (score["win_calibration"], score["top3_calibration"]):
        if set(calibration) != {"ece", "bins"} or len(calibration["bins"]) != 10:
            raise EvaluationRejected("calibration schema is incomplete")
        weighted, observations = 0.0, 0
        for index, bin_ in enumerate(calibration["bins"]):
            if (
                set(bin_) != {"lower", "upper", "count", "mean_probability", "observed_rate"}
                or not math.isclose(bin_["lower"], index / 10)
                or not math.isclose(bin_["upper"], (index + 1) / 10)
            ):
                raise EvaluationRejected("calibration bin boundary disagrees")
            if type(bin_["count"]) is not int or bin_["count"] < 0:
                raise EvaluationRejected("calibration bin count is invalid")
            if bin_["count"] == 0:
                if bin_["mean_probability"] is not None or bin_["observed_rate"] is not None:
                    raise EvaluationRejected("empty calibration bin contains observations")
            else:
                if not 0 <= bin_["mean_probability"] <= 1 or not 0 <= bin_["observed_rate"] <= 1:
                    raise EvaluationRejected("calibration observation is invalid")
                weighted += bin_["count"] * abs(bin_["mean_probability"] - bin_["observed_rate"])
                observations += bin_["count"]
        if observations < count or not math.isclose(calibration["ece"], weighted / observations):
            raise EvaluationRejected("calibration denominator or ECE disagrees")
    if include_population:
        if (
            score["coverage_denominator"] < count
            or not math.isclose(score["coverage"], count / score["coverage_denominator"])
            or not math.isclose(score["abstention"], 1 - score["coverage"])
        ):
            raise EvaluationRejected("coverage or abstention denominator disagrees")
        if set(score["slices"]) != {"venue", "distance", "grade", "field_size"}:
            raise EvaluationRejected("required metric slices are incomplete")
        for slices in score["slices"].values():
            if sum(item["race_count"] for item in slices.values()) != count:
                raise EvaluationRejected("slice population does not partition the horizon")
            for item in slices.values():
                _validate_score(item, include_population=False)


class PromotionAuthority:
    """Persists evidence and atomically assigns an already-registered bundle next day."""

    def __init__(self, store: SQLiteOperationsStore, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts

    def promote(
        self,
        operation_id: OperationId,
        *,
        evidence_id: str,
        report: Mapping[str, Any],
        challenger_bundle_id: str,
        assignment_id: str,
        promotion_record_id: str,
        approved_at: datetime,
        effective_racing_day: str,
        approver: str,
        reason: str,
        probation_id: str,
        promotion_run_id: OperationId,
        approval_racing_day_id: str,
    ) -> ServingAssignment:
        require_aware(approved_at, "approved_at")
        try:
            _validate_report(report)
        except EvaluationRejected as error:
            raise PromotionRejected(str(error)) from error
        decision = promotion_decision(report, challenger_bundle_id)
        if not decision["promote"]:
            raise PromotionRejected("promotion gates failed: " + ",".join(decision["reasons"]))
        content = json.dumps(
            report, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        artifact_checksum = _digest(report)
        payload = {
            "promotion_record_id": promotion_record_id,
            "evidence_id": evidence_id,
            "challenger": challenger_bundle_id,
            "assignment_id": assignment_id,
            "approved_at": iso_timestamp(approved_at),
            "effective_racing_day": effective_racing_day,
            "approver": approver,
            "reason": reason,
            "probation_id": probation_id,
            "artifact_checksum": str(artifact_checksum),
            "promotion_run_id": str(promotion_run_id),
            "approval_racing_day_id": approval_racing_day_id,
        }
        with self.store._operation(operation_id, "phase6_promote_next_day", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT next_assignment_id FROM phase6_promotion_records WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or row[0] != assignment_id:
                    raise PromotionRejected("promotion replay lacks exact durable result")
            else:
                evidence_row = db.execute(
                    "SELECT * FROM phase6_evaluation_evidence WHERE evidence_id=? "
                    "AND champion_bundle_id=? AND challenger_bundle_id=?",
                    (evidence_id, report["champion_bundle_id"], challenger_bundle_id),
                ).fetchone()
                trusted = db.execute(
                    "SELECT * FROM phase6_trusted_evaluations WHERE evidence_id=?",
                    (evidence_id,),
                ).fetchone()
                promotion_run = db.execute(
                    "SELECT * FROM phase6_runs WHERE run_id=? AND run_kind='promotion'",
                    (str(promotion_run_id),),
                ).fetchone()
                if (
                    evidence_row is None
                    or trusted is None
                    or promotion_run is None
                    or str(promotion_run_id) == trusted["evaluation_run_id"]
                    or evidence_row["artifact_checksum"] != str(artifact_checksum)
                    or self.artifacts.read(artifact_checksum) != content
                ):
                    raise PromotionRejected(
                        "promotion report is not exact sealed evaluation evidence"
                    )
                policy_row = db.execute(
                    "SELECT * FROM phase6_policy_registry WHERE policy_checksum=?",
                    (trusted["policy_checksum"],),
                ).fetchone()
                policy_content = json.dumps(
                    report["policy"], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
                if (
                    policy_row is None
                    or policy_row["policy_id"] != report["policy"]["policy_id"]
                    or policy_row["artifact_checksum"] != trusted["policy_checksum"]
                    or self.artifacts.read(ArtifactChecksum(policy_row["artifact_checksum"]))
                    != policy_content
                ):
                    raise PromotionRejected("sealed policy artifact is unavailable or corrupt")
                pointer = db.execute("SELECT * FROM champion_pointer WHERE singleton=1").fetchone()
                bundle = db.execute(
                    "SELECT * FROM canonical_model_bundles WHERE bundle_id=?",
                    (challenger_bundle_id,),
                ).fetchone()
                components = db.execute(
                    "SELECT component_kind,artifact_checksum FROM canonical_bundle_components WHERE bundle_id=? ORDER BY component_kind",
                    (challenger_bundle_id,),
                ).fetchall()
                probation = db.execute(
                    "SELECT * FROM phase6_probation_states WHERE probation_id=?", (probation_id,)
                ).fetchone()
                if pointer is None or bundle is None or len(components) != len(COMPONENT_KINDS):
                    raise PromotionRejected(
                        "champion or complete registered challenger is unavailable"
                    )
                if (
                    pointer["assignment_id"] != trusted["champion_assignment_id"]
                    or pointer["bundle_id"] != report["champion_bundle_id"]
                    or bundle["created_at"] != trusted["challenger_registered_at"]
                    or bundle["created_at"]
                    >= db.execute(
                        "SELECT started_at FROM phase6_runs WHERE run_id=?",
                        (trusted["evaluation_run_id"],),
                    ).fetchone()[0]
                ):
                    raise PromotionRejected(
                        "champion pointer drift or post-hoc challenger registration"
                    )
                run_provenance = db.execute(
                    "SELECT lineage.*,registration.started_at AS registration_started_at,"
                    "source.started_at AS source_started_at FROM phase6_bundle_lineage_v2 lineage "
                    "JOIN phase6_runs registration ON registration.run_id=lineage.registration_run_id "
                    "JOIN phase6_runs source ON source.run_id=lineage.source_run_id "
                    "WHERE bundle_id=?",
                    (challenger_bundle_id,),
                ).fetchone()
                if (
                    run_provenance is None
                    or run_provenance["bundle_registration_operation_id"] != bundle["operation_id"]
                    or len(
                        {
                            run_provenance["registration_run_id"],
                            run_provenance["source_run_id"],
                            trusted["evaluation_run_id"],
                            str(promotion_run_id),
                        }
                    )
                    != 4
                    or not (
                        run_provenance["source_started_at"]
                        < run_provenance["registration_started_at"]
                        <= bundle["created_at"]
                        < db.execute(
                            "SELECT started_at FROM phase6_runs WHERE run_id=?",
                            (trusted["evaluation_run_id"],),
                        ).fetchone()[0]
                        <= trusted["sealed_at"]
                        < promotion_run["started_at"]
                        <= iso_timestamp(approved_at)
                    )
                ):
                    raise PromotionRejected(
                        "candidate training, registration, evaluation, and promotion runs overlap"
                    )
                successors = db.execute(
                    "SELECT s.racing_day_id,d.local_date FROM phase6_racing_day_schedule s "
                    "JOIN racing_days d USING(racing_day_id) WHERE s.predecessor_racing_day_id=?",
                    (approval_racing_day_id,),
                ).fetchall()
                if len(successors) != 1:
                    raise PromotionRejected("next durable Racing Day is missing or ambiguous")
                successor = successors[0]
                if successor["local_date"] != effective_racing_day:
                    raise PromotionRejected("effective date is not the next durable Racing Day")
                if approved_at.date() >= date.fromisoformat(effective_racing_day):
                    raise PromotionRejected("approval is backdated or post-effective")
                probation_days = db.execute(
                    "SELECT * FROM phase6_probation_days WHERE probation_id=? ORDER BY racing_day",
                    (probation_id,),
                ).fetchall()
                authenticated_days = db.execute(
                    "SELECT a.racing_day_id,a.programme_checksum,d.local_date "
                    "FROM phase6_probation_day_auth a JOIN racing_days d USING(racing_day_id) "
                    "WHERE a.probation_id=? ORDER BY d.local_date",
                    (probation_id,),
                ).fetchall()
                chain = []
                cursor = approval_racing_day_id
                for _ in range(14):
                    scheduled = db.execute(
                        "SELECT predecessor_racing_day_id,programme_checksum FROM phase6_racing_day_schedule WHERE racing_day_id=?",
                        (cursor,),
                    ).fetchone()
                    if scheduled is None:
                        break
                    chain.append((cursor, scheduled["programme_checksum"]))
                    cursor = scheduled["predecessor_racing_day_id"]
                expected_days = list(reversed(chain))
                if (
                    probation is None
                    or len(expected_days) != 14
                    or sorted(
                        (row["racing_day_id"], row["programme_checksum"])
                        for row in authenticated_days
                    )
                    != sorted(expected_days)
                ):
                    raise PromotionRejected(
                        "independently durable fourteen-day probation is incomplete"
                    )
                phase7_seals = db.execute(
                    "SELECT * FROM phase7_probation_seals WHERE probation_id=?",
                    (probation_id,),
                ).fetchall()
                if len(phase7_seals) != 1:
                    raise PromotionRejected("exactly one Phase 7 probation seal is required")
                phase7_seal = phase7_seals[0]
                control = db.execute(
                    "SELECT * FROM phase7_probation_control WHERE singleton=1"
                ).fetchone()
                release_pointer = db.execute(
                    "SELECT * FROM phase7_release_pointer WHERE singleton=1 "
                    "AND authority='race_collection_service'"
                ).fetchone()
                activation = db.execute(
                    "SELECT * FROM phase7_release_history WHERE operation_id=? "
                    "AND action='activate'",
                    (phase7_seal["cutover_operation_id"],),
                ).fetchone()
                acceptances = db.execute(
                    "SELECT a.*,e.release_id,e.reconciliation_checksum,e.restart_checksum,"
                    "e.ordering_checksum,e.determinism_checksum,e.complete,e.critical_failure "
                    "FROM phase7_probation_acceptances a "
                    "JOIN phase7_day_evidence e USING(racing_day_id) "
                    "WHERE a.generation=? ORDER BY a.local_date",
                    (phase7_seal["generation"],),
                ).fetchall()
                if (
                    control is None
                    or control["state"] != "complete"
                    or control["generation"] != phase7_seal["generation"]
                    or release_pointer is None
                    or release_pointer["release_id"] != phase7_seal["release_id"]
                    or activation is None
                    or activation["release_id"] != phase7_seal["release_id"]
                    or activation["effective_racing_day_id"]
                    != release_pointer["effective_racing_day_id"]
                    or probation["state_checksum"] != phase7_seal["state_checksum"]
                    or phase7_seal["sealed_at"] > iso_timestamp(approved_at)
                    or len(acceptances) != 14
                    or any(
                        row["release_id"] != phase7_seal["release_id"]
                        or row["complete"] != 1
                        or row["critical_failure"] != 0
                        for row in acceptances
                    )
                    or acceptances[-1]["racing_day_id"] != approval_racing_day_id
                ):
                    raise PromotionRejected("Phase 7 probation seal is stale, mixed, or misaligned")
                predecessor = activation["effective_racing_day_id"]
                for acceptance in acceptances:
                    scheduled = db.execute(
                        "SELECT predecessor_racing_day_id FROM phase6_racing_day_schedule "
                        "WHERE racing_day_id=?",
                        (acceptance["racing_day_id"],),
                    ).fetchone()
                    if scheduled is None or scheduled[0] != predecessor:
                        raise PromotionRejected("Phase 7 probation schedule chain is invalid")
                    predecessor = acceptance["racing_day_id"]
                try:
                    verify_release_authority(db, self.artifacts, phase7_seal["release_id"])
                except (OperationalRejected, ArtifactStoreError, ValueError) as error:
                    raise PromotionRejected("Phase 7 release authority is unavailable") from error
                operational_authority = OperationalAuthority(self.store, self.artifacts)
                try:
                    for acceptance in acceptances:
                        self.artifacts.verify(ArtifactChecksum(acceptance["programme_checksum"]))
                        reconciliation = db.execute(
                            "SELECT report_checksum,report_json FROM phase7_reconciliation "
                            "WHERE racing_day_id=?",
                            (acceptance["racing_day_id"],),
                        ).fetchone()
                        reconciliation_checksum = ArtifactChecksum(
                            acceptance["reconciliation_checksum"]
                        )
                        if (
                            reconciliation is None
                            or reconciliation["report_checksum"] != str(reconciliation_checksum)
                            or self.artifacts.read(reconciliation_checksum)
                            != json.dumps(
                                json.loads(reconciliation["report_json"]),
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            ).encode()
                        ):
                            raise OperationalRejected("reconciliation report authority is invalid")
                        for kind, column in (
                            ("restart", "restart_checksum"),
                            ("ordering", "ordering_checksum"),
                            ("determinism", "determinism_checksum"),
                        ):
                            operational_authority.verify_operational_evidence(
                                db,
                                racing_day_id=acceptance["racing_day_id"],
                                release_id=acceptance["release_id"],
                                evidence_kind=kind,
                                checksum=ArtifactChecksum(acceptance[column]),
                            )
                except (OperationalRejected, ArtifactStoreError, ValueError) as error:
                    raise PromotionRejected(
                        "Phase 7 acceptance evidence is unavailable or corrupt"
                    ) from error
                manifest_content = self.artifacts.read(
                    ArtifactChecksum(probation["state_checksum"])
                )
                try:
                    manifest = json.loads(manifest_content)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PromotionRejected("probation manifest is invalid") from error
                durable_days = [
                    {
                        "racing_day": row["racing_day"],
                        "reconciliation_checksum": row["reconciliation_checksum"],
                        "restart_checksum": row["restart_checksum"],
                        "ordering_checksum": row["ordering_checksum"],
                        "determinism_checksum": row["determinism_checksum"],
                    }
                    for row in probation_days
                ]
                phase7_projection = [
                    (
                        row["local_date"],
                        row["racing_day_id"],
                        row["programme_checksum"],
                        row["reconciliation_checksum"],
                        row["restart_checksum"],
                        row["ordering_checksum"],
                        row["determinism_checksum"],
                    )
                    for row in acceptances
                ]
                phase6_days_by_date = {row["racing_day"]: row for row in probation_days}
                phase6_auth_by_date = {row["local_date"]: row for row in authenticated_days}
                phase6_projection = []
                for local_date in sorted(phase6_days_by_date):
                    day_row = phase6_days_by_date[local_date]
                    auth_row = phase6_auth_by_date.get(local_date)
                    if auth_row is None:
                        raise PromotionRejected("Phase 6 probation day authentication is omitted")
                    phase6_projection.append(
                        (
                            local_date,
                            auth_row["racing_day_id"],
                            auth_row["programme_checksum"],
                            day_row["reconciliation_checksum"],
                            day_row["restart_checksum"],
                            day_row["ordering_checksum"],
                            day_row["determinism_checksum"],
                        )
                    )
                if (
                    len(phase6_days_by_date) != 14
                    or len(phase6_auth_by_date) != 14
                    or phase6_projection != phase7_projection
                ):
                    raise PromotionRejected(
                        "Phase 7 evidence and Phase 6 probation projection are not one-to-one"
                    )
                if manifest != {
                    "schema_version": "phase7-probation-v1",
                    "probation_id": probation_id,
                    "through_racing_day": probation["through_racing_day"],
                    "days": durable_days,
                }:
                    raise PromotionRejected(
                        "probation manifest disagrees with durable day evidence"
                    )
                approval_day = db.execute(
                    "SELECT local_date FROM racing_days WHERE racing_day_id=?",
                    (approval_racing_day_id,),
                ).fetchone()
                if (
                    approval_day is None
                    or probation["through_racing_day"] != approval_day["local_date"]
                    or len(probation_days) != 14
                ):
                    raise PromotionRejected("probation evidence is stale or misaligned")
                for row in probation_days:
                    for field in (
                        "reconciliation_checksum",
                        "restart_checksum",
                        "ordering_checksum",
                        "determinism_checksum",
                    ):
                        self.artifacts.verify(ArtifactChecksum(row[field]))
                if (
                    report["bundle_checksums"].get(challenger_bundle_id)
                    != bundle["bundle_checksum"]
                ):
                    raise PromotionRejected("challenger checksum disagrees with evidence")
                registrations = {}
                for model_id, claimed_checksum in report["bundle_checksums"].items():
                    registered = db.execute(
                        "SELECT bundle_checksum FROM canonical_model_bundles WHERE bundle_id=?",
                        (model_id,),
                    ).fetchone()
                    if registered is None or registered["bundle_checksum"] != claimed_checksum:
                        raise PromotionRejected("evaluation model registration has changed")
                    registered_components = db.execute(
                        "SELECT artifact_checksum FROM canonical_bundle_components WHERE bundle_id=?",
                        (model_id,),
                    ).fetchall()
                    if len(registered_components) != len(COMPONENT_KINDS):
                        raise PromotionRejected("evaluation model registration is incomplete")
                    self.artifacts.verify(ArtifactChecksum(claimed_checksum))
                    for registered_component in registered_components:
                        self.artifacts.verify(
                            ArtifactChecksum(registered_component["artifact_checksum"])
                        )
                    registrations[model_id] = registered["bundle_checksum"]
                if report.get("registry_authentication") != str(_digest(registrations)):
                    raise PromotionRejected("evaluation lacks exact registry authentication")
                component_json = json.dumps(
                    {r["component_kind"]: r["artifact_checksum"] for r in components},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                # The assignment's operation FK needs a durable operation identity of its own.
                assignment_operation = (
                    "op_"
                    + hashlib.sha256((str(operation_id) + assignment_id).encode()).hexdigest()[:32]
                )
                db.execute(
                    "INSERT INTO operations VALUES(?,?,?,?)",
                    (
                        assignment_operation,
                        "phase6_assignment",
                        hashlib.sha256(assignment_id.encode()).hexdigest(),
                        iso_timestamp(approved_at),
                    ),
                )
                db.execute(
                    "INSERT INTO canonical_serving_assignments VALUES(?,?,?,?,?,?,?,?)",
                    (
                        assignment_id,
                        challenger_bundle_id,
                        bundle["bundle_checksum"],
                        iso_timestamp(approved_at),
                        effective_racing_day,
                        promotion_record_id,
                        iso_timestamp(approved_at),
                        assignment_operation,
                    ),
                )
                db.execute(
                    "INSERT INTO phase6_promotion_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        promotion_record_id,
                        evidence_id,
                        pointer["assignment_id"],
                        assignment_id,
                        challenger_bundle_id,
                        bundle["bundle_checksum"],
                        component_json,
                        iso_timestamp(approved_at),
                        effective_racing_day,
                        approver,
                        report["policy"]["policy_id"],
                        reason,
                        probation_id,
                        str(operation_id),
                    ),
                )
                db.execute(
                    "INSERT INTO phase6_next_day_assignments VALUES(?,?,?,?,?)",
                    (
                        assignment_id,
                        successor["racing_day_id"],
                        pointer["assignment_id"],
                        promotion_record_id,
                        str(operation_id),
                    ),
                )
                db.execute(
                    "INSERT INTO phase6_assignment_history VALUES(?,?,?,?,?,?,?)",
                    (
                        promotion_record_id + ":history",
                        successor["racing_day_id"],
                        assignment_id,
                        "promoted",
                        None,
                        iso_timestamp(approved_at),
                        str(operation_id),
                    ),
                )
        return ServingAssignment(
            assignment_id,
            challenger_bundle_id,
            ArtifactChecksum(report["bundle_checksums"][challenger_bundle_id]),
            iso_timestamp(approved_at),
            effective_racing_day,
            promotion_record_id,
        )

    def rollback_staged(
        self,
        operation_id: OperationId,
        *,
        rollback_id: str,
        staged_assignment_id: str,
        reason: str,
        rolled_back_at: datetime,
    ) -> str:
        payload = {
            "rollback_id": rollback_id,
            "staged_assignment_id": staged_assignment_id,
            "reason": reason,
            "rolled_back_at": iso_timestamp(rolled_back_at),
        }
        with self.store._operation(operation_id, "rollback_phase6_staged_assignment", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT restored_assignment_id FROM phase6_rollback_records WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None:
                    raise PromotionRejected("rollback replay lacks exact durable result")
                return row[0]
            staged = db.execute(
                "SELECT rollback_assignment_id,effective_racing_day_id,promotion_record_id "
                "FROM phase6_next_day_assignments WHERE assignment_id=?",
                (staged_assignment_id,),
            ).fetchone()
            if staged is None:
                raise PromotionRejected("staged assignment is unavailable")
            db.execute(
                "INSERT INTO phase6_rollback_records VALUES(?,?,?,?,?,?)",
                (
                    rollback_id,
                    staged_assignment_id,
                    staged["rollback_assignment_id"],
                    reason,
                    iso_timestamp(rolled_back_at),
                    str(operation_id),
                ),
            )
            prior = db.execute(
                "SELECT history_id FROM phase6_assignment_history WHERE assignment_id=? "
                "AND effective_racing_day_id=? ORDER BY rowid DESC LIMIT 1",
                (staged_assignment_id, staged["effective_racing_day_id"]),
            ).fetchone()
            db.execute(
                "INSERT INTO phase6_assignment_history VALUES(?,?,?,?,?,?,?)",
                (
                    rollback_id + ":history",
                    staged["effective_racing_day_id"],
                    staged["rollback_assignment_id"],
                    "rollback_restored",
                    prior["history_id"] if prior else None,
                    iso_timestamp(rolled_back_at),
                    str(operation_id),
                ),
            )
            return staged["rollback_assignment_id"]

    def resolve_scheduled_assignment(self, racing_day_id: str) -> str:
        with self.store._connect() as db:
            row = db.execute(
                "SELECT assignment_id FROM phase6_assignment_history "
                "WHERE effective_racing_day_id=? ORDER BY rowid DESC LIMIT 1",
                (racing_day_id,),
            ).fetchone()
        if row is None:
            raise PromotionRejected("no authenticated assignment exists for Racing Day")
        return row["assignment_id"]
