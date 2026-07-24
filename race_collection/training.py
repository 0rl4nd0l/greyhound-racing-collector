"""Immutable corpus construction and deterministic Phase-5 challenger training."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import ArtifactStore
from .domain import ArtifactChecksum, OperationId, require_aware
from .features import DerivationResult, derive_features
from .model_bundle import (
    BundleComponent,
    CanonicalBundle,
    SUPPORTED_FEATURE_CONTRACT,
)
from .operations import BarrierNotSatisfied, SQLiteOperationsStore, iso_timestamp
from .ordered_finish import ORDERED_FINISH_CONTRACT, forecast_ordered_finish, ordered_finish_nll


class CorpusRejected(ValueError):
    """Corpus evidence is ambiguous, mutable, leaky, or lacks provenance."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checksum(content: bytes) -> ArtifactChecksum:
    return ArtifactChecksum(f"sha256:{hashlib.sha256(content).hexdigest()}")


@dataclass(frozen=True, slots=True)
class LegacyRow:
    source: str
    source_record_id: str
    dog_id: str
    local_racing_date: str
    observed_at: str
    authoritative: bool
    artifact_checksum: ArtifactChecksum
    values: Mapping[str, Any]


def audit_legacy_corpus(rows: Iterable[LegacyRow]) -> Mapping[str, Any]:
    """Deduplicate Dog Runs deterministically while retaining every observation."""
    ordered = sorted(
        rows,
        key=lambda row: (
            row.dog_id,
            row.local_racing_date,
            not row.authoritative,
            row.source,
            row.source_record_id,
            str(row.artifact_checksum),
        ),
    )
    observations: list[dict[str, Any]] = []
    by_run: dict[tuple[str, str], list[LegacyRow]] = {}
    seen_observations: set[tuple[str, str, str]] = set()
    for row in ordered:
        date.fromisoformat(row.local_racing_date)
        require_aware(datetime.fromisoformat(row.observed_at), "legacy observed_at")
        if not all((row.source, row.source_record_id, row.dog_id)):
            raise CorpusRejected("legacy observation identity is incomplete")
        observation_key = (row.source, row.source_record_id, str(row.artifact_checksum))
        if observation_key in seen_observations:
            continue
        seen_observations.add(observation_key)
        by_run.setdefault((row.dog_id, row.local_racing_date), []).append(row)
        observations.append(
            {
                "source": row.source,
                "source_record_id": row.source_record_id,
                "dog_id": row.dog_id,
                "local_racing_date": row.local_racing_date,
                "authoritative": row.authoritative,
                "artifact_checksum": str(row.artifact_checksum),
                "observed_at": row.observed_at,
            }
        )
    dog_runs: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for (dog_id, racing_date), candidates in sorted(by_run.items()):
        authoritative = [row for row in candidates if row.authoritative]
        preferred = authoritative or candidates
        value_encodings = {canonical_json(row.values) for row in preferred}
        if len(value_encodings) > 1:
            conflicts.append(
                {"dog_id": dog_id, "local_racing_date": racing_date, "reason": "source_conflict"}
            )
            continue
        selected = preferred[0]
        dog_runs.append(
            {
                "dog_run_id": f"{dog_id}:{racing_date}",
                "dog_id": dog_id,
                "local_racing_date": racing_date,
                "authoritative": bool(authoritative),
                "selected_observation": {
                    "source": selected.source,
                    "source_record_id": selected.source_record_id,
                    "artifact_checksum": str(selected.artifact_checksum),
                },
                "superseded_observations": [
                    {
                        "source": row.source,
                        "source_record_id": row.source_record_id,
                        "artifact_checksum": str(row.artifact_checksum),
                    }
                    for row in candidates
                    if row is not selected
                ],
                "values": selected.values,
            }
        )
    manifest = {
        "schema_version": "legacy-training-corpus-audit-v1",
        "promotion_grade": False,
        "observations": observations,
        "dog_runs": dog_runs,
        "conflicts": conflicts,
        "counts": {
            "input_rows": len(ordered),
            "unique_observations": len(observations),
            "deduplicated_dog_runs": len(dog_runs),
            "quarantined_conflicts": len(conflicts),
        },
    }
    return {**manifest, "audit_checksum": str(checksum(canonical_json(manifest)))}


def validate_evaluation_outcome(
    outcome: Mapping[str, Any],
    *,
    evidence_frozen_at: datetime,
    joined_at: datetime,
    runner_ids: Sequence[str],
    box_by_runner: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return one official total order or fail with an explicit exclusion."""
    if type(outcome) is not dict:
        raise CorpusRejected("official result envelope must be an object")
    exclusions = outcome.get("exclusions")
    if type(exclusions) is not list or any(
        type(item) is not str or not item for item in exclusions
    ):
        raise CorpusRejected("official result exclusions envelope is malformed")
    if exclusions or outcome.get("official") is not True:
        raise CorpusRejected("official result is not evaluation eligible")
    order = outcome.get("order")
    provenance = outcome.get("provenance")
    required_provenance = {"source", "source_record_id", "observed_at"}
    if (
        type(order) is not list
        or type(provenance) is not dict
        or not required_provenance <= set(provenance)
        or any(
            type(provenance[key]) is not str or not provenance[key] for key in required_provenance
        )
    ):
        raise CorpusRejected("official result order/provenance is incomplete")
    if any(type(box) is not int or box <= 0 for box in order) or len(set(order)) != len(order):
        raise CorpusRejected("official result order is ambiguous")
    if type(outcome.get("published_at")) is not str:
        raise CorpusRejected("official result publication timestamp is invalid")
    try:
        observed_at = datetime.fromisoformat(provenance["observed_at"])
        published_at = datetime.fromisoformat(outcome["published_at"])
        require_aware(evidence_frozen_at, "evidence frozen_at")
        require_aware(joined_at, "joined_at")
        require_aware(observed_at, "result observed_at")
        require_aware(published_at, "result published_at")
    except (TypeError, ValueError) as error:
        raise CorpusRejected("official result timestamp is invalid") from error
    if (
        published_at <= evidence_frozen_at
        or observed_at < published_at
        or observed_at > joined_at
        or joined_at < published_at
    ):
        raise CorpusRejected("result/evidence temporal order is invalid")
    runners = tuple(runner_ids)
    if (
        type(box_by_runner) is not dict
        or set(box_by_runner) != set(runners)
        or any(type(box) is not int or box <= 0 for box in box_by_runner.values())
        or len(set(box_by_runner.values())) != len(runners)
    ):
        raise CorpusRejected("sealed evidence box mapping is incomplete or ambiguous")
    runner_by_box = {box: runner for runner, box in box_by_runner.items()}
    if set(order) != set(runner_by_box):
        raise CorpusRejected("official box order and sealed runner set disagree")
    return tuple(runner_by_box[box] for box in order)


_RESULT_DERIVED_KEYS = {"result_order", "finish_order", "winner", "result", "place"}


def _reject_result_derived_evidence(value: Any) -> None:
    if type(value) is dict:
        for key, nested in value.items():
            if type(key) is str and key.casefold() in _RESULT_DERIVED_KEYS:
                raise CorpusRejected("sealed evidence contains a post-result feature")
            _reject_result_derived_evidence(nested)
    elif type(value) is list:
        for nested in value:
            _reject_result_derived_evidence(nested)


def _authoritative_box_mapping(evidence: Mapping[str, Any], fields: Mapping[str, Any]) -> Any:
    box_mapping = fields.get("box")
    provenance = evidence.get("field_provenance")

    def is_authoritative(item: Any) -> bool:
        if not (
            type(item) is dict
            and item.get("field") == "box"
            and item.get("authority") in {"official_programme", "official_card"}
            and item.get("value") == box_mapping
            and type(item.get("source")) is str
            and bool(item["source"])
            and type(item.get("artifact_checksum")) is str
        ):
            return False
        try:
            ArtifactChecksum(item["artifact_checksum"])
        except ValueError:
            return False
        return True

    if type(provenance) is not list or not any(is_authoritative(item) for item in provenance):
        raise CorpusRejected("sealed evidence box mapping lacks authoritative provenance")
    return box_mapping


@dataclass(frozen=True, slots=True)
class TrainingExample:
    example_id: str
    race_id: str
    racing_date: str
    evidence_checksum: ArtifactChecksum
    result_checksum: ArtifactChecksum
    runner_ids: tuple[str, ...]
    official_order: tuple[str, ...]
    feature_matrix_checksum: ArtifactChecksum
    artifact_checksum: ArtifactChecksum


class TrainingCorpusAuthority:
    def __init__(self, store: SQLiteOperationsStore, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts

    def build_forward_example(
        self,
        operation_id: OperationId,
        *,
        phase3_example_id: str,
        example_id: str,
        schema_bytes: bytes,
        schema_checksum: ArtifactChecksum,
        missingness_bytes: bytes,
        missingness_checksum: ArtifactChecksum,
        joined_at: datetime,
    ) -> TrainingExample:
        require_aware(joined_at, "joined_at")
        with self.store._connect() as db:
            row = db.execute(
                "SELECT t.*,s.normalized_checksum,s.frozen_at,a.artifact_checksum AS result_checksum,"
                "a.outcome_json,r.racing_day_id,d.local_date "
                "FROM training_examples t JOIN deferred_predictions p "
                "ON p.prediction_id=t.prediction_id AND p.race_id=t.race_id "
                "JOIN sealed_evidence s ON s.seal_id=p.seal_id AND s.race_id=t.race_id "
                "JOIN result_attempts a ON a.attempt_id=t.result_attempt_id "
                "AND a.race_id=t.race_id AND a.status='collected' "
                "JOIN races r USING(race_id) JOIN racing_days d USING(racing_day_id) "
                "WHERE t.training_example_id=? AND t.eligibility='eligible' AND a.status='collected'",
                (phase3_example_id,),
            ).fetchone()
        if row is None:
            raise BarrierNotSatisfied("forward example requires one eligible Phase-3 join")
        evidence_checksum = ArtifactChecksum(row["normalized_checksum"])
        result_checksum = ArtifactChecksum(row["result_checksum"])
        evidence = self.artifacts.read(evidence_checksum)
        result_bytes = self.artifacts.read(result_checksum)
        try:
            outcome = json.loads(row["outcome_json"])
            result_document = json.loads(result_bytes)
            evidence_document = json.loads(evidence)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CorpusRejected("forward example artifacts must be JSON") from error
        if result_document != outcome:
            raise CorpusRejected("official result artifact disagrees with collected outcome")
        if type(evidence_document) is not dict:
            raise CorpusRejected("sealed evidence envelope must be an object")
        freeze_at = datetime.fromisoformat(row["frozen_at"])
        fields = evidence_document.get("fields", {})
        _reject_result_derived_evidence(fields)
        derived = derive_features(
            evidence,
            expected_evidence_checksum=evidence_checksum,
            schema_bytes=schema_bytes,
            expected_schema_checksum=schema_checksum,
            missingness_policy_bytes=missingness_bytes,
            expected_missingness_checksum=missingness_checksum,
        )
        order = validate_evaluation_outcome(
            outcome,
            evidence_frozen_at=freeze_at,
            joined_at=joined_at,
            runner_ids=derived.matrix.runner_ids,
            box_by_runner=(
                _authoritative_box_mapping(evidence_document, fields)
                if type(fields) is dict
                else None
            ),
        )
        document = {
            "schema_version": "canonical-training-example-v1",
            "origin": "forward-sealed",
            "promotion_evidence_eligible": True,
            "example_id": example_id,
            "phase3_example_id": phase3_example_id,
            "race_id": row["race_id"],
            "racing_date": row["local_date"],
            "evidence_checksum": str(evidence_checksum),
            "result_checksum": str(result_checksum),
            "feature_matrix_checksum": str(derived.matrix.checksum),
            "runner_ids": list(derived.matrix.runner_ids),
            "official_order": list(order),
            "evidence_frozen_at": row["frozen_at"],
            "result_published_at": outcome["published_at"],
        }
        content = canonical_json(document)
        artifact = self.artifacts.put(content, media_type="application/json")
        payload = {"example": document, "artifact_checksum": str(artifact.checksum)}
        with self.store._operation(operation_id, "register_forward_training_example", payload) as (
            db,
            replay,
        ):
            expected = (
                example_id,
                phase3_example_id,
                row["race_id"],
                str(evidence_checksum),
                str(result_checksum),
                str(derived.matrix.checksum),
                str(artifact.checksum),
                row["local_date"],
                iso_timestamp(joined_at),
                str(operation_id),
            )
            if replay:
                durable = db.execute(
                    "SELECT * FROM canonical_training_examples WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if durable is None or tuple(durable) != expected:
                    raise CorpusRejected("forward example replay conflicts with durable identity")
            else:
                db.execute(
                    "INSERT INTO canonical_training_examples VALUES(?,?,?,?,?,?,?,?,?,?)", expected
                )
        return TrainingExample(
            example_id,
            row["race_id"],
            row["local_date"],
            evidence_checksum,
            result_checksum,
            tuple(derived.matrix.runner_ids),
            order,
            derived.matrix.checksum,
            artifact.checksum,
        )


@dataclass(frozen=True, slots=True)
class LinearStrengthModel:
    coefficients: tuple[float, ...]
    intercept: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.coefficients) is not tuple
            or not self.coefficients
            or any(
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in self.coefficients
            )
            or type(self.intercept) not in (int, float)
            or isinstance(self.intercept, bool)
            or not math.isfinite(self.intercept)
        ):
            raise CorpusRejected("linear strength parameters must be nonempty and finite")

    def latent_strengths(self, rows: Sequence[Sequence[float]]) -> list[float]:
        strengths = []
        for row in rows:
            if type(row) not in (list, tuple) or len(row) != len(self.coefficients):
                raise CorpusRejected("feature row width disagrees with model dimensions")
            if any(
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in row
            ):
                raise CorpusRejected("feature row values must be finite numeric values")
            strength = self.intercept + math.fsum(
                coefficient * value for coefficient, value in zip(self.coefficients, row)
            )
            if not math.isfinite(strength):
                raise CorpusRejected("latent strength is not finite")
            strengths.append(strength)
        return strengths

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[list[float]]:
        # Phase-4's loader capability probe remains supported; canonical serving uses latent_strengths.
        return [
            [1.0 - p, p]
            for p in (
                1.0 / (1.0 + math.exp(-max(min(s, 700), -700))) for s in self.latent_strengths(rows)
            )
        ]

    def to_bytes(self) -> bytes:
        return canonical_json(
            {
                "schema_version": "linear-runner-strength-v1",
                "coefficients": self.coefficients,
                "intercept": self.intercept,
            }
        )

    @classmethod
    def from_bytes(cls, content: bytes) -> "LinearStrengthModel":
        if type(content) is not bytes:
            raise CorpusRejected("model artifact must be exact bytes")
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CorpusRejected("model artifact is not valid JSON") from error
        if (
            type(value) is not dict
            or set(value) != {"schema_version", "coefficients", "intercept"}
            or value.get("schema_version") != "linear-runner-strength-v1"
            or type(value.get("coefficients")) is not list
            or not value["coefficients"]
        ):
            raise CorpusRejected("model artifact envelope is invalid")
        return cls(tuple(value["coefficients"]), value["intercept"])


@dataclass(frozen=True, slots=True)
class PreparedExample:
    identity: TrainingExample
    features: DerivationResult


def prepare_training_example(
    example: TrainingExample,
    *,
    artifacts: ArtifactStore,
    schema_bytes: bytes,
    schema_checksum: ArtifactChecksum,
    missingness_bytes: bytes,
    missingness_checksum: ArtifactChecksum,
) -> PreparedExample:
    """Replay an immutable example through the exact Phase-4 feature contract."""
    document = json.loads(artifacts.read(example.artifact_checksum))
    expected = {
        "example_id": example.example_id,
        "race_id": example.race_id,
        "racing_date": example.racing_date,
        "evidence_checksum": str(example.evidence_checksum),
        "result_checksum": str(example.result_checksum),
        "feature_matrix_checksum": str(example.feature_matrix_checksum),
        "runner_ids": list(example.runner_ids),
        "official_order": list(example.official_order),
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise CorpusRejected("immutable training example identity disagrees")
    derived = derive_features(
        artifacts.read(example.evidence_checksum),
        expected_evidence_checksum=example.evidence_checksum,
        schema_bytes=schema_bytes,
        expected_schema_checksum=schema_checksum,
        missingness_policy_bytes=missingness_bytes,
        expected_missingness_checksum=missingness_checksum,
    )
    if (
        derived.matrix.checksum != example.feature_matrix_checksum
        or derived.matrix.runner_ids != example.runner_ids
    ):
        raise CorpusRejected("training feature replay disagrees with immutable example")
    return PreparedExample(example, derived)


def fit_linear_strength_model(
    examples: Sequence[PreparedExample],
    *,
    seed: int,
    epochs: int = 200,
    learning_rate: float = 0.02,
) -> LinearStrengthModel:
    """Deterministic full-batch gradient descent on PL ordered-finish likelihood."""
    if not examples:
        raise CorpusRejected("training corpus is empty")
    width = len(examples[0].features.matrix.columns)
    coefficients = np.zeros(width, dtype=float)
    # Seed is recorded and controls a deterministic, tiny symmetry-breaking start.
    coefficients += np.random.default_rng(seed).normal(0.0, 1e-6, size=width)
    for _ in range(epochs):
        gradient = np.zeros(width, dtype=float)
        count = 0
        for example in examples:
            matrix = np.asarray(example.features.matrix.rows, dtype=float)
            runner_index = {
                runner: index for index, runner in enumerate(example.features.matrix.runner_ids)
            }
            remaining = [runner_index[runner] for runner in example.identity.official_order]
            for chosen in tuple(remaining):
                logits = matrix[remaining] @ coefficients
                logits -= np.max(logits)
                probabilities = np.exp(logits) / np.sum(np.exp(logits))
                gradient += matrix[chosen] - probabilities @ matrix[remaining]
                remaining.remove(chosen)
                count += 1
        coefficients += learning_rate * gradient / max(count, 1)
        if not np.all(np.isfinite(coefficients)):
            raise CorpusRejected("training became numerically unstable")
    return LinearStrengthModel(tuple(float(value) for value in coefficients))


def expanding_window_validation(
    examples: Sequence[PreparedExample], *, seed: int, minimum_train: int = 2
) -> Mapping[str, Any]:
    ordered = sorted(
        examples, key=lambda item: (item.identity.racing_date, item.identity.example_id)
    )
    folds: list[dict[str, Any]] = []
    for held_out_date in sorted({item.identity.racing_date for item in ordered}):
        training = [item for item in ordered if item.identity.racing_date < held_out_date]
        if len(training) < minimum_train:
            continue
        model = fit_linear_strength_model(training, seed=seed)
        for held_out in (item for item in ordered if item.identity.racing_date == held_out_date):
            forecast = forecast_ordered_finish(
                held_out.features.matrix.runner_ids,
                model.latent_strengths(held_out.features.matrix.rows),
            )
            folds.append(
                {
                    "trained_through": max(item.identity.racing_date for item in training),
                    "evaluation_racing_date": held_out_date,
                    "evaluation_example_id": held_out.identity.example_id,
                    "ordered_finish_nll": ordered_finish_nll(
                        forecast, held_out.identity.official_order
                    ),
                }
            )
    return {
        "method": "expanding-window-temporal-v1",
        "folds": folds,
        "mean_ordered_finish_nll": (
            math.fsum(fold["ordered_finish_nll"] for fold in folds) / len(folds) if folds else None
        ),
    }


def train_challenger_bundle(
    *,
    examples: Sequence[PreparedExample],
    model_id: str,
    bundle_id: str,
    schema_bytes: bytes,
    missingness_bytes: bytes,
    validation: Mapping[str, Any],
    seed: int,
    artifacts: ArtifactStore,
    epochs: int = 200,
    learning_rate: float = 0.02,
) -> CanonicalBundle:
    """Fit and publish one reproducible bundle, without any serving assignment."""
    if not examples:
        raise CorpusRejected("challenger bundle requires immutable training examples")
    if type(epochs) is not int or epochs <= 0:
        raise CorpusRejected("epochs must be a positive integer")
    if (
        type(learning_rate) not in (int, float)
        or isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise CorpusRejected("learning_rate must be finite and positive")
    identities = sorted(examples, key=lambda item: item.identity.example_id)
    if len({item.identity.example_id for item in identities}) != len(identities):
        raise CorpusRejected("challenger training example identities are duplicated")
    model = fit_linear_strength_model(
        identities, seed=seed, epochs=epochs, learning_rate=float(learning_rate)
    )
    trained_through = max(item.identity.racing_date for item in identities)
    date.fromisoformat(trained_through)
    corpus_entries = [
        {
            "training_example_id": item.identity.example_id,
            "artifact_checksum": str(item.identity.artifact_checksum),
            "evidence_checksum": str(item.identity.evidence_checksum),
            "result_checksum": str(item.identity.result_checksum),
            "feature_matrix_checksum": str(item.identity.feature_matrix_checksum),
            "racing_date": item.identity.racing_date,
        }
        for item in identities
    ]
    documents: dict[str, bytes] = {
        "model": model.to_bytes(),
        "feature_schema": schema_bytes,
        "missingness_policy": missingness_bytes,
        "training_configuration": canonical_json(
            {
                "model_id": model_id,
                "feature_contract_version": SUPPORTED_FEATURE_CONTRACT,
                "forecast_contract_version": ORDERED_FINISH_CONTRACT,
                "algorithm": "full-batch-plackett-luce-linear-v1",
                "optimizer": "deterministic-full-batch-gradient-ascent",
                "seed": seed,
                "epochs": epochs,
                "learning_rate": float(learning_rate),
            }
        ),
        "dependency_manifest": canonical_json(
            {"model_id": model_id, "packages": {"numpy": np.__version__}}
        ),
        "training_corpus": canonical_json(
            {
                "model_id": model_id,
                "corpus_id": str(checksum(canonical_json(corpus_entries))),
                "training_example_ids": [item.identity.example_id for item in identities],
                "training_examples": corpus_entries,
                "origins": ["forward-sealed"],
                "legacy_promotion_evidence": False,
            }
        ),
        "calibration": canonical_json(
            {
                "model_id": model_id,
                "forecast_contract_version": ORDERED_FINISH_CONTRACT,
                "method": "native-plackett-luce-normalization",
                "status": "initial-challenger-unpromoted",
            }
        ),
        "evaluation": canonical_json(
            {
                "model_id": model_id,
                "forecast_contract_version": ORDERED_FINISH_CONTRACT,
                "population": "evaluation-eligible-unambiguous-forward-examples",
                "validation": validation,
                "promotion_decision": None,
            }
        ),
        "runtime_requirements": canonical_json(
            {
                "model_id": model_id,
                "python_implementation": platform.python_implementation(),
                "python_major_minor": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
            }
        ),
    }
    components = []
    for kind, content in documents.items():
        artifact = artifacts.put(
            content,
            media_type=(
                "application/json"
                if kind != "model"
                else "application/vnd.canonical.linear-model+json"
            ),
        )
        components.append(BundleComponent(f"{kind}.json", kind, artifact.checksum, len(content)))
    provisional = CanonicalBundle(
        bundle_id,
        model_id,
        "canonical",
        checksum(b"placeholder"),
        SUPPORTED_FEATURE_CONTRACT,
        ORDERED_FINISH_CONTRACT,
        tuple(components),
        trained_through,
    )
    manifest_bytes = canonical_json(provisional.manifest())
    # The checksum is not a field of manifest(), so replacing it is non-recursive.
    bundle_checksum = artifacts.put(manifest_bytes, media_type="application/json").checksum
    return CanonicalBundle(
        bundle_id,
        model_id,
        "canonical",
        bundle_checksum,
        SUPPORTED_FEATURE_CONTRACT,
        ORDERED_FINISH_CONTRACT,
        tuple(components),
        trained_through,
    )


def wagering_strategy_report(
    forecast: Mapping[str, float], sealed_decimal_odds: Mapping[str, float], *, threshold: float
) -> Mapping[str, Any]:
    """Pure report-only win-ticket illustration; never part of model quality or execution."""
    tickets = []
    for runner in sorted(forecast):
        odds = sealed_decimal_odds.get(runner)
        if type(odds) in (int, float) and not isinstance(odds, bool) and odds > 1:
            edge = forecast[runner] * float(odds) - 1.0
            if edge >= threshold:
                tickets.append({"runner_id": runner, "decimal_odds": odds, "model_edge": edge})
    return {
        "schema_version": "wagering-strategy-report-v1",
        "report_only": True,
        "places_bets": False,
        "forecast_quality_metric": False,
        "tickets": tickets,
    }
