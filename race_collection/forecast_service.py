"""One canonical forecast application service and compatibility adapters."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import numpy as np

from .artifacts import ArtifactStore
from .domain import ArtifactChecksum, OperationId, require_aware
from .features import (
    derive_features,
    validate_feature_availability_manifest_document,
)
from .forecasting import PredictionRequest
from .model_bundle import (
    SUPPORTED_FORECAST_CONTRACT,
    BundleUnavailable,
    ChampionLoader,
    PredictionProvenance,
)
from .operations import iso_timestamp
from .ordered_finish import (
    ORDERED_FINISH_CONTRACT,
    OrderedFinishError,
    forecast_ordered_finish,
    ordered_finish_from_probabilities,
)


class ForecastUnavailable(RuntimeError):
    pass


def _probability(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _validate_prediction_output_schema(document: Mapping[str, Any]) -> None:
    ordered = document.get("forecast_contract_version") == ORDERED_FINISH_CONTRACT
    prediction_fields = {"dog_id", "win_probability", "rank"} | (
        {"top_2_probability", "top_3_probability"} if ordered else set()
    )
    predictions = document.get("predictions")
    if (
        type(predictions) is not list
        or not predictions
        or any(type(row) is not dict or set(row) != prediction_fields for row in predictions)
        or any(
            type(row["dog_id"]) is not str
            or not row["dog_id"]
            or type(row["rank"]) is not int
            or row["rank"] <= 0
            or not _probability(row["win_probability"])
            or (ordered and not _probability(row["top_2_probability"]))
            or (ordered and not _probability(row["top_3_probability"]))
            for row in predictions
        )
    ):
        raise ForecastUnavailable("prediction artifact schema is invalid")
    runner_ids = tuple(row["dog_id"] for row in predictions)
    ranks = tuple(row["rank"] for row in predictions)
    if len(set(runner_ids)) != len(runner_ids) or set(ranks) != set(
        range(1, len(predictions) + 1)
    ):
        raise ForecastUnavailable("prediction artifact schema is invalid")
    if not ordered:
        return
    ordered_shapes = {
        "ordered_finish_probabilities": len(runner_ids),
        "exacta_probabilities": 2,
        "trifecta_probabilities": 3,
        "most_likely_orders": len(runner_ids),
    }
    for field, order_size in ordered_shapes.items():
        rows = document.get(field)
        if type(rows) is not list or any(
            type(row) is not dict
            or set(row) != {"order", "probability"}
            or type(row["order"]) is not list
            or len(row["order"]) != order_size
            or len(set(row["order"])) != order_size
            or any(runner not in runner_ids for runner in row["order"])
            or not _probability(row["probability"])
            for row in rows
        ):
            raise ForecastUnavailable("prediction artifact schema is invalid")
    try:
        full_rows = document["ordered_finish_probabilities"]
        full = {tuple(row["order"]): row["probability"] for row in full_rows}
        if len(full) != len(full_rows):
            raise OrderedFinishError("ordered probabilities are duplicated")
        reconstructed = ordered_finish_from_probabilities(runner_ids, full)
    except (KeyError, TypeError, ValueError, OrderedFinishError) as error:
        raise ForecastUnavailable("prediction ordered distribution is invalid") from error
    expected_projections = {
        "exacta_probabilities": reconstructed.exacta,
        "trifecta_probabilities": reconstructed.trifecta,
    }
    for field, expected in expected_projections.items():
        actual = {tuple(row["order"]): row["probability"] for row in document[field]}
        if len(actual) != len(document[field]) or actual != expected:
            raise ForecastUnavailable("prediction ordered distribution is invalid")
    most_likely = [
        {"order": list(order), "probability": probability}
        for order, probability in reconstructed.most_likely_orders
    ]
    if document["most_likely_orders"] != most_likely:
        raise ForecastUnavailable("prediction ordered distribution is invalid")
    expected_predictions = [
        {
            "dog_id": runner,
            "win_probability": reconstructed.win[runner],
            "top_2_probability": reconstructed.top_2[runner],
            "top_3_probability": reconstructed.top_3[runner],
            "rank": reconstructed.ranking.index(runner) + 1,
        }
        for runner in reconstructed.ranking
    ]
    if predictions != expected_predictions:
        raise ForecastUnavailable("prediction ordered distribution is invalid")


@dataclass(frozen=True, slots=True)
class ForecastRequest:
    evidence_checksum: ArtifactChecksum
    seal_id: int | None = None
    race_id: str | None = None


class CanonicalForecastService:
    def __init__(
        self,
        loader: ChampionLoader,
        artifacts: ArtifactStore,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self.loader, self.artifacts = loader, artifacts
        self.clock = clock

    def emit_training_request(
        self,
        operation_id: OperationId,
        *,
        request_id: str,
        reason: str,
        requested_at: datetime,
        evidence_id: str | None = None,
        service_run_id: OperationId,
    ) -> bool:
        """Durably request an external workflow; this service has no training authority."""
        payload = {
            "request_id": request_id,
            "reason": reason,
            "requested_at": iso_timestamp(requested_at),
            "evidence_id": evidence_id,
            "service_run_id": str(service_run_id),
        }
        with self.loader.store._operation(operation_id, "emit_training_request", payload) as (
            db,
            replay,
        ):
            if replay:
                return False
            db.execute(
                "INSERT INTO phase6_training_requests VALUES(?,?,?,?,?)",
                (request_id, reason, evidence_id, iso_timestamp(requested_at), str(operation_id)),
            )
            run = db.execute(
                "SELECT 1 FROM phase6_runs WHERE run_id=? AND run_kind='forecast_service'",
                (str(service_run_id),),
            ).fetchone()
            if run is None:
                raise ForecastUnavailable("training request lacks a forecast-service run")
            db.execute(
                "INSERT INTO phase6_service_training_requests VALUES(?,?,?,?)",
                (
                    request_id,
                    str(service_run_id),
                    "canonical_forecast_service",
                    str(operation_id),
                ),
            )
        return True

    def persist_evaluation_forecast(
        self,
        operation_id: OperationId,
        *,
        service_run_id: OperationId,
        race_id: str,
        bundle_id: str,
        bundle_checksum: ArtifactChecksum,
        evidence_checksum: ArtifactChecksum,
        computed_at: datetime,
        computation_id: str,
    ) -> ArtifactChecksum:
        """Load and score one registered bundle before persisting its replay artifact."""
        require_aware(computed_at, "computed_at")
        with self.loader.store._connect() as db:
            deferred = db.execute(
                "SELECT prediction_id,seal_id FROM deferred_predictions WHERE race_id=? "
                "AND evidence_checksum=? AND computed_at=?",
                (
                    race_id,
                    str(evidence_checksum),
                    iso_timestamp(computed_at),
                ),
            ).fetchone()
            cohort = db.execute(
                "SELECT c.operation_id,m.service_run_id FROM "
                "phase7_day_forecast_commands c "
                "JOIN phase7_day_forecast_cohort_members m "
                "ON m.racing_day_id=c.racing_day_id AND m.bundle_id=c.bundle_id "
                "WHERE c.race_id=? AND c.bundle_id=?",
                (race_id, bundle_id),
            ).fetchone()
        if deferred is None:
            raise ForecastUnavailable("Phase-3 replay authority is unavailable")
        if cohort is not None and (
            cohort["operation_id"] != str(operation_id)
            or cohort["service_run_id"] != str(service_run_id)
        ):
            raise ForecastUnavailable(
                "forecast operation or service run disagrees with day cohort authority"
            )
        loaded = self.loader.load_registered(bundle_id, bundle_checksum)
        result = self.forecast_with_champion(
            loaded,
            ForecastRequest(evidence_checksum, deferred["seal_id"], race_id),
            computed_at=computed_at,
        )
        orders = result.get("ordered_finish_probabilities")
        if type(orders) is not list:
            raise ForecastUnavailable("registered bundle did not emit an ordered distribution")
        distribution = {
            "runner_ids": [item["dog_id"] for item in result["predictions"]],
            "orders": [[item["order"], item["probability"]] for item in orders],
        }
        document = {
            "schema_version": "phase6-ordered-forecast-v1",
            "race_id": race_id,
            "bundle_id": bundle_id,
            "bundle_checksum": str(bundle_checksum),
            "evidence_checksum": str(evidence_checksum),
            "computed_at": iso_timestamp(computed_at),
            "distribution": distribution,
        }
        artifact = self.artifacts.put(
            json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
            media_type="application/json",
        )
        forecast_checksum = artifact.checksum
        payload = {
            "race_id": race_id,
            "bundle_id": bundle_id,
            "artifact": str(artifact.checksum),
            "computation_id": computation_id,
            "service_run_id": str(service_run_id),
        }
        with self.loader.store._operation(operation_id, "persist_evaluation_forecast", payload) as (
            db,
            replay,
        ):
            if not replay:
                db.execute(
                    "INSERT INTO phase6_service_computations VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        computation_id,
                        race_id,
                        bundle_id,
                        str(bundle_checksum),
                        str(evidence_checksum),
                        iso_timestamp(computed_at),
                        str(service_run_id),
                        deferred["prediction_id"],
                        str(operation_id),
                    ),
                )
                db.execute(
                    "INSERT INTO phase6_forecast_service_artifacts VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(forecast_checksum),
                        race_id,
                        bundle_id,
                        str(bundle_checksum),
                        str(evidence_checksum),
                        str(artifact.checksum),
                        iso_timestamp(computed_at),
                        str(service_run_id),
                        deferred["prediction_id"],
                        str(operation_id),
                    ),
                )
                db.execute(
                    "INSERT INTO phase6_forecast_computation_bindings VALUES(?,?)",
                    (str(forecast_checksum), computation_id),
                )
        return forecast_checksum

    def forecast(self, request: ForecastRequest) -> Mapping[str, Any]:
        self._authenticate_evidence(request)
        try:
            champion = self.loader.load()
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ForecastUnavailable(str(error)) from error
        return self.forecast_with_champion(champion, request, computed_at=self.clock())

    def _authenticate_evidence(self, request: ForecastRequest) -> None:
        """Authenticate deferred seal/race/freeze before loading or scoring a model."""
        if request.seal_id is None and request.race_id is None:
            return
        try:
            document = json.loads(self.artifacts.read(request.evidence_checksum))
            if type(request.seal_id) is not int or type(request.race_id) is not str:
                raise ForecastUnavailable("deferred evidence authority context is incomplete")
            if document.get("race_id") != request.race_id:
                raise ForecastUnavailable("sealed evidence race identity disagrees")
            durable = self.loader.authenticate_seal(
                seal_id=request.seal_id,
                race_id=request.race_id,
                evidence_checksum=request.evidence_checksum,
            )
            freeze = document.get("freeze")
            if type(freeze) is not dict or type(freeze.get("at")) is not str:
                raise ForecastUnavailable("sealed evidence freeze envelope is missing")
            frozen = datetime.fromisoformat(freeze["at"])
            require_aware(frozen, "sealed evidence frozen_at")
            require_aware(durable, "durable sealed evidence frozen_at")
            if frozen != durable:
                raise ForecastUnavailable("sealed evidence freeze disagrees with durable seal")
        except ForecastUnavailable:
            raise
        except Exception as error:
            raise ForecastUnavailable(str(error)) from error

    def derive_with_champion(self, champion, request: ForecastRequest):
        """Authenticate and derive the exact sealed inputs without invoking the model."""
        try:
            evidence = self.artifacts.read(request.evidence_checksum)
            evidence_document = json.loads(evidence)
            if request.race_id is not None and evidence_document.get("race_id") != request.race_id:
                raise ForecastUnavailable("sealed evidence race identity disagrees")
            if request.seal_id is not None or request.race_id is not None:
                if type(request.seal_id) is not int or type(request.race_id) is not str:
                    raise ForecastUnavailable("deferred evidence authority context is incomplete")
                durable_frozen_at = self.loader.authenticate_seal(
                    seal_id=request.seal_id,
                    race_id=request.race_id,
                    evidence_checksum=request.evidence_checksum,
                )
            else:
                durable_frozen_at = None
            freeze = evidence_document.get("freeze")
            if type(freeze) is not dict or type(freeze.get("at")) is not str:
                raise ForecastUnavailable("sealed evidence freeze envelope is missing")
            evidence_frozen_at = datetime.fromisoformat(freeze["at"])
            require_aware(evidence_frozen_at, "sealed evidence frozen_at")
            if durable_frozen_at is not None:
                require_aware(durable_frozen_at, "durable sealed evidence frozen_at")
                if evidence_frozen_at != durable_frozen_at:
                    raise ForecastUnavailable("sealed evidence freeze disagrees with durable seal")
            schema_component = champion.bundle.component("feature_schema")
            missing_component = champion.bundle.component("missingness_policy")
            derived = derive_features(
                evidence,
                expected_evidence_checksum=request.evidence_checksum,
                schema_bytes=self.artifacts.read(schema_component.checksum),
                expected_schema_checksum=schema_component.checksum,
                missingness_policy_bytes=self.artifacts.read(missing_component.checksum),
                expected_missingness_checksum=missing_component.checksum,
                expected_feature_cutoff_at=durable_frozen_at,
                raw_evidence_reader=self.artifacts.read,
            )
            return derived, evidence_frozen_at
        except ForecastUnavailable:
            raise
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ForecastUnavailable(str(error)) from error

    def forecast_with_champion(
        self, champion, request: ForecastRequest, *, computed_at: datetime
    ) -> Mapping[str, Any]:
        require_aware(computed_at, "computed_at")
        try:
            derived, evidence_frozen_at = self.derive_with_champion(champion, request)
            contract = champion.bundle.forecast_contract_version
            ordered = None
            legacy_wins = None
            if contract == ORDERED_FINISH_CONTRACT:
                if not callable(getattr(champion.model, "latent_strengths", None)):
                    raise BundleUnavailable("ordered-finish model lacks latent_strengths")
                strengths = champion.model.latent_strengths(derived.matrix.rows)
                ordered = forecast_ordered_finish(derived.matrix.runner_ids, strengths)
                ranking = ordered.ranking
            elif contract == SUPPORTED_FORECAST_CONTRACT:
                probabilities = champion.model.predict_proba(derived.matrix.rows)
                if isinstance(probabilities, np.ndarray):
                    if probabilities.ndim != 2:
                        raise BundleUnavailable("model returned malformed predict_proba shape")
                    probabilities = probabilities.tolist()
                if type(probabilities) not in (list, tuple) or any(
                    type(row) not in (list, tuple) or len(row) != 2 for row in probabilities
                ):
                    raise BundleUnavailable("model returned malformed predict_proba shape")
                try:
                    numeric = [tuple(float(value) for value in row) for row in probabilities]
                except (TypeError, ValueError) as error:
                    raise BundleUnavailable("model returned non-numeric probabilities") from error
                if len(numeric) != len(derived.matrix.runner_ids) or any(
                    not math.isfinite(value) or value < 0 or value > 1
                    for row in numeric
                    for value in row
                ):
                    raise BundleUnavailable("model returned invalid probabilities")
                if any(
                    not math.isclose(sum(row), 1.0, rel_tol=1e-9, abs_tol=1e-12) for row in numeric
                ):
                    raise BundleUnavailable("model probability rows do not sum to one")
                total = math.fsum(row[1] for row in numeric)
                if total <= 0:
                    raise BundleUnavailable("model returned zero probability mass")
                legacy_wins = tuple(row[1] / total for row in numeric)
                ranking = tuple(
                    runner
                    for _, runner in sorted(
                        zip(legacy_wins, derived.matrix.runner_ids),
                        key=lambda item: (-item[0], item[1]),
                    )
                )
            else:  # loader also guards this; keep direct service use fail-closed.
                raise BundleUnavailable("forecast contract is unsupported")
            ranks = {runner: rank for rank, runner in enumerate(ranking, 1)}
            provenance = PredictionProvenance(
                champion.bundle.model_id,
                str(champion.bundle.bundle_checksum),
                champion.bundle.trained_through,
                champion.assignment.promotion_approved_at,
                champion.assignment.promotion_effective_from_racing_day,
                champion.assignment.promotion_record_id,
                iso_timestamp(computed_at),
                iso_timestamp(evidence_frozen_at),
            ).as_dict()
            return {
                "success": True,
                "forecast_contract_version": champion.bundle.forecast_contract_version,
                **(
                    {
                        "feature_contract": {
                            "version": derived.contract.version,
                            "schema_checksum": str(derived.contract.schema_checksum),
                            "missingness_policy_checksum": str(
                                derived.contract.missingness_policy_checksum
                            ),
                        },
                        "feature_availability_manifest_checksum": str(
                            derived.availability_manifest.checksum
                        ),
                        "feature_availability_manifest": (
                            derived.availability_manifest.as_dict()
                        ),
                    }
                    if derived.availability_manifest is not None
                    else {}
                ),
                "feature_matrix_checksum": str(derived.matrix.checksum),
                "evidence_checksum": str(request.evidence_checksum),
                "predictions": [
                    {
                        "dog_id": runner,
                        "win_probability": (
                            ordered.win[runner] if ordered is not None else legacy_wins[index]
                        ),
                        **(
                            {
                                "top_2_probability": ordered.top_2[runner],
                                "top_3_probability": ordered.top_3[runner],
                            }
                            if ordered is not None
                            else {}
                        ),
                        "rank": ranks[runner],
                    }
                    for index, runner in enumerate(derived.matrix.runner_ids)
                ],
                **(
                    {
                        "ordered_finish_probabilities": [
                            {"order": list(order), "probability": probability}
                            for order, probability in sorted(ordered.order_probabilities.items())
                        ],
                        "exacta_probabilities": [
                            {"order": list(order), "probability": probability}
                            for order, probability in sorted(ordered.exacta.items())
                        ],
                        "trifecta_probabilities": [
                            {"order": list(order), "probability": probability}
                            for order, probability in sorted(ordered.trifecta.items())
                        ],
                        "most_likely_orders": [
                            {"order": list(order), "probability": probability}
                            for order, probability in ordered.most_likely_orders
                        ],
                    }
                    if ordered is not None
                    else {}
                ),
                "provenance": provenance,
            }
        except Exception as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise ForecastUnavailable(str(error)) from error


class CanonicalDeferredPredictor:
    """Phase-3 seam: failures propagate into its per-race quarantine authority."""

    def __init__(
        self,
        service: CanonicalForecastService,
        artifacts: ArtifactStore,
        *,
        clock: Callable[[], datetime],
    ):
        self.service, self.artifacts, self.clock = service, artifacts, clock

    def predict(self, request: PredictionRequest) -> ArtifactChecksum:
        champion = self.service.loader.load_day_pin(request)
        computed_at = self.clock()
        require_aware(computed_at, "deferred prediction clock")
        result = self.service.forecast_with_champion(
            champion,
            ForecastRequest(request.evidence_checksum, request.seal_id, str(request.race_id)),
            computed_at=computed_at,
        )
        entries = result.get("feature_availability_manifest", {}).get("entries")
        if type(entries) is not list or any(
            type(entry) is not dict
            or (
                entry.get("semantics") in {"identity-critical", "forecast-required"}
                and entry.get("status") != "READY_NOW"
            )
            for entry in entries
        ):
            raise ForecastUnavailable(
                "deferred prediction requires READY_NOW identity-critical and required inputs"
            )
        result = {
            **result,
            "deferred_identity": {
                "race_id": str(request.race_id),
                "racing_day_id": str(request.racing_day_id),
                "seal_id": request.seal_id,
                "sealed_evidence_checksum": str(request.evidence_checksum),
                "model_bundle_id": request.bundle.bundle_id,
                "model_release_id": request.release.release_id,
                "promotion_policy_id": request.policy_id,
            },
        }
        artifact = self.artifacts.put(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
            media_type="application/vnd.canonical-race-forecast+json",
        )
        return artifact.checksum

    def authenticate(self, checksum: ArtifactChecksum, expected_computed_at: datetime) -> None:
        """Authenticate the returned document against Phase-3's commit time."""
        document = json.loads(self.artifacts.read(checksum))
        if type(document) is not dict:
            raise ForecastUnavailable("prediction artifact schema is invalid")
        actual = document.get("provenance", {}).get("prediction_computed_at")
        if actual != iso_timestamp(expected_computed_at):
            raise ForecastUnavailable("prediction artifact computation time disagrees")

    def authenticate_request(
        self,
        checksum: ArtifactChecksum,
        expected_computed_at: datetime,
        request: PredictionRequest,
    ) -> None:
        """Bind the sealed prediction bytes to the exact lifecycle authority snapshot."""
        self.authenticate(checksum, expected_computed_at)
        document = json.loads(self.artifacts.read(checksum))
        champion = self.service.loader.load_day_pin(request)
        base_fields = {
            "success",
            "forecast_contract_version",
            "feature_contract",
            "feature_availability_manifest_checksum",
            "feature_availability_manifest",
            "feature_matrix_checksum",
            "evidence_checksum",
            "predictions",
            "provenance",
            "deferred_identity",
        }
        ordered_fields = {
            "ordered_finish_probabilities",
            "exacta_probabilities",
            "trifecta_probabilities",
            "most_likely_orders",
        }
        expected_fields = (
            base_fields | ordered_fields
            if document.get("forecast_contract_version") == ORDERED_FINISH_CONTRACT
            else base_fields
        )
        if set(document) != expected_fields or document.get("success") is not True:
            raise ForecastUnavailable("prediction artifact schema is invalid")
        if document.get("forecast_contract_version") != champion.bundle.forecast_contract_version:
            raise ForecastUnavailable("prediction forecast contract is invalid")
        _validate_prediction_output_schema(document)
        identity = document.get("deferred_identity")
        expected = {
            "race_id": str(request.race_id),
            "racing_day_id": str(request.racing_day_id),
            "seal_id": request.seal_id,
            "sealed_evidence_checksum": str(request.evidence_checksum),
            "model_bundle_id": request.bundle.bundle_id,
            "model_release_id": request.release.release_id,
            "promotion_policy_id": request.policy_id,
        }
        if type(identity) is not dict or set(identity) != set(expected) or identity != expected:
            raise ForecastUnavailable("prediction artifact authority identity disagrees")
        provenance = document.get("provenance")
        if type(provenance) is not dict or set(provenance) != set(
            PredictionProvenance.__dataclass_fields__
        ):
            raise ForecastUnavailable("prediction provenance envelope is incomplete")
        try:
            PredictionProvenance(**provenance)
            ArtifactChecksum(provenance["artifact_checksum"])
        except (TypeError, ValueError) as error:
            raise ForecastUnavailable("prediction provenance envelope is invalid") from error
        evidence_frozen_at = self.service.loader.authenticate_seal(
            seal_id=request.seal_id,
            race_id=str(request.race_id),
            evidence_checksum=request.evidence_checksum,
        )
        expected_provenance = PredictionProvenance(
            champion.bundle.model_id,
            str(champion.bundle.bundle_checksum),
            champion.bundle.trained_through,
            champion.assignment.promotion_approved_at,
            champion.assignment.promotion_effective_from_racing_day,
            champion.assignment.promotion_record_id,
            iso_timestamp(expected_computed_at),
            iso_timestamp(evidence_frozen_at),
        ).as_dict()
        if provenance != expected_provenance:
            raise ForecastUnavailable("prediction provenance disagrees with durable authority")
        if document.get("evidence_checksum") != str(request.evidence_checksum):
            raise ForecastUnavailable("prediction evidence identity disagrees")
        contract = document.get("feature_contract")
        if type(contract) is not dict or set(contract) != {
            "version",
            "schema_checksum",
            "missingness_policy_checksum",
        }:
            raise ForecastUnavailable("prediction feature contract provenance is incomplete")
        schema_component = champion.bundle.component("feature_schema")
        missing_component = champion.bundle.component("missingness_policy")
        expected_contract = {
            "version": champion.bundle.feature_contract_version,
            "schema_checksum": str(schema_component.checksum),
            "missingness_policy_checksum": str(missing_component.checksum),
        }
        if contract != expected_contract:
            raise ForecastUnavailable("prediction feature contract version is invalid")
        ArtifactChecksum(contract["schema_checksum"])
        ArtifactChecksum(contract["missingness_policy_checksum"])
        manifest = document.get("feature_availability_manifest")
        if (
            type(manifest) is not dict
            or manifest.get("race_id") != str(request.race_id)
            or manifest.get("evidence_checksum") != str(request.evidence_checksum)
        ):
            raise ForecastUnavailable("prediction feature availability manifest is invalid")
        try:
            manifest_features = validate_feature_availability_manifest_document(manifest)
        except (TypeError, ValueError) as error:
            raise ForecastUnavailable(
                "prediction feature availability manifest is invalid"
            ) from error
        try:
            schema = json.loads(self.artifacts.read(schema_component.checksum))
            declared_features = [
                (item["name"], item["family"], item["semantics"])
                for item in [*schema["fields"], *schema.get("candidate_features", [])]
            ]
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ForecastUnavailable(
                "prediction feature availability manifest authority is invalid"
            ) from error
        manifest_declarations = [
            (entry["feature"], entry["family"], entry["semantics"])
            for entry in manifest["entries"]
        ]
        if list(manifest_features) != [item[0] for item in declared_features] or (
            manifest_declarations != declared_features
        ):
            raise ForecastUnavailable(
                "prediction feature availability manifest membership disagrees"
            )
        manifest_checksum = ArtifactChecksum(document["feature_availability_manifest_checksum"])
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if (
            ArtifactChecksum("sha256:" + hashlib.sha256(manifest_bytes).hexdigest())
            != manifest_checksum
        ):
            raise ForecastUnavailable("prediction feature availability manifest checksum disagrees")
        matrix_checksum = ArtifactChecksum(document["feature_matrix_checksum"])
        derived, _frozen_at = self.service.derive_with_champion(
            champion,
            ForecastRequest(request.evidence_checksum, request.seal_id, str(request.race_id)),
        )
        if (
            derived.availability_manifest is None
            or manifest != derived.availability_manifest.as_dict()
            or manifest_checksum != derived.availability_manifest.checksum
            or matrix_checksum != derived.matrix.checksum
        ):
            raise ForecastUnavailable(
                "prediction feature availability manifest or matrix disagrees with sealed evidence"
            )


def canonical_endpoint(
    service: CanonicalForecastService, payload: Mapping[str, Any]
) -> tuple[Mapping[str, Any], int]:
    try:
        if "evidence_frozen_at" in payload or "prediction_computed_at" in payload:
            return {
                "success": False,
                "status": "unavailable",
                "error": "provenance timestamps are server-authenticated",
            }, 400
        request = ForecastRequest(ArtifactChecksum(payload["evidence_checksum"]))
        return service.forecast(request), 200
    except (KeyError, TypeError, ValueError) as error:
        return {"success": False, "status": "unavailable", "error": str(error)}, 400
    except ForecastUnavailable as error:
        return {"success": False, "status": "unavailable", "error": str(error)}, 503


def training_request_endpoint(
    service: CanonicalForecastService,
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], int]:
    """The only mutation exposed by the forecast-service boundary."""
    try:
        created = service.emit_training_request(
            OperationId(payload["operation_id"]),
            request_id=payload["request_id"],
            reason=payload["reason"],
            evidence_id=payload.get("evidence_id"),
            requested_at=datetime.fromisoformat(payload["requested_at"]),
            service_run_id=OperationId(payload["service_run_id"]),
        )
        return {"success": True, "created": created}, 202
    except (KeyError, TypeError, ValueError, ForecastUnavailable) as error:
        return {"success": False, "status": "unavailable", "error": str(error)}, 400


class CanonicalForecastApplication:
    """Mounted application boundary for canonical forecasts and training requests."""

    def __init__(self, service: CanonicalForecastService):
        self.service = service

    def handle(
        self, method: str, path: str, payload: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], int]:
        if method == "POST" and path == "/v1/forecasts":
            return canonical_endpoint(self.service, payload)
        if method == "POST" and path == "/v1/training-requests":
            return training_request_endpoint(self.service, payload)
        return {"success": False, "status": "not_found"}, 404


def legacy_prediction_adapter(
    service: CanonicalForecastService, payload: Mapping[str, Any]
) -> tuple[Mapping[str, Any], int]:
    """Compatibility surface; intentionally returns the canonical result unchanged."""
    return canonical_endpoint(service, payload)
