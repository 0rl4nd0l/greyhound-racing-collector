"""One canonical forecast application service and compatibility adapters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

import numpy as np

from .artifacts import ArtifactStore
from .domain import ArtifactChecksum, OperationId, require_aware
from .features import derive_features
from .forecasting import PredictionRequest
from .model_bundle import (
    SUPPORTED_FORECAST_CONTRACT,
    BundleUnavailable,
    ChampionLoader,
    PredictionProvenance,
)
from .operations import iso_timestamp
from .ordered_finish import ORDERED_FINISH_CONTRACT, forecast_ordered_finish


class ForecastUnavailable(RuntimeError):
    pass


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

    def forecast_with_champion(
        self, champion, request: ForecastRequest, *, computed_at: datetime
    ) -> Mapping[str, Any]:
        require_aware(computed_at, "computed_at")
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
            )
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
        artifact = self.artifacts.put(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(),
            media_type="application/vnd.canonical-race-forecast+json",
        )
        return artifact.checksum

    def authenticate(self, checksum: ArtifactChecksum, expected_computed_at: datetime) -> None:
        """Authenticate the returned document against Phase-3's commit time."""
        document = json.loads(self.artifacts.read(checksum))
        actual = document.get("provenance", {}).get("prediction_computed_at")
        if actual != iso_timestamp(expected_computed_at):
            raise ForecastUnavailable("prediction artifact computation time disagrees")


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
