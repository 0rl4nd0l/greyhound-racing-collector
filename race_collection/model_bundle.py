"""Immutable canonical model-bundle authority and fail-closed loader."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable, Mapping

from .artifacts import ArtifactStore
from .domain import ArtifactChecksum, OperationId, require_aware
from .forecasting import PredictionRequest
from .operations import (
    OperationsStoreError,
    SQLiteOperationsStore,
    iso_timestamp,
)
from .ordered_finish import ORDERED_FINISH_CONTRACT

SUPPORTED_FEATURE_CONTRACT = "sealed-race-features-v1"
SUPPORTED_FORECAST_CONTRACT = "runner-win-probability-v1"
SUPPORTED_FORECAST_CONTRACTS = {SUPPORTED_FORECAST_CONTRACT, ORDERED_FINISH_CONTRACT}
COMPONENT_KINDS = (
    "model",
    "feature_schema",
    "missingness_policy",
    "training_configuration",
    "dependency_manifest",
    "training_corpus",
    "calibration",
    "evaluation",
    "runtime_requirements",
)


class BundleUnavailable(OperationsStoreError):
    """The exact pinned champion cannot be safely loaded."""


def validate_training_corpus_manifest(corpus: Mapping[str, Any], bundle: "CanonicalBundle") -> None:
    identities = corpus.get("training_example_ids")
    corpus_entries = corpus.get("training_examples")
    ordered_corpus = bundle.forecast_contract_version == ORDERED_FINISH_CONTRACT
    if (
        type(identities) is not list
        or not identities
        or any(type(item) is not str or not item for item in identities)
        or len(set(identities)) != len(identities)
        or (ordered_corpus and identities != sorted(identities))
        or type(corpus.get("corpus_id")) is not str
        or not corpus.get("corpus_id")
        or (
            ordered_corpus
            and (type(corpus_entries) is not list or len(corpus_entries) != len(identities))
        )
    ):
        raise BundleUnavailable("training corpus identity is incomplete")
    if not ordered_corpus:
        return
    required_entry_fields = {
        "training_example_id",
        "artifact_checksum",
        "evidence_checksum",
        "result_checksum",
        "feature_matrix_checksum",
        "racing_date",
    }
    try:
        invalid = (
            any(type(entry) is not dict for entry in corpus_entries)
            or any(set(entry) != required_entry_fields for entry in corpus_entries)
            or [entry["training_example_id"] for entry in corpus_entries] != identities
            or any(
                not all(
                    type(entry[field]) is str and entry[field] for field in required_entry_fields
                )
                for entry in corpus_entries
            )
            or any(
                ArtifactChecksum(entry[field]) is None
                for entry in corpus_entries
                for field in (
                    "artifact_checksum",
                    "evidence_checksum",
                    "result_checksum",
                    "feature_matrix_checksum",
                )
            )
            or max(date.fromisoformat(entry["racing_date"]) for entry in corpus_entries)
            != date.fromisoformat(bundle.trained_through)
            or corpus["corpus_id"]
            != "sha256:"
            + hashlib.sha256(
                json.dumps(
                    corpus_entries,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
        )
    except (KeyError, TypeError, ValueError) as error:
        raise BundleUnavailable("training corpus identities or cutoff disagree") from error
    if invalid:
        raise BundleUnavailable("training corpus identities or cutoff disagree")


def _required(value: str, name: str) -> str:
    if type(value) is not str or not value.strip() or value.strip().lower() == "unknown":
        raise ValueError(f"{name} must be known nonblank text")
    return value


@dataclass(frozen=True, slots=True)
class BundleComponent:
    name: str
    kind: str
    checksum: ArtifactChecksum
    byte_size: int

    def __post_init__(self) -> None:
        _required(self.name, "component name")
        if self.kind not in COMPONENT_KINDS:
            raise ValueError("unsupported component kind")
        if not isinstance(self.checksum, ArtifactChecksum) or self.byte_size <= 0:
            raise ValueError("component checksum and positive size are required")


@dataclass(frozen=True, slots=True)
class PredictionProvenance:
    champion_model_id: str
    artifact_checksum: str
    trained_through: str
    promotion_approved_at: str
    promotion_effective_from_racing_day: str
    promotion_record_id: str
    prediction_computed_at: str
    evidence_frozen_at: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _required(getattr(self, name), name)
        for name in ("promotion_approved_at", "prediction_computed_at", "evidence_frozen_at"):
            parsed = datetime.fromisoformat(getattr(self, name))
            require_aware(parsed, name)
        date.fromisoformat(self.trained_through)
        date.fromisoformat(self.promotion_effective_from_racing_day)
        ArtifactChecksum(self.artifact_checksum)

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class CanonicalBundle:
    bundle_id: str
    model_id: str
    origin: str
    bundle_checksum: ArtifactChecksum
    feature_contract_version: str
    forecast_contract_version: str
    components: tuple[BundleComponent, ...]
    trained_through: str
    legacy_model_bundle_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "bundle_id",
            "model_id",
            "feature_contract_version",
            "forecast_contract_version",
            "trained_through",
        ):
            _required(getattr(self, name), name)
        if self.origin not in {"canonical", "legacy-origin"}:
            raise ValueError("invalid bundle origin")
        if (self.origin == "canonical" and self.legacy_model_bundle_id is not None) or (
            self.origin == "legacy-origin"
            and (
                type(self.legacy_model_bundle_id) is not str
                or not self.legacy_model_bundle_id.strip()
            )
        ):
            raise ValueError("legacy origin requires one explicit Phase-3 bundle binding")
        if not isinstance(self.bundle_checksum, ArtifactChecksum):
            raise ValueError("bundle checksum is required")
        kinds = [component.kind for component in self.components]
        if sorted(kinds) != sorted(COMPONENT_KINDS) or len(set(kinds)) != len(kinds):
            raise ValueError("bundle requires exactly one component of every kind")
        date.fromisoformat(self.trained_through)

    def component(self, kind: str) -> BundleComponent:
        return next(component for component in self.components if component.kind == kind)

    def manifest(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "model_id": self.model_id,
            "origin": self.origin,
            "legacy_model_bundle_id": self.legacy_model_bundle_id,
            "feature_contract_version": self.feature_contract_version,
            "forecast_contract_version": self.forecast_contract_version,
            "trained_through": self.trained_through,
            "components": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "checksum": str(c.checksum),
                    "byte_size": c.byte_size,
                }
                for c in sorted(self.components, key=lambda item: item.kind)
            ],
        }


@dataclass(frozen=True, slots=True)
class ServingAssignment:
    assignment_id: str
    bundle_id: str
    bundle_checksum: ArtifactChecksum
    promotion_approved_at: str
    promotion_effective_from_racing_day: str
    promotion_record_id: str

    def __post_init__(self) -> None:
        for name in (
            "assignment_id",
            "bundle_id",
            "promotion_approved_at",
            "promotion_effective_from_racing_day",
            "promotion_record_id",
        ):
            _required(getattr(self, name), name)
        require_aware(datetime.fromisoformat(self.promotion_approved_at), "promotion_approved_at")
        date.fromisoformat(self.promotion_effective_from_racing_day)
        if not isinstance(self.bundle_checksum, ArtifactChecksum):
            raise ValueError("assignment bundle checksum is required")


class ModelBundleAuthority:
    def __init__(self, store: SQLiteOperationsStore):
        self.store = store

    def register(self, operation_id: OperationId, bundle: CanonicalBundle, at: datetime) -> bool:
        require_aware(at, "at")
        payload = {
            "bundle": bundle.manifest(),
            "checksum": str(bundle.bundle_checksum),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "register_canonical_bundle", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT * FROM canonical_model_bundles " "WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                durable = db.execute(
                    "SELECT component_name,component_kind,artifact_checksum,byte_size "
                    "FROM canonical_bundle_components WHERE bundle_id=? ORDER BY component_kind",
                    (bundle.bundle_id,),
                ).fetchall()
                expected = sorted(
                    [(c.name, c.kind, str(c.checksum), c.byte_size) for c in bundle.components],
                    key=lambda item: item[1],
                )
                checksums = {c.kind: str(c.checksum) for c in bundle.components}
                expected_row = (
                    bundle.bundle_id,
                    bundle.model_id,
                    bundle.origin,
                    bundle.legacy_model_bundle_id,
                    str(bundle.bundle_checksum),
                    bundle.feature_contract_version,
                    bundle.forecast_contract_version,
                    checksums["feature_schema"],
                    checksums["missingness_policy"],
                    checksums["training_configuration"],
                    checksums["dependency_manifest"],
                    checksums["training_corpus"],
                    bundle.trained_through,
                    checksums["calibration"],
                    checksums["evaluation"],
                    checksums["runtime_requirements"],
                    iso_timestamp(at),
                    str(operation_id),
                )
                if (
                    row is None
                    or tuple(row) != expected_row
                    or [tuple(item) for item in durable] != expected
                ):
                    raise OperationsStoreError("canonical bundle replay lacks exact durable result")
                return False
            checksums = {c.kind: str(c.checksum) for c in bundle.components}
            db.execute(
                "INSERT INTO canonical_model_bundles VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    bundle.bundle_id,
                    bundle.model_id,
                    bundle.origin,
                    bundle.legacy_model_bundle_id,
                    str(bundle.bundle_checksum),
                    bundle.feature_contract_version,
                    bundle.forecast_contract_version,
                    checksums["feature_schema"],
                    checksums["missingness_policy"],
                    checksums["training_configuration"],
                    checksums["dependency_manifest"],
                    checksums["training_corpus"],
                    bundle.trained_through,
                    checksums["calibration"],
                    checksums["evaluation"],
                    checksums["runtime_requirements"],
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            db.executemany(
                "INSERT INTO canonical_bundle_components VALUES(?,?,?,?,?)",
                [
                    (bundle.bundle_id, c.name, c.kind, str(c.checksum), c.byte_size)
                    for c in bundle.components
                ],
            )
            db.execute(
                "INSERT INTO phase6_runs VALUES(?,?,?,?)",
                (str(operation_id), "registration", iso_timestamp(at), str(operation_id)),
            )
        return True

    def register_assignment(
        self, operation_id: OperationId, assignment: ServingAssignment, at: datetime
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "assignment": assignment.assignment_id,
            "bundle": assignment.bundle_id,
            "checksum": str(assignment.bundle_checksum),
            "promotion_approved_at": assignment.promotion_approved_at,
            "promotion_effective_from_racing_day": assignment.promotion_effective_from_racing_day,
            "promotion_record_id": assignment.promotion_record_id,
            "at": iso_timestamp(at),
        }
        expected = (
            assignment.assignment_id,
            assignment.bundle_id,
            str(assignment.bundle_checksum),
            assignment.promotion_approved_at,
            assignment.promotion_effective_from_racing_day,
            assignment.promotion_record_id,
            iso_timestamp(at),
            str(operation_id),
        )
        with self.store._operation(operation_id, "register_serving_assignment", payload) as (
            db,
            replay,
        ):
            if replay:
                row = db.execute(
                    "SELECT * FROM canonical_serving_assignments WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or tuple(row) != expected:
                    raise OperationsStoreError("assignment replay lacks exact durable result")
                return False
            db.execute(
                "INSERT INTO canonical_serving_assignments VALUES(?,?,?,?,?,?,?,?)", expected
            )
        return True

    def bootstrap_champion(
        self, operation_id: OperationId, assignment: ServingAssignment, at: datetime
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "assignment": assignment.assignment_id,
            "bundle": assignment.bundle_id,
            "checksum": str(assignment.bundle_checksum),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "bootstrap_champion", payload) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT * FROM champion_pointer WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or tuple(row) != (
                    1,
                    assignment.assignment_id,
                    assignment.bundle_id,
                    str(assignment.bundle_checksum),
                    iso_timestamp(at),
                    str(operation_id),
                ):
                    raise OperationsStoreError("champion replay lacks exact durable pointer")
                return False
            db.execute(
                "INSERT INTO champion_pointer VALUES(1,?,?,?,?,?)",
                (
                    assignment.assignment_id,
                    assignment.bundle_id,
                    str(assignment.bundle_checksum),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
        return True

    def bind_day_assignment(
        self,
        operation_id: OperationId,
        racing_day_id: str,
        assignment: ServingAssignment,
        at: datetime,
    ) -> bool:
        require_aware(at, "at")
        payload = {
            "day": racing_day_id,
            "assignment": assignment.assignment_id,
            "bundle": assignment.bundle_id,
            "checksum": str(assignment.bundle_checksum),
            "at": iso_timestamp(at),
        }
        expected = (
            racing_day_id,
            assignment.assignment_id,
            assignment.bundle_id,
            str(assignment.bundle_checksum),
            iso_timestamp(at),
            str(operation_id),
        )
        with self.store._operation(operation_id, "bind_day_assignment", payload) as (db, replay):
            if replay:
                row = db.execute(
                    "SELECT * FROM canonical_day_assignments WHERE operation_id=?",
                    (str(operation_id),),
                ).fetchone()
                if row is None or tuple(row) != expected:
                    raise OperationsStoreError("day assignment replay lacks exact durable result")
                return False
            db.execute("INSERT INTO canonical_day_assignments VALUES(?,?,?,?,?,?)", expected)
        return True


@dataclass(frozen=True, slots=True)
class LoadedChampion:
    bundle: CanonicalBundle
    assignment: ServingAssignment
    model: Any
    feature_schema: Mapping[str, Any]
    missingness_policy: Mapping[str, Any]


class ChampionLoader:
    """Loads only the singleton pointer after verifying the complete immutable bundle."""

    def __init__(
        self,
        store: SQLiteOperationsStore,
        artifacts: ArtifactStore,
        *,
        deserializer: Callable[[bytes], Any],
    ):
        self.store, self.artifacts, self.deserializer = store, artifacts, deserializer

    @staticmethod
    def _object(content: bytes, name: str) -> Mapping[str, Any]:
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise BundleUnavailable(f"{name} is not valid JSON") from error
        if type(value) is not dict:
            raise BundleUnavailable(f"{name} must be a JSON object")
        return value

    def load(self) -> LoadedChampion:
        """Resolve the current pointer for on-demand use or initial day pinning."""
        with self.store._connect() as db:
            pointer = db.execute("SELECT * FROM champion_pointer WHERE singleton=1").fetchone()
        if pointer is None:
            raise BundleUnavailable("champion pointer is missing")
        with self.store._connect() as db:
            assignment_row = db.execute(
                "SELECT * FROM canonical_serving_assignments WHERE assignment_id=? "
                "AND bundle_id=? AND bundle_checksum=?",
                (
                    pointer["assignment_id"],
                    pointer["bundle_id"],
                    pointer["bundle_checksum"],
                ),
            ).fetchone()
        if assignment_row is None:
            raise BundleUnavailable("champion assignment is unavailable")
        assignment = self._assignment(assignment_row)
        return self._load_exact(assignment)

    @staticmethod
    def _assignment(row: Any) -> ServingAssignment:
        return ServingAssignment(
            row["assignment_id"],
            row["bundle_id"],
            ArtifactChecksum(row["bundle_checksum"]),
            row["promotion_approved_at"],
            row["promotion_effective_from_racing_day"],
            row["promotion_record_id"],
        )

    def load_day_pin(
        self,
        request: PredictionRequest,
    ) -> LoadedChampion:
        """Authenticate the complete Phase-3 day-pin relation, then load it exactly."""
        with self.store._connect() as db:
            relation = db.execute(
                "SELECT a.* FROM racing_day_pins p "
                "JOIN model_releases r ON r.release_id=p.release_id "
                "AND r.bundle_id=p.bundle_id AND r.policy_id=p.policy_id "
                "JOIN model_bundles b ON b.bundle_id=p.bundle_id "
                "JOIN canonical_day_assignments d ON d.racing_day_id=p.racing_day_id "
                "JOIN canonical_serving_assignments a ON a.assignment_id=d.assignment_id "
                "AND a.bundle_id=d.bundle_id AND a.bundle_checksum=d.bundle_checksum "
                "JOIN canonical_model_bundles c ON c.bundle_id=a.bundle_id "
                "AND c.bundle_checksum=a.bundle_checksum AND c.origin='legacy-origin' "
                "AND c.legacy_model_bundle_id=b.bundle_id "
                "JOIN canonical_bundle_components component ON component.bundle_id=c.bundle_id "
                "AND component.component_kind='model' "
                "WHERE p.racing_day_id=? AND p.bundle_id=? AND p.release_id=? "
                "AND p.policy_id=? AND b.artifact_checksum=? AND b.artifact_size=? "
                "AND component.artifact_checksum=b.artifact_checksum "
                "AND component.byte_size=b.artifact_size",
                (
                    str(request.racing_day_id),
                    request.bundle.bundle_id,
                    request.release.release_id,
                    request.policy_id,
                    str(request.bundle.artifact_checksum),
                    request.bundle.artifact_size,
                ),
            ).fetchone()
        if relation is None:
            raise BundleUnavailable("prediction request does not authenticate its Racing Day pin")
        return self._load_exact(self._assignment(relation))

    def load_registered(self, bundle_id: str, bundle_checksum: ArtifactChecksum) -> LoadedChampion:
        """Load an immutable registered challenger for result-blind shadow scoring."""
        with self.store._connect() as db:
            row = db.execute(
                "SELECT created_at,trained_through FROM canonical_model_bundles "
                "WHERE bundle_id=? AND bundle_checksum=?",
                (bundle_id, str(bundle_checksum)),
            ).fetchone()
        if row is None:
            raise BundleUnavailable("registered evaluation bundle is unavailable")
        assignment = ServingAssignment(
            "evaluation-shadow:" + bundle_id,
            bundle_id,
            bundle_checksum,
            row["created_at"],
            row["trained_through"],
            "immutable-registration",
        )
        return self._load_exact(assignment, require_assignment=False)

    def authenticate_seal(
        self, *, seal_id: int, race_id: str, evidence_checksum: ArtifactChecksum
    ) -> datetime:
        """Authenticate the exact durable Phase-3 seal relation for a request."""
        with self.store._connect() as db:
            seal = db.execute(
                "SELECT frozen_at FROM sealed_evidence WHERE seal_id=? AND race_id=? "
                "AND normalized_checksum=?",
                (
                    seal_id,
                    race_id,
                    str(evidence_checksum),
                ),
            ).fetchone()
        if seal is None:
            raise BundleUnavailable("deferred evidence seal relation is unavailable")
        frozen_at = datetime.fromisoformat(seal["frozen_at"])
        require_aware(frozen_at, "durable sealed evidence frozen_at")
        return frozen_at

    def _load_exact(
        self, assignment: ServingAssignment, *, require_assignment: bool = True
    ) -> LoadedChampion:
        """Authenticate and load one exact immutable canonical bundle."""
        with self.store._connect() as db:
            durable_assignment = db.execute(
                "SELECT 1 FROM canonical_serving_assignments WHERE assignment_id=? "
                "AND bundle_id=? AND bundle_checksum=? AND promotion_approved_at=? "
                "AND promotion_effective_from_racing_day=? AND promotion_record_id=?",
                (
                    assignment.assignment_id,
                    assignment.bundle_id,
                    str(assignment.bundle_checksum),
                    assignment.promotion_approved_at,
                    assignment.promotion_effective_from_racing_day,
                    assignment.promotion_record_id,
                ),
            ).fetchone()
            row = db.execute(
                "SELECT * FROM canonical_model_bundles WHERE bundle_id=? AND bundle_checksum=?",
                (assignment.bundle_id, str(assignment.bundle_checksum)),
            ).fetchone()
            components = db.execute(
                "SELECT * FROM canonical_bundle_components WHERE bundle_id=? ORDER BY component_kind",
                (assignment.bundle_id,),
            ).fetchall()
        if (
            (require_assignment and durable_assignment is None)
            or row is None
            or len(components) != len(COMPONENT_KINDS)
        ):
            raise BundleUnavailable("pinned bundle is missing or incomplete")
        bundle = CanonicalBundle(
            row["bundle_id"],
            row["model_id"],
            row["origin"],
            ArtifactChecksum(row["bundle_checksum"]),
            row["feature_contract_version"],
            row["forecast_contract_version"],
            tuple(
                BundleComponent(
                    c["component_name"],
                    c["component_kind"],
                    ArtifactChecksum(c["artifact_checksum"]),
                    c["byte_size"],
                )
                for c in components
            ),
            row["trained_through"],
            row["legacy_model_bundle_id"],
        )
        if (
            bundle.feature_contract_version != SUPPORTED_FEATURE_CONTRACT
            or bundle.forecast_contract_version not in SUPPORTED_FORECAST_CONTRACTS
        ):
            raise BundleUnavailable("pinned bundle contract version is unsupported")
        manifest_bytes = self.artifacts.read(bundle.bundle_checksum)
        if self._object(manifest_bytes, "bundle manifest") != bundle.manifest():
            raise BundleUnavailable("bundle manifest disagrees with authoritative relations")
        content: dict[str, bytes] = {}
        for component in bundle.components:
            value = self.artifacts.read(component.checksum)
            if len(value) != component.byte_size:
                raise BundleUnavailable(f"component size mismatch: {component.name}")
            content[component.kind] = value
        documents = {
            kind: self._object(value, kind.replace("_", " "))
            for kind, value in content.items()
            if kind != "model"
        }
        schema = documents["feature_schema"]
        missingness = documents["missingness_policy"]
        dependencies = documents["dependency_manifest"]
        runtime = documents["runtime_requirements"]
        if (
            schema.get("bundle_id") != bundle.bundle_id
            or schema.get("contract_version") != bundle.feature_contract_version
        ):
            raise BundleUnavailable("feature schema contract disagrees with bundle")
        if (
            type(schema.get("fields")) is not list
            or type(missingness.get("imputation")) is not dict
        ):
            raise BundleUnavailable("feature schema or missingness policy is invalid")
        corpus = documents["training_corpus"]
        validate_training_corpus_manifest(corpus, bundle)
        packages = dependencies.get("packages")
        if type(packages) is not dict or any(
            type(name) is not str or not name or type(version) is not str or not version
            for name, version in packages.items()
        ):
            raise BundleUnavailable("dependency manifest is invalid")
        training = documents["training_configuration"]
        calibration = documents["calibration"]
        evaluation = documents["evaluation"]
        for name in (
            "training_configuration",
            "dependency_manifest",
            "training_corpus",
            "calibration",
            "evaluation",
            "runtime_requirements",
        ):
            if documents[name].get("model_id") != bundle.model_id:
                raise BundleUnavailable(f"{name.replace('_', ' ')} model binding disagrees")
        if (
            missingness.get("bundle_id") != bundle.bundle_id
            or missingness.get("feature_contract_version") != bundle.feature_contract_version
        ):
            raise BundleUnavailable("missingness policy contract disagrees with bundle")
        if (
            not training
            or training.get("feature_contract_version") != bundle.feature_contract_version
            or training.get("forecast_contract_version") != bundle.forecast_contract_version
            or any(type(key) is not str or not key for key in training)
        ):
            raise BundleUnavailable("training configuration is invalid")
        if bundle.forecast_contract_version == ORDERED_FINISH_CONTRACT and (
            training.get("algorithm") != "full-batch-plackett-luce-linear-v1"
            or training.get("optimizer") != "deterministic-full-batch-gradient-ascent"
            or type(training.get("seed")) is not int
            or type(training.get("epochs")) is not int
            or training["epochs"] <= 0
            or type(training.get("learning_rate")) not in (int, float)
            or isinstance(training["learning_rate"], bool)
            or not math.isfinite(training["learning_rate"])
            or training["learning_rate"] <= 0
        ):
            raise BundleUnavailable("ordered-finish training configuration is incomplete")
        if (
            not calibration
            or calibration.get("forecast_contract_version") != bundle.forecast_contract_version
            or not all(calibration.get(key) for key in ("method", "status"))
        ):
            raise BundleUnavailable("calibration metadata is invalid")
        if (
            not evaluation
            or evaluation.get("forecast_contract_version") != bundle.forecast_contract_version
            or type(evaluation.get("population")) is not str
            or not evaluation["population"]
        ):
            raise BundleUnavailable("evaluation metadata is invalid")
        if (
            runtime.get("python_implementation") != platform.python_implementation()
            or runtime.get("python_major_minor")
            != f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}"
        ):
            raise BundleUnavailable("runtime is incompatible")
        for package, expected in dependencies.get("packages", {}).items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as error:
                raise BundleUnavailable(f"dependency unavailable: {package}") from error
            if actual != expected:
                raise BundleUnavailable(f"dependency version mismatch: {package}")
        model = self.deserializer(content["model"])
        if bundle.forecast_contract_version == ORDERED_FINISH_CONTRACT:
            if not callable(getattr(model, "latent_strengths", None)):
                raise BundleUnavailable("ordered-finish model does not implement latent_strengths")
        elif not callable(getattr(model, "predict_proba", None)):
            raise BundleUnavailable("runner-win model does not implement predict_proba")
        return LoadedChampion(bundle, assignment, model, schema, missingness)


def legacy_incumbent_conversion_status() -> Mapping[str, Any]:
    """Source-proven conversion result: quarantine, never fabricated completion."""
    return {
        "model_id": "V4_ExtraTrees_ExtraTreesClassifier_Calibrated_20260329_212033",
        "classification": "sklearn.calibration.CalibratedClassifierCV",
        "status": "quarantined",
        "missing_mandatory_facts": [
            "trained_through",
            "promotion_approved_at",
            "promotion_effective_from_racing_day",
            "promotion_record_id",
            "feature_schema",
            "missingness_policy",
            "dependency_manifest",
            "training_corpus",
            "calibration",
            "evaluation",
            "runtime_requirements",
        ],
    }
