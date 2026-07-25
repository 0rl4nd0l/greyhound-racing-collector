"""Transactional Phase 7 backups and isolated, application-level restore drills."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .artifacts import ArtifactStore, ArtifactStoreError, LocalArtifactStore
from .domain import ArtifactChecksum, OperationId, require_aware
from .operations import BarrierNotSatisfied, SQLiteOperationsStore, iso_timestamp


class RecoveryRejected(RuntimeError):
    """A backup or restore could not prove recoverability."""


def _digest(content: bytes) -> ArtifactChecksum:
    return ArtifactChecksum(f"sha256:{hashlib.sha256(content).hexdigest()}")


ARTIFACT_REFERENCE_CONTRACT = "phase7-artifact-references-v1"
ARTIFACT_REFERENCE_SCHEMA_VERSION = 29

# Closed schema contract: only content-addressed object references belong here.
# Operational/result/database hashes are intentionally absent.
RELATIONAL_ARTIFACT_REFERENCES = (
    ("expected_races", "programme_checksum"),
    ("programme_race_observations", "programme_checksum"),
    ("run_observations", "artifact_checksum"),
    ("field_evidence", "artifact_checksum"),
    ("odds_attempts", "artifact_checksum"),
    ("odds_attempts", "runner_mapping_checksum"),
    ("sealed_evidence", "raw_manifest_checksum"),
    ("sealed_evidence", "normalized_checksum"),
    ("sealed_evidence", "odds_checksum"),
    ("model_bundles", "artifact_checksum"),
    ("model_bundles", "metadata_checksum"),
    ("model_bundles", "scaler_checksum"),
    ("deferred_predictions", "evidence_checksum"),
    ("deferred_predictions", "artifact_checksum"),
    ("prediction_quarantines", "evidence_checksum"),
    ("result_attempts", "artifact_checksum"),
    ("training_examples", "artifact_checksum"),
    ("on_demand_forecasts", "artifact_checksum"),
    ("on_demand_forecasts", "evidence_checksum"),
    ("canonical_model_bundles", "bundle_checksum"),
    ("canonical_model_bundles", "feature_schema_checksum"),
    ("canonical_model_bundles", "missingness_policy_checksum"),
    ("canonical_model_bundles", "training_configuration_checksum"),
    ("canonical_model_bundles", "dependency_manifest_checksum"),
    ("canonical_model_bundles", "training_corpus_checksum"),
    ("canonical_model_bundles", "calibration_checksum"),
    ("canonical_model_bundles", "evaluation_checksum"),
    ("canonical_model_bundles", "runtime_requirements_checksum"),
    ("canonical_bundle_components", "artifact_checksum"),
    ("canonical_training_examples", "evidence_checksum"),
    ("canonical_training_examples", "result_checksum"),
    ("canonical_training_examples", "artifact_checksum"),
    ("phase6_evaluation_evidence", "artifact_checksum"),
    ("phase6_forecast_artifacts", "forecast_checksum"),
    ("phase6_forecast_service_artifacts", "artifact_checksum"),
    ("phase6_policy_registry", "artifact_checksum"),
    ("phase6_probation_states", "state_checksum"),
    ("phase6_probation_days", "reconciliation_checksum"),
    ("phase6_probation_days", "restart_checksum"),
    ("phase6_probation_days", "ordering_checksum"),
    ("phase6_probation_days", "determinism_checksum"),
    ("phase6_trusted_evaluations", "report_checksum"),
    ("phase7_release_configurations", "config_checksum"),
    ("phase7_release_manifests", "manifest_checksum"),
    ("phase7_release_manifests", "config_checksum"),
    ("phase7_operational_evidence", "artifact_checksum"),
    ("phase7_operational_evidence", "manifest_checksum"),
    ("phase7_determinism_executions", "input_checksum"),
    ("phase7_determinism_executions", "output_checksum"),
    ("phase7_day_forecast_cohort_members", "bundle_checksum"),
    ("phase7_day_forecast_cohort_components", "artifact_checksum"),
    ("phase7_reconciliation", "report_checksum"),
    ("phase7_cutover_eligibility", "evidence_checksum"),
    ("phase7_probation_seals", "state_checksum"),
)

JSON_ARTIFACT_REFERENCES = (("phase6_promotion_records", "component_checksums_json"),)


def artifact_inventory(connection: sqlite3.Connection) -> tuple[str, ...]:
    """Resolve the exact versioned artifact-reference contract from a snapshot."""
    schema_version = connection.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]
    if schema_version != ARTIFACT_REFERENCE_SCHEMA_VERSION:
        raise RecoveryRejected(
            "artifact-reference contract does not cover snapshot schema " f"{schema_version!r}"
        )
    references: set[str] = set()
    for table, column in RELATIONAL_ARTIFACT_REFERENCES:
        for (value,) in connection.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'
        ):
            references.add(str(ArtifactChecksum(value)))
    for table, column in JSON_ARTIFACT_REFERENCES:
        for (document,) in connection.execute(
            f'SELECT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'
        ):
            parsed = json.loads(document)
            if not isinstance(parsed, dict):
                raise RecoveryRejected(f"{table}.{column} violates its typed JSON contract")
            references.update(str(ArtifactChecksum(value)) for value in parsed.values())
    return tuple(sorted(references))


class RecoveryAuthority:
    def __init__(self, store: SQLiteOperationsStore, artifacts: ArtifactStore):
        self.store, self.artifacts = store, artifacts

    def backup(
        self,
        operation_id: OperationId,
        *,
        backup_id: str,
        racing_day_id: str,
        snapshot_path: Path,
        replica: LocalArtifactStore,
        at: datetime,
    ) -> ArtifactChecksum:
        """Create a SQLite snapshot after reconciliation, then checksum and replicate references."""
        require_aware(at, "at")
        payload = {
            "backup": backup_id,
            "day": racing_day_id,
            "snapshot": str(snapshot_path.resolve()),
            "at": iso_timestamp(at),
        }
        # sqlite3.Connection.backup must not run from the writer transaction it
        # is copying. A dedicated read connection produces one consistent DB
        # image; publication of its verified identity is a separate atomic op.
        source = self.store._connect()
        try:
            if (
                source.execute(
                    "SELECT 1 FROM phase7_reconciliation WHERE racing_day_id=? AND mismatch_count=0",
                    (racing_day_id,),
                ).fetchone()
                is None
            ):
                raise BarrierNotSatisfied("only a complete reconciled Racing Day may be backed up")
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            target = sqlite3.connect(snapshot_path)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        database_checksum = _digest(snapshot_path.read_bytes())
        recovered = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
        try:
            ordered_checksums = artifact_inventory(recovered)
        finally:
            recovered.close()
        for checksum in ordered_checksums:
            content = self.artifacts.read(ArtifactChecksum(checksum))
            replica.put(
                content,
                media_type="application/octet-stream",
                expected_checksum=ArtifactChecksum(checksum),
            )
        inventory = json.dumps(ordered_checksums, separators=(",", ":")).encode()
        inventory_artifact = replica.put(inventory, media_type="application/json")
        with self.store._operation(operation_id, "phase7_backup", payload) as (db, replay):
            if replay:
                return ArtifactChecksum(
                    db.execute(
                        "SELECT database_checksum FROM phase7_backups WHERE operation_id=?",
                        (str(operation_id),),
                    ).fetchone()[0]
                )
            if (
                db.execute(
                    "SELECT 1 FROM phase7_reconciliation WHERE racing_day_id=? AND mismatch_count=0",
                    (racing_day_id,),
                ).fetchone()
                is None
            ):
                raise BarrierNotSatisfied("reconciliation changed before backup publication")
            db.execute(
                "INSERT INTO phase7_backups VALUES(?,?,?,?,?,?,?)",
                (
                    backup_id,
                    racing_day_id,
                    str(database_checksum),
                    str(inventory_artifact.checksum),
                    iso_timestamp(at),
                    str(operation_id),
                    ARTIFACT_REFERENCE_CONTRACT,
                ),
            )
            return database_checksum

    def restore_drill(
        self,
        operation_id: OperationId,
        *,
        drill_id: str,
        backup_id: str,
        snapshot_path: Path,
        replica: LocalArtifactStore,
        at: datetime,
    ) -> bool:
        """Verify bytes, DB integrity, every replicated artifact, and an application query."""
        require_aware(at, "at")
        payload = {
            "drill": drill_id,
            "backup": backup_id,
            "snapshot": str(snapshot_path.resolve()),
            "at": iso_timestamp(at),
        }
        with self.store._operation(operation_id, "phase7_restore_drill", payload) as (db, replay):
            if replay:
                return bool(
                    db.execute(
                        "SELECT successful FROM phase7_restore_drills WHERE operation_id=?",
                        (str(operation_id),),
                    ).fetchone()[0]
                )
            backup = db.execute(
                "SELECT * FROM phase7_backups WHERE backup_id=?", (backup_id,)
            ).fetchone()
            database_ok = (
                backup is not None
                and snapshot_path.exists()
                and str(_digest(snapshot_path.read_bytes())) == backup["database_checksum"]
                and backup["artifact_reference_contract"] == ARTIFACT_REFERENCE_CONTRACT
            )
            artifacts_ok = False
            readable = False
            if database_ok:
                recovered: sqlite3.Connection | None = None
                try:
                    recovered = sqlite3.connect(f"file:{snapshot_path}?mode=ro", uri=True)
                    recovered.row_factory = sqlite3.Row
                    integrity = recovered.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                    reconciliation = recovered.execute(
                        "SELECT mismatch_count FROM phase7_reconciliation WHERE racing_day_id=?",
                        (backup["racing_day_id"],),
                    ).fetchone()
                    readable = integrity and reconciliation is not None and reconciliation[0] == 0
                    inventory_bytes = replica.read(
                        ArtifactChecksum(backup["artifact_inventory_checksum"])
                    )
                    inventory = json.loads(inventory_bytes)
                    if not isinstance(inventory, list) or any(
                        not isinstance(item, str) for item in inventory
                    ):
                        raise RecoveryRejected("backup artifact inventory has invalid JSON shape")
                    typed_inventory = tuple(str(ArtifactChecksum(item)) for item in inventory)
                    if (
                        typed_inventory != tuple(sorted(set(typed_inventory)))
                        or inventory_bytes
                        != json.dumps(typed_inventory, separators=(",", ":")).encode()
                    ):
                        raise RecoveryRejected(
                            "backup artifact inventory is not canonical sorted unique JSON"
                        )
                    restored_inventory = artifact_inventory(recovered)
                    if typed_inventory != restored_inventory:
                        raise RecoveryRejected(
                            "backup artifact inventory does not exactly match restored snapshot"
                        )
                    artifacts_ok = all(
                        replica.verify(ArtifactChecksum(item)) for item in restored_inventory
                    )
                except (
                    OSError,
                    sqlite3.Error,
                    TypeError,
                    ValueError,
                    KeyError,
                    ArtifactStoreError,
                    RecoveryRejected,
                ):
                    readable = artifacts_ok = False
                finally:
                    if recovered is not None:
                        recovered.close()
            successful = database_ok and artifacts_ok and readable
            db.execute(
                "INSERT INTO phase7_restore_drills VALUES(?,?,?,?,?,?,?,?)",
                (
                    drill_id,
                    backup_id,
                    int(database_ok),
                    int(artifacts_ok),
                    int(readable),
                    int(successful),
                    iso_timestamp(at),
                    str(operation_id),
                ),
            )
            return successful
