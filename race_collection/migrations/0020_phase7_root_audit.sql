DROP TRIGGER phase7_progress_append_only_delete;
DROP TRIGGER phase7_progress_append_only_update;
DROP TRIGGER phase7_progress_exact;
DROP TABLE phase7_scheduler_progress;
CREATE TABLE phase7_scheduler_progress (
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id), phase_ordinal INTEGER NOT NULL CHECK(phase_ordinal BETWEEN 1 AND 9),
 phase_name TEXT NOT NULL, lease_generation INTEGER NOT NULL REFERENCES phase7_scheduler_history(generation),
 command_operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 result_json TEXT NOT NULL CHECK(json_valid(result_json)), result_checksum TEXT NOT NULL,
 completed_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,phase_ordinal), UNIQUE(racing_day_id,phase_name)
);
CREATE TRIGGER phase7_progress_exact BEFORE INSERT ON phase7_scheduler_progress WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_advance_phase')
 OR NEW.command_operation_id=NEW.operation_id
 OR NOT EXISTS (SELECT 1 FROM operations c WHERE c.operation_id=NEW.command_operation_id)
 OR NEW.phase_ordinal<>(SELECT count(*)+1 FROM phase7_scheduler_progress p WHERE p.racing_day_id=NEW.racing_day_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease l WHERE l.singleton=1 AND l.generation=NEW.lease_generation AND l.expires_at>NEW.completed_at)
 OR length(NEW.result_checksum)<>71 OR substr(NEW.result_checksum,1,7)<>'sha256:'
BEGIN SELECT RAISE(ABORT,'scheduler advancement lacks ordered idempotent command authority'); END;
CREATE TRIGGER phase7_progress_append_only_update BEFORE UPDATE ON phase7_scheduler_progress BEGIN SELECT RAISE(ABORT,'scheduler progress is append-only'); END;
CREATE TRIGGER phase7_progress_append_only_delete BEFORE DELETE ON phase7_scheduler_progress BEGIN SELECT RAISE(ABORT,'scheduler progress is append-only'); END;

DROP TRIGGER phase7_day_exact;
DROP TRIGGER phase7_evidence_append_only_delete;
DROP TRIGGER phase7_evidence_append_only_update;
DROP TRIGGER phase7_evidence_authority;
DROP TABLE phase7_operational_evidence;
CREATE TABLE phase7_operational_evidence (
 artifact_checksum TEXT PRIMARY KEY, evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('restart','ordering','determinism','reconciliation','cutover')),
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id), release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 manifest_checksum TEXT NOT NULL UNIQUE, manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json) AND json_type(manifest_json)='object'),
 byte_size INTEGER NOT NULL CHECK(byte_size>0), verified_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id), UNIQUE(evidence_kind,racing_day_id,release_id)
);
CREATE TRIGGER phase7_evidence_authority BEFORE INSERT ON phase7_operational_evidence WHEN
 json_extract(NEW.manifest_json,'$.schema_version')<>'phase7-operational-evidence-v1'
 OR json_extract(NEW.manifest_json,'$.evidence_kind')<>NEW.evidence_kind
 OR json_extract(NEW.manifest_json,'$.racing_day_id')<>NEW.racing_day_id
 OR json_extract(NEW.manifest_json,'$.release_id')<>NEW.release_id
 OR json_extract(NEW.manifest_json,'$.artifact_checksum')<>NEW.artifact_checksum
 OR NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind IN ('phase7_record_operational_evidence','phase7_reconciliation','phase7_cutover_eligibility'))
BEGIN SELECT RAISE(ABORT,'operational evidence lacks exact typed authority'); END;
CREATE TRIGGER phase7_evidence_append_only_update BEFORE UPDATE ON phase7_operational_evidence BEGIN SELECT RAISE(ABORT,'operational evidence is append-only'); END;
CREATE TRIGGER phase7_evidence_append_only_delete BEFORE DELETE ON phase7_operational_evidence BEGIN SELECT RAISE(ABORT,'operational evidence is append-only'); END;
CREATE TRIGGER phase7_day_exact BEFORE INSERT ON phase7_day_evidence WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_reconciliation')
 OR EXISTS (SELECT 1 FROM (SELECT 'reconciliation' kind,NEW.reconciliation_checksum checksum UNION ALL SELECT 'restart',NEW.restart_checksum UNION ALL SELECT 'ordering',NEW.ordering_checksum UNION ALL SELECT 'determinism',NEW.determinism_checksum) required
  WHERE NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a WHERE a.artifact_checksum=required.checksum AND a.evidence_kind=required.kind AND a.racing_day_id=NEW.racing_day_id AND a.release_id=NEW.release_id))
BEGIN SELECT RAISE(ABORT,'Phase 7 day lacks exact day/release evidence'); END;

CREATE TABLE phase7_release_configurations (
 config_checksum TEXT PRIMARY KEY, schema_version TEXT NOT NULL CHECK(schema_version='phase7-config-v1'),
 config_json TEXT NOT NULL CHECK(json_valid(config_json)), service_root TEXT NOT NULL,
 recorded_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_manifest_config_exact BEFORE INSERT ON phase7_release_manifests WHEN NOT EXISTS (
 SELECT 1 FROM phase7_release_configurations c WHERE c.config_checksum=NEW.config_checksum AND c.service_root=NEW.service_root)
BEGIN SELECT RAISE(ABORT,'release manifest lacks exact typed configuration bytes'); END;

ALTER TABLE phase7_release_history ADD COLUMN prior_effective_racing_day_id TEXT REFERENCES racing_days(racing_day_id);
