CREATE TABLE canonical_model_bundles (
    bundle_id TEXT PRIMARY KEY CHECK(length(trim(bundle_id)) > 0),
    model_id TEXT NOT NULL CHECK(length(trim(model_id)) > 0),
    origin TEXT NOT NULL CHECK(origin IN ('canonical','legacy-origin')),
    legacy_model_bundle_id TEXT REFERENCES model_bundles(bundle_id),
    bundle_checksum TEXT NOT NULL UNIQUE CHECK(length(bundle_checksum)=71 AND substr(bundle_checksum,1,7)='sha256:' AND substr(bundle_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    feature_contract_version TEXT NOT NULL,
    forecast_contract_version TEXT NOT NULL,
    feature_schema_checksum TEXT NOT NULL CHECK(length(feature_schema_checksum)=71 AND substr(feature_schema_checksum,1,7)='sha256:' AND substr(feature_schema_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    missingness_policy_checksum TEXT NOT NULL CHECK(length(missingness_policy_checksum)=71 AND substr(missingness_policy_checksum,1,7)='sha256:' AND substr(missingness_policy_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    training_configuration_checksum TEXT NOT NULL CHECK(length(training_configuration_checksum)=71 AND substr(training_configuration_checksum,1,7)='sha256:' AND substr(training_configuration_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    dependency_manifest_checksum TEXT NOT NULL CHECK(length(dependency_manifest_checksum)=71 AND substr(dependency_manifest_checksum,1,7)='sha256:' AND substr(dependency_manifest_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    training_corpus_checksum TEXT NOT NULL CHECK(length(training_corpus_checksum)=71 AND substr(training_corpus_checksum,1,7)='sha256:' AND substr(training_corpus_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    trained_through TEXT NOT NULL CHECK(length(trim(trained_through)) > 0 AND lower(trained_through) <> 'unknown'),
    calibration_checksum TEXT NOT NULL CHECK(length(calibration_checksum)=71 AND substr(calibration_checksum,1,7)='sha256:' AND substr(calibration_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    evaluation_checksum TEXT NOT NULL CHECK(length(evaluation_checksum)=71 AND substr(evaluation_checksum,1,7)='sha256:' AND substr(evaluation_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    runtime_requirements_checksum TEXT NOT NULL CHECK(length(runtime_requirements_checksum)=71 AND substr(runtime_requirements_checksum,1,7)='sha256:' AND substr(runtime_requirements_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
    ,UNIQUE(bundle_id,bundle_checksum),
    CHECK((origin='canonical' AND legacy_model_bundle_id IS NULL) OR
          (origin='legacy-origin' AND legacy_model_bundle_id IS NOT NULL))
);

CREATE TABLE canonical_bundle_components (
    bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
    component_name TEXT NOT NULL,
    component_kind TEXT NOT NULL CHECK(component_kind IN ('model','feature_schema','missingness_policy','training_configuration','dependency_manifest','training_corpus','calibration','evaluation','runtime_requirements')),
    artifact_checksum TEXT NOT NULL CHECK(length(artifact_checksum)=71 AND substr(artifact_checksum,1,7)='sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    byte_size INTEGER NOT NULL CHECK(byte_size > 0),
    PRIMARY KEY(bundle_id, component_name),
    UNIQUE(bundle_id, component_kind)
);

CREATE TABLE canonical_serving_assignments (
    assignment_id TEXT PRIMARY KEY CHECK(length(trim(assignment_id)) > 0),
    bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
    bundle_checksum TEXT NOT NULL,
    promotion_approved_at TEXT NOT NULL CHECK(length(trim(promotion_approved_at)) > 0 AND lower(promotion_approved_at) <> 'unknown'),
    promotion_effective_from_racing_day TEXT NOT NULL CHECK(length(trim(promotion_effective_from_racing_day)) > 0 AND lower(promotion_effective_from_racing_day) <> 'unknown'),
    promotion_record_id TEXT NOT NULL CHECK(length(trim(promotion_record_id)) > 0 AND lower(promotion_record_id) <> 'unknown'),
    assigned_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    UNIQUE(assignment_id,bundle_id,bundle_checksum),
    FOREIGN KEY(bundle_id,bundle_checksum) REFERENCES canonical_model_bundles(bundle_id,bundle_checksum)
);

CREATE TABLE canonical_day_assignments (
    racing_day_id TEXT PRIMARY KEY REFERENCES racing_day_pins(racing_day_id),
    assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
    bundle_id TEXT NOT NULL,
    bundle_checksum TEXT NOT NULL,
    bound_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    FOREIGN KEY(assignment_id,bundle_id,bundle_checksum)
      REFERENCES canonical_serving_assignments(assignment_id,bundle_id,bundle_checksum)
);

CREATE TABLE champion_pointer (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    assignment_id TEXT NOT NULL UNIQUE REFERENCES canonical_serving_assignments(assignment_id),
    bundle_id TEXT NOT NULL UNIQUE,
    bundle_checksum TEXT NOT NULL,
    set_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    FOREIGN KEY(assignment_id,bundle_id,bundle_checksum)
      REFERENCES canonical_serving_assignments(assignment_id,bundle_id,bundle_checksum)
);

CREATE UNIQUE INDEX canonical_bundle_identity ON canonical_model_bundles(bundle_id,bundle_checksum);
CREATE UNIQUE INDEX canonical_legacy_binding_unique
  ON canonical_model_bundles(legacy_model_bundle_id)
  WHERE legacy_model_bundle_id IS NOT NULL;

CREATE TRIGGER canonical_model_bundles_append_only_update BEFORE UPDATE ON canonical_model_bundles BEGIN SELECT RAISE(ABORT,'canonical model bundles are append-only'); END;
CREATE TRIGGER canonical_model_bundles_append_only_delete BEFORE DELETE ON canonical_model_bundles BEGIN SELECT RAISE(ABORT,'canonical model bundles are append-only'); END;
CREATE TRIGGER canonical_bundle_components_append_only_update BEFORE UPDATE ON canonical_bundle_components BEGIN SELECT RAISE(ABORT,'canonical bundle components are append-only'); END;
CREATE TRIGGER canonical_bundle_components_append_only_delete BEFORE DELETE ON canonical_bundle_components BEGIN SELECT RAISE(ABORT,'canonical bundle components are append-only'); END;
CREATE TRIGGER canonical_serving_assignments_append_only_update BEFORE UPDATE ON canonical_serving_assignments BEGIN SELECT RAISE(ABORT,'canonical serving assignments are append-only'); END;
CREATE TRIGGER canonical_serving_assignments_append_only_delete BEFORE DELETE ON canonical_serving_assignments BEGIN SELECT RAISE(ABORT,'canonical serving assignments are append-only'); END;
CREATE TRIGGER canonical_day_assignments_append_only_update BEFORE UPDATE ON canonical_day_assignments BEGIN SELECT RAISE(ABORT,'canonical day assignments are append-only'); END;
CREATE TRIGGER canonical_day_assignments_append_only_delete BEFORE DELETE ON canonical_day_assignments BEGIN SELECT RAISE(ABORT,'canonical day assignments are append-only'); END;
CREATE TRIGGER champion_pointer_append_only_update BEFORE UPDATE ON champion_pointer BEGIN SELECT RAISE(ABORT,'champion pointer is append-only in phase 4'); END;
CREATE TRIGGER champion_pointer_append_only_delete BEFORE DELETE ON champion_pointer BEGIN SELECT RAISE(ABORT,'champion pointer is append-only'); END;

CREATE TRIGGER canonical_bundle_complete_components BEFORE INSERT ON canonical_serving_assignments
WHEN (SELECT COUNT(*) FROM canonical_bundle_components c WHERE c.bundle_id=NEW.bundle_id) <> 9
 OR NOT EXISTS (SELECT 1 FROM canonical_model_bundles b WHERE b.bundle_id=NEW.bundle_id AND b.bundle_checksum=NEW.bundle_checksum
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='feature_schema' AND c.artifact_checksum=b.feature_schema_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='missingness_policy' AND c.artifact_checksum=b.missingness_policy_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='training_configuration' AND c.artifact_checksum=b.training_configuration_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='dependency_manifest' AND c.artifact_checksum=b.dependency_manifest_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='training_corpus' AND c.artifact_checksum=b.training_corpus_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='calibration' AND c.artifact_checksum=b.calibration_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='evaluation' AND c.artifact_checksum=b.evaluation_checksum)
   AND EXISTS(SELECT 1 FROM canonical_bundle_components c WHERE c.bundle_id=b.bundle_id AND c.component_kind='runtime_requirements' AND c.artifact_checksum=b.runtime_requirements_checksum))
BEGIN SELECT RAISE(ABORT,'champion pointer bundle relations disagree'); END;

CREATE TRIGGER canonical_legacy_bundle_authenticates_phase3 BEFORE INSERT ON canonical_model_bundles
WHEN NEW.origin='legacy-origin' AND NOT EXISTS (
  SELECT 1 FROM model_bundles b WHERE b.bundle_id=NEW.legacy_model_bundle_id AND b.model_id=NEW.model_id)
BEGIN SELECT RAISE(ABORT,'legacy canonical bundle does not authenticate phase-3 identity'); END;

CREATE TRIGGER canonical_model_component_authenticates_phase3 BEFORE INSERT ON canonical_bundle_components
WHEN NEW.component_kind='model'
AND EXISTS (SELECT 1 FROM canonical_model_bundles c WHERE c.bundle_id=NEW.bundle_id AND c.origin='legacy-origin')
AND NOT EXISTS (
  SELECT 1 FROM canonical_model_bundles c JOIN model_bundles b ON b.bundle_id=c.legacy_model_bundle_id
  WHERE c.bundle_id=NEW.bundle_id AND c.origin='legacy-origin'
    AND b.artifact_checksum=NEW.artifact_checksum AND b.artifact_size=NEW.byte_size
)
BEGIN SELECT RAISE(ABORT,'legacy canonical model component does not authenticate phase-3 artifact'); END;

CREATE TRIGGER canonical_day_assignment_exact_pin BEFORE INSERT ON canonical_day_assignments
WHEN NOT EXISTS (
  SELECT 1 FROM canonical_serving_assignments a
  JOIN canonical_model_bundles c ON c.bundle_id=a.bundle_id AND c.bundle_checksum=a.bundle_checksum
  JOIN racing_day_pins p ON p.racing_day_id=NEW.racing_day_id
  JOIN model_releases r ON r.release_id=p.release_id AND r.bundle_id=p.bundle_id AND r.policy_id=p.policy_id
  JOIN model_bundles b ON b.bundle_id=p.bundle_id
  JOIN canonical_bundle_components component ON component.bundle_id=c.bundle_id AND component.component_kind='model'
  WHERE a.assignment_id=NEW.assignment_id AND a.bundle_id=NEW.bundle_id
    AND a.bundle_checksum=NEW.bundle_checksum AND c.origin='legacy-origin'
    AND c.legacy_model_bundle_id=p.bundle_id
    AND component.artifact_checksum=b.artifact_checksum AND component.byte_size=b.artifact_size
)
BEGIN SELECT RAISE(ABORT,'canonical day assignment disagrees with immutable pin'); END;
