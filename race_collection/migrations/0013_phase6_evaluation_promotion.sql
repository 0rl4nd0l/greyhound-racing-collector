CREATE TABLE phase6_probation_states (
    probation_id TEXT PRIMARY KEY,
    through_racing_day TEXT NOT NULL CHECK(through_racing_day GLOB '????-??-??'),
    state_checksum TEXT NOT NULL CHECK(length(state_checksum)=71 AND substr(state_checksum,1,7)='sha256:' AND substr(state_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    recorded_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_probation_days (
    probation_id TEXT NOT NULL REFERENCES phase6_probation_states(probation_id),
    racing_day TEXT NOT NULL CHECK(racing_day GLOB '????-??-??'),
    reconciliation_checksum TEXT NOT NULL CHECK(length(reconciliation_checksum)=71 AND substr(reconciliation_checksum,1,7)='sha256:'),
    restart_checksum TEXT NOT NULL CHECK(length(restart_checksum)=71 AND substr(restart_checksum,1,7)='sha256:'),
    ordering_checksum TEXT NOT NULL CHECK(length(ordering_checksum)=71 AND substr(ordering_checksum,1,7)='sha256:'),
    determinism_checksum TEXT NOT NULL CHECK(length(determinism_checksum)=71 AND substr(determinism_checksum,1,7)='sha256:'),
    successful INTEGER NOT NULL CHECK(successful=1),
    PRIMARY KEY(probation_id,racing_day)
);

CREATE TABLE phase6_evaluation_evidence (
    evidence_id TEXT PRIMARY KEY,
    champion_bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
    challenger_bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
    population_checksum TEXT NOT NULL,
    artifact_checksum TEXT NOT NULL UNIQUE,
    policy_id TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    CHECK(champion_bundle_id <> challenger_bundle_id),
    CHECK(length(population_checksum)=71 AND substr(population_checksum,1,7)='sha256:'),
    CHECK(length(artifact_checksum)=71 AND substr(artifact_checksum,1,7)='sha256:')
);

CREATE TABLE phase6_forward_evaluation_races (
    race_id TEXT PRIMARY KEY,
    training_example_id TEXT NOT NULL UNIQUE,
    racing_day TEXT NOT NULL CHECK(racing_day GLOB '????-??-??'),
    evidence_checksum TEXT NOT NULL,
    result_checksum TEXT NOT NULL,
    training_artifact_checksum TEXT NOT NULL,
    venue TEXT NOT NULL CHECK(length(trim(venue)) > 0),
    distance_m INTEGER NOT NULL CHECK(distance_m > 0),
    grade TEXT NOT NULL CHECK(length(trim(grade)) > 0),
    field_size INTEGER NOT NULL CHECK(field_size > 1),
    registered_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TRIGGER phase6_forward_evaluation_exact_source BEFORE INSERT ON phase6_forward_evaluation_races
WHEN NOT EXISTS (
  SELECT 1 FROM canonical_training_examples c
  JOIN training_examples t ON t.training_example_id=c.phase3_training_example_id
  JOIN races r ON r.race_id=c.race_id
  WHERE c.training_example_id=NEW.training_example_id AND c.race_id=NEW.race_id
    AND c.evidence_checksum=NEW.evidence_checksum AND c.result_checksum=NEW.result_checksum
    AND c.racing_date=NEW.racing_day
    AND c.artifact_checksum=NEW.training_artifact_checksum
    AND t.eligibility='eligible' AND r.state='training_example_ready'
    AND EXISTS(SELECT 1 FROM field_evidence f WHERE f.race_id=NEW.race_id AND f.field_name='venue' AND json_extract(f.value_json,'$')=NEW.venue)
    AND EXISTS(SELECT 1 FROM field_evidence f WHERE f.race_id=NEW.race_id AND f.field_name='distance' AND CAST(json_extract(f.value_json,'$') AS INTEGER)=NEW.distance_m)
    AND EXISTS(SELECT 1 FROM field_evidence f WHERE f.race_id=NEW.race_id AND f.field_name='grade' AND json_extract(f.value_json,'$')=NEW.grade)
    AND EXISTS(SELECT 1 FROM field_evidence f WHERE f.race_id=NEW.race_id AND f.field_name='field_size' AND CAST(json_extract(f.value_json,'$') AS INTEGER)=NEW.field_size)
)
BEGIN SELECT RAISE(ABORT,'forward evaluation race lacks eligible Phase-5 source'); END;

CREATE TABLE phase6_promotion_records (
    promotion_record_id TEXT PRIMARY KEY,
    evidence_id TEXT NOT NULL UNIQUE REFERENCES phase6_evaluation_evidence(evidence_id),
    prior_assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
    next_assignment_id TEXT NOT NULL UNIQUE REFERENCES canonical_serving_assignments(assignment_id),
    challenger_bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
    challenger_bundle_checksum TEXT NOT NULL,
    component_checksums_json TEXT NOT NULL,
    approved_at TEXT NOT NULL,
    effective_racing_day TEXT NOT NULL CHECK(effective_racing_day GLOB '????-??-??'),
    approver TEXT NOT NULL CHECK(length(trim(approver)) > 0),
    policy_id TEXT NOT NULL,
    reason TEXT NOT NULL CHECK(length(trim(reason)) > 0),
    probation_id TEXT NOT NULL REFERENCES phase6_probation_states(probation_id),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase6_training_requests (
    training_request_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    evidence_id TEXT REFERENCES phase6_evaluation_evidence(evidence_id),
    requested_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE next_champion_pointer (
    assignment_id TEXT PRIMARY KEY REFERENCES canonical_serving_assignments(assignment_id),
    bundle_id TEXT NOT NULL,
    bundle_checksum TEXT NOT NULL,
    effective_racing_day TEXT NOT NULL,
    rollback_assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    UNIQUE(effective_racing_day),
    FOREIGN KEY(assignment_id,bundle_id,bundle_checksum) REFERENCES canonical_serving_assignments(assignment_id,bundle_id,bundle_checksum)
);

CREATE TRIGGER phase6_probation_append_only_update BEFORE UPDATE ON phase6_probation_states BEGIN SELECT RAISE(ABORT,'probation states are append-only'); END;
CREATE TRIGGER phase6_probation_append_only_delete BEFORE DELETE ON phase6_probation_states BEGIN SELECT RAISE(ABORT,'probation states are append-only'); END;
CREATE TRIGGER phase6_probation_days_append_only_update BEFORE UPDATE ON phase6_probation_days BEGIN SELECT RAISE(ABORT,'probation days are append-only'); END;
CREATE TRIGGER phase6_probation_days_append_only_delete BEFORE DELETE ON phase6_probation_days BEGIN SELECT RAISE(ABORT,'probation days are append-only'); END;
CREATE TRIGGER phase6_evidence_append_only_update BEFORE UPDATE ON phase6_evaluation_evidence BEGIN SELECT RAISE(ABORT,'evaluation evidence is append-only'); END;
CREATE TRIGGER phase6_evidence_append_only_delete BEFORE DELETE ON phase6_evaluation_evidence BEGIN SELECT RAISE(ABORT,'evaluation evidence is append-only'); END;
CREATE TRIGGER phase6_forward_evaluation_append_only_update BEFORE UPDATE ON phase6_forward_evaluation_races BEGIN SELECT RAISE(ABORT,'forward evaluation races are append-only'); END;
CREATE TRIGGER phase6_forward_evaluation_append_only_delete BEFORE DELETE ON phase6_forward_evaluation_races BEGIN SELECT RAISE(ABORT,'forward evaluation races are append-only'); END;
CREATE TRIGGER phase6_promotion_append_only_update BEFORE UPDATE ON phase6_promotion_records BEGIN SELECT RAISE(ABORT,'promotion records are append-only'); END;
CREATE TRIGGER phase6_promotion_append_only_delete BEFORE DELETE ON phase6_promotion_records BEGIN SELECT RAISE(ABORT,'promotion records are append-only'); END;
CREATE TRIGGER phase6_training_requests_append_only_update BEFORE UPDATE ON phase6_training_requests BEGIN SELECT RAISE(ABORT,'training requests are append-only'); END;
CREATE TRIGGER phase6_training_requests_append_only_delete BEFORE DELETE ON phase6_training_requests BEGIN SELECT RAISE(ABORT,'training requests are append-only'); END;

CREATE TRIGGER next_champion_pointer_complete BEFORE INSERT ON next_champion_pointer
WHEN NOT EXISTS (
  SELECT 1 FROM phase6_promotion_records p
  JOIN canonical_serving_assignments a ON a.assignment_id=p.next_assignment_id
  WHERE p.prior_assignment_id=NEW.rollback_assignment_id AND a.assignment_id=NEW.assignment_id
    AND a.bundle_id=NEW.bundle_id AND a.bundle_checksum=NEW.bundle_checksum
    AND p.effective_racing_day=NEW.effective_racing_day
)
BEGIN SELECT RAISE(ABORT,'next champion pointer lacks complete promotion record'); END;
CREATE TRIGGER next_champion_pointer_append_only_update BEFORE UPDATE ON next_champion_pointer BEGIN SELECT RAISE(ABORT,'next champion pointer is append-only'); END;
CREATE TRIGGER next_champion_pointer_append_only_delete BEFORE DELETE ON next_champion_pointer BEGIN SELECT RAISE(ABORT,'next champion pointer is append-only'); END;
