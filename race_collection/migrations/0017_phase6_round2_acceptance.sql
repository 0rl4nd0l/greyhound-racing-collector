DROP TRIGGER phase6_forward_evaluation_exact_source;
CREATE VIEW phase6_resolved_field_evidence AS
WITH ranked AS (
 SELECT f.*,
  CASE f.authority
   WHEN 'official_programme' THEN 60 WHEN 'official_jump' THEN 60
   WHEN 'official_card' THEN 50 WHEN 'source_card' THEN 40
   WHEN 'market' THEN 30 WHEN 'embedded_form' THEN 10 ELSE -1 END AS authority_rank,
  CASE WHEN f.field_name IN (
   'identity','race_identity','race_number','runner_identity','runner_set',
   'runner_features','box','venue','field_size','scheduled_jump','actual_jump',
   'jump_time','result_order') THEN 1 ELSE 0 END AS intrinsic_critical
 FROM field_evidence f
 WHERE f.authority IN (
  'official_programme','official_jump','official_card','source_card','market','embedded_form'
 ) AND f.field_name IN (
  'identity','race_identity','race_number','runner_identity','runner_set',
  'runner_features','box','venue','distance','grade','field_size','scheduled_jump',
  'actual_jump','jump_time','result_order'
 )
), highest AS (
 SELECT race_id,field_name,max(authority_rank) AS authority_rank
 FROM ranked GROUP BY race_id,field_name
), top_ranked AS (
 SELECT r.*,
  row_number() OVER (
   PARTITION BY r.race_id,r.field_name
   ORDER BY json(r.value_json),r.authority,r.source,r.intrinsic_critical,r.artifact_checksum
  ) AS winner_number
 FROM ranked r JOIN highest h USING(race_id,field_name,authority_rank)
), top_summary AS (
 SELECT race_id,field_name,count(DISTINCT json(value_json)) AS distinct_top_values
 FROM top_ranked GROUP BY race_id,field_name
)
SELECT winner.race_id,winner.field_name,json(winner.value_json) AS value_json,
       summary.distinct_top_values,winner.intrinsic_critical AS critical
FROM top_ranked winner JOIN top_summary summary USING(race_id,field_name)
WHERE winner.winner_number=1;

CREATE TRIGGER phase6_forward_evaluation_exact_source BEFORE INSERT ON phase6_forward_evaluation_races
WHEN NOT EXISTS (
 SELECT 1 FROM canonical_training_examples c
 JOIN training_examples t ON t.training_example_id=c.phase3_training_example_id
 JOIN races r ON r.race_id=c.race_id
 JOIN phase6_resolved_field_evidence venue ON venue.race_id=r.race_id AND venue.field_name='venue'
 JOIN phase6_resolved_field_evidence distance ON distance.race_id=r.race_id AND distance.field_name='distance'
 JOIN phase6_resolved_field_evidence grade ON grade.race_id=r.race_id AND grade.field_name='grade'
 JOIN phase6_resolved_field_evidence field_size ON field_size.race_id=r.race_id AND field_size.field_name='field_size'
 WHERE c.training_example_id=NEW.training_example_id AND c.race_id=NEW.race_id
  AND c.evidence_checksum=NEW.evidence_checksum AND c.result_checksum=NEW.result_checksum
  AND c.racing_date=NEW.racing_day AND c.artifact_checksum=NEW.training_artifact_checksum
  AND t.eligibility='eligible' AND r.state='training_example_ready'
  AND (venue.critical=0 OR venue.distinct_top_values=1) AND json_extract(venue.value_json,'$')=NEW.venue
  AND (distance.critical=0 OR distance.distinct_top_values=1) AND CAST(json_extract(distance.value_json,'$') AS INTEGER)=NEW.distance_m
  AND (grade.critical=0 OR grade.distinct_top_values=1) AND json_extract(grade.value_json,'$')=NEW.grade
  AND (field_size.critical=0 OR field_size.distinct_top_values=1) AND CAST(json_extract(field_size.value_json,'$') AS INTEGER)=NEW.field_size
)
BEGIN SELECT RAISE(ABORT,'forward evaluation race lacks resolved eligible Phase-5 authority'); END;

CREATE TABLE phase6_bundle_lineage_v2 (
 bundle_id TEXT PRIMARY KEY REFERENCES canonical_model_bundles(bundle_id),
 bundle_registration_operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 registration_run_id TEXT NOT NULL UNIQUE REFERENCES phase6_runs(run_id),
 source_run_id TEXT NOT NULL REFERENCES phase6_runs(run_id),
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 CHECK(operation_id GLOB 'op_*' AND length(operation_id)=35 AND substr(operation_id,4) NOT GLOB '*[^0-9a-f]*')
);

CREATE TABLE phase6_service_computations (
 computation_id TEXT PRIMARY KEY CHECK(length(trim(computation_id))>0),
 race_id TEXT NOT NULL REFERENCES races(race_id),
 bundle_id TEXT NOT NULL REFERENCES canonical_model_bundles(bundle_id),
 bundle_checksum TEXT NOT NULL,
 evidence_checksum TEXT NOT NULL,
 computed_at TEXT NOT NULL,
 service_run_id TEXT NOT NULL REFERENCES phase6_runs(run_id),
 phase3_prediction_id TEXT NOT NULL REFERENCES deferred_predictions(prediction_id),
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 UNIQUE(race_id,bundle_id),
 FOREIGN KEY(bundle_id,bundle_checksum) REFERENCES canonical_model_bundles(bundle_id,bundle_checksum)
);
CREATE TABLE phase6_forecast_computation_bindings (
 forecast_checksum TEXT PRIMARY KEY REFERENCES phase6_forecast_service_artifacts(forecast_checksum),
 computation_id TEXT NOT NULL UNIQUE REFERENCES phase6_service_computations(computation_id)
);
CREATE TRIGGER phase6_service_computation_exact BEFORE INSERT ON phase6_service_computations
WHEN NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
 OR length(NEW.bundle_checksum)<>71 OR substr(NEW.bundle_checksum,1,7)<>'sha256:'
 OR substr(NEW.bundle_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.evidence_checksum)<>71 OR substr(NEW.evidence_checksum,1,7)<>'sha256:'
 OR substr(NEW.evidence_checksum,8) GLOB '*[^0-9a-f]*'
 OR NOT EXISTS (
  SELECT 1 FROM phase6_runs s JOIN deferred_predictions p ON p.prediction_id=NEW.phase3_prediction_id
  WHERE s.run_id=NEW.service_run_id AND s.run_kind='forecast_service'
   AND s.started_at<=NEW.computed_at AND p.race_id=NEW.race_id
   AND p.evidence_checksum=NEW.evidence_checksum AND p.computed_at=NEW.computed_at
 )
BEGIN SELECT RAISE(ABORT,'service computation identity or authority is invalid'); END;
CREATE TRIGGER phase6_service_computation_append_only_update BEFORE UPDATE ON phase6_service_computations BEGIN SELECT RAISE(ABORT,'service computations are append-only'); END;
CREATE TRIGGER phase6_service_computation_append_only_delete BEFORE DELETE ON phase6_service_computations BEGIN SELECT RAISE(ABORT,'service computations are append-only'); END;
CREATE TRIGGER phase6_forecast_binding_append_only_update BEFORE UPDATE ON phase6_forecast_computation_bindings BEGIN SELECT RAISE(ABORT,'forecast bindings are append-only'); END;
CREATE TRIGGER phase6_forecast_binding_append_only_delete BEFORE DELETE ON phase6_forecast_computation_bindings BEGIN SELECT RAISE(ABORT,'forecast bindings are append-only'); END;

CREATE TRIGGER phase6_bundle_lineage_v2_exact BEFORE INSERT ON phase6_bundle_lineage_v2
WHEN NOT EXISTS (
 SELECT 1 FROM canonical_model_bundles b
 JOIN phase6_runs registration ON registration.run_id=NEW.registration_run_id AND registration.run_kind='registration'
 JOIN phase6_runs source ON source.run_id=NEW.source_run_id AND source.run_kind IN ('training','tuning')
 WHERE b.bundle_id=NEW.bundle_id AND b.operation_id=NEW.bundle_registration_operation_id
  AND NEW.registration_run_id=NEW.bundle_registration_operation_id
  AND source.started_at<registration.started_at AND registration.started_at<=b.created_at
)
BEGIN SELECT RAISE(ABORT,'bundle lineage does not match its real registration operation'); END;

CREATE TRIGGER phase6_bundle_lineage_v2_append_only_update BEFORE UPDATE ON phase6_bundle_lineage_v2 BEGIN SELECT RAISE(ABORT,'bundle lineage is append-only'); END;
CREATE TRIGGER phase6_bundle_lineage_v2_append_only_delete BEFORE DELETE ON phase6_bundle_lineage_v2 BEGIN SELECT RAISE(ABORT,'bundle lineage is append-only'); END;

CREATE TRIGGER phase6_policy_exact_artifact BEFORE INSERT ON phase6_policy_registry
WHEN NEW.policy_checksum<>NEW.artifact_checksum
 OR length(NEW.policy_checksum)<>71 OR substr(NEW.policy_checksum,1,7)<>'sha256:'
 OR substr(NEW.policy_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'policy checksum or operation identity is invalid'); END;

CREATE TRIGGER phase6_service_forecast_identity BEFORE INSERT ON phase6_forecast_service_artifacts
WHEN length(NEW.forecast_checksum)<>71 OR substr(NEW.forecast_checksum,1,7)<>'sha256:'
 OR substr(NEW.forecast_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.artifact_checksum)<>71 OR substr(NEW.artifact_checksum,1,7)<>'sha256:'
 OR substr(NEW.artifact_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.bundle_checksum)<>71 OR substr(NEW.bundle_checksum,1,7)<>'sha256:'
 OR substr(NEW.bundle_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.evidence_checksum)<>71 OR substr(NEW.evidence_checksum,1,7)<>'sha256:'
 OR substr(NEW.evidence_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'service forecast checksum or operation identity is invalid'); END;

DROP TRIGGER phase6_forecast_service_before_result;
CREATE TRIGGER phase6_forecast_service_before_result BEFORE INSERT ON phase6_forecast_service_artifacts
WHEN NEW.deferred_prediction_id IS NULL OR NOT EXISTS (
 SELECT 1 FROM phase6_runs s
 JOIN deferred_predictions p ON p.prediction_id=NEW.deferred_prediction_id
 JOIN expected_races e ON e.race_id=NEW.race_id
 JOIN races r ON r.race_id=p.race_id JOIN racing_days d ON d.racing_day_id=r.racing_day_id
 JOIN sealed_evidence z ON z.seal_id=p.seal_id AND z.race_id=p.race_id
 WHERE s.run_id=NEW.service_run_id AND s.run_kind='forecast_service'
  AND s.started_at<=NEW.generated_at
  AND p.race_id=NEW.race_id AND p.evidence_checksum=NEW.evidence_checksum
  AND p.computed_at=NEW.generated_at AND d.closed_at IS NOT NULL AND d.closed_at<=NEW.generated_at
  AND e.programme_checksum IS NOT NULL AND z.normalized_checksum=NEW.evidence_checksum
  AND NOT EXISTS (SELECT 1 FROM result_attempts x WHERE x.race_id=NEW.race_id AND x.attempted_at<=NEW.generated_at)
)
BEGIN SELECT RAISE(ABORT,'forecast lacks its genuine result-blind service authority'); END;

CREATE UNIQUE INDEX phase6_one_promoted_history_per_day ON phase6_assignment_history(effective_racing_day_id) WHERE action='promoted';
CREATE UNIQUE INDEX phase6_one_rollback_history_per_day ON phase6_assignment_history(effective_racing_day_id) WHERE action='rollback_restored';
CREATE TRIGGER phase6_history_exact_chain BEFORE INSERT ON phase6_assignment_history
WHEN (NEW.action='promoted' AND (NEW.prior_history_id IS NOT NULL OR NOT EXISTS (
 SELECT 1 FROM phase6_next_day_assignments n JOIN phase6_promotion_records p USING(promotion_record_id)
 JOIN operations o ON o.operation_id=NEW.operation_id
 WHERE n.effective_racing_day_id=NEW.effective_racing_day_id AND n.assignment_id=NEW.assignment_id
  AND p.next_assignment_id=NEW.assignment_id AND o.kind='phase6_promote_next_day'
))) OR (NEW.action='rollback_restored' AND NOT EXISTS (
 SELECT 1 FROM phase6_rollback_records r JOIN phase6_next_day_assignments n ON n.assignment_id=r.staged_assignment_id
 JOIN phase6_assignment_history prior ON prior.history_id=NEW.prior_history_id
 JOIN operations o ON o.operation_id=NEW.operation_id
 WHERE n.effective_racing_day_id=NEW.effective_racing_day_id
  AND r.restored_assignment_id=NEW.assignment_id AND n.rollback_assignment_id=NEW.assignment_id
  AND prior.effective_racing_day_id=NEW.effective_racing_day_id AND prior.action='promoted'
  AND prior.assignment_id=n.assignment_id AND o.kind='rollback_phase6_staged_assignment'
))
BEGIN SELECT RAISE(ABORT,'assignment history is not the exact authoritative chain'); END;

CREATE TRIGGER phase6_rollback_exact_target BEFORE INSERT ON phase6_rollback_records
WHEN NOT EXISTS (
 SELECT 1 FROM phase6_next_day_assignments n JOIN operations o ON o.operation_id=NEW.operation_id
 WHERE n.assignment_id=NEW.staged_assignment_id AND n.rollback_assignment_id=NEW.restored_assignment_id
  AND o.kind='rollback_phase6_staged_assignment'
)
BEGIN SELECT RAISE(ABORT,'rollback does not restore the exact staged target'); END;

CREATE TRIGGER phase6_service_training_append_only_update BEFORE UPDATE ON phase6_service_training_requests BEGIN SELECT RAISE(ABORT,'service training requests are append-only'); END;
CREATE TRIGGER phase6_service_training_append_only_delete BEFORE DELETE ON phase6_service_training_requests BEGIN SELECT RAISE(ABORT,'service training requests are append-only'); END;
CREATE TRIGGER phase6_service_training_identity BEFORE INSERT ON phase6_service_training_requests
WHEN NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'service training request operation identity is invalid'); END;

CREATE TRIGGER phase6_runs_strict_operation_id BEFORE INSERT ON phase6_runs
WHEN NEW.run_id NOT GLOB 'op_*' OR length(NEW.run_id)<>35
 OR substr(NEW.run_id,4) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id<>NEW.run_id
BEGIN SELECT RAISE(ABORT,'phase6 run operation identity is invalid'); END;

-- Bundles registered before Phase 6 already have the truthful registration
-- authority: their immutable registration operation and creation timestamp.
INSERT OR IGNORE INTO phase6_runs(run_id,run_kind,started_at,operation_id)
SELECT operation_id,'registration',created_at,operation_id
FROM canonical_model_bundles;

CREATE TRIGGER phase6_probation_state_strict BEFORE INSERT ON phase6_probation_states
WHEN length(NEW.state_checksum)<>71 OR substr(NEW.state_checksum,1,7)<>'sha256:'
 OR substr(NEW.state_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'probation state identity is invalid'); END;
CREATE TRIGGER phase6_probation_day_strict BEFORE INSERT ON phase6_probation_days
WHEN length(NEW.reconciliation_checksum)<>71 OR substr(NEW.reconciliation_checksum,1,7)<>'sha256:'
 OR substr(NEW.reconciliation_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.restart_checksum)<>71 OR substr(NEW.restart_checksum,1,7)<>'sha256:'
 OR substr(NEW.restart_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.ordering_checksum)<>71 OR substr(NEW.ordering_checksum,1,7)<>'sha256:'
 OR substr(NEW.ordering_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.determinism_checksum)<>71 OR substr(NEW.determinism_checksum,1,7)<>'sha256:'
 OR substr(NEW.determinism_checksum,8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'probation day checksum is invalid'); END;
CREATE TRIGGER phase6_evaluation_evidence_strict BEFORE INSERT ON phase6_evaluation_evidence
WHEN length(NEW.population_checksum)<>71 OR substr(NEW.population_checksum,1,7)<>'sha256:'
 OR substr(NEW.population_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.artifact_checksum)<>71 OR substr(NEW.artifact_checksum,1,7)<>'sha256:'
 OR substr(NEW.artifact_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'evaluation evidence identity is invalid'); END;
CREATE TRIGGER phase6_forward_race_strict BEFORE INSERT ON phase6_forward_evaluation_races
WHEN length(NEW.evidence_checksum)<>71 OR substr(NEW.evidence_checksum,1,7)<>'sha256:'
 OR substr(NEW.evidence_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.result_checksum)<>71 OR substr(NEW.result_checksum,1,7)<>'sha256:'
 OR substr(NEW.result_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.training_artifact_checksum)<>71 OR substr(NEW.training_artifact_checksum,1,7)<>'sha256:'
 OR substr(NEW.training_artifact_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'forward evaluation identity is invalid'); END;
CREATE TRIGGER phase6_trusted_evaluation_strict BEFORE INSERT ON phase6_trusted_evaluations
WHEN length(NEW.report_checksum)<>71 OR substr(NEW.report_checksum,1,7)<>'sha256:'
 OR substr(NEW.report_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.policy_checksum)<>71 OR substr(NEW.policy_checksum,1,7)<>'sha256:'
 OR substr(NEW.policy_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'trusted evaluation identity is invalid'); END;
CREATE TRIGGER phase6_schedule_strict BEFORE INSERT ON phase6_racing_day_schedule
WHEN length(NEW.programme_checksum)<>71 OR substr(NEW.programme_checksum,1,7)<>'sha256:'
 OR substr(NEW.programme_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'Racing Day schedule identity is invalid'); END;
CREATE TRIGGER phase6_probation_auth_strict BEFORE INSERT ON phase6_probation_day_auth
WHEN length(NEW.programme_checksum)<>71 OR substr(NEW.programme_checksum,1,7)<>'sha256:'
 OR substr(NEW.programme_checksum,8) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'probation programme checksum is invalid'); END;
CREATE TRIGGER phase6_promotion_strict BEFORE INSERT ON phase6_promotion_records
WHEN length(NEW.challenger_bundle_checksum)<>71 OR substr(NEW.challenger_bundle_checksum,1,7)<>'sha256:'
 OR substr(NEW.challenger_bundle_checksum,8) GLOB '*[^0-9a-f]*'
 OR NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
 OR EXISTS (SELECT 1 FROM json_each(NEW.component_checksums_json) value
            WHERE length(value.value)<>71 OR substr(value.value,1,7)<>'sha256:'
             OR substr(value.value,8) GLOB '*[^0-9a-f]*')
BEGIN SELECT RAISE(ABORT,'promotion identity or component checksum is invalid'); END;
CREATE TRIGGER phase6_training_request_strict BEFORE INSERT ON phase6_training_requests
WHEN NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'training request operation identity is invalid'); END;
CREATE TRIGGER phase6_next_assignment_strict BEFORE INSERT ON phase6_next_day_assignments
WHEN NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'next assignment operation identity is invalid'); END;
CREATE TRIGGER phase6_rollback_strict BEFORE INSERT ON phase6_rollback_records
WHEN NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'rollback operation identity is invalid'); END;
CREATE TRIGGER phase6_history_strict BEFORE INSERT ON phase6_assignment_history
WHEN NEW.operation_id NOT GLOB 'op_*' OR length(NEW.operation_id)<>35
 OR substr(NEW.operation_id,4) GLOB '*[^0-9a-f]*'
BEGIN SELECT RAISE(ABORT,'assignment history operation identity is invalid'); END;

DROP TABLE phase6_race_start_authority;

DROP TABLE next_champion_pointer;
CREATE VIEW next_champion_pointer AS
SELECT history.assignment_id,assignment.bundle_id,assignment.bundle_checksum,
       day.local_date AS effective_racing_day,next.rollback_assignment_id,
       history.operation_id
FROM phase6_assignment_history history
JOIN canonical_serving_assignments assignment USING(assignment_id)
JOIN phase6_next_day_assignments next
  ON next.effective_racing_day_id=history.effective_racing_day_id
JOIN racing_days day ON day.racing_day_id=history.effective_racing_day_id
WHERE history.rowid=(
 SELECT max(latest.rowid) FROM phase6_assignment_history latest
 WHERE latest.effective_racing_day_id=history.effective_racing_day_id
);
