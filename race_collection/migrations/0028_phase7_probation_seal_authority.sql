-- The supported populated 17->latest path may contain immutable canonical
-- registrations committed after migration 17 was applied. Reassert the same
-- truthful, idempotent provenance backfill introduced by migration 17.
INSERT OR IGNORE INTO phase6_runs(run_id,run_kind,started_at,operation_id)
SELECT operation_id,'registration',created_at,operation_id
FROM canonical_model_bundles;

CREATE TABLE phase7_phase6_probation_authority (
 probation_id TEXT PRIMARY KEY,
 generation INTEGER NOT NULL UNIQUE,
 candidate_release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 activation_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 seal_operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 state_checksum TEXT NOT NULL UNIQUE,
 CHECK(length(state_checksum)=71
  AND state_checksum GLOB 'sha256:[0-9a-f]*')
);
CREATE TRIGGER phase7_phase6_probation_authority_exact
BEFORE INSERT ON phase7_phase6_probation_authority WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.seal_operation_id
  AND o.kind='phase7_seal_probation')
 OR NOT EXISTS (SELECT 1 FROM phase7_probation_control c WHERE c.singleton=1
  AND c.generation=NEW.generation AND c.state='complete')
 OR NOT EXISTS (SELECT 1 FROM phase7_release_pointer p
  JOIN phase7_release_history h ON h.operation_id=NEW.activation_operation_id
  WHERE p.singleton=1 AND p.authority='race_collection_service'
   AND p.release_id=NEW.candidate_release_id
   AND h.action='activate' AND h.release_id=NEW.candidate_release_id
   AND h.effective_racing_day_id=p.effective_racing_day_id)
 OR (SELECT count(*) FROM phase7_probation_acceptances a
     WHERE a.generation=NEW.generation)<>14
BEGIN SELECT RAISE(ABORT,'Phase 6 probation authority lacks exact Phase 7 chain'); END;
INSERT INTO phase7_phase6_probation_authority
SELECT probation_id,generation,release_id,cutover_operation_id,operation_id,state_checksum
FROM phase7_probation_seals;
CREATE TABLE phase7_probation_migration_validation (
 orphan_count INTEGER NOT NULL CHECK(orphan_count=0)
);
INSERT INTO phase7_probation_migration_validation SELECT count(*) FROM (
 SELECT 1 FROM phase6_probation_states state
 LEFT JOIN phase7_phase6_probation_authority authority
  ON authority.probation_id=state.probation_id
  AND authority.state_checksum=state.state_checksum
  AND authority.seal_operation_id=state.operation_id
 WHERE authority.probation_id IS NULL);
DROP TABLE phase7_probation_migration_validation;
CREATE TRIGGER phase7_phase6_probation_authority_append_only_update
BEFORE UPDATE ON phase7_phase6_probation_authority
BEGIN SELECT RAISE(ABORT,'Phase 6 probation authority is append-only'); END;
CREATE TRIGGER phase7_phase6_probation_authority_append_only_delete
BEFORE DELETE ON phase7_phase6_probation_authority
BEGIN SELECT RAISE(ABORT,'Phase 6 probation authority is append-only'); END;

CREATE TRIGGER phase7_probation_state_to_phase6
BEFORE INSERT ON phase6_probation_states WHEN NOT EXISTS (
 SELECT 1 FROM phase7_phase6_probation_authority a
 WHERE a.probation_id=NEW.probation_id
  AND a.state_checksum=NEW.state_checksum
  AND a.seal_operation_id=NEW.operation_id)
BEGIN SELECT RAISE(ABORT,'probation state was not issued by exact Phase 7 authority'); END;

DROP TRIGGER phase7_probation_to_phase6;
CREATE TRIGGER phase7_probation_to_phase6
BEFORE INSERT ON phase6_probation_days WHEN
 length(NEW.reconciliation_checksum)<>71
 OR substr(NEW.reconciliation_checksum,1,7)<>'sha256:'
 OR substr(NEW.reconciliation_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.restart_checksum)<>71
 OR substr(NEW.restart_checksum,1,7)<>'sha256:'
 OR substr(NEW.restart_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.ordering_checksum)<>71
 OR substr(NEW.ordering_checksum,1,7)<>'sha256:'
 OR substr(NEW.ordering_checksum,8) GLOB '*[^0-9a-f]*'
 OR length(NEW.determinism_checksum)<>71
 OR substr(NEW.determinism_checksum,1,7)<>'sha256:'
 OR substr(NEW.determinism_checksum,8) GLOB '*[^0-9a-f]*'
 OR NOT EXISTS (
  SELECT 1 FROM phase7_phase6_probation_authority authority
  JOIN phase7_probation_acceptances a ON a.generation=authority.generation
  JOIN phase7_day_evidence e USING(racing_day_id)
  WHERE authority.probation_id=NEW.probation_id
   AND a.local_date=NEW.racing_day
   AND e.reconciliation_checksum=NEW.reconciliation_checksum
   AND e.restart_checksum=NEW.restart_checksum
   AND e.ordering_checksum=NEW.ordering_checksum
   AND e.determinism_checksum=NEW.determinism_checksum)
BEGIN SELECT RAISE(ABORT,'probation evidence was not issued by Phase 7 authority'); END;

DROP TRIGGER phase6_probation_auth_exact_day;
CREATE TRIGGER phase6_probation_auth_exact_day
BEFORE INSERT ON phase6_probation_day_auth WHEN NOT EXISTS (
 SELECT 1 FROM phase7_phase6_probation_authority authority
 JOIN phase7_probation_acceptances accepted
  ON accepted.generation=authority.generation
 JOIN phase6_probation_days day
  ON day.probation_id=authority.probation_id
  AND day.racing_day=accepted.local_date
 WHERE authority.probation_id=NEW.probation_id
  AND accepted.racing_day_id=NEW.racing_day_id
  AND accepted.programme_checksum=NEW.programme_checksum)
BEGIN SELECT RAISE(ABORT,'probation day authentication lacks exact Phase 7 chain'); END;

CREATE TABLE phase7_legacy_retirement_eligibility (
 eligibility_id TEXT PRIMARY KEY,
 probation_id TEXT NOT NULL UNIQUE,
 probation_generation INTEGER NOT NULL UNIQUE,
 candidate_release_id TEXT NOT NULL,
 legacy_release_id TEXT NOT NULL,
 activation_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 probation_seal_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 probation_state_checksum TEXT NOT NULL,
 recorded_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 CHECK(candidate_release_id<>legacy_release_id)
);
CREATE TRIGGER phase7_legacy_retirement_eligibility_exact
BEFORE INSERT ON phase7_legacy_retirement_eligibility WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_record_legacy_retirement_eligibility')
 OR NOT EXISTS (
  SELECT 1 FROM phase7_probation_seals seal
  JOIN phase7_probation_control control ON control.singleton=1
   AND control.generation=seal.generation AND control.state='complete'
  JOIN phase7_release_pointer pointer ON pointer.singleton=1
   AND pointer.authority='race_collection_service'
   AND pointer.release_id=seal.release_id
  JOIN phase7_release_history activation
   ON activation.operation_id=seal.cutover_operation_id
   AND activation.action='activate'
   AND activation.release_id=seal.release_id
  WHERE seal.probation_id=NEW.probation_id
   AND seal.generation=NEW.probation_generation
   AND seal.release_id=NEW.candidate_release_id
   AND seal.cutover_operation_id=NEW.activation_operation_id
   AND seal.operation_id=NEW.probation_seal_operation_id
   AND seal.state_checksum=NEW.probation_state_checksum
   AND activation.prior_authority='legacy'
   AND activation.prior_release_id=NEW.legacy_release_id
   AND pointer.legacy_preserved=1)
BEGIN SELECT RAISE(ABORT,'retirement eligibility lacks exact sealed probation authority'); END;
CREATE TRIGGER phase7_legacy_retirement_eligibility_append_only_update
BEFORE UPDATE ON phase7_legacy_retirement_eligibility
BEGIN SELECT RAISE(ABORT,'retirement eligibility is append-only'); END;
CREATE TRIGGER phase7_legacy_retirement_eligibility_append_only_delete
BEFORE DELETE ON phase7_legacy_retirement_eligibility
BEGIN SELECT RAISE(ABORT,'retirement eligibility is append-only'); END;

DROP TRIGGER phase7_admin_audit_exact;
CREATE TRIGGER phase7_admin_audit_exact BEFORE INSERT ON phase7_admin_audit WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND (
  (NEW.command IN ('pause','resume') AND o.kind='phase7_admin_pause') OR
  (NEW.command='resolve_alert' AND o.kind='phase7_resolve_alert') OR
  (NEW.command='reset' AND o.kind='phase7_reset_probation') OR
  (NEW.command='initialize_legacy' AND o.kind='phase7_initialize_legacy') OR
  (NEW.command='activate' AND o.kind='phase7_activate_release') OR
  (NEW.command='rollback' AND o.kind='phase7_rollback_release') OR
  (NEW.command='record_legacy_retirement_eligibility'
   AND o.kind='phase7_record_legacy_retirement_eligibility')))
BEGIN SELECT RAISE(ABORT,'admin audit lacks exact application command authority'); END;

CREATE TABLE phase7_day_command_plan (
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 phase_ordinal INTEGER NOT NULL CHECK(phase_ordinal BETWEEN 1 AND 9),
 phase_name TEXT NOT NULL,
 command_operation_id TEXT NOT NULL UNIQUE,
 lease_generation INTEGER NOT NULL,
 planned_at TEXT NOT NULL,
 operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,phase_ordinal),
 UNIQUE(racing_day_id,phase_name)
);

CREATE TABLE phase7_scheduler_renewals (
 operation_id TEXT PRIMARY KEY REFERENCES operations(operation_id),
 lease_generation INTEGER NOT NULL,
 lease_token TEXT NOT NULL,
 renewed_at TEXT NOT NULL,
 expires_at TEXT NOT NULL
);
CREATE TRIGGER phase7_scheduler_lease_renewal_monotonic
BEFORE UPDATE ON phase7_scheduler_lease WHEN
 NEW.owner_id<>OLD.owner_id OR NEW.lease_token<>OLD.lease_token
 OR NEW.acquired_at<>OLD.acquired_at OR NEW.generation<>OLD.generation
 OR NEW.operation_id<>OLD.operation_id OR NEW.expires_at<=OLD.expires_at
BEGIN SELECT RAISE(ABORT,'scheduler lease renewal must monotonically extend expiry'); END;
CREATE TRIGGER phase7_scheduler_renewal_exact
BEFORE INSERT ON phase7_scheduler_renewals WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_renew_scheduler_lease')
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease l WHERE l.singleton=1
  AND l.generation=NEW.lease_generation AND l.lease_token=NEW.lease_token
  AND l.expires_at=NEW.expires_at)
 OR EXISTS (SELECT 1 FROM phase7_scheduler_history h
  WHERE h.generation=NEW.lease_generation
   AND (NEW.renewed_at<h.acquired_at OR NEW.expires_at<=h.expires_at))
 OR EXISTS (SELECT 1 FROM phase7_scheduler_renewals prior
  WHERE prior.lease_generation=NEW.lease_generation
   AND (NEW.renewed_at<=prior.renewed_at OR NEW.expires_at<=prior.expires_at))
 OR NEW.expires_at<=NEW.renewed_at
BEGIN SELECT RAISE(ABORT,'scheduler renewal lacks exact lease authority'); END;
CREATE TRIGGER phase7_scheduler_renewals_append_only_update
BEFORE UPDATE ON phase7_scheduler_renewals
BEGIN SELECT RAISE(ABORT,'scheduler renewals are append-only'); END;
CREATE TRIGGER phase7_scheduler_renewals_append_only_delete
BEFORE DELETE ON phase7_scheduler_renewals
BEGIN SELECT RAISE(ABORT,'scheduler renewals are append-only'); END;

CREATE TRIGGER phase7_day_command_plan_append_only_update
BEFORE UPDATE ON phase7_day_command_plan
BEGIN SELECT RAISE(ABORT,'day command plans are append-only'); END;
CREATE TRIGGER phase7_day_command_plan_append_only_delete
BEFORE DELETE ON phase7_day_command_plan
BEGIN SELECT RAISE(ABORT,'day command plans are append-only'); END;

-- Version 27 had exact, append-only completed prefixes but no full-day plan.
-- Materialise the only truth it can support: retain every completed identity
-- and define the unstarted suffix explicitly as deterministic migration work.
-- The canonical JSON below is key-sorted and compact, matching _payload_hash.
CREATE TEMP TABLE phase7_v27_plan_assertion (
 exact INTEGER NOT NULL CHECK(exact=1)
);
INSERT INTO phase7_v27_plan_assertion
SELECT CASE WHEN EXISTS (
 SELECT 1
 FROM phase7_scheduler_progress p
 LEFT JOIN operations advance ON advance.operation_id=p.operation_id
 LEFT JOIN operations command ON command.operation_id=p.command_operation_id
 LEFT JOIN phase7_application_command_receipts receipt
  ON receipt.command_operation_id=p.command_operation_id
 LEFT JOIN phase7_scheduler_history lease ON lease.generation=p.lease_generation
 WHERE advance.kind IS NOT 'phase7_advance_phase'
    OR command.kind IS NOT 'phase7_command_'||p.phase_name
    OR receipt.racing_day_id IS NOT p.racing_day_id
    OR receipt.phase_name IS NOT p.phase_name
    OR receipt.result_json IS NOT p.result_json
    OR receipt.result_checksum IS NOT p.result_checksum
    OR lease.acquired_at IS NULL OR lease.acquired_at>p.completed_at
    OR lease.expires_at<=p.completed_at
    OR p.phase_name IS NOT CASE p.phase_ordinal
      WHEN 1 THEN 'discover_programme' WHEN 2 THEN 'collect_cards_and_form'
      WHEN 3 THEN 'collect_adaptive_odds' WHEN 4 THEN 'close_and_seal'
      WHEN 5 THEN 'deferred_prediction' WHEN 6 THEN 'collect_results'
      WHEN 7 THEN 'join_training_examples' WHEN 8 THEN 'reconcile'
      WHEN 9 THEN 'request_training' END
    OR p.phase_ordinal<>(SELECT count(*) FROM phase7_scheduler_progress prefix
       WHERE prefix.racing_day_id=p.racing_day_id
        AND prefix.phase_ordinal<=p.phase_ordinal)
) THEN 0 ELSE 1 END;

CREATE TEMP TABLE phase7_v27_plan_rows AS
WITH RECURSIVE ordinals(ordinal) AS (
 VALUES(1) UNION ALL SELECT ordinal+1 FROM ordinals WHERE ordinal<9
),
days AS (
 SELECT racing_day_id,max(completed_at) suffix_planned_at
 FROM phase7_scheduler_progress GROUP BY racing_day_id
)
SELECT d.racing_day_id,o.ordinal AS phase_ordinal,
 CASE o.ordinal
  WHEN 1 THEN 'discover_programme' WHEN 2 THEN 'collect_cards_and_form'
  WHEN 3 THEN 'collect_adaptive_odds' WHEN 4 THEN 'close_and_seal'
  WHEN 5 THEN 'deferred_prediction' WHEN 6 THEN 'collect_results'
  WHEN 7 THEN 'join_training_examples' WHEN 8 THEN 'reconcile'
  WHEN 9 THEN 'request_training' END AS phase_name,
 COALESCE(p.command_operation_id,
  'op_'||substr(sha256_text(
   'phase7-v27-suffix-command-v1:'||d.racing_day_id||':'||o.ordinal),1,32)
 ) AS command_operation_id,
 COALESCE(p.lease_generation,(SELECT lease_generation
   FROM phase7_scheduler_progress last WHERE last.racing_day_id=d.racing_day_id
   ORDER BY phase_ordinal DESC LIMIT 1)) AS lease_generation,
 COALESCE(p.completed_at,d.suffix_planned_at) AS planned_at,
 p.phase_ordinal IS NOT NULL AS completed,
 'op_'||substr(sha256_text(
  'phase7-v27-plan-migration-v1:'||d.racing_day_id),1,32) AS operation_id
FROM days d CROSS JOIN ordinals o
LEFT JOIN phase7_scheduler_progress p
 ON p.racing_day_id=d.racing_day_id AND p.phase_ordinal=o.ordinal
ORDER BY d.racing_day_id,o.ordinal;

CREATE TEMP TABLE phase7_v27_plan_days AS
SELECT day.racing_day_id,day.operation_id,
 strftime('%Y-%m-%dT%H:%M:%f000+00:00','now') AS operation_created_at,
 json_object(
  'completed_prefix',json(COALESCE((
   SELECT json_group_array(json_object(
    'command_operation_id',prefix.command_operation_id,
    'completed_at',prefix.planned_at,
    'lease_generation',prefix.lease_generation,
    'migration_plan_anchor_at',prefix.planned_at,
    'phase_name',prefix.phase_name,
    'phase_ordinal',prefix.phase_ordinal))
   FROM (
    SELECT * FROM phase7_v27_plan_rows
    WHERE racing_day_id=day.racing_day_id AND completed=1
    ORDER BY phase_ordinal
   ) prefix),'[]')),
  'migrated_suffix',json(COALESCE((
   SELECT json_group_array(json_object(
    'command_operation_id',suffix.command_operation_id,
    'lease_generation',suffix.lease_generation,
    'phase_name',suffix.phase_name,
    'phase_ordinal',suffix.phase_ordinal,
    'planned_at',suffix.planned_at))
   FROM (
    SELECT * FROM phase7_v27_plan_rows
    WHERE racing_day_id=day.racing_day_id AND completed=0
    ORDER BY phase_ordinal
   ) suffix),'[]')),
  'provenance_version','phase7-v27-day-plan-migration-v1',
  'racing_day_id',day.racing_day_id
 ) AS payload_json
FROM phase7_v27_plan_rows day
GROUP BY day.racing_day_id;

INSERT INTO operations(operation_id,kind,payload_sha256,created_at)
SELECT operation_id,'phase7_migrate_v27_day_command_plan',
 sha256_text(payload_json),operation_created_at
FROM phase7_v27_plan_days;

INSERT INTO phase7_day_command_plan
SELECT racing_day_id,phase_ordinal,phase_name,command_operation_id,
 lease_generation,planned_at,operation_id
FROM phase7_v27_plan_rows;

DROP TABLE phase7_v27_plan_days;
DROP TABLE phase7_v27_plan_rows;
DROP TABLE phase7_v27_plan_assertion;

-- Migration authority is closed after the transactional backfill.  All later
-- plan rows must be genuine live planning operations with provable lease time.
CREATE TABLE phase7_day_plan_authorities (
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 lease_generation INTEGER NOT NULL,
 lease_token TEXT NOT NULL,
 planned_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,operation_id)
);
CREATE TRIGGER phase7_day_plan_authority_exact
BEFORE INSERT ON phase7_day_plan_authorities WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_plan_racing_day')
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease lease
  WHERE lease.singleton=1 AND lease.generation=NEW.lease_generation
   AND lease.lease_token=NEW.lease_token
   AND lease.acquired_at<=NEW.planned_at AND lease.expires_at>NEW.planned_at)
BEGIN SELECT RAISE(ABORT,'day plan authority lacks exact live scheduler lease'); END;
CREATE TRIGGER phase7_day_plan_authorities_append_only_update
BEFORE UPDATE ON phase7_day_plan_authorities
BEGIN SELECT RAISE(ABORT,'day plan authorities are append-only'); END;
CREATE TRIGGER phase7_day_plan_authorities_append_only_delete
BEFORE DELETE ON phase7_day_plan_authorities
BEGIN SELECT RAISE(ABORT,'day plan authorities are append-only'); END;

CREATE TRIGGER phase7_day_command_plan_exact
BEFORE INSERT ON phase7_day_command_plan WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_plan_racing_day')
 OR NOT EXISTS (SELECT 1 FROM phase7_day_plan_authorities authority
  WHERE authority.racing_day_id=NEW.racing_day_id
   AND authority.lease_generation=NEW.lease_generation
   AND authority.planned_at=NEW.planned_at
   AND authority.operation_id=NEW.operation_id)
 OR NEW.phase_name<>CASE NEW.phase_ordinal
  WHEN 1 THEN 'discover_programme' WHEN 2 THEN 'collect_cards_and_form'
  WHEN 3 THEN 'collect_adaptive_odds' WHEN 4 THEN 'close_and_seal'
  WHEN 5 THEN 'deferred_prediction' WHEN 6 THEN 'collect_results'
  WHEN 7 THEN 'join_training_examples' WHEN 8 THEN 'reconcile'
  WHEN 9 THEN 'request_training' END
BEGIN SELECT RAISE(ABORT,'day command plan lacks scheduler authority'); END;

CREATE TABLE phase7_migrated_plan_adoptions (
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 lease_generation INTEGER NOT NULL,
 adopted_at TEXT NOT NULL,
 migration_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,lease_generation)
);
CREATE TRIGGER phase7_migrated_plan_adoption_exact
BEFORE INSERT ON phase7_migrated_plan_adoptions WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_adopt_migrated_day_command_plan')
 OR NOT EXISTS (SELECT 1 FROM operations migration
  WHERE migration.operation_id=NEW.migration_operation_id
   AND migration.kind='phase7_migrate_v27_day_command_plan')
 OR (SELECT count(*) FROM phase7_day_command_plan p
  WHERE p.racing_day_id=NEW.racing_day_id
   AND p.operation_id=NEW.migration_operation_id)<>9
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease lease
  WHERE lease.singleton=1 AND lease.generation=NEW.lease_generation
   AND lease.acquired_at<=NEW.adopted_at AND lease.expires_at>NEW.adopted_at)
BEGIN SELECT RAISE(ABORT,'migrated day plan adoption lacks exact live authority'); END;
CREATE TRIGGER phase7_migrated_plan_adoptions_append_only_update
BEFORE UPDATE ON phase7_migrated_plan_adoptions
BEGIN SELECT RAISE(ABORT,'migrated plan adoptions are append-only'); END;
CREATE TRIGGER phase7_migrated_plan_adoptions_append_only_delete
BEFORE DELETE ON phase7_migrated_plan_adoptions
BEGIN SELECT RAISE(ABORT,'migrated plan adoptions are append-only'); END;

-- Every immutable plan, including an ordinary live plan, must be explicitly
-- adopted before a later lease generation can execute or advance it.
CREATE TABLE phase7_day_plan_adoptions (
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 lease_generation INTEGER NOT NULL,
 lease_token TEXT NOT NULL,
 adopted_at TEXT NOT NULL,
 plan_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 plan_kind TEXT NOT NULL CHECK(plan_kind IN (
  'phase7_plan_racing_day','phase7_migrate_v27_day_command_plan')),
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,lease_generation)
);
CREATE TRIGGER phase7_day_plan_adoption_exact
BEFORE INSERT ON phase7_day_plan_adoptions WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind=CASE NEW.plan_kind
   WHEN 'phase7_migrate_v27_day_command_plan'
    THEN 'phase7_adopt_migrated_day_command_plan'
   ELSE 'phase7_adopt_day_command_plan' END)
 OR NOT EXISTS (SELECT 1 FROM operations original
  WHERE original.operation_id=NEW.plan_operation_id
   AND original.kind=NEW.plan_kind)
 OR (SELECT count(*) FROM phase7_day_command_plan p
  WHERE p.racing_day_id=NEW.racing_day_id
   AND p.operation_id=NEW.plan_operation_id)<>9
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease lease
  WHERE lease.singleton=1 AND lease.generation=NEW.lease_generation
   AND lease.lease_token=NEW.lease_token
   AND lease.acquired_at<=NEW.adopted_at AND lease.expires_at>NEW.adopted_at)
BEGIN SELECT RAISE(ABORT,'day plan adoption lacks exact live authority'); END;
CREATE TRIGGER phase7_day_plan_adoptions_append_only_update
BEFORE UPDATE ON phase7_day_plan_adoptions
BEGIN SELECT RAISE(ABORT,'day plan adoptions are append-only'); END;
CREATE TRIGGER phase7_day_plan_adoptions_append_only_delete
BEFORE DELETE ON phase7_day_plan_adoptions
BEGIN SELECT RAISE(ABORT,'day plan adoptions are append-only'); END;

CREATE TABLE phase7_application_command_claims (
 command_operation_id TEXT PRIMARY KEY,
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 phase_name TEXT NOT NULL,
 command_payload_sha256 TEXT NOT NULL
  CHECK(length(command_payload_sha256)=64
   AND command_payload_sha256 NOT GLOB '*[^0-9a-f]*'),
 lease_generation INTEGER NOT NULL,
 lease_token TEXT NOT NULL,
 claimed_at TEXT NOT NULL,
 CHECK(phase_name IN (
  'discover_programme','collect_cards_and_form','collect_adaptive_odds',
  'close_and_seal','deferred_prediction','collect_results',
  'join_training_examples','reconcile','request_training'))
);
CREATE TRIGGER phase7_application_command_claim_exact
BEFORE INSERT ON phase7_application_command_claims WHEN
 NOT EXISTS (SELECT 1 FROM phase7_day_command_plan p
  WHERE p.command_operation_id=NEW.command_operation_id
   AND p.racing_day_id=NEW.racing_day_id AND p.phase_name=NEW.phase_name)
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease l WHERE l.singleton=1
  AND l.generation=NEW.lease_generation AND l.lease_token=NEW.lease_token
  AND l.acquired_at<=NEW.claimed_at AND l.expires_at>NEW.claimed_at)
BEGIN SELECT RAISE(ABORT,'application command claim lacks live scheduler authority'); END;
CREATE TRIGGER phase7_application_command_claims_append_only_update
BEFORE UPDATE ON phase7_application_command_claims
BEGIN SELECT RAISE(ABORT,'application command claims are append-only'); END;

CREATE TABLE phase7_determinism_executions (
 operation_id TEXT PRIMARY KEY REFERENCES operations(operation_id),
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 release_manifest_checksum TEXT NOT NULL,
 config_checksum TEXT NOT NULL,
 bundle_authority_checksum TEXT NOT NULL,
 runner_identity TEXT NOT NULL,
 runner_implementation_version TEXT NOT NULL,
 input_checksum TEXT NOT NULL,
 output_checksum TEXT NOT NULL,
 executed_at TEXT NOT NULL,
 CHECK(input_checksum<>output_checksum),
 CHECK(length(release_manifest_checksum)=71
  AND release_manifest_checksum GLOB 'sha256:[0-9a-f]*'),
 CHECK(length(config_checksum)=71 AND config_checksum GLOB 'sha256:[0-9a-f]*'),
 CHECK(length(bundle_authority_checksum)=71
  AND bundle_authority_checksum GLOB 'sha256:[0-9a-f]*'),
 CHECK(length(input_checksum)=71 AND input_checksum GLOB 'sha256:[0-9a-f]*'),
 CHECK(length(output_checksum)=71 AND output_checksum GLOB 'sha256:[0-9a-f]*'),
 CHECK(length(trim(runner_identity))>0),
 CHECK(length(trim(runner_implementation_version))>0)
);
CREATE TRIGGER phase7_determinism_execution_exact
BEFORE INSERT ON phase7_determinism_executions WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_record_determinism_execution')
 OR NOT EXISTS (SELECT 1 FROM phase7_release_manifests r
  WHERE r.release_id=NEW.release_id
   AND r.manifest_checksum=NEW.release_manifest_checksum
   AND r.config_checksum=NEW.config_checksum)
 OR NEW.runner_identity<>'race_collection.phase7.closed_replay'
 OR NEW.runner_implementation_version<>'phase7-determinism-runner-v1'
BEGIN SELECT RAISE(ABORT,'determinism execution lacks exact operation authority'); END;
CREATE TRIGGER phase7_determinism_executions_append_only_update
BEFORE UPDATE ON phase7_determinism_executions
BEGIN SELECT RAISE(ABORT,'determinism executions are append-only'); END;
CREATE TRIGGER phase7_determinism_executions_append_only_delete
BEFORE DELETE ON phase7_determinism_executions
BEGIN SELECT RAISE(ABORT,'determinism executions are append-only'); END;

DROP TRIGGER phase7_receipt_append_only_update;
ALTER TABLE phase7_application_command_receipts
 ADD COLUMN command_payload_sha256 TEXT NOT NULL DEFAULT
 '0000000000000000000000000000000000000000000000000000000000000000'
 CHECK(length(command_payload_sha256)=64
 AND command_payload_sha256 NOT GLOB '*[^0-9a-f]*');
UPDATE phase7_application_command_receipts SET command_payload_sha256=sha256_text(
 CASE phase_name
 WHEN 'discover_programme' THEN json_object(
  'programme_checksum',json_extract(result_json,'$.programme_checksum'),
  'source',json_extract(result_json,'$.source'),
  'type','discover_programme')
 WHEN 'request_training' THEN json_object(
  'authorization_operation_id',
   json_extract(result_json,'$.request.authorization_operation_id'),
  'binding_operation_id',json_extract(result_json,'$.request.operation_id'),
  'request_id',json_extract(result_json,'$.request.training_request_id'),
  'request_operation_id',
   json_extract(result_json,'$.request.request_operation_id'),
  'type','request_training')
 ELSE json_object('type',phase_name) END);
CREATE TRIGGER phase7_receipt_append_only_update
BEFORE UPDATE ON phase7_application_command_receipts
BEGIN SELECT RAISE(ABORT,'application receipts are append-only'); END;
DROP TRIGGER phase7_receipt_exact;
CREATE TRIGGER phase7_receipt_exact
BEFORE INSERT ON phase7_application_command_receipts WHEN
 length(NEW.result_checksum)<>71 OR substr(NEW.result_checksum,1,7)<>'sha256:'
 OR length(NEW.command_payload_sha256)<>64
 OR NEW.command_payload_sha256 GLOB '*[^0-9a-f]*'
 OR NOT EXISTS (SELECT 1 FROM operations o
  WHERE o.operation_id=NEW.command_operation_id
   AND o.kind='phase7_command_'||NEW.phase_name)
 OR NOT EXISTS (SELECT 1 FROM phase7_application_command_claims c
  WHERE c.command_operation_id=NEW.command_operation_id
   AND c.racing_day_id=NEW.racing_day_id
   AND c.phase_name=NEW.phase_name
   AND c.command_payload_sha256=NEW.command_payload_sha256)
BEGIN SELECT RAISE(ABORT,'application receipt lacks exact command authority'); END;

CREATE TABLE phase7_observation_authority_events (
 event_id INTEGER PRIMARY KEY AUTOINCREMENT,
 candidate_release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 action TEXT NOT NULL CHECK(action IN ('authorize','revoke')),
 actor TEXT NOT NULL CHECK(length(trim(actor))>0),
 reason TEXT NOT NULL CHECK(length(trim(reason))>0),
 occurred_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_observation_authority_event_exact
BEFORE INSERT ON phase7_observation_authority_events WHEN
 (NEW.action='authorize' AND (
  NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
   AND o.kind='phase7_authorize_observation')
  OR NOT EXISTS (SELECT 1 FROM phase7_release_pointer p WHERE p.singleton=1
   AND p.authority='legacy' AND p.legacy_preserved=1)
  OR EXISTS (SELECT 1 FROM phase7_observation_authority_events current
   WHERE current.event_id=(SELECT max(event_id)
    FROM phase7_observation_authority_events)
    AND current.action='authorize')))
 OR (NEW.action='revoke' AND (
  NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
   AND o.kind IN ('phase7_revoke_observation','phase7_activate_release'))
  OR NOT EXISTS (SELECT 1 FROM phase7_observation_authority_events current
   WHERE current.event_id=(SELECT max(event_id)
    FROM phase7_observation_authority_events)
    AND current.action='authorize'
    AND current.candidate_release_id=NEW.candidate_release_id)))
BEGIN SELECT RAISE(ABORT,'observation authority transition is invalid'); END;
CREATE TRIGGER phase7_observation_authority_events_append_only_update
BEFORE UPDATE ON phase7_observation_authority_events
BEGIN SELECT RAISE(ABORT,'observation authority events are append-only'); END;
CREATE TRIGGER phase7_observation_authority_events_append_only_delete
BEFORE DELETE ON phase7_observation_authority_events
BEGIN SELECT RAISE(ABORT,'observation authority events are append-only'); END;
CREATE TRIGGER phase7_cutover_eligibility_observation_exact
BEFORE INSERT ON phase7_cutover_eligibility WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_cutover_eligibility')
 OR NOT EXISTS (
  SELECT 1 FROM phase7_observation_authority_events current
  WHERE current.event_id=(SELECT max(event_id)
   FROM phase7_observation_authority_events)
   AND current.action='authorize'
   AND current.candidate_release_id=NEW.candidate_release_id)
 OR NOT EXISTS (
  SELECT 1 FROM phase7_day_evidence first
  JOIN phase7_day_evidence second
  WHERE first.racing_day_id=NEW.first_racing_day_id
   AND second.racing_day_id=NEW.second_racing_day_id
   AND first.release_id=NEW.candidate_release_id
   AND second.release_id=NEW.candidate_release_id
   AND first.complete=1 AND first.critical_failure=0
   AND second.complete=1 AND second.critical_failure=0
   AND first.recorded_at>=(SELECT occurred_at
    FROM phase7_observation_authority_events ORDER BY event_id DESC LIMIT 1)
   AND second.recorded_at>=(SELECT occurred_at
    FROM phase7_observation_authority_events ORDER BY event_id DESC LIMIT 1)
   AND NEW.eligible_at>=first.recorded_at
   AND NEW.eligible_at>=second.recorded_at)
BEGIN SELECT RAISE(ABORT,'cutover eligibility lacks prospective observation authority'); END;
CREATE TRIGGER phase7_application_command_claims_append_only_delete
BEFORE DELETE ON phase7_application_command_claims
BEGIN SELECT RAISE(ABORT,'application command claims are append-only'); END;

CREATE TABLE phase7_application_command_attempts (
 attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
 command_operation_id TEXT NOT NULL
  REFERENCES phase7_application_command_claims(command_operation_id),
 lease_generation INTEGER NOT NULL,
 lease_token TEXT NOT NULL,
 state TEXT NOT NULL CHECK(state IN (
  'claimed','recovering','handler_failed','postcondition_failed','fenced','completed')),
 recorded_at TEXT NOT NULL,
 details TEXT NOT NULL
);
CREATE TRIGGER phase7_application_command_attempt_exact
BEFORE INSERT ON phase7_application_command_attempts WHEN
 NEW.state IN ('claimed','recovering','completed') AND
 NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease l WHERE l.singleton=1
  AND l.generation=NEW.lease_generation AND l.lease_token=NEW.lease_token
  AND l.acquired_at<=NEW.recorded_at AND l.expires_at>NEW.recorded_at)
BEGIN SELECT RAISE(ABORT,'application command attempt lacks live scheduler authority'); END;
CREATE TRIGGER phase7_application_command_attempts_append_only_update
BEFORE UPDATE ON phase7_application_command_attempts
BEGIN SELECT RAISE(ABORT,'application command attempts are append-only'); END;
CREATE TRIGGER phase7_application_command_attempts_append_only_delete
BEFORE DELETE ON phase7_application_command_attempts
BEGIN SELECT RAISE(ABORT,'application command attempts are append-only'); END;

CREATE TABLE phase7_training_request_intents (
 racing_day_id TEXT NOT NULL REFERENCES phase7_reconciliation(racing_day_id),
 training_request_id TEXT NOT NULL UNIQUE,
 request_operation_id TEXT NOT NULL UNIQUE,
 authorized_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,training_request_id,request_operation_id)
);
CREATE TRIGGER phase7_training_request_intent_exact
BEFORE INSERT ON phase7_training_request_intents WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_authorize_training_request')
 OR NOT EXISTS (SELECT 1 FROM phase7_day_evidence e
  JOIN phase7_reconciliation r USING(racing_day_id)
  WHERE e.racing_day_id=NEW.racing_day_id AND e.complete=1
   AND r.mismatch_count=0 AND NEW.authorized_at>=r.reconciled_at)
BEGIN SELECT RAISE(ABORT,'training request intent lacks exact Racing Day authority'); END;
CREATE TRIGGER phase7_training_request_intents_append_only_update
BEFORE UPDATE ON phase7_training_request_intents
BEGIN SELECT RAISE(ABORT,'training request intents are append-only'); END;
CREATE TRIGGER phase7_training_request_intents_append_only_delete
BEFORE DELETE ON phase7_training_request_intents
BEGIN SELECT RAISE(ABORT,'training request intents are append-only'); END;

CREATE TABLE phase7_day_training_requests (
 racing_day_id TEXT PRIMARY KEY REFERENCES phase7_reconciliation(racing_day_id),
 training_request_id TEXT NOT NULL UNIQUE REFERENCES phase6_service_training_requests(training_request_id),
 request_operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 bound_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_day_training_request_exact
BEFORE INSERT ON phase7_day_training_requests WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_bind_training_request')
 OR NOT EXISTS (
  SELECT 1 FROM phase6_training_requests r
  JOIN phase6_service_training_requests s USING(training_request_id)
  JOIN phase7_training_request_intents i
   ON i.racing_day_id=NEW.racing_day_id
   AND i.training_request_id=NEW.training_request_id
   AND i.request_operation_id=NEW.request_operation_id
  JOIN phase7_day_evidence e ON e.racing_day_id=NEW.racing_day_id
  JOIN phase7_reconciliation x ON x.racing_day_id=e.racing_day_id
  WHERE r.training_request_id=NEW.training_request_id
   AND r.operation_id=NEW.request_operation_id
   AND s.operation_id=NEW.request_operation_id
   AND e.complete=1 AND x.mismatch_count=0
   AND NEW.bound_at>=x.reconciled_at AND NEW.bound_at>=r.requested_at
   AND NEW.bound_at>=i.authorized_at)
BEGIN SELECT RAISE(ABORT,'training request lacks exact Racing Day authority'); END;
CREATE TRIGGER phase7_day_training_requests_append_only_update
BEFORE UPDATE ON phase7_day_training_requests
BEGIN SELECT RAISE(ABORT,'day training requests are append-only'); END;
CREATE TRIGGER phase7_day_training_requests_append_only_delete
BEFORE DELETE ON phase7_day_training_requests
BEGIN SELECT RAISE(ABORT,'day training requests are append-only'); END;

DROP TRIGGER phase7_probation_seal_exact;
CREATE TRIGGER phase7_probation_seal_exact BEFORE INSERT ON phase7_probation_seals WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_seal_probation')
 OR NOT EXISTS (SELECT 1 FROM phase7_probation_control c WHERE c.singleton=1
  AND c.generation=NEW.generation AND c.state='complete')
 OR NOT EXISTS (SELECT 1 FROM phase6_probation_states s
  WHERE s.probation_id=NEW.probation_id AND s.state_checksum=NEW.state_checksum
   AND s.operation_id=NEW.operation_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_release_pointer p
  WHERE p.singleton=1 AND p.authority='race_collection_service'
   AND p.release_id=NEW.release_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_release_history h
  JOIN phase7_release_pointer p ON p.singleton=1
  WHERE h.operation_id=NEW.cutover_operation_id AND h.release_id=NEW.release_id
   AND h.action='activate' AND h.effective_racing_day_id=p.effective_racing_day_id)
 OR (SELECT count(*) FROM phase7_probation_acceptances a
     WHERE a.generation=NEW.generation)<>14
 OR EXISTS (SELECT 1 FROM phase7_probation_acceptances a
  JOIN phase7_day_evidence e USING(racing_day_id)
  WHERE a.generation=NEW.generation AND e.release_id<>NEW.release_id)
 OR EXISTS (SELECT 1 FROM phase7_probation_acceptances a
  JOIN racing_days d USING(racing_day_id)
  WHERE a.generation=NEW.generation AND
   (a.local_date<>d.local_date OR NOT EXISTS (
    SELECT 1 FROM operations o WHERE o.operation_id=a.operation_id
     AND o.kind='phase7_accept_probation_day')))
 OR EXISTS (SELECT 1 FROM phase7_probation_acceptances a
  WHERE a.generation=NEW.generation AND NOT EXISTS (
   SELECT 1 FROM phase6_racing_day_schedule s
   WHERE s.racing_day_id=a.racing_day_id
    AND s.programme_checksum=a.programme_checksum
    AND s.predecessor_racing_day_id=CASE
     WHEN a.local_date=(SELECT min(local_date) FROM phase7_probation_acceptances
       WHERE generation=NEW.generation)
     THEN (SELECT effective_racing_day_id FROM phase7_release_history
       WHERE operation_id=NEW.cutover_operation_id)
     ELSE (SELECT prior.racing_day_id FROM phase7_probation_acceptances prior
       WHERE prior.generation=NEW.generation AND prior.local_date<a.local_date
       ORDER BY prior.local_date DESC LIMIT 1)
    END))
BEGIN SELECT RAISE(ABORT,'probation seal lacks exact generation, release, cutover and chain'); END;

CREATE TRIGGER phase7_probation_acceptance_temporal BEFORE INSERT ON phase7_probation_acceptances
WHEN NOT EXISTS (
 SELECT 1 FROM racing_days d
 JOIN phase7_day_evidence e USING(racing_day_id)
 JOIN phase7_reconciliation r USING(racing_day_id)
 WHERE d.racing_day_id=NEW.racing_day_id
  AND NEW.accepted_at>=d.closed_at
  AND NEW.accepted_at>=e.recorded_at
  AND NEW.accepted_at>=r.reconciled_at
  AND date(NEW.accepted_at)>=d.local_date)
BEGIN SELECT RAISE(ABORT,'probation acceptance predates its durable Racing Day evidence'); END;

CREATE TRIGGER phase7_probation_acceptance_exact_chain
BEFORE INSERT ON phase7_probation_acceptances WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_accept_probation_day')
 OR NOT EXISTS (SELECT 1 FROM phase7_probation_control c WHERE c.singleton=1
  AND c.generation=NEW.generation AND c.state='running')
 OR (SELECT count(*) FROM phase7_probation_acceptances a
     WHERE a.generation=NEW.generation)>=14
 OR NOT EXISTS (
  SELECT 1 FROM racing_days d
  JOIN phase7_day_evidence e USING(racing_day_id)
  JOIN phase7_reconciliation r USING(racing_day_id)
  JOIN phase6_racing_day_schedule s USING(racing_day_id)
  WHERE d.racing_day_id=NEW.racing_day_id
   AND d.local_date=NEW.local_date
   AND e.complete=1 AND e.critical_failure=0
   AND r.mismatch_count=0
   AND s.programme_checksum=NEW.programme_checksum
   AND s.predecessor_racing_day_id=CASE
    WHEN NOT EXISTS (SELECT 1 FROM phase7_probation_acceptances prior
      WHERE prior.generation=NEW.generation)
    THEN (SELECT effective_racing_day_id FROM phase7_release_pointer
      WHERE singleton=1 AND authority='race_collection_service'
       AND release_id=e.release_id)
    ELSE (SELECT prior.racing_day_id FROM phase7_probation_acceptances prior
      WHERE prior.generation=NEW.generation
      ORDER BY prior.local_date DESC LIMIT 1)
   END
   AND NEW.local_date>COALESCE(
    (SELECT prior.local_date FROM phase7_probation_acceptances prior
     WHERE prior.generation=NEW.generation
     ORDER BY prior.local_date DESC LIMIT 1),
    (SELECT d0.local_date FROM phase7_release_pointer p
     JOIN racing_days d0 ON d0.racing_day_id=p.effective_racing_day_id
     WHERE p.singleton=1 AND p.authority='race_collection_service'
      AND p.release_id=e.release_id)))
BEGIN SELECT RAISE(ABORT,'probation acceptance lacks exact schedule authority'); END;

-- A Racing-Day forecast cohort is operational coverage authority only.  It is
-- deliberately separate from the long-horizon Phase-6 evaluation verdict.
CREATE TABLE phase7_day_forecast_cohorts (
 racing_day_id TEXT PRIMARY KEY REFERENCES racing_days(racing_day_id),
 assignment_id TEXT NOT NULL REFERENCES canonical_serving_assignments(assignment_id),
 authorized_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_day_forecast_cohort_members (
 racing_day_id TEXT NOT NULL REFERENCES phase7_day_forecast_cohorts(racing_day_id),
 role TEXT NOT NULL CHECK(role IN ('champion','challenger')),
 bundle_id TEXT NOT NULL,
 bundle_checksum TEXT NOT NULL,
 service_run_id TEXT NOT NULL UNIQUE,
 PRIMARY KEY(racing_day_id,bundle_id),
 UNIQUE(racing_day_id,role,bundle_id),
 FOREIGN KEY(bundle_id,bundle_checksum)
  REFERENCES canonical_model_bundles(bundle_id,bundle_checksum)
);
CREATE TABLE phase7_day_forecast_cohort_components (
 racing_day_id TEXT NOT NULL,
 bundle_id TEXT NOT NULL,
 component_kind TEXT NOT NULL,
 artifact_checksum TEXT NOT NULL,
 byte_size INTEGER NOT NULL,
 PRIMARY KEY(racing_day_id,bundle_id,component_kind),
 FOREIGN KEY(racing_day_id,bundle_id)
  REFERENCES phase7_day_forecast_cohort_members(racing_day_id,bundle_id)
);
CREATE TABLE phase7_day_forecast_commands (
 racing_day_id TEXT NOT NULL,
 race_id TEXT NOT NULL REFERENCES races(race_id),
 bundle_id TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE,
 PRIMARY KEY(racing_day_id,race_id,bundle_id),
 FOREIGN KEY(racing_day_id,bundle_id)
  REFERENCES phase7_day_forecast_cohort_members(racing_day_id,bundle_id)
);
CREATE TRIGGER phase7_day_forecast_cohort_exact
BEFORE INSERT ON phase7_day_forecast_cohorts WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_authorize_day_forecast_cohort')
 OR NOT EXISTS (SELECT 1 FROM canonical_day_assignments d
  WHERE d.racing_day_id=NEW.racing_day_id AND d.assignment_id=NEW.assignment_id)
 OR EXISTS (SELECT 1 FROM result_attempts r JOIN races x USING(race_id)
  WHERE x.racing_day_id=NEW.racing_day_id)
 OR EXISTS (SELECT 1 FROM phase6_forecast_service_artifacts f JOIN races x USING(race_id)
  WHERE x.racing_day_id=NEW.racing_day_id)
BEGIN SELECT RAISE(ABORT,'day forecast cohort lacks pre-result assignment authority'); END;
CREATE TRIGGER phase7_day_forecast_cohorts_append_only_update
BEFORE UPDATE ON phase7_day_forecast_cohorts
BEGIN SELECT RAISE(ABORT,'day forecast cohorts are append-only'); END;
CREATE TRIGGER phase7_day_forecast_cohorts_append_only_delete
BEFORE DELETE ON phase7_day_forecast_cohorts
BEGIN SELECT RAISE(ABORT,'day forecast cohorts are append-only'); END;
CREATE TRIGGER phase7_day_forecast_cohort_member_exact
BEFORE INSERT ON phase7_day_forecast_cohort_members WHEN
 NOT EXISTS (SELECT 1 FROM phase7_day_forecast_cohorts c
  JOIN canonical_model_bundles b
   ON b.bundle_id=NEW.bundle_id AND b.bundle_checksum=NEW.bundle_checksum
  WHERE c.racing_day_id=NEW.racing_day_id AND b.created_at<c.authorized_at)
 OR (NEW.role='champion' AND NOT EXISTS (
  SELECT 1 FROM phase7_day_forecast_cohorts c
  JOIN canonical_day_assignments d USING(racing_day_id)
  WHERE c.racing_day_id=NEW.racing_day_id
   AND d.bundle_id=NEW.bundle_id AND d.bundle_checksum=NEW.bundle_checksum))
 OR (NEW.role='challenger' AND EXISTS (
  SELECT 1 FROM canonical_day_assignments d WHERE d.racing_day_id=NEW.racing_day_id
   AND d.bundle_id=NEW.bundle_id AND d.bundle_checksum=NEW.bundle_checksum))
BEGIN SELECT RAISE(ABORT,'day forecast cohort member lacks exact bundle authority'); END;
CREATE TRIGGER phase7_day_forecast_cohort_members_append_only_update
BEFORE UPDATE ON phase7_day_forecast_cohort_members
BEGIN SELECT RAISE(ABORT,'day forecast cohort members are append-only'); END;
CREATE TRIGGER phase7_day_forecast_cohort_members_append_only_delete
BEFORE DELETE ON phase7_day_forecast_cohort_members
BEGIN SELECT RAISE(ABORT,'day forecast cohort members are append-only'); END;
CREATE TRIGGER phase7_day_forecast_cohort_components_append_only_update
BEFORE UPDATE ON phase7_day_forecast_cohort_components
BEGIN SELECT RAISE(ABORT,'day forecast cohort components are append-only'); END;
CREATE TRIGGER phase7_day_forecast_cohort_component_exact
BEFORE INSERT ON phase7_day_forecast_cohort_components WHEN
 NOT EXISTS (SELECT 1 FROM canonical_bundle_components c
  WHERE c.bundle_id=NEW.bundle_id AND c.component_kind=NEW.component_kind
   AND c.artifact_checksum=NEW.artifact_checksum AND c.byte_size=NEW.byte_size)
BEGIN SELECT RAISE(ABORT,'day forecast cohort component disagrees with registration'); END;
CREATE TRIGGER phase7_day_forecast_cohort_components_append_only_delete
BEFORE DELETE ON phase7_day_forecast_cohort_components
BEGIN SELECT RAISE(ABORT,'day forecast cohort components are append-only'); END;
CREATE TRIGGER phase7_day_forecast_commands_append_only_update
BEFORE UPDATE ON phase7_day_forecast_commands
BEGIN SELECT RAISE(ABORT,'day forecast commands are append-only'); END;
CREATE TRIGGER phase7_day_forecast_command_exact
BEFORE INSERT ON phase7_day_forecast_commands WHEN
 NOT EXISTS (SELECT 1 FROM races r WHERE r.race_id=NEW.race_id
  AND r.racing_day_id=NEW.racing_day_id)
 OR EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id)
BEGIN SELECT RAISE(ABORT,'day forecast command lacks exact future race authority'); END;
CREATE TRIGGER phase7_day_forecast_commands_append_only_delete
BEFORE DELETE ON phase7_day_forecast_commands
BEGIN SELECT RAISE(ABORT,'day forecast commands are append-only'); END;
CREATE TRIGGER phase7_day_forecast_service_identity
BEFORE INSERT ON phase6_forecast_service_artifacts WHEN EXISTS (
 SELECT 1 FROM phase7_day_forecast_commands c
 WHERE c.race_id=NEW.race_id AND c.bundle_id=NEW.bundle_id
) AND NOT EXISTS (
 SELECT 1 FROM phase7_day_forecast_commands c
 JOIN phase7_day_forecast_cohort_members m
  ON m.racing_day_id=c.racing_day_id AND m.bundle_id=c.bundle_id
 WHERE c.race_id=NEW.race_id AND c.bundle_id=NEW.bundle_id
  AND c.operation_id=NEW.operation_id
  AND m.bundle_checksum=NEW.bundle_checksum
  AND m.service_run_id=NEW.service_run_id
)
BEGIN SELECT RAISE(ABORT,'service forecast disagrees with day cohort identity'); END;
