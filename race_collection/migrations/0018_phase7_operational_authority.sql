CREATE TABLE phase7_scheduler_lease (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner_id TEXT NOT NULL,
 lease_token TEXT NOT NULL UNIQUE, acquired_at TEXT NOT NULL, expires_at TEXT NOT NULL,
 generation INTEGER NOT NULL CHECK(generation>0), operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_scheduler_history (
 generation INTEGER PRIMARY KEY, owner_id TEXT NOT NULL, lease_token TEXT NOT NULL UNIQUE,
 acquired_at TEXT NOT NULL, expires_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_one_lease_insert BEFORE INSERT ON phase7_scheduler_lease
WHEN NEW.singleton<>1 BEGIN SELECT RAISE(ABORT,'one scheduler lease authority'); END;

CREATE TABLE phase7_admin_audit (
 audit_id INTEGER PRIMARY KEY, actor TEXT NOT NULL CHECK(length(trim(actor))>0),
 reason TEXT NOT NULL CHECK(length(trim(reason))>0), command TEXT NOT NULL,
 scope TEXT NOT NULL, before_json TEXT NOT NULL CHECK(json_valid(before_json)),
 after_json TEXT NOT NULL CHECK(json_valid(after_json)), occurred_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_pauses (
 scope TEXT PRIMARY KEY CHECK(scope IN ('results','joins','training_requests','promotion','cutover')),
 paused INTEGER NOT NULL CHECK(paused IN (0,1)), reason TEXT NOT NULL,
 changed_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_alerts (
 alert_id TEXT PRIMARY KEY, category TEXT NOT NULL CHECK(category IN
 ('source_wide_outage','day_blocker','checksum_failure','post_freeze_contamination','result_before_prediction','champion_failure')),
 racing_day_id TEXT REFERENCES racing_days(racing_day_id), details TEXT NOT NULL,
 raised_at TEXT NOT NULL, resolved_at TEXT, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase7_release_manifests (
 release_id TEXT PRIMARY KEY, manifest_checksum TEXT NOT NULL UNIQUE,
 code_commit TEXT NOT NULL CHECK(length(code_commit)=40), config_checksum TEXT NOT NULL,
 schema_version INTEGER NOT NULL CHECK(schema_version>=18), artifact_contract TEXT NOT NULL,
 policy_version TEXT NOT NULL, bundle_versions_json TEXT NOT NULL CHECK(json_valid(bundle_versions_json)),
 service_root TEXT NOT NULL CHECK(substr(service_root,1,1)='/' AND service_root NOT LIKE '%20__/%'),
 observed_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_release_pointer (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 authority TEXT NOT NULL CHECK(authority IN ('legacy','race_collection_service')),
 legacy_preserved INTEGER NOT NULL CHECK(legacy_preserved=1), effective_racing_day_id TEXT REFERENCES racing_days(racing_day_id),
 changed_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_release_history (
 history_id INTEGER PRIMARY KEY, release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 authority TEXT NOT NULL, prior_release_id TEXT, prior_authority TEXT,
 effective_racing_day_id TEXT REFERENCES racing_days(racing_day_id), action TEXT NOT NULL CHECK(action IN ('initial','activate','rollback')),
 changed_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE phase7_day_evidence (
 racing_day_id TEXT PRIMARY KEY REFERENCES racing_days(racing_day_id),
 reconciliation_checksum TEXT NOT NULL, restart_checksum TEXT NOT NULL,
 ordering_checksum TEXT NOT NULL, determinism_checksum TEXT NOT NULL,
 complete INTEGER NOT NULL CHECK(complete IN (0,1)), critical_failure INTEGER NOT NULL CHECK(critical_failure IN (0,1)),
 release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id), recorded_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 CHECK(complete=0 OR critical_failure=0)
);
CREATE TABLE phase7_reconciliation (
 racing_day_id TEXT PRIMARY KEY REFERENCES phase7_day_evidence(racing_day_id),
 report_checksum TEXT NOT NULL UNIQUE, mismatch_count INTEGER NOT NULL CHECK(mismatch_count>=0),
 report_json TEXT NOT NULL CHECK(json_valid(report_json)), reconciled_at TEXT NOT NULL
);
CREATE TABLE phase7_cutover_eligibility (
 candidate_release_id TEXT PRIMARY KEY REFERENCES phase7_release_manifests(release_id),
 first_racing_day_id TEXT NOT NULL REFERENCES phase7_day_evidence(racing_day_id),
 second_racing_day_id TEXT NOT NULL REFERENCES phase7_day_evidence(racing_day_id),
 evidence_checksum TEXT NOT NULL, eligible_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id), CHECK(first_racing_day_id<>second_racing_day_id)
);

CREATE TABLE phase7_probation_control (
 singleton INTEGER PRIMARY KEY CHECK(singleton=1), state TEXT NOT NULL CHECK(state IN ('running','paused','reset','complete')),
 reason TEXT NOT NULL, generation INTEGER NOT NULL CHECK(generation>0), changed_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_probation_acceptances (
 generation INTEGER NOT NULL, racing_day_id TEXT NOT NULL UNIQUE REFERENCES phase7_day_evidence(racing_day_id),
 local_date TEXT NOT NULL, programme_checksum TEXT NOT NULL, accepted_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id), PRIMARY KEY(generation,local_date)
);

CREATE TABLE phase7_backups (
 backup_id TEXT PRIMARY KEY, racing_day_id TEXT NOT NULL REFERENCES phase7_reconciliation(racing_day_id),
 database_checksum TEXT NOT NULL, artifact_inventory_checksum TEXT NOT NULL,
 created_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TABLE phase7_restore_drills (
 drill_id TEXT PRIMARY KEY, backup_id TEXT NOT NULL REFERENCES phase7_backups(backup_id),
 database_verified INTEGER NOT NULL CHECK(database_verified IN (0,1)), artifacts_verified INTEGER NOT NULL CHECK(artifacts_verified IN (0,1)),
 application_readable INTEGER NOT NULL CHECK(application_readable IN (0,1)), successful INTEGER NOT NULL CHECK(successful IN (0,1)),
 verified_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 CHECK(successful=0 OR (database_verified=1 AND artifacts_verified=1 AND application_readable=1))
);

CREATE TRIGGER phase7_day_exact BEFORE INSERT ON phase7_day_evidence WHEN
 EXISTS (SELECT 1 FROM (SELECT NEW.reconciliation_checksum v UNION ALL SELECT NEW.restart_checksum UNION ALL SELECT NEW.ordering_checksum UNION ALL SELECT NEW.determinism_checksum) WHERE length(v)<>71 OR substr(v,1,7)<>'sha256:' OR substr(v,8) GLOB '*[^0-9a-f]*')
BEGIN SELECT RAISE(ABORT,'Phase 7 evidence checksum invalid'); END;
CREATE TRIGGER phase7_probation_auth BEFORE INSERT ON phase7_probation_acceptances WHEN NOT EXISTS (
 SELECT 1 FROM phase7_day_evidence e JOIN racing_days d USING(racing_day_id)
 JOIN races r USING(racing_day_id) JOIN expected_races x USING(race_id)
 WHERE e.racing_day_id=NEW.racing_day_id AND e.complete=1 AND e.critical_failure=0 AND d.local_date=NEW.local_date
 GROUP BY e.racing_day_id HAVING min(x.programme_checksum)=max(x.programme_checksum) AND min(x.programme_checksum)=NEW.programme_checksum)
BEGIN SELECT RAISE(ABORT,'probation day lacks authentic complete Racing Day authority'); END;
CREATE TRIGGER phase7_probation_to_phase6 BEFORE INSERT ON phase6_probation_days WHEN NOT EXISTS (
 SELECT 1 FROM phase7_probation_acceptances a JOIN phase7_day_evidence e USING(racing_day_id)
 WHERE a.local_date=NEW.racing_day AND e.reconciliation_checksum=NEW.reconciliation_checksum
 AND e.restart_checksum=NEW.restart_checksum AND e.ordering_checksum=NEW.ordering_checksum AND e.determinism_checksum=NEW.determinism_checksum)
BEGIN SELECT RAISE(ABORT,'probation evidence was not issued by Phase 7 authority'); END;
CREATE TRIGGER phase7_day_append_only_update BEFORE UPDATE ON phase7_day_evidence BEGIN SELECT RAISE(ABORT,'day evidence is append-only'); END;
CREATE TRIGGER phase7_day_append_only_delete BEFORE DELETE ON phase7_day_evidence BEGIN SELECT RAISE(ABORT,'day evidence is append-only'); END;
CREATE TRIGGER phase7_reconciliation_append_only_update BEFORE UPDATE ON phase7_reconciliation BEGIN SELECT RAISE(ABORT,'reconciliation is append-only'); END;
CREATE TRIGGER phase7_reconciliation_append_only_delete BEFORE DELETE ON phase7_reconciliation BEGIN SELECT RAISE(ABORT,'reconciliation is append-only'); END;
CREATE TRIGGER phase7_release_manifest_append_only_update BEFORE UPDATE ON phase7_release_manifests BEGIN SELECT RAISE(ABORT,'release manifests are append-only'); END;
CREATE TRIGGER phase7_release_manifest_append_only_delete BEFORE DELETE ON phase7_release_manifests BEGIN SELECT RAISE(ABORT,'release manifests are append-only'); END;
