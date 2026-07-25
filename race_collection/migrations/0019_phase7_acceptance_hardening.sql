CREATE TABLE phase7_operational_evidence (
 artifact_checksum TEXT PRIMARY KEY,
 evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('restart','ordering','determinism','reconciliation','cutover')),
 byte_size INTEGER NOT NULL CHECK(byte_size>0), verified_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_evidence_authority BEFORE INSERT ON phase7_operational_evidence
WHEN NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
 AND o.kind IN ('phase7_record_operational_evidence','phase7_reconciliation','phase7_cutover_eligibility'))
BEGIN SELECT RAISE(ABORT,'operational evidence lacks application operation authority'); END;
CREATE TRIGGER phase7_evidence_append_only_update BEFORE UPDATE ON phase7_operational_evidence BEGIN SELECT RAISE(ABORT,'operational evidence is append-only'); END;
CREATE TRIGGER phase7_evidence_append_only_delete BEFORE DELETE ON phase7_operational_evidence BEGIN SELECT RAISE(ABORT,'operational evidence is append-only'); END;

DROP TRIGGER phase7_day_exact;
CREATE TRIGGER phase7_day_exact BEFORE INSERT ON phase7_day_evidence WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_reconciliation')
 OR NOT EXISTS (SELECT 1 FROM phase7_release_manifests m WHERE m.release_id=NEW.release_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a WHERE a.artifact_checksum=NEW.reconciliation_checksum AND a.evidence_kind='reconciliation')
 OR NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a WHERE a.artifact_checksum=NEW.restart_checksum AND a.evidence_kind='restart')
 OR NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a WHERE a.artifact_checksum=NEW.ordering_checksum AND a.evidence_kind='ordering')
 OR NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a WHERE a.artifact_checksum=NEW.determinism_checksum AND a.evidence_kind='determinism')
BEGIN SELECT RAISE(ABORT,'Phase 7 day lacks verified operational evidence authority'); END;

CREATE TRIGGER phase7_cutover_exact BEFORE INSERT ON phase7_cutover_eligibility WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_cutover_eligibility')
 OR NOT EXISTS (SELECT 1 FROM phase7_operational_evidence a WHERE a.artifact_checksum=NEW.evidence_checksum AND a.evidence_kind='cutover')
 OR NOT EXISTS (SELECT 1 FROM phase7_day_evidence first JOIN racing_days d1 ON d1.racing_day_id=first.racing_day_id
   JOIN phase7_day_evidence second ON second.racing_day_id=NEW.second_racing_day_id
   JOIN racing_days d2 ON d2.racing_day_id=second.racing_day_id
   WHERE first.racing_day_id=NEW.first_racing_day_id AND first.complete=1 AND first.critical_failure=0
   AND second.complete=1 AND second.critical_failure=0 AND julianday(d2.local_date)-julianday(d1.local_date)=1)
BEGIN SELECT RAISE(ABORT,'cutover lacks two consecutive authenticated complete days'); END;

DROP TRIGGER phase7_probation_auth;
CREATE TRIGGER phase7_probation_auth BEFORE INSERT ON phase7_probation_acceptances WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_accept_probation_day')
 OR NOT EXISTS (
 SELECT 1 FROM phase7_day_evidence e JOIN racing_days d USING(racing_day_id)
 JOIN races r USING(racing_day_id) JOIN expected_races x USING(race_id)
 WHERE e.racing_day_id=NEW.racing_day_id AND e.complete=1 AND e.critical_failure=0 AND d.local_date=NEW.local_date
 GROUP BY e.racing_day_id HAVING min(x.programme_checksum)=max(x.programme_checksum) AND min(x.programme_checksum)=NEW.programme_checksum)
BEGIN SELECT RAISE(ABORT,'probation day lacks authentic complete Racing Day authority'); END;

CREATE TABLE phase7_scheduler_progress (
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id), phase_ordinal INTEGER NOT NULL CHECK(phase_ordinal BETWEEN 1 AND 9),
 phase_name TEXT NOT NULL, lease_generation INTEGER NOT NULL REFERENCES phase7_scheduler_history(generation),
 completed_at TEXT NOT NULL, operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 PRIMARY KEY(racing_day_id,phase_ordinal), UNIQUE(racing_day_id,phase_name)
);
CREATE TRIGGER phase7_progress_exact BEFORE INSERT ON phase7_scheduler_progress WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_advance_phase')
 OR NEW.phase_ordinal<>(SELECT count(*)+1 FROM phase7_scheduler_progress p WHERE p.racing_day_id=NEW.racing_day_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease l WHERE l.singleton=1 AND l.generation=NEW.lease_generation AND l.expires_at>NEW.completed_at)
BEGIN SELECT RAISE(ABORT,'scheduler advancement lacks live ordered lease authority'); END;
CREATE TRIGGER phase7_progress_append_only_update BEFORE UPDATE ON phase7_scheduler_progress BEGIN SELECT RAISE(ABORT,'scheduler progress is append-only'); END;
CREATE TRIGGER phase7_progress_append_only_delete BEFORE DELETE ON phase7_scheduler_progress BEGIN SELECT RAISE(ABORT,'scheduler progress is append-only'); END;
