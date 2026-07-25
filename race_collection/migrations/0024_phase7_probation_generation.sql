CREATE TABLE phase7_probation_seals (
 generation INTEGER PRIMARY KEY,
 probation_id TEXT NOT NULL UNIQUE REFERENCES phase6_probation_states(probation_id),
 release_id TEXT NOT NULL REFERENCES phase7_release_manifests(release_id),
 cutover_operation_id TEXT NOT NULL REFERENCES operations(operation_id),
 state_checksum TEXT NOT NULL,
 sealed_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);
CREATE TRIGGER phase7_probation_seal_exact BEFORE INSERT ON phase7_probation_seals WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_seal_probation')
 OR NOT EXISTS (SELECT 1 FROM phase7_probation_control c WHERE c.singleton=1
  AND c.generation=NEW.generation AND c.state='complete')
 OR NOT EXISTS (SELECT 1 FROM phase7_release_history h
  WHERE h.operation_id=NEW.cutover_operation_id AND h.release_id=NEW.release_id
  AND h.action='activate')
BEGIN SELECT RAISE(ABORT,'probation seal lacks exact generation and cutover authority'); END;
CREATE TRIGGER phase7_probation_seals_append_only_update BEFORE UPDATE ON phase7_probation_seals
BEGIN SELECT RAISE(ABORT,'probation seals are append-only'); END;
CREATE TRIGGER phase7_probation_seals_append_only_delete BEFORE DELETE ON phase7_probation_seals
BEGIN SELECT RAISE(ABORT,'probation seals are append-only'); END;
