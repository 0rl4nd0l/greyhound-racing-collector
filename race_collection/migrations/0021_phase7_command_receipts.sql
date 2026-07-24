CREATE TABLE phase7_application_command_receipts (
 command_operation_id TEXT PRIMARY KEY REFERENCES operations(operation_id),
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 phase_name TEXT NOT NULL CHECK(phase_name IN (
  'discover_programme','collect_cards_and_form','collect_adaptive_odds',
  'close_and_seal','deferred_prediction','collect_results',
  'join_training_examples','reconcile','request_training')),
 result_json TEXT NOT NULL CHECK(json_valid(result_json)), result_checksum TEXT NOT NULL,
 committed_at TEXT NOT NULL, UNIQUE(racing_day_id,phase_name)
);
CREATE TRIGGER phase7_receipt_exact BEFORE INSERT ON phase7_application_command_receipts WHEN
 length(NEW.result_checksum)<>71 OR substr(NEW.result_checksum,1,7)<>'sha256:'
 OR NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.command_operation_id
  AND o.kind='phase7_command_'||NEW.phase_name)
BEGIN SELECT RAISE(ABORT,'application receipt lacks exact phase operation authority'); END;
CREATE TRIGGER phase7_receipt_append_only_update BEFORE UPDATE ON phase7_application_command_receipts BEGIN SELECT RAISE(ABORT,'application receipts are append-only'); END;
CREATE TRIGGER phase7_receipt_append_only_delete BEFORE DELETE ON phase7_application_command_receipts BEGIN SELECT RAISE(ABORT,'application receipts are append-only'); END;

DROP TRIGGER phase7_progress_exact;
CREATE TRIGGER phase7_progress_exact BEFORE INSERT ON phase7_scheduler_progress WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id AND o.kind='phase7_advance_phase')
 OR NEW.command_operation_id=NEW.operation_id
 OR NOT EXISTS (SELECT 1 FROM phase7_application_command_receipts c
  WHERE c.command_operation_id=NEW.command_operation_id AND c.racing_day_id=NEW.racing_day_id
   AND c.phase_name=NEW.phase_name AND c.result_json=NEW.result_json AND c.result_checksum=NEW.result_checksum)
 OR NEW.phase_ordinal<>(SELECT count(*)+1 FROM phase7_scheduler_progress p WHERE p.racing_day_id=NEW.racing_day_id)
 OR NOT EXISTS (SELECT 1 FROM phase7_scheduler_lease l WHERE l.singleton=1 AND l.generation=NEW.lease_generation AND l.expires_at>NEW.completed_at)
BEGIN SELECT RAISE(ABORT,'scheduler advancement lacks exact ordered command receipt'); END;
