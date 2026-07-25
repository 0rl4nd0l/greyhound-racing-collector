CREATE TABLE phase7_rejected_result_commands (
 operation_id TEXT PRIMARY KEY REFERENCES operations(operation_id),
 racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
 race_id TEXT NOT NULL REFERENCES races(race_id),
 reason TEXT NOT NULL CHECK(reason='result_before_prediction'),
 rejected_at TEXT NOT NULL,
 UNIQUE(race_id,operation_id)
);

CREATE TRIGGER phase7_rejected_result_exact BEFORE INSERT ON phase7_rejected_result_commands WHEN
 NOT EXISTS (SELECT 1 FROM operations o WHERE o.operation_id=NEW.operation_id
  AND o.kind='phase7_reject_result_before_prediction')
 OR NOT EXISTS (SELECT 1 FROM races r WHERE r.race_id=NEW.race_id
  AND r.racing_day_id=NEW.racing_day_id)
 OR EXISTS (SELECT 1 FROM deferred_predictions p WHERE p.race_id=NEW.race_id)
BEGIN SELECT RAISE(ABORT,'result rejection lacks an authentic application barrier'); END;

CREATE TRIGGER phase7_rejected_result_append_only_update
BEFORE UPDATE ON phase7_rejected_result_commands
BEGIN SELECT RAISE(ABORT,'result rejection receipts are append-only'); END;
CREATE TRIGGER phase7_rejected_result_append_only_delete
BEFORE DELETE ON phase7_rejected_result_commands
BEGIN SELECT RAISE(ABORT,'result rejection receipts are append-only'); END;
