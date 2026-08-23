CREATE TABLE forward_baseline_prediction_quarantines (
 race_id TEXT PRIMARY KEY REFERENCES forward_baseline_cohort_members(race_id),
 cohort_id TEXT NOT NULL REFERENCES forward_baseline_cohorts(cohort_id),
 source_collection_quarantine_id INTEGER NOT NULL REFERENCES collection_quarantines(quarantine_id),
 code TEXT NOT NULL CHECK(length(trim(code)) > 0),
 details TEXT NOT NULL CHECK(length(trim(details)) > 0),
 quarantined_at TEXT NOT NULL,
 operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
 FOREIGN KEY(cohort_id,race_id)
  REFERENCES forward_baseline_cohort_members(cohort_id,race_id)
);

CREATE TRIGGER forward_baseline_prediction_quarantines_append_only_update
BEFORE UPDATE ON forward_baseline_prediction_quarantines
BEGIN SELECT RAISE(ABORT,'forward baseline prediction quarantine is append-only'); END;
CREATE TRIGGER forward_baseline_prediction_quarantines_append_only_delete
BEFORE DELETE ON forward_baseline_prediction_quarantines
BEGIN SELECT RAISE(ABORT,'forward baseline prediction quarantine is append-only'); END;
