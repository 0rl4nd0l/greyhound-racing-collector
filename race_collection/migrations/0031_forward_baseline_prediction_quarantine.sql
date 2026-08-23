CREATE TABLE forward_baseline_prediction_quarantines (
 cohort_id TEXT NOT NULL,
 race_id TEXT PRIMARY KEY,
 prediction_id TEXT NOT NULL UNIQUE CHECK(length(trim(prediction_id))>0),
 stage TEXT NOT NULL CHECK(stage IN ('identity','collection','sealing')),
 code TEXT NOT NULL CHECK(length(trim(code))>0),
 details TEXT NOT NULL CHECK(length(trim(details))>0),
 quarantined_at TEXT NOT NULL,
 collection_quarantine_operation_id TEXT NOT NULL UNIQUE,
 FOREIGN KEY(cohort_id,race_id)
  REFERENCES forward_baseline_cohort_members(cohort_id,race_id),
 FOREIGN KEY(collection_quarantine_operation_id)
  REFERENCES collection_quarantines(operation_id)
);

INSERT INTO forward_baseline_prediction_quarantines
SELECT m.cohort_id,q.race_id,'cohort-quarantine-'||q.operation_id,
 q.stage,q.code,q.details,q.created_at,q.operation_id
FROM collection_quarantines q
JOIN forward_baseline_cohort_members m ON m.race_id=q.race_id
WHERE q.quarantine_id=(
 SELECT MIN(first_q.quarantine_id) FROM collection_quarantines first_q
 WHERE first_q.race_id=q.race_id
);

CREATE TRIGGER forward_baseline_prediction_quarantines_exact_source
BEFORE INSERT ON forward_baseline_prediction_quarantines
WHEN NOT EXISTS (
 SELECT 1 FROM collection_quarantines q
 JOIN forward_baseline_cohort_members m ON m.race_id=q.race_id
 WHERE q.operation_id=NEW.collection_quarantine_operation_id
 AND q.race_id=NEW.race_id
 AND q.stage=NEW.stage
 AND q.code=NEW.code
 AND q.details=NEW.details
 AND q.created_at=NEW.quarantined_at
 AND m.cohort_id=NEW.cohort_id
)
BEGIN SELECT RAISE(ABORT,'baseline prediction quarantine source disagrees'); END;

CREATE TRIGGER forward_baseline_prediction_quarantines_append_only_update
BEFORE UPDATE ON forward_baseline_prediction_quarantines
BEGIN SELECT RAISE(ABORT,'baseline prediction quarantine is append-only'); END;
CREATE TRIGGER forward_baseline_prediction_quarantines_append_only_delete
BEFORE DELETE ON forward_baseline_prediction_quarantines
BEGIN SELECT RAISE(ABORT,'baseline prediction quarantine is append-only'); END;
