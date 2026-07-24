CREATE TABLE canonical_training_examples (
    training_example_id TEXT PRIMARY KEY CHECK(length(trim(training_example_id)) > 0),
    phase3_training_example_id TEXT NOT NULL UNIQUE REFERENCES training_examples(training_example_id),
    race_id TEXT NOT NULL UNIQUE REFERENCES races(race_id),
    evidence_checksum TEXT NOT NULL CHECK(length(evidence_checksum)=71 AND substr(evidence_checksum,1,7)='sha256:' AND substr(evidence_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    result_checksum TEXT NOT NULL CHECK(length(result_checksum)=71 AND substr(result_checksum,1,7)='sha256:' AND substr(result_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    feature_matrix_checksum TEXT NOT NULL CHECK(length(feature_matrix_checksum)=71 AND substr(feature_matrix_checksum,1,7)='sha256:' AND substr(feature_matrix_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    artifact_checksum TEXT NOT NULL UNIQUE CHECK(length(artifact_checksum)=71 AND substr(artifact_checksum,1,7)='sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    racing_date TEXT NOT NULL CHECK(racing_date GLOB '????-??-??'),
    joined_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TRIGGER canonical_training_example_exact_relations
BEFORE INSERT ON canonical_training_examples
WHEN NOT EXISTS (
  SELECT 1 FROM training_examples t
  JOIN deferred_predictions p ON p.prediction_id=t.prediction_id AND p.race_id=t.race_id
  JOIN sealed_evidence s ON s.seal_id=p.seal_id AND s.race_id=t.race_id
  JOIN result_attempts a ON a.attempt_id=t.result_attempt_id AND a.race_id=t.race_id
  JOIN races r ON r.race_id=t.race_id
  JOIN racing_days d ON d.racing_day_id=r.racing_day_id
  WHERE t.training_example_id=NEW.phase3_training_example_id
    AND t.race_id=NEW.race_id AND t.eligibility='eligible'
    AND r.state='training_example_ready'
    AND s.normalized_checksum=NEW.evidence_checksum
    AND a.status='collected' AND a.artifact_checksum=NEW.result_checksum
    AND d.local_date=NEW.racing_date
)
BEGIN SELECT RAISE(ABORT,'canonical training example relations disagree'); END;

CREATE TRIGGER canonical_training_examples_append_only_update
BEFORE UPDATE ON canonical_training_examples
BEGIN SELECT RAISE(ABORT,'canonical training examples are append-only'); END;

CREATE TRIGGER canonical_training_examples_append_only_delete
BEFORE DELETE ON canonical_training_examples
BEGIN SELECT RAISE(ABORT,'canonical training examples are append-only'); END;
