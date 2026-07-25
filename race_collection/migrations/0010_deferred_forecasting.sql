CREATE TABLE model_bundles (
    bundle_id TEXT PRIMARY KEY CHECK (length(trim(bundle_id)) > 0),
    origin TEXT NOT NULL CHECK (origin = 'legacy-origin'),
    model_id TEXT NOT NULL CHECK (length(trim(model_id)) > 0),
    artifact_checksum TEXT NOT NULL CHECK (length(artifact_checksum) = 71 AND substr(artifact_checksum,1,7) = 'sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    artifact_size INTEGER NOT NULL CHECK (artifact_size > 0),
    metadata_checksum TEXT NOT NULL CHECK (length(metadata_checksum) = 71 AND substr(metadata_checksum,1,7) = 'sha256:' AND substr(metadata_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    scaler_checksum TEXT CHECK (scaler_checksum IS NULL OR (length(scaler_checksum) = 71 AND substr(scaler_checksum,1,7) = 'sha256:' AND substr(scaler_checksum,8) NOT GLOB '*[^0-9a-f]*')),
    envelope_kind TEXT NOT NULL CHECK (envelope_kind IN ('raw_registry_model', 'v4_full_envelope')),
    provenance_json TEXT NOT NULL CHECK (json_valid(provenance_json) AND json_type(provenance_json) = 'object'),
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE model_releases (
    release_id TEXT PRIMARY KEY CHECK (length(trim(release_id)) > 0),
    bundle_id TEXT NOT NULL REFERENCES model_bundles(bundle_id),
    policy_id TEXT NOT NULL CHECK (length(trim(policy_id)) > 0),
    descriptor_json TEXT NOT NULL CHECK (json_valid(descriptor_json) AND json_type(descriptor_json) = 'object'),
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE racing_day_pins (
    racing_day_id TEXT PRIMARY KEY REFERENCES racing_days(racing_day_id),
    bundle_id TEXT NOT NULL REFERENCES model_bundles(bundle_id),
    release_id TEXT NOT NULL REFERENCES model_releases(release_id),
    policy_id TEXT NOT NULL CHECK (length(trim(policy_id)) > 0),
    pinned_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE prediction_begins (
    race_id TEXT PRIMARY KEY REFERENCES races(race_id),
    authority_snapshot_json TEXT NOT NULL CHECK (json_valid(authority_snapshot_json) AND json_type(authority_snapshot_json)='object'),
    begun_at TEXT NOT NULL,
    request_intent_sha256 TEXT NOT NULL CHECK (length(request_intent_sha256)=64 AND request_intent_sha256 NOT GLOB '*[^0-9a-f]*'),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE deferred_predictions (
    prediction_id TEXT PRIMARY KEY CHECK (length(trim(prediction_id)) > 0),
    race_id TEXT NOT NULL UNIQUE REFERENCES races(race_id),
    racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
    bundle_id TEXT NOT NULL REFERENCES model_bundles(bundle_id),
    release_id TEXT NOT NULL REFERENCES model_releases(release_id),
    policy_id TEXT NOT NULL CHECK (length(trim(policy_id)) > 0),
    seal_id INTEGER NOT NULL REFERENCES sealed_evidence(seal_id),
    evidence_checksum TEXT NOT NULL CHECK (length(evidence_checksum) = 71 AND substr(evidence_checksum,1,7) = 'sha256:' AND substr(evidence_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    artifact_checksum TEXT NOT NULL CHECK (length(artifact_checksum) = 71 AND substr(artifact_checksum,1,7) = 'sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    computed_at TEXT NOT NULL,
    request_intent_sha256 TEXT NOT NULL CHECK (length(request_intent_sha256) = 64 AND request_intent_sha256 NOT GLOB '*[^0-9a-f]*'),
    authority_snapshot_json TEXT NOT NULL CHECK (json_valid(authority_snapshot_json) AND json_type(authority_snapshot_json) = 'object'),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE prediction_quarantines (
    race_id TEXT PRIMARY KEY REFERENCES races(race_id),
    prediction_id TEXT NOT NULL UNIQUE CHECK (length(trim(prediction_id)) > 0),
    racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
    bundle_id TEXT NOT NULL REFERENCES model_bundles(bundle_id),
    release_id TEXT NOT NULL REFERENCES model_releases(release_id),
    policy_id TEXT NOT NULL CHECK (length(trim(policy_id)) > 0),
    seal_id INTEGER NOT NULL REFERENCES sealed_evidence(seal_id),
    evidence_checksum TEXT NOT NULL CHECK (length(evidence_checksum) = 71 AND substr(evidence_checksum,1,7) = 'sha256:' AND substr(evidence_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    code TEXT NOT NULL CHECK (length(trim(code)) > 0),
    details TEXT NOT NULL CHECK (length(trim(details)) > 0),
    quarantined_at TEXT NOT NULL,
    request_intent_sha256 TEXT NOT NULL CHECK (length(request_intent_sha256) = 64 AND request_intent_sha256 NOT GLOB '*[^0-9a-f]*'),
    authority_snapshot_json TEXT NOT NULL CHECK (json_valid(authority_snapshot_json) AND json_type(authority_snapshot_json) = 'object'),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE result_attempts (
    attempt_id TEXT PRIMARY KEY CHECK (length(trim(attempt_id)) > 0),
    race_id TEXT NOT NULL REFERENCES deferred_predictions(race_id),
    attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
    max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
    deadline TEXT NOT NULL CHECK (length(trim(deadline)) > 0),
    retry_policy_version TEXT NOT NULL CHECK (retry_policy_version = 'result-retry-v1'),
    min_backoff_seconds INTEGER NOT NULL CHECK (min_backoff_seconds = 1),
    status TEXT NOT NULL CHECK (status IN ('failed', 'collected', 'quarantined')),
    artifact_checksum TEXT CHECK (artifact_checksum IS NULL OR (length(artifact_checksum) = 71 AND substr(artifact_checksum,1,7) = 'sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*')),
    outcome_json TEXT CHECK (outcome_json IS NULL OR json_valid(outcome_json)),
    error TEXT,
    attempted_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    UNIQUE (race_id, attempt_number),
    CHECK ((status = 'collected' AND artifact_checksum IS NOT NULL AND outcome_json IS NOT NULL AND error IS NULL) OR
           (status <> 'collected' AND artifact_checksum IS NULL AND outcome_json IS NULL AND error IS NOT NULL AND length(trim(error)) > 0))
);

CREATE TABLE training_examples (
    training_example_id TEXT PRIMARY KEY CHECK (length(trim(training_example_id)) > 0),
    race_id TEXT NOT NULL UNIQUE REFERENCES deferred_predictions(race_id),
    prediction_id TEXT NOT NULL UNIQUE REFERENCES deferred_predictions(prediction_id),
    result_attempt_id TEXT NOT NULL UNIQUE REFERENCES result_attempts(attempt_id),
    artifact_checksum TEXT NOT NULL CHECK (length(artifact_checksum) = 71 AND substr(artifact_checksum,1,7) = 'sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    eligibility TEXT NOT NULL CHECK (eligibility IN ('eligible', 'evaluation_ineligible')),
    reason TEXT,
    joined_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    CHECK ((eligibility='eligible' AND reason IS NULL) OR
           (eligibility='evaluation_ineligible' AND reason IS NOT NULL AND length(trim(reason)) > 0))
);

CREATE TABLE on_demand_forecasts (
    forecast_id TEXT PRIMARY KEY CHECK (length(trim(forecast_id)) > 0),
    race_id TEXT NOT NULL REFERENCES races(race_id),
    artifact_checksum TEXT NOT NULL CHECK (length(artifact_checksum) = 71 AND substr(artifact_checksum,1,7) = 'sha256:' AND substr(artifact_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    evidence_checksum TEXT NOT NULL CHECK (length(evidence_checksum) = 71 AND substr(evidence_checksum,1,7) = 'sha256:' AND substr(evidence_checksum,8) NOT GLOB '*[^0-9a-f]*'),
    computed_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE INDEX result_attempts_by_race ON result_attempts(race_id, attempt_number);

CREATE TRIGGER deferred_predictions_exact_relations
BEFORE INSERT ON deferred_predictions
WHEN EXISTS (SELECT 1 FROM prediction_quarantines q WHERE q.race_id=NEW.race_id)
  OR NOT EXISTS (
    SELECT 1 FROM races r
    JOIN sealed_evidence s ON s.seal_id = NEW.seal_id AND s.race_id = r.race_id
    JOIN racing_day_pins p ON p.racing_day_id = r.racing_day_id
    WHERE r.race_id = NEW.race_id
      AND r.state = 'prediction_pending'
      AND r.racing_day_id = NEW.racing_day_id
      AND p.bundle_id = NEW.bundle_id
      AND p.release_id = NEW.release_id
      AND p.policy_id = NEW.policy_id
      AND s.normalized_checksum = NEW.evidence_checksum
)
BEGIN SELECT RAISE(ABORT, 'deferred prediction relations disagree'); END;

CREATE TRIGGER prediction_quarantines_exact_day
BEFORE INSERT ON prediction_quarantines
WHEN EXISTS (SELECT 1 FROM deferred_predictions p WHERE p.race_id=NEW.race_id)
  OR NOT EXISTS (
    SELECT 1 FROM races r
    JOIN sealed_evidence s ON s.seal_id=NEW.seal_id AND s.race_id=r.race_id
    JOIN racing_day_pins p ON p.racing_day_id=r.racing_day_id
    WHERE r.race_id=NEW.race_id AND r.state='prediction_pending'
      AND r.racing_day_id=NEW.racing_day_id
      AND p.bundle_id=NEW.bundle_id AND p.release_id=NEW.release_id
      AND p.policy_id=NEW.policy_id AND s.normalized_checksum=NEW.evidence_checksum
)
BEGIN SELECT RAISE(ABORT, 'prediction quarantine relations disagree'); END;

CREATE TRIGGER deferred_predictions_exact_snapshot BEFORE INSERT ON deferred_predictions
WHEN NOT EXISTS (
  SELECT 1 FROM races r JOIN racing_days d USING(racing_day_id)
  JOIN racing_day_pins p USING(racing_day_id)
  JOIN sealed_evidence s ON s.seal_id=NEW.seal_id AND s.race_id=r.race_id
  JOIN model_releases m ON m.release_id=p.release_id
  WHERE r.race_id=NEW.race_id
    AND json_extract(NEW.authority_snapshot_json,'$.race')=NEW.race_id
    AND json_extract(NEW.authority_snapshot_json,'$.prediction')=NEW.prediction_id
    AND json_extract(NEW.authority_snapshot_json,'$.day')=NEW.racing_day_id
    AND json_extract(NEW.authority_snapshot_json,'$.seal')=NEW.seal_id
    AND json_extract(NEW.authority_snapshot_json,'$.evidence')=NEW.evidence_checksum
    AND json_extract(NEW.authority_snapshot_json,'$.bundle')=NEW.bundle_id
    AND json_extract(NEW.authority_snapshot_json,'$.release')=NEW.release_id
    AND json_extract(NEW.authority_snapshot_json,'$.policy')=NEW.policy_id
    AND json_extract(NEW.authority_snapshot_json,'$.closed_at')=d.closed_at
    AND json_extract(NEW.authority_snapshot_json,'$.race_updated_at')=r.updated_at
    AND json(json_extract(NEW.authority_snapshot_json,'$.descriptor'))=json(m.descriptor_json)
)
BEGIN SELECT RAISE(ABORT, 'deferred prediction snapshot disagrees'); END;

CREATE TRIGGER prediction_quarantines_exact_snapshot BEFORE INSERT ON prediction_quarantines
WHEN NOT EXISTS (
  SELECT 1 FROM races r JOIN racing_days d USING(racing_day_id)
  JOIN racing_day_pins p USING(racing_day_id)
  JOIN sealed_evidence s ON s.seal_id=NEW.seal_id AND s.race_id=r.race_id
  JOIN model_releases m ON m.release_id=p.release_id
  WHERE r.race_id=NEW.race_id
    AND json_extract(NEW.authority_snapshot_json,'$.race')=NEW.race_id
    AND json_extract(NEW.authority_snapshot_json,'$.prediction')=NEW.prediction_id
    AND json_extract(NEW.authority_snapshot_json,'$.day')=NEW.racing_day_id
    AND json_extract(NEW.authority_snapshot_json,'$.seal')=NEW.seal_id
    AND json_extract(NEW.authority_snapshot_json,'$.evidence')=NEW.evidence_checksum
    AND json_extract(NEW.authority_snapshot_json,'$.bundle')=NEW.bundle_id
    AND json_extract(NEW.authority_snapshot_json,'$.release')=NEW.release_id
    AND json_extract(NEW.authority_snapshot_json,'$.policy')=NEW.policy_id
    AND json_extract(NEW.authority_snapshot_json,'$.closed_at')=d.closed_at
    AND json_extract(NEW.authority_snapshot_json,'$.race_updated_at')=r.updated_at
    AND json(json_extract(NEW.authority_snapshot_json,'$.descriptor'))=json(m.descriptor_json)
)
BEGIN SELECT RAISE(ABORT, 'prediction quarantine snapshot disagrees'); END;

CREATE TRIGGER model_releases_exact_bundle BEFORE INSERT ON model_releases
WHEN NOT EXISTS (SELECT 1 FROM model_bundles b WHERE b.bundle_id=NEW.bundle_id)
BEGIN SELECT RAISE(ABORT, 'model release bundle disagrees'); END;

CREATE TRIGGER racing_day_pins_exact_release BEFORE INSERT ON racing_day_pins
WHEN NOT EXISTS (SELECT 1 FROM model_releases m WHERE m.release_id=NEW.release_id
  AND m.bundle_id=NEW.bundle_id AND m.policy_id=NEW.policy_id)
BEGIN SELECT RAISE(ABORT, 'day pin release disagrees'); END;

CREATE TRIGGER prediction_begins_exact_snapshot BEFORE INSERT ON prediction_begins
WHEN NOT EXISTS (
  SELECT 1 FROM races r JOIN racing_days d USING(racing_day_id)
  JOIN racing_day_pins p USING(racing_day_id)
  JOIN sealed_evidence s ON s.race_id=r.race_id
  JOIN model_releases m ON m.release_id=p.release_id
  WHERE r.race_id=NEW.race_id AND r.state='awaiting_day_close'
    AND json_extract(NEW.authority_snapshot_json,'$.race')=r.race_id
    AND json_extract(NEW.authority_snapshot_json,'$.prediction')='begin-'||r.race_id
    AND json_extract(NEW.authority_snapshot_json,'$.day')=r.racing_day_id
    AND json_extract(NEW.authority_snapshot_json,'$.seal')=s.seal_id
    AND json_extract(NEW.authority_snapshot_json,'$.evidence')=s.normalized_checksum
    AND json_extract(NEW.authority_snapshot_json,'$.bundle')=p.bundle_id
    AND json_extract(NEW.authority_snapshot_json,'$.release')=p.release_id
    AND json_extract(NEW.authority_snapshot_json,'$.policy')=p.policy_id
    AND json_extract(NEW.authority_snapshot_json,'$.closed_at')=d.closed_at
    AND json_extract(NEW.authority_snapshot_json,'$.race_updated_at')=r.updated_at
    AND json(json_extract(NEW.authority_snapshot_json,'$.descriptor'))=json(m.descriptor_json)
)
BEGIN SELECT RAISE(ABORT, 'prediction begin snapshot disagrees'); END;

CREATE TRIGGER result_attempts_exact_prediction BEFORE INSERT ON result_attempts
WHEN NOT EXISTS (SELECT 1 FROM deferred_predictions p JOIN races r USING(race_id)
  WHERE p.race_id=NEW.race_id AND r.state='result_pending')
BEGIN SELECT RAISE(ABORT, 'result race lacks exact prediction'); END;

CREATE TRIGGER result_attempts_exact_shape BEFORE INSERT ON result_attempts
WHEN NEW.status='collected' AND (
  json_type(NEW.outcome_json) <> 'object'
  OR json_type(NEW.outcome_json,'$.order') <> 'array'
  OR json_array_length(NEW.outcome_json,'$.order') = 0
  OR EXISTS (SELECT 1 FROM json_each(NEW.outcome_json,'$.order')
             WHERE type <> 'integer' OR value <= 0)
  OR EXISTS (SELECT value FROM json_each(NEW.outcome_json,'$.order')
             GROUP BY value HAVING COUNT(*) > 1)
)
BEGIN SELECT RAISE(ABORT, 'collected result shape is invalid'); END;

CREATE TRIGGER result_attempts_exact_retry_policy BEFORE INSERT ON result_attempts
WHEN julianday(NEW.attempted_at) IS NULL OR julianday(NEW.deadline) IS NULL
  OR NEW.attempt_number <> 1 + (SELECT COUNT(*) FROM result_attempts a WHERE a.race_id=NEW.race_id)
  OR EXISTS (SELECT 1 FROM result_attempts a WHERE a.race_id=NEW.race_id
             AND a.status IN ('collected','quarantined'))
  OR EXISTS (
    SELECT 1 FROM result_attempts a WHERE a.race_id=NEW.race_id
      AND (a.max_attempts<>NEW.max_attempts OR julianday(a.deadline)<>julianday(NEW.deadline)
           OR a.retry_policy_version<>NEW.retry_policy_version
           OR a.min_backoff_seconds<>NEW.min_backoff_seconds)
  )
  OR EXISTS (
    SELECT 1 FROM result_attempts a WHERE a.race_id=NEW.race_id
      AND a.attempt_number=NEW.attempt_number-1
      AND (julianday(NEW.attempted_at)-julianday(a.attempted_at))*86400.0 < NEW.min_backoff_seconds
  )
  OR (NEW.status='failed' AND
      (NEW.attempt_number>=NEW.max_attempts OR julianday(NEW.attempted_at)>=julianday(NEW.deadline)))
  OR (NEW.status='quarantined' AND
      NEW.attempt_number<NEW.max_attempts AND julianday(NEW.attempted_at)<julianday(NEW.deadline))
BEGIN SELECT RAISE(ABORT, 'result retry policy disagrees'); END;

CREATE TRIGGER training_examples_exact_relations
BEFORE INSERT ON training_examples
WHEN NOT EXISTS (
    SELECT 1 FROM deferred_predictions p JOIN result_attempts a USING (race_id)
    WHERE p.race_id = NEW.race_id
      AND p.prediction_id = NEW.prediction_id
      AND a.attempt_id = NEW.result_attempt_id
      AND a.status = 'collected'
      AND EXISTS (SELECT 1 FROM races r WHERE r.race_id=NEW.race_id
                  AND r.state='result_collected')
)
BEGIN SELECT RAISE(ABORT, 'training example relations disagree'); END;

CREATE TRIGGER model_bundles_append_only_update BEFORE UPDATE ON model_bundles BEGIN SELECT RAISE(ABORT, 'model_bundles is append-only'); END;
CREATE TRIGGER model_bundles_append_only_delete BEFORE DELETE ON model_bundles BEGIN SELECT RAISE(ABORT, 'model_bundles is append-only'); END;
CREATE TRIGGER model_releases_append_only_update BEFORE UPDATE ON model_releases BEGIN SELECT RAISE(ABORT, 'model_releases is append-only'); END;
CREATE TRIGGER model_releases_append_only_delete BEFORE DELETE ON model_releases BEGIN SELECT RAISE(ABORT, 'model_releases is append-only'); END;
CREATE TRIGGER racing_day_pins_append_only_update BEFORE UPDATE ON racing_day_pins BEGIN SELECT RAISE(ABORT, 'racing_day_pins is append-only'); END;
CREATE TRIGGER racing_day_pins_append_only_delete BEFORE DELETE ON racing_day_pins BEGIN SELECT RAISE(ABORT, 'racing_day_pins is append-only'); END;
CREATE TRIGGER prediction_begins_append_only_update BEFORE UPDATE ON prediction_begins BEGIN SELECT RAISE(ABORT, 'prediction_begins is append-only'); END;
CREATE TRIGGER prediction_begins_append_only_delete BEFORE DELETE ON prediction_begins BEGIN SELECT RAISE(ABORT, 'prediction_begins is append-only'); END;
CREATE TRIGGER deferred_predictions_append_only_update BEFORE UPDATE ON deferred_predictions BEGIN SELECT RAISE(ABORT, 'deferred_predictions is append-only'); END;
CREATE TRIGGER deferred_predictions_append_only_delete BEFORE DELETE ON deferred_predictions BEGIN SELECT RAISE(ABORT, 'deferred_predictions is append-only'); END;
CREATE TRIGGER prediction_quarantines_append_only_update BEFORE UPDATE ON prediction_quarantines BEGIN SELECT RAISE(ABORT, 'prediction_quarantines is append-only'); END;
CREATE TRIGGER prediction_quarantines_append_only_delete BEFORE DELETE ON prediction_quarantines BEGIN SELECT RAISE(ABORT, 'prediction_quarantines is append-only'); END;
CREATE TRIGGER result_attempts_append_only_update BEFORE UPDATE ON result_attempts BEGIN SELECT RAISE(ABORT, 'result_attempts is append-only'); END;
CREATE TRIGGER result_attempts_append_only_delete BEFORE DELETE ON result_attempts BEGIN SELECT RAISE(ABORT, 'result_attempts is append-only'); END;
CREATE TRIGGER training_examples_append_only_update BEFORE UPDATE ON training_examples BEGIN SELECT RAISE(ABORT, 'training_examples is append-only'); END;
CREATE TRIGGER training_examples_append_only_delete BEFORE DELETE ON training_examples BEGIN SELECT RAISE(ABORT, 'training_examples is append-only'); END;
CREATE TRIGGER on_demand_forecasts_append_only_update BEFORE UPDATE ON on_demand_forecasts BEGIN SELECT RAISE(ABORT, 'on_demand_forecasts is append-only'); END;
CREATE TRIGGER on_demand_forecasts_append_only_delete BEFORE DELETE ON on_demand_forecasts BEGIN SELECT RAISE(ABORT, 'on_demand_forecasts is append-only'); END;
