CREATE TABLE expected_races (
    race_id TEXT PRIMARY KEY REFERENCES races(race_id),
    source TEXT NOT NULL,
    source_race_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_number INTEGER NOT NULL CHECK (race_number > 0),
    scheduled_jump TEXT NOT NULL,
    programme_checksum TEXT NOT NULL CHECK (programme_checksum GLOB 'sha256:[0-9a-f]*'),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    UNIQUE (source, source_race_id),
    UNIQUE (race_id, venue, race_number)
);

CREATE TABLE dog_identity_decisions (
    decision_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_alias TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    tier TEXT NOT NULL CHECK (tier IN (
        'authoritative', 'high_confidence_provisional', 'ambiguous'
    )),
    dog_id TEXT REFERENCES dogs(dog_id),
    reason TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    CHECK ((tier = 'ambiguous' AND dog_id IS NULL) OR
           (tier <> 'ambiguous' AND dog_id IS NOT NULL))
);

CREATE TABLE odds_attempts (
    attempt_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    source TEXT NOT NULL,
    attempted_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    artifact_checksum TEXT,
    runner_mapping_checksum TEXT,
    error TEXT,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    CHECK ((status = 'succeeded' AND artifact_checksum IS NOT NULL AND
            runner_mapping_checksum IS NOT NULL AND error IS NULL) OR
           (status = 'failed' AND artifact_checksum IS NULL AND
            runner_mapping_checksum IS NULL AND error IS NOT NULL))
);

CREATE TABLE field_evidence (
    evidence_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    field_name TEXT NOT NULL,
    authority TEXT NOT NULL,
    value_json TEXT NOT NULL,
    artifact_checksum TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    critical INTEGER NOT NULL CHECK (critical IN (0, 1)),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE sealed_evidence (
    seal_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    raw_manifest_checksum TEXT NOT NULL,
    normalized_checksum TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    normalization_version TEXT NOT NULL,
    frozen_at TEXT NOT NULL,
    freeze_authority TEXT NOT NULL CHECK (freeze_authority IN (
        'actual_jump', 'scheduled_minus_buffer'
    )),
    odds_checksum TEXT NOT NULL,
    sealed_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    UNIQUE (race_id, normalized_checksum)
);

CREATE TABLE collection_quarantines (
    quarantine_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    stage TEXT NOT NULL CHECK (stage IN ('identity', 'collection', 'sealing')),
    code TEXT NOT NULL,
    details TEXT NOT NULL,
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE INDEX expected_races_by_jump ON expected_races(scheduled_jump);
CREATE INDEX odds_attempts_by_race_time ON odds_attempts(race_id, attempted_at);
CREATE INDEX field_evidence_by_race_field ON field_evidence(race_id, field_name, authority);
CREATE INDEX identity_decisions_by_alias ON dog_identity_decisions(source, source_alias);
