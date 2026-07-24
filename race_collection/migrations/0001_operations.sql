CREATE TABLE racing_days (
    racing_day_id TEXT PRIMARY KEY CHECK (racing_day_id GLOB 'day_[0-9a-f]*'),
    local_date TEXT NOT NULL,
    timezone TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    UNIQUE (local_date, timezone)
);

CREATE TABLE races (
    race_id TEXT PRIMARY KEY CHECK (race_id GLOB 'race_[0-9a-f]*'),
    racing_day_id TEXT NOT NULL REFERENCES racing_days(racing_day_id),
    state TEXT NOT NULL CHECK (state IN (
        'discovered', 'card_collected', 'collecting_odds', 'evidence_sealed',
        'awaiting_day_close', 'prediction_pending', 'prediction_committed',
        'prediction_quarantined', 'result_pending', 'result_collected',
        'result_quarantined', 'training_example_ready', 'evaluation_ineligible'
    )),
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE race_aliases (
    race_id TEXT NOT NULL REFERENCES races(race_id),
    source TEXT NOT NULL,
    alias TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source, alias),
    UNIQUE (race_id, source, alias)
);

CREATE TABLE dogs (
    dog_id TEXT PRIMARY KEY CHECK (dog_id GLOB 'dog_[0-9a-f]*'),
    created_at TEXT NOT NULL
);

CREATE TABLE dog_aliases (
    dog_id TEXT NOT NULL REFERENCES dogs(dog_id),
    source TEXT NOT NULL,
    alias TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source, alias)
);

CREATE TABLE dog_runs (
    dog_run_id INTEGER PRIMARY KEY,
    dog_id TEXT NOT NULL REFERENCES dogs(dog_id),
    local_racing_date TEXT NOT NULL,
    authoritative INTEGER NOT NULL CHECK (authoritative IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE (dog_id, local_racing_date)
);

CREATE TABLE run_observations (
    observation_id INTEGER PRIMARY KEY,
    dog_run_id INTEGER NOT NULL REFERENCES dog_runs(dog_run_id),
    source TEXT NOT NULL,
    artifact_checksum TEXT NOT NULL CHECK (artifact_checksum GLOB 'sha256:[0-9a-f]*'),
    observed_at TEXT NOT NULL,
    starts INTEGER CHECK (starts IS NULL OR starts >= 0),
    wins INTEGER CHECK (wins IS NULL OR wins >= 0),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    CHECK (starts IS NULL OR wins IS NULL OR wins <= starts)
);

CREATE TABLE quarantines (
    quarantine_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE TABLE supersessions (
    supersession_id INTEGER PRIMARY KEY,
    entity_type TEXT NOT NULL,
    prior_id TEXT NOT NULL,
    replacement_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    UNIQUE (entity_type, prior_id, replacement_id),
    CHECK (prior_id <> replacement_id)
);

CREATE TABLE lifecycle_events (
    event_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    prior_state TEXT,
    target_state TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE INDEX races_by_day_state ON races(racing_day_id, state);
CREATE INDEX observations_by_run ON run_observations(dog_run_id, observed_at);
