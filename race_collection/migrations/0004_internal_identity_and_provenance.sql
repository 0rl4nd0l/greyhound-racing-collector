CREATE TABLE programme_race_observations (
    observation_id INTEGER PRIMARY KEY,
    race_id TEXT NOT NULL REFERENCES races(race_id),
    source TEXT NOT NULL,
    source_race_id TEXT NOT NULL,
    venue TEXT NOT NULL,
    race_number INTEGER NOT NULL CHECK (race_number > 0),
    scheduled_jump TEXT NOT NULL,
    programme_checksum TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    collision INTEGER NOT NULL CHECK (collision IN (0, 1)),
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE INDEX programme_observations_by_alias
    ON programme_race_observations(source, source_race_id);
