CREATE TABLE dog_identity_aliases (
    provisional_dog_id TEXT PRIMARY KEY REFERENCES dogs(dog_id),
    canonical_dog_id TEXT NOT NULL REFERENCES dogs(dog_id),
    upgraded_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id),
    CHECK (provisional_dog_id <> canonical_dog_id)
);

CREATE TABLE identity_quarantines (
    identity_quarantine_id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    source_alias TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    operation_id TEXT NOT NULL UNIQUE REFERENCES operations(operation_id)
);

CREATE INDEX dog_identity_aliases_by_canonical
    ON dog_identity_aliases(canonical_dog_id);
CREATE INDEX identity_quarantines_by_alias
    ON identity_quarantines(source, source_alias);
