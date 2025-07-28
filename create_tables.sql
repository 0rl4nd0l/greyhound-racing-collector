-- SQL schema for greyhound racing database

-- Race metadata table
CREATE TABLE IF NOT EXISTS race_metadata (
    race_id TEXT PRIMARY KEY,
    venue TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    race_date TEXT NOT NULL,
    race_name TEXT,
    grade TEXT,
    distance TEXT,
    field_size INTEGER,
    winner_name TEXT,
    winner_odds TEXT,
    winner_margin TEXT,
    url TEXT,
    extraction_timestamp TEXT NOT NULL,
    track_condition TEXT
);

-- Create index on common query fields
CREATE INDEX IF NOT EXISTS idx_race_metadata_date ON race_metadata(race_date);
CREATE INDEX IF NOT EXISTS idx_race_metadata_venue ON race_metadata(venue);
CREATE INDEX IF NOT EXISTS idx_race_metadata_extraction ON race_metadata(extraction_timestamp);

-- Dog race data table
CREATE TABLE IF NOT EXISTS dog_race_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id TEXT NOT NULL,
    box_number INTEGER,
    dog_name TEXT,
    trainer TEXT,
    form TEXT,
    weight TEXT,
    winning_time TEXT,
    placing INTEGER,
    margin TEXT,
    odds TEXT,
    FOREIGN KEY (race_id) REFERENCES race_metadata(race_id)
);

-- Create index for faster race lookups
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race ON dog_race_data(race_id);
