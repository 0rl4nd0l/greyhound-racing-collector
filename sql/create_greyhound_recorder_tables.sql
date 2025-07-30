-- Create tables for The Greyhound Recorder data
-- This extends the existing database schema with additional data from The Greyhound Recorder

-- Table to store additional data from The Greyhound Recorder
CREATE TABLE IF NOT EXISTS races_gr_extra (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    meeting_id INTEGER,
    long_form_url TEXT,
    short_form_url TEXT,
    fields_url TEXT,
    data_source TEXT DEFAULT 'the_greyhound_recorder',
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES race_metadata (id)
);

-- Table to store detailed race data from long form pages
CREATE TABLE IF NOT EXISTS gr_race_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    race_number INTEGER,
    race_name TEXT,
    race_time TEXT,
    race_distance INTEGER,
    race_grade TEXT,
    prize_money DECIMAL(10,2),
    conditions TEXT,
    track_condition TEXT,
    weather TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES race_metadata (id)
);

-- Table to store dog entries from The Greyhound Recorder
CREATE TABLE IF NOT EXISTS gr_dog_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id INTEGER NOT NULL,
    dog_name TEXT NOT NULL,
    box_number INTEGER,
    trainer_name TEXT,
    owner_name TEXT,
    form_guide TEXT,
    recent_form TEXT,
    best_time DECIMAL(5,2),
    last_start TEXT,
    weight DECIMAL(4,1),
    age_months INTEGER,
    color TEXT,
    sex TEXT,
    sire TEXT,
    dam TEXT,
    comment TEXT,
    jockey_name TEXT,
    barrier_trial_info TEXT,
    scratched BOOLEAN DEFAULT 0,
    emergency BOOLEAN DEFAULT 0,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES race_metadata (id)
);

-- Table to store form guide data for individual dogs
CREATE TABLE IF NOT EXISTS gr_dog_form (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_entry_id INTEGER NOT NULL,
    race_date DATE,
    venue TEXT,
    distance INTEGER,
    box_number INTEGER,
    finishing_position INTEGER,
    race_time DECIMAL(5,2),
    split_times TEXT, -- JSON string for split times
    margin TEXT,
    weight DECIMAL(4,1),
    comments TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (dog_entry_id) REFERENCES gr_dog_entries (id)
);

-- Index for performance
CREATE INDEX IF NOT EXISTS idx_races_gr_extra_race_id ON races_gr_extra (race_id);
CREATE INDEX IF NOT EXISTS idx_gr_race_details_race_id ON gr_race_details (race_id);
CREATE INDEX IF NOT EXISTS idx_gr_dog_entries_race_id ON gr_dog_entries (race_id);
CREATE INDEX IF NOT EXISTS idx_gr_dog_entries_dog_name ON gr_dog_entries (dog_name);
CREATE INDEX IF NOT EXISTS idx_gr_dog_form_dog_entry_id ON gr_dog_form (dog_entry_id);
CREATE INDEX IF NOT EXISTS idx_gr_dog_form_race_date ON gr_dog_form (race_date);

-- Add updated_at trigger for races_gr_extra
CREATE TRIGGER IF NOT EXISTS trigger_races_gr_extra_updated_at
    AFTER UPDATE ON races_gr_extra
    FOR EACH ROW
BEGIN
    UPDATE races_gr_extra SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Add updated_at trigger for gr_race_details
CREATE TRIGGER IF NOT EXISTS trigger_gr_race_details_updated_at
    AFTER UPDATE ON gr_race_details
    FOR EACH ROW
BEGIN
    UPDATE gr_race_details SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Add updated_at trigger for gr_dog_entries
CREATE TRIGGER IF NOT EXISTS trigger_gr_dog_entries_updated_at
    AFTER UPDATE ON gr_dog_entries
    FOR EACH ROW
BEGIN
    UPDATE gr_dog_entries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
