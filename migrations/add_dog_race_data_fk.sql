-- Add Foreign Key Constraint to dog_race_data table
-- This requires recreating the table in SQLite

PRAGMA foreign_keys=OFF;

BEGIN TRANSACTION;

-- Create new table with foreign key constraint
CREATE TABLE dog_race_data_new (
    id INT,
    race_id TEXT,
    dog_name TEXT,
    dog_clean_name TEXT,
    dog_id INT,
    box_number INT,
    trainer_name TEXT,
    trainer_id INT,
    weight REAL,
    running_style TEXT,
    odds_decimal REAL,
    odds_fractional TEXT,
    starting_price REAL,
    individual_time TEXT,
    sectional_1st TEXT,
    sectional_2nd TEXT,
    sectional_3rd TEXT,
    margin TEXT,
    beaten_margin REAL,
    was_scratched NUM,
    blackbook_link TEXT,
    extraction_timestamp NUM,
    data_source TEXT,
    form_guide_json TEXT,
    historical_records TEXT,
    performance_rating REAL,
    speed_rating REAL,
    class_rating REAL,
    recent_form TEXT,
    win_probability REAL,
    place_probability REAL,
    scraped_trainer_name TEXT,
    scraped_reaction_time TEXT,
    scraped_nbtt TEXT,
    scraped_race_classification TEXT,
    scraped_raw_result TEXT,
    scraped_finish_position TEXT,
    best_time REAL,
    data_quality_note TEXT,
    finish_position INTEGER,
    odds TEXT,
    trainer TEXT,
    winning_time TEXT,
    placing INTEGER,
    form TEXT,
    FOREIGN KEY (race_id) REFERENCES race_metadata (race_id) ON DELETE CASCADE
);

-- Copy data from old table to new table
INSERT INTO dog_race_data_new SELECT * FROM dog_race_data;

-- Drop old table
DROP TABLE dog_race_data;

-- Rename new table to original name
ALTER TABLE dog_race_data_new RENAME TO dog_race_data;

-- Recreate indexes
CREATE INDEX IF NOT EXISTS idx_dog_race_data_dog_name ON dog_race_data(dog_name);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race_id ON dog_race_data(race_id);
CREATE INDEX IF NOT EXISTS idx_dog_name ON dog_race_data(dog_clean_name);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dog_race_unique ON dog_race_data(race_id, dog_clean_name, box_number);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race ON dog_race_data (race_id);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_finish_position ON dog_race_data(finish_position);

COMMIT;

PRAGMA foreign_keys=ON;

-- Verify foreign key constraint was added
PRAGMA foreign_key_list(dog_race_data);
