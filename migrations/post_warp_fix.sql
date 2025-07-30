-- Non-destructive database migration to fix schema mismatches

-- Create missing indexes for race_metadata
CREATE INDEX IF NOT EXISTS idx_race_metadata_venue ON race_metadata (venue);
CREATE INDEX IF NOT EXISTS idx_race_metadata_extraction ON race_metadata (extraction_timestamp);

-- Add missing columns to dog_race_data
ALTER TABLE dog_race_data ADD COLUMN odds TEXT;
ALTER TABLE dog_race_data ADD COLUMN winning_time TEXT;
ALTER TABLE dog_race_data ADD COLUMN form TEXT;
ALTER TABLE dog_race_data ADD COLUMN placing INTEGER;
ALTER TABLE dog_race_data ADD COLUMN trainer TEXT;

-- Create missing indexes for dog_race_data
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race ON dog_race_data (race_id);

-- Add missing columns to dogs
ALTER TABLE dogs ADD COLUMN color TEXT;
ALTER TABLE dogs ADD COLUMN weight DECIMAL(5,2);
ALTER TABLE dogs ADD COLUMN age INTEGER;
ALTER TABLE dogs ADD COLUMN id INTEGER;
ALTER TABLE dogs ADD COLUMN trainer TEXT;
ALTER TABLE dogs ADD COLUMN owner TEXT;
ALTER TABLE dogs ADD COLUMN sex TEXT;

-- NOTE: Modifying column types, constraints, or foreign keys in SQLite often requires recreating the table.
-- These changes should be reviewed and applied manually if needed.