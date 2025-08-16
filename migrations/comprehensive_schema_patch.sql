-- Comprehensive Schema Patch for Greyhound Racing Database
-- Date: 2025-08-01
-- Purpose: Non-destructive migration to align schema with ORM expectations

-- Add missing columns to dog_race_data
BEGIN;
ALTER TABLE dog_race_data ADD COLUMN odds TEXT;
ALTER TABLE dog_race_data ADD COLUMN trainer TEXT;
ALTER TABLE dog_race_data ADD COLUMN winning_time TEXT;
ALTER TABLE dog_race_data ADD COLUMN placing INTEGER;
ALTER TABLE dog_race_data ADD COLUMN form TEXT;
COMMIT;

-- Create missing indexes for better performance
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race ON dog_race_data (race_id);
CREATE INDEX IF NOT EXISTS idx_race_metadata_venue ON race_metadata (venue);
CREATE INDEX IF NOT EXISTS idx_race_metadata_extraction ON race_metadata (extraction_timestamp);

-- Add missing columns to dogs table
BEGIN;
ALTER TABLE dogs ADD COLUMN weight DECIMAL(5,2);
ALTER TABLE dogs ADD COLUMN age INTEGER;
ALTER TABLE dogs ADD COLUMN id INTEGER;
ALTER TABLE dogs ADD COLUMN color TEXT;
ALTER TABLE dogs ADD COLUMN owner TEXT;
ALTER TABLE dogs ADD COLUMN trainer TEXT;
ALTER TABLE dogs ADD COLUMN sex TEXT;
COMMIT;

-- Note: start_datetime already exists, skipping

-- Create views for better data access
CREATE VIEW IF NOT EXISTS venue_resolver AS
SELECT 
    venue_key,
    official_name,
    venue_codes,
    track_codes,
    location
FROM venue_mappings
WHERE active = 1;

-- Additional integrity checks and performance improvements
-- Note: Direct foreign key constraints on SQLite require recreating tables
-- These are handled by Alembic migrations for safer data preservation

-- Create additional indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_enhanced_expert_data_race_date ON enhanced_expert_data(race_date);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_finish_position ON dog_race_data(finish_position);
CREATE INDEX IF NOT EXISTS idx_race_metadata_race_date ON race_metadata(race_date);

-- Update statistics for query optimizer
ANALYZE;

-- Verify schema integrity
PRAGMA integrity_check;
