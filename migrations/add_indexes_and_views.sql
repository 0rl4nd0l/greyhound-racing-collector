-- Safe Migration: Add indexes and views only
-- Date: 2025-08-01
-- Purpose: Add missing indexes and views without modifying existing schema

-- Create missing indexes for better performance
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race ON dog_race_data (race_id);
CREATE INDEX IF NOT EXISTS idx_race_metadata_venue ON race_metadata (venue);
CREATE INDEX IF NOT EXISTS idx_race_metadata_extraction ON race_metadata (extraction_timestamp);

-- Create additional indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_enhanced_expert_data_race_date ON enhanced_expert_data(race_date);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_finish_position ON dog_race_data(finish_position);
CREATE INDEX IF NOT EXISTS idx_race_metadata_race_date ON race_metadata(race_date);

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

-- Update statistics for query optimizer
ANALYZE;

-- Verify schema integrity
PRAGMA integrity_check;
