-- Initialize the greyhound_test database with production schema
-- This file will be automatically executed by PostgreSQL during container startup

-- Enable UUID extension for unique identifiers
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create race_metadata table with same structure as production
CREATE TABLE IF NOT EXISTS race_metadata (
    id SERIAL PRIMARY KEY,
    race_id VARCHAR(255) UNIQUE NOT NULL,
    venue VARCHAR(100) NOT NULL,
    race_date DATE NOT NULL,
    race_name VARCHAR(255),
    grade VARCHAR(50),
    distance VARCHAR(50),
    track_condition VARCHAR(100),
    weather VARCHAR(100),
    temperature REAL,
    humidity REAL,
    wind_speed REAL,
    wind_direction VARCHAR(50),
    track_record VARCHAR(50),
    prize_money_total REAL,
    prize_money_breakdown TEXT,
    race_time TIME,
    field_size INTEGER,
    track_variant VARCHAR(50),
    number_of_runners INTEGER,
    race_type VARCHAR(50),
    start_datetime TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create dog_race_data table
CREATE TABLE IF NOT EXISTS dog_race_data (
    id SERIAL PRIMARY KEY,
    race_id VARCHAR(255) NOT NULL,
    dog_name VARCHAR(255) NOT NULL,
    dog_clean_name VARCHAR(255),
    trap_number INTEGER,
    finish_position INTEGER,
    starting_price VARCHAR(50),
    individual_time REAL,
    sectional_1 REAL,
    sectional_2 REAL,
    sectional_3 REAL,
    sectional_4 REAL,
    weight REAL,
    trainer VARCHAR(255),
    age VARCHAR(10),
    form_comment TEXT,
    margin VARCHAR(50),
    prize_money REAL,
    box_number INTEGER,
    odds_win REAL,
    odds_place REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES race_metadata(race_id) ON DELETE CASCADE
);

-- Create dogs table
CREATE TABLE IF NOT EXISTS dogs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    clean_name VARCHAR(255),
    trainer VARCHAR(255),
    sire VARCHAR(255),
    dam VARCHAR(255),
    color VARCHAR(50),
    sex VARCHAR(10),
    date_of_birth DATE,
    weight REAL,
    career_starts INTEGER DEFAULT 0,
    career_wins INTEGER DEFAULT 0,
    career_places INTEGER DEFAULT 0,
    career_shows INTEGER DEFAULT 0,
    career_earnings REAL DEFAULT 0.0,
    best_time REAL,
    last_start_date DATE,
    retirement_date DATE,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_race_metadata_race_id ON race_metadata(race_id);
CREATE INDEX IF NOT EXISTS idx_race_metadata_date ON race_metadata(race_date);
CREATE INDEX IF NOT EXISTS idx_race_metadata_venue ON race_metadata(venue);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_race_id ON dog_race_data(race_id);
CREATE INDEX IF NOT EXISTS idx_dog_race_data_dog_name ON dog_race_data(dog_clean_name);
CREATE INDEX IF NOT EXISTS idx_dogs_clean_name ON dogs(clean_name);
CREATE INDEX IF NOT EXISTS idx_dogs_trainer ON dogs(trainer);

-- Create additional tables for comprehensive testing
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    race_id VARCHAR(255) NOT NULL,
    dog_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    win_probability REAL NOT NULL CHECK (win_probability >= 0 AND win_probability <= 1),
    place_probability REAL NOT NULL CHECK (place_probability >= 0 AND place_probability <= 1),
    predicted_rank INTEGER,
    confidence_score REAL,
    expected_value REAL,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (race_id) REFERENCES race_metadata(race_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    evaluation_date DATE NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    dataset_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to update updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_race_metadata_updated_at BEFORE UPDATE ON race_metadata FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dog_race_data_updated_at BEFORE UPDATE ON dog_race_data FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_dogs_updated_at BEFORE UPDATE ON dogs FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO test_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO test_user;
