CREATE TABLE race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_name TEXT,
                grade TEXT,
                distance TEXT,
                track_condition TEXT,
                weather TEXT,
                temperature REAL,
                humidity REAL,
                wind_speed REAL,
                wind_direction TEXT,
                track_record TEXT,
                prize_money_total REAL,
                prize_money_breakdown TEXT,
                race_time TEXT,
                field_size INTEGER,
                url TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                winner_name TEXT,
                winner_odds REAL,
                winner_margin REAL,
                race_status TEXT,
                data_quality_note TEXT, actual_field_size INTEGER, scratched_count INTEGER, scratch_rate REAL, box_analysis TEXT, weather_condition TEXT, precipitation REAL, pressure REAL, visibility REAL, weather_location TEXT, weather_timestamp DATETIME, weather_adjustment_factor REAL, sportsbet_url TEXT, venue_slug TEXT, start_datetime DATETIME,
                UNIQUE(race_id)
            );
CREATE TABLE sqlite_sequence(name,seq);
CREATE TABLE IF NOT EXISTS "dog_race_data_backup" (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                dog_id INTEGER,
                box_number INTEGER,
                finish_position TEXT,
                trainer_name TEXT,
                trainer_id INTEGER,
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
                was_scratched BOOLEAN DEFAULT FALSE,
                blackbook_link TEXT,
                extraction_timestamp DATETIME,
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
                best_time REAL, data_quality_note TEXT,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id),
                UNIQUE(race_id, dog_clean_name, box_number)
            );
CREATE TABLE race_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                analysis_type TEXT,
                analysis_data TEXT,
                confidence_score REAL,
                predicted_winner TEXT,
                predicted_odds REAL,
                analysis_timestamp DATETIME,
                model_version TEXT,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            );
CREATE TABLE track_conditions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue TEXT,
                date DATE,
                condition TEXT,
                rail_position TEXT,
                track_rating REAL,
                weather_conditions TEXT,
                temperature REAL,
                humidity REAL,
                wind_conditions TEXT,
                track_bias TEXT,
                extraction_timestamp DATETIME
            );
CREATE TABLE track_condition_backup_20250724_185411(
  race_id TEXT,
  track_condition TEXT,
  weather TEXT,
  url TEXT,
  extraction_timestamp NUM
);
CREATE TABLE live_odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                venue TEXT,
                race_number INTEGER,
                race_date DATE,
                race_time TEXT,
                dog_name TEXT,
                dog_clean_name TEXT,
                box_number INTEGER,
                odds_decimal REAL,
                odds_fractional TEXT,
                market_type TEXT DEFAULT 'win',
                source TEXT DEFAULT 'sportsbet',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_current BOOLEAN DEFAULT TRUE
            );
CREATE TABLE odds_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                odds_decimal REAL,
                odds_change REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'sportsbet'
            );
CREATE TABLE value_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                predicted_probability REAL,
                market_odds REAL,
                implied_probability REAL,
                value_percentage REAL,
                confidence_level TEXT,
                bet_recommendation TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT,
                dog_clean_name TEXT,
                predicted_probability REAL,
                confidence_level TEXT,
                prediction_source TEXT DEFAULT 'ml_model',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
CREATE TABLE weather_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    race_time DATETIME,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction TEXT,
                    pressure REAL,
                    condition TEXT,
                    precipitation REAL,
                    visibility REAL,
                    uv_index REAL,
                    data_source TEXT DEFAULT 'MOCK_API',
                    collection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 1.0,
                    UNIQUE(venue_code, race_date, race_time)
                );
CREATE TABLE weather_impact_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    weather_condition TEXT NOT NULL,
                    temperature_range TEXT NOT NULL,
                    humidity_range TEXT NOT NULL,
                    wind_range TEXT NOT NULL,
                    avg_winning_time REAL,
                    time_variance REAL,
                    favorite_strike_rate REAL,
                    avg_winning_margin REAL,
                    track_bias_impact TEXT,
                    sample_size INTEGER,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(venue_code, weather_condition, temperature_range, humidity_range, wind_range)
                );
CREATE TABLE weather_forecast_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    forecast_date DATE NOT NULL,
                    forecast_data TEXT NOT NULL,
                    cache_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    UNIQUE(venue_code, forecast_date)
                );
CREATE TABLE weather_data_v2 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_code TEXT NOT NULL,
                    race_date DATE NOT NULL,
                    race_time DATETIME,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    wind_direction INTEGER,
                    pressure REAL,
                    condition TEXT,
                    precipitation REAL,
                    visibility REAL,
                    weather_code INTEGER,
                    data_source TEXT DEFAULT 'OPEN_METEO',
                    collection_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.95,
                    UNIQUE(venue_code, race_date, race_time)
                );
CREATE TABLE venue_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    venue_key TEXT NOT NULL,
                    official_name TEXT NOT NULL,
                    venue_codes TEXT NOT NULL,
                    track_codes TEXT NOT NULL,
                    location TEXT NOT NULL,
                    active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
CREATE TABLE enhanced_expert_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT,
                    dog_name TEXT,
                    dog_clean_name TEXT,
                    venue TEXT,
                    race_date TEXT,
                    race_number INTEGER,
                    position INTEGER,
                    box_number INTEGER,
                    weight REAL,
                    distance INTEGER,
                    grade TEXT,
                    race_time REAL,
                    win_time REAL,
                    bonus_time REAL,
                    first_sectional REAL,
                    margin REAL,
                    pir_rating INTEGER,
                    starting_price REAL,
                    extraction_timestamp TEXT,
                    data_source TEXT DEFAULT 'enhanced_expert_form',
                    UNIQUE(race_id, dog_clean_name)
                );
CREATE TABLE enhanced_expert_data_backup_dog_day_fix(
  id INT,
  race_id TEXT,
  dog_name TEXT,
  dog_clean_name TEXT,
  venue TEXT,
  race_date TEXT,
  race_number INT,
  position INT,
  box_number INT,
  weight REAL,
  distance INT,
  grade TEXT,
  race_time REAL,
  win_time REAL,
  bonus_time REAL,
  first_sectional REAL,
  margin REAL,
  pir_rating INT,
  starting_price REAL,
  extraction_timestamp TEXT,
  data_source TEXT
);
CREATE TABLE dog_race_data_backup_box_number_fix(
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
  finish_position INT
);
CREATE TABLE race_metadata_backup_dedup_race_metadata(
  id INT,
  race_id TEXT,
  venue TEXT,
  race_number INT,
  race_date NUM,
  race_name TEXT,
  grade TEXT,
  distance TEXT,
  track_condition TEXT,
  weather TEXT,
  temperature REAL,
  humidity REAL,
  wind_speed REAL,
  wind_direction TEXT,
  track_record TEXT,
  prize_money_total REAL,
  prize_money_breakdown TEXT,
  race_time TEXT,
  field_size INT,
  url TEXT,
  extraction_timestamp NUM,
  data_source TEXT,
  winner_name TEXT,
  winner_odds REAL,
  winner_margin REAL,
  race_status TEXT,
  data_quality_note TEXT,
  actual_field_size INT,
  scratched_count INT,
  scratch_rate REAL,
  box_analysis TEXT,
  weather_condition TEXT,
  precipitation REAL,
  pressure REAL,
  visibility REAL,
  weather_location TEXT,
  weather_timestamp NUM,
  weather_adjustment_factor REAL,
  sportsbet_url TEXT
);
CREATE TABLE dog_race_data_backup_dedup_dog_race_data(
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
  finish_position INT
);
CREATE TABLE sqlite_stat1(tbl,idx,stat);
CREATE TABLE sqlite_stat4(tbl,idx,neq,nlt,ndlt,sample);
CREATE INDEX idx_enhanced_dog_name ON enhanced_expert_data(dog_clean_name);
CREATE INDEX idx_enhanced_race_date ON enhanced_expert_data(race_date);
CREATE INDEX idx_enhanced_venue ON enhanced_expert_data(venue);
CREATE UNIQUE INDEX idx_race_metadata_unique 
                ON race_metadata(race_id)
            ;
CREATE UNIQUE INDEX idx_enhanced_expert_unique 
                ON enhanced_expert_data(race_id, dog_clean_name)
            ;
CREATE INDEX idx_dog_date_check 
                ON enhanced_expert_data(dog_clean_name, race_date)
            ;
CREATE INDEX idx_race_date 
                ON race_metadata(race_date)
            ;
CREATE VIEW venue_resolver AS
            SELECT 
                venue_key,
                official_name,
                venue_codes,
                track_codes,
                location
            FROM venue_mappings
            WHERE active = 1
/* venue_resolver(venue_key,official_name,venue_codes,track_codes,location) */;
CREATE INDEX idx_race_metadata_date ON race_metadata(race_date);
CREATE TABLE dogs (
                dog_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT UNIQUE NOT NULL,
                total_races INTEGER DEFAULT 0,
                total_wins INTEGER DEFAULT 0,
                total_places INTEGER DEFAULT 0,
                best_time REAL,
                average_position REAL,
                last_race_date TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            , weight DECIMAL(5,2), age INTEGER, id INTEGER, color TEXT, owner TEXT, trainer TEXT, sex TEXT);
CREATE TABLE dog_performances (
                performance_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT NOT NULL,
                race_id TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                race_time REAL,
                weight REAL,
                trainer TEXT,
                odds REAL,
                margin TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            , dog_id INTEGER);
CREATE TABLE dogs_ft_extra (
                id INTEGER NOT NULL PRIMARY KEY,
                dog_id INTEGER NOT NULL UNIQUE,
                sire_name TEXT,
                sire_id TEXT,
                dam_name TEXT,
                dam_id TEXT,
                whelping_date DATE,
                age_days INTEGER,
                color TEXT,
                sex TEXT,
                ear_brand TEXT,
                career_starts INTEGER,
                career_wins INTEGER,
                career_places INTEGER,
                career_win_percent REAL,
                winning_boxes_json TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'fasttrack',
                FOREIGN KEY(dog_id) REFERENCES dogs(id) ON DELETE CASCADE
            );
CREATE TABLE races_ft_extra (
                id INTEGER NOT NULL PRIMARY KEY,
                race_id INTEGER NOT NULL UNIQUE,
                track_rating TEXT,
                weather_description TEXT,
                race_comment TEXT,
                split_1_time_winner REAL,
                split_2_time_winner REAL,
                run_home_time_winner REAL,
                video_url TEXT,
                stewards_report_url TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'fasttrack',
                FOREIGN KEY(race_id) REFERENCES races(id) ON DELETE CASCADE
            );
CREATE TABLE dog_performance_ft_extra (
                id INTEGER NOT NULL PRIMARY KEY,
                performance_id INTEGER NOT NULL UNIQUE,
                pir_rating TEXT,
                split_1_time REAL,
                split_2_time REAL,
                run_home_time REAL,
                beaten_margin REAL,
                in_race_comment TEXT,
                fixed_odds_sp REAL,
                prize_money REAL,
                pre_race_weight REAL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                data_source TEXT DEFAULT 'fasttrack',
                FOREIGN KEY(performance_id) REFERENCES dog_performances(id) ON DELETE CASCADE
            );
CREATE TABLE expert_form_analysis (
                id INTEGER NOT NULL PRIMARY KEY,
                race_id INTEGER NOT NULL,
                pdf_url TEXT,
                analysis_text TEXT,
                expert_selections TEXT,
                confidence_ratings TEXT,
                key_insights TEXT,
                analysis_date DATETIME,
                expert_name TEXT,
                extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                extraction_confidence REAL,
                data_source TEXT DEFAULT 'fasttrack_expert',
                processing_status TEXT DEFAULT 'pending',
                processing_notes TEXT,
                FOREIGN KEY(race_id) REFERENCES races(id) ON DELETE CASCADE
            );
CREATE TABLE races_gr_extra (
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
CREATE TABLE gr_race_details (
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
CREATE TABLE gr_dog_entries (
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
CREATE TABLE gr_dog_form (
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
CREATE INDEX idx_races_gr_extra_race_id ON races_gr_extra (race_id);
CREATE INDEX idx_gr_race_details_race_id ON gr_race_details (race_id);
CREATE INDEX idx_gr_dog_entries_race_id ON gr_dog_entries (race_id);
CREATE INDEX idx_gr_dog_entries_dog_name ON gr_dog_entries (dog_name);
CREATE INDEX idx_gr_dog_form_dog_entry_id ON gr_dog_form (dog_entry_id);
CREATE INDEX idx_gr_dog_form_race_date ON gr_dog_form (race_date);
CREATE TRIGGER trigger_races_gr_extra_updated_at
    AFTER UPDATE ON races_gr_extra
    FOR EACH ROW
BEGIN
    UPDATE races_gr_extra SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
CREATE TRIGGER trigger_gr_race_details_updated_at
    AFTER UPDATE ON gr_race_details
    FOR EACH ROW
BEGIN
    UPDATE gr_race_details SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
CREATE TRIGGER trigger_gr_dog_entries_updated_at
    AFTER UPDATE ON gr_dog_entries
    FOR EACH ROW
BEGIN
    UPDATE gr_dog_entries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
CREATE TABLE alembic_version (
	version_num VARCHAR(32) NOT NULL, 
	CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);
CREATE INDEX idx_race_metadata_venue_date ON race_metadata (venue, race_date);
CREATE INDEX idx_race_metadata_race_id ON race_metadata (race_id);
CREATE TABLE trainers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trainer_name TEXT UNIQUE NOT NULL,
    trainer_id INTEGER UNIQUE,
    total_races INTEGER DEFAULT 0,
    total_wins INTEGER DEFAULT 0,
    total_places INTEGER DEFAULT 0,
    win_percentage REAL DEFAULT 0.0,
    place_percentage REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE gpt_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                confidence_score REAL,
                tokens_used INTEGER,
                timestamp TEXT NOT NULL,
                model_used TEXT,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            );
CREATE INDEX idx_gpt_analysis_race_id ON gpt_analysis(race_id)
        ;
CREATE INDEX idx_gpt_analysis_timestamp ON gpt_analysis(timestamp)
        ;
CREATE TABLE comprehensive_dog_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT UNIQUE,
                dog_clean_name TEXT,
                career_races INTEGER DEFAULT 0,
                career_wins INTEGER DEFAULT 0,
                career_places INTEGER DEFAULT 0,
                career_earnings REAL DEFAULT 0,
                best_time REAL,
                average_time REAL,
                win_percentage REAL,
                place_percentage REAL,
                track_preferences TEXT,  -- JSON
                distance_preferences TEXT,  -- JSON
                grade_performance TEXT,  -- JSON
                trainer_history TEXT,  -- JSON
                recent_form_extended TEXT,  -- JSON - 20+ races
                sectional_data TEXT,  -- JSON
                injury_history TEXT,  -- JSON
                breeding_info TEXT,  -- JSON
                last_updated DATETIME,
                data_completeness_score REAL,
                profile_source TEXT,
                UNIQUE(dog_clean_name)
            );
CREATE TABLE detailed_race_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_name TEXT,
                race_date DATE,
                venue TEXT,
                race_number INTEGER,
                distance INTEGER,
                grade TEXT,
                track_condition TEXT,
                weather TEXT,
                box_number INTEGER,
                finish_position INTEGER,
                race_time REAL,
                sectional_times TEXT,  -- JSON
                margin REAL,
                starting_odds REAL,
                prize_money REAL,
                field_size INTEGER,
                trainer_name TEXT,
                weight REAL,
                race_class TEXT,
                track_record BOOLEAN,
                data_source TEXT,
                extraction_timestamp DATETIME,
                FOREIGN KEY (dog_name) REFERENCES comprehensive_dog_profiles (dog_clean_name)
            );
CREATE TABLE trainer_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trainer_name TEXT,
                total_races INTEGER,
                total_wins INTEGER,
                win_percentage REAL,
                speciality_tracks TEXT,  -- JSON
                speciality_distances TEXT,  -- JSON
                recent_form TEXT,  -- JSON
                last_updated DATETIME,
                UNIQUE(trainer_name)
            );
CREATE INDEX idx_race_metadata_venue ON race_metadata (venue);
CREATE INDEX idx_race_metadata_extraction ON race_metadata (extraction_timestamp);
CREATE INDEX idx_enhanced_expert_data_race_date ON enhanced_expert_data(race_date);
CREATE INDEX idx_race_metadata_race_date ON race_metadata(race_date);
CREATE TABLE IF NOT EXISTS "dog_race_data" (
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
CREATE INDEX idx_dog_race_data_dog_name ON dog_race_data(dog_name);
CREATE INDEX idx_dog_race_data_race_id ON dog_race_data(race_id);
CREATE INDEX idx_dog_name ON dog_race_data(dog_clean_name);
CREATE UNIQUE INDEX idx_dog_race_unique ON dog_race_data(race_id, dog_clean_name, box_number);
CREATE INDEX idx_dog_race_data_race ON dog_race_data (race_id);
CREATE INDEX idx_dog_race_data_finish_position ON dog_race_data(finish_position);
