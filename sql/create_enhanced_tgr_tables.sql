-- Enhanced TGR (The Greyhound Recorder) Database Tables
-- =====================================================
-- Comprehensive schema for storing enhanced TGR data

-- Enhanced dog form data with comprehensive tracking
CREATE TABLE IF NOT EXISTS gr_dog_form (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT NOT NULL,
    race_date TEXT,
    venue TEXT,
    grade TEXT,
    distance TEXT,
    box_number INTEGER,
    recent_form TEXT, -- JSON array of recent form positions
    weight REAL,
    comments TEXT,
    odds REAL,
    odds_text TEXT,
    trainer TEXT,
    profile_url TEXT,
    race_url TEXT,
    field_size INTEGER,
    race_number INTEGER,
    expert_comments TEXT, -- JSON array of expert comments
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dog_name, race_date, venue, race_number)
);

-- TGR performance summary for quick lookups
CREATE TABLE IF NOT EXISTS tgr_dog_performance_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT NOT NULL UNIQUE,
    performance_data TEXT, -- JSON with win rates, averages, etc.
    venue_analysis TEXT, -- JSON with venue-specific performance
    distance_analysis TEXT, -- JSON with distance-specific performance
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_entries INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    places INTEGER DEFAULT 0,
    win_percentage REAL DEFAULT 0.0,
    place_percentage REAL DEFAULT 0.0,
    average_position REAL DEFAULT 0.0,
    best_position INTEGER DEFAULT 8,
    consistency_score REAL DEFAULT 0.0,
    form_trend TEXT DEFAULT 'stable', -- improving, declining, stable
    distance_versatility INTEGER DEFAULT 0,
    venues_raced INTEGER DEFAULT 0
);

-- Expert insights and comments from TGR
CREATE TABLE IF NOT EXISTS tgr_expert_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT NOT NULL,
    comment_type TEXT, -- 'dog_comment', 'expert_insight', 'race_preview'
    race_date TEXT,
    venue TEXT,
    comment_text TEXT NOT NULL,
    source TEXT, -- 'form_guide', 'expert_analysis', 'race_preview'
    sentiment_score REAL, -- calculated sentiment (-1 to 1)
    keywords TEXT, -- JSON array of extracted keywords
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dog_name, comment_text, race_date)
);

-- Venue-specific performance analysis
CREATE TABLE IF NOT EXISTS tgr_venue_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT NOT NULL,
    venue TEXT NOT NULL,
    starts INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    places INTEGER DEFAULT 0,
    positions TEXT, -- JSON array of positions
    win_rate REAL DEFAULT 0.0,
    place_rate REAL DEFAULT 0.0,
    average_position REAL DEFAULT 0.0,
    best_position INTEGER DEFAULT 8,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dog_name, venue)
);

-- Distance-specific performance analysis
CREATE TABLE IF NOT EXISTS tgr_distance_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT NOT NULL,
    distance TEXT NOT NULL,
    starts INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    places INTEGER DEFAULT 0,
    positions TEXT, -- JSON array of positions
    win_rate REAL DEFAULT 0.0,
    place_rate REAL DEFAULT 0.0,
    average_position REAL DEFAULT 0.0,
    best_position INTEGER DEFAULT 8,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dog_name, distance)
);

-- TGR race details (full race information)
CREATE TABLE IF NOT EXISTS tgr_race_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_url TEXT NOT NULL UNIQUE,
    venue TEXT,
    race_date TEXT,
    race_number INTEGER,
    race_title TEXT,
    grade TEXT,
    distance TEXT,
    track_condition TEXT,
    field_size INTEGER DEFAULT 0,
    expert_comments TEXT, -- JSON array of expert comments
    race_preview TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TGR scraping log and status
CREATE TABLE IF NOT EXISTS tgr_scraping_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT,
    scrape_type TEXT, -- 'form_data', 'performance_summary', 'enhanced_data'
    status TEXT, -- 'success', 'failed', 'partial'
    entries_found INTEGER DEFAULT 0,
    comments_found INTEGER DEFAULT 0,
    error_message TEXT,
    scrape_duration REAL, -- seconds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced feature cache with metadata
CREATE TABLE IF NOT EXISTS tgr_enhanced_feature_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dog_name TEXT NOT NULL,
    race_timestamp TIMESTAMP,
    tgr_features TEXT NOT NULL, -- JSON with all 18+ TGR features
    performance_summary TEXT, -- JSON with detailed performance data
    venue_insights TEXT, -- JSON with venue-specific insights
    distance_insights TEXT, -- JSON with distance-specific insights
    expert_sentiment REAL DEFAULT 0.0,
    form_trend TEXT DEFAULT 'stable',
    cache_version INTEGER DEFAULT 1,
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    UNIQUE(dog_name, race_timestamp)
);

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_gr_dog_form_dog_name ON gr_dog_form(dog_name);
CREATE INDEX IF NOT EXISTS idx_gr_dog_form_race_date ON gr_dog_form(race_date);
CREATE INDEX IF NOT EXISTS idx_gr_dog_form_venue ON gr_dog_form(venue);

CREATE INDEX IF NOT EXISTS idx_tgr_performance_summary_dog_name ON tgr_dog_performance_summary(dog_name);
CREATE INDEX IF NOT EXISTS idx_tgr_performance_summary_updated ON tgr_dog_performance_summary(last_updated);

CREATE INDEX IF NOT EXISTS idx_tgr_expert_insights_dog_name ON tgr_expert_insights(dog_name);
CREATE INDEX IF NOT EXISTS idx_tgr_expert_insights_date ON tgr_expert_insights(race_date);
CREATE INDEX IF NOT EXISTS idx_tgr_expert_insights_type ON tgr_expert_insights(comment_type);

CREATE INDEX IF NOT EXISTS idx_tgr_venue_performance_dog_name ON tgr_venue_performance(dog_name);
CREATE INDEX IF NOT EXISTS idx_tgr_venue_performance_venue ON tgr_venue_performance(venue);

CREATE INDEX IF NOT EXISTS idx_tgr_distance_performance_dog_name ON tgr_distance_performance(dog_name);
CREATE INDEX IF NOT EXISTS idx_tgr_distance_performance_distance ON tgr_distance_performance(distance);

CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_cache_dog_name ON tgr_enhanced_feature_cache(dog_name);
CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_cache_timestamp ON tgr_enhanced_feature_cache(race_timestamp);
CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_cache_expires ON tgr_enhanced_feature_cache(expires_at);

-- Create views for easy access to enhanced data
CREATE VIEW IF NOT EXISTS vw_tgr_dog_summary AS
SELECT 
    ps.dog_name,
    ps.wins,
    ps.places,
    ps.win_percentage,
    ps.place_percentage,
    ps.average_position,
    ps.best_position,
    ps.consistency_score,
    ps.form_trend,
    ps.distance_versatility,
    ps.venues_raced,
    COUNT(df.id) as total_form_entries,
    COUNT(ei.id) as total_comments,
    ps.last_updated
FROM tgr_dog_performance_summary ps
LEFT JOIN gr_dog_form df ON ps.dog_name = df.dog_name
LEFT JOIN tgr_expert_insights ei ON ps.dog_name = ei.dog_name
GROUP BY ps.dog_name;

-- Create view for recent TGR activity
CREATE VIEW IF NOT EXISTS vw_tgr_recent_activity AS
SELECT 
    'form_data' as activity_type,
    dog_name,
    race_date as activity_date,
    venue,
    'Form entry added' as description,
    created_at
FROM gr_dog_form
UNION ALL
SELECT 
    'expert_insight' as activity_type,
    dog_name,
    race_date as activity_date,
    venue,
    'Expert comment added: ' || substr(comment_text, 1, 50) || '...' as description,
    created_at
FROM tgr_expert_insights
ORDER BY created_at DESC
LIMIT 100;
