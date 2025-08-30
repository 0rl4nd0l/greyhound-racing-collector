#!/usr/bin/env python3
"""
TGR Enhancement Migration Script
===============================

Safely migrates the database to support enhanced TGR data capture
while preserving existing data.
"""

import sqlite3
import logging
import json
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_tgr_enhancement(db_path: str = "greyhound_racing_data.db"):
    """Migrate database to support enhanced TGR functionality."""
    
    logger.info("🚀 Starting TGR enhancement migration...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if migration is needed
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tgr_dog_performance_summary'")
        if cursor.fetchone():
            logger.info("✅ TGR enhancement tables already exist")
            conn.close()
            return
        
        logger.info("📦 Creating enhanced TGR tables...")
        
        # Create enhanced TGR tables (avoiding conflicts with existing ones)
        
        # 1. TGR performance summary for quick lookups
        cursor.execute("""
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
            )
        """)
        
        # 2. Expert insights and comments from TGR
        cursor.execute("""
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
            )
        """)
        
        # 3. Venue-specific performance analysis
        cursor.execute("""
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
            )
        """)
        
        # 4. Distance-specific performance analysis
        cursor.execute("""
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
            )
        """)
        
        # 5. TGR scraping log and status
        cursor.execute("""
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
            )
        """)
        
        # 6. Enhanced feature cache with metadata
        cursor.execute("""
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
            )
        """)
        
        # 7. Enhanced dog form table (supplement to existing gr_dog_form)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tgr_enhanced_dog_form (
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
                finishing_position INTEGER,
                race_time TEXT,
                split_times TEXT,
                margin TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(dog_name, race_date, venue, race_number)
            )
        """)
        
        # Create performance indices
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_tgr_performance_dog_name ON tgr_dog_performance_summary(dog_name)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_performance_updated ON tgr_dog_performance_summary(last_updated)",
            
            "CREATE INDEX IF NOT EXISTS idx_tgr_insights_dog_name ON tgr_expert_insights(dog_name)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_insights_date ON tgr_expert_insights(race_date)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_insights_type ON tgr_expert_insights(comment_type)",
            
            "CREATE INDEX IF NOT EXISTS idx_tgr_venue_dog_name ON tgr_venue_performance(dog_name)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_venue_venue ON tgr_venue_performance(venue)",
            
            "CREATE INDEX IF NOT EXISTS idx_tgr_distance_dog_name ON tgr_distance_performance(dog_name)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_distance_distance ON tgr_distance_performance(distance)",
            
            "CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_cache_dog_name ON tgr_enhanced_feature_cache(dog_name)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_cache_timestamp ON tgr_enhanced_feature_cache(race_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_cache_expires ON tgr_enhanced_feature_cache(expires_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_form_dog_name ON tgr_enhanced_dog_form(dog_name)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_form_date ON tgr_enhanced_dog_form(race_date)",
            "CREATE INDEX IF NOT EXISTS idx_tgr_enhanced_form_venue ON tgr_enhanced_dog_form(venue)"
        ]
        
        for index in indices:
            cursor.execute(index)
        
        # Create enhanced views
        cursor.execute("""
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
                COUNT(DISTINCT df.id) as form_entries_count,
                COUNT(DISTINCT ei.id) as insights_count,
                ps.last_updated
            FROM tgr_dog_performance_summary ps
            LEFT JOIN tgr_enhanced_dog_form df ON ps.dog_name = df.dog_name
            LEFT JOIN tgr_expert_insights ei ON ps.dog_name = ei.dog_name
            GROUP BY ps.dog_name
        """)
        
        cursor.execute("""
            CREATE VIEW IF NOT EXISTS vw_tgr_recent_activity AS
            SELECT 
                'enhanced_form' as activity_type,
                dog_name,
                race_date as activity_date,
                venue,
                'Enhanced form entry: ' || COALESCE(grade, 'Unknown grade') || ' at ' || COALESCE(venue, 'Unknown venue') as description,
                created_at
            FROM tgr_enhanced_dog_form
            WHERE created_at >= datetime('now', '-30 days')
            UNION ALL
            SELECT 
                'expert_insight' as activity_type,
                dog_name,
                race_date as activity_date,
                venue,
                'Expert insight: ' || substr(comment_text, 1, 50) || '...' as description,
                created_at
            FROM tgr_expert_insights
            WHERE created_at >= datetime('now', '-30 days')
            UNION ALL
            SELECT 
                'performance_update' as activity_type,
                dog_name,
                last_updated as activity_date,
                '' as venue,
                'Performance summary updated: ' || CAST(wins AS TEXT) || ' wins, ' || CAST(places AS TEXT) || ' places' as description,
                last_updated as created_at
            FROM tgr_dog_performance_summary
            WHERE last_updated >= datetime('now', '-30 days')
            ORDER BY created_at DESC
            LIMIT 100
        """)
        
        # Migrate existing TGR cache data to enhanced format
        logger.info("🔄 Migrating existing TGR cache data...")
        
        cursor.execute("SELECT COUNT(*) FROM tgr_feature_cache")
        existing_cache_count = cursor.fetchone()[0]
        
        if existing_cache_count > 0:
            logger.info(f"Found {existing_cache_count} existing cache entries to migrate")
            
            cursor.execute("""
                INSERT OR IGNORE INTO tgr_enhanced_feature_cache 
                (dog_name, race_timestamp, tgr_features, cached_at, expires_at)
                SELECT 
                    dog_name,
                    race_timestamp,
                    tgr_features,
                    cached_at,
                    datetime(cached_at, '+24 hours') as expires_at
                FROM tgr_feature_cache
                WHERE tgr_features IS NOT NULL
            """)
            
            migrated_count = cursor.rowcount
            logger.info(f"✅ Migrated {migrated_count} cache entries to enhanced format")
        
        # Create migration record
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS migration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                migration_name TEXT NOT NULL,
                status TEXT NOT NULL,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            INSERT INTO migration_log (migration_name, status, notes)
            VALUES (?, ?, ?)
        """, [
            'tgr_enhancement_v1',
            'success', 
            f'Enhanced TGR tables created with {existing_cache_count} cache entries migrated'
        ])
        
        conn.commit()
        conn.close()
        
        logger.info("✅ TGR enhancement migration completed successfully!")
        
        # Verify the migration
        verify_migration(db_path)
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        if conn:
            conn.rollback()
            conn.close()
        raise

def verify_migration(db_path: str = "greyhound_racing_data.db"):
    """Verify the migration was successful."""
    
    logger.info("🔍 Verifying TGR enhancement migration...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    expected_tables = [
        'tgr_dog_performance_summary',
        'tgr_expert_insights', 
        'tgr_venue_performance',
        'tgr_distance_performance',
        'tgr_scraping_log',
        'tgr_enhanced_feature_cache',
        'tgr_enhanced_dog_form'
    ]
    
    for table in expected_tables:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", [table])
        if cursor.fetchone():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"✅ {table}: {count} records")
        else:
            logger.error(f"❌ Missing table: {table}")
    
    # Check views
    expected_views = ['vw_tgr_dog_summary', 'vw_tgr_recent_activity']
    for view in expected_views:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name=?", [view])
        if cursor.fetchone():
            logger.info(f"✅ View created: {view}")
        else:
            logger.error(f"❌ Missing view: {view}")
    
    # Check enhanced cache migration
    cursor.execute("SELECT COUNT(*) FROM tgr_enhanced_feature_cache")
    enhanced_cache_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM tgr_feature_cache")
    original_cache_count = cursor.fetchone()[0]
    
    logger.info(f"📊 Cache migration: {original_cache_count} original → {enhanced_cache_count} enhanced")
    
    conn.close()
    logger.info("✅ Migration verification completed")

def populate_sample_enhanced_data(db_path: str = "greyhound_racing_data.db"):
    """Populate some sample enhanced TGR data for testing."""
    
    logger.info("🧪 Populating sample enhanced TGR data...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Sample performance summaries
    sample_dogs = [
        ("BALLARAT STAR", 15, 3, 8, 20.0, 53.3, 3.2, 1, 85.5, "improving", 3, 5),
        ("SWIFT THUNDER", 22, 5, 12, 22.7, 54.5, 2.8, 1, 78.2, "stable", 2, 4),
        ("RACING LEGEND", 18, 2, 6, 11.1, 33.3, 4.1, 2, 92.1, "declining", 4, 6)
    ]
    
    for dog_data in sample_dogs:
        (dog_name, total_entries, wins, places, win_pct, place_pct, avg_pos, 
         best_pos, consistency, trend, dist_variety, venues) = dog_data
        
        performance_data = {
            "total_starts": total_entries,
            "wins": wins,
            "places": places,
            "win_percentage": win_pct,
            "place_percentage": place_pct,
            "average_position": avg_pos,
            "best_position": best_pos,
            "consistency_score": consistency,
            "recent_form_trend": trend,
            "distance_versatility": dist_variety
        }
        
        venue_analysis = {
            "BALLARAT": {"starts": 8, "wins": 2, "win_rate": 25.0},
            "GEELONG": {"starts": 7, "wins": 1, "win_rate": 14.3}
        }
        
        distance_analysis = {
            "400m": {"starts": 12, "wins": 2, "win_rate": 16.7},
            "500m": {"starts": 6, "wins": 1, "win_rate": 16.7}
        }
        
        cursor.execute("""
            INSERT OR REPLACE INTO tgr_dog_performance_summary
            (dog_name, performance_data, venue_analysis, distance_analysis,
             total_entries, wins, places, win_percentage, place_percentage,
             average_position, best_position, consistency_score, form_trend,
             distance_versatility, venues_raced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            dog_name, json.dumps(performance_data), json.dumps(venue_analysis), 
            json.dumps(distance_analysis), total_entries, wins, places, win_pct,
            place_pct, avg_pos, best_pos, consistency, trend, dist_variety, venues
        ])
    
    # Sample expert insights
    sample_insights = [
        ("BALLARAT STAR", "expert_insight", "2025-08-20", "BALLARAT", 
         "Strong recent form with improved box speed. Watch for early pace.", "expert_analysis", 0.7),
        ("SWIFT THUNDER", "dog_comment", "2025-08-22", "GEELONG",
         "Struggled with the slow early pace but finished well. Better suited to faster tracks.", "form_guide", 0.2),
        ("RACING LEGEND", "race_preview", "2025-08-23", "SANDOWN",
         "Veteran performer showing signs of decline but still competitive in lower grades.", "expert_analysis", -0.1)
    ]
    
    for insight in sample_insights:
        cursor.execute("""
            INSERT OR IGNORE INTO tgr_expert_insights
            (dog_name, comment_type, race_date, venue, comment_text, source, sentiment_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, insight)
    
    # Sample enhanced form entries
    sample_form_entries = [
        ("BALLARAT STAR", "2025-08-20", "BALLARAT", "Grade 5", "400m", 3, 
         '["2", "1", "3", "4", "2"]', 30.5, "Strong early, held on well", 4.20, "T. Johnson"),
        ("SWIFT THUNDER", "2025-08-22", "GEELONG", "Maiden", "500m", 1,
         '["1", "3", "2", "5", "1"]', 32.1, "Impressive debut win", 8.50, "M. Smith"),
        ("RACING LEGEND", "2025-08-23", "SANDOWN", "Grade 6", "450m", 5,
         '["4", "5", "3", "6", "4"]', 29.8, "Battled but outclassed", 12.00, "R. Brown")
    ]
    
    for form_entry in sample_form_entries:
        cursor.execute("""
            INSERT OR IGNORE INTO tgr_enhanced_dog_form
            (dog_name, race_date, venue, grade, distance, box_number, recent_form,
             weight, comments, odds, trainer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, form_entry)
    
    conn.commit()
    conn.close()
    
    logger.info("✅ Sample enhanced TGR data populated")

if __name__ == "__main__":
    # Run the migration
    migrate_tgr_enhancement()
    
    # Populate sample data
    populate_sample_enhanced_data()
    
    print("🎉 TGR Enhancement Migration Complete!")
    print("   - Enhanced TGR tables created")
    print("   - Existing cache data migrated") 
    print("   - Sample data populated")
    print("   - Ready for enhanced TGR data capture!")
