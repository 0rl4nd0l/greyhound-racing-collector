#!/usr/bin/env python3
"""
Complete System Reset Script
============================

This script performs a complete reset of the greyhound racing collector system:
1. Clears all databases
2. Removes all form guide directories and files
3. Creates 3 clean directories: unprocessed, processed, upcoming_races
4. Moves test files and unnecessary scripts to archive
5. Cleans up temporary files and caches

Author: AI Assistant
Date: July 11, 2025
"""

import os
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime

def log_action(message):
    """Log reset actions with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def main():
    """Perform complete system reset"""
    log_action("🔄 Starting complete system reset...")
    
    # Get current directory
    base_dir = Path.cwd()
    log_action(f"Working in: {base_dir}")
    
    # 1. Clear all databases
    log_action("🗑️  Clearing databases...")
    databases_dir = base_dir / "databases"
    if databases_dir.exists():
        for db_file in databases_dir.glob("*.db"):
            try:
                db_file.unlink()
                log_action(f"   Deleted: {db_file.name}")
            except Exception as e:
                log_action(f"   Error deleting {db_file.name}: {e}")
    
    # Remove standalone db files
    for db_file in base_dir.glob("*.db"):
        try:
            db_file.unlink()
            log_action(f"   Deleted: {db_file.name}")
        except Exception as e:
            log_action(f"   Error deleting {db_file.name}: {e}")
    
    # 2. Remove all form guide directories
    log_action("🗑️  Removing form guide directories...")
    form_dirs_to_remove = [
        "form_guides",
        "historical_races", 
        "advanced_results",
        "data",
        "prediction_agent",
        "analysis_agent",
        "predictions",
        "ai_models"
    ]
    
    for dir_name in form_dirs_to_remove:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                log_action(f"   Removed directory: {dir_name}")
            except Exception as e:
                log_action(f"   Error removing {dir_name}: {e}")
    
    # 3. Create clean directory structure
    log_action("📁 Creating clean directory structure...")
    clean_dirs = [
        "unprocessed",
        "processed", 
        "upcoming_races",
        "databases"
    ]
    
    for dir_name in clean_dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        log_action(f"   Created: {dir_name}/")
        
        # Add README to each directory
        readme_path = dir_path / "README.md"
        readme_content = {
            "unprocessed": "# Unprocessed Forms\n\nThis directory contains downloaded CSV files that need to be processed.\nFiles are moved here after download and before processing.",
            "processed": "# Processed Forms\n\nThis directory contains CSV files that have been fully processed.\nFiles are moved here after complete race data has been gathered and logged.",
            "upcoming_races": "# Upcoming Races\n\nThis directory contains CSV files for races that are yet to run.\nThese files contain race information but no results yet.",
            "databases": "# Databases\n\nThis directory contains SQLite database files for the racing system.\nMain database: comprehensive_greyhound_data.db"
        }
        
        with open(readme_path, "w") as f:
            f.write(readme_content[dir_name])
    
    # 4. Move test files and unnecessary scripts to archive
    log_action("📦 Moving test files and unnecessary scripts to archive...")
    
    # Create archive directory
    archive_dir = base_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    # Files to archive
    files_to_archive = [
        "test_*.py",
        "fix_database.py",
        "data_cleanup.py", 
        "comprehensive_database_fix.py",
        "enhanced_database_rebuilder.py",
        "comprehensive_data_rebuilder.py",
        "move_files_to_unprocessed.py",
        "race_form_guide.csv",
        "comprehensive_greyhound_data.db",
        "greyhound_data.db",
        "*.json",
        "test_url_fix.py"
    ]
    
    for pattern in files_to_archive:
        for file_path in base_dir.glob(pattern):
            if file_path.is_file():
                try:
                    dest_path = archive_dir / file_path.name
                    if dest_path.exists():
                        dest_path.unlink()
                    shutil.move(str(file_path), str(dest_path))
                    log_action(f"   Archived: {file_path.name}")
                except Exception as e:
                    log_action(f"   Error archiving {file_path.name}: {e}")
    
    # 5. Clean up temporary files and caches
    log_action("🧹 Cleaning up temporary files and caches...")
    
    # Remove __pycache__ directories
    for pycache_dir in base_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            log_action(f"   Removed: {pycache_dir}")
        except Exception as e:
            log_action(f"   Error removing {pycache_dir}: {e}")
    
    # Remove .DS_Store files
    for ds_store in base_dir.rglob(".DS_Store"):
        try:
            ds_store.unlink()
            log_action(f"   Removed: {ds_store}")
        except Exception as e:
            log_action(f"   Error removing {ds_store}: {e}")
    
    # 6. Update app.py to use correct database path
    log_action("⚙️  Updating app.py configuration...")
    app_py = base_dir / "app.py"
    if app_py.exists():
        with open(app_py, "r") as f:
            content = f.read()
        
        # Update database path
        content = content.replace(
            "DATABASE_PATH = './databases/greyhound_racing.db'",
            "DATABASE_PATH = './databases/comprehensive_greyhound_data.db'"
        )
        content = content.replace(
            "DATABASE_PATH = './databases/comprehensive_greyhound_data.db'",
            "DATABASE_PATH = './databases/comprehensive_greyhound_data.db'"
        )
        
        # Update directory paths
        content = content.replace(
            "UNPROCESSED_DIR = './unprocessed'",
            "UNPROCESSED_DIR = './unprocessed'"
        )
        content = content.replace(
            "PROCESSED_DIR = './form_guides/processed'",
            "PROCESSED_DIR = './processed'"
        )
        content = content.replace(
            "UPCOMING_DIR = './upcoming_races'",
            "UPCOMING_DIR = './upcoming_races'"
        )
        
        with open(app_py, "w") as f:
            f.write(content)
        
        log_action("   Updated app.py configuration")
    
    # 7. Create initial database
    log_action("🔧 Creating initial database...")
    db_path = base_dir / "databases" / "comprehensive_greyhound_data.db"
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create race_metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS race_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT UNIQUE NOT NULL,
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
                race_status TEXT
            )
        ''')
        
        # Create dog_race_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dog_race_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
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
                was_scratched BOOLEAN,
                blackbook_link TEXT,
                extraction_timestamp DATETIME,
                data_source TEXT,
                form_guide_json TEXT,
                performance_rating REAL,
                speed_rating REAL,
                class_rating REAL,
                recent_form TEXT,
                win_probability REAL,
                place_probability REAL,
                FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_race_id ON dog_race_data(race_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dog_name ON dog_race_data(dog_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_race_date ON race_metadata(race_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_venue ON race_metadata(venue)')
        
        conn.commit()
        conn.close()
        
        log_action("   Created initial database with clean schema")
        
    except Exception as e:
        log_action(f"   Error creating database: {e}")
    
    # 8. Create summary of reset
    log_action("📋 Creating reset summary...")
    
    summary_path = base_dir / "RESET_SUMMARY.md"
    summary_content = f"""# System Reset Summary

**Reset Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Actions Performed

### 1. Database Cleanup
- Removed all existing database files
- Created fresh database with clean schema
- Database location: `databases/comprehensive_greyhound_data.db`

### 2. Directory Structure Reset
- Removed all form guide directories
- Created 3 clean directories:
  - `unprocessed/` - For downloaded CSV files awaiting processing
  - `processed/` - For CSV files with complete race data
  - `upcoming_races/` - For races yet to run

### 3. File Management
- Moved test files and outdated scripts to `archive/`
- Cleaned up temporary files and caches
- Updated app.py configuration

### 4. Next Steps
1. Run the CSV download process
2. Process the downloaded files
3. Start the web application

## File Structure
```
greyhound_racing_collector/
├── unprocessed/          # Downloaded CSVs awaiting processing
├── processed/            # Fully processed CSVs
├── upcoming_races/       # Future race CSVs
├── databases/           # SQLite databases
├── archive/             # Archived test files and scripts
├── logs/                # System logs
├── static/              # Web app assets
├── templates/           # Web app templates
└── app.py              # Main Flask application
```

## Key Scripts Remaining
- `app.py` - Flask web application
- `form_guide_csv_scraper.py` - CSV download and processing
- `enhanced_race_analyzer.py` - Race analysis
- `logger.py` - Logging system
- `run.py` - Main execution script

System is now ready for a fresh start!
"""
    
    with open(summary_path, "w") as f:
        f.write(summary_content)
    
    log_action("✅ Complete system reset finished!")
    log_action("📖 See RESET_SUMMARY.md for details")
    log_action("🚀 System is ready for fresh CSV download and processing")

if __name__ == "__main__":
    main()
