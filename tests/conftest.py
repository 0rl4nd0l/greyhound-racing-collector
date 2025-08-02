"""
Pytest Configuration for Comprehensive Backend Tests
====================================================

This file provides shared fixtures and configuration for all backend tests.
"""

import pytest
import os
import sqlite3
import tempfile
import shutil
from flask import Flask
from flask.testing import FlaskClient

# Import the Flask app
from app import app as flask_app


@pytest.fixture(scope="session")
def temp_db():
    """Create a temporary database for testing session"""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    yield db_path
    
    # Cleanup
    os.close(db_fd)
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture(scope="function")
def test_app(temp_db):
    """Create Flask app configured for testing"""
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    upcoming_dir = os.path.join(temp_dir, "upcoming_races")
    processed_dir = os.path.join(temp_dir, "processed")
    
    os.makedirs(upcoming_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Configure app for testing
    flask_app.config.update({
        'TESTING': True,
        'DATABASE_PATH': temp_db,
        'UPCOMING_DIR': upcoming_dir,
        'PROCESSED_DIR': processed_dir,
        'UPLOAD_FOLDER': upcoming_dir,
        'SECRET_KEY': 'test-secret-key',
        'WTF_CSRF_ENABLED': False,  # Disable CSRF for testing
    })
    
    # Create application context
    ctx = flask_app.app_context()
    ctx.push()
    
    yield flask_app
    
    # Cleanup
    ctx.pop()
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def client(test_app):
    """Create test client"""
    return test_app.test_client()


@pytest.fixture
def runner(test_app):
    """Create test CLI runner"""
    return test_app.test_cli_runner()


def setup_test_data(db_path):
    """Setup test data in the test database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create basic schema for testing
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dogs (
            dog_id TEXT PRIMARY KEY,
            dog_name TEXT UNIQUE,
            total_races INTEGER DEFAULT 0,
            total_wins INTEGER DEFAULT 0,
            total_places INTEGER DEFAULT 0,
            best_time TEXT,
            average_position REAL,
            last_race_date TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS race_metadata (
            race_id TEXT PRIMARY KEY,
            venue TEXT,
            race_number INTEGER,
            race_date TEXT,
            race_name TEXT,
            grade TEXT,
            distance TEXT,
            field_size INTEGER,
            winner_name TEXT,
            winner_odds TEXT,
            winner_margin TEXT,
            url TEXT,
            extraction_timestamp TEXT,
            track_condition TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dog_race_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id TEXT,
            dog_name TEXT,
            box_number INTEGER,
            finish_position INTEGER,
            individual_time TEXT,
            weight REAL,
            trainer_name TEXT,
            odds_decimal REAL,
            margin TEXT,
            sectional_1st TEXT,
            sectional_2nd TEXT,
            FOREIGN KEY (race_id) REFERENCES race_metadata (race_id)
        )
    ''')
    
    # Insert test data
    test_dogs = [
        ('test_dog_1', 'Test Greyhound One', 15, 5, 8, '18.45', 3.8, '2025-01-15'),
        ('test_dog_2', 'Test Greyhound Two', 12, 3, 6, '18.72', 4.2, '2025-01-14'),
        ('test_dog_3', 'Test Greyhound Three', 8, 2, 4, '19.15', 3.5, '2025-01-13'),
    ]
    
    for dog in test_dogs:
        cursor.execute('''
            INSERT OR REPLACE INTO dogs 
            (dog_id, dog_name, total_races, total_wins, total_places, best_time, average_position, last_race_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', dog)
    
    test_races = [
        ('test_race_1', 'Test Track A', 1, '2025-01-15', 'Test Race One', 'Grade 5', '500m', 8, 
         'Test Greyhound One', '2.50', '1.2L', 'http://test.url/1', '2025-01-15T19:00:00', 'Good'),
        ('test_race_2', 'Test Track B', 2, '2025-01-14', 'Test Race Two', 'Grade 4', '520m', 6, 
         'Test Greyhound Two', '3.20', '0.8L', 'http://test.url/2', '2025-01-14T20:00:00', 'Slow'),
    ]
    
    for race in test_races:
        cursor.execute('''
            INSERT OR REPLACE INTO race_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', race)
    
    test_dog_races = [
        ('test_race_1', 'Test Greyhound One', 1, 1, '18.45', 30.2, 'Test Trainer A', 2.50, 'Win'),
        ('test_race_1', 'Test Greyhound Two', 2, 3, '18.72', 30.8, 'Test Trainer B', 4.80, '2.5L'),
        ('test_race_2', 'Test Greyhound Two', 1, 1, '18.85', 30.5, 'Test Trainer B', 3.20, 'Win'),
        ('test_race_2', 'Test Greyhound Three', 3, 2, '19.15', 29.9, 'Test Trainer C', 5.40, '1.2L'),
    ]
    
    for dog_race in test_dog_races:
        cursor.execute('''
            INSERT OR REPLACE INTO dog_race_data 
            (race_id, dog_name, box_number, finish_position, individual_time, weight, trainer_name, odds_decimal, margin)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', dog_race)
    
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def setup_database(temp_db):
    """Automatically setup test database for each test"""
    setup_test_data(temp_db)
    yield
    # Database cleanup is handled by temp_db fixture


@pytest.fixture
def sample_csv_content():
    """Provide sample CSV content for upload tests"""
    return """Dog Name,Box,Weight,Trainer
1. Sample Dog One,1,30.0,Sample Trainer A
2. Sample Dog Two,2,31.5,Sample Trainer B
3. Sample Dog Three,3,29.8,Sample Trainer C
4. Sample Dog Four,4,30.2,Sample Trainer D
"""


@pytest.fixture
def create_test_race_file(test_app):
    """Factory fixture to create test race files"""
    created_files = []
    
    def _create_file(filename, content=None):
        if content is None:
            content = """Dog Name,Box,Weight,Trainer
1. Test Race Dog One,1,30.0,Test Trainer A
2. Test Race Dog Two,2,31.0,Test Trainer B
3. Test Race Dog Three,3,29.5,Test Trainer C
"""
        
        filepath = os.path.join(test_app.config['UPCOMING_DIR'], filename)
        with open(filepath, 'w') as f:
            f.write(content)
        
        created_files.append(filepath)
        return filepath
    
    yield _create_file
    
    # Cleanup created files
    for filepath in created_files:
        if os.path.exists(filepath):
            os.remove(filepath)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "upload: marks tests that test file upload functionality"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark database tests
        if "database" in item.name.lower() or "db" in item.name.lower():
            item.add_marker(pytest.mark.database)
        
        # Mark upload tests
        if "upload" in item.name.lower():
            item.add_marker(pytest.mark.upload)
        
        # Mark integration tests
        if "integration" in item.name.lower() or "workflow" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        
        # Mark slow tests (typically integration or large data tests)
        if "large" in item.name.lower() or "integration" in item.name.lower():
            item.add_marker(pytest.mark.slow)
