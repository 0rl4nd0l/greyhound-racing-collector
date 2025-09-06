#!/usr/bin/env python3
"""
Integration tests for API duplicate detection and end-to-end flow.

Tests cover:
- API endpoint /api/upcoming_races_csv duplicate detection
- End-to-end flow with sample CSVs in UPCOMING_RACES_DIR
- Validation that API returns one race per CSV with unique race_ids
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import requests


class TestAPIIntegrationDuplicateDetection:
    """Test API integration for duplicate detection and race processing."""
    
    def test_api_upcoming_races_csv_no_duplicates(self, tmp_path):
        """Test that /api/upcoming_races_csv returns one race per CSV with unique race_ids."""
        # Create temporary upcoming races directory
        upcoming_dir = tmp_path / "upcoming_races"
        upcoming_dir.mkdir()
        
        # Create multiple test CSV files
        test_csvs = [
            ("Race_1_GEE_2025-09-03.csv", "Dog Name,BOX,WGT\nDog A,1,32.0\nDog B,2,31.5"),
            ("Race_2_RICH_2025-09-03.csv", "Dog Name,BOX,WGT\nDog C,1,30.8\nDog D,2,31.2"),
            ("Race_3_DAPT_2025-09-03.csv", "Dog Name,BOX,WGT\nDog E,1,32.5\nDog F,2,30.9"),
        ]
        
        # Write CSV files
        for filename, content in test_csvs:
            csv_file = upcoming_dir / filename
            csv_file.write_text(content)
        
        # Mock the upcoming directory in the app
        with patch('app.UPCOMING_DIR', str(upcoming_dir)):
            try:
                # Try to test with the actual Flask app if available
                from app import app
                
                # Set up test client
                app.config['TESTING'] = True
                with app.test_client() as client:
                    # Make request to the API endpoint
                    response = client.get('/api/upcoming_races_csv')
                    
                    # Should get successful response
                    assert response.status_code == 200
                    
                    # Parse JSON response
                    data = json.loads(response.data)
                    assert data['success'] is True
                    
                    races = data.get('races', [])
                    
                    # Should have one race per CSV file
                    assert len(races) == len(test_csvs), f"Expected {len(test_csvs)} races, got {len(races)}"
                    
                    # Extract race_ids and check for uniqueness
                    race_ids = [race.get('race_id') for race in races if 'race_id' in race]
                    unique_race_ids = set(race_ids)
                    assert len(race_ids) == len(unique_race_ids), "Found duplicate race_ids in API response"
                    
                    # Extract filenames and check for uniqueness
                    filenames = [race.get('filename') for race in races if 'filename' in race]
                    unique_filenames = set(filenames)
                    assert len(filenames) == len(unique_filenames), "Found duplicate filenames in API response"
                    
                    # Verify each CSV file is represented by exactly one race
                    expected_filenames = set([csv[0] for csv in test_csvs])
                    actual_filenames = set(filenames)
                    assert actual_filenames == expected_filenames, "API response doesn't match expected CSV files"
                    
            except ImportError:
                pytest.skip("Flask app not available for testing")
    
    def test_end_to_end_csv_placement_and_api_response(self, tmp_path):
        """Test end-to-end flow: place CSVs, refresh, validate API response."""
        # Create temporary upcoming races directory
        upcoming_dir = tmp_path / "upcoming_races"  
        upcoming_dir.mkdir()
        
        # Sample CSV content with proper form guide structure
        sample_csv_content = """Dog Name,BOX,WGT,DIST,TRACK,DATE,TIME,G,SP,PLC
Super Fast,1,32.0,500,GEE,2025-09-03,30.15,5,2.40,1
Lightning Bolt,2,31.8,500,GEE,2025-09-03,30.28,5,3.50,2
Rocket Dog,3,30.9,500,GEE,2025-09-03,30.45,5,8.10,3
Speed Demon,4,33.2,500,GEE,2025-09-03,30.52,5,5.20,4
"""
        
        # Create sample CSV files with different race numbers/venues
        sample_races = [
            ("Race 1 - GEE - 2025-09-03.csv", sample_csv_content.replace("GEE", "GEE")),
            ("Race 2 - RICH - 2025-09-03.csv", sample_csv_content.replace("GEE", "RICH")),
        ]
        
        # Place CSV files in upcoming races directory
        for filename, content in sample_races:
            csv_file = upcoming_dir / filename
            csv_file.write_text(content)
        
        # Mock the upcoming directory and test
        with patch('app.UPCOMING_DIR', str(upcoming_dir)):
            try:
                from app import app
                
                app.config['TESTING'] = True
                with app.test_client() as client:
                    # Test with refresh=true to force reload
                    response = client.get('/api/upcoming_races_csv?refresh=true')
                    
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert data['success'] is True
                    
                    races = data.get('races', [])
                    
                    # Should have found our CSV files
                    assert len(races) == len(sample_races)
                    
                    # Validate response structure
                    for race in races:
                        assert 'race_id' in race
                        assert 'filename' in race
                        assert 'venue' in race
                        assert 'race_name' in race
                        
                        # race_id should be non-empty
                        assert race['race_id'].strip() != ""
                        
                        # filename should match one of our test files
                        assert race['filename'] in [sr[0] for sr in sample_races]
                    
                    # Test that race_ids are unique
                    race_ids = [race['race_id'] for race in races]
                    assert len(race_ids) == len(set(race_ids)), "Race IDs are not unique"
                    
                    # Test response metadata
                    assert 'count' in data
                    assert data['count'] == len(races)
                    assert 'timestamp' in data
                    
            except ImportError:
                pytest.skip("Flask app not available for testing")
    
    def test_api_response_shape_and_semantics(self, tmp_path):
        """Test that API response has correct shape and semantics."""
        # Create minimal test setup
        upcoming_dir = tmp_path / "upcoming_races"
        upcoming_dir.mkdir()
        
        # Single CSV file for testing response structure
        csv_content = """Dog Name,BOX,WGT,DIST,TRACK,DATE
Test Dog,1,32.0,500,TEST,2025-09-03
Another Dog,2,31.5,500,TEST,2025-09-03
"""
        
        csv_file = upcoming_dir / "Race 1 - TEST - 2025-09-03.csv"
        csv_file.write_text(csv_content)
        
        with patch('app.UPCOMING_DIR', str(upcoming_dir)):
            try:
                from app import app
                
                app.config['TESTING'] = True
                with app.test_client() as client:
                    response = client.get('/api/upcoming_races_csv')
                    
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    
                    # Test top-level response structure
                    assert isinstance(data, dict)
                    assert 'success' in data
                    assert 'races' in data
                    assert 'count' in data
                    assert 'timestamp' in data
                    
                    assert data['success'] is True
                    assert isinstance(data['races'], list)
                    assert isinstance(data['count'], int)
                    assert data['count'] == len(data['races'])
                    
                    # Test race semantics - should not have outcome fields
                    races = data['races']
                    if races:
                        race = races[0]
                        
                        # Should have race identification fields
                        expected_fields = ['race_id', 'filename', 'race_name']
                        for field in expected_fields:
                            assert field in race, f"Missing field: {field}"
                        
                        # Should NOT have outcome fields (this is upcoming races, not results)
                        forbidden_fields = ['winner_name', 'winner_margin', 'winning_time', 'result']
                        for field in forbidden_fields:
                            assert field not in race, f"Should not have outcome field: {field}"
                        
                        # race_id should be consistent format (hex string)
                        race_id = race['race_id']
                        assert isinstance(race_id, str)
                        assert len(race_id) > 0
                        # Should be hexadecimal (from MD5 hash)
                        try:
                            int(race_id, 16)  # Should parse as hex
                        except ValueError:
                            pytest.fail(f"race_id '{race_id}' is not a valid hexadecimal string")
            
            except ImportError:
                pytest.skip("Flask app not available for testing")
    
    def test_empty_upcoming_races_directory(self, tmp_path):
        """Test API behavior with empty upcoming races directory."""
        # Create empty upcoming races directory
        upcoming_dir = tmp_path / "upcoming_races"
        upcoming_dir.mkdir()
        
        with patch('app.UPCOMING_DIR', str(upcoming_dir)):
            try:
                from app import app
                
                app.config['TESTING'] = True
                with app.test_client() as client:
                    response = client.get('/api/upcoming_races_csv')
                    
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    
                    # Should still be successful but with no races
                    assert data['success'] is True
                    assert data['races'] == []
                    assert data['count'] == 0
                    
            except ImportError:
                pytest.skip("Flask app not available for testing")
    
    def test_invalid_csv_files_handling(self, tmp_path):
        """Test that invalid CSV files are handled gracefully."""
        upcoming_dir = tmp_path / "upcoming_races"
        upcoming_dir.mkdir()
        
        # Create mix of valid and invalid files
        files_to_create = [
            ("valid_race.csv", "Dog Name,BOX\nDog A,1\nDog B,2"),
            ("empty_file.csv", ""),
            ("not_a_csv.txt", "This is not a CSV"),
            ("malformed.csv", "Dog Name,BOX\n\"Unclosed quote,1"),
        ]
        
        for filename, content in files_to_create:
            file_path = upcoming_dir / filename
            file_path.write_text(content)
        
        with patch('app.UPCOMING_DIR', str(upcoming_dir)):
            try:
                from app import app
                
                app.config['TESTING'] = True
                with app.test_client() as client:
                    response = client.get('/api/upcoming_races_csv')
                    
                    # API should still work and return successful response
                    assert response.status_code == 200
                    data = json.loads(response.data)
                    assert 'success' in data
                    assert 'races' in data
                    
                    # Should only process valid CSV files (txt files should be ignored)
                    # Empty and malformed CSVs might be processed or skipped depending on validation
                    races = data.get('races', [])
                    
                    # At minimum, should not crash and should return list
                    assert isinstance(races, list)
                    
                    # If any races processed, they should have proper structure
                    for race in races:
                        assert 'race_id' in race
                        assert 'filename' in race
                        # Only .csv files should be processed
                        assert race['filename'].endswith('.csv')
                    
            except ImportError:
                pytest.skip("Flask app not available for testing")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
