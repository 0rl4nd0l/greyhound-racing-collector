#!/usr/bin/env python3
"""
Unit tests for CSV validation pipeline enhancements.

Tests cover:
- load_upcoming_races no duplicates
- CSV processing single race per file  
- Venue/date extraction
- Header drift detection and fuzzy Dog Name matching
- Invisible Unicode cleanup logging
- Mixed delimiters detection and parsing
- Continuation rows extreme forward fill
- Non-CSV files quarantined

Tests are aligned with FORM_GUIDE_SPEC and Data semantics rules.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

# Import modules under test
from src.parsers.csv_ingestion import CsvIngestion, ParsedRace, ValidationReport
from tools.form_guide_validator import FormGuideValidator, ValidationSeverity, validate_form_guide_csv


class TestLoadUpcomingRacesNoDuplicates:
    """Test that load_upcoming_races produces no duplicates - one race per CSV file."""
    
    def test_load_upcoming_races_no_duplicates(self, tmp_path):
        """Test that load_upcoming_races returns unique race_ids with one race per CSV file."""
        # Create temporary upcoming races directory
        upcoming_dir = tmp_path / "upcoming_races"
        upcoming_dir.mkdir()
        
        # Create test CSV files
        csv_files = []
        expected_races = 3
        
        for i in range(expected_races):
            csv_content = f"""Dog Name,BOX,WGT,DIST,TRACK,DATE
Dog A,1,32.0,500,GEE,2025-01-{15+i}
Dog B,2,31.5,500,GEE,2025-01-{15+i}
Dog C,3,32.2,500,GEE,2025-01-{15+i}
"""
            csv_file = upcoming_dir / f"Race_{i+1}_GEE_2025-01-{15+i}.csv"
            csv_file.write_text(csv_content)
            csv_files.append(csv_file)
        
        # Mock the app's load_upcoming_races function
        with patch('app.UPCOMING_DIR', str(upcoming_dir)):
            try:
                from app import load_upcoming_races
                races = load_upcoming_races()
                
                # Test expectations
                assert len(races) == expected_races, f"Expected {expected_races} races, got {len(races)}"
                
                # Test unique race_ids
                race_ids = [r.get('race_id') for r in races if 'race_id' in r]
                unique_race_ids = set(race_ids)
                assert len(race_ids) == len(unique_race_ids), "Found duplicate race_ids"
                
                # Test one race per file
                filenames = [r.get('filename') for r in races if 'filename' in r]
                unique_filenames = set(filenames)
                assert len(filenames) == len(unique_filenames), "Found duplicate filenames"
                
            except ImportError:
                pytest.skip("app.load_upcoming_races not available for testing")


class TestCSVProcessingSingleRacePerFile:
    """Test that CSV processing creates exactly one race record per CSV file."""
    
    def test_csv_processing_single_race_per_file(self):
        """Test that each CSV file produces exactly one race record."""
        csv_content = """Dog Name,BOX,WGT,DIST,TRACK,DATE
Dog A,1,32.0,500,GEE,2025-01-15
Dog B,2,31.5,500,GEE,2025-01-15
Dog C,3,32.2,500,GEE,2025-01-15
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            f.flush()
            
            try:
                ingestion = CsvIngestion(f.name)
                parsed_race, validation_report = ingestion.parse_csv()
                
                # Should have parsed successfully
                assert validation_report.is_valid
                assert len(parsed_race.headers) > 0
                assert len(parsed_race.records) > 0
                
                # The records represent individual dog records, not separate races
                # The race-level processing should aggregate these into one race
                dog_names = set()
                if parsed_race.headers and 'Dog Name' in parsed_race.headers:
                    dog_name_idx = parsed_race.headers.index('Dog Name')
                    for record in parsed_race.records:
                        if len(record) > dog_name_idx and record[dog_name_idx]:
                            dog_names.add(record[dog_name_idx])
                
                # Should have multiple dogs in one race file
                assert len(dog_names) > 1, "Expected multiple dogs in form guide"
                
            finally:
                os.unlink(f.name)


class TestVenueDateExtraction:
    """Test venue and date extraction from filenames and CSV content."""
    
    def test_venue_date_extraction_from_filename(self):
        """Test extraction of venue and date from standardized filenames."""
        # Test cases for different filename patterns
        test_cases = [
            ("Race 1 - GEE - 2025-01-15.csv", {"venue": "GEE", "date": "2025-01-15", "race_number": 1}),
            ("Race 5 - RICH - 2025-02-20.csv", {"venue": "RICH", "date": "2025-02-20", "race_number": 5}),
            ("Race 12 - DAPT - 2025-03-10.csv", {"venue": "DAPT", "date": "2025-03-10", "race_number": 12}),
        ]
        
        for filename, expected in test_cases:
            # Import the extraction utility
            try:
                from utils.csv_metadata import parse_race_csv_meta
                
                # Create a temporary file with the expected filename
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    # Write minimal CSV content
                    f.write("Dog Name,BOX\nDog A,1\n")
                    f.flush()
                    
                    # Rename to expected filename
                    temp_path = Path(f.name)
                    target_path = temp_path.parent / filename
                    temp_path.rename(target_path)
                    
                    try:
                        metadata = parse_race_csv_meta(str(target_path))
                        
                        # Verify extraction
                        assert metadata['status'] == 'success'
                        if 'venue' in expected:
                            # Allow for case variations and normalization
                            assert metadata['venue'].upper() == expected['venue'].upper()
                        if 'date' in expected:
                            assert expected['date'] in metadata['race_date']
                        if 'race_number' in expected:
                            assert metadata['race_number'] == expected['race_number']
                            
                    finally:
                        target_path.unlink()
                        
            except ImportError:
                pytest.skip("utils.csv_metadata not available for testing")


class TestHeaderDriftDetectionFuzzyDogName:
    """Test header drift detection with fuzzy Dog Name matching."""
    
    def test_fuzzy_dog_name_detection(self):
        """Test fuzzy matching for Dog Name header variations."""
        csv_content_with_variant_header = """Doggie Name,BOX,WGT
Super Dog,1,32.0
Fast Dog,2,31.5
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content_with_variant_header)
            f.flush()
            
            try:
                ingestion = CsvIngestion(f.name)
                parsed_race, validation_report = ingestion.parse_csv()
                
                # Should detect and correct the fuzzy header match
                assert 'Dog Name' in parsed_race.headers
                
                # Should have warnings about header corrections
                warning_messages = validation_report.warnings
                header_corrections = [w for w in warning_messages if 'header corrections' in w.lower()]
                assert len(header_corrections) > 0
                
            finally:
                os.unlink(f.name)
    
    def test_header_drift_tolerance(self):
        """Test 30% header drift tolerance threshold."""
        # CSV with headers that mostly don't match expected (should trigger drift warning)
        csv_content_drift = """Unknown1,Unknown2,Unknown3,Dog Name,BOX
Val1,Val2,Val3,Dog A,1
Val1,Val2,Val3,Dog B,2
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content_drift)
            f.flush()
            
            try:
                ingestion = CsvIngestion(f.name)
                parsed_race, validation_report = ingestion.parse_csv()
                
                # Should have drift warnings
                drift_warnings = [w for w in validation_report.warnings if 'drift' in w.lower()]
                assert len(drift_warnings) > 0
                
            finally:
                os.unlink(f.name)


class TestInvisibleUnicodeCleanupLogging:
    """Test invisible Unicode character detection, cleaning, and logging."""
    
    def test_invisible_unicode_cleanup_logging(self):
        """Test that invisible Unicode characters are detected, cleaned, and logged."""
        # CSV content with invisible Unicode characters
        csv_content_with_unicode = "Dog Name,BOX,WGT\u200BHidden\nSuper\u200C Dog,1,32.0\u200D\nFast Dog,2,31.5"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write(csv_content_with_unicode)
            f.flush()
            
            try:
                ingestion = CsvIngestion(f.name)
                
                # Test the Unicode cleaning function directly
                cleaned_text, removed_count = ingestion.clean_invisible_unicode(csv_content_with_unicode)
                assert removed_count > 0
                assert '\u200B' not in cleaned_text
                assert '\u200C' not in cleaned_text  
                assert '\u200D' not in cleaned_text
                
                # Test full parsing with logging
                parsed_race, validation_report = ingestion.parse_csv()
                
                # Should have warnings about Unicode cleanup
                unicode_warnings = [w for w in validation_report.warnings if 'unicode' in w.lower()]
                assert len(unicode_warnings) > 0
                
                # Should contain count of removed characters
                cleanup_warnings = [w for w in unicode_warnings if 'cleaned' in w.lower() and 'characters' in w.lower()]
                assert len(cleanup_warnings) > 0
                
            finally:
                os.unlink(f.name)


class TestMixedDelimitersDetectionAndParsing:
    """Test mixed delimiter detection and proper parsing."""
    
    def test_csv_sniffer_delimiter_detection(self):
        """Test CSV Sniffer-based delimiter detection."""
        # Test different delimiter types
        test_cases = [
            ("Dog Name|BOX|WGT\nSuper Dog|1|32.0", "|"),
            ("Dog Name;BOX;WGT\nSuper Dog;1;32.0", ";"),  
            ("Dog Name\tBOX\tWGT\nSuper Dog\t1\t32.0", "\t"),
            ("Dog Name,BOX,WGT\nSuper Dog,1,32.0", ","),
        ]
        
        for content, expected_delimiter in test_cases:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
                f.write(content)
                f.flush()
                
                try:
                    ingestion = CsvIngestion(f.name)
                    
                    # Test delimiter detection
                    detected_delimiter = ingestion.choose_delimiter(content)
                    assert detected_delimiter == expected_delimiter
                    
                    # Test full parsing
                    parsed_race, validation_report = ingestion.parse_csv()
                    assert validation_report.is_valid
                    assert len(parsed_race.records) > 0
                    
                finally:
                    os.unlink(f.name)
    
    def test_mixed_delimiter_detection_in_form_guide_validator(self):
        """Test mixed delimiter detection in FormGuideValidator."""
        # Content with mixed delimiters (should be detected)
        mixed_content = """Dog Name,BOX;WGT
Super Dog,1;32.0
Fast Dog,2;31.5"""
        
        validator = FormGuideValidator()
        cleaned_content, issues = validator.preprocess_csv_content(mixed_content)
        
        # Should detect mixed delimiters
        mixed_delimiter_issues = [issue for issue in issues if 'mixed delimiters' in issue.message.lower()]
        assert len(mixed_delimiter_issues) > 0
        assert validator.parsing_stats['mixed_delimiters_detected'] > 0


class TestContinuationRowsExtremeForwardFill:
    """Test continuation row handling with extreme forward-fill scenarios."""
    
    def test_continuation_rows_extreme_forward_fill(self):
        """Test extreme forward-fill scenarios with multiple continuation rows."""
        # CSV with multiple continuation rows per dog
        csv_content_continuation = """Dog Name,BOX,WGT,DATE,TRACK
Super Dog,1,32.0,2025-01-15,GEE
,,31.9,2025-01-08,GEE
,,32.1,2025-01-01,GEE
Fast Dog,2,31.5,2025-01-15,GEE
,,31.4,2025-01-08,GEE
,,31.6,2025-01-01,GEE
Quick Dog,3,30.8,2025-01-15,GEE
,,30.9,2025-01-08,GEE
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content_continuation)
            f.flush()
            
            try:
                ingestion = CsvIngestion(f.name)
                parsed_race, validation_report = ingestion.parse_csv()
                
                # Should process successfully
                assert validation_report.is_valid
                
                # Check forward-fill warnings
                forward_fill_warnings = [w for w in validation_report.warnings if 'forward-filled' in w.lower()]
                assert len(forward_fill_warnings) > 0
                
                # Verify forward-fill worked - check that continuation rows have dog names
                if 'Dog Name' in parsed_race.headers:
                    dog_name_idx = parsed_race.headers.index('Dog Name')
                    for record in parsed_race.records:
                        if len(record) > dog_name_idx:
                            # Every record should have a dog name after forward-fill
                            assert record[dog_name_idx].strip() != ""
                
            finally:
                os.unlink(f.name)
    
    def test_max_dogs_constraint_enforcement(self):
        """Test that max 10 unique dogs constraint is enforced."""
        # Create CSV with more than 10 unique dogs
        csv_content_many_dogs = "Dog Name,BOX,WGT\n"
        for i in range(12):  # 12 dogs > 10 limit
            csv_content_many_dogs += f"Dog {i+1:02d},1,32.{i}\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content_many_dogs)
            f.flush()
            
            try:
                ingestion = CsvIngestion(f.name)
                parsed_race, validation_report = ingestion.parse_csv()
                
                # Should have error about too many dogs
                too_many_dogs_errors = [e for e in validation_report.errors if 'too many unique dogs' in e.lower()]
                assert len(too_many_dogs_errors) > 0
                
                # Should not be valid due to constraint violation
                assert not validation_report.is_valid
                
            finally:
                os.unlink(f.name)


class TestNonCSVFilesQuarantined:
    """Test that non-CSV files are properly quarantined."""
    
    def test_non_csv_files_quarantined(self):
        """Test that files with non-CSV extensions are quarantined."""
        non_csv_content = "This is not a CSV file"
        
        # Test with .txt file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(non_csv_content)
            f.flush()
            
            try:
                # Use the centralized validation function
                ok, df, validation_report, quarantine_reason = validate_form_guide_csv(f.name)
                
                # Should not pass validation
                assert not ok
                assert quarantine_reason is not None
                assert 'extension' in quarantine_reason.lower()
                assert validation_report['status'] == 'error'
                
            finally:
                os.unlink(f.name)
    
    def test_empty_files_quarantined(self):
        """Test that empty files are quarantined."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write nothing (empty file)
            f.flush()
            
            try:
                ok, df, validation_report, quarantine_reason = validate_form_guide_csv(f.name)
                
                # Should not pass validation
                assert not ok
                assert quarantine_reason is not None
                assert 'empty' in quarantine_reason.lower()
                assert validation_report['status'] == 'error'
                
            finally:
                os.unlink(f.name)
    
    def test_malformed_csv_quarantined(self):
        """Test that malformed CSV files are quarantined."""
        malformed_content = """Dog Name,BOX,WGT
"Unclosed quote dog,1,32.0
Another dog,2,31.5"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(malformed_content)
            f.flush()
            
            try:
                ok, df, validation_report, quarantine_reason = validate_form_guide_csv(f.name)
                
                # May or may not be quarantined depending on CSV parser tolerance
                # But should at least detect the parsing issues
                if not ok:
                    assert quarantine_reason is not None
                    assert validation_report['status'] == 'error'
                else:
                    # If parsed, should have warnings or errors about format issues
                    assert len(validation_report.get('errors', [])) + len(validation_report.get('warnings', [])) > 0
                    
            finally:
                os.unlink(f.name)


class TestFormGuideValidatorIntegration:
    """Integration tests for the complete FormGuideValidator pipeline."""
    
    def test_complete_validation_pipeline(self):
        """Test the complete validation pipeline with a realistic form guide."""
        # Use the existing test fixture
        fixture_path = Path(__file__).parent / "fixtures" / "test_race.csv"
        if not fixture_path.exists():
            pytest.skip("Test fixture not found")
        
        ok, df, validation_report, quarantine_reason = validate_form_guide_csv(fixture_path)
        
        # Should pass validation
        assert ok
        assert df is not None
        assert not df.empty
        assert validation_report['status'] == 'success'
        assert quarantine_reason is None
        
        # Should have processed dogs
        assert validation_report['unique_dogs_found'] > 0
        assert validation_report['total_rows'] > 0
        
        # Should be within dog limit
        assert validation_report['unique_dogs_found'] <= 10
    
    def test_centralized_validation_function_comprehensive(self):
        """Test the centralized validate_form_guide_csv function comprehensively.""" 
        # Test with good CSV
        good_csv = """Dog Name,BOX,WGT,DATE,TRACK
Super Dog,1,32.0,2025-01-15,GEE
Fast Dog,2,31.5,2025-01-15,GEE
Quick Dog,3,30.8,2025-01-15,GEE
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(good_csv)
            f.flush()
            
            try:
                ok, df, validation_report, quarantine_reason = validate_form_guide_csv(f.name)
                
                # Should pass all validations
                assert ok
                assert df is not None
                assert not df.empty
                assert validation_report['status'] == 'success'
                assert quarantine_reason is None
                
                # Should have proper structure
                assert 'file' in validation_report
                assert 'errors' in validation_report
                assert 'warnings' in validation_report
                assert 'parsing_stats' in validation_report
                assert 'unique_dogs_found' in validation_report
                assert 'total_rows' in validation_report
                
                # Should have found dogs
                assert validation_report['unique_dogs_found'] == 3
                assert validation_report['total_rows'] == 3
                
            finally:
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
