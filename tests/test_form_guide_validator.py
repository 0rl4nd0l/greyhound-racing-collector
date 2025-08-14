import json
import tempfile
from pathlib import Path
import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tools.form_guide_validator import (
    FormGuideValidator, 
    GuideIssue, 
    ValidationSeverity,
    validate_only
)


class TestFormGuideValidator:
    """Test suite for FormGuideValidator following FORM_GUIDE_SPEC.md requirements."""
    
    def test_bom_removal(self, tmp_path):
        """Test BOM (Byte Order Mark) removal."""
        content = "\ufeffDog Name,Box,Weight\nLightning Bolt,1,30.0"
        file_path = tmp_path / "test_bom.csv"
        file_path.write_text(content, encoding='utf-8-sig')
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that BOM removal was detected
        bom_issues = [issue for issue in issues if "BOM" in issue.message]
        assert len(bom_issues) == 1
        assert bom_issues[0].severity == ValidationSeverity.INFO
    
    def test_line_ending_normalization(self, tmp_path):
        """Test line ending normalization."""
        content = "Dog Name,Box,Weight\r\nLightning Bolt,1,30.0\r\n"
        file_path = tmp_path / "test_line_endings.csv"
        file_path.write_bytes(content.encode('utf-8'))
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that line ending normalization was detected
        line_ending_issues = [issue for issue in issues if "line endings" in issue.message]
        assert len(line_ending_issues) == 1
        assert line_ending_issues[0].severity == ValidationSeverity.INFO
    
    def test_mixed_delimiters_detection(self, tmp_path):
        """Test mixed delimiters detection."""
        content = "Dog Name,Box;Weight\nLightning Bolt,1;30.0\n"
        file_path = tmp_path / "test_mixed_delimiters.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that mixed delimiters were detected
        mixed_delimiter_issues = [issue for issue in issues if "Mixed delimiters" in issue.message]
        assert len(mixed_delimiter_issues) > 0
        assert all(issue.severity == ValidationSeverity.WARNING for issue in mixed_delimiter_issues)
    
    def test_invisible_unicode_characters(self, tmp_path):
        """Test invisible Unicode character detection and cleaning."""
        # Add some invisible control characters
        content = "Dog Name,Box,Weight\nLightning\u200bBolt,1,30.0\n"  # Zero-width space
        file_path = tmp_path / "test_invisible_chars.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that invisible characters were detected and cleaned
        invisible_char_issues = [issue for issue in issues if "invisible Unicode" in issue.message]
        assert len(invisible_char_issues) > 0
        assert all(issue.severity == ValidationSeverity.WARNING for issue in invisible_char_issues)
    
    def test_embedded_newlines_detection(self, tmp_path):
        """Test embedded newlines detection."""
        content = '"Dog Name",Box,Weight\n"Lightning\nBolt",1,30.0\n'
        file_path = tmp_path / "test_embedded_newlines.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that embedded newlines were detected
        embedded_newline_issues = [issue for issue in issues if "Embedded newline" in issue.message]
        assert len(embedded_newline_issues) > 0
        assert all(issue.severity == ValidationSeverity.WARNING for issue in embedded_newline_issues)
    
    def test_forward_fill_continuation_rows(self, tmp_path):
        """Test forward-fill rule for continuation rows."""
        content = """Dog Name,Box,Weight,Trainer
Lightning Bolt,1,30.0,J. Smith
,,,Additional comments
Speedy Susan,2,31.5,M. Jones
,,,More comments
"""
        file_path = tmp_path / "test_forward_fill.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that forward-fill was applied
        forward_fill_issues = [issue for issue in issues if "Forward-filled" in issue.message]
        assert len(forward_fill_issues) > 0
        
        # Check that dog names were forward-filled correctly
        assert df.iloc[1]['Dog Name'] == 'Lightning Bolt'
        assert df.iloc[3]['Dog Name'] == 'Speedy Susan'
    
    def test_blank_rows_handling(self, tmp_path):
        """Test handling of blank rows."""
        content = """Dog Name,Box,Weight
Lightning Bolt,1,30.0


Speedy Susan,2,31.5
"""
        file_path = tmp_path / "test_blank_rows.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Verify that blank rows don't break parsing
        assert len(df) >= 2  # Should have at least the two valid rows
    
    def test_dog_name_validation(self, tmp_path):
        """Test dog name validation (required field)."""
        content = """Dog Name,Box,Weight
,1,30.0
Lightning Bolt,2,31.5
"""
        file_path = tmp_path / "test_dog_name_validation.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that missing dog name was flagged as error
        dog_name_errors = [issue for issue in issues if "dog name" in issue.message.lower() and issue.severity == ValidationSeverity.ERROR]
        assert len(dog_name_errors) > 0
    
    def test_box_number_validation(self, tmp_path):
        """Test box number validation (1-8 range)."""
        content = """Dog Name,Box,Weight
Lightning Bolt,0,30.0
Speedy Susan,9,31.5
Thunder Strike,5,32.0
"""
        file_path = tmp_path / "test_box_validation.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that invalid box numbers were flagged
        box_errors = [issue for issue in issues if "Box number" in issue.message and "outside valid range" in issue.message]
        assert len(box_errors) == 2  # Should have errors for box 0 and box 9
    
    def test_weight_validation(self, tmp_path):
        """Test weight validation (20.0-40.0 range)."""
        content = """Dog Name,Box,Weight
Lightning Bolt,1,15.0
Speedy Susan,2,45.0
Thunder Strike,3,30.0
"""
        file_path = tmp_path / "test_weight_validation.csv"
        file_path.write_text(content)
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Check that out-of-range weights were flagged as warnings
        weight_warnings = [issue for issue in issues if "Weight" in issue.message and "outside typical range" in issue.message]
        assert len(weight_warnings) == 2  # Should have warnings for 15.0 and 45.0
    
    def test_encoding_detection(self, tmp_path):
        """Test automatic encoding detection."""
        content = "Dog Name,Box,Weight\nLightning Bolt,1,30.0\n"
        
        # Test with different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            file_path = tmp_path / f"test_encoding_{encoding}.csv"
            file_path.write_text(content, encoding=encoding)
            
            validator = FormGuideValidator()
            df, issues = validator.validate_csv_file(file_path)
            
            # Should successfully parse regardless of encoding
            assert len(df) == 1
            assert df.iloc[0]['Dog Name'] == 'Lightning Bolt'
    
    def test_delimiter_detection(self):
        """Test automatic delimiter detection."""
        validator = FormGuideValidator()
        
        # Test comma delimiter
        comma_content = "Name,Box,Weight\nDog,1,30.0"
        assert validator.detect_delimiter(comma_content) == ','
        
        # Test semicolon delimiter
        semicolon_content = "Name;Box;Weight\nDog;1;30.0"
        assert validator.detect_delimiter(semicolon_content) == ';'
        
        # Test tab delimiter
        tab_content = "Name\tBox\tWeight\nDog\t1\t30.0"
        assert validator.detect_delimiter(tab_content) == '\t'
    
    def test_column_pattern_matching(self):
        """Test column pattern matching for dog name and box columns."""
        validator = FormGuideValidator()
        
        # Test dog name column detection
        columns = ['Dog Name', 'Box', 'Weight']
        assert validator._find_dog_name_column(columns) == 'Dog Name'
        
        columns = ['Name', 'Box', 'Weight']
        assert validator._find_dog_name_column(columns) == 'Name'
        
        columns = ['Greyhound', 'Box', 'Weight']
        assert validator._find_dog_name_column(columns) == 'Greyhound'
        
        # Test box column detection
        columns = ['Dog Name', 'Box', 'Weight']
        assert validator._find_box_column(columns) == 'Box'
        
        columns = ['Dog Name', 'Trap', 'Weight']
        assert validator._find_box_column(columns) == 'Trap'
    
    def test_comprehensive_edge_case(self, tmp_path):
        """Test a comprehensive edge case with multiple issues."""
        # BOM + mixed delimiters + embedded newlines + invalid data
        content = '\ufeff"Dog Name",Box;Weight\n"Lightning\nBolt",0;15.0\n"Speedy Susan",9,45.0\n'
        file_path = tmp_path / "test_comprehensive.csv"
        file_path.write_text(content, encoding='utf-8-sig')
        
        validator = FormGuideValidator()
        df, issues = validator.validate_csv_file(file_path)
        
        # Should have multiple types of issues
        issue_types = [issue.severity for issue in issues]
        assert ValidationSeverity.INFO in issue_types  # BOM removal
        assert ValidationSeverity.WARNING in issue_types  # Mixed delimiters, embedded newlines, weight range
        assert ValidationSeverity.ERROR in issue_types  # Invalid box numbers
    
    def test_validate_only_function(self, tmp_path):
        """Test the validate_only CLI function."""
        content = """Dog Name,Box,Weight
Lightning Bolt,1,30.0
Speedy Susan,2,31.5
"""
        file_path = tmp_path / "test_validate_only.csv"
        file_path.write_text(content)
        
        # Run validate_only function
        report = validate_only(file_path)
        
        # Check report structure
        assert "file_path" in report
        assert "timestamp" in report
        assert "success" in report
        assert "data_records_count" in report
        assert "issues" in report
        assert "issues_summary" in report
        
        # Check that JSON report file was created
        json_report_path = file_path.with_suffix('.report.json')
        assert json_report_path.exists()
        
        # Load and verify JSON report
        with json_report_path.open('r') as f:
            json_report = json.load(f)
        assert json_report["data_records_count"] == 2


# Fixture for creating test files
@pytest.fixture
def mock_file(tmp_path):
    content = """Dog Name,Box,Weight,Trainer
1. Sample Dog One,1,30.0,Sample Trainer A
2. Sample Dog Two,2,31.5,Sample Trainer B
3. Sample Dog Three,3,29.8,Sample Trainer C
4. Sample Dog Four,4,30.2,Sample Trainer D
"""
    file_path = tmp_path / "test_form_guide.csv"
    file_path.write_text(content)
    return file_path

