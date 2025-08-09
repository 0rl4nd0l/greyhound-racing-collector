import csv
import re
import unicodedata
import hashlib
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import sys
import os

# Add utils directory to path for guardian import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.file_integrity_guardian import FileIntegrityGuardian

@dataclass
class ParsedRace:
    headers: List[str]
    records: List[List[str]]
    file_path: str = ""
    file_hash: str = ""
    
@dataclass
class ValidationReport:
    errors: List[str]
    warnings: List[str] = None
    is_valid: bool = True
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        self.is_valid = len(self.errors) == 0

class CsvIngestion:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def remove_bom(self, text: str) -> str:
        return text.replace('\ufeff', '')

    def detect_delimiter(self, text: str) -> str:
        if ',' in text and ';' in text:
            return ',' if text.count(',') > text.count(';') else ';'
        elif ',' in text:
            return ','
        elif ';' in text:
            return ';'
        return ','

    def has_invisible_chars(self, text: str) -> bool:
        return any(unicodedata.category(char) in ['Cf', 'Cc'] for char in text)
    
    def pre_process(self, text: str) -> str:
        text = self.remove_bom(text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        delimiter = self.detect_delimiter(text)
        return text, delimiter

    def detect_headers(self, first_line: str, delimiter: str) -> List[str]:
        return first_line.split(delimiter)

    def calculate_file_hash(self) -> str:
        """Calculate SHA-256 hash of file content"""
        with open(self.file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    def validate_csv_structure(self, headers: List[str], records: List[List[str]]) -> List[str]:
        """Validate CSV structure and return list of errors"""
        errors = []
        
        if not headers:
            errors.append("CSV file has no headers")
            return errors
            
        if not records:
            errors.append("CSV file has no data records")
            return errors
        
        expected_column_count = len(headers)
        
        for i, record in enumerate(records, start=2):  # Start at 2 since headers are row 1
            if len(record) != expected_column_count:
                errors.append(f"Row {i}: Expected {expected_column_count} columns, got {len(record)}")
        
        return errors
    
    def detect_malformed_entries(self, records: List[List[str]]) -> List[str]:
        """Detect malformed entries that should trigger quarantine"""
        errors = []
        
        for i, record in enumerate(records, start=2):
            # Check for completely empty rows
            if all(cell.strip() == '' for cell in record):
                errors.append(f"Row {i}: Empty record found")
            
            # Check for records with too many empty cells
            empty_cells = sum(1 for cell in record if cell.strip() == '')
            if empty_cells > len(record) / 2:  # More than half empty
                errors.append(f"Row {i}: Too many empty cells ({empty_cells}/{len(record)})")
        
        return errors

    def parse_csv(self) -> Tuple['ParsedRace', 'ValidationReport']:
        # Pre-validate file with integrity guardian
        try:
            guardian = FileIntegrityGuardian()
            validation_result = guardian.validate_file(self.file_path)
            
            if validation_result.should_quarantine:
                # File is problematic, quarantine it
                issues_summary = "; ".join(validation_result.issues)
                guardian.quarantine_file(self.file_path, f"Pre-parse validation failed: {issues_summary}")
                return ParsedRace([], [], self.file_path, ""), ValidationReport([f"File quarantined: {issues_summary}"])
                
        except Exception as guardian_error:
            print(f"⚠️ Guardian validation failed for {self.file_path}: {guardian_error}")
            # Continue with normal parsing if guardian fails
        
        # Increase CSV field size limit to handle large fields
        current_limit = csv.field_size_limit()
        csv.field_size_limit(10 * 1024 * 1024)  # Set to 10MB limit
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                content, delimiter = self.pre_process(content)

                # Calculate file hash for duplicate detection
                file_hash = self.calculate_file_hash()

                # Prepare CSV reader
                reader = csv.reader(content.splitlines(), delimiter=delimiter)
                
                # Read header
                try:
                    headers = next(reader)
                except StopIteration:
                    return ParsedRace([], [], self.file_path, file_hash), ValidationReport(["Empty CSV file"])
                
                records = list(reader)
                
                # Comprehensive validation
                errors = []
                warnings = []
                
                # Check for invisible characters
                if self.has_invisible_chars(content):
                    warnings.append('Invisible characters found and removed')
                
                # Validate CSV structure
                structure_errors = self.validate_csv_structure(headers, records)
                errors.extend(structure_errors)
                
                # Detect malformed entries
                malformed_errors = self.detect_malformed_entries(records)
                errors.extend(malformed_errors)
                
                # Create parsed race object
                parsed_race = ParsedRace(
                    headers=headers, 
                    records=records, 
                    file_path=self.file_path,
                    file_hash=file_hash
                )
                
                validation_report = ValidationReport(errors=errors, warnings=warnings)
                return parsed_race, validation_report
                
        except Exception as e:
            return ParsedRace([], [], self.file_path, ""), ValidationReport([f"Error reading CSV file: {str(e)}"])
        finally:
            # Restore original field size limit
            csv.field_size_limit(current_limit)

# Further expansion needed for complete dog block extraction, validation, etc.
