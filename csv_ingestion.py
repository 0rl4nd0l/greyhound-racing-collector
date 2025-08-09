#!/usr/bin/env python3
"""
CSV Ingestion Layer for Form Guide Data
=======================================

Debug mode allows logging of detailed CSV parsing and processing steps.

This module provides robust CSV ingestion capabilities with:
1. Schema validation to ensure required columns exist
2. Proper mapping of "Dog Name" column to `dog_name`
3. Comprehensive error handling and descriptive error messages
4. Flexible column alias support for different CSV formats

The ingestion layer handles the form guide format where:
- The first column is "Dog Name" which maps to `dog_name`
- Blank rows belong to the dog above them (greyhound race form format)
- Multiple historical races are stored per dog

Author: AI Assistant
Date: January 2025
Version: 1.0.0
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # All required columns must be present
    MODERATE = "moderate"  # Core columns must be present, others optional
    LENIENT = "lenient"    # Only dog_name required


@dataclass
class ColumnMapping:
    """Column mapping configuration"""
    source_name: str
    target_name: str
    required: bool = True
    aliases: List[str] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class ValidationResult:
    """Result of CSV validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_required: List[str]
    available_columns: List[str]
    file_info: Dict[str, Any]


class FormGuideCsvIngestionError(Exception):
    """Custom exception for CSV ingestion errors"""
    pass


class FormGuideCsvIngestor:
    """
    Robust CSV ingestion system for form guide data with schema validation.
    
    Features:
    - Maps "Dog Name" to `dog_name` consistently
    - Schema validation with descriptive error messages
    - Flexible column alias support
    - Handles greyhound form guide format with blank rows
    - Comprehensive error reporting
    """
    
    def __init__(self, db_path: str = 'greyhound_racing_data.db', validation_level: ValidationLevel = ValidationLevel.MODERATE):
        """
        Initialize the CSV ingestor.
        
        Args:
            db_path: Path to the database for caching
            validation_level: How strict to be with column validation
        """
        self.db_path = db_path
        self.validation_level = validation_level
        self.logger = logging.getLogger(__name__)
        
        # Define the standard column mappings for form guide CSVs
        self.column_mappings = {
            # Primary dog identifier - ALWAYS REQUIRED
            "dog_name": ColumnMapping(
                source_name="Dog Name",
                target_name="dog_name", 
                required=True,
                aliases=["DOG NAME", "Dog", "Name", "HOUND", "Greyhound"]
            ),
            
            # Core race performance data
            "sex": ColumnMapping(
                source_name="Sex",
                target_name="sex",
                required=False,
                aliases=["SEX", "S", "Gender"]
            ),
            "place": ColumnMapping(
                source_name="PLC",
                target_name="place",
                required=True,
                aliases=["PLACE", "Position", "POS", "Fin", "FINISH"]
            ),
            "box": ColumnMapping(
                source_name="BOX",
                target_name="box",
                required=True,
                aliases=["Box", "BOX_NUMBER", "Start", "Starting_Box"]
            ),
            "weight": ColumnMapping(
                source_name="WGT",
                target_name="weight",
                required=False,
                aliases=["WEIGHT", "Weight", "Wt", "KG"]
            ),
            "distance": ColumnMapping(
                source_name="DIST",
                target_name="distance",
                required=True,
                aliases=["DISTANCE", "Distance", "Metres", "M", "METERS"]
            ),
            "date": ColumnMapping(
                source_name="DATE",
                target_name="date",
                required=True,
                aliases=["Date", "RACE_DATE", "RaceDate"]
            ),
            "track": ColumnMapping(
                source_name="TRACK",
                target_name="track",
                required=True,
                aliases=["Track", "VENUE", "Venue", "Course"]
            ),
            "grade": ColumnMapping(
                source_name="G",
                target_name="grade",
                required=False,
                aliases=["GRADE", "Grade", "Class", "CLASS"]
            ),
            
            # Performance metrics
            "time": ColumnMapping(
                source_name="TIME",
                target_name="time",
                required=False,
                aliases=["Time", "RACE_TIME", "Individual_Time"]
            ),
            "win_time": ColumnMapping(
                source_name="WIN",
                target_name="win_time",
                required=False,
                aliases=["WIN_TIME", "Winning_Time", "WINNER_TIME"]
            ),
            "bonus": ColumnMapping(
                source_name="BON",
                target_name="bonus",
                required=False,
                aliases=["BONUS", "Bonus"]
            ),
            "first_sectional": ColumnMapping(
                source_name="1 SEC",
                target_name="first_sectional",
                required=False,
                aliases=["FIRST_SEC", "Split", "1ST_SEC", "Sectional"]
            ),
            "margin": ColumnMapping(
                source_name="MGN",
                target_name="margin",
                required=False,
                aliases=["MARGIN", "Margin", "MGN"]
            ),
            "runner_up": ColumnMapping(
                source_name="W/2G",
                target_name="runner_up",
                required=False,
                aliases=["RUNNER_UP", "Second", "2ND"]
            ),
            "pir": ColumnMapping(
                source_name="PIR",
                target_name="pir",
                required=False,
                aliases=["PIR", "Performance_Rating"]
            ),
            "starting_price": ColumnMapping(
                source_name="SP",
                target_name="starting_price",
                required=False,
                aliases=["STARTING_PRICE", "SP", "Odds", "Price"]
            )
        }
        
        # Define required columns based on validation level
        self.required_columns_by_level = {
            ValidationLevel.STRICT: [
                "dog_name", "place", "box", "distance", "date", "track"
            ],
            ValidationLevel.MODERATE: [
                "dog_name", "place", "distance", "date", "track"
            ],
            ValidationLevel.LENIENT: [
                "dog_name"
            ]
        }
    
    def validate_csv_schema(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate CSV schema and return detailed validation results.
        
        if self.logger.debug_mode:
            self.logger.process_logger.debug(f"Validating CSV schema for {file_path}")
        
        Args:
            file_path: Path to the CSV file to validate
            
        Returns:
            ValidationResult with validation details
        """
        file_path = Path(file_path)
        errors = []
        warnings = []
        missing_required = []
        available_columns = []
        
        # File existence check
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"File does not exist: {file_path}"],
                warnings=[],
                missing_required=[],
                available_columns=[],
                file_info={"exists": False, "size": 0}
            )
        
        file_info = {
            "exists": True,
            "size": file_path.stat().st_size,
            "name": file_path.name
        }
        
        try:
            # Read first few rows to check structure
            df = pd.read_csv(file_path, nrows=10)
            available_columns = list(df.columns)
            
            # Check if file appears to be HTML (common issue)
            # TODO: Gap - Improve HTML detection for corrupted CSV files
            # TODO: File corruption handling is limited to basic DOCTYPE check
            # TODO: Add detection for other file formats (JSON, XML, plain text)
            if len(available_columns) == 1 and "DOCTYPE html" in str(df.iloc[0, 0]):
                errors.append("File appears to be HTML, not CSV data")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    missing_required=missing_required,
                    available_columns=available_columns,
                    file_info=file_info
                )
            
            # Check for Dog Name column specifically
            # TODO: Gap - Dog Name column validation needs enhancement
            # TODO: Current aliases may be incomplete for different CSV sources
            # TODO: Add fuzzy matching for slightly misspelled headers
            dog_name_found = False
            dog_name_column = None
            
            for col in available_columns:
                if col.strip() == "Dog Name":
                    dog_name_found = True
                    dog_name_column = col
                    break
                # Check aliases
                for alias in self.column_mappings["dog_name"].aliases:
                    if col.strip().upper() == alias.upper():
                        dog_name_found = True
                        dog_name_column = col
                        warnings.append(f"Found dog name column using alias: '{col}' -> 'dog_name'")
                        break
                
                if dog_name_found:
                    break
            
            if not dog_name_found:
                errors.append(
                    f"Required 'Dog Name' column not found. Available columns: {available_columns}. "
                    f"Expected first column to be 'Dog Name' or one of: {self.column_mappings['dog_name'].aliases}"
                )
                missing_required.append("dog_name")
            
            # Check for other required columns based on validation level
            required_cols = self.required_columns_by_level[self.validation_level]
            
            for req_col in required_cols:
                if req_col == "dog_name":
                    continue  # Already checked above
                    
                mapping = self.column_mappings.get(req_col)
                if not mapping:
                    continue
                    
                found = False
                # Check primary name
                if mapping.source_name in available_columns:
                    found = True
                else:
                    # Check aliases
                    for alias in mapping.aliases:
                        if alias in available_columns:
                            found = True
                            warnings.append(f"Found column '{req_col}' using alias: '{alias}'")
                            break
                
                if not found:
                    missing_required.append(req_col)
                    errors.append(
                        f"Required column '{mapping.source_name}' (maps to '{req_col}') not found. "
                        f"Aliases: {mapping.aliases}"
                    )
            
            # Check if first column is actually Dog Name (form guide format requirement)
            if available_columns and not available_columns[0].strip() == "Dog Name":
                if dog_name_found:
                    warnings.append(
                        f"Dog name column found but not in first position. "
                        f"Found at position: {available_columns.index(dog_name_column) + 1}"
                    )
                else:
                    errors.append(
                        f"First column should be 'Dog Name' but found '{available_columns[0]}'. "
                        "Form guide format requires Dog Name as first column."
                    )
            
            # Additional data quality checks
            if df.empty:
                errors.append("CSV file is empty (no data rows)")
            elif len(df) < 2:
                warnings.append("CSV file has very few rows - may be incomplete")
            
            # Check for obvious formatting issues
            if len(available_columns) < 3:
                warnings.append(f"Only {len(available_columns)} columns found - file may be malformed")
            
        except pd.errors.EmptyDataError:
            errors.append("CSV file is empty or contains no data")
        except pd.errors.ParserError as e:
            errors.append(f"CSV parsing error: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error reading CSV: {str(e)}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            missing_required=missing_required,
            available_columns=available_columns,
            file_info=file_info
        )
    
    def validate_headers(self, headers: List[str]) -> None:
        """
        Validate that all required columns are present.
        
        if self.logger.debug_mode:
            self.logger.process_logger.debug(f"Validating headers: {headers}")
        
        Args:
            headers: The list of column headers from the CSV
        
        Raises:
            ValueError: if any required column is missing or misplaced
        """
        required_columns = self.required_columns_by_level[self.validation_level]
        for target_col in required_columns:
            mapping = self.column_mappings.get(target_col)
            if not mapping:
                continue
                
            found = False
            # Check primary name
            if mapping.source_name in headers:
                found = True
            else:
                # Check aliases
                for alias in mapping.aliases:
                    if alias in headers:
                        found = True
                        break
            
            if not found:
                raise ValueError(f"Missing required column: {mapping.source_name} (maps to {target_col})")

    def map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map CSV columns to standardized names, with Dog Name -> dog_name.
        
        Args:
            df: DataFrame with original column names
            
        Returns:
            DataFrame with mapped column names
        """
        column_mapping_dict = {}
        
        for target_name, mapping in self.column_mappings.items():
            # Check primary name first
            if mapping.source_name in df.columns:
                column_mapping_dict[mapping.source_name] = mapping.target_name
                continue
            
            # Check aliases
            found = False
            for alias in mapping.aliases:
                if alias in df.columns:
                    column_mapping_dict[alias] = mapping.target_name
                    found = True
                    break
            
            if not found and mapping.required:
                raise FormGuideCsvIngestionError(
                    f"Required column '{mapping.source_name}' (target: '{target_name}') not found. "
                    f"Available columns: {list(df.columns)}"
                )
        
        # Apply the mapping
        df_mapped = df.rename(columns=column_mapping_dict)
        
        # Ensure dog_name is properly mapped (critical requirement)
        if "dog_name" not in df_mapped.columns:
            raise FormGuideCsvIngestionError(
                "Failed to map 'Dog Name' column to 'dog_name'. "
                "This is a critical requirement for form guide processing."
            )
        
        return df_mapped
    
    def process_form_guide_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process form guide format where blank rows belong to dog above them.

        if self.logger.debug_mode:
            self.logger.process_logger.debug("Processing form guide format")
        
        Args:
            df: DataFrame with mapped columns
            
        Returns:
            List of dictionaries with processed dog data
        """
        processed_data = []
        current_dog_name = None
        
        for idx, row in df.iterrows():
            if pd.isna(row["dog_name"]) or row["dog_name"].strip() == "":
                row["dog_name"] = current_dog_name
            else:
                dog_name_raw = str(row["dog_name"]).strip()
                current_dog_name = dog_name_raw
                if ". " in current_dog_name:
                    current_dog_name = current_dog_name.split(". ", 1)[1]

            # Skip if we don't have a current dog
            if current_dog_name is None:
                continue
            
            # Create record for this row (historical race data)
            record = {"dog_name": current_dog_name}
            
            # Add all other mapped columns
            for col in df.columns:
                if col != "dog_name":
                    value = row[col]
                    # Clean up empty values
                    if pd.isna(value) or str(value).strip() in ['""', '', 'nan']:
                        value = None
                    else:
                        value = str(value).strip()
                    record[col] = value
            
            # Only add if we have meaningful data (at least place and date)
            if record.get("place") and record.get("date"):
                processed_data.append(record)
        
        return processed_data
    
    def ingest_csv(self, file_path: Union[str, Path], 
                   skip_validation: bool = False) -> Tuple[List[Dict[str, Any]], ValidationResult]:
        """
        Complete CSV ingestion pipeline with validation and processing.
        
        if self.logger.debug_mode:
            self.logger.process_logger.debug(f"Starting CSV ingestion for {file_path}")
        
        Args:
            file_path: Path to CSV file to ingest
            skip_validation: Skip schema validation (not recommended)
            
        Returns:
            Tuple of (processed_data, validation_result)
            
        Raises:
            FormGuideCsvIngestionError: If ingestion fails
        """
        file_path = Path(file_path)
        
        # Step 1: Validate schema
        if not skip_validation:
            validation_result = self.validate_csv_schema(file_path)
            
            if not validation_result.is_valid:
                error_msg = f"CSV validation failed for {file_path}:\n"
                for error in validation_result.errors:
                    error_msg += f"  ❌ {error}\n"
                if validation_result.warnings:
                    error_msg += "Warnings:\n"
                    for warning in validation_result.warnings:
                        error_msg += f"  ⚠️ {warning}\n"
                
                raise FormGuideCsvIngestionError(error_msg)
        else:
            validation_result = ValidationResult(
                is_valid=True, errors=[], warnings=["Validation skipped"], 
                missing_required=[], available_columns=[], file_info={}
            )
        
        try:
            # Check if file is already processed in cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            file_hash = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
            cursor.execute("SELECT 1 FROM processed_race_files WHERE file_hash = ?", (file_hash,))
            if cursor.fetchone() is not None:
                self.logger.debug(f"File {file_path} is already processed, skipping.")
                conn.close()
                return [], ValidationResult(is_valid=True, errors=[], warnings=["Already processed"], missing_required=[], available_columns=[], file_info={})
            conn.close()
            
            # Step 2: Load CSV
            df = pd.read_csv(file_path, on_bad_lines="skip", encoding="utf-8")
            
            # Step 3: Map columns (including Dog Name -> dog_name)
            df_mapped = self.map_columns(df)
            
            # Step 4: Process form guide format
            processed_data = self.process_form_guide_format(df_mapped)
            
            if not processed_data:
                raise FormGuideCsvIngestionError(
                    f"No valid data extracted from {file_path}. "
                    "Check file format and ensure it contains race data."
                )

            # Cache the processed file
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO processed_race_files (file_hash, file_path) VALUES (?, ?)", (file_hash, str(file_path)))
            conn.commit()
            conn.close()
            
            # Update mtime heuristic for optimization
            try:
                from utils.mtime_heuristic import create_mtime_heuristic
                heuristic = create_mtime_heuristic(self.db_path)
                heuristic.update_processed_mtime_from_files([str(file_path)])
            except Exception as e:
                self.logger.debug(f"Mtime heuristic update failed (non-critical): {e}")
            
            self.logger.info(f"Successfully ingested {len(processed_data)} records from {file_path}")
            
            return processed_data, validation_result
            
        except FormGuideCsvIngestionError:
            raise  # Re-raise our custom errors
        except Exception as e:
            raise FormGuideCsvIngestionError(
                f"Unexpected error during CSV ingestion of {file_path}: {str(e)}"
            )
    
    def batch_ingest(self, file_paths: List[Union[str, Path]], 
                     continue_on_error: bool = True) -> Dict[str, Any]:
        """
        Ingest multiple CSV files in batch.
        
        Args:
            file_paths: List of file paths to ingest
            continue_on_error: Continue processing if individual files fail
            
        Returns:
            Dictionary with batch results
        """
        results = {
            "successful": [],
            "failed": [],
            "total_records": 0,
            "total_files": len(file_paths),
            "errors": {}
        }
        
        for file_path in file_paths:
            try:
                processed_data, validation_result = self.ingest_csv(file_path)
                
                results["successful"].append({
                    "file": str(file_path),
                    "records": len(processed_data),
                    "warnings": validation_result.warnings
                })
                results["total_records"] += len(processed_data)
                
            except FormGuideCsvIngestionError as e:
                results["failed"].append(str(file_path))
                results["errors"][str(file_path)] = str(e)
                
                if not continue_on_error:
                    raise
        
        return results


def create_ingestor(validation_level: str = "moderate") -> FormGuideCsvIngestor:
    """
    Factory function to create a CSV ingestor with specified validation level.
    
    Args:
        validation_level: "strict", "moderate", or "lenient"
        
    Returns:
        FormGuideCsvIngestor instance
    """
    level_map = {
        "strict": ValidationLevel.STRICT,
        "moderate": ValidationLevel.MODERATE,
        "lenient": ValidationLevel.LENIENT
    }
    
    level = level_map.get(validation_level.lower(), ValidationLevel.MODERATE)
    return FormGuideCsvIngestor(validation_level=level)


def save_to_database(processed_data: List[Dict[str, Any]], db_path: str = 'greyhound_racing_data.db'):
    """
    Save the processed CSV data into the database.
    Creates race metadata and dog race data records from form guide data.
    """
    if not processed_data:
        print("No data to save")
        return
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Group records by race (track + date + distance + grade)
        races = {}
        for record in processed_data:
            # Generate race ID from available data
            track = record.get('track', 'unknown').lower()
            date = record.get('date', 'unknown')
            distance = record.get('distance', 'unknown')
            grade = record.get('grade', 'unknown')
            
            race_key = f"{track}_{date}_{distance}_{grade}"
            race_id = race_key.replace(' ', '_').replace('/', '_')
            
            if race_key not in races:
                races[race_key] = {
                    'race_id': race_id,
                    'venue': track.title(),
                    'race_date': date,
                    'distance': distance,
                    'grade': grade,
                    'dogs': []
                }
            
            # Add dog data to race
            races[race_key]['dogs'].append({
                'race_id': race_id,
                'dog_name': record.get('dog_name', ''),
                'dog_clean_name': record.get('dog_name', '').strip() if record.get('dog_name') else '',
                'box_number': record.get('box'),
                'finish_position': record.get('place'),
                'weight': record.get('weight'),
                'starting_price': record.get('starting_price'),
                'individual_time': record.get('time'),
                'sectional_1st': record.get('first_sectional'),
                'margin': record.get('margin'),
                'extraction_timestamp': datetime.now().isoformat(),
                'data_source': 'csv_ingestion'
            })
        
        # Insert race metadata and dog data
        races_saved = 0
        dogs_saved = 0
        
        for race_key, race_data in races.items():
            # Insert race metadata
            cursor.execute("""
                INSERT OR IGNORE INTO race_metadata 
                (race_id, venue, race_date, distance, grade, field_size, extraction_timestamp, data_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_data['race_id'],
                race_data['venue'],
                race_data['race_date'],
                race_data['distance'],
                race_data['grade'],
                len(race_data['dogs']),
                datetime.now().isoformat(),
                'csv_ingestion'
            ))
            
            if cursor.rowcount > 0:
                races_saved += 1
            
            # Insert dog race data
            for dog in race_data['dogs']:
                cursor.execute("""
                    INSERT OR IGNORE INTO dog_race_data 
                    (race_id, dog_name, dog_clean_name, box_number, finish_position, weight, 
                     starting_price, individual_time, sectional_1st, margin, extraction_timestamp, data_source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    dog['race_id'],
                    dog['dog_name'],
                    dog['dog_clean_name'],
                    dog['box_number'],
                    dog['finish_position'],
                    dog['weight'],
                    dog['starting_price'],
                    dog['individual_time'],
                    dog['sectional_1st'],
                    dog['margin'],
                    dog['extraction_timestamp'],
                    dog['data_source']
                ))
                
                if cursor.rowcount > 0:
                    dogs_saved += 1
        
        # Commit changes
        conn.commit()
        print(f"✅ Saved {races_saved} races and {dogs_saved} dog records to database")
        
    except Exception as e:
        print(f"❌ Error saving to database: {e}")
        conn.rollback()
    finally:
        conn.close()


# Enhanced CSV ingestion pipeline
class EnhancedFormGuideCsvIngestor(FormGuideCsvIngestor):
    def ingest_csv(self, file_path: Union[str, Path], skip_validation: bool = False) -> Tuple[List[Dict[str, Any]], ValidationResult]:
        processed_data, validation_result = super().ingest_csv(file_path, skip_validation)
        save_to_database(processed_data)
        return processed_data, validation_result


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create ingestor
    ingestor = create_ingestor("moderate")
    
    if len(sys.argv) > 1:
        # Test with provided file
        test_file = sys.argv[1]
        print(f"Testing CSV ingestion with: {test_file}")
        
        try:
            # Validate first
            validation_result = ingestor.validate_csv_schema(test_file)
            print(f"\n📊 Validation Results:")
            print(f"Valid: {validation_result.is_valid}")
            
            if validation_result.errors:
                print("❌ Errors:")
                for error in validation_result.errors:
                    print(f"  - {error}")
            
            if validation_result.warnings:
                print("⚠️ Warnings:")
                for warning in validation_result.warnings:
                    print(f"  - {warning}")
            
            if validation_result.is_valid:
                # Ingest the file
                processed_data, _ = ingestor.ingest_csv(test_file)
                print(f"\n✅ Successfully processed {len(processed_data)} records")
                
                # Show first few records
                print("\n📋 Sample records:")
                for i, record in enumerate(processed_data[:3]):
                    print(f"Record {i+1}: {record}")
            
        except FormGuideCsvIngestionError as e:
            print(f"❌ Ingestion failed: {e}")
    
    else:
        print("Usage: python csv_ingestion.py <csv_file_path>")
        print("\nThis module provides robust CSV ingestion for greyhound form guide data.")
        print("Features:")
        print("- Maps 'Dog Name' column to 'dog_name'")
        print("- Schema validation with descriptive errors")  
        print("- Handles greyhound form guide format")
        print("- Flexible validation levels (strict/moderate/lenient)")
