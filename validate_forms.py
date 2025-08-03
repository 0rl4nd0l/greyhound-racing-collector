import os
import multiprocessing
import pandas as pd
import json
import shutil
import csv
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import chardet
import numpy as np
from collections import defaultdict

# Define directories to search for CSVs (excluding system directories)
CSV_DIRECTORIES = ["unprocessed", "processed", "form_guides", "data", "samples"]
AUDIT_DIR = "audit"
OUTPUT_SUMMARY = os.path.join(AUDIT_DIR, "validation_summary.parquet")
QUARANTINE_DIR = os.path.join(AUDIT_DIR, "quarantine")
ERROR_TYPES_FILE = os.path.join(AUDIT_DIR, "error_types_summary.json")

# Ensure audit directories exist
os.makedirs(QUARANTINE_DIR, exist_ok=True)

# Expected columns for form guide CSVs
EXPECTED_COLUMNS = ['Dog Name', 'Sex', 'PLC', 'BOX', 'WGT', 'DIST', 'DATE', 'TRACK', 'G', 'TIME', 'WIN', 'BON', '1 SEC', 'MGN', 'W/2G', 'PIR', 'SP']

def detect_delimiter(file_path):
    """Detect the delimiter used in the CSV file"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(1024)
        sniffer = csv.Sniffer()
        try:
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter
        except:
            return ','

def validate_csv(file_path):
    """Comprehensive CSV validation for form guide files"""
    error_types = []
    warnings = 0
    quarantined_rows = 0
    success = True
    
    try:
        # Detect file encoding
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
        
        # Detect delimiter
        delimiter = detect_delimiter(file_path)
        
        # Try reading the CSV
        df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter, on_bad_lines='skip')
        
        if df.empty:
            error_types.append('empty_file')
            success = False
            warnings += 1
        
        row_count = len(df)
        
        # Check for missing columns
        missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing_columns:
            error_types.append('missing_columns')
            warnings += 1
            if len(missing_columns) > len(EXPECTED_COLUMNS) // 2:  # More than half missing
                success = False
        
        # Check for extra unexpected columns
        extra_columns = set(df.columns) - set(EXPECTED_COLUMNS)
        if extra_columns:
            error_types.append('extra_columns')
            warnings += 1
        
        # Data quality checks
        for index, row in df.iterrows():
            row_issues = 0
            
            # Check for too many null values in a row
            null_count = row.isnull().sum()
            if null_count > len(row) * 0.5:  # More than 50% null
                quarantined_rows += 1
                row_issues += 1
                if 'high_null_percentage' not in error_types:
                    error_types.append('high_null_percentage')
            
            # Check specific column validations if they exist
            if 'BOX' in df.columns and pd.notnull(row.get('BOX')):
                try:
                    box_num = float(row['BOX'])
                    if box_num < 1 or box_num > 8:  # Typical box numbers
                        row_issues += 1
                        if 'invalid_box_number' not in error_types:
                            error_types.append('invalid_box_number')
                except (ValueError, TypeError):
                    row_issues += 1
                    if 'invalid_box_format' not in error_types:
                        error_types.append('invalid_box_format')
            
            # Check weight format
            if 'WGT' in df.columns and pd.notnull(row.get('WGT')):
                try:
                    weight = float(row['WGT'])
                    if weight < 20 or weight > 40:  # Typical greyhound weights
                        row_issues += 1
                        if 'invalid_weight' not in error_types:
                            error_types.append('invalid_weight')
                except (ValueError, TypeError):
                    row_issues += 1
                    if 'invalid_weight_format' not in error_types:
                        error_types.append('invalid_weight_format')
            
            # Check for outcome data leakage (PLC should be present for historical data)
            if 'PLC' in df.columns and pd.isnull(row.get('PLC')):
                row_issues += 1
                if 'missing_outcome_data' not in error_types:
                    error_types.append('missing_outcome_data')
            
            if row_issues > 0:
                warnings += 1
        
        if quarantined_rows > row_count * 0.3:  # More than 30% quarantined
            success = False
            
        validation_result = {
            "file_path": file_path,
            "success": success,
            "warnings": warnings,
            "rows_total": row_count,
            "rows_quarantined": quarantined_rows,
            "pct_rows_quarantined": quarantined_rows / row_count if row_count > 0 else 0,
            "delimiter": delimiter,
            "encoding": encoding,
            "error_types": error_types,
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }
        
        return validation_result
    
    except Exception as e:
        error_types.append('read_error')
        print(f"Error processing {file_path}: {e}")
        return {
            "file_path": file_path,
            "success": False,
            "warnings": 1,
            "rows_total": 0,
            "rows_quarantined": 0,
            "pct_rows_quarantined": 0,
            "delimiter": 'unknown',
            "encoding": 'unknown',
            "error_types": error_types,
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else 0
        }

if __name__ == "__main__":
    print("Starting batch validation of form guide CSVs...")
    
    # Find relevant CSV files (form guide specific)
    csv_files = []
    
    # Search specific directories for form guide files
    for directory in CSV_DIRECTORIES:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.csv'):
                        csv_files.append(os.path.join(root, file))
    
    # Find additional race-related CSVs in current directory
    for file in os.listdir('.'):
        if file.endswith('.csv') and any(keyword in file.lower() for keyword in 
                                       ['race', 'form', 'guide', 'greyhound', 'track']):
            csv_files.append(file)
    
    # Remove duplicates and filter out system files
    csv_files = list(set(csv_files))
    csv_files = [f for f in csv_files if not any(skip_dir in f for skip_dir in 
                                                ['node_modules', 'venv', '.git', '__pycache__', 
                                                 'site-packages', 'coverage', 'test-results'])]
    
    print(f"Found {len(csv_files)} relevant CSV files to validate")
    
    # Process files in parallel
    print("Processing files with multiprocessing...")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Limit the number of files to a subset for faster processing
        results = list(tqdm(executor.map(validate_csv, csv_files), total=len(csv_files)))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    print(f"\nValidation Results:")
    print(f"Total files processed: {len(results)}")
    
    # Filter successful validations
    successful_validations = [result for result in results if result['success']]
    failures = [result for result in results if not result['success']]
    
    print(f"Successful validations: {len(successful_validations)}")
    print(f"Failed validations: {len(failures)}")
    
    # Analyze error types for heat map
    error_type_counts = defaultdict(int)
    for result in results:
        for error_type in result.get('error_types', []):
            error_type_counts[error_type] += 1
    
    # Save error types summary
    with open(ERROR_TYPES_FILE, 'w') as f:
        json.dump({k: int(v) for k, v in error_type_counts.items()}, f, indent=2)
    
    print(f"\nMost common error types:")
    for error_type, count in sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {error_type}: {count} files")
    
    # Identify top-N offending files (highest quarantine rate or warnings)
    if failures:
        top_offenders = sorted(failures, key=lambda x: (x['pct_rows_quarantined'], x['warnings']), reverse=True)[:20]
        
        print(f"\nQuarantining top {len(top_offenders)} offending files...")
        
        # Copy (don't move) offending files to quarantine
        for i, offender in enumerate(top_offenders):
            file_name = os.path.basename(offender['file_path'])
            quarantine_path = os.path.join(QUARANTINE_DIR, f"top{i+1:02d}_{file_name}")
            try:
                if os.path.exists(offender['file_path']):
                    shutil.copy2(offender['file_path'], quarantine_path)
                    print(f"  Copied {file_name} (warnings: {offender['warnings']}, quarantined: {offender['pct_rows_quarantined']:.1%})")
            except Exception as e:
                print(f"  Failed to copy {file_name}: {e}")
    
    # Create summary DataFrame and save
    df_summary = pd.DataFrame(results)
    df_summary.to_parquet(OUTPUT_SUMMARY, index=False)
    
    # Generate additional summary statistics
    summary_stats = {
        'total_files': len(results),
        'successful_files': len(successful_validations),
        'failed_files': len(failures),
        'success_rate': len(successful_validations) / len(results) if results else 0,
        'average_warnings_per_file': df_summary['warnings'].mean(),
        'average_quarantine_rate': df_summary['pct_rows_quarantined'].mean(),
        'total_rows_processed': df_summary['rows_total'].sum(),
        'total_rows_quarantined': df_summary['rows_quarantined'].sum(),
        'most_common_delimiter': df_summary['delimiter'].mode().iloc[0] if not df_summary.empty else 'unknown',
        'error_type_distribution': dict(error_type_counts)
    }
    
    # Save summary statistics
    with open(os.path.join(AUDIT_DIR, 'validation_statistics.json'), 'w') as f:
        # Ensure all values are convertible to native Python types
        json_serializable_summary = {k: (float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v) for k, v in summary_stats.items()}
        json.dump(json_serializable_summary, f, indent=2)
    
    print(f"\nValidation Summary:")
    print(f"  Success rate: {summary_stats['success_rate']:.1%}")
    print(f"  Average warnings per file: {summary_stats['average_warnings_per_file']:.1f}")
    print(f"  Average quarantine rate: {summary_stats['average_quarantine_rate']:.1%}")
    print(f"  Total rows processed: {summary_stats['total_rows_processed']:,}")
    print(f"  Total rows quarantined: {summary_stats['total_rows_quarantined']:,}")
    print(f"  Most common delimiter: '{summary_stats['most_common_delimiter']}'")
    
    print(f"\nResults saved to:")
    print(f"  - Summary: {OUTPUT_SUMMARY}")
    print(f"  - Error types: {ERROR_TYPES_FILE}")
    print(f"  - Statistics: {os.path.join(AUDIT_DIR, 'validation_statistics.json')}")
    print(f"  - Quarantined files: {QUARANTINE_DIR}/")
