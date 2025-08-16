#!/usr/bin/env python3
"""
Race Data Cross-Reference Analysis
Step 3: Cross-reference race data with log files to flag inconsistencies

This script performs left joins on race identifiers and timestamps to detect:
• Missing log entries for races in the CSV
• Log-only races not present in the CSV  
• Conflicting fields (distance, dog name, result times, etc.)
"""

import pandas as pd
import glob
import os
import re
from datetime import datetime
from pathlib import Path

def extract_race_info_from_filename(filename):
    """Extract race information from CSV filename"""
    basename = os.path.basename(filename)
    
    # Common patterns in race filenames
    patterns = [
        r'Race\s*(\d+)\s*-\s*([A-Z_]+)\s*-\s*(.+)\.csv',  # Race 1 - BEN - 02 July 2025.csv
        r'race_(\d+)_([^_]+)_(.+)\.csv',  # race_01_test_venue_2025-08-03.csv
        r'([^/]+)\.csv'  # fallback
    ]
    
    for pattern in patterns:
        match = re.search(pattern, basename, re.IGNORECASE)
        if match:
            if len(match.groups()) >= 3:
                race_num = match.group(1)
                venue = match.group(2)
                date_str = match.group(3)
                return {
                    'race_number': race_num,
                    'venue': venue,
                    'date_info': date_str,
                    'filename': basename,
                    'full_path': filename
                }
    
    return {
        'race_number': 'unknown',
        'venue': 'unknown', 
        'date_info': 'unknown',
        'filename': basename,
        'full_path': filename
    }

def normalize_date(date_str):
    """Normalize various date formats to YYYY-MM-DD"""
    if pd.isna(date_str) or date_str == 'unknown':
        return None
        
    # Common date patterns
    patterns = [
        (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),  # 2025-08-03
        (r'(\d{2})\s+(\w+)\s+(\d{4})', '%d %B %Y'),  # 02 July 2025
        (r'(\d{1,2})\s+(\w+)\s+(\d{4})', '%d %B %Y'),  # 2 July 2025
    ]
    
    for pattern, format_str in patterns:
        match = re.search(pattern, str(date_str))
        if match:
            try:
                if format_str == '%d %B %Y':
                    date_obj = datetime.strptime(match.group(0), format_str)
                else:
                    date_obj = datetime.strptime(match.group(0), format_str)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
    
    return None

def load_and_process_csv_data():
    """Load all CSV race files and extract race information"""
    csv_files = glob.glob('./processed/step6_cleanup/*.csv')
    race_data = []
    
    print(f"Processing {len(csv_files)} CSV files...")
    
    for csv_file in csv_files:
        try:
            # Extract info from filename
            race_info = extract_race_info_from_filename(csv_file)
            
            # Load CSV data
            df = pd.read_csv(csv_file)
            
            # Get unique dates, tracks, distances from the data
            if not df.empty:
                unique_dates = df['DATE'].dropna().unique() if 'DATE' in df.columns else []
                unique_tracks = df['TRACK'].dropna().unique() if 'TRACK' in df.columns else []
                unique_distances = df['DIST'].dropna().unique() if 'DIST' in df.columns else []
                dog_names = df['Dog Name'].dropna().unique() if 'Dog Name' in df.columns else []
                
                race_record = {
                    'source_file': race_info['filename'],
                    'race_number': race_info['race_number'],
                    'venue_from_filename': race_info['venue'],
                    'date_from_filename': normalize_date(race_info['date_info']),
                    'csv_row_count': len(df),
                    'csv_dates': list(unique_dates),
                    'csv_tracks': list(unique_tracks),
                    'csv_distances': list(unique_distances),
                    'csv_dog_count': len(dog_names),
                    'csv_dog_names': list(dog_names[:5]),  # First 5 dog names for verification
                    'data_source': 'csv'
                }
                
                race_data.append(race_record)
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    return pd.DataFrame(race_data)

def load_and_process_log_data():
    """Load and process log data"""
    try:
        logs_df = pd.read_csv('race_logs_normalized.csv')
        
        # Filter for relevant log entries (those with race-related information)
        race_logs = logs_df[
            (logs_df['race_id'].notna()) |
            (logs_df['event'].notna()) |
            (logs_df['details.file'].notna() & logs_df['details.file'].str.contains('.csv', na=False))
        ].copy()
        
        print(f"Found {len(race_logs)} relevant log entries")
        
        # Extract race information from logs
        log_race_data = []
        
        for _, row in race_logs.iterrows():
            log_record = {
                'log_timestamp': row.get('timestamp'),
                'log_race_id': row.get('race_id'),
                'log_event': row.get('event'),
                'log_action': row.get('action'),
                'log_file': row.get('details.file'),
                'log_status': row.get('status'),
                'data_source': 'log'
            }
            log_race_data.append(log_record)
        
        return pd.DataFrame(log_race_data)
        
    except Exception as e:
        print(f"Error loading log data: {e}")
        return pd.DataFrame()

def detect_discrepancies(csv_data, log_data):
    """Detect discrepancies between CSV and log data"""
    discrepancies = {
        'missing_log_entries': [],
        'log_only_races': [],
        'conflicting_fields': [],
        'summary': {}
    }
    
    # 1. Find CSV races missing from logs
    csv_files = set(csv_data['source_file'].unique())
    log_files = set()
    
    if not log_data.empty and 'log_file' in log_data.columns:
        log_files = set(log_data['log_file'].dropna().apply(lambda x: os.path.basename(x)).unique())
    
    missing_in_logs = csv_files - log_files
    for missing_file in missing_in_logs:
        csv_record = csv_data[csv_data['source_file'] == missing_file].iloc[0]
        discrepancies['missing_log_entries'].append({
            'type': 'missing_log_entry',
            'csv_file': missing_file,
            'race_number': csv_record['race_number'],
            'venue': csv_record['venue_from_filename'],
            'date': csv_record['date_from_filename'],
            'dog_count': csv_record['csv_dog_count'],
            'description': f"CSV race file {missing_file} has no corresponding log entries"
        })
    
    # 2. Find log entries for files not in CSV directory
    log_only_files = log_files - csv_files
    for log_file in log_only_files:
        if log_file and log_file != 'nan':
            log_records = log_data[log_data['log_file'].str.contains(log_file, na=False)]
            if not log_records.empty:
                log_record = log_records.iloc[0]
                discrepancies['log_only_races'].append({
                    'type': 'log_only_race',
                    'log_file': log_file,
                    'log_timestamp': log_record['log_timestamp'],
                    'log_action': log_record['log_action'],
                    'log_status': log_record['log_status'],
                    'description': f"Log references file {log_file} not found in CSV directory"
                })
    
    # 3. Check for potential conflicts in matched files
    for csv_file in csv_files.intersection(log_files):
        csv_record = csv_data[csv_data['source_file'] == csv_file].iloc[0]
        matching_logs = log_data[log_data['log_file'].str.contains(csv_file, na=False)] if not log_data.empty else pd.DataFrame()
        
        if not matching_logs.empty:
            for _, log_record in matching_logs.iterrows():
                # Look for potential conflicts
                conflict_found = False
                conflict_details = []
                
                # Check timestamps if available
                if log_record['log_timestamp'] and csv_record['date_from_filename']:
                    log_date = pd.to_datetime(log_record['log_timestamp']).date()
                    csv_date = pd.to_datetime(csv_record['date_from_filename']).date()
                    if log_date != csv_date:
                        conflict_found = True
                        conflict_details.append(f"Date mismatch: CSV={csv_date}, Log={log_date}")
                
                if conflict_found:
                    discrepancies['conflicting_fields'].append({
                        'type': 'field_conflict',
                        'file': csv_file,
                        'conflicts': conflict_details,
                        'csv_data': csv_record.to_dict(),
                        'log_data': log_record.to_dict(),
                        'description': f"Field conflicts found in {csv_file}: {'; '.join(conflict_details)}"
                    })
    
    # Generate summary
    discrepancies['summary'] = {
        'total_csv_files': len(csv_files),
        'total_log_entries': len(log_data) if not log_data.empty else 0,
        'missing_log_entries_count': len(discrepancies['missing_log_entries']),
        'log_only_races_count': len(discrepancies['log_only_races']),
        'conflicting_fields_count': len(discrepancies['conflicting_fields']),
        'files_with_logs': len(csv_files.intersection(log_files)),
        'files_without_logs': len(missing_in_logs)
    }
    
    return discrepancies

def save_discrepancies_report(discrepancies):
    """Save discrepancies to CSV and JSON files"""
    
    # Create a comprehensive DataFrame for all discrepancies
    all_discrepancies = []
    
    # Add missing log entries
    for item in discrepancies['missing_log_entries']:
        item['discrepancy_category'] = 'missing_log_entry'
        all_discrepancies.append(item)
    
    # Add log-only races
    for item in discrepancies['log_only_races']:
        item['discrepancy_category'] = 'log_only_race'
        all_discrepancies.append(item)
    
    # Add conflicting fields
    for item in discrepancies['conflicting_fields']:
        item['discrepancy_category'] = 'field_conflict'
        all_discrepancies.append(item)
    
    # Save to CSV
    if all_discrepancies:
        discrepancies_df = pd.DataFrame(all_discrepancies)
        discrepancies_df.to_csv('race_data_discrepancies.csv', index=False)
        print(f"Saved {len(all_discrepancies)} discrepancies to race_data_discrepancies.csv")
    else:
        # Create empty DataFrame with expected columns
        discrepancies_df = pd.DataFrame(columns=['discrepancy_category', 'description', 'type'])
        discrepancies_df.to_csv('race_data_discrepancies.csv', index=False)
        print("No discrepancies found - saved empty report to race_data_discrepancies.csv")
    
    # Save summary to separate file
    summary_df = pd.DataFrame([discrepancies['summary']])
    summary_df.to_csv('race_data_discrepancies_summary.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("RACE DATA CROSS-REFERENCE ANALYSIS SUMMARY")
    print("="*60)
    for key, value in discrepancies['summary'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print(f"\nDiscrepancy Details:")
    print(f"• Missing log entries: {len(discrepancies['missing_log_entries'])}")
    print(f"• Log-only races: {len(discrepancies['log_only_races'])}")
    print(f"• Field conflicts: {len(discrepancies['conflicting_fields'])}")
    
    return discrepancies_df

def main():
    """Main execution function"""
    print("Starting Race Data Cross-Reference Analysis...")
    print("="*60)
    
    # Load and process CSV data
    print("\n1. Loading CSV race data...")
    csv_data = load_and_process_csv_data()
    print(f"Processed {len(csv_data)} CSV race files")
    
    # Load and process log data
    print("\n2. Loading log data...")
    log_data = load_and_process_log_data()
    print(f"Processed log data with {len(log_data)} entries")
    
    # Detect discrepancies
    print("\n3. Detecting discrepancies...")
    discrepancies = detect_discrepancies(csv_data, log_data)
    
    # Save results
    print("\n4. Saving discrepancy report...")
    discrepancies_df = save_discrepancies_report(discrepancies)
    
    print("\n5. Analysis complete!")
    print("Files generated:")
    print("• race_data_discrepancies.csv - Detailed discrepancy report")
    print("• race_data_discrepancies_summary.csv - Summary statistics")
    
    return discrepancies_df

if __name__ == "__main__":
    discrepancies_df = main()
