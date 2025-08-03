#!/usr/bin/env python3
"""
Enhanced Schema Conformance Checker
Validates data against database schema constraints with comprehensive checks
"""

import pandas as pd
import sqlite3
from datetime import datetime
import numpy as np

def connect_db():
    """Connect to the race data database"""
    return sqlite3.connect('/Users/orlandolee/greyhound_racing_collector/databases/race_data.db')

def load_sample_data(conn, table_name, limit=5):
    """Load random sample data from specified table"""
    query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT {limit};"
    return pd.read_sql(query, conn)

def check_not_null_constraints(df, required_columns):
    """Check NOT NULL constraint violations"""
    violations = {}
    for col in required_columns:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            violations[col] = null_count
    return violations

def check_numeric_ranges(df, range_checks):
    """Check numeric range constraint violations"""
    violations = {}
    for col, (min_val, max_val) in range_checks.items():
        if col in df.columns and not df[col].isnull().all():
            # Filter out null values for range checking
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                out_of_range = ~valid_data.between(min_val, max_val)
                violations[col] = out_of_range.sum()
            else:
                violations[col] = 0
    return violations

def check_categorical_values(df, categorical_checks):
    """Check categorical constraint violations"""
    violations = {}
    for col, valid_values in categorical_checks.items():
        if col in df.columns and not df[col].isnull().all():
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                invalid_values = ~valid_data.isin(valid_values)
                violations[col] = invalid_values.sum()
            else:
                violations[col] = 0
    return violations

def analyze_races_table(conn):
    """Analyze races table for schema conformance"""
    print("Analyzing races table...")
    df = load_sample_data(conn, 'races', 10)
    
    results = {
        'table': 'races',
        'total_rows': len(df),
        'sample_data': df.head(3).to_dict('records')
    }
    
    # NOT NULL checks
    required_cols = ['race_name', 'venue', 'race_date']
    results['not_null_violations'] = check_not_null_constraints(df, required_cols)
    
    # Range checks
    range_checks = {
        'distance': (300, 800)  # Race distances 300m to 800m
    }
    results['range_violations'] = check_numeric_ranges(df, range_checks)
    
    return results

def analyze_dog_performances_table(conn):
    """Analyze dog_performances table for schema conformance"""
    print("Analyzing dog_performances table...")
    df = load_sample_data(conn, 'dog_performances', 10)
    
    results = {
        'table': 'dog_performances',
        'total_rows': len(df),
        'sample_data': df.head(3).to_dict('records')
    }
    
    # NOT NULL checks
    required_cols = ['race_id', 'dog_name']
    results['not_null_violations'] = check_not_null_constraints(df, required_cols)
    
    # Range checks
    range_checks = {
        'box_number': (1, 8),      # Trap numbers 1-8
        'weight': (20, 40),        # Weights 20-40 kg
        'finish_position': (1, 8), # Finishing positions 1-8
        'race_time': (0, 120)      # Race times 0-120 seconds (reasonable range)
    }
    results['range_violations'] = check_numeric_ranges(df, range_checks)
    
    return results

def analyze_dogs_table(conn):
    """Analyze dogs table for schema conformance"""
    print("Analyzing dogs table...")
    df = load_sample_data(conn, 'dogs', 10)
    
    results = {
        'table': 'dogs',
        'total_rows': len(df),
        'sample_data': df.head(3).to_dict('records')
    }
    
    # NOT NULL checks
    required_cols = ['dog_name']
    results['not_null_violations'] = check_not_null_constraints(df, required_cols)
    
    # Range checks - logical constraints
    range_checks = {
        'total_races': (0, 10000),     # Reasonable race count
        'total_wins': (0, 10000),      # Wins should be <= total_races
        'total_places': (0, 10000),    # Places should be <= total_races
        'best_time': (0, 120),         # Best time in reasonable range
        'average_position': (1, 8)     # Average position 1-8
    }
    results['range_violations'] = check_numeric_ranges(df, range_checks)
    
    # Business logic checks
    business_violations = {}
    if 'total_wins' in df.columns and 'total_races' in df.columns:
        wins_exceed_races = (df['total_wins'] > df['total_races']).sum()
        business_violations['wins_exceed_races'] = wins_exceed_races
    
    if 'total_places' in df.columns and 'total_races' in df.columns:
        places_exceed_races = (df['total_places'] > df['total_races']).sum()
        business_violations['places_exceed_races'] = places_exceed_races
    
    results['business_logic_violations'] = business_violations
    
    return results

def generate_comprehensive_report(all_results):
    """Generate comprehensive schema conformance report"""
    report_path = '/Users/orlandolee/greyhound_racing_collector/audit/schema_conformance_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Schema & Data-Type Conformance Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Executive Summary\n\n")
        
        total_violations = 0
        for result in all_results:
            not_null_total = sum(result.get('not_null_violations', {}).values())
            range_total = sum(result.get('range_violations', {}).values())
            business_total = sum(result.get('business_logic_violations', {}).values())
            total_violations += not_null_total + range_total + business_total
        
        f.write(f"- **Total Tables Analyzed:** {len(all_results)}\n")
        f.write(f"- **Total Violation Count:** {total_violations}\n")
        f.write(f"- **Data Quality Status:** {'PASS' if total_violations == 0 else 'FAIL'}\n\n")
        
        f.write("## Detailed Analysis\n\n")
        
        for result in all_results:
            f.write(f"### {result['table'].upper()} Table\n\n")
            f.write(f"**Sample Size:** {result['total_rows']} rows\n\n")
            
            # NOT NULL Violations
            f.write("#### NOT NULL Constraint Violations\n\n")
            not_null_violations = result.get('not_null_violations', {})
            if any(not_null_violations.values()):
                f.write("| Column | Violation Count |\n")
                f.write("|--------|----------------|\n")
                for col, count in not_null_violations.items():
                    f.write(f"| {col} | {count} |\n")
            else:
                f.write("✅ No NOT NULL violations found\n")
            f.write("\n")
            
            # Range Violations
            f.write("#### Numeric Range Violations\n\n")
            range_violations = result.get('range_violations', {})
            if any(range_violations.values()):
                f.write("| Column | Violation Count |\n")
                f.write("|--------|----------------|\n")
                for col, count in range_violations.items():
                    f.write(f"| {col} | {count} |\n")
            else:
                f.write("✅ No range violations found\n")
            f.write("\n")
            
            # Business Logic Violations
            if 'business_logic_violations' in result:
                f.write("#### Business Logic Violations\n\n")
                business_violations = result['business_logic_violations']
                if any(business_violations.values()):
                    f.write("| Rule | Violation Count |\n")
                    f.write("|------|----------------|\n")
                    for rule, count in business_violations.items():
                        f.write(f"| {rule} | {count} |\n")
                else:
                    f.write("✅ No business logic violations found\n")
                f.write("\n")
            
            # Sample Data
            f.write("#### Sample Data\n\n")
            f.write("```json\n")
            f.write(str(result['sample_data']))
            f.write("\n```\n\n")
        
        f.write("## Recommendations\n\n")
        if total_violations > 0:
            f.write("- Review and clean data with violations\n")
            f.write("- Implement data validation at ingestion\n")
            f.write("- Add database constraints to prevent future violations\n")
        else:
            f.write("- Data quality is excellent\n")
            f.write("- Continue monitoring with regular audits\n")
        
        f.write("\n---\n")
        f.write("*Report generated by Enhanced Schema Conformance Checker*\n")

def main():
    """Main execution function"""
    print("Starting Enhanced Schema Conformance Check...")
    
    conn = connect_db()
    all_results = []
    
    try:
        # Analyze each table
        all_results.append(analyze_races_table(conn))
        all_results.append(analyze_dog_performances_table(conn))
        all_results.append(analyze_dogs_table(conn))
        
        # Generate comprehensive report
        generate_comprehensive_report(all_results)
        
        print("✅ Schema conformance check completed successfully!")
        print("📊 Report generated: audit/schema_conformance_report.md")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
