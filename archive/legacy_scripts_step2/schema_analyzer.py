#!/usr/bin/env python3

import re
import sqlite3
import pandas as pd
import subprocess

def get_schema_from_db(db_path):
    """Get schema directly from database using sqlite3"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()
        
        schema_info = []
        
        for (table_name,) in tables:
            # Get column info for each table
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns_info = cursor.fetchall()
            
            # Extract column names
            columns = [col[1] for col in columns_info]  # col[1] is the column name
            
            schema_info.append({
                'table_name': table_name,
                'columns': columns,
                'column_count': len(columns)
            })
        
        conn.close()
        return pd.DataFrame(schema_info)
    
    except Exception as e:
        print(f"Error accessing database: {e}")
        return None

def create_column_mapping(schema_df):
    """Create a mapping of all columns across all tables for easier lookup"""
    column_mapping = {}
    
    for _, row in schema_df.iterrows():
        table_name = row['table_name']
        columns = row['columns']
        
        for col in columns:
            if col not in column_mapping:
                column_mapping[col] = []
            column_mapping[col].append(table_name)
    
    return column_mapping

def save_schema_analysis(schema_df, column_mapping, output_file="schema_analysis.txt"):
    """Save detailed schema analysis to file"""
    with open(output_file, 'w') as f:
        f.write("=== DATABASE SCHEMA ANALYSIS ===\n\n")
        
        f.write("TABLES AND COLUMNS:\n")
        f.write("-" * 50 + "\n")
        for _, row in schema_df.iterrows():
            f.write(f"\nTable: {row['table_name']} ({row['column_count']} columns)\n")
            f.write("Columns: " + ", ".join(row['columns']) + "\n")
        
        f.write("\n\nCOLUMN FREQUENCY ANALYSIS:\n")
        f.write("-" * 50 + "\n")
        
        # Sort columns by frequency (how many tables they appear in)
        sorted_columns = sorted(column_mapping.items(), key=lambda x: len(x[1]), reverse=True)
        
        for col, tables in sorted_columns:
            f.write(f"{col}: appears in {len(tables)} table(s) - {', '.join(tables)}\n")
        
        f.write("\n\nCRITICAL TABLES (likely main data tables):\n")
        f.write("-" * 50 + "\n")
        
        # Identify main tables (exclude backup and temp tables)
        main_tables = schema_df[
            ~schema_df['table_name'].str.contains('backup|temp|sqlite_', case=False, na=False)
        ].copy()
        
        main_tables = main_tables.sort_values('column_count', ascending=False)
        
        for _, row in main_tables.iterrows():
            f.write(f"- {row['table_name']}: {row['column_count']} columns\n")

def main():
    db_path = "greyhound_racing_data.db"
    
    print("Analyzing database schema...")
    schema_df = get_schema_from_db(db_path)
    
    if schema_df is not None:
        print(f"Found {len(schema_df)} tables in the database")
        
        # Create column mapping
        column_mapping = create_column_mapping(schema_df)
        
        # Save analysis
        save_schema_analysis(schema_df, column_mapping)
        
        print("\nSchema analysis saved to schema_analysis.txt")
        
        # Display summary
        print("\nSUMMARY:")
        print(f"Total tables: {len(schema_df)}")
        print(f"Total unique columns: {len(column_mapping)}")
        
        # Show main tables (non-backup)
        main_tables = schema_df[
            ~schema_df['table_name'].str.contains('backup|temp|sqlite_', case=False, na=False)
        ]
        
        print(f"Main tables (excluding backups): {len(main_tables)}")
        
        print("\nMain tables:")
        for _, row in main_tables.head(10).iterrows():
            print(f"  - {row['table_name']}: {row['column_count']} columns")
        
        return schema_df, column_mapping
    else:
        print("Failed to analyze database schema")
        return None, None

if __name__ == "__main__":
    schema_df, column_mapping = main()
