#!/usr/bin/env python3
"""
Database Schema Introspection Script
Analyzes all tables in greyhound_racing_data.db and their foreign key relationships
"""

import sqlite3
import json
from datetime import datetime

def introspect_database(db_path):
    """Introspect the database to get complete schema information"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    schema_info = {
        'database_path': db_path,
        'introspection_timestamp': datetime.now().isoformat(),
        'total_tables': len(tables),
        'tables': {}
    }
    
    for table_name in tables:
        print(f"Introspecting table: {table_name}")
        
        table_info = {
            'name': table_name,
            'columns': [],
            'foreign_keys': [],
            'indexes': []
        }
        
        # Get table info (columns)
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for col in columns:
            column_info = {
                'cid': col[0],
                'name': col[1],
                'type': col[2],
                'notnull': bool(col[3]),
                'default_value': col[4],
                'pk': bool(col[5])
            }
            table_info['columns'].append(column_info)
        
        # Get foreign keys
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        foreign_keys = cursor.fetchall()
        
        for fk in foreign_keys:
            fk_info = {
                'id': fk[0],
                'seq': fk[1],
                'table': fk[2],
                'from': fk[3],
                'to': fk[4],
                'on_update': fk[5],
                'on_delete': fk[6],
                'match': fk[7]
            }
            table_info['foreign_keys'].append(fk_info)
        
        # Get indexes
        cursor.execute(f"PRAGMA index_list({table_name})")
        indexes = cursor.fetchall()
        
        for idx in indexes:
            # Get index info
            cursor.execute(f"PRAGMA index_info({idx[1]})")
            index_columns = cursor.fetchall()
            
            index_info = {
                'seq': idx[0],
                'name': idx[1],
                'unique': bool(idx[2]),
                'origin': idx[3],
                'partial': bool(idx[4]),
                'columns': [{'seqno': col[0], 'cid': col[1], 'name': col[2]} for col in index_columns]
            }
            table_info['indexes'].append(index_info)
        
        schema_info['tables'][table_name] = table_info
    
    conn.close()
    return schema_info

def save_schema_info(schema_info, output_file):
    """Save schema information to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(schema_info, f, indent=2)
    print(f"Schema information saved to: {output_file}")

if __name__ == "__main__":
    db_path = "greyhound_racing_data.db"
    output_file = "database_schema_info.json"
    
    print(f"Starting database introspection of: {db_path}")
    schema_info = introspect_database(db_path)
    save_schema_info(schema_info, output_file)
    
    print(f"\nIntrospection Summary:")
    print(f"Total tables: {schema_info['total_tables']}")
    
    # Count tables with foreign keys
    tables_with_fks = sum(1 for table in schema_info['tables'].values() if table['foreign_keys'])
    print(f"Tables with foreign keys: {tables_with_fks}")
    
    # List tables with foreign keys
    if tables_with_fks > 0:
        print("\nTables with foreign key constraints:")
        for table_name, table_info in schema_info['tables'].items():
            if table_info['foreign_keys']:
                print(f"  - {table_name}: {len(table_info['foreign_keys'])} FK(s)")
                for fk in table_info['foreign_keys']:
                    print(f"    * {fk['from']} -> {fk['table']}.{fk['to']}")
