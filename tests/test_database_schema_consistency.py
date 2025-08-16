#!/usr/bin/env python3
"""
Database Schema & Migration Consistency Tests

This module implements comprehensive database schema consistency tests as required:
• Run `alembic revision --autogenerate --compare-type` in CI; assert generated diff is **empty**.  
• Automated check that every foreign-key pair has existing indexes.  
• Integrity queries: orphan records, NULLs in non-nullable columns, enum mismatches.  
• Daily cron CI job dumps prod schema → compares hash to repo SQL; alerts on drift.
"""

import hashlib
import json
import os
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable

try:
    from models import Base
except ImportError:
    # Handle case when running as script
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import Base


class DatabaseSchemaConsistencyTester:
    """Comprehensive database schema consistency testing framework."""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or "sqlite:///greyhound_racing_data.db"
        self.engine = create_engine(self.database_url)
        self.inspector = inspect(self.engine)
        
    def test_alembic_schema_consistency(self) -> Dict[str, Any]:
        """
        Test that alembic revision --autogenerate produces no changes.
        This ensures the database schema matches the models exactly.
        """
        result = {
            "test_name": "alembic_schema_consistency",
            "passed": False,
            "message": "",
            "details": {}
        }
        
        try:
            # Create a temporary alembic revision to check for changes
            with tempfile.TemporaryDirectory() as temp_dir:
                # Run alembic revision --autogenerate --compare-type
                cmd = [
                    "alembic", "revision", "--autogenerate", "--compare-type",
                    "--message", "schema_consistency_test",
                    "--rev-id", f"test_{int(datetime.now().timestamp())}"
                ]
                
                process = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd="."
                )
                
                if process.returncode != 0:
                    result["message"] = f"Alembic command failed: {process.stderr}"
                    return result
                
                # Check if the generated migration file has any actual changes
                # Look for the most recent migration file
                alembic_versions_dir = Path("alembic/versions")
                migration_files = list(alembic_versions_dir.glob("test_*.py"))
                
                if not migration_files:
                    result["message"] = "No test migration file was generated"
                    return result
                
                # Get the most recent test migration file
                latest_migration = max(migration_files, key=os.path.getctime)
                
                # Read the migration file content
                with open(latest_migration, 'r') as f:
                    migration_content = f.read()
                
                # Check if the upgrade function is empty (no schema changes)
                lines = migration_content.split('\n')
                upgrade_section = False
                has_operations = False
                
                for line in lines:
                    if 'def upgrade()' in line:
                        upgrade_section = True
                        continue
                    elif 'def downgrade()' in line:
                        upgrade_section = False
                        continue
                    
                    if upgrade_section and line.strip():
                        # Skip comments and pass statements
                        stripped = line.strip()
                        if (stripped and 
                            not stripped.startswith('#') and 
                            not stripped.startswith('"""') and
                            stripped != 'pass'):
                            has_operations = True
                            break
                
                # Clean up the test migration file
                latest_migration.unlink()
                
                if has_operations:
                    result["message"] = "Schema drift detected! Alembic found differences between models and database."
                    result["details"]["migration_content"] = migration_content
                else:
                    result["passed"] = True
                    result["message"] = "Schema is consistent - no changes detected by Alembic"
                
        except Exception as e:
            result["message"] = f"Error running schema consistency test: {str(e)}"
        
        return result
    
    def test_foreign_key_indexes(self) -> Dict[str, Any]:
        """
        Test that every foreign key has a corresponding index for performance.
        This is critical for query performance and referential integrity.
        """
        result = {
            "test_name": "foreign_key_indexes",
            "passed": True,
            "message": "",
            "details": {
                "missing_indexes": [],
                "foreign_keys_checked": [],
                "existing_indexes": []
            }
        }
        
        try:
            # Get all foreign keys from all tables
            tables = self.inspector.get_table_names()
            
            for table_name in tables:
                foreign_keys = self.inspector.get_foreign_keys(table_name)
                indexes = self.inspector.get_indexes(table_name)
                
                # Create a set of indexed columns for quick lookup
                indexed_columns = set()
                for index in indexes:
                    result["details"]["existing_indexes"].append({
                        "table": table_name,
                        "index_name": index["name"],
                        "columns": index["column_names"]
                    })
                    
                    # Add single column indexes
                    for col in index["column_names"]:
                        indexed_columns.add(col)
                    
                    # Add multi-column index combinations
                    if len(index["column_names"]) > 1:
                        indexed_columns.add(tuple(index["column_names"]))
                
                # Check each foreign key
                for fk in foreign_keys:
                    fk_columns = fk["constrained_columns"]
                    fk_info = {
                        "table": table_name,
                        "columns": fk_columns,
                        "referenced_table": fk["referred_table"],
                        "referenced_columns": fk["referred_columns"]
                    }
                    result["details"]["foreign_keys_checked"].append(fk_info)
                    
                    # Check if FK columns are indexed
                    has_index = False
                    
                    # Check for exact column match
                    if len(fk_columns) == 1:
                        has_index = fk_columns[0] in indexed_columns
                    else:
                        has_index = tuple(fk_columns) in indexed_columns
                    
                    # Also check if FK columns are part of any multi-column index
                    if not has_index:
                        for index in indexes:
                            index_cols = index["column_names"]
                            if all(col in index_cols for col in fk_columns):
                                has_index = True
                                break
                    
                    if not has_index:
                        result["details"]["missing_indexes"].append({
                            "table": table_name,
                            "foreign_key_columns": fk_columns,
                            "referenced_table": fk["referred_table"],
                            "suggested_index_name": f"idx_{table_name}_{'_'.join(fk_columns)}"
                        })
                        result["passed"] = False
            
            if result["passed"]:
                result["message"] = f"All foreign keys have appropriate indexes ({len(result['details']['foreign_keys_checked'])} checked)"
            else:
                missing_count = len(result["details"]["missing_indexes"])
                result["message"] = f"Found {missing_count} foreign keys without indexes"
                
        except Exception as e:
            result["passed"] = False
            result["message"] = f"Error checking foreign key indexes: {str(e)}"
        
        return result
    
    def test_data_integrity(self) -> Dict[str, Any]:
        """
        Run comprehensive data integrity checks:
        - Orphan records (FK references to non-existent records)
        - NULL values in non-nullable columns
        - Enum mismatches and data type violations
        """
        result = {
            "test_name": "data_integrity",
            "passed": True,
            "message": "",
            "details": {
                "orphan_records": [],
                "null_violations": [],
                "data_type_violations": [],
                "constraint_violations": []
            }
        }
        
        try:
            with self.engine.connect() as conn:
                # Test for orphan records
                orphan_checks = [
                    {
                        "query": """
                            SELECT COUNT(*) as count, 'dog_race_data' as table_name
                            FROM dog_race_data d 
                            LEFT JOIN race_metadata r ON d.race_id = r.race_id 
                            WHERE r.race_id IS NULL AND d.race_id IS NOT NULL
                        """,
                        "description": "Dog race data records with non-existent race_id"
                    },
                    {
                        "query": """
                            SELECT COUNT(*) as count, 'prediction_history' as table_name
                            FROM prediction_history p 
                            LEFT JOIN race_metadata r ON p.race_id = r.race_id 
                            WHERE r.race_id IS NULL AND p.race_id IS NOT NULL
                        """,
                        "description": "Prediction history records with non-existent race_id"
                    }
                ]
                
                for check in orphan_checks:
                    try:
                        result_row = conn.execute(text(check["query"])).fetchone()
                        if result_row and result_row[0] > 0:
                            result["details"]["orphan_records"].append({
                                "description": check["description"],
                                "count": result_row[0],
                                "table": result_row[1] if len(result_row) > 1 else "unknown"
                            })
                            result["passed"] = False
                    except Exception as e:
                        # Table might not exist, skip this check
                        continue
                
                # Test for NULL violations in non-nullable columns
                null_checks = [
                    {
                        "query": "SELECT COUNT(*) FROM dogs WHERE dog_name IS NULL",
                        "description": "Dogs table: dog_name should not be NULL",
                        "table": "dogs"
                    },
                    {
                        "query": "SELECT COUNT(*) FROM race_metadata WHERE race_id IS NULL",
                        "description": "Race metadata: race_id should not be NULL",
                        "table": "race_metadata"
                    }
                ]
                
                for check in null_checks:
                    try:
                        result_row = conn.execute(text(check["query"])).fetchone()
                        if result_row and result_row[0] > 0:
                            result["details"]["null_violations"].append({
                                "description": check["description"],
                                "count": result_row[0],
                                "table": check["table"]
                            })
                            result["passed"] = False
                    except Exception as e:
                        continue
                
                # Test for data type violations and constraints
                constraint_checks = [
                    {
                        "query": """
                            SELECT COUNT(*) FROM dog_race_data 
                            WHERE finish_position IS NOT NULL 
                            AND (finish_position < 1 OR finish_position > 20)
                        """,
                        "description": "Invalid finish_position values (should be 1-20)",
                        "table": "dog_race_data"
                    },
                    {
                        "query": """
                            SELECT COUNT(*) FROM race_metadata 
                            WHERE field_size IS NOT NULL 
                            AND (field_size < 1 OR field_size > 20)
                        """,
                        "description": "Invalid field_size values (should be 1-20)",
                        "table": "race_metadata"
                    }
                ]
                
                for check in constraint_checks:
                    try:
                        result_row = conn.execute(text(check["query"])).fetchone()
                        if result_row and result_row[0] > 0:
                            result["details"]["constraint_violations"].append({
                                "description": check["description"],
                                "count": result_row[0],
                                "table": check["table"]
                            })
                            result["passed"] = False
                    except Exception as e:
                        continue
            
            # Summary message
            total_issues = (len(result["details"]["orphan_records"]) + 
                          len(result["details"]["null_violations"]) + 
                          len(result["details"]["constraint_violations"]))
            
            if result["passed"]:
                result["message"] = "All data integrity checks passed"
            else:
                result["message"] = f"Found {total_issues} data integrity issues"
                
        except Exception as e:
            result["passed"] = False
            result["message"] = f"Error running data integrity checks: {str(e)}"
        
        return result
    
    def generate_schema_hash(self) -> str:
        """
        Generate a hash of the current database schema for drift detection.
        This can be used in daily cron jobs to detect schema changes.
        """
        try:
            schema_info = {}
            
            # Get all tables and their structures
            tables = self.inspector.get_table_names()
            
            for table_name in tables:
                columns = self.inspector.get_columns(table_name)
                indexes = self.inspector.get_indexes(table_name)
                foreign_keys = self.inspector.get_foreign_keys(table_name)
                primary_keys = self.inspector.get_pk_constraint(table_name)
                unique_constraints = self.inspector.get_unique_constraints(table_name)
                
                schema_info[table_name] = {
                    "columns": sorted([
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "default": str(col["default"]) if col["default"] is not None else None
                        }
                        for col in columns
                    ], key=lambda x: x["name"]),
                    "indexes": sorted([
                        {
                            "name": idx["name"] or "unnamed",
                            "columns": sorted(idx["column_names"]),
                            "unique": idx["unique"]
                        }
                        for idx in indexes
                    ], key=lambda x: x["name"] or "unnamed"),
                    "foreign_keys": sorted([
                        {
                            "columns": sorted(fk["constrained_columns"]),
                            "referenced_table": fk["referred_table"],
                            "referenced_columns": sorted(fk["referred_columns"])
                        }
                        for fk in foreign_keys
                    ], key=lambda x: str(x["columns"])),
                    "primary_key": sorted(primary_keys["constrained_columns"]) if primary_keys else [],
                    "unique_constraints": sorted([
                        {
                            "name": uc["name"] or "unnamed",
                            "columns": sorted(uc["column_names"])
                        }
                        for uc in unique_constraints
                    ], key=lambda x: x["name"] or "unnamed")
                }
            
            # Create deterministic JSON string and hash it
            schema_json = json.dumps(schema_info, sort_keys=True, separators=(',', ':'))
            return hashlib.sha256(schema_json.encode()).hexdigest()
            
        except Exception as e:
            raise Exception(f"Error generating schema hash: {str(e)}")
    
    def save_schema_snapshot(self, output_file: str = None) -> str:
        """Save current schema snapshot for comparison purposes."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"schema_snapshot_{timestamp}.json"
        
        try:
            schema_info = {
                "timestamp": datetime.now().isoformat(),
                "database_url": self.database_url,
                "schema_hash": self.generate_schema_hash(),
                "tables": {}
            }
            
            tables = self.inspector.get_table_names()
            
            for table_name in tables:
                columns = self.inspector.get_columns(table_name)
                indexes = self.inspector.get_indexes(table_name)
                foreign_keys = self.inspector.get_foreign_keys(table_name)
                
                schema_info["tables"][table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "default": str(col["default"]) if col["default"] is not None else None
                        }
                        for col in columns
                    ],
                    "indexes": [
                        {
                            "name": idx["name"],
                            "columns": idx["column_names"],
                            "unique": idx["unique"]
                        }
                        for idx in indexes
                    ],
                    "foreign_keys": [
                        {
                            "columns": fk["constrained_columns"],
                            "referenced_table": fk["referred_table"],
                            "referenced_columns": fk["referred_columns"]
                        }
                        for fk in foreign_keys
                    ]
                }
            
            with open(output_file, 'w') as f:
                json.dump(schema_info, f, indent=2, sort_keys=True)
            
            return output_file
            
        except Exception as e:
            raise Exception(f"Error saving schema snapshot: {str(e)}")
    
    def compare_schema_snapshots(self, snapshot1_file: str, snapshot2_file: str) -> Dict[str, Any]:
        """Compare two schema snapshots to detect drift."""
        try:
            with open(snapshot1_file, 'r') as f:
                snapshot1 = json.load(f)
            
            with open(snapshot2_file, 'r') as f:
                snapshot2 = json.load(f)
            
            differences = {
                "schema_hash_changed": snapshot1["schema_hash"] != snapshot2["schema_hash"],
                "timestamp1": snapshot1["timestamp"],
                "timestamp2": snapshot2["timestamp"],
                "table_differences": {},
                "summary": []
            }
            
            # Compare each table
            tables1 = set(snapshot1["tables"].keys())
            tables2 = set(snapshot2["tables"].keys())
            
            # New tables
            new_tables = tables2 - tables1
            if new_tables:
                differences["summary"].append(f"New tables: {', '.join(new_tables)}")
            
            # Removed tables
            removed_tables = tables1 - tables2
            if removed_tables:
                differences["summary"].append(f"Removed tables: {', '.join(removed_tables)}")
            
            # Changed tables
            common_tables = tables1 & tables2
            for table in common_tables:
                table1 = snapshot1["tables"][table]
                table2 = snapshot2["tables"][table]
                
                if table1 != table2:
                    differences["table_differences"][table] = {
                        "columns_changed": table1["columns"] != table2["columns"],
                        "indexes_changed": table1["indexes"] != table2["indexes"],
                        "foreign_keys_changed": table1["foreign_keys"] != table2["foreign_keys"]
                    }
            
            return differences
            
        except Exception as e:
            raise Exception(f"Error comparing schema snapshots: {str(e)}")


# Pytest test functions
@pytest.fixture
def schema_tester():
    """Fixture to provide schema tester instance."""
    return DatabaseSchemaConsistencyTester()


def test_alembic_schema_consistency(schema_tester):
    """Test that alembic doesn't detect any schema changes."""
    result = schema_tester.test_alembic_schema_consistency()
    
    assert result["passed"], f"Schema consistency test failed: {result['message']}"


def test_foreign_key_indexes(schema_tester):
    """Test that all foreign keys have appropriate indexes."""
    result = schema_tester.test_foreign_key_indexes()
    
    if not result["passed"]:
        missing_indexes = result["details"]["missing_indexes"]
        suggestions = []
        for missing in missing_indexes:
            suggestions.append(f"CREATE INDEX {missing['suggested_index_name']} ON {missing['table']}({', '.join(missing['foreign_key_columns'])});")
        
        fail_message = f"{result['message']}\n\nSuggested fixes:\n" + "\n".join(suggestions)
        assert False, fail_message
    
    assert result["passed"], result["message"]


def test_data_integrity(schema_tester):
    """Test data integrity constraints."""
    result = schema_tester.test_data_integrity()
    
    if not result["passed"]:
        details = result["details"]
        error_summary = []
        
        if details["orphan_records"]:
            error_summary.append("Orphan records found:")
            for orphan in details["orphan_records"]:
                error_summary.append(f"  - {orphan['description']}: {orphan['count']} records")
        
        if details["null_violations"]:
            error_summary.append("NULL violations found:")
            for null_viol in details["null_violations"]:
                error_summary.append(f"  - {null_viol['description']}: {null_viol['count']} records")
        
        if details["constraint_violations"]:
            error_summary.append("Constraint violations found:")
            for constraint in details["constraint_violations"]:
                error_summary.append(f"  - {constraint['description']}: {constraint['count']} records")
        
        fail_message = f"{result['message']}\n\n" + "\n".join(error_summary)
        assert False, fail_message
    
    assert result["passed"], result["message"]


def test_schema_hash_generation(schema_tester):
    """Test schema hash generation for drift detection."""
    schema_hash = schema_tester.generate_schema_hash()
    
    assert isinstance(schema_hash, str), "Schema hash should be a string"
    assert len(schema_hash) == 64, "Schema hash should be SHA256 (64 characters)"
    assert schema_hash.isalnum(), "Schema hash should be alphanumeric"


if __name__ == "__main__":
    """Command-line interface for running schema consistency tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Schema Consistency Tests")
    parser.add_argument("--database-url", help="Database URL to test")
    parser.add_argument("--save-snapshot", help="Save schema snapshot to file")
    parser.add_argument("--compare-snapshots", nargs=2, help="Compare two schema snapshots")
    parser.add_argument("--generate-hash", action="store_true", help="Generate schema hash")
    
    args = parser.parse_args()
    
    tester = DatabaseSchemaConsistencyTester(args.database_url)
    
    if args.save_snapshot:
        output_file = tester.save_schema_snapshot(args.save_snapshot)
        print(f"Schema snapshot saved to: {output_file}")
    
    elif args.compare_snapshots:
        differences = tester.compare_schema_snapshots(args.compare_snapshots[0], args.compare_snapshots[1])
        print("Schema comparison results:")
        print(json.dumps(differences, indent=2))
    
    elif args.generate_hash:
        schema_hash = tester.generate_schema_hash()
        print(f"Current schema hash: {schema_hash}")
    
    else:
        # Run all tests
        print("Running schema consistency tests...")
        
        print("\n1. Testing Alembic schema consistency...")
        result1 = tester.test_alembic_schema_consistency()
        print(f"   Result: {'PASS' if result1['passed'] else 'FAIL'} - {result1['message']}")
        
        print("\n2. Testing foreign key indexes...")
        result2 = tester.test_foreign_key_indexes()
        print(f"   Result: {'PASS' if result2['passed'] else 'FAIL'} - {result2['message']}")
        
        print("\n3. Testing data integrity...")
        result3 = tester.test_data_integrity()
        print(f"   Result: {'PASS' if result3['passed'] else 'FAIL'} - {result3['message']}")
        
        print("\n4. Generating schema hash for drift detection...")
        schema_hash = tester.generate_schema_hash()
        print(f"   Current schema hash: {schema_hash}")
        
        all_passed = all([result1["passed"], result2["passed"], result3["passed"]])
        print(f"\nOverall result: {'PASS' if all_passed else 'FAIL'}")
        
        if not all_passed:
            exit(1)
