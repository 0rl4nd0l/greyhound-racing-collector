#!/usr/bin/env python3
"""
Database Quality Assessment Script
=================================

Comprehensive assessment of data quality for ML training.
"""

import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")

def assess_database_quality(db_path="greyhound_racing_data.db"):
    """Run comprehensive database quality checks."""
    
    if not os.path.exists(db_path):
        return {"error": f"Database not found: {db_path}"}
    
    print(f"🔍 Assessing database quality: {db_path}")
    
    assessment = {
        "database_path": db_path,
        "assessment_timestamp": datetime.now().isoformat(),
        "file_size_mb": round(os.path.getsize(db_path) / 1024 / 1024, 2),
        "tables": {},
        "data_quality": {},
        "issues": [],
        "recommendations": []
    }
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get all tables
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
        
        assessment["tables"] = {}
        
        # Assess each table
        for table in tables:
            print(f"   Analyzing {table}...")
            table_info = assess_table(conn, table)
            assessment["tables"][table] = table_info
            
            # Collect issues
            if table_info.get("issues"):
                assessment["issues"].extend([f"{table}: {issue}" for issue in table_info["issues"]])
        
        # Key table assessments
        if "dog_race_data" in tables and "race_metadata" in tables:
            assessment["data_quality"] = assess_data_relationships(conn)
        
        # Generate recommendations
        assessment["recommendations"] = generate_recommendations(assessment)
        
        conn.close()
        
        # Save assessment
        output_path = "reports/data_quality_report.json"
        os.makedirs("reports", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(assessment, f, indent=2, default=str)
        
        print(f"✅ Assessment complete. Report saved to {output_path}")
        return assessment
        
    except Exception as e:
        assessment["error"] = str(e)
        print(f"❌ Error assessing database: {e}")
        return assessment

def assess_table(conn, table_name):
    """Assess individual table quality."""
    info = {
        "row_count": 0,
        "columns": {},
        "issues": [],
        "data_types": {},
        "sample_data": {}
    }
    
    try:
        # Get row count
        cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        info["row_count"] = cursor.fetchone()[0]
        
        # Get table schema
        cursor = conn.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            info["columns"][col_name] = {
                "type": col_type,
                "nullable": bool(col[3]),
                "default": col[4]
            }
            
            # Check for nulls and data quality
            null_count = conn.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NULL").fetchone()[0]
            info["columns"][col_name]["null_count"] = null_count
            info["columns"][col_name]["null_rate"] = null_count / max(info["row_count"], 1)
            
            # Sample values (non-null)
            try:
                sample = conn.execute(f"SELECT DISTINCT {col_name} FROM {table_name} WHERE {col_name} IS NOT NULL LIMIT 5").fetchall()
                info["columns"][col_name]["sample_values"] = [str(row[0]) for row in sample]
            except:
                info["columns"][col_name]["sample_values"] = []
        
        # Check for high null rates
        high_null_columns = [col for col, data in info["columns"].items() 
                           if data["null_rate"] > 0.5 and info["row_count"] > 100]
        if high_null_columns:
            info["issues"].append(f"High null rates in columns: {', '.join(high_null_columns)}")
        
        # Special checks for key tables
        if table_name == "dog_race_data":
            info.update(assess_dog_race_data(conn))
        elif table_name == "race_metadata":
            info.update(assess_race_metadata(conn))
            
    except Exception as e:
        info["error"] = str(e)
        info["issues"].append(f"Error analyzing table: {str(e)}")
    
    return info

def assess_dog_race_data(conn):
    """Special assessment for dog_race_data table."""
    issues = []
    stats = {}
    
    try:
        # Check for duplicate (race_id, box_number) combinations
        dup_count = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT race_id, box_number, COUNT(*) as cnt 
                FROM dog_race_data 
                WHERE race_id IS NOT NULL AND box_number IS NOT NULL
                GROUP BY race_id, box_number 
                HAVING cnt > 1
            )
        """).fetchone()[0]
        
        if dup_count > 0:
            issues.append(f"Found {dup_count} duplicate (race_id, box_number) combinations")
        
        # Check finish position validity
        invalid_positions = conn.execute("""
            SELECT COUNT(*) FROM dog_race_data 
            WHERE finish_position IS NOT NULL 
            AND (finish_position < 1 OR finish_position > 20)
        """).fetchone()[0]
        
        if invalid_positions > 0:
            issues.append(f"Found {invalid_positions} invalid finish positions")
        
        # Count races with missing critical data
        missing_odds = conn.execute("SELECT COUNT(*) FROM dog_race_data WHERE starting_price IS NULL").fetchone()[0]
        stats["missing_odds_count"] = missing_odds
        
        missing_positions = conn.execute("SELECT COUNT(*) FROM dog_race_data WHERE finish_position IS NULL").fetchone()[0]
        stats["missing_positions_count"] = missing_positions
        
    except Exception as e:
        issues.append(f"Error in dog_race_data assessment: {str(e)}")
    
    return {"dog_race_issues": issues, "dog_race_stats": stats}

def assess_race_metadata(conn):
    """Special assessment for race_metadata table."""
    issues = []
    stats = {}
    
    try:
        # Check for races missing winners
        missing_winners = conn.execute("SELECT COUNT(*) FROM race_metadata WHERE winner_name IS NULL").fetchone()[0]
        if missing_winners > 0:
            issues.append(f"Found {missing_winners} races without winner names")
        
        # Check date/time consistency
        invalid_dates = conn.execute("""
            SELECT COUNT(*) FROM race_metadata 
            WHERE race_date IS NULL OR race_time IS NULL
        """).fetchone()[0]
        
        if invalid_dates > 0:
            issues.append(f"Found {invalid_dates} races with missing date/time")
        
        # Check for reasonable date ranges
        try:
            date_range = conn.execute("""
                SELECT MIN(race_date) as min_date, MAX(race_date) as max_date 
                FROM race_metadata WHERE race_date IS NOT NULL
            """).fetchone()
            stats["date_range"] = {"min": date_range[0], "max": date_range[1]}
        except:
            pass
        
        # Venue distribution
        try:
            venue_count = conn.execute("SELECT COUNT(DISTINCT venue) FROM race_metadata WHERE venue IS NOT NULL").fetchone()[0]
            stats["unique_venues"] = venue_count
        except:
            pass
            
    except Exception as e:
        issues.append(f"Error in race_metadata assessment: {str(e)}")
    
    return {"race_metadata_issues": issues, "race_metadata_stats": stats}

def assess_data_relationships(conn):
    """Assess relationships between key tables."""
    relationships = {}
    
    try:
        # Join coverage between dog_race_data and race_metadata
        orphaned_dogs = conn.execute("""
            SELECT COUNT(*) FROM dog_race_data d
            LEFT JOIN race_metadata r ON d.race_id = r.race_id
            WHERE r.race_id IS NULL
        """).fetchone()[0]
        
        orphaned_races = conn.execute("""
            SELECT COUNT(*) FROM race_metadata r
            LEFT JOIN dog_race_data d ON r.race_id = d.race_id  
            WHERE d.race_id IS NULL
        """).fetchone()[0]
        
        relationships["orphaned_dog_records"] = orphaned_dogs
        relationships["orphaned_race_records"] = orphaned_races
        
        # Race integrity - check if race winners match finish positions
        winner_mismatch = conn.execute("""
            SELECT COUNT(*) FROM race_metadata r
            JOIN dog_race_data d ON r.race_id = d.race_id
            WHERE r.winner_name IS NOT NULL 
            AND d.finish_position = 1
            AND LOWER(TRIM(r.winner_name)) != LOWER(TRIM(d.dog_name))
        """).fetchone()[0]
        
        relationships["winner_mismatches"] = winner_mismatch
        
        # Field size consistency
        field_size_check = conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT r.race_id, r.field_size, COUNT(d.race_id) as actual_dogs
                FROM race_metadata r
                LEFT JOIN dog_race_data d ON r.race_id = d.race_id
                WHERE r.field_size IS NOT NULL
                GROUP BY r.race_id, r.field_size
                HAVING r.field_size != actual_dogs
            )
        """).fetchone()[0]
        
        relationships["field_size_mismatches"] = field_size_check
        
    except Exception as e:
        relationships["error"] = str(e)
    
    return relationships

def generate_recommendations(assessment):
    """Generate actionable recommendations based on assessment."""
    recommendations = []
    
    # High-level issues
    total_issues = len(assessment.get("issues", []))
    if total_issues > 10:
        recommendations.append("HIGH: Critical data quality issues detected - prioritize data cleaning")
    
    # Specific recommendations
    tables = assessment.get("tables", {})
    
    # Dog race data recommendations
    if "dog_race_data" in tables:
        dog_info = tables["dog_race_data"]
        if dog_info.get("dog_race_stats", {}).get("missing_odds_count", 0) > 1000:
            recommendations.append("MED: High number of missing starting prices - affects EV calculations")
        if dog_info.get("dog_race_issues"):
            recommendations.append("HIGH: Fix duplicate race/box combinations and invalid positions")
    
    # Race metadata recommendations  
    if "race_metadata" in tables:
        race_info = tables["race_metadata"]
        if race_info.get("race_metadata_issues"):
            recommendations.append("HIGH: Fix missing winner names - required for proper labeling")
    
    # Relationship recommendations
    data_quality = assessment.get("data_quality", {})
    if data_quality.get("orphaned_dog_records", 0) > 100:
        recommendations.append("MED: Many dog records without race metadata - affects feature engineering")
    if data_quality.get("winner_mismatches", 0) > 50:
        recommendations.append("HIGH: Winner name mismatches - verify race result scraping accuracy")
    
    # General recommendations
    if not recommendations:
        recommendations.append("LOW: Data quality appears acceptable for ML training")
    
    return recommendations

if __name__ == "__main__":
    import sys
    
    # Use environment variable or default path
    db_path = os.getenv("GREYHOUND_DB_PATH", "greyhound_racing_data.db")
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    assessment = assess_database_quality(db_path)
    
    # Print summary
    print("\n📋 Assessment Summary:")
    print(f"Database: {assessment['database_path']} ({assessment.get('file_size_mb', 0)}MB)")
    
    if "error" in assessment:
        print(f"❌ Error: {assessment['error']}")
    else:
        table_count = len(assessment.get("tables", {}))
        issue_count = len(assessment.get("issues", []))
        print(f"Tables: {table_count}")
        print(f"Issues: {issue_count}")
        
        if assessment.get("recommendations"):
            print("\n🎯 Top Recommendations:")
            for rec in assessment["recommendations"][:5]:
                print(f"   • {rec}")
