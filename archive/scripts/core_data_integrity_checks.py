#!/usr/bin/env python3
"""
Step 4: Core Data Integrity Checks
=================================

Comprehensive data integrity analysis for the unified greyhound racing database:
1. Null / missing value audit per column; classify as Allowed vs Unexpected
2. Duplicate detection using composite keys (e.g., race_id + dog_box)
3. Data-type range checks (negative times, impossible dates, unrealistic speeds)
4. Produce a heat-map style report of integrity issues ranked by severity

Author: AI Assistant
Date: January 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
import re
from collections import defaultdict, Counter
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoreDataIntegrityAnalyzer:
    """Comprehensive data integrity analyzer for the greyhound racing database"""
    
    def __init__(self, database_paths: List[str]):
        """
        Initialize the analyzer with multiple database paths
        
        Args:
            database_paths: List of database file paths to analyze
        """
        self.database_paths = database_paths
        self.integrity_issues = {}
        self.severity_weights = {
            'CRITICAL': 100,
            'HIGH': 50,
            'MEDIUM': 20,
            'LOW': 5,
            'INFO': 1
        }
        self.analysis_results = {}
        
        # Define expected ranges and rules
        self.data_rules = {
            'race_metadata': {
                'field_size': {'min': 1, 'max': 16, 'type': 'int'},
                'distance': {'pattern': r'^\d+m?$|^Unknown$', 'type': 'str'},
                'race_number': {'min': 1, 'max': 20, 'type': 'int'},
                'temperature': {'min': -10, 'max': 50, 'type': 'float'},
                'humidity': {'min': 0, 'max': 100, 'type': 'float'},
                'wind_speed': {'min': 0, 'max': 100, 'type': 'float'},
                'winner_odds': {'min': 1.0, 'max': 999.0, 'type': 'float'},
                'race_date': {'type': 'date', 'min_date': '2020-01-01', 'max_date': '2030-12-31'}
            },
            'dog_race_data': {
                'box_number': {'min': 1, 'max': 16, 'type': 'int'},
                'weight': {'min': 20.0, 'max': 45.0, 'type': 'float'},
                'odds_decimal': {'min': 1.0, 'max': 999.0, 'type': 'float'},
                'finish_position': {'min': 1, 'max': 16, 'type': 'int'},
                'individual_time': {'pattern': r'^\d+\.\d+$|^Unknown$|^NULL$', 'type': 'str'},
                'beaten_margin': {'min': 0.0, 'max': 50.0, 'type': 'float'}
            },
            'dogs': {
                'total_races': {'min': 0, 'max': 1000, 'type': 'int'},
                'total_wins': {'min': 0, 'max': 1000, 'type': 'int'},
                'total_places': {'min': 0, 'max': 1000, 'type': 'int'},
                'best_time': {'min': 20.0, 'max': 60.0, 'type': 'float'},
                'average_position': {'min': 1.0, 'max': 16.0, 'type': 'float'},
                'weight': {'min': 20.0, 'max': 45.0, 'type': 'float'},
                'age': {'min': 1, 'max': 10, 'type': 'int'}
            }
        }
        
        # Define mandatory fields (should not be null)
        self.mandatory_fields = {
            'race_metadata': ['race_id', 'venue', 'race_date'],
            'dog_race_data': ['race_id', 'dog_name', 'dog_clean_name'],
            'dogs': ['dog_name']
        }
        
        # Define composite keys for duplicate detection
        self.composite_keys = {
            'race_metadata': ['race_id'],
            'dog_race_data': ['race_id', 'dog_clean_name', 'box_number'],
            'dogs': ['dog_name']
        }

    def analyze_all_databases(self) -> Dict[str, Any]:
        """
        Analyze all databases and generate comprehensive integrity report
        
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Starting comprehensive data integrity analysis...")
        
        for db_path in self.database_paths:
            if os.path.exists(db_path):
                logger.info(f"Analyzing database: {db_path}")
                try:
                    self.analyze_database(db_path)
                except Exception as e:
                    logger.error(f"Error analyzing {db_path}: {e}")
                    self.integrity_issues[db_path] = {'error': str(e)}
            else:
                logger.warning(f"Database not found: {db_path}")
        
        # Generate consolidated report
        self.generate_integrity_report()
        return self.analysis_results

    def analyze_database(self, db_path: str) -> None:
        """Analyze a single database for integrity issues"""
        
        with sqlite3.connect(db_path) as conn:
            # Get database schema information
            tables = self.get_table_info(conn)
            
            db_results = {
                'database_path': db_path,
                'tables': tables,
                'null_value_audit': {},
                'duplicate_detection': {},
                'range_validation': {},
                'data_quality_issues': [],
                'severity_summary': defaultdict(int)
            }
            
            for table_name in tables:
                logger.info(f"  Analyzing table: {table_name}")
                
                # 1. Null/Missing Value Audit
                null_audit = self.audit_null_values(conn, table_name)
                db_results['null_value_audit'][table_name] = null_audit
                
                # 2. Duplicate Detection
                duplicates = self.detect_duplicates(conn, table_name)
                db_results['duplicate_detection'][table_name] = duplicates
                
                # 3. Data Range Validation
                range_issues = self.validate_data_ranges(conn, table_name)
                db_results['range_validation'][table_name] = range_issues
                
                # 4. Data Quality Assessment
                quality_issues = self.assess_data_quality(conn, table_name)
                db_results['data_quality_issues'].extend(quality_issues)
            
            # Calculate severity summary
            for issue in db_results['data_quality_issues']:
                severity = issue.get('severity', 'LOW')
                db_results['severity_summary'][severity] += 1
            
            self.analysis_results[db_path] = db_results

    def get_table_info(self, conn: sqlite3.Connection) -> List[str]:
        """Get list of tables in the database"""
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return [table for table in tables if not table.startswith('sqlite_')]

    def audit_null_values(self, conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """
        Audit null and missing values per column
        Classify as Allowed vs Unexpected
        """
        
        # Get table schema
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Get total row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        null_audit = {
            'total_rows': total_rows,
            'columns': {},
            'summary': {
                'critical_nulls': 0,
                'allowed_nulls': 0,
                'unexpected_nulls': 0
            }
        }
        
        for column in columns:
            # Count nulls and empty strings
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN {column} IS NULL THEN 1 END) as null_count,
                    COUNT(CASE WHEN {column} = '' THEN 1 END) as empty_count,
                    COUNT(CASE WHEN LOWER(CAST({column} AS TEXT)) IN ('nan', 'null', 'none', 'n/a', 'unknown') THEN 1 END) as placeholder_count
                FROM {table_name}
            """)
            
            result = cursor.fetchone()
            total, null_count, empty_count, placeholder_count = result
            
            missing_count = null_count + empty_count + placeholder_count
            missing_percentage = (missing_count / total * 100) if total > 0 else 0
            
            # Classify as allowed or unexpected
            is_mandatory = column in self.mandatory_fields.get(table_name, [])
            
            if is_mandatory and missing_count > 0:
                classification = 'UNEXPECTED'
                severity = 'CRITICAL'
                null_audit['summary']['critical_nulls'] += 1
            elif missing_percentage > 50:
                classification = 'UNEXPECTED'
                severity = 'HIGH'
                null_audit['summary']['unexpected_nulls'] += 1
            elif missing_percentage > 20:
                classification = 'UNEXPECTED'
                severity = 'MEDIUM'
                null_audit['summary']['unexpected_nulls'] += 1
            else:
                classification = 'ALLOWED'
                severity = 'LOW'
                null_audit['summary']['allowed_nulls'] += 1
            
            null_audit['columns'][column] = {
                'null_count': null_count,
                'empty_count': empty_count,
                'placeholder_count': placeholder_count,
                'total_missing': missing_count,
                'missing_percentage': round(missing_percentage, 2),
                'classification': classification,
                'severity': severity,
                'is_mandatory': is_mandatory
            }
        
        return null_audit

    def detect_duplicates(self, conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """
        Detect duplicates using composite keys
        """
        
        composite_key = self.composite_keys.get(table_name, [])
        if not composite_key:
            return {'message': 'No composite key defined for this table'}
        
        # Build the GROUP BY clause
        key_clause = ', '.join(composite_key)
        
        cursor = conn.cursor()
        
        # Find duplicates
        query = f"""
            SELECT {key_clause}, COUNT(*) as duplicate_count
            FROM {table_name}
            WHERE {' AND '.join(f'{key} IS NOT NULL' for key in composite_key)}
            GROUP BY {key_clause}
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC
        """
        
        cursor.execute(query)
        duplicates = cursor.fetchall()
        
        # Get total count for context
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        duplicate_analysis = {
            'composite_key': composite_key,
            'total_rows': total_rows,
            'duplicate_groups': len(duplicates),
            'total_duplicate_records': sum(count for *_, count in duplicates),
            'duplicates': []
        }
        
        for duplicate in duplicates[:50]:  # Limit to top 50 for reporting
            *key_values, count = duplicate
            duplicate_analysis['duplicates'].append({
                'key_values': dict(zip(composite_key, key_values)),
                'count': count
            })
        
        # Calculate severity
        duplicate_percentage = (duplicate_analysis['total_duplicate_records'] / total_rows * 100) if total_rows > 0 else 0
        
        if duplicate_percentage > 10:
            duplicate_analysis['severity'] = 'CRITICAL'
        elif duplicate_percentage > 5:
            duplicate_analysis['severity'] = 'HIGH'
        elif duplicate_percentage > 1:
            duplicate_analysis['severity'] = 'MEDIUM'
        else:
            duplicate_analysis['severity'] = 'LOW'
        
        duplicate_analysis['duplicate_percentage'] = round(duplicate_percentage, 2)
        
        return duplicate_analysis

    def validate_data_ranges(self, conn: sqlite3.Connection, table_name: str) -> Dict[str, Any]:
        """
        Validate data types and ranges
        Check for negative times, impossible dates, unrealistic values
        """
        
        table_rules = self.data_rules.get(table_name, {})
        if not table_rules:
            return {'message': 'No validation rules defined for this table'}
        
        cursor = conn.cursor()
        range_validation = {
            'total_validations': 0,
            'total_violations': 0,
            'field_violations': {}
        }
        
        for field, rules in table_rules.items():
            # Check if field exists in table
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [row[1] for row in cursor.fetchall()]
            
            if field not in columns:
                continue
            
            field_violations = {
                'rules': rules,
                'violations': [],
                'violation_count': 0,
                'total_checked': 0
            }
            
            # Get total non-null values for this field
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {field} IS NOT NULL")
            total_non_null = cursor.fetchone()[0]
            field_violations['total_checked'] = total_non_null
            
            if rules.get('type') == 'int':
                # Integer range validation
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None or max_val is not None:
                    conditions = []
                    if min_val is not None:
                        conditions.append(f"CAST({field} AS INTEGER) < {min_val}")
                    if max_val is not None:
                        conditions.append(f"CAST({field} AS INTEGER) > {max_val}")
                    
                    where_clause = ' OR '.join(conditions)
                    query = f"""
                        SELECT {field}, COUNT(*) as violation_count
                        FROM {table_name}
                        WHERE {field} IS NOT NULL AND ({where_clause})
                        GROUP BY {field}
                        ORDER BY violation_count DESC
                        LIMIT 10
                    """
                    
                    cursor.execute(query)
                    violations = cursor.fetchall()
                    
                    for value, count in violations:
                        field_violations['violations'].append({
                            'value': value,
                            'count': count,
                            'issue': f'Value {value} outside range [{min_val}, {max_val}]'
                        })
                        field_violations['violation_count'] += count
            
            elif rules.get('type') == 'float':
                # Float range validation
                min_val = rules.get('min')
                max_val = rules.get('max')
                
                if min_val is not None or max_val is not None:
                    conditions = []
                    if min_val is not None:
                        conditions.append(f"CAST({field} AS FLOAT) < {min_val}")
                    if max_val is not None:
                        conditions.append(f"CAST({field} AS FLOAT) > {max_val}")
                    
                    where_clause = ' OR '.join(conditions)
                    query = f"""
                        SELECT {field}, COUNT(*) as violation_count
                        FROM {table_name}
                        WHERE {field} IS NOT NULL AND ({where_clause})
                        GROUP BY {field}
                        ORDER BY violation_count DESC
                        LIMIT 10
                    """
                    
                    cursor.execute(query)
                    violations = cursor.fetchall()
                    
                    for value, count in violations:
                        field_violations['violations'].append({
                            'value': value,
                            'count': count,
                            'issue': f'Value {value} outside range [{min_val}, {max_val}]'
                        })
                        field_violations['violation_count'] += count
            
            elif rules.get('type') == 'date':
                # Date validation
                min_date = rules.get('min_date')
                max_date = rules.get('max_date')
                
                if min_date or max_date:
                    conditions = []
                    if min_date:
                        conditions.append(f"date({field}) < date('{min_date}')")
                    if max_date:
                        conditions.append(f"date({field}) > date('{max_date}')")
                    
                    where_clause = ' OR '.join(conditions)
                    query = f"""
                        SELECT {field}, COUNT(*) as violation_count
                        FROM {table_name}
                        WHERE {field} IS NOT NULL AND ({where_clause})
                        GROUP BY {field}
                        ORDER BY violation_count DESC
                        LIMIT 10
                    """
                    
                    try:
                        cursor.execute(query)
                        violations = cursor.fetchall()
                        
                        for value, count in violations:
                            field_violations['violations'].append({
                                'value': value,
                                'count': count,
                                'issue': f'Date {value} outside range [{min_date}, {max_date}]'
                            })
                            field_violations['violation_count'] += count
                    except Exception as e:
                        field_violations['violations'].append({
                            'value': 'ERROR',
                            'count': 0,
                            'issue': f'Date validation error: {str(e)}'
                        })
            
            elif rules.get('pattern'):
                # Pattern validation
                pattern = rules['pattern']
                
                # Use SQLite REGEXP if available, otherwise use LIKE patterns
                try:
                    query = f"""
                        SELECT {field}, COUNT(*) as violation_count
                        FROM {table_name}
                        WHERE {field} IS NOT NULL 
                        AND {field} NOT REGEXP '{pattern}'
                        GROUP BY {field}
                        ORDER BY violation_count DESC
                        LIMIT 10
                    """
                    cursor.execute(query)
                    violations = cursor.fetchall()
                    
                    for value, count in violations:
                        field_violations['violations'].append({
                            'value': value,
                            'count': count,
                            'issue': f'Value "{value}" does not match pattern {pattern}'
                        })
                        field_violations['violation_count'] += count
                        
                except sqlite3.OperationalError:
                    # REGEXP not available, skip pattern validation
                    field_violations['violations'].append({
                        'value': 'SKIPPED',
                        'count': 0,
                        'issue': 'Pattern validation skipped (REGEXP not available)'
                    })
            
            # Calculate violation percentage
            if field_violations['total_checked'] > 0:
                violation_percentage = (field_violations['violation_count'] / field_violations['total_checked']) * 100
                field_violations['violation_percentage'] = round(violation_percentage, 2)
                
                # Determine severity
                if violation_percentage > 20:
                    field_violations['severity'] = 'CRITICAL'
                elif violation_percentage > 10:
                    field_violations['severity'] = 'HIGH'
                elif violation_percentage > 5:
                    field_violations['severity'] = 'MEDIUM'
                else:
                    field_violations['severity'] = 'LOW'
            else:
                field_violations['violation_percentage'] = 0
                field_violations['severity'] = 'INFO'
            
            range_validation['field_violations'][field] = field_violations
            range_validation['total_validations'] += field_violations['total_checked']
            range_validation['total_violations'] += field_violations['violation_count']
        
        return range_validation

    def assess_data_quality(self, conn: sqlite3.Connection, table_name: str) -> List[Dict[str, Any]]:
        """
        Assess overall data quality and identify specific issues
        """
        
        quality_issues = []
        cursor = conn.cursor()
        
        # Get table row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]
        
        if total_rows == 0:
            quality_issues.append({
                'table': table_name,
                'issue_type': 'EMPTY_TABLE',
                'description': f'Table {table_name} is empty',
                'severity': 'CRITICAL',
                'affected_rows': 0,
                'recommendation': 'Investigate why table has no data'
            })
            return quality_issues
        
        # Check for specific data quality issues
        
        # 1. Check for suspicious race times (if applicable)
        if table_name == 'dog_race_data' and 'individual_time' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]:
            try:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM dog_race_data
                    WHERE individual_time IS NOT NULL 
                    AND individual_time != 'Unknown'
                    AND individual_time != ''
                    AND (
                        CAST(individual_time AS FLOAT) < 20.0 
                        OR CAST(individual_time AS FLOAT) > 60.0
                    )
                """)
                
                suspicious_times = cursor.fetchone()[0]
                if suspicious_times > 0:
                    quality_issues.append({
                        'table': table_name,
                        'issue_type': 'SUSPICIOUS_RACE_TIMES',
                        'description': f'Found {suspicious_times} races with suspicious times (< 20s or > 60s)',
                        'severity': 'MEDIUM' if suspicious_times < total_rows * 0.05 else 'HIGH',
                        'affected_rows': suspicious_times,
                        'recommendation': 'Review race time data collection and validation'
                    })
            except Exception as e:
                logger.warning(f"Could not validate race times: {e}")
        
        # 2. Check for impossible odds
        if table_name == 'dog_race_data' and 'odds_decimal' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]:
            try:
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM dog_race_data
                    WHERE odds_decimal IS NOT NULL 
                    AND (odds_decimal < 1.0 OR odds_decimal > 999.0)
                """)
                
                impossible_odds = cursor.fetchone()[0]
                if impossible_odds > 0:
                    quality_issues.append({
                        'table': table_name,
                        'issue_type': 'IMPOSSIBLE_ODDS',
                        'description': f'Found {impossible_odds} records with impossible odds',
                        'severity': 'HIGH',
                        'affected_rows': impossible_odds,
                        'recommendation': 'Review odds data collection process'
                    })
            except Exception as e:
                logger.warning(f"Could not validate odds: {e}")
        
        # 3. Check for future race dates
        if table_name == 'race_metadata' and 'race_date' in [col[1] for col in cursor.execute(f"PRAGMA table_info({table_name})")]:
            try:
                future_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM race_metadata
                    WHERE race_date > '{future_date}'
                """)
                
                future_races = cursor.fetchone()[0]
                if future_races > 0:
                    quality_issues.append({
                        'table': table_name,
                        'issue_type': 'FUTURE_RACE_DATES',
                        'description': f'Found {future_races} races with dates more than 1 year in the future',
                        'severity': 'MEDIUM',
                        'affected_rows': future_races,
                        'recommendation': 'Verify race date extraction logic'
                    })
            except Exception as e:
                logger.warning(f"Could not validate race dates: {e}")
        
        # 4. Check for inconsistent dog names
        if table_name == 'dog_race_data':
            try:
                cursor.execute("""
                    SELECT COUNT(DISTINCT dog_name), COUNT(DISTINCT dog_clean_name)
                    FROM dog_race_data
                    WHERE dog_name IS NOT NULL AND dog_clean_name IS NOT NULL
                """)
                
                raw_names, clean_names = cursor.fetchone()
                if raw_names and clean_names and raw_names > clean_names * 1.5:
                    quality_issues.append({
                        'table': table_name,
                        'issue_type': 'INCONSISTENT_DOG_NAMES',
                        'description': f'High variation in dog names: {raw_names} raw vs {clean_names} clean',
                        'severity': 'MEDIUM',
                        'affected_rows': raw_names - clean_names,
                        'recommendation': 'Review dog name cleaning and standardization process'
                    })
            except Exception as e:
                logger.warning(f"Could not validate dog names: {e}")
        
        return quality_issues

    def generate_integrity_report(self) -> None:
        """Generate comprehensive integrity report with heat map visualization"""
        
        logger.info("Generating comprehensive integrity report...")
        
        # Create output directory
        output_dir = Path('./integrity_analysis_reports')
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Generate detailed JSON report
        json_report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'databases_analyzed': len(self.analysis_results),
            'summary': self.generate_summary_statistics(),
            'detailed_results': self.analysis_results,
            'recommendations': self.generate_recommendations()
        }
        
        json_path = output_dir / f'integrity_analysis_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        logger.info(f"Detailed JSON report saved to: {json_path}")
        
        # 2. Generate heat map visualization
        self.generate_heatmap_visualization(output_dir, timestamp)
        
        # 3. Generate HTML summary report
        self.generate_html_report(output_dir, timestamp, json_report)
        
        # 4. Generate console summary
        self.print_console_summary(json_report['summary'])

    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics across all databases"""
        
        summary = {
            'total_databases': len(self.analysis_results),
            'total_tables': 0,
            'total_records': 0,
            'severity_distribution': defaultdict(int),
            'issue_types': defaultdict(int),
            'database_health_scores': {}
        }
        
        for db_path, results in self.analysis_results.items():
            if 'error' in results:
                continue
                
            summary['total_tables'] += len(results['tables'])
            
            # Count total records
            for table_name in results['tables']:
                null_audit = results['null_value_audit'].get(table_name, {})
                if 'total_rows' in null_audit:
                    summary['total_records'] += null_audit['total_rows']
            
            # Count severity distribution
            for severity, count in results['severity_summary'].items():
                summary['severity_distribution'][severity] += count
            
            # Count issue types
            for issue in results['data_quality_issues']:
                issue_type = issue.get('issue_type', 'UNKNOWN')
                summary['issue_types'][issue_type] += 1
            
            # Calculate database health score
            health_score = self.calculate_health_score(results)
            summary['database_health_scores'][db_path] = health_score
        
        return summary

    def calculate_health_score(self, results: Dict[str, Any]) -> float:
        """Calculate a health score (0-100) for a database"""
        
        total_penalty = 0
        max_penalty = 1000  # Maximum possible penalty
        
        # Penalty for severity distribution
        for severity, count in results['severity_summary'].items():
            weight = self.severity_weights.get(severity, 1)
            total_penalty += count * weight
        
        # Additional penalties
        for table_name, duplicate_info in results['duplicate_detection'].items():
            if isinstance(duplicate_info, dict) and 'duplicate_percentage' in duplicate_info:
                duplicate_penalty = duplicate_info['duplicate_percentage'] * 5
                total_penalty += duplicate_penalty
        
        # Calculate score (higher is better)
        health_score = max(0, 100 - (total_penalty / max_penalty * 100))
        return round(health_score, 1)

    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on analysis"""
        
        recommendations = []
        
        # Analyze patterns across all databases
        all_issues = []
        for results in self.analysis_results.values():
            if 'data_quality_issues' in results:
                all_issues.extend(results['data_quality_issues'])
        
        # Count issue types
        issue_counts = Counter(issue['issue_type'] for issue in all_issues)
        
        # Generate recommendations based on most common issues
        for issue_type, count in issue_counts.most_common(10):
            if issue_type == 'SUSPICIOUS_RACE_TIMES':
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Data Validation',
                    'issue': f'Suspicious race times found in {count} cases',
                    'recommendation': 'Implement race time validation rules during data ingestion',
                    'technical_details': 'Add checks for times outside 20-60 second range'
                })
            
            elif issue_type == 'IMPOSSIBLE_ODDS':
                recommendations.append({
                    'priority': 'HIGH',
                    'category': 'Data Quality',
                    'issue': f'Impossible odds values found in {count} cases',
                    'recommendation': 'Add odds validation to prevent values < 1.0 or > 999.0',
                    'technical_details': 'Implement odds range validation in scraping pipeline'
                })
            
            elif issue_type == 'INCONSISTENT_DOG_NAMES':
                recommendations.append({
                    'priority': 'MEDIUM',
                    'category': 'Data Standardization',
                    'issue': f'Inconsistent dog names found in {count} cases',
                    'recommendation': 'Improve dog name cleaning and standardization',
                    'technical_details': 'Enhance name normalization algorithms and create master dog registry'
                })
        
        # Add general recommendations
        recommendations.extend([
            {
                'priority': 'HIGH',
                'category': 'Data Integrity',
                'issue': 'Multiple database files with potential inconsistencies',
                'recommendation': 'Consolidate to single authoritative database with proper migration',
                'technical_details': 'Design unified schema and migrate data with deduplication'
            },
            {
                'priority': 'MEDIUM',
                'category': 'Data Monitoring',
                'issue': 'No automated data quality monitoring',
                'recommendation': 'Implement automated daily data integrity checks',
                'technical_details': 'Schedule this analysis to run automatically and alert on issues'
            }
        ])
        
        return recommendations

    def generate_heatmap_visualization(self, output_dir: Path, timestamp: str) -> None:
        """Generate heat map visualization of integrity issues"""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Prepare data for heatmap
            databases = []
            tables = []
            severity_scores = []
            
            for db_path, results in self.analysis_results.items():
                if 'error' in results:
                    continue
                
                db_name = Path(db_path).stem
                
                for table_name in results['tables']:
                    # Calculate severity score for this table
                    score = 0
                    
                    # Add null value issues
                    null_audit = results['null_value_audit'].get(table_name, {})
                    for col_info in null_audit.get('columns', {}).values():
                        severity = col_info.get('severity', 'LOW')
                        score += self.severity_weights.get(severity, 1)
                    
                    # Add duplicate issues
                    duplicate_info = results['duplicate_detection'].get(table_name, {})
                    if isinstance(duplicate_info, dict) and 'severity' in duplicate_info:
                        severity = duplicate_info['severity']
                        score += self.severity_weights.get(severity, 1) * 2
                    
                    # Add range validation issues
                    range_info = results['range_validation'].get(table_name, {})
                    for field_info in range_info.get('field_violations', {}).values():
                        severity = field_info.get('severity', 'LOW')
                        score += self.severity_weights.get(severity, 1)
                    
                    databases.append(db_name)
                    tables.append(table_name)
                    severity_scores.append(score)
            
            if not severity_scores:
                logger.warning("No data available for heatmap generation")
                return
            
            # Create pivot table for heatmap
            df = pd.DataFrame({
                'Database': databases,
                'Table': tables,
                'Severity_Score': severity_scores
            })
            
            pivot_df = df.pivot_table(
                index='Table', 
                columns='Database', 
                values='Severity_Score', 
                fill_value=0
            )
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_df,
                annot=True,
                cmap='RdYlBu_r',
                center=0,
                fmt='.0f',
                cbar_kws={'label': 'Integrity Issue Severity Score'}
            )
            
            plt.title('Data Integrity Issues Heat Map\n(Higher scores indicate more severe issues)')
            plt.xlabel('Database')
            plt.ylabel('Table')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            heatmap_path = output_dir / f'integrity_heatmap_{timestamp}.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Heat map visualization saved to: {heatmap_path}")
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping heatmap generation")
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")

    def generate_html_report(self, output_dir: Path, timestamp: str, json_report: Dict[str, Any]) -> None:
        """Generate HTML summary report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Integrity Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .severity-critical {{ color: #d73027; }}
                .severity-high {{ color: #fc8d59; }}
                .severity-medium {{ color: #fee08b; }}
                .severity-low {{ color: #91d5ff; }}
                .severity-info {{ color: #99d8c9; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #e6f3ff; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Integrity Analysis Report</h1>
                <p><strong>Analysis Date:</strong> {json_report['analysis_timestamp']}</p>
                <p><strong>Databases Analyzed:</strong> {json_report['databases_analyzed']}</p>
            </div>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
        """
        
        summary = json_report['summary']
        html_content += f"""
                    <tr><td>Total Databases</td><td>{summary['total_databases']}</td></tr>
                    <tr><td>Total Tables</td><td>{summary['total_tables']}</td></tr>
                    <tr><td>Total Records</td><td>{summary['total_records']:,}</td></tr>
                </table>
                
                <h3>Severity Distribution</h3>
                <table>
                    <tr><th>Severity</th><th>Count</th></tr>
        """
        
        for severity, count in summary['severity_distribution'].items():
            severity_class = f"severity-{severity.lower()}"
            html_content += f'<tr><td class="{severity_class}">{severity}</td><td>{count}</td></tr>'
        
        html_content += """
                </table>
                
                <h3>Database Health Scores</h3>
                <table>
                    <tr><th>Database</th><th>Health Score</th></tr>
        """
        
        for db_path, score in summary['database_health_scores'].items():
            db_name = Path(db_path).name
            color = 'green' if score > 80 else 'orange' if score > 60 else 'red'
            html_content += f'<tr><td>{db_name}</td><td style="color: {color};">{score}/100</td></tr>'
        
        html_content += """
                </table>
            </div>
            
            <div class="recommendations">
                <h2>Recommendations</h2>
        """
        
        for rec in json_report['recommendations']:
            priority_color = 'red' if rec['priority'] == 'HIGH' else 'orange' if rec['priority'] == 'MEDIUM' else 'blue'
            html_content += f"""
                <div class="recommendation">
                    <h4 style="color: {priority_color};">[{rec['priority']}] {rec['category']}</h4>
                    <p><strong>Issue:</strong> {rec['issue']}</p>
                    <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                    <p><strong>Technical Details:</strong> {rec['technical_details']}</p>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        html_path = output_dir / f'integrity_report_{timestamp}.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {html_path}")

    def print_console_summary(self, summary: Dict[str, Any]) -> None:
        """Print a summary to console"""
        
        print("\n" + "="*80)
        print("DATA INTEGRITY ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 OVERVIEW:")
        print(f"   • Databases Analyzed: {summary['total_databases']}")
        print(f"   • Total Tables: {summary['total_tables']}")
        print(f"   • Total Records: {summary['total_records']:,}")
        
        print(f"\n🚨 SEVERITY DISTRIBUTION:")
        for severity, count in summary['severity_distribution'].items():
            icon = "🔴" if severity == "CRITICAL" else "🟠" if severity == "HIGH" else "🟡" if severity == "MEDIUM" else "🔵" if severity == "LOW" else "⚪"
            print(f"   {icon} {severity}: {count}")
        
        print(f"\n💯 DATABASE HEALTH SCORES:")
        for db_path, score in summary['database_health_scores'].items():
            db_name = Path(db_path).name
            status = "🟢 Excellent" if score > 90 else "🟡 Good" if score > 70 else "🟠 Fair" if score > 50 else "🔴 Poor"
            print(f"   • {db_name}: {score}/100 {status}")
        
        print(f"\n🔍 TOP ISSUE TYPES:")
        for issue_type, count in list(summary['issue_types'].items())[:5]:
            print(f"   • {issue_type}: {count} occurrences")
        
        print("\n" + "="*80)


def main():
    """Main execution function"""
    
    # Define database paths to analyze
    database_paths = [
        './databases/race_data.db',
        './databases/greyhound_racing.db',
        './databases/comprehensive_greyhound_data.db',
        './databases/unified_racing.db',
        './databases/unified_data.db',
        './race_data.db',
        'greyhound_racing_data.db'  # From app.py configuration
    ]
    
    # Filter to only existing databases
    existing_databases = [path for path in database_paths if os.path.exists(path)]
    
    if not existing_databases:
        logger.error("No database files found! Please check database paths.")
        return
    
    logger.info(f"Found {len(existing_databases)} database files to analyze")
    
    # Initialize analyzer
    analyzer = CoreDataIntegrityAnalyzer(existing_databases)
    
    # Run comprehensive analysis
    results = analyzer.analyze_all_databases()
    
    logger.info("Data integrity analysis completed!")
    logger.info("Check the './integrity_analysis_reports' directory for detailed reports.")


if __name__ == "__main__":
    main()
