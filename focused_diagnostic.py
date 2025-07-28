#!/usr/bin/env python3
"""
Focused Greyhound Analysis Predictor Diagnostic
===============================================

Targeted diagnostic focusing on core application files and critical issues.

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import sqlite3
import json
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime
import pandas as pd

class FocusedDiagnostic:
    def __init__(self, base_dir="/Users/orlandolee/greyhound_racing_collector"):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "greyhound_racing_data.db"
        self.issues = []
        
        # Core files to analyze
        self.core_files = [
            "app.py",
            "weather_enhanced_predictor.py", 
            "unified_predictor.py",
            "comprehensive_enhanced_ml_system.py",
            "enhanced_pipeline_v2.py",
            "advanced_ml_system_v2.py",
            "json_utils.py",
            "logger.py"
        ]
        
        print("🎯 Focused Diagnostic for Greyhound Analysis Predictor")
        print("=" * 60)

    def test_database_connectivity(self):
        """Test database connection and basic queries"""
        print("\n🗄️ DATABASE CONNECTIVITY TEST")
        print("-" * 40)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Test basic connection
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            print(f"✅ Database connected: {table_count} tables found")
            
            # Test critical tables
            critical_tables = ['race_metadata', 'dog_race_data']
            for table in critical_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    print(f"✅ Table {table}: {count} records")
                except Exception as e:
                    print(f"❌ Table {table}: {e}")
                    self.issues.append({
                        'type': 'missing_table',
                        'table': table,
                        'severity': 'critical'
                    })
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            self.issues.append({
                'type': 'database_connection_error',
                'error': str(e),
                'severity': 'critical'
            })

    def test_core_imports(self):
        """Test importing core modules"""
        print("\n📦 CORE MODULE IMPORT TEST")
        print("-" * 40)
        
        # Add project to path
        sys.path.insert(0, str(self.base_dir))
        
        for file_name in self.core_files:
            file_path = self.base_dir / file_name
            if not file_path.exists():
                print(f"⚠️  {file_name}: File not found")
                continue
            
            try:
                module_name = file_name.replace('.py', '')
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    print(f"✅ {file_name}: Import successful")
                else:
                    print(f"❌ {file_name}: Could not create spec")
                    
            except Exception as e:
                print(f"❌ {file_name}: {str(e)[:100]}...")
                self.issues.append({
                    'type': 'import_error',
                    'file': file_name,
                    'error': str(e),
                    'severity': 'high'
                })

    def test_flask_app(self):
        """Test Flask application"""
        print("\n🌐 FLASK APPLICATION TEST")
        print("-" * 40)
        
        try:
            sys.path.insert(0, str(self.base_dir))
            import app
            
            if hasattr(app, 'app'):
                flask_app = app.app
                print("✅ Flask app instance found")
                
                # Test app configuration
                with flask_app.test_client() as client:
                    flask_app.config['TESTING'] = True
                    
                    # Test basic routes
                    test_routes = [
                        ('/', 'GET'),
                        ('/races', 'GET'),
                        ('/api/stats', 'GET')
                    ]
                    
                    for route, method in test_routes:
                        try:
                            if method == 'GET':
                                response = client.get(route)
                            else:
                                response = client.post(route)
                            
                            print(f"   {method:4} {route:20} -> {response.status_code}")
                            
                            if response.status_code >= 500:
                                self.issues.append({
                                    'type': 'endpoint_error',
                                    'path': route,
                                    'status_code': response.status_code,
                                    'severity': 'high'
                                })
                                
                        except Exception as e:
                            print(f"   {method:4} {route:20} -> ERROR: {str(e)[:50]}...")
                            self.issues.append({
                                'type': 'endpoint_test_error',
                                'path': route,
                                'error': str(e),
                                'severity': 'medium'
                            })
            else:
                print("❌ Flask app instance not found")
                self.issues.append({
                    'type': 'flask_app_not_found',
                    'severity': 'critical'
                })
                
        except Exception as e:
            print(f"❌ Flask import failed: {e}")
            self.issues.append({
                'type': 'flask_import_error',
                'error': str(e),
                'severity': 'critical'
            })

    def test_prediction_pipeline(self):
        """Test prediction components"""
        print("\n🎯 PREDICTION PIPELINE TEST")
        print("-" * 40)
        
        # Test weather enhanced predictor
        try:
            sys.path.insert(0, str(self.base_dir))
            from weather_enhanced_predictor import WeatherEnhancedPredictor
            
            predictor = WeatherEnhancedPredictor()
            print("✅ WeatherEnhancedPredictor: Initialization successful")
            
            # Test with sample file
            sample_files = list(self.base_dir.glob("upcoming_races/*.csv"))
            if sample_files:
                sample_file = sample_files[0]
                print(f"   Testing with: {sample_file.name}")
                
                # This should work if everything is properly set up
                # result = predictor.predict_race(str(sample_file))
                print("   ⚠️  Prediction test skipped (would require full setup)")
            else:
                print("   ⚠️  No sample files found for testing")
                
        except Exception as e:
            print(f"❌ WeatherEnhancedPredictor: {str(e)[:100]}...")
            self.issues.append({
                'type': 'predictor_error',
                'component': 'WeatherEnhancedPredictor',
                'error': str(e),
                'severity': 'high'
            })

    def test_dependencies(self):
        """Test critical dependencies"""
        print("\n📚 DEPENDENCY TEST")
        print("-" * 40)
        
        critical_deps = [
            'pandas', 'numpy', 'sklearn', 'flask', 'sqlite3', 
            'json', 'datetime', 'pathlib', 'logging'
        ]
        
        for dep in critical_deps:
            try:
                __import__(dep)
                print(f"✅ {dep}")
            except ImportError as e:
                print(f"❌ {dep}: {e}")
                self.issues.append({
                    'type': 'missing_dependency',
                    'dependency': dep,
                    'severity': 'high'
                })

    def analyze_data_quality(self):
        """Analyze data quality issues"""
        print("\n📊 DATA QUALITY ANALYSIS")
        print("-" * 40)
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check for missing winners
            missing_winners = pd.read_sql_query("""
                SELECT COUNT(*) as count FROM race_metadata 
                WHERE winner_name IS NULL OR winner_name = '' OR winner_name = 'nan'
            """, conn)
            
            count = missing_winners.iloc[0]['count']
            print(f"Races missing winners: {count}")
            if count > 0:
                self.issues.append({
                    'type': 'missing_race_winners',
                    'count': count,
                    'severity': 'medium'
                })
            
            # Check for missing box numbers
            missing_boxes = pd.read_sql_query("""
                SELECT COUNT(*) as count FROM dog_race_data 
                WHERE box_number IS NULL
            """, conn)
            
            count = missing_boxes.iloc[0]['count']
            print(f"Dog entries missing box numbers: {count}")
            if count > 0:
                self.issues.append({
                    'type': 'missing_box_numbers',
                    'count': count,
                    'severity': 'medium'
                })
            
            # Check data freshness
            recent_races = pd.read_sql_query("""
                SELECT COUNT(*) as count FROM race_metadata 
                WHERE race_date >= date('now', '-7 days')
            """, conn)
            
            count = recent_races.iloc[0]['count']
            print(f"Recent races (last 7 days): {count}")
            if count == 0:
                self.issues.append({
                    'type': 'stale_data',
                    'severity': 'low'
                })
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Data quality analysis failed: {e}")
            self.issues.append({
                'type': 'data_analysis_error',
                'error': str(e),
                'severity': 'medium'
            })

    def check_file_structure(self):
        """Check critical file structure"""
        print("\n📁 FILE STRUCTURE CHECK")
        print("-" * 40)
        
        critical_dirs = [
            'upcoming_races',
            'predictions', 
            'processed',
            'logs',
            'templates',
            'static'
        ]
        
        for dir_name in critical_dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                file_count = len(list(dir_path.glob("*")))
                print(f"✅ {dir_name}/: {file_count} files")
            else:
                print(f"❌ {dir_name}/: Missing")
                self.issues.append({
                    'type': 'missing_directory',
                    'directory': dir_name,
                    'severity': 'medium'
                })

    def generate_repair_recommendations(self):
        """Generate specific repair recommendations"""
        print("\n🔧 REPAIR RECOMMENDATIONS")
        print("-" * 40)
        
        # Group issues by type
        issue_types = {}
        for issue in self.issues:
            issue_type = issue['type']
            if issue_type not in issue_types:
                issue_types[issue_type] = []
            issue_types[issue_type].append(issue)
        
        recommendations = []
        
        # Database issues
        if 'database_connection_error' in issue_types:
            recommendations.append({
                'priority': 1,
                'title': 'Fix Database Connection',
                'description': 'Database connection is failing',
                'action': 'Check database file exists and permissions are correct'
            })
        
        if 'missing_table' in issue_types:
            recommendations.append({
                'priority': 1,
                'title': 'Restore Missing Tables',
                'description': f"Missing tables: {[i['table'] for i in issue_types['missing_table']]}",
                'action': 'Run database schema recreation script'
            })
        
        # Import issues
        if 'import_error' in issue_types:
            recommendations.append({
                'priority': 2,
                'title': 'Fix Import Errors',
                'description': f"Files with import errors: {len(issue_types['import_error'])}",
                'action': 'Install missing dependencies or fix import paths'
            })
        
        # Flask issues
        if 'flask_app_not_found' in issue_types:
            recommendations.append({
                'priority': 1,
                'title': 'Fix Flask Application',
                'description': 'Flask app instance not found',
                'action': 'Check app.py structure and Flask initialization'
            })
        
        # Endpoint issues
        if 'endpoint_error' in issue_types:
            recommendations.append({
                'priority': 2,
                'title': 'Fix Endpoint Errors',
                'description': f"Endpoints with 5xx errors: {len(issue_types['endpoint_error'])}",
                'action': 'Debug individual endpoint handlers'
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['title']} (Priority {rec['priority']})")
            print(f"   Description: {rec['description']}")
            print(f"   Action: {rec['action']}")
        
        return recommendations

    def run_diagnostic(self):
        """Run focused diagnostic"""
        print("Starting focused diagnostic...\n")
        
        # Run all tests
        self.test_database_connectivity()
        self.test_dependencies()
        self.test_core_imports()
        self.test_flask_app()
        self.test_prediction_pipeline()
        self.analyze_data_quality()
        self.check_file_structure()
        
        # Generate summary
        print(f"\n🎯 DIAGNOSTIC SUMMARY")
        print("=" * 40)
        print(f"Total issues found: {len(self.issues)}")
        
        # Count by severity
        severity_counts = {}
        for issue in self.issues:
            severity = issue['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for severity, count in severity_counts.items():
            print(f"{severity.capitalize()}: {count}")
        
        # Generate recommendations
        recommendations = self.generate_repair_recommendations()
        
        # Save diagnostic report
        report = {
            'timestamp': datetime.now().isoformat(),
            'issues': self.issues,
            'recommendations': recommendations,
            'summary': {
                'total_issues': len(self.issues),
                'severity_breakdown': severity_counts
            }
        }
        
        # Convert any pandas int64 types to native Python int for JSON serialization
        def convert_pandas_types(obj):
            if hasattr(obj, 'dtype') and 'int64' in str(obj.dtype):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_pandas_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_pandas_types(item) for item in obj]
            return obj
        
        report = convert_pandas_types(report)
        
        report_file = self.base_dir / 'diagnostic_logs' / f"focused_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📊 Report saved: {report_file}")
        
        return report

if __name__ == "__main__":
    diagnostic = FocusedDiagnostic()
    diagnostic.run_diagnostic()
