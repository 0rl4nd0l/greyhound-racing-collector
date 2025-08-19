#!/usr/bin/env python3
"""
Greyhound Analysis Predictor - Full Stack Diagnostic System
==========================================================

Comprehensive diagnostic tool to analyze and repair the broken Flask application
after database schema changes and project restructuring.

Author: AI Assistant
Date: July 28, 2025
"""

import os
import sys
import ast
import sqlite3
import json
import traceback
import subprocess
import re
import inspect
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging

class GreyhoundDiagnosticSystem:
    def __init__(self, base_dir="/Users/orlandolee/greyhound_racing_collector"):
        self.base_dir = Path(base_dir)
        self.db_path = self.base_dir / "greyhound_racing_data.db"
        self.logs_dir = self.base_dir / "diagnostic_logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup comprehensive logging
        self.setup_logging()
        
        # Data structures for analysis
        self.routes = {}
        self.models = {}
        self.database_queries = []
        self.imports = defaultdict(list)
        self.functions = {}
        self.classes = {}
        self.issues = []
        
        self.logger.info("🔧 Greyhound Diagnostic System Initialized")
        print("🧠 Starting Full-Stack Diagnostic of Greyhound Analysis Predictor")
        print("=" * 70)
    
    def setup_logging(self):
        """Setup comprehensive logging system"""
        log_file = self.logs_dir / f"diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def scan_python_files(self):
        """Comprehensive scan of all Python files"""
        print("\n🔍 PHASE 1: CODEBASE RECONNAISSANCE")
        print("=" * 50)
        
        python_files = list(self.base_dir.glob("**/*.py"))
        self.logger.info(f"Found {len(python_files)} Python files")
        
        for py_file in python_files:
            try:
                self.analyze_python_file(py_file)
            except Exception as e:
                self.logger.error(f"Error analyzing {py_file}: {e}")
                self.issues.append({
                    'type': 'file_analysis_error',
                    'file': str(py_file),
                    'error': str(e),
                    'severity': 'medium'
                })
        
        self.print_codebase_summary()
    
    def analyze_python_file(self, file_path):
        """Analyze individual Python file using AST"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract information
            for node in ast.walk(tree):
                self.extract_node_info(node, file_path, content)
                
        except SyntaxError as e:
            self.logger.error(f"Syntax error in {file_path}: {e}")
            self.issues.append({
                'type': 'syntax_error',
                'file': str(file_path),
                'error': str(e),
                'severity': 'high'
            })
        except Exception as e:
            self.logger.error(f"Error parsing {file_path}: {e}")
    
    def extract_node_info(self, node, file_path, content):
        """Extract relevant information from AST nodes"""
        relative_path = str(file_path.relative_to(self.base_dir))
        
        # Flask routes
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (isinstance(decorator, ast.Call) and 
                    isinstance(decorator.func, ast.Attribute) and
                    decorator.func.attr == 'route'):
                    
                    route_info = self.extract_route_info(node, decorator, file_path)
                    self.routes[f"{relative_path}:{node.name}"] = route_info
        
        # Class definitions (models)
        elif isinstance(node, ast.ClassDef):
            self.classes[f"{relative_path}:{node.name}"] = {
                'file': relative_path,
                'name': node.name,
                'bases': [self.get_node_name(base) for base in node.bases],
                'methods': [method.name for method in node.body if isinstance(method, ast.FunctionDef)]
            }
        
        # Function definitions
        elif isinstance(node, ast.FunctionDef):
            self.functions[f"{relative_path}:{node.name}"] = {
                'file': relative_path,
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'line': node.lineno
            }
        
        # Import statements
        elif isinstance(node, ast.Import):
            for alias in node.names:
                self.imports[relative_path].append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname
                })
        
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                self.imports[relative_path].append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname
                })
        
        # SQL queries and database operations
        elif isinstance(node, ast.Str):
            if self.contains_sql(node.s):
                self.database_queries.append({
                    'file': relative_path,
                    'query': node.s,
                    'line': node.lineno
                })
    
    def extract_route_info(self, func_node, decorator, file_path):
        """Extract Flask route information"""
        route_path = None
        methods = ['GET']
        
        if decorator.args:
            if isinstance(decorator.args[0], ast.Str):
                route_path = decorator.args[0].s
        
        for keyword in decorator.keywords:
            if keyword.arg == 'methods':
                if isinstance(keyword.value, ast.List):
                    methods = [elt.s for elt in keyword.value.elts if isinstance(elt, ast.Str)]
        
        return {
            'file': str(file_path.relative_to(self.base_dir)),
            'function': func_node.name,
            'path': route_path,
            'methods': methods,
            'line': func_node.lineno
        }
    
    def get_node_name(self, node):
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_node_name(node.value)}.{node.attr}"
        else:
            return str(node)
    
    def contains_sql(self, text):
        """Check if string contains SQL"""
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        return any(keyword in text.upper() for keyword in sql_keywords)
    
    def analyze_database_schema(self):
        """Deep analysis of database schema"""
        print("\n🗄️ PHASE 2: DATABASE DEEP-DIVE")
        print("=" * 50)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_info = {}
            
            for (table_name,) in tables:
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                
                # Get foreign keys
                cursor.execute(f"PRAGMA foreign_key_list({table_name});")
                foreign_keys = cursor.fetchall()
                
                # Get indexes
                cursor.execute(f"PRAGMA index_list({table_name});")
                indexes = cursor.fetchall()
                
                schema_info[table_name] = {
                    'columns': columns,
                    'foreign_keys': foreign_keys,
                    'indexes': indexes
                }
                
                print(f"📊 Table: {table_name}")
                print(f"   Columns: {len(columns)}")
                print(f"   Foreign Keys: {len(foreign_keys)}")
                print(f"   Indexes: {len(indexes)}")
            
            conn.close()
            
            # Validate against code expectations
            self.validate_schema_expectations(schema_info)
            
            return schema_info
            
        except Exception as e:
            self.logger.error(f"Database analysis failed: {e}")
            self.issues.append({
                'type': 'database_connection_error',
                'error': str(e),
                'severity': 'critical'
            })
            return {}
    
    def validate_schema_expectations(self, schema_info):
        """Validate database schema against code expectations"""
        print("\n🔍 Schema Validation:")
        
        # Check for common table expectations based on queries
        expected_tables = set()
        expected_columns = defaultdict(set)
        
        for query_info in self.database_queries:
            query = query_info['query'].upper()
            
            # Extract table names from queries (basic parsing)
            table_matches = re.findall(r'FROM\s+(\w+)', query)
            expected_tables.update(table_matches)
            
            # Extract column references
            column_matches = re.findall(r'(\w+)\s*=', query)
            for table in table_matches:
                expected_columns[table].update(column_matches)
        
        # Validate expectations
        actual_tables = set(schema_info.keys())
        missing_tables = expected_tables - actual_tables
        
        if missing_tables:
            for table in missing_tables:
                self.issues.append({
                    'type': 'missing_table',
                    'table': table,
                    'severity': 'high'
                })
                print(f"   ❌ Missing table: {table}")
        
        # Validate columns
        for table, expected_cols in expected_columns.items():
            if table in schema_info:
                actual_cols = {col[1] for col in schema_info[table]['columns']}
                missing_cols = expected_cols - actual_cols
                
                if missing_cols:
                    for col in missing_cols:
                        self.issues.append({
                            'type': 'missing_column',
                            'table': table,
                            'column': col,
                            'severity': 'medium'
                        })
                        print(f"   ⚠️ Missing column: {table}.{col}")
    
    def test_flask_endpoints(self):
        """Test all Flask endpoints"""
        print("\n🌐 PHASE 3: ENDPOINT TESTING")
        print("=" * 50)
        
        try:
            # Import the Flask app
            sys.path.insert(0, str(self.base_dir))
            
            # Try different common app names
            app_candidates = ['app', 'application', 'main']
            flask_app = None
            
            for app_name in app_candidates:
                try:
                    if (self.base_dir / f"{app_name}.py").exists():
                        module = __import__(app_name)
                        if hasattr(module, 'app'):
                            flask_app = module.app
                            break
                except Exception as e:
                    self.logger.debug(f"Could not import {app_name}: {e}")
            
            if not flask_app:
                self.issues.append({
                    'type': 'flask_app_not_found',
                    'severity': 'critical'
                })
                print("❌ Could not locate Flask app instance")
                return
            
            # Test endpoints
            with flask_app.test_client() as client:
                flask_app.config['TESTING'] = True
                
                for route_key, route_info in self.routes.items():
                    if route_info['path']:
                        self.test_endpoint(client, route_info)
        
        except Exception as e:
            self.logger.error(f"Flask testing failed: {e}")
            self.issues.append({
                'type': 'flask_testing_error',
                'error': str(e),
                'severity': 'high'
            })
    
    def test_endpoint(self, client, route_info):
        """Test individual endpoint"""
        try:
            path = route_info['path']
            methods = route_info['methods']
            
            for method in methods:
                if method in ['GET', 'POST']:
                    print(f"   Testing {method} {path}")
                    
                    if method == 'GET':
                        response = client.get(path)
                    else:
                        response = client.post(path, json={})
                    
                    print(f"     Status: {response.status_code}")
                    
                    if response.status_code >= 500:
                        self.issues.append({
                            'type': 'endpoint_error',
                            'path': path,
                            'method': method,
                            'status_code': response.status_code,
                            'severity': 'high'
                        })
                    elif response.status_code >= 400:
                        self.issues.append({
                            'type': 'endpoint_client_error',
                            'path': path,
                            'method': method,
                            'status_code': response.status_code,
                            'severity': 'medium'
                        })
        
        except Exception as e:
            self.logger.error(f"Error testing {route_info['path']}: {e}")
            self.issues.append({
                'type': 'endpoint_test_error',
                'path': route_info['path'],
                'error': str(e),
                'severity': 'medium'
            })
    
    def analyze_prediction_pipeline(self):
        """Analyze prediction-specific components"""
        print("\n🎯 PHASE 4: PREDICTION PIPELINE ANALYSIS")
        print("=" * 50)
        
        # Look for prediction-related files
        prediction_files = []
        for py_file in self.base_dir.glob("**/*.py"):
            if any(keyword in py_file.name.lower() for keyword in 
                   ['predict', 'model', 'ml', 'weather', 'enhanced']):
                prediction_files.append(py_file)
        
        print(f"Found {len(prediction_files)} prediction-related files:")
        
        for file_path in prediction_files:
            print(f"   📄 {file_path.relative_to(self.base_dir)}")
            self.analyze_prediction_file(file_path)
    
    def analyze_prediction_file(self, file_path):
        """Analyze individual prediction file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for model loading patterns
            model_patterns = [
                r'joblib\.load',
                r'pickle\.load',
                r'load_model',
                r'\.pkl',
                r'\.joblib'
            ]
            
            for pattern in model_patterns:
                if re.search(pattern, content):
                    print(f"     🤖 Found model loading pattern: {pattern}")
            
            # Check for database queries
            if 'sqlite3' in content or 'pd.read_sql' in content:
                print(f"     🗄️ Database operations detected")
            
            # Test importability
            try:
                relative_path = file_path.relative_to(self.base_dir)
                module_name = str(relative_path).replace('/', '.').replace('.py', '')
                
                # Skip if in venv or similar
                if 'venv' not in str(relative_path) and 'env' not in str(relative_path):
                    sys.path.insert(0, str(self.base_dir))
                    __import__(module_name)
                    print(f"     ✅ Module imports successfully")
            except Exception as e:
                print(f"     ❌ Import error: {e}")
                self.issues.append({
                    'type': 'import_error',
                    'file': str(relative_path),
                    'error': str(e),
                    'severity': 'high'
                })
        
        except Exception as e:
            self.logger.error(f"Error analyzing prediction file {file_path}: {e}")
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n📋 PHASE 5: DIAGNOSTIC REPORT GENERATION")
        print("=" * 50)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': len(list(self.base_dir.glob("**/*.py"))),
                'routes_found': len(self.routes),
                'classes_found': len(self.classes),
                'functions_found': len(self.functions),
                'database_queries': len(self.database_queries),
                'issues_found': len(self.issues)
            },
            'routes': self.routes,
            'classes': self.classes,
            'database_queries': self.database_queries,
            'issues': self.issues,
            'recommendations': self.generate_recommendations()
        }
        
        # Save report
        report_file = self.logs_dir / f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Diagnostic report saved: {report_file}")
        
        return report
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Group issues by type
        issue_types = defaultdict(int)
        for issue in self.issues:
            issue_types[issue['type']] += 1
        
        # Generate recommendations
        if issue_types.get('missing_table', 0) > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'database',
                'description': 'Missing database tables detected. Run schema migration.',
                'action': 'Create missing tables or update queries to match new schema'
            })
        
        if issue_types.get('import_error', 0) > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'dependencies',
                'description': 'Import errors detected in prediction modules.',
                'action': 'Install missing dependencies or fix import paths'
            })
        
        if issue_types.get('endpoint_error', 0) > 0:
            recommendations.append({
                'priority': 'medium',
                'category': 'endpoints',
                'description': 'Flask endpoints returning server errors.',
                'action': 'Debug endpoint handlers and fix underlying issues'
            })
        
        return recommendations
    
    def print_codebase_summary(self):
        """Print summary of codebase analysis"""
        print(f"\n📊 CODEBASE SUMMARY:")
        print(f"   Routes found: {len(self.routes)}")
        print(f"   Classes found: {len(self.classes)}")
        print(f"   Functions found: {len(self.functions)}")
        print(f"   Database queries: {len(self.database_queries)}")
        
        print(f"\n🛣️ FLASK ROUTES:")
        for route_key, route_info in self.routes.items():
            methods_str = ', '.join(route_info['methods'])
            print(f"   {methods_str:8} {route_info['path']:30} -> {route_info['function']}")
    
    def run_full_diagnostic(self):
        """Run complete diagnostic workflow"""
        try:
            # Phase 1: Codebase reconnaissance
            self.scan_python_files()
            
            # Phase 2: Database analysis
            self.analyze_database_schema()
            
            # Phase 3: Endpoint testing
            self.test_flask_endpoints()
            
            # Phase 4: Prediction pipeline analysis
            self.analyze_prediction_pipeline()
            
            # Phase 5: Generate report
            report = self.generate_diagnostic_report()
            
            print(f"\n🎯 DIAGNOSTIC COMPLETE")
            print(f"   Issues found: {len(self.issues)}")
            print(f"   Critical: {len([i for i in self.issues if i['severity'] == 'critical'])}")
            print(f"   High: {len([i for i in self.issues if i['severity'] == 'high'])}")
            print(f"   Medium: {len([i for i in self.issues if i['severity'] == 'medium'])}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Diagnostic failed: {e}")
            print(f"❌ Diagnostic failed: {e}")
            return None

if __name__ == "__main__":
    diagnostic = GreyhoundDiagnosticSystem()
    diagnostic.run_full_diagnostic()
