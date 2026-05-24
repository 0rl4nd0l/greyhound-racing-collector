#!/usr/bin/env python3
"""
Database Usage Analysis
=======================

Analyzes how database connections are made throughout the codebase to determine
if the database routing is applied universally or if there are inconsistencies.
"""

import os
import re
from pathlib import Path

def analyze_database_usage():
    """Analyze database connection patterns across the codebase."""
    
    print("🔍 Analyzing Database Usage Patterns...")
    print("=" * 60)
    
    # Pattern categories to search for
    patterns = {
        'Direct sqlite3.connect': [
            r'sqlite3\.connect\s*\(\s*["\']([^"\']+)["\']',
            r'sqlite3\.connect\s*\(\s*([A-Z_]+)\s*\)',
            r'sqlite3\.connect\s*\(\s*app\.config\.get\(["\']([^"\']+)["\']',
        ],
        'Database routing functions': [
            r'open_sqlite_readonly\s*\(',
            r'open_sqlite_writable\s*\(',
            r'get_analytics_db_path\s*\(',
            r'get_staging_db_path\s*\(',
        ],
        'Config-based connections': [
            r'app\.config\.get\s*\(\s*["\']DATABASE_PATH["\']',
            r'DATABASE_PATH',
            r'GREYHOUND_DB_PATH',
            r'ANALYTICS_DB_PATH',
            r'STAGING_DB_PATH',
        ]
    }
    
    # Files to analyze
    code_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules', 'htmlcov']):
            continue
            
        for file in files:
            if file.endswith(('.py', '.js', '.ts')):
                code_files.append(os.path.join(root, file))
    
    results = {}
    
    for category, pattern_list in patterns.items():
        results[category] = {}
        for pattern in pattern_list:
            results[category][pattern] = []
            
            for file_path in code_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\\n') + 1
                        context = content[max(0, match.start()-50):match.end()+50].replace('\\n', ' ')
                        
                        results[category][pattern].append({
                            'file': file_path,
                            'line': line_num,
                            'match': match.group(),
                            'context': context.strip()
                        })
                        
                except Exception:
                    continue
    
    # Generate report
    print("📊 DATABASE CONNECTION ANALYSIS REPORT")
    print("=" * 60)
    
    total_direct_connections = 0
    total_routed_connections = 0
    
    for category, pattern_data in results.items():
        print(f"\\n🔍 {category.upper()}")
        print("-" * 50)
        
        category_count = 0
        for pattern, matches in pattern_data.items():
            if matches:
                print(f"\\n  Pattern: {pattern}")
                print(f"  Matches: {len(matches)}")
                category_count += len(matches)
                
                # Show first few matches
                for i, match in enumerate(matches[:3]):
                    file_short = match['file'].replace('./', '')
                    print(f"    {i+1}. {file_short}:{match['line']} - {match['match']}")
                
                if len(matches) > 3:
                    print(f"    ... and {len(matches) - 3} more matches")
        
        print(f"\\n  📈 Category total: {category_count} matches")
        
        if 'Direct sqlite3.connect' in category:
            total_direct_connections = category_count
        elif 'Database routing functions' in category:
            total_routed_connections = category_count
    
    # Analysis summary
    print(f"\\n📋 SUMMARY ANALYSIS")
    print("=" * 50)
    print(f"Direct database connections: {total_direct_connections}")
    print(f"Routed database connections: {total_routed_connections}")
    
    if total_routed_connections == 0:
        print("❌ NO ROUTING: All connections are direct - routing not applied")
        universally_applied = False
    elif total_direct_connections == 0:
        print("✅ FULL ROUTING: All connections use routing system")
        universally_applied = True
    else:
        print("⚠️  MIXED USAGE: Both direct and routed connections found")
        routing_percentage = (total_routed_connections / (total_direct_connections + total_routed_connections)) * 100
        print(f"Routing adoption: {routing_percentage:.1f}%")
        universally_applied = routing_percentage > 80
    
    # Check specific app.py patterns
    print(f"\\n🎯 UI APPLICATION (app.py) ANALYSIS")
    print("-" * 50)
    
    app_py_path = './app.py'
    if os.path.exists(app_py_path):
        with open(app_py_path, 'r') as f:
            app_content = f.read()
        
        # Check for routing imports
        routing_imports = bool(re.search(r'from scripts\.db_utils import', app_content))
        print(f"Database routing imported: {'✅ Yes' if routing_imports else '❌ No'}")
        
        # Check for fallback functions
        fallback_functions = bool(re.search(r'def get_analytics_db_path', app_content))
        print(f"Fallback functions present: {'✅ Yes' if fallback_functions else '❌ No'}")
        
        # Check DatabaseManager usage
        db_manager_usage = len(re.findall(r'DatabaseManager', app_content))
        print(f"DatabaseManager usage: {db_manager_usage} occurrences")
        
        # Check specific routing usage in endpoints
        endpoint_routing = len(re.findall(r'open_sqlite_readonly|open_sqlite_writable', app_content))
        direct_sqlite = len(re.findall(r'sqlite3\.connect', app_content))
        print(f"Endpoint routing calls: {endpoint_routing}")
        print(f"Direct sqlite3.connect calls: {direct_sqlite}")
        
        if endpoint_routing > direct_sqlite:
            print("✅ UI predominantly uses database routing")
        elif endpoint_routing == 0:
            print("❌ UI does not use database routing")
        else:
            print("⚠️  UI has mixed database connection patterns")
    
    # Environment configuration check
    print(f"\\n⚙️  ENVIRONMENT CONFIGURATION")
    print("-" * 50)
    
    env_file = '.env'
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()
        
        staging_path = re.search(r'STAGING_DB_PATH=(.+)', env_content)
        analytics_path = re.search(r'ANALYTICS_DB_PATH=(.+)', env_content)
        greyhound_path = re.search(r'GREYHOUND_DB_PATH=(.+)', env_content)
        
        print(f"STAGING_DB_PATH: {staging_path.group(1) if staging_path else 'Not set'}")
        print(f"ANALYTICS_DB_PATH: {analytics_path.group(1) if analytics_path else 'Not set'}")
        print(f"GREYHOUND_DB_PATH: {greyhound_path.group(1) if greyhound_path else 'Not set'}")
        
        if staging_path and analytics_path:
            if staging_path.group(1) == analytics_path.group(1):
                print("⚠️  Both staging and analytics point to same database")
            else:
                print("✅ Staging and analytics use separate databases")
    
    print(f"\\n🎯 CONCLUSION")
    print("=" * 50)
    
    if universally_applied:
        print("✅ Database routing is UNIVERSALLY APPLIED")
        print("   The application consistently uses the routing system")
    else:
        print("❌ Database routing is NOT universally applied")
        print("   Mixed patterns found - some direct connections remain")
    
    return universally_applied

if __name__ == "__main__":
    universally_applied = analyze_database_usage()
    
    print(f"\\n🔍 RECOMMENDATION:")
    if universally_applied:
        print("✅ Your database configuration is properly applied throughout the system")
    else:
        print("⚠️  Consider migrating remaining direct connections to use the routing system")
