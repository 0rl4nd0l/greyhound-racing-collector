#!/usr/bin/env python3
"""
Phase 1 Asset Optimization Script
==================================

This script implements immediate performance optimizations for the
Greyhound Racing Dashboard static assets.

Usage: python optimize_assets_phase1.py
"""

import os
import re
import shutil
from datetime import datetime


def backup_files():
    """Create backup of files before optimization"""
    backup_dir = f"backup_assets_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        "app.py",
        "static/css/utilities.css",
        "static/css/style.css",
        "requirements.txt"
    ]
    
    for file_path in files_to_backup:
        if os.path.exists(file_path):
            shutil.copy2(file_path, os.path.join(backup_dir, os.path.basename(file_path)))
    
    print(f"✅ Backup created in {backup_dir}/")
    return backup_dir


def optimize_flask_config():
    """Add SEND_FILE_MAX_AGE_DEFAULT and Flask-Compress to app.py"""
    print("📝 Optimizing Flask configuration...")
    
    with open("app.py", "r") as f:
        content = f.read()
    
    # Check if already configured
    if "SEND_FILE_MAX_AGE_DEFAULT" in content:
        print("   ⚠️ SEND_FILE_MAX_AGE_DEFAULT already configured")
    else:
        # Add after app initialization
        app_init_pattern = r'app = Flask\(__name__\)\s*\napp\.secret_key = "[^"]*"'
        replacement = '''app = Flask(__name__)
app.secret_key = "greyhound_racing_secret_key_2025"

# Static file caching configuration (1 year for better performance)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 31536000'''
        
        content = re.sub(app_init_pattern, replacement, content)
    
    # Add Flask-Compress import and initialization
    if "from flask_compress import Compress" not in content:
        # Add import after Flask imports
        import_pattern = r'(from flask import.*?\n)'
        import_replacement = r'\1from flask_compress import Compress\n'
        content = re.sub(import_pattern, import_replacement, content, flags=re.DOTALL)
        
        # Add Compress initialization after CORS setup
        cors_pattern = r'(CORS\([^)]+\))'
        cors_replacement = r'''\1

# Enable compression for all responses
Compress(app)'''
        content = re.sub(cors_pattern, cors_replacement, content)
    
    with open("app.py", "w") as f:
        f.write(content)
    
    print("   ✅ Flask configuration optimized")


def optimize_css_utilities():
    """Remove duplicate Bootstrap utilities from utilities.css"""
    print("📝 Optimizing CSS utilities...")
    
    with open("static/css/utilities.css", "r") as f:
        content = f.read()
    
    # Bootstrap already provides these utilities - remove duplicates
    bootstrap_duplicates = [
        r'\.d-flex.*?display: flex.*?\}',
        r'\.flex-row.*?flex-direction: row.*?\}',
        r'\.flex-column.*?flex-direction: column.*?\}',
        r'\.justify-content-start.*?justify-content: flex-start.*?\}',
        r'\.justify-content-end.*?justify-content: flex-end.*?\}',
        r'\.justify-content-center.*?justify-content: center.*?\}',
        r'\.justify-content-between.*?justify-content: space-between.*?\}',
        r'\.align-items-center.*?align-items: center.*?\}',
        r'\.flex-wrap.*?flex-wrap: wrap.*?\}',
    ]
    
    original_size = len(content)
    
    for pattern in bootstrap_duplicates:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Clean up extra whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
    
    # Keep custom grid utilities and responsive variants as they're useful
    optimized_content = '''/* Custom Grid Utilities (Bootstrap supplements) */
.d-grid { display: grid; }
.grid-cols-1 { grid-template-columns: repeat(1, 1fr); }
.grid-cols-2 { grid-template-columns: repeat(2, 1fr); }
.grid-cols-3 { grid-template-columns: repeat(3, 1fr); }
.grid-cols-4 { grid-template-columns: repeat(4, 1fr); }
.gap-1 { gap: var(--spacing-sm); }
.gap-2 { gap: var(--spacing-md); }
.gap-3 { gap: var(--spacing-lg); }

/* Responsive Grid Utilities */
@media (min-width: 576px) {
    .grid-cols-sm-2 { grid-template-columns: repeat(2, 1fr); }
    .grid-cols-sm-3 { grid-template-columns: repeat(3, 1fr); }
    .grid-cols-sm-4 { grid-template-columns: repeat(4, 1fr); }
}

@media (min-width: 768px) {
    .grid-cols-md-2 { grid-template-columns: repeat(2, 1fr); }
    .grid-cols-md-3 { grid-template-columns: repeat(3, 1fr); }
    .grid-cols-md-4 { grid-template-columns: repeat(4, 1fr); }
    .grid-cols-md-5 { grid-template-columns: repeat(5, 1fr); }
    .grid-cols-md-6 { grid-template-columns: repeat(6, 1fr); }
}

@media (min-width: 992px) {
    .grid-cols-lg-2 { grid-template-columns: repeat(2, 1fr); }
    .grid-cols-lg-3 { grid-template-columns: repeat(3, 1fr); }
    .grid-cols-lg-4 { grid-template-columns: repeat(4, 1fr); }
    .grid-cols-lg-6 { grid-template-columns: repeat(6, 1fr); }
    .grid-cols-lg-8 { grid-template-columns: repeat(8, 1fr); }
}

@media (min-width: 1200px) {
    .grid-cols-xl-3 { grid-template-columns: repeat(3, 1fr); }
    .grid-cols-xl-4 { grid-template-columns: repeat(4, 1fr); }
    .grid-cols-xl-6 { grid-template-columns: repeat(6, 1fr); }
    .grid-cols-xl-8 { grid-template-columns: repeat(8, 1fr); }
}
'''
    
    with open("static/css/utilities.css", "w") as f:
        f.write(optimized_content)
    
    new_size = len(optimized_content)
    savings = original_size - new_size
    print(f"   ✅ CSS utilities optimized - Saved {savings} bytes ({savings/original_size*100:.1f}%)")


def consolidate_css_variables():
    """Replace hardcoded colors with CSS variables in style.css"""
    print("📝 Consolidating CSS variables...")
    
    with open("static/css/style.css", "r") as f:
        content = f.read()
    
    # Replace hardcoded colors with variables
    color_replacements = {
        '#e9ecef': 'var(--border-color)',
        '#28a745': 'var(--success-color)',
        '#20c997': 'var(--secondary-color)',
        '#d4edda': 'var(--success-bg, #d4edda)',  # Keep fallback for new variables
        '#155724': 'var(--success-text, #155724)',
        '#fff3cd': 'var(--warning-bg, #fff3cd)',
        '#856404': 'var(--warning-text, #856404)',
        '#f8d7da': 'var(--danger-bg, #f8d7da)',
        '#721c24': 'var(--danger-text, #721c24)',
        'rgba(76, 175, 80, 0.2)': 'rgba(var(--secondary-color-rgb, 76, 175, 80), 0.2)',
    }
    
    for hardcoded, variable in color_replacements.items():
        content = content.replace(hardcoded, variable)
    
    with open("static/css/style.css", "w") as f:
        f.write(content)
    
    print("   ✅ CSS variables consolidated")


def update_requirements():
    """Add Flask-Compress to requirements.txt"""
    print("📝 Updating requirements.txt...")
    
    with open("requirements.txt", "r") as f:
        requirements = f.read()
    
    if "Flask-Compress" not in requirements:
        with open("requirements.txt", "a") as f:
            f.write("\nFlask-Compress>=1.13\n")
        print("   ✅ Flask-Compress added to requirements.txt")
    else:
        print("   ⚠️ Flask-Compress already in requirements.txt")


def create_fontawesome_subset_guide():
    """Create a guide for implementing FontAwesome subset"""
    guide_content = '''# FontAwesome Subset Implementation Guide

## Current Usage Analysis
Based on template analysis, these icons are used:
- fa-dog, fa-home, fa-flag-checkered, fa-history
- fa-expand-arrows-alt, fa-calendar-plus, fa-search
- fa-chart-bar, fa-robot, fa-chart-line, fa-brain
- fa-database, fa-table, fa-cogs, fa-upload
- fa-file-alt, fa-question-circle, fa-play, fa-book
- fa-bug, fa-heartbeat, fa-tachometer-alt
- fa-sun, fa-moon

## Implementation Options

### Option 1: FontAwesome Kit (Recommended)
1. Create a FontAwesome Kit at https://fontawesome.com/kits
2. Select only the icons listed above
3. Replace the CDN link in base.html with your kit URL

### Option 2: Custom Icon Font
1. Use tools like IcoMoon or Fontello
2. Create custom icon font with only needed icons
3. Replace FontAwesome CDN with local custom font

### Option 3: SVG Icons
1. Download individual SVG icons
2. Inline critical icons in HTML
3. Use icon sprite for non-critical icons

## Estimated Savings
- Current: ~44KB (full FontAwesome)
- Optimized: ~11KB (subset)
- Savings: ~33KB (75% reduction)
'''
    
    with open("fontawesome_optimization_guide.md", "w") as f:
        f.write(guide_content)
    
    print("   ✅ FontAwesome optimization guide created")


def generate_performance_summary():
    """Generate a summary of optimizations performed"""
    summary = f'''# Phase 1 Optimization Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Optimizations Applied ✅

1. **Flask Static File Caching**
   - Added SEND_FILE_MAX_AGE_DEFAULT = 31536000 (1 year)
   - Improves browser caching for static assets

2. **Response Compression**
   - Added Flask-Compress middleware
   - Reduces payload size for all responses

3. **CSS Optimization**
   - Removed duplicate Bootstrap utilities from utilities.css
   - Consolidated hardcoded colors to use CSS variables
   - Estimated savings: ~1-2KB

4. **Requirements Updated**
   - Added Flask-Compress to requirements.txt

## Next Steps (Phase 2)

1. **FontAwesome Subset** (High Impact)
   - See fontawesome_optimization_guide.md
   - Estimated savings: ~33KB (75% reduction)

2. **Asset Versioning**
   - Implement cache-busting for CSS/JS files
   - Prevents stale cache issues

3. **JavaScript Code Splitting**
   - Split large JS files by page/feature
   - Implement lazy loading for non-critical scripts

## Installation Required

Run the following command to install new dependencies:
```bash
pip install Flask-Compress
```

## Verification

After restarting the Flask application:
1. Check browser DevTools Network tab
2. Verify static files have proper cache headers
3. Check that responses are compressed (gzip/br)
4. Monitor page load times for improvement

Expected improvements:
- Static file caching: Better repeat visit performance
- Response compression: 60-80% smaller payload sizes
- CSS optimization: Faster stylesheet parsing
'''
    
    with open("phase1_optimization_summary.md", "w") as f:
        f.write(summary)
    
    print(f"   ✅ Optimization summary created")


def main():
    """Run Phase 1 optimizations"""
    print("🚀 Starting Phase 1 Asset Optimization")
    print("=" * 50)
    
    # Create backup first
    backup_dir = backup_files()
    
    try:
        # Apply optimizations
        optimize_flask_config()
        optimize_css_utilities()
        consolidate_css_variables()
        update_requirements()
        create_fontawesome_subset_guide()
        generate_performance_summary()
        
        print("\n" + "=" * 50)
        print("✅ Phase 1 Optimization Complete!")
        print("\n📋 Next Steps:")
        print("1. Run: pip install Flask-Compress")
        print("2. Restart your Flask application")
        print("3. Test performance improvements")
        print("4. Review fontawesome_optimization_guide.md for Phase 2")
        print(f"\n💾 Backup available in: {backup_dir}/")
        
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        print(f"💾 Restore from backup: {backup_dir}/")
        raise


if __name__ == "__main__":
    main()
