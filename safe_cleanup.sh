#!/bin/bash

# Safe Cleanup Script for Greyhound Racing Project
# ===============================================
# This script removes clearly unnecessary files while preserving all active system components
# IMPORTANT: Review ACTIVE_SCRIPTS_GUIDE.md and CLEANUP_ANALYSIS_REPORT.md first!

echo "🧹 SAFE CLEANUP SCRIPT"
echo "======================"
echo ""
echo "This script will remove:"
echo "1. Log files (*.log)"
echo "2. Debug/test files (Race_01_UNKNOWN_*.json)"
echo "3. Empty databases (0 byte .db files)"
echo "4. HTML debug files"
echo "5. Large archive directories (after confirmation)"
echo ""
echo "⚠️  IMPORTANT: This will preserve all active scripts and the main database!"
echo ""

# Function to ask for confirmation
confirm() {
    read -p "$1 (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

# Count files to be removed
log_count=$(find . -maxdepth 1 -name "*.log" -type f | wc -l | xargs)
debug_count=$(find . -maxdepth 1 -name "Race_01_UNKNOWN_*.json" -type f | wc -l | xargs)
empty_db_count=$(find . -maxdepth 1 -name "*.db" -size 0 -type f | wc -l | xargs)
html_count=$(find . -maxdepth 1 -name "*.html" -type f | wc -l | xargs)

echo "Files to be removed:"
echo "- Log files: $log_count"
echo "- Debug files: $debug_count" 
echo "- Empty databases: $empty_db_count"
echo "- HTML files: $html_count"
echo ""

# Confirm before starting
if ! confirm "Do you want to proceed with safe cleanup?"; then
    echo "❌ Cleanup cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting cleanup..."

# 1. Remove log files
echo "📋 Removing log files..."
if [ $log_count -gt 0 ]; then
    find . -maxdepth 1 -name "*.log" -type f -delete
    echo "✅ Removed $log_count log files"
else
    echo "ℹ️  No log files to remove"
fi

# 2. Remove debug files
echo "🔍 Removing debug files..."
if [ $debug_count -gt 0 ]; then
    find . -maxdepth 1 -name "Race_01_UNKNOWN_*.json" -type f -delete
    echo "✅ Removed $debug_count debug files"
else
    echo "ℹ️  No debug files to remove"
fi

# 3. Remove empty databases (but preserve the main one!)
echo "🗄️  Removing empty databases..."
if [ $empty_db_count -gt 0 ]; then
    # Explicitly list the empty databases to avoid accidents
    find . -maxdepth 1 -name "enhanced_greyhound_racing.db" -size 0 -type f -delete 2>/dev/null || true
    find . -maxdepth 1 -name "greyhound_data.db" -size 0 -type f -delete 2>/dev/null || true
    find . -maxdepth 1 -name "greyhound_racing.db" -size 0 -type f -delete 2>/dev/null || true
    echo "✅ Removed empty databases"
else
    echo "ℹ️  No empty databases to remove"
fi

# 4. Remove HTML debug files
echo "🌐 Removing HTML debug files..."
if [ $html_count -gt 0 ]; then
    find . -maxdepth 1 -name "*.html" -type f -delete
    echo "✅ Removed $html_count HTML files"
else
    echo "ℹ️  No HTML files to remove"
fi

# 5. Remove some debug text files
echo "📄 Removing debug text files..."
rm -f debug_*.txt 2>/dev/null && echo "✅ Removed debug text files" || echo "ℹ️  No debug text files to remove"

echo ""
echo "🔍 Checking large directories for optional cleanup..."

# Check virtual environments
if [ -d "venv" ]; then
    venv_size=$(du -sh venv 2>/dev/null | cut -f1)
    echo "📁 Found virtual environment: venv ($venv_size)"
    if confirm "Remove venv directory? (can be recreated with pip)"; then
        rm -rf venv
        echo "✅ Removed venv directory"
    fi
fi

if [ -d "ml_env" ]; then
    ml_env_size=$(du -sh ml_env 2>/dev/null | cut -f1)
    echo "📁 Found ML environment: ml_env ($ml_env_size)"
    if confirm "Remove ml_env directory? (can be recreated)"; then
        rm -rf ml_env
        echo "✅ Removed ml_env directory"
    fi
fi

# Check archive directories
if [ -d "cleanup_archive" ]; then
    cleanup_size=$(du -sh cleanup_archive 2>/dev/null | cut -f1)
    echo "📁 Found cleanup archive: cleanup_archive ($cleanup_size)"
    if confirm "Remove cleanup_archive directory? (already archived content)"; then
        rm -rf cleanup_archive
        echo "✅ Removed cleanup_archive directory"
    fi
fi

if [ -d "archive_unused_scripts" ]; then
    archive_size=$(du -sh archive_unused_scripts 2>/dev/null | cut -f1)
    echo "📁 Found script archive: archive_unused_scripts ($archive_size)"
    if confirm "Remove archive_unused_scripts directory? (archived scripts)"; then
        rm -rf archive_unused_scripts
        echo "✅ Removed archive_unused_scripts directory"
    fi
fi

if [ -d "outdated_scripts" ]; then
    outdated_size=$(du -sh outdated_scripts 2>/dev/null | cut -f1)
    echo "📁 Found outdated scripts: outdated_scripts ($outdated_size)"
    if confirm "Remove outdated_scripts directory? (old scripts)"; then
        rm -rf outdated_scripts
        echo "✅ Removed outdated_scripts directory"
    fi
fi

if [ -d "backup_before_cleanup" ]; then
    backup_size=$(du -sh backup_before_cleanup 2>/dev/null | cut -f1)
    echo "📁 Found backup directory: backup_before_cleanup ($backup_size)"
    if confirm "Remove backup_before_cleanup directory? (old backup)"; then
        rm -rf backup_before_cleanup
        echo "✅ Removed backup_before_cleanup directory"
    fi
fi

echo ""
echo "✅ CLEANUP COMPLETE!"
echo ""
echo "📊 Summary:"
echo "✅ All active scripts preserved"
echo "✅ Main database (greyhound_racing_data.db) preserved"
echo "✅ Prediction results preserved"
echo "✅ Web app assets preserved"
echo ""
echo "🔍 Remaining structure:"
ls -la | grep -E "^d|weather_enhanced_predictor|upcoming_race_predictor|app.py|greyhound_racing_data.db" | head -10
echo ""
echo "💡 For detailed active scripts info, see: ACTIVE_SCRIPTS_GUIDE.md"
echo "📋 For full cleanup analysis, see: CLEANUP_ANALYSIS_REPORT.md"
