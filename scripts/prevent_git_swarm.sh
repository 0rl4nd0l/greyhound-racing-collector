#!/bin/bash
# Git Swarm Prevention Maintenance Script

echo "🛡️  Git Swarm Prevention System"
echo "================================"

# Function to check current Git status
check_git_status() {
    file_count=$(git status --porcelain | wc -l)
    echo "📊 Current Git status: $file_count files"
    
    if [ "$file_count" -gt 500 ]; then
        echo "⚠️  WARNING: High file count detected!"
        echo "🔧 Recommended actions:"
        echo "   1. Review files with: git status --porcelain | head -20"
        echo "   2. Update .gitignore if needed"
        echo "   3. Consider 'git reset' for unstaged files"
        return 1
    elif [ "$file_count" -gt 100 ]; then
        echo "⚠️  CAUTION: Moderate file count"
        return 1
    else
        echo "✅ Git status looks healthy"
        return 0
    fi
}

# Function to clean up problematic files
cleanup_problematic_files() {
    echo "🧹 Cleaning up problematic files..."
    
    # Remove common problematic files/dirs
    rm -rf .venv311/ .venv312/ node_modules/ __pycache__/ 2>/dev/null
    find . -name "*.pyc" -delete 2>/dev/null
    find . -name "*.log" -size +1M -delete 2>/dev/null
    find . -name "*.pid" -delete 2>/dev/null
    find . -name ".DS_Store" -delete 2>/dev/null
    
    echo "✅ Cleanup completed"
}

# Function to validate .gitignore
validate_gitignore() {
    echo "🔍 Validating .gitignore..."
    
    required_patterns=(
        ".venv*/"
        "node_modules/"
        "__pycache__/"
        "*.db*"
        "*.log"
        "*.pid"
        "archive/"
        "processed/"
        "upcoming_races_temp/"
    )
    
    missing_patterns=()
    for pattern in "${required_patterns[@]}"; do
        if ! grep -q "^${pattern//\*/\\*}$" .gitignore; then
            missing_patterns+=("$pattern")
        fi
    done
    
    if [ ${#missing_patterns[@]} -gt 0 ]; then
        echo "⚠️  Missing .gitignore patterns:"
        printf '   %s\n' "${missing_patterns[@]}"
        return 1
    else
        echo "✅ .gitignore looks comprehensive"
        return 0
    fi
}

# Main execution
echo "Starting Git swarm prevention check..."
echo

check_git_status
git_status_result=$?

validate_gitignore
gitignore_result=$?

if [ $git_status_result -ne 0 ] || [ $gitignore_result -ne 0 ]; then
    echo
    echo "🚨 Issues detected! Run cleanup? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cleanup_problematic_files
        echo "♻️  Re-checking after cleanup..."
        check_git_status
    fi
fi

echo
echo "📝 Prevention measures active:"
echo "   ✅ Enhanced .gitignore (190+ patterns)"
echo "   ✅ Pre-commit hook (1000+ file limit)"
echo "   ✅ Maintenance script available"
echo
echo "🔧 To run manual check: ./scripts/prevent_git_swarm.sh"
echo "🔧 To bypass pre-commit: git commit --no-verify"
