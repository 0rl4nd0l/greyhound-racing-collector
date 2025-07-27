#!/bin/bash

# Automated Git Backup Script for Greyhound Racing Collector
# This script performs regular backups of the project to Git

set -e  # Exit on any error

# Configuration
PROJECT_DIR="/Users/orlandolee/greyhound_racing_collector"
LOG_FILE="$PROJECT_DIR/logs/git_backup.log"
BACKUP_BRANCH="backup/$(date +%Y-%m-%d)"
MAX_LOG_SIZE=10485760  # 10MB in bytes

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to rotate log file if it gets too large
rotate_log_if_needed() {
    if [ -f "$LOG_FILE" ] && [ $(stat -f%z "$LOG_FILE" 2>/dev/null || echo 0) -gt $MAX_LOG_SIZE ]; then
        mv "$LOG_FILE" "${LOG_FILE}.old"
        log_message "Log file rotated due to size limit"
    fi
}

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_message "ERROR: Not in a git repository"
        exit 1
    fi
}

# Function to check for uncommitted changes
check_for_changes() {
    if git diff-index --quiet HEAD --; then
        log_message "No changes to commit"
        return 1
    fi
    return 0
}

# Function to create a backup branch
create_backup_branch() {
    local branch_name="$1"
    if git show-ref --verify --quiet "refs/heads/$branch_name"; then
        log_message "Backup branch $branch_name already exists, switching to it"
        git checkout "$branch_name"
    else
        log_message "Creating new backup branch: $branch_name"
        git checkout -b "$branch_name"
    fi
}

# Function to perform the backup
perform_backup() {
    local backup_type="$1"
    local commit_message="$2"
    
    # Add all tracked and new files (respecting .gitignore)
    git add .
    
    # Check if there are staged changes
    if git diff-cached --quiet; then
        log_message "No staged changes to commit"
        return 0
    fi
    
    # Get commit statistics
    local files_changed=$(git diff --cached --name-only | wc -l | tr -d ' ')
    local insertions=$(git diff --cached --numstat | awk '{inserted+=$1} END {print inserted+0}')
    local deletions=$(git diff --cached --numstat | awk '{deleted+=$2} END {print deleted+0}')
    
    # Commit changes
    if git commit -m "$commit_message"; then
        log_message "SUCCESS: Committed $files_changed files (+$insertions -$deletions) - $backup_type"
        
        # Try to push to remote if it exists
        if git remote | grep -q origin; then
            if git push origin HEAD; then
                log_message "SUCCESS: Pushed to remote repository"
            else
                log_message "WARNING: Failed to push to remote repository (network issue?)"
            fi
        else
            log_message "INFO: No remote repository configured"
        fi
        
        return 0
    else
        log_message "ERROR: Failed to commit changes"
        return 1
    fi
}

# Function to clean up old backup branches (keep last 7 days)
cleanup_old_backups() {
    local cutoff_date=$(date -v-7d +%Y-%m-%d)
    log_message "Cleaning up backup branches older than $cutoff_date"
    
    git branch | grep "backup/" | while read -r branch; do
        local branch_name=$(echo "$branch" | sed 's/^[ *]*//')
        local branch_date=$(echo "$branch_name" | sed 's/backup\///')
        
        if [[ "$branch_date" < "$cutoff_date" ]]; then
            log_message "Deleting old backup branch: $branch_name"
            git branch -D "$branch_name" 2>/dev/null || true
        fi
    done
}

# Function to generate backup summary
generate_summary() {
    local repo_size=$(du -sh .git 2>/dev/null | cut -f1 || echo "unknown")
    local total_commits=$(git rev-list --all --count 2>/dev/null || echo "unknown")
    local branches=$(git branch -a | wc -l | tr -d ' ')
    
    log_message "=== BACKUP SUMMARY ==="
    log_message "Repository size: $repo_size"
    log_message "Total commits: $total_commits"
    log_message "Total branches: $branches"
    log_message "Current branch: $(git branch --show-current)"
    log_message "Last commit: $(git log -1 --format='%h - %s (%cr)' 2>/dev/null || echo 'none')"
    log_message "======================="
}

# Main backup function
main() {
    local backup_type="${1:-scheduled}"
    
    # Start logging
    rotate_log_if_needed
    log_message "=== Starting Git Backup ($backup_type) ==="
    
    # Change to project directory
    cd "$PROJECT_DIR" || {
        log_message "ERROR: Cannot change to project directory: $PROJECT_DIR"
        exit 1
    }
    
    # Check if we're in a git repository
    check_git_repo
    
    # Store current branch
    local original_branch=$(git branch --show-current)
    
    # Determine commit message based on backup type
    local commit_message
    case "$backup_type" in
        "scheduled")
            commit_message="Automated scheduled backup - $(date '+%Y-%m-%d %H:%M:%S')"
            ;;
        "manual")
            commit_message="Manual backup - $(date '+%Y-%m-%d %H:%M:%S')"
            ;;
        "hourly")
            commit_message="Hourly automated backup - $(date '+%Y-%m-%d %H:%M:%S')"
            ;;
        "daily")
            commit_message="Daily automated backup - $(date '+%Y-%m-%d %H:%M:%S')"
            ;;
        *)
            commit_message="Backup ($backup_type) - $(date '+%Y-%m-%d %H:%M:%S')"
            ;;
    esac
    
    # Check for changes
    if check_for_changes; then
        # Perform backup on main branch first
        if perform_backup "$backup_type" "$commit_message"; then
            log_message "Main branch backup completed successfully"
            
            # Create daily backup branch for important snapshots
            if [[ "$backup_type" == "daily" || "$backup_type" == "scheduled" ]]; then
                create_backup_branch "$BACKUP_BRANCH"
                git checkout "$original_branch"
                log_message "Created backup branch: $BACKUP_BRANCH"
            fi
        else
            log_message "ERROR: Main branch backup failed"
            exit 1
        fi
    else
        log_message "No changes detected, skipping backup"
    fi
    
    # Cleanup old backup branches
    cleanup_old_backups
    
    # Generate summary
    generate_summary
    
    log_message "=== Git Backup Completed ($backup_type) ==="
}

# Check command line arguments
case "${1:-}" in
    "manual"|"hourly"|"daily"|"scheduled")
        main "$1"
        ;;
    "--help"|"-h")
        echo "Usage: $0 [manual|hourly|daily|scheduled]"
        echo "  manual    - Manual backup triggered by user"
        echo "  hourly    - Hourly automated backup"
        echo "  daily     - Daily automated backup (creates backup branch)"
        echo "  scheduled - Default scheduled backup"
        echo "  --help    - Show this help message"
        exit 0
        ;;
    "")
        main "scheduled"
        ;;
    *)
        echo "ERROR: Unknown backup type: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
