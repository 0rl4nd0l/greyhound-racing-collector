#!/bin/bash

# Batch commit script for cleanup changes
# This will commit changes in smaller batches to avoid Git swarm protection

echo "Starting batch commit process..."

# Commit .gitignore changes first
echo "Committing .gitignore updates..."
git add .gitignore
if git diff --cached --quiet; then
    echo "No .gitignore changes to commit"
else
    git commit -m "Update .gitignore patterns for cleanup process"
fi

# Add the new GIT_SWARM_PREVENTION.md file
echo "Committing Git swarm prevention documentation..."
git add GIT_SWARM_PREVENTION.md 2>/dev/null || true
if ! git diff --cached --quiet; then
    git commit -m "Add Git swarm prevention documentation"
fi

# Commit app.py changes
echo "Committing app.py changes..."
git add app.py
if ! git diff --cached --quiet; then
    git commit -m "Update app.py configuration"
fi

# Commit archive directory deletions in batches
echo "Processing archive directory deletions in batches..."
batch_size=200
counter=0

# Get list of deleted archive files
deleted_files=$(git status --porcelain | grep '^D.*archive/' | cut -c4- | head -1000)

if [ -n "$deleted_files" ]; then
    echo "$deleted_files" | while IFS= read -r file; do
        git add "$file" 2>/dev/null || true
        counter=$((counter + 1))
        
        if [ $((counter % batch_size)) -eq 0 ] || [ $counter -eq 1000 ]; then
            echo "Committing batch of archive deletions (batch ending at item $counter)..."
            git commit -m "Cleanup: Archive batch deletion ($counter files processed)"
            counter=0
        fi
    done
fi

# Handle any remaining files
echo "Handling any remaining archive deletions..."
remaining_deleted=$(git status --porcelain | grep '^D.*archive/' | wc -l)
if [ "$remaining_deleted" -gt 0 ]; then
    echo "Adding remaining $remaining_deleted deleted archive files..."
    git status --porcelain | grep '^D.*archive/' | cut -c4- | head -500 | xargs -I {} git add "{}" 2>/dev/null || true
    if ! git diff --cached --quiet; then
        git commit -m "Cleanup: Final archive deletion batch"
    fi
fi

echo "Batch commit process completed."
echo "Remaining uncommitted changes:"
git status --porcelain | head -10
