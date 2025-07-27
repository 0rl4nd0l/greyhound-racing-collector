#!/bin/bash

# Script to set up GitHub remote repository for automated backups
# Run this script to add a remote GitHub repository for your project

echo "🔧 GitHub Remote Repository Setup"
echo "================================="
echo ""

# Check if git is available
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ This directory is not a Git repository."
    exit 1
fi

# Check if remote already exists
if git remote | grep -q origin; then
    echo "ℹ️  Remote 'origin' already exists:"
    git remote -v
    echo ""
    read -p "Do you want to update the existing remote? (y/N): " update_remote
    if [[ $update_remote =~ ^[Yy]$ ]]; then
        read -p "Enter new GitHub repository URL: " repo_url
        git remote set-url origin "$repo_url"
        echo "✅ Remote 'origin' updated to: $repo_url"
    else
        echo "ℹ️  Keeping existing remote configuration."
    fi
else
    echo "📝 To set up automated backups to GitHub:"
    echo "1. Create a new repository on GitHub"
    echo "2. Copy the repository URL (HTTPS or SSH)"
    echo "3. Enter it below"
    echo ""
    
    read -p "Enter GitHub repository URL (or press Enter to skip): " repo_url
    
    if [[ -n "$repo_url" ]]; then
        git remote add origin "$repo_url"
        echo "✅ Remote 'origin' added: $repo_url"
        
        # Try to push current branch
        current_branch=$(git branch --show-current)
        echo "🚀 Attempting to push current branch: $current_branch"
        
        if git push -u origin "$current_branch"; then
            echo "✅ Successfully pushed to remote repository"
        else
            echo "⚠️  Failed to push to remote. You may need to:"
            echo "   - Set up authentication (SSH keys or personal access token)"
            echo "   - Check repository permissions"
            echo "   - Verify the repository URL"
        fi
    else
        echo "ℹ️  Skipping remote repository setup."
        echo "   You can add it later with: git remote add origin <repository-url>"
    fi
fi

echo ""
echo "🔍 Current Git configuration:"
echo "Repository: $(pwd)"
echo "Branch: $(git branch --show-current)"
echo "Remotes:"
git remote -v || echo "  No remotes configured"

echo ""
echo "📋 Next steps:"
echo "1. The automated backup system is now active"
echo "2. Backups will run automatically at:"
echo "   - Every 6 hours during active development (9 AM - 11 PM)"
echo "   - Daily at 2 AM (creates backup branch)"
echo "   - Evening backup at 10 PM"
echo "3. Manual backup: ./git_backup.sh manual"
echo "4. Check backup logs: tail -f logs/git_backup.log"

if git remote | grep -q origin; then
    echo "5. Your code will be automatically pushed to GitHub!"
else
    echo "5. Add a GitHub remote to enable cloud backups"
fi

echo ""
echo "🎉 Git backup system is ready!"
