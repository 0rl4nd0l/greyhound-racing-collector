# Git Swarm Prevention System

## 🚨 What is Git Swarm?
Git swarm occurs when thousands of files are tracked by Git, causing:
- **Massive CPU usage** from Git diff operations
- **Warp/IDE performance degradation** 
- **System slowdown** from continuous Git processes
- **Memory exhaustion** from large Git operations

## 🛡️ Prevention Layers

### Layer 1: Enhanced .gitignore
- **190+ patterns** covering all problematic file types
- Virtual environments (`.venv*`, `node_modules/`)
- Database files (`*.db*`, `*.sqlite*`)
- Large data directories (`processed/`, `archive/`)
- Build artifacts, logs, caches

### Layer 2: Pre-commit Hook
- **Automatic blocking** of commits >1000 files
- **Pattern detection** for problematic files
- **Interactive warnings** for suspicious patterns
- Can be bypassed with `--no-verify` if needed

### Layer 3: Maintenance Script
```bash
./scripts/prevent_git_swarm.sh
```
- Monitors current Git status
- Validates .gitignore completeness
- Automated cleanup of problematic files
- Health checks and recommendations

### Layer 4: Monitoring Commands
```bash
# Quick health check
git status --porcelain | wc -l

# Find large directories
du -sh */ | sort -hr | head -10

# Check for ignored files that might be tracked
git ls-files --ignored --exclude-standard
```

## 🚨 Emergency Procedures

### If Git Swarm Occurs:
1. **Kill runaway processes**: `killall git`
2. **Remove lock files**: `rm -f .git/*.lock`
3. **Reset unstaged files**: `git reset`
4. **Run cleanup script**: `./scripts/prevent_git_swarm.sh`
5. **Commit with bypass**: `git commit --no-verify`

### Critical File Patterns to Always Ignore:
- `.venv*/` - Virtual environments
- `node_modules/` - Node.js dependencies  
- `__pycache__/` - Python cache
- `*.db*` - Database files
- `processed/` - Large CSV datasets
- `archive/` - Archive directories
- `*.log` - Log files
- `test-results/` - Test artifacts

## 📊 Performance Metrics
- **Healthy**: <100 files in Git status
- **Caution**: 100-500 files 
- **Warning**: 500-1000 files
- **Critical**: >1000 files (triggers pre-commit block)

## 🔧 Maintenance
Run weekly health check:
```bash
./scripts/prevent_git_swarm.sh
```

## 📞 Emergency Contact
If Git swarm occurs despite these measures:
1. Immediately run emergency procedures above
2. Review .gitignore for missing patterns
3. Consider repository restructuring if chronic issues persist
