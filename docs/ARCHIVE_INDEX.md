# Archive Index - Greyhound Racing Collector
## External Archive Location Reference

> **📁 Archive Status**: Moved to external hard drive  
> **📅 Archive Date**: September 1, 2025  
> **📊 Archive Size**: ~19.2GB  
> **🔄 Restoration**: See restoration instructions below  

---

## 📋 Quick Reference

### Archive Location
- **Original Path**: `/Users/test/Desktop/greyhound_racing_collector_archive/`
- **Current Status**: ⚠️ **MOVED TO EXTERNAL HARD DRIVE** 
- **Hard Drive Label**: `[TO BE UPDATED WHEN MOVED]`

### Key Stats
- **Total Files Archived**: 5,094+ files
- **Code Lines Archived**: 370,800+ lines
- **Space Saved**: 19.2GB (83% reduction)
- **Project Size**: 23GB → 3.8GB

---

## 🗂️ Complete Archive Directory Structure

```
greyhound_racing_collector_archive/
├── README.md                          # Restoration instructions
├── MANIFEST.txt                       # Complete file listing
└── archived_content/
    ├── large_folders/2025-09/
    │   ├── archive/                   # 16GB - Historical codebase
    │   │   ├── archived_configs/      # Configuration backups
    │   │   ├── backup_before_cleanup/ # Pre-cleanup snapshots
    │   │   ├── corrupt_historical_race_data/  # 1 corrupt CSV
    │   │   ├── corrupt_or_legacy_race_files/  # 4,800+ race CSVs
    │   │   ├── backup_verification.py
    │   │   ├── cleanup_duplicate_predictions.py.disabled
    │   │   ├── complete_reset.py
    │   │   ├── comprehensive_data_rebuilder.py
    │   │   ├── comprehensive_database_fix.py
    │   │   └── com.greyhound.git-backup.plist
    │   ├── node_modules/              # 377MB - Frontend dependencies
    │   │   ├── @playwright/           # Browser testing
    │   │   ├── @types/                # TypeScript definitions
    │   │   ├── cypress/               # E2E testing framework
    │   │   ├── jest/                  # JavaScript testing
    │   │   └── [1000+ npm packages]   # All Node.js dependencies
    │   └── comprehensive_form_cache/  # 57MB - Cached form data
    ├── backups/2025-09/
    │   ├── backups/                   # 361MB - Various system backups
    │   │   ├── archive_logs.py
    │   │   └── manifests/
    │   │       └── main_workflow.jsonl
    │   └── database_backups/          # 91MB - Database snapshots
    ├── models/2025-09/
    │   ├── model_registry.bak-20250830-123307/  # 251MB - Model backup
    │   ├── ai_models/                 # Legacy AI models
    │   └── ml_models_v3/              # Version 3 ML models
    ├── logs/2025-09/
    │   └── old_logs/                  # Historical log files (>30 days)
    ├── data/2025-09/
    │   ├── Race_01_UNKNOWN_*.json     # Debug race files
    │   └── *.html                     # Debug HTML outputs
    ├── cache/2025-09/
    │   ├── __pycache__/              # Python bytecode cache
    │   ├── .pytest_cache*/           # Pytest cache files
    │   ├── .mypy_cache/              # MyPy type checking cache
    │   └── .ruff_cache/              # Ruff linter cache
    ├── environments/2025-09/
    │   ├── .venv311/                 # Old Python 3.11 venv
    │   └── venv/                     # Legacy virtual environment
    └── testing/2025-09/
        ├── cypress/
        │   ├── e2e/endpoints-menu.cy.js
        │   └── videos/endpoints-menu.cy.js.mp4
        └── playwright/
            └── e2e/endpoints-menu.spec.ts
```

---

## 📊 Detailed File Inventory

### Large Folders (16.4GB total)
| Directory | Size | Description | Restoration Priority |
|-----------|------|-------------|---------------------|
| `archive/` | 16.0GB | Historical codebase with 193,519 Python files | LOW - Legacy code |
| `node_modules/` | 377MB | Frontend dependencies | MEDIUM - Regenerable |
| `comprehensive_form_cache/` | 57MB | Cached form processing data | LOW - Cache data |

### Backup Directories (452MB total)
| Directory | Size | Description | Restoration Priority |
|-----------|------|-------------|---------------------|
| `backups/` | 361MB | System and workflow backups | MEDIUM - Recovery data |
| `database_backups/` | 91MB | Database snapshots | HIGH - Data integrity |

### Model Storage (251MB+ total)
| Directory | Size | Description | Restoration Priority |
|-----------|------|-------------|---------------------|
| `model_registry.bak-*` | 251MB | Complete model registry backup | HIGH - ML models |
| `ai_models/` | Variable | Legacy AI model files | LOW - Deprecated |
| `ml_models_v3/` | Variable | Version 3 ML models | MEDIUM - May be useful |

### Development Artifacts
| Category | Count | Description | Restoration Priority |
|----------|-------|-------------|---------------------|
| Race CSV files | 4,800+ | Historical race data files | MEDIUM - Historical data |
| Debug JSON files | 50+ | Race debugging outputs | LOW - Debug artifacts |
| HTML files | 20+ | Debug HTML outputs | LOW - Debug artifacts |
| Cache directories | 4 types | Python/linter caches | NONE - Regenerable |
| Virtual environments | 2 | Old Python environments | NONE - Regenerable |
| Test artifacts | 3 files | Cypress/Playwright test files | LOW - Test outputs |

---

## 🔄 Restoration Instructions

### Quick Restoration Commands

**Full Archive Restoration** (if hard drive mounted as `/Volumes/ArchiveDrive`):
```bash
# Navigate to project directory
cd /Users/test/Desktop/greyhound_racing_collector

# Restore specific high-priority items
cp -r /Volumes/ArchiveDrive/greyhound_racing_collector_archive/archived_content/backups/2025-09/database_backups/ ./
cp -r /Volumes/ArchiveDrive/greyhound_racing_collector_archive/archived_content/models/2025-09/model_registry.bak-*/ ./

# Restore development dependencies (if needed)
cp -r /Volumes/ArchiveDrive/greyhound_racing_collector_archive/archived_content/large_folders/2025-09/node_modules/ ./
npm install  # Alternative to copying node_modules

# Restore historical data (if research needed)
cp -r /Volumes/ArchiveDrive/greyhound_racing_collector_archive/archived_content/large_folders/2025-09/archive/ ./
```

**Selective Restoration by Priority**:

**HIGH PRIORITY** - Critical for system operation:
```bash
# Database backups
cp -r /Volumes/ArchiveDrive/.../database_backups/ ./backups/
# Model registry
cp -r /Volumes/ArchiveDrive/.../model_registry.bak-*/ ./
```

**MEDIUM PRIORITY** - Useful for development:
```bash
# System backups
cp -r /Volumes/ArchiveDrive/.../backups/ ./
# Historical race data
cp -r /Volumes/ArchiveDrive/.../corrupt_or_legacy_race_files/ ./data/historical/
# Node dependencies (or just run npm install)
npm install
```

**LOW PRIORITY** - Research/debugging only:
```bash
# Full historical archive
cp -r /Volumes/ArchiveDrive/.../archive/ ./
# Debug artifacts
cp -r /Volumes/ArchiveDrive/.../data/2025-09/ ./debug/
```

---

## 🎯 Agent Guidelines

### For Future Development Agents

**What's Safe to Ignore**:
- Cache directories (`__pycache__`, `.pytest_cache`, etc.) - Always regenerable
- Virtual environments (`.venv311`, `venv`) - Create new ones
- Debug artifacts (`Race_*.json`, `*.html`) - Temporary debug files
- Test videos/screenshots - Generated during testing

**What May Be Needed**:
- **Database backups** - Critical for data recovery
- **Model registry backup** - Contains trained ML models
- **Historical race CSVs** - Research data (4,800+ files)
- **Node modules** - Can restore or regenerate with `npm install`

**What's Definitely Important**:
- **Model registry backup** - Contains production ML models
- **Database backups** - Data integrity and recovery
- **System backups** - Configuration and workflow data

### Current Active System (Post-Cleanup)

**✅ Still Available in Main Project**:
- All active Python scripts per `ACTIVE_SCRIPTS_GUIDE.md`
- Main database: `greyhound_racing_data.db`
- Current model registry with 11 tracked models
- All templates, static files, and core directories
- Working virtual environment (`.venv`)
- All tests and test fixtures

**❌ Moved to Archive**:
- Historical/legacy code (16GB of old scripts)
- Development dependencies (node_modules - 377MB)
- Debug and cache files
- Old virtual environments
- Test artifacts and videos

---

## 📝 Restoration Checklist

When restoring from hard drive archive:

- [ ] **Verify hard drive is mounted and accessible**
- [ ] **Check archive integrity** - Verify key directories exist
- [ ] **Restore by priority** - Start with HIGH priority items
- [ ] **Test functionality** - Run tests after each restoration
- [ ] **Update paths** - Ensure restored files are in correct locations
- [ ] **Regenerate caches** - Clear and rebuild development caches
- [ ] **Verify dependencies** - Ensure all required packages are available

## 🔍 Finding Specific Files

**Common Restoration Scenarios**:

1. **Need historical ML model**: Look in `models/2025-09/model_registry.bak-*`
2. **Need old race data**: Look in `large_folders/2025-09/archive/corrupt_or_legacy_race_files/`
3. **Need system configuration**: Look in `backups/2025-09/backups/`
4. **Need database backup**: Look in `backups/2025-09/database_backups/`
5. **Need to restore frontend**: Look in `large_folders/2025-09/node_modules/` or run `npm install`

---

**Last Updated**: September 1, 2025  
**Next Review**: When archive is moved to hard drive (update mount path)
