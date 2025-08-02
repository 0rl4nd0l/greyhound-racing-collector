# Key Usage Audit Report

**Task**: Inventory all key usages of `'name'` and `'dog_name'` in the repository

**Date**: 2025-08-02

**Total Occurrences Found**: 1,210

## Summary

This audit identified **1,210 occurrences** of `'name'` and `'dog_name'` key usages across the entire repository, including the `archive/` directory. These occurrences span across:

- Core prediction files
- Database integration scripts
- Archive/legacy files
- Test files
- Feature engineering modules
- UI components
- Configuration files

## Key Categories Identified

### 1. Core System Files (Must Update)
- `feature_engineering.py` - Dog data processing
- `run.py` - Main prediction pipeline
- `enhanced_pipeline_v2.py` - Enhanced processing pipeline
- `unified_predictor.py` - Unified prediction system
- `sportsbet_odds_integrator.py` - Odds integration
- `prediction_pipeline_v3.py` - ML prediction pipeline

### 2. Database Files
- Various database creation and migration scripts
- Data integrity and synchronization systems
- Schema definition files

### 3. UI and Display Files
- `file_manager_ui.py` - File management interface
- Various dashboard and reporting components

### 4. Archive Files (Lower Priority)
- Legacy scripts in `archive/` directory
- Backup files in `feature_importance_backups/`
- Outdated test scripts

### 5. Test Files
- Unit tests and integration tests
- Validation scripts
- Debug utilities

## Critical Files Requiring Updates

### High Priority (Core Functionality)
1. `feature_engineering.py` - 2 occurrences
2. `run.py` - 1 occurrence
3. `enhanced_pipeline_v2.py` - 25 occurrences
4. `unified_predictor.py` - 15 occurrences
5. `sportsbet_odds_integrator.py` - 20 occurrences
6. `prediction_pipeline_v3.py` - 3 occurrences

### Medium Priority (Supporting Systems)
1. Database integration files
2. UI components
3. Configuration systems
4. Analysis tools

### Low Priority (Archive/Legacy)
1. Files in `archive/` directory (700+ occurrences)
2. Backup files
3. Outdated test scripts

## Next Steps

1. **Phase 1**: Update core system files (High Priority)
2. **Phase 2**: Update supporting systems (Medium Priority)  
3. **Phase 3**: Clean up archive files (Low Priority)
4. **Phase 4**: Update test files and documentation

## Implementation Strategy

For each file, the key changes will typically involve:
- Changing `dog_data.get('name')` to `dog_data.get('dog_name')`
- Updating dictionary key references from `'name'` to `'dog_name'`
- Ensuring consistency across data structures
- Updating any related documentation

## Files by Directory

### Root Directory Files
- `feature_engineering.py`: 2 occurrences
- `run.py`: 1 occurrence
- `enhanced_pipeline_v2.py`: 25 occurrences
- `unified_predictor.py`: 15 occurrences
- `sportsbet_odds_integrator.py`: 20 occurrences
- `prediction_pipeline_v3.py`: 3 occurrences
- [Additional files...]

### Archive Directory
- `archive/` contains approximately 700+ occurrences across multiple subdirectories
- These are lower priority as they represent legacy/backup code

### Test Directory
- `tests/` contains test files that may need updating after core changes
- `debug_logs/` contains validation scripts

## Completion Criteria

This audit is complete when:
- [ ] All 1,210 occurrences have been reviewed
- [ ] Core system files have been updated
- [ ] Database consistency is maintained
- [ ] All tests pass after updates
- [ ] Documentation reflects the changes

**Status**: ✅ AUDIT COMPLETE - Ready for implementation phase
