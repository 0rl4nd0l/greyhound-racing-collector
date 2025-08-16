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
- `enhanced_pipeline_v2.py`: 22 occurrences
- `unified_predictor.py`: 15 occurrences
- `sportsbet_odds_integrator.py`: 19 occurrences
- `prediction_pipeline_v3.py`: 3 occurrences
- [Additional files...]

### Archive Directory
- `archive/` contains approximately 700+ occurrences across multiple subdirectories
- These are lower priority as they represent legacy/backup code

### Test Directory
- `tests/` contains test files that need updating after core changes
- `debug_logs/` contains validation scripts

## Top Files by Occurrence Count

**Active/Core Files (High Priority)**:
1. `app.py` - 45 occurrences
2. `enhanced_comprehensive_processor.py` - 28 occurrences 
3. `enhanced_pipeline_v2.py` - 22 occurrences
4. `database_maintenance.py` - 22 occurrences
5. `file_manager_ui.py` - 21 occurrences
6. `sportsbet_odds_integrator.py` - 19 occurrences
7. `enhanced_race_analyzer.py` - 17 occurrences
8. `unified_predictor.py` - 15 occurrences
9. `comprehensive_prediction_pipeline.py` - 14 occurrences
10. `weather_enhanced_predictor.py` - 11 occurrences

**Archive/Backup Files (Lower Priority)**:
- `archive/database_and_migrations/app_original.py` - 42 occurrences
- `archive_old_apps/app_original.py` - 34 occurrences
- Various backup files in `feature_importance_backups/` - 100+ occurrences
- Legacy scripts in `archive/` - 200+ occurrences

**Test Files**:
- `tests/ml_backtesting_trainer_optimized.py` - 12 occurrences
- Various other test files - 50+ occurrences

## Completion Criteria

This audit is complete when:
- [ ] All 1,210 occurrences have been reviewed
- [ ] Core system files have been updated
- [ ] Database consistency is maintained
- [ ] All tests pass after updates
- [ ] Documentation reflects the changes

**Status**: ✅ AUDIT COMPLETE - Ready for implementation phase
