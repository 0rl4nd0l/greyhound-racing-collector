# Data Integrity Cleanup Report
**Date**: 2025-09-01T22:11:25.242219

## Issues Found
- **box_duplicates**: 7636
- **dog_duplicates**: 223
- **multiple_winners**: 1108
- **races_without_winners**: 1451

## Fixes Applied
- **box_duplicates_removed**: 10191 records affected
- **dog_duplicates_removed**: 635 records affected
- **multiple_winners_corrected**: 1792 records affected
- **missing_winners_populated**: 0 records affected

## Constraints Added
- `CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_box_per_race ON dog_race_data(race_id, box_number)`
- `CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_dog_per_race ON dog_race_data(race_id, dog_clean_name)`

## Backup Location
Pre-cleanup backup: `docs/analysis/pre_cleanup_backup_20250901_221125.sqlite`
