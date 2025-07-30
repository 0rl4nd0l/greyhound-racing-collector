# Database Schema Difference Report

**Generated on:** 2025-07-30  
**Database:** `greyhound_racing_data.db`  
**Analysis Tool:** Step 2 - Database Schema Validation  

## Summary

This report compares the expected database schema (based on `initialize_database.py`, `create_unified_database.py`, and `create_tables.sql`) against the actual schema found in the live database.

**Key Findings:**
- **Expected Tables:** 8 tables defined in migration/initialization scripts
- **Actual Tables:** 25 tables found in the live database  
- **Missing Tables:** 5 tables from expected schema not found
- **Extra Tables:** 22 tables in live database not in expected schema
- **Tables with Differences:** 3 tables have structural mismatches

**Recommendation:** The database has evolved significantly beyond the original schema definitions. The missing core tables (`races`, `race_entries`, `form_guides`, `enhanced_analysis`, `venues`) suggest either incomplete migration or schema drift. Review and update schema documentation to match the actual working database structure.
## Missing Tables
- `races`
- `race_entries`
- `form_guides`
- `enhanced_analysis`
- `venues`
## Extra Tables
- `predictions`
- `venue_mappings`
- `odds_history`
- `sqlite_stat4`
- `race_analytics`
- `enhanced_expert_data`
- `weather_data_v2`
- `sqlite_sequence`
- `dog_race_data_backup_box_number_fix`
- `track_conditions`
- `weather_impact_analysis`
- `enhanced_expert_data_backup_dog_day_fix`
- `weather_forecast_cache`
- `value_bets`
- `weather_data`
- `race_metadata_backup_dedup_race_metadata`
- `dog_performances`
- `dog_race_data_backup_dedup_dog_race_data`
- `dog_race_data_backup`
- `track_condition_backup_20250724_185411`
- `live_odds`
- `sqlite_stat1`
## Table Differences
### Table: `race_metadata`
#### Extra Columns
- `actual_field_size` (INTEGER)
- `box_analysis` (TEXT)
- `weather_timestamp` (DATETIME)
- `scratched_count` (INTEGER)
- `weather_location` (TEXT)
- `visibility` (REAL)
- `prize_money_breakdown` (TEXT)
- `pressure` (REAL)
- `prize_money_total` (REAL)
- `wind_speed` (REAL)
- `sportsbet_url` (TEXT)
- `race_time` (TEXT)
- `precipitation` (REAL)
- `data_quality_note` (TEXT)
- `weather` (TEXT)
- `track_record` (TEXT)
- `weather_adjustment_factor` (REAL)
- `humidity` (REAL)
- `race_status` (TEXT)
- `venue_slug` (TEXT)
- `weather_condition` (TEXT)
- `id` (INTEGER)
- `data_source` (TEXT)
- `scratch_rate` (REAL)
- `wind_direction` (TEXT)
- `temperature` (REAL)
#### Column Mismatches
- **`venue`**: nullability mismatch (expected: True, actual: False)
- **`winner_margin`**: type mismatch (expected: TEXT, actual: REAL)
- **`extraction_timestamp`**: type mismatch (expected: TEXT, actual: DATETIME), nullability mismatch (expected: True, actual: False)
- **`race_number`**: nullability mismatch (expected: True, actual: False)
- **`race_date`**: type mismatch (expected: TEXT, actual: DATE), nullability mismatch (expected: True, actual: False)
- **`race_id`**: PK mismatch (expected: True, actual: False)
- **`winner_odds`**: type mismatch (expected: TEXT, actual: REAL)
#### Missing Indexes
- Index `idx_race_metadata_venue` on `(venue)`
- Index `idx_race_metadata_extraction` on `(extraction_timestamp)`
### Table: `dog_race_data`
#### Missing Columns
- `odds` (TEXT)
- `winning_time` (TEXT)
- `form` (TEXT)
- `placing` (INTEGER)
- `trainer` (TEXT)
#### Extra Columns
- `scraped_nbtt` (TEXT)
- `recent_form` (TEXT)
- `beaten_margin` (REAL)
- `dog_id` (INT)
- `scraped_trainer_name` (TEXT)
- `odds_fractional` (TEXT)
- `blackbook_link` (TEXT)
- `speed_rating` (REAL)
- `extraction_timestamp` (NUM)
- `scraped_reaction_time` (TEXT)
- `running_style` (TEXT)
- `individual_time` (TEXT)
- `was_scratched` (NUM)
- `win_probability` (REAL)
- `sectional_3rd` (TEXT)
- `starting_price` (REAL)
- `performance_rating` (REAL)
- `sectional_2nd` (TEXT)
- `place_probability` (REAL)
- `dog_clean_name` (TEXT)
- `historical_records` (TEXT)
- `trainer_id` (INT)
- `odds_decimal` (REAL)
- `data_quality_note` (TEXT)
- `class_rating` (REAL)
- `best_time` (REAL)
- `form_guide_json` (TEXT)
- `trainer_name` (TEXT)
- `sectional_1st` (TEXT)
- `scraped_raw_result` (TEXT)
- `scraped_race_classification` (TEXT)
- `finish_position` (INTEGER)
- `data_source` (TEXT)
- `scraped_finish_position` (TEXT)
#### Column Mismatches
- **`weight`**: type mismatch (expected: TEXT, actual: REAL)
- **`box_number`**: type mismatch (expected: INTEGER, actual: INT)
- **`id`**: type mismatch (expected: INTEGER, actual: INT), PK mismatch (expected: True, actual: False)
- **`race_id`**: nullability mismatch (expected: True, actual: False)
#### Missing Foreign Keys
- `race_id` -> `race_metadata(race_id)`
#### Missing Indexes
- Index `idx_dog_race_data_race` on `(race_id)`
### Table: `dogs`
#### Missing Columns
- `color` (TEXT)
- `weight` (DECIMAL(5,2))
- `age` (INTEGER)
- `id` (INTEGER)
- `trainer` (TEXT)
- `owner` (TEXT)
- `sex` (TEXT)
#### Extra Columns
- `dog_id` (INTEGER)
- `last_race_date` (TEXT)
- `total_races` (INTEGER)
- `total_wins` (INTEGER)
- `average_position` (REAL)
- `total_places` (INTEGER)
- `best_time` (REAL)