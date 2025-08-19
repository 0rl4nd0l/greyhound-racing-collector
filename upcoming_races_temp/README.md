# Upcoming Races

This directory contains CSV files for races that are yet to run.
These files contain race information but no results yet.

Folder path and configuration
- Default location: `./upcoming_races_temp`
- Configurable via environment variable: `UPCOMING_RACES_DIR`

Filename convention
- Pattern: `Race {race_number} - {VENUE_CODE} - YYYY-MM-DD.csv`
  - Example: `Race 4 - GOSF - 2025-07-28.csv`
- Extension: `.csv` (lowercase)

CSV schema (race data)
- Encoding: UTF-8 (BOM discouraged)
- Delimiter: comma `,`
- Header row: required

Required columns
- `race_date` (YYYY-MM-DD)
- `venue_code`
- `race_number`
- `dog_name`
- `box`

Optional columns
- `trainer`, `weight`, `distance`, `grade`, `meeting_name`, `scheduled_time_local`

Forbidden/empty-at-ingest fields
- Outcome fields (`PLC`, `finish_position`, `winner`, `winning_time`, `margin`) must be absent or blank.

Archiving guidance
- Outdated or redundant files should be moved to `archive/upcoming_races/YYYY/MM/`
- Preserve filenames; append `_archived-YYYYMMDD-HHMMSS` if duplicates occur
- See also: `docs/ARCHIVAL_SYSTEM_SUMMARY.md`

Notes
- Historical form guides do not belong here; they live under `./unprocessed` and are processed separately.
- Race winners must be scraped from the race page after the event, not inferred from form guides.
