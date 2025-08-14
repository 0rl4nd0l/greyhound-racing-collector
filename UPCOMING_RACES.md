Upcoming Races CSVs
====================

Purpose
- This folder and its rules define how the application discovers, validates, and consumes upcoming race CSV files for the UI, API, and prediction pipelines.

Folder path
- Live directory: ./upcoming_races
- The backend ensures this directory exists on startup.
- Uploads also use this path by default.

Consumers
- API endpoints:
  - /api/upcoming_races: Live data (default) with CSV fallback; in tests it forces CSV to enable mocking.
  - /api/upcoming_races_csv: Lists files from ./upcoming_races with pagination/search and metadata extraction.
- UI:
  - /upcoming page renders the list using /api/upcoming_races and /api/upcoming_races_csv.
- Prediction:
  - The predict page lists CSVs directly from ./upcoming_races.

File requirements
- Extension: .csv (lowercase)
- Hidden or non-CSV files are ignored.
- CSVs should represent upcoming racecards (no result columns like PLC/WIN).
- Encoding: UTF-8 recommended.
- Delimiter: commonly pipe (|) or comma (,); the loader reads headers and is defensive.

Filename patterns supported
- Canonical: "Race {num} - {VENUE} - {YYYY-MM-DD}.csv"
  - Example: "Race 1 - WPK - 2025-02-01.csv"
- Alternative underscore styles:
  - "Race_{num}_{VENUE}_{YYYY-MM-DD}.csv" (e.g., "Race_3_WPK_2025-02-01.csv")
  - "{VENUE}_Race_{num}_{YYYY-MM-DD}.csv" (e.g., "MEA_Race_8_2025-03-15.csv")
- The loader also attempts best-effort parsing for:
  - "Race {num} - {VENUE} - 01 February 2025.csv" (converts to YYYY-MM-DD)

Metadata extraction and precedence
- The loader extracts race_number, venue, race_date from:
  1) CSV header fields when present (preferred): race_name, venue/track, race_date/date, race_number/number, distance, grade
  2) Filename patterns as fallbacks
- Additional fields: distance and grade are read from headers when available.

Sanitization and normalization
- NaN-like values ("nan", "none", "null", empty) for venue, race_date, distance, grade are normalized to "Unknown".
- Venue tokens have stray underscores trimmed (e.g., "_MEA_" -> "MEA").
- Duplicate races are de-duplicated by the composite key: {venue}_{race_date}_{race_number}.
- race_id is constructed stably using an MD5 of the filename to match tests.

Pagination, sorting, search (/api/upcoming_races_csv)
- Query params: page (default 1), per_page (max 50), sort_by (race_date default), order (asc/desc), search.
- Returns JSON with pagination info and an array of races including filename and parsed metadata.

Caching (/api/upcoming_races)
- In-memory cache with 5-minute TTL keyed by query params (days, page, per_page, source).
- Response includes: count, timestamp, from_cache, cache_expires_in_minutes.
- Supports refresh=true to bypass cache.

Testing behavior
- In test mode, /api/upcoming_races forces CSV source and calls the CSV loader early so test mocks/patches surface correctly (e.g., for error handling tests).

Operational guidance
- Place only valid upcoming race CSVs in ./upcoming_races.
- Archive outdated CSVs (past-dated) to your chosen archive folder to keep the list clean.
- Keep file names predictable to help parsing and search.

Examples
- Race 5 - WPK - 2025-02-01.csv
- Race_1_APK_2025-08-04.csv
- MEA_Race_8_2025-03-15.csv

Troubleshooting
- If /api/upcoming_races_csv returns 0 files:
  - Ensure ./upcoming_races exists and contains .csv files.
  - Check permissions and that files aren’t hidden.
- If fields are "Unknown":
  - Confirm headers contain race metadata or adjust filename to a supported pattern.
- If duplicates appear:
  - Ensure unique combination of {venue}, {race_date}, {race_number}; rename or remove duplicates.

