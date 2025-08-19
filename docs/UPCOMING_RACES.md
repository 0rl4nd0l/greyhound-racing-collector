# Upcoming Races CSV Folder

This project uses a single, well-defined folder for all upcoming race CSV files:

- Path: ./upcoming_races
- Extension: .csv only (no JSON or other formats for ingestion)
- Filename pattern (strict): "Race {number} - {VENUE} - {YYYY-MM-DD}.csv"
  - Examples:
    - Race 1 - WPK - 2025-02-01.csv
    - Race 8 - MEA - 2025-03-15.csv
    - Race 4 - GOSF - 2025-02-03.csv

Notes:
- The application creates the folder automatically on startup if it does not exist.
- The API endpoints that consume these files are:
  - /api/upcoming_races (CSV fallback/source)
  - /api/upcoming_races_csv (direct listing and parsing of CSVs)
- The predictions pipeline uses the same folder to locate files for batch and single-race prediction endpoints.

Data hygiene rules applied by the loader:
- NaN-like strings ("nan", "NaN", "null", "None", empty strings) are normalized to "Unknown" for fields such as venue, distance, grade, and race_date.
- Venue values are normalized (slashes replaced with underscores, surrounding underscores stripped).
- Duplicate races are skipped using the unique key: {venue}_{race_date}_{race_number}.
- race_id is constructed using the MD5 hash of the filename (first 12 hex chars) to ensure stable IDs.

How the UI picks up races automatically:
- The Upcoming UI uses /api/upcoming_races (which falls back to CSVs) and /api/upcoming_races_csv to discover and display races with no manual steps required.

CI and tests:
- Tests validate the folder path, CSV parsing, NaN sanitation, duplicate-skipping, and ID construction.
- Keep CSVs in ./upcoming_races and ensure they use the .csv extension to pass validation.
