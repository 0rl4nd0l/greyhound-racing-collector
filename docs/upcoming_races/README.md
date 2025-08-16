# Upcoming Races CSVs

This directory is the canonical source for upcoming race CSV files used by:
- The /api/upcoming_races and /api/upcoming_races_csv API endpoints
- The Upcoming UI (predict page) for listing selectable races
- The prediction pipelines (PredictionPipelineV4 primary, V3 fallback)

Folder path
- ./upcoming_races

Filename requirements
- Extension: .csv (lowercase)
- Naming pattern: "Race {number} - {VENUE} - {YYYY-MM-DD}.csv"
  - Examples:
    - Race 1 - WPK - 2025-02-01.csv
    - Race 10 - MEA - 2025-02-02.csv
- Notes:
  - VENUE should be 2–5 uppercase characters (e.g., WPK, MEA, GOSF)
  - Use a hyphen/dash surrounded by spaces between parts
  - If your source file is named differently, use the helper scripts to alias/normalize (see below)

Helper scripts
- scripts/normalize_upcoming_to_api_pattern.py — creates symlink aliases inside ./upcoming_races that match the API pattern, preserving originals in place
- scripts/alias_upcoming_api_names_safe.py — safely aliases only confidently-parseable filenames already in ./upcoming_races

UI and pipeline integration
- The Upcoming UI (predict page) enumerates CSVs in this folder and shows them for selection automatically; no manual listing is needed
- The prediction UI triggers PredictionPipelineV4 by default against the selected CSV; V3 and legacy fallbacks are used if needed

CI validation
- Unit and integration tests exercise /api/upcoming_races and /api/upcoming_races_csv and validate structure, pagination, and naming
- Tests also use temporary directories via configuration; however, this repo keeps an empty ./upcoming_races folder tracked to avoid regressions

See also
- utils/file_naming.py (build_upcoming_csv_filename)
- file_naming_standards/NAMING_STANDARDS.md
- docs/api/endpoints.md (Upcoming endpoints)

