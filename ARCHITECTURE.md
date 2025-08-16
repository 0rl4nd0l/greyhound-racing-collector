# Scraping & ETL Architecture (Inventory)

This document inventories the current, canonical scripts, modules, endpoints, and outputs for the greyhound scraping and ETL flows.

Last updated: 2025-08-14

## Repo layout (relevant dirs)
- upcoming_races/ — downloaded form guide CSVs (current)
- processed/ — normalized outputs and artifacts
- archive/ — archived and legacy scripts/data
- gpt_enhanced_predictions/ — JSON predictions and analysis artefacts
- services/ — guardians/cron utilities
- tools/ — validators and MCP utilities

The repo is flat (no src/ package). Scripts are invoked with `python <script>.py`.

---

## A) Historical data (form guide) scraping/parsing

Primary (canonical)
- upcoming_race_browser.py
  - Purpose: Download/refresh form guide CSVs for upcoming races; enhance cached races with scraped live times.
  - How to run: `python upcoming_race_browser.py`
  - External endpoints used:
    - Base: `https://www.thedogs.com.au`
    - Race pages: `/racing/{venue_slug}/{YYYY-MM-DD}/{race_number}`
    - Expert form subpage: `.../expert-form`
    - CSV discovery (attempted variants): `?format=csv`, `?export=csv`, `/export/csv`, `/download/csv`, `/csv`, `/form-guide.csv` (plus fallback patterns)
  - Outputs:
    - CSV files in `upcoming_races/` named `Race {race_number} - {VENUE_CODE} - YYYY-MM-DD.csv`
    - Enhances cached races in memory; validated downstream by validators

Supporting/validation
- tools/form_guide_validator.py
  - Purpose: Validate structure/headers and content of form guide CSVs (pipe-delimited assumptions per project rules)
  - How to run: `python tools/form_guide_validator.py`

Legacy (fallbacks)
- form_guide_csv_scraper.py
- expert_form_csv_scraper.py
  - How to run: `python <script>.py`
  - Status: Superseded by `upcoming_race_browser.py`. Keep for fallback/diagnostics, or move under `archive/outdated_scripts/` if fully retired.

---

## B) Race data scraping (winners, weather) & post-processing

Primary (canonical)
- sportsbet_recent_races_scraper.py
  - Purpose: Scrape recent completed race results (winners, odds, margins, metadata) and persist to DB
  - How to run: `python sportsbet_recent_races_scraper.py`
  - External endpoints used:
    - Base: `https://www.sportsbet.com.au`
    - Results index: `/racing/results/greyhound-racing`
    - Also inspects Sportsbet racing pages (links/DOM) to gather completed race info (indicators like "FINAL")
  - Outputs:
    - SQLite DB updates (table: `race_metadata`) — fields include: race_id, venue, race_number, race_date, race_name, grade, distance, winner_name, winner_odds, winner_margin, url, extraction_timestamp

Supporting
- sportsbet_race_time_scraper.py
  - Purpose: Normalize/augment race times (complementary to recent results)
  - How to run: `python sportsbet_race_time_scraper.py`

Post-processing
- process_race_data.py
  - Purpose: Normalize/transform scraped artifacts to `processed/` and/or DB
  - How to run: `python process_race_data.py`

Compliance note
- Winner extraction is sourced from the race/result page (Sportsbet), NOT from form guides, matching project rules.

---

## C) Shared utilities, services & configuration
- config_loader.py — configuration bootstrap (has `__main__`)
- endpoint_cache.py — caches HTTP results and endpoints (has `__main__`)
- database_manager.py — DB connectivity and schema interactions
- services/guardian_service.py — watch `./upcoming_races` and `./processed` for integrity/automation
- services/guardian_cron_service.py — cron-friendly guardian wrapper with graceful shutdown
- file_naming_standards/file_naming_validator.py — ensures file naming like `Race N - VENUE - YYYY-MM-DD.csv`

---

## App/API integration (for reference)
Front-end and automation scripts consume internal app endpoints (served by the Flask app):
- GET `/api/upcoming_races_csv` (supports `?refresh=`, pagination and `?search=`)
- POST `/api/predict_single_race_enhanced` (accepts `race_id` or `race_filename`)
- POST `/api/predict_all_upcoming_races_enhanced`
- GET `/api/recent_races`

These are used by UI scripts (interactive-races.js, predictions_v2.js, ml-dashboard.js) and appear in code coverage reports.

---

## Outputs & file locations
- Historical (form guides): `upcoming_races/` with filenames `Race {race_number} - {VENUE_CODE} - YYYY-MM-DD.csv`
  - Staging/cache: `upcoming_races_temp/` (see README for archive policy)
  - Archival: `archive/upcoming_races/YYYY/MM/` (move outdated/redundant files here per repo rules)
- Processed (normalized): `processed/` (e.g., `processed/test_race_for_prediction.csv`)
- Database: `greyhound_racing_data.db` (tables include `race_metadata` and related)
- Predictions/analysis: `gpt_enhanced_predictions/` (JSON)
- Logs/diagnostics: `inventory.csv`, `debug_call_chain_*.log`, `referential_integrity_results.json`

---

## Canonical commands (quick reference)
- Download/refresh form guides (upcoming) — thedogs:
  - `python upcoming_race_browser.py`
- Validate form guide CSVs:
  - `python tools/form_guide_validator.py`
- Scrape recent winners/odds (Sportsbet) and store to DB:
  - `python sportsbet_recent_races_scraper.py`
- Post-process race data:
  - `python process_race_data.py`
- Optional: validate file naming
  - `python file_naming_standards/file_naming_validator.py`

---

## Notes / follow-ups
- API `/api/upcoming_races_csv` duplicates were observed in debug logs (same race_id repeated). Investigate upstream ID generation in the loader and the browser to ensure stable unique `race_id` per race (venue+date+number).
- Consider moving fully superseded scrapers (e.g., `form_guide_csv_scraper.py`, `expert_form_csv_scraper.py`) into `archive/outdated_scripts/` to declutter the root while retaining history.
- Ensure validators are part of the regular pipeline (pre-commit or post-download) to maintain CSV quality and naming standards.

