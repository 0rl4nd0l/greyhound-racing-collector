# Odds Coverage & Normalization HOWTO

This short guide shows how to normalize live_odds race IDs to the canonical format and generate the coverage report.

Canonical race_id format
- VENUECODE_YYYY-MM-DD_RN (e.g., AP_K_2025-09-10_3)
- Ensures consistent joins between predictions, race_metadata, and live_odds.

Run normalization (idempotent)
- Uses config/venue_mapping.py to standardize venues and a best-effort date parser to enforce ISO dates.

Commands:
- Makefile target (recommended):
  make normalize-odds DATABASE_PATH=./greyhound_racing_data_writable.db

- Direct script invocation:
  python scripts/normalize_live_odds_race_ids.py --db ./greyhound_racing_data_writable.db

Notes:
- Non-destructive and idempotent: upserts race_metadata minimal rows and updates live_odds.race_id when needed.
- Works without scraping; transforms existing DB rows.

Generate coverage report (extended)
- Includes:
  - predictions (last N hours)
  - predictions_latest (latest per (race_id,dog))
  - odds_coverage_by_venue (top venues with current odds joined to predictions_latest)

Commands:
- Makefile target:
  make report-coverage HOURS=24 DATABASE_PATH=./greyhound_racing_data_writable.db

- Direct:
  python scripts/report_prediction_coverage.py --hours 24 --db ./greyhound_racing_data_writable.db --save docs/analysis/prediction_coverage_report_$(date +%Y%m%d_%H%M%S).json

Tips
- Consider running normalize-odds periodically (e.g., hourly) to keep live_odds consistent as new data arrives.
- To improve odds coverage further, run your odds integrator/refresh and then re-run the coverage report.

