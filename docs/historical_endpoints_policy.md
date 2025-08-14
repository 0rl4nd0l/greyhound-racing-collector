Historical data and results scraping access policy

Summary
- Historical data = dog past performances (form guides). Race data = the race’s own results (winner, margins, weather, etc.).
- Only endpoints that analyze past races or enrich dog histories may use the comprehensive collector and TheGreyhoundRecorderScraper.
- Prediction endpoints for upcoming races must not trigger results scraping or post-race data collection.

Endpoints allowed to access comprehensive collector (historical workflows)
- GET /api/dogs/<dog_name>/form
  Purpose: Return historical form for a dog; may enrich with detailed race history if available.
  Access: May read comprehensive_dog_profiles and detailed_race_history. May use lazy-loaded collector only when COMPREHENSIVE_COLLECTOR_ALLOWED is true (ENABLE_RESULTS_SCRAPERS and not prediction_only mode). Does not scrape race winners for current races.

- POST /api/test_historical_prediction
  Purpose: Evaluate model on a historical race already in DB and compare to the stored winner.
  Access: Reads stored winner_name from race_metadata only. No scraping. The collector not needed here; relies on DB.

- GET /api/model/historical_accuracy
  Purpose: Aggregate historical model accuracy from logs/DB.
  Access: Read-only from DB. No scraping.

Endpoints that must NOT access results scraping
- /predict_page, POST /predict, POST /api/predict_batch, GET /api/prediction_results
  Purpose: Upcoming race predictions and viewing stored prediction files.
  Restriction: Must not import or initialize results scrapers. Guarded via module_guard and feature flags. Uses only form guide CSVs and historical data already persisted.

- Sportsbet odds endpoints: /api/sportsbet/*
  Purpose: Odds only. No winner/results scraping.

Lazy loading behavior validation
- ComprehensiveFormDataCollector uses lazy loading for TheGreyhoundRecorderScraper via _load_greyhound_recorder().
- Calls to collect_comprehensive_form_data() now ensure the scraper is initialized before use; if unavailable, collection gracefully skips recorder and returns success: false for that source without breaking the workflow.
- app.py uses get_comprehensive_collector_class(), gated by COMPREHENSIVE_COLLECTOR_ALLOWED. Prediction endpoints do not import the collector in prediction_only mode.

What was tested
- Ran historical-related tests (historical core/filtering and historical prediction). These passed in the current environment, indicating historical endpoints and logic remain functional under lazy loading.

Notes and enforcement
- Ensure ENABLE_RESULTS_SCRAPERS=false in environments where only upcoming predictions are desired. This keeps the collector and external scrapers out of memory and avoids accidental result scraping.
- The winner of a race must come from the race page itself and be stored in DB during dedicated results ingestion tasks, not from form guides. Historical endpoints should read winners from DB; they should not scrape live sites during prediction flows.

