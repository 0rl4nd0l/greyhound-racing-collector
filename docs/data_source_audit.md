## Data Source Audit

| File/Module Name | Data Fields Provided | Storage Destination | Downstream Dependencies | FastTrack Overlap |
| :--- | :--- | :--- | :--- | :--- |
| `direct_racing_scraper.py` | Race metadata (date, venue, race number, time, name, URL) | In-memory list, passed to other components | `app.py` (for displaying today's races) | Yes |
| `event_scraper.py` | Odds data (event name, market ID, selections, odds) | CSV file or in-memory DataFrame | `hybrid_odds_scraper.py` | Yes |
| `form_guide_csv_scraper.py` | Greyhound form data (dog name, sex, placing, box, weight, distance, date, track, grade, time, win time, bonus, first split, margin, PIR, starting price) | CSV files in `unprocessed` and `form_guides/downloaded` directories, `greyhound_racing.db` database | `comprehensive_prediction_pipeline.py` | Yes |
| `hybrid_odds_scraper.py` | Odds data | In-memory DataFrame | `app.py` | Yes |
| `odds_scraper_system.py` | Odds data | `greyhound_racing_data.db` database | `app.py` | Yes |
| `safe_data_ingestion.py` | Generic data ingestion for various tables | `greyhound_racing_data.db` database | `app.py` | No |
| `sportsbet_odds_integrator.py` | Live odds, value bets, race metadata | `greyhound_racing_data.db` database | `app.py` | Yes |
| `sportsbet_recent_races_scraper.py` | Recent race data | `greyhound_racing_data.db` database | `app.py` | Yes |
| `sportsbet_race_time_scraper.py` | Race times | In-memory list, passed to other components | `comprehensive_prediction_pipeline.py` | Yes |
| `enhanced_data_integration.py` | Enhanced dog data (sectional times, PIR ratings, weight history, margins, performance indicators) | In-memory dictionary, passed to other components | `comprehensive_prediction_pipeline.py` | No |
| `comprehensive_prediction_pipeline.py` | Predictions and analysis | `predictions` directory (JSON files) | `app.py` | No |
| `ml_system_v3.py` | Machine learning models | `ml_models_v3` directory (joblib files) | `comprehensive_prediction_pipeline.py` | No |
| `app.py` | Web application | `greyhound_racing_data.db` database | None | No |

