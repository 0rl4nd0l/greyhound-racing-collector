# FastTrack Redundancy Matrix

This document analyzes data overlap between the new FastTrack scraper system and existing data collection scripts.

## Methodology

We compared data fields, output formats, and ingestion patterns across all existing collector scripts against the FastTrack system capabilities. Each script was evaluated on three criteria:

- **Full Overlap**: Data is completely redundant with FastTrack
- **Partial Overlap**: Some data overlap, but unique fields exist  
- **New Unique Information**: Provides data not available in FastTrack

## Data Field Comparison

### FastTrack Capabilities
- Race metadata (date, venue, race number, name)
- Dog profiles (pedigree, age, ear brand, career statistics)
- Performance data (sectional times, margins, PIR ratings)
- Expert form analysis (PDF extracts)
- Enhanced race conditions and weather
- Detailed prize money and odds data

## Redundancy Analysis

| Existing Script | Full Overlap | Partial Overlap | Unique Info | Data Fields | Dependencies | Recommendation |
| :--- | :---: | :---: | :---: | :--- | :--- | :--- |
| `direct_racing_scraper.py` | ✓ | | | Race metadata, basic race info | `app.py` | **DEPRECATE** - FastTrack provides superior race data |
| `event_scraper.py` | ✓ | | | Basic odds data, market selections | `hybrid_odds_scraper.py` | **DEPRECATE** - FastTrack includes comprehensive odds |
| `form_guide_csv_scraper.py` | | ✓ | | Historical form, venue mapping | Multiple pipelines | **MERGE** - Keep for historical backfill, migrate to FastTrack |
| `hybrid_odds_scraper.py` | ✓ | | | Live odds compilation | `app.py` | **DEPRECATE** - FastTrack provides real-time odds |
| `odds_scraper_system.py` | ✓ | | | Structured odds storage | `app.py` | **DEPRECATE** - Redundant with FastTrack odds system |
| `safe_data_ingestion.py` | | | ✓ | Generic validation framework | System-wide | **KEEP** - Critical for FastTrack adapter validation |
| `sportsbet_odds_integrator.py` | ✓ | | | Live odds, value bets | `app.py` | **DEPRECATE** - FastTrack supersedes sportsbook-specific data |
| `sportsbet_recent_races_scraper.py` | | ✓ | | Sportsbet-specific race results | `app.py` | **MERGE** - Evaluate unique sportsbook fields |
| `sportsbet_race_time_scraper.py` | ✓ | | | Race timing data | `comprehensive_prediction_pipeline.py` | **DEPRECATE** - FastTrack includes superior timing data |
| `enhanced_data_integration.py` | | ✓ | | Sectional times, performance indicators | `comprehensive_prediction_pipeline.py` | **MERGE** - Enhanced analysis may complement FastTrack |

