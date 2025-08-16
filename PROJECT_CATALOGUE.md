# Greyhound Racing Predictor - Project Catalogue

## Project Overview
**Comprehensive AI-powered system for analyzing and predicting greyhound racing outcomes**

---

## Database Systems

### Primary Database
| Database | Type | Size | Description |
|----------|------|------|-------------|
| `greyhound_racing_data.db` | SQLite | 39.4 MB | Main unified database with 24 tables including race_metadata, dog_race_data, comprehensive_dog_profiles, gpt_analysis, weather_data, etc. |

### Legacy Databases
| Database | Type | Description |
|----------|------|-------------|
| `databases/race_data.db` | SQLite | Legacy race data storage |
| `databases/greyhound_racing.db` | SQLite | Empty legacy database |
| `databases/comprehensive_greyhound_data.db` | SQLite | Historical comprehensive data |
| `databases/unified_racing.db` | SQLite | Unified racing database |
| `databases/unified_data.db` | SQLite | Unified data storage |

---

## Prediction Methods

### Primary Prediction Systems
| System | File | Description | Key Features |
|--------|------|-------------|--------------|
| **PredictionPipelineV3** | `prediction_pipeline_v3.py` | Comprehensive integrated system with ML V4, weather, GPT | ML System V4, Weather Enhancement, GPT Integration, Fallback Hierarchy |
| **UnifiedPredictor** | `unified_predictor.py` | Unified interface with intelligent fallbacks | Intelligent Fallbacks, Centralized Config, Performance Monitoring, Caching |
| **ComprehensivePredictionPipeline** | `comprehensive_prediction_pipeline.py` | Multi-source data integration pipeline | Multi-source Integration, Data Quality Checks, Real-time Validation, Ensemble Methods |

### Machine Learning Systems
| System | File | Description | Algorithms |
|--------|------|-------------|------------|
| **MLSystemV3** | `ml_system_v3.py` | Comprehensive ML with drift monitoring, SHAP | GradientBoosting, XGBoost, RandomForest, LogisticRegression |
| **MLSystemV4** | `ml_system_v4.py` | Advanced ML with enhanced feature engineering | Advanced optimization |
| **ComprehensiveEnhancedMLSystem** | `comprehensive_enhanced_ml_system.py` | Enhanced ML with comprehensive features | Comprehensive feature engineering |

### Supporting Systems
| System | File | Description |
|--------|------|-------------|
| **TraditionalRaceAnalyzer** | `traditional_analysis.py` | Traditional race analysis as fallback |
| **WeatherEnhancedPredictor** | `weather_enhanced_predictor.py` | Weather-aware prediction enhancement |
| **GPTPredictionEnhancer** | `gpt_prediction_enhancer.py` | GPT-4 powered analysis and enhancement |

---

## Entry Points

### Primary Entry Points
| File | Type | Description | Features |
|------|------|-------------|----------|
| **app.py** | Flask Web Application | Main web app with comprehensive API endpoints (port 5000) | Web Dashboard, REST API, Real-time Monitoring, Prediction Endpoints |
| **run.py** | CLI Entry Point | Command-line interface for data tasks | Commands: collect, analyze, predict |

### Secondary Entry Points
| File | Type | Description |
|------|------|-------------|
| **main.py** | Event Scraper | EventScraper integration for sportsbook odds |

---

## Configuration Artifacts

### Docker & Containerization
| File | Description |
|------|-------------|
| `Dockerfile` | Docker config for Flask app deployment (python:3.11-slim) |
| `docker-compose.test.yml` | Testing environment with PostgreSQL, Redis, Chrome, Flask, Celery, RQ, Playwright |

### Build Tools
| File | Description |
|------|-------------|
| `Makefile` | Build scripts for install, test, lint, run, docker operations |

### CI/CD
| File | Description |
|------|-------------|
| `.github/workflows/ci.yml` | Main CI/CD pipeline with PostgreSQL, Redis, multi-Python testing |
| `.github/workflows/backend-tests.yml` | Backend-specific test workflows |
| `.github/workflows/ci-fasttrack.yml` | Fast-track CI workflow |

### Database
| File | Description |
|------|-------------|
| `alembic.ini` | Database migration configuration |
| `schema_contract.yaml` | Database schema contract for race_metadata table |

### Dependencies
| File | Description |
|------|-------------|
| `requirements.txt` | Python dependencies (Flask, pandas, scikit-learn, selenium, playwright, etc.) |
| `package.json` | Node.js frontend dependencies and build scripts |

---

## Data Sources

### Data Scrapers
| Scraper | Purpose | Target | Output/Features |
|---------|---------|--------|-----------------|
| **form_guide_csv_scraper.py** | Historical race data | Training data for ML models | CSV files in ./unprocessed |
| **upcoming_race_browser.py** | Live/upcoming races | Real-time prediction pipeline | Real race times, venue mapping, future filtering |
| **direct_racing_scraper.py** | Quick race overview | Today's and tomorrow's races | Race summaries with estimated times |
| **hybrid_odds_scraper.py** | Betting odds collection | Live betting odds | API-first approach, Selenium fallback, DataFrame output |

### Data Processors
| Processor | Description |
|-----------|-------------|
| **enhanced_comprehensive_processor.py** | Enhanced data processing with comprehensive analysis |
| **csv_ingestion.py** | CSV data ingestion and processing |
| **enhanced_data_integration.py** | Advanced data integration from multiple sources |

---

## API Endpoints

### Prediction Endpoints
- `/api/predict_single_race_enhanced` - Enhanced single race prediction
- `/api/predict_all_upcoming_races_enhanced` - Enhanced batch prediction
- `/api/predict` - Legacy unified prediction
- `/api/ml-predict` - ML prediction simulation

### Data Endpoints
- `/api/dogs/search` - Search dogs by name
- `/api/dogs/<dog_name>/details` - Detailed dog statistics
- `/api/dogs/<dog_name>/form` - Comprehensive form guide
- `/api/dogs/top_performers` - Top performing dogs
- `/api/dogs/all` - Paginated dog listing
- `/api/races/paginated` - Paginated race browsing
- `/api/upcoming_races_csv` - Upcoming races from CSV
- `/api/races` - All races listing

### Batch Operations
- `/api/batch/predict` - Create batch prediction job
- `/api/batch/status/<job_id>` - Check job status
- `/api/batch/cancel/<job_id>` - Cancel job
- `/api/batch/progress/<job_id>` - Job progress
- `/api/batch/stream` - Stream batch results

### System Management
- `/api/health` - Health check
- `/api/system_status` - System monitoring
- `/api/enable-explain-analyze` - Enable query analysis
- `/ws` - WebSocket endpoint

### GPT Enhancement
- `/api/gpt/enhance_race` - GPT race enhancement
- `/api/gpt/daily_insights` - Daily GPT insights
- `/api/gpt/enhance_multiple` - Multi-race GPT enhancement
- `/api/gpt/comprehensive_report` - Comprehensive GPT report
- `/api/gpt/status` - GPT integration status

---

## Feature Engineering

### FeatureStore System (`features.py`)
- **V3BoxPositionFeatures** - Box position analysis
- **V3CompetitionFeatures** - Competition level analysis
- **V3DistanceStatsFeatures** - Distance performance statistics
- **V3RecentFormFeatures** - Recent form analysis
- **V3TrainerFeatures** - Trainer performance features
- **V3VenueAnalysisFeatures** - Venue-specific analysis
- **V3WeatherTrackFeatures** - Weather and track condition features

---

## Directory Structure

### Data Directories
`databases/`, `unprocessed/`, `processed/`, `upcoming_races/`, `predictions/`, `form_guides/`, `enhanced_expert_data/`, `comprehensive_form_cache/`

### Model Directories
`model_registry/`, `comprehensive_trained_models/`, `advanced_models/`

### Archive Directories
`archive/`, `archive_unused_scripts/`, `quarantine/`

### Testing Directories
`tests/`, `cypress/`, `test_upcoming_races/`

### Documentation Directories
`docs/`, `debug_logs/`, `backups/`

---

## Technologies

### Backend
Python, Flask, SQLite, Pandas, Scikit-learn, XGBoost, SHAP

### Frontend
JavaScript, HTML, CSS

### Testing
pytest, Playwright, Cypress, Locust

### Infrastructure
Docker, GitHub Actions, Alembic

### AI/ML
OpenAI GPT-4, Optuna, MLflow, SMOTE-NC

---

## Summary

This is a sophisticated, production-ready greyhound racing prediction system with:
- **Multiple prediction methods** with intelligent fallback hierarchy
- **Comprehensive data integration** from various sources
- **Advanced ML systems** with drift monitoring and explainability
- **Full-stack web application** with REST API
- **Robust testing and CI/CD** infrastructure
- **Container-ready deployment** with Docker
- **GPT-4 integration** for enhanced analysis
