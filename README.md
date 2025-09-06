# Greyhound Racing Predictor

> **📁 Archive Notice**: As of September 1, 2025, this project has been optimized from 23GB to 3.8GB.  
> **19.2GB of historical data** has been safely archived to external storage.  
> See **[ARCHIVE_INDEX.md](docs/ARCHIVE_INDEX.md)** for complete archive details and restoration instructions.

## Overview

A comprehensive machine learning system for predicting greyhound race outcomes using advanced feature engineering, temporal leakage protection, and probability calibration.

For common commands and a high-level architecture overview, see WARP.md.

See also: [V4 Prediction System Analysis](reports/prediction_system_analysis.md) — full prediction flow, formulas (normalization, confidence, EV), environment toggles, and failure modes.

## Key Features

This pipeline provides end-to-end functionality for:
- **CSV Data Parsing**: Parse race data from various CSV formats
- **Feature Engineering**: Generate 50+ features with temporal leakage protection
- **ML Model Scoring**: Use trained models to predict race outcomes
- **Probability Calibration**: Convert raw scores to calibrated win probabilities
- **Ranked Output**: Generate professional race predictions with confidence metrics

---

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone repository-url
cd greyhound_racing_collector
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify installation:**
```bash
python greyhound_prediction_pipeline.py --help
```

---

## Core Functions

### 1. `parse_csv`

Parses CSV race data and extracts structured information.
- Output: Parsed JSON and statistics

### 2. `feature_engineer`

Generates features for prediction:
- Includes recent form, venue analysis, competition level, etc.

### 3. `score`

Scores dogs using trained models.
- Uses ensemble methods and feature importance analysis

### 4. `probabilities`

Converts scores to calibrated win probabilities.

---

### Basic Usage

#### Parse CSV Race Data
```bash
python greyhound_prediction_pipeline.py parse --input races.csv --output ./parsed/
```

#### Predict Single Race
```bash
python greyhound_prediction_pipeline.py predict --race race_data.json --output predictions.csv
```

#### Run Full Pipeline
```bash
python greyhound_prediction_pipeline.py full-pipeline --input races.csv --output ./results/
```

## Advanced Usage

### Custom Model Training

```python
from greyhound_prediction_pipeline import GreyhoundPredictionPipeline
from ml_system_v4 import MLSystemV4

# Initialize systems
pipeline = GreyhoundPredictionPipeline()
ml_system = MLSystemV4()

# Train new model
ml_system.train_model(training_data_path="historical_races.csv")
```

### Batch Processing

```python
import pandas as pd
from pathlib import Path

# Process multiple race files
race_files = Path("./races/").glob("*.csv")
results = []

for race_file in race_files:
    result = pipeline.run_full_pipeline(
        str(race_file), 
        f"./outputs/{race_file.stem}/"
    )
    results.append(result)
```

### API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = GreyhoundPredictionPipeline()

@app.route('/predict', methods=['POST'])
def predict_race():
    race_data = request.json
    
    features = pipeline.feature_engineer(race_data)
    scores = pipeline.score(features)
    probabilities = pipeline.probabilities(scores)
    
    return jsonify(probabilities)
```

## License

This project is proprietary. All rights reserved.

## Support

For issues and questions:
- Create GitHub issue for bugs
- Check existing documentation
- Review test cases for examples

## Version History

- **v4.0**: Current version with temporal leakage protection
- **v3.0**: Enhanced feature engineering
- **v2.0**: Initial ML pipeline
- **v1.0**: Basic prediction system

---

## Acknowledgments

- Built using scikit-learn, XGBoost, and pandas
- Feature engineering inspired by racing domain expertise
- Temporal validation methodology from time series forecasting best practices
- **GPT Enhancement**: AI-powered race analysis using OpenAI GPT-4 for narrative insights and betting strategies

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js and npm (for frontend development)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/greyhound-racing-predictor.git
   cd greyhound-racing-predictor
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies (if you need to modify the frontend):**
   ```bash
   npm install
   ```

5. **Initialize the unified database:**
   ```bash
   python create_unified_database.py
   ```
   This script will create `greyhound_racing_data.db` and populate it with data from any existing legacy databases or CSV files.

### Running the Application

- **Start the Flask server:**
  ```bash
  python app.py
  ```
The application will be available at `http://127.0.0.1:5002` (or the port specified by the `PORT` environment variable).

#### Port Configuration & Troubleshooting

The application uses **port 5002** by default. You can override this using the `PORT` environment variable:

```bash
# Run on a different port
PORT=8080 python app.py

# Or set it persistently
export PORT=8080
python app.py
```

### UI Modes and Feature Flags

The web UI supports a simplified and an advanced mode, plus an opt-in dynamic endpoints menu.

- UI_MODE: Controls navigation complexity and some heavy assets.
  - simple (default): Minimal top-level nav (Dashboard, Upcoming, Predict, Logs).
  - advanced: Full navigation (Races, Analysis, AI/ML, System, Help, etc.).

  Examples:
  - macOS/Linux
    - UI_MODE=simple python app.py
    - UI_MODE=advanced python app.py
  - Windows PowerShell
    - $env:UI_MODE='simple'; python app.py
    - $env:UI_MODE='advanced'; python app.py

- ENABLE_ENDPOINT_DROPDOWNS: Toggle the dynamic endpoints dropdown toolbar (opt-in).
  - 0 (default): Disabled.
  - 1: Enabled. Injects a menu that enumerates available Flask endpoints for quick navigation (primarily for dev/testing).

  Examples:
  - macOS/Linux: ENABLE_ENDPOINT_DROPDOWNS=1 python app.py
  - Windows PowerShell: $env:ENABLE_ENDPOINT_DROPDOWNS='1'; python app.py

Notes:
- All advanced routes remain accessible by direct URL in both modes; only visibility in the navbar changes.
- CI can set UI_MODE=advanced to exercise the full UI surface.

##### Port Conflict Resolution

If you encounter "port already in use" errors, use these troubleshooting commands:

```bash
# Check what's using a specific port
lsof -i :5002

# Find all Python processes using ports
lsof -i -P | grep python

# Kill a specific process by PID (replace XXXX with actual PID)
kill XXXX

# Kill all Python processes (use with caution)
pkill -f python

# Check if the service is responding
curl http://localhost:5002/api/health
```

**Health Check Endpoint**: The application provides a health check at `/api/health` which returns JSON with system status, component availability, and version information.

- **Dashboard Access**: Visit `/gpt-enhancement` for the GPT Enhancement Dashboard.

- **Data Processing and Predictions:**
  The `run.py` script provides command-line access to core functions:
  ```bash
  # Scrape the latest race data
  python run.py collect

  # Process all unprocessed data and populate the database
  python run.py analyze

  # Run predictions on all upcoming races
  python run.py predict
  ```

## Advisory Workflow

The application includes an integrated advisory system that provides quality assessment and warnings for prediction results.

### Advisory Features

- **Automatic Quality Assessment**: Analyzes prediction confidence, calibration, and data quality
- **Color-Coded Messages**: INFO (green), WARNING (yellow), CRITICAL (red) message types
- **Collapsible Details**: Expandable sections for detailed issue explanations
- **OpenAI Integration**: AI-powered summaries when API key is available, template fallback
- **Real-time Integration**: Non-blocking workflow integration with prediction pipeline

### Advisory API Endpoints

#### Generate Advisory

- **POST /api/generate_advisory**

  Generate advisory messages for prediction data or file.

  **Request Body** (JSON):
  ```json
  {
    "prediction_data": {
      "race_id": "race_identifier",
      "race_date": "2025-08-04",
      "predictions": [
        {"dog_name": "Test Dog", "box_number": 1, "win_prob": 0.4, "confidence": 0.8}
      ]
    }
    // OR
    "file_path": "/path/to/prediction.json"
  }
  ```

  **Response**:
  ```json
  {
    "success": true,
    "messages": [
      {
        "type": "WARNING",
        "category": "quality_assessment",
        "title": "Moderate Quality Predictions",
        "message": "Prediction quality score: 75/100 - Some issues detected",
        "timestamp": "2025-08-04T12:00:00Z"
      }
    ],
    "human_readable_summary": "WARNING: 2 warnings identified. Overall quality score: 75/100. Review recommended.",
    "ml_json": {
      "summary": {"total_messages": 2, "quality_score": 75},
      "feature_flags": {"has_quality_issues": true, "low_quality_score": false}
    },
    "processing_time_ms": 23.5,
    "openai_used": false
  }
  ```

### Frontend Integration

The advisory system includes JavaScript utilities for frontend integration:

```javascript
// Load advisory utilities
<script src="/static/js/advisoryUtils.js"></script>

// Render advisory in UI
const advisoryData = {
  title: "Quality Assessment",
  message: "Prediction analysis completed",
  type: "warning",
  details: ["Low confidence detected", "Class imbalance present"],
  helpText: "These issues may affect prediction accuracy"
};

const container = document.getElementById('advisory-container');
AdvisoryUtils.renderAdvisory(advisoryData, container);
```

### Usage Examples

#### Command Line Testing
```bash
# Test advisory with sample data
python3 advisory.py --test

# Generate advisory for specific file
python3 advisory.py --file prediction_result.json
```

#### API Integration
```bash
# Test advisory API endpoint
curl -X POST http://127.0.0.1:5002/api/generate_advisory \
  -H "Content-Type: application/json" \
  -d '{"prediction_data": {"race_id": "test", "predictions": []}}'
```

#### Workflow Integration
The advisory system is designed to integrate seamlessly with the prediction workflow:

1. **Non-blocking**: Advisory generation runs in parallel, not blocking predictions
2. **Error-tolerant**: Falls back gracefully when OpenAI unavailable
3. **Performance-optimized**: Completes in under 100ms for typical cases
4. **UI-ready**: Provides structured data for frontend rendering

## API Documentation

The application provides a comprehensive RESTful API for interacting with the prediction system and its data. Below are the key endpoints.

Tip: To validate V4 feature contracts via the UI, see docs/CONTRACT_VALIDATION.md (UI-based validation).

### Upcoming Races CSVs (source of truth)
- Folder path: `./upcoming_races`
- File extension: `.csv`
- Required naming pattern: `Race {number} - {VENUE} - {YYYY-MM-DD}.csv`
  - Example: `Race 1 - WPK - 2025-02-01.csv`
- The Upcoming UI enumerates files from this folder automatically; no manual list needed
- The predictions UI uses PredictionPipelineV4 on the selected CSV, with fallbacks (V3, legacy)
- Helper scripts to maintain naming consistency:
  - `scripts/normalize_upcoming_to_api_pattern.py`
  - `scripts/alias_upcoming_api_names_safe.py`

### Main Prediction Endpoints

#### Enhanced Single Race Prediction

-   **POST /api/predict_single_race_enhanced**

    **Primary endpoint** for getting predictions for a single race with automatic data enhancement and intelligent pipeline selection.

    **Request Body** (JSON):

    ```json
    {
        "race_filename": "Race 4 - GOSF - 2025-07-28.csv"
        // OR
        "race_id": "race_identifier"
    }
    ```

    **Key Features**:
    - Accepts either `race_filename` or `race_id` parameter
    - Automatically searches multiple directories (upcoming, historical)
    - Intelligent prediction pipeline selection:
      1. **PredictionPipelineV3** (primary - most advanced)
      2. **UnifiedPredictor** (fallback)
      3. **ComprehensivePredictionPipeline** (final fallback)
    - Enhanced error handling with detailed failure information

    **Response**:

    ```json
    {
        "success": true,
        "race_id": "extracted_or_provided_id",
        "race_filename": "Race 4 - GOSF - 2025-07-28.csv",
        "predictions": [
            {
                "dog_name": "Lightning Bolt",
                "box_number": 1,
                "win_probability": 0.35,
                "place_probability": 0.67,
                "confidence_score": 0.84,
                "predicted_position": 1,
                "reasoning": "Strong recent form and favorable track conditions"
            }
        ],
        "predictor_used": "PredictionPipelineV3",
        "file_path": "/path/to/race/file.csv",
        "enhancement_applied": true,
        "timestamp": "2025-01-15T10:30:00Z"
    }
    ```

#### Enhanced Batch Prediction

-   **POST /api/predict_all_upcoming_races_enhanced**

    **Batch endpoint** for predicting all upcoming races with comprehensive error handling, logging, and performance monitoring.

    **Request Body** (Optional JSON):

    ```json
    {
        "max_races": 10,
        "force_rerun": false
    }
    ```

    **Key Features**:
    - Automatically discovers all CSV files in upcoming races directory
    - Intelligent pipeline selection with fallbacks
    - Comprehensive error tracking and recovery
    - Performance monitoring and timing metrics
    - Detailed success/failure reporting

    **Response**:

    ```json
    {
        "success": true,
        "total_races": 5,
        "success_count": 4,
        "predictions": [
            {
                "race_filename": "Race 1 - GOSF - 2025-01-15.csv",
                "success": true,
                "predictions": [...],
                "processing_time_ms": 2340,
                "pipeline_used": "ComprehensivePredictionPipeline"
            }
        ],
        "errors": [
            "Race 5 prediction failed: insufficient historical data"
        ],
        "pipeline_type": "ComprehensivePredictionPipeline",
        "processing_time_seconds": 45.2
    }
    ```

    **Pipeline Selection Logic**:
    1. **ComprehensivePredictionPipeline** (primary - handles batch operations)
    2. **PredictionPipelineV3** (fallback - individual race processing)
    3. Detailed error reporting if both fail

### Data Flow Architecture

The enhanced prediction system follows this data flow:

```
[Race CSV Files] → [Pipeline Selection] → [Data Enhancement] → [ML Processing] → [Prediction Results]
                           ↓
        ┌─ PredictionPipelineV3 (Advanced ML + Features)
        ├─ ComprehensivePredictionPipeline (Batch + Comprehensive)
        └─ UnifiedPredictor (Legacy Fallback)
```

**Key Improvements**:
- **Automatic Enhancement**: All predictions include data enrichment from multiple sources
- **Intelligent Fallbacks**: System gracefully degrades through available prediction methods
- **Error Recovery**: Comprehensive error handling prevents total system failures
- **Performance Monitoring**: Detailed timing and success metrics for optimization

### Data Endpoints

-   **GET /api/dogs/search?q=<query>**: Search for greyhounds by name.
-   **GET /api/dogs/<dog_name>/details**: Get detailed statistics and historical performance for a specific dog.
-   **GET /api/races/paginated**: Browse historical races with powerful search, sorting, and pagination options.

### System Management

-   **GET /api/system_status**: Real-time monitoring of logs, model performance, and database health.
-   **POST /api/process_data**: Trigger a background task to process all unprocessed data files.

### GPT Enhancement Endpoints

-   **POST /api/gpt/enhance_race**: Enhance a race with GPT analysis
    ```bash
    curl -X POST http://127.0.0.1:5002/api/gpt/enhance_race \
      -H "Content-Type: application/json" \
      -d '{"race_file_path": "./upcoming_races/Race 1 - GOSF - 2025-07-31.csv"}'
    ```

-   **GET /api/gpt/daily_insights**: Get GPT daily insights for a specific date
    ```bash
    curl "http://127.0.0.1:5002/api/gpt/daily_insights?date=2025-07-31"
    ```

-   **POST /api/gpt/enhance_multiple**: Enhance multiple races with GPT analysis
    ```bash
    curl -X POST http://127.0.0.1:5002/api/gpt/enhance_multiple \
      -H "Content-Type: application/json" \
      -d '{"race_files": ["race1.csv", "race2.csv"], "max_races": 3}'
    ```

-   **POST /api/gpt/comprehensive_report**: Generate comprehensive GPT report
    ```bash
    curl -X POST http://127.0.0.1:5002/api/gpt/comprehensive_report \
      -H "Content-Type: application/json" \
      -d '{"race_ids": ["race123", "race456"]}'
    ```

-   **GET /api/gpt/status**: Check GPT integration status
    ```bash
    curl "http://127.0.0.1:5002/api/gpt/status"
    ```

## Recommendations for Long-Term Stability

To ensure the continued stability, maintainability, and performance of the Greyhound Racing Predictor, the following are highly recommended:

### 1. Database Schema Migrations with Alembic

As the system evolves, the database schema will inevitably change. Manually managing these changes is error-prone and can lead to data integrity issues. **Alembic** is the official database migration tool for SQLAlchemy and provides a robust, version-controlled system for managing schema updates.

**Benefits**:

-   **Version Control**: Schema changes are stored as scripts in your repository.
-   **Repeatability**: Easily upgrade and downgrade database schemas across different environments.
-   **Safety**: Reduces the risk of manual errors during database updates.

### 2. Automated Testing with GitHub Actions

To maintain code quality and prevent regressions, it is crucial to automate the testing process. **GitHub Actions** provides a simple yet powerful way to build a continuous integration (CI) pipeline directly within your GitHub repository.

**Recommended Workflow**:

1.  **On every `push` or `pull_request`**:
2.  **Set up Python** and install dependencies from `requirements.txt`.
3.  **Run the `pytest` test suite** to ensure all existing functionality works as expected.
4.  (Optional) **Deploy to a staging environment** for further testing.

This will ensure that any new changes are automatically validated, providing confidence and stability for the project.

## Testing

To ensure the system works correctly after setup or changes:

### Backend Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test files
pytest tests/test_prediction_api.py
```

### System Integration Tests

```bash
# Test the prediction pipeline
python test_unified_system.py

# Test database integrity
python test_unified_schema.py
```

### Model Comparison Testing

The system includes a comprehensive model comparison harness for evaluating ML models against historical race data:

```bash
# Compare all models against historical forms
python tests/model_comparison_harness.py --model all --csv-dir archive/historical_forms

# Test specific model (v3, v3s, v4)
python tests/model_comparison_harness.py --model v4 --csv-dir data/test_races/

# Enable verbose logging
python tests/model_comparison_harness.py --model all --csv-dir archive/historical_forms --verbose
```

#### Expected Folder Layout

The model comparison harness expects CSV files organized in the following structure:

```
archive/historical_forms/               # Root directory for historical race data
├── Race_1_GOSF_2025-07-28.csv        # Individual race files
├── Race_2_RICH_2025-07-29.csv        # Format: Race_[Number]_[Venue]_[Date].csv
├── Race_3_APWE_2025-07-30.csv
└── ...

 data/test_races/                       # Alternative test data directory
├── test_race_1.csv
├── test_race_2.csv
└── ...
```

**CSV File Requirements:**
- Must contain `Dog Name` or `dog_name` column
- Should include racing metadata: `BOX`, `Weight`, `Distance`, `Venue`, etc.
- Post-outcome columns (like `PLC`, `finish_position`) are automatically stripped to prevent temporal leakage
- Files are preprocessed according to FORM_GUIDE_SPEC.md standards

#### Output Artifacts

The harness generates comprehensive comparison reports:

1. **JSON Results File**: `model_comparison_results_YYYYMMDD_HHMMSS.json`
   - Detailed predictions from each model
   - Performance metrics and timing data
   - Model metadata and configuration info
   - Success/failure tracking per race

2. **Console Output**:
   - Race-by-race prediction summaries
   - Model performance statistics:
     - Success rates and prediction counts
     - Top-1 accuracy and Brier scores
     - Expected Value (EV) correlations
     - Confidence vs. dispersion metrics
     - Calibration usage statistics

3. **Performance Metrics**:
   - **Success Rate**: Percentage of races successfully predicted
   - **Top-1 Accuracy**: How often the top-ranked dog wins
   - **Brier Score**: Probabilistic prediction accuracy (lower is better)
   - **EV Correlation**: Alignment between predicted and realized expected value
   - **Variance Spread**: Consistency between raw and normalized probabilities
   - **Calibration Count**: Number of races using probability calibration

#### Model Types

- **v3**: Full ML System with comprehensive features
- **v3s**: Simplified ML System (basic configuration)
- **v4**: Leakage-safe ML System with enhanced data validation
- **all**: Tests all available models and provides comparative analysis

### Frontend Tests (if applicable)

```bash
# Run JavaScript unit tests
npm test

# Run end-to-end tests
npm run cypress:run
```

## Data Collection Scrapers

The system includes several specialized scrapers for different data collection needs:

### Historical Data Collection

- **`form_guide_csv_scraper.py`**
  - **Purpose**: Downloads CSV form guides for **historical races only** (previous day or earlier)
  - **Target**: Training data collection for machine learning models
  - **Usage**: `python form_guide_csv_scraper.py`
  - **Output**: CSV files in `./unprocessed` directory
  - **Note**: Automatically skips current day and future races

### Live/Upcoming Race Data

- **`upcoming_race_browser.py`**
  - **Purpose**: Fetches **upcoming races** with real-time race scheduling
  - **Target**: Live prediction data pipeline
  - **Usage**: Import and use `UpcomingRaceBrowser` class
  - **Features**: Real race times, venue mapping, future race filtering
  - **Output**: Race metadata for prediction pipeline

### Quick Race Listing

- **`direct_racing_scraper.py`**
  - **Purpose**: Quick scrape of today's and tomorrow's races from main page
  - **Target**: Immediate race overview
  - **Usage**: `python direct_racing_scraper.py`
  - **Output**: Race summaries with estimated times

### Odds Collection

- **`hybrid_odds_scraper.py`**
  - **Purpose**: Reliable odds scraping with API + Selenium fallback
  - **Target**: Live betting odds for any race
  - **Features**: Professional API-first approach with DOM scraping backup
  - **Usage**: Import `HybridOddsScraper` class
  - **Output**: Structured odds data as pandas DataFrame

### Which Scraper to Use?

| Need | Scraper | When to Use |
|------|---------|-------------|
| **Training Data** | `form_guide_csv_scraper.py` | Building ML models with historical race results |
| **Live Predictions** | `upcoming_race_browser.py` | Getting upcoming races for real-time predictions |
| **Quick Overview** | `direct_racing_scraper.py` | Checking what races are available today/tomorrow |
| **Betting Odds** | `hybrid_odds_scraper.py` | Getting current odds for any specific race |

## Upcoming Race CSVs: Location, Naming, Schema, and Archiving

To standardize developer workflows and keep the repository tidy, use the following conventions for upcoming race CSVs.

- Folder path (default): `./upcoming_races_temp` (configurable via UPCOMING_RACES_DIR)
- Environment variable: `UPCOMING_RACES_DIR` points to the folder where upcoming race CSVs are stored

Folder path
- Default: `./upcoming_races_temp`
- Override: set `UPCOMING_RACES_DIR` in your `.env` or environment (see below)

Filename convention
- Pattern: `Race {race_number} - {VENUE_CODE} - YYYY-MM-DD.csv`
  - Example: `Race 4 - GOSF - 2025-07-28.csv`
- Allowed characters: letters, numbers, spaces, dashes, and underscores
- Extension: `.csv` (lowercase)

CSV schema and dialect (race data, not form guides)
- Purpose: These files contain race data for a not-yet-run race. They must NOT include results-only fields. Per project rules, winners come from the race page, not historical form guides.
- Encoding: UTF-8 (no BOM preferred)
- Delimiter: comma `,`
- Quoting: standard CSV quoting for fields containing commas or newlines
- Line endings: `\n` (LF) preferred; `\r\n` accepted
- Header row: required

Required columns (minimum)
- `race_date` (YYYY-MM-DD)
- `venue_code` (e.g., GOSF, RICH)
- `race_number` (1-12)
- `dog_name`
- `box` (1-8)

Recommended columns (optional)
- `trainer`, `weight`, `distance`, `grade`, `meeting_name`, `scheduled_time_local`

Forbidden/empty-at-ingest columns
- Outcome fields such as `PLC`, `finish_position`, `winner`, `margin`, `winning_time` must be absent or blank for upcoming races.

Validation notes
- Dog blocks: Expect up to 8 unique dogs; blank continuation rows are not used for upcoming race CSVs.
- Mixed delimiters and invisible unicode should be avoided. See `docs/FORM_GUIDE_SPEC.md` for detection tips if needed.

Setting UPCOMING_RACES_DIR
- Local `.env` (preferred during development):
  - Add `UPCOMING_RACES_DIR=./upcoming_races_temp`
- Shell session (temporary):
  - `export UPCOMING_RACES_DIR=./upcoming_races_temp`
- Docker Compose:
  - `environment:` section of the service should include `UPCOMING_RACES_DIR=/app/upcoming_races_temp`
  - Mount the host folder to the same container path, e.g.:
    - `- ./upcoming_races_temp:/app/upcoming_races_temp` (add `:Z` on SELinux hosts)
- Systemd or hosting envs:
  - Add an Environment entry: `Environment=UPCOMING_RACES_DIR=/srv/greyhound/upcoming_races`
  - Ensure the service user has read and execute permission on the directory (e.g., `chmod 750` and group membership)

Archiving procedure for outdated upcoming CSVs
- Archive-first policy: move outdated or redundant CSVs to the archive, do not delete.
- Suggested structure: `archive/upcoming_races/YYYY/MM/`
- Process:
  1. Move file into `archive/upcoming_races/{YYYY}/{MM}/`
  2. Preserve original filename
  3. Optionally append a timestamp if a name conflict occurs: `..._archived-YYYYMMDD-HHMMSS.csv`
  4. Record move in your commit message (and manifest if you maintain one)
- Rationale: Keeps main directories clean and complies with project archival rules. Always check the archive before creating a new file with the same race identity.

Note on historical vs race data
- Historical data (form guides) live under `./unprocessed` → processed → database; format described in `docs/FORM_GUIDE_SPEC.md`
- Race data (upcoming CSVs) use the schema above and are for prediction inputs; winners and race outcomes are scraped from the race page, not inferred from the form guide.

## Dependencies

### Backend (pip)

-   `pandas`
-   `numpy`
-   `requests`
-   `urllib3`
-   `pytest`
-   `beautifulsoup4`
-   `flask`
-   `flask-cors`
-   `python-dotenv`
-   `scikit-learn`
-   `joblib`
-   `selenium`

### Frontend (npm)

-   `jspdf`
-   `papaparse`
-   (See `package.json` for a full list of dev dependencies)

## Additional Features

### Header Requirements

The following headers are required for input CSV files:

- `Dog Name`: The name of the dog
- `PLC`: Place of the dog in the race
- `BOX`: The box number from which the dog started

### Manifest Behavior

Each batch job generates a manifest file capturing:

- Workflow steps
- Timestamps
- Model versions and hyperparameters
- Data checksums

These files ensure reproducibility and tracking of job progress.

### Debug Mode Usage

Enable detailed logging by setting the environment variable `DEBUG=1`.

**Example Commands:**
```bash
# Enable debug mode for single file prediction
DEBUG=1 python3 cli_batch_predictor.py --file upcoming_races/test_race.csv

# Enable debug mode for batch prediction
DEBUG=1 python3 batch_prediction_cli.py --input ./upcoming_races --output ./batch_results
```

**Expected Debug Log Excerpts:**
```
DEBUG: Processing CSV file: test_race.csv
DEBUG: Found headers: ['Dog Name', 'PLC', 'BOX', 'Trainer']
DEBUG: Dog count: 8 (expected: 8)
DEBUG: Validation passed - all required headers present
DEBUG: Feature extraction complete for 8 dogs
WARNING: Dog count deviation detected: expected 8, found 7
DEBUG: Model prediction complete - top 3 predictions generated
```

## Directory Layout and Data Types

Clear separation between historical data (form guides) and race data (upcoming races) is enforced throughout the repo.

- Historical data (form guides)
  - Purpose: past-race records used for training/analysis only
  - Source: form guide CSVs and scraped historical tables
  - Storage: unprocessed/ → processed/ → databases/
  - Rules: Never use historical outcome columns as inputs when predicting the same race (no leakage)
- Race data (upcoming races)
  - Purpose: inputs for predictions for not-yet-run races
  - Source: manual CSVs or scraper outputs for future races
  - Storage: UPCOMING_RACES_DIR (default: ./upcoming_races or ./upcoming_races_temp, see config)
  - Rules: Must not contain post-race fields; winners are scraped from the race page after the race, never from form guides

Related directories in this repo
- upcoming_races/ or upcoming_races_temp/ — upcoming race CSVs (race data)
- unprocessed/ and processed/ — ingestion lanes for historical data (form guides)
- predictions/ — generated prediction artifacts (JSON/CSV)
- archive/ — archive-first storage for retired/legacy files and outdated CSVs
- logs/ — application and data quality logs

Note on archive-first policy
- Before creating new files, search under archive/ folders for an existing version.
- When deprecating or superseding files/scripts, move them into the appropriate archive/ subfolder to keep the root clean.

## Configuration

Set these environment variables (e.g., in a .env file or via the shell) to control paths and behavior.

- DISABLE_ASSET_MINIFY
  - Description: When set to 1, skip webassets minification filters to avoid optional deps (jsmin/cssmin). Recommended for local/dev unless you need minified bundles.
  - Default: 1
  - Example: DISABLE_ASSET_MINIFY=1
- ENABLE_ENDPOINT_DROPDOWNS
  - Description: Enables a dev-only dropdown toolbar in the UI that lists all server endpoints by category. Useful for QA; keep disabled in prod.
  - Default: 0
  - Example: ENABLE_ENDPOINT_DROPDOWNS=1
  - Note (2025-08-28): The dropdowns are no longer auto-enabled in testing/debug modes; enable explicitly via the env var when needed. The /api/endpoints route and endpoints-menu.js remain available behind this flag.
  - CI: The UI E2E job in .github/workflows/backend-tests.yml is currently disabled with `if: ${{ false }}`. Remove that guard to re-enable the UI E2E job.
- DISABLE_NAV_DROPDOWNS
  - Description: Hides the main top navigation dropdowns (e.g., Races, ML, System, Help) even when UI_MODE=advanced. Useful for demos or a simplified UI while retaining advanced pages.
  - Default: 0
  - Example: DISABLE_NAV_DROPDOWNS=1
- TESTING
  - Description: Enables various test helpers and routes when true. Keep false in normal runs.
  - Default: false
  - Example: TESTING=false
- UPCOMING_RACES_DIR
  - Description: Directory the UI/API enumerates for upcoming race CSVs
  - Default: ./upcoming_races (some setups use ./upcoming_races_temp)
  - Example: UPCOMING_RACES_DIR=./upcoming_races
- DOWNLOADS_WATCH_DIR
  - Description: Optional folder a file-watcher monitors (e.g., your browser Downloads directory) to auto-move/copy new race CSVs into UPCOMING_RACES_DIR
  - Default: unset (manual mode)
  - Example: DOWNLOADS_WATCH_DIR=~/Downloads
- PREDICTIONS_DIR
  - Description: Where generated predictions are written (JSON/CSV/UI caches)
  - Default: ./predictions
  - Example: PREDICTIONS_DIR=./predictions
- ARCHIVE_ROOT
  - Description: Root folder for archived items per the archive-first policy
  - Default: ./archive
  - Example: ARCHIVE_ROOT=./archive
- LOG_PATH
  - Description: Central log file or folder for system logs
  - Default: ./logs
  - Example: LOG_PATH=./logs

Example .env

# Core
UPCOMING_RACES_DIR=./upcoming_races
PREDICTIONS_DIR=./predictions
ARCHIVE_ROOT=./archive

# UI/dev toggles
DISABLE_ASSET_MINIFY=1
ENABLE_ENDPOINT_DROPDOWNS=0
TESTING=false

# Optional automation (set only if you run a watcher)
DOWNLOADS_WATCH_DIR=~/Downloads

## Manual “Upcoming Races” Flow and Immediate Prediction Visibility

How the manual flow works
1) Prepare a CSV per race following the naming pattern: "Race {race_number} - {VENUE_CODE} - YYYY-MM-DD.csv".
2) Place the CSV into UPCOMING_RACES_DIR.
3) The UI lists files directly from this directory. No database insert is required for visibility.
4) When you click a race in the UI or call /api/predict_single_race_enhanced with the file name, the backend runs PredictionPipelineV4 (with fallbacks) and returns predictions.
5) Results can be persisted under PREDICTIONS_DIR and rendered in the UI immediately.

Why predictions appear immediately
- The UI enumerates the UPCOMING_RACES_DIR and requests predictions on demand, so newly added files are instantly discoverable without a separate ingest step.
- No post-race outcome fields are used as inputs (historical data is separate), ensuring leakage-safe predictions.

Optional automation via watcher
- If DOWNLOADS_WATCH_DIR is set and a local watcher process is running, newly downloaded CSVs can be auto-copied or moved into UPCOMING_RACES_DIR, making them appear in the UI instantly.
- If not set, the system operates in manual mode: just copy/move files yourself.

## Troubleshooting (Upcoming Races)

Permissions
- Symptom: CSVs placed in UPCOMING_RACES_DIR do not appear in the UI.
- Checks:
  - Ensure the application user can read and execute the directory and read the files (e.g., chmod 750 on the directory, chmod 640 on files).
  - On Docker: confirm the volume mount path matches UPCOMING_RACES_DIR inside the container.

Watcher not running (automation setups)
- Symptom: New files downloaded into your browser Downloads don’t appear in UPCOMING_RACES_DIR automatically.
- Checks:
  - Verify DOWNLOADS_WATCH_DIR is set.
  - Ensure your watcher service/process is running without errors (review logs under logs/).
  - Fall back to manual copy: move the CSV into UPCOMING_RACES_DIR and refresh the UI.

Path issues
- Symptom: API returns “file not found” or wrong directory.
- Checks:
  - Confirm the exact filename (including spaces and dashes) matches the UI listing and the filesystem.
  - Print your effective configuration (env) and verify UPCOMING_RACES_DIR/PREDICTIONS_DIR.
  - Normalize symlinks/relative paths if running under different working directories.

CSV structure errors
- Symptom: Backend rejects the file or returns validation errors.
- Checks:
  - Ensure a header row exists and only pre-race fields are provided (no outcomes like PLC/finish_position).
  - Validate required fields: race_date, venue_code, race_number, dog_name, box.
  - Use UTF-8 encoding and .csv extension.

Archive-first policy
- Before creating a new race CSV for the same identity, search under archive/ to avoid duplicates.
- When a race becomes outdated, move the CSV to archive/upcoming_races/YYYY/MM/ rather than deleting.

## Link: Upcoming Races User Guide (with screenshots)

For a step-by-step visual walkthrough, see docs/Upcoming_Races_User_Guide.md

## Production Hardening and Safety Defaults

This repository enforces a strict “no fabricated outputs” policy in production paths. All mock, placeholder, or random-based logic is removed or explicitly gated behind development-only environment flags. API endpoints fail fast (HTTP 503) when predictors are unavailable.

Key environment flags (default OFF):
- UNIFIED_ALLOW_BASIC_FALLBACK — allow dev-only basic fallback in unified_predictor.py
- ML_V4_ALLOW_HEURISTIC — allow dev-only single-dog heuristic in ml_system_v4.py
- ML_V4_ALLOW_SIMULATED_ODDS — allow dev-only simulated odds for EV learning in ml_system_v4.py
- TGR_ALLOW_PLACEHOLDER — allow dev-only race insights placeholder in TGR scraper
- ALLOW_SYNTHETIC_TEST_MODEL — allow synthetic-data test trainer script

See docs/hardening.md for full details.

## Repository Structure

-   `app.py`: Main Flask application.
-   `run.py`: Entry point for command-line tasks (collect, analyze, predict).
-   `database/`: Database files and schemas.
-   `static/`: Frontend assets (JavaScript, CSS).
-   `templates/`: HTML templates for the web interface.
-   `tests/`: Unit and integration tests.
-   `archive/`: Older or superseded scripts.

## Superseded Files

The following files have been moved to the `/archive` directory as they have been superseded by newer implementations:

### Legacy Prediction and Debug Scripts
-   `run_pipeline_debug.py`
-   `integrity_test.py`
-   `flask_api_endpoint_test.py`
-   `quick_flask_test.py`
-   `retry_weather_service_test.py`
-   `debug_model_features.py`
-   `debug_scaler.py`
-   `debug_race_extraction.py`
-   `train_test_data.py`

### Obsolete Upcoming Race Scripts (Archive: `/archive/obsolete_upcoming_race_scripts/`)
-   `upcoming_race_predictor.py` - Superseded by enhanced prediction endpoints
-   `upcoming_race_predictor_clean.py` - Superseded by PredictionPipelineV3
-   `upcoming_race_predictor_test.py` - Superseded by modern pytest test suite
-   `integrated_race_collector.py` - Superseded by ComprehensiveFormDataCollector
-   `enhanced_odds_collector.py` - Superseded by hybrid_odds_scraper.py

**Note**: These scripts have been replaced by the enhanced API endpoints (`/api/predict_single_race_enhanced`, `/api/predict_all_upcoming_races_enhanced`) which provide intelligent pipeline selection, comprehensive error handling, and better integration with the main application.

---

For details on the production hardening policy and safety defaults, see docs/hardening.md.
