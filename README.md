# Greyhound Racing Predictor

## Overview

The Greyhound Racing Predictor is a comprehensive, AI-powered system for analyzing and predicting greyhound racing outcomes. It combines data from various sources, including FastTrack and other sportsbooks, to provide detailed race analysis and predictions using advanced machine learning techniques.

### Key Features

- **Unified Database**: Consolidates data from multiple sources into a single, efficient SQLite database (`greyhound_racing_data.db`)
### Advanced ML Pipeline: Utilizes Optuna for Bayesian optimization with stratified TimeSeriesSplit, feature engineering, and intelligent model selection. Incorporates class imbalance handling with SMOTE-NC and focal loss, and outputs to MLflow.
- **Unified Prediction System**: Intelligent prediction engine that automatically selects the best available prediction method
- **Flask Web Interface**: A web-based dashboard for monitoring races, viewing predictions, and managing the system
- **Comprehensive API**: RESTful API endpoints for predictions, dog statistics, race data, and system management
- **Automated Data Processing**: Scripts for automated collection, analysis, and processing of racing data
- **Frontend Integration**: Includes a JavaScript frontend for enhanced user interaction
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
The application will be available at `http://127.0.0.1:5000`.

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

## API Documentation

The application provides a comprehensive RESTful API for interacting with the prediction system and its data. Below are the key endpoints.

### Main Prediction Endpoint

-   **POST /api/predict_single_race_enhanced**

    This is the primary endpoint for getting predictions for a single race. It automatically fetches the latest data, runs the most advanced prediction pipeline available, and returns detailed results.

    **Request Body**:

    ```json
    {
        "race_filename": "Race 4 - GOSF - 2025-07-28.csv"
    }
    ```

    **Response**:

    -   A detailed JSON object containing the race summary, top picks, and a full list of predictions for each dog with confidence scores and reasoning.

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
    curl -X POST http://127.0.0.1:5000/api/gpt/enhance_race \
      -H "Content-Type: application/json" \
      -d '{"race_file_path": "./upcoming_races/Race 1 - GOSF - 2025-07-31.csv"}'
    ```

-   **GET /api/gpt/daily_insights**: Get GPT daily insights for a specific date
    ```bash
    curl "http://127.0.0.1:5000/api/gpt/daily_insights?date=2025-07-31"
    ```

-   **POST /api/gpt/enhance_multiple**: Enhance multiple races with GPT analysis
    ```bash
    curl -X POST http://127.0.0.1:5000/api/gpt/enhance_multiple \
      -H "Content-Type: application/json" \
      -d '{"race_files": ["race1.csv", "race2.csv"], "max_races": 3}'
    ```

-   **POST /api/gpt/comprehensive_report**: Generate comprehensive GPT report
    ```bash
    curl -X POST http://127.0.0.1:5000/api/gpt/comprehensive_report \
      -H "Content-Type: application/json" \
      -d '{"race_ids": ["race123", "race456"]}'
    ```

-   **GET /api/gpt/status**: Check GPT integration status
    ```bash
    curl "http://127.0.0.1:5000/api/gpt/status"
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

-   `run_pipeline_debug.py`
-   `integrity_test.py`
-   `flask_api_endpoint_test.py`
-   `quick_flask_test.py`
-   `retry_weather_service_test.py`
-   `debug_model_features.py`
-   `debug_scaler.py`
-   `debug_race_extraction.py`
-   `train_test_data.py`
