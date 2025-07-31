# Greyhound Predictor Changelog

## [v3.0.1] - 2025-07-31

### Fixed
-   **Form Guide CSV Scraper**: Fixed regex patterns to correctly recognize race dates and filenames, resolving "Unknown" entries in data processing.

## [v3.0.0] - 2025-07-26

This major update focuses on a comprehensive refactoring of the entire system, from the database to the prediction pipeline and the Flask API. The primary goals were to unify scattered data sources, improve prediction accuracy, enhance system stability, and provide a more robust and developer-friendly platform.

### Added

-   **Unified Database Schema**:
    -   Introduced a single SQLite database (`greyhound_racing_data.db`) to consolidate all historical, race, and dog data.
    -   Created `create_unified_database.py` to build and populate the new schema from legacy data sources.

-   **Unified Prediction System**:
    -   Developed `unified_predictor.py`, a new core prediction engine that intelligently selects the best available prediction method based on available system components.
    -   Implemented `prediction_pipeline_v3.py`, a state-of-the-art machine learning pipeline with advanced feature engineering, data validation, and model management.

-   **Enhanced Flask API (`app.py`)**:
    -   **New Prediction Endpoint**: Added `/api/predict_single_race_enhanced`, which automatically enriches input data and runs the most advanced prediction pipeline available.
    -   **Detailed Data Endpoints**:
        -   `/api/dogs/search`: Search for greyhounds.
        -   `/api/dogs/<dog_name>/details`: Get comprehensive statistics and performance history for a specific dog.
        -   `/api/races/paginated`: A powerful endpoint for browsing historical races with search, sorting, and pagination.
    -   **System Management**: Added endpoints for monitoring logs, managing data processing workflows, and viewing model performance.

-   **Configuration & Stability**:
    -   Introduced `UnifiedPredictorConfig` to centralize all paths, feature names, and system settings.
    -   Implemented caching for prediction results to improve performance.
    -   Added robust error handling and fallback mechanisms across the entire stack.

### Changed

-   **Project Structure**:
    -   Reorganized the project by moving dozens of outdated, redundant, and test-specific scripts into the `archive/` directory to clean up the root folder.
    -   Standardized file naming and module structures for better clarity.

-   **Data Processing**:
    -   The `run.py` script and background processing tasks in `app.py` were updated to work with the new unified database and prediction system.

### Future Improvements

-   **Database Migrations**:
    -   It is highly recommended to integrate **Alembic** to manage future database schema changes. This provides a version-controlled, repeatable, and safe way to evolve the database without manual SQL scripts.

-   **Automated Testing & CI/CD**:
    -   To ensure long-term stability and code quality, setting up **GitHub Actions** for continuous integration is recommended. An automated workflow should be configured to:
        1.  Install Python dependencies.
        2.  Run the `pytest` test suite on every push and pull request.
        3.  (Optional) Deploy the application to a staging environment.

-   **Frontend Enhancements**:
    -   The frontend application can be significantly enhanced by integrating the new detailed API endpoints to provide richer visualizations and deeper insights into dog and race data.

