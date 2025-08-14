# ML System V4 Documentation

## Architecture
- PredictionPipelineV4: Handles data processing and model inference
- MLSystemV4: Orchestrates the pipeline and integrates with backend

## Data Flow
1. Historical data (form guides) loaded from CSVs
2. Race data scraped from webpage (weather, track, etc.)
3. Data preprocessed and features engineered
4. Model generates predictions
5. Results formatted for frontend consumption

## API Endpoints
- POST /api/v4/predict - Generate predictions for a race
- GET /api/v4/health - System health check

## Testing
Run tests: `python test_scripts/test_e2e_ml_v4.py`
