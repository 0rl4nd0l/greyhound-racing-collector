# API Architecture

The Greyhound Racing Predictor system includes a comprehensive set of API endpoints that facilitate various functionalities, such as predictions, data enhancement, and real-time updates.

## Key Endpoints

- **/api/dogs/search**: Allows searching for dogs by name with optional pagination and filtering.
- **/api/predict_single_race**: Provides predictions for a single race file, compatible with frontend interfaces.
- **/api/predict_stream**: SSE endpoint for real-time streaming of race predictions.
- **/api/feature_analysis**: Analyzes and retrieves detailed information about engineered features.
- **/api/database/integrity_check**: Runs integrity checks on the database schema and data quality.
- **/api/check_performance_drift**: Monitors model and feature performance drift, returning JSON reports.

## Architecture

- **RESTful Design**: API architecture follows standard REST principles, allowing for stateless interactions and clear resource representations.
- **CORS Management**: The app includes CORS support to enable cross-origin requests from web clients.
- **Flask Framework**: Built using Flask, with distinct routes handling different resources and actions.
- **Error Handling**: Consistent error management and logging to ensure clarity in failure scenarios.

## Real-time Integration

The integration of SSE for real-time prediction updates allows clients to maintain an open connection for instant update streaming without requiring HTTP polling.

## Security and Authentication

- **Secure File Handling**: Ensures files are securely uploaded and processed.
- **Environment Configurations**: All sensitive data such as API keys and database paths are managed through environment variables.
- **Rate Limiting and Throttling**: Protects the system from overuse by implementing request limits.

This architecture is designed to balance flexibility, real-time capability, and security, supporting the complex operations of greyhound race data analysis and prediction.
