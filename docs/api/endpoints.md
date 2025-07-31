# API Documentation

This document provides a detailed reference for the Greyhound Racing Predictor's API endpoints.

## Overview

The API allows interaction with the prediction system, providing endpoints for predictions, data analysis, and system management.

## Core Prediction Endpoints

### Single Race Prediction (Enhanced)
- **Endpoint**: `/api/predict_single_race_enhanced`
- **Method**: POST
- **Description**: Get enhanced predictions for a single race with advanced analysis
- **Request Body**:
  ```json
  {
    "race_filename": "Race 4 - GOSF - 2025-07-28.csv"
  }
  ```
- **Response**:
  ```json
  {
    "race_summary": {
      "race_name": "Race 4",
      "venue": "GOSF",
      "date": "2025-07-28",
      "distance": "500m",
      "field_size": 8
    },
    "top_picks": [
      {
        "dog_name": "Lightning Bolt",
        "trap_number": 1,
        "win_probability": 0.35,
        "confidence": "High"
      }
    ],
    "predictions": [
      {
        "dog_name": "Lightning Bolt",
        "trap_number": 1,
        "win_probability": 0.35,
        "place_probability": 0.65,
        "predicted_rank": 1,
        "confidence_score": 0.87,
        "reasoning": "Strong recent form and favorable conditions"
      }
    ]
  }
  ```

### All Upcoming Races Prediction (Enhanced)
- **Endpoint**: `/api/predict_all_upcoming_races_enhanced`
- **Method**: POST
- **Description**: Get enhanced predictions for all upcoming races with comprehensive analysis
- **Request Body** (Optional):
  ```json
  {
    "max_races": 5,
    "skip_sanity_checks": false,
    "enable_drift_monitoring": true
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "total_races": 3,
    "successful_predictions": 2,
    "failed_predictions": 1,
    "processing_time_seconds": 45.2,
    "drift_warnings": [],
    "sanity_check_results": {
      "total_checks": 15,
      "passed": 14,
      "failed": 1,
      "warnings": ["Feature correlation drift detected"]
    },
    "races": [
      {
        "race_filename": "Race 1 - GOSF - 2025-01-15.csv",
        "status": "success",
        "prediction": {
          "race_summary": {},
          "top_picks": [],
          "predictions": []
        },
        "confidence_score": 0.87,
        "drift_score": 0.12,
        "processing_time_ms": 2340
      }
    ]
  }
  ```

### Monitor Metrics
- **Endpoint**: `/metrics`
- **Method**: GET
- **Description**: Provides Prometheus metrics for system monitoring.
- **Response**: Exposes metrics in a format suitable for Prometheus scraping.

### Submit Feedback
- **Endpoint**: `/feedback`
- **Method**: POST
- **Description**: Submit feedback or error reports related to predictions.
- **Request Body**:
  ```json
  {
    "race_id": "integer",
    "feedback": "string"
  }
  ```
- **Response**: Acknowledgment of received feedback.

## Authentication

- Some endpoints may require authentication or API tokens. Ensure to check headers and status codes.

## Error Handling

- Standard HTTP status codes are used to indicate success or failure.
- Common error codes include 400 (Bad Request), 404 (Not Found), 500 (Internal Server Error).

## Rate Limiting

- To ensure service quality, rate limiting policies may apply to certain endpoints.

## Versioning

- The API follows semantic versioning principles. Ensure to check the API version before integration.

