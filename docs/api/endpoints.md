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
    ],
    "prediction_tier": "comprehensive_pipeline",
    "fallback_reasons": []
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

## Prediction Fallback Mechanism

The prediction system uses an intelligent fallback hierarchy to ensure robust predictions even when higher-tier systems fail. The response includes metadata about which prediction tier was used and any fallback reasons.

### Prediction Tiers (in order of preference)

1. **comprehensive_pipeline** - Full comprehensive analysis with all enhancements
2. **weather_enhanced** - Weather-enhanced predictor with meteorological data
3. **unified_predictor** - Unified prediction system combining multiple approaches
4. **ml_system_v3_basic** - Basic ML system as final fallback

### Response Fields Related to Fallbacks

- **prediction_tier**: String indicating which prediction tier was successfully used
- **fallback_reasons**: Array of objects describing why higher tiers failed (if any)

### Fallback Reason Structure

```json
{
  "tier": "comprehensive_pipeline",
  "reason": "Comprehensive pipeline exception: Model file not found",
  "timestamp": "2025-01-15T10:30:45.123456"
}
```

### Example Response with Fallbacks

```json
{
  "success": true,
  "prediction_tier": "unified_predictor",
  "fallback_reasons": [
    {
      "tier": "comprehensive_pipeline",
      "reason": "Comprehensive pipeline exception: Database connection failed",
      "timestamp": "2025-01-15T10:30:45.123456"
    },
    {
      "tier": "weather_enhanced",
      "reason": "Weather-enhanced predictor returned unsuccessful result: Weather API timeout",
      "timestamp": "2025-01-15T10:30:46.789012"
    }
  ],
  "predictions": [...],
  "note": "Successfully predicted using unified predictor after higher tiers failed"
}
```

### Error Response with Fallbacks

If all prediction tiers fail, the error response will include all fallback reasons:

```json
{
  "success": false,
  "error": "All prediction methods failed. Final error: No valid predictions generated",
  "predictions": [],
  "prediction_method": "ml_system_v3",
  "fallback_reasons": [
    {
      "tier": "comprehensive_pipeline",
      "reason": "Comprehensive pipeline exception: Model file corrupted",
      "timestamp": "2025-01-15T10:30:45.123456"
    },
    {
      "tier": "weather_enhanced",
      "reason": "Weather-enhanced predictor exception: API key expired",
      "timestamp": "2025-01-15T10:30:46.789012"
    },
    {
      "tier": "unified_predictor",
      "reason": "Unified predictor returned unsuccessful result: Invalid race data format",
      "timestamp": "2025-01-15T10:30:47.345678"
    },
    {
      "tier": "ml_system_v3_basic",
      "reason": "All prediction methods failed. Final error: No valid predictions generated",
      "timestamp": "2025-01-15T10:30:48.901234"
    }
  ]
}
```

