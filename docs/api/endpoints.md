# API Documentation

This document provides a detailed reference for the Greyhound Racing Predictor's API endpoints.

## Overview

The API allows interaction with the prediction system, providing endpoints for predictions, data analysis, and system management.

## Core Prediction Endpoints

### Upcoming Races CSVs (folder, naming, and consumption)
- Folder path: `./upcoming_races`
- Files: `.csv` only
- Required naming pattern: `Race {number} - {VENUE} - {YYYY-MM-DD}.csv`
- The Upcoming UI automatically lists CSVs from this folder
- Predictions pipeline consumes these CSVs directly (V4 primary, V3 fallback)
- Utilities: `scripts/normalize_upcoming_to_api_pattern.py`, `scripts/alias_upcoming_api_names_safe.py`

### Single Race Prediction (Enhanced)
- **Endpoint**: `/api/predict_single_race_enhanced`
- **Method**: POST
- **Description**: Primary endpoint for single race prediction with intelligent pipeline selection and automatic file discovery
- **Request Body**:
  ```json
  {
    "race_filename": "Race 4 - GOSF - 2025-07-28.csv"
    // OR
    "race_id": "race_identifier"
  }
  ```
- **Key Features**:
  - Accepts either `race_filename` or `race_id` parameter
  - Automatically searches multiple directories (upcoming, historical)
  - Intelligent pipeline selection: PredictionPipelineV3 → UnifiedPredictor → ComprehensivePredictionPipeline
  - Enhanced error handling with detailed failure information

- **Response**:
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

- **Error Response**:
  ```json
  {
    "success": false,
    "message": "Race file 'invalid.csv' not found in upcoming or historical directories",
    "error_type": "file_not_found",
    "race_filename": "invalid.csv",
    "searched_directories": ["./upcoming_races", "./historical_races"]
  }
  ```

### All Upcoming Races Prediction (Enhanced)
- **Endpoint**: `/api/predict_all_upcoming_races_enhanced`
- **Method**: POST
- **Description**: Batch endpoint for predicting all upcoming races with comprehensive error handling and performance monitoring
- **Request Body** (Optional JSON):
  ```json
  {
    "max_races": 10,
    "force_rerun": false
  }
  ```
- **Key Features**:
  - Automatically discovers all CSV files in upcoming races directory
  - Intelligent pipeline selection: ComprehensivePredictionPipeline (primary) → PredictionPipelineV3 (fallback)
  - Comprehensive error tracking and recovery
  - Performance monitoring and timing metrics
  - Detailed success/failure reporting

- **Response**:
  ```json
  {
    "success": true,
    "total_races": 5,
    "success_count": 4,
    "predictions": [
      {
        "race_filename": "Race 1 - GOSF - 2025-01-15.csv",
        "success": true,
        "predictions": [
          {
            "dog_name": "Lightning Bolt",
            "box_number": 1,
            "win_probability": 0.35,
            "predicted_position": 1
          }
        ],
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

- **Empty Directory Response**:
  ```json
  {
    "success": true,
    "message": "No upcoming races found",
    "total_races": 0,
    "success_count": 0,
    "predictions": [],
    "errors": []
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

