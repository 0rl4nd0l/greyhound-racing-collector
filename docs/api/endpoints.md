# API Documentation

This document provides a detailed reference for the Greyhound Racing Predictor's API endpoints.

## Overview

The API allows interaction with the prediction system. It provides endpoints for submitting prediction requests, retrieving results, and accessing relevant metadata.

## Endpoints

### Predict Race Outcome
- **Endpoint**: `/predict`
- **Method**: POST
- **Description**: Predict the outcome of a race given the required input data.
- **Request Body**:
  ```json
  {
    "race_id": "integer",
    "predictors": {
        "track_condition": "string",
        "weather": "object",
        // additional predictors...
    }
  }
  ```
- **Response**:
  ```json
  {
    "predictions": [
       {
         "greyhound_id": "integer",
         "probability": "float"
       },
       // more predictions...
    ]
  }
  ```

### Get Race Details
- **Endpoint**: `/race/{race_id}`
- **Method**: GET
- **Description**: Retrieve detailed information about a specific race.
- **Response**:
  ```json
  {
    "race": {
      "id": "integer",
      "track": "string",
      "weather": "object",
      // additional details...
    }
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

