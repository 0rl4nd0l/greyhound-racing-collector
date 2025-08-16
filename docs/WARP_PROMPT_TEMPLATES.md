# Prompt Templates Library

## Overview
This document contains reusable prompt templates for common tasks in the Greyhound Racing system.

## Templates

### Add Endpoint
Create a new API endpoint with validation.

- **Template**
  ```yaml
  add_endpoint:
    description: "Add a new endpoint to the API."
    fields:
      - method: POST
      - path: /api/new_endpoint
      - validation: ["field1", "field2"]
    acceptance_criteria:
      - [ ] Endpoint returns 200 on success
      - [ ] Responds with 400 for invalid input
      - [ ] Includes correct error messages in response
  ```

### Feature Engineering
Outline new feature engineering process.

- **Template**
  ```yaml
  feature_engineering:
    description: "Define feature engineering steps."
    steps:
      - "Identify target column"
      - "Select input features"
      - "Apply transformations"
      - "Validate against benchmark"
    acceptance_criteria:
      - [ ] Features improve model accuracy
      - [ ] No leakage from target
      - [ ] Documented transformations
  ```

### Calibration
Define calibration process checklist.

- **Template**
  ```yaml
  calibration:
    description: "Setup model calibration."
    steps:
      - "Run reliability analysis"
      - "Compute Brier score"
      - "Refit with best parameters"
      - "Validate using test set"
    acceptance_criteria:
      - [ ] Calibration slope between 0.9 and 1.1
      - [ ] Brier score under 0.1
  ```

### Backtesting
Plan backtesting procedure for predictive models.

- **Template**
  ```yaml
  backtesting:
    description: "Backtest model predictions."
    steps:
      - "Select historical data period"
      - "Run backtest simulation"
      - "Aggregate statistics"
      - "Review against expected outcomes"
    acceptance_criteria:
      - [ ] Predicted outcomes fall within expected range
      - [ ] Statistically significant improvement over baseline
  ```

### Container/Test Pipeline
Detail containerization and testing setup.

- **Template**
  ```yaml
  container_test_pipeline:
    description: "Setup containers and testing pipelines."
    steps:
      - "Create Dockerfile"
      - "Setup CI/CD pipeline"
      - "Integrate with testing suite"
    acceptance_criteria:
      - [ ] Automated testing on all merges
      - [ ] Containers pass smoke tests
  ```

### Archiving
Define file archival process.

- **Template**
  ```yaml
  archiving:
    description: "Describe archiving protocol."
    steps:
      - "Identify obsolete files"
      - "Compress and store securely"
      - "Update archive logs"
    acceptance_criteria:
      - [ ] All identified files archived
      - [ ] Archive retrievable without data loss
  ```
