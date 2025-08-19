# Prometheus Metrics

This document outlines the metrics exposed by the Greyhound Racing Predictor through Prometheus.

## Overview

Prometheus is integrated to collect valuable metrics that help monitor the application and its performance. The `/metrics` endpoint provides insights into various system metrics.

## Key Metrics

### Scrape Time Per Job
- **Metric Name**: `scrape_duration_seconds`
- **Description**: Measures the time taken to scrape each job.
- **Type**: Histogram
- **Interactions**: Helps identify slow scrape targets.

### Model Latency
- **Metric Name**: `model_prediction_latency_seconds`
- **Description**: Measures the latency of model predictions.
- **Type**: Histogram
- **Usage**: Tracks real-time delay in model responses, allowing for performance tuning.

### Queue Length
- **Metric Name**: `queue_length`
- **Description**: Represents the length of the job queue.
- **Type**: Gauge
- **Usage**: Indicates the number of jobs waiting for processing, useful for capacity planning.

## Alerts

You can set alerts based on these metrics to notify you when conditions are outside expected thresholds:

- **High Latency**: Alert when `model_prediction_latency_seconds` exceeds a set value for an extended period.
- **Long Queues**: Alert when the `queue_length` consistently stays above a certain threshold.

## Configuration

Ensure that Prometheus is configured to scrape metrics from the `/metrics` endpoint of the application. Update your `prometheus.yml` file with the following job configuration:

```yaml
scrape_configs:
  - job_name: 'greyhound_predictor'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:5000']  # Update to your server's address if necessary
```

## Performance Monitoring

Use the metrics to establish baseline performance indicators and gradually refine them as the system matures. Leverage Grafana or other visualization tools to create dashboards and track these metrics over time.
