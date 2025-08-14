# Performance Monitoring

The performance monitoring system provides comprehensive insights into system health, prediction accuracy, and operational metrics.

## Key Metrics

### System Health Metrics
- **CPU Usage**: Real-time CPU utilization monitoring using psutil
- **Memory Usage**: RAM consumption and available memory tracking
- **Disk Usage**: Storage utilization and available space monitoring
- **Process Health**: Monitoring of critical system processes

### Prediction Performance Metrics
- **Prediction Accuracy**: Real-time tracking of prediction success rates
- **Response Times**: API endpoint latency monitoring
- **Throughput**: Requests per second and concurrent prediction handling
- **Error Rates**: Failed prediction attempts and error categorization

### Model Performance Metrics
- **ROC AUC**: Area Under the Curve for model performance assessment
- **Precision/Recall**: Classification accuracy metrics
- **Feature Importance**: Tracking of most influential features
- **Drift Detection**: Automated monitoring for model performance degradation

### Database Performance
- **Query Performance**: SQL query execution time tracking
- **Connection Pool**: Database connection utilization
- **Index Efficiency**: Database index usage and optimization
- **Data Quality**: Automated data integrity checks

## Monitoring API Endpoints

### `/api/system_health`
Returns comprehensive system health status including:
- Resource utilization (CPU, memory, disk)
- Component health status
- Overall system status assessment

### `/api/performance_metrics`
Provides real-time performance metrics with caching:
- Live calculation of key performance indicators
- Cached results for improved response times
- Timestamp information for metric freshness

### `/api/recent_predictions`
Lists recent predictions with their status:
- Prediction history with accuracy tracking
- Success/failure status for each prediction
- Performance trend analysis

## Prometheus Integration

The system integrates with Prometheus for advanced monitoring:

```python
from prometheus_client import Counter, Histogram, Gauge

# Key metrics tracked
prediction_counter = Counter('predictions_total', 'Total predictions made')
prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
active_connections = Gauge('active_connections', 'Active database connections')
```

### Custom Metrics
- `scrape_duration_seconds`: Time taken for data scraping operations
- `model_prediction_latency_seconds`: ML model prediction latency
- `queue_length`: Background processing queue length
- `active_models`: Number of active prediction models

## Performance Thresholds

### Critical Thresholds
- **CPU Usage**: Alert if > 90% for 5+ minutes
- **Memory Usage**: Alert if > 85% for 3+ minutes
- **Disk Usage**: Alert if > 90% used space
- **Prediction Latency**: Alert if > 1 second average

### Warning Thresholds
- **CPU Usage**: Warning if > 70% for 10+ minutes
- **Memory Usage**: Warning if > 70% for 5+ minutes
- **Error Rate**: Warning if > 5% of requests fail
- **Model Accuracy**: Warning if accuracy drops > 2%

## Dashboard Integration

Performance metrics are integrated into the monitoring dashboard:
- Real-time metric visualization
- Historical trend analysis
- Alert status indicators
- System component health overview

The performance monitoring system ensures optimal system operation through continuous tracking, alerting, and automated health assessments.
