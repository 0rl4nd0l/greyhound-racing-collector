# Alerting System

The alerting system provides automated notifications for critical system events, performance degradation, and operational issues.

## Alert Categories

### System Health Alerts
- **High CPU Usage**: Triggered when CPU usage exceeds 90% for 5+ minutes
- **Memory Exhaustion**: Alert when available memory drops below 15%
- **Disk Space Critical**: Warning when disk usage exceeds 90%
- **Process Failures**: Immediate alerts for critical process crashes

### Model Performance Alerts
- **Accuracy Degradation**: Alert when model accuracy drops more than 2% from baseline
- **Prediction Latency**: Warning when prediction time exceeds 1 second average
- **Model Drift**: Notification when feature or performance drift is detected
- **Training Failures**: Immediate alerts for failed model training runs

### Data Quality Alerts
- **Data Ingestion Failures**: Alerts for failed data scraping or ingestion
- **Schema Violations**: Notifications for database schema inconsistencies
- **Missing Data**: Warnings for incomplete or missing critical data
- **Data Freshness**: Alerts when data becomes stale beyond acceptable thresholds

### API and Service Alerts
- **High Error Rates**: Alert when API error rate exceeds 5%
- **Service Unavailability**: Immediate notifications for service downtime
- **Rate Limiting**: Warnings when rate limits are approached or exceeded
- **Authentication Failures**: Security alerts for repeated authentication failures

## Alert Channels

### Logging Integration
All alerts are integrated with the comprehensive logging system:
```python
from logger import logger

# Critical system alerts
logger.log_system("High CPU usage detected", "ERROR", "SYSTEM_HEALTH", 
                 context={'cpu_percent': 95.2, 'duration': '6 minutes'})

# Model performance alerts
logger.log_model("Model accuracy below threshold", "WARNING", "PERFORMANCE",
                context={'current_accuracy': 0.82, 'threshold': 0.85})
```

### Real-time Dashboard
- Visual alert indicators on monitoring dashboard
- Color-coded severity levels (Info, Warning, Error, Critical)
- Historical alert timeline and resolution tracking
- Alert acknowledgment and resolution workflow

### API Endpoints
- `/api/alerts/active`: Get current active alerts
- `/api/alerts/history`: Retrieve alert history with filtering
- `/api/alerts/acknowledge`: Mark alerts as acknowledged
- `/api/alerts/resolve`: Mark alerts as resolved

## Alert Severity Levels

### Critical (Red)
- System outages or failures
- Security breaches or authentication issues
- Data corruption or loss
- Model performance below acceptable thresholds

### Warning (Yellow)
- Performance degradation within acceptable limits
- Resource usage approaching thresholds
- Non-critical service disruptions
- Data quality issues that don't affect core functionality

### Info (Blue)
- Routine operational notifications
- Scheduled maintenance notifications
- Performance milestone achievements
- System optimization recommendations

## Alert Configuration

### Thresholds
Configurable alert thresholds for different metrics:
```python
ALERT_THRESHOLDS = {
    'cpu_usage': {'warning': 70, 'critical': 90},
    'memory_usage': {'warning': 70, 'critical': 85},
    'disk_usage': {'warning': 80, 'critical': 90},
    'prediction_latency': {'warning': 0.5, 'critical': 1.0},
    'model_accuracy': {'warning': 0.02, 'critical': 0.05}  # Degradation thresholds
}
```

### Suppression Rules
- **Duplicate Suppression**: Prevent repeated alerts for the same issue
- **Maintenance Windows**: Suppress alerts during scheduled maintenance
- **Dependency Suppression**: Suppress downstream alerts when upstream issues are detected
- **Time-based Suppression**: Reduce alert frequency during known problematic periods

## Integration with Monitoring

The alerting system is tightly integrated with the monitoring infrastructure:

### Prometheus Alertmanager
- Integration with Prometheus for metric-based alerting
- Custom alert rules for specific business logic
- Alert routing based on severity and type
- Automated alert grouping and deduplication

### Health Check Integration
```python
def check_system_health():
    health_status = monitoring_api.get_system_health()
    
    if health_status['resources']['cpu_usage'] > ALERT_THRESHOLDS['cpu_usage']['critical']:
        trigger_alert('HIGH_CPU_USAGE', severity='critical', 
                     context=health_status['resources'])
```

## Alert Response Workflows

### Automated Responses
- **Auto-scaling**: Automatic resource allocation for performance issues
- **Circuit Breaker**: Automatic service isolation for failing components  
- **Fallback Activation**: Automatic fallback to backup systems
- **Cache Warming**: Preemptive cache warming for performance issues

### Manual Response Procedures
- Detailed runbooks for common alert scenarios
- Escalation procedures for unresolved critical alerts
- Step-by-step diagnostic and resolution guides
- Post-incident review and documentation requirements

The alerting system ensures rapid detection and response to system issues, maintaining high availability and performance standards for the greyhound racing prediction system.
