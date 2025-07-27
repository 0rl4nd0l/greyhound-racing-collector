// Model Monitoring Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeMonitoring();
    
    // Setup event listeners
    document.getElementById('refresh-events-btn').addEventListener('click', refreshMonitoring);
    
    // Auto-refresh every 2 minutes
    setInterval(refreshMonitoring, 120000);
});

let performanceTrendChart = null;
let driftTrendChart = null;
let featureDistributionChart = null;
let predictionDistributionChart = null;

function initializeMonitoring() {
    // Initialize charts
    initializeCharts();
    // Load initial data
    refreshMonitoring();
}

function refreshMonitoring() {
    // Get best model performance data
    fetch('/api/model/performance')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updatePerformanceMetrics(data);
                updatePerformanceTrendChart(data.performance_metrics);
                updateMonitoringEvents(data.monitoring_events);
            }
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
    
    // Trigger drift detection
    fetch('/api/model/monitoring/drift', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateDriftMetrics(data);
            updateDriftTrendChart(data.drift_results);
        }
    })
    .catch(error => {
        console.error('Error fetching drift data:', error);
    });
}

function updatePerformanceMetrics(data) {
    const container = document.getElementById('performance-metrics');
    const metrics = data.performance_metrics;
    
    let html = `
        <div class="row g-2 mb-3">
            <div class="col-6">
                <div class="border rounded p-2 text-center">
                    <small class="text-muted d-block">Accuracy</small>
                    <strong class="text-success">${formatPercentage(metrics.accuracy)}</strong>
                </div>
            </div>
            <div class="col-6">
                <div class="border rounded p-2 text-center">
                    <small class="text-muted d-block">Precision</small>
                    <strong class="text-info">${formatPercentage(metrics.precision)}</strong>
                </div>
            </div>
            <div class="col-6">
                <div class="border rounded p-2 text-center">
                    <small class="text-muted d-block">Recall</small>
                    <strong class="text-warning">${formatPercentage(metrics.recall)}</strong>
                </div>
            </div>
            <div class="col-6">
                <div class="border rounded p-2 text-center">
                    <small class="text-muted d-block">F1 Score</small>
                    <strong class="text-primary">${formatPercentage(metrics.f1_score)}</strong>
                </div>
            </div>
        </div>
        <div class="text-center">
            <small class="text-muted">Last Updated: ${formatDate(data.timestamp)}</small>
        </div>
    `;
    
    container.innerHTML = html;
}

function updateDriftMetrics(data) {
    const container = document.getElementById('drift-metrics');
    const drift = data.drift_results;
    
    const driftStatus = drift.drift_detected ? 'Detected' : 'None';
    const statusClass = drift.drift_detected ? 'danger' : 'success';
    const driftScore = drift.drift_score || 0;
    
    let html = `
        <div class="mb-3">
            <div class="d-flex justify-content-between align-items-center">
                <span>Drift Status:</span>
                <span class="badge bg-${statusClass}">${driftStatus}</span>
            </div>
        </div>
        <div class="mb-3">
            <div class="d-flex justify-content-between align-items-center">
                <span>Drift Score:</span>
                <strong>${driftScore.toFixed(3)}</strong>
            </div>
            <div class="progress mt-2">
                <div class="progress-bar bg-${driftScore > 0.5 ? 'danger' : 'success'}" 
                     role="progressbar" 
                     style="width: ${Math.min(driftScore * 100, 100)}%">
                    ${formatPercentage(driftScore)}
                </div>
            </div>
        </div>
        <div class="text-center">
            <button class="btn btn-sm btn-outline-primary" onclick="showDriftAnalysis('${data.model_id}')">
                View Details
            </button>
        </div>
    `;
    
    container.innerHTML = html;
}

function updateMonitoringEvents(events) {
    const tbody = document.getElementById('monitoring-events-body');
    tbody.innerHTML = '';
    
    events.forEach(event => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${formatDate(event.timestamp)}</td>
            <td><code>${event.model_id}</code></td>
            <td>
                <span class="badge bg-info">${formatEventType(event.event_type)}</span>
            </td>
            <td>
                ${formatEventMetrics(event)}
            </td>
            <td>
                <span class="badge bg-${getEventStatusColor(event)}">
                    ${getEventStatus(event)}
                </span>
            </td>
            <td>
                <button class="btn btn-sm btn-outline-primary" onclick="showEventDetails('${event.id}')">
                    Details
                </button>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function initializeCharts() {
    // Performance Trend Chart
    const perfCtx = document.getElementById('performance-trend-chart');
    performanceTrendChart = new Chart(perfCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Accuracy',
                data: [],
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.1
            }, {
                label: 'Precision',
                data: [],
                borderColor: 'rgba(54, 162, 235, 1)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Drift Trend Chart
    const driftCtx = document.getElementById('drift-trend-chart');
    driftTrendChart = new Chart(driftCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Drift Score',
                data: [],
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Feature Distribution Chart
    const featureCtx = document.getElementById('feature-distribution-chart');
    featureDistributionChart = new Chart(featureCtx, {
        type: 'radar',
        data: {
            labels: ['Speed', 'Form', 'Weight', 'Odds', 'Distance', 'Grade'],
            datasets: [{
                label: 'Current',
                data: [0.8, 0.7, 0.6, 0.9, 0.5, 0.7],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }, {
                label: 'Training',
                data: [0.75, 0.8, 0.65, 0.85, 0.55, 0.75],
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Prediction Distribution Chart
    const predCtx = document.getElementById('prediction-distribution-chart');
    predictionDistributionChart = new Chart(predCtx, {
        type: 'histogram',
        data: {
            labels: ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
            datasets: [{
                label: 'Win Probability Distribution',
                data: [5, 12, 18, 25, 20, 15, 8, 4, 2, 1],
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function updatePerformanceTrendChart(performanceData) {
    // Update chart with historical performance data
    if (performanceData && performanceData.history) {
        const labels = performanceData.history.map(h => formatDate(h.date));
        const accuracy = performanceData.history.map(h => h.accuracy);
        const precision = performanceData.history.map(h => h.precision);
        
        performanceTrendChart.data.labels = labels;
        performanceTrendChart.data.datasets[0].data = accuracy;
        performanceTrendChart.data.datasets[1].data = precision;
        performanceTrendChart.update();
    }
}

function updateDriftTrendChart(driftData) {
    // Update chart with drift history
    if (driftData && driftData.history) {
        const labels = driftData.history.map(h => formatDate(h.date));
        const scores = driftData.history.map(h => h.drift_score);
        
        driftTrendChart.data.labels = labels;
        driftTrendChart.data.datasets[0].data = scores;
        driftTrendChart.update();
    }
}

function showDriftAnalysis(modelId) {
    fetch(`/api/model/performance?model_id=${modelId}&days_back=30`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showDriftModal(data);
            }
        })
        .catch(error => {
            console.error('Error fetching drift analysis:', error);
            showNotification('Error loading drift analysis', 'error');
        });
}

function showDriftModal(data) {
    const container = document.getElementById('drift-analysis-container');
    
    let html = `
        <h6>Model: ${data.model_info.model_id}</h6>
        <p class="text-muted">Analysis Period: ${data.analysis_period_days} days</p>
        
        <div class="row mb-3">
            <div class="col-md-6">
                <h6>Performance Metrics</h6>
                <ul class="list-unstyled">
                    <li>Accuracy: ${formatPercentage(data.performance_metrics.accuracy)}</li>
                    <li>Precision: ${formatPercentage(data.performance_metrics.precision)}</li>
                    <li>Recall: ${formatPercentage(data.performance_metrics.recall)}</li>
                    <li>F1 Score: ${formatPercentage(data.performance_metrics.f1_score)}</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6>Drift Indicators</h6>
                <ul class="list-unstyled">
                    <li>Feature Drift: ${data.drift_score ? 'Detected' : 'None'}</li>
                    <li>Prediction Drift: ${data.pred_drift ? 'Detected' : 'None'}</li>
                    <li>Target Drift: ${data.target_drift ? 'Detected' : 'None'}</li>
                </ul>
            </div>
        </div>
        
        <div class="alert alert-info">
            <strong>Recommendation:</strong> 
            ${getDriftRecommendation(data)}
        </div>
    `;
    
    container.innerHTML = html;
    
    const modal = new bootstrap.Modal(document.getElementById('driftModal'));
    modal.show();
}

function showEventDetails(eventId) {
    // Implementation for showing detailed event information
    console.log('Show event details for:', eventId);
}

// Utility functions
function formatPercentage(value) {
    if (!value && value !== 0) return 'N/A';
    return (value * 100).toFixed(1) + '%';
}

function formatDate(dateString) {
    return new Date(dateString).toLocaleString();
}

function formatEventType(type) {
    return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
}

function formatEventMetrics(event) {
    if (event.drift_score) {
        return `Drift: ${event.drift_score.toFixed(3)}`;
    }
    if (event.accuracy) {
        return `Acc: ${formatPercentage(event.accuracy)}`;
    }
    return 'N/A';
}

function getEventStatusColor(event) {
    if (event.drift_detected) return 'danger';
    if (event.event_type === 'performance_check') return 'success';
    return 'info';
}

function getEventStatus(event) {
    if (event.drift_detected) return 'Alert';
    if (event.event_type === 'performance_check') return 'OK';
    return 'Info';
}

function getDriftRecommendation(data) {
    const driftScore = data.drift_score || 0;
    
    if (driftScore > 0.7) {
        return 'High drift detected. Consider retraining the model immediately.';
    } else if (driftScore > 0.4) {
        return 'Moderate drift detected. Monitor closely and consider retraining soon.';
    } else {
        return 'Model performance is stable. Continue monitoring.';
    }
}

function showNotification(message, type = 'info') {
    // Use your notification system here
    console.log(`${type.toUpperCase()}: ${message}`);
}
