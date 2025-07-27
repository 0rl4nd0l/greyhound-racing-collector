// Predictions Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializePredictions();
    
    // Setup event listeners
    document.getElementById('refresh-predictions-btn').addEventListener('click', refreshPredictions);
    
    // Auto-refresh every minute
    setInterval(refreshPredictions, 60000);
});

let winProbabilityChart = null;
let confidenceChart = null;

function initializePredictions() {
    // Initialize charts
    initializeCharts();
    // Load initial data
    refreshPredictions();
}

function refreshPredictions() {
    // Load predictions for upcoming races
    fetch('/api/predictions/upcoming')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateActiveModelInfo(data);
                updateUpcomingRaces(data);
                updatePredictionsTable(data);
                updateCharts(data);
            }
        })
        .catch(error => {
            console.error('Error fetching predictions:', error);
            showNotification('Error loading predictions data', 'error');
        });
}

function updateActiveModelInfo(data) {
    const container = document.getElementById('active-model-info');
    const models = {};
    
    // Extract unique models used for predictions
    Object.values(data.predictions).forEach(racePredictions => {
        Object.values(racePredictions).forEach(prediction => {
            if (prediction.model_info) {
                models[prediction.model_info.model_id] = prediction.model_info;
            }
        });
    });
    
    let html = '';
    Object.values(models).forEach(model => {
        html += `
            <div class="mb-2">
                <small class="text-muted">Model ID:</small> ${model.model_id}<br>
                <small class="text-muted">Performance:</small> ${formatScore(model.performance_score)}
            </div>
        `;
    });
    
    container.innerHTML = html || '<p class="text-muted">No active models</p>';
}

function updateUpcomingRaces(data) {
    const container = document.getElementById('upcoming-races-container');
    const races = Object.entries(data.predictions);
    
    if (races.length === 0) {
        container.innerHTML = '<p class="text-muted">No upcoming races found</p>';
        return;
    }
    
    let html = '<div class="row g-2">';
    races.forEach(([raceId, predictions]) => {
        const raceInfo = getRaceInfo(raceId, predictions);
        html += `
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-body">
                        <h6 class="card-title">${raceInfo.venue} - Race ${raceInfo.number}</h6>
                        <p class="card-text">
                            <small class="text-muted">Time:</small> ${formatTime(raceInfo.time)}<br>
                            <small class="text-muted">Predictions:</small> ${Object.keys(predictions).length} available
                        </p>
                        <button class="btn btn-sm btn-outline-primary" onclick="showRacePredictions('${raceId}')">
                            View Details
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    container.innerHTML = html;
}

function updatePredictionsTable(data) {
    const tbody = document.getElementById('predictions-table-body');
    tbody.innerHTML = '';
    
    Object.entries(data.predictions).forEach(([raceId, racePredictions]) => {
        const raceInfo = getRaceInfo(raceId, racePredictions);
        
        // Get predictions for each runner
        Object.entries(racePredictions).forEach(([predType, prediction]) => {
            if (prediction.predictions) {
                prediction.predictions.forEach(runnerPred => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${raceInfo.venue} R${raceInfo.number}</td>
                        <td>${formatTime(raceInfo.time)}</td>
                        <td>${runnerPred.dog_name || runnerPred.box}</td>
                        <td>${formatProbability(runnerPred.win_probability)}</td>
                        <td>${formatProbability(runnerPred.place_probability)}</td>
                        <td>${formatTime(runnerPred.predicted_time)}</td>
                        <td>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar bg-success" 
                                     role="progressbar" 
                                     style="width: ${runnerPred.confidence * 100}%">
                                    ${formatProbability(runnerPred.confidence)}
                                </div>
                            </div>
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        });
    });
}

function initializeCharts() {
    // Initialize Win Probability Distribution chart
    const winProbCtx = document.getElementById('win-probability-chart');
    winProbabilityChart = new Chart(winProbCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Win Probability',
                data: [],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
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
    
    // Initialize Confidence Distribution chart
    const confidenceCtx = document.getElementById('confidence-chart');
    confidenceChart = new Chart(confidenceCtx, {
        type: 'doughnut',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(255, 99, 132, 0.2)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(255, 99, 132, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true
        }
    });
}

function updateCharts(data) {
    // Collect all predictions
    const allPredictions = [];
    Object.values(data.predictions).forEach(racePredictions => {
        Object.values(racePredictions).forEach(prediction => {
            if (prediction.predictions) {
                allPredictions.push(...prediction.predictions);
            }
        });
    });
    
    // Update Win Probability chart
    const winProbs = allPredictions.map(p => p.win_probability).sort((a, b) => a - b);
    winProbabilityChart.data.labels = winProbs.map((_, i) => `Dog ${i + 1}`);
    winProbabilityChart.data.datasets[0].data = winProbs;
    winProbabilityChart.update();
    
    // Update Confidence chart
    const confidenceCounts = [0, 0, 0]; // High, Medium, Low
    allPredictions.forEach(p => {
        if (p.confidence >= 0.7) confidenceCounts[0]++;
        else if (p.confidence >= 0.4) confidenceCounts[1]++;
        else confidenceCounts[2]++;
    });
    confidenceChart.data.datasets[0].data = confidenceCounts;
    confidenceChart.update();
}

function showRacePredictions(raceId) {
    fetch(`/api/predictions/upcoming`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            race_ids: [raceId],
            prediction_types: ['win_probability', 'place_probability', 'race_time']
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const predictions = data.predictions[raceId];
            showPredictionModal(raceId, predictions);
        }
    })
    .catch(error => {
        console.error('Error fetching race predictions:', error);
        showNotification('Error loading race predictions', 'error');
    });
}

function showPredictionModal(raceId, predictions) {
    const container = document.getElementById('prediction-details-container');
    const raceInfo = getRaceInfo(raceId, predictions);
    
    let html = `
        <h6>${raceInfo.venue} - Race ${raceInfo.number}</h6>
        <p class="text-muted">Scheduled: ${formatTime(raceInfo.time)}</p>
        
        <div class="table-responsive">
            <table class="table table-sm">
                <thead>
                    <tr>
                        <th>Box</th>
                        <th>Dog</th>
                        <th>Win %</th>
                        <th>Place %</th>
                        <th>Pred. Time</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    // Combine predictions for each runner
    const runnerPredictions = {};
    Object.entries(predictions).forEach(([predType, prediction]) => {
        if (prediction.predictions) {
            prediction.predictions.forEach(runnerPred => {
                const box = runnerPred.box;
                if (!runnerPredictions[box]) {
                    runnerPredictions[box] = runnerPred;
                } else {
                    Object.assign(runnerPredictions[box], runnerPred);
                }
            });
        }
    });
    
    // Sort by box number and add to table
    Object.values(runnerPredictions)
        .sort((a, b) => a.box - b.box)
        .forEach(pred => {
            html += `
                <tr>
                    <td>${pred.box}</td>
                    <td>${pred.dog_name || '-'}</td>
                    <td>${formatProbability(pred.win_probability)}</td>
                    <td>${formatProbability(pred.place_probability)}</td>
                    <td>${formatTime(pred.predicted_time)}</td>
                    <td>
                        <div class="progress" style="height: 20px;">
                            <div class="progress-bar bg-success" 
                                 role="progressbar" 
                                 style="width: ${pred.confidence * 100}%">
                                ${formatProbability(pred.confidence)}
                            </div>
                        </div>
                    </td>
                </tr>
            `;
        });
    
    html += `
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = html;
    
    const modal = new bootstrap.Modal(document.getElementById('predictionModal'));
    modal.show();
}

// Utility functions
function getRaceInfo(raceId, predictions) {
    // Extract race info from ID and predictions
    const [venue, raceNum] = raceId.split('_r');
    return {
        venue: venue.replace(/-/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        number: raceNum.split('_')[0],
        time: predictions.race_time || 'TBD'
    };
}

function formatTime(time) {
    if (!time) return 'TBD';
    // Implement time formatting based on your data format
    return time;
}

function formatProbability(prob) {
    if (!prob) return '-';
    return (prob * 100).toFixed(1) + '%';
}

function formatScore(score) {
    if (!score) return '-';
    return (score * 100).toFixed(1) + '%';
}

function showNotification(message, type = 'info') {
    // Use your notification system here
    console.log(`${type.toUpperCase()}: ${message}`);
}
