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
    // Load race files status and prediction results
    Promise.allSettled([
        fetch('/api/race_files_status').then(response => response.json()).catch(e => ({success: false, error: e})),
        fetch('/api/prediction_results').then(response => response.json()).catch(e => ({success: false, error: e}))
    ])
    .then(([raceFilesResult, predictionsResult]) => {
        const raceFilesData = raceFilesResult.status === 'fulfilled' ? raceFilesResult.value : {success: false};
        const predictionsData = predictionsResult.status === 'fulfilled' ? predictionsResult.value : {success: false};
        
        // Update upcoming races if race files data is available
        if (raceFilesData.success) {
            updateUpcomingRaces(raceFilesData);
        } else {
            console.warn('Race files data not available:', raceFilesData.error);
            document.getElementById('upcoming-races-container').innerHTML = '<p class="text-muted">Error loading race data</p>';
        }
        
        // Update predictions table and charts if prediction data is available
        if (predictionsData.success) {
            updatePredictionsTable(predictionsData);
            updateCharts(predictionsData);
        } else {
            console.warn('Predictions data not available:', predictionsData.error || predictionsData.message);
            // Use race files data to show basic prediction table if available
            if (raceFilesData.success && raceFilesData.predicted_races) {
                updatePredictionsTableFromRaceFiles(raceFilesData);
            } else {
                document.getElementById('predictions-table-body').innerHTML = '<tr><td colspan="7" class="text-center text-muted">No predictions available</td></tr>';
            }
            // Clear charts
            updateCharts({predictions: []});
        }
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
    const upcomingRaces = data.unpredicted_races || [];
    const predictedRaces = data.predicted_races || [];
    
    if (upcomingRaces.length === 0 && predictedRaces.length === 0) {
        container.innerHTML = '<p class="text-muted">No upcoming races found</p>';
        return;
    }
    
    let html = '<div class="row g-2">';
    
    // Show unpredicted races first
    upcomingRaces.slice(0, 4).forEach(race => {
        const raceInfo = {
            venue: race.race_id.replace(/_/g, ' ').toUpperCase(),
            number: 'TBD',
            time: 'Awaiting prediction'
        };
        html += `
            <div class="col-md-6">
                <div class="card h-100 border-warning">
                    <div class="card-body">
                        <h6 class="card-title">${raceInfo.venue}</h6>
                        <p class="card-text">
                            <small class="text-muted">Status:</small> <span class="badge bg-warning">Pending Prediction</span><br>
                            <small class="text-muted">File:</small> ${race.filename}
                        </p>
                        <button class="btn btn-sm btn-warning" onclick="predictRace('${race.filename}')">
                            Run Prediction
                        </button>
                    </div>
                </div>
            </div>
        `;
    });
    
    // Show recent predicted races
    predictedRaces.slice(0, 4).forEach(race => {
        html += `
            <div class="col-md-6">
                <div class="card h-100 border-success">
                    <div class="card-body">
                        <h6 class="card-title">${race.venue} - Race ${race.race_date}</h6>
                        <p class="card-text">
                            <small class="text-muted">Top Pick:</small> ${race.top_pick?.dog_name || 'N/A'}<br>
                            <small class="text-muted">Method:</small> ${race.prediction_method}
                        </p>
                        <button class="btn btn-sm btn-success" onclick="showRaceDetails('${race.race_name}')">
                            View Results
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
    
    const predictions = data.predictions || [];
    
    if (predictions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">No predictions available</td></tr>';
        return;
    }
    
    // Take recent predictions and limit to show top runners
    predictions.slice(0, 20).forEach(race => {
        const runners = race.predictions || [];
        
        // Sort runners by win probability and show top performers
        runners.sort((a, b) => (b.win_probability || 0) - (a.win_probability || 0))
               .slice(0, 3) // Show top 3 per race
               .forEach(runner => {
                   const tr = document.createElement('tr');
                   tr.innerHTML = `
                       <td>${race.venue} - ${race.race_date}</td>
                       <td>${formatTime(race.scheduled_time || 'TBD')}</td>
                       <td>${runner.dog_name || `Box ${runner.box || 'N/A'}`}</td>
                       <td>${formatProbability(runner.win_probability)}</td>
                       <td>${formatProbability(runner.place_probability)}</td>
                       <td>${formatTime(runner.predicted_time)}</td>
                       <td>
                           <div class="progress" style="height: 20px;">
                               <div class="progress-bar bg-success" 
                                    role="progressbar" 
                                    style="width: ${(runner.confidence || 0) * 100}%">
                                   ${formatProbability(runner.confidence)}
                               </div>
                           </div>
                       </td>
                   `;
                   tbody.appendChild(tr);
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
    // Collect all predictions from the race files
    const allPredictions = [];
    const predictions = data.predictions || [];
    
    predictions.forEach(race => {
        const runners = race.predictions || [];
        runners.forEach(runner => {
            if (runner.win_probability !== undefined) {
                allPredictions.push({
                    win_probability: runner.win_probability,
                    confidence: runner.confidence || 0.5
                });
            }
        });
    });
    
    if (allPredictions.length === 0) {
        // Show empty charts if no data
        winProbabilityChart.data.labels = [];
        winProbabilityChart.data.datasets[0].data = [];
        winProbabilityChart.update();
        
        confidenceChart.data.datasets[0].data = [0, 0, 0];
        confidenceChart.update();
        return;
    }
    
    // Update Win Probability chart
    const winProbs = allPredictions.map(p => p.win_probability).sort((a, b) => a - b);
    winProbabilityChart.data.labels = winProbs.slice(0, 20).map((_, i) => `Runner ${i + 1}`);
    winProbabilityChart.data.datasets[0].data = winProbs.slice(0, 20);
    winProbabilityChart.update();
    
    // Update Confidence chart - use prediction_score as confidence proxy
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
