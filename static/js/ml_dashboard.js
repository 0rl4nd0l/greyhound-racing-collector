// Greyhound Racing ML Dashboard - Enhanced Interactivity
document.addEventListener('DOMContentLoaded', () => {
    const state = {
        races: [],
        predictions: [],
        autoRefreshInterval: null,
        isLoading: false,
        charts: {},
        lastPredictionData: null,
        currentFilePath: null
    };

    const elements = {
        racesContainer: document.getElementById('races-container'),
        progressBarContainer: document.getElementById('progress-bar-container'),
        progressBar: document.getElementById('progress-bar'),
        runAllButton: document.getElementById('run-all-predictions'),
        autoRefreshToggle: document.getElementById('auto-refresh-toggle'),
        refreshIntervalSelect: document.getElementById('refresh-interval'),
        exportCsvButton: document.getElementById('export-csv'),
        exportPdfButton: document.getElementById('export-pdf'),
        autoAdvisoryToggle: document.getElementById('auto-advisory-toggle'),
        generateAdvisoryBtn: document.getElementById('generate-advisory-btn'),
        advisoryMessagesContainer: document.getElementById('advisory-messages')
    };

    // Initialize the dashboard
    function init() {
        setupEventListeners();
        initCharts();
        loadDashboardData();
        loadAutoRefreshState();
    }

    // Setup all event listeners
    function setupEventListeners() {
        if (elements.runAllButton) {
            elements.runAllButton.addEventListener('click', runAllPredictions);
        }
        if (elements.autoRefreshToggle) {
            elements.autoRefreshToggle.addEventListener('change', handleAutoRefresh);
        }
        if (elements.refreshIntervalSelect) {
            elements.refreshIntervalSelect.addEventListener('change', handleAutoRefresh);
        }
        if (elements.exportCsvButton) {
            elements.exportCsvButton.addEventListener('click', exportToCsv);
        }
        if (elements.exportPdfButton) {
            elements.exportPdfButton.addEventListener('click', exportToPdf);
        }
        if (elements.generateAdvisoryBtn) {
            elements.generateAdvisoryBtn.addEventListener('click', generateAdvisoryManually);
        }
    }

    // Load all dashboard data
    async function loadDashboardData() {
        if (state.isLoading) return;
        
        state.isLoading = true;
        try {
            await Promise.all([
                loadRaces(),
                loadPredictions(),
                loadHistoricalAccuracy()
            ]);
            updateUI();
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            showAlert('Error loading dashboard data', 'danger');
        } finally {
            state.isLoading = false;
        }
    }

    // Load races from API
    async function loadRaces() {
        try {
            const response = await fetch('/api/race_files_status');
            const data = await response.json();
            
            if (data.success) {
                state.races = [...(data.predicted_races || []), ...(data.unpredicted_races || [])];
            }
        } catch (error) {
            console.error('Error loading races:', error);
        }
    }

    // Load predictions from API
    async function loadPredictions() {
        try {
            const response = await fetch('/api/prediction_results');
            const data = await response.json();
            
            if (data.success) {
                state.predictions = data.predictions || [];
            }
        } catch (error) {
            console.error('Error loading predictions:', error);
        }
    }

    // Generate advisory automatically after successful prediction
    async function generateAdvisoryForPrediction(predictionData, filePath = null) {
        // Check if auto-advisory is enabled
        if (!elements.autoAdvisoryToggle || !elements.autoAdvisoryToggle.checked) {
            return;
        }

        // Store prediction data for manual generation
        state.lastPredictionData = predictionData;
        state.currentFilePath = filePath;

        await generateAdvisory(predictionData, filePath);
    }

    // Generate advisory manually via button click
    async function generateAdvisoryManually() {
        if (state.lastPredictionData) {
            await generateAdvisory(state.lastPredictionData, state.currentFilePath);
        } else {
            // Try to get the most recent prediction from the current state
            if (state.predictions && state.predictions.length > 0) {
                await generateAdvisory({ prediction_data: state.predictions[0] });
            } else {
                showAlert('No prediction data available to generate advisory', 'warning');
            }
        }
    }

    // Core advisory generation function
    async function generateAdvisory(predictionData, filePath = null) {
        if (!elements.advisoryMessagesContainer) {
            console.warn('Advisory messages container not found');
            return;
        }

        // Show loading spinner
        if (window.AdvisoryUtils && window.AdvisoryUtils.showAdvisoryLoading) {
            window.AdvisoryUtils.showAdvisoryLoading(
                elements.advisoryMessagesContainer,
                'Generating AI advisory...'
            );
        }

        // Prepare payload
        const payload = {};
        if (filePath) {
            payload.file_path = filePath;
        } else {
            payload.prediction_data = predictionData;
        }

        try {
            // Make non-blocking API call
            const response = await fetch('/api/generate_advisory', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.success) {
                // Clear loading state and render advisory
                if (window.AdvisoryUtils && window.AdvisoryUtils.clearAdvisory) {
                    window.AdvisoryUtils.clearAdvisory(elements.advisoryMessagesContainer);
                }
                
                if (window.AdvisoryUtils && window.AdvisoryUtils.renderAdvisory) {
                    window.AdvisoryUtils.renderAdvisory(result, elements.advisoryMessagesContainer);
                } else {
                    // Fallback rendering if AdvisoryUtils not available
                    renderAdvisoryFallback(result);
                }
            } else {
                throw new Error(result.message || 'Advisory generation failed');
            }
        } catch (error) {
            console.error('Error generating advisory:', error);
            
            // Clear loading state
            if (window.AdvisoryUtils && window.AdvisoryUtils.clearAdvisory) {
                window.AdvisoryUtils.clearAdvisory(elements.advisoryMessagesContainer);
            }
            
            // Show error alert
            showAlert(`Advisory generation failed: ${error.message}`, 'danger', elements.advisoryMessagesContainer);
        }
    }

    // Fallback advisory rendering if AdvisoryUtils is not available
    function renderAdvisoryFallback(result) {
        const alertClass = result.type === 'success' ? 'alert-success' :
                          result.type === 'warning' ? 'alert-warning' :
                          result.type === 'danger' ? 'alert-danger' : 'alert-info';
        
        const html = `
            <div class="alert ${alertClass} alert-dismissible fade show" role="alert">
                <h6><i class="fas fa-lightbulb"></i> ${result.title || 'Advisory'}</h6>
                <p>${result.message || 'Advisory generated successfully'}</p>
                ${result.details && result.details.length > 0 ?
                    `<ul class="mb-0">${result.details.map(detail => `<li>${detail}</li>`).join('')}</ul>` :
                    ''
                }
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            </div>
        `;
        
        elements.advisoryMessagesContainer.innerHTML = html;
    }

    // Enhanced alert function that can target specific containers
    function showAlert(message, type, container = null) {
        // Use the enhanced alert logic for containers
        if (container) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            container.innerHTML = '';
            container.appendChild(alertDiv);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, 5000);
            return;
        }
        
        // Global alert fallback - check if global showAlert exists
        if (typeof window.showAlert === 'function' && window.showAlert !== showAlert) {
            window.showAlert(message, type);
            return;
        }
        
        // Final fallback - create floating alert
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 1050; max-width: 400px;';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(alertDiv);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, 5000);
    }

    // Expose functions globally for template access
    window.generateAdvisoryForPrediction = generateAdvisoryForPrediction;
    window.generateAdvisoryManually = generateAdvisoryManually;

    init();
});

function initializeCharts() {
    // Win Probability Chart
    const winProbCtx = document.getElementById('winProbabilityChart').getContext('2d');
    const winProbabilityChart = new Chart(winProbCtx, {
        type: 'bar',
        data: {
            labels: [], // Race names
            datasets: [{
                label: 'Win Probability',
                data: [], // Probabilities
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            },
            onClick: (event, elements) => {
                if (elements.length > 0) {
                    const raceIndex = elements[0].index;
                    const race = raceData.predicted[raceIndex];
                    showPredictionDetail(race.race_name);
                }
            }
        }
    });

    // Confidence Pie Chart
    const confidenceCtx = document.getElementById('confidencePieChart').getContext('2d');
    const confidencePieChart = new Chart(confidenceCtx, {
        type: 'pie',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                label: 'Prediction Confidence',
                data: [0, 0, 0],
                backgroundColor: [
                    'rgba(40, 167, 69, 0.7)',
                    'rgba(255, 193, 7, 0.7)',
                    'rgba(220, 53, 69, 0.7)'
                ]
            }]
        }
    });

    // Historical Accuracy Chart
    const historicalCtx = document.getElementById('historicalAccuracyChart').getContext('2d');
    const historicalAccuracyChart = new Chart(historicalCtx, {
        type: 'line',
        data: {
            labels: [], // Dates
            datasets: [{
                label: 'Model Accuracy',
                data: [], // Accuracy values
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.1
            }]
        }
    });

    // Store charts for later updates
    window.charts = {
        winProbabilityChart,
        confidencePieChart,
        historicalAccuracyChart
    };

    // Fetch data for historical accuracy chart
    fetch('/api/model/historical_accuracy')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateHistoricalAccuracyChart(data.accuracy_data);
            }
        });
}

function updateCharts(predictedRaces) {
    if (!window.charts) return;

    // Update Win Probability Chart
    const winProbChart = window.charts.winProbabilityChart;
    winProbChart.data.labels = predictedRaces.map(r => r.race_name);
    winProbChart.data.datasets[0].data = predictedRaces.map(r => r.top_pick ? r.top_pick.prediction_score : 0);
    winProbabilityChart.update();

    // Update Confidence Pie Chart
    const confidencePieChart = window.charts.confidencePieChart;
    const confidenceCounts = { 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0 };
    predictedRaces.forEach(race => {
        if (race.top_pick && race.top_pick.confidence_level) {
            confidenceCounts[race.top_pick.confidence_level]++;
        }
    });
    confidencePieChart.data.datasets[0].data = [confidenceCounts.HIGH, confidenceCounts.MEDIUM, confidenceCounts.LOW];
    confidencePieChart.update();
}

function updateHistoricalAccuracyChart(accuracyData) {
    if (!window.charts) return;

    const historicalAccuracyChart = window.charts.historicalAccuracyChart;
    historicalAccuracyChart.data.labels = accuracyData.map(d => d.date);
    historicalAccuracyChart.data.datasets[0].data = accuracyData.map(d => d.accuracy);
    historicalAccuracyChart.update();
}

function showPredictionDetail(raceName) {
    const race = raceData.predicted.find(r => r.race_name === raceName);
    if (!race) return;

    const modalBody = document.getElementById('predictionDetailContent');
    modalBody.innerHTML = `<h5>${race.race_name}</h5><p>Top pick: ${race.top_pick.dog_name}</p>`; // Simple example
    
    new bootstrap.Modal(document.getElementById('predictionDetailModal')).show();
}
