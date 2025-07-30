// Greyhound Racing ML Dashboard - Enhanced Interactivity
document.addEventListener('DOMContentLoaded', () => {
    const state = {
        races: [],
        predictions: [],
        autoRefreshInterval: null,
        isLoading: false,
        charts: {}
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
