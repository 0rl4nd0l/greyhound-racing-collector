/* JavaScript for ML Dashboard Functionality */
function initializeDashboard() {
    refreshRaceCards();
    loadFeatureImportanceChart();
    loadConfidenceChart();
    loadPerformanceTrendChart();
}

function refreshRaceCards() {
    fetch('/api/recent_races')
        .then(response => response.json())
        .then(data => updateRaceCards(data.races));
}

function updateRaceCards(races) {
    const container = document.getElementById('race-cards-container');
    container.innerHTML = '';
    const racesArray = Array.isArray(races) ? races : Object.values(races || {});
    racesArray.forEach(race => {
        const card = document.createElement('div');
        card.className = 'race-card';
        card.innerHTML = `
            <div class="race-header">
                <span class="race-title">${race.race_name} - ${race.venue}</span>
                <span class="race-time">${new Date(race.race_date).toLocaleDateString()}</span>
            </div>
            <div class="dogs-grid">
                <!-- Loop through dogs for the race -->
            </div>
        `;
        container.appendChild(card);
    });
}

function loadFeatureImportanceChart() {
    const ctx = document.getElementById('feature-importance-chart');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Speed', 'Form', 'Weight', 'Odds'],
            datasets: [{
                label: 'Feature Importance',
                data: [0.2, 0.5, 0.1, 0.2],
                backgroundColor: [
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(75, 192, 192, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        }
    });
}

function loadConfidenceChart() {
    const ctx = document.getElementById('confidence-chart');
    new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['High', 'Medium', 'Low'],
            datasets: [{
                label: 'Model Confidence',
                data: [60, 30, 10],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.6)',
                    'rgba(255, 206, 86, 0.6)',
                    'rgba(255, 99, 132, 0.6)'
                ],
            }]
        }
    });
}

function loadPerformanceTrendChart() {
    const ctx = document.getElementById('performance-trend-chart');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['January', 'February', 'March', 'April', 'May'],
            datasets: [{
                label: 'Performance Over Time',
                data: [85, 90, 80, 88, 95],
                fill: false,
                borderColor: 'rgba(75, 192, 192, 1)',
                tension: 0.1
            }]
        }
    });
}
