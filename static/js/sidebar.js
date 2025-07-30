
function updateSidebar() {
    fetch('/api/system_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Update Logs
                const logsContainer = document.getElementById('sidebar-logs');
                logsContainer.innerHTML = '';
                data.logs.forEach(log => {
                    const logEntry = document.createElement('div');
                    logEntry.classList.add('list-group-item');
                    let logLevelClass = '';
                    switch (log.level) {
                        case 'ERROR':
                            logLevelClass = 'text-danger';
                            break;
                        case 'WARNING':
                            logLevelClass = 'text-warning';
                            break;
                        case 'INFO':
                            logLevelClass = 'text-info';
                            break;
                    }
                    logEntry.innerHTML = `<span class="${logLevelClass}">[${log.level}]</span> ${log.message}`;
                    logsContainer.appendChild(logEntry);
                });

                // Update Model Metrics
                const metricsContainer = document.getElementById('sidebar-model-metrics');
                metricsContainer.innerHTML = '';
                data.model_metrics.forEach(model => {
                    const metricEntry = document.createElement('div');
                    metricEntry.innerHTML = `<strong>${model.model_name}:</strong> Accuracy - ${model.accuracy.toFixed(2)}`;
                    metricsContainer.appendChild(metricEntry);
                });

                // Update System Health
                const healthContainer = document.getElementById('sidebar-system-health');
                healthContainer.innerHTML = `<strong>Total Races:</strong> ${data.db_stats.total_races}`;
            } else {
                const errorBanner = document.createElement('div');
                errorBanner.classList.add('alert', 'alert-danger');
                errorBanner.textContent = `Error updating sidebar: ${data.message}`;
                document.body.appendChild(errorBanner);
                setTimeout(() => errorBanner.remove(), 5000);
            }
        })
        .catch(error => {
            const errorBanner = document.createElement('div');
            errorBanner.classList.add('alert', 'alert-danger');
            errorBanner.textContent = `Error updating sidebar: ${error}`;
            document.body.appendChild(errorBanner);
            setTimeout(() => errorBanner.remove(), 5000);
        });
}

document.addEventListener('DOMContentLoaded', () => {
    updateSidebar();
    setInterval(updateSidebar, 5000); // Poll every 5 seconds
});

