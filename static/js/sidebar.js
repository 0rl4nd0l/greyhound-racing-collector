
function updateSidebar() {
    fetch('/api/system_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateSidebarWithData(data);
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

function updateSidebarWithData(data) {
    if (!data) return;

    // Update Logs
    const logsContainer = document.getElementById('sidebar-logs');
    if (logsContainer && data.logs) {
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
    }

    // Update Model Metrics
    const metricsContainer = document.getElementById('sidebar-model-metrics');
    if (metricsContainer && data.model_metrics) {
        metricsContainer.innerHTML = '';
        data.model_metrics.forEach(model => {
            const metricEntry = document.createElement('div');
            metricEntry.innerHTML = `<strong>${model.model_name}:</strong> Accuracy - ${model.accuracy.toFixed(2)}`;
            metricsContainer.appendChild(metricEntry);
        });
    }

    // Update System Health
    const healthContainer = document.getElementById('sidebar-system-health');
    if (healthContainer && data.db_stats) {
        let healthHtml = `<strong>Total Races:</strong> ${data.db_stats.total_races}`;
        // Append GPT diagnostics summary if available
        try {
            const gpt = data.gpt_diagnostics || {};
            if (typeof gpt === 'object' && ('json_files_count' in gpt || 'archives_count' in gpt)) {
                const jsonCount = typeof gpt.json_files_count === 'number' ? gpt.json_files_count : 0;
                const zipCount = typeof gpt.archives_count === 'number' ? gpt.archives_count : 0;
                healthHtml += `<div class="mt-2"><strong>GPT Diagnostics:</strong> JSON ${jsonCount}, Archives ${zipCount}</div>`;
            }
        } catch (e) {
            // non-fatal
        }
        healthContainer.innerHTML = healthHtml;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const eventSourceSupported = !!window.EventSource;

    // Delay establishing long-lived realtime connections slightly so initial page load
    // can reach a stable state (helps automated tests that wait for networkidle).
    const connectRealtime = () => {
        if (eventSourceSupported) {
            const eventSource = new EventSource('/api/predict_stream');
            eventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data || '{}');
                    if (data.type === 'result') {
                        updateSidebarWithData(data);
                    } else if (data.type === 'error') {
                        console.error('Prediction Stream Error:', data.message);
                    }
                } catch (e) {
                    // non-fatal
                }
            };
            eventSource.onerror = function(event) {
                console.error('Prediction EventSource failed. Falling back to polling.', event);
                try { eventSource.close(); } catch {}
                initiatePolling();
            };
        } else {
            console.log('EventSource not supported. Using fallback polling.');
            initiatePolling();
        }
    };

    // 1.25s strike balance: snappy UI but avoids blocking networkidle in tests
    setTimeout(connectRealtime, 1250);

    function initiatePolling() {
        updateSidebar();
        let interval = 5000;
        const poll = () => {
            updateSidebarWithBackoff()
                .finally(() => {
                    setTimeout(poll, interval);
                    interval = Math.min(interval * 2, 30000); // Back-off strategy
                });
        };
        poll();
    }
});

function updateSidebarWithBackoff() {
    return fetch('/api/system_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateSidebarWithData(data);
            } else {
                console.error('Failed to update sidebar:', data.message);
            }
        })
        .catch(error => {
            console.error('Error updating sidebar:', error);
        });
}

