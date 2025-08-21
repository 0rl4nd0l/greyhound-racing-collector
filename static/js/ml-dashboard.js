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

// --- Backtesting logs compatibility for ML training dashboard (hyphen file) ---
(function(){
    function byIdsCompat(ids) {
        for (const id of ids) {
            const el = document.getElementById(id);
            if (el) return el;
        }
        return null;
    }
    function initBacktestingLogsIntegrationCompat() {
        const logsEl = byIdsCompat(['backtesting-logs','training-logs','trainingLogs','ml-training-logs']);
        if (!logsEl) return;
        const statusEl = byIdsCompat(['backtesting-status','training-status','trainingStatus','ml-training-status']);
        const refreshBtn = byIdsCompat(['refresh-backtesting-logs','refresh-training-logs']);
        const limitSel = byIdsCompat(['backtesting-log-limit','training-log-limit']);

        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
                await pollBacktestingLogsCompat(limit, logsEl, statusEl);
            });
        }
        startBacktestingSSECompat(logsEl, statusEl, limitSel);
        const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
        pollBacktestingLogsCompat(limit, logsEl, statusEl).catch(() => {});
    }
    function startBacktestingSSECompat(logsEl, statusEl, limitSel) {
        let es;
        let delay = 1000;
        const maxDelay = 30000;
        const connect = () => {
            try {
                if (statusEl) statusEl.textContent = 'Connecting to backtesting log stream...';
                es = new EventSource('/api/backtesting/logs/stream');
                es.onopen = () => { if (statusEl) statusEl.textContent = 'Connected to backtesting log stream'; delay = 1000; };
                es.onmessage = (ev) => appendBacktestingLogLineCompat(ev.data, logsEl);
                es.addEventListener('completed', () => {
                    if (statusEl) statusEl.textContent = 'Backtesting run completed';
                    const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
                    pollBacktestingLogsCompat(limit, logsEl, statusEl).catch(() => {});
                });
                es.onerror = () => {
                    if (statusEl) statusEl.textContent = 'Stream error, falling back to polling';
                    try { es.close(); } catch {}
                    const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
                    pollBacktestingLogsCompat(limit, logsEl, statusEl).catch(() => {});
                    setTimeout(connect, delay);
                    delay = Math.min(delay * 2, maxDelay);
                };
            } catch (e) {
                if (statusEl) statusEl.textContent = 'Failed to open log stream';
            }
        };
        connect();
    }
    async function pollBacktestingLogsCompat(limit, logsEl, statusEl) {
        try {
            if (statusEl) statusEl.textContent = 'Loading backtesting logs...';
            const resp = await fetch(`/api/backtesting/logs?limit=${encodeURIComponent(limit)}`);
            const data = await resp.json();
            if (!data || data.success === false) throw new Error((data && (data.error || data.message)) || 'Unknown error');
            if (statusEl) {
                const parts = [];
                parts.push(data.running ? 'Running' : (data.completed ? 'Completed' : 'Idle'));
                if (typeof data.progress === 'number') parts.push(`Progress: ${Math.round(data.progress * 100)}%`);
                if (data.current_task) parts.push(`Task: ${data.current_task}`);
                statusEl.textContent = parts.join(' • ');
            }
            renderBacktestingLogsCompat(data.logs || [], logsEl);
        } catch (err) {
            if (statusEl) statusEl.textContent = `Failed to load logs: ${err.message}`;
            console.error('Backtesting logs fetch error:', err);
        }
    }
    function renderBacktestingLogsCompat(entries, logsEl) {
        if (!Array.isArray(entries)) entries = [];
        const lines = entries.map(sanitizeBacktestingEntryToLineCompat).join('\n');
        if (logsEl.tagName === 'PRE' || logsEl.tagName === 'TEXTAREA' || logsEl.tagName === 'CODE') {
            logsEl.textContent = lines;
        } else {
            logsEl.innerHTML = '';
            const pre = document.createElement('pre');
            pre.className = 'mb-0 small';
            pre.textContent = lines;
            logsEl.appendChild(pre);
        }
        if (typeof logsEl.scrollTop === 'number') logsEl.scrollTop = logsEl.scrollHeight;
    }
    function appendBacktestingLogLineCompat(raw, logsEl) {
        const line = typeof raw === 'string' ? raw : JSON.stringify(raw);
        if (logsEl.tagName === 'PRE' || logsEl.tagName === 'TEXTAREA' || logsEl.tagName === 'CODE') {
            logsEl.textContent += (logsEl.textContent ? '\n' : '') + line;
        } else {
            let pre = logsEl.querySelector('pre');
            if (!pre) { pre = document.createElement('pre'); pre.className = 'mb-0 small'; logsEl.appendChild(pre); }
            pre.textContent += (pre.textContent ? '\n' : '') + line;
        }
        if (typeof logsEl.scrollTop === 'number') logsEl.scrollTop = logsEl.scrollHeight;
    }
    function sanitizeBacktestingEntryToLineCompat(entry) {
        try {
            const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : new Date().toLocaleString();
            const lvl = entry.level || (entry.severity || 'INFO');
            const msg = entry.message || entry.msg || entry.text || '';
            const extra = Object.keys(entry).filter(k => !['timestamp','level','message','msg','text'].includes(k)).reduce((a,k)=>{a[k]=entry[k];return a;},{});
            const extraStr = Object.keys(extra).length ? ` | ${JSON.stringify(extra)}` : '';
            return `[${ts}] ${lvl}: ${msg}${extraStr}`;
        } catch (_) {
            try { return typeof entry === 'string' ? entry : JSON.stringify(entry); } catch { return String(entry); }
        }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initBacktestingLogsIntegrationCompat);
    } else {
        initBacktestingLogsIntegrationCompat();
    }
})();
