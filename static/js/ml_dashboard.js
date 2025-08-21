// Greyhound Racing ML Dashboard - Enhanced Interactivity
(function(){
    const run = () => {
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
        // Initialize backtesting log integration if DOM elements exist
        initBacktestingLogsIntegration();
        // Bind Backtesting triggers so clicking the button starts streaming into the panel
        bindBacktestingTriggers();
    }

    // Setup all event listeners
    function setupEventListeners() {
        // track which promotion timestamps we've toasted to avoid duplicates
        window.__promotionToastSeen = window.__promotionToastSeen || new Set();
        // Diagnostics run button (if present)
        const diagRunBtn = document.getElementById('diagnostics-run-btn');
        const diagMaxRaces = document.getElementById('diagnostics-max-races');
        const diagStatus = document.getElementById('diagnostics-status');
        const diagLogs = document.getElementById('diagnostics-logs');
        if (diagRunBtn) {
            diagRunBtn.addEventListener('click', async () => {
                try {
                    const maxRaces = diagMaxRaces ? parseInt(diagMaxRaces.value || '400', 10) : 400;
                    const modelsSel = document.getElementById('diag-models');
                    const calsSel = document.getElementById('diag-cals');
                    const tuneChk = document.getElementById('diag-tune');
                    const tuneIter = document.getElementById('diag-tune-iter');
                    const tuneCv = document.getElementById('diag-tune-cv');
                    const autoPromoteChk = document.getElementById('diag-auto-promote');
                    const models = modelsSel ? Array.from(modelsSel.selectedOptions).map(o => o.value).join(',') : undefined;
                    const calibrations = calsSel ? Array.from(calsSel.selectedOptions).map(o => o.value).join(',') : undefined;
                    const tune = !!(tuneChk && tuneChk.checked);
                    const tune_iter = tuneIter ? parseInt(tuneIter.value || '20', 10) : 20;
                    const tune_cv = tuneCv ? parseInt(tuneCv.value || '3', 10) : 3;
                    const auto_promote = !!(autoPromoteChk && autoPromoteChk.checked);
                    const payload = { max_races: maxRaces, models, calibrations, tune, tune_iter, tune_cv, auto_promote };
                    const resp = await fetch('/api/diagnostics/run', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await resp.json();
                    if (!data.success) throw new Error(data.error || 'Failed to start diagnostics');
                    const jobId = data.job_id;
                    if (diagStatus) diagStatus.textContent = `Running diagnostics job ${jobId}...`;
                    // Start SSE stream for logs
                    if (diagLogs) { diagLogs.textContent = ''; }
                    const es = new EventSource(`/api/jobs/${jobId}/logs/stream`);
                    es.onmessage = (ev) => {
                        if (!diagLogs) return;
                        diagLogs.textContent += ev.data + '\n';
                        diagLogs.scrollTop = diagLogs.scrollHeight;
                        try {
                            const txt = ev.data || '';
                            if (txt.includes('Promotion audit written') || txt.includes('model_promotion')) {
                                // Attempt to fetch last promotion and toast once
                                maybeToastLastPromotion();
                            }
                        } catch(_) {}
                    };
                    es.addEventListener('completed', () => {
                        if (diagStatus) diagStatus.textContent = `Diagnostics job ${jobId} completed.`;
                        // Give backend a moment to flush audit then fetch
                        setTimeout(maybeToastLastPromotion, 500);
                        es.close();
                    });
                    es.addEventListener('error', (e) => {
                        if (diagStatus) diagStatus.textContent = `Diagnostics stream error.`;
                    });
                    // Poll status for stage/progress
                    const poll = async () => {
                        try {
                            const s = await fetch(`/api/jobs/${jobId}/status`);
                            const st = await s.json();
                            if (st.success && st.job) {
                                const j = st.job;
                                if (diagStatus) diagStatus.textContent = `Status: ${j.status}${j.stage ? ' • ' + j.stage : ''}`;
                                if (j.status === 'completed' || j.status === 'failed' || j.status === 'canceled') return;
                            }
                        } catch (_) {}
                        setTimeout(poll, 1500);
                    };
                    poll();
                } catch (err) {
                    console.error('Diagnostics run error:', err);
                    if (diagStatus) diagStatus.textContent = `Error: ${err.message}`;
                }
            });
            // Autostart via URL params
            try {
                const params = new URLSearchParams(window.location.search);
                const autostart = params.get('autostart');
                const maxParam = params.get('max_races');
                const modelsParam = params.get('models');
                const calsParam = params.get('calibrations');
                const tuneParam = params.get('tune');
                const tuneIterParam = params.get('tune_iter');
                const tuneCvParam = params.get('tune_cv');
                const autoPromoteParam = params.get('auto_promote');
                if (diagMaxRaces && maxParam) {
                    const mv = parseInt(maxParam, 10);
                    if (!Number.isNaN(mv)) diagMaxRaces.value = String(mv);
                }
                const modelsSel = document.getElementById('diag-models');
                if (modelsSel && modelsParam) {
                    const wanted = new Set(modelsParam.split(',').map(s => s.trim()));
                    Array.from(modelsSel.options).forEach(opt => opt.selected = wanted.has(opt.value));
                }
                const calsSel = document.getElementById('diag-cals');
                if (calsSel && calsParam) {
                    const wanted = new Set(calsParam.split(',').map(s => s.trim()));
                    Array.from(calsSel.options).forEach(opt => opt.selected = wanted.has(opt.value));
                }
                const tuneChk = document.getElementById('diag-tune');
                if (tuneChk && (tuneParam === '1' || tuneParam === 'true')) tuneChk.checked = true;
                const tuneIter = document.getElementById('diag-tune-iter');
                if (tuneIter && tuneIterParam) tuneIter.value = String(parseInt(tuneIterParam, 10));
                const tuneCv = document.getElementById('diag-tune-cv');
                if (tuneCv && tuneCvParam) tuneCv.value = String(parseInt(tuneCvParam, 10));
                const autoPromoteChk = document.getElementById('diag-auto-promote');
                if (autoPromoteChk && (autoPromoteParam === '0' || autoPromoteParam === 'false')) autoPromoteChk.checked = false;
                if (autostart === '1' || autostart === 'true') {
                    setTimeout(() => diagRunBtn.click(), 300);
                }
            } catch (_) {}
        }
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

    // Helper: fetch last promotion and toast once
    async function maybeToastLastPromotion() {
        try {
            const resp = await fetch('/api/diagnostics/last_promotion');
            const data = await resp.json();
            if (!data.success || !data.found) return;
            const entry = data.entry || {};
            const key = `${entry.timestamp}|${entry.event}|${entry.details && entry.details.model_id || ''}`;
            if (window.__promotionToastSeen && window.__promotionToastSeen.has(key)) return;
            if (window.__promotionToastSeen) window.__promotionToastSeen.add(key);
            const eid = entry.event === 'model_promoted' ? 'success' : 'warning';
            const model = (entry.details && entry.details.model) || 'unknown';
            const cal = (entry.details && entry.details.calibration) || 'unknown';
            const mid = (entry.details && entry.details.model_id) || 'N/A';
            const roc = (entry.details && entry.details.roc_auc != null) ? `, ROC AUC=${entry.details.roc_auc.toFixed ? entry.details.roc_auc.toFixed(3) : entry.details.roc_auc}` : '';
            showAlert(`Model ${entry.event === 'model_promoted' ? 'promoted' : 'promotion failed'}: ${model}/${cal} (id=${mid})${roc}`, eid);
        } catch (_) {}
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

    // Backtesting logs integration
    function byIds(ids) {
        for (const id of ids) {
            const el = document.getElementById(id);
            if (el) return el;
        }
        return null;
    }
    function findTrainingPanelFallback() {
        // Try to locate the Training Progress & Logs panel heuristically
        const candidates = Array.from(document.querySelectorAll('.card, .panel, .container, div'));
        for (const el of candidates) {
            try {
                const header = el.querySelector('.card-header, header, h1, h2, h3, h4, .panel-title, .section-title');
                const headerText = (header && header.textContent || '').toLowerCase();
                if (!headerText) continue;
                if (headerText.includes('training') && (headerText.includes('progress') || headerText.includes('logs'))) {
                    // Within this container, find a likely logs area
                    const logsEl = el.querySelector('#backtesting-logs, #training-logs, #trainingLogs, #ml-training-logs, pre, code, textarea, .log, .logs, .console, .terminal');
                    // Status area is often a muted/alert area just above logs
                    const statusEl = el.querySelector('#backtesting-status, #training-status, #trainingStatus, #ml-training-status, .alert, .text-muted, .status');
                    // Optional control elements
                    const refreshBtn = el.querySelector('#refresh-backtesting-logs, #refresh-training-logs, button.refresh, .btn-refresh');
                    const limitSel = el.querySelector('#backtesting-log-limit, #training-log-limit, select.limit');
                    if (logsEl) {
                        return { logsEl, statusEl, refreshBtn, limitSel };
                    }
                }
            } catch (_) {}
        }
        return { logsEl: null, statusEl: null, refreshBtn: null, limitSel: null };
    }

    function initBacktestingLogsIntegration() {
        let logsEl = byIds(['backtesting-logs','training-logs','trainingLogs','ml-training-logs']);
        let statusEl = byIds(['backtesting-status','training-status','trainingStatus','ml-training-status']);
        let refreshBtn = byIds(['refresh-backtesting-logs','refresh-training-logs']);
        let limitSel = byIds(['backtesting-log-limit','training-log-limit']);

        if (!logsEl) {
            const fb = findTrainingPanelFallback();
            logsEl = fb.logsEl; statusEl = fb.statusEl; refreshBtn = fb.refreshBtn; limitSel = fb.limitSel;
        }

        if (!logsEl) return; // Activate only when a container is found

        // Attach manual refresh polling if button exists
        if (refreshBtn) {
            refreshBtn.addEventListener('click', async () => {
                const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
                await pollBacktestingLogs(limit, logsEl, statusEl);
            });
        }

        // Start SSE stream for live logs; fallback to polling on error
        startBacktestingSSE(logsEl, statusEl, limitSel);

        // Also do an initial poll to show recent logs immediately
        const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
        pollBacktestingLogs(limit, logsEl, statusEl).catch(() => {});
    }

    // Bind click handlers so pressing "Backtesting" on the page wires streaming into the log panel
    function bindBacktestingTriggers() {
        const tryBind = () => {
            const candidates = [
                '#backtesting-btn', '#start-backtesting', '#run-backtest', '#run-backtesting',
                'button[data-action="backtesting"]', 'a[data-action="backtesting"]'
            ];
            const els = candidates.flatMap(sel => Array.from(document.querySelectorAll(sel)));
            const textMatches = Array.from(document.querySelectorAll('button, a, .btn'))
                .filter(el => /backtest|backtesting/i.test((el.textContent || '').trim()));
            const all = Array.from(new Set([...els, ...textMatches]));
            all.forEach(el => {
                if (el.__backtestingBound) return;
                el.addEventListener('click', () => {
                    // Prepare the panel
                    const fb = findTrainingPanelFallback();
                    const logsEl = fb.logsEl || byIds(['backtesting-logs','training-logs','trainingLogs','ml-training-logs']);
                    const statusEl = fb.statusEl || byIds(['backtesting-status','training-status','trainingStatus','ml-training-status']);
                    if (logsEl) {
                        // Clear and set status
                        if (logsEl.tagName === 'PRE' || logsEl.tagName === 'TEXTAREA' || logsEl.tagName === 'CODE') {
                            logsEl.textContent = '';
                        } else {
                            const pre = logsEl.querySelector('pre');
                            if (pre) pre.textContent = ''; else logsEl.innerHTML = '';
                        }
                        if (statusEl) statusEl.textContent = 'Starting backtesting...';
                    }
                    // Reinitialize streaming
                    initBacktestingLogsIntegration();
                });
                el.__backtestingBound = true;
            });
        };
        // Initial bind and observe for dynamic buttons
        tryBind();
        const mo = new MutationObserver(() => tryBind());
        mo.observe(document.body, { childList: true, subtree: true });
    }

    function startBacktestingSSE(logsEl, statusEl, limitSel) {
        let es;
        let reconnectDelay = 1000; // start at 1s, backoff up to 30s
        const maxDelay = 30000;

        const connect = () => {
            try {
                if (statusEl) statusEl.textContent = 'Connecting to backtesting log stream...';
                es = new EventSource('/api/backtesting/logs/stream');

                es.onopen = () => {
                    if (statusEl) statusEl.textContent = 'Connected to backtesting log stream';
                    reconnectDelay = 1000; // reset backoff on success
                };

                es.onmessage = (ev) => {
                    appendBacktestingLogLine(ev.data, logsEl);
                };

                es.addEventListener('heartbeat', (ev) => {
                    // Optional: update UI on heartbeat if needed
                });

                es.addEventListener('completed', () => {
                    if (statusEl) statusEl.textContent = 'Backtesting run completed';
                    // Do a final poll to capture any missed entries
                    const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
                    pollBacktestingLogs(limit, logsEl, statusEl).catch(() => {});
                });

                es.onerror = () => {
                    if (statusEl) statusEl.textContent = 'Stream error, falling back to polling';
                    try { es.close(); } catch (_) {}
                    // Fallback to polling immediately, then try to reconnect with backoff
                    const limit = limitSel ? parseInt(limitSel.value || '200', 10) : 200;
                    pollBacktestingLogs(limit, logsEl, statusEl).catch(() => {});
                    setTimeout(connect, reconnectDelay);
                    reconnectDelay = Math.min(reconnectDelay * 2, maxDelay);
                };
            } catch (e) {
                if (statusEl) statusEl.textContent = 'Failed to open log stream';
            }
        };

        connect();
    }

    async function pollBacktestingLogs(limit, logsEl, statusEl) {
        try {
            if (statusEl) statusEl.textContent = 'Loading backtesting logs...';
            const resp = await fetch(`/api/backtesting/logs?limit=${encodeURIComponent(limit)}`);
            const data = await resp.json();
            if (!data || data.success === false) {
                throw new Error((data && (data.error || data.message)) || 'Unknown error');
            }
            if (statusEl) {
                const parts = [];
                parts.push(data.running ? 'Running' : (data.completed ? 'Completed' : 'Idle'));
                if (typeof data.progress === 'number') parts.push(`Progress: ${Math.round(data.progress * 100)}%`);
                if (data.current_task) parts.push(`Task: ${data.current_task}`);
                statusEl.textContent = parts.join(' • ');
            }
            renderBacktestingLogs(data.logs || [], logsEl);
        } catch (err) {
            if (statusEl) statusEl.textContent = `Failed to load logs: ${err.message}`;
            console.error('Backtesting logs fetch error:', err);
        }
    }

    function renderBacktestingLogs(entries, logsEl) {
        if (!Array.isArray(entries)) entries = [];
        const isTrainingPanel = logsEl.id === 'training-logs';
        if (isTrainingPanel) {
            logsEl.innerHTML = '';
            entries.forEach(entry =e {
                const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
                const level = entry.level || 'INFO';
                const levelClass = level === 'ERROR' ? 'text-danger' : level === 'WARNING' ? 'text-warning' : 'text-info';
                const div = document.createElement('div');
                div.className = 'mb-1';
                div.innerHTML = `\u003cspan class=\"text-muted\"\u003e[${ts}]\u003c/span\u003e \u003cspan class=\"${levelClass}\"\u003e${entry.message || ''}\u003c/span\u003e`;
                logsEl.appendChild(div);
            });
        } else {
            // Render as simple preformatted lines for efficiency
            const lines = entries.map(sanitizeBacktestingEntryToLine).join('\n');
            if (logsEl.tagName === 'PRE' || logsEl.tagName === 'TEXTAREA' || logsEl.tagName === 'CODE') {
                logsEl.textContent = lines;
            } else {
                logsEl.innerHTML = '';
                const pre = document.createElement('pre');
                pre.className = 'mb-0 small';
                pre.textContent = lines;
                logsEl.appendChild(pre);
            }
        }
        // Auto-scroll to bottom
        if (typeof logsEl.scrollTop === 'number') {
            logsEl.scrollTop = logsEl.scrollHeight;
        } else if (logsEl.firstChild e typeof logsEl.firstChild.scrollTop === 'number') {
            logsEl.firstChild.scrollTop = logsEl.firstChild.scrollHeight;
        }
    }

    function appendBacktestingLogLine(raw, logsEl) {
        // Try to parse JSON line from SSE if applicable; otherwise treat as text
        let msg = raw;
        try {
            const maybe = JSON.parse(raw);
            if (maybe e typeof maybe === 'object') msg = maybe;
        } catch(_) {}
        const isTrainingPanel = logsEl.id === 'training-logs';
        if (isTrainingPanel) {
            const entry = typeof msg === 'string' ? { message: msg } : msg;
            const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
            const level = entry.level || 'INFO';
            const levelClass = level === 'ERROR' ? 'text-danger' : 'text-info';
            const div = document.createElement('div');
            div.className = 'mb-1';
            div.innerHTML = `\u003cspan class=\"text-muted\"\u003e[${ts}]\u003c/span\u003e \u003cspan class=\"${levelClass}\"\u003e${entry.message || ''}\u003c/span\u003e`;
            logsEl.appendChild(div);
        } else {
            const line = typeof msg === 'string' ? msg : sanitizeBacktestingEntryToLine(msg);
            if (logsEl.tagName === 'PRE' || logsEl.tagName === 'TEXTAREA' || logsEl.tagName === 'CODE') {
                logsEl.textContent += (logsEl.textContent ? '\n' : '') + line;
            } else {
                let pre = logsEl.querySelector('pre');
                if (!pre) { pre = document.createElement('pre'); pre.className = 'mb-0 small'; logsEl.appendChild(pre); }
                pre.textContent += (pre.textContent ? '\n' : '') + line;
            }
        }
        if (typeof logsEl.scrollTop === 'number') logsEl.scrollTop = logsEl.scrollHeight;
    }

    function sanitizeBacktestingEntryToLine(entry) {
        // entry is already sanitized by backend; ensure robust string formatting
        try {
            const ts = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : new Date().toLocaleString();
            const lvl = entry.level || (entry.severity || 'INFO');
            const msg = entry.message || entry.msg || entry.text || '';
            // Include extra data keys compactly, excluding common ones
            const extra = Object.keys(entry)
                .filter(k => !['timestamp','level','message','msg','text'].includes(k))
                .reduce((acc, k) => { acc[k] = entry[k]; return acc; }, {});
            const extraStr = Object.keys(extra).length ? ` | ${JSON.stringify(extra)}` : '';
            return `[${ts}] ${lvl}: ${msg}${extraStr}`;
        } catch (_) {
            try { return typeof entry === 'string' ? entry : JSON.stringify(entry); } catch { return String(entry); }
        }
    }

    // Expose functions globally for template access
    window.generateAdvisoryForPrediction = generateAdvisoryForPrediction;
    window.generateAdvisoryManually = generateAdvisoryManually;
    window.initBacktestingLogsIntegration = initBacktestingLogsIntegration;

    init();
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', run);
    } else {
        run();
    }
})();

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
