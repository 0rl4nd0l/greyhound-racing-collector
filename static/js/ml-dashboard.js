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

// --- V4 Extras: Evaluation summary + Feature contracts UI widgets (compat file) ---
(function(){
  function el(tag, attrs = {}, children = []) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
    (Array.isArray(children) ? children : [children]).forEach(c => {
      if (c == null) return;
      if (typeof c === 'string') node.appendChild(document.createTextNode(c));
      else node.appendChild(c);
    });
    return node;
  }

  async function fetchJSON(url) {
    const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
    if (!res.ok) throw new Error(`Request failed: ${res.status}`);
    return await res.json();
  }

  async function renderEvalSummary(containerId = 'v4-eval-summary', win = 500) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    try {
      const data = await fetchJSON(`/api/v4/eval/summary/latest?window=${encodeURIComponent(String(win))}`);
      if (!data || data.success === false) throw new Error((data && (data.error || data.message)) || 'Unknown error');
      const pre = el('pre', { class: 'code' }, JSON.stringify(data.summary || data, null, 2));
      container.appendChild(pre);
    } catch (e) {
      container.appendChild(el('div', { class: 'error text-danger small' }, `Failed to load evaluation summary: ${e.message}`));
    }
  }

  async function renderContractsList(containerId = 'v4-contracts-list') {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    try {
      const data = await fetchJSON('/api/v4/models/contracts');
      if (!data || data.success === false) throw new Error((data && (data.error || data.message)) || 'Unknown error');
      const ul = el('ul', { class: 'list-unstyled mb-0' });
      const contracts = Array.isArray(data.contracts) ? data.contracts : [];
      if (contracts.length === 0) {
        container.appendChild(el('div', { class: 'text-muted' }, 'No contracts found. Train a model to generate one.'));
      } else {
        contracts.forEach(c => {
          const li = el('li', { class: 'mb-1' });
          const link = el('a', { href: '#', 'data-name': c.name }, c.name + (c.modified ? ` (updated ${c.modified})` : ''));
          link.addEventListener('click', async (ev) => {
            ev.preventDefault();
            await renderContractDetail('v4-contract-detail', c.name);
          });
          li.appendChild(link);
          ul.appendChild(li);
        });
        container.appendChild(ul);
      }
    } catch (e) {
      container.appendChild(el('div', { class: 'error text-danger small' }, `Failed to list contracts: ${e.message}`));
    }
  }

  async function renderContractDetail(containerId = 'v4-contract-detail', contractName = 'v4_feature_contract.json') {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    try {
      const data = await fetchJSON(`/api/v4/models/contracts/${encodeURIComponent(contractName)}`);
      if (!data || data.success === false) throw new Error((data && (data.error || data.message)) || 'Unknown error');
      const pre = el('pre', { class: 'code' }, JSON.stringify(data.contract || data, null, 2));
      container.appendChild(pre);
    } catch (e) {
      container.appendChild(el('div', { class: 'error text-danger small' }, `Failed to load contract: ${e.message}`));
    }
  }

  function bindRefreshContract(buttonId = 'refresh-v4-contract') {
    const btn = document.getElementById(buttonId);
    if (!btn) return;
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      const orig = btn.textContent;
      btn.textContent = 'Refreshing...';
      try {
        const res = await fetch('/api/v4/models/contracts/refresh', { method: 'POST' });
        const data = await res.json();
        if (!data || data.success === false) throw new Error((data && (data.error || data.message)) || 'Unknown error');
        await renderContractsList('v4-contracts-list');
        await renderContractDetail('v4-contract-detail', 'v4_feature_contract.json');
      } catch (e) {
        alert('Failed to refresh contract: ' + e.message);
      } finally {
        btn.disabled = false;
        btn.textContent = orig || 'Refresh Contract';
      }
    });
  }

  // Render validation result into a container
  function renderContractValidation(containerId, payload, statusCode) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';
    if (!payload) { container.textContent = 'No response'; return; }

    if (payload.error) {
      container.innerHTML = `<div class="text-danger">${payload.error}</div>`;
      return;
    }

    const { matched, strict, diff, path } = payload;
    const hdr = document.createElement('div');
    hdr.className = 'mb-2';
    const statusText = matched ? (strict ? 'Matched (Strict)' : 'Matched') : (strict ? 'Mismatch (Strict)' : 'Mismatched');
    hdr.innerHTML = `<strong>Status:</strong> ${statusText} ${path ? `<small class=\"text-muted\">(${path})</small>` : ''}`;
    container.appendChild(hdr);

    if (!diff) return;

    const makeList = (title, arr) => {
      const wrap = document.createElement('div');
      wrap.className = 'mb-2';
      const h = document.createElement('div');
      h.innerHTML = `<strong>${title}</strong>`;
      wrap.appendChild(h);
      if (!arr || arr.length === 0) {
        const p = document.createElement('div'); p.className = 'text-muted small'; p.textContent = 'None'; wrap.appendChild(p);
      } else {
        const ul = document.createElement('ul'); ul.className = 'small mb-0';
        arr.forEach(v => { const li = document.createElement('li'); li.textContent = v; ul.appendChild(li); });
        wrap.appendChild(ul);
      }
      return wrap;
    };

    const blocks = [];
    try {
      blocks.push(makeList('Categorical - Missing', (diff.categorical && diff.categorical.missing) || []));
      blocks.push(makeList('Categorical - Extra', (diff.categorical && diff.categorical.extra) || []));
      blocks.push(makeList('Numerical - Missing', (diff.numerical && diff.numerical.missing) || []));
      blocks.push(makeList('Numerical - Extra', (diff.numerical && diff.numerical.extra) || []));
    } catch(_) {}

    const sig = document.createElement('div');
    sig.className = 'mb-2';
    const sMatch = diff.signature_match === true;
    sig.innerHTML = `<strong>Signature:</strong> ${sMatch ? 'Match' : 'Mismatch'}${diff.expected_signature ? `<br><small class=\"text-muted\">expected: ${diff.expected_signature}</small>` : ''}${diff.current_signature ? `<br><small class=\"text-muted\">current: ${diff.current_signature}</small>` : ''}`;

    container.appendChild(sig);
    blocks.forEach(b => container.appendChild(b));

    // Footer note
    const note = document.createElement('div'); note.className = 'mt-2 text-muted small';
    note.textContent = statusCode === 409 ? 'Strict check returned 409 due to mismatch.' : 'OK';
    container.appendChild(note);
  }

  function updateContractStatusBadge(badgeId, matched, strict, statusCode) {
    const badge = document.getElementById(badgeId);
    if (!badge) return;
    badge.className = 'badge';
    if (matched === true) {
      badge.classList.add('bg-success');
      badge.textContent = strict ? 'Stable (strict)' : 'Stable';
    } else if (statusCode === 409 || strict) {
      badge.classList.add('bg-danger');
      badge.textContent = 'Mismatch';
    } else {
      badge.classList.add('bg-warning');
      badge.textContent = 'Diffs';
    }
  }

  async function runContractCheck({ strict = false, containerId = 'v4-contract-validation-result', badgeId = 'v4-contract-status-badge' } = {}) {
    try {
      const url = strict ? '/api/v4/models/contracts/check?strict=1' : '/api/v4/models/contracts/check';
      const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
      const body = await res.json();
      renderContractValidation(containerId, body, res.status);
      if (typeof body.matched === 'boolean') updateContractStatusBadge(badgeId, body.matched, !!body.strict, res.status);
    } catch (e) {
      renderContractValidation('v4-contract-validation-result', { error: e.message }, 0);
      updateContractStatusBadge('v4-contract-status-badge', false, false, 0);
    }
  }

  function bindValidateContract(buttonId = 'validate-v4-contract', strictToggleId = 'validate-v4-strict', containerId = 'v4-contract-validation-result', badgeId = 'v4-contract-status-badge') {
    const btn = document.getElementById(buttonId);
    const chk = document.getElementById(strictToggleId);
    if (!btn) return;
    btn.addEventListener('click', async () => {
      btn.disabled = true; const orig = btn.textContent; btn.textContent = 'Validating...';
      try {
        const strict = !!(chk && chk.checked);
        await runContractCheck({ strict, containerId, badgeId });
      } catch (e) {
        renderContractValidation(containerId, { error: e.message }, 0);
      } finally {
        btn.disabled = false; btn.textContent = orig || 'Validate';
      }
    });
  }

  function autoInit() {
    const any = document.getElementById('v4-eval-summary') || document.getElementById('v4-contracts-list') || document.getElementById('v4-contract-detail') || document.getElementById('refresh-v4-contract');
    if (!any) return;
    renderEvalSummary('v4-eval-summary', 500);
    renderContractsList('v4-contracts-list');
    renderContractDetail('v4-contract-detail', 'v4_feature_contract.json');
    bindRefreshContract('refresh-v4-contract');
    bindValidateContract('validate-v4-contract', 'validate-v4-strict', 'v4-contract-validation-result', 'v4-contract-status-badge');
    // Initial non-strict check to populate badge/status
    runContractCheck({ strict: false, containerId: 'v4-contract-validation-result', badgeId: 'v4-contract-status-badge' });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
  } else {
    autoInit();
  }
})();
