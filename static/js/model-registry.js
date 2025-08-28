// Model Registry Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeModelRegistry();

    // Setup event listeners
    const startBtn = document.getElementById('start-training-btn');
    if (startBtn) startBtn.addEventListener('click', startModelTraining);

    const refreshBtn = document.getElementById('refresh-best-btn');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshBestModel);

    // Refresh data and panels every 30 seconds
    setInterval(() => {
        refreshModelRegistry();
        loadPromotionPanels();
    }, 30000);
});

function initializeModelRegistry() {
    refreshModelRegistry();
    loadPromotionPanels();
}

function refreshModelRegistry() {
    // Load registry status
    fetch('/api/model_registry/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateBestModels(data.best_models || {});
                updateRegistryStatus({
                    total_models: data.model_count ?? (data.all_models?.length || 0),
                    best_models: data.best_models || {}
                });
                if (Array.isArray(data.all_models)) {
                    updateModelsTable(data.all_models);
                }
            }
        })
        .catch(error => {
            console.error('Error fetching registry status:', error);
            showNotification('Error loading model registry data', 'error');
        });
}

function updateBestModels(bestModels) {
    const container = document.getElementById('best-models-container');
    if (!container) return;
    let html = '';

    for (const [predType, model] of Object.entries(bestModels)) {
        const perf = (model.performance_score != null) ? `${(model.performance_score*100).toFixed(1)}%` : 'N/A';
        const created = model.created_at ? formatDate(model.created_at) : 'N/A';
        html += `
            <div class="mb-3">
                <h6>${formatPredictionType(predType)}</h6>
                <div class="card bg-light">
                    <div class="card-body p-2">
                        <small class="text-muted">Model ID:</small> ${escapeHtml(model.model_id)}<br>
                        <small class="text-muted">Version:</small> ${escapeHtml(model.version)}<br>
                        <small class="text-muted">Score:</small> ${perf}<br>
                        <small class="text-muted">Created:</small> ${created}
                    </div>
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function updateRegistryStatus(data) {
    const container = document.getElementById('registry-status-container');
    if (!container) return;

    const html = `
        <div class="d-flex justify-content-between mb-3">
            <div>
                <h6 class="mb-2">Registry Status</h6>
                <span class="badge bg-success">Active</span>
            </div>
            <div class="text-end">
                <button class="btn btn-primary btn-sm" onclick="showTrainingModal()">
                    Train New Model
                </button>
            </div>
        </div>
        <div class="row g-2">
            <div class="col-6">
                <div class="border rounded p-2">
                    <small class="text-muted d-block">Total Models</small>
                    <strong>${data.total_models ?? '—'}</strong>
                </div>
            </div>
            <div class="col-6">
                <div class="border rounded p-2">
                    <small class="text-muted d-block">Active Models</small>
                    <strong>${Object.keys(data.best_models || {}).length}</strong>
                </div>
            </div>
        </div>
    `;

    container.innerHTML = html;
}

function updateModelsTable(models) {
    const tbody = document.getElementById('models-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    models.forEach(model => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${escapeHtml(model.model_name || model.model_id || 'N/A')}</td>
            <td>${escapeHtml(formatPredictionType(model.prediction_type || 'unknown'))}</td>
            <td>${model.version ? 'v' + escapeHtml(model.version) : '—'}</td>
            <td>${model.performance_score != null ? (model.performance_score*100).toFixed(1) + '%' : '—'}</td>
            <td>${model.created_at ? formatDate(model.created_at) : '—'}</td>
            <td>
                <span class="badge bg-${model.is_active ? 'success' : 'secondary'}">
                    ${model.is_active ? 'Active' : 'Inactive'}
                </span>
            </td>
            <td>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary" onclick="showModelDetails('${escapeAttr(model.model_id)}')">
                        Details
                    </button>
                    <button class="btn btn-outline-secondary" onclick="downloadModel('${escapeAttr(model.model_id)}')">
                        Download
                    </button>
                </div>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

function loadTrainableModels() {
    fetch('/api/model_registry/models')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const select = document.getElementById('train-model-select');
                if (!select) return;
                select.innerHTML = (data.models || []).map(model => 
                    `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.name || model.model_id)}</option>`
                ).join('');
            } else {
                showNotification('Error loading trainable models', 'error');
            }
        })
        .catch(error => {
            console.error('Error fetching trainable models:', error);
            showNotification('Error loading trainable models', 'error');
        });
}

function showTrainingModal() {
    loadTrainableModels();
    const modalEl = document.getElementById('trainingModal');
    if (!modalEl) return;
    const modal = new bootstrap.Modal(modalEl);
    modal.show();
}

function startModelTraining() {
    const form = document.getElementById('training-form');
    if (!form) return;
    const formData = new FormData(form);

    const data = {
        prediction_type: formData.get('prediction_type'),
        training_data_days: parseInt(formData.get('training_data_days')),
        force_retrain: formData.get('force_retrain') === 'on',
        model_id: formData.get('model_id')
    };

    fetch('/api/model_registry/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Model training initiated successfully', 'success');
            const modalEl = document.getElementById('trainingModal');
            if (modalEl) bootstrap.Modal.getInstance(modalEl)?.hide();
            pollTrainingStatus(data.job_id);
        } else {
            showNotification(`Training error: ${data.error}`, 'error');
        }
    })
    .catch(error => {
        console.error('Error starting training:', error);
        showNotification('Error starting model training', 'error');
    });
}

function pollTrainingStatus(jobId) {
    const intervalId = setInterval(() => {
        fetch(`/api/model_registry/status?job_id=${encodeURIComponent(jobId)}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (data.status === 'completed') {
                        clearInterval(intervalId);
                        showNotification('Training completed successfully', 'success');
                        refreshModelRegistry();
                    } else {
                        updateTrainingProgress(data);
                    }
                } else {
                    showNotification('Error polling training status', 'error');
                }
            })
            .catch(error => {
                console.error('Error polling training status:', error);
                showNotification('Error polling training status', 'error');
            });
    }, 3000);

    function updateTrainingProgress(data) {
        const barWrap = document.getElementById('training-progress');
        const progressBar = document.querySelector('#training-progress .progress-bar');
        if (barWrap) barWrap.classList.remove('d-none');
        if (progressBar) {
            const p = Number(data.progress || 0);
            progressBar.style.width = `${p}%`;
            progressBar.textContent = `${p}%`;
        }
    }
}

async function showModelDetails(modelId) {
    try {
        const [performanceResponse, monitoringResponse] = await Promise.all([
            fetch(`/api/model/performance?model_id=${encodeURIComponent(modelId)}`),
            fetch(`/api/model/monitoring/drift`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            })
        ]);

        const performanceData = await performanceResponse.json();
        const monitoringData = await monitoringResponse.json();
        console.log(performanceData, monitoringData);
    } catch (error) {
        console.error('Error fetching model details:', error);
        showNotification('Error loading model details', 'error');
    }
}

function downloadModel(modelId) {
    window.location.href = `/api/model/download/${encodeURIComponent(modelId)}`;
}

// Promotion panels
async function loadPromotionPanels(){
    try{
        const now = new Date();
        const [sigRes, lastRes, statusRes] = await Promise.all([
            fetch('/api/model_registry/refresh_signal', { cache:'no-store' }),
            fetch('/api/diagnostics/last_promotion', { cache:'no-store' }),
            fetch('/api/model_registry/status', { cache:'no-store' })
        ]);
        const [sig, last, status] = await Promise.all([
            sigRes.json().catch(()=>({})),
            lastRes.json().catch(()=>({})),
            statusRes.json().catch(()=>({}))
        ]);
        renderCurrentBestPanel(sig, status, now);
        renderLastPromotionPanel(last, now);
    }catch(err){
        showNotification('Failed to load promotion panels', 'error');
    }
}

function renderCurrentBestPanel(signalData, statusData, refreshedAt){
    const body = document.getElementById('current-best-body');
    if (!body) return;
    const lastRef = refreshedAt ? new Date(refreshedAt).toLocaleTimeString() : new Date().toLocaleTimeString();
    if (!signalData || !signalData.exists || !signalData.signal){
        body.innerHTML = '<p class="text-muted mb-1">No promotion signal available yet.</p>' +
                         `<div class="text-muted small">Last refresh: ${escapeHtml(lastRef)}</div>`;
        return;
    }
    const s = signalData.signal || {};
    const m = s.best_metadata || {};
    const acc = (m.accuracy != null) ? (m.accuracy*100).toFixed(2) + '%' : '—';
    const auc = (m.auc != null) ? Number(m.auc).toFixed(3) : '—';
    const top1 = (m.top1_rate != null) ? (m.top1_rate*100).toFixed(2) + '%' : '—';

    const sync = computeSyncStatus(signalData, statusData);
    const badge = sync.inSync
        ? `<span class="badge bg-success">In Sync</span>`
        : `<span class="badge bg-warning text-dark">Out of Sync</span>`;
    const syncDetail = (!sync.inSync && (sync.registryBestId || sync.signalId))
        ? `<div class="small text-muted">Registry: ${escapeHtml(formatShortId(sync.registryBestId || '—'))} vs Signal: ${escapeHtml(formatShortId(sync.signalId || '—'))}</div>`
        : '';
    const badgeHelp = `<span class="ms-2 small text-muted" title="In Sync means the in-memory registry best model matches the latest broadcast signal. Out of Sync means parts of the app may still be refreshing; it should resolve automatically shortly.">?</span>`;

    body.innerHTML = `
      <div class="d-flex justify-content-between align-items-center mb-2">
        <div>${badge}${badgeHelp}${syncDetail ? '<span class="ms-2"></span>' : ''}</div>
        <div class="small text-muted">Last refresh: ${escapeHtml(lastRef)}</div>
      </div>`
      <div class="row g-2">
        <div class="col-12"><small class="text-muted">Model ID</small><div class="fw-semibold">${escapeHtml(s.promoted_model_id || '—')}</div></div>
        <div class="col-6"><small class="text-muted">Model Name</small><div>${escapeHtml(m.model_name || '—')}</div></div>
        <div class="col-6"><small class="text-muted">Type</small><div>${escapeHtml(m.model_type || '—')}</div></div>
        <div class="col-4"><small class="text-muted">Accuracy</small><div>${acc}</div></div>
        <div class="col-4"><small class="text-muted">AUC</small><div>${auc}</div></div>
        <div class="col-4"><small class="text-muted">Top-1</small><div>${top1}</div></div>
        <div class="col-6"><small class="text-muted">Policy</small><div>${escapeHtml(s.selection_policy || 'correct_winners')}</div></div>
        <div class="col-6"><small class="text-muted">Prediction</small><div><span class="badge bg-info">${escapeHtml(s.prediction_type || m.prediction_type || 'win')}</span></div></div>
        <div class=\"col-12\"><small class=\"text-muted\">Updated</small><div>${escapeHtml(s.timestamp || '—')}</div></div>\n      </div>\n      <div class=\"mt-2 small text-muted\">Legend: <span class=\"badge bg-success\">In Sync</span> registry equals broadcast; <span class=\"badge bg-warning text-dark\">Out of Sync</span> registry update pending.</div>\n    `;
      </div>
    `;
}

function renderLastPromotionPanel(data, refreshedAt){
    const body = document.getElementById('last-promotion-body');
    if (!body) return;
    const lastRef = refreshedAt ? new Date(refreshedAt).toLocaleTimeString() : new Date().toLocaleTimeString();
    if (!data || data.success !== true || data.found !== true){
        body.innerHTML = '<p class="text-muted mb-1">No promotion record found.</p>' +
                         `<div class="text-muted small">Last refresh: ${escapeHtml(lastRef)}</div>`;
        return;
    }
    const e = data.entry || {};
    const brier = e.brier_score != null ? Number(e.brier_score).toFixed(4) : '—';
    const slope = e.reliability_slope != null ? Number(e.reliability_slope).toFixed(3) : '—';

    body.innerHTML = `
      <div class="d-flex justify-content-end mb-2">
        <div class="small text-muted">Last refresh: ${escapeHtml(lastRef)}</div>
      </div>
      <div class="row g-2">
        <div class="col-6"><small class="text-muted">Status</small><div>${e.success ? '<span class="badge bg-success">Promoted</span>' : '—'}</div></div>
        <div class="col-6"><small class="text-muted">Time</small><div>${escapeHtml(e.timestamp || '—')}</div></div>
        <div class="col-12"><small class="text-muted">Message</small><div>${escapeHtml(e.message || '—')}</div></div>
        <div class="col-6"><small class="text-muted">Brier</small><div>${brier}</div></div>
        <div class="col-6"><small class="text-muted">Reliability Slope</small><div>${slope}</div></div>
        ${e.artifact_path ? `<div class="col-12"><small class="text-muted">Artifact</small><div class="text-truncate">${escapeHtml(e.artifact_path)}</div></div>` : ''}
      </div>
    `;
}

async function refreshBestModel(){
    const btn = document.getElementById('refresh-best-btn');
    if (btn){ btn.disabled = true; btn.innerText = 'Refreshing...'; }
    try{
        const res = await fetch('/api/model_registry/refresh_best', { method:'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        const data = await res.json().catch(()=>({}));
        if(res.ok && data && data.success){
            showNotification('Best model refreshed: ' + (data.promoted_model_id || 'OK'), 'success');
            await loadPromotionPanels();
            refreshModelRegistry();
        } else {
            showNotification('Refresh failed: ' + (data?.error || res.status), 'error');
        }
    }catch(err){
        showNotification('Refresh error: ' + (err?.message || err), 'error');
    }finally{
        if (btn){ btn.disabled = false; btn.innerHTML = '<i class="fas fa-bolt"></i> Refresh Best'; }
    }
}

function _extractModelId(entry){
    if (!entry) return null;
    if (typeof entry === 'object'){
        return entry.model_id || entry.modelId || entry.id || null;
    }
    if (typeof entry === 'string'){
        // Attempt to parse Python repr like: "ModelMetadata(model_id='XYZ', ...)"
        const m = entry.match(/model_id=['\"]([^'\"]+)['\"]/);
        if (m && m[1]) return m[1];
        // Fallback: try to find something that looks like an ID token
        const idLike = entry.match(/[A-Za-z0-9_\-]{8,}/);
        return idLike ? idLike[0] : null;
    }
    return null;
}

function computeSyncStatus(signalData, statusData){
    try{
        const s = signalData && signalData.signal ? signalData.signal : {};
        const signalId = s.promoted_model_id || s.model_id || null;
        let registryBestId = null;
        const bm = (statusData && (statusData.best_models || statusData.best_model)) ? (statusData.best_models || { _single: statusData.best_model }) : null;
        if (bm){
            const keys = Object.keys(bm || {});
            const winKey = keys.find(k => String(k).toLowerCase() === 'win');
            const entry = winKey ? bm[winKey] : bm[keys[0]];
            registryBestId = _extractModelId(entry);
        }
        if (!registryBestId || !signalId){
            return { inSync: false, registryBestId, signalId };
        }
        return { inSync: String(registryBestId) === String(signalId), registryBestId, signalId };
    } catch (e){
        return { inSync: false, registryBestId: null, signalId: null };
    }
}

function formatShortId(id, n = 8){
    const s = String(id || '');
    if (s.length <= n) return s;
    return s.slice(0, Math.floor(n/2)) + '…' + s.slice(-Math.ceil(n/2));
}

// Utility functions
function formatPredictionType(type) {
    if (!type) return 'Unknown';
    return String(type)
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function formatScore(score) {
    if (score == null) return '—';
    return (score * 100).toFixed(1) + '%';
}

function formatDate(dateString) {
    try { return new Date(dateString).toLocaleString(); } catch { return String(dateString || '—'); }
}

function showNotification(message, type = 'info') {
    const toastEl = document.getElementById('training-toast');
    if (!toastEl){ console.log(`${type.toUpperCase()}: ${message}`); return; }
    const body = toastEl.querySelector('.toast-body');
    const header = toastEl.querySelector('.toast-header .me-auto');
    if (header) header.textContent = 'Model Registry';
    if (body) body.textContent = String(message || '');
    try {
        const toast = bootstrap.Toast.getOrCreateInstance(toastEl);
        toast.show();
    } catch (e) {
        console.log(`${type.toUpperCase()}: ${message}`);
    }
}

function escapeHtml(s){ return String(s ?? '').replace(/[&<>"]+/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch])); }
function escapeAttr(s){ return String(s ?? '').replace(/"/g, '&quot;'); }
