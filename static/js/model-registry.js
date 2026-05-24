// Model Registry Dashboard JavaScript

(function initModelRegistryUI(){
    const runner = () => {
        initializeModelRegistry();

        // Setup event listeners
        const startBtn = document.getElementById('start-training-btn');
        if (startBtn && !startBtn.dataset._bound) {
            startBtn.addEventListener('click', startModelTraining);
            startBtn.dataset._bound = '1';
        }

        const refreshBtn = document.getElementById('refresh-best-btn');
        if (refreshBtn && !refreshBtn.dataset._bound) {
            refreshBtn.addEventListener('click', refreshBestModel);
            refreshBtn.dataset._bound = '1';
        }

        // Refresh data and panels every 30 seconds
        if (!window.__modelRegistryRefreshInterval) {
            window.__modelRegistryRefreshInterval = setInterval(() => {
                try { refreshModelRegistry(); } catch (e) { /* noop */ }
                try { loadPromotionPanels(); } catch (e) { /* noop */ }
            }, 30000);
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runner, { once: true });
    } else {
        runner();
    }
})();

function initializeModelRegistry() {
    refreshModelRegistry();
    loadPromotionPanels();
}

function refreshModelRegistry() {
    // Load registry status (use registry-backed endpoint)
    fetch('/api/model/registry/status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                updateBestModels(data.best_models || {});
                const total = (data.total_models ?? data.model_count ?? (Array.isArray(data.all_models) ? data.all_models.length : 0));
                updateRegistryStatus({
                    total_models: total,
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
    // Use registry-backed list of trainable models
    fetch('/api/model/list_trainable')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const select = document.getElementById('train-model-select');
                if (!select) return;
                select.innerHTML = (data.models || []).map(model => 
                    `<option value="${escapeAttr(model.model_id)}">${escapeHtml(model.model_name || model.name || model.model_id)}</option>`
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

    // Use registry-backed training trigger
    fetch('/api/model/training/trigger', {
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
        // Poll registry-backed job status
        fetch(`/api/model/registry/status?job_id=${encodeURIComponent(jobId)}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    if (data.status === 'completed') {
                        clearInterval(intervalId);
                        showNotification('Training completed successfully', 'success');
                        refreshModelRegistry();
                    } else if (data.status === 'failed') {
                        clearInterval(intervalId);
                        showNotification(`Training failed: ${data.error_message || 'Unknown error'}`, 'error');
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
        const modalEl = ensureDetailsModal();
        setDetailsModalContent('Loading model details…', '<div class="text-muted">Fetching performance and monitoring data…</div>');

        const [perfRes, driftRes, detailsRes] = await Promise.all([
            fetch(`/api/model/performance?model_id=${encodeURIComponent(modelId)}`),
            fetch(`/api/model/monitoring/drift`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_id: modelId })
            }),
            fetch(`/api/model/details?model_id=${encodeURIComponent(modelId)}`)
        ]);

        const performanceData = await perfRes.json().catch(() => ({}));
        const monitoringData = await driftRes.json().catch(() => ({}));
        const detailsJson = await detailsRes.json().catch(() => ({}));
        const details = (detailsJson && (detailsJson.details || detailsJson.model || detailsJson)) || {};

        renderDetailsModal(performanceData, monitoringData, modelId, details);

        try {
            const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
            modal.show();
        } catch (e) {
            // Fallback if Bootstrap JS is unavailable
            modalEl.classList.add('show');
            modalEl.style.display = 'block';
        }
    } catch (error) {
        console.error('Error fetching model details:', error);
        showNotification('Error loading model details', 'error');
    }
}

function ensureDetailsModal() {
    let modalEl = document.getElementById('model-details-modal');
    if (modalEl) return modalEl;

    const tpl = document.createElement('div');
    tpl.innerHTML = `
    <div class="modal fade" id="model-details-modal" tabindex="-1" aria-hidden="true">
      <div class="modal-dialog modal-lg modal-dialog-scrollable">
        <div class="modal-content">
          <div class="modal-header">
            <h5 class="modal-title" id="model-details-title">Model Details</h5>
            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body" id="model-details-body">
            <div class="text-muted">Loading…</div>
          </div>
          <div class="modal-footer">
            <a id="model-details-download" class="btn btn-outline-secondary" href="#" download>
              <i class="fas fa-download"></i> Download
            </a>
            <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Close</button>
          </div>
        </div>
      </div>
    </div>`;
    modalEl = tpl.firstElementChild;
    document.body.appendChild(modalEl);

    // Fallback close handling when Bootstrap JS isn't present
    modalEl.addEventListener('click', (ev) => {
        const target = ev.target;
        if (!target) return;
        if (target.classList && target.classList.contains('modal')) {
            modalEl.classList.remove('show');
            modalEl.style.display = 'none';
        }
    }, { passive: true });

    return modalEl;
}

function setDetailsModalContent(title, html) {
    const titleEl = document.getElementById('model-details-title');
    const bodyEl = document.getElementById('model-details-body');
    if (titleEl) titleEl.textContent = String(title || 'Model Details');
    if (bodyEl) bodyEl.innerHTML = String(html || '');
}

function renderDetailsModal(performanceData, monitoringData, modelId, details) {
    const title = `Model Details — ${escapeHtml(String(modelId || ''))}`;

    // Prefer registry details for true per-model metrics; fallback to perf shim
    const perfShim = (performanceData && performanceData.performance_metrics) || {};
    const accVal = (details && details.accuracy != null) ? Number(details.accuracy) : (perfShim.accuracy != null ? Number(perfShim.accuracy) : null);
    const precVal = (details && details.precision != null) ? Number(details.precision) : (perfShim.precision != null ? Number(perfShim.precision) : null);
    const recallVal = (details && details.recall != null) ? Number(details.recall) : (perfShim.recall != null ? Number(perfShim.recall) : null);
    const f1Val = (details && details.f1_score != null) ? Number(details.f1_score) : (perfShim.f1_score != null ? Number(perfShim.f1_score) : null);
    const accuracy = accVal != null ? (accVal * 100).toFixed(2) + '%' : '—';
    const precision = precVal != null ? (precVal * 100).toFixed(2) + '%' : '—';
    const recall = recallVal != null ? (recallVal * 100).toFixed(2) + '%' : '—';
    const f1 = f1Val != null ? (f1Val * 100).toFixed(2) + '%' : '—';

    // Drift section
    const drift = (monitoringData && monitoringData.drift_results) || {};
    const driftScore = drift.drift_score != null ? String(drift.drift_score) : '—';
    const driftDetected = drift.drift_detected ? '<span class="badge bg-warning text-dark">Yes</span>' : '<span class="badge bg-success">No</span>';

    // Registry details
    const mId = String(modelId || details?.model_id || '—');
    const mName = details?.model_name || '—';
    const mType = details?.model_type || details?.prediction_type || '—';
    const predType = details?.prediction_type || '—';
    const created = details?.created_at ? formatDate(details.created_at) : '—';
    const featCount = (details && details.features_count != null) ? String(details.features_count) : '—';
    const trainSamples = (details && details.training_samples != null) ? String(details.training_samples) : '—';
    const modelSizeMb = (details && details.model_size_mb != null) ? (Number(details.model_size_mb).toFixed(2) + ' MB') : '—';
    const perfScore = (details && details.performance_score != null) ? (Number(details.performance_score) * 100).toFixed(2) + '%' : '—';
    const cw = (details && details.correct_winners != null) ? String(details.correct_winners) : '—';
    const re = (details && details.races_evaluated != null) ? String(details.races_evaluated) : '—';
    const t1 = (details && details.top1_rate != null) ? (Number(details.top1_rate) * 100).toFixed(2) + '%' : '—';
    const mSizeBytes = (details && details.model_file_size != null) ? humanBytes(details.model_file_size) : '—';
    const sSizeBytes = (details && details.scaler_file_size != null) ? humanBytes(details.scaler_file_size) : '—';

    const html = `
      <div class="container-fluid">
        <div class="row g-3">
          <div class="col-12 col-md-6">
            <div class="border rounded p-2">
              <h6 class="mb-2">Performance</h6>
              <div class="small text-muted">Snapshot</div>
              <ul class="mb-0">
                <li>Accuracy: <strong>${accuracy}</strong></li>
                <li>Precision: <strong>${precision}</strong></li>
                <li>Recall: <strong>${recall}</strong></li>
                <li>F1 Score: <strong>${f1}</strong></li>
              </ul>
            </div>
          </div>
          <div class="col-12 col-md-6">
            <div class="border rounded p-2">
              <h6 class="mb-2">Drift</h6>
              <div>Detected: ${driftDetected}</div>
              <div>Score: <strong>${escapeHtml(driftScore)}</strong></div>
            </div>
          </div>
        </div>

        <div class="row g-3 mt-1">
          <div class="col-12 col-md-6">
            <div class="border rounded p-2">
              <h6 class="mb-2">Model Metadata</h6>
              <div class="row g-2 small">
                <div class="col-6"><small class="text-muted">Model ID</small><div class="text-truncate">${escapeHtml(mId)}</div></div>
                <div class="col-6"><small class="text-muted">Name</small><div class="text-truncate">${escapeHtml(mName)}</div></div>
                <div class="col-6"><small class="text-muted">Type</small><div>${escapeHtml(mType)}</div></div>
                <div class="col-6"><small class="text-muted">Prediction</small><div><span class="badge bg-info">${escapeHtml(predType)}</span></div></div>
                <div class="col-6"><small class="text-muted">Created</small><div>${escapeHtml(created)}</div></div>
                <div class="col-6"><small class="text-muted">Performance Score</small><div>${escapeHtml(perfScore)}</div></div>
                <div class="col-6"><small class="text-muted">Features</small><div>${escapeHtml(featCount)}</div></div>
                <div class="col-6"><small class="text-muted">Training Samples</small><div>${escapeHtml(trainSamples)}</div></div>
                <div class="col-6"><small class="text-muted">Model Size</small><div>${escapeHtml(modelSizeMb)}</div></div>
                <div class="col-6"><small class="text-muted">Validation Method</small><div>${escapeHtml(details?.validation_method || '—')}</div></div>
                <div class="col-6"><small class="text-muted">Data Quality</small><div>${details?.data_quality_score!=null ? (Number(details.data_quality_score)*100).toFixed(1)+'%' : '—'}</div></div>
                <div class="col-6"><small class="text-muted">Ensemble</small><div>${details?.is_ensemble ? '<span class="badge bg-secondary">Yes</span>' : 'No'}</div></div>
                <div class="col-6"><small class="text-muted">Components</small><div>${Array.isArray(details?.ensemble_components)? details.ensemble_components.length : '—'}</div></div>
                <div class="col-6"><small class="text-muted">Inference Time</small><div>${details?.inference_time_ms!=null ? escapeHtml(String(details.inference_time_ms)+' ms') : '—'}</div></div>
              </div>
            </div>
          </div>
          <div class="col-12 col-md-6">
            <div class="border rounded p-2">
              <h6 class="mb-2">Evaluation</h6>
              <div class="row g-2 small">
                <div class="col-6"><small class="text-muted">Top-1 Rate</small><div>${escapeHtml(t1)}</div></div>
                <div class="col-6"><small class="text-muted">Correct Winners</small><div>${escapeHtml(cw)}</div></div>
                <div class="col-6"><small class="text-muted">Races Evaluated</small><div>${escapeHtml(re)}</div></div>
              </div>
            </div>
          </div>
        </div>

        <div class="row g-3 mt-1">
          <div class="col-12">
            <div class="border rounded p-2">
              <h6 class="mb-2">Artifacts</h6>
              <div class="row g-2 small">
                <div class="col-12 col-md-6">
                  <small class="text-muted">Model File</small>
                  <div>${escapeHtml(mSizeBytes)}</div>
                  ${details?.model_file_path ? `<div class="text-truncate"><code>${escapeHtml(details.model_file_path)}</code></div>` : ''}
                  <div class="mt-1">
                    ${details?.model_file_path ? `<button class="btn btn-sm btn-outline-secondary me-1" onclick="copyText('${escapeAttr(details.model_file_path)}')">Copy Path</button>` : ''}
                    ${details?.model_file_path ? `<button class="btn btn-sm btn-outline-primary" onclick="revealInFinder('${escapeAttr(String(modelId))}','model')">Reveal in Finder</button>` : ''}
                  </div>
                </div>
                <div class="col-12 col-md-6">
                  <small class="text-muted">Scaler File</small>
                  <div>${escapeHtml(sSizeBytes)}</div>
                  ${details?.scaler_file_path ? `<div class="text-truncate"><code>${escapeHtml(details.scaler_file_path)}</code></div>` : ''}
                  <div class="mt-1">
                    ${details?.scaler_file_path ? `<button class="btn btn-sm btn-outline-secondary me-1" onclick="copyText('${escapeAttr(details.scaler_file_path)}')">Copy Path</button>` : ''}
                    ${details?.scaler_file_path ? `<button class="btn btn-sm btn-outline-primary" onclick="revealInFinder('${escapeAttr(String(modelId))}','scaler')">Reveal in Finder</button>` : ''}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    setDetailsModalContent(title, html);

    const a = document.getElementById('model-details-download');
    if (a) a.href = `/api/model/download/${encodeURIComponent(String(modelId || ''))}`;
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
            // Use registry-backed status for consistent JSON
            fetch('/api/model/registry/status', { cache:'no-store' })
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
        ? `<span class=\"badge bg-success\">In Sync</span>`
        : `<span class=\"badge bg-warning text-dark\">Out of Sync</span>`;
    const syncDetail = (!sync.inSync && (sync.registryBestId || sync.signalId))
        ? `<div class=\"small text-muted\">Registry: ${escapeHtml(formatShortId(sync.registryBestId || '—'))} vs Signal: ${escapeHtml(formatShortId(sync.signalId || '—'))}</div>`
        : '';
    const badgeHelp = `<span class=\"ms-2 small text-muted\" title=\"In Sync means the in-memory registry best model matches the latest broadcast signal. Out of Sync means parts of the app may still be refreshing; it should resolve automatically shortly.\">?</span>`;

    body.innerHTML = `
      <div class=\"d-flex justify-content-between align-items-center mb-2\">
        <div>${badge}${badgeHelp}${syncDetail ? '<span class=\"ms-2\"></span>' : ''}</div>
        <div class=\"small text-muted\">Last refresh: ${escapeHtml(lastRef)}</div>
      </div>
      <div class=\"row g-2\">
        <div class=\"col-12\"><small class=\"text-muted\">Model ID</small><div class=\"fw-semibold\">${escapeHtml(s.promoted_model_id || '—')}</div></div>
        <div class=\"col-6\"><small class=\"text-muted\">Model Name</small><div>${escapeHtml(m.model_name || '—')}</div></div>
        <div class=\"col-6\"><small class=\"text-muted\">Type</small><div>${escapeHtml(m.model_type || '—')}</div></div>
        <div class=\"col-4\"><small class=\"text-muted\">Accuracy</small><div>${acc}</div></div>
        <div class=\"col-4\"><small class=\"text-muted\">AUC</small><div>${auc}</div></div>
        <div class=\"col-4\"><small class=\"text-muted\">Top-1</small><div>${top1}</div></div>
        <div class=\"col-6\"><small class=\"text-muted\">Policy</small><div>${escapeHtml(s.selection_policy || 'correct_winners')}</div></div>
        <div class=\"col-6\"><small class=\"text-muted\">Prediction</small><div><span class=\"badge bg-info\">${escapeHtml(s.prediction_type || m.prediction_type || 'win')}</span></div></div>
        <div class=\"col-12\"><small class=\"text-muted\">Updated</small><div>${escapeHtml(s.timestamp || '—')}</div></div>
      </div>
      <div class=\"mt-2 small text-muted\">Legend: <span class=\"badge bg-success\">In Sync</span> registry equals broadcast; <span class=\"badge bg-warning text-dark\">Out of Sync</span> registry update pending.</div>
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

function humanBytes(bytes){
    try{
        const n = Number(bytes);
        if (!isFinite(n) || n < 0) return '—';
        const u = ['B','KB','MB','GB','TB'];
        let i = 0, v = n;
        while (v >= 1024 && i < u.length - 1){ v /= 1024; i++; }
        const prec = (v >= 10 || i === 0) ? 0 : 1;
        return v.toFixed(prec) + ' ' + u[i];
    }catch{ return '—'; }
}

async function revealInFinder(modelId, kind){
    try{
        const res = await fetch(`/api/model/reveal/${encodeURIComponent(String(modelId||''))}`, {
            method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({kind})
        });
        const j = await res.json().catch(()=>({}));
        if (j && j.success){ showNotification('Revealed in Finder', 'success'); }
        else { showNotification('Reveal failed: ' + (j?.error || res.status), 'error'); }
    }catch(err){ showNotification('Reveal error: ' + (err?.message || err), 'error'); }
}

function copyText(text){
    try{
        if (navigator.clipboard && navigator.clipboard.writeText){
            navigator.clipboard.writeText(text).then(()=>{ showNotification('Path copied to clipboard','success'); })
            .catch(()=>{ fallbackCopy(text); });
        } else { fallbackCopy(text); }
    }catch(e){ fallbackCopy(text); }
    function fallbackCopy(t){
        try{
            const ta = document.createElement('textarea');
            ta.value = String(t||'');
            ta.style.position='fixed'; ta.style.left='-9999px';
            document.body.appendChild(ta); ta.select();
            try{ document.execCommand('copy'); showNotification('Path copied to clipboard','success'); }catch{}
            document.body.removeChild(ta);
        }catch{}
    }
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
