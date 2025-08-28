/*
  ML Training stub bundle (via Vite)
  - Safe to include on /ml-training and /ml_training_simple pages
  - Minimal DOM hooks; no external dependencies
*/

(function () {
  const log = (...args) => console.log('[model-training]', ...args);
  const qs = (sel) => document.querySelector(sel);

  async function safeFetchJSON(url, options = {}) {
    try {
      const res = await fetch(url, { credentials: 'same-origin', ...options });
      const ct = res.headers.get('content-type') || '';
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      if (ct.includes('application/json')) return await res.json();
      return { success: true, raw: await res.text() };
    } catch (err) {
      log('fetch error', err);
      return { success: false, error: String(err) };
    }
  }

  async function refreshModelStatus() {
    const statusEl = qs('#model-status');
    const badgeEl = qs('#model-status-badge');
    const res = await safeFetchJSON('/api/model_status');
    if (!statusEl) return res; // no-op if no target

    if (res && res.success) {
      const d = res;
      statusEl.textContent = `Model: ${d.model_type || 'N/A'} | Accuracy: ${fmtPct(d.accuracy)} | AUC: ${fmtPct(d.auc_score)} | Last Trained: ${d.last_trained || 'N/A'}`;
      if (badgeEl) badgeEl.textContent = d.model_type || 'No model';
    } else {
      statusEl.textContent = `Model status unavailable${res && res.error ? `: ${res.error}` : ''}`;
    }
    return res;
  }

  function fmtPct(v) {
    const n = Number(v);
    if (!isFinite(n) || n <= 0) return '—';
    return `${(n * (n <= 1 ? 100 : 1)).toFixed(2)}%`;
    // Supports already-in-[0,1] and already-in-[0,100]
  }

  async function startAutomatedTraining() {
    const btn = qs('#btn-start-training');
    if (btn) btn.disabled = true;
    const res = await safeFetchJSON('/api/automated_training', { method: 'POST' });
    if (!res.success) alert(`Training start failed: ${res.error || 'Unknown error'}`);
    await wait(600);
    await pollTrainingStatus();
    if (btn) btn.disabled = false;
  }

  async function pollTrainingStatus() {
    const el = qs('#training-status');
    const logEl = qs('#training-log');
    if (!el && !logEl) return; // no-op if nothing to update

    // Poll a few times; this is just a stub UX
    for (let i = 0; i < 20; i++) {
      const res = await safeFetchJSON('/api/training_status');
      if (el && res) {
        const s = res.running ? 'running' : (res.completed ? 'completed' : (res.error ? 'error' : 'idle'));
        el.textContent = `Status: ${s} | Progress: ${Number(res.progress || 0)}%`;
      }
      if (logEl && res && Array.isArray(res.log)) {
        logEl.innerHTML = res.log.slice(-10).map(e => `<div>[${e.timestamp || ''}] ${escapeHtml(e.message || '')}</div>`).join('');
      }
      if (res && (res.completed || res.error)) break;
      await wait(2000);
    }
  }

  function wait(ms) { return new Promise(r => setTimeout(r, ms)); }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

  function wireEvents() {
    const trainBtn = qs('#btn-start-training');
    if (trainBtn) trainBtn.addEventListener('click', startAutomatedTraining);
    const refreshBtn = qs('#btn-refresh-status');
    if (refreshBtn) refreshBtn.addEventListener('click', refreshModelStatus);
  }

  document.addEventListener('DOMContentLoaded', async () => {
    log('bundle loaded');
    wireEvents();
    await refreshModelStatus();
  });

  // Expose a tiny API for debugging
  window.ModelTraining = { refreshModelStatus, startAutomatedTraining, pollTrainingStatus };
})();

