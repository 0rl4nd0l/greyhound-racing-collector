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

  // Show a compact registry summary instead of legacy single-model status
  async function refreshModelStatus() {
    const statusEl = qs('#model-status');
    const badgeEl = qs('#model-status-badge');
    const res = await safeFetchJSON('/api/model/registry/status');
    if (!statusEl) return res; // no-op if no target

    if (res && res.success) {
      const total = Number(res.total_models ?? res.model_count ?? (Array.isArray(res.all_models) ? res.all_models.length : 0));
      const activeJobs = Number((res.registry_info && res.registry_info.active_jobs) || 0);
      const ts = (res.registry_info && res.registry_info.timestamp) || '';
      statusEl.textContent = `Models: ${total} | Active jobs: ${activeJobs} | ${ts}`;
      if (badgeEl) badgeEl.textContent = `Models: ${total}`;
    } else {
      statusEl.textContent = `Registry status unavailable${res && res.error ? `: ${res.error}` : ''}`;
    }
    return res;
  }

  function fmtPct(v) {
    const n = Number(v);
    if (!isFinite(n) || n <= 0) return '—';
    return `${(n * (n <= 1 ? 100 : 1)).toFixed(2)}%`;
  }

  async function startAutomatedTraining() {
    const btn = qs('#btn-start-training');
    if (btn) btn.disabled = true;
    // Trigger a registry-backed job for comprehensive training by default
    const res = await safeFetchJSON('/api/model/training/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: 'comprehensive_training', prediction_type: 'win', training_data_days: 7, force_retrain: true })
    });
    if (!res.success) {
      alert(`Training start failed: ${res.error || 'Unknown error'}`);
      if (btn) btn.disabled = false;
      return;
    }
    const jobId = res.job_id;
    await wait(600);
    await pollTrainingStatus(jobId);
    if (btn) btn.disabled = false;
  }

  async function pollTrainingStatus(jobId) {
    const el = qs('#training-status');
    const logEl = qs('#training-log');
    if (!el && !logEl) return; // no-op if nothing to update

    // Poll until terminal state or for up to ~60 seconds
    const deadline = Date.now() + 60 * 1000;
    while (Date.now() < deadline) {
      const res = await safeFetchJSON(`/api/model/registry/status${jobId ? `?job_id=${encodeURIComponent(jobId)}` : ''}`);
      if (el && res) {
        const s = res.status || (res.registry_info && res.registry_info.active_jobs > 0 ? 'running' : 'idle');
        const pct = Number(res.progress || 0);
        el.textContent = `Status: ${s} | Progress: ${pct}%`;
      }
      if (logEl && res) {
        // Registry-backed status doesn't stream logs; show a minimal status line
        const now = new Date().toISOString();
        const s = res.status || 'checking';
        logEl.innerHTML = `<div>[${now}] ${escapeHtml(`status=${s}, progress=${Number(res.progress||0)}%`)}</div>`;
      }
      if (res && (res.status === 'completed' || res.status === 'failed')) break;
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

  async function maybeAutoStart() {
    try {
      const res = await safeFetchJSON('/api/model/registry/status');
      if (res && res.success) {
        const active = (res.registry_info && res.registry_info.active_jobs) || 0;
        // Only auto-start if no active jobs
        if (!active) {
          await startAutomatedTraining();
        }
      }
    } catch (_) { /* ignore */ }
  }

  document.addEventListener('DOMContentLoaded', async () => {
    log('bundle loaded');
    wireEvents();
    await refreshModelStatus();
    // Help tests by auto-starting a training job when page loads and no job is active
    await maybeAutoStart();
  });

  // Expose a tiny API for debugging
  window.ModelTraining = { refreshModelStatus, startAutomatedTraining, pollTrainingStatus };
})();

