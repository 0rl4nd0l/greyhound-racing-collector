(function(){
  'use strict';
  const btn = document.getElementById('btn-reset-db');
  const bar = document.getElementById('db-reset-progress');
  const statusEl = document.getElementById('db-reset-status');
  const logEl = document.getElementById('db-reset-log');
  const backupEl = document.getElementById('db-reset-backup');
  const optCollect = document.getElementById('opt-collect');
  const optAnalyze = document.getElementById('opt-analyze');

  // Ensure a toast container exists
  function ensureToastContainer(){
    let c = document.querySelector('.toast-container');
    if (!c) {
      c = document.createElement('div');
      c.className = 'toast-container position-fixed top-0 end-0 p-3';
      c.style.zIndex = '2000';
      document.body.appendChild(c);
    }
    return c;
  }
  function showToast(message, variant){
    const c = ensureToastContainer();
    const t = document.createElement('div');
    const klass = variant === 'success' ? 'text-bg-success' : variant === 'danger' ? 'text-bg-danger' : 'text-bg-dark';
    t.className = `toast align-items-center ${klass} border-0 show`;
    t.role = 'alert'; t.ariaLive = 'assertive'; t.ariaAtomic = 'true';
    t.innerHTML = `<div class="d-flex"><div class="toast-body">${message}</div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button></div>`;
    c.appendChild(t);
    setTimeout(()=>{ try{ t.remove(); }catch(e){} }, 4000);
  }

  function log(line) {
    try {
      const ts = new Date().toISOString();
      if (logEl) {
        logEl.textContent += `[${ts}] ${line}\n`;
        logEl.scrollTop = logEl.scrollHeight;
      } else {
        console.log('[db-reset]', line);
      }
    } catch (e) { /* noop */ }
  }

  function setProgress(pct, step) {
    const v = Math.max(0, Math.min(100, Number(pct) || 0));
    if (bar) {
      bar.style.width = v + '%';
      bar.textContent = v + '%';
    }
    if (statusEl) statusEl.textContent = step ? `${step} (${v}%)` : `${v}%`;
  }

  function updateBackup(path){
    if (!backupEl || !path) return;
    // Only set once
    if (backupEl.dataset.set === '1') return;
    backupEl.dataset.set = '1';
    const linkId = 'db-backup-copy';
    backupEl.innerHTML = `Backup created: <code title="Backup path">${escapeHtml(path)}</code> <button id="${linkId}" class="btn btn-sm btn-outline-secondary ms-2">Copy path</button>`;
    const btnCopy = document.getElementById(linkId);
    if (btnCopy) {
      btnCopy.addEventListener('click', async ()=>{
        try { await navigator.clipboard.writeText(path); showToast('Backup path copied', 'success'); } catch(e) { showToast('Failed to copy path', 'danger'); }
      });
    }
  }

  function escapeHtml(s){
    return String(s).replace(/[&<>"]/g, c=> ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  }

  async function postReset() {
    if (!btn) return;
    btn.disabled = true;
    if (bar) { bar.classList.remove('bg-danger','bg-success','bg-warning'); }
    setProgress(1, 'queued');
    log('Submitting reset job...');
    showToast('Starting database reset...', 'dark');
    try {
      const payload = {
        collect: !!(optCollect && optCollect.checked),
        analyze: !!(optAnalyze ? optAnalyze.checked : true)
      };
      const resp = await fetch('/api/database/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await safeJson(resp);
      if (!resp.ok || !data?.success) {
        throw new Error(data?.error || data?.message || `HTTP ${resp.status}`);
      }
      log(`Task started: ${data.task_id}`);
      await poll(data.task_id);
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      log(`ERROR: ${msg}`);
      if (bar) bar.classList.add('bg-danger');
      if (statusEl) statusEl.textContent = 'failed';
      showToast(`Database reset failed: ${msg}`, 'danger');
    } finally {
      btn.disabled = false;
    }
  }

  async function poll(taskId) {
    let lastStep = null;
    const start = Date.now();
    for (;;) {
      const done = await tick();
      if (done) break;
      await sleep(1000);
    }

    async function tick() {
      try {
        const res = await fetch(`/api/background/status/${encodeURIComponent(taskId)}`, { cache: 'no-store' });
        const s = await safeJson(res);
        if (!s?.success) throw new Error(s?.error || 'status error');
        const step = s.step || s.status || 'running';
        const pct = typeof s.progress === 'number' ? s.progress : (step === 'completed' ? 100 : 0);
        if (step !== lastStep) {
          let line = `step: ${step}`;
          if (s.backup) { line += ` (backup: ${s.backup})`; updateBackup(s.backup); }
          if (s.error) line += ` — error: ${s.error}`;
          log(line);
          lastStep = step;
        }
        setProgress(pct, step);
        if (s.status === 'completed') {
          log('✅ Completed.');
          if (bar) { bar.classList.remove('bg-danger'); bar.classList.add('bg-success'); }
          showToast('Database reset completed', 'success');
          return true;
        }
        if (s.status === 'failed') {
          log(`❌ Failed: ${s.error || 'unknown error'}`);
          if (bar) bar.classList.add('bg-danger');
          showToast(`Database reset failed: ${s.error || 'unknown error'}`, 'danger');
          return true;
        }
        // Safety timeout: 20 minutes
        if (Date.now() - start > 20 * 60 * 1000) {
          log('⏰ Timeout waiting for completion.');
          if (bar) bar.classList.add('bg-warning');
          showToast('Database reset timed out', 'danger');
          return true;
        }
        return false;
      } catch (e) {
        const msg = e && e.message ? e.message : String(e);
        log(`poll error: ${msg}`);
        if (bar) bar.classList.add('bg-danger');
        showToast(`Status polling failed: ${msg}`, 'danger');
        return true;
      }
    }
  }

  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
  async function safeJson(resp) { try { return await resp.json(); } catch { return {}; } }

  if (btn) { btn.addEventListener('click', postReset); }

  // Expose a simple starter so other UI (the red card) can trigger a real reset
  window.startDbReset = function startDbReset() {
    try {
      if (optAnalyze) optAnalyze.checked = true;
      if (optCollect) optCollect.checked = false;
    } catch(e){}
    if (btn) btn.click(); else showToast('Reset control not available on this page', 'danger');
  };
})();

