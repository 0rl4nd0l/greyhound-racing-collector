(function(){
  function fmtTs(ts){
    try { return new Date(ts * 1000).toLocaleString(); } catch(e){ return String(ts); }
  }

  // Heuristic summary extractor for cohort reports with flexible schemas
  function deriveCohortSummary(payload){
    const p = payload || {};
    const g = (obj, paths, dflt) => {
      if(!obj) return dflt;
      for(const path of paths){
        try{
          const parts = path.split('.');
          let cur = obj;
          for(const part of parts){
            if(cur == null) { cur = undefined; break; }
            cur = cur[part];
          }
          if(cur !== undefined && cur !== null && !(typeof cur === 'number' && Number.isNaN(cur))) return cur;
        }catch(_){/*noop*/}
      }
      return dflt;
    };
    const asCount = (v) => {
      if(typeof v === 'number' && isFinite(v)) return v;
      if(Array.isArray(v)) return v.length;
      return undefined;
    };

    // Total races
    let total = g(p, ['total_races','n_races','num_races','races_evaluated','count','num_samples'], undefined);
    if(total === undefined){
      total = asCount(g(p, ['races','rows','results','entries','samples'], undefined));
    }

    // Accuracy-like fields
    const accuracy = g(p, ['accuracy','metrics.accuracy','summary.accuracy','overall.accuracy','top1_accuracy','avg_accuracy','balanced_accuracy'], undefined);

    // ROI-like fields
    const roi = g(p, ['roi','metrics.roi','summary.roi','overall.roi','avg_roi'], undefined);

    // Period / date range
    const start = g(p, ['period_start','start_date','start','min_date'], undefined);
    const end = g(p, ['period_end','end_date','end','max_date'], undefined);
    const period = (start || end) ? `${start ?? '?'} → ${end ?? '?'}` : g(p, ['period','date_range'], undefined);

    // Optional loss metrics
    const logloss = g(p, ['logloss','metrics.logloss','summary.logloss'], undefined);
    const auc = g(p, ['roc_auc','metrics.roc_auc','summary.roc_auc','auc'], undefined);

    const summary = {};
    if(total !== undefined) summary.total_races = total;
    if(accuracy !== undefined) summary.accuracy = accuracy;
    if(roi !== undefined) summary.roi = roi;
    if(period !== undefined) summary.period = period;
    if(logloss !== undefined) summary.logloss = logloss;
    if(auc !== undefined) summary.roc_auc = auc;

    return summary;
  }

  function renderCohortSummary(summary, updatedAt){
    const box = document.getElementById('cohort-summary');
    if(!box) return;
    const keysOrder = ['period','total_races','accuracy','roi','roc_auc','logloss'];
    const pretty = {
      period: 'Period',
      total_races: 'Total Races',
      accuracy: 'Accuracy',
      roi: 'ROI',
      roc_auc: 'ROC AUC',
      logloss: 'LogLoss'
    };
    const fmtVal = (k, v) => {
      if(v == null) return '';
      if(typeof v === 'number'){
        if(k === 'accuracy' || k === 'roc_auc') return (v*1).toFixed(4);
        if(k === 'roi') return (v*1).toFixed(3);
        if(k === 'logloss') return (v*1).toFixed(4);
      }
      return String(v);
    };

    const pieces = [];
    for(const k of keysOrder){
      if(Object.prototype.hasOwnProperty.call(summary, k)){
        const v = summary[k];
        pieces.push(`<span class="tag" aria-label="${pretty[k]}"><strong>${pretty[k]}:</strong> ${fmtVal(k, v)}</span>`);
      }
    }
    if(pieces.length === 0){
      box.innerHTML = '<span class="text-muted">No summary available.</span>';
      return;
    }
    box.innerHTML = pieces.join(' ');
  }

  function renderCohort(data){
    const fn = document.getElementById('cohort-filename');
    const up = document.getElementById('cohort-updated');
    const st = document.getElementById('cohort-status');
    const pre = document.getElementById('cohort-json');
    if(!data || !data.success){
      if(st) st.textContent = 'Not found';
      if(pre) pre.textContent = 'No cohort report available.';
      renderCohortSummary({}, null);
      return;
    }
    if(fn) fn.textContent = data.filename || 'cohort_report.json';
    if(up) up.textContent = data.updated_at ? '('+fmtTs(data.updated_at)+')' : '';
    if(st) st.textContent = 'OK';
    try{
      const payload = data.cohort_report ?? {};
      // Summary
      const summary = deriveCohortSummary(payload);
      renderCohortSummary(summary, data.updated_at || null);
      // Raw JSON (truncated)
      const text = JSON.stringify(payload, null, 2);
      pre.textContent = text.length > 60000 ? (text.slice(0, 60000) + '\n... [truncated]') : text;
    }catch(e){
      pre.textContent = 'Failed to render cohort report.';
      renderCohortSummary({}, null);
    }
  }

  function renderRegistry(data){
    const fn = document.getElementById('registry-filename');
    const up = document.getElementById('registry-updated');
    const st = document.getElementById('registry-status');
    const thead = document.getElementById('registry-thead');
    const tbody = document.getElementById('registry-tbody');
    if(!data || !data.success){
      if(st) st.textContent = 'Not found';
      thead.innerHTML = '';
      tbody.innerHTML = '<tr><td class="text-muted">No registry report available.</td></tr>';
      return;
    }
    if(fn) fn.textContent = data.filename || 'registry_report.csv';
    if(up) up.textContent = data.updated_at ? '('+fmtTs(data.updated_at)+')' : '';
    if(st) st.textContent = 'OK';

    const rows = Array.isArray(data.rows) ? data.rows : [];
    if(rows.length === 0){
      thead.innerHTML = '';
      tbody.innerHTML = '<tr><td class="text-muted">Empty report.</td></tr>';
      return;
    }

    const cols = Object.keys(rows[0]);
    thead.innerHTML = '<tr>' + cols.map(c => `<th scope="col">${c}</th>`).join('') + '</tr>';
    const html = rows.map(r => '<tr>' + cols.map(c => `<td>${(r[c] ?? '')}</td>`).join('') + '</tr>').join('');
    tbody.innerHTML = html;
  }

  async function loadCohort(){
    try{
      const res = await fetch('/api/backtests/cohort/latest');
      const data = await res.json();
      renderCohort(data);
    }catch(e){ renderCohort(null); }
  }

  async function loadRegistry(){
    try{
      const res = await fetch('/api/registry/report');
      const data = await res.json();
      renderRegistry(data);
    }catch(e){ renderRegistry(null); }
  }

  function bind(){
    document.getElementById('refresh-cohort')?.addEventListener('click', loadCohort);
    document.getElementById('refresh-registry')?.addEventListener('click', loadRegistry);
  }

  document.addEventListener('DOMContentLoaded', function(){
    bind();
    loadCohort();
    loadRegistry();
  });
})();

