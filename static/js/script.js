// Greyhound Racing Dashboard JavaScript
// =====================================


document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initializeDashboard();

    // Inject Sportsbet controls on the odds dashboard
    try { maybeInjectSportsbetControls(); } catch (e) { console.warn('SB controls inject failed', e); }
    
    // Auto-refresh stats every 30 seconds
    setInterval(refreshStats, 30000);
});

function initializeDashboard() {
    console.log('🐾 Greyhound Racing Dashboard initialized');
    
    // Add any initialization code here
    highlightActiveNavItem();
    initializeRefreshButtons();
}

function highlightActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('nav a');
    
    // Use Array.from to ensure forEach is available on all browsers
    Array.from(navLinks).forEach(link => {
        const href = link.getAttribute('href');
        // Clear any legacy inline background to maintain WCAG contrast
        if (link.style && link.style.backgroundColor) {
            link.style.backgroundColor = '';
        }
        if (href === currentPath) {
            // Prefer semantic active state and accessible name
            link.classList.add('active');
            if (!link.getAttribute('aria-current')) {
                link.setAttribute('aria-current', 'page');
            }
            // Ensure high-contrast text color is enforced via CSS, not inline styles
        }
    });
}

function initializeRefreshButtons() {
    const refreshButtons = document.querySelectorAll('[data-refresh]');
    
    // Use Array.from to ensure forEach is available on all browsers
    Array.from(refreshButtons).forEach(button => {
        button.addEventListener('click', function() {
            const target = this.getAttribute('data-refresh');
            refreshSection(target);
        });
    });
}

function refreshStats() {
    fetch('/api/stats')
        .then(response => response.json())
        .then(data => {
            console.log('📊 Stats refreshed:', data);
            // Update stats on the page if needed
        })
        .catch(error => {
            console.error('Error refreshing stats:', error);
        });
}

function refreshSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.style.opacity = '0.5';
        
        // Simulate refresh - in reality, this would update the content
        setTimeout(() => {
            section.style.opacity = '1';
        }, 500);
    }
}

// Utility functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 3000);
}

// API interaction functions
async function fetchRecentRaces(limit = 10) {
    try {
        const response = await fetch(`/api/recent_races?limit=${limit}`);
        const data = await response.json();
        const racesArray = Array.isArray(data.races) ? data.races : Object.values(data.races || {});
        return racesArray;
    } catch (error) {
        console.error('Error fetching recent races:', error);
        return [];
    }
}

async function fetchRaceDetails(raceId) {
    try {
        const response = await fetch(`/api/race/${raceId}`);
        const data = await response.json();
        return data.race_data;
    } catch (error) {
        console.error('Error fetching race details:', error);
        return null;
    }
}

function maybeInjectSportsbetControls(){
    const path = (window.location && window.location.pathname) || '';
    if (!/\/odds_dashboard\b/.test(path)) return;
    const container = document.querySelector('main') || document.body;
    const panel = document.createElement('div');
    panel.className = 'sb-controls';
    panel.innerHTML = `
      <div class="sb-header">
        <span class="sb-title">Sportsbet Controls</span>
        <span id="sb-status-badge" class="badge sb-badge">Checking…</span>
      </div>
      <div class="sb-actions">
<button type=\"button\" class=\"btn btn-sm btn-primary btn-contrast\" id=\"od-ctl-update\">Update odds & value bets</button>
<button type=\"button\" class=\"btn btn-sm btn-outline-secondary btn-contrast\" id=\"od-ctl-seed-quick\">Seed quick (3)</button>
<button type=\"button\" class=\"btn btn-sm btn-outline-dark btn-contrast\" id=\"od-ctl-seed-preds\">Seed for predictions</button>
      </div>
      <div class="sb-metrics">
        <div>
          <div class="sb-metric-value" id="sb-races-updated">0</div>
          <div class="sb-metric-label">Races updated</div>
        </div>
        <div>
          <div class="sb-metric-value" id="sb-value-count">0</div>
          <div class="sb-metric-label">Value bets</div>
        </div>
      </div>
      <div class="sb-options">
        <label style="display:flex; align-items:center; gap:6px;">
          <input type="checkbox" id="sb-auto-refresh" /> Auto refresh
        </label>
        <div style="display:flex; gap:8px;">
          <a href="/api/sportsbet/live_odds" target="_blank" rel="noreferrer">live_odds</a>
          <a href="/api/sportsbet/value_bets" target="_blank" rel="noreferrer">value_bets</a>
        </div>
      </div>
      <div id="od-ctl-out" class="sb-out small"></div>
    `;
    container.appendChild(panel);

    const out = panel.querySelector('#od-ctl-out');
    const setOut = (s)=>{ try{ out.textContent = s; }catch(_){} };
    const disable = (b, v)=>{ try{ b.disabled = !!v; if(v) b.classList.add('btn-loading'); else b.classList.remove('btn-loading'); }catch(_){} };

    const btnUpdate = panel.querySelector('#od-ctl-update');
    const btnQuick = panel.querySelector('#od-ctl-seed-quick');
    const btnPreds = panel.querySelector('#od-ctl-seed-preds');
    const badge = panel.querySelector('#sb-status-badge');
    const racesEl = panel.querySelector('#sb-races-updated');
    const valuesEl = panel.querySelector('#sb-value-count');
    const autoChk = panel.querySelector('#sb-auto-refresh');

    async function refreshStatus(){
      try{
        const r = await fetch('/api/sportsbet/status', {cache:'no-store'});
        const j = await r.json();
        badge.className = 'badge sb-badge';
        if(j && j.success){
          const st = (j.status||{});
          if(st.available){ badge.classList.add('success'); badge.textContent = 'Active'; }
          else if(st.disabled_env){ badge.classList.add('warning'); badge.textContent = 'Disabled'; }
          else { badge.classList.add('danger'); badge.textContent = 'Unavailable'; }
        }else{ badge.classList.add('danger'); badge.textContent = 'Unavailable'; }
      }catch(e){ badge.className = 'badge sb-badge danger'; badge.textContent='Unavailable'; }
    }

    async function refreshMetrics(){
      try{
        const [oddsRes, vbRes] = await Promise.all([
          fetch('/api/sportsbet/live_odds', {cache:'no-store'}),
          fetch('/api/sportsbet/value_bets', {cache:'no-store'})
        ]);
        const odds = await oddsRes.json();
        const vbs = await vbRes.json();
        if(odds && odds.success && racesEl) racesEl.textContent = String((odds.odds_summary||[]).length);
        if(vbs && vbs.success && valuesEl) valuesEl.textContent = String((vbs.value_bets||[]).length);
      }catch(e){ /* ignore */ }
    }

    let autoTimer = null;
    function applyAutoRefresh(){
      try{ if(autoTimer) { clearInterval(autoTimer); autoTimer = null; } }catch(_){ }
      if(autoChk && autoChk.checked){
        autoTimer = setInterval(()=>{ refreshMetrics(); }, 30000);
      }
      try{ localStorage.setItem('sb_auto_refresh', autoChk && autoChk.checked ? '1' : '0'); }catch(_){ }
    }
    try{ autoChk.checked = (localStorage.getItem('sb_auto_refresh')||'0') === '1'; }catch(_){ }
    autoChk.addEventListener('change', applyAutoRefresh);

    btnUpdate && btnUpdate.addEventListener('click', async ()=>{
      disable(btnUpdate, true); setOut('Updating odds and computing value bets…');
      try{
        const r = await fetch('/api/sportsbet/update_odds', { method: 'POST' });
        const j = await r.json();
        if(j && j.success){
          setOut(`Updated ${j.races_updated||0} races; found ${j.value_bets_found||0} value bets.`);
          if(racesEl) racesEl.textContent = String(j.races_updated||0);
          if(valuesEl) valuesEl.textContent = String(j.value_bets_found||0);
        } else {
          setOut(`Update failed: ${(j && (j.message||j.error)) || 'Unknown error'}`);
        }
      }catch(e){ setOut('Update failed.'); }
      finally{ disable(btnUpdate, false); }
    });

    btnQuick && btnQuick.addEventListener('click', async ()=>{
      disable(btnQuick, true); setOut('Seeding a few upcoming races…');
      try{
        const r = await fetch('/api/sportsbet/seed_quick', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ limit: 3, statuses: ['LIVE','UPCOMING','SOON','LATER'] }) });
        const j = await r.json();
        if(j && j.success){ setOut(`Seeded ${j.seeded||0} race(s).`); }
        else { setOut(`Seed failed: ${(j && (j.errors?.join(', ') || j.message || j.error)) || 'Unknown error'}`); }
      }catch(e){ setOut('Seed failed.'); }
      finally{ disable(btnQuick, false); refreshMetrics(); }
    });

    btnPreds && btnPreds.addEventListener('click', async ()=>{
      disable(btnPreds, true); setOut("Seeding races linked to today's predictions…");
      try{
        const r = await fetch('/api/sportsbet/seed_for_predictions_today', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ limit: 8 }) });
        const j = await r.json();
        if(j && j.success){ setOut(`Seeded ${j.seeded||0}/${j.attempted||0} prediction-linked races.`); }
        else { setOut(`Seed-for-predictions failed: ${(j && (j.errors?.join(', ') || j.message || j.error)) || 'Unknown error'}`); }
      }catch(e){ setOut('Seed-for-predictions failed.'); }
      finally{ disable(btnPreds, false); refreshMetrics(); }
    });

    // Initial
    refreshStatus();
    refreshMetrics();
    applyAutoRefresh();
}

// Export functions for global use
window.dashboardUtils = {
    refreshStats,
    refreshSection,
    formatFileSize,
    formatDate,
    showNotification,
    fetchRecentRaces,
    fetchRaceDetails
};
