(function(){
  if (!window.ENDPOINTS_MENU_ENABLED) return;
  const FLAG_KEY = 'ENDPOINTS_MENU_ENABLED';
  try {
    // Avoid duplicate injection
    if (window.__endpointsMenuInjected) return;
    window.__endpointsMenuInjected = true;

    const ready = (cb)=>{
      if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', cb, {once:true});
      else cb();
    };

    const createEl = (tag, attrs={}, children=[])=>{
      const el = document.createElement(tag);
      for (const [k,v] of Object.entries(attrs)){
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = v;
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.substring(2), v);
        else el.setAttribute(k, v);
      }
      for (const c of (Array.isArray(children)?children:[children])){
        if (c == null) continue;
        el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      }
      return el;
    };

    const groupBy = (arr, key)=>{
      const m = new Map();
      for (const it of arr){
        const k = (typeof key === 'function') ? key(it) : (it[key] ?? 'Other');
        if (!m.has(k)) m.set(k, []);
        m.get(k).push(it);
      }
      return m;
    };

    const fetchEndpoints = async ()=>{
      const res = await fetch('/api/endpoints', { headers: { 'Accept': 'application/json' }});
      if (!res.ok) throw new Error('Failed to load endpoints');
      const data = await res.json();
      if (!data || data.success === false) throw new Error(data && data.error || 'No endpoints');
      return data.endpoints || [];
    };

    const catOrder = [
      'Predictions',
      'Monitoring & Health',
      'Model Registry & Training',
      'Upcoming & Data',
      'TGR Enrichment',
      'Background Tasks',
      'Utilities'
    ];

    const ensureMount = ()=>{
      // Try to find an existing navbar; otherwise create a compact floating toolbar
      let navBar = document.querySelector('.navbar .container, .navbar .container-fluid, header .navbar');
      if (navBar) return navBar;
      // Create minimal bar at top-right
      const bar = createEl('div', { id: 'endpoints-menu-toolbar', style: 'position:fixed;top:8px;right:8px;z-index:2147483647;background:rgba(255,255,255,0.96);border:1px solid #ddd;border-radius:8px;padding:6px 8px;font:14px/1.3 system-ui,-apple-system,Segoe UI,Roboto;box-shadow:0 2px 8px rgba(0,0,0,0.12);' });
      document.body.appendChild(bar);
      return bar;
    };

    const buildMenu = (items)=>{
      // Normalize and sort
      items = items.filter(e => e && e.path && Array.isArray(e.methods));
      items.sort((a,b)=>{
        const ac = a.category || 'Utilities';
        const bc = b.category || 'Utilities';
        const ai = catOrder.indexOf(ac);
        const bi = catOrder.indexOf(bc);
        if (ai !== bi) return (ai<0?999:ai) - (bi<0?999:bi);
        return (a.path||'').localeCompare(b.path||'');
      });

      const byCat = groupBy(items, it => it.category || 'Utilities');

      const container = createEl('div', { class: 'endpoints-menu d-flex flex-wrap gap-2 align-items-center' });

      const makeSelect = (label, opts)=>{
        const wrap = createEl('div', { class: 'endpoints-select-wrap', style: 'margin:4px;' });
        const lab = createEl('label', { class: 'me-2', style: 'font-weight:600;' , text: label});
        const sel = createEl('select', { class: 'form-select form-select-sm', style: 'min-width: 260px; display:inline-block;' });
        sel.appendChild(createEl('option', { value: '', text: `Select ${label}…` }));
        for (const o of opts){
          const txt = `${o.path} ${o.methods ? '['+o.methods.join(',')+']' : ''}`;
          sel.appendChild(createEl('option', { value: o.path, text: txt }));
        }
        wrap.appendChild(lab);
        wrap.appendChild(sel);
        return { wrap, sel };
      };

      const runGET = async (path)=>{
        try {
          window.localStorage.setItem('lastEndpointPath', path);
          const res = await fetch(path, { headers: { 'Accept': 'application/json' }});
          const ct = res.headers.get('Content-Type') || '';
          const isJSON = ct.includes('application/json');
          const body = isJSON ? await res.json() : await res.text();
          notify(`${res.status} ${res.statusText} - ${path}`, isJSON ? JSON.stringify(body).slice(0, 2000) : String(body).slice(0, 2000));
        } catch (e){
          notify(`Request failed - ${path}`, String(e && e.message || e), true);
        }
      };

      const runPOST = async (def)=>{
        const path = def.path;
        try {
          window.localStorage.setItem('lastEndpointPath', path);
          if (def.requires_body){
            const payload = prompt(`POST ${path}\nEnter JSON body (or leave empty for {}):`, '{}');
            if (payload === null) return; // cancelled
            let json = {};
            try { json = payload ? JSON.parse(payload) : {}; } catch(e){ return alert('Invalid JSON body'); }
            const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify(json) });
            const body = await res.text();
            notify(`${res.status} ${res.statusText} - ${path}`, body.slice(0, 2000));
          } else {
            const res = await fetch(path, { method: 'POST', headers: { 'Accept': 'application/json' } });
            const body = await res.text();
            notify(`${res.status} ${res.statusText} - ${path}`, body.slice(0, 2000));
          }
        } catch (e){
          notify(`POST failed - ${path}`, String(e && e.message || e), true);
        }
      };

      const notify = (title, body, isError)=>{
        try {
          // Use Bootstrap toast container when present, otherwise fallback alert
          let cont = document.querySelector('.toast-container');
          if (!cont){
            cont = createEl('div', { class: 'toast-container position-fixed top-0 end-0 p-3' });
            document.body.appendChild(cont);
          }
          const toast = createEl('div', { class: 'toast align-items-center text-white ' + (isError?'bg-danger':'bg-primary'), role: 'status', 'aria-live':'polite', 'aria-atomic':'true', style:'min-width:360px; max-width: 520px;' });
          const bodyDiv = createEl('div', { class: 'd-flex' }, [
            createEl('div', { class: 'toast-body', style: 'white-space: pre-wrap; max-height: 320px; overflow:auto;' }, [createEl('div', { text: title }), createEl('div', { text: body })]),
            createEl('button', { type: 'button', class: 'btn-close btn-close-white me-2 m-auto', 'data-bs-dismiss': 'toast', 'aria-label': 'Close' })
          ]);
          toast.appendChild(bodyDiv);
          cont.appendChild(toast);
          // Bootstrap 5 toast init if available
          const any = window.bootstrap && window.bootstrap.Toast ? new window.bootstrap.Toast(toast, { delay: 5000 }) : null;
          if (any) any.show(); else toast.classList.add('show');
          setTimeout(()=>{ try{ toast.remove(); } catch(_e){} }, 8000);
        } catch(e){
          alert(title + '\n' + body);
        }
      };

      // Build one dropdown per category
      const orderedCats = Array.from(byCat.keys()).sort((a,b)=>{
        const ai = catOrder.indexOf(a);
        const bi = catOrder.indexOf(b);
        return (ai<0?999:ai) - (bi<0?999:bi);
      });

      orderedCats.forEach(cat => {
        const entries = byCat.get(cat) || [];
        // Separate GET and POST entries for clarity
        const gets = entries.filter(e => e.methods.includes('GET'));
        const posts = entries.filter(e => e.methods.includes('POST'));
        // GET select
        if (gets.length){
          const { wrap, sel } = makeSelect(cat + ' (GET)', gets);
          sel.addEventListener('change', async (e)=>{
            const path = e.target.value;
            if (!path) return;
            await runGET(path);
            sel.value = '';
          });
          container.appendChild(wrap);
        }
        // POST select
        if (posts.length){
          const { wrap, sel } = makeSelect(cat + ' (POST)', posts);
          sel.addEventListener('change', async (e)=>{
            const path = e.target.value;
            if (!path) return;
            const def = posts.find(p => p.path === path) || { path };
            await runPOST(def);
            sel.value = '';
          });
          container.appendChild(wrap);
        }
      });

      // Small help link
      container.appendChild(createEl('a', { href:'#', class:'ms-2 link-secondary', title:'About endpoint menus', onClick:(e)=>{e.preventDefault(); alert('This toolbar lists all server routes grouped by category. GET routes fetch and show responses. POST routes may prompt for a JSON body.');}}, 'help'));

      return container;
    };

    ready(async ()=>{
      try {
        const endpoints = await fetchEndpoints();
        const mount = ensureMount();
        const menu = buildMenu(endpoints);
        // If mounting into a navbar, wrap in a nav-item container when possible
        if (mount.classList.contains('container') || mount.classList.contains('container-fluid')){
          const holder = createEl('div', { class: 'd-flex flex-wrap align-items-center ms-auto', style: 'gap:6px;' });
          holder.appendChild(menu);
          mount.appendChild(holder);
        } else {
          mount.appendChild(menu);
        }
      } catch (e){
        // Silent fail – do not break the page
        console.warn('Endpoints menu failed to initialize:', e);
      }
    });
  } catch(e){
    // Never throw
    console.warn('Endpoints menu fatal:', e);
  }
})();

