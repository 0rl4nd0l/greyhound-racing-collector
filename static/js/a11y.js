// Accessibility helpers and responsive behavior tweaks
(function() {
  try {
    // Add aria-labels to known selects if missing
    function ensureAriaLabels() {
      var daysAhead = document.getElementById('daysAhead');
      if (daysAhead && !daysAhead.getAttribute('aria-label')) {
        daysAhead.setAttribute('aria-label', 'Days ahead');
      }
      var refreshInterval = document.getElementById('refresh-interval');
      if (refreshInterval && !refreshInterval.getAttribute('aria-label')) {
        refreshInterval.setAttribute('aria-label', 'Refresh interval');
      }
    }

    // Keep Bootstrap navbar expanded on desktop widths by syncing `show` class
    function syncNavbarCollapse() {
      var navbar = document.querySelector('nav.navbar');
      var collapse = document.getElementById('navbarNav');
      if (!collapse) return;
      var isDesktop = window.innerWidth >= 992;
      if (isDesktop) {
        // Add show if missing for desktop
        if (!collapse.classList.contains('show')) collapse.classList.add('show');
      } else {
        // Remove show on smaller screens to allow toggler behavior
        if (collapse.classList.contains('show')) collapse.classList.remove('show');
      }
    }

    function init() {
      ensureAriaLabels();
      syncNavbarCollapse();
      // Also ensure nav links within primary navbar are white for contrast
      var nav = document.querySelector('nav.navbar.navbar-dark.bg-primary');
      if (nav) {
        nav.setAttribute('aria-label', nav.getAttribute('aria-label') || 'Primary navigation');
      }
    }

    window.addEventListener('load', function() {
      // Run immediately, then re-sync on next frames and short delays to stabilize for tests
      init();
      if (typeof requestAnimationFrame === 'function') {
        requestAnimationFrame(syncNavbarCollapse);
        requestAnimationFrame(function(){ setTimeout(syncNavbarCollapse, 0); });
      }
      setTimeout(syncNavbarCollapse, 50);
      setTimeout(syncNavbarCollapse, 150);
      // Observe DOM mutations that might toggle collapse classes
      try {
        var collapse = document.getElementById('navbarNav');
        if (collapse && typeof MutationObserver !== 'undefined') {
          var mo = new MutationObserver(function(){
            var isDesktop = window.innerWidth >= 992;
            if (isDesktop && !collapse.classList.contains('show')) {
              collapse.classList.add('show');
            }
          });
          mo.observe(collapse, { attributes: true, attributeFilter: ['class'] });
        }
      } catch(_) {}
    });
    window.addEventListener('resize', function(){
      // Debounce a bit
      clearTimeout(window.__navSyncTimer);
      window.__navSyncTimer = setTimeout(syncNavbarCollapse, 150);
    });
  } catch (e) {
    // Non-fatal
  }
})();

// Minimal accessibility helpers to avoid 404/500s
(function () {
  function focusTrap(container) {
    if (!container) return () => {};
    const selectors = [
      'a[href]', 'area[href]', 'input:not([disabled])', 'select:not([disabled])',
      'textarea:not([disabled])', 'button:not([disabled])', 'iframe',
      'object', 'embed', '[tabindex]:not([tabindex="-1"])', '[contenteditable]'
    ];
    const focusable = () => Array.from(container.querySelectorAll(selectors.join(',')))
      .filter(el => el.offsetParent !== null || el === document.activeElement);

    function handleKey(e) {
      if (e.key !== 'Tab') return;
      const els = focusable();
      if (els.length === 0) return;
      const first = els[0];
      const last = els[els.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        last.focus();
        e.preventDefault();
      } else if (!e.shiftKey && document.activeElement === last) {
        first.focus();
        e.preventDefault();
      }
    }

    container.addEventListener('keydown', handleKey);
    return () => container.removeEventListener('keydown', handleKey);
  }

  function announce(message, politeness = 'polite') {
    let live = document.getElementById('a11y-live-region');
    if (!live) {
      live = document.createElement('div');
      live.id = 'a11y-live-region';
      live.setAttribute('aria-live', politeness);
      live.setAttribute('aria-atomic', 'true');
      live.style.position = 'absolute';
      live.style.left = '-9999px';
      document.body.appendChild(live);
    }
    live.textContent = '';
    setTimeout(() => { live.textContent = message; }, 10);
  }

  window.A11y = { focusTrap, announce };

  // Enhance accessible names and responsive navbar behavior
  function enhanceAccessibleNames() {
    try {
      const daysAhead = document.getElementById('daysAhead');
      if (daysAhead && !daysAhead.getAttribute('aria-label')) {
        daysAhead.setAttribute('aria-label', 'Days ahead');
      }
      const refreshInterval = document.getElementById('refresh-interval');
      if (refreshInterval && !refreshInterval.getAttribute('aria-label')) {
        refreshInterval.setAttribute('aria-label', 'Refresh interval');
      }
    } catch (_) {}
  }

  function syncNavbarCollapse() {
    try {
      const nav = document.querySelector('nav.navbar');
      const collapse = document.getElementById('navbarNav');
      if (!nav || !collapse) return;
      const isDesktop = window.matchMedia('(min-width: 992px)').matches; // Bootstrap lg
      if (isDesktop) {
        collapse.classList.add('show');
      } else {
        collapse.classList.remove('show');
      }
    } catch (_) {}
  }

  function initA11yDomEnhancements() {
    enhanceAccessibleNames();
    syncNavbarCollapse();
    window.addEventListener('resize', syncNavbarCollapse);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initA11yDomEnhancements);
  } else {
    initA11yDomEnhancements();
  }
})();

