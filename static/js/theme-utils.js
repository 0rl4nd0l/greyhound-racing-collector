// Minimal theme utilities to avoid 404/500s and provide basic functionality
(function () {
  const STORAGE_KEY = 'theme';

  function applyTheme(theme) {
    const t = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
    // Update icon and button title for accessibility
    try {
      var icon = document.getElementById('theme-icon');
      if (icon) {
        icon.className = t === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
      }
      var btn = document.getElementById('theme-toggle');
      if (btn) {
        btn.title = t === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
        btn.setAttribute('aria-label', btn.title);
      }
    } catch (_) {}
    // Update theme badge text and classes
    try {
      var tBadge = document.getElementById('theme-badge');
      if (tBadge) {
        tBadge.textContent = 'Theme: ' + (t === 'dark' ? 'Dark' : 'Light');
        // reset base classes, then apply theme variant
        tBadge.className = 'badge ms-2 ' + (t === 'dark' ? 'bg-dark text-light' : 'bg-light text-dark');
        tBadge.setAttribute('title', 'Current theme');
      }
    } catch(_) {}
  }

  function getStoredTheme() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (_) {
      return null;
    }
  }

  function setStoredTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (_) {}
  }

  function detectPreferredTheme() {
    try {
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
    } catch (_) {
      return 'light';
    }
  }

  function initTheme() {
    const saved = getStoredTheme();
    const initial = saved || detectPreferredTheme();
    applyTheme(initial);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    setStoredTheme(next);
  }

  // Expose minimal API
  window.ThemeUtils = {
    init: initTheme,
    toggle: toggleTheme,
    apply: applyTheme,
  };

  // Auto-init on DOM ready; also bind toggle if button exists
  function bindToggle(){
    try {
      var btn = document.getElementById('theme-toggle');
      if (btn && !btn.dataset.boundTheme) {
        btn.addEventListener('click', function(e){
          e.preventDefault();
          if (window.ThemeUtils && typeof window.ThemeUtils.toggle === 'function') {
            window.ThemeUtils.toggle();
          } else if (typeof window.toggleTheme === 'function') {
            window.toggleTheme();
          }
        }, { passive: true });
        btn.dataset.boundTheme = '1';
      }
    } catch (_) {}
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function(){ initTheme(); bindToggle(); });
  } else {
    initTheme();
    bindToggle();
  }
})();

