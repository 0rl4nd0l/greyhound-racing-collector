// Minimal theme utilities to avoid 404/500s and provide basic functionality
(function () {
  const STORAGE_KEY = 'ui.theme';

  function applyTheme(theme) {
    const t = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', t);
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

  // Auto-init on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
  } else {
    initTheme();
  }
})();

