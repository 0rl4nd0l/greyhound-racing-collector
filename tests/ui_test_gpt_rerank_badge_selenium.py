#!/usr/bin/env python3
"""
Selenium test: verify GPT Rerank badge renders in interactive-races summary card.
This test:
- Starts a local Flask server on port 5555
- Opens /interactive-races
- Enables the test export hook
- Calls window.displayPredictionResults with a synthetic result that includes gpt_rerank.applied
- Verifies the "GPT Rerank" badge appears
"""

import os
import sys
import time
import threading
import pytest

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from drivers import get_chrome_driver
    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False

from werkzeug.serving import make_server
from app import app as flask_app
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


@pytest.fixture(scope="function")
def flask_server():
    # Configure app for testing
    flask_app.config.update({
        'TESTING': True,
        'DEBUG': False,
        'SERVER_NAME': 'localhost:5555'
    })
    server = make_server('localhost', 5555, flask_app)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)
    try:
        yield 'http://localhost:5555'
    finally:
        try:
            server.shutdown()
        except Exception:
            pass


@pytest.fixture(scope="function")
def driver():
    if not SELENIUM_AVAILABLE:
        pytest.skip("Selenium WebDriver not available")
    import shutil as _shutil
    # Require a system ChromeDriver to avoid network downloads in constrained environments
    chromedriver_path = _shutil.which('chromedriver')
    if not chromedriver_path:
        pytest.skip("System chromedriver not found; skipping UI badge test (install via: brew install chromedriver)")
    # Ensure JS is enabled by default; DISABLE_JS gate is supported in drivers.py
    os.environ.pop('DISABLE_JS', None)
    _driver = None
    try:
        _driver = get_chrome_driver(headless=True)
        _driver.set_window_size(1280, 900)
        _driver.implicitly_wait(5)
        yield _driver
    finally:
        if _driver:
            try:
                _driver.quit()
            except Exception:
                pass


@pytest.mark.skipif(not SELENIUM_AVAILABLE, reason="Selenium WebDriver not available")
def test_gpt_rerank_badge_summary(driver, flask_server):
    base = flask_server
    driver.get(f"{base}/interactive-races")

    # Wait for document ready (basic element like body exists)
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )

    # Enable test export and call the display function with a synthetic result
    driver.execute_script("window.ENABLE_UI_EXPORTS = true;")

    js = """
      (function(){
        try {
          // Ensure the prediction results container exists
          var container = document.getElementById('prediction-results-container');
          if (!container) {
            var main = document.querySelector('main') || document.body;
            var wrapper = document.createElement('div');
            wrapper.id = 'prediction-results-container';
            wrapper.innerHTML = '<div class="card"><div class="card-header">Prediction Results</div><div class="card-body"><div id="prediction-results-body"></div></div></div>';
            main.appendChild(wrapper);
          }
          // Build synthetic result with GPT rerank
          var results = [{
            success: true,
            predictions: [
              { dog_name: 'Alpha', win_prob: 0.55 },
              { dog_name: 'Bravo', win_prob: 0.31 }
            ],
            gpt_rerank: { applied: true, alpha: 0.6, tokens_used: 64 },
            race_filename: 'selenium_race.csv',
            predictor_used: 'PredictionPipelineV4'
          }];
          if (typeof window.displayPredictionResults === 'function') {
            window.displayPredictionResults(results);
            return true;
          }
          return false;
        } catch(e) { return String(e); }
      })();
    """

    ok = driver.execute_script(js)
    assert ok is True, "displayPredictionResults not available (test export hook failed)"

    # Wait until badge appears
    badge = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, 'span.badge.bg-info'))
    )

    assert badge is not None
    assert 'GPT Rerank' in badge.text
    title = badge.get_attribute('title') or ''
    assert 'GPT rerank applied' in title
    # Note: alpha and tokens placeholders are included; exact numeric values may vary


