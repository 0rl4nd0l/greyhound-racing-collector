#!/usr/bin/env python3
"""
Standalone Selenium check for GPT Rerank badge rendering in interactive-races.
- Starts Flask app on localhost:5560
- Opens /interactive-races
- Uses test export hook to call displayPredictionResults with gpt_rerank.applied
- Prints PASS/FAIL summary
Note: Requires system chromedriver available on PATH.
"""
import os
import sys
import threading
import time

# Ensure project root on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from werkzeug.serving import make_server

try:
    from drivers import get_chrome_driver
except Exception as e:
    print(f"Selenium unavailable: {e}")
    sys.exit(2)

import shutil

CHROMEDRIVER = shutil.which("chromedriver")
if not CHROMEDRIVER:
    print(
        "SKIP: system chromedriver not found (install via: brew install chromedriver)"
    )
    sys.exit(3)

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from app import app as flask_app


def start_server():
    flask_app.config.update(
        {"TESTING": True, "DEBUG": False, "SERVER_NAME": "localhost:5560"}
    )
    server = make_server("localhost", 5560, flask_app)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(2)
    return server


def main():
    # Ensure JS is enabled
    os.environ.pop("DISABLE_JS", None)

    server = start_server()
    base = "http://localhost:5560"
    driver = None
    try:
        driver = get_chrome_driver(headless=True)
        driver.set_window_size(1280, 900)
        driver.get(f"{base}/interactive-races")
        WebDriverWait(driver, 10).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        # Enable test export and render a synthetic result
        driver.execute_script("window.ENABLE_UI_EXPORTS = true;")
        ok = driver.execute_script(
            """
            (function(){
              try {
                var container = document.getElementById('prediction-results-container');
                if (!container) {
                  var main = document.querySelector('main') || document.body;
                  var wrapper = document.createElement('div');
                  wrapper.id = 'prediction-results-container';
                  wrapper.innerHTML = '<div class="card"><div class="card-header">Prediction Results</div><div class="card-body"><div id="prediction-results-body"></div></div></div>';
                  main.appendChild(wrapper);
                }
                var results = [{
                  success: true,
                  predictions: [ { dog_name: 'Alpha', win_prob: 0.57 } ],
                  gpt_rerank: { applied: true, alpha: 0.62, tokens_used: 88 },
                  race_filename: 'standalone_selenium.csv',
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
        )
        if ok is not True:
            print(
                "FAIL: displayPredictionResults not exported (set window.ENABLE_UI_EXPORTS=true)"
            )
            return 1

        badge = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "span.badge.bg-info"))
        )
        if badge and "GPT Rerank" in (badge.text or ""):
            title = badge.get_attribute("title") or ""
            if "GPT rerank applied" in title:
                print("PASS: GPT Rerank badge rendered with tooltip")
                return 0
        print("FAIL: GPT Rerank badge not found")
        return 1
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    finally:
        try:
            server.shutdown()
        except Exception:
            pass
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    code = main()
    sys.exit(code)
