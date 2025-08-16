import os
import sys
import importlib
import types
import pytest

# Ensure we import app after setting env to prediction_only
@pytest.fixture(autouse=True)
def set_prediction_only_env(monkeypatch):
    monkeypatch.setenv('PREDICTION_IMPORT_MODE', 'prediction_only')
    # Ensure strict guard so violations raise
    monkeypatch.setenv('MODULE_GUARD_STRICT', '1')
    # Disable live scraping flags by default in tests
    monkeypatch.setenv('ENABLE_RESULTS_SCRAPERS', '0')
    monkeypatch.setenv('ENABLE_LIVE_SCRAPING', '0')

    # Provide a stub for ml_system_v4 to avoid import-time IndentationError in real file during tests
    if 'ml_system_v4' not in sys.modules:
        stub = types.ModuleType('ml_system_v4')
        class MLSystemV4:  # minimal stub
            def __init__(self, *args, **kwargs):
                pass
        def train_leakage_safe_model(*args, **kwargs):
            return None
        stub.MLSystemV4 = MLSystemV4
        stub.train_leakage_safe_model = train_leakage_safe_model
        sys.modules['ml_system_v4'] = stub


def test_predict_page_post_does_not_trigger_results_scraper(monkeypatch):
    # Import the Flask app
    import app as app_module
    app = app_module.app
    app.config['TESTING'] = True

    # Sanity: ensure the greyhound recorder scraper module is not importable during the request
    # We simulate a prior accidental import to ensure the guard will clear it before prediction
    sys.modules['src.collectors.the_greyhound_recorder_scraper'] = types.ModuleType('src.collectors.the_greyhound_recorder_scraper')

    client = app.test_client()

    # Create a dummy CSV in upcoming dir
    upcoming_dir = app_module.UPCOMING_DIR
    os.makedirs(upcoming_dir, exist_ok=True)
    race_filename = 'Race 1 - TESTVENUE - 2025-08-04.csv'
    with open(os.path.join(upcoming_dir, race_filename), 'w') as f:
        f.write('Dog Name,Box,Weight,Trainer\n1. Test Dog,1,30.0,Trainer A\n')

    # POST to /predict_page should NOT try to load results scrapers and should not error with module guard
    resp = client.post('/predict_page', data={'race_files': race_filename, 'action': 'single'}, follow_redirects=True)
    assert resp.status_code in (200, 302)
    # The page should render or redirect back without module guard error message
    assert b"Results scraping module loaded" not in resp.data


def test_module_guard_blocks_when_results_scraper_loaded(monkeypatch):
    # Import a fresh app module to reset state
    import importlib
    app_module = importlib.reload(importlib.import_module('app'))
    app = app_module.app
    app.config['TESTING'] = True

    # Force-load a disallowed results scraper module in sys.modules to trigger the guard
    sys.modules['src.collectors.fasttrack_scraper'] = types.ModuleType('src.collectors.fasttrack_scraper')

    client = app.test_client()

    # Prepare a valid CSV
    upcoming_dir = app_module.UPCOMING_DIR
    os.makedirs(upcoming_dir, exist_ok=True)
    race_filename = 'Race 2 - TESTVENUE - 2025-08-04.csv'
    with open(os.path.join(upcoming_dir, race_filename), 'w') as f:
        f.write('Dog Name,Box,Weight,Trainer\n1. Another Dog,1,30.0,Trainer B\n')

    resp = client.post('/predict_page', data={'race_files': race_filename}, follow_redirects=False)
    # Expect a redirect due to guard flash/redirect path
    assert resp.status_code in (302, 303)
    # Follow redirect and check for flash message in final page
    follow = client.get('/predict_page')
    assert b"Prediction blocked due to unsafe modules" in follow.data


def test_manual_prediction_flow_works_without_tgr_scraper(monkeypatch):
    # Ensure TheGreyhoundRecorderScraper is NOT importable
    if 'src.collectors.the_greyhound_recorder_scraper' in sys.modules:
        del sys.modules['src.collectors.the_greyhound_recorder_scraper']

    import importlib
    app_module = importlib.reload(importlib.import_module('app'))
    app = app_module.app
    app.config['TESTING'] = True

    client = app.test_client()

    # Prepare a valid CSV
    upcoming_dir = app_module.UPCOMING_DIR
    os.makedirs(upcoming_dir, exist_ok=True)
    race_filename = 'Race 3 - TESTVENUE - 2025-08-04.csv'
    with open(os.path.join(upcoming_dir, race_filename), 'w') as f:
        f.write('Dog Name,Box,Weight,Trainer\n1. Third Dog,1,30.0,Trainer C\n')

    resp = client.post('/predict_page', data={'race_files': race_filename}, follow_redirects=True)
    # Either it renders the results or at least returns the page without any results-scraper related errors
    assert resp.status_code in (200, 302)
    assert b"Results scraping module loaded" not in resp.data
    # Also verify we did not import the TGR scraper as a side effect
    assert 'src.collectors.the_greyhound_recorder_scraper' not in sys.modules

