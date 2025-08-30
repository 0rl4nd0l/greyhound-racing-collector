#!/usr/bin/env python3
import os
import sys

# Minimize side-effects during import
os.environ.setdefault('WATCH_DOWNLOADS', '0')
os.environ.setdefault('WATCH_UPCOMING', '0')
os.environ.setdefault('DISABLE_STARTUP_GUARD', '1')
os.environ.setdefault('TESTING', '1')

# Ensure current project directory is on sys.path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

try:
    import app as app_module
except Exception as e:
    print(f"IMPORT_FAIL: {e}")
    sys.exit(1)

client = app_module.app.test_client()

failures = []

def check(mode: str):
    os.environ['UI_MODE'] = mode
    # Hit home page
    resp = client.get('/')
    if resp.status_code != 200:
        failures.append(f"/{mode}: status {resp.status_code}")
        return
    html = resp.get_data(as_text=True)
    # Basic sanity
    if '<nav' not in html:
        failures.append(f"{mode}: navbar missing")
    # Mode-specific expectations
    if mode == 'simple':
        # Expect minimal links
        if '/upcoming' not in html:
            failures.append("simple: missing Upcoming link")
        if '/predict_page' not in html:
            failures.append("simple: missing Predict link")
        # Should not show heavy advanced controls
        if '/ml_dashboard' in html or 'Run Diagnostics' in html:
            failures.append("simple: advanced ML controls visible")
    else:  # advanced
        # Expect dropdowns and ML controls to be present
        if 'id="racesDropdown"' not in html:
            failures.append("advanced: Races dropdown missing")
        if '/ml_dashboard' not in html:
            failures.append("advanced: ML Dashboard link missing")

# Run checks
check('simple')
check('advanced')

# Also ensure upcoming page renders in both modes
for mode in ('simple', 'advanced'):
    os.environ['UI_MODE'] = mode
    r = client.get('/upcoming')
    if r.status_code != 200:
        failures.append(f"/upcoming {mode}: status {r.status_code}")

if failures:
    print("SMOKE_TEST_FAIL")
    for f in failures:
        print(f" - {f}")
    sys.exit(2)
else:
    print("SMOKE_TEST_PASS")
    sys.exit(0)

