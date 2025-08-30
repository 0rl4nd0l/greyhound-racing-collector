#!/usr/bin/env python3
import os
import json
from pathlib import Path

# Ensure test mode so heavy services defer initialization
os.environ['TESTING'] = '1'

# Import the Flask app after setting TESTING
from app import app  # noqa: E402


def main():
    client = app.test_client()

    # Prepare a minimal CSV in the configured upcoming directory
    upcoming_dir = app.config.get('UPCOMING_DIR', './upcoming_races')
    Path(upcoming_dir).mkdir(parents=True, exist_ok=True)
    filename = 'smoke_test_race.csv'
    fpath = Path(upcoming_dir) / filename

    # Minimal 8-runner CSV
    rows = [
        'Dog Name,Box,Weight,Trainer',
        '1. Smoke Alpha,1,30.1,Trainer A',
        '2. Smoke Bravo,2,29.8,Trainer B',
        '3. Smoke Charlie,3,31.0,Trainer C',
        '4. Smoke Delta,4,30.5,Trainer D',
        '5. Smoke Echo,5,29.9,Trainer E',
        '6. Smoke Foxtrot,6,30.2,Trainer F',
        '7. Smoke Golf,7,30.0,Trainer G',
        '8. Smoke Hotel,8,30.3,Trainer H',
    ]
    fpath.write_text('\n'.join(rows) + '\n', encoding='utf-8')

    # Call the enhanced endpoint
    resp = client.post(
        '/api/predict_single_race_enhanced',
        json={'race_filename': filename},
        headers={'Content-Type': 'application/json'}
    )

    try:
        data = resp.get_json(force=True)
    except Exception:
        data = None

    # Build concise summary
    summary = {
        'status_code': resp.status_code,
        'success': bool(data and data.get('success')),
        'degraded': bool(data.get('degraded')) if isinstance(data, dict) else None,
        'top_level_predictions_count': (len(data.get('predictions')) if isinstance(data, dict) and isinstance(data.get('predictions'), list) else 0),
        'has_gpt_rerank_key': (isinstance(data, dict) and ('gpt_rerank' in data)),
        'predictor_used': (data.get('predictor_used') if isinstance(data, dict) else None),
        'message': (data.get('message') if isinstance(data, dict) else None),
        # Optional: surface alpha if present for quick visibility
        'gpt_rerank_alpha': (data.get('gpt_rerank', {}) or {}).get('alpha') if isinstance(data, dict) else None,
    }

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

