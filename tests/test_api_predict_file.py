import json
import pytest

import app as app_module


def test_predict_file_top_pick_order_by_win_prob(monkeypatch):
    # Always resolve to a dummy path
    monkeypatch.setattr(app_module, 'resolve_race_file_path', lambda fn: '/tmp/dummy.csv', raising=True)
    # Avoid CSV meta enrichment side-effects
    monkeypatch.setattr(app_module, 'enhance_prediction_with_csv_meta', lambda pr, p: pr, raising=True)

    def fake_run_prediction(path: str):
        # Out-of-order probabilities to ensure endpoint sorts by win_prob desc
        return {
            'success': True,
            'predictions': [
                {'dog_name': 'Alpha', 'win_prob': 0.10},
                {'dog_name': 'Bravo', 'win_prob': 0.30},
                {'dog_name': 'Charlie', 'win_prob': 0.20},
            ]
        }

    monkeypatch.setattr(app_module, 'run_prediction_for_race_file', fake_run_prediction, raising=True)

    client = app_module.app.test_client()
    resp = client.post('/api/predict_file', json={'race_file': 'Race 4 - DARW - 2025-08-24.csv'})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    assert data.get('resolved_path') == '/tmp/dummy.csv'

    computed = data.get('computed') or {}
    assert computed.get('top_pick', {}).get('name') == 'Bravo'

    top3 = computed.get('top3') or []
    names = [e.get('name') for e in top3]
    assert names == ['Bravo', 'Charlie', 'Alpha']

    probs = [e.get('win_prob') for e in top3]
    assert probs == sorted(probs, reverse=True)


def test_predict_file_percentage_conversion(monkeypatch):
    # Always resolve to a dummy path
    monkeypatch.setattr(app_module, 'resolve_race_file_path', lambda fn: '/tmp/dummy.csv', raising=True)
    # Avoid CSV meta enrichment side-effects
    monkeypatch.setattr(app_module, 'enhance_prediction_with_csv_meta', lambda pr, p: pr, raising=True)

    def fake_run_prediction_pct(path: str):
        # Provide win probabilities as percentages; endpoint should normalize to 0-1
        return {
            'success': True,
            'predictions': [
                {'dog_name': 'DogA', 'win_probability': 55.0},
                {'dog_name': 'DogB', 'win_probability': 30.0},
                {'dog_name': 'DogC', 'win_probability': 15.0},
            ]
        }

    monkeypatch.setattr(app_module, 'run_prediction_for_race_file', fake_run_prediction_pct, raising=True)

    client = app_module.app.test_client()
    resp = client.post('/api/predict_file', json={'race_file': 'Any.csv'})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True

    comp = data.get('computed') or {}
    top3 = comp.get('top3') or []
    names = [e.get('name') for e in top3]
    assert names == ['DogA', 'DogB', 'DogC']

    probs = [e.get('win_prob') for e in top3]
    assert all(0.0 <= float(p) <= 1.0 for p in probs)
    assert abs(float(probs[0]) - 0.55) < 1e-6


def test_predict_file_not_found_404(monkeypatch):
    # Simulate unresolved path
    monkeypatch.setattr(app_module, 'resolve_race_file_path', lambda fn: None, raising=True)
    client = app_module.app.test_client()
    resp = client.post('/api/predict_file', json={'race_file': 'no_such_file.csv'})
    assert resp.status_code == 404
    data = resp.get_json()
    assert data.get('success') is False
    assert 'not found' in (data.get('error') or '').lower()


def test_predict_file_empty_predictions_degrades_gracefully(monkeypatch):
    # Resolve to a dummy path and return empty predictions
    monkeypatch.setattr(app_module, 'resolve_race_file_path', lambda fn: '/tmp/dummy.csv', raising=True)
    monkeypatch.setattr(app_module, 'enhance_prediction_with_csv_meta', lambda pr, p: pr, raising=True)

    def fake_run_prediction_empty(path: str):
        return {
            'success': True,
            'predictions': []  # no runners
        }

    monkeypatch.setattr(app_module, 'run_prediction_for_race_file', fake_run_prediction_empty, raising=True)

    client = app_module.app.test_client()
    resp = client.post('/api/predict_file', json={'race_file': 'empty.csv'})

    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get('success') is True
    # No computed top picks when predictions are empty
    assert ('computed' not in data) or (not data.get('computed'))

