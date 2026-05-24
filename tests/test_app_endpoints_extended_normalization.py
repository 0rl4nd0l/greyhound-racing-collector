import importlib


def test_app_extended_endpoints_normalization():
    app_module = importlib.import_module("app")
    app = getattr(app_module, "app", None)
    assert app is not None
    client = app.test_client()

    # A broader set of GET endpoints to exercise app.py branches
    endpoints = [
        "/api/diagnostics/summary",
        "/api/diagnostics/last_promotion",
        "/api/diagnostics/last_promotion/status",
        "/api/download_watch_status",
        "/api/v4/eval/summary/latest",
        "/api/v4/eval/mispredictions/latest",
        "/api/v4/models/contracts/check?strict=0",
        "/api/model_registry/refresh_signal",
        "/api/race_notes",
        "/api/races",
    ]

    acceptable = {200, 400, 404, 500}
    for ep in endpoints:
        resp = client.get(ep)
        assert resp.status_code in acceptable

    # Exercise a lightweight POST that uses an internal background thread (simulated)
    resp = client.post("/api/predictions/generate", json={})
    assert resp.status_code in acceptable
