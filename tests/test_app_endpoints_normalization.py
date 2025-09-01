import importlib


def test_app_basic_endpoints_normalization():
    app_module = importlib.import_module("app")
    app = getattr(app_module, "app", None)
    assert app is not None
    client = app.test_client()

    # Hit lightweight endpoints that do not require DB writes
    endpoints = [
        "/api/health",
        "/health",
        "/ping",
        "/api/server-port",
        "/api/v4/models/contracts",
        "/api/model_health",
    ]

    for ep in endpoints:
        resp = client.get(ep)
        # Accept 200; for contracts endpoint 200 even if no directory exists
        assert resp.status_code in (200,)
