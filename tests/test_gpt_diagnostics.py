import json

# Use the shared client fixture from tests/conftest.py


def test_gpt_diagnostics_endpoint_basic(client):
    """Verify /api/gpt/diagnostics returns a non-mutating summary structure.
    Ensures keys exist and types are correct even when no files exist.
    """
    resp = client.get("/api/gpt/diagnostics")
    assert resp.status_code == 200
    data = resp.get_json() if hasattr(resp, "get_json") else json.loads(resp.data)

    assert data.get("success") is True
    diag = data.get("diagnostics")
    assert isinstance(diag, dict)

    # Expected keys
    for key in (
        "json_files_count",
        "archives_count",
        "total_json_bytes",
        "total_archives_bytes",
        "latest_json_path",
        "latest_archive_path",
    ):
        assert key in diag, f"Missing diagnostics key: {key}"

    # Type checks
    assert isinstance(diag["json_files_count"], int)
    assert isinstance(diag["archives_count"], int)
    assert isinstance(diag["total_json_bytes"], int)
    assert isinstance(diag["total_archives_bytes"], int)
    # paths may be None when no files exist
    assert (diag["latest_json_path"] is None) or isinstance(
        diag["latest_json_path"], str
    )
    assert (diag["latest_archive_path"] is None) or isinstance(
        diag["latest_archive_path"], str
    )


def test_system_status_includes_gpt_diagnostics(client):
    """Verify /api/system_status includes a gpt_diagnostics summary."""
    resp = client.get("/api/system_status")
    # Flask may return dict directly; test client still wraps it
    assert resp.status_code == 200 or resp.status_code == 304
    # If 304 due to conditional, re-request without condition
    if resp.status_code == 304:
        resp = client.get("/api/system_status")
        assert resp.status_code == 200

    data = resp.get_json() if hasattr(resp, "get_json") else json.loads(resp.data)

    assert data.get("success") is True
    assert "gpt_diagnostics" in data, "system_status missing gpt_diagnostics"
    gpt_diag = data.get("gpt_diagnostics") or {}

    assert isinstance(gpt_diag, dict)
    # keys exist with integer counts
    assert "json_files_count" in gpt_diag and isinstance(
        gpt_diag.get("json_files_count"), int
    )
    assert "archives_count" in gpt_diag and isinstance(
        gpt_diag.get("archives_count"), int
    )
