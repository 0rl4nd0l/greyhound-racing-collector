import json
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize("sample_name", [
    "tests/sample_csv_data/test_race_2.csv",
    "tests/sample_csv_data/test_race_with_outcomes.csv",
])
def test_api_predict_rank_monotonic(sample_name, tmp_path):
    """
    API-level regression: ensure the top pick equals the max probability and
    predicted_rank is consistent with descending probability, when predictions are present.
    """
    # Import here to avoid module import side-effects during collection
    import app as app_module

    app = app_module.app
    app.config["TESTING"] = True

    # Prepare UPCOMING_DIR with the sample file
    src = Path(sample_name)
    assert src.exists(), f"Missing test sample: {src}"
    dst = tmp_path / src.name
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    # Point the app to our temp directory
    app.config["UPCOMING_DIR"] = str(tmp_path)

    client = app.test_client()
    resp = client.post(
        "/api/predict_single_race_enhanced",
        data=json.dumps({"race_filename": src.name}),
        content_type="application/json",
    )

    assert resp.status_code == 200, resp.data
    data = resp.get_json() or {}

    # Degraded responses are allowed in some environments; skip strict checks in that case
    if data.get("degraded"):
        pytest.skip("Degraded prediction path returned (environment-dependent)")

    # Prefer nested prediction payload if present
    pred = data.get("prediction") if isinstance(data.get("prediction"), dict) else data
    predictions = pred.get("predictions") if isinstance(pred, dict) else []
    if not predictions:
        pytest.skip("No predictions produced by backend (environment-dependent)")

    # Extract probabilities and ranks
    probs = []
    ranks = []
    for p in predictions:
        if not isinstance(p, dict):
            continue
        wp = (
            p.get("win_prob")
            or p.get("win_prob_norm")
            or p.get("normalized_win_probability")
            or p.get("win_probability")
        )
        if wp is None:
            continue
        try:
            probs.append(float(wp))
            ranks.append(p.get("predicted_rank"))
        except Exception:
            continue

    assert len(probs) >= 2, "Need at least two runners with probabilities"

    # Top pick equals argmax probability
    top = predictions[0]
    top_wp = (
        top.get("win_prob")
        or top.get("win_prob_norm")
        or top.get("normalized_win_probability")
        or top.get("win_probability")
        or 0.0
    )
    assert float(top_wp) == max(probs), "Top pick is not the highest probability"

    # Monotone ranks with descending probability (when ranks are present)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    ordered_ranks = [ranks[i] for i in order if ranks[i] is not None]
    if ordered_ranks:
        assert all(
            ordered_ranks[i] <= ordered_ranks[i + 1] for i in range(len(ordered_ranks) - 1)
        ), f"Ranks not monotone: {ordered_ranks}"

