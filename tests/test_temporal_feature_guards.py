import os
import re

import pytest


def test_forbidden_postrace_fields_are_referenced_in_builders():
    # We avoid importing heavy modules; instead scan source files for forbidden fields
    repo_root = os.getcwd()
    candidates = [
        os.path.join(repo_root, "temporal_feature_builder.py"),
        os.path.join(repo_root, "temporal_feature_builder_optimized.py"),
        os.path.join(repo_root, "ml_system_v4.py"),
        os.path.join(repo_root, "src", "temporal_feature_builder.py"),
        os.path.join(repo_root, "src", "temporal_feature_builder_optimized.py"),
    ]

    files = [p for p in candidates if os.path.exists(p)]
    if not files:
        pytest.skip("Temporal feature builder sources not present in repository")

    patterns = [
        re.compile(r"finish_position", re.IGNORECASE),
        re.compile(r"margin", re.IGNORECASE),
        re.compile(r"sectional", re.IGNORECASE),
    ]

    matched_any = False
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Ensure the forbidden signals appear in the code (as part of guards or comments)
        for pat in patterns:
            assert pat.search(
                text
            ), f"Expected pattern {pat.pattern} in {os.path.basename(path)}"
        matched_any = True

    assert matched_any, "No builder files scanned"
