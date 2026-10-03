"""Process-start policy for the isolated Operator UI R3 runtime."""
from __future__ import annotations

import os
from collections.abc import Mapping


def legacy_startup_disabled(environment: Mapping[str, str] | None = None) -> bool:
    """Return whether this process is the generated, connected R3 service."""
    values = os.environ if environment is None else environment
    return (
        values.get("OPERATOR_UI_CONNECTED_MODE") == "1"
        and values.get("OPERATOR_UI_R3_PROFILE") == "repository-v1"
        and values.get("OPERATOR_UI_DISABLE_LEGACY_STARTUP") == "1"
    )
