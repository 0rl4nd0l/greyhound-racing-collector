import re

from src.ai.prompts import DOMAIN_RULES, system_prompt


def test_domain_rules_wording():
    text = DOMAIN_RULES
    assert "historical data" in text
    assert "race data" in text
    assert "Never infer winners" in text or "winners must" in text


def test_system_prompt_includes_rules_for_advisory():
    p = system_prompt("advisory")
    # Must include domain rules
    assert "historical data" in p
    assert "race data" in p
    # Ensure winner source rule present
    assert re.search(
        r"winners?\s+must\s+come\s+from\s+the\s+race\s+webpage", p, re.IGNORECASE
    )


def test_system_prompt_roles_minimal_length():
    # Keep prompts short/testable
    for role in ("advisory", "analyst", "bettor", "daily"):
        p = system_prompt(role)
        assert len(p) < 400, "Prompt should be concise"
