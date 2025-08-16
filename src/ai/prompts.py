"""
Shared prompt fragments for AI interactions.

- Centralizes domain rules and role-specific system prompts
- Keeps prompts short, consistent, and testable
"""
from __future__ import annotations

from typing import Literal

# Project domain rules (short, explicit)
DOMAIN_RULES = (
    "Use 'historical data' for form guide CSVs and past performance. "
    "Use 'race data' for scraped data for the race itself (weather, winners, etc.). "
    "Never infer winners from historical data; winners must come from the race webpage scrape."
)


def system_prompt(role: Literal["advisory", "analyst", "bettor", "daily"] = "analyst") -> str:
    """Return a concise system prompt with embedded domain rules.

    Roles:
    - advisory: data quality/advisory summaries
    - analyst: race analysis and reasoning
    - bettor: betting strategy guidance
    - daily: day-level portfolio/overview
    """
    base = {
        "advisory": "You analyze prediction quality and produce concise advisories.",
        "analyst": "You analyze greyhound races with statistical rigor.",
        "bettor": "You propose sustainable betting strategies.",
        "daily": "You summarize daily racing insights and portfolio guidance.",
    }[role]

    # Keep short and testable
    return (
        f"{base} Follow domain rules: {DOMAIN_RULES} "
        "Keep outputs brief and structured when requested."
    )


def instructions_for_json_output() -> str:
    """Short instruction snippet for structured JSON outputs when needed."""
    return (
        "Return valid JSON only. Include keys: 'summary', 'highlights', 'risks' where applicable."
    )
