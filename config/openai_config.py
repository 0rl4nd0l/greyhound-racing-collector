"""
Centralized OpenAI configuration
- Provides a single source of truth for model selection and defaults
- Avoids hardcoded model strings across the codebase
"""

import os
from dataclasses import dataclass
from typing import Optional

# Reasonable defaults; can be overridden via environment
# Prefer fast/affordable model by default; upgrade per workload if needed
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
DEFAULT_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2000"))


@dataclass(frozen=True)
class OpenAIConfig:
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


def get_openai_config(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> OpenAIConfig:
    return OpenAIConfig(
        model=model or DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE if temperature is None else float(temperature),
        max_tokens=DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens),
    )
