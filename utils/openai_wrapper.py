"""
Lightweight OpenAI wrapper to standardize usage across the codebase.

- respond_text: one-off prompt using Responses API when available; falls back to chat.completions
- respond_json: like respond_text but parses JSON output safely
- chat: multi-message chat using chat.completions
- Centralizes model/temperature/max_tokens via config.openai_config
- Includes minimal retry/backoff for transient 429/5xx errors
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import time

from config.openai_config import get_openai_config, OpenAIConfig


@dataclass
class OpenAIResponse:
    text: str
    model: str
    usage: Dict[str, Any]


class OpenAIWrapper:
    def __init__(self, client: Any, config: Optional[OpenAIConfig] = None):
        self.client = client
        self.cfg = config or get_openai_config()

    def _with_retries(self, func, *, retries: int = 2, base_delay: float = 0.2):
        last_err = None
        for attempt in range(retries + 1):
            try:
                return func()
            except Exception as e:  # pragma: no cover - branch inspected by tests
                last_err = e
                # Inspect for HTTP-like status on error objects
                status = getattr(e, "status", None) or getattr(e, "status_code", None)
                msg = str(e).lower()
                transient = status in (429, 500, 502, 503, 504) or any(s in msg for s in ["rate limit", "timeout", "temporarily"])
                if attempt < retries and transient:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        raise last_err  # safety

    def respond_text(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> OpenAIResponse:
        """One-shot text response using Responses API if available, else chat.completions.

        For gpt-5 models, prefer the Responses API. If falling back to chat.completions,
        use 'max_tokens' (Chat Completions) and 'max_output_tokens' (Responses API).
        """
        temp = self.cfg.temperature if temperature is None else float(temperature)
        max_tok = self.cfg.max_tokens if max_tokens is None else int(max_tokens)
        mdl = model or self.cfg.model
        is_gpt5 = isinstance(mdl, str) and (mdl.startswith("gpt-5") or "gpt-5" in mdl)

        # Try Responses API first
        text = None
        usage: Dict[str, Any] = {"total_tokens": None}

        def call_responses():
            responses = getattr(self.client, "responses", None)
            if responses and hasattr(responses, "create"):
                kwargs = dict(
                    model=mdl,
                    input=prompt if not system else [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=max_tok,
                )
                # For gpt-5, temperature must be default (1). Prefer omitting the parameter entirely.
                if not is_gpt5:
                    kwargs["temperature"] = temp
                # Hint JSON-only formatting when caller appended the JSON directive
                try:
                    if system and "Return valid JSON only." in system:
                        kwargs["response_format"] = {"type": "json_object"}
                except Exception:
                    pass
                try:
                    return self.client.responses.create(**kwargs)
                except TypeError:
                    # Retry without response_format for older SDKs
                    kwargs.pop("response_format", None)
                    return self.client.responses.create(**kwargs)
            raise AttributeError("responses.create not available")

        try:
            resp = self._with_retries(call_responses)
            # Extract text from Responses API
            if hasattr(resp, "output_text"):
                text = resp.output_text
            elif hasattr(resp, "output") and resp.output:
                parts = []
                for item in resp.output:
                    if getattr(item, "type", "") == "output_text":
                        parts.append(getattr(item, "text", ""))
                text = "".join(parts) if parts else None
            usage = getattr(resp, "usage", usage)
        except Exception:
            text = None

        if text is None:
            # Fallback to Chat Completions
            def call_chat():
                messages: List[Dict[str, str]] = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                # Choose correct token parameter based on model family
                token_param = "max_tokens"
                kwargs = dict(
                    model=mdl,
                    messages=messages,
                )
                # For gpt-5, omit temperature to use default (1) as per API constraints
                if not is_gpt5:
                    kwargs["temperature"] = temp
                kwargs[token_param] = max_tok
                # Enforce JSON output if caller indicated JSON requirement
                try:
                    if system and "Return valid JSON only." in system:
                        kwargs["response_format"] = {"type": "json_object"}
                except Exception:
                    pass
                try:
                    return self.client.chat.completions.create(**kwargs)
                except TypeError:
                    # Retry without response_format if SDK doesn't support it
                    kwargs.pop("response_format", None)
                    return self.client.chat.completions.create(**kwargs)

            resp = self._with_retries(call_chat)
            text = resp.choices[0].message.content
            usage = getattr(resp, "usage", usage)

        return OpenAIResponse(text=text, model=mdl, usage=dict(usage) if usage else {})

    def respond_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Respond and parse JSON; returns a dict. Raises ValueError on invalid JSON."""
        sys = (system + "\nReturn valid JSON only.") if system else "Return valid JSON only."
        resp = self.respond_text(
            prompt=prompt,
            system=sys,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        try:
            return json.loads(resp.text)
        except Exception as e:
            raise ValueError(f"Invalid JSON from model: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> OpenAIResponse:
        temp = self.cfg.temperature if temperature is None else float(temperature)
        max_tok = self.cfg.max_tokens if max_tokens is None else int(max_tokens)
        mdl = model or self.cfg.model
        is_gpt5 = isinstance(mdl, str) and (mdl.startswith("gpt-5") or "gpt-5" in mdl)

        def call_chat():
            token_param = "max_tokens"
            kwargs = dict(
                model=mdl,
                messages=messages,
            )
            # For gpt-5, omit temperature to default to 1 (only supported value)
            if not is_gpt5:
                kwargs["temperature"] = temp
            kwargs[token_param] = max_tok
            return self.client.chat.completions.create(**kwargs)

        resp = self._with_retries(call_chat)
        text = resp.choices[0].message.content
        usage = getattr(resp, "usage", {"total_tokens": None})
        return OpenAIResponse(text=text, model=mdl, usage=dict(usage) if usage else {})
