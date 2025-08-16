Outdated OpenAI integration scripts archived here.

What moved and why:
- gpt_prediction_enhancer.py
- openai_connectivity_verifier.py
- openai_enhanced_analyzer.py

These modules used legacy direct OpenAI usage patterns and have been superseded by a centralized wrapper and config.

Use this instead:
- utils/openai_wrapper.py — unified API for chat/responses with retry and config
- config/openai_config.py — single source for model/temperature/max_tokens

Notes:
- Any new AI calls should import OpenAIWrapper and get_openai_config.
- If you need connectivity checks, consider lightweight health checks via the wrapper rather than bespoke scripts.

Example usage:
from utils.openai_wrapper import OpenAIWrapper
from config.openai_config import get_openai_config

# obtain a client (e.g., from your service wiring)
wrapper = OpenAIWrapper(client, get_openai_config())
resp = wrapper.chat([
    {"role": "system", "content": "You are an assistant."},
    {"role": "user", "content": "Hello"},
])
print(resp.text)

