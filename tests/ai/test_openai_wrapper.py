import json
from unittest.mock import MagicMock
import types
import pytest

from utils.openai_wrapper import OpenAIWrapper


class DummyHTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status = status


@pytest.fixture
def mock_client_success_text():
    client = types.SimpleNamespace()
    # Responses API style
    resp_obj = types.SimpleNamespace(output_text="hello world", usage={"total_tokens": 10})
    client.responses = types.SimpleNamespace(create=MagicMock(return_value=resp_obj))
    # Chat fallback
    choice_msg = types.SimpleNamespace(content="unused")
    choice = types.SimpleNamespace(message=choice_msg)
    chat_resp = types.SimpleNamespace(choices=[choice], usage={"total_tokens": 10})
    client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=MagicMock(return_value=chat_resp)))
    return client


@pytest.fixture
def mock_client_rate_limit_then_ok():
    client = types.SimpleNamespace()

    # First two attempts raise 429, then succeed
    calls = {"n": 0}

    def flappy_create(**kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise DummyHTTPError(429)
        return types.SimpleNamespace(output_text="recovered", usage={"total_tokens": 12})

    client.responses = types.SimpleNamespace(create=MagicMock(side_effect=flappy_create))

    # Chat fallback (shouldn't be used here)
    choice_msg = types.SimpleNamespace(content="chat recovered")
    choice = types.SimpleNamespace(message=choice_msg)
    chat_resp = types.SimpleNamespace(choices=[choice], usage={"total_tokens": 12})
    client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=MagicMock(return_value=chat_resp)))
    return client


@pytest.fixture
def mock_client_force_chat():
    client = types.SimpleNamespace()
    # Responses missing -> force chat path
    client.responses = None
    choice_msg = types.SimpleNamespace(content="from chat")
    choice = types.SimpleNamespace(message=choice_msg)
    chat_resp = types.SimpleNamespace(choices=[choice], usage={"total_tokens": 8})
    client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=MagicMock(return_value=chat_resp)))
    return client


def test_respond_text_responses_ok(mock_client_success_text):
    w = OpenAIWrapper(mock_client_success_text)
    out = w.respond_text("hi")
    assert out.text == "hello world"
    assert out.usage["total_tokens"] == 10


def test_respond_text_rate_limit_backoff(mock_client_rate_limit_then_ok):
    w = OpenAIWrapper(mock_client_rate_limit_then_ok)
    out = w.respond_text("test")
    assert out.text == "recovered"
    # ensure multiple attempts happened
    assert mock_client_rate_limit_then_ok.responses.create.call_count == 3


def test_respond_text_falls_back_to_chat(mock_client_force_chat):
    w = OpenAIWrapper(mock_client_force_chat)
    out = w.respond_text("hi", system="sys")
    assert out.text == "from chat"
    assert mock_client_force_chat.chat.completions.create.called


def test_respond_json_parses_valid_json(mock_client_success_text):
    # Make responses return JSON string
    obj = {"a": 1, "b": [1, 2]}
    mock_client_success_text.responses.create.return_value = types.SimpleNamespace(
        output_text=json.dumps(obj), usage={"total_tokens": 5}
    )
    w = OpenAIWrapper(mock_client_success_text)
    data = w.respond_json("return json")
    assert data == obj


def test_respond_json_invalid_raises(mock_client_success_text):
    mock_client_success_text.responses.create.return_value = types.SimpleNamespace(
        output_text="not json", usage={"total_tokens": 5}
    )
    w = OpenAIWrapper(mock_client_success_text)
    with pytest.raises(ValueError):
        _ = w.respond_json("return json")

