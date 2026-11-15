from __future__ import annotations

import json

import httpx
import pytest
import respx

import agent.tools  # noqa: F401
from agent import llm as llm_mod
from agent.config import get_settings
from agent.llm import GeminiLLM, LLMUnavailable, _gemini_schema, _to_openai_messages
from agent.registry import registry

URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


@pytest.fixture
def gemini(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "gemini_api_key", "test-key")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda *_: None)
    return GeminiLLM()


def test_gemini_requires_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "gemini_api_key", "")
    with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
        GeminiLLM()


def test_schema_reduced_for_every_registered_tool():
    for spec in registry.anthropic_tool_specs():
        text = json.dumps(_gemini_schema(spec["input_schema"]))
        assert "additionalProperties" not in text and '"null"' not in text and "$defs" not in text


def test_optional_fields_become_plain_types():
    out = _gemini_schema(
        {
            "type": "object",
            "properties": {
                "x": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "X",
                }
            },
        }
    )
    assert out["properties"]["x"] == {"type": "string"}


def test_message_conversion_roundtrip():
    msgs = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "t",
                    "input": {"a": 1},
                    "extra_content": {"google": {"thought_signature": "sig"}},
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "{}"}],
        },
    ]
    out = _to_openai_messages("sys", msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert out[2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "sig"
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}


@respx.mock
def test_tool_call_response_parsed(gemini):
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c9",
                                    "type": "function",
                                    "function": {
                                        "name": "mlflow_find_best_run",
                                        "arguments": '{"metric": "f1"}',
                                    },
                                    "extra_content": {"google": {"thought_signature": "s"}},
                                }
                            ],
                        },
                    }
                ]
            },
        )
    )
    r = gemini.create(
        "sys", [{"role": "user", "content": "best?"}], registry.anthropic_tool_specs()
    )
    assert r.stop_reason == "tool_use" and r.tool_uses[0].name == "mlflow_find_best_run"
    assert r.tool_uses[0].input == {"metric": "f1"} and r.content[0]["extra_content"]
    sent = json.loads(respx.calls[0].request.content)
    assert sent["tools"] and sent["tool_choice"] == "auto"
    assert respx.calls[0].request.headers["authorization"] == "Bearer test-key"


@respx.mock
def test_final_text_and_429_retry(gemini):
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(
                200, json={"choices": [{"finish_reason": "stop", "message": {"content": "done"}}]}
            ),
        ]
    )
    r = gemini.create("s", [{"role": "user", "content": "x"}], [])
    assert r.text == "done" and not r.tool_uses and route.call_count == 2


@respx.mock
def test_persistent_rate_limit_and_errors_map_to_unavailable(gemini):
    respx.post(URL).mock(return_value=httpx.Response(429))
    with pytest.raises(LLMUnavailable, match="rate limited"):
        gemini.create("s", [{"role": "user", "content": "x"}], [])
    respx.post(URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(LLMUnavailable, match="400"):
        gemini.create("s", [{"role": "user", "content": "x"}], [])
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    with pytest.raises(LLMUnavailable, match="connection"):
        gemini.create("s", [{"role": "user", "content": "x"}], [])


def test_openrouter_provider_selected(monkeypatch):
    from agent.llm import OpenRouterLLM, make_llm

    s = get_settings()
    monkeypatch.setattr(s, "llm_provider", "openrouter")
    monkeypatch.setattr(s, "openrouter_api_key", "test-key")
    llm = make_llm()
    assert isinstance(llm, OpenRouterLLM)
    assert llm.url == "https://openrouter.ai/api/v1/chat/completions"
    monkeypatch.setattr(s, "openrouter_api_key", "")
    with pytest.raises(LLMUnavailable):
        make_llm()
