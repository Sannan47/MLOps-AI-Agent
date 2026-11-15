"""LLM client protocol, Anthropic implementation and a scripted FakeLLM for tests/CI."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from agent.config import get_settings


class LLMUnavailable(Exception):
    """The LLM could not be reached or refused the request."""


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    content: list[dict]  # full assistant content blocks, appended back into the history
    stop_reason: str
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)


class LLMClient(Protocol):
    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class AnthropicLLM:
    def __init__(self) -> None:
        import anthropic

        s = get_settings()
        if not s.anthropic_api_key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic(api_key=s.anthropic_api_key, timeout=120.0)
        self.model = s.agent_model

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        import anthropic

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"},
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
            )
        except anthropic.RateLimitError as e:
            raise LLMUnavailable("LLM rate limited; try again shortly") from e
        except anthropic.APIConnectionError as e:
            raise LLMUnavailable("LLM connection failed") from e
        except anthropic.APIStatusError as e:
            raise LLMUnavailable(f"LLM API error {e.status_code}") from e
        blocks = [b.model_dump(exclude_none=True) for b in resp.content]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        uses = [
            ToolUse(b["id"], b["name"], b.get("input", {}))
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        if resp.stop_reason == "refusal":
            raise LLMUnavailable("the model refused this request")
        if resp.stop_reason == "max_tokens" and not uses:
            text += "\n(response truncated: max_tokens reached)"
        return LLMResponse(blocks, resp.stop_reason or "end_turn", text, uses)


Step = Callable[[list[dict]], LLMResponse] | LLMResponse | tuple | str


class FakeLLM:
    """Scripted LLM. Each step is a final text (str), a list of (tool, args) calls, or a callable
    receiving the message history."""

    def __init__(self, steps: list[Any]):
        self.steps = list(steps)
        self.calls: list[list[dict]] = []

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.steps:
            return _text("(script exhausted)")
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, LLMResponse):
            return step
        if isinstance(step, str):
            return _text(step)
        uses = [ToolUse("toolu_" + uuid.uuid4().hex[:10], n, a) for n, a in step]
        blocks = [{"type": "tool_use", "id": u.id, "name": u.name, "input": u.input} for u in uses]
        return LLMResponse(blocks, "tool_use", "", uses)


def _text(t: str) -> LLMResponse:
    return LLMResponse([{"type": "text", "text": t}], "end_turn", t, [])


# ---------------------------------------------------------------------------
# Gemini (OpenAI-compatible chat-completions endpoint). Same loop, different wire format.
# ---------------------------------------------------------------------------
_DROP_KEYS = {"additionalProperties", "title", "default", "$defs", "strict", "examples"}


def _gemini_schema(node: Any) -> Any:
    """Reduce a pydantic JSON schema to the subset Gemini accepts."""
    if isinstance(node, dict):
        if "anyOf" in node:  # Optional[X] -> X
            options = [o for o in node["anyOf"] if o.get("type") != "null"]
            if len(options) == 1:
                merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **options[0]}
                return _gemini_schema(merged)
        return {k: _gemini_schema(v) for k, v in node.items() if k not in _DROP_KEYS}
    if isinstance(node, list):
        return [_gemini_schema(v) for v in node]
    return node


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            calls = []
            for b in content:
                if b.get("type") != "tool_use":
                    continue
                call = {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                if b.get("extra_content"):  # Gemini thought signatures must be echoed back
                    call["extra_content"] = b["extra_content"]
                calls.append(call)
            msg: dict = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    out.append(
                        {"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]}
                    )
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


class GeminiLLM:
    """LLMClient backed by Gemini's OpenAI-compatible endpoint (free tier friendly)."""

    def __init__(self, max_retries: int = 4) -> None:
        self.key, self.model, base = self._config()
        self.url = base.rstrip("/") + "/chat/completions"
        self.max_retries = max_retries

    def _config(self) -> tuple[str, str, str]:
        s = get_settings()
        if not s.gemini_api_key:
            raise LLMUnavailable("GEMINI_API_KEY is not set")
        return s.gemini_api_key, s.gemini_model, s.gemini_base_url

    def create(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        body = {
            "model": self.model,
            "messages": _to_openai_messages(system, messages),
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": _gemini_schema(t["input_schema"]),
                    },
                }
                for t in tools
            ],
            "tool_choice": "auto",
        }
        headers = {"Authorization": f"Bearer {self.key}"}
        resp = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(self.url, json=body, headers=headers, timeout=120)
            except httpx.HTTPError as e:
                raise LLMUnavailable("LLM connection failed") from e
            if resp.status_code in (429, 500, 503) and attempt < self.max_retries:
                time.sleep(min(2 ** (attempt + 2), 60))  # free tiers rate-limit; back off
                continue
            break
        assert resp is not None
        if resp.status_code == 429:
            raise LLMUnavailable("LLM rate limited; try again shortly")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"LLM API error {resp.status_code}: {resp.text[:200]}")
        choice = resp.json()["choices"][0]
        msg = choice["message"]
        text = msg.get("content") or ""
        blocks: list[dict] = [{"type": "text", "text": text}] if text else []
        uses: list[ToolUse] = []
        for tc in msg.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            uses.append(ToolUse(tc["id"], tc["function"]["name"], args))
            block = {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": args,
            }
            if tc.get("extra_content"):
                block["extra_content"] = tc["extra_content"]
            blocks.append(block)
        if choice.get("finish_reason") == "content_filter" and not uses:
            raise LLMUnavailable("the model refused this request")
        return LLMResponse(blocks, "tool_use" if uses else "end_turn", text, uses)


class OpenRouterLLM(GeminiLLM):
    """Same OpenAI-compatible chat/tool protocol, served by OpenRouter."""

    def _config(self) -> tuple[str, str, str]:
        s = get_settings()
        if not s.openrouter_api_key:
            raise LLMUnavailable("OPENROUTER_API_KEY is not set")
        return s.openrouter_api_key, s.openrouter_model, s.openrouter_base_url


def make_llm() -> LLMClient:
    """Pick the provider from LLM_PROVIDER (anthropic | gemini | openrouter)."""
    provider = get_settings().llm_provider.lower()
    if provider == "gemini":
        return GeminiLLM()
    if provider == "openrouter":
        return OpenRouterLLM()
    if provider == "anthropic":
        return AnthropicLLM()
    raise LLMUnavailable(f"unknown LLM_PROVIDER '{provider}'")
