"""LLM client protocol, Anthropic implementation and a scripted FakeLLM for tests/CI."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

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
