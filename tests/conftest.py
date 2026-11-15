from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent.registry import Registry, ToolSpec
from agent.schemas import NoInput


@pytest.fixture
def reg() -> Registry:
    """Fresh registry (own pending store) so tests do not share state."""
    return Registry()


def add_tool(reg: Registry, name, fn, module="mlflow", risk="read", model=NoInput, timeout=None):
    reg.register(ToolSpec(name, name, model, fn, risk, module, timeout))


def evidence_ids(messages: list[dict]) -> dict[str, str]:
    """Map tool name -> evidence id from all tool_result blocks in a message history."""
    out: dict[str, str] = {}
    for m in messages:
        if m["role"] == "user" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    d = json.loads(b["content"].split('..."truncated"')[0])
                    out[d["tool"]] = d["evidence_id"]
    return out


class Flexible(BaseModel):
    model_config = {"extra": "allow"}


def ns(**kw):
    return SimpleNamespace(**kw)
