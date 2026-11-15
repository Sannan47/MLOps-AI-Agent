"""Replay tests built from the saved real evidence (no network, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.orchestrator import Evidence, find_unsupported
from agent.schemas import ToolResult

ROOT = Path(__file__).resolve().parent.parent
CONVERSATIONS = sorted((ROOT / "evidence" / "conversations").glob("conv*.json"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", CONVERSATIONS, ids=lambda p: p.stem)
def test_saved_conversation_is_grounded_in_its_own_evidence(path):
    """Every number, id, version and citation in a recorded answer is backed by its tool results."""
    doc = _load(path)
    ev = Evidence()
    for r in doc["tool_results"]:
        ev.add(ToolResult.model_validate(r))
    assert find_unsupported(doc["response"]["answer"], ev) == []
    assert doc["response"]["tool_calls"], "a recorded conversation must show tool calls"


def test_failure_c_trace_shows_validation_failure_and_no_promotion():
    t = _load(ROOT / "evidence" / "failures" / "C" / "airflow_trace.json")
    text = json.dumps(t)
    assert "validate_dataset" in text and "upstream_failed" in text
    assert "label" in text and "missing" in text
    assert t["registry_before"] == t["registry_after"]


def test_failure_b_trace_shows_missing_model_file():
    t = _load(ROOT / "evidence" / "failures" / "B" / "tool_results.json")
    text = json.dumps(t)
    assert "/models/missing.joblib" in text
    assert "503" in text
