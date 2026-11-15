"""CLI: python -m agent.main "question" """

from __future__ import annotations

import argparse
import sys

from agent.llm import LLMUnavailable, make_llm
from agent.logging_utils import new_request_id
from agent.orchestrator import Agent
from agent.registry import registry


def main(argv: list[str] | None = None, llm=None) -> int:
    p = argparse.ArgumentParser(prog="agent")
    p.add_argument("question")
    a = p.parse_args(argv)
    try:
        llm = llm or make_llm()
    except LLMUnavailable as e:
        print(f"LLM unavailable: {e}", file=sys.stderr)
        return 2
    resp = Agent(llm).run(a.question, new_request_id())
    print(resp.answer)
    print(f"\nrequest_id: {resp.request_id}")
    print("\nEvidence:")
    for c in resp.tool_calls:
        print(f"  [{c.evidence_id}] {c.tool} -> {c.status} ({c.duration_ms:.0f} ms)")
    for act in resp.pending_actions:
        ans = input(f"\nConfirm action {act.action_id} ({act.summary})? [y/N] ").strip().lower()
        if ans == "y":
            res = registry.confirm(act.action_id, resp.request_id)
            print(f"-> {res.status}: {res.data or res.error}")
        else:
            registry.reject(act.action_id)
            print("-> rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
