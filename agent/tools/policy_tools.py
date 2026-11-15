"""Helper (non-module) tool wrapping the codified retraining policy."""

from __future__ import annotations

from agent.policy import evaluate_retraining_policy
from agent.registry import tool
from agent.schemas import PolicyInput


@tool(
    "policy_evaluate_retraining",
    "Evaluate the codified retraining policy against facts gathered from other tools. "
    "Omit unknown facts. This is the only basis for recommending retraining.",
    PolicyInput,
    module="policy",
)
def policy_evaluate_retraining(inp: PolicyInput) -> dict:
    return evaluate_retraining_policy(inp.model_dump())
