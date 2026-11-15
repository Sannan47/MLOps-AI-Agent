"""Codified retraining policy: the only rules the agent may use to recommend retraining."""

from __future__ import annotations

from pathlib import Path

import yaml

POLICY_PATH = Path(__file__).with_name("policy.yaml")


def load_policy(path: Path | None = None) -> dict:
    return yaml.safe_load((path or POLICY_PATH).read_text(encoding="utf-8"))


def evaluate_retraining_policy(facts: dict, policy: dict | None = None) -> dict:
    """Return {recommend, reasons, policy_rules_triggered, insufficient_evidence}."""
    policy = policy or load_policy()
    provided = {k: v for k, v in facts.items() if v is not None}
    if not provided:
        return {
            "recommend": False,
            "insufficient_evidence": True,
            "reasons": ["insufficient evidence: no facts supplied"],
            "policy_rules_triggered": [],
        }
    reasons: list[str] = []
    rules: list[str] = []
    age = facts.get("model_age_days")
    if age is not None and age > policy["max_model_age_days"]:
        rules.append("max_model_age_days")
        reasons.append(f"model is {age:.0f} days old (> {policy['max_model_age_days']})")
    f1 = facts.get("champion_f1")
    if f1 is not None and f1 < policy["min_f1"]:
        rules.append("min_f1")
        reasons.append(f"champion F1 {f1} < policy min_f1 {policy['min_f1']}")
    best = facts.get("best_f1")
    if f1 is not None and best is not None and best - f1 > policy["max_f1_drop_vs_champion"]:
        rules.append("max_f1_drop_vs_champion")
        reasons.append(f"champion F1 {f1} trails best F1 {best} by more than the allowed drop")
    latest, trained = facts.get("latest_data_version"), facts.get("trained_data_version")
    if policy["retrain_if_new_data_version"] and latest and trained and latest != trained:
        rules.append("retrain_if_new_data_version")
        reasons.append(f"newer data version {latest} exists (model trained on {trained})")
    if policy["retrain_if_last_gate_failed"] and facts.get("last_gate_failed"):
        rules.append("retrain_if_last_gate_failed")
        reasons.append("last quality gate failed")
    if not rules:
        reasons = ["no policy rule triggered by the supplied evidence"]
    return {
        "recommend": bool(rules),
        "insufficient_evidence": False,
        "reasons": reasons,
        "policy_rules_triggered": rules,
        "facts_considered": sorted(provided),
    }
