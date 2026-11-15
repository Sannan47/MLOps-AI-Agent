"""Namespace/DAG allowlists and the two-phase confirmation store for mutating tools."""

from __future__ import annotations

import threading
import uuid
from datetime import timedelta

from agent.config import Settings, get_settings
from agent.logging_utils import log_event
from agent.schemas import PendingAction, utcnow


class NamespacePolicy:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def check(self, module: str, tool: str, args: dict) -> str | None:
        """Return a rejection reason, or None if the call is permitted."""
        if module == "k8s":
            ns = args.get("namespace") or self.settings.k8s_namespace
            if ns not in self.settings.allowed_namespaces:
                return (
                    f"namespace '{ns}' is not approved; allowed: {self.settings.allowed_namespaces}"
                )
        if tool == "airflow_trigger_dag":
            dag = args.get("dag_id")
            if dag not in self.settings.approved_dag_list:
                return f"DAG '{dag}' is not approved; approved: {self.settings.approved_dag_list}"
        return None


class PendingStore:
    """In-memory pending actions. Only a human (API/CLI) may confirm; the LLM cannot."""

    def __init__(self, ttl_seconds: int | None = None):
        self.ttl = ttl_seconds if ttl_seconds is not None else get_settings().action_ttl_seconds
        self._actions: dict[str, PendingAction] = {}
        self._lock = threading.Lock()

    def create(self, tool: str, args: dict, summary: str, namespace: str | None) -> PendingAction:
        now = utcnow()
        action = PendingAction(
            action_id=uuid.uuid4().hex[:12],
            tool=tool,
            args=args,
            namespace=namespace,
            summary=summary,
            created_at=now,
            expires_at=now + timedelta(seconds=self.ttl),
        )
        with self._lock:
            self._actions[action.action_id] = action
        log_event("action_pending", tool=tool, action_id=action.action_id, summary=summary)
        return action

    def list(self) -> list[PendingAction]:
        with self._lock:
            for a in self._actions.values():
                self._expire(a)
            return [a for a in self._actions.values() if a.status == "pending"]

    def _expire(self, a: PendingAction) -> None:
        if a.status == "pending" and utcnow() > a.expires_at:
            a.status = "expired"

    def take(self, action_id: str, new_status: str) -> tuple[PendingAction | None, str | None]:
        """Atomically claim a pending action. Returns (action, error)."""
        with self._lock:
            a = self._actions.get(action_id)
            if a is None:
                return None, "unknown action id"
            self._expire(a)
            if a.status != "pending":
                return a, f"action is {a.status}, cannot be used again"
            a.status = new_status  # type: ignore[assignment]
            log_event("action_decision", action_id=action_id, decision=new_status, tool=a.tool)
            return a, None
