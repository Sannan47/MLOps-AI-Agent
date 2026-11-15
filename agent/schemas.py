"""Pydantic models for every tool input/output and the agent response."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ToolStatus = Literal["ok", "error", "unavailable", "timeout", "rejected", "pending_confirmation"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolResult(BaseModel):
    tool: str
    status: ToolStatus
    data: dict | list | None = None
    error: str | None = None
    duration_ms: float = 0.0
    evidence_id: str
    request_id: str
    timestamp: datetime = Field(default_factory=utcnow)


# ---- DVC inputs ----
class NoInput(Strict):
    pass


class DvcCompareInput(Strict):
    rev_a: str
    rev_b: str
    path: str | None = None


class DvcRunInput(Strict):
    run_id: str


class DvcReproduceInput(Strict):
    rev: str
    verify: bool = True


# ---- MLflow inputs ----
class ListRunsInput(Strict):
    experiment: str | None = None
    max_results: int = Field(default=10, ge=1, le=100)
    order_by: str = "start_time DESC"
    filter: str | None = None


class RunIdInput(Strict):
    run_id: str


class FindBestRunInput(Strict):
    experiment: str | None = None
    metric: str = "f1"
    mode: Literal["max", "min"] = "max"
    max_results: int = Field(default=20, ge=1, le=200)


class CompareRunsInput(Strict):
    run_ids: list[str] | None = None
    last_n: int = Field(default=5, ge=2, le=20)
    metrics: list[str] = Field(default_factory=lambda: ["f1", "precision", "recall", "accuracy"])
    experiment: str | None = None


class CompareExperimentsInput(Strict):
    exp_a: str
    exp_b: str
    metric: str = "f1"


class ModelRegistryInput(Strict):
    model_name: str | None = None


# ---- Airflow inputs ----
class DagRunsInput(Strict):
    dag_id: str
    limit: int = Field(default=5, ge=1, le=50)


class DagRunInput(Strict):
    dag_id: str
    run_id: str


class TriggerDagInput(Strict):
    dag_id: str
    conf: dict[str, Any] = Field(default_factory=dict)


class WaitDagRunInput(Strict):
    dag_id: str
    run_id: str
    timeout_s: int = Field(default=600, ge=10, le=900)


class FailedTasksInput(Strict):
    dag_id: str
    run_id: str | None = None


class TaskLogsInput(Strict):
    dag_id: str
    run_id: str
    task_id: str
    try_number: int = Field(default=1, ge=1)
    tail_lines: int = Field(default=80, ge=1, le=500)


# ---- Kubernetes inputs ----
class K8sBase(Strict):
    namespace: str = "mlops-app"


class PodsInput(K8sBase):
    label_selector: str | None = None


class PodInput(K8sBase):
    pod: str


class PodLogsInput(K8sBase):
    pod: str
    container: str | None = None
    tail_lines: int = Field(default=100, ge=1, le=1000)
    previous: bool = False


class EventsInput(K8sBase):
    involved_object: str | None = None
    warning_only: bool = True


class DeploymentInput(K8sBase):
    deployment: str = "prediction-api"


class ServiceInput(K8sBase):
    service: str = "prediction-api"


class ScaleInput(K8sBase):
    deployment: str = "prediction-api"
    replicas: int = Field(ge=1, le=5)


# ---- Policy ----
class PolicyInput(Strict):
    model_age_days: float | None = None
    champion_f1: float | None = None
    latest_data_version: str | None = None
    trained_data_version: str | None = None
    last_gate_failed: bool | None = None
    best_f1: float | None = None


# ---- Pending actions and agent response ----
class PendingAction(BaseModel):
    action_id: str
    tool: str
    args: dict
    namespace: str | None = None
    summary: str
    created_at: datetime
    expires_at: datetime
    status: Literal["pending", "confirmed", "rejected", "expired"] = "pending"
    result: dict | None = None


class Fact(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    tool: str
    args: dict
    status: ToolStatus
    duration_ms: float
    evidence_id: str


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    facts: list[Fact] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    pending_actions: list[PendingAction] = Field(default_factory=list)
