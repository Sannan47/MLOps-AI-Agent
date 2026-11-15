"""FastAPI interface for the agent: /chat, /actions, /health, /tools."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

import agent.tools  # noqa: F401
from agent.config import get_settings
from agent.llm import AnthropicLLM, LLMClient, LLMUnavailable
from agent.logging_utils import log_event, new_request_id, request_id_var
from agent.orchestrator import Agent
from agent.registry import registry

app = FastAPI(title="MLOps AI Agent", version="1.0.0")

# Overridable in tests: returns the LLM client used for each /chat call.
llm_factory: Callable[[], LLMClient] = AnthropicLLM


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or new_request_id()
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    if not req.message.strip():
        raise HTTPException(422, "message must not be blank")
    rid = request.headers.get("X-Request-ID") or new_request_id()
    try:
        llm = llm_factory()
    except LLMUnavailable as e:
        raise HTTPException(503, f"LLM unavailable: {e}") from e
    log_event("chat_request", length=len(req.message))
    return Agent(llm).run(req.message, rid)


@app.get("/actions")
def list_actions():
    return registry.store.list()


@app.post("/actions/{action_id}/confirm")
def confirm_action(action_id: str):
    res = registry.confirm(action_id, request_id_var.get())
    if res.status == "rejected":
        raise HTTPException(409, res.error or "action cannot be confirmed")
    return res


@app.post("/actions/{action_id}/reject")
def reject_action(action_id: str):
    if not registry.reject(action_id):
        raise HTTPException(409, "action unknown, expired or already decided")
    return {"action_id": action_id, "status": "rejected"}


@app.get("/tools")
def list_tools():
    return [
        {
            "name": t.name,
            "module": t.module,
            "risk": t.risk,
            "confirmation_required": t.risk == "mutate",
            "description": t.description,
        }
        for t in registry.tools.values()
    ]


def check_mlflow() -> str:
    r = httpx.get(f"{get_settings().mlflow_tracking_uri}/health", timeout=3)
    r.raise_for_status()
    return "up"


def check_airflow() -> str:
    r = httpx.get(f"{get_settings().airflow_base_url}/api/v2/monitor/health", timeout=3)
    r.raise_for_status()
    return "up"


def check_k8s() -> str:
    from agent.tools.kubernetes_tools import get_apis

    core, _ = get_apis()
    core.list_namespaced_pod(get_settings().k8s_namespace, limit=1, _request_timeout=3)
    return "up"


def check_dvc() -> str:
    if not (Path(get_settings().dvc_repo_path) / ".dvc").exists():
        raise FileNotFoundError("no .dvc directory")
    return "up"


@app.get("/health")
def health():
    out = {}
    for name, fn in (
        ("mlflow", check_mlflow),
        ("airflow", check_airflow),
        ("kubernetes", check_k8s),
        ("dvc", check_dvc),
    ):
        try:
            out[name] = {"status": fn()}
        except Exception as e:  # noqa: BLE001 - health must never fail
            out[name] = {"status": "down", "error": type(e).__name__}
    return {"status": "ok", "services": out}
