"""Airflow 3 tools (REST API v2, JWT auth via POST /auth/token)."""

from __future__ import annotations

import re
import time

import httpx

from agent.config import get_settings
from agent.registry import tool
from agent.schemas import (
    DagRunInput,
    DagRunsInput,
    FailedTasksInput,
    NoInput,
    TaskLogsInput,
    TriggerDagInput,
    WaitDagRunInput,
)

TERMINAL = {"success", "failed"}
POLL_INTERVAL_S = 10.0
_token: dict[str, str | None] = {"value": None}
ERROR_RE = re.compile(r"(ERROR|Traceback|CRITICAL|\b\w+(Error|Exception)\b)")


class AirflowHTTPError(RuntimeError):
    pass


def _fetch_token(client: httpx.Client) -> str:
    s = get_settings()
    r = client.post(
        f"{s.airflow_base_url}/auth/token",
        json={"username": s.airflow_username, "password": s.airflow_password},
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    _token["value"] = token
    return token


def _request(method: str, path: str, **kw) -> dict:
    s = get_settings()
    with httpx.Client(timeout=15) as client:
        for attempt in range(2):
            token = _token["value"] or _fetch_token(client)
            r = client.request(
                method,
                f"{s.airflow_base_url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                **kw,
            )
            if r.status_code == 401 and attempt == 0:
                _token["value"] = None  # refresh once
                continue
            if r.status_code >= 400:
                raise AirflowHTTPError(f"Airflow API {method} {path} returned {r.status_code}")
            if "json" in r.headers.get("content-type", ""):
                return r.json()
            return {"text": r.text}
    raise AirflowHTTPError("authentication failed")


def _run_info(r: dict) -> dict:
    keys = ["dag_run_id", "state", "start_date", "end_date", "logical_date", "run_type", "conf"]
    return {k: r.get(k) for k in keys}


@tool(
    "airflow_list_dags",
    "List Airflow DAGs (paused flag, tags, last parse).",
    NoInput,
    module="airflow",
)
def list_dags(_: NoInput) -> list[dict]:
    d = _request("GET", "/api/v2/dags")
    return [
        {
            "dag_id": x["dag_id"],
            "is_paused": x.get("is_paused"),
            "tags": [t["name"] if isinstance(t, dict) else t for t in x.get("tags", [])],
            "last_parsed_time": x.get("last_parsed_time"),
        }
        for x in d.get("dags", [])
    ]


@tool(
    "airflow_get_dag_runs",
    "Recent runs of a DAG (state, times, conf).",
    DagRunsInput,
    module="airflow",
)
def get_dag_runs(inp: DagRunsInput) -> list[dict]:
    d = _request(
        "GET",
        f"/api/v2/dags/{inp.dag_id}/dagRuns",
        params={"order_by": "-start_date", "limit": inp.limit},
    )
    return [_run_info(r) for r in d.get("dag_runs", [])]


def _task_states(dag_id: str, run_id: str) -> list[dict]:
    d = _request("GET", f"/api/v2/dags/{dag_id}/dagRuns/{run_id}/taskInstances")
    return [
        {"task_id": t["task_id"], "state": t.get("state"), "try_number": t.get("try_number")}
        for t in d.get("task_instances", [])
    ]


@tool(
    "airflow_get_dag_run_status",
    "State of one DAG run plus per-task states.",
    DagRunInput,
    module="airflow",
)
def get_dag_run_status(inp: DagRunInput) -> dict:
    r = _request("GET", f"/api/v2/dags/{inp.dag_id}/dagRuns/{inp.run_id}")
    return {**_run_info(r), "tasks": _task_states(inp.dag_id, inp.run_id)}


@tool(
    "airflow_trigger_dag",
    "Request a run of an approved DAG. Requires user confirmation before it executes.",
    TriggerDagInput,
    risk="mutate",
    module="airflow",
)
def trigger_dag(inp: TriggerDagInput) -> dict:
    r = _request(
        "POST", f"/api/v2/dags/{inp.dag_id}/dagRuns", json={"logical_date": None, "conf": inp.conf}
    )
    return _run_info(r)


@tool(
    "airflow_wait_for_dag_run",
    "Poll a DAG run until it finishes or the timeout elapses.",
    WaitDagRunInput,
    module="airflow",
    timeout_s=900,
)
def wait_for_dag_run(inp: WaitDagRunInput) -> dict:
    deadline = time.monotonic() + inp.timeout_s
    while True:
        r = _request("GET", f"/api/v2/dags/{inp.dag_id}/dagRuns/{inp.run_id}")
        if r.get("state") in TERMINAL:
            return {**_run_info(r), "finished": True, "tasks": _task_states(inp.dag_id, inp.run_id)}
        if time.monotonic() >= deadline:
            return {**_run_info(r), "finished": False, "message": "wait timed out"}
        time.sleep(POLL_INTERVAL_S)


@tool(
    "airflow_get_failed_tasks",
    "Failed / upstream_failed task instances of a run (latest if no run id).",
    FailedTasksInput,
    module="airflow",
)
def get_failed_tasks(inp: FailedTasksInput) -> dict:
    run_id = inp.run_id
    if run_id is None:
        runs = _request(
            "GET",
            f"/api/v2/dags/{inp.dag_id}/dagRuns",
            params={"order_by": "-start_date", "limit": 1},
        ).get("dag_runs", [])
        if not runs:
            return {"run_id": None, "failed_tasks": [], "message": "no runs found"}
        run_id = runs[0]["dag_run_id"]
    tasks = _task_states(inp.dag_id, run_id)
    return {
        "run_id": run_id,
        "failed_tasks": [t for t in tasks if t["state"] in ("failed", "upstream_failed")],
    }


def extract_diagnostics(lines: list[str]) -> list[str]:
    return [ln.strip() for ln in lines if ERROR_RE.search(ln)][:20]


@tool(
    "airflow_get_task_logs",
    "Tail of a task's logs plus extracted error lines.",
    TaskLogsInput,
    module="airflow",
)
def get_task_logs(inp: TaskLogsInput) -> dict:
    d = _request(
        "GET",
        f"/api/v2/dags/{inp.dag_id}/dagRuns/{inp.run_id}/taskInstances/{inp.task_id}"
        f"/logs/{inp.try_number}",
    )
    content = d.get("content", d.get("text", ""))
    if isinstance(content, list):
        lines = [c if isinstance(c, str) else str(c.get("event", c)) for c in content]
    else:
        lines = str(content).splitlines()
    return {
        "task_id": inp.task_id,
        "try_number": inp.try_number,
        "log_tail": lines[-inp.tail_lines :],
        "diagnostics": extract_diagnostics(lines),
    }
