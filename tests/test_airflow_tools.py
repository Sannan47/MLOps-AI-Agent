from __future__ import annotations

import httpx
import pytest
import respx

from agent.registry import registry
from agent.tools import airflow_tools

BASE = "http://localhost:8080"


@pytest.fixture(autouse=True)
def _reset_token():
    airflow_tools._token["value"] = None
    yield
    airflow_tools._token["value"] = None


def auth(mock):
    return mock.post(f"{BASE}/auth/token").mock(
        return_value=httpx.Response(200, json={"access_token": "jwt-token"})
    )


@respx.mock
def test_airflow_token_flow_and_401_refresh():
    token_route = auth(respx)
    route = respx.get(f"{BASE}/api/v2/dags").mock(
        side_effect=[
            httpx.Response(401),
            httpx.Response(
                200,
                json={
                    "dags": [
                        {
                            "dag_id": "ml_training_pipeline",
                            "is_paused": False,
                            "tags": [{"name": "ml"}],
                        }
                    ]
                },
            ),
        ]
    )
    res = registry.execute("airflow_list_dags", {}, "r")
    assert res.status == "ok" and res.data[0]["dag_id"] == "ml_training_pipeline"
    assert token_route.call_count == 2 and route.call_count == 2
    assert "jwt-token" not in res.model_dump_json()


@respx.mock
def test_airflow_500_returns_error_status():
    auth(respx)
    respx.get(f"{BASE}/api/v2/dags").mock(return_value=httpx.Response(500))
    res = registry.execute("airflow_list_dags", {}, "r")
    assert res.status == "error" and "500" in res.error


@respx.mock
def test_airflow_down_returns_unavailable():
    respx.post(f"{BASE}/auth/token").mock(side_effect=httpx.ConnectError("refused"))
    assert registry.execute("airflow_list_dags", {}, "r").status == "unavailable"


@respx.mock
def test_airflow_trigger_pending_until_confirmed():
    auth(respx)
    post = respx.post(f"{BASE}/api/v2/dags/ml_training_pipeline/dagRuns").mock(
        return_value=httpx.Response(200, json={"dag_run_id": "manual__1", "state": "queued"})
    )
    res = registry.execute("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"}, "r")
    assert res.status == "pending_confirmation" and post.call_count == 0
    done = registry.confirm(res.data["action_id"])
    assert done.status == "ok" and post.call_count == 1
    assert done.data["dag_run_id"] == "manual__1"


@respx.mock
def test_airflow_trigger_non_approved_dag_makes_no_http_call():
    post = respx.post(f"{BASE}/api/v2/dags/other/dagRuns")
    res = registry.execute("airflow_trigger_dag", {"dag_id": "other"}, "r")
    assert res.status == "rejected" and post.call_count == 0


@respx.mock
def test_airflow_failed_tasks_and_log_diagnostics():
    auth(respx)
    d = "ml_training_pipeline"
    respx.get(f"{BASE}/api/v2/dags/{d}/dagRuns").mock(
        return_value=httpx.Response(
            200, json={"dag_runs": [{"dag_run_id": "r1", "state": "failed"}]}
        )
    )
    respx.get(f"{BASE}/api/v2/dags/{d}/dagRuns/r1/taskInstances").mock(
        return_value=httpx.Response(
            200,
            json={
                "task_instances": [
                    {"task_id": "validate_dataset", "state": "failed", "try_number": 1},
                    {"task_id": "preprocess_data", "state": "upstream_failed", "try_number": 0},
                    {"task_id": "x", "state": "success", "try_number": 1},
                ]
            },
        )
    )
    res = registry.execute("airflow_get_failed_tasks", {"dag_id": d}, "r")
    assert [t["task_id"] for t in res.data["failed_tasks"]] == [
        "validate_dataset",
        "preprocess_data",
    ]
    respx.get(f"{BASE}/api/v2/dags/{d}/dagRuns/r1/taskInstances/validate_dataset/logs/1").mock(
        return_value=httpx.Response(
            200,
            json={
                "content": [
                    "INFO starting",
                    "ERROR DataValidationError: column 'label' missing",
                    "Traceback ...",
                ]
            },
        )
    )
    logs = registry.execute(
        "airflow_get_task_logs", {"dag_id": d, "run_id": "r1", "task_id": "validate_dataset"}, "r"
    )
    assert any("DataValidationError" in x for x in logs.data["diagnostics"])


@respx.mock
def test_airflow_wait_for_dag_run_polls_until_terminal(monkeypatch):
    monkeypatch.setattr(airflow_tools, "POLL_INTERVAL_S", 0.01)
    auth(respx)
    d = "ml_training_pipeline"
    respx.get(f"{BASE}/api/v2/dags/{d}/dagRuns/r1").mock(
        side_effect=[
            httpx.Response(200, json={"dag_run_id": "r1", "state": "running"}),
            httpx.Response(200, json={"dag_run_id": "r1", "state": "success"}),
        ]
    )
    respx.get(f"{BASE}/api/v2/dags/{d}/dagRuns/r1/taskInstances").mock(
        return_value=httpx.Response(200, json={"task_instances": []})
    )
    res = registry.execute("airflow_wait_for_dag_run", {"dag_id": d, "run_id": "r1"}, "r")
    assert res.status == "ok" and res.data["state"] == "success" and res.data["finished"]
