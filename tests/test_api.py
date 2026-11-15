from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agent.llm import FakeLLM
from agent.registry import registry

client = TestClient(api_main.app)


@pytest.fixture(autouse=True)
def fake_llm(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "llm_factory",
        lambda: FakeLLM(
            ["**Known facts**\n- hello\n**Uncertainties**\n- none\n**Recommended actions**\n- none"]
        ),
    )


def test_api_chat_rejects_empty_message_422():
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 4001}).status_code == 422
    assert client.post("/chat", json={}).status_code == 422


def test_api_chat_returns_structured_response_and_request_id():
    r = client.post("/chat", json={"message": "hi"}, headers={"X-Request-ID": "abc123"})
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == "abc123" and r.headers["X-Request-ID"] == "abc123"
    assert body["facts"][0]["statement"] == "hello"


def test_api_confirm_and_reject_flow():
    apps = MagicMock()
    ca = (MagicMock(), apps)
    res = registry.execute("k8s_restart_deployment", {"deployment": "prediction-api"}, "r")
    aid = res.data["action_id"]
    assert any(a["action_id"] == aid for a in client.get("/actions").json())
    with patch("agent.tools.kubernetes_tools.get_apis", lambda: ca):
        ok = client.post(f"/actions/{aid}/confirm")
        again = client.post(f"/actions/{aid}/confirm")
    assert ok.status_code == 200 and ok.json()["status"] == "ok"
    assert again.status_code == 409
    assert apps.patch_namespaced_deployment.call_count == 1
    res2 = registry.execute("k8s_scale_deployment", {"replicas": 2}, "r")
    assert client.post(f"/actions/{res2.data['action_id']}/reject").status_code == 200
    assert client.post("/actions/unknown/confirm").status_code == 409


def test_api_health_reports_per_service_status_with_services_down():
    def down():
        raise ConnectionError("nope")

    with (
        patch.object(api_main, "check_mlflow", down),
        patch.object(api_main, "check_airflow", down),
        patch.object(api_main, "check_k8s", lambda: "up"),
        patch.object(api_main, "check_dvc", lambda: "up"),
    ):
        r = client.get("/health")
    s = r.json()["services"]
    assert r.status_code == 200
    assert s["mlflow"]["status"] == "down" and s["kubernetes"]["status"] == "up"


def test_api_tools_listing_marks_mutating_tools():
    tools = {t["name"]: t for t in client.get("/tools").json()}
    assert tools["k8s_restart_deployment"]["confirmation_required"] is True
    assert tools["mlflow_find_best_run"]["risk"] == "read"
    assert NS  # keep import used
