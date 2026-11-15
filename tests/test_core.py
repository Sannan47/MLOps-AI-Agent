"""Schemas, registry, permissions, logging, policy."""

from __future__ import annotations

import io
import json
import logging
import time

import pytest
from pydantic import ValidationError

import agent.tools  # noqa: F401
from agent.logging_utils import JsonFormatter, get_logger, log_event, redact, request_id_var
from agent.permissions import PendingStore
from agent.policy import evaluate_retraining_policy
from agent.registry import Registry, ToolSpec, registry
from agent.schemas import (
    FindBestRunInput,
    NoInput,
    ScaleInput,
    TriggerDagInput,
)
from tests.conftest import add_tool


def test_schema_rejects_unknown_field_and_wrong_type():
    with pytest.raises(ValidationError):
        FindBestRunInput(metric="f1", bogus=1)
    with pytest.raises(ValidationError):
        FindBestRunInput(max_results="many")
    with pytest.raises(ValidationError):
        ScaleInput(replicas=50)


def test_json_schema_export_for_every_tool():
    specs = registry.anthropic_tool_specs()
    assert specs
    for s in specs:
        assert s["strict"] is True
        assert s["input_schema"]["additionalProperties"] is False


def test_registry_has_min_three_tools_per_module():
    counts = {k: len(v) for k, v in registry.by_module().items()}
    for module in ("dvc", "mlflow", "airflow", "k8s"):
        assert counts[module] >= 3, counts


def test_k8s_no_delete_tool_exists_and_unknown_tool_errors():
    assert not [n for n in registry.tools if "delete" in n or "exec" in n]
    res = registry.execute("k8s_delete_pod", {"pod": "x"}, "r1")
    assert res.status == "error" and "unknown tool" in res.error


def test_invalid_tool_args_rejected(reg):
    add_tool(reg, "t", lambda i: {}, model=FindBestRunInput)
    res = reg.execute("t", {"metric": "f1", "nope": 1}, "r")
    assert res.status == "error" and "invalid request" in res.error


def test_tool_timeout_returns_timeout_status(reg):
    add_tool(reg, "slow", lambda i: time.sleep(1.0), timeout=0.1)
    res = reg.execute("slow", {}, "r")
    assert res.status == "timeout"


def test_tool_exception_and_connection_error_statuses(reg):
    def boom(_):
        raise RuntimeError("kaput")

    def down(_):
        raise ConnectionError("Connection refused")

    add_tool(reg, "boom", boom)
    add_tool(reg, "down", down)
    assert reg.execute("boom", {}, "r").status == "error"
    assert reg.execute("down", {}, "r").status == "unavailable"


def test_unauthorized_k8s_namespace_rejected(reg):
    called = []
    from agent.schemas import K8sBase

    add_tool(reg, "k8s_x", lambda i: called.append(i), module="k8s", model=K8sBase)
    res = reg.execute("k8s_x", {"namespace": "kube-system"}, "r")
    assert res.status == "rejected"
    assert called == []


def test_mutating_tool_requires_confirmation_and_runs_once(reg):
    from agent.schemas import DeploymentInput

    calls = []
    add_tool(
        reg,
        "k8s_restart",
        lambda i: calls.append(i) or {"ok": 1},
        module="k8s",
        risk="mutate",
        model=DeploymentInput,
    )
    res = reg.execute("k8s_restart", {"deployment": "prediction-api"}, "r")
    assert res.status == "pending_confirmation" and calls == []
    action_id = res.data["action_id"]
    done = reg.confirm(action_id)
    assert done.status == "ok" and len(calls) == 1
    again = reg.confirm(action_id)
    assert again.status == "rejected" and len(calls) == 1


def test_confirm_expired_or_unknown_action_rejected(reg):
    assert reg.confirm("nope").status == "rejected"
    reg.store = PendingStore(ttl_seconds=0)
    a = reg.store.create("t", {}, "s", None)
    time.sleep(0.01)
    add_tool(reg, "t", lambda i: {})
    assert reg.confirm(a.action_id).status == "rejected"


def test_reject_action(reg):
    a = reg.store.create("t", {}, "s", None)
    assert reg.reject(a.action_id) is True
    assert reg.reject(a.action_id) is False


def test_airflow_trigger_non_approved_dag_rejected(reg):
    add_tool(
        reg,
        "airflow_trigger_dag",
        lambda i: {},
        module="airflow",
        risk="mutate",
        model=TriggerDagInput,
    )
    res = reg.execute("airflow_trigger_dag", {"dag_id": "evil_dag"}, "r")
    assert res.status == "rejected"
    ok = reg.execute("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"}, "r")
    assert ok.status == "pending_confirmation"


def _capture():
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(JsonFormatter())
    get_logger().addHandler(h)
    return buf, h


def test_structured_log_fields_present(reg):
    buf, h = _capture()
    try:
        add_tool(reg, "ok_tool", lambda i: {"a": 1})
        token = request_id_var.set("req-123")
        reg.execute("ok_tool", {}, "req-123")
        request_id_var.reset(token)
    finally:
        get_logger().removeHandler(h)
    line = [json.loads(ln) for ln in buf.getvalue().splitlines() if '"tool_call"' in ln][0]
    for k in ("tool", "status", "duration_ms", "request_id", "timestamp", "level"):
        assert k in line, k
    assert line["request_id"] == "req-123"


def test_logs_redact_secrets():
    buf, h = _capture()
    try:
        log_event(
            "tool_call",
            args={"password": "hunter2", "headers": {"Authorization": "Bearer abc"}},
            note="key sk-ant-api03-SECRETVALUE and Bearer abc.def.ghi",
        )
    finally:
        get_logger().removeHandler(h)
    out = buf.getvalue()
    for secret in ("hunter2", "sk-ant-api03", "SECRETVALUE"):
        assert secret not in out
    assert "***" in out
    assert redact({"token": "x", "n": 1}) == {"token": "***", "n": 1}


def test_retraining_policy_rules():
    none = evaluate_retraining_policy({})
    assert none["recommend"] is False and none["insufficient_evidence"] is True
    fresh = evaluate_retraining_policy(
        {
            "model_age_days": 3,
            "champion_f1": 0.95,
            "latest_data_version": "data-v1.1",
            "trained_data_version": "data-v1.1",
        }
    )
    assert fresh["recommend"] is False and fresh["policy_rules_triggered"] == []
    old = evaluate_retraining_policy({"model_age_days": 45})
    assert old["policy_rules_triggered"] == ["max_model_age_days"]
    newdata = evaluate_retraining_policy(
        {"latest_data_version": "data-v2.0", "trained_data_version": "data-v1.0"}
    )
    assert (
        newdata["recommend"] and "retrain_if_new_data_version" in newdata["policy_rules_triggered"]
    )
    lowf1 = evaluate_retraining_policy({"champion_f1": 0.8})
    assert "min_f1" in lowf1["policy_rules_triggered"]


def test_registry_never_raises_for_none_args(reg):
    reg.register(ToolSpec("n", "n", NoInput, lambda i: [], "read", "mlflow"))
    assert reg.execute("n", None, "r").status == "ok"
    assert isinstance(reg, Registry)
