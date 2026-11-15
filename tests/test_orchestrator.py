from __future__ import annotations

import json

from agent.llm import FakeLLM, LLMUnavailable
from agent.orchestrator import Agent, Evidence, find_unsupported, parse_sections
from agent.registry import Registry
from agent.schemas import (
    DagRunsInput,
    DeploymentInput,
    DvcRunInput,
    FindBestRunInput,
    K8sBase,
    NoInput,
    PolicyInput,
    TriggerDagInput,
)
from tests.conftest import add_tool, evidence_ids

RUN_ID = "a1b2c3d4e5f60718293a4b5c6d7e8f90"


def scenario_registry(mlflow_ok=True):
    reg = Registry()
    calls: list[str] = []

    def rec(name, data):
        def fn(inp):
            calls.append(name)
            if isinstance(data, Exception):
                raise data
            return data

        return fn

    mlflow_data = (
        {"best": {"run_id": RUN_ID, "metrics": {"f1": 0.9412}, "data.version": "data-v1.1"}}
        if mlflow_ok
        else ConnectionError("Connection refused")
    )
    add_tool(
        reg,
        "mlflow_find_best_run",
        rec("mlflow_find_best_run", mlflow_data),
        model=FindBestRunInput,
    )
    add_tool(
        reg,
        "dvc_get_dataset_for_run",
        rec("dvc_get_dataset_for_run", {"data.version": "data-v1.1", "data.dvc_md5": "ff00aa11"}),
        module="dvc",
        model=DvcRunInput,
    )
    add_tool(
        reg,
        "dvc_list_versions",
        rec("dvc_list_versions", [{"tag": "data-v1.1"}]),
        module="dvc",
        model=NoInput,
    )
    add_tool(
        reg,
        "k8s_get_deployed_version",
        rec("k8s_get_deployed_version", {"image_tags": ["1.0.0"], "ready_replicas": 2}),
        module="k8s",
        model=DeploymentInput,
    )
    add_tool(
        reg,
        "k8s_find_unhealthy_pods",
        rec("k8s_find_unhealthy_pods", {"unhealthy_pods": []}),
        module="k8s",
        model=K8sBase,
    )
    add_tool(
        reg,
        "airflow_get_dag_runs",
        rec("airflow_get_dag_runs", [{"state": "success"}]),
        module="airflow",
        model=DagRunsInput,
    )
    add_tool(
        reg,
        "airflow_trigger_dag",
        rec("airflow_trigger_dag", {}),
        module="airflow",
        risk="mutate",
        model=TriggerDagInput,
    )
    add_tool(
        reg,
        "policy_evaluate_retraining",
        rec("policy_evaluate_retraining", {"recommend": False, "policy_rules_triggered": []}),
        module="policy",
        model=PolicyInput,
    )
    return reg, calls


def test_tool_selection_best_model_calls_mlflow():
    reg, calls = scenario_registry()
    llm = FakeLLM(
        [
            [("mlflow_find_best_run", {"metric": "f1"})],
            lambda m: (
                "**Known facts**\n- Best run "
                + RUN_ID
                + f" has F1 0.9412 [{evidence_ids(m)['mlflow_find_best_run']}]\n"
                "**Uncertainties**\n- none\n**Recommended actions**\n- none"
            ),
        ]
    )
    resp = Agent(llm, reg).run("best model?", "req1")
    assert calls == ["mlflow_find_best_run"]
    assert resp.tool_calls[0].status == "ok"
    assert resp.facts and resp.facts[0].evidence_ids


def test_multi_tool_integration_scenario_calls_all_modules_and_no_trigger():
    reg, calls = scenario_registry()

    def final(m):
        e = evidence_ids(m)
        return (
            f"**Known facts**\n- Best run {RUN_ID} F1 0.9412 on data-v1.1 [{e['mlflow_find_best_run']}]\n"
            f"- Deployed image tag 1.0.0 with ready replicas 2 [{e['k8s_get_deployed_version']}]\n"
            f"- Latest DAG run state success [{e['airflow_get_dag_runs']}]\n"
            f"- Policy: no rule triggered [{e['policy_evaluate_retraining']}]\n"
            "**Uncertainties**\n- No drift or degradation data supplied.\n"
            "**Recommended actions**\n- No retraining needed now."
        )

    llm = FakeLLM(
        [
            [("mlflow_find_best_run", {})],
            [
                ("dvc_get_dataset_for_run", {"run_id": RUN_ID}),
                ("dvc_list_versions", {}),
                ("k8s_get_deployed_version", {}),
                ("k8s_find_unhealthy_pods", {}),
                ("airflow_get_dag_runs", {"dag_id": "ml_training_pipeline"}),
            ],
            [("policy_evaluate_retraining", {"trained_data_version": "data-v1.1"})],
            final,
        ]
    )
    resp = Agent(llm, reg).run("best model, dataset, deployment, retrain?", "req2")
    modules = {reg.tools[c.tool].module for c in resp.tool_calls}
    assert {"mlflow", "dvc", "k8s", "airflow"} <= modules
    assert "airflow_trigger_dag" not in calls
    assert not resp.pending_actions
    for section in ("Known facts", "Uncertainties", "Recommended actions"):
        assert section in resp.answer
    assert resp.facts and resp.uncertainties and resp.recommendations
    # all results from one assistant turn go back in ONE user message
    second_turn_history = llm.calls[2]
    assert len(second_turn_history[-1]["content"]) == 5


def test_grounding_validator_flags_unsupported_numbers():
    ev = Evidence()
    reg, _ = scenario_registry()
    ev.add(reg.execute("mlflow_find_best_run", {}, "r"))
    eid = ev.results[0].evidence_id
    assert find_unsupported(f"F1 is 0.9412 [{eid}]", ev) == []
    assert "0.97" in find_unsupported("F1 is 0.97", ev)
    assert "[E-ffffff]" in find_unsupported("see [E-ffffff]", ev)
    assert "cafe1234" in find_unsupported("commit cafe1234", ev)
    assert find_unsupported("tag data-v1.1 and 94.12%", ev) == []


def test_agent_does_not_invent_results_when_tool_unavailable():
    reg, _ = scenario_registry(mlflow_ok=False)
    claim = "**Known facts**\n- Best F1 = 0.97 [E-abcdef]\n**Uncertainties**\n- none"
    llm = FakeLLM([[("mlflow_find_best_run", {})], claim, claim])
    resp = Agent(llm, reg).run("best model?", "req3")
    assert resp.tool_calls[0].status == "unavailable"
    assert "0.97" not in resp.answer
    assert "could not be retrieved" in resp.answer


def test_unavailable_guard_adds_note_when_llm_ignores_failure():
    reg, _ = scenario_registry(mlflow_ok=False)
    llm = FakeLLM([[("mlflow_find_best_run", {})], "**Known facts**\n- Nothing to report"])
    resp = Agent(llm, reg).run("best?", "req4")
    assert "mlflow information could not be retrieved" in resp.answer


def test_iteration_cap_enforced():
    reg, calls = scenario_registry()
    llm = FakeLLM([[("dvc_list_versions", {})]] * 30)
    resp = Agent(llm, reg).run("loop forever", "req5")
    assert len(calls) == 12
    assert "stopped before completion" in resp.answer


def test_mutating_tool_yields_pending_action_not_execution():
    reg, calls = scenario_registry()
    llm = FakeLLM(
        [
            [("airflow_trigger_dag", {"dag_id": "ml_training_pipeline"})],
            "**Known facts**\n- A trigger request was created and awaits your confirmation.\n"
            "**Uncertainties**\n- Not executed yet.\n**Recommended actions**\n- Confirm the action.",
        ]
    )
    resp = Agent(llm, reg).run("trigger training", "req6")
    assert calls == []
    assert len(resp.pending_actions) == 1
    assert resp.pending_actions[0].tool == "airflow_trigger_dag"


def test_llm_unavailable_is_reported_cleanly():
    class Down:
        def create(self, *a, **k):
            raise LLMUnavailable("rate limited")

    resp = Agent(Down(), scenario_registry()[0]).run("hi", "req7")
    assert "LLM is unavailable" in resp.answer and resp.tool_calls == []


def test_tool_result_blocks_flag_errors_and_are_redacted():
    reg = Registry()

    def leak(_):
        return {"password": "hunter2", "ok": 1}

    add_tool(reg, "dvc_leak", leak, module="dvc")
    llm = FakeLLM([[("dvc_leak", {})], "**Known facts**\n- done"])
    Agent(llm, reg).run("x", "req8")
    block = llm.calls[1][-1]["content"][0]
    body = json.loads(block["content"])
    assert block["is_error"] is False and "hunter2" not in block["content"]
    assert body["data"]["password"] == "***"


def test_parse_sections():
    facts, unc, rec = parse_sections(
        "**Known facts**\n- a [E-abcdef]\n**Uncertainties**\n- b\n**Recommended actions**\n- c"
    )
    assert facts[0].evidence_ids == ["E-abcdef"] and unc == ["b"] and rec == ["c"]


def test_removed_claims_note_is_merged_into_existing_uncertainties():
    reg, _ = scenario_registry(mlflow_ok=True)
    bad = (
        "**Known facts**\n- F1 is 0.55 [E-abcdef]\n**Uncertainties**\n- none\n"
        "**Recommended actions**\n- none"
    )
    llm = FakeLLM([[("mlflow_find_best_run", {})], bad, bad])
    resp = Agent(llm, reg).run("best?", "req9")
    assert resp.answer.count("**Uncertainties**") == 1
    assert "removed" in resp.answer and "0.55" not in resp.answer


def test_multi_id_citations_are_not_flagged_as_numbers():
    reg, _ = scenario_registry()
    ev = Evidence()
    a = reg.execute("mlflow_find_best_run", {}, "r")
    b = reg.execute("dvc_list_versions", {}, "r")
    ev.add(a)
    ev.add(b)
    text = (
        f"Best F1 0.9412 on data-v1.1 [{a.evidence_id}, {b.evidence_id}] and again {b.evidence_id}"
    )
    assert find_unsupported(text, ev) == []
    assert "[E-ffffff]" in find_unsupported(f"x [{a.evidence_id}; E-ffffff]", ev)
    facts, _, _ = parse_sections(f"**Known facts**\n- ok [{a.evidence_id}, {b.evidence_id}]")
    assert facts[0].evidence_ids == [a.evidence_id, b.evidence_id]
