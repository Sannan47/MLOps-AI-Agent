from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from mlflow.exceptions import MlflowException

from agent.registry import registry


def make_run(rid, f1, start, gate="passed", version="data-v1.1", precision=0.95, recall=0.95):
    return NS(
        info=NS(run_id=rid, experiment_id="1", status="FINISHED", start_time=start),
        data=NS(
            metrics={"f1": f1, "precision": precision, "recall": recall, "accuracy": 0.9},
            tags={"quality_gate": gate, "data.version": version, "mlflow.runName": rid},
            params={"train.C": str(f1)},
        ),
    )


class FakeClient:
    def __init__(self, runs, models=None):
        self.runs = runs
        self.models = models or []

    def search_experiments(self):
        return [NS(experiment_id="1", name="exp", lifecycle_stage="active")]

    def get_experiment_by_name(self, name):
        return NS(experiment_id="1") if name == "exp" else None

    def search_runs(self, ids, filter_string="", max_results=100, order_by=None):
        return sorted(self.runs, key=lambda r: -r.info.start_time)[:max_results]

    def get_run(self, rid):
        return next(r for r in self.runs if r.info.run_id == rid)

    def list_artifacts(self, rid):
        return [NS(path="model")]

    def search_registered_models(self):
        return self.models

    def search_model_versions(self, q):
        return [NS(version="1", run_id="a", creation_timestamp=1700000000000)]


def call(client, tool, args=None):
    with patch("agent.tools.mlflow_tools.get_client", lambda: client):
        return registry.execute(tool, args or {}, "r")


def test_mlflow_find_best_run_max_and_min_and_ties():
    runs = [make_run("a", 0.93, 1), make_run("b", 0.97, 2), make_run("c", 0.97, 3)]
    c = FakeClient(runs)
    res = call(c, "mlflow_find_best_run", {"experiment": "exp"})
    assert res.status == "ok"
    assert res.data["best"]["run_id"] == "c"  # tie -> newest
    assert set(res.data["tied_run_ids"]) == {"b", "c"}
    res_min = call(c, "mlflow_find_best_run", {"experiment": "exp", "mode": "min"})
    assert res_min.data["best"]["run_id"] == "a"
    assert res.data["margin"] == 0.0


def test_mlflow_best_run_includes_readiness_checks():
    runs = [make_run("a", 0.99, 5, gate="failed", precision=0.5), make_run("b", 0.9, 1)]
    models = [NS(name="m", aliases={"champion": "1"})]
    res = call(FakeClient(runs, models), "mlflow_find_best_run", {})
    rc = res.data["readiness_checks"]
    assert res.data["best"]["run_id"] == "a"
    assert rc["quality_gate_passed"] is False and rc["secondary_metrics_ok"] is False
    assert rc["production_ready"] is False and "quality_gate_passed" in rc["failed_checks"]


def test_mlflow_empty_experiment_returns_no_invented_data():
    res = call(FakeClient([]), "mlflow_find_best_run", {"experiment": "exp"})
    assert res.status == "ok" and res.data["best"] is None
    assert call(FakeClient([]), "mlflow_list_runs", {"experiment": "exp"}).data == []


def test_mlflow_unknown_experiment_rejected():
    assert call(FakeClient([]), "mlflow_list_runs", {"experiment": "nope"}).status == "rejected"


def test_mlflow_unavailable_returns_unavailable_status():
    class Down:
        def search_experiments(self):
            raise MlflowException("API request failed: Max retries exceeded, Connection refused")

    res = call(Down(), "mlflow_list_experiments")
    assert res.status == "unavailable"


def test_mlflow_compare_runs_last_n_and_param_diff():
    runs = [make_run(str(i), 0.9 + i / 100, i) for i in range(1, 7)]
    res = call(FakeClient(runs), "mlflow_compare_runs", {"last_n": 5})
    assert res.status == "ok" and len(res.data["runs"]) == 5
    assert res.data["best_per_metric"]["f1"] == "6"
    assert "train.C" in res.data["param_differences"]


def test_mlflow_compare_experiments_and_run_details():
    c = FakeClient([make_run("a", 0.9, 1), make_run("b", 0.94, 2)])
    res = call(c, "mlflow_compare_experiments", {"exp_a": "exp", "exp_b": "exp"})
    assert res.data["a"]["best"] == 0.94 and res.data["a"]["run_count"] == 2
    det = call(c, "mlflow_get_run_details", {"run_id": "a"})
    assert det.data["artifacts"] == ["model"] and det.data["tags"]["data.version"] == "data-v1.1"
    reg = call(c, "mlflow_get_model_registry", {})
    assert reg.status == "ok"


@pytest.mark.integration
def test_integration_real_mlflow():
    assert registry.execute("mlflow_list_experiments", {}, "r").status == "ok"


def test_mlflow_registry_unknown_model_name_is_rejected_not_empty():
    models = [NS(name="breast_cancer_clf", aliases={"champion": "1"})]
    res = call(FakeClient([], models), "mlflow_get_model_registry", {"model_name": "my-experiment"})
    assert res.status == "rejected" and "breast_cancer_clf" in res.error
