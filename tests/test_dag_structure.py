"""Optional (needs apache-airflow installed, e.g. the CI dag-check job)."""

from __future__ import annotations

import pytest

pytest.importorskip("airflow")


def test_dag_structure():
    from airflow.models import DagBag

    bag = DagBag("dags", include_examples=False)
    assert not bag.import_errors
    dag = bag.dags["ml_training_pipeline"]
    assert [t.task_id for t in dag.topological_sort()] == [
        "validate_dataset",
        "preprocess_data",
        "train_model",
        "evaluate_and_log_mlflow",
        "apply_quality_gate",
    ]
    assert dag.default_args["retries"] >= 1 and dag.catchup is False
