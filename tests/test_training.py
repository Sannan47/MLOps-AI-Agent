"""Training modules: validation, provenance and quality gate (isolated MLflow sqlite store)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from training import common, evaluate, preprocess, quality_gate, train, validate
from training.common import DataValidationError, QualityGateError, load_params


@pytest.fixture
def params():
    return load_params()


def make_df(n=200, null_ratio=0.0, drop_label=False):
    import numpy as np

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({f: rng.normal(y * 0.6, 1.0) for f in load_params()["data"]["features"]})
    df["label"] = y
    if null_ratio:
        df.loc[: int(n * null_ratio), "mean_radius"] = None
    return df.drop(columns=["label"]) if drop_label else df


def test_validate_passes_on_good_data(params):
    assert validate.validate_frame(make_df(), params)["ok"]


def test_validate_dataset_fails_on_corrupted_data(params, tmp_path):
    p = tmp_path / "bad.csv"
    make_df(drop_label=True).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="label"):
        validate.validate_file(p, params, tmp_path / "r.json")
    assert (tmp_path / "r.json").exists()
    p2 = tmp_path / "nulls.csv"
    make_df(null_ratio=0.35).to_csv(p2, index=False)
    with pytest.raises(DataValidationError, match="null ratio"):
        validate.validate_file(p2, params)


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    return uri


def run_pipeline(tmp_path, params, name="r"):
    csv = tmp_path / "d.csv"
    make_df().to_csv(csv, index=False)
    preprocess.preprocess(csv, tmp_path / "proc", params)
    run_id = train.train(
        tmp_path / "proc", tmp_path / "m.joblib", params, run_name=name, validation_report=None
    )
    evaluate.evaluate(
        tmp_path / "proc", tmp_path / "m.joblib", run_id, params, tmp_path / "metrics.json"
    )
    return run_id


@pytest.mark.slow
def test_provenance_tags_present(mlflow_env, tmp_path, params):
    run_id = run_pipeline(tmp_path, params)
    tags = MlflowClient(mlflow_env).get_run(run_id).data.tags
    for k in (
        "data.path",
        "data.dvc_md5",
        "data.version",
        "data.git_commit",
        "data.dvc_remote",
        "code.git_commit",
        "code.git_dirty",
        "pipeline.trigger",
    ):
        assert tags.get(k), k
    assert "f1" in MlflowClient(mlflow_env).get_run(run_id).data.metrics


@pytest.mark.slow
def test_quality_gate_blocks_promotion(mlflow_env, tmp_path, params):
    run_id = run_pipeline(tmp_path, params)
    client = MlflowClient(mlflow_env)
    with pytest.raises(QualityGateError):
        quality_gate.apply_gate(run_id, params, min_f1=0.9999, client=client)
    assert client.get_run(run_id).data.tags["quality_gate"] == "failed"
    assert client.search_registered_models() == []


@pytest.mark.slow
def test_quality_gate_promotes_champion_then_challenger(mlflow_env, tmp_path, params):
    client = MlflowClient(mlflow_env)
    mlflow.set_tracking_uri(mlflow_env)
    r1 = run_pipeline(tmp_path, params, "a")
    out1 = quality_gate.apply_gate(r1, params, min_f1=0.5, client=client)
    assert out1["alias"] == "champion"
    r2 = run_pipeline(tmp_path, params, "b")
    out2 = quality_gate.apply_gate(r2, params, min_f1=0.5, client=client)
    assert out2["alias"] == "challenger"
    assert common.REPO_ROOT.exists()
    assert Path(common.__file__).exists()


def test_supersede_previous_attempts_marks_earlier_runs_failed():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    def run(rid, status, tags=None):
        return SimpleNamespace(
            info=SimpleNamespace(run_id=rid, status=status),
            data=SimpleNamespace(tags=tags or {}),
        )

    client = MagicMock()
    client.search_runs.return_value = [
        run("try1", "FINISHED"),
        run("old", "FAILED", {"airflow.superseded_by_retry": "true"}),
    ]
    out = train.supersede_previous_attempts("1", "manual__x'y", client=client)
    assert out == ["try1"]
    expected = "tags.`airflow.dag_run_id` = 'manual__x" + "\\" + "'y'"
    assert client.search_runs.call_args.kwargs["filter_string"] == expected
    client.set_tag.assert_called_once_with("try1", "airflow.superseded_by_retry", "true")
    client.set_terminated.assert_called_once_with("try1", status="FAILED")


@pytest.mark.slow
def test_airflow_retry_does_not_leave_orphan_finished_run(
    mlflow_env, tmp_path, params, monkeypatch
):
    monkeypatch.setenv("AIRFLOW_DAG_RUN_ID", "manual__retry_test")
    first = run_pipeline(tmp_path, params, "attempt1")
    second = run_pipeline(tmp_path, params, "attempt2")
    client = MlflowClient(mlflow_env)
    r1, r2 = client.get_run(first), client.get_run(second)
    assert r1.info.status == "FAILED"
    assert r1.data.tags["airflow.superseded_by_retry"] == "true"
    assert r2.info.status == "FINISHED"
    assert "airflow.superseded_by_retry" not in r2.data.tags
