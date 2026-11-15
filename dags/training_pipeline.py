"""ml_training_pipeline: validate -> preprocess -> train -> evaluate -> quality gate (Airflow 3)."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

from airflow.sdk import Param, dag, get_current_context, task

try:
    from airflow.sdk.exceptions import AirflowFailException
except ImportError:  # pragma: no cover
    from airflow.exceptions import AirflowFailException

PROJECT = Path(os.environ.get("PROJECT_DIR", "/opt/airflow/project"))
sys.path.insert(0, str(PROJECT))
log = logging.getLogger(__name__)

default_args = {
    "retries": 2,
    "retry_delay": timedelta(seconds=30),
    "retry_exponential_backoff": True,
    "execution_timeout": timedelta(minutes=10),
}


def _dataset_path(params: dict) -> Path:
    """Resolve the dataset: explicit path override, or `dvc get --rev` for a non-HEAD revision."""
    override = params.get("dataset_path")
    if override:
        return PROJECT / override
    rev = params.get("dataset_rev", "HEAD")
    if rev == "HEAD":
        return PROJECT / "data/raw/dataset.csv"
    dest = Path(tempfile.mkdtemp()) / "dataset.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "dvc",
            "get",
            str(PROJECT),
            "data/raw/dataset.csv",
            "--rev",
            rev,
            "-o",
            str(dest),
        ],
        check=True,
        cwd=PROJECT,
        timeout=120,
    )
    return dest


@dag(
    dag_id="ml_training_pipeline",
    schedule=None,
    catchup=False,
    tags=["ml", "training"],
    default_args=default_args,
    params={
        "dataset_rev": Param("HEAD", type="string"),
        "dataset_path": Param("", type=["string", "null"]),
        "min_f1": Param(0.90, type="number"),
    },
)
def ml_training_pipeline():
    @task
    def validate_dataset() -> str:
        from training.common import DataValidationError, load_params
        from training.validate import validate_file

        ctx = get_current_context()
        os.chdir(PROJECT)
        path = _dataset_path(ctx["params"])
        try:
            res = validate_file(path, load_params(), PROJECT / "validation_report.json")
        except DataValidationError as e:  # deterministic: do not retry
            log.error("validation failed for %s: %s", path, e)
            raise AirflowFailException(str(e)) from e
        log.info("validated %s rows=%s", path, res["rows"])
        return str(path)

    @task
    def preprocess_data(path: str) -> str:
        from training.common import load_params
        from training.preprocess import preprocess

        os.chdir(PROJECT)
        out = preprocess(path, PROJECT / "data/processed", load_params())
        log.info("preprocessed %s -> %s", path, out)
        return path

    @task
    def train_model(path: str) -> str:
        from training.common import load_params
        from training.train import train

        ctx = get_current_context()
        os.environ["PIPELINE_TRIGGER"] = "airflow"
        os.environ["AIRFLOW_DAG_RUN_ID"] = ctx["run_id"]
        # record the dataset revision actually trained on, not the workspace copy
        rev = ctx["params"].get("dataset_rev", "HEAD")
        if rev != "HEAD" and not ctx["params"].get("dataset_path"):
            os.environ["DATA_REV"] = rev
        else:
            os.environ.pop("DATA_REV", None)
        os.chdir(PROJECT)
        run_id = train(
            PROJECT / "data/processed",
            PROJECT / "models/model.joblib",
            load_params(),
            run_name=f"airflow-{ctx['run_id']}",
        )
        log.info("trained model run_id=%s", run_id)
        return run_id

    @task
    def evaluate_and_log_mlflow(mlflow_run_id: str) -> str:
        from training.common import load_params
        from training.evaluate import evaluate

        os.chdir(PROJECT)
        m = evaluate(
            PROJECT / "data/processed",
            PROJECT / "models/model.joblib",
            mlflow_run_id,
            load_params(),
            PROJECT / "metrics.json",
        )
        log.info("evaluated mlflow_run_id=%s metrics=%s", mlflow_run_id, m)
        return mlflow_run_id

    @task
    def apply_quality_gate(mlflow_run_id: str) -> dict:
        from training.common import QualityGateError, load_params
        from training.quality_gate import apply_gate

        ctx = get_current_context()
        min_f1 = float(ctx["params"].get("min_f1", 0.90))
        try:
            res = apply_gate(mlflow_run_id, load_params(), min_f1=min_f1)
        except QualityGateError as e:  # deterministic: fail immediately, no alias set
            log.error("quality gate failed mlflow_run_id=%s: %s", mlflow_run_id, e)
            raise AirflowFailException(str(e)) from e
        log.info("quality gate passed mlflow_run_id=%s result=%s", mlflow_run_id, res)
        return {"mlflow_run_id": mlflow_run_id, **{k: str(v) for k, v in res.items()}}

    path = validate_dataset()
    pre = preprocess_data(path)
    run = train_model(pre)
    evald = evaluate_and_log_mlflow(run)
    apply_quality_gate(evald)


ml_training_pipeline()
