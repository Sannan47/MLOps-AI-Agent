"""Train a scikit-learn model and log it to MLflow (run id saved for later stages)."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from training.common import REPO_ROOT, capture_provenance, load_params, tracking_uri


def build_model(t: dict):
    if t["model"] == "logistic_regression":
        return LogisticRegression(C=t["C"], max_iter=t["max_iter"])
    if t["model"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=t["n_estimators"], max_depth=t["max_depth"], random_state=42
        )
    raise ValueError(f"unknown model {t['model']}")


def train(
    processed: str | Path,
    model_out: str | Path,
    params: dict,
    experiment: str | None = None,
    run_name: str | None = None,
    validation_report: str | Path | None = "validation_report.json",
    params_path: str | Path | None = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(experiment or params["mlflow"]["experiment"])
    label = params["data"]["label"]
    df = pd.read_csv(Path(processed) / "train.csv")
    model = build_model(params["train"])
    model.fit(df.drop(columns=[label]), df[label])
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(capture_provenance())
        mlflow.log_params({f"train.{k}": v for k, v in params["train"].items()})
        mlflow.log_param("split.seed", params["split"]["seed"])
        mlflow.log_artifact(str(params_path or REPO_ROOT / "params.yaml"))
        if validation_report and Path(validation_report).exists():
            mlflow.log_artifact(str(validation_report))
        mlflow.log_artifact(str(model_out), artifact_path="model_file")
        mlflow.sklearn.log_model(model, name="model")
        return run.info.run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--model-out", default="models/model.joblib")
    p.add_argument("--run-id-out", default="models/run_id.txt")
    p.add_argument("--experiment", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--params", default=None)
    a = p.parse_args()
    params = load_params(a.params)
    run_id = train(a.processed, a.model_out, params, a.experiment, a.run_name, params_path=a.params)
    Path(a.run_id_out).write_text(run_id, encoding="utf-8")
    print(f"run_id={run_id}")


if __name__ == "__main__":
    main()
